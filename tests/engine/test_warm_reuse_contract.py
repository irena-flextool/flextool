"""Warm LP reuse contract: a built LP is reused only for an identical sub-solve.

The in-process warm path (``run_chain_from_db(warm=True)``, the default of the CLI and GUI) keeps the
built :class:`polar_high.WarmProblem` and may re-solve it for the next sub-solve.  A reused LP keeps the
``Var`` frames and HiGHS names of the sub-solve it was built for, and the output writer and the
handoff read the solution by those labels.  So the LP is reused only when the next sub-solve's
FlexData (labels included) and solver configuration are identical, and the warm path pushes nothing.
Rolls shift ``dt`` on every roll, so they are always rebuilt.

What went wrong before, with the reuse of rolls (each case below fails on that code):

* the ``p_inflow`` RHS push ignored the autoscale Layer-2 row factor (objectives 4-6x cold);
* the outputs of a reused roll were read from the labels of the roll the LP was built for (values
  of other hours, or zeros), and the storage carry-over to the next roll was lost;
* ``p_step_duration`` that varies between steps was pushed by label into matrix cells only: the warm
  run stopped as infeasible;
* a second solve with other solver settings (``solve.scaling``) re-used the LP and its settings.

Fixtures: ``tests.json`` plus an override alternative per test (built from JSON, never a checked-in
DB).  Run under ``FLEXTOOL_AUTOSCALE_STRICT=1`` in the area gate.
"""
from __future__ import annotations

import copy
import dataclasses
import logging

import polars as pl
import pytest

from tests.engine._unit_db import TIMESTEPS, assert_outputs_equal, b64, make_db, time_map, utf8

pytestmark = pytest.mark.solver

DISPATCH = "y2020_2day_dispatch"
TWIN = "y2020_2day_twin"
LAST = "y2020_2day_last"


def _roll(i: int) -> str:
    return f"{DISPATCH}_roll_{i}"


def _rolling(horizon: float, jump: float) -> list:
    return [["solve", DISPATCH, "solve_mode", b64("rolling_window", "str")],
            ["solve", DISPATCH, "rolling_solve_horizon", b64(horizon, "float")],
            ["solve", DISPATCH, "rolling_solve_jump", b64(jump, "float")]]


def _single_p2020(solve: str) -> list:
    return [["solve", solve, "period_timeset",
             b64({"index_type": "str", "index_name": "period", "data": [["p2020", "2day"]]}, "map")],
            ["solve", solve, "realized_periods",
             b64({"value_type": "str", "index_name": "period", "data": ["p2020"]}, "array")],
            ["solve", solve, "solve_mode", b64("single_solve", "str")]]


def _co2_cap(caps: dict[str, float]) -> list:
    return [["group", "co2_limit", "co2_max_period",
             b64({"index_type": "str", "index_name": "period", "data": [[d, v] for d, v in caps.items()]}, "map")]]


def _warm_and_cold(tmp_path_factory, url: str, scen: str, tag: str) -> dict:
    from flextool.engine import run_chain_from_db

    out = {}
    for warm in (True, False):
        wf = tmp_path_factory.mktemp(f"{tag}_w{int(warm)}")
        out[warm] = (run_chain_from_db(url, scen, warm=warm, keep_solutions=True, work_folder=wf), wf)
    assert list(out[True][0]) == list(out[False][0])
    return out


def _assert_rolls_equal(runs: dict) -> None:
    (hot, wf_hot), (cold, wf_cold) = runs[True], runs[False]
    for name in cold:
        h, c = hot[name].solution.obj, cold[name].solution.obj
        assert h == pytest.approx(c, rel=1e-9), f"{name}: warm {h} vs cold {c}"
        assert_outputs_equal(wf_hot, wf_cold, name)


def _spy_layer2(mp: pytest.MonkeyPatch) -> list:
    """Record every Layer-2 plan the cascade computes from a range walk."""
    from flextool.engine.solve import _orchestration

    plans: list = []
    orig = _orchestration._autoscale_apply_layer2_pre_solve

    def spy(*args, **kwargs):
        plan, ranges = orig(*args, **kwargs)
        plans.append(plan)
        return plan, ranges

    mp.setattr(_orchestration, "_autoscale_apply_layer2_pre_solve", spy)
    return plans


@pytest.fixture
def layer2_plans(monkeypatch) -> list:
    return _spy_layer2(monkeypatch)


def _layer2_active(plans: list) -> bool:
    return any(p is not None and any(int(e) != 0 for e in p.type_exponents.values()) for p in plans)


# ---------------------------------------------------------------------------
# Rolls: always rebuilt, warm == cold in objective, outputs and carry-over


def test_rolling_layer2_warm_equals_cold(tmp_path_factory, layer2_plans) -> None:
    """Rolling 30 h / 6 h over p2020 + p2025 with a 1e12 / 0.5e12 CO2 cap: Layer 2 scales the energy
    family.  Before: rolls 1-3 and 9-11 were reused and their demand RHS lost the row factor."""
    values = (_co2_cap({"p2020": 1e12, "p2025": 0.5e12})
              + [["solve", DISPATCH, "period_timeset",
                  b64({"index_type": "str", "index_name": "period",
                       "data": [["p2020", "2day"], ["p2025", "2day"]]}, "map")],
                 ["solve", DISPATCH, "realized_periods",
                  b64({"value_type": "str", "index_name": "period", "data": ["p2020", "p2025"]}, "array")]]
              + _rolling(30.0, 6.0))
    url, scen = make_db(tmp_path_factory, alt="reuse_l2roll", values=values,
                        base_chain=["init", "west", "coal", "co2_limit"])
    runs = _warm_and_cold(tmp_path_factory, url, scen, "reuse_l2roll")
    assert _layer2_active(layer2_plans), (
        "Layer 2 did not scale the LP: the fixture no longer tests the Layer-2 RHS scaling")
    hot = runs[True][0]
    assert list(hot) == [_roll(i) for i in range(16)]
    _assert_rolls_equal(runs)
    assert not any(st.warm_used for st in hot.values()), "a roll reused an LP built for another window"


def test_rolling_outputs_and_storage_carry_match_cold(tmp_path_factory) -> None:
    """Battery, rolling 6 h / 6 h.  Before: rolls 3 and 6 were reused, their outputs were zeros and
    the next roll started without the stored energy (roll 4 objective off by 8.7e4)."""
    url, scen = make_db(tmp_path_factory, alt="reuse_battery", values=_rolling(6.0, 6.0),
                        base_chain=["init", "west", "coal", "battery"])
    runs = _warm_and_cold(tmp_path_factory, url, scen, "reuse_battery")
    hot, cold = runs[True][0], runs[False][0]
    assert len(cold) == 8
    carried = 0
    for name in cold:
        h, c = hot[name].flex_data.p_roll_continue_state, cold[name].flex_data.p_roll_continue_state
        assert (h is None) == (c is None), f"{name}: carry-in warm {h} vs cold {c}"
        if c is None:
            continue
        hf, cf = utf8(h.frame).sort(h.dims), utf8(c.frame).sort(c.dims)
        assert hf.drop("value").equals(cf.drop("value")), name
        assert hf["value"].to_list() == pytest.approx(cf["value"].to_list(), rel=1e-9, abs=1e-6), name
        carried += 1
    assert carried >= 6, "the fixture carries no storage state between rolls"
    _assert_rolls_equal(runs)
    assert not any(st.warm_used for st in hot.values())


def test_rolling_online_carry_matches_cold(tmp_path_factory) -> None:
    """Unit commitment, rolling 12 h / 6 h: ``coal_plant`` (``startup_method`` linear, ``min_load`` 0.5) serves a
    demand that is 0 for 9 h and 400 MW for 9 h, so it starts and stops across roll boundaries.  The carried
    commitment (``p_roll_continue_online``, from the previous roll's ``v_online`` read by label) and every roll's
    objective and outputs equal cold; the carry takes both an off (0) and a partly-on value."""
    demand = time_map({t: (0.0 if (i // 9) % 2 == 0 else -400.0) for i, t in enumerate(TIMESTEPS)}, -400.0)
    url, scen = make_db(tmp_path_factory, alt="reuse_online",
                        values=_rolling(12.0, 6.0) + [["node", "west", "inflow", demand]],
                        base_chain=["init", "west", "coal", "coal_min_load"])
    runs = _warm_and_cold(tmp_path_factory, url, scen, "reuse_online")
    hot, cold = runs[True][0], runs[False][0]
    assert len(cold) == 8
    carried = []
    for name in cold:
        h, c = hot[name].flex_data.p_roll_continue_online, cold[name].flex_data.p_roll_continue_online
        assert (h is None) == (c is None), f"{name}: online carry-in warm {h} vs cold {c}"
        if c is None:
            continue
        hf, cf = utf8(h.frame).sort(h.dims), utf8(c.frame).sort(c.dims)
        assert hf.drop("value").equals(cf.drop("value")), name
        assert hf["value"].to_list() == pytest.approx(cf["value"].to_list(), rel=1e-9, abs=1e-9), name
        carried.extend(cf["value"].to_list())
    assert len(carried) >= 6, "the fixture carries no commitment between rolls"
    assert min(carried) == pytest.approx(0.0, abs=1e-9) and max(carried) > 0.5, (
        f"the carried commitment does not change between rolls: {carried}")
    _assert_rolls_equal(runs)
    assert not any(st.warm_used for st in hot.values())


def test_rolling_variable_step_duration_warm_equals_cold(tmp_path_factory) -> None:
    """Step durations alternate 1 h / 3 h, rolling 8 h / 1 h: equal-shape rolls differ positionally in
    ``p_step_duration``.  Before: the warm run stopped (roll 1 infeasible)."""
    dur = [[f"t{i:04d}", (1.0, 3.0)[(i - 1) % 2]] for i in range(1, 8761)]
    values = [["timeline", "y2020", "timestep_duration", b64({"index_type": "str", "rank": 1, "data": dur}, "map")]]
    url, scen = make_db(tmp_path_factory, alt="reuse_stepdur", values=values + _rolling(8.0, 1.0),
                        base_chain=["init", "west", "coal"])
    runs = _warm_and_cold(tmp_path_factory, url, scen, "reuse_stepdur")
    hot = runs[True][0]
    assert len(hot) >= 20
    steps = {tuple(utf8(st.flex_data.p_step_duration.frame).sort("d", "t")["value"].to_list()[:2])
             for st in hot.values()}
    assert {(1.0, 3.0), (3.0, 1.0)} <= steps, f"step durations do not alternate between rolls: {steps}"
    _assert_rolls_equal(runs)
    assert not any(st.warm_used for st in hot.values())


# ---------------------------------------------------------------------------
# Identical sub-solves: reuse is kept, and only for the same solver configuration


def _three_solves(extra: list) -> list:
    """Solves A (``y2020_2day_dispatch``), B and C over p2020 / 2day: B is identical to A, C is last
    (only the last solve realizes the dispatch outputs, so C differs from B)."""
    return ([["model", "flexTool", "solves",
              b64({"value_type": "str", "index_name": "sequence_index", "data": [DISPATCH, TWIN, LAST]}, "array")]]
            + _single_p2020(TWIN) + _single_p2020(LAST) + extra)


THREE_ENTITIES = [["solve", TWIN, None], ["solve", LAST, None]]


@pytest.fixture(scope="module")
def twin_runs(tmp_path_factory) -> dict:
    """A, B, C with a 1e12 CO2 cap (Layer 2 scales the energy family); ``"plans"`` holds the Layer-2
    plans of both runs."""
    url, scen = make_db(tmp_path_factory, alt="reuse_twin", values=_three_solves(_co2_cap({"p2020": 1e12})),
                        base_chain=["init", "west", "coal", "co2_limit"], entities=THREE_ENTITIES)
    with pytest.MonkeyPatch.context() as mp:
        plans = _spy_layer2(mp)
        runs = _warm_and_cold(tmp_path_factory, url, scen, "reuse_twin")
    runs["plans"] = plans
    return runs


def test_label_identical_solve_reuses_with_layer2(twin_runs) -> None:
    from flextool.engine.solve._warm import _reuse_blocker

    hot = twin_runs[True][0]
    assert list(hot) == [DISPATCH, TWIN, LAST]
    assert _layer2_active(twin_runs["plans"]), "Layer 2 did not scale the LP: the reuse is not tested under Layer 2"
    # Non-vacuity: the real A -> B pair is identical, so the reuse branch is exercised ...
    assert _reuse_blocker(hot[DISPATCH].flex_data, hot[TWIN].flex_data) is None
    assert hot[TWIN].warm_used, "the identical solve B did not reuse A's LP"
    # ... and B -> C legitimately differs (only the last solve realizes the dispatch).
    assert _reuse_blocker(hot[TWIN].flex_data, hot[LAST].flex_data) is not None
    assert not hot[LAST].warm_used
    _assert_rolls_equal(twin_runs)


def test_solver_settings_change_prevents_reuse(tmp_path_factory, monkeypatch, caplog) -> None:
    """B has ``solve.scaling = basic``: its data equal A's, but the solver configuration does not, so it
    builds its own LP.  Before: B re-used A's LP and ran under A's settings."""
    from flextool.engine.solve._warm import _reuse_blocker

    monkeypatch.delenv("FLEXTOOL_SCALING", raising=False)
    url, scen = make_db(tmp_path_factory, alt="reuse_scaling",
                        values=_three_solves([["solve", TWIN, "scaling", b64("basic", "str")]]),
                        base_chain=["init", "west", "coal"], entities=THREE_ENTITIES)
    with caplog.at_level(logging.INFO):
        runs = _warm_and_cold(tmp_path_factory, url, scen, "reuse_scaling")
    hot = runs[True][0]
    assert _reuse_blocker(hot[DISPATCH].flex_data, hot[TWIN].flex_data) is None, (
        "A and B differ in their data: the test would not isolate the solver configuration")
    assert not hot[TWIN].warm_used, "B re-used A's LP despite another solve.scaling"
    reasons = [r.getMessage() for r in caplog.records
               if "COLD REBUILD" in r.getMessage() and TWIN in r.getMessage()]
    assert reasons and all("solver configuration differs" in m for m in reasons), reasons
    _assert_rolls_equal(runs)


# ---------------------------------------------------------------------------
# Unit level: every field counts


def _relabel_first_row(frame: pl.DataFrame) -> pl.DataFrame | None:
    """``frame`` with the first row's value of one column replaced by another value of the same dtype, or
    ``None`` when no column allows it."""
    if frame.height == 0:
        return None
    first = pl.int_range(pl.len()) == 0
    for c, dtype in frame.schema.items():
        if dtype == pl.Utf8:
            new = pl.lit("__changed__")
        elif isinstance(dtype, pl.Enum) and len(dtype.categories) >= 2:
            cur = frame[c][0]
            new = pl.lit(next(x for x in dtype.categories if x != cur)).cast(dtype)
        elif dtype.is_numeric():
            new = pl.col(c) + 1
        else:
            continue
        return frame.with_columns(pl.when(first).then(new).otherwise(pl.col(c)).cast(dtype).alias(c))
    return None


def _shift_t(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(pl.col("t").cast(pl.Utf8).str.replace(r"^t", "s").alias("t"))


def test_reuse_blocker_names_each_change(twin_runs) -> None:
    from flextool.engine.solve._warm import (
        _STRUCTURAL_FIELDS,
        _IncompatibleUpdate,
        _apply_warm_updates,
        _reuse_blocker,
    )
    from polar_high import Param

    data = twin_runs[True][0][DISPATCH].flex_data
    assert _apply_warm_updates(None, data, copy.deepcopy(data)) == 0

    inflow = data.p_inflow
    sd = data.p_step_duration
    first = pl.int_range(pl.len()) == 0
    # A set field outside the fingerprint's ``_STRUCTURAL_FIELDS``, with one label changed (same height,
    # same dtype): before, such a field was not compared at all.
    some_set, changed = next(
        (f.name, alt) for f in dataclasses.fields(data)
        if f.name not in _STRUCTURAL_FIELDS and isinstance(getattr(data, f.name), pl.DataFrame)
        and (alt := _relabel_first_row(getattr(data, f.name))) is not None)
    changes = {
        "p_inflow": Param(inflow.dims, inflow.frame.with_columns(pl.col("value") * 1.5)),
        "p_step_duration": Param(sd.dims, sd.frame.with_columns(
            pl.when(first).then(pl.col("value") * 2).otherwise(pl.col("value")).alias("value"))),
        "p_storage_state_reference_value": Param(("n",), pl.DataFrame({"n": ["west"], "value": [1.0]})),
        "dt": _shift_t(data.dt),
        some_set: changed,
    }
    assert data.p_storage_state_reference_value is None
    for name, value in changes.items():
        nxt = dataclasses.replace(data, **{name: value})
        assert _reuse_blocker(data, nxt) == name
        with pytest.raises(_IncompatibleUpdate, match=repr(name)):
            _apply_warm_updates(None, data, nxt)


# ---------------------------------------------------------------------------
# Memory: the previous WarmProblem is released before an LP that is not a reuse of it


def test_cold_built_solve_releases_previous_warm_problem(tmp_path_factory, monkeypatch) -> None:
    """A warm HiGHS solve followed by a save-memory (cold, subprocess) solve: the previous WarmProblem
    (and its HiGHS instance) must be gone before the cold build, not kept alive for nothing."""
    import gc
    import weakref

    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core
    from flextool.engine.solve import _warm
    from flextool.run_options import Resolved, RunOptions

    # The run's options are resolved once; the test switches save-memory on for the later sub-solves through
    # the run's own resolution (its ``resolved`` mapping), as a flag would have set it.
    run_options = RunOptions.from_env({})
    url, scen = make_db(tmp_path_factory, alt="reuse_release", values=_three_solves([]),
                        base_chain=["init", "west", "coal"], entities=THREE_ENTITIES)
    built: list = []
    alive_at_cold_build: list[bool] = []
    orig_build_warm = _warm._build_warm_problem
    orig_build = core.build_flextool

    def build_warm(*args, **kwargs):
        wp = orig_build_warm(*args, **kwargs)
        built.append(weakref.ref(wp))
        # Every later sub-solve takes the save-memory (cold) path.
        run_options.resolved["save_memory"] = Resolved(True, "cli")
        return wp

    def build(pb, data, **kwargs):
        if built and len(alive_at_cold_build) < 2:
            gc.collect()
            alive_at_cold_build.append(built[0]() is not None)
        return orig_build(pb, data, **kwargs)

    monkeypatch.setattr(_warm, "_build_warm_problem", build_warm)
    monkeypatch.setattr(core, "build_flextool", build)
    steps = run_chain_from_db(url, scen, warm=True, keep_solutions=False,
                              work_folder=tmp_path_factory.mktemp("reuse_release_w"), run_options=run_options)
    assert list(steps) == [DISPATCH, TWIN, LAST]
    assert len(built) == 1, "expected one warm build (the first solve) before save-memory took over"
    assert alive_at_cold_build and not any(alive_at_cold_build), (
        f"the first solve's WarmProblem was still alive at a cold build: {alive_at_cold_build}")
