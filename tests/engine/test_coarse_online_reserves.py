"""Coarse processes with online variables: units online, start-ups and shut-downs once per coarse step.

Owner decision 2026-11-04 ("Coarse flows for units with online variables / reserves"): a unit (or connection)
whose every balance node is coarse keeps one flow per coarse step also when it has online variables; its units
online, start-ups and shut-downs are decided once per coarse step (the start / stop at the step's start), a
minimum up / down time counts whole coarse steps, the online capacity and minimum-load rows are averaged over the
step, and the outputs show units online on every solve step and each start-up / shut-down at its step's first
solve step.  With ramp limits, its start-up / shut-down capacity reduction uses the mean share over the whole coarse
step and its units online ramp between coarse steps.  A coarse process holds one reserve per coarse step (owner
decision (5)–(7)): its reserve limits (per process, per connection position, minimum flow, storage energy when every
member is coarse) are averaged over the step, the reserve's ramp room is the smallest of the step's solve steps, and
the requirement rows stay per solve step.  A process whose forecast tie or realization pin covers only part of a
coarse step keeps the solve's step.

Fixtures: ``tests/fixtures/lh2_three_region.json`` (liquefier_A h2_A -> lh2_A between daily nodes, one week of
hourly steps), ``tests/fixtures/stochastics.json`` and ``tests.json`` (hand-built model), with override
alternatives built from JSON under the pytest tmp dir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from tests.engine._lp_rows import row_bounds, row_terms
from tests.engine._unit_db import (
    BASE_FIXTURE_JSON, FLEXTOOL_ROOT, RunCache, b64, build, is_realization, last_stochastic, make_db, run, utf8,
    without_realization,
)
from tests.engine.test_coarse_flows import (
    LH2, LH2_COARSE, LH2_RP, STOCH, _chain, _coarse_procs, _f, _forecasts, _keys, _s, _stoch_values,
    _STOCH_ENTITIES, _tmap_lh2, _total_objective,
)

pytestmark = pytest.mark.solver

_DAYS = [f"t{24 * k + 1:04d}" for k in range(7)]
# liquefier_A as the lp_snapshot fixture coarse_uc: linear online, min_load 0.3, its 200 MW as 4 units of 50 MW.
_UC = [["unit", "liquefier_A", "startup_method", _s("linear")], ["unit", "liquefier_A", "min_load", _f(0.3)],
       ["unit", "liquefier_A", "existing", _f(4.0)], ["unit", "liquefier_A", "virtual_unitsize", _f(50.0)]]
# ... with a start-up cost, a minimum up / down time longer than a day, and a liquid hydrogen demand at lh2_A that
# alternates between 10 MW and 150 MW from day to day (10 MW on the first) with almost no storage there, so the
# units start and stop.
_UC_FULL = _UC + [["unit", "liquefier_A", "startup_cost", _f(50.0)],
                  ["node", "lh2_A", "existing", _f(1.0)],
                  ["node", "lh2_A", "inflow", _tmap_lh2(lambda h: -10.0 if (h // 24) % 2 == 0 else -150.0)],
                  ["unit", "liquefier_A", "minimum_time_method", _s("both")],
                  ["unit", "liquefier_A", "min_uptime", _f(30.0)], ["unit", "liquefier_A", "min_downtime", _f(30.0)]]


# The coarse models two tests solve alike share one run.  ``_tied_fine`` sets ``_STATE["tied"]`` (undone with its
# monkeypatch at the test's end): every run of the tied hourly model is made afresh, never from the cache.
_RUNS = RunCache()
_STATE = {"tied": False}


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _model(tmp_path_factory, *, alt: str, values: list, base_chain: list[str], entities: list | None = None,
           fixture_json: Path = BASE_FIXTURE_JSON) -> dict:
    """``run`` of the model: shared within the module unless the tied hourly model is being built (read only)."""
    if _STATE["tied"]:
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=base_chain, entities=entities,
                            fixture_json=fixture_json)
        return run(tmp_path_factory, url, scen)
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=base_chain,
                                  entities=entities, fixture_json=fixture_json)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


def _lh2(tmp_path_factory, alt, values=(), entities=None) -> dict:
    return _model(tmp_path_factory, alt=alt, values=list(values), base_chain=_chain(LH2, "lh2_three_region"),
                  entities=entities, fixture_json=LH2)


def _cols(pb, prefix: str) -> list[str]:
    return [c for c in pb.canonicalise().col_names if c.startswith(prefix)]


# ---------------------------------------------------------------------------
# Structure


def test_one_step_reader_equals_the_full_reader(tmp_path_factory) -> None:
    """``var_values_at`` (the roll-end handoffs' reader of one step) gives the rows of ``var_values`` (every fine
    step) at that step, for a level (copy mode) and an event (first mode), at a coarse step's first hour, inside
    it and at its last hour, for coarse and fine processes."""
    from flextool.engine.output._coarse_flows import var_values, var_values_at

    step = last_stochastic(_lh2(tmp_path_factory, "co_reader", _UC))
    sol, fd = step.solution, step.flex_data
    for name in ("v_flow", "v_online_linear", "v_startup_linear"):
        full = utf8(var_values(sol, fd, name))
        for t in ("t0025", "t0030", "t0048", "t0168"):
            one = utf8(var_values_at(sol, fd, name, "y2030", t))
            want = full.filter((pl.col("d") == "y2030") & (pl.col("t") == t))
            cols = [c for c in want.columns if c != "value"]
            assert one.height == want.height > 0, (name, t)
            assert one.sort(cols).equals(want.sort(cols)), (name, t)


def test_online_unit_is_coarse_with_one_column_per_day(tmp_path_factory) -> None:
    """coarse_uc: liquefier_A (linear online, min_load 0.3) is coarse.  Its units online, start-ups and shut-downs
    have 7 columns (one per day, at the day's first hour), online_change / maxOnline / maxStartup 7 rows, the online
    capacity and minimum load one averaged row per day (maxFlow_online_linear_coarse, minFlow_minload_linear_coarse)
    and no hourly row; its v_flow has 7 columns."""
    fd = last_stochastic(_lh2(tmp_path_factory, "co_struct", _UC)).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    pb = build(fd)
    for var in ("v_online_linear", "v_startup_linear", "v_shutdown_linear"):
        cols = _cols(pb, f"{var}[liquefier_A,")
        assert [_keys(c)[-1] for c in cols] == _DAYS, var
    assert [_keys(c)[-1] for c in _cols(pb, "v_flow[liquefier_A,h2_A,lh2_A,")] == _DAYS
    for fam in ("online_change_linear", "maxOnline_linear", "maxStartup_linear", "minFlow_minload_linear_coarse"):
        assert sorted(_keys(r)[-1] for r in row_terms(pb, fam, "liquefier_A,")) == _DAYS, fam
    assert len(row_terms(pb, "maxFlow_online_linear_coarse", "liquefier_A,h2_A,lh2_A,")) == 7
    assert not row_terms(pb, "maxFlow_online_linear", "liquefier_A,")
    assert not row_terms(pb, "minFlow_minload_linear", "liquefier_A,")
    # The day-2 row: online − (online of day 1) − startup + shutdown = 0; day 1 wraps to day 7 (stretch start).
    rows = row_terms(pb, "online_change_linear", "liquefier_A,y2030,t0025")
    assert rows["online_change_linear[liquefier_A,y2030,t0025]"] == pytest.approx({
        "v_online_linear[liquefier_A,y2030,t0025]": 1.0, "v_online_linear[liquefier_A,y2030,t0001]": -1.0,
        "v_startup_linear[liquefier_A,y2030,t0025]": -1.0, "v_shutdown_linear[liquefier_A,y2030,t0025]": 1.0})
    first = row_terms(pb, "online_change_linear",
                      "liquefier_A,y2030,t0001")["online_change_linear[liquefier_A,y2030,t0001]"]
    assert first["v_online_linear[liquefier_A,y2030,t0145]"] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Exact: the coarse model == the fine model with every coarse-process value tied over each coarse step

_LEVELS = ("v_flow", "v_flow_back", "v_online_linear", "v_online_integer", "v_reserve")
_EVENTS = ("v_startup_linear", "v_startup_integer", "v_shutdown_linear", "v_shutdown_integer")


def _add_ties(pb, fd) -> None:
    """In the fine LP ``pb`` of ``fd`` (every process at the solve's step): for each coarse process of
    ``fd.process_coarse_step`` and each interior fine step t of a coarse step b, ``x[…,t] = x[…,b]`` for a level x
    (flows, units online, reserves) and ``x[…,t] = 0`` for an event x (start-ups, shut-downs)."""
    from polar_high.engine import Var

    steps = getattr(fd, "process_coarse_step", None)
    if steps is None or steps.height == 0:
        return
    st = steps.select(*(pl.col(c).cast(pl.Utf8).alias(f"_{c}") for c in ("p", "d", "b_first", "t")))
    for name in (*_LEVELS, *_EVENTS):
        var = pb._vars.get(name)
        if var is None:
            continue
        dims = var.dims
        fr = var.frame.with_columns(*(pl.col(c).cast(pl.Utf8).alias(f"_{c}") for c in dims))
        inner = (fr.join(st, on=["_p", "_d", "_t"], how="inner")
                 .filter(pl.col("_t") != pl.col("_b_first")))
        if inner.height == 0:
            continue
        over = inner.select(dims)
        lhs = {"a": Var(name, dims, inner.select(*dims, "col_id"))}
        if name in _LEVELS:
            first = fr.select(*[f"_{c}" for c in dims], pl.col("col_id").alias("__first"))
            keys = [f"_{c}" for c in dims[:-1]]
            paired = inner.join(first.rename({"_t": "_b_first"}), on=[*keys, "_b_first"], how="inner")
            assert paired.height == inner.height, name
            lhs["b"] = -Var(name, dims, paired.select(*dims, pl.col("__first").alias("col_id")))
        pb.add_cstr(f"tie_{name}", over=over, sense="==", lhs_terms=lhs, rhs_terms={})


def _tied_fine(monkeypatch) -> None:
    """Build every solve fine (the model ignores processCoarse) with the coarse steps tied (:func:`_add_ties`).
    The tie rows are a test device outside the autoscale registry: not strict there."""
    import flextool.engine.model.core as core
    from flextool.engine.model import _coarse_flows

    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    monkeypatch.setitem(_STATE, "tied", True)
    monkeypatch.setattr(_coarse_flows, "prepare", lambda b: None)
    original = core.build_flextool

    def build_tied(pb, data, *args, **kwargs):
        out = original(pb, data, *args, **kwargs)
        _add_ties(pb, data)
        return out

    monkeypatch.setattr(core, "build_flextool", build_tied)


_ROLLING = [["solve", "lh2_week", "solve_mode", _s("rolling_window")],
            ["solve", "lh2_week", "rolling_solve_horizon", _f(72.0)],
            ["solve", "lh2_week", "rolling_solve_jump", _f(48.0)]]
# Integer units online (startup_method binary in the fixture's schema version; migrated to integer).
_INTEGER = [["unit", "liquefier_A", "startup_method", _s("binary")],
            ["solve", "lh2_week", "solver_mip_gap", _f(1e-9)]]
# min_load_efficiency: the section input (units online × section) in the conversion and the balance.
_SECTION = [["unit", "liquefier_A", "conversion_method", _s("min_load_efficiency")],
            ["unit", "liquefier_A", "efficiency_at_min_load", _f(0.6)]]


def _uc_case(tmp_path_factory, case: str, alt: str) -> dict:
    if case == "rp":
        return _model(tmp_path_factory, alt=alt, values=_UC_FULL, fixture_json=LH2_RP,
                      base_chain=_chain(LH2_RP, "lh2_three_region_rp_invest"))
    extra = {"single": [], "rolling": _ROLLING, "integer": _INTEGER, "section": _SECTION}[case]
    base = [v for v in _UC_FULL if case != "integer" or v[2] != "startup_method"]
    return _lh2(tmp_path_factory, alt, base + extra)


def _startups(steps: dict, kind: str = "linear") -> float:
    total = 0.0
    for s in without_realization(steps).values():
        v = s.solution.value(f"v_startup_{kind}") if f"v_startup_{kind}" in s.solution._vars else None
        if v is not None:
            total += float(utf8(v).filter(pl.col("p") == "liquefier_A")["value"].sum())
    return total


@pytest.mark.parametrize("case", ["single", "rolling", "rp", "integer", "section"])
def test_coarse_online_equals_tied_fine_model(tmp_path_factory, monkeypatch, case) -> None:
    """liquefier_A with linear (or integer) online variables, min_load 0.3, a start-up cost and 30-h minimum up /
    down times, every parameter constant within a day: the coarse model has the objective of the hourly model whose
    flows and units online are tied over each day and whose start-ups / shut-downs are 0 after a day's first hour —
    in a single solve, a rolling solve (72 h / 48 h: the units online and the 30-h minimum times carried across the
    roll), a representative-period invest solve, with integer units online and with a min_load_efficiency section.
    The start-ups are counted once per day (the coarse model starts units; a representative day is one coarse
    step that wraps to itself, so it has none)."""
    kind = "integer" if case == "integer" else "linear"
    coarse = _uc_case(tmp_path_factory, case, f"co_tie_{case}")
    assert all("liquefier_A" in _coarse_procs(s.flex_data) for s in without_realization(coarse).values())
    _tied_fine(monkeypatch)
    tied = _uc_case(tmp_path_factory, case, f"co_tie_{case}_fine")
    assert len(coarse) == len(tied)
    assert _total_objective(coarse) == pytest.approx(_total_objective(tied), rel=1e-9)
    assert (_startups(coarse, kind) > 1e-6) == (case != "rp")
    assert _startups(coarse, kind) == pytest.approx(_startups(tied, kind), rel=1e-6, abs=1e-9)
    if case == "integer":
        on = utf8(last_stochastic(coarse).solution.value("v_online_integer")).filter(pl.col("p") == "liquefier_A")
        assert on.height == 7
        assert all(abs(v - round(v)) < 1e-6 for v in on["value"].to_list())


# ---------------------------------------------------------------------------
# Roll carry: units online and the minimum up / down time history across a roll


def _value_at(step, var: str, t: str) -> float:
    v = utf8(step.solution.value(var)).filter((pl.col("p") == "liquefier_A") & (pl.col("t") == t))
    assert v.height == 1, (var, t, v)
    return float(v["value"][0])


def test_roll_carry_of_coarse_units_online(tmp_path_factory) -> None:
    """Rolling 72 h / 48 h: roll 0 realizes t0001–t0048, whose last step t0048 lies inside liquefier_A's day-2
    coarse step (first hour t0025).  Roll 1 continues from it: units online carried = roll 0's day-2 value; the
    on-line history has every realized hour that ends less than 30 h before the end, ends at t0048 (its reference row:
    hours_to_end = its own 1-h duration) and holds the day-2 start-up at t0025 (hours_to_end 24) with 0 at the
    other hours of the day; the 30-h minimum up time carries that start-up to roll 1's first hour t0049
    (the hours 24 + 0 … 24 + 5 < 30 of the carry), whose row is day 3's minimum-uptime row; the rows of days 4
    and 5 look back one day each (24 < 30)."""
    from flextool.engine.preprocess.params._mintime import history_reference_step

    rolls = list(without_realization(_uc_case(tmp_path_factory, "rolling", "co_roll_carry")).values())
    r0, r1 = rolls[0], rolls[1]
    fd = r1.flex_data
    on = utf8(fd.p_roll_continue_online.frame).filter(pl.col("p") == "liquefier_A")["value"].to_list()
    assert on == pytest.approx([_value_at(r0, "v_online_linear", "t0025")])
    hist = (fd.online_startup_history.with_columns(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "t")))
            .filter(pl.col("p") == "liquefier_A").sort("hours_to_end"))
    assert history_reference_step(hist) == ("y2030", "t0048")
    assert hist["t"].to_list() == [f"t{h:04d}" for h in range(48, 19, -1)]
    assert hist["hours_to_end"].to_list() == pytest.approx([float(k) for k in range(1, 30)])
    su = _value_at(r0, "v_startup_linear", "t0025")
    assert su > 1e-6
    by_t = dict(zip(hist["t"].to_list(), hist["startup"].to_list()))
    assert by_t.pop("t0025") == pytest.approx(su)
    assert all(v == 0.0 for t, v in by_t.items() if t >= "t0025")
    carry = utf8(fd.uptime_carry).filter(pl.col("p") == "liquefier_A").sort("t")
    assert carry["t"].to_list() == [f"t{h:04d}" for h in range(49, 55)]
    assert carry["value"].to_list() == pytest.approx([su] * 6)
    # The coarse model reads it once, on day 3's row: units online − its own start-up ≥ the carried start-up.
    pb = build(fd)
    rows = row_terms(pb, "minimum_uptime_linear", "liquefier_A,")
    assert sorted(_keys(r)[-1] for r in rows) == ["t0049", "t0073", "t0097"]
    lo = row_bounds(pb, "minimum_uptime_linear", "liquefier_A,y2030,t0049")[
        "minimum_uptime_linear[liquefier_A,y2030,t0049]"][0]
    assert lo == pytest.approx(su)


# ---------------------------------------------------------------------------
# Hand-built model (tests.json, 2 days of 1-h steps, period p2020): coal_market -> gen_d -> hd
#
# hd: a balance node in a 24-h group, demand 25 MW for the 24 hours of day 1 (600 MWh), 0 on day 2; penalty_up
# 10000 and penalty_down 1000 €/MWh (P = 1000: a surplus costs P).  gen_d: coal_market (commodity, no balance: neutral;
# coal price 20 €/MWh) -> hd, efficiency 1, existing 1 = one unit of 100 MW (virtual_unitsize 100), linear
# online, min_load 0.5, startup_cost 1000 €/MW (1e5 € per unit started).  gen_d is coarse: on_k, su_k, sd_k once
# per day k (k = 1, 2), flow F_k.
# Rows (EQ. headers of online_change, maxFlow_online_coarse, minFlow_minload_coarse, minimum_uptime):
#   on_1 − on_2 = su_1 − sd_1 (day 1 is the stretch start: its predecessor wraps to the stretch's last step, day 2),
#   on_2 − on_1 = su_2 − sd_2;  50·on_k ≤ F_k ≤ 100·on_k (MW);  day 1 balance F_1 = 25 (a shortfall costs 10000);
#   day 2 surplus 24·F_2 MWh at P + 20 €/MWh (penalty and fuel).
# (a) min_uptime 24 h: the day-2 row looks back to day 1 only if elapsed(day 1 -> day 2) = 24 < 24: no; the day-1 row
#     has no ancestor (no wrap in the look-back).  Both hold only their own step: no minimum_uptime row.
#     Cost 1e5·(su_1 + su_2) + 1200·(P + 20)·on_2 (F_2 = 50·on_2 at least) + fuel of day 1.  0.25 ≤ on_1 ≤ 0.5;
#     su_1 ≥ on_1 − on_2.  on_2 = x > 0 saves at most 1e5·x of start-up and costs 1200·1020·x: on_2 = 0, on_1 = 0.25,
#     su_1 = 0.25, sd_1 = 0, su_2 = 0, sd_2 = 0.25; F_1 = 25 MW, F_2 = 0.
# (b) min_uptime 30 h: the day-2 row looks back to day 1 (24 < 30): on_2 ≥ su_1 + su_2.  With on_2 = y ≤ 0.25:
#     su_1 ≥ 0.25 − y and y ≥ su_1 ⇒ y ≥ 0.125; cost 1e5·(0.25 − y) + 1200·1020·y rises with y ⇒ y = 0.125 (for any
#     P > 1e5/1200 − 20 = 63.3).  on_1 = 0.25 (a higher on_1 needs a larger su_1 and y), su_1 = 0.125, sd_1 = 0,
#     on_2 = 0.125, su_2 = 0, sd_2 = 0.125; F_1 = 25 MW, F_2 = 50·0.125 = 6.25 MW (150 MWh of surplus on day 2).

_HAND_CHAIN = ["init", "west", "coal"]
_HAND_ENTITIES = [["node", "hd", None], ["unit", "gen_d", None], ["group", "daily", None],
                  ["unit__inputNode", ["gen_d", "coal_market"], None], ["unit__outputNode", ["gen_d", "hd"], None],
                  ["group__node", ["daily", "hd"], None]]
_HOURS = [f"t{h:04d}" for h in range(1, 49)]


def _hand_uc_values(min_uptime: float) -> list:
    demand = b64({"index_type": "str", "index_name": "time",
                  "data": [[t, -25.0 if h < 24 else 0.0] for h, t in enumerate(_HOURS)]}, "map")
    vals = [["group", "daily", "new_stepduration", _f(24.0)],
            ["node", "hd", "node_type", _s("balance")], ["node", "hd", "inflow", demand],
            ["node", "hd", "penalty_up", _f(10000.0)], ["node", "hd", "penalty_down", _f(1000.0)],
            ["unit", "gen_d", "conversion_method", _s("constant_efficiency")], ["unit", "gen_d", "efficiency", _f(1.0)],
            ["unit", "gen_d", "existing", _f(1.0)], ["unit", "gen_d", "virtual_unitsize", _f(100.0)],
            ["unit", "gen_d", "startup_method", _s("linear")], ["unit", "gen_d", "min_load", _f(0.5)],
            ["unit", "gen_d", "startup_cost", _f(1000.0)],
            ["unit", "gen_d", "minimum_time_method", _s("min_uptime")],
            ["unit", "gen_d", "min_uptime", _f(min_uptime)]]
    return vals


def _hand_uc(tmp_path_factory, alt: str, min_uptime: float):
    return last_stochastic(_model(tmp_path_factory, alt=alt, values=_hand_uc_values(min_uptime),
                                  base_chain=_HAND_CHAIN, entities=_HAND_ENTITIES))


def _hourly(step, var: str) -> list[float]:
    from flextool.engine.output._coarse_flows import var_values

    v = var_values(step.solution, step.flex_data, var)
    rows = utf8(v.drop("value")).with_columns(value=v["value"]).filter(pl.col("p") == "gen_d").sort("t")
    assert rows["t"].to_list() == _HOURS
    return rows["value"].to_list()


@pytest.mark.parametrize("min_uptime, on, su, sd", [
    (24.0, (0.25, 0.0), (0.25, 0.0), (0.0, 0.25)),
    (30.0, (0.25, 0.125), (0.125, 0.0), (0.0, 0.125)),
])
def test_hand_built_coarse_unit_commitment(tmp_path_factory, min_uptime, on, su, sd) -> None:
    """The hand-built model (derivation above): units online, start-ups and shut-downs once per day, on every hour of
    the day for units online and at the day's first hour for start-ups / shut-downs (0 at its other hours); the
    flow 25 MW on day 1 and 50 · on_2 MW on day 2; minimum_uptime rows only where the look-back reaches the previous
    day (min_uptime 30 h: the day-2 row; 24 h: none)."""
    step = _hand_uc(tmp_path_factory, f"co_hand_{int(min_uptime)}", min_uptime)
    assert "gen_d" in _coarse_procs(step.flex_data)
    assert _hourly(step, "v_online_linear") == pytest.approx([on[0]] * 24 + [on[1]] * 24, abs=1e-9)
    for var, (first, second) in (("v_startup_linear", su), ("v_shutdown_linear", sd)):
        assert _hourly(step, var) == pytest.approx([first] + [0.0] * 23 + [second] + [0.0] * 23, abs=1e-9), var
    assert [f * 100.0 for f in _hourly(step, "v_flow")] == pytest.approx([25.0] * 24 + [50.0 * on[1]] * 24, abs=1e-7)
    rows = row_terms(build(step.flex_data), "minimum_uptime_linear", "gen_d,")
    if min_uptime == 24.0:
        assert not rows
    else:
        assert rows == {"minimum_uptime_linear[gen_d,p2020,t0025]": pytest.approx(
            {"v_online_linear[gen_d,p2020,t0025]": 1.0, "v_startup_linear[gen_d,p2020,t0001]": -1.0,
             "v_startup_linear[gen_d,p2020,t0025]": -1.0})}


# ---------------------------------------------------------------------------
# Stochastic: units online tied once per wholly tied coarse step; a partly tied step keeps the process fine

# stochastics.json with the coarse 'upper' reservoir of test_coarse_flows and a second daily reservoir 'top'
# feeding 'upper' through 'top_turbine' (linear online).  Neither node is in the stochastic group, so only the
# units-online tie (non_anticipativity_online_linear, every online process) covers top_turbine.
_TOP_ENTITIES = _STOCH_ENTITIES + [["node", "top", None], ["group__node", ["daily", "top"], None],
                                   ["unit", "top_turbine", None], ["unit__inputNode", ["top_turbine", "top"], None],
                                   ["unit__outputNode", ["top_turbine", "upper"], None]]


def _top_values(step: float) -> list:
    return _stoch_values(step) + [
        ["node", "top", "node_type", _s("storage")], ["node", "top", "existing", _f(3000.0)],
        ["node", "top", "inflow", _f(100.0)], ["node", "top", "storage_start_end_method", _s("fix_start")],
        ["node", "top", "storage_state_start", _f(0.5)],
        ["node", "top", "penalty_up", _f(5000.0)], ["node", "top", "penalty_down", _f(5000.0)],
        ["unit", "top_turbine", "efficiency", _f(1.0)], ["unit", "top_turbine", "existing", _f(3.0)],
        ["unit", "top_turbine", "virtual_unitsize", _f(100.0)], ["unit", "top_turbine", "startup_method", _s("linear")],
        ["unit", "top_turbine", "min_load", _f(0.2)], ["unit", "top_turbine", "startup_cost", _f(10.0)]]


def _top(tmp_path_factory, alt: str, scenario: str, step: float) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_top_values(step), base_chain=_chain(STOCH, scenario),
                        entities=_TOP_ENTITIES, fixture_json=STOCH)
    return run(tmp_path_factory, url, scen)


def _online_tie_reach(fd) -> list[tuple[str, str, str]]:
    """``(column, d, t)``: each units-online column of a non_anticipativity_online_linear row with a fine step it
    holds that is not tied (a coarse column holds every fine step of its coarse step)."""
    tie = utf8(fd.dt_non_anticipativity)
    tied = {(f, t) for a, fs in _forecasts(fd).items() for f in fs
            for t in tie.filter(pl.col("d") == a)["t"].to_list()}
    held: dict[tuple[str, str, str], list[str]] = {}
    if fd.process_coarse_step is not None:
        for p, d, b, t in utf8(fd.process_coarse_step).select("p", "d", "b_first", "t").iter_rows():
            held.setdefault((p, d, b), []).append(t)
    out = []
    for terms in row_terms(build(fd), "non_anticipativity_online_linear").values():
        for col in terms:
            p, d, t = _keys(col)
            out += [(col, d, s) for s in held.get((p, d, t), [t]) if (d, s) not in tied]
    return out


@pytest.mark.parametrize("step", [24.0, 48.0])
def test_stochastic_online_tie_stays_within_the_committed_steps(tmp_path_factory, step) -> None:
    """Rolling stochastic dispatch (96-h horizon, 24-h jump): each roll ties the units online of every online
    process between the forecasts over its 24 committed hours.  With 24-h coarse steps a committed day is one
    coarse step: top_turbine is coarse in every roll, tied by one row per forecast pair at the day's first hour,
    and every forecast cuts the tied hours into the same coarse steps as its reference forecast.  With 48-h steps a
    roll's first coarse step holds 24 committed and 24 uncommitted hours: top_turbine keeps the solve's step there
    (no tie row holds an uncommitted hour)."""
    steps = without_realization(_top(tmp_path_factory, f"co_sto_roll{int(step)}", "1_week_rolling_wind", step))
    rolls = [s.flex_data for s in steps.values() if _forecasts(s.flex_data)]
    assert len(rolls) >= 5
    coarse_in = []
    for fd in rolls:
        assert _online_tie_reach(fd) == []
        coarse_in.append("top_turbine" in _coarse_procs(fd))
    if step == 24.0:
        assert all(coarse_in)
        for fd in rolls:
            rows = [r for r in row_terms(build(fd), "non_anticipativity_online_linear") if "top_turbine" in r]
            pairs = sum(len(fs) - 1 for fs in _forecasts(fd).values())
            assert len(rows) == pairs, rows
            st = utf8(fd.process_coarse_step).filter(pl.col("p") == "top_turbine")
            tie_t = set(utf8(fd.dt_non_anticipativity)["t"].to_list())
            for fs in _forecasts(fd).values():
                cut = [st.filter((pl.col("d") == f) & pl.col("t").is_in(sorted(tie_t))).select("t", "b_first")
                       .sort("t").rows() for f in fs]
                assert all(c == cut[0] for c in cut)
    else:
        assert not coarse_in[0] and not all(coarse_in)


# ---------------------------------------------------------------------------
# Realization solve: units online pinned to the committed values


def test_pin_rule_keeps_a_partly_or_varying_pinned_process_fine() -> None:
    """flex_data.coarse_pin_varies: a coarse process whose units-online pin covers only part of one of its coarse
    steps, or changes within one, keeps the solve's step; one pinned to the same value on every hour of each coarse
    step (or not pinned) does not."""
    from types import SimpleNamespace

    from polar_high import Param

    from flextool.engine.preprocess.flex_data import coarse_pin_varies

    hours = [f"t{h}" for h in range(1, 7)]
    steps = pl.DataFrame({"p": [p for p in ("const", "part", "vary", "free") for _ in hours],
                          "d": ["d1"] * 24, "b_first": (["t1"] * 3 + ["t4"] * 3) * 4, "t": hours * 4})
    pins = pl.DataFrame({"p": ["const"] * 6 + ["part"] * 4 + ["vary"] * 6, "d": ["d1"] * 16,
                         "t": hours + hours[:4] + hours,
                         "value": [1.0] * 3 + [2.0] * 3 + [1.0] * 4 + [1.0, 1.0, 2.0, 2.0, 2.0, 2.0]})
    fd = SimpleNamespace(p_fix_online=Param(("p", "d", "t"), pins))
    assert coarse_pin_varies(fd, steps) == {"part", "vary"}
    assert coarse_pin_varies(SimpleNamespace(p_fix_online=None), steps) == set()


def _realized_branches() -> list:
    """2day_dispatch's ``stochastic_branches`` with branch ``realized`` flagged ``realized: yes``: the engine adds a
    realization solve that dispatches ``period1`` with the committed (tied) units online pinned."""
    branches = ("realized", "upper", "lower", "mid")
    return b64({"index_type": "str", "data": [
        ["period1", {"type": "map", "index_type": "str", "data": [
            [b, {"type": "map", "index_type": "str", "data": [
                ["t0001", {"type": "map", "index_type": "str",
                           "data": [["yes" if b == "realized" else "no", 1.0]]}]]}]
            for b in branches]}]]}, "map")


def test_realization_solve_pins_coarse_units_online_once_per_day(tmp_path_factory) -> None:
    """2-day stochastic dispatch with a realized designation (every hour tied): top_turbine is coarse in the
    stochastic solve, its committed units online are handed to the realization solve on every hour (the value of
    its day), so the pin is the same on every hour of each day and top_turbine is coarse there too, pinned by one
    online_fix_eq row per day; its realized units online equal the committed ones."""
    url, scen = make_db(tmp_path_factory, alt="co_pin",
                        values=_top_values(24.0) + [["solve", "2day_dispatch", "stochastic_branches",
                                                     _realized_branches()]],
                        base_chain=_chain(STOCH, "2_day_stochastic_dispatch"), entities=_TOP_ENTITIES,
                        fixture_json=STOCH)
    steps = run(tmp_path_factory, url, scen)
    real = [s for k, s in steps.items() if "__realized_dispatch" in k]
    assert len(real) == 1
    fd = real[0].flex_data
    assert "top_turbine" in _coarse_procs(fd)
    pin = utf8(fd.p_fix_online.frame).filter(pl.col("p") == "top_turbine").sort("t")
    assert pin["t"].to_list() == [f"t{h:04d}" for h in range(1, 49)]
    days = [pin["value"][:24].unique().to_list(), pin["value"][24:].unique().to_list()]
    assert all(len(v) == 1 for v in days)
    pb = build(fd)
    rows = row_terms(pb, "online_fix_eq_linear", "top_turbine,")
    assert sorted(_keys(r)[-1] for r in rows) == ["t0001", "t0025"]
    on = utf8(real[0].solution.value("v_online_linear")).filter(pl.col("p") == "top_turbine").sort("t")
    assert on["value"].to_list() == pytest.approx([days[0][0], days[1][0]], abs=1e-9)


# ---------------------------------------------------------------------------
# Outputs


def test_outputs_equal_tied_fine_model(tmp_path_factory, monkeypatch) -> None:
    """The hand-built model with min_uptime 30 h (unique optimum): every written primal output (output_raw v_* /
    vq_*: flows, units online, start-ups, shut-downs, slacks) of the coarse model equals that of the hourly model
    tied over each day — units online on every hour of the day, each start-up / shut-down at the day's first hour
    and 0 at its other hours (a copied start-up would count it 24 times in the start-up totals and costs)."""
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import assert_outputs_equal, output_frames

    folders = {}
    for kind in ("coarse", "tied"):
        alt = f"co_out_{kind}"
        if kind == "coarse":
            # The coarse run is test_hand_built_coarse_unit_commitment's (min_uptime 30): read only.
            steps, folders[kind], _scen = _RUNS.get(tmp_path_factory, alt=alt, values=_hand_uc_values(30.0),
                                                    base_chain=_HAND_CHAIN, entities=_HAND_ENTITIES)
        else:
            _tied_fine(monkeypatch)
            url, scen = make_db(tmp_path_factory, alt=alt, values=_hand_uc_values(30.0), base_chain=_HAND_CHAIN,
                                entities=_HAND_ENTITIES)
            folders[kind] = tmp_path_factory.mktemp(f"{alt}_w")
            steps = run_chain_from_db(url, scen, work_folder=folders[kind], keep_solutions=True)
        assert "gen_d" in _coarse_procs(last_stochastic(steps).flex_data)
    assert_outputs_equal(folders["coarse"], folders["tied"], "y2020_2day_dispatch")
    outs = output_frames(folders["coarse"], "y2020_2day_dispatch")
    assert {"v_online_linear", "v_startup_linear", "v_shutdown_linear"} <= set(outs)


def test_registry_has_explicit_online_coarse_families() -> None:
    """Every online <family>{sfx}_coarse resolves to its own registry entry with its fine family's type."""
    from flextool.engine.autoscale._layer2_types import CONSTRAINT_FAMILIES, lookup_cstr

    for fam in ("maxFlow_online", "maxFlow_back_online", "maxOutputSum_online", "minFlow_minload",
                "minFlow_output_floor"):
        for kind in ("linear", "integer"):
            name = f"{fam}_{kind}_coarse"
            assert lookup_cstr(name) is CONSTRAINT_FAMILIES[name]
            assert CONSTRAINT_FAMILIES[name] == lookup_cstr(f"{fam}_{kind}")


def test_notice_reasons_of_a_coarse_online_unit(tmp_path_factory) -> None:
    """The behaviour notice names a coarse online unit with the reasons its results can differ from the solve's
    step: units online, start-ups and shut-downs once per coarse step, and (with a minimum up / down time) the
    minimum times counted in whole coarse steps; a min load that varies within a day is named too.  Units without
    online variables keep their reasons (none here)."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    fd = last_stochastic(_lh2(tmp_path_factory, "co_reasons", [v for v in _UC_FULL if v[2] != "min_load"] + [
        ["unit", "liquefier_A", "min_load", _tmap_lh2(lambda h: 0.2 + 0.2 * (h % 2))]])).flex_data
    why = coarse_process_reasons(fd, fd.process_coarse_step)
    assert why["liquefier_A"] == ["its units online, start-ups and shut-downs are decided once per coarse step",
                                  "its minimum up / down time counts whole coarse steps",
                                  "its minimum load varies within a coarse step"]
    assert "pipe_AB" not in why


# ---------------------------------------------------------------------------
# Start-up / shut-down capacity reduction and the ramps of units online on the coarse step
#
# The hand-built model above with min_uptime 24 h and a ramp limit on gen_d -> hd, ramp_speed_up = ramp_speed_down
# = s per minute.  On a 24-h coarse step ρ_b = s·60·24 (the share of capacity the speed covers in the step) and a
# unit started at the step's start averages SU_b = ml + ½·ρ_b over the step while ρ_b ≤ 1 − ml (it has not reached
# full load), else 1 − (1 − ml)²/(2·ρ_b); SD_b likewise for a unit that stops at the step's end (EQ. headers of
# maxFlow_online and _eq_ramp).  ml = 0.5, availability 1.
# Rows (v_flow f_k in units of 100 MW): maxFlow_online_coarse day 1: f_1 ≤ on_1 − (1 − SU)·su_1 (red_sd − red_su = 0);
#   maxFlow_online_shutdown_coarse day 1 (successor day 2): f_1 ≤ on_1 − (1 − SD)·sd_2; day 2 has no successor (the
#   solve's last step: no shut-down reduction) and no start-up (below).  Ramp rows only between day 1 and day 2
#   (ρ̄ = s·60·(24 + 24)/2 < 1; day 1 has no predecessor): down f_2 − f_1 ≥ −(ρ̄·(on_2 − su_2) + SD·sd_2) + F·su_2,
#   up f_2 − f_1 ≤ ρ̄·(on_2 − su_2) + SU·su_2 − F·sd_2, F = ml = 0.5.
# Optimum as case (a): on_2 = 0 (a day-2 unit costs 1200·1020 € per unit of surplus, more than its start-up), so
#   su_1 = on_1 (the wrap), sd_2 = on_1, f_1 = 0.25 and the cheapest on_1 meets f_1 ≤ SU·on_1, f_1 ≤ SD·on_1 and the
#   down ramp −0.25 ≥ −SD·on_1, all three on_1 ≥ 0.25 / SU (SU = SD); the up ramp −0.25 ≤ −0.5·on_1 holds; the min
#   load 0.25 ≥ 0.5·on_1 holds for SU ≥ 0.5.
#   s = 1e-4: ρ_b = 0.144 ≤ 0.5: SU = 0.5 + 0.072 = 0.572, on_1 = 0.25 / 0.572 = 0.437062937…
#   s = 5e-4: ρ_b = 0.72 > 0.5 (full load after 16.7 h): SU = 1 − 0.25 / 1.44 = 0.826388…, on_1 = 0.302521008…

def _ramp_values(speed: float) -> list:
    rel = ["gen_d", "hd"]
    return [["unit__outputNode", rel, "ramp_method", _s("ramp_limit")],
            ["unit__outputNode", rel, "ramp_speed_up", _f(speed)],
            ["unit__outputNode", rel, "ramp_speed_down", _f(speed)]]


def _hand_ramp(tmp_path_factory, alt: str, speed: float, min_uptime: float = 24.0, extra=()):
    return _model(tmp_path_factory, alt=alt, values=_hand_uc_values(min_uptime) + _ramp_values(speed) + list(extra),
                  base_chain=_HAND_CHAIN, entities=_HAND_ENTITIES)


@pytest.mark.parametrize("speed, su_share", [(1e-4, 0.572), (5e-4, 1.0 - 0.25 / 1.44)])
def test_hand_built_coarse_startup_reduction(tmp_path_factory, speed, su_share) -> None:
    """The hand-built model with a ramp limit (derivation above): gen_d is coarse, its day-1 units online are
    0.25 / SU with SU the mean start-up share over the day (also once the ramp reaches full load within the day),
    started on day 1 and shut down at day 2's start; the reduction rows read the start-up of day 1 and the shut-down
    of day 2 with the coefficient 1 − SU, the down ramp of day 2 its shut-down with SD = SU."""
    step = last_stochastic(_hand_ramp(tmp_path_factory, f"co_red_{speed:g}", speed))
    assert "gen_d" in _coarse_procs(step.flex_data)
    on_1 = 0.25 / su_share
    assert _hourly(step, "v_online_linear") == pytest.approx([on_1] * 24 + [0.0] * 24, abs=1e-9)
    assert _hourly(step, "v_startup_linear") == pytest.approx([on_1] + [0.0] * 47, abs=1e-9)
    assert _hourly(step, "v_shutdown_linear") == pytest.approx([0.0] * 24 + [on_1] + [0.0] * 23, abs=1e-9)
    assert [f * 100.0 for f in _hourly(step, "v_flow")] == pytest.approx([25.0] * 24 + [0.0] * 24, abs=1e-7)
    pb = build(step.flex_data)
    arc = "gen_d,coal_market,hd,p2020,"
    red = row_terms(pb, "maxFlow_online_linear_coarse", arc)
    assert sorted(red) == [f"maxFlow_online_linear_coarse[{arc}t0001]", f"maxFlow_online_linear_coarse[{arc}t0025]"]
    assert red[f"maxFlow_online_linear_coarse[{arc}t0001]"]["v_startup_linear[gen_d,p2020,t0001]"] == pytest.approx(
        1.0 - su_share)
    sd_rows = row_terms(pb, "maxFlow_online_shutdown_linear_coarse", arc)
    assert list(sd_rows) == [f"maxFlow_online_shutdown_linear_coarse[{arc}t0001]"]
    assert sd_rows[f"maxFlow_online_shutdown_linear_coarse[{arc}t0001]"][
        "v_shutdown_linear[gen_d,p2020,t0025]"] == pytest.approx(1.0 - su_share)
    assert not row_terms(pb, "maxFlow_online_linear", "gen_d,")
    assert not row_terms(pb, "maxFlow_online_shutdown_linear", "gen_d,")
    down = row_terms(pb, "ramp_sink_down_constraint", "gen_d,")
    assert list(down) == [f"ramp_sink_down_constraint[{arc}t0025]"]
    assert abs(down[f"ramp_sink_down_constraint[{arc}t0025]"]["v_shutdown_linear[gen_d,p2020,t0025]"]) == \
        pytest.approx(su_share)


def _step_values(values: list) -> list:
    """``values`` of the hand-built model with the 24-h step on the solve (solve-level new_stepduration) instead of
    on the group of hd: every process and node on 24-h solve steps."""
    return [v for v in values if not (v[0] == "group" and v[2] == "new_stepduration")] + [
        ["solve", "y2020_2day_dispatch", "new_stepduration", _f(24.0)]]


_HAND_ROLLING = [["solve", "y2020_2day_dispatch", "solve_mode", _s("rolling_window")],
                 ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", _f(48.0)],
                 ["solve", "y2020_2day_dispatch", "rolling_solve_jump", _f(24.0)]]


@pytest.mark.parametrize("min_uptime, rolling", [(24.0, False), (30.0, False), (30.0, True)])
def test_coarse_ramp_reduction_equals_24h_solve(tmp_path_factory, min_uptime, rolling) -> None:
    """With the ramp short of full load within a day (s = 1e-4 / min), the coarse unit's start-up / shut-down
    reduction, online ramps and minimum up time are those of the same model solved on 24-h steps (the fine rule
    min(1, ml + ½·ρ) equals the mean share there): gen_d's units online, start-ups, shut-downs and flows are the
    same in every solve, also rolling (48-h horizon, 24-h jump: units online, the ramp's flow and the 30-h minimum up
    time carried into day 2's roll).  gen_d and hd are a separate part of the model: the rest of tests.json changes
    with the solve's step, so the objectives differ.  (Once the ramp reaches full load within the step the coarse
    step takes the exact mean share 1 − (1 − ml)²/(2ρ), the fine rule caps at 1.)"""
    tag = f"{int(min_uptime)}{'r' if rolling else ''}"
    extra = _HAND_ROLLING if rolling else []
    coarse = without_realization(_hand_ramp(tmp_path_factory, f"co_red24_{tag}", 1e-4, min_uptime, extra))
    url, scen = make_db(tmp_path_factory, alt=f"co_red24s_{tag}",
                        values=_step_values(_hand_uc_values(min_uptime) + _ramp_values(1e-4) + extra),
                        base_chain=_HAND_CHAIN, entities=_HAND_ENTITIES)
    daily = without_realization(run(tmp_path_factory, url, scen))
    assert len(coarse) == len(daily) == (2 if rolling else 1)
    started = 0.0
    for c, f in zip(coarse.values(), daily.values()):
        assert "gen_d" in _coarse_procs(c.flex_data) and not _coarse_procs(f.flex_data)
        for var in ("v_online_linear", "v_startup_linear", "v_shutdown_linear", "v_flow"):
            got = [utf8(s.solution.value(var)).filter(pl.col("p") == "gen_d").sort("t")["value"].to_list()
                   for s in (c, f)]
            assert len(got[0]) == len(got[1]) and got[0] == pytest.approx(got[1], abs=1e-9), var
        started += utf8(c.solution.value("v_startup_linear")).filter(pl.col("p") == "gen_d")["value"].sum()
    assert started > 1e-6


def _top_ramp(speed: float) -> list:
    rel = ["top_turbine", "upper"]
    return [["unit__outputNode", rel, "ramp_method", _s("ramp_limit")],
            ["unit__outputNode", rel, "ramp_speed_up", _f(speed)],
            ["unit__outputNode", rel, "ramp_speed_down", _f(speed)]]


def test_stochastic_coarse_unit_with_ramps(tmp_path_factory) -> None:
    """Rolling stochastic dispatch with 24-h coarse steps and a ramp limit on top_turbine (min_load 0.2, speed
    1e-4 / min: SU = 0.2 + ½·0.144 = 0.272 < 1, a start-up / shut-down reduction): top_turbine is coarse in every
    roll, tied once per committed day per forecast pair, with its capacity reduction once per coarse step of every
    forecast (maxFlow_online_linear_coarse) and no hourly online capacity row."""
    url, scen = make_db(tmp_path_factory, alt="co_sto_ramp", values=_top_values(24.0) + _top_ramp(1e-4),
                        base_chain=_chain(STOCH, "1_week_rolling_wind"), entities=_TOP_ENTITIES, fixture_json=STOCH)
    steps = without_realization(run(tmp_path_factory, url, scen))
    rolls = [s.flex_data for s in steps.values() if _forecasts(s.flex_data)]
    assert len(rolls) >= 5
    for fd in rolls:
        assert "top_turbine" in _coarse_procs(fd)
        assert _online_tie_reach(fd) == []
        pb = build(fd)
        ties = [r for r in row_terms(pb, "non_anticipativity_online_linear") if "top_turbine" in r]
        assert len(ties) == sum(len(fs) - 1 for fs in _forecasts(fd).values())
        steps_tt = utf8(fd.process_coarse_step).filter(pl.col("p") == "top_turbine").select("d", "b_first").unique()
        red = row_terms(pb, "maxFlow_online_linear_coarse", "top_turbine,")
        assert len(red) == steps_tt.height
        assert all(any(c.startswith("v_startup_linear[top_turbine,") for c in terms) for terms in red.values())
        assert not row_terms(pb, "maxFlow_online_linear", "top_turbine,")


# Mid-horizon stochastic fixture (stoch_two_period_midhorizon_nreal0.json: elec, period p2035 = t0001–t0003 and its
# branches p2040_high / _low = t0004–t0006, 1-h steps): elec in a 3-h group makes each period one coarse step, so the
# trunk's coarse step has one coarse successor per branch (a branch point).  base (invested, unitsize 1) gets linear
# online variables, min_load 0.4, a start-up cost and a ramp limit of 1e-3 / min on base -> elec:
# ρ_b = 1e-3·60·3 = 0.18 ≤ 0.6, SU = SD = 0.4 + 0.09 = 0.49, the value of the fine rule min(1, ml + ½·ρ) on a 3-h step.
_MID = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_midhorizon_nreal0.json"
_MID_UC = [["unit", "base", "startup_method", _s("linear")], ["unit", "base", "min_load", _f(0.4)],
           ["unit", "base", "startup_cost", _f(5.0)],
           ["unit__outputNode", ["base", "elec"], "ramp_method", _s("ramp_limit")],
           ["unit__outputNode", ["base", "elec"], "ramp_speed_up", _f(1e-3)],
           ["unit__outputNode", ["base", "elec"], "ramp_speed_down", _f(1e-3)]]


def test_coarse_branch_point_reduction_equals_3h_solve(tmp_path_factory) -> None:
    """At the branch point the shut-down reduction of the trunk's coarse step is one
    maxFlow_online_shutdown_branch_linear_coarse row per branch (each with that branch's shut-down), and the whole
    stochastic model equals the model solved on 3-h steps (solve-level new_stepduration): the same objective (every
    parameter of a period is averaged or summed over it in both)."""
    url, scen = make_db(tmp_path_factory, alt="co_mid", values=_MID_UC + [["group", "g3", "new_stepduration", _f(3.0)]],
                        base_chain=["init", "base", "stoch"], fixture_json=_MID,
                        entities=[["group", "g3", None], ["group__node", ["g3", "elec"], None]])
    coarse = last_stochastic(run(tmp_path_factory, url, scen))
    assert {"base", "wind"} <= _coarse_procs(coarse.flex_data)
    url, scen = make_db(tmp_path_factory, alt="co_mid3",
                        values=_MID_UC + [["solve", "stoch_2p_mid", "new_stepduration", _f(3.0)]],
                        base_chain=["init", "base", "stoch"], fixture_json=_MID)
    step3 = last_stochastic(run(tmp_path_factory, url, scen))
    assert not _coarse_procs(step3.flex_data)
    pb = build(coarse.flex_data)
    branch = row_terms(pb, "maxFlow_online_shutdown_branch_linear_coarse", "base,")
    assert len(branch) == 2
    sd_cols = sorted(c for terms in branch.values() for c in terms if c.startswith("v_shutdown_linear"))
    assert [c.split(",")[1] for c in sd_cols] == ["p2040_high", "p2040_low"]
    assert len(row_terms(build(step3.flex_data), "maxFlow_online_shutdown_branch_linear", "base,")) == 2
    assert float(coarse.obj) == pytest.approx(float(step3.obj), rel=1e-9)



# ---------------------------------------------------------------------------
# Reserves: one reserve per coarse step, the requirement checked at every solve step

def _reserve(r: str, ud: str, group: str, providers: list, *, reservation, penalty: float = 1000.0,
             duration: "float | None" = None) -> tuple[list, list]:
    """Entities and values of reserve ``r`` (timeseries method) for ``group`` with ``providers`` ``(class, p, n)``
    (max_share 1, reliability 1); ``reservation`` a float or a time-map value list."""
    grp = [r, ud, group]
    ents = [["upDown", ud, None], ["reserve", r, None], ["reserve__upDown__group", grp, None]]
    vals = [["reserve__upDown__group", grp, "reserve_method", _s("timeseries_only")],
            ["reserve__upDown__group", grp, "penalty_reserve", _f(penalty)],
            ["reserve__upDown__group", grp, "reservation",
             reservation if isinstance(reservation, list) else _f(reservation)]]
    if duration is not None:
        vals.append(["reserve__upDown__group", grp, "reserve_duration", _f(duration)])
    for cls, p, n in providers:
        rel = f"reserve__upDown__{cls}__node"
        ents.append([rel, [r, ud, p, n], None])
        vals += [[rel, [r, ud, p, n], "is_enabled", _s("yes")], [rel, [r, ud, p, n], "max_share", _f(1.0)],
                 [rel, [r, ud, p, n], "reliability", _f(1.0)]]
    return ents, vals


# lh2: upward reserve r_up for lh2_A / lh2_B by liquefier_A (at lh2_A) and pipe_AB (at lh2_B, with a 2-h reserve
# duration: pipe_AB draws on the storage lh2_A, reserve_storage_up_floor), downward reserve r_dn by both; 20 MW each,
# constant.  With liquefier_A's units online (_UC_FULL) and every parameter constant within a day.
_LH2_RES_GROUP = [["group", "lh2_res", None], ["group__node", ["lh2_res", "lh2_A"], None],
                  ["group__node", ["lh2_res", "lh2_B"], None]]


def _lh2_reserves() -> tuple[list, list]:
    providers = [("unit", "liquefier_A", "lh2_A"), ("connection", "pipe_AB", "lh2_B")]
    e_up, v_up = _reserve("r_up", "up", "lh2_res", providers, reservation=20.0, duration=2.0)
    e_dn, v_dn = _reserve("r_dn", "down", "lh2_res", providers, reservation=20.0)
    ents = _LH2_RES_GROUP + e_up + [e for e in e_dn if e[0] != "upDown" or e[1] != "up"]
    return ents, v_up + v_dn


def _lh2_res_case(tmp_path_factory, alt: str, case: str) -> dict:
    ents, vals = _lh2_reserves()
    if case == "rp":
        return _model(tmp_path_factory, alt=alt, values=_UC_FULL + vals, fixture_json=LH2_RP, entities=ents,
                      base_chain=_chain(LH2_RP, "lh2_three_region_rp_invest"))
    return _lh2(tmp_path_factory, alt, _UC_FULL + vals + (_ROLLING if case == "rolling" else []), entities=ents)


def test_coarse_reserve_structure(tmp_path_factory) -> None:
    """lh2 with reserves: liquefier_A and pipe_AB stay coarse; v_reserve has one column per day for each provider
    (p, r, ud, n); the per-process reserve limits are one averaged row per day (reserve_process_*_coarse,
    connection_reserve_*_coarse) while the requirement rows stay hourly (168 per reserve and group); the storage
    row of lh2_A (its state and pipe_AB's reserve both coarse) is one row per day."""
    fd = last_stochastic(_lh2_res_case(tmp_path_factory, "co_res_struct", "single")).flex_data
    assert _coarse_procs(fd) == LH2_COARSE
    pb = build(fd)
    for prefix in ("v_reserve[liquefier_A,r_up,up,lh2_A,", "v_reserve[pipe_AB,r_dn,down,lh2_B,"):
        assert [_keys(c)[-1] for c in _cols(pb, prefix)] == _DAYS, prefix
    assert len(row_terms(pb, "reserve_process_upward_coarse", "liquefier_A,")) == 7
    assert not row_terms(pb, "reserve_process_upward", "liquefier_A,")
    for fam in ("connection_reserve_up_coarse", "connection_reserve_down_coarse"):
        assert len(row_terms(pb, fam, "pipe_AB,")) == 7, fam
    for r in ("r_up,up,lh2_res,", "r_dn,down,lh2_res,"):
        assert len(row_terms(pb, "reserveBalance_timeseries_eq", r)) == 168, r
    assert sorted(_keys(r)[-1] for r in row_terms(pb, "reserve_storage_up_floor_coarse", "lh2_A,")) == _DAYS
    assert not row_terms(pb, "reserve_storage_up_floor", "lh2_A,")


@pytest.mark.parametrize("case", ["single", "rolling", "rp"])
def test_coarse_reserves_equal_tied_fine_model(tmp_path_factory, monkeypatch, case) -> None:
    """lh2 with reserves (above) and liquefier_A's units online: the coarse model has the objective of the hourly
    model with every coarse-process flow, units online and reserve tied over each day (start-ups 0 after a day's
    first hour), in a single, a rolling and a representative-period invest solve: averaged rows with parameters
    constant within a day are the fine rows, and the requirement is checked every hour in both."""
    coarse = _lh2_res_case(tmp_path_factory, f"co_res_tie_{case}", case)
    held = 0.0
    for s in without_realization(coarse).values():
        assert {"liquefier_A", "pipe_AB"} <= _coarse_procs(s.flex_data)
        held += float(s.solution.value("v_reserve")["value"].sum())
    assert held > 1e-6
    _tied_fine(monkeypatch)
    tied = _lh2_res_case(tmp_path_factory, f"co_res_tie_{case}_fine", case)
    assert len(coarse) == len(tied)
    assert _total_objective(coarse) == pytest.approx(_total_objective(tied), rel=1e-9)


# Hand-built reserve model (tests.json, 2 days of 1-h steps, period p2020): coal_market -> gen_d -> hd as above but
# without online variables: gen_d 50 MW existing, one unit of 50 MW (no virtual_unitsize: the unit size is the
# existing capacity; v in units of 50 MW), efficiency 1; hd
# demand 25 MW for the 24 hours of day 1, 0 on day 2.  Upward reserve r_up for the group hd_res = {hd} by gen_d
# (max_share 1, reliability 1): 10 MW in hours 1–12, 30 MW in hours 13–24, 60 MW on day 2; penalty_reserve 1000.
# gen_d is coarse: f_k and R_k once per day k.
# Rows: day 1 balance 24·50·f_1 = 600 MWh ⇒ f_1 = 0.5 (25 MW; more costs penalty_down + fuel, less penalty_up);
#   maxFlow_coarse f_1 + R_1 ≤ 1 ⇒ R_1 ≤ 25 MW; reserveBalance per hour 50·R_1 + shortfall_t ≥ requirement_t.
# (a) no ramp limit: the shortfall 30 − 50·R_1 of hours 13–24 falls with R_1 ⇒ R_1 = 25 MW (the cap), shortfall 0 in
#     hours 1–12 and 5 MW in hours 13–24 (an hourly flow could not avoid it either: hd is balanced per day, but one
#     reserve per day under one flow per day holds 25 MW).  Day 2: f_2 = 0 and the requirement 60 MW is above the
#     cap, so R_2 = 50 MW (unique: every output is determined).
# (b) upward ramp limit 0.2/60 per minute on gen_d -> hd: every fine row of day 1 has the room ρ_t = 0.2 (1-h
#     steps: (1 + 1)/2 h, and t0001 without a predecessor 1 h), so the one row of day 1 is R_1 ≤ 0.2 · 1 unit = 10 MW
#     (ramp_sink_up_reserve_coarse); no ramp row between the days (ρ̄ = 0.2·24 ≥ 1).  ⇒ R_1 = 10 MW, shortfall 0 in
#     hours 1–12 and 20 MW in hours 13–24.  The hourly model tied over each day has the same rows: R_1 ≤ 0.2 at
#     t0001 and inside day 1 (flow change 0), and at t0025 f_2 − f_1 + R_2 ≤ 0.2 with f_2 − f_1 = −0.5.  (Upward only:
#     a downward limit would hold the hourly flow above 0 in hour 25, which the coarse ramp between days does not.)

_RES_HAND_ENTITIES = [["node", "hd", None], ["unit", "gen_d", None], ["group", "daily", None],
                      ["unit__inputNode", ["gen_d", "coal_market"], None], ["unit__outputNode", ["gen_d", "hd"], None],
                      ["group__node", ["daily", "hd"], None], ["group", "hd_res", None],
                      ["group__node", ["hd_res", "hd"], None]]


def _hand_res(tmp_path_factory, alt: str, ramp: bool):
    return _model(tmp_path_factory, alt=alt, **_hand_res_model(ramp))


def _hand_res_db(tmp_path_factory, alt: str, ramp: bool) -> tuple[str, str]:
    return make_db(tmp_path_factory, alt=alt, **_hand_res_model(ramp))


def _hand_res_model(ramp: bool) -> dict:
    demand = b64({"index_type": "str", "index_name": "time",
                  "data": [[t, -25.0 if h < 24 else 0.0] for h, t in enumerate(_HOURS)]}, "map")
    req = b64({"index_type": "str", "index_name": "time",
               "data": [[t, (10.0 if h < 12 else 30.0) if h < 24 else 60.0] for h, t in enumerate(_HOURS)]}, "map")
    ents, res_vals = _reserve("r_up", "up", "hd_res", [("unit", "gen_d", "hd")], reservation=req)
    vals = [["group", "daily", "new_stepduration", _f(24.0)],
            ["node", "hd", "node_type", _s("balance")], ["node", "hd", "inflow", demand],
            ["node", "hd", "penalty_up", _f(10000.0)], ["node", "hd", "penalty_down", _f(1000.0)],
            ["unit", "gen_d", "conversion_method", _s("constant_efficiency")], ["unit", "gen_d", "efficiency", _f(1.0)],
            ["unit", "gen_d", "existing", _f(50.0)]] + res_vals
    if ramp:
        vals += [["unit__outputNode", ["gen_d", "hd"], "ramp_method", _s("ramp_limit")],
                 ["unit__outputNode", ["gen_d", "hd"], "ramp_speed_up", _f(0.2 / 60.0)]]
    return {"values": vals, "base_chain": _HAND_CHAIN, "entities": _RES_HAND_ENTITIES + ents}


def _hourly_of(step, var: str, **keys) -> list[float]:
    from flextool.engine.output._coarse_flows import var_values

    v = var_values(step.solution, step.flex_data, var)
    rows = utf8(v.drop("value")).with_columns(value=v["value"])
    for k, x in keys.items():
        rows = rows.filter(pl.col(k) == x)
    rows = rows.sort("t")
    assert rows["t"].to_list() == _HOURS, var
    return rows["value"].to_list()


@pytest.mark.parametrize("ramp, held", [(False, 25.0), (True, 10.0)])
def test_hand_built_coarse_reserve(tmp_path_factory, monkeypatch, ramp, held) -> None:
    """The hand-built reserve model (derivation above): gen_d holds one reserve per day, 25 MW on day 1 (10 MW with
    the ramp limit), written on every hour of the day; the hourly requirement leaves a shortfall of 30 MW − that
    in hours 13–24 and none in hours 1–12; the coarse model has the objective of the hourly model with gen_d's flow
    and reserve tied over each day."""
    from flextool.engine.model._reserve import reserve_shortfall_scale

    steps = _hand_res(tmp_path_factory, f"co_res_hand{int(ramp)}", ramp)
    step = last_stochastic(steps)
    fd = step.flex_data
    assert "gen_d" in _coarse_procs(fd)
    res = _hourly_of(step, "v_reserve", p="gen_d")
    assert [50.0 * v for v in res[:24]] == pytest.approx([held] * 24, abs=1e-7)
    assert [50.0 * f for f in _hourly(step, "v_flow")[:24]] == pytest.approx([25.0] * 24, abs=1e-7)
    keys = ["r", "ud", "g", "d", "t"]
    vq = utf8(step.solution.value("vq_reserve")).filter(pl.col("g") == "hd_res")
    scale = utf8(reserve_shortfall_scale(fd).frame).filter(pl.col("g") == "hd_res").rename({"value": "__s"})
    short = (vq.join(scale, on=[c for c in keys if c in scale.columns], how="inner")
             .with_columns(mw=pl.col("value") * pl.col("__s")).sort("t"))
    assert short["t"].to_list() == _HOURS
    assert short["mw"].to_list()[:24] == pytest.approx([0.0] * 12 + [30.0 - held] * 12, abs=1e-6)
    pb = build(fd)
    assert [_keys(c)[-1] for c in _cols(pb, "v_reserve[gen_d,")] == ["t0001", "t0025"]
    assert len(row_terms(pb, "reserveBalance_timeseries_eq", "r_up,up,hd_res,")) == 48
    room = row_terms(pb, "ramp_sink_up_reserve_coarse", "gen_d,")
    if ramp:
        assert sorted(room) == [f"ramp_sink_up_reserve_coarse[gen_d,coal_market,hd,p2020,{t}]"
                                for t in ("t0001", "t0025")]
        lo, hi = row_bounds(pb, "ramp_sink_up_reserve_coarse", "gen_d,coal_market,hd,p2020,t0001")[
            "ramp_sink_up_reserve_coarse[gen_d,coal_market,hd,p2020,t0001]"]
        assert hi == pytest.approx(0.2)
    else:
        assert not room
    _tied_fine(monkeypatch)
    tied = _hand_res(tmp_path_factory, f"co_res_hand{int(ramp)}_fine", ramp)
    assert _total_objective(steps) == pytest.approx(_total_objective(tied), rel=1e-9)


def test_reserve_ramp_room_is_the_smallest_room_times_capacity_factor(tmp_path_factory) -> None:
    """liquefier_A with a second output (lh2_B: an indirect unit, so its input arc h2_A -> liquefier_A is capped by
    p_indirect_input_cap) and an efficiency of 0.85 in hours 1–12 and 0.5 in hours 13–24 of each day holds
    downward reserve r_dn at its input h2_A, with an upward ramp limit of 0.1 / 60 per minute on the input arc.
    Hand derivation (EQ. ramp_source_up_reserve_coarse): 1-h steps, so every fine step's room is
    ρ_t = 0.1 · 60 · 1 / 60 = 0.1 (with or without a predecessor); the input arc's capacity factor is the full-load
    input per unit of capacity, c_t = input_share_max · 1 / efficiency_t = 1 / 0.85 in hours 1–12 and 2 in hours
    13–24; the one row per day bounds the reserve by min_t ρ_t · c_t = 0.1 / 0.85 = 0.117647 per unit (one unit:
    unitsize = existing).  Before: the smallest room times the day's mean factor, 0.1 · (1/0.85 + 2) / 2 = 0.158824,
    looser than every hour of the tied model in hours 1–12."""
    ents, vals = _reserve("r_dn", "down", "h2_res", [("unit", "liquefier_A", "h2_A")], reservation=500.0)
    ents = [["group", "h2_res", None], ["group__node", ["h2_res", "h2_A"], None],
            ["unit__outputNode", ["liquefier_A", "lh2_B"], None]] + ents
    vals += [["unit", "liquefier_A", "efficiency", _tmap_lh2(lambda h: 0.85 if h % 24 < 12 else 0.5)],
             ["unit__inputNode", ["liquefier_A", "h2_A"], "ramp_method", _s("ramp_limit")],
             ["unit__inputNode", ["liquefier_A", "h2_A"], "ramp_speed_up", _f(0.1 / 60.0)]]
    fd = last_stochastic(_lh2(tmp_path_factory, "co_res_room", vals, entities=ents)).flex_data
    assert "liquefier_A" in _coarse_procs(fd)
    pb = build(fd)
    bounds = row_bounds(pb, "ramp_source_up_reserve_coarse", "liquefier_A,")
    assert sorted(_keys(k)[-1] for k in bounds) == _DAYS
    assert all(hi == pytest.approx(0.1 / 0.85, rel=1e-12) for _, hi in bounds.values())
    terms = row_terms(pb, "ramp_source_up_reserve_coarse", "liquefier_A,h2_A,liquefier_A,y2030,t0025")
    assert terms == {"ramp_source_up_reserve_coarse[liquefier_A,h2_A,liquefier_A,y2030,t0025]":
                     {"v_reserve[liquefier_A,r_dn,down,h2_A,y2030,t0025]": pytest.approx(1.0)}}


# Stochastic: top_turbine (above) also holds upward reserve r_up for the group upper_res = {upper} (10 MW).
def _top_reserve() -> tuple[list, list]:
    ents, vals = _reserve("r_up", "up", "upper_res", [("unit", "top_turbine", "upper")], reservation=10.0)
    return [["group", "upper_res", None], ["group__node", ["upper_res", "upper"], None]] + ents, vals


def _reserve_tie_reach(fd) -> list[tuple[str, str, str]]:
    """``(column, d, t)``: each reserve column of a non_anticipativity_reserve row with a fine step it holds that is
    not tied."""
    tie = utf8(fd.dt_non_anticipativity)
    tied = {(f, t) for a, fs in _forecasts(fd).items() for f in fs
            for t in tie.filter(pl.col("d") == a)["t"].to_list()}
    held: dict[tuple[str, str, str], list[str]] = {}
    if fd.process_coarse_step is not None:
        for p, d, b, t in utf8(fd.process_coarse_step).select("p", "d", "b_first", "t").iter_rows():
            held.setdefault((p, d, b), []).append(t)
    out = []
    for terms in row_terms(build(fd), "non_anticipativity_reserve").values():
        for col in terms:
            k = _keys(col)
            p, d, t = k[0], k[-2], k[-1]
            out += [(col, d, s) for s in held.get((p, d, t), [t]) if (d, s) not in tied]
    return out


@pytest.mark.parametrize("step", [24.0, 48.0])
def test_stochastic_reserve_tie_stays_within_the_committed_steps(tmp_path_factory, step) -> None:
    """Rolling stochastic dispatch (96-h horizon, 24-h jump) with top_turbine holding reserve: each roll ties the
    reserve between the forecasts over its 24 committed hours.  24-h coarse steps: top_turbine is coarse in every
    roll, tied by one row per forecast pair at the day's first hour.  48-h steps: a roll's first coarse step holds
    committed and uncommitted hours, so top_turbine keeps the solve's step there; no tie row holds an uncommitted
    hour."""
    ents, vals = _top_reserve()
    url, scen = make_db(tmp_path_factory, alt=f"co_sto_res{int(step)}", values=_top_values(step) + vals,
                        base_chain=_chain(STOCH, "1_week_rolling_wind"), entities=_TOP_ENTITIES + ents,
                        fixture_json=STOCH)
    steps = without_realization(run(tmp_path_factory, url, scen))
    rolls = [s.flex_data for s in steps.values() if _forecasts(s.flex_data)]
    assert len(rolls) >= 5
    coarse_in = []
    for fd in rolls:
        assert _reserve_tie_reach(fd) == []
        coarse_in.append("top_turbine" in _coarse_procs(fd))
    if step == 24.0:
        assert all(coarse_in)
        for fd in rolls:
            rows = [r for r in row_terms(build(fd), "non_anticipativity_reserve") if "top_turbine" in r]
            assert len(rows) == sum(len(fs) - 1 for fs in _forecasts(fd).values()), rows
    else:
        assert not coarse_in[0] and not all(coarse_in)


def test_realization_solve_pins_coarse_reserve_once_per_day(tmp_path_factory) -> None:
    """2-day stochastic dispatch with a realized designation (every hour tied) and top_turbine holding reserve: the
    committed reserve is handed to the realization solve on every hour (the value of its day), so the pin is the
    same on every hour of each day, top_turbine is coarse in the realization solve and pinned by one reserve_fix_eq
    row per day; its realized reserve equals the committed one."""
    ents, vals = _top_reserve()
    url, scen = make_db(tmp_path_factory, alt="co_pin_res",
                        values=_top_values(24.0) + vals + [["solve", "2day_dispatch", "stochastic_branches",
                                                            _realized_branches()]],
                        base_chain=_chain(STOCH, "2_day_stochastic_dispatch"), entities=_TOP_ENTITIES + ents,
                        fixture_json=STOCH)
    steps = run(tmp_path_factory, url, scen)
    real = [s for k, s in steps.items() if "__realized_dispatch" in k]
    assert len(real) == 1
    fd = real[0].flex_data
    assert "top_turbine" in _coarse_procs(fd)
    pin = utf8(fd.p_fix_reserve.frame).filter(pl.col("p") == "top_turbine").sort("t")
    assert pin["t"].to_list() == [f"t{h:04d}" for h in range(1, 49)]
    days = [pin["value"][:24].unique().to_list(), pin["value"][24:].unique().to_list()]
    assert all(len(v) == 1 for v in days)
    rows = row_terms(build(fd), "reserve_fix_eq", "top_turbine,")
    assert sorted(_keys(r)[-1] for r in rows) == ["t0001", "t0025"]
    res = utf8(real[0].solution.value("v_reserve")).filter(pl.col("p") == "top_turbine").sort("t")
    assert res["value"].to_list() == pytest.approx([days[0][0], days[1][0]], abs=1e-9)


def test_pin_rule_covers_reserve_pins() -> None:
    """flex_data.coarse_pin_varies: a coarse process whose reserve pin (p_fix_reserve) covers only part of one of its
    coarse steps, or changes within one, keeps the solve's step, per (p, r, ud, n); one pinned to the same value on
    every hour of each coarse step does not."""
    from types import SimpleNamespace

    from polar_high import Param

    from flextool.engine.preprocess.flex_data import coarse_pin_varies

    hours = [f"t{h}" for h in range(1, 7)]
    steps = pl.DataFrame({"p": [p for p in ("const", "part", "vary") for _ in hours], "d": ["d1"] * 18,
                          "b_first": (["t1"] * 3 + ["t4"] * 3) * 3, "t": hours * 3})
    rows = ([("const", "n1", t, 1.0 if i < 3 else 2.0) for i, t in enumerate(hours)]
            + [("const", "n2", t, 3.0) for t in hours]
            + [("part", "n1", t, 1.0) for t in hours[:4]]
            + [("vary", "n1", t, 1.0) for t in hours] + [("vary", "n2", t, float(i)) for i, t in enumerate(hours)])
    pins = pl.DataFrame({"p": [r[0] for r in rows], "r": ["r_up"] * len(rows), "ud": ["up"] * len(rows),
                         "n": [r[1] for r in rows], "d": ["d1"] * len(rows), "t": [r[2] for r in rows],
                         "value": [r[3] for r in rows]})
    fd = SimpleNamespace(p_fix_online=None, p_fix_reserve=Param(("p", "r", "ud", "n", "d", "t"), pins))
    assert coarse_pin_varies(fd, steps) == {"part", "vary"}



def test_reserve_outputs_equal_tied_fine_model(tmp_path_factory, monkeypatch) -> None:
    """The hand-built reserve model (unique optimum): every written primal output (output_raw v_* / vq_*: flows,
    reserves, shortfalls) of the coarse model equals that of the hourly model tied over each day — the reserve on
    every hour of its day (25 MW on day 1, 50 MW on day 2)."""
    from flextool.engine import run_chain_from_db
    from tests.engine._unit_db import assert_outputs_equal, output_frames

    folders = {}
    for kind in ("coarse", "tied"):
        if kind == "coarse":
            # The coarse run is test_hand_built_coarse_reserve's (no ramp): read only.
            steps, folders[kind], _scen = _RUNS.get(tmp_path_factory, alt=f"co_res_out_{kind}",
                                                    **_hand_res_model(False))
        else:
            _tied_fine(monkeypatch)
            url, scen = _hand_res_db(tmp_path_factory, f"co_res_out_{kind}", False)
            folders[kind] = tmp_path_factory.mktemp(f"co_res_out_{kind}_w")
            steps = run_chain_from_db(url, scen, work_folder=folders[kind], keep_solutions=True)
        assert "gen_d" in _coarse_procs(last_stochastic(steps).flex_data)
    assert_outputs_equal(folders["coarse"], folders["tied"], "y2020_2day_dispatch")
    outs = output_frames(folders["coarse"], "y2020_2day_dispatch")
    assert "v_reserve" in outs



def test_notice_reasons_of_a_coarse_reserve_provider(tmp_path_factory) -> None:
    """The behaviour notice names a coarse reserve provider with "one reserve per coarse step", and, when its
    availability varies within a day, the step-mean headroom; pipe_AB (a provider with a constant availability)
    gets only the first."""
    from flextool.engine.preprocess.flex_data import coarse_process_reasons

    ents, vals = _lh2_reserves()
    fd = last_stochastic(_lh2(tmp_path_factory, "co_res_reasons", _UC_FULL + vals + [
        ["unit", "liquefier_A", "availability", _tmap_lh2(lambda h: 0.8 + 0.2 * (h % 2))]], entities=ents)).flex_data
    why = coarse_process_reasons(fd, fd.process_coarse_step)
    one = "it holds one reserve per coarse step (the requirement is checked at every time step)"
    assert one in why["liquefier_A"]
    assert "its reserve headroom uses the step-mean availability / profile" in why["liquefier_A"]
    assert why["pipe_AB"] == [one]


# ---------------------------------------------------------------------------
# Constant vs equivalent map: min_load and availability of a coarse online reserve provider


def _tmap_const(value: float, hours: int) -> list:
    """A ``time`` map over ``hours`` hourly steps t0001 … with the same ``value`` at every step."""
    return b64({"index_type": "str", "index_name": "time",
                "data": [[f"t{h + 1:04d}", value] for h in range(hours)]}, "map")


def _same_lp(a, b) -> None:
    """The canonical LPs of FlexData ``a`` and ``b`` are identical: names, matrix, bounds, costs, integrality."""
    ma, mb = build(a).canonicalise(), build(b).canonicalise()
    assert list(ma.row_names) == list(mb.row_names)
    assert list(ma.col_names) == list(mb.col_names)
    for field in ("col_ptr", "row_idx", "val", "row_lb", "row_ub", "sense_char", "col_obj", "col_lb", "col_ub",
                  "col_int"):
        assert np.array_equal(getattr(ma, field), getattr(mb, field)), field


def _shaped_case(tmp_path_factory, case: str, shape: str) -> tuple[dict, str]:
    """``case`` single / rolling (lh2: liquefier_A with linear units online, a 30-h minimum up / down time and
    reserves r_up / r_dn) or stochastic (stochastics.json rolling stochastic dispatch, 24-h coarse steps:
    top_turbine with linear units online holding reserve r_up), with the provider's min_load and availability
    given as constants or (``shape`` "map") as time maps holding the same value at every step.  Returns the run's
    steps and the provider."""
    alt = f"co_shape_{case}_{shape}"
    if case == "stochastic":
        unit, hours, min_load, avail = "top_turbine", 8760, 0.2, 0.9
        ents, vals = _top_reserve()
        base = [v for v in _top_values(24.0) + vals if not (v[1] == unit and v[2] == "min_load")]
        chain, fixture, entities = _chain(STOCH, "1_week_rolling_wind"), STOCH, _TOP_ENTITIES + ents
    else:
        unit, hours, min_load, avail = "liquefier_A", 168, 0.3, 0.9
        ents, vals = _lh2_reserves()
        base = [v for v in _UC_FULL + vals if not (v[1] == unit and v[2] == "min_load")]
        base += _ROLLING if case == "rolling" else []
        chain, fixture, entities = _chain(LH2, "lh2_three_region"), LH2, ents
    shaped = [["unit", unit, "min_load", _f(min_load) if shape == "constant" else _tmap_const(min_load, hours)],
              ["unit", unit, "availability", _f(avail) if shape == "constant" else _tmap_const(avail, hours)]]
    url, scen = make_db(tmp_path_factory, alt=alt, values=base + shaped, base_chain=chain, entities=entities,
                        fixture_json=fixture)
    return run(tmp_path_factory, url, scen), unit


@pytest.mark.parametrize("case", ["single", "rolling", "stochastic"])
def test_constant_and_equivalent_map_give_the_same_coarse_lp(tmp_path_factory, case) -> None:
    """min_load and availability of a coarse unit with units online that holds reserve, given as constants or as
    time maps with the same value at every step: every solve of the run (each roll; each stochastic roll and its
    realization solve) builds the same LP — the averaged online-capacity, minimum-load and reserve rows read the
    step mean of a map that equals the constant."""
    const, unit = _shaped_case(tmp_path_factory, case, "constant")
    mapped, _ = _shaped_case(tmp_path_factory, case, "map")
    assert list(const) == list(mapped)
    assert len(const) >= (1 if case == "single" else 3)
    for name in const:
        a, b = const[name].flex_data, mapped[name].flex_data
        if not is_realization(name):
            assert unit in _coarse_procs(a) and unit in _coarse_procs(b), name
        _same_lp(a, b)
