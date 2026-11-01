"""The realization solve of a stochastic fan (owner decision "Stochastic realization model (2026-11-01)").

A stochastic solve optimises over forecasts only.  Its realized operations come from a realization solve that runs
right after it over the same committed window, with its investments fixed and its non-anticipativity-tied unit
commitment and reserves inherited (the value of the first sorted forecast, on the tied cells).  A ``realized: yes``
designation whose fan reaches the committed window makes the engine add that solve when none is configured.

Fixtures (built from JSON, CLAUDE.md invariant 3): ``stoch_t0_commit.json`` — a single solve over ``p2035`` (3 hourly
steps), branches ``rlz`` (realized: yes, weight 0), ``high`` / ``low`` (0.5 each) revealed at t0001; a committable
``peaker`` (2 units of 50 MW, min_load 0.5) backs ``wind`` (100 MW, CF rlz 0.5 / high 0.8 / low 0.0) against 100 MW
demand; it authors ``dispatch_realization_method = separate_dispatch``, which the variants here drop.
``w4b_reserve_dispatch.json`` — the same structure with an up-reserve of 10 MW on the peaker.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
T0_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_t0_commit.json"
RESERVE_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "w4b_reserve_dispatch.json"

PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"
ANCHOR = "p2035"
FIRST_FORECAST = "p2035_high"       # sorted forecasts of p2035: p2035_high < p2035_low
STEPS = ("t0001", "t0002", "t0003")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _enc(value) -> list:
    return [base64.b64encode(json.dumps(value).encode()).decode(), "str"]


def _variant(src: Path, tmp: Path, name: str, mutate) -> Path:
    data = json.loads(src.read_text())
    mutate(data)
    out = tmp / f"{name}.json"
    out.write_text(json.dumps(data))
    return out


def _drop_separate_dispatch(data: dict) -> None:
    data["parameter_values"] = [r for r in data["parameter_values"] if r[2] != "dispatch_realization_method"]


def _set_profile(data: dict, values: dict) -> None:
    """Set the ``wprof`` CF per branch (constant over the steps)."""
    for r in data["parameter_values"]:
        if r[1] == "wprof" and r[2] == "profile":
            m = json.loads(base64.b64decode(r[3][0]).decode())
            for branch in m["data"]:
                for ts in branch[1]["data"]:
                    for cell in ts[1]["data"]:
                        cell[1] = values[branch[0]]
            r[3] = [base64.b64encode(json.dumps(m).encode()).decode(), r[3][1]]


def _hard_node(data: dict) -> None:
    data["parameter_values"].append(["node", "elec", "penalty_method", _enc("off"), "base"])


def _run(fixture_json: Path, tmp: Path, *, scenario: str = "stoch_t0_commit", capture_mps: bool = False):
    """Run the chain (FLEXTOOL_AUTOSCALE_STRICT=1, warm=False); return ``(steps, {solve: mps bytes})``."""
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.engine import run_chain_from_db
    from flextool.engine.model import core as model_core
    from flextool.update_flextool.db_migration import migrate_database

    url = json_to_db(fixture_json, tmp / f"{fixture_json.stem}.sqlite")
    migrate_database(url)
    work = tmp / "work"
    work.mkdir()
    mps: dict[str, bytes] = {}
    original = model_core.build_flextool

    def capturing(m, d, *args, **kwargs):
        result = original(m, d, *args, **kwargs)
        frame = sys._getframe(1)
        while frame is not None and not isinstance(frame.f_locals.get("complete_solve_name"), str):
            frame = frame.f_back
        name = frame.f_locals["complete_solve_name"] if frame is not None else "solve"
        path = tmp / f"{name}.mps"
        m.write_mps(path, emit_names=True)
        mps[name] = path.read_bytes()
        return result

    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    if capture_mps:
        model_core.build_flextool = capturing
    try:
        steps = run_chain_from_db(url, scenario, work_folder=work, warm=False, keep_solutions=True)
    finally:
        model_core.build_flextool = original
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    return steps, mps


def _value(steps, solve: str, var: str) -> pl.DataFrame:
    assert solve in steps, f"no {solve!r} step in {list(steps)}"
    sol = steps[solve].solution
    assert var in sol._vars, f"{var} not in {solve}: {sorted(sol._vars)}"
    return sol.value(var).with_columns(pl.col(c).cast(pl.Utf8) for c in sol._vars[var].dims)


def _at(frame: pl.DataFrame, **keys) -> float:
    row = frame.filter(pl.all_horizontal(*(pl.col(k) == v for k, v in keys.items())))
    assert row.height == 1, (keys, row)
    return float(row["value"][0])


@pytest.fixture(scope="module")
def designated(tmp_path_factory):
    """The designated single solve WITHOUT dispatch_realization_method (the engine adds the realization solve)."""
    tmp = tmp_path_factory.mktemp("realization_designated")
    return _run(_variant(T0_FIXTURE, tmp, "designated", _drop_separate_dispatch), tmp, capture_mps=True)


@pytest.fixture(scope="module")
def separate(tmp_path_factory):
    """The same solve WITH dispatch_realization_method = separate_dispatch."""
    tmp = tmp_path_factory.mktemp("realization_separate")
    return _run(T0_FIXTURE, tmp, capture_mps=True)


# ---------------------------------------------------------------------------
# auto-spawn
# ---------------------------------------------------------------------------

@pytest.mark.solver
def test_designation_adds_the_realization_solve(designated) -> None:
    """A realized: yes designation reaching the committed window adds ``<solve>__realized_dispatch`` right after
    the stochastic solve, though no dispatch_realization_method is authored."""
    steps, _ = designated
    assert list(steps) == [PARENT, SIBLING]


@pytest.mark.solver
def test_auto_spawned_realization_solve_equals_separate_dispatch(designated, separate) -> None:
    """The auto-spawned realization solve builds the same LP as the one separate_dispatch adds, and the stochastic
    solve's LP does not depend on which of the two added it."""
    _, mps_auto = designated
    _, mps_sep = separate
    assert set(mps_auto) == set(mps_sep) == {PARENT, SIBLING}
    assert mps_auto[SIBLING] == mps_sep[SIBLING]
    assert mps_auto[PARENT] == mps_sep[PARENT]


@pytest.mark.solver
def test_realization_online_is_first_sorted_forecast(designated) -> None:
    """The realization solve's units online equal the stochastic solve's first sorted forecast (``p2035_high``) on
    the tied cells (the committed window p2035 t0001-t0003), re-keyed to the anchor period."""
    steps, _ = designated
    parent = _value(steps, PARENT, "v_online_linear")
    sib = _value(steps, SIBLING, "v_online_linear")
    for t in STEPS:
        assert _at(sib, p="peaker", d=ANCHOR, t=t) == pytest.approx(
            _at(parent, p="peaker", d=FIRST_FORECAST, t=t), abs=1e-9)


@pytest.mark.solver
def test_realization_reserve_is_first_sorted_forecast(tmp_path_factory) -> None:
    """The realization solve's reserve is pinned (reserve_fix_eq) to the stochastic solve's first sorted forecast on
    the tied cells: v_reserve of the realization solve == v_reserve of ``p2035_high`` at every step."""
    tmp = tmp_path_factory.mktemp("realization_reserve")
    steps, mps = _run(RESERVE_FIXTURE, tmp, scenario="w8_realized_dispatch", capture_mps=True)
    assert b"reserve_fix_eq" in mps[SIBLING]
    parent = _value(steps, PARENT, "v_reserve")
    sib = _value(steps, SIBLING, "v_reserve")
    for t in STEPS:
        assert _at(sib, p="peaker", d=ANCHOR, t=t) == pytest.approx(
            _at(parent, p="peaker", d=FIRST_FORECAST, t=t), abs=1e-9)


# ---------------------------------------------------------------------------
# feasibility pre-check on a hard-balance node
# ---------------------------------------------------------------------------

def _hard_variant(cf: dict):
    def mutate(data: dict) -> None:
        _drop_separate_dispatch(data)
        _hard_node(data)
        _set_profile(data, cf)
    return mutate


@pytest.mark.solver
def test_hard_node_realized_shortfall_raises_before_solve(tmp_path_factory) -> None:
    """``elec`` has no balance slack.  Forecast CF high 0.8 / low 0.6: high needs 20 MW of peaker, so the min_load
    floor 0.5 · 50 · v_online ≤ 20 keeps v_online ≤ 0.8; low needs 40 MW, so v_online ≥ 0.8 — the tied commitment is
    0.8 units = 40 MW.  Realized CF 0.0: the realized need is 100 MWh per step against at most 40 (peaker) + 0 (wind)
    MWh, so the realization solve raises, naming the node and the first step, before it is solved."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    tmp = tmp_path_factory.mktemp("realization_hard_short")
    jf = _variant(T0_FIXTURE, tmp, "hard_short", _hard_variant({"rlz": 0.0, "high": 0.8, "low": 0.6}))
    with pytest.raises(FlexToolConfigError, match=r"cannot balance node 'elec'.*'p2035', step 't0001'"):
        _run(jf, tmp)


@pytest.mark.solver
def test_hard_node_realized_within_pin_solves(tmp_path_factory) -> None:
    """Same commitment (0.8 units = 40 MW); realized CF 0.7: wind ≤ 70 MW and the peaker 20..40 MW cover the 100 MW
    demand (peaker 30 MW), so the check passes and the realization solve reaches an optimum."""
    tmp = tmp_path_factory.mktemp("realization_hard_ok")
    jf = _variant(T0_FIXTURE, tmp, "hard_ok", _hard_variant({"rlz": 0.7, "high": 0.8, "low": 0.6}))
    steps, _ = _run(jf, tmp)
    assert SIBLING in steps and steps[SIBLING].solution is not None
    sib = _value(steps, SIBLING, "v_online_linear")
    assert _at(sib, p="peaker", d=ANCHOR, t="t0001") == pytest.approx(0.8, abs=1e-9)


# ---------------------------------------------------------------------------
# registration rules (config level, no solve)
# ---------------------------------------------------------------------------

class _Reader:
    def __init__(self):
        self.registered: list[tuple[str, str]] = []

    def entities(self, _cls):
        return pl.DataFrame({"name": ["p"]})

    def register_derived_solve(self, parent, sibling):
        self.registered.append((parent, sibling))


def _config(**over):
    from collections import defaultdict
    from flextool.engine.input._solve_config import SolveConfig
    cfg = SimpleNamespace(
        model_solve={"m": ["p"]}, contains_solves={}, solve_modes={"p": "single_solve"}, rolling_times={},
        stochastic_branches={"p": [["d1", "rlz", "t0001", "yes", 0.0], ["d1", "a", "t0001", "no", 1.0]]},
        realized_periods=defaultdict(list, {"p": [("d1", "d1")]}),
        timesets_used_by_solves={"p": [("d1", "ts")]}, stochastic_invest_method={},
        dispatch_realization_method={}, fix_storage_periods={}, realized_dispatch_parent={},
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    for name in ("dispatch_realization_method_for", "stochastic_invest_method_for", "register_realization_solve",
                 "spawn_realized_dispatch_sibling"):
        setattr(cfg, name, getattr(SolveConfig, name).__get__(cfg))
    return cfg


def test_configured_realization_solve_is_registered_not_spawned() -> None:
    """A deterministic solve right after the stochastic solve, same mode and rolls, realizing its committed fan
    period, is registered as its realization solve; no sibling is added."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(model_solve={"m": ["p", "q"]}, solve_modes={"p": "single_solve", "q": "single_solve"})
    cfg.realized_periods["q"] = [("d1", "d1")]
    reader = _Reader()
    _spawn_realized_dispatch_siblings(cfg, reader, logger=None)
    assert cfg.realized_dispatch_parent == {"q": "p"}
    assert reader.registered == []
    assert cfg.model_solve["m"] == ["p", "q"]


def test_configured_solve_missing_the_fan_period_is_not_registered() -> None:
    """A following deterministic solve that does not realize the fan's committed period is not a realization
    solve: the designation then adds the engine sibling right after the stochastic solve."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(model_solve={"m": ["p", "q"]}, solve_modes={"p": "single_solve", "q": "single_solve"})
    cfg.realized_periods["q"] = [("d2", "d2")]
    reader = _Reader()
    _spawn_realized_dispatch_siblings(cfg, reader, logger=None)
    assert cfg.realized_dispatch_parent == {"p__realized_dispatch": "p"}
    assert cfg.model_solve["m"] == ["p", "p__realized_dispatch", "q"]


def test_recourse_designation_warns_and_adds_nothing(caplog) -> None:
    """Under recourse there is no realized post-reveal investment to fix: a warning, no realization solve."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(stochastic_invest_method={"p": "recourse"})
    with caplog.at_level(logging.WARNING, logger="flextool.engine.solve._realization"):
        _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)
    assert cfg.realized_dispatch_parent == {}
    assert any("recourse" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("nesting", ["parent", "child"])
def test_nested_designation_raises(nesting: str) -> None:
    """A designated solve that contains solves, or is nested in another solve, cannot get a realization solve."""
    from flextool.engine.input._solve_state import FlexToolConfigError
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    if nesting == "parent":
        cfg = _config(contains_solves={"p": ["c"]})
    else:
        cfg = _config(model_solve={"m": ["top"]}, contains_solves={"top": ["p"]})
    with pytest.raises(FlexToolConfigError, match="realized: yes designation"):
        _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)


def test_designation_outside_the_committed_window_adds_nothing() -> None:
    """A designated fan opened at a period after every realized period does not reach the committed window."""
    from flextool.engine.solve._orchestration import _spawn_realized_dispatch_siblings
    cfg = _config(stochastic_branches={"p": [["d2", "rlz", "t0001", "yes", 0.0], ["d2", "a", "t0001", "no", 1.0]]},
                  timesets_used_by_solves={"p": [("d1", "ts"), ("d2", "ts")]})
    _spawn_realized_dispatch_siblings(cfg, _Reader(), logger=None)
    assert cfg.realized_dispatch_parent == {}
