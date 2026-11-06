"""``invest_cost`` / ``salvage_value`` shapes in the ed_invest / ed_divest predicate.

``unit/node/connection.invest_cost`` and ``.salvage_value`` are "constant or period"
(docs/reference.md).  The predicate that decides which ``(entity, period)`` pairs get an
invest / divest variable (``_per_entity_period_cost``) must read them through the shape
resolver (CLAUDE.md invariant #2):

* a period map on one entity class next to a constant on another (or within one class)
  must not crash (it raised a polars schema mismatch in ``pl.concat``);
* a period map authored with Spine's silent ``x`` index must keep its per-period values
  (it was cross-joined, so a 0 in one period was ignored);
* map keys outside the solve's invest periods must not create invest variables (in a
  multi-solve model every solve got variables for every map period);
* under recourse each anchor period's value is inherited by its branch periods, and a
  branch's own explicit 0 wins over the anchor's value (the period label itself has no
  invest variable under recourse).

Every DB is built from a JSON fixture plus an override alternative (never a checked-in
``.sqlite``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"
BASE = "network_coal_wind_battery_invest_cumulative"
MULTI = "wind_battery_invest_lifetime_renew_4solve"
PERIODS = ["p2020", "p2025", "p2030", "p2035"]


def _lp_snapshot():
    """``tools/lp_snapshot.py`` (fixture overlay helpers), loaded by path."""
    name = "_lp_snapshot_for_tests"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "lp_snapshot.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _pmap(values: dict[str, float], index_name: str = "period") -> dict:
    return _lp_snapshot()._pmap(values, index_name)


def _solve(tmp_path_factory, fixture: str, patch) -> dict:
    """Build the DB from ``fixture`` + ``patch(data) -> (data, scenario)`` and run it."""
    from flextool.engine import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data, scenario = patch(json.loads((FIXTURES / fixture).read_text()))
    work = tmp_path_factory.mktemp("invest_cost_shapes")
    json_path = work / "fixture.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / "fixture.sqlite")
    migrate_database(url)
    steps = run_chain_from_db(url, scenario, work_folder=work / "work", keep_solutions=True)
    assert steps, "run_chain_from_db produced no orchestration steps"
    for step in steps.values():
        assert step.solution is not None, "solve produced no solution"
    return steps


def _overlay(values: list, base: str = BASE):
    def patch(data):
        return _lp_snapshot()._overlay(data, name="custom", base=base, values=values), "custom"
    return patch


def _pairs(frame: pl.DataFrame | None) -> set[tuple[str, str]]:
    if frame is None or frame.height == 0:
        return set()
    return set(frame.select(pl.col("e").cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).iter_rows())


def _periods_of(pairs: set[tuple[str, str]], entity: str) -> set[str]:
    return {d for e, d in pairs if e == entity}


def _single(steps: dict):
    assert len(steps) == 1, f"expected a single solve, got {list(steps)}"
    return next(iter(steps.values()))


def _invest_domain(step) -> set[tuple[str, str]]:
    """``(entity, period)`` columns of the solved ``v_invest_p`` / ``v_invest_n``."""
    out: set[tuple[str, str]] = set()
    for var, col in (("v_invest_p", "p"), ("v_invest_n", "n")):
        try:
            frame = step.solution.value(var)
        except (KeyError, ValueError):
            continue
        if frame is None or frame.height == 0:
            continue
        out |= set(frame.select(pl.col(col).cast(pl.Utf8), pl.col("d").cast(pl.Utf8)).iter_rows())
    return out


@pytest.fixture(scope="module")
def control_invest(tmp_path_factory) -> set[tuple[str, str]]:
    """All-scalar control: the scenario as authored."""
    step = _single(_solve(tmp_path_factory, "tests.json", _overlay([])))
    pairs = _pairs(step.flex_data.ed_invest_set)
    assert _periods_of(pairs, "coal_plant") == set(PERIODS)
    return pairs


# ── mixed shapes: no crash, same set as the all-scalar control ─────────────


@pytest.mark.parametrize("values", [
    pytest.param([["unit", "coal_plant", "invest_cost", 1000.0],
                  ["node", "battery", "invest_cost", _pmap({d: 500.0 for d in PERIODS})]],
                 id="scalar_unit_plus_period_map_node"),
    pytest.param([["unit", "coal_plant", "invest_cost", _pmap({d: 1000.0 for d in PERIODS})],
                  ["unit", "wind_plant", "invest_cost", 900.0]],
                 id="period_map_and_scalar_within_unit_class"),
    pytest.param([["unit", "coal_plant", "invest_cost", _pmap({d: 1000.0 for d in PERIODS}, "x")],
                  ["node", "battery", "invest_cost", _pmap({d: 500.0 for d in PERIODS})]],
                 id="x_map_unit_plus_period_map_node"),
])
def test_mixed_invest_cost_shapes_solve(tmp_path_factory, control_invest, values):
    step = _single(_solve(tmp_path_factory, "tests.json", _overlay(values)))
    pairs = _pairs(step.flex_data.ed_invest_set)
    assert pairs == control_invest
    # coal_plant has no capacity-constraint coefficient, so its periods come from the cost.
    assert _periods_of(pairs, "coal_plant") == set(PERIODS)


# ── per-period zero: the pair is dropped, also for a silent-'x' map ─────────


@pytest.mark.parametrize("index_name", ["period", "x"])
def test_invest_cost_zero_in_one_period(tmp_path_factory, index_name):
    cost = _pmap({"p2020": 1000.0, "p2025": 0.0, "p2030": 1000.0, "p2035": 1000.0}, index_name)
    step = _single(_solve(tmp_path_factory, "tests.json",
                          _overlay([["unit", "coal_plant", "invest_cost", cost]])))
    pairs = _pairs(step.flex_data.ed_invest_set)
    assert _periods_of(pairs, "coal_plant") == {"p2020", "p2030", "p2035"}
    assert _periods_of(_invest_domain(step), "coal_plant") == {"p2020", "p2030", "p2035"}


# ── salvage_value (divest path): mixed shapes, per-period zero ──────────────


def test_salvage_value_mixed_shapes(tmp_path_factory):
    values = [
        ["unit", "coal_plant", "invest_method", "invest_retire_total"],
        ["unit", "coal_plant", "salvage_value",
         _pmap({"p2020": 10.0, "p2025": 0.0, "p2030": 10.0, "p2035": 10.0})],
        ["unit", "wind_plant", "invest_method", "invest_retire_total"],
        ["unit", "wind_plant", "salvage_value", 5.0],
    ]
    step = _single(_solve(tmp_path_factory, "tests.json", _overlay(values)))
    divest = _pairs(step.flex_data.ed_divest_set)
    assert _periods_of(divest, "coal_plant") == {"p2020", "p2030", "p2035"}
    # wind_plant has no capacity-constraint coefficient, so its divest periods come from the
    # constant salvage_value alone (a broken scalar broadcast would leave it with none).
    assert _periods_of(divest, "wind_plant") == set(PERIODS)


# ── multi-solve: map keys outside a solve's invest periods are dropped ──────


def _per_solve_invest(steps: dict) -> dict[str, set[tuple[str, str]]]:
    return {name: _invest_domain(step) for name, step in steps.items()}


def test_period_map_restricted_to_solve_invest_periods(tmp_path_factory):
    baseline = _per_solve_invest(_solve(tmp_path_factory, "tests.json", _overlay([], base=MULTI)))
    maps = [
        ["unit", "wind_plant", "invest_cost", _pmap({d: 1000.0 for d in PERIODS})],
        ["node", "battery", "invest_cost", _pmap({d: 200.0 for d in PERIODS})],
        ["connection", "battery_inverter", "invest_cost", _pmap({d: 0.0 for d in PERIODS})],
    ]
    mapped = _per_solve_invest(_solve(tmp_path_factory, "tests.json", _overlay(maps, base=MULTI)))
    assert set(mapped) == set(baseline)
    assert len(baseline) > 1
    for solve, pairs in baseline.items():
        for entity in ("wind_plant", "battery"):
            assert _periods_of(pairs, entity), f"{solve}: baseline has no {entity} invest"
            assert _periods_of(mapped[solve], entity) == _periods_of(pairs, entity), solve
    first = min(baseline)
    assert _periods_of(mapped[first], "wind_plant") == {"p2020", "p2025"}


# ── recourse: anchor values reach the branch periods; branch-native 0 wins ──


def _stoch_store(cost_map: dict[str, float]):
    def patch(data):
        lps = _lp_snapshot()
        data = lps._patch_stoch_invest_storage(data)
        for pv in data["parameter_values"]:
            if pv[0] == "node" and pv[1] == "store" and pv[2] == "invest_cost":
                pv[3] = lps._pack(lps._pmap(cost_map))
        data["scenarios"].append(["custom", False, ""])
        for row in [r for r in data["scenario_alternatives"] if r[0] == "stoch_invest_storage"]:
            data["scenario_alternatives"].append(["custom", row[1], row[2]])
        return data, "custom"
    return patch


# Under recourse the period labels p2035 / p2040 have no LP column: only the forecast branches invest.
STOCH_PERIODS = {"p2035_high", "p2035_low", "p2040_high", "p2040_low"}


def test_recourse_period_map_reaches_branch_periods(tmp_path_factory):
    steps = _solve(tmp_path_factory, "stoch_two_period_hedge.json",
                   _stoch_store({"p2035": 1.0, "p2040": 1.0}))
    pairs = _pairs(_single(steps).flex_data.ed_invest_set)
    assert _periods_of(pairs, "store") == STOCH_PERIODS


def test_recourse_branch_native_zero_wins_over_anchor(tmp_path_factory):
    steps = _solve(tmp_path_factory, "stoch_two_period_hedge.json",
                   _stoch_store({"p2035": 1.0, "p2040": 1.0, "p2040_high": 0.0}))
    pairs = _pairs(_single(steps).flex_data.ed_invest_set)
    assert _periods_of(pairs, "store") == STOCH_PERIODS - {"p2040_high"}
