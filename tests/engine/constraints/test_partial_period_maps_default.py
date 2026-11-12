"""A period map that names only some periods: the other periods take the parameter's default (item 79, pinning).

``availability`` (default 1) and ``efficiency`` (default 1) given as a period map ``{d: v}`` for one of two
periods must give the default in the other period -- not 0, not the authored value spread over both periods --
in a single solve, in rolls, in nested sub-solves and on a stochastic branch period (which inherits its
anchor's value).  Audited without a defect (``_common._availability_factor``, ``_derived`` efficiency,
``PdtLookup`` DEF1); these tests pin it.

Vehicle (``tests/fixtures/tests.json`` + one override alternative on ``init, west, coal``; independent oracle
``specs/bugfix/v1/oracle.md`` item 79): periods p2020 and p2025, each on the 1-step timeset ``one_hour``
(t0001, 1 h); ``years_represented`` 5 each, ``inflation_rate`` 0, timestep weight 1.  Node west: inflow
-100 MW, penalty_up 1e4.  coal_market price 10 CUR/MWh, gas_market price 50.  coal_plant 100 MW from
coal_market; gas_plant 100 MW from gas_market, efficiency 1.  Operating-cost weight of a period::

    w_op[d] = step_duration * timestep_weight * years_represented / period_share = 1 * 1 * 5 / (1 / 8760) = 43 800

    case A  coal availability {p2020: 0.8}: p2020 coal 80 + gas 20 = 800 + 1000; p2025 coal 100 = 1000
            objective 43 800 * 2800 = 122 640 000
    case B  coal efficiency {p2025: 0.5}:   p2020 slope 1, coal 100 = 1000; p2025 slope 2, coal 100 = 2000
            objective 43 800 * 3000 = 131 400 000
    case C  case A + coal2 (100 MW, coal_market, efficiency 0.8, no availability): p2020 coal 80 (800) + coal2 20
            (fuel 25 MWh, 250); p2025 coal 100 (1000)
            objective 43 800 * 2050 = 89 790 000
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import RunCache, b64, is_realization, utf8

W_OP = 43_800.0
COST = {"A": {"p2020": 1800.0, "p2025": 1000.0},
        "B": {"p2020": 1000.0, "p2025": 2000.0},
        "C": {"p2020": 1050.0, "p2025": 1000.0}}
OBJECTIVE = {"A": 122_640_000.0, "B": 131_400_000.0, "C": 89_790_000.0}
PERIODS = ["p2020", "p2025"]
SOLVE = "y2020_2day_dispatch"
CHAIN = ["init", "west", "coal"]

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _num(v: float) -> list:
    return b64(float(v), "float")


def _str(v: str) -> list:
    return b64(v, "str")


def _pmap(values: dict) -> list:
    return b64({"type": "map", "index_type": "str", "index_name": "period",
                "data": [[k, v] for k, v in values.items()]}, "map")


def _arr(items: list[str]) -> list:
    return b64({"type": "array", "value_type": "str", "data": items}, "array")


ENTITIES = [
    ["timeset", "one_hour", None],
    ["commodity", "gas", None],
    ["node", "gas_market", None],
    ["commodity__node", ["gas", "gas_market"], None],
    ["unit", "gas_plant", None],
    ["unit__inputNode", ["gas_plant", "gas_market"], None],
    ["unit__outputNode", ["gas_plant", "west"], None],
]
COAL2 = [
    ["unit", "coal2", None],
    ["unit__inputNode", ["coal2", "coal_market"], None],
    ["unit__outputNode", ["coal2", "west"], None],
]


def _base_values() -> list:
    return [
        ["model", "flexTool", "inflation_rate", _num(0.0)],
        ["timeset", "one_hour", "timeline", _str("y2020")],
        ["timeset", "one_hour", "timeset_duration", b64({"type": "map", "index_type": "str",
                                                         "data": [["t0001", 1.0]]}, "map")],
        ["solve", SOLVE, "period_timeset", _pmap({p: "one_hour" for p in PERIODS})],
        ["solve", SOLVE, "realized_periods", _arr(PERIODS)],
        ["solve", SOLVE, "years_represented", _pmap({p: 5.0 for p in PERIODS})],
        ["node", "west", "inflow", _num(-100.0)],
        ["node", "west", "penalty_up", _num(1e4)],
        ["node", "gas_market", "node_type", _str("commodity")],
        ["commodity", "coal", "price", _num(10.0)],
        ["commodity", "gas", "price", _num(50.0)],
        ["unit", "coal_plant", "existing", _num(100.0)],
        ["unit", "gas_plant", "existing", _num(100.0)],
        ["unit", "gas_plant", "efficiency", _num(1.0)],
    ]


def _case_values(case: str) -> tuple[list, list]:
    """``(values, entities)`` of case A / B / C."""
    values, entities = _base_values(), list(ENTITIES)
    if case in ("A", "C"):
        values += [["unit", "coal_plant", "availability", _pmap({"p2020": 0.8})],
                   ["unit", "coal_plant", "efficiency", _num(1.0)]]
    if case == "B":
        values += [["unit", "coal_plant", "efficiency", _pmap({"p2025": 0.5})]]
    if case == "C":
        entities += COAL2
        values += [["unit", "coal2", "existing", _num(100.0)], ["unit", "coal2", "efficiency", _num(0.8)]]
    return values, entities


def _mode_values(mode: str) -> list:
    if mode == "single":
        return []
    if mode == "rolling":
        return [["solve", SOLVE, "solve_mode", _str("rolling_window")],
                ["solve", SOLVE, "rolling_solve_horizon", _num(1.0)],
                ["solve", SOLVE, "rolling_solve_jump", _num(1.0)]]
    if mode == "nested":
        # A 2-D map keyed by the sub-solve: one sub-solve per period, each holding its own period only.
        per_sub = b64({"type": "map", "index_type": "str", "data": [
            [p, {"type": "map", "index_type": "str", "data": [[p, "yes"]]}] for p in PERIODS]}, "map")
        return [["solve", SOLVE, "invest_periods", per_sub],
                ["solve", SOLVE, "realized_invest_periods", per_sub],
                ["solve", SOLVE, "realized_periods", per_sub]]
    if mode == "stochastic":
        branch = {"type": "map", "index_type": "str", "data": [["b", {
            "type": "map", "index_type": "str", "data": [["t0001", {
                "type": "map", "index_type": "str", "data": [["no", 1.0]]}]]}]]}
        return [["solve", SOLVE, "stochastic_branches",
                 b64({"type": "map", "index_type": "str", "data": [["p2025", branch]]}, "map")]]
    raise ValueError(mode)


def _steps(tmp_path_factory, case: str, mode: str) -> dict:
    values, entities = _case_values(case)
    mode_values = _mode_values(mode)
    replaced = {tuple(map(str, v[:3])) for v in mode_values}
    values = [v for v in values if tuple(map(str, v[:3])) not in replaced] + mode_values
    steps, _wf, _scen = _RUNS.get(tmp_path_factory, alt=f"c79_{case}_{mode}", values=values,
                                  base_chain=CHAIN, entities=entities)
    return {k: s for k, s in steps.items() if not is_realization(k)}


def _cells(step, field: str, unit: str) -> dict[str, float]:
    """``{d: value}`` of *field* for *unit* on the step's ``dt`` (one time step per period), as the LP uses it.

    ``p_process_availability`` is sparse (authored rows only): the LP reads it through
    ``_common._availability_factor``, which fills the schema default 1 for an entity or period without a row (and
    no availability Param at all is a factor of 1 everywhere).  ``p_slope`` is dense."""
    from flextool.engine.input._param_shapes import promote_param_to_dt
    from flextool.engine.model._common import _availability_factor

    fd = step.flex_data
    grid = utf8(fd.dt).select("d", "t").unique()
    if field == "p_process_availability":
        if fd.p_process_availability is None:
            return {d: 1.0 for d in grid["d"]}
        param = _availability_factor(fd, pl.DataFrame({"p": [unit]}))
    else:
        param = getattr(fd, field)
    f = promote_param_to_dt(param, fd.dt)
    f = utf8(f.collect() if isinstance(f, pl.LazyFrame) else f).filter(pl.col("p") == unit)
    f = f.join(grid, on=["d", "t"], how="inner")
    out: dict[str, float] = {}
    for d, v in f.select("d", pl.col("value").cast(pl.Float64)).iter_rows():
        assert out.get(d, v) == v, (field, unit, d)
        out[d] = round(v, 12)
    assert set(out) == set(grid["d"]), (field, unit, out)
    return out


#: (case, field, unit, {d: expected}) -- the period without a value takes the default 1.
PARAMS = [
    ("A", "p_process_availability", "coal_plant", {"p2020": 0.8, "p2025": 1.0}),
    ("B", "p_slope", "coal_plant", {"p2020": 1.0, "p2025": 2.0}),
    ("C", "p_process_availability", "coal_plant", {"p2020": 0.8, "p2025": 1.0}),
    ("C", "p_process_availability", "coal2", {"p2020": 1.0, "p2025": 1.0}),
]


def _assert_params(steps: dict, case: str, *, branch_period: str | None = None) -> None:
    """Every step's cells of the case's parameters equal the oracle's for the periods of that step; a branch
    period takes its anchor's value."""
    seen: set[str] = set()
    for name, step in steps.items():
        for c, field, unit, expected in PARAMS:
            if c != case:
                continue
            cells = _cells(step, field, unit)
            assert cells, (name, field, unit)
            for d, v in cells.items():
                anchor = "p2025" if d == branch_period else d
                assert v == pytest.approx(expected[anchor], rel=1e-12), (name, field, unit, d)
                seen.add(d)
    assert seen, steps


@pytest.mark.parametrize("case", ["A", "B", "C"])
def test_single_solve(tmp_path_factory, case: str) -> None:
    steps = _steps(tmp_path_factory, case, "single")
    assert list(steps) == [SOLVE]
    (step,) = steps.values()
    _assert_params(steps, case)
    assert sorted(set(utf8(step.flex_data.dt)["d"])) == PERIODS
    assert step.solution.obj == pytest.approx(OBJECTIVE[case], rel=1e-9)


@pytest.mark.parametrize("case", ["A", "B", "C"])
def test_rolling(tmp_path_factory, case: str) -> None:
    """Rolls of one step (one per period): each roll's parameters are the single solve's for its (d, t), its
    objective is 43 800 * C_d, and the roll objectives add up to the single solve's."""
    steps = _steps(tmp_path_factory, case, "rolling")
    assert len(steps) > 1 and all("_roll_" in k for k in steps), list(steps)
    _assert_params(steps, case)
    covered = sorted({d for s in steps.values() for d in utf8(s.flex_data.dt)["d"]})
    assert covered == PERIODS
    for name, step in steps.items():
        (d,) = set(utf8(step.flex_data.dt)["d"])  # a one-step roll holds one period
        assert step.solution.obj == pytest.approx(W_OP * COST[case][d], rel=1e-9), (name, d)
    assert sum(s.solution.obj for s in steps.values()) == pytest.approx(OBJECTIVE[case], rel=1e-9)


@pytest.mark.parametrize("case", ["A", "B", "C"])
def test_nested_sub_solves(tmp_path_factory, case: str) -> None:
    """One sub-solve per invest period; sub-solve objective = 43 800 * C_d."""
    steps = _steps(tmp_path_factory, case, "nested")
    assert list(steps) == [f"{SOLVE}_{p}" for p in PERIODS], list(steps)
    _assert_params(steps, case)
    for p in PERIODS:
        assert steps[f"{SOLVE}_{p}"].solution.obj == pytest.approx(W_OP * COST[case][p], rel=1e-9), p


@pytest.mark.parametrize("case", ["A", "B", "C"])
def test_stochastic_branch_inherits_the_partial_value(tmp_path_factory, case: str) -> None:
    """One forecast b of p2025 (weight 1): operations of p2025 run on p2025_b, which takes p2025's value -- the
    default where the map names only p2020, the authored value where it names p2025.  Objective = single."""
    steps = _steps(tmp_path_factory, case, "stochastic")
    (step,) = steps.values()
    periods = set(utf8(step.flex_data.dt)["d"])
    assert "p2025_b" in periods, periods
    _assert_params(steps, case, branch_period="p2025_b")
    assert step.solution.obj == pytest.approx(OBJECTIVE[case], rel=1e-9)
