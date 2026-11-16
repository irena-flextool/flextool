"""``lifetime_technical``, ``payback_time``, ``invest_cost``, ``net_salvage_value`` and ``discount_rate`` of an entity:
a constant, a ``period`` map and a map with the unnamed ``x`` index give the same LP when they give every period the
same value.

Fixtures (built from JSON):

* single: ``tests.json``, the four periods p2020 … p2035 (run years 0, 5, 10, 15) in one solve (``SINGLE``);
  coal_plant has 500 MW existing, ``lifetime_method = reinvest_choice`` and may invest and retire
  (``invest_retire_no_limit``), so its technical lifetime sets the end of life of the existing capacity and the
  windows of the investment and retirement annuities, and its payback time their annuity factors.
* rolling: ``tests.json`` ``dispatch_fullYear_roll`` with a 48 h horizon and a 36 h jump (8 rolls over
  p2020 … p2035) and coal_plant's 500 MW existing (``reinvest_choice``, no investment): the end of life in every
  roll.
* stochastic: ``stoch_two_period_invest.json`` ``recourse`` (periods p2035, p2040, a fan of branches); peaker
  invests and retires: the branch periods inherit their anchor's value.
* single_payback_mid / stochastic_payback_mid: the single and stochastic cases with a payback time that ends inside
  a period (12 on 5-year periods; 0.5 on the one-year p2035), so the annuity is paid for part of that period.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import (
    FLEXTOOL_ROOT,
    PLANT,
    RunCache,
    b64,
    build,
    utf8,
)
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_constraint_constant_period_map import ROLL_LONG_JUMP
from tests.engine.test_user_constraint_capacity_terms import SINGLE

PERIODS = ["p2020", "p2025", "p2030", "p2035"]
ROLL_CHAIN = ["init", "west", "coal", "battery", "wind", "fullYear", "dispatch_fullYear_roll"]
STOCH_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_invest.json"
STOCH_CHAIN = ["init", "base", "stoch", "recourse"]
STOCH_PERIODS = ["p2035", "p2040"]
PEAKER = "peaker"

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _map(values: dict[str, float], index_name: str) -> list:
    return b64({"index_type": "str", "index_name": index_name, "data": [[k, v] for k, v in values.items()]}, "map")


def _shaped(value: float, shape: str, periods: list[str]) -> list:
    """``value`` as a constant (``shape`` "constant") or as a map over ``periods`` with the index ``shape``."""
    if shape == "constant":
        return b64(value, "float")
    return _map(dict.fromkeys(periods, value), shape)


def _lp(fd) -> tuple[dict, dict]:
    """``({row: [lb, ub, {col: coef}]}, {col: (lb, ub, obj)})`` of the LP built from ``fd``."""
    m = build(fd).canonicalise()
    rows = {nm: [float(m.row_lb[i]), float(m.row_ub[i]), {}] for i, nm in enumerate(m.row_names)}
    for c in range(m.n_cols):
        for k in range(m.col_ptr[c], m.col_ptr[c + 1]):
            rows[m.row_names[m.row_idx[k]]][2][m.col_names[c]] = float(m.val[k])
    cols = {m.col_names[c]: (float(m.col_lb[c]), float(m.col_ub[c]), float(m.col_obj[c])) for c in range(m.n_cols)}
    return rows, cols


# The constant of each parameter in each case; a map variant gives every period this value.
SINGLE_BASE = {"lifetime_technical": 10.0, "payback_time": 10.0, "discount_rate": 0.05, "invest_cost": 400.0,
               "net_salvage_value": 50.0}
STOCH_BASE = {"lifetime_technical": 1.0, "payback_time": 1.0, "discount_rate": 0.05, "invest_cost": 1.0,
              "net_salvage_value": 0.1}
ROLL_BASE = {"lifetime_technical": 10.0, "payback_time": 10.0}
SINGLE_MID = {**SINGLE_BASE, "payback_time": 12.0}
STOCH_MID = {**STOCH_BASE, "payback_time": 0.5}

CASES = {
    # case: (entity, base values, extra fixed values, chain, periods, fixture json)
    "single": (PLANT, SINGLE_BASE,
               [["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
                ["unit", PLANT, "invest_method", b64("invest_retire_no_limit", "str")]],
               _no_battery_invest(SINGLE), PERIODS, None),
    "rolling": (PLANT, ROLL_BASE,
                ROLL_LONG_JUMP + [["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")]],
                ROLL_CHAIN, PERIODS, None),
    "stochastic": (PEAKER, STOCH_BASE,
                   [["unit", PEAKER, "lifetime_method", b64("reinvest_choice", "str")],
                    ["unit", PEAKER, "invest_method", b64("invest_retire_no_limit", "str")]],
                   STOCH_CHAIN, STOCH_PERIODS, STOCH_JSON),
}
CASES["single_payback_mid"] = (PLANT, SINGLE_MID) + CASES["single"][2:]
CASES["stochastic_payback_mid"] = (PEAKER, STOCH_MID) + CASES["stochastic"][2:]


def _steps(tmp_path_factory, case: str, shaped: str | None = None, shape: str = "constant") -> dict:
    """The run of ``case`` with every parameter of its base a constant, except ``shaped`` given as ``shape``."""
    entity, base, extra, chain, periods, fixture = CASES[case]
    values = list(extra) + [["unit", entity, name, _shaped(v, shape if name == shaped else "constant", periods)]
                            for name, v in base.items()]
    kwargs = {} if fixture is None else {"fixture_json": fixture}
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"life_{case}_{shaped}_{shape}", values=values,
                            base_chain=chain, **kwargs)
    return steps


@pytest.mark.solver
@pytest.mark.parametrize(("case", "parameter"), [
    ("single", "lifetime_technical"), ("single", "payback_time"), ("single", "discount_rate"),
    ("single", "invest_cost"), ("single", "net_salvage_value"),
    ("rolling", "lifetime_technical"), ("rolling", "payback_time"),
    ("stochastic", "lifetime_technical"), ("stochastic", "payback_time"), ("stochastic", "discount_rate"),
    ("stochastic", "invest_cost"), ("stochastic", "net_salvage_value"),
    ("single_payback_mid", "payback_time"), ("stochastic_payback_mid", "payback_time"),
])
@pytest.mark.parametrize("shape", ["period", "x"])
def test_lifetime_constant_equals_maps(tmp_path_factory, case, parameter, shape) -> None:
    """A map (``period`` or unnamed ``x`` index) with the constant's value in every period builds, in every
    solve / roll / branch, the LP of the constant: rows, coefficients, bounds and objective coefficients."""
    constant = _steps(tmp_path_factory, case)
    mapped = _steps(tmp_path_factory, case, parameter, shape)
    assert list(mapped) == list(constant) and constant
    if case == "rolling":
        assert len(constant) == 8
    for name in constant:
        assert _lp(mapped[name].flex_data) == _lp(constant[name].flex_data), name


def _all_existing(step, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(step.flex_data.p_entity_all_existing.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


NO_INVEST = [["unit", PLANT, "invest_method", b64("not_allowed", "str")],
             ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")]]


@pytest.mark.solver
def test_unnamed_lifetime_map_reads_each_period(tmp_path_factory) -> None:
    """A lifetime map with the unnamed ``x`` index {10, 10, 10, 40} is read like the same ``period`` map: the end
    of life of the existing 500 MW is yr[p2020] + lifetime[p2020] = 0 + 10 = 10, so the capacity is 500 MW in
    p2020 and p2025 (yr 0, 5 < 10) and 0 in p2030 and p2035 (yr 10, 15 ≥ 10).  (Before: the ``x`` map was read
    as its last value, 40, and the capacity never retired.)"""
    lifetimes = dict(zip(PERIODS, (10.0, 10.0, 10.0, 40.0)))
    runs = {}
    for shape in ("period", "x"):
        steps, _, _ = _RUNS.get(tmp_path_factory, alt=f"life_eol_{shape}",
                                values=NO_INVEST + [["unit", PLANT, "lifetime_technical", _map(lifetimes, shape)]],
                                base_chain=_no_battery_invest(SINGLE))
        runs[shape] = list(steps.values())[-1]
    want = {"p2020": 500.0, "p2025": 500.0, "p2030": 0.0, "p2035": 0.0}
    for shape, step in runs.items():
        got = _all_existing(step)
        assert {"p2020", "p2025"} <= set(got), shape
        assert {d: got.get(d, 0.0) for d in want} == want, shape
    assert _lp(runs["x"].flex_data) == _lp(runs["period"].flex_data)

