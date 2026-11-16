"""A period-varying ``existing`` in every solve of a run: chained, rolling, nested and stochastic.

Each solve's existing capacity is the existing cohort with the user's ``existing`` of each of its periods, plus the
investments earlier solves realized (``realized_cohort``), minus the carried retirements.  Before the cohort
handoff, a later solve of a chain (and every roll after the first) took the map's first-period value for every
period: the falling map {500, 300, 100, 0} gave 500 MW in each later solve.

Built from ``tests/fixtures/tests.json`` (``coal_plant``, virtual_unitsize unset: MW; periods p2020 … p2035, five
years each; ``lifetime_method`` reinvest_automatic, so the existing capacity and every investment stay in service to
the end of the horizon).  Expected values, by hand:

* coal not investable → ``p_entity_all_existing[coal, d] = map[d]`` in every solve that holds d;
* coal investable → ``map[d] + Σ`` (the coal investments realized by the earlier solves, decided in a period
  ``≤ d``: ``v_invest · unitsize``, read from the solved run);
* the p2035 flow of coal is 0 under the falling map (no capacity there).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import b64, make_db, run
from tests.engine.test_chain_costs import _write_csv
from tests.engine.test_construction_lead_time_calendar import _same_lp

pytestmark = pytest.mark.solver

PLANT = "coal_plant"
PERIODS = ["p2020", "p2025", "p2030", "p2035"]
FALLING = [500.0, 300.0, 100.0, 0.0]
RISING = [100.0, 300.0, 500.0, 500.0]
# Four solves y2020_5week … y2035_5week, each realizing its first period (invest_solveSequence_5weeks).
CHAIN = ["init", "west", "wind", "coal", "battery", "5weeks", "invest_solveSequence_5weeks"]
CHAIN_INVEST = ["init", "west", "wind", "coal", "coal_invest", "battery", "5weeks", "invest_solveSequence_5weeks"]
# The same model as one solve over the four periods (y2020_2035_5week).
SINGLE_INVEST = CHAIN_INVEST + ["invest_allYears_5weeks"]
# invest_24h → storage_fullYear_6h → dispatch_fullYear_roll (rolls of 36 h, jumps of 24 h): a nested chain.
NESTED = ["init", "west", "coal", "wind", "battery", "battery_invest", "invest_solveSequence_5weeks",
          "invest_allYears_5weeks", "fullYear", "dispatch_fullYear_roll", "storage_fullYear_6h", "invest_24h",
          "bind_battery_forward", "unlimited_investment"]
STOCH_JSON = Path(__file__).resolve().parents[1] / "fixtures" / "stoch_realized_dispatch.json"


def _map(values: list[float], periods: list[str] = PERIODS) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[d, v] for d, v in zip(periods, values)]}, "map")


def _existing(values: list[float], entity: str = PLANT) -> list:
    return ["unit", entity, "existing", _map(values)]


NOT_INVESTABLE = ["unit", PLANT, "invest_method", b64("not_allowed", "str")]


def _frame(step, key: str, entity: str = PLANT) -> dict[str, float]:
    """``{period: value}`` of the emitted ``solve_data/<key>`` of ``entity``."""
    f = step.flex_data_provider.get(f"solve_data/{key}")
    f = f.filter(pl.col("entity").cast(pl.Utf8) == entity)
    return {str(d): float(v) for d, v in f.select("period", "value").iter_rows()}


def _realized(step) -> set[str]:
    f = step.flex_data_provider.get("solve_data/realized_invest_periods_of_current_solve")
    return {str(d) for d in f[f.columns[0]].to_list()}


def _invested_mw(step, entity: str = PLANT) -> dict[str, float]:
    """``{decision period: MW}`` the solve realized for ``entity`` (``v_invest`` · unitsize of its realized
    decision periods)."""
    sol = step.solution
    if "v_invest_p" not in sol._vars:
        return {}
    us = step.flex_data_provider.get("solve_data/p_entity_unitsize")
    unitsize = float(us.filter(pl.col(us.columns[0]).cast(pl.Utf8) == entity)[us.columns[-1]][0])
    v = sol.value("v_invest_p").filter(pl.col("p").cast(pl.Utf8) == entity)
    return {str(d): float(x) * unitsize for d, x in v.select("d", "value").iter_rows()
            if str(d) in _realized(step) and float(x) > 1e-12}


def _assert_map_plus_carried(steps: dict, values: list[float], invests: bool) -> None:
    by_period = dict(zip(PERIODS, values))
    carried: dict[str, float] = {}
    for name, step in steps.items():
        got = _frame(step, "p_entity_all_existing")
        assert got, name
        for d, v in got.items():
            want = by_period[d] + sum(mw for d_inv, mw in carried.items() if PERIODS.index(d_inv) <= PERIODS.index(d))
            assert v == pytest.approx(want, rel=1e-12, abs=1e-9), (name, d, v, want)
        if invests:
            for d_inv, mw in _invested_mw(step).items():
                carried[d_inv] = carried.get(d_inv, 0.0) + mw


# ---------------------------------------------------------------------------
# Chained solves


def test_chain_falling_map_is_read_in_every_solve(tmp_path_factory) -> None:
    steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt="map_falling", values=[_existing(FALLING),
                                                                                       NOT_INVESTABLE],
                                           base_chain=CHAIN))
    assert list(steps) == ["y2020_5week", "y2025_5week", "y2030_5week", "y2035_5week"]
    for name, step in steps.items():
        assert _frame(step, "p_entity_all_existing") == {d: v for d, v in zip(PERIODS, FALLING)
                                                          if d in _frame(step, "p_entity_all_existing")}, name
    # The LP: no coal capacity in p2035, so no coal flow (the map's first value gave 500 MW there).
    last = steps["y2035_5week"]
    count = last.flex_data.p_process_existing_count.frame
    count = count.filter((pl.col("p").cast(pl.Utf8) == PLANT) & (pl.col("d").cast(pl.Utf8) == "p2035"))
    assert count["value"].to_list() in ([], [0.0])
    flow = last.solution.value("v_flow")
    proc = [c for c in flow.columns if c not in ("d", "t", "value")][0]
    flow = flow.filter(pl.col(proc).cast(pl.Utf8) == PLANT)
    assert flow.height == 0 or flow["value"].abs().max() == pytest.approx(0.0, abs=1e-9)


def test_chain_rising_map_adds_the_carried_investments(tmp_path_factory) -> None:
    steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt="map_rising", values=[_existing(RISING)],
                                           base_chain=CHAIN_INVEST))
    assert any(_invested_mw(s) for s in steps.values()), "fixture invariant: the chain invests in coal"
    _assert_map_plus_carried(steps, RISING, invests=True)


# ---------------------------------------------------------------------------
# Rolling solve and nested solves

ROLL = "roll_small"


def _rolling_values(values: list[float]) -> list:
    return [_existing(values), NOT_INVESTABLE,
            ["model", "flexTool", "solves", b64({"value_type": "str", "data": [ROLL]}, "array")],
            ["solve", ROLL, "period_timeset", b64({"index_type": "str", "index_name": "period",
                                                   "data": [[d, "2day"] for d in PERIODS]}, "map")],
            ["solve", ROLL, "realized_periods", b64({"value_type": "str", "data": PERIODS}, "array")],
            ["solve", ROLL, "years_represented", _map([5.0] * 4)],
            ["solve", ROLL, "solve_mode", b64("rolling_window", "str")],
            ["solve", ROLL, "rolling_solve_horizon", b64(36.0, "float")],
            ["solve", ROLL, "rolling_solve_jump", b64(24.0, "float")]]


def _rolling(tmp_path_factory, alt: str, values: list[float]) -> dict:
    url, scen = make_db(tmp_path_factory, alt=alt, values=_rolling_values(values), base_chain=["init", "west", "coal"],
                        entities=[["solve", ROLL, None]])
    return run(tmp_path_factory, url, scen)


def test_every_roll_reads_the_map(tmp_path_factory) -> None:
    steps = _rolling(tmp_path_factory, "map_roll", FALLING)
    assert len(steps) > len(PERIODS)
    by_period = dict(zip(PERIODS, FALLING))
    for name, step in steps.items():
        for d, v in _frame(step, "p_entity_all_existing").items():
            assert v == by_period[d], (name, d)
    holds_2035 = [s for s in steps.values() if "p2035" in _frame(s, "p_entity_all_existing")]
    assert holds_2035
    for step in holds_2035:
        count = step.flex_data.p_process_existing_count.frame.filter(
            (pl.col("p").cast(pl.Utf8) == PLANT) & (pl.col("d").cast(pl.Utf8) == "p2035"))
        assert count["value"].to_list() in ([], [0.0])


def test_nested_sub_solves_read_the_map(tmp_path_factory) -> None:
    values = [_existing(FALLING), NOT_INVESTABLE,
              ["solve", "dispatch_fullYear_roll", "rolling_solve_horizon", b64(36.0, "float")],
              ["solve", "dispatch_fullYear_roll", "rolling_solve_jump", b64(24.0, "float")]]
    steps = run(tmp_path_factory, *make_db(tmp_path_factory, alt="map_nested", values=values, base_chain=NESTED))
    names = list(steps)
    assert any("dispatch" in n for n in names) and any("storage" in n for n in names), names
    by_period = dict(zip(PERIODS, FALLING))
    for name, step in steps.items():
        got = _frame(step, "p_entity_all_existing")
        assert got, name
        for d, v in got.items():
            assert v == by_period[d], (name, d)


# ---------------------------------------------------------------------------
# Stochastic solve with a realized-dispatch solve


def test_stochastic_dispatch_reads_the_map_and_the_trunk_investment(tmp_path_factory) -> None:
    """``stoch_realized_dispatch``: the stochastic solve invests in the peaker in p2035 (one real-named trunk
    decision), the dispatch solve realizes it.  With the peaker's existing a map {p2035: 20}: every period of the
    stochastic solve (its branch periods too) holds 20 MW, and the dispatch solve 20 + the trunk investment."""
    values = [["unit", "peaker", "existing", _map([20.0], ["p2035"])]]
    url, scen = make_db(tmp_path_factory, alt="map_stoch", values=values, base_chain=["init", "base", "stoch"],
                        fixture_json=STOCH_JSON)
    steps = run(tmp_path_factory, url, scen)
    stoch, dispatch = steps["stoch_rd"], steps[[n for n in steps if n != "stoch_rd"][0]]
    assert set(_frame(stoch, "p_entity_all_existing", "peaker").values()) == {20.0}
    trunk = _invested_mw(stoch, "peaker")
    assert set(trunk) == {"p2035"}
    carried = stoch.handoff.realized_cohort
    assert carried["period_decided"].to_list() == ["p2035"]
    assert _frame(dispatch, "p_entity_all_existing", "peaker") == {"p2035": pytest.approx(20.0 + trunk["p2035"])}


# ---------------------------------------------------------------------------
# Relational: a constant and the equivalent map build the same LP; a chain and one solve the same capacity


@pytest.mark.parametrize("case", ["chain", "rolling"])
def test_constant_equals_period_map(tmp_path_factory, case) -> None:
    if case == "chain":
        def _go(alt, existing):
            return run(tmp_path_factory, *make_db(tmp_path_factory, alt=alt, values=[existing, NOT_INVESTABLE],
                                                  base_chain=CHAIN))
        const = _go("ex_const_chain", ["unit", PLANT, "existing", b64(300.0, "float")])
        as_map = _go("ex_map_chain", _existing([300.0] * 4))
    else:
        const_values = _rolling_values([300.0] * 4)
        const_values[0] = ["unit", PLANT, "existing", b64(300.0, "float")]
        url, scen = make_db(tmp_path_factory, alt="ex_const_roll", values=const_values,
                            base_chain=["init", "west", "coal"], entities=[["solve", ROLL, None]])
        const = run(tmp_path_factory, url, scen)
        as_map = _rolling(tmp_path_factory, "ex_map_roll", [300.0] * 4)
    assert list(const) == list(as_map)
    for name in const:
        _same_lp(const[name], as_map[name])


# coal: a 50 MW investment forced in p2025 (invest_min_period = invest_max_period) and 100 MW of the existing
# capacity retired in p2025 (retire_min_period = retire_max_period); the map falls from 500 to 300.
STEP_DOWN = [500.0, 300.0, 300.0, 300.0]
FORCED = [_existing(STEP_DOWN),
          ["unit", PLANT, "invest_method", b64("invest_retire_period", "str")],
          ["unit", PLANT, "invest_min_period", _map([0.0, 50.0, 0.0, 0.0])],
          ["unit", PLANT, "invest_max_period", _map([0.0, 50.0, 0.0, 0.0])],
          ["unit", PLANT, "retire_min_period", _map([0.0, 100.0, 0.0, 0.0])],
          ["unit", PLANT, "retire_max_period", _map([0.0, 100.0, 0.0, 0.0])],
          # wind: 100 MW coming online in p2025 (the limits count the capacity starting to operate), decided in p2020
          # with a 5-year construction lead time.
          ["unit", "wind_plant", "invest_method", b64("invest_period", "str")],
          ["unit", "wind_plant", "invest_cost", b64(1000.0, "float")],
          ["unit", "wind_plant", "lifetime_technical", b64(30.0, "float")],
          ["unit", "wind_plant", "payback_time", b64(30.0, "float")],
          ["unit", "wind_plant", "construction_lead_time", b64(5.0, "float")],
          ["unit", "wind_plant", "invest_min_period", _map([0.0, 100.0, 0.0, 0.0])],
          ["unit", "wind_plant", "invest_max_period", _map([0.0, 100.0, 0.0, 0.0])]]


def _capacity(tmp_path_factory, alt: str, chain: list[str]) -> pd.DataFrame:
    out = _write_csv(tmp_path_factory, alt, FORCED, chain)
    frame = pd.read_csv(out / "unit_capacity__d.csv")
    frame = frame[frame["unit"].isin([PLANT, "wind_plant"])]
    return frame[["unit", "period", "total"]].sort_values(["unit", "period"]).reset_index(drop=True)


def test_chain_and_single_solve_have_the_same_capacity_in_service(tmp_path_factory) -> None:
    """The forced investments and retirement give one capacity in service per period, whether the periods are
    solved one after the other or in one solve.  coal: 500 (p2020) and 300 + 50 − 100 = 250 (p2025 … p2035);
    wind (existing 1000 MW): 1000 in p2020, 1000 + 100 once the p2020 decision starts to operate (p2025 on)."""
    chain = _capacity(tmp_path_factory, "forced_chain", CHAIN_INVEST)
    single = _capacity(tmp_path_factory, "forced_single", SINGLE_INVEST)
    coal = chain[chain["unit"] == PLANT]
    wind = chain[chain["unit"] == "wind_plant"]
    assert list(coal["period"]) == PERIODS and list(wind["period"]) == PERIODS
    assert coal["total"].tolist() == pytest.approx([500.0, 250.0, 250.0, 250.0])
    assert wind["total"].tolist() == pytest.approx([1000.0, 1100.0, 1100.0, 1100.0])
    pd.testing.assert_frame_equal(chain, single, check_exact=False, rtol=1e-9, atol=1e-9)
