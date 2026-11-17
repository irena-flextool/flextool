"""``end_of_life_method``: the end of life of every cohort rounded to a period start, in a solved model.

Built from ``tests/fixtures/tests.json`` (alternatives init, west, wind, coal): coal_plant has 500 MW existing,
wind_plant 1000 MW.  The solve ``life_L`` holds p2020, p2025, p2030 representing 5, 5 and 10 years: their run years
are 0, 5, 10, the horizon ends at 20, the seams are {0, 5, 10, 20}.  A cohort is in service in ``d`` while
``y_on <= y[d] < y_eol``.  Expected values, by hand:

* coal_plant reinvest_choice, lifetime_technical 9: Y = 0 + 9 = 9.  closest_seam: 9 − 5 = 4 > 10 − 9 = 1 → 10;
  next_seam → 10; previous_seam → 5.  Existing capacity per period: closest / next 500, 500, 0; previous 500, 0, 0.
* lifetime_technical 12: Y = 12 → next_seam 20 (500, 500, 500); closest (12 − 10 = 2 < 20 − 12 = 8) and previous
  → 10 (500, 500, 0).
* wind_plant invests in p2025 (reinvest_choice, lifetime_technical 7): y_on = 5, Y = 12 → closest / previous 10
  (in service in p2025 only), next 20 (p2025 and p2030).
* An explicit next_seam builds the same LP as no end_of_life_method (the default).
* wind_plant lifetime_technical 4, decided in p2025 under previous_seam: Y = 9 → 5 = y_on, in service nowhere: a
  warning names it, the decision is still offered.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import RunCache, b64
from tests.engine.test_construction_lead_time_calendar import _same_lp

pytestmark = pytest.mark.solver

PLANT = "coal_plant"
WIND = "wind_plant"
PERIODS = ["p2020", "p2025", "p2030"]
YEARS = [5.0, 5.0, 10.0]
CHAIN = ["init", "west", "wind", "coal"]
SOLVE = "life_L"

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _map(values: dict, kind: str = "map") -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, kind)


def _array(values: list[str]) -> list:
    return b64({"value_type": "str", "data": values}, "array")


def solve_values(solve: str = SOLVE, periods: list[str] = PERIODS, years: list[float] = YEARS,
                 realized: list[str] | None = None, invest: list[str] | None = None,
                 mode: str = "single_solve") -> list:
    """The values of a solve ``solve`` (``solve_mode`` ``mode``) over ``periods`` (timeset 2day) representing
    ``years``."""
    out = [["solve", solve, "period_timeset", _map(dict.fromkeys(periods, "2day"))],
           ["solve", solve, "realized_periods", _array(realized if realized is not None else periods)],
           ["solve", solve, "years_represented", _map(dict(zip(periods, years)))],
           ["solve", solve, "solve_mode", b64(mode, "str")]]
    if invest:
        out.append(["solve", solve, "invest_periods", _array(invest)])
    return out


def model_solves(solves: list[str]) -> list:
    return ["model", "flexTool", "solves", _array(solves)]


def coal(life: float, method: str | None = None) -> list:
    out = [["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
           ["unit", PLANT, "lifetime_technical", b64(life, "float")]]
    if method is not None:
        out.append(["unit", PLANT, "end_of_life_method", b64(method, "str")])
    return out


def wind_invest(life: float, method: str | None = None) -> list:
    out = [["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
           ["unit", WIND, "invest_method", b64("invest_period", "str")],
           ["unit", WIND, "invest_cost", b64(1000.0, "float")],
           ["unit", WIND, "invest_max_period", _map({"p2025": 100.0})],
           ["unit", WIND, "lifetime_technical", b64(life, "float")],
           ["unit", WIND, "payback_time", b64(life, "float")]]
    if method is not None:
        out.append(["unit", WIND, "end_of_life_method", b64(method, "str")])
    return out


def steps_of(tmp_path_factory, alt: str, values: list, solves: list[str] = (SOLVE,)) -> dict:
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=[model_solves(list(solves))] + values,
                            base_chain=CHAIN, entities=[["solve", s, None] for s in solves])
    return steps


def capacity(step, entity: str = PLANT) -> dict[str, float]:
    """``{period: MW}`` of the solve's ``p_entity_all_existing`` of ``entity`` (the capacity the LP builds on)."""
    f = step.flex_data.p_entity_all_existing.frame.filter(pl.col("e").cast(pl.Utf8) == entity)
    return {str(d): float(v) for d, v in f.select("d", "value").iter_rows()}


def invest_alive(step, entity: str = WIND) -> list[tuple[str, str]]:
    """``[(decision period, period in service)]`` of ``entity``'s investments (the solve's cohort_alive)."""
    alive = step.flex_data.cohort_alive.filter((pl.col("kind") == "invest") & (pl.col("e").cast(pl.Utf8) == entity))
    return sorted((str(a), str(b)) for a, b in alive.select("d_build", "d").iter_rows())


@pytest.mark.parametrize("life,method,want", [
    (9.0, "closest_seam", [500.0, 500.0, 0.0]),
    (9.0, "next_seam", [500.0, 500.0, 0.0]),
    (9.0, "previous_seam", [500.0, 0.0, 0.0]),
    (12.0, "closest_seam", [500.0, 500.0, 0.0]),
    (12.0, "previous_seam", [500.0, 500.0, 0.0]),
    (12.0, "next_seam", [500.0, 500.0, 500.0]),
])
def test_existing_capacity_by_end_of_life_method(tmp_path_factory, life, method, want) -> None:
    steps = steps_of(tmp_path_factory, f"eol_{method}_{life:g}", solve_values() + coal(life, method))
    (step,) = steps.values()
    assert capacity(step) == dict(zip(PERIODS, want))
    count = step.flex_data.p_process_existing_count.frame.filter(pl.col("p").cast(pl.Utf8) == PLANT)
    gone = [d for d, v in zip(PERIODS, want) if v == 0.0]
    assert all(v == 0.0 for d, v in count.select(pl.col("d").cast(pl.Utf8), "value").iter_rows() if d in gone)


@pytest.mark.parametrize("method,alive", [
    ("closest_seam", [("p2025", "p2025")]),
    ("previous_seam", [("p2025", "p2025")]),
    ("next_seam", [("p2025", "p2025"), ("p2025", "p2030")]),
])
def test_investment_in_service_by_end_of_life_method(tmp_path_factory, method, alive) -> None:
    steps = steps_of(tmp_path_factory, f"eol_inv_{method}",
                     solve_values(invest=["p2025"]) + wind_invest(7.0, method))
    (step,) = steps.values()
    assert invest_alive(step) == alive
    edd = step.flex_data_provider.get("solve_data/edd_invest")
    edd = edd.filter(pl.col("entity").cast(pl.Utf8) == WIND)
    assert sorted((str(a), str(b)) for a, b in edd.select("period_history", "period").iter_rows()) == alive


def test_explicit_next_seam_is_the_default(tmp_path_factory) -> None:
    unset = steps_of(tmp_path_factory, "eol_unset", solve_values(invest=["p2025"]) + coal(12.0) + wind_invest(7.0))
    explicit = steps_of(tmp_path_factory, "eol_next", solve_values(invest=["p2025"]) + coal(12.0, "next_seam")
                        + wind_invest(7.0, "next_seam"))
    assert list(unset) == list(explicit)
    for name in unset:
        _same_lp(unset[name], explicit[name])


def test_investment_that_never_operates_warns_and_is_offered(tmp_path_factory, capsys) -> None:
    from flextool.engine import _behaviour_changes
    capsys.readouterr()
    steps = steps_of(tmp_path_factory, "eol_never", solve_values(invest=["p2025"]) + wind_invest(4.0,
                                                                                                "previous_seam"))
    (step,) = steps.values()
    printed = capsys.readouterr().out
    assert _behaviour_changes.already_warned(("investment_never_operates", WIND))
    assert ("Warning: unit wind_plant: investments decided in p2025 would never operate (lifetime_technical 4 "
            "years, end_of_life_method previous_seam)") in printed
    assert printed.count("investments decided in") == 1
    assert invest_alive(step) == []
    offered = step.flex_data.ed_invest_set
    assert (WIND, "p2025") in {(str(e), str(d)) for e, d in offered.select("e", "d").iter_rows()}


@pytest.mark.parametrize("method,periods", [
    ("previous_seam", ["p2025"]),
    ("closest_seam", ["p2025"]),
    ("next_seam", ["p2025", "p2030"]),
])
def test_fixed_cost_of_an_investment_stops_at_its_rounded_end_of_life(tmp_path_factory, method, periods) -> None:
    """wind_plant decided in p2025 (y_on 5, lifetime_technical 7, fixed_cost 10 CUR/kW/a): its end of life
    Y = 12 lies inside p2030 [10, 20).  previous_seam / closest_seam end the cohort at 10, so it pays its fixed cost
    in p2025 only: ``lifetime_fixed_cost = 10 · 1000 · inflation_op[p2025]``; next_seam keeps p2030:
    ``10 · 1000 · (inflation_op[p2025] + inflation_op[p2030])``.  The reported operating window has the same
    periods."""
    values = solve_values(invest=["p2025"]) + wind_invest(7.0, method) + [
        ["unit", WIND, "fixed_cost", b64(10.0, "float")]]
    (step,) = steps_of(tmp_path_factory, f"eol_fc_{method}", values).values()
    fd = step.flex_data
    op = {str(d): float(v) for d, v in fd.p_inflation_op.frame.select(pl.col("d").cast(pl.Utf8), "value").iter_rows()}
    lfc = fd.ed_lifetime_fixed_cost.frame.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                                 & (pl.col("d").cast(pl.Utf8) == "p2025"))
    assert lfc["value"].to_list() == [pytest.approx(10.0 * 1000.0 * sum(op[d] for d in periods), rel=1e-12)]
    window = fd.edd_operating_window.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                            & (pl.col("d_invest").cast(pl.Utf8) == "p2025"))
    assert sorted(str(d) for d in window["d"].to_list()) == periods
    assert invest_alive(step) == [("p2025", d) for d in periods]


def test_lead_time_investment_takes_the_lifetime_of_its_decision_period(tmp_path_factory) -> None:
    """wind_plant decided in p2020 (y 0) with a 5-year construction lead time starts to operate in p2025 (y_on 5);
    lifetime_technical is a map {p2020: 3, p2025: 20, p2030: 20}, end_of_life_method next_seam, fixed_cost 10
    CUR/kW/a.  The lifetime is that of the decision period, 3: Y = 5 + 3 = 8 → next_seam 10, so the cohort is in
    service in p2025 only and pays its fixed cost there: ``lifetime_fixed_cost[p2020] = 10 · 1000 ·
    inflation_op[p2025]``, the reported operating window p2025.  (The lifetime of the start of operation, 20, would
    give Y = 25, past the horizon end 20: p2025 and p2030.)"""
    values = solve_values(invest=["p2020"]) + [
        ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", WIND, "invest_method", b64("invest_period", "str")],
        ["unit", WIND, "invest_cost", b64(1000.0, "float")],
        ["unit", WIND, "invest_max_period", _map({"p2025": 100.0})],
        ["unit", WIND, "lifetime_technical", _map({"p2020": 3.0, "p2025": 20.0, "p2030": 20.0})],
        ["unit", WIND, "payback_time", b64(20.0, "float")],
        ["unit", WIND, "construction_lead_time", b64(5.0, "float")],
        ["unit", WIND, "end_of_life_method", b64("next_seam", "str")],
        ["unit", WIND, "fixed_cost", b64(10.0, "float")]]
    (step,) = steps_of(tmp_path_factory, "eol_lead_life", values).values()
    fd = step.flex_data
    assert invest_alive(step) == [("p2020", "p2025")]
    op = {str(d): float(v) for d, v in fd.p_inflation_op.frame.select(pl.col("d").cast(pl.Utf8), "value").iter_rows()}
    lfc = fd.ed_lifetime_fixed_cost.frame.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                                 & (pl.col("d").cast(pl.Utf8) == "p2020"))
    assert lfc["value"].to_list() == [pytest.approx(10.0 * 1000.0 * op["p2025"], rel=1e-12)]
    window = fd.edd_operating_window.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                            & (pl.col("d_invest").cast(pl.Utf8) == "p2020"))
    assert sorted(str(d) for d in window["d"].to_list()) == ["p2025"]
