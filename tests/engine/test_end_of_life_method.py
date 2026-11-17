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
* An explicit closest_seam builds the same LP as no end_of_life_method (the default).
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
           ["unit", WIND, "lifetime_technical", b64(life, "float")]]
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


def test_explicit_closest_seam_is_the_default(tmp_path_factory) -> None:
    unset = steps_of(tmp_path_factory, "eol_unset", solve_values(invest=["p2025"]) + coal(12.0) + wind_invest(7.0))
    explicit = steps_of(tmp_path_factory, "eol_closest", solve_values(invest=["p2025"]) + coal(12.0, "closest_seam")
                        + wind_invest(7.0, "closest_seam"))
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
    (None, ["p2025"]),
])
def test_fixed_cost_of_an_investment_stops_at_its_rounded_end_of_life(tmp_path_factory, method, periods) -> None:
    """wind_plant decided in p2025 (y_on 5, lifetime_technical 7, fixed_cost 10 CUR/kW/a): its end of life
    Y = 12 lies inside p2030 [10, 20).  previous_seam / closest_seam (also the default, no end_of_life_method) end
    the cohort at 10, so it pays its fixed cost in p2025 only: ``lifetime_fixed_cost = 10 · 1000 ·
    inflation_op[p2025]``; next_seam keeps p2030: ``10 · 1000 · (inflation_op[p2025] + inflation_op[p2030])``.  The
    reported operating window has the same periods."""
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


# ---------------------------------------------------------------------------
# The default closest_seam: a tie, a lead time, and the notice of an upgraded database


def test_tie_and_seam_ends_of_life_are_rounded_alike_by_the_default(tmp_path_factory) -> None:
    """coal_plant (existing) lifetime_technical 7.5: Y = 7.5, a tie (7.5 − 5 = 10 − 7.5) goes to the later start
    10; wind_plant decided in p2025 with lifetime_technical 15: Y = 5 + 15 = 20, the horizon end (a seam).  Both
    round as next_seam does, so the default builds the same LP as an explicit next_seam."""
    unset = steps_of(tmp_path_factory, "eol_tie_unset", solve_values(invest=["p2025"]) + coal(7.5)
                     + wind_invest(15.0))
    explicit = steps_of(tmp_path_factory, "eol_tie_next", solve_values(invest=["p2025"]) + coal(7.5, "next_seam")
                        + wind_invest(15.0, "next_seam"))
    for name in unset:
        _same_lp(unset[name], explicit[name])
    assert capacity(next(iter(unset.values()))) == dict(zip(PERIODS, [500.0, 500.0, 0.0]))


def _lead_time_wind(life: float, method: str | None) -> list:
    out = [["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
           ["unit", WIND, "invest_method", b64("invest_period", "str")],
           ["unit", WIND, "invest_cost", b64(1000.0, "float")],
           ["unit", WIND, "invest_max_period", _map({"p2025": 100.0})],
           ["unit", WIND, "lifetime_technical", b64(life, "float")],
           ["unit", WIND, "construction_lead_time", b64(5.0, "float")],
           ["unit", WIND, "fixed_cost", b64(10.0, "float")]]
    if method is not None:
        out.append(["unit", WIND, "end_of_life_method", b64(method, "str")])
    return out


@pytest.mark.parametrize("method,periods", [(None, ["p2025"]), ("next_seam", ["p2025", "p2030"])])
def test_lead_time_investment_end_of_life_by_the_default(tmp_path_factory, method, periods) -> None:
    """wind_plant ordered in p2020 (y 0) with a 5-year construction lead time starts to operate in p2025 (y_on 5);
    lifetime_technical 6: Y = 5 + 6 = 11 lies inside p2030 [10, 20).  The default closest_seam rounds it to 10
    (11 − 10 = 1 < 20 − 11 = 9): in service in p2025 only, fixed cost ``10 · 1000 · inflation_op[p2025]``;
    next_seam rounds it to 20: p2025 and p2030."""
    values = solve_values(invest=["p2020"]) + _lead_time_wind(6.0, method)
    (step,) = steps_of(tmp_path_factory, f"eol_lead_default_{method}", values).values()
    assert invest_alive(step) == [("p2020", d) for d in periods]
    fd = step.flex_data
    op = {str(d): float(v) for d, v in fd.p_inflation_op.frame.select(pl.col("d").cast(pl.Utf8), "value").iter_rows()}
    lfc = fd.ed_lifetime_fixed_cost.frame.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                                 & (pl.col("d").cast(pl.Utf8) == "p2020"))
    assert lfc["value"].to_list() == [pytest.approx(10.0 * 1000.0 * sum(op[d] for d in periods), rel=1e-12)]


NOTICE_ID = "end_of_life_closest_seam"
#: A chain of two solves on the layout: life_a holds p2020, life_b p2025 and p2030 (run years 0, 5, 10; the horizon
#: ends at 20).
CHAIN_SOLVES = (solve_values("life_a", ["p2020"], [5.0], invest=["p2020"])
                + solve_values("life_b", ["p2025", "p2030"], [5.0, 10.0]))


def _notice_lines(tmp_path_factory, capsys, alt: str, values: list, solves: tuple[str, ...] = (SOLVE,),
                  marker: bool = True) -> list[str]:
    """The printed lines of the ``end_of_life_closest_seam`` notice of a run of ``solves`` whose database carries
    the upgrade's marker naming coal_plant and wind_plant (none with ``marker`` False)."""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, MARKER_PREFIX
    from tests.engine._unit_db import make_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=[model_solves(list(solves))] + values, base_chain=CHAIN,
                        entities=[["solve", s, None] for s in solves])
    if marker:
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=MARKER_PREFIX + NOTICE_ID,
                        value='{"since": 81, "entities": ["coal_plant", "wind_plant"]}')
            db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]


def test_notice_names_each_entity_whose_end_of_life_moves_once(tmp_path_factory, capsys) -> None:
    """Over the chain life_a | life_b, coal_plant (existing 500 MW, lifetime_technical 12: Y = 12 → 10 instead of
    20) is named once, when the run places its existing cohort; wind_plant (no existing capacity, decided in p2020
    with lifetime_technical 12: Y = 12 → 10) is named by the solve that offers the decision."""
    values = CHAIN_SOLVES + coal(12.0) + [
        ["unit", WIND, "existing", b64(0.0, "float")],
        ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", WIND, "invest_method", b64("invest_period", "str")],
        ["unit", WIND, "invest_cost", b64(1000.0, "float")],
        ["unit", WIND, "invest_max_period", _map({"p2020": 100.0})],
        ["unit", WIND, "lifetime_technical", b64(12.0, "float")]]
    lines = _notice_lines(tmp_path_factory, capsys, "eol_notice_chain", values, solves=("life_a", "life_b"))
    assert len(lines) == 2, lines
    assert "Affected entities: coal_plant." in lines[0]
    assert ("coal_plant (existing): end of life at year 12 of the run, out of service from p2030 (next_seam: in "
            "service to the end of the horizon).") in lines[0]
    assert "Affected entities: wind_plant." in lines[1]
    assert ("wind_plant (decided in p2020): end of life at year 12 of the run, out of service from p2030 "
            "(next_seam: in service to the end of the horizon).") in lines[1]


@pytest.mark.parametrize("alt,values,marker", [
    # a tie (7.5 → 10 = next_seam) and an investment ending past the horizon (5 + 18 = 23: in service to the end)
    ("eol_quiet_tie", coal(7.5) + wind_invest(18.0), True),
    # an explicit end_of_life_method, either one
    ("eol_quiet_explicit", coal(12.0, "closest_seam") + wind_invest(7.0, "next_seam"), True),
    # reinvest_automatic: renewed, no end of life
    ("eol_quiet_automatic", [["unit", PLANT, "lifetime_method", b64("reinvest_automatic", "str")],
                             ["unit", PLANT, "lifetime_technical", b64(12.0, "float")]], True),
    # no existing capacity and no investment
    ("eol_quiet_no_existing", coal(12.0) + [["unit", PLANT, "existing", b64(0.0, "float")]], True),
    # no marker (a database that was not upgraded)
    ("eol_quiet_no_marker", coal(12.0) + wind_invest(7.0), False),
])
def test_notice_is_silent_where_nothing_moves(tmp_path_factory, capsys, alt, values, marker) -> None:
    lines = _notice_lines(tmp_path_factory, capsys, alt, solve_values(invest=["p2025"]) + values, marker=marker)
    assert lines == []


def test_notice_is_registered() -> None:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    assert NOTICE_ID in BEHAVIOUR_CHANGES


def _lead_time_annuity(tmp_path_factory, alt: str, payback: float | None) -> tuple[float, float]:
    """``(ed_annuity, ed_entity_annual_discounted)`` of wind_plant decided in p2020 with a 5-year construction lead
    time (online in p2025), lifetime_technical {p2020: 3, p2025: 20, p2030: 20}, and ``payback`` as its constant
    payback_time (none when None)."""
    values = solve_values(invest=["p2020"]) + [
        ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", WIND, "invest_method", b64("invest_period", "str")],
        ["unit", WIND, "invest_cost", b64(1000.0, "float")],
        ["unit", WIND, "discount_rate", b64(0.05, "float")],
        ["unit", WIND, "invest_max_period", _map({"p2025": 100.0})],
        ["unit", WIND, "lifetime_technical", _map({"p2020": 3.0, "p2025": 20.0, "p2030": 20.0})],
        ["unit", WIND, "construction_lead_time", b64(5.0, "float")],
        ["unit", WIND, "end_of_life_method", b64("next_seam", "str")]]
    if payback is not None:
        values.append(["unit", WIND, "payback_time", b64(payback, "float")])
    (step,) = steps_of(tmp_path_factory, alt, values).values()
    fd = step.flex_data

    def at_decision(frame: pl.DataFrame) -> float:
        rows = frame.filter((pl.col("e").cast(pl.Utf8) == WIND) & (pl.col("d").cast(pl.Utf8) == "p2020"))
        (value,) = rows["value"].to_list()
        return float(value)

    return at_decision(fd.ed_annuity.frame), at_decision(fd.ed_entity_annual_discounted.frame)


def test_lead_time_annuity_without_payback_time_takes_the_lifetime_of_its_decision_period(tmp_path_factory) -> None:
    """Without payback_time the annuity of wind_plant (decided in p2020, online in p2025) is over the
    lifetime_technical of its decision period, 3 years (not that of p2025, 20), as its end of life is: the same as
    with payback_time 3.  By hand: 1.05^-3 = 0.8638376, factor 0.05 / (1 − 0.8638376) = 0.3672086, annuity 1000 ·
    1000 · 0.3672086 = 367 208.56 CUR/MW/a; reinvest_choice pays it for 3 years from y_on 5 (years 5, 6, 7 of p2025,
    no inflation): 3 · 367 208.56 = 1 101 625.69.  (Over 20 years: 80 242.59 a year for the 15 years to the
    horizon end.)"""
    ann, discounted = _lead_time_annuity(tmp_path_factory, "lead_annuity_no_payback", None)
    twin = _lead_time_annuity(tmp_path_factory, "lead_annuity_payback_3", 3.0)
    assert ann == pytest.approx(367_208.56, rel=1e-7)
    assert discounted == pytest.approx(1_101_625.69, rel=1e-7)
    assert (ann, discounted) == pytest.approx(twin, rel=1e-12)
