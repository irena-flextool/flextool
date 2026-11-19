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
  warning names it, and the decision is not offered (no investment column).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import RunCache, b64, build, utf8
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


def _invest_columns(step, entity: str = WIND) -> list[str]:
    """The decision periods of ``entity``'s investment columns in the LP of ``step``."""
    var = build(step.flex_data)._vars.get("v_invest_p")
    if var is None:
        return []
    return sorted(str(d) for d in utf8(var.frame).filter(pl.col("p") == entity)["d"].to_list())


def _offered(step, entity: str = WIND) -> list[str]:
    """The decision periods of ``entity`` in the solve's investment set."""
    offered = step.flex_data.ed_invest_set
    if offered is None:
        return []
    return sorted(str(d) for e, d in offered.select("e", "d").iter_rows() if str(e) == entity)


def _cap_every_period(values: list) -> list:
    """``values`` with wind_plant's ``invest_max_period`` 100 MW in every period of the solve."""
    return [row if row[2] != "invest_max_period" else ["unit", WIND, "invest_max_period",
                                                        _map(dict.fromkeys(PERIODS, 100.0))]
            for row in values]


def test_investment_that_never_operates_is_not_offered(tmp_path_factory, coded_log) -> None:
    """wind_plant may invest in p2020, p2025 and p2030 (lifetime_technical 7, previous_seam).  Hand: p2020 y_on 0,
    Y 7 → 5 (in service in p2020); p2025 y_on 5, Y 12 → 10 (p2025); p2030 y_on 10, Y 17 → 10 = y_on: it would
    never operate, so the solve has no investment column for it; a warning names p2030 once."""
    coded_log()
    values = solve_values(invest=PERIODS) + _cap_every_period(wind_invest(7.0, "previous_seam"))
    (step,) = steps_of(tmp_path_factory, "eol_never_one", values).values()
    printed = coded_log()
    assert "Warning: investments would never operate (end of life at start) ('wind_plant') [W-0522]" in printed
    assert ("W-0522 wind_plant: unit: investments decided in p2030 (lifetime_technical 7 years, "
            "end_of_life_method previous_seam)") in printed
    assert printed.count("investments decided in") == 1
    assert _offered(step) == ["p2020", "p2025"]
    assert _invest_columns(step) == ["p2020", "p2025"]
    assert invest_alive(step) == [("p2020", "p2020"), ("p2025", "p2025")]


@pytest.mark.parametrize("lead", [False, True, "map"])
def test_entity_whose_every_investment_never_operates_solves(tmp_path_factory, lead) -> None:
    """wind_plant (the only entity that invests) lifetime_technical 4, previous_seam.  Decided in p2025: y_on 5,
    Y 9 → 5 = y_on.  With a 5-year construction lead time, decided in p2020: starts to operate in p2025, y_on 5,
    Y 9 → 5 = y_on; also with the lifetime the map {p2020: 20, p2025: 4, p2030: 20}, whose online period's value
    describes the plant (as constructed; the decision period's 20 would give Y 25 → 20, in service in p2025
    and p2030).  Each way the one decision would never operate: the solve has no investment set and no investment
    column, and it solves."""
    if lead:
        values = solve_values(invest=["p2020"]) + _cap_every_period(_lead_time_wind(4.0, "previous_seam"))
        if lead == "map":
            values = [row if row[2] != "lifetime_technical" else
                      ["unit", WIND, "lifetime_technical", _map({"p2020": 20.0, "p2025": 4.0, "p2030": 20.0})]
                      for row in values]
    else:
        values = solve_values(invest=["p2025"]) + wind_invest(4.0, "previous_seam")
    steps = steps_of(tmp_path_factory, f"eol_never_all_{lead}", values)
    (step,) = steps.values()
    assert _offered(step) == []
    assert _invest_columns(step) == []
    assert step.flex_data.ed_invest_commission is None


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
    inflation_op[p2025]``; next_seam keeps p2030: ``10 · 1000 · (inflation_op[p2025] + inflation_op[p2030])``, the
    periods the cohort is in service in."""
    values = solve_values(invest=["p2025"]) + wind_invest(7.0, method) + [
        ["unit", WIND, "fixed_cost", b64(10.0, "float")]]
    (step,) = steps_of(tmp_path_factory, f"eol_fc_{method}", values).values()
    fd = step.flex_data
    op = {str(d): float(v) for d, v in fd.p_inflation_op.frame.select(pl.col("d").cast(pl.Utf8), "value").iter_rows()}
    lfc = fd.ed_lifetime_fixed_cost.frame.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                                 & (pl.col("d").cast(pl.Utf8) == "p2025"))
    assert lfc["value"].to_list() == [pytest.approx(10.0 * 1000.0 * sum(op[d] for d in periods), rel=1e-12)]
    assert invest_alive(step) == [("p2025", d) for d in periods]


def test_lead_time_investment_takes_the_lifetime_of_its_online_period(tmp_path_factory) -> None:
    """wind_plant decided in p2020 (y 0) with a 5-year construction lead time starts to operate in p2025 (y_on 5);
    lifetime_technical is a map {p2020: 3, p2025: 20, p2030: 20}, end_of_life_method next_seam, fixed_cost 10
    CUR/kW/a.  The period map describes the plant as constructed, so the lifetime is that of
    the online period p2025, 20: Y = 5 + 20 = 25, past the horizon end 20 (the last seam) → next_seam +inf, so the
    cohort is in service in p2025 and p2030 and pays its fixed cost there: ``lifetime_fixed_cost[p2020] = 10 · 1000
    · (inflation_op[p2025] + inflation_op[p2030])``.  (The lifetime of
    the decision period, 3, would give Y = 8 → 10: p2025 only.)  The per-solve emit's investment window
    (``edd_invest``) is the same."""
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
    assert invest_alive(step) == [("p2020", "p2025"), ("p2020", "p2030")]
    # The solve's emitted investment window (edd_invest, the bounds of the per-solve emit) is the same window.
    edd = step.flex_data_provider.get("solve_data/edd_invest").filter(pl.col("entity").cast(pl.Utf8) == WIND)
    assert sorted((str(a), str(b)) for a, b in edd.select("period_history", "period").iter_rows()) == [
        ("p2020", "p2025"), ("p2020", "p2030")]
    op = {str(d): float(v) for d, v in fd.p_inflation_op.frame.select(pl.col("d").cast(pl.Utf8), "value").iter_rows()}
    lfc = fd.ed_lifetime_fixed_cost.frame.filter((pl.col("e").cast(pl.Utf8) == WIND)
                                                 & (pl.col("d").cast(pl.Utf8) == "p2020"))
    assert lfc["value"].to_list() == [pytest.approx(10.0 * 1000.0 * (op["p2025"] + op["p2030"]), rel=1e-12)]


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


def _notice_lines(tmp_path_factory, log, alt: str, values: list, solves: tuple[str, ...] = (SOLVE,),
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
    log()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in log.notices() if summary in ln]


def test_notice_names_each_entity_whose_end_of_life_moves_once(tmp_path_factory, coded_log) -> None:
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
    lines = _notice_lines(tmp_path_factory, coded_log, "eol_notice_chain", values, solves=("life_a", "life_b"))
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
def test_notice_is_silent_where_nothing_moves(tmp_path_factory, coded_log, alt, values, marker) -> None:
    lines = _notice_lines(tmp_path_factory, coded_log, alt, solve_values(invest=["p2025"]) + values, marker=marker)
    assert lines == []


def test_notice_is_registered() -> None:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    assert NOTICE_ID in BEHAVIOUR_CHANGES


def _lead_time_annuity_step(tmp_path_factory, alt: str, payback: list | None):
    """The step of wind_plant decided in p2020 with a 5-year construction lead time (online in p2025),
    lifetime_technical {p2020: 3, p2025: 20, p2030: 20}, and ``payback`` (an encoded value) as its payback_time
    (none when None)."""
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
        values.append(["unit", WIND, "payback_time", payback])
    (step,) = steps_of(tmp_path_factory, alt, values).values()
    return step


def _at_decision(frame: pl.DataFrame, d: str = "p2020") -> float:
    """The value of wind_plant's decision ``d`` in ``frame`` (``[e, d, value]``)."""
    rows = frame.filter((pl.col("e").cast(pl.Utf8) == WIND) & (pl.col("d").cast(pl.Utf8) == d))
    (value,) = rows["value"].to_list()
    return float(value)


def _lead_time_annuity(step) -> tuple[float, float]:
    """``(the price per year in service, ed_entity_annual_discounted)`` of wind_plant's decision p2020 in ``step``."""
    fd = step.flex_data
    return _price(fd), _at_decision(fd.ed_entity_annual_discounted.frame)


def _price(fd) -> float:
    """The price per year in service a_cap − a_salv of wind_plant's investment cohort of p2020 (the cohort table:
    the plant as constructed, its period maps read at its online period)."""
    rows = fd.cohort.filter((pl.col("e").cast(pl.Utf8) == WIND) & (pl.col("cohort") == "invest:p2020"))
    (value,) = (rows["a_cap"] - rows["a_salv"]).to_list()
    return float(value)


def test_lead_time_annuity_without_payback_time_takes_the_lifetime_of_its_online_period(tmp_path_factory) -> None:
    """Without payback_time the annuity of wind_plant (decided in p2020, online in p2025) is over the
    lifetime_technical of its online period, 20 years (not that of p2020, 3): the period map describes the plant as
    constructed, as for its end of life.  The same as with payback_time 20.  By hand:
    1.05^-20 = 0.3768895, factor 0.05 / (1 − 0.3768895) = 0.0802426, annuity 1000 · 1000 · 0.0802426 / 1.05 (paid
    at the start of each year, the default discount_offset_investment 0) = 76 421.51 CUR/MW/a; reinvest_choice pays
    it for 20 years from y_on 5, of which the 15 to the horizon end 20 are in the model (no inflation): 15 · 76 421.51
    = 1 146 322.67.  (Over 3 years: 349 722.44 a year for 3 years.)"""
    ann, discounted = _lead_time_annuity(_lead_time_annuity_step(tmp_path_factory, "lead_annuity_no_payback", None))
    twin = _lead_time_annuity(_lead_time_annuity_step(tmp_path_factory, "lead_annuity_payback_20",
                                                      b64(20.0, "float")))
    assert ann == pytest.approx(76_421.51, rel=1e-7)
    assert discounted == pytest.approx(1_146_322.67, rel=1e-7)
    assert (ann, discounted) == pytest.approx(twin, rel=1e-12)


def test_payback_time_copy_of_the_lifetime_is_the_empty_payback_time(tmp_path_factory) -> None:
    """A payback_time equal to lifetime_technical (the period map {p2020: 3, p2025: 20, p2030: 20}, as a database
    migrated by an earlier development build of v81 holds it) builds the same annuity, lumps and LP as an empty
    payback_time, with a construction lead time and period-varying values: both are read at the period the
    capacity comes online."""
    empty = _lead_time_annuity_step(tmp_path_factory, "lead_payback_empty", None)
    copy = _lead_time_annuity_step(tmp_path_factory, "lead_payback_copy",
                                   _map({"p2020": 3.0, "p2025": 20.0, "p2030": 20.0}))
    assert _lead_time_annuity(copy) == pytest.approx(_lead_time_annuity(empty), rel=1e-12)
    _same_lp(empty, copy)


def test_lead_time_investment_reads_every_period_map_at_its_online_period(tmp_path_factory) -> None:
    """wind_plant decided in p2020 (y 0) with a 5-year construction lead time comes online in p2025 (y_on 5).  Its
    period maps differ between the decision and the online period: invest_cost {p2020: 2000, p2025: 1000, p2030:
    1500} CUR/kW, payback_time {10, 4, 10}, lifetime_technical {3, 7, 30}; discount_rate 0.05, fixed_cost 10
    CUR/kW/a, end_of_life_method next_seam, reinvest_choice, no inflation.  The maps describe the plant as
    constructed: every value is that of p2025.  By hand:

    * annuity = invest_cost · 1000 · r / (1 − (1+r)^-n) / (1 + r) = 1000 · 1000 · 0.05 / (1 − 1.05^-4) / 1.05 =
      268 582.70 CUR/MW/a, paid back over payback_time 4 years at the start of each year (the default
      discount_offset_investment 0);
    * end of life Y = 5 + 7 = 12 → next_seam 20: the cohort (y_life 7, y_eol 20) is in service in p2025 and
      p2030 (5 + 10 = 15 years), the fixed cost ``10 · 1000 · (inflation_op[p2025] + inflation_op[p2030])``
      (``ed_lifetime_fixed_cost``);
    * the level price over lifetime_technical 7 (no inflation: G(x) = x) is a = 268 582.70 · 4 / 7 = 153 475.83
      CUR/MW a year in service (the cohort's price; before: the annuity for 4 of p2025's 5 years), so the lump
      ``ed_entity_annual_discounted`` (and its run-level twin ``ed_entity_annual_discounted_run``: the solve is the
      run) = 15 · a = 2 302 137.41 (before: the 4 payback years, 4 · 268 582.70 = 1 074 330.79), the reported
      investment at its start of operation p2025.

    (At the decision period: annuity 2000 · 1000 · 0.05 / (1 − 1.05^-10) over 10 years, end of life 0 + 5 + 3 =
    8 → 10, in service and paying fixed cost in p2025 only.)"""
    values = solve_values(invest=["p2020"]) + [
        ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", WIND, "invest_method", b64("invest_period", "str")],
        ["unit", WIND, "invest_cost", _map({"p2020": 2000.0, "p2025": 1000.0, "p2030": 1500.0})],
        ["unit", WIND, "payback_time", _map({"p2020": 10.0, "p2025": 4.0, "p2030": 10.0})],
        ["unit", WIND, "lifetime_technical", _map({"p2020": 3.0, "p2025": 7.0, "p2030": 30.0})],
        ["unit", WIND, "discount_rate", b64(0.05, "float")],
        ["unit", WIND, "invest_max_period", _map({"p2025": 100.0})],
        ["unit", WIND, "construction_lead_time", b64(5.0, "float")],
        ["unit", WIND, "end_of_life_method", b64("next_seam", "str")],
        ["unit", WIND, "fixed_cost", b64(10.0, "float")]]
    (step,) = steps_of(tmp_path_factory, "lead_online_maps", values).values()
    fd = step.flex_data
    ann = 1000.0 * 1000.0 * 0.05 / (1.0 - 1.05 ** -4) / 1.05
    assert ann == pytest.approx(268_582.70, rel=1e-8)
    price = ann * 4.0 / 7.0
    assert price == pytest.approx(153_475.83, abs=0.01)
    assert _price(fd) == pytest.approx(price, rel=1e-12)
    assert 15.0 * price == pytest.approx(2_302_137.41, abs=0.01)
    lines = fd.edl_invest_lines_run.filter((pl.col("e") == WIND) & (pl.col("d") == "p2020"))
    assert {(li, d): v for li, d, v in lines.select("line", "d_report", "value").iter_rows()} == pytest.approx(
        {("investment", "p2025"): 15.0 * price}, rel=1e-12)
    assert _at_decision(fd.ed_entity_annual_discounted.frame) == pytest.approx(15.0 * price, rel=1e-12)
    assert _at_decision(fd.ed_entity_annual_discounted_run.frame) == pytest.approx(15.0 * price, rel=1e-12)
    (cohort,) = fd.cohort.filter((pl.col("kind") == "invest") & (pl.col("e").cast(pl.Utf8) == WIND)).iter_rows(
        named=True)
    assert (str(cohort["d_on"]), cohort["y_on"], cohort["y_life"], cohort["y_eol"]) == ("p2025", 5.0, 7.0, 20.0)
    assert invest_alive(step) == [("p2020", "p2025"), ("p2020", "p2030")]
    op = {str(d): float(v) for d, v in fd.p_inflation_op.frame.select(pl.col("d").cast(pl.Utf8), "value").iter_rows()}
    fixed = 10.0 * 1000.0 * (op["p2025"] + op["p2030"])
    assert _at_decision(fd.ed_lifetime_fixed_cost.frame) == pytest.approx(fixed, rel=1e-12)


@pytest.mark.parametrize(("cost", "named"), [
    ({"p2020": 0.0, "p2025": 1000.0, "p2030": 1000.0}, False),
    ({"p2020": 1000.0, "p2025": 0.0, "p2030": 0.0}, True),
], ids=["cost_online", "no_cost_online"])
def test_zero_cost_notice_reads_the_cost_of_the_online_period(tmp_path_factory, coded_log, cost, named) -> None:
    """With the ``invest_without_cost_offered`` marker, wind_plant decided in p2020 with a 5-year construction lead
    time comes online in p2025 and pays the invest_cost of p2025 (the plant as constructed): an invest_cost map
    with 0 only in p2020 has a cost, so wind_plant is not named; a map with 0 from p2025 on has none, so it is named
    with its decision period p2020.  (Before, the notice read the decision period's cost: the other way round.)"""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import MARKER_PREFIX
    from tests.engine._unit_db import make_db

    values = [model_solves([SOLVE])] + solve_values(invest=["p2020"]) + [
        ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
        ["unit", WIND, "invest_method", b64("invest_period", "str")],
        ["unit", WIND, "invest_cost", _map(cost)],
        ["unit", WIND, "invest_max_period", _map({"p2025": 100.0})],
        ["unit", WIND, "lifetime_technical", b64(20.0, "float")],
        ["unit", WIND, "construction_lead_time", b64(5.0, "float")]]
    alt = f"zero_cost_online_{named}"
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=CHAIN,
                        entities=[["solve", SOLVE, None]])
    with DatabaseMapping(url) as db:
        db.add_item("metadata", name=MARKER_PREFIX + "invest_without_cost_offered", value="{}")
        db.commit_session("behaviour-change marker")
    coded_log()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    lines = [ln for ln in coded_log.notices() if "invest_without_cost_offered" in ln]
    listed = [ln for ln in lines if f"{WIND} (p2020)" in ln]
    assert bool(listed) == named, lines
