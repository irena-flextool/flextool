"""The investment cost of ``reinvest_choice`` / ``no_reinvestment`` capacity is the level yearly cost of a plant of
its real lifetime, charged for each year its cohort is in service inside the horizon.

Fixture ``tests.json`` (built from JSON), single solve ``y2020_2035_5week``: periods p2020 … p2035 of 5 years each,
run years 0, 5, 10, 15, horizon 20.  coal_plant: invest_cost 400 CUR/kW, discount_rate 0.05, ``invest_no_limit``,
fixed_cost 0, end_of_life_method the default closest_seam.  The model sets ``social_discount_rate`` 0.05 and
``discount_offset_investment`` 1 (both explicitly: the defaults are 0 and 0), so the investment factor of year k
(0-based, from the run start) is 1.05^-(k+1).

Hand derivation.  The annuity over n years is A(n) = 400 · 1000 · r / (1 − (1+r)^-n) (r = discount_rate).  The
price per year in service is a = A(n) · G(n) / G(L), G(x) = Σ_{k<⌈x⌉} min(1, x − k) · 1.05^-k (L =
lifetime_technical): the loan's present value spread evenly over the real lifetime.  With r = 0.05 = the inflation
rate, G(n) = (1 − 1.05^-n) / (1 − 1.05^-1), so a = A(L) for every n.  The capacity of a decision is in service from
its start of operation y_on to its end of life y_on + L rounded to the closest period start (one of 0, 5, 10, 15,
20; past the horizon: 20).  With S(a, b) = Σ_{j=a..b} 1.05^-j (year k in service ↔ j = k + 1) and A12 = 45 130.164:

====================  ==================  ============================  ======================================
decision, L = n = 12  years in service    ed_entity_annual_discounted   before (the payback window, 12 years)
====================  ==================  ============================  ======================================
p2020                 0 … 9 (12 → 10)     A12 · S(1, 10) = 348 483.16   A12 · S(1, 12) = 400 000
p2025                 5 … 14 (17 → 15)    A12 · S(6, 15) = 273 045.68   400 000 · 1.05^-5 = 313 410.47
p2030                 10 … 19 (22 → 20)   A12 · S(11, 20)               unchanged
p2035                 15 … 19             A12 · S(16, 20)               unchanged
====================  ==================  ============================  ======================================
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

from tests.engine._unit_db import BASE_FIXTURE_JSON, PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import MEGA, _no_battery_invest, _results
from tests.engine.test_intraperiod_blocks import _chain
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

PERIODS = ["p2020", "p2025", "p2030", "p2035"]
R = 1.05


def _s(a: int, b: int) -> float:
    """Σ_{j=a..b} 1.05^-j: the investment factors of years a−1 … b−1."""
    return sum(R ** -j for j in range(a, b + 1))


def _annuity(n: float, r: float = 0.05) -> float:
    return 400.0 * 1000.0 * r / (1.0 - (1.0 + r) ** -n)


def _g(x: float) -> float:
    """G(x) = Σ_{k<⌈x⌉} min(1, x − k) · 1.05^-k, written out year by year."""
    total, k = 0.0, 0
    while k < x:
        total += min(1.0, x - k) * R ** -k
        k += 1
    return total


def _price(n: float, life: float, r: float = 0.05) -> float:
    """a = A(n) · G(n) / G(L): the price per year in service."""
    return _annuity(n, r) * _g(n) / _g(life)


A12 = _annuity(12.0)

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _period_map(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _economics(method: str = "reinvest_choice", technical: float = 12.0, payback=12.0,
               invest_method: str = "invest_no_limit", rate: float = 0.05,
               end_of_life: str | None = None) -> list:
    payback_value = _period_map(payback) if isinstance(payback, dict) else b64(float(payback), "float")
    out = [["model", "flexTool", "social_discount_rate", b64(0.05, "float")],
           ["model", "flexTool", "discount_offset_investment", b64(1.0, "float")],
           ["unit", PLANT, "invest_cost", b64(400.0, "float")],
           ["unit", PLANT, "discount_rate", b64(rate, "float")],
           ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
           ["unit", PLANT, "invest_method", b64(invest_method, "str")],
           ["unit", PLANT, "lifetime_method", b64(method, "str")],
           ["unit", PLANT, "lifetime_technical", b64(technical, "float")],
           ["unit", PLANT, "payback_time", payback_value]]
    if end_of_life is not None:
        out.append(["unit", PLANT, "end_of_life_method", b64(end_of_life, "str")])
    return out


def _step(tmp_path_factory, alt: str, values: list, chain: list[str] | None = None):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values,
                            base_chain=_no_battery_invest(SINGLE) if chain is None else chain)
    return list(steps.values())[-1]


def _ed(param, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


EXPECTED_12 = {"p2020": A12 * _s(1, 10), "p2025": A12 * _s(6, 15), "p2030": A12 * _s(11, 20),
               "p2035": A12 * _s(16, 20)}


@pytest.mark.solver
@pytest.mark.parametrize("method", ["reinvest_choice", "no_reinvestment"])
def test_level_price_for_each_year_in_service(tmp_path_factory, method) -> None:
    """lifetime_technical = payback_time = 12 (table above): the price is A12 a year, and the closest period start
    ends the life of the p2020 decision at 10 and of the p2025 decision at 15, so they pay 10 years, A12 · S(1, 10)
    = 348 483.16 and A12 · S(6, 15) = 273 045.68 (before: the 12 payback years, 400 000 and 313 410.47, also the
    years after the plant was gone); the later ones are cut by the horizon.  no_reinvestment allows investment
    only within the first technical lifetime (decisions p2020, p2025, p2030), at the same values."""
    got = _ed(_step(tmp_path_factory, f"pb_{method}", _economics(method)).flex_data.ed_entity_annual_discounted)
    assert EXPECTED_12["p2020"] == pytest.approx(348_483.16, abs=0.01)
    assert EXPECTED_12["p2025"] == pytest.approx(273_045.68, abs=0.01)
    periods = PERIODS if method == "reinvest_choice" else PERIODS[:3]
    for d in periods:
        assert got[d] == pytest.approx(EXPECTED_12[d], rel=1e-12), d


@pytest.mark.solver
def test_reinvest_automatic_pays_to_the_horizon(tmp_path_factory) -> None:
    """reinvest_automatic keeps paying the annuity to the end of the horizon: the p2020 decision pays years 0 … 19,
    A12 · S(1, 20) = 562 421.60."""
    got = _ed(_step(tmp_path_factory, "pb_automatic", _economics("reinvest_automatic")).flex_data
              .ed_entity_annual_discounted)
    assert got["p2020"] == pytest.approx(A12 * _s(1, 20), rel=1e-12)
    assert got["p2020"] == pytest.approx(562_421.5967, rel=1e-9)


@pytest.mark.solver
@pytest.mark.parametrize("technical", [30.0, 5.0], ids=["technical_30", "technical_5"])
def test_price_is_over_the_technical_lifetime(tmp_path_factory, technical) -> None:
    """payback_time 12 with lifetime_technical 30 or 5, discount_rate = social_discount_rate: the price is a =
    A12 · G(12) / G(L) = A(L) (A30 = 26 020.57, A5 = 92 389.92), paid for the years in service: with 30 every year
    to the horizon (S(y_on + 1, 20)), with 5 the decision's own period only (S(y_on + 1, y_on + 5)).  (Before: the
    12 payback years, 400 000 for the p2020 decision with either lifetime.)"""
    got = _ed(_step(tmp_path_factory, f"pb_technical_{technical:g}", _economics(technical=technical)).flex_data
              .ed_entity_annual_discounted)
    a = _price(12.0, technical)
    assert a == pytest.approx(_annuity(technical), rel=1e-12)
    assert a == pytest.approx({30.0: 26_020.57, 5.0: 92_389.92}[technical], abs=0.01)
    for d, y_on in zip(PERIODS, (0, 5, 10, 15)):
        end = 20 if technical == 30.0 else y_on + 5
        assert got[d] == pytest.approx(a * _s(y_on + 1, end), rel=1e-12), d


@pytest.mark.solver
@pytest.mark.parametrize(("rate", "payback", "want"), [
    (0.03, 20.0, 243_347.43), (0.03, 40.0, 215_657.94),
    (0.05, 20.0, 290_509.88), (0.05, 40.0, 290_509.88),
    (0.08, 20.0, 368_745.25), (0.08, 40.0, 418_033.19),
])
def test_payback_time_sets_the_size_of_the_price(tmp_path_factory, rate, payback, want) -> None:
    """lifetime_technical 40 (in service from p2020 to the horizon, years 0 … 19), payback_time 20 or 40,
    discount_rate 3, 5 or 8 %: the p2020 decision costs a · S(1, 20), a = A(n) · G(n) / G(40).  With the discount
    rate below the inflation rate (3 %) the shorter payback time is dearer (243 347.43 > 215 657.94), at 5 % both
    are A40 · S(1, 20) = 290 509.88, above it (8 %) cheaper (368 745.25 < 418 033.19).  (Before, payback 20: the 20
    years of the horizon at A(20): 335 062.51, 400 000, 507 721.46.)"""
    got = _ed(_step(tmp_path_factory, f"pb_size_{rate:g}_{payback:g}",
                    _economics(technical=40.0, payback=payback, rate=rate)).flex_data.ed_entity_annual_discounted)
    assert _price(payback, 40.0, rate) * _s(1, 20) == pytest.approx(want, abs=0.01)
    assert got["p2020"] == pytest.approx(_price(payback, 40.0, rate) * _s(1, 20), rel=1e-12)


A7 = _annuity(7.0)
A40 = _annuity(40.0)


@pytest.mark.solver
@pytest.mark.parametrize(("technical", "end_of_life", "want"), [
    (7.0, "closest_seam", (299_287.75, 234_499.78, 183_736.72, 143_962.52)),
    (7.0, "next_seam", (533_787.53, 418_236.50, 327_699.24, 143_962.52)),
    (40.0, "closest_seam", (290_509.88, 189_584.31, 110_506.48, 48_546.93)),
])
def test_end_of_life_method_sets_the_years_not_the_price(tmp_path_factory, technical, end_of_life, want) -> None:
    """lifetime_technical = payback_time = 7 or 40: the price is A7 = 69 127.93 (A40 = 23 311.26) a year whatever
    the rounding; end_of_life_method sets the years in service.  L 7, closest: ends 7, 12, 17, 22 → 5, 10, 15, 20
    (each decision its own period); next_seam → 10, 15, 20 and +inf (two periods each, the p2035 one to the
    horizon).  L 40: every decision to the horizon.  (Before, L 7: the 7 payback years, 400 000 for the p2020
    decision under both methods.)"""
    got = _ed(_step(tmp_path_factory, f"pb_eol_{technical:g}_{end_of_life}",
                    _economics(technical=technical, payback=technical, end_of_life=end_of_life)).flex_data
              .ed_entity_annual_discounted)
    a = A7 if technical == 7.0 else A40
    assert (A7, A40) == (pytest.approx(69_127.93, abs=0.01), pytest.approx(23_311.26, abs=0.01))
    if technical == 40.0:
        ends = (20, 20, 20, 20)
    elif end_of_life == "closest_seam":
        ends = (5, 10, 15, 20)
    else:
        ends = (10, 15, 20, 20)
    for d, y_on, end, value in zip(PERIODS, (0, 5, 10, 15), ends, want):
        assert a * _s(y_on + 1, end) == pytest.approx(value, abs=0.01), d
        assert got[d] == pytest.approx(a * _s(y_on + 1, end), rel=1e-12), d


@pytest.mark.solver
def test_payback_time_of_the_online_period(tmp_path_factory) -> None:
    """A payback_time map {p2020: 12, p2025: 7, p2030: 12, p2035: 12}, lifetime_technical 12: the p2025 decision
    pays back over 7 years, a = A(7) · G(7) / G(12), which with discount_rate = social_discount_rate is A12: the
    payback time does not change the cost.  It is in service from 5 to 17 → 15: A12 · S(6, 15) = 273 045.68 (before: the 7
    payback years, A(7) · S(6, 12) = 400 000 · 1.05^-5)."""
    payback = {"p2020": 12.0, "p2025": 7.0, "p2030": 12.0, "p2035": 12.0}
    got = _ed(_step(tmp_path_factory, "pb_map", _economics(payback=payback)).flex_data.ed_entity_annual_discounted)
    assert _price(7.0, 12.0) == pytest.approx(A12, rel=1e-12)
    assert got["p2025"] == pytest.approx(_price(7.0, 12.0) * _s(6, 15), rel=1e-12)
    assert got["p2025"] == pytest.approx(EXPECTED_12["p2025"], rel=1e-12)
    assert got["p2020"] == pytest.approx(EXPECTED_12["p2020"], rel=1e-12)


@pytest.mark.solver
def test_fractional_payback_time(tmp_path_factory) -> None:
    """payback_time 12.5, lifetime_technical 12: a = A(12.5) · G(12.5) / G(12), G(12.5) = Σ_{k<12} 1.05^-k + 0.5 ·
    1.05^-12; the p2020 decision is in service 10 years: a · S(1, 10) = 348 359.74 (before: the payback years 0 …
    11 and half of 12, A(12.5) · (S(1, 12) + 0.5 · 1.05^-13) = 399 858.33)."""
    got = _ed(_step(tmp_path_factory, "pb_half", _economics(payback=12.5)).flex_data.ed_entity_annual_discounted)
    want = _price(12.5, 12.0) * _s(1, 10)
    assert got["p2020"] == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(348_359.74, abs=0.01)


@pytest.mark.solver
def test_fractional_period_and_lifetime(tmp_path_factory) -> None:
    """p2020 represents 2.5 years (rows of width 1, 1, 0.5 at years 0, 1, 2; the last discounted 0.5 · 1.05^-(2 +
    0.5)), so p2025 starts at 2.5; payback_time = lifetime_technical = 2.25: the end of life 2.25 is closer to 2.5
    than to 0, so the p2020 decision is in service in the whole of p2020 at the price A(2.25):
    A(2.25) · (1.05^-1 + 1.05^-2 + 0.5 · 1.05^-2.5) = 442 832.59 (before: the payback years 0, 1 and half of the
    half-year row, 400 262.77)."""
    years = _period_map({"p2020": 2.5, "p2025": 5.0, "p2030": 5.0, "p2035": 5.0})
    got = _ed(_step(tmp_path_factory, "pb_frac_period", _economics(technical=2.25, payback=2.25) + [
        ["solve", "y2020_2035_5week", "years_represented", years]]).flex_data.ed_entity_annual_discounted)
    want = _annuity(2.25) * (R ** -1 + R ** -2 + 0.5 * R ** -2.5)
    assert got["p2020"] == pytest.approx(want, rel=1e-12)
    assert want == pytest.approx(442_832.59, abs=0.01)


@pytest.mark.solver
def test_lead_time_prices_the_years_from_the_start_of_operation(tmp_path_factory) -> None:
    """construction_lead_time 5: the p2020 decision starts to operate in p2025 (y_on 5, end of life 17 → 15) and
    pays A12 for years 5 … 14, the value of a p2025 decision without a lead time, A12 · S(6, 15) (before: its 12
    payback years, 400 000 · 1.05^-5)."""
    got = _ed(_step(tmp_path_factory, "pb_lead", _economics() + [
        ["unit", PLANT, "construction_lead_time", b64(5.0, "float")]]).flex_data.ed_entity_annual_discounted)
    assert got["p2020"] == pytest.approx(EXPECTED_12["p2025"], rel=1e-12)


@pytest.mark.solver
def test_unequal_periods_pay_the_same_price_per_year(tmp_path_factory) -> None:
    """A solve of five periods starting at 0, 5, 10, 20, 30 (years_represented 5, 5, 10, 10, 10; horizon 40),
    lifetime_technical = payback_time = 7, closest_seam: the ends 7, 12, 17, 27, 37 round to 5, 10, 20, 30, 40, so
    every decision is in service in its own period, 5, 5, 10, 10 and 10 years, and pays A7 = 69 127.93 for each:
    A7 · S(1, 5) = 299 287.75, A7 · S(6, 10) = 234 499.78, A7 · S(11, 20) = 327 699.24, A7 · S(21, 30) =
    201 178.91, A7 · S(31, 40) = 123 506.40.  (Before: the 7 payback years, so 1.3365 × A7 a year in service for
    the decisions of 5-year periods and 0.7494 × A7 for those of 10-year periods.)"""
    from tests.engine.test_end_of_life_method import CHAIN as LIFE_CHAIN, model_solves, solve_values
    periods = ["p2020", "p2025", "p2030", "p2040", "p2050"]
    values = [model_solves(["life_L"])] + solve_values(periods=periods, years=[5.0, 5.0, 10.0, 10.0, 10.0],
                                                       invest=periods)
    values += _economics(technical=7.0, payback=7.0)
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="pb_unequal", values=values, base_chain=LIFE_CHAIN,
                            entities=[["solve", "life_L", None]])
    got = _ed(list(steps.values())[-1].flex_data.ed_entity_annual_discounted)
    spans = ((1, 5), (6, 10), (11, 20), (21, 30), (31, 40))
    want = (299_287.75, 234_499.78, 327_699.24, 201_178.91, 123_506.40)
    for d, (a, b), value in zip(periods, spans, want):
        assert A7 * _s(a, b) == pytest.approx(value, abs=0.01), d
        assert got[d] == pytest.approx(A7 * _s(a, b), rel=1e-12), d


FIXED_COST_MAP = {"p2020": 10.0, "p2025": 20.0, "p2030": 30.0, "p2035": 40.0}


def _with_fixed_cost(fixed) -> list:
    value = _period_map(fixed) if isinstance(fixed, dict) else b64(float(fixed), "float")
    return [v for v in _economics() if v[2] != "fixed_cost"] + [
        ["model", "flexTool", "discount_offset_operations", b64(0.5, "float")],
        ["unit", PLANT, "fixed_cost", value]]


def _s_op(a: int, b: int) -> float:
    """Σ_{k=a..b−1} 1.05^-(k + 0.5): the operations factors of years a … b−1 (offset 0.5)."""
    return sum(R ** -(k + 0.5) for k in range(a, b))


@pytest.mark.solver
def test_fixed_cost_of_each_period_in_service(tmp_path_factory) -> None:
    """fixed_cost map {p2020: 10, p2025: 20, p2030: 30, p2035: 40} CUR/kW/a, lifetime_technical = payback_time = 12
    (in service 0 … 9, 5 … 14, 10 … 19, 15 … 19), discount_offset_operations 0.5 (set explicitly), so the
    operations factor of year k is 1.05^-(k + 0.5): S_op(0, 5) = 4.436393, S_op(5, 10) = 3.476030.  New capacity
    pays the fixed cost of each period it is in service, as existing capacity does:
    ``ed_lifetime_fixed_cost`` = 1000 · Σ fixed_cost[d′] · S_op(d′) = 113 884.54 (p2020: 10 · S_op(0, 5) + 20 ·
    S_op(5, 10)), 151 227.43 (p2025), 167 066.07 (p2030), 85 359.24 (p2035).  (FlexTool 4.5.0: the fixed cost of the
    period it comes online for every period, 79 124.24, 123 991.82, 145 726.25, 85 359.24.)"""
    got = _ed(_step(tmp_path_factory, "pb_fixed_map", _with_fixed_cost(FIXED_COST_MAP)).flex_data
              .ed_lifetime_fixed_cost)
    starts = {"p2020": 0, "p2025": 5, "p2030": 10, "p2035": 15}
    ends = {"p2020": 10, "p2025": 15, "p2030": 20, "p2035": 20}
    want = {"p2020": 113_884.54, "p2025": 151_227.43, "p2030": 167_066.07, "p2035": 85_359.24}
    assert (_s_op(0, 5), _s_op(5, 10)) == (pytest.approx(4.436393, abs=1e-6), pytest.approx(3.476030, abs=1e-6))
    for d in PERIODS:
        by_hand = 1000.0 * sum(FIXED_COST_MAP[p] * _s_op(y, y + 5) for p, y in starts.items()
                               if starts[d] <= y < ends[d])
        assert by_hand == pytest.approx(want[d], abs=0.01), d
        assert got[d] == pytest.approx(by_hand, rel=1e-12), d


@pytest.mark.solver
def test_constant_fixed_cost_map_is_the_constant(tmp_path_factory) -> None:
    """A fixed_cost map with 25 in every period builds the lumps and the LP of the constant 25."""
    from tests.engine.test_construction_lead_time_calendar import _same_lp
    constant = _step(tmp_path_factory, "pb_fixed_const", _with_fixed_cost(25.0))
    mapped = _step(tmp_path_factory, "pb_fixed_const_map", _with_fixed_cost(dict.fromkeys(PERIODS, 25.0)))
    for name in ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost"):
        assert _ed(getattr(mapped.flex_data, name)) == _ed(getattr(constant.flex_data, name)), name
    _same_lp(constant, mapped)


FIXED_NOTICE_ID = "fixed_cost_per_operating_period"


@pytest.mark.solver
def test_fixed_cost_notice_names_entities_whose_fixed_cost_changes(tmp_path_factory, capsys) -> None:
    """With the upgrade's marker naming coal_plant, the fixed_cost map above changes its lifetime fixed cost and the
    notice names it once; a map with the same value in every period changes nothing and prints no notice."""
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, MARKER_PREFIX
    from tests.engine._unit_db import make_db

    def lines(alt: str, fixed) -> list[str]:
        url, scen = make_db(tmp_path_factory, alt=alt, values=_with_fixed_cost(fixed),
                            base_chain=_no_battery_invest(SINGLE))
        with DatabaseMapping(url) as db:
            db.add_item("metadata", name=MARKER_PREFIX + FIXED_NOTICE_ID,
                        value='{"since": 81, "entities": ["coal_plant"]}')
            db.commit_session("behaviour-change marker")
        capsys.readouterr()
        run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
        summary = BEHAVIOUR_CHANGES[FIXED_NOTICE_ID].summary
        return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]

    named = lines("pb_fixed_notice", FIXED_COST_MAP)
    assert len(named) == 1 and "Affected entities: coal_plant." in named[0], named
    assert lines("pb_fixed_notice_flat", dict.fromkeys(PERIODS, 25.0)) == []


def _seed(step, name: str) -> dict[str, float]:
    from flextool.engine.preprocess.params._derived import _provider_has_key, _provider_read

    path = Path("solve_data") / f"{name}.csv"
    provider = step.flex_data_provider
    assert _provider_has_key(provider, path), name
    frame = _provider_read(provider, path)
    frame = frame.filter(pl.col(frame.columns[0]) == PLANT)
    return {str(d): float(v) for d, v in zip(frame[frame.columns[1]].to_list(), frame[frame.columns[2]].to_list())}


SEED_CASES = {
    "payback_12": _economics(),
    "payback_map": _economics(payback={"p2020": 12.0, "p2025": 7.0, "p2030": 12.0, "p2035": 12.0}),
    "payback_12_5": _economics(payback=12.5),
    # A construction lead time with a fixed cost: both lumps start at the start of operation.
    "lead_time_fixed_cost": [v for v in _economics() if v[2] != "fixed_cost"] + [
        ["unit", PLANT, "fixed_cost", b64(100.0, "float")],
        ["unit", PLANT, "construction_lead_time", b64(5.0, "float")]],
}


@pytest.mark.solver
@pytest.mark.parametrize("case", list(SEED_CASES))
def test_solve_frames_of_the_lumps_are_the_objective_coefficients(tmp_path_factory, case) -> None:
    """The solve's ``solve_data/ed_entity_annual_discounted`` and ``solve_data/ed_lifetime_fixed_cost`` frames are
    the coefficients the objective uses.  Before, the CSV generation derived them a second time, anchored at the
    decision period: with a 5-year lead time the p2020 decision's frames summed the window from p2020 and the
    annuity of p2020, not from its start of operation p2025."""
    step = _step(tmp_path_factory, f"pb_seed_{case}", SEED_CASES[case])
    for name in ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost"):
        want = _ed(getattr(step.flex_data, name))
        got = _seed(step, name)
        assert set(want) <= set(got), name
        for d, v in want.items():
            assert got[d] == pytest.approx(v, rel=1e-12, abs=1e-9), (name, d)


NESTED = "multi_fullYear_battery_nested_multi_invest"


def _child_invests() -> list:
    """``storage_fullYear_6h`` (the child of ``invest_5weeks``) gets ``invest_5weeks``'s array invest periods: each
    nested anchor ``storage_fullYear_6h_<period>`` (not a Spine solve) then invests in its own period."""
    data = json.loads(BASE_FIXTURE_JSON.read_text())
    return [["solve", "storage_fullYear_6h", pv[2], pv[3]] for pv in data["parameter_values"]
            if pv[:2] == ["solve", "invest_5weeks"] and pv[2] in ("invest_periods", "realized_invest_periods")
            and pv[4] == "invest_oneYear4Times_5weeks"]


def _all_ed(param) -> dict[tuple[str, str], float]:
    frame = utf8(param.frame)
    return {(e, d): float(v) for e, d, v in frame.select("e", "d", "value").iter_rows()}


@pytest.mark.solver
def test_nested_sub_solve_pays_for_its_investments(tmp_path_factory) -> None:
    """A nested sub-solve that invests (``storage_fullYear_6h_<P>``, one period P, no Spine solve) prices its
    investments by the rule of a Spine solve.  Oracle: the split solves ``invest_5weeks_<P>`` (Spine solves of the
    reader, periods P and the next) over the same run calendar.  A decision in the last period of such a solve pays
    over that one period, the window of the child's decision; for P = p2020 the window additivity gives it (the p2020
    decision pays over p2020 and p2025, the p2025 decision over p2025; the costs are constant).  The payback times
    (10, 25, 30) end on period boundaries or after the horizon, so these windows are whole periods.  The
    sub-solve's ``solve_data`` frames are its coefficients.  Before: the sub-solve had neither lump, so its investments were
    free in the objective."""
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="pb_nested_child", values=_child_invests(),
                            base_chain=_chain(NESTED))
    nxt = dict(zip(PERIODS, PERIODS[1:]))
    prev = {b: a for a, b in nxt.items()}
    for P in PERIODS:
        child = steps[f"storage_fullYear_6h_{P}"]
        decisions = sorted((e, d) for e, d in utf8(child.flex_data.ed_invest_set).select("e", "d").iter_rows())
        assert decisions and {d for _, d in decisions} == {P}, (P, decisions)
        for name in ("ed_entity_annual_discounted", "ed_lifetime_fixed_cost"):
            param = getattr(child.flex_data, name)
            assert param is not None, (P, name)
            got = _all_ed(param)
            if P == "p2020":
                spine = _all_ed(getattr(steps["invest_5weeks_p2020"].flex_data, name))
                want = {(e, d): spine[(e, "p2020")] - spine[(e, "p2025")] for e, d in decisions}
            else:
                spine = _all_ed(getattr(steps[f"invest_5weeks_{prev[P]}"].flex_data, name))
                want = {k: spine[k] for k in decisions}
            for k, v in want.items():
                assert got[k] == pytest.approx(v, rel=1e-12, abs=1e-6), (P, name, k)
            assert any(got[k] > 0.0 for k in decisions), (P, name)
            if P == PERIODS[-1]:
                # The anchors of one level share a Provider: after the run its frames are the last anchor's.
                assert _seed(child, name)[P] == pytest.approx(got[(PLANT, P)], rel=1e-12), (P, name)


PIN_100_IN_P2020 = {"p2020": 100.0, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0}


def _pinned(technical: float) -> list:
    return _economics(technical=technical, invest_method="invest_period") + [
        ["unit", PLANT, "invest_min_period", _period_map(PIN_100_IN_P2020)],
        ["unit", PLANT, "invest_max_period", _period_map(PIN_100_IN_P2020)]]


def _yearly_investment(results: dict[str, pd.DataFrame]) -> dict[str, float]:
    unit = results["cost_unit_annualized__d"]
    out = {}
    for d in PERIODS:
        rows = unit[(unit["period"] == d) & (unit["unit"] == PLANT)]
        out[d] = float(rows["investment"].sum()) * MEGA if len(rows) else 0.0
    return out


def _yearly_fixed_cost(results: dict[str, pd.DataFrame]) -> dict[str, float]:
    unit = results["cost_unit_annualized__d"]
    out = {}
    for d in PERIODS:
        rows = unit[(unit["period"] == d) & (unit["unit"] == PLANT)]
        out[d] = float(rows["fixed cost invested"].sum()) * MEGA if len(rows) else 0.0
    return out


@pytest.mark.solver
def test_yearly_investment_cost_for_each_year_in_service(tmp_path_factory) -> None:
    """100 MW decided in p2020 (pinned), payback_time 12 and lifetime_technical 5: the capacity is in service in
    p2020 only, at the price a = A12 · G(12) / G(5) = A(5) = 92 389.92 a year (discount_rate = social_discount_rate).
    The annualized investment cost is 100 · a = 9 238 991.93 a year in p2020 and 0 in the later periods, the
    yearly amount the objective charges for each year in service (before: 100 · A12 a year in p2020 and p2025 and
    2/5 of that in p2030, the 12 payback years, also after the capacity was gone)."""
    results = _results(tmp_path_factory, "pb_yearly_single", _pinned(5.0), _no_battery_invest(SINGLE))
    got = _yearly_investment(results)
    a = _price(12.0, 5.0)
    assert 100.0 * a == pytest.approx(9_238_991.93, abs=0.01)
    want = {"p2020": 100.0 * a, "p2025": 0.0, "p2030": 0.0, "p2035": 0.0}
    for d in PERIODS:
        assert got[d] == pytest.approx(want[d], rel=1e-7, abs=1e-6), d


@pytest.mark.solver
def test_chain_pays_the_years_in_service_of_the_run(tmp_path_factory) -> None:
    """The chain ``invest_solveSequence_5weeks`` (each solve holds its own period and the next) with 100 MW decided in
    p2020, payback_time = lifetime_technical = 12: the end of life 12 is rounded to 10, so the capacity is in
    service in p2020 and p2025.  The y2020 solve (p2020, p2025) prices the decision over its years 0 … 9: A12 ·
    S(1, 10) = 348 483.16 per MW, and so does the reported lump over the run (before: the payback years of the run,
    0 … 11, A12 · S(1, 12) = 400 000).  The investment is reported as one event at its start of operation, p2020:
    discounted 100 · A12 · S(1, 10), annualized that event ÷ p2020's investment factor S(1, 5) = 100 · A12 ·
    S(1, 10) / S(1, 5) = 8.049083 M, and 0 in the later periods (before: the yearly annuity, 100 · A12 in p2020 and
    p2025; earlier still 100 · A12 · 2/5 = 1.805 M in p2030 too)."""
    values = _pinned(12.0)
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="pb_chain_steps", values=values,
                            base_chain=_no_battery_invest(CHAIN))
    fd = steps["y2020_5week"].flex_data
    assert _ed(fd.ed_entity_annual_discounted)["p2020"] == pytest.approx(A12 * _s(1, 10), rel=1e-12)
    assert A12 * _s(1, 10) == pytest.approx(348_483.1638, rel=1e-9)
    assert _ed(fd.ed_entity_annual_discounted_run)["p2020"] == pytest.approx(A12 * _s(1, 10), rel=1e-12)
    got = _yearly_investment(_results(tmp_path_factory, "pb_yearly_chain", values, _no_battery_invest(CHAIN)))
    want = {"p2020": 100.0 * A12 * _s(1, 10) / _s(1, 5), "p2025": 0.0, "p2030": 0.0, "p2035": 0.0}
    assert want["p2020"] == pytest.approx(8_049_082.8, abs=1.0)
    for d in PERIODS:
        assert got[d] == pytest.approx(want[d], rel=1e-7, abs=1e-6), d


@pytest.mark.solver
def test_chain_reports_the_fixed_cost_of_each_period(tmp_path_factory) -> None:
    """The chain above with the fixed_cost map {p2020: 10, p2025: 20, p2030: 30, p2035: 40} CUR/kW/a (100 MW decided
    in p2020, in service in p2020 and p2025): the fixed cost invested is reported in each period it is paid,
    discounted 100 · 1000 · 10 · S_op(0, 5) = 4.436393 M in p2020 and 100 · 1000 · 20 · S_op(5, 10) = 6.952060 M in
    p2025, and the two add up to the lump the deciding solve's objective charges, 100 MW · ed_lifetime_fixed_cost
    = 100 · 113 884.54 (the y2020 solve holds p2020 and p2025, the plant's whole life).  Annualized, 100 · 1000 · 10
    = 1 M a year in p2020 and 2 M in p2025 (FlexTool 4.5.0: the p2020 value in both, 79 124.24 per MW and 1 M a
    year)."""
    values = [v for v in _pinned(12.0) if v[2] != "fixed_cost"] + [
        ["model", "flexTool", "discount_offset_operations", b64(0.5, "float")],
        ["unit", PLANT, "fixed_cost", _period_map(FIXED_COST_MAP)]]
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="pb_chain_fixed_steps", values=values,
                            base_chain=_no_battery_invest(CHAIN))
    lump = _ed(steps["y2020_5week"].flex_data.ed_lifetime_fixed_cost)["p2020"]
    assert lump == pytest.approx(113_884.54, abs=0.01)
    results = _results(tmp_path_factory, "pb_yearly_chain_fixed", values, _no_battery_invest(CHAIN))
    unit = results["cost_unit_discounted__d"]
    unit = unit[unit["unit"] == PLANT].set_index("period")["fixed cost invested"] * MEGA
    want = {"p2020": 100.0 * 1000.0 * 10.0 * _s_op(0, 5), "p2025": 100.0 * 1000.0 * 20.0 * _s_op(5, 10),
            "p2030": 0.0, "p2035": 0.0}
    assert (want["p2020"], want["p2025"]) == (pytest.approx(4.436393e6, abs=1.0), pytest.approx(6.952060e6, abs=1.0))
    for d in PERIODS:
        assert unit.get(d, 0.0) == pytest.approx(want[d], rel=1e-7, abs=1e-6), d
    assert unit.sum() == pytest.approx(100.0 * lump, rel=1e-7)
    got = _yearly_fixed_cost(results)
    want = {"p2020": 1.0e6, "p2025": 2.0e6, "p2030": 0.0, "p2035": 0.0}
    for d in PERIODS:
        assert got[d] == pytest.approx(want[d], rel=1e-7, abs=1e-6), d


NOTICE_ID = "annuity_over_payback_time"


def _notice_lines(tmp_path_factory, capsys, alt: str, values: list) -> list[str]:
    from spinedb_api import DatabaseMapping

    from flextool.engine import run_chain_from_db
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, MARKER_PREFIX
    from tests.engine._unit_db import make_db

    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    with DatabaseMapping(url) as db:
        db.add_item("metadata", name=MARKER_PREFIX + NOTICE_ID, value='{"since": 81, "entities": ["coal_plant"]}')
        db.commit_session("behaviour-change marker")
    capsys.readouterr()
    run_chain_from_db(url, scen, work_folder=tmp_path_factory.mktemp(f"{alt}_w"))
    summary = BEHAVIOUR_CHANGES[NOTICE_ID].summary
    return [ln for ln in capsys.readouterr().out.splitlines() if summary in ln]


@pytest.mark.solver
def test_notice_names_entities_whose_annuity_changes(tmp_path_factory, capsys) -> None:
    """With the upgrade's marker, coal_plant (lifetime_technical = payback_time = 12) is named once: before, its
    annuity was paid in every period that started before the end of the lifetime (p2020 decision: years 0 … 14),
    now for its 10 years in service.  With a lifetime of 10 (ending on a period start) the years and the price are
    those of before and no notice is printed.  Nor for reinvest_automatic, whose annuity is paid to the horizon
    whatever the lifetimes (here a payback time of 12 against a technical lifetime of 30)."""
    lines = _notice_lines(tmp_path_factory, capsys, "pb_notice", _economics())
    assert len(lines) == 1 and "coal_plant" in lines[0]
    assert _notice_lines(tmp_path_factory, capsys, "pb_notice_seam", _economics(technical=10.0, payback=10.0)) == []
    assert _notice_lines(tmp_path_factory, capsys, "pb_notice_automatic",
                         _economics(method="reinvest_automatic", technical=30.0)) == []


def test_notice_is_registered() -> None:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES

    assert NOTICE_ID in BEHAVIOUR_CHANGES
