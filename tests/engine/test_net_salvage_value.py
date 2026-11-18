"""The net salvage value: a level credit in the yearly price of new capacity; received in full when existing
capacity retires.

Fixture ``tests.json`` (built from JSON), single solve ``y2020_2035_5week``: periods p2020 … p2035 of 5 years each,
run years 0, 5, 10, 15, horizon 20.  The model sets ``social_discount_rate`` i = 0.05 and ``discount_offset_investment``
1, so the investment factor of year k (0-based) is 1.05^-(k+1) and a period's factor is the sum of its five years.
coal_plant: discount_rate r = 0.05 (= i), fixed_cost 0, net_salvage_value 200 CUR/kW (S = 200 000 CUR/MW),
lifetime_method reinvest_choice, existing 500 MW.  D(d) = 1.05^-y[d] discounts a one-off at the start of d.

Hand values (every sum written out year by year below, not the closed forms of the engine):

* new capacity (invest_cost 400, lifetime_technical 40, built p2020): its salvage at the end of its 40-year life is
  levelled over the 40 years at r: a_salv = S · 1.05^-40 · 0.05 / (1 − 1.05^-40) = 1 655.63 CUR/MW a year; the
  capital A = 400 000 · 0.05 / (1 − 1.05^-40) = 23 311.27; the price per year in service a_cap − a_salv =
  21 655.63, charged over its 20 years in the horizon: Σ_{j=1..20} 1.05^-j · 21 655.63 = 269 877.04.
* existing capacity, lifetime 40 (outlives the horizon): its investment is sunk; its salvage value is a level credit
  a_x = 1 655.63 over its 40 years from the run start.  Retiring a MW in p2025 receives S at the start of p2025 and
  gives back every credit inside the horizon: S · 1.05^-5 − a_x · Σ_{j=1..20} 1.05^-j = 156 705.23 − 20 632.84 =
  136 072.39 (the ``divest_salvage`` term; the objective subtracts it).
* existing capacity, lifetime 15 (ends inside the horizon, at the start of p2035): a_x over 15 years, whose credits
  sum to S · 1.05^-15 = 96 203.42 (the objective constant per MW with include_existing_fixed_cost: the salvage
  once); retiring in p2025: S · 1.05^-5 − 96 203.42 = 60 501.81, so constant and retirement together give the MW
  exactly −S · 1.05^-5 = −156 705.23.
* existing capacity, lifetime 30 and age 29, next_seam: the 1 remaining year is rounded to the period start at year
  5, so it serves 5 years; its credit is levelled over those 5 service years (not over the 1 exact year, which would
  pay 5 credits of a 1-year levelling, 4.4 times the salvage) and sums to S · 1.05^-5 = 156 705.23; retiring in
  p2020: S − 156 705.23 = 43 294.77.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, make_db, run, utf8
from tests.engine.test_chain_costs import _no_battery_invest, _period_map
from tests.engine.test_construction_lead_time_calendar import _same_lp
from tests.engine.test_user_constraint_capacity_terms import CHAIN, SINGLE

R = 1.05
S = 200.0 * 1000.0

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _s(a: int, b: int) -> float:
    """Σ_{j=a..b} 1.05^-j: the investment factors of years a−1 … b−1."""
    return sum(R ** -j for j in range(a, b + 1))


def _level(amount: float, years: int) -> float:
    """The level yearly amount, paid at the end of each of ``years`` years at r = 0.05, worth ``amount`` paid at the
    end of the last year: amount · 1.05^-years / Σ_{j=1..years} 1.05^-j."""
    return amount * R ** -years / _s(1, years)


def _values(lifetime: float, invest_method: str, *, salvage=None, extra: list | None = None) -> list:
    return ([["model", "flexTool", "social_discount_rate", b64(0.05, "float")],
             ["model", "flexTool", "discount_offset_investment", b64(1.0, "float")],
             ["unit", PLANT, "discount_rate", b64(0.05, "float")],
             ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
             ["unit", PLANT, "invest_method", b64(invest_method, "str")],
             ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
             ["unit", PLANT, "lifetime_technical", b64(lifetime, "float")],
             ["unit", PLANT, "net_salvage_value", b64(200.0, "float") if salvage is None else salvage]]
            + (extra or []))


def _step(tmp_path_factory, alt: str, values: list, chain=None):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values,
                            base_chain=_no_battery_invest(SINGLE if chain is None else chain))
    return list(steps.values())[-1]


def _plant(param) -> dict[str, float]:
    assert param is not None
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


@pytest.mark.solver
def test_salvage_lowers_the_yearly_price_of_new_capacity(tmp_path_factory) -> None:
    """New capacity, lifetime 40, built p2020 (module docstring): a_salv 1 655.63, price 21 655.63 a year in service,
    objective coefficient 269 877.04; the cohort carries the capital and salvage parts."""
    step = _step(tmp_path_factory, "nsv_built", _values(40.0, "invest_no_limit",
                                                        extra=[["unit", PLANT, "invest_cost", b64(400.0, "float")]]))
    a_salv = _level(S, 40)
    a_cap = 400.0 * 1000.0 / _s(1, 40)
    assert a_salv == pytest.approx(1_655.63, abs=0.01)
    assert a_cap - a_salv == pytest.approx(21_655.63, abs=0.01)
    assert (a_cap - a_salv) * _s(1, 20) == pytest.approx(269_877.04, abs=0.01)
    assert _plant(step.flex_data.ed_entity_annual_discounted)["p2020"] == pytest.approx(
        (a_cap - a_salv) * _s(1, 20), rel=1e-12)
    (cohort,) = step.flex_data.cohort.filter((pl.col("kind") == "invest") & (pl.col("d_build").cast(pl.Utf8) == "p2020")
                                             & (pl.col("e").cast(pl.Utf8) == PLANT)).iter_rows(named=True)
    assert cohort["a_cap"] == pytest.approx(a_cap, rel=1e-12)
    assert cohort["a_salv"] == pytest.approx(a_salv, rel=1e-12)
    assert cohort["pay"] == pytest.approx(a_cap, rel=1e-12)
    assert (cohort["n_pay"], cohort["r_loan"], cohort["nsv_eol"]) == (40.0, 0.05, S)


@pytest.mark.solver
def test_retiring_existing_capacity_that_outlives_the_horizon(tmp_path_factory) -> None:
    """Existing capacity, lifetime 40, retired in p2025: S · 1.05^-5 − a_x · Σ_{j=1..20} 1.05^-j = 136 072.39."""
    step = _step(tmp_path_factory, "nsv_existing_40", _values(40.0, "retire_no_limit"))
    want = S * R ** -5 - _level(S, 40) * _s(1, 20)
    assert want == pytest.approx(136_072.39, abs=0.01)
    assert _plant(step.flex_data.ed_divest_salvage)["p2025"] == pytest.approx(want, rel=1e-12)


@pytest.mark.solver
def test_existing_capacity_is_salvaged_once(tmp_path_factory) -> None:
    """Existing capacity, lifetime 15: its credits (the objective constant with include_existing_fixed_cost) sum to
    S · 1.05^-15 = 96 203.42 a MW, and retiring a MW in p2025 earns S · 1.05^-5 − 96 203.42 = 60 501.81, so the two
    give the MW exactly S · 1.05^-5.  The model constant of the 500 MW is −500 · 96 203.42."""
    from polar_high import Problem

    from flextool.engine import build_flextool

    step = _step(tmp_path_factory, "nsv_existing_15", _values(15.0, "retire_no_limit"))
    credits = sum(_plant(step.flex_data.ed_existing_salvage_credit).values())
    by_hand = sum(_level(S, 15) * R ** -j for j in range(1, 16))
    assert by_hand == pytest.approx(S * R ** -15, rel=1e-12)
    assert credits == pytest.approx(96_203.42, abs=0.01)
    assert credits == pytest.approx(by_hand, rel=1e-12)
    retire = _plant(step.flex_data.ed_divest_salvage)["p2025"]
    assert retire == pytest.approx(60_501.81, abs=0.01)
    assert credits + retire == pytest.approx(S * R ** -5, rel=1e-12)

    def objective(include: bool) -> float:
        pb = Problem()
        build_flextool(pb, step.flex_data, include_existing_fixed_cost=include)
        sol = pb.solve()
        assert sol.optimal
        return sol.obj

    assert objective(True) - objective(False) == pytest.approx(-500.0 * by_hand, rel=1e-9)


@pytest.mark.solver
@pytest.mark.parametrize("method", ["next_seam", "closest_seam"])
def test_credit_of_a_rounded_life_is_levelled_over_its_service_years(tmp_path_factory, method) -> None:
    """Existing capacity, lifetime 30 and age 29: 1 year remains.  next_seam rounds it to year 5, so it serves 5
    years: its credits over those 5 service years sum to S · 1.05^-5 = 156 705.23 (the salvage once), and retiring
    in p2020 earns S − 156 705.23 = 43 294.77.  closest_seam rounds it to the run start (year 1 is nearer 0 than 5):
    no service, no credit."""
    step = _step(tmp_path_factory, f"nsv_rounded_{method}", _values(30.0, "retire_no_limit", extra=[
        ["unit", PLANT, "age_at_model_start", b64(29.0, "float")],
        ["unit", PLANT, "end_of_life_method", b64(method, "str")]]))
    credit = step.flex_data.ed_existing_salvage_credit
    if method == "closest_seam":
        assert credit is None or not _plant(credit)
        return
    credits = sum(_plant(credit).values())
    assert credits == pytest.approx(S * R ** -5, rel=1e-12)
    assert credits == pytest.approx(156_705.23, abs=0.01)
    retire = _plant(step.flex_data.ed_divest_salvage)["p2020"]
    assert retire == pytest.approx(S - S * R ** -5, rel=1e-12)
    assert retire == pytest.approx(43_294.77, abs=0.01)


@pytest.mark.solver
@pytest.mark.parametrize("chain", ["single", "chain"])
def test_constant_salvage_value_equals_its_period_map(tmp_path_factory, chain) -> None:
    """A net_salvage_value map with 200 in every period builds the LP of the constant 200 (new and existing
    capacity, lifetime 15), in a single solve and in the chain y2020 … y2035."""
    base = SINGLE if chain == "single" else CHAIN
    extra = [["unit", PLANT, "invest_cost", b64(400.0, "float")]]
    constant = _RUNS.get(tmp_path_factory, alt=f"nsv_const_{chain}", base_chain=_no_battery_invest(base),
                         values=_values(15.0, "invest_retire_no_limit", extra=extra))[0]
    mapped = _RUNS.get(tmp_path_factory, alt=f"nsv_map_{chain}", base_chain=_no_battery_invest(base),
                       values=_values(15.0, "invest_retire_no_limit", extra=extra,
                                      salvage=_period_map(dict.fromkeys(["p2020", "p2025", "p2030", "p2035"],
                                                                        200.0))))[0]
    assert list(constant) == list(mapped)
    for name in constant:
        _same_lp(constant[name], mapped[name])


@pytest.mark.solver
def test_constant_salvage_value_equals_its_period_map_in_a_fan(tmp_path_factory) -> None:
    """The same in the two-period stochastic fan of ``stoch_two_period_invest.json`` (each branch period takes its
    anchor's value): the existing wind (100 MW, lifetime 40) may retire with a net_salvage_value 200 given as a
    constant or as a map over p2035 / p2040."""
    from tests.engine.test_chain_costs import FAN_JSON, FAN_VALUES

    def steps(alt: str, salvage) -> dict:
        values = FAN_VALUES + [["unit", "wind", "invest_method", b64("retire_no_limit", "str")],
                               ["unit", "wind", "lifetime_method", b64("reinvest_choice", "str")],
                               ["unit", "wind", "lifetime_technical", b64(40.0, "float")],
                               ["unit", "wind", "net_salvage_value", salvage]]
        url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=["init", "base", "stoch"],
                            fixture_json=FAN_JSON)
        return run(tmp_path_factory, url, scen)

    constant = steps("nsv_fan_const", b64(200.0, "float"))
    mapped = steps("nsv_fan_map", _period_map({"p2035": 200.0, "p2040": 200.0}))
    assert list(constant) == list(mapped)
    for name in constant:
        assert constant[name].flex_data.ed_divest_salvage is not None, name
        _same_lp(constant[name], mapped[name])
