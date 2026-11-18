"""Retirement of model-built capacity (#70): a built cohort can be retired; a retired MW pays its contract as it ran.

Fixture ``tests.json`` (built from JSON), single solve ``y2020_2035_5week``: periods p2020 … p2035 of 5 one-year rows
each, run years 0, 5, 10, 15, horizon 20.  The model sets ``social_discount_rate`` i = 0.05 and
``discount_offset_investment`` 1 (0 where stated), so a payment of service year k (0-based, from the run start)
weighs v^(k+off) with v = 1/1.05, and a one-off at the start of year t weighs D(t) = v^t.  coal_plant: existing 0
(unless stated), invest_cost 400 CUR/kW (C = 400 000 CUR/MW), fixed_cost 0, net_salvage_value 200 (S = 200 000),
lifetime_method reinvest_choice.

Every expected value is an explicit sum of the investor's contract, year by year (:func:`_contract`), never the
engine's closed forms:

* the loan: C repaid in n = payback_time years at r = discount_rate, a payment A = C·r/(1 − (1+r)^−n)·(1+r)^(off−1)
  falling off years into each year (worth C at r for every off);
* the level price of a kept plant: the loan's value at i spread over the lifetime L, less the salvage S received
  at the end of the lifetime levelled over the L years at r, one price a year in service (inside the horizon);
* a plant retired at the start of year m after its start pays the payments due before m, then at m the loan
  still owed (the payments of the years ≥ m discounted at r to m), and receives S_m at m; the level price it was
  charged is given back and the fixed cost from m on is saved.

So building and retiring at once costs (C − S)·D(m) whatever r, n, L and off are (check A).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, PLANT, RunCache, b64, build, cstr_over, make_db, run, utf8
from tests.engine.test_chain_costs import _no_battery_invest, _period_map, _solve, _solves
from tests.engine.test_construction_lead_time_calendar import _same_lp
from tests.engine.test_user_constraint_capacity_terms import SINGLE

V = 1.0 / 1.05
C = 400.0 * 1000.0
S = 200.0 * 1000.0
PERIODS = ["p2020", "p2025", "p2030", "p2035"]
YEAR = {"p2020": 0, "p2025": 5, "p2030": 10, "p2035": 15}
H = 20

_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def _contract(r: float, n: int, life: int, off: float, built: int, retired: int | None, *, s_end: float = S,
              s_retire: float = S, cost: float = C) -> tuple[float, float]:
    """``(kept, retired)``: per MW, the objective weight of a plant built at the start of year ``built`` and kept
    (its level price over its years in service inside the horizon), and the change when it is retired at the start
    of year ``retired`` instead (its cash less the level price given back); explicit yearly sums."""
    pay = cost * r / (1.0 - (1.0 + r) ** -n) * (1.0 + r) ** (off - 1.0) if r else cost / n
    loan_at_i = sum(pay * V ** (k + off) for k in range(n))  # the loan's payments weighed at i (from year 0)
    level = sum(V ** (k + off) for k in range(life))  # the same per unit over the lifetime
    salvage_at_r = s_end * (1.0 + r) ** -life  # S at the end of the lifetime, at r, from the start
    level_r = sum((1.0 + r) ** -(k + off) for k in range(life))
    a_b = loan_at_i / level - salvage_at_r / level_r
    service = range(built, min(built + life, H))
    kept = sum(a_b * V ** (y + off) for y in service)
    if retired is None:
        return kept, 0.0
    m = retired - built
    paid = sum(pay * V ** (built + k + off) for k in range(min(m, n)))
    owed = sum(pay * (1.0 + r) ** -(k + off - m) for k in range(m, n))
    cash = paid + V ** retired * (owed - s_retire)
    return kept, cash - kept


def _values(*, r: float = 0.05, life: float = 10.0, off: float = 1.0, payback: float | None = None,
            method: str = "invest_retire_period", existing: "float | list" = 0.0, fixed_cost: list | None = None,
            salvage: list | None = None, extra: list | None = None) -> list:
    out = [["model", "flexTool", "social_discount_rate", b64(0.05, "float")],
           ["model", "flexTool", "discount_offset_investment", b64(off, "float")],
           ["unit", PLANT, "discount_rate", b64(r, "float")],
           ["unit", PLANT, "fixed_cost", b64(0.0, "float") if fixed_cost is None else fixed_cost],
           ["unit", PLANT, "existing", existing if isinstance(existing, list) else b64(existing, "float")],
           ["unit", PLANT, "invest_cost", b64(400.0, "float")],
           ["unit", PLANT, "invest_method", b64(method, "str")],
           ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
           ["unit", PLANT, "lifetime_technical", b64(life, "float")],
           ["unit", PLANT, "net_salvage_value", b64(200.0, "float") if salvage is None else salvage]]
    if payback is not None:
        out.append(["unit", PLANT, "payback_time", b64(payback, "float")])
    return out + (extra or [])


def _pinned(invest: dict[str, float], retire: dict[str, float]) -> list:
    """Investment and retirement fixed per period (min = max; 0 in the periods not given)."""
    inv = {d: invest.get(d, 0.0) for d in PERIODS}
    ret = {d: retire.get(d, 0.0) for d in PERIODS}
    return [["unit", PLANT, "invest_min_period", _period_map(inv)],
            ["unit", PLANT, "invest_max_period", _period_map(inv)],
            ["unit", PLANT, "retire_min_period", _period_map(ret)],
            ["unit", PLANT, "retire_max_period", _period_map(ret)]]


def _step(tmp_path_factory, alt: str, values: list):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    return list(steps.values())[-1]


def _plant(param, cohort: str | None = None) -> dict[str, float]:
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    if cohort is not None:
        frame = frame.filter(pl.col("cohort") == cohort)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


def _built_mw(step, var: str = "v_divest_built_p") -> dict:
    """MW of ``var`` per (cohort, period) (``v_divest_built_p``) or per period (``v_divested_built_p``)."""
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == PLANT)["value"][0])
    frame = utf8(step.solution.value(var)).filter(pl.col("p") == PLANT)
    if "cohort" not in frame.columns:
        return {d: v * us for d, v in frame.select("d", "value").iter_rows()}
    return {(c, d): v * us for c, d, v in frame.select("cohort", "d", "value").iter_rows()}


def _divest_mw(step) -> dict[str, float]:
    if "v_divest_p" not in step.solution._vars:
        return {}
    us = float(utf8(step.flex_data.p_unitsize.frame).filter(pl.col("p") == PLANT)["value"][0])
    frame = utf8(step.solution.value("v_divest_p")).filter(pl.col("p") == PLANT)
    return {d: v * us for d, v in frame.select("d", "value").iter_rows()}


@pytest.mark.solver
@pytest.mark.parametrize(("r", "life", "payback", "off"), [(0.05, 10.0, None, 1.0), (0.03, 40.0, None, 1.0),
                                                          (0.08, 40.0, 20.0, 1.0), (0.05, 40.0, None, 0.0),
                                                          (0.0, 10.0, None, 0.0)])
def test_build_and_retire_at_once_costs_invest_cost_less_salvage(tmp_path_factory, r, life, payback, off) -> None:
    """Built in p2030 and retired there at once: (C − S)·D(10) = 200 000·1.05^-10 = 122 782.65 per MW, for every
    discount rate, lifetime, payback time and offset (the kept plant's level price is given back in full, the loan
    is owed in full at its start, worth C at r).  Measured: the objective of 100 MW built and retired in p2030 less
    that of the twin without the plant (capacity 0 in both: a retirement counts in its own period)."""
    n = int(payback or life)
    kept, change = _contract(r, n, int(life), off, 10, 10)
    assert kept + change == pytest.approx((C - S) * V ** 10, rel=1e-12)
    assert (C - S) * V ** 10 == pytest.approx(122_782.65, abs=0.01)
    tag = f"{r}_{life}_{payback}_{off}".replace(".", "p")
    case = _step(tmp_path_factory, f"rb_once_{tag}",
                 _values(r=r, life=life, off=off, payback=payback,
                         extra=_pinned({"p2030": 100.0}, {"p2030": 100.0})))
    twin = _step(tmp_path_factory, f"rb_none_{tag}", _values(r=r, life=life, off=off, payback=payback,
                                                              extra=_pinned({}, {})))
    assert _built_mw(case)[("invest:p2030", "p2030")] == pytest.approx(100.0, rel=1e-9)
    assert case.solution.obj - twin.solution.obj == pytest.approx(100.0 * (C - S) * V ** 10, rel=1e-8)


@pytest.mark.solver
def test_retirement_after_the_payback_time_with_unequal_rates(tmp_path_factory) -> None:
    """payback_time 20 < lifetime 40, r 3 %, i 5 %, built p2020: the change of retiring a MW at the start of p2025 /
    p2030 / p2035 instead of keeping it: +892.39 / +15 332.59 / +31 803.55 (the loan is paid off at par, weighed at
    i), and 200 000 = C − S when retired at once."""
    step = _step(tmp_path_factory, "rb_payback", _values(r=0.03, life=40.0, payback=20.0))
    cost = _plant(step.flex_data.cd_divest_built_cost, "invest:p2020")
    fixed = _plant(step.flex_data.cd_divest_built_fixed, "invest:p2020")
    lump = _plant(step.flex_data.ed_entity_annual_discounted)["p2020"]
    for d, want in (("p2025", 892.39), ("p2030", 15_332.59), ("p2035", 31_803.55)):
        _, change = _contract(0.03, 20, 40, 1.0, 0, YEAR[d])
        assert change == pytest.approx(want, abs=0.01)
        assert cost[d] - fixed[d] == pytest.approx(change, rel=1e-9)
    kept, change = _contract(0.03, 20, 40, 1.0, 0, 0)
    assert lump == pytest.approx(kept, rel=1e-12)
    assert lump + cost["p2020"] == pytest.approx(C - S, rel=1e-9)


@pytest.mark.solver
@pytest.mark.parametrize(("life", "rest"), [(10.0, 277_217.35), (40.0, 269_877.04)])
def test_partly_retired_cohort(tmp_path_factory, life, rest) -> None:
    """100 MW built in p2020, 40 MW retired in p2025 (r = i): each retired MW costs C − S·D(5) = 243 294.77; each
    of the 60 kept MW its level price: lifetime 10 ends inside the horizon (p2030), so it pays its whole contract
    including the salvage at the end of life, C − S·D(10) = 277 217.35; lifetime 40 outlives it, so it pays the 20
    years inside, 21 655.63 · Σ_{k<20} 1.05^-(k+1) = 269 877.04.  The capacity is 60 MW from p2025 to the end of
    life."""
    step = _step(tmp_path_factory, f"rb_part_{int(life)}",
                 _values(life=life, extra=_pinned({"p2020": 100.0}, {"p2025": 40.0})))
    kept, change = _contract(0.05, int(life), int(life), 1.0, 0, 5)
    assert kept + change == pytest.approx(C - S * V ** 5, rel=1e-12)
    assert C - S * V ** 5 == pytest.approx(243_294.77, abs=0.01)
    assert kept == pytest.approx(rest, abs=0.01)
    fd = step.flex_data
    lump = _plant(fd.ed_entity_annual_discounted)["p2020"]
    retire = _plant(fd.cd_divest_built_cost, "invest:p2020")["p2025"] - _plant(fd.cd_divest_built_fixed,
                                                                             "invest:p2020")["p2025"]
    assert 100.0 * lump + 40.0 * retire == pytest.approx(40.0 * (C - S * V ** 5) + 60.0 * kept, rel=1e-12)
    assert _built_mw(step)[("invest:p2020", "p2025")] == pytest.approx(40.0, rel=1e-9)
    counted = {d: v for d, v in _built_mw(step, "v_divested_built_p").items() if v > 1e-6}
    alive = ["p2025"] if life == 10.0 else ["p2025", "p2030", "p2035"]
    assert sorted(counted) == alive
    assert all(v == pytest.approx(40.0, rel=1e-9) for v in counted.values())


#: Existing 100 MW with lifetime 10 (in service p2020, p2025); 200 MW built in p2025 (lifetime 10: p2025, p2030).
_RETIRE_ALL = _values(existing=100.0, method="invest_retire_period_total", extra=[
    ["unit", PLANT, "invest_min_period", _period_map({"p2020": 0.0, "p2025": 200.0, "p2030": 0.0, "p2035": 0.0})],
    ["unit", PLANT, "invest_max_period", _period_map({"p2020": 0.0, "p2025": 200.0, "p2030": 0.0, "p2035": 0.0})]])


def _retire_total(mw: float) -> list:
    return [["unit", PLANT, "retire_min_total", b64(mw, "float")], ["unit", PLANT, "retire_max_total",
                                                                   b64(mw, "float")]]


@pytest.mark.solver
def test_a_cohort_retires_at_most_what_it_has(tmp_path_factory) -> None:
    """#70: a retirement total of 300 MW is met by the existing 100 MW and the 200 MW built in p2025, each cohort
    retiring at most what it has (maxDivest_cohort: one row per cohort and retirement period)."""
    step = _step(tmp_path_factory, "rb_total_300", _RETIRE_ALL + _retire_total(300.0))
    assert step.solution.optimal
    assert sum(_divest_mw(step).values()) == pytest.approx(100.0, rel=1e-9)
    built = _built_mw(step)
    assert {c for (c, _), v in built.items() if v > 1e-6} == {"invest:p2025"}
    assert sum(built.values()) == pytest.approx(200.0, rel=1e-9)
    rows = cstr_over(build(step.flex_data), "maxDivest_cohort_p").filter(pl.col("cohort") == "invest:p2025")
    assert sorted(rows.select("cohort", "d").iter_rows()) == [("invest:p2025", "p2025"), ("invest:p2025", "p2030")]


def _config_error(tmp_path_factory, alt: str, values: list, chain: list | None = None, entities=None) -> str:
    from flextool.engine.input._solve_state import FlexToolConfigError
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=chain or _no_battery_invest(SINGLE),
                        entities=entities)
    with pytest.raises(FlexToolConfigError) as err:
        run(tmp_path_factory, url, scen)
    return str(err.value)


def test_a_retirement_minimum_above_what_the_cohorts_have_is_an_error(tmp_path_factory) -> None:
    """301 MW required in total, more than the existing 100 MW and the 200 MW the model can build there: an error
    naming both; a minimum in p2035, where no cohort is in service any more, is one too."""
    msg = _config_error(tmp_path_factory, "rb_total_301", _RETIRE_ALL + _retire_total(301.0))
    assert PLANT in msg and "301" in msg and "100" in msg and "200" in msg
    msg = _config_error(tmp_path_factory, "rb_p2035", _RETIRE_ALL + [
        ["unit", PLANT, "retire_min_period", _period_map({"p2020": 0.0, "p2025": 0.0, "p2030": 0.0, "p2035": 1.0})]])
    assert PLANT in msg and "p2035" in msg and "retire_min_period" in msg


#: A chain: rb_first (p2020, p2025) builds 200 MW in p2025 and retires 50 MW of them there, rb_second (p2030,
#: p2035) follows and must retire ``p2030_min`` MW in p2030.  The existing 100 MW (lifetime 40) is 0 in p2025, so the
#: p2025 retirement can only be of the new capacity.
_CHAIN_ENTITIES = [["solve", "rb_first", None], ["solve", "rb_second", None]]


def _chain_values(p2030_min: float) -> list:
    return (_values(life=40.0, existing=_period_map({"p2020": 100.0, "p2025": 0.0, "p2030": 100.0, "p2035": 100.0}),
                    extra=[
        ["unit", PLANT, "invest_min_period", _period_map({"p2020": 0.0, "p2025": 200.0, "p2030": 0.0, "p2035": 0.0})],
        ["unit", PLANT, "invest_max_period", _period_map({"p2020": 0.0, "p2025": 200.0, "p2030": 0.0, "p2035": 0.0})],
        ["unit", PLANT, "retire_min_period",
         _period_map({"p2020": 0.0, "p2025": 50.0, "p2030": p2030_min, "p2035": 0.0})],
        ["unit", PLANT, "retire_max_period",
         _period_map({"p2020": 0.0, "p2025": 50.0, "p2030": 1000.0, "p2035": 1000.0})]])
        + _solves(["rb_first", "rb_second"])
        + _solve("rb_first", ["p2020", "p2025"], invest_periods=["p2020", "p2025"],
                 realized_periods=["p2020", "p2025"], realized_invest_periods=["p2020", "p2025"])
        + _solve("rb_second", ["p2030", "p2035"], invest_periods=["p2030", "p2035"],
                 realized_periods=["p2030", "p2035"], realized_invest_periods=["p2030", "p2035"]))


@pytest.mark.solver
def test_built_retirements_carry_to_the_later_solve(tmp_path_factory) -> None:
    """The 50 MW retired in rb_first are carried per cohort: rb_second starts from 150 MW of the p2025 cohort and
    meets its 250 MW in p2030 with them (#70: at most those 150 MW) and the whole existing 100 MW (the built
    retirements do not lower the existing capacity's limit).  The chain's retirement total counts the built 50."""
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="rb_chain", values=_chain_values(250.0),
                            base_chain=_no_battery_invest(SINGLE), entities=_CHAIN_ENTITIES)
    first, second = steps["rb_first"], steps["rb_second"]
    assert _built_mw(first)[("invest:p2025", "p2025")] == pytest.approx(50.0, rel=1e-9)
    retire = utf8(first.handoff.realized_retire)
    assert retire.select("entity", "cohort", "period").rows() == [(PLANT, "invest:p2025", "p2025")]
    assert retire["value"][0] == pytest.approx(50.0, rel=1e-9)
    cohorts = second.flex_data.cohort_divest_built.filter((pl.col("e").cast(pl.Utf8) == PLANT)
                                                          & (pl.col("cohort") == "invest:p2025"))
    assert cohorts.height == 1
    assert cohorts["remaining"][0] == pytest.approx(150.0, rel=1e-9)
    assert cohorts["d_build"][0] is None
    assert sum(_built_mw(second).values()) == pytest.approx(150.0, rel=1e-9)
    assert sum(_divest_mw(second).values()) == pytest.approx(100.0, rel=1e-9)
    divested = utf8(second.flex_data.p_entity_divested.frame).filter(pl.col("e") == PLANT)
    assert divested["value"][0] == pytest.approx(50.0, rel=1e-9)
    capacity = utf8(second.flex_data.p_entity_all_existing.frame).filter(pl.col("e") == PLANT)
    assert dict(zip(capacity["d"], capacity["value"])) == pytest.approx({"p2030": 250.0, "p2035": 250.0})
    assert _built_mw(second)[("invest:p2025", "p2030")] == pytest.approx(150.0, rel=1e-9)


@pytest.mark.solver
def test_carried_built_retirements_leave_the_reported_existing_capacity(tmp_path_factory) -> None:
    """In rb_second the reported existing capacity is the user's existing 100 MW of p2030 and p2035: rb_first
    retired 50 MW of its own p2025 cohort and none of the existing capacity, so those 50 MW come off the carried
    cohort (150 MW of the 200 built), not off the existing column."""
    import pandas as pd

    from tests.engine.test_chain_costs import _write_csv

    out = _write_csv(tmp_path_factory, "rb_chain_results", _chain_values(250.0), _no_battery_invest(SINGLE),
                     entities=_CHAIN_ENTITIES)
    capacity = pd.read_csv(out / "unit_capacity__d.csv")
    plant = capacity[(capacity["unit"] == PLANT) & (capacity["solve"] == "rb_second")].set_index("period")
    assert plant.loc["p2030", "existing"] == pytest.approx(100.0, rel=1e-9)
    assert plant.loc["p2035", "existing"] == pytest.approx(100.0, rel=1e-9)


def test_a_carried_cohort_retires_at_most_its_remainder(tmp_path_factory) -> None:
    """rb_second must retire 251 MW in p2030: more than the existing 100 MW and the 150 MW left of the carried
    cohort (an error naming them, 250); 250 MW is met."""
    msg = _config_error(tmp_path_factory, "rb_chain_251", _chain_values(251.0), entities=_CHAIN_ENTITIES)
    assert PLANT in msg and "p2030" in msg and "251" in msg and "250" in msg


@pytest.mark.solver
def test_constant_salvage_value_equals_its_period_map_with_built_retirement(tmp_path_factory) -> None:
    """With retirable built capacity (lifetime 10, built p2020, 40 MW retired in p2025) a net_salvage_value map of
    200 in every period builds the LP of the constant 200."""
    pinned = _pinned({"p2020": 100.0}, {"p2025": 40.0})
    constant = _step(tmp_path_factory, "rb_part_10", _values(life=10.0, extra=pinned))
    mapped = _step(tmp_path_factory, "rb_part_10_map", _values(life=10.0, extra=pinned,
                                                               salvage=_period_map(dict.fromkeys(PERIODS, 200.0))))
    assert constant.flex_data.cd_divest_built_cost is not None
    _same_lp(constant, mapped)


@pytest.mark.solver
def test_building_to_retire_is_warned(tmp_path_factory, capsys) -> None:
    """net_salvage_value 450 above invest_cost 400: building and retiring at once earns (C − 450 000)·D(m) < 0, so
    the model builds to retire up to the limit; a warning names the entity."""
    from flextool.engine import _behaviour_changes

    _behaviour_changes.start_run({})
    values = _values(extra=_pinned({"p2030": 10.0}, {"p2030": 10.0}), salvage=b64(450.0, "float"))
    url, scen = make_db(tmp_path_factory, alt="rb_warn", values=values, base_chain=_no_battery_invest(SINGLE))
    run(tmp_path_factory, url, scen)
    out = capsys.readouterr().out
    assert f"building and retiring {PLANT} earns money" in out


#: The two-period fan of ``stoch_two_period_hedge.json`` (p2035 at year 0, p2040 at year 1, one year each, horizon 2)
#: under recourse investment: ``base`` (unitsize 1 MW, invest_cost 1 CUR/kW in p2035: C = 1000 CUR/MW, r 0.05)
#: may invest and retire, lifetime 2 (in service in both periods), net_salvage_value 0.5 (S = 500).
HEDGE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "stoch_two_period_hedge.json"
HEDGE_C, HEDGE_S = 1000.0, 500.0


def _hedge_values(invest: dict[str, float], retire_min: dict[str, float]) -> list:
    return [["model", "flextool", "social_discount_rate", b64(0.05, "float")],
            ["model", "flextool", "discount_offset_investment", b64(1.0, "float")],
            ["unit", "base", "invest_method", b64("invest_retire_period", "str")],
            ["unit", "base", "lifetime_method", b64("reinvest_choice", "str")],
            ["unit", "base", "lifetime_technical", b64(2.0, "float")],
            ["unit", "base", "invest_cost", _period_map({"p2035": 1.0, "p2040": 3.0})],
            ["unit", "base", "discount_rate", b64(0.05, "float")],
            ["unit", "base", "net_salvage_value", b64(0.5, "float")],
            ["unit", "base", "invest_min_period", _period_map(invest)],
            ["unit", "base", "invest_max_period", _period_map(invest)],
            ["unit", "base", "retire_min_period", _period_map(retire_min)],
            ["unit", "base", "retire_max_period", _period_map({"p2035": 50.0, "p2040": 50.0})]]


def _hedge(tmp_path_factory, alt: str, chain: list[str], values: list):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=chain, fixture_json=HEDGE_JSON)
    return [s for name, s in steps.items() if "realized" not in name][-1]


def _base(param, cohort: str) -> dict[str, float]:
    frame = utf8(param.frame).filter((pl.col("e") == "base") & (pl.col("cohort") == cohort))
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


@pytest.mark.solver
def test_retiring_a_shared_cohort_in_every_branch_pays_its_contract_once(tmp_path_factory) -> None:
    """The fan opens in p2040; 50 MW of base built in the shared period p2035.  Retired there at once a MW costs
    (C − S)·D(0) = 500; retired in each branch of p2040 instead, the probability-weighted retirements and the
    investment add up to the contract as it ran: C − S·D(1) = 1000 − 500/1.05 = 523.81 (the level price is given
    back along each branch's own lineage).  Each calendar period counts once, also beside the realized
    designation's copy of p2040 in the solve's history: the existing 50 MW (lifetime 2) retired in p2035 earns
    S − S·D(2) = 500·(1 − 1.05^-2) = 46.49 (its level credits sum to its salvage at the end of life once)."""
    step = _hedge(tmp_path_factory, "rb_fan_trunk", ["init", "base", "stoch", "rp"],
                  _hedge_values({"p2035": 50.0, "p2040": 0.0}, {"p2035": 0.0, "p2040": 0.0})
                  + [["unit", "base", "existing", b64(50.0, "float")]])
    fd = step.flex_data
    existing = dict(zip(*[utf8(fd.ed_divest_salvage.frame).filter(pl.col("e") == "base")[c].to_list()
                          for c in ("d", "value")]))
    assert existing["p2035"] == pytest.approx(HEDGE_S * (1.0 - 1.05 ** -2), rel=1e-12)
    assert HEDGE_S * (1.0 - 1.05 ** -2) == pytest.approx(46.49, abs=0.01)
    lump = dict(zip(*[utf8(fd.ed_entity_annual_discounted.frame).filter(pl.col("e") == "base")[c].to_list()
                      for c in ("d", "value")]))["p2035"]
    cost = _base(fd.cd_divest_built_cost, "invest:p2035")
    fixed = _base(fd.cd_divest_built_fixed, "invest:p2035")
    assert lump + cost["p2035"] - fixed["p2035"] == pytest.approx(HEDGE_C - HEDGE_S, rel=1e-12)
    weights = dict(zip(*[utf8(fd.pd_branch_weight.frame)[c].to_list() for c in ("d", "value")]))
    branches = sorted(d for d in cost if d != "p2035")
    assert len(branches) == 2
    assert sum(weights[d] for d in branches) == pytest.approx(1.0, rel=1e-12)
    total = lump + sum(weights[d] * (cost[d] - fixed[d]) for d in branches)
    assert total == pytest.approx(HEDGE_C - HEDGE_S / 1.05, rel=1e-12)
    assert HEDGE_C - HEDGE_S / 1.05 == pytest.approx(523.81, abs=0.01)


@pytest.mark.solver
def test_retirement_inside_the_shared_investment_window_is_the_same_in_every_branch(tmp_path_factory) -> None:
    """The fan opens in p2035 and p2035 is a shared investment period (shared_invest_periods): each branch has its
    own copy of the 50 MW built there, and the retirement of at least 10 MW in p2035 is the same in every branch —
    one tie per branch pair, between the cohorts the branches build in their copies of p2035."""
    step = _hedge(tmp_path_factory, "rb_fan_window", ["init", "base", "stoch", "ws", "na_window"],
                  _hedge_values({"p2035": 50.0, "p2040": 0.0}, {"p2035": 10.0, "p2040": 0.0}))
    rows = cstr_over(build(step.flex_data), "non_anticipativity_divest_built_p")
    assert rows is not None and rows.height == 1
    ((p, cohort, d, cohort_b, b),) = rows.select("p", "cohort", "d", "cohort_b", "b").iter_rows()
    assert (p, cohort, cohort_b) == ("base", f"invest:{d}", f"invest:{b}") and d != b
    built = {(c, dd): v for c, dd, v in utf8(step.solution.value("v_divest_built_p"))
             .filter(pl.col("p") == "base").select("cohort", "d", "value").iter_rows()}
    assert built[(cohort, d)] == pytest.approx(built[(cohort_b, b)], abs=1e-9)
    assert built[(cohort, d)] >= 10.0 - 1e-9


@pytest.mark.solver
@pytest.mark.parametrize("retired", [12.0, 30.0])
def test_an_investment_and_the_extensions_of_one_shared_period_each_tie_to_their_own_twin(tmp_path_factory,
                                                                                          retired) -> None:
    """The fan opens in p2035; p2035 and p2040 are shared investment periods.  base has 50 MW existing and 50 MW
    built in p2035 (each branch's copy), both with a 1-year life that ends at p2040, a life extension of 1 year,
    and at most 5 MW built in p2040.  In each branch's p2040 three different cohorts are decided: the extension of
    the existing cohort, the extension of the branch's own p2035 investment, and the p2040 investment.  Each ties
    only to its own twin in the other branch (same kind, twin parent): four ties in all with the p2035 investment.
    A retirement of ``retired`` MW in p2040 is then free to come from any cohort (30 MW needs more than the 5 MW the
    p2040 investment can hold, so tying every cohort to every other one made it infeasible), the same in each
    branch."""
    values = [row for row in _hedge_values({"p2035": 50.0, "p2040": 0.0}, {"p2035": 0.0, "p2040": retired})
              if row[2] not in ("lifetime_technical", "invest_max_period", "retire_max_period")]
    values += [["unit", "base", "existing", b64(50.0, "float")],
               ["unit", "base", "lifetime_technical", b64(1.0, "float")],
               ["unit", "base", "invest_max_period", _period_map({"p2035": 50.0, "p2040": 5.0})],
               ["unit", "base", "retire_max_period", _period_map({"p2035": 0.0, "p2040": retired})],
               ["unit", "base", "life_extension_cost", b64(0.5, "float")],
               ["unit", "base", "life_extension_years", b64(1.0, "float")],
               ["solve", "stoch_2p_hedge", "shared_invest_periods",
                b64({"value_type": "str", "data": ["p2035", "p2040"]}, "array")]]
    step = _hedge(tmp_path_factory, f"rb_fan_twins_{int(retired)}", ["init", "base", "stoch", "ws", "na_window"],
                  values)
    rows = cstr_over(build(step.flex_data), "non_anticipativity_divest_built_p")
    assert rows is not None
    ties = {(c, d, cb, b) for c, d, cb, b in utf8(rows).select("cohort", "d", "cohort_b", "b").iter_rows()}

    shared = [("invest:p2035_{}", "p2035_{}"), ("invest:p2040_{}", "p2040_{}"),
              ("extend:p2040_{}:existing:p2035", "p2040_{}"), ("extend:p2040_{}:invest:p2035_{}", "p2040_{}")]
    want = {(c.format("high", "high"), d.format("high"), c.format("low", "low"), d.format("low")) for c, d in shared}
    assert ties == want, ties
    built = {(c, dd): v for c, dd, v in utf8(step.solution.value("v_divest_built_p"))
             .filter(pl.col("p") == "base").select("cohort", "d", "value").iter_rows()}
    for c, d, cb, b in want:
        assert built.get((c, d), 0.0) == pytest.approx(built.get((cb, b), 0.0), abs=1e-9)
    for branch in ("high", "low"):
        total = sum(v for (c, dd), v in built.items() if dd == f"p2040_{branch}")
        assert total == pytest.approx(retired, abs=1e-6)


@pytest.mark.solver
def test_built_retirements_carry_into_a_rolling_solve(tmp_path_factory) -> None:
    """rb_first builds 200 MW in p2025 and retires 50 MW of them there; a rolling dispatch solve of p2030 that
    follows starts from the existing 100 MW and the 150 MW left of that cohort in every roll."""
    base = _values(life=40.0, existing=_period_map({"p2020": 100.0, "p2025": 0.0, "p2030": 100.0, "p2035": 100.0}),
                   extra=[
        ["unit", PLANT, "invest_min_period", _period_map({"p2020": 0.0, "p2025": 200.0, "p2030": 0.0, "p2035": 0.0})],
        ["unit", PLANT, "invest_max_period", _period_map({"p2020": 0.0, "p2025": 200.0, "p2030": 0.0, "p2035": 0.0})],
        ["unit", PLANT, "retire_min_period", _period_map({"p2020": 0.0, "p2025": 50.0, "p2030": 0.0, "p2035": 0.0})],
        ["unit", PLANT, "retire_max_period", _period_map({"p2020": 0.0, "p2025": 50.0, "p2030": 0.0, "p2035": 0.0})]])
    values = (base + _solves(["rb_first", "rb_roll"])
              + _solve("rb_first", ["p2020", "p2025"], invest_periods=["p2020", "p2025"],
                       realized_periods=["p2020", "p2025"], realized_invest_periods=["p2020", "p2025"])
              + [row for row in _solve("rb_roll", ["p2030"], realized_periods=["p2030"])
                 if row[2] not in ("solve_mode", "period_timeset")]
              # a rolling solve needs a continuous timeline (5weeks has gaps): the 48 h of timeset 2day
              + [["solve", "rb_roll", "period_timeset", _period_map({"p2030": "2day"})],
                 ["solve", "rb_roll", "solve_mode", b64("rolling_window", "str")],
                 ["solve", "rb_roll", "rolling_solve_horizon", b64(8.0, "float")],
                 ["solve", "rb_roll", "rolling_solve_jump", b64(4.0, "float")]])
    steps, _, _ = _RUNS.get(tmp_path_factory, alt="rb_roll", values=values, base_chain=_no_battery_invest(SINGLE),
                            entities=[["solve", "rb_first", None], ["solve", "rb_roll", None]])
    rolls = [s for name, s in steps.items() if name.startswith("rb_roll")]
    assert len(rolls) > 1
    for roll in rolls:
        capacity = utf8(roll.flex_data.p_entity_all_existing.frame).filter(pl.col("e") == PLANT)
        assert dict(zip(capacity["d"], capacity["value"])) == pytest.approx({"p2030": 250.0})


@pytest.mark.solver
def test_retirement_of_built_capacity_in_the_results(tmp_path_factory) -> None:
    """100 MW built in p2020 (lifetime 10, fixed_cost 100 CUR/kW), 40 MW retired in p2025.  The unit capacity
    reports 40 MW divested in p2025 and 60 MW in service then (0 from p2030, the end of life).  The retirement's
    term in the objective, 40 MW × the change of retiring a MW (C − S·D(5) less the kept plant's level price,
    :func:`_contract`), is reported by event (r = i, off 1, v = 1/1.05):

    * *retirement* at p2025: the contract's payments to the start of p2025 and its balance then, less the capital
      its years in service were charged, A·Σ_{k=0..9} v^(k+1): both are worth C at r = i, so 0;
    * *salvage economic* at p2025: the salvage value received, −40·S·v^5;
    * *salvage end of life* at p2030 (its end of life): the 100 MW plant's credit −100·S·v^10 (the investment's
      line) and the retired 40 MW's credit given back +40·S·v^10: −60·S·v^10;

    the retirement's lines add up to 40 × change; the cost tables show the salvage causes as one *salvage* line.
    The fixed cost no longer paid, in p2025 (the retired 40 MW count until the end of life, the start of p2030):
    −40 · 100 000 · Σ_{k=5..9} 1.05^-(k+0.5) (discount_offset_operations 0.5).  Annualized: each line divided by
    its period's investment factor (p2025: Σ_{k=5..9} v^(k+1), p2030: Σ_{k=10..14} v^(k+1)); the fixed cost
    reduction −40 · 100 000 a year in p2025."""
    import pandas as pd

    from tests.engine.test_chain_costs import MEGA, _row, _write_csv

    values = _values(life=10.0, fixed_cost=b64(100.0, "float"), extra=_pinned({"p2020": 100.0}, {"p2025": 40.0}))
    out = _write_csv(tmp_path_factory, "rb_results", values, _no_battery_invest(SINGLE))
    capacity = pd.read_csv(out / "unit_capacity__d.csv")
    plant = capacity[capacity["unit"] == PLANT].set_index("period")
    assert plant.loc["p2025", "divested"] == pytest.approx(40.0, rel=1e-9)
    assert plant.loc["p2025", "total"] == pytest.approx(60.0, rel=1e-9)
    assert plant.loc["p2030", "total"] == pytest.approx(0.0, abs=1e-9)
    _, change = _contract(0.05, 10, 10, 1.0, 0, 5)
    economic, back = -40.0 * S * V ** 5, 40.0 * S * V ** 10
    assert economic + back == pytest.approx(40.0 * change, rel=1e-9)
    fixed = 100.0 * 1000.0 * sum(V ** (k + 0.5) for k in range(5, 10))
    discounted = pd.read_csv(out / "cost_unit_discounted__d.csv")
    annualized = pd.read_csv(out / "cost_unit_annualized__d.csv")
    factor = {"p2025": sum(V ** (k + 1) for k in range(5, 10)), "p2030": sum(V ** (k + 1) for k in range(10, 15))}
    want = {("p2025", "retirement"): 0.0, ("p2025", "salvage"): economic,
            ("p2030", "salvage"): -100.0 * S * V ** 10 + back}
    for (d, line), amount in want.items():
        assert _row(discounted, d)[line] * MEGA == pytest.approx(amount, rel=1e-7, abs=1e-3), (d, line)
        assert _row(annualized, d)[line] * MEGA == pytest.approx(amount / factor[d], rel=1e-7, abs=1e-3), (d, line)
    retired = _row(discounted, "p2025")
    assert retired["fixed cost reduction of divestments"] * MEGA == pytest.approx(-40.0 * fixed, rel=1e-7)
    assert _row(annualized, "p2025")["fixed cost reduction of divestments"] * MEGA == pytest.approx(
        -40.0 * 100.0 * 1000.0, rel=1e-7)


LH2_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"
#: lh2_three_region_invest: lh2_invest (one period y2030, decomposition benders over three benders_regional
#: regions) then the monolithic lh2_dispatch of y2030.  wind_A (region_A, unitsize 1000 MW) gets no existing
#: capacity and is reinvest_choice with retirement, at most 100 MW built and 40 MW retired in y2030.  Its
#: net_salvage_value of 100 000 CUR/kW (invest_cost 50) makes building and retiring at once earn far more than the
#: 40 MW would produce, so the solve builds the 100 MW and retires 40 MW of them: the limits bind (a pinned minimum
#: is not used: a non-owner region of a Benders solve cannot meet another region's investment minimum).
WIND = "wind_A"


def _lh2_retire(tmp_path_factory, alt: str, extra: list) -> dict:
    values = [["unit", WIND, "existing", b64(0.0, "float")],
              ["unit", WIND, "lifetime_method", b64("reinvest_choice", "str")],
              ["unit", WIND, "invest_method", b64("invest_retire_period", "str")],
              ["unit", WIND, "net_salvage_value", b64(100_000.0, "float")],
              ["unit", WIND, "invest_max_period", _period_map({"y2030": 100.0})],
              ["unit", WIND, "retire_max_period", _period_map({"y2030": 40.0})]] + extra
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values,
                            base_chain=["lh2_three_region", "lh2_three_region_invest"], fixture_json=LH2_JSON)
    return steps


def _wind_existing(steps: dict) -> dict[str, float]:
    dispatch = next(s for name, s in steps.items() if "lh2_dispatch" in name)
    capacity = utf8(dispatch.flex_data.p_entity_all_existing.frame).filter(pl.col("e") == WIND)
    return dict(zip(capacity["d"], capacity["value"]))


@pytest.mark.solver
def test_built_retirements_under_benders_carry_to_the_later_solve(tmp_path_factory) -> None:
    """The Benders invest solve builds 100 MW of wind_A and retires 40 MW of that cohort in y2030; the dispatch
    solve that follows starts from 100 − 40 = 60 MW (the cohort's 100 MW less its carried retirement), not 100,
    as after the same invest solve without decomposition."""
    steps = _lh2_retire(tmp_path_factory, "rb_benders", [])
    invest = next(s for name, s in steps.items() if "lh2_invest" in name)
    assert invest.is_benders
    retired = utf8(invest.handoff.realized_retire).filter(pl.col("entity") == WIND)
    assert retired.select("cohort", "period").rows() == [("invest:y2030", "y2030")]
    assert retired["value"][0] == pytest.approx(40.0, rel=1e-9)
    assert _wind_existing(steps) == pytest.approx({"y2030": 60.0}, rel=1e-9)
    monolith = _lh2_retire(tmp_path_factory, "rb_benders_twin",
                           [["solve", "lh2_invest", "decomposition", b64("none", "str")]])
    assert not next(s for name, s in monolith.items() if "lh2_invest" in name).is_benders
    assert _wind_existing(monolith) == pytest.approx(_wind_existing(steps), rel=1e-9)
