"""Life extension of capacity reaching its end of life (``_extend``, ``_npv.extension_prices``), hand-derived.

Model: ``tests.json``, the single solve over p2020 … p2035 (5-year periods, run years 0, 5, 10, 15, horizon end
20).  coal_plant: 500 MW existing (age 0, one unit of 500 MW), lifetime_method reinvest_choice, lifetime_technical
L = 10 (the existing cohort serves [0, 10) and ends at the start of p2030), invest_method invest_no_limit,
invest_cost C = 400 CUR/kW, discount_rate r = 5 %, social_discount_rate i = 5 %, discount_offset_investment 1,
fixed_cost 0.  Per MW, v = 1/(1+i), D(t) = v^t, S[a, b) = Σ_{y=a}^{b−1} v^(y+1), ann(n) = r/(1 − (1+r)^−n);
E = 1000·life_extension_cost, L_x = life_extension_years, S = 1000·net_salvage_value::

    a new plant online in p2030:   C·1000·ann(10)·S[10, 20) = C·1000·v^10            (r = i)
    the extension at p2030:        ((E + S_c)·ann(L_x) − S_x·(1+r)^−L_x·ann(L_x))·S[10, min(10 + L_x, 20))

the extension a plant of its own (its cost plus the parent's salvage value it gives up, its own salvage at the
extended end of life, financed at r over L_x years), its yearly price charged for its years in service inside the
horizon.  With r = i and a flat S the extension at p2030 costs E·D(10) + S·(D(10) − D(10 + L_x)) inside the
horizon: the cash flows (E at the start, the parent's salvage S received at the extended end of life instead of at
10).  Values per MW:

* E 200 / 399 / 401 / 450, L_x 10, no salvage: 1000·E·v^10 = 122 782.65 / 244 951.39 / 246 179.21 / 276 260.96
  against a new plant's 400 000·v^10 = 245 565.30: extending pays below E = 400, building above;
* a flat net_salvage_value 200: the new plant costs (C − S·v^10)·v^10 = 170 187.40 and so does the extension at
  E = 200 = C − S (extending gives up the scrap value a new plant would get);
* built at p2020, S 200, E 100: L_x 5: 100 000·v^10 + 200 000·(v^10 − v^15) = 87 970.56; L_x 10: 108 796.08;
* extend at p2030 (L_x 10, E 100, S 200) and retire at p2035: the retirement's coefficient brings the total to the
  5-year extension, 87 970.56; extend and retire at once: E·D(10) = 61 391.33.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import PLANT, RunCache, b64, utf8
from tests.engine.test_chain_costs import _no_battery_invest
from tests.engine.test_user_constraint_capacity_terms import SINGLE

R = 0.05
V = 1.0 / 1.05
MW = 500.0
_RUNS = RunCache()


@pytest.fixture(scope="module", autouse=True)
def _free_runs():
    yield
    _RUNS.clear()


def ann(n: float) -> float:
    return R / (1.0 - (1.0 + R) ** -n)


def span(a: int, b: int) -> float:
    return sum(V ** (y + 1) for y in range(a, b))


def _values(*, cost: float = 200.0, years: float = 10.0, salvage: float | None = None,
            method: str = "invest_no_limit", extra: list | None = None) -> list:
    values = [["model", "flexTool", "social_discount_rate", b64(R, "float")],
              ["model", "flexTool", "discount_offset_investment", b64(1.0, "float")],
              ["unit", PLANT, "discount_rate", b64(R, "float")],
              ["unit", PLANT, "fixed_cost", b64(0.0, "float")],
              ["unit", PLANT, "invest_method", b64(method, "str")],
              ["unit", PLANT, "lifetime_method", b64("reinvest_choice", "str")],
              ["unit", PLANT, "lifetime_technical", b64(10.0, "float")],
              ["unit", PLANT, "invest_cost", b64(400.0, "float")],
              ["unit", PLANT, "life_extension_cost", b64(cost, "float")],
              ["unit", PLANT, "life_extension_years", b64(years, "float")]]
    if salvage is not None:
        values.append(["unit", PLANT, "net_salvage_value", b64(salvage, "float")])
    return values + (extra or [])


def _step(tmp_path_factory, alt: str, values: list):
    steps, _, _ = _RUNS.get(tmp_path_factory, alt=alt, values=values, base_chain=_no_battery_invest(SINGLE))
    return list(steps.values())[-1]


def _param(param, keys=("cohort", "d")) -> dict:
    frame = utf8(param.frame).filter(pl.col("e") == PLANT)
    return {tuple(row[:-1]): row[-1] for row in frame.select(*keys, "value").iter_rows()}


def _solution(step, var: str, keys=("cohort", "d")) -> dict:
    sol = step.solution
    if var not in sol._vars:
        return {}
    frame = utf8(sol.value(var)).filter(pl.col("p") == PLANT)
    return {tuple(row[:-1]): row[-1] for row in frame.select(*keys, "value").iter_rows()}


@pytest.mark.solver
@pytest.mark.parametrize("cost, want, extends", [(200.0, 122_782.65, True), (399.0, 244_951.39, True),
                                                 (401.0, 246_179.21, False), (450.0, 276_260.96, False)])
def test_extension_against_a_new_build(tmp_path_factory, cost, want, extends) -> None:
    """The existing cohort ends at p2030: extending it for 10 years costs 1000·E·v^10, a new plant 400 000·v^10."""
    build = 400_000.0 * ann(10.0) * span(10, 20)
    assert build == pytest.approx(400_000.0 * V ** 10, rel=1e-12) == pytest.approx(245_565.30, abs=0.01)
    extension = 1000.0 * cost * ann(10.0) * span(10, 20)
    assert extension == pytest.approx(want, abs=0.01)
    step = _step(tmp_path_factory, f"le_build_{cost:g}", _values(cost=cost))
    assert _param(step.flex_data.cd_extend_cost)[("existing:p2020", "p2030")] == pytest.approx(extension, rel=1e-12)
    assert _param(step.flex_data.ed_entity_annual_discounted, ("d",))[("p2030",)] == pytest.approx(build, rel=1e-12)
    extended = _solution(step, "v_extend_p").get(("existing:p2020", "p2030"), 0.0) * MW
    assert extended == pytest.approx(MW if extends else 0.0, abs=1e-6)


@pytest.mark.solver
def test_break_even_with_salvage_is_invest_cost_less_salvage(tmp_path_factory) -> None:
    """A flat net_salvage_value 200: the extension at E = 200 costs what a new plant does, 170 187.40 per MW."""
    s = 200_000.0
    build = (400_000.0 - s * 1.05 ** -10) * ann(10.0) * span(10, 20)
    extension = ((200_000.0 + s) - s * 1.05 ** -10) * ann(10.0) * span(10, 20)
    assert build == pytest.approx(extension, rel=1e-12) == pytest.approx(170_187.40, abs=0.01)
    step = _step(tmp_path_factory, "le_salvage_tie", _values(cost=200.0, salvage=200.0))
    assert _param(step.flex_data.cd_extend_cost)[("existing:p2020", "p2030")] == pytest.approx(extension, rel=1e-12)
    assert _param(step.flex_data.ed_entity_annual_discounted, ("d",))[("p2030",)] == pytest.approx(build, rel=1e-12)


@pytest.mark.solver
@pytest.mark.parametrize("years, want", [(5.0, 87_970.56), (10.0, 108_796.08)])
def test_the_salvage_moves_to_the_extended_end_of_life(tmp_path_factory, years, want) -> None:
    """A plant built at p2020 (lifetime 10) extended at p2030 by E 100 with a flat salvage value 200: the extension
    costs E·D(10) + S·(D(10) − D(10 + L_x)), the parent's salvage received at the extended end of life."""
    s = 200_000.0
    end = int(10 + years)
    cash = 100_000.0 * V ** 10 + s * (V ** 10 - V ** end)
    contract = ((100_000.0 + s) - s * 1.05 ** -years) * ann(years) * span(10, end)
    assert contract == pytest.approx(cash, rel=1e-12) == pytest.approx(want, abs=0.01)
    step = _step(tmp_path_factory, f"le_moved_{years:g}", _values(cost=100.0, years=years, salvage=200.0))
    got = _param(step.flex_data.cd_extend_cost)
    assert got[("invest:p2020", "p2030")] == pytest.approx(cash, rel=1e-12)
    prices = (step.flex_data.cohort.filter(pl.col("cohort") == "extend:p2030:invest:p2020")
              .select("y_on", "y_eol", "n_pay", "r_loan", "nsv_eol").row(0))
    assert prices == pytest.approx((10.0, end, years, R, s))


@pytest.mark.solver
def test_extend_then_retire_is_the_shorter_extension(tmp_path_factory) -> None:
    """Extend the built plant at p2030 for 10 years (E 100, S 200) and retire the extension at p2035: the
    extension's cost plus its retirement's is the 5-year extension, 87 970.56; retiring at once costs E·D(10)."""
    step = _step(tmp_path_factory, "le_retire", _values(cost=100.0, salvage=200.0, method="invest_retire_no_limit"))
    extension = _param(step.flex_data.cd_extend_cost)[("invest:p2020", "p2030")]
    retire = _param(step.flex_data.cd_divest_built_cost)
    later = extension + retire[("extend:p2030:invest:p2020", "p2035")]
    now = extension + retire[("extend:p2030:invest:p2020", "p2030")]
    assert later == pytest.approx(100_000.0 * V ** 10 + 200_000.0 * (V ** 10 - V ** 15), rel=1e-12)
    assert later == pytest.approx(87_970.56, abs=0.01)
    assert now == pytest.approx(100_000.0 * V ** 10, rel=1e-12) == pytest.approx(61_391.33, abs=0.01)


@pytest.mark.solver
def test_only_what_is_still_in_service_is_extended(tmp_path_factory) -> None:
    """200 of the 500 MW existing are retired at p2025 (retire_min_period): at most the 300 MW left in service at the
    end of life are extended, and the cheap extension takes all of them."""
    values = _values(cost=200.0, method="invest_retire_period", extra=[
        ["unit", PLANT, "retire_min_period", b64({"index_type": "str", "data": [["p2020", 0.0], ["p2025", 200.0],
                                                                                  ["p2030", 0.0], ["p2035", 0.0]]},
                                                 "map")],
        ["unit", PLANT, "retire_max_period", b64({"index_type": "str", "data": [["p2020", 0.0], ["p2025", 200.0],
                                                                                  ["p2030", 0.0], ["p2035", 0.0]]},
                                                 "map")],
        ["unit", PLANT, "invest_max_period", b64(1e6, "float")]])
    step = _step(tmp_path_factory, "le_bound", values)
    assert _solution(step, "v_divest_p", ("d",)).get(("p2025",), 0.0) * MW == pytest.approx(200.0, abs=1e-6)
    assert _solution(step, "v_extend_p")[("existing:p2020", "p2030")] * MW == pytest.approx(300.0, abs=1e-6)


@pytest.mark.solver
def test_an_extension_is_never_extended(tmp_path_factory) -> None:
    """A 5-year extension at p2030 ends at p2035, a period start: it has no option of its own there (nor has any
    extension), only the cohorts that end their life there (the investment of p2025)."""
    step = _step(tmp_path_factory, "le_once", _values(cost=200.0, years=5.0))
    options = utf8(step.flex_data.cd_extend).filter(pl.col("e") == PLANT).select("cohort", "d").rows()
    assert sorted(options) == [("existing:p2020", "p2030"), ("invest:p2020", "p2030"), ("invest:p2025", "p2035")]


@pytest.mark.solver
def test_extension_without_an_invest_method_is_reported_as_investment(tmp_path_factory) -> None:
    """invest_method not_allowed (no investment, no retirement): the existing 500 MW are still extended at p2030 (E 100, 10
    years, the plant's only way to keep producing), and the extension's cost is reported as investment of p2030:
    500 · 100 000·v^10 = 30.695663 M CUR (discounted) and 500 · 100 000·ann(10) = 6.475229 M CUR a year."""
    from tests.engine.test_chain_costs import MEGA, _results
    values = _values(cost=100.0, method="not_allowed")
    step = _step(tmp_path_factory, "le_no_method", values)
    assert _solution(step, "v_extend_p")[("existing:p2020", "p2030")] * MW == pytest.approx(MW, abs=1e-6)
    assert not _solution(step, "v_invest_p", ("d",))
    results = _results(tmp_path_factory, "le_no_method_out", values, _no_battery_invest(SINGLE))
    discounted = results["cost_unit_discounted__d"]
    row = discounted[(discounted["unit"] == PLANT) & (discounted["period"] == "p2030")].iloc[0]
    assert row["investment"] * MEGA == pytest.approx(MW * 100_000.0 * V ** 10, rel=1e-7)
    assert MW * 100_000.0 * V ** 10 / MEGA == pytest.approx(30.695663, abs=1e-6)
    yearly = results["cost_unit_annualized__d"]
    for d in ("p2030", "p2035"):
        row = yearly[(yearly["unit"] == PLANT) & (yearly["period"] == d)].iloc[0]
        assert row["investment"] * MEGA == pytest.approx(MW * 100_000.0 * ann(10.0), rel=1e-7), d
    assert MW * 100_000.0 * ann(10.0) / MEGA == pytest.approx(6.475229, abs=1e-6)
