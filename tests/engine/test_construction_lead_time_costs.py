"""Construction lead time: the annuity and the fixed cost of new capacity are paid only while it operates, at the
values of the period it comes online (invest_cost is an overnight cost).

* tests.json, single solve ``y2020_2035_5week`` (p2020 … p2035, 5 years apart): with a 5-year lead time a
  decision of p2020 comes online in p2025 and operates over the same years as a p2025 decision without a lead
  time, so their cost coefficients are equal; the annuity of coal_plant (discount_rate 0.05, lifetime 30) is
  invest_cost · 1000 · 0.05 / (1 − 1.05^−30), proportional to invest_cost.
* stoch_two_period_invest.json (recourse, p2035 and p2040 of one year each, a two-branch fan, no inflation):
  peaker invest_cost 1, discount_rate 0.05, lifetime 1 → annuity 1 · 1000 · 0.05 / (1 − 1.05^−1) = 1050 per
  operating year.
* lh2_three_region.json (Benders over three regions) with two 5-year periods.
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine._unit_db import FLEXTOOL_ROOT, b64, build, make_db, run, utf8
from tests.engine.test_user_constraint_capacity_terms import (
    PLANT,
    SINGLE,
    STOCH_CHAIN,
    STOCH_JSON,
    TWIN_ENTITIES,
    _add_values,
    GE,
    _period_rows,
    _constraint,
    _twin_values,
)

LEAD = "construction_lead_time"


def _period_map(data: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in data.items()]}, "map")


def _single(tmp_path_factory, alt: str, values: list):
    url, scen = make_db(tmp_path_factory, alt=alt, values=values, base_chain=SINGLE)
    return list(run(tmp_path_factory, url, scen).values())[-1]


def _col_obj(step) -> dict[str, float]:
    m = build(step.flex_data).canonicalise()
    return dict(zip(m.col_names, (float(v) for v in m.col_obj)))


def _param(step, field: str, entity: str = PLANT) -> dict[str, float]:
    frame = utf8(getattr(step.flex_data, field).frame).filter(pl.col("e") == entity)
    return dict(zip(frame["d"].to_list(), frame["value"].to_list()))


COST_MAP = {"p2020": 400.0, "p2025": 800.0, "p2030": 800.0, "p2035": 800.0}


def test_invest_cost_of_the_build_period(tmp_path_factory) -> None:
    """invest_cost {p2020: 400, later: 800}, fixed_cost 0.  With L = 5 the p2020 decision is built (comes online)
    in p2025, so it pays the p2025 invest cost over p2025 … p2035: its objective coefficient equals that of the
    p2025 decision without a lead time, and is twice the coefficient with a constant invest_cost 400 (the same
    operating years, annuity proportional to invest_cost)."""
    no_fixed = [["unit", PLANT, "fixed_cost", b64(0.0, "float")]]
    lagged = _single(tmp_path_factory, "lt_cost_map", no_fixed + [
        ["unit", PLANT, LEAD, b64(5.0, "float")], ["unit", PLANT, "invest_cost", _period_map(COST_MAP)]])
    unlagged = _single(tmp_path_factory, "lt_cost_map0", no_fixed + [
        ["unit", PLANT, "invest_cost", _period_map(COST_MAP)]])
    constant = _single(tmp_path_factory, "lt_cost_const", no_fixed + [
        ["unit", PLANT, LEAD, b64(5.0, "float")], ["unit", PLANT, "invest_cost", b64(400.0, "float")]])
    col = f"v_invest_p[{PLANT},p2020]"
    a, b, c = _col_obj(lagged), _col_obj(unlagged), _col_obj(constant)
    assert a[col] > 0.0
    assert a[col] == pytest.approx(b[f"v_invest_p[{PLANT},p2025]"], rel=1e-12)
    assert a[col] / c[col] == pytest.approx(2.0, rel=1e-12)


def test_fixed_cost_only_while_operating(tmp_path_factory) -> None:
    """fixed_cost {p2020: 100, later: 200}.  With L = 5 the p2020 decision pays the p2025 fixed cost from p2025 on:
    its lifetime fixed cost equals that of the p2025 decision without a lead time, and is twice the value with a
    constant fixed_cost 100."""
    fixed = {"p2020": 100.0, "p2025": 200.0, "p2030": 200.0, "p2035": 200.0}
    lagged = _single(tmp_path_factory, "lt_fc_map", [
        ["unit", PLANT, LEAD, b64(5.0, "float")], ["unit", PLANT, "fixed_cost", _period_map(fixed)]])
    unlagged = _single(tmp_path_factory, "lt_fc_map0", [["unit", PLANT, "fixed_cost", _period_map(fixed)]])
    constant = _single(tmp_path_factory, "lt_fc_const", [
        ["unit", PLANT, LEAD, b64(5.0, "float")], ["unit", PLANT, "fixed_cost", b64(100.0, "float")]])
    a = _param(lagged, "ed_lifetime_fixed_cost")
    b = _param(unlagged, "ed_lifetime_fixed_cost")
    c = _param(constant, "ed_lifetime_fixed_cost")
    assert a["p2020"] > 0.0
    assert a["p2020"] == pytest.approx(b["p2025"], rel=1e-12)
    assert a["p2020"] / c["p2020"] == pytest.approx(2.0, rel=1e-12)


def test_recourse_annuity_counted_once_and_identity(tmp_path_factory) -> None:
    """peaker with L = 1 under recourse: the p2035 decisions of each branch come online in that branch's p2040
    (the p2040 decisions would start at the end of the horizon and are not offered).  The annuity of a p2035
    decision is 1050 × one operating year = 1050, not 2100 (counting both branch copies of p2040, or p2035 too,
    as without the lead time).  The user-constraint identity capacity = pre-built + invested − retired holds."""
    values = ([["unit", "peaker", "invest_method", b64("invest_retire_no_limit", "str")]]
              + _constraint("c_cap") + _constraint("c_twin"))
    url, scen = make_db(tmp_path_factory, alt="lt_recourse", values=values, base_chain=STOCH_CHAIN,
                        entities=TWIN_ENTITIES, fixture_json=STOCH_JSON)
    _add_values(url, "lt_recourse", _twin_values("unit", "peaker") + [["unit", "peaker", LEAD, b64(1.0, "float")]])
    step = list(run(tmp_path_factory, url, scen).values())[0]
    ann = _param(step, "ed_entity_annual_discounted", "peaker")
    offered = sorted(utf8(step.flex_data.ed_invest_set).filter(pl.col("e") == "peaker")["d"].to_list())
    assert offered == ["p2035_high", "p2035_low"]
    assert ann["p2035_high"] == pytest.approx(1050.0, rel=1e-9)
    assert ann["p2035_low"] == pytest.approx(1050.0, rel=1e-9)
    commission = utf8(step.flex_data.edd_invest_commission).filter(pl.col("e") == "peaker")
    assert sorted(commission.select("d_invest", "d").rows()) == [("p2035_high", "p2040_high"),
                                                                 ("p2035_low", "p2040_low")]
    pb = build(step.flex_data)
    cap = _period_rows(pb, GE + "_period", "c_cap")
    twin = _period_rows(pb, GE + "_period", "c_twin")
    assert cap and sorted(cap) == sorted(twin)
    for d in cap:
        assert cap[d][0] == pytest.approx(twin[d][0], rel=1e-12), d
        assert cap[d][1] == pytest.approx(twin[d][1], rel=1e-12), d
    # The branch p2040 rows carry the branch's p2035 decision (online there).
    assert all(f"v_invest_p[peaker,p2035_{b}]" in cap[f"p2040_{b}"][0] for b in ("high", "low"))


LH2_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"


def _period_values(solve: str, decomposition: str) -> list:
    def arr(items: list[str]) -> list:
        return b64({"type": "array", "value_type": "str", "data": items, "index_name": "period"}, "array")
    both = ["y2030", "y2035"]
    return [["solve", solve, "period_timeset", _period_map({"y2030": "week168", "y2035": "week168"})],
            ["solve", solve, "realized_periods", arr(both)],
            ["solve", solve, "invest_periods", arr(both)],
            ["solve", solve, "realized_invest_periods", arr(both)],
            ["solve", solve, "years_represented", _period_map({"y2030": 5.0, "y2035": 5.0})],
            ["solve", solve, "decomposition", b64(decomposition, "str")],
            ["model", "flexTool", "solves", b64({"type": "array", "value_type": "str", "data": [solve]}, "array")],
            ["unit", "wind_A", LEAD, b64(5.0, "float")]]


def test_benders_with_lead_time_matches_monolith(tmp_path_factory) -> None:
    """Three-region Benders investment solve over two 5-year periods with a 5-year lead time on wind_A (its y2030
    decision comes online in y2035): the decomposed solve reaches the monolithic objective."""
    objective = {}
    for decomposition in ("none", "benders"):
        url, scen = make_db(tmp_path_factory, alt=f"lt_benders_{decomposition}",
                            values=_period_values("lh2_invest", decomposition),
                            base_chain=["lh2_three_region", "lh2_three_region_invest"], fixture_json=LH2_JSON)
        step = list(run(tmp_path_factory, url, scen).values())[0]
        assert step.is_benders is (decomposition == "benders")
        commission = utf8(step.flex_data.ed_invest_commission)
        assert commission.select("e", "d", "d_c").rows() == [("wind_A", "y2030", "y2035")]
        objective[decomposition] = step.obj
    assert objective["benders"] == pytest.approx(objective["none"], rel=1e-6)
