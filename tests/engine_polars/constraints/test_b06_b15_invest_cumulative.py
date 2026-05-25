"""Surface B.6 / B.15 — investment / retirement caps and group-invest /
cumulative-invest bounds.

Focused constraint tests on ``toy_invest_3d`` (3 periods × 1 investable
process):

* ``test_max_divest_var_bound`` — covers **B6-2** (`maxDivest_var_bound`):
  per-unit cap on ``v_divest`` (NOT ``v_invest``).  We give the process
  an existing fleet of 3, set ``entity_max_units = 2``, and add a
  positive ``ed_lifetime_fixed_cost_divest`` so divesting is profitable.
  The structural cap pins v_divest = 2 (the cap), even though existing=3.

* ``test_max_invest_and_divest_entity_period`` — covers **B6-3**
  (`maxInvest_entity_period`) consolidated with **B6-4**
  (`maxDivest_entity_period`).  One mutation sets both per-period caps
  for invest (d1: 1 unit) and divest (d2: 1 unit) — both bounds bind
  while sister periods stay loose.

* ``test_group_invest_max_total`` — covers **B15-3**
  (`maxInvestGroup_entity_total`): cap Σ_{e ∈ g, d} v_invest[e, d]
  ≤ 2 across two entities sharing a group, with high demand that would
  otherwise drive each above 2 individually.

* ``test_min_invest_and_no_investment_eq`` — covers **B15-5**
  (`minInvest_entity_period` + `e_invest_min_total`) consolidated with
  **B15-6** (`fix_v_invest_no_investment_eq_p`).  In one fixture: pin
  v_invest[u, d1] = 0 via `ed_invest_forbidden_no_investment` (equality),
  force v_invest[u, d2] ≥ 1 via `ed_invest_min_period`, and force
  Σ_d v_invest[u, d] ≥ 2 via `e_invest_min_total`.  Optimum picks
  v_invest = (0, 1, 1).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars._pdt_join import compute_nodeBalance_dt

from .conftest import solver_options


def _solve(data) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# B6-2 — maxDivest_var_bound (per-unit cap on v_divest).

def test_max_divest_var_bound(toy_invest_3d):
    """Covers B6-2 — `maxDivest_var_bound` per-unit cap on v_divest.

    Hand-calc: ed_lifetime_fixed_cost_divest=10 per unit makes divest
    profitable (objective subtracts 10·v_divest_p); structural cap is
    p_entity_max_units=2.  No per-period or total caps active.
    Optimum: v_divest_p[d1]=2 (binds against entity_max_units).
    """
    d = toy_invest_3d
    periods = ["d1", "d2", "d3"]
    # Tighten entity_max_units to 2 (the per-unit divest bound RHS).
    p_entity_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [2.0]*3}))
    # Existing capacity 3 — divest reduces flow capacity (v_flow ≤ existing
    # − Σ v_divest), so we need ≥ cap+ε existing to allow v_divest=2.
    pss = d.process_source_sink
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss.join(pl.DataFrame({"d": periods}), how="cross")
           .with_columns(value=pl.lit(3.0))
           .select("p", "source", "sink", "d", "value"))
    # Divest decision space mirrors invest set.
    pd_divest_set = d.pd_invest_set.clone()
    ed_divest_set = d.ed_invest_set.clone()
    # edd_divest_active: divest at d_divest is "active" from d_divest onward.
    edd_divest_active = pl.DataFrame({
        "p": ["u"]*3, "d_divest": periods, "d": periods})
    # Profitable divest: -10 per unit per period in the objective.
    ed_lifetime_fixed_cost_divest = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [10.0]*3}))
    data = dataclasses.replace(d,
        p_entity_max_units=p_entity_max_units,
        p_flow_upper_existing=p_flow_upper_existing,
        pd_divest_set=pd_divest_set,
        ed_divest_set=ed_divest_set,
        edd_divest_active=edd_divest_active,
        ed_lifetime_fixed_cost_divest=ed_lifetime_fixed_cost_divest,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "maxDivest_var_bound" in set(pb.cstr_names())
    # Hand-calc: cap=2 binds; v_invest stays at 0 (no demand).
    v_divest = sol.value("v_divest_p").sort("d")["value"].to_list()
    assert v_divest == [pytest.approx(2.0, rel=1e-7)] * 3


# ---------------------------------------------------------------------------
# B6-3 + B6-4 — per-period invest/divest caps.

def test_max_invest_and_divest_entity_period(toy_invest_3d):
    """Covers B6-3 (`maxInvest_entity_period`) + B6-4
    (`maxDivest_entity_period`).

    Hand-calc: ed_invest_max_period[d1]=1 (others=10), demand=10 in d1
    (1 step) so cost-min wants invest>1 but cap binds: v_invest[d1]=1.
    Mirror for divest: existing (via ed_lifetime_fixed_cost_divest=10)
    incentivises divest; ed_divest_max_period[d2]=1 binds v_divest[d2]=1.
    """
    d = toy_invest_3d
    periods = ["d1", "d2", "d3"]
    # Per-period invest cap: tight at d1, loose at d2/d3.
    ed_invest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods,
                      "value": [1.0, 10.0, 10.0]}))
    # Demand at d1/t01 (10 units) so the d1 invest cap binds.
    nb_dt = compute_nodeBalance_dt(d)
    p_inflow_new = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.when(pl.col("d") == "d1")
                                  .then(-10.0).otherwise(0.0))
            .select("n", "d", "t", "value"))
    # Existing capacity = 5 so v_divest can be positive without infeasibility.
    pss = d.process_source_sink
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss.join(pl.DataFrame({"d": periods}), how="cross")
           .with_columns(value=pl.lit(5.0))
           .select("p", "source", "sink", "d", "value"))
    # Same divest setup as B6-2 but with per-period cap binding at d2.
    pd_divest_set = d.pd_invest_set.clone()
    ed_divest_set = d.ed_invest_set.clone()
    ed_divest_period_set = d.ed_invest_set.clone()
    edd_divest_active = pl.DataFrame({
        "p": ["u"]*3, "d_divest": periods, "d": periods})
    ed_divest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods,
                      "value": [10.0, 1.0, 10.0]}))
    ed_lifetime_fixed_cost_divest = Param(("e", "d"),
        pl.DataFrame({"e": ["u"]*3, "d": periods, "value": [10.0]*3}))
    data = dataclasses.replace(d,
        p_inflow=p_inflow_new,
        p_flow_upper_existing=p_flow_upper_existing,
        ed_invest_max_period=ed_invest_max_period,
        pd_divest_set=pd_divest_set,
        ed_divest_set=ed_divest_set,
        edd_divest_active=edd_divest_active,
        ed_divest_period_set=ed_divest_period_set,
        ed_divest_max_period=ed_divest_max_period,
        ed_lifetime_fixed_cost_divest=ed_lifetime_fixed_cost_divest,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    cstr = set(pb.cstr_names())
    assert "maxInvest_entity_period_p" in cstr
    assert "maxDivest_entity_period_p" in cstr
    # Hand-calc: v_invest[d1]=1 (cap), v_divest[d2]=1 (cap).
    v_inv = sol.value("v_invest_p").sort("d")
    v_div = sol.value("v_divest_p").sort("d")
    assert v_inv.filter(pl.col("d") == "d1")["value"][0] == pytest.approx(1.0, rel=1e-7)
    assert v_div.filter(pl.col("d") == "d2")["value"][0] == pytest.approx(1.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B15-3 — group total invest cap.

def test_group_invest_max_total(toy_invest_3d):
    """Covers B15-3 — `maxInvestGroup_entity_total` group-level cap on
    Σ_{e ∈ g, d_invest ≤ d} v_invest[e, d_invest] ≤ 2.

    Hand-calc: 2 entities (``u1``, ``u2``) in group ``g``; large demand
    that the LP would otherwise meet by investing >2 units total.  Cap
    = 2 ⇒ at the latest period d3 the LP must satisfy
    Σ_{e ∈ {u1,u2}, d_invest ∈ {d1,d2,d3}} v_invest[e, d_invest] ≤ 2.
    Cost-min picks total = 2.
    """
    d = toy_invest_3d
    periods = ["d1", "d2", "d3"]
    # Two processes share the same source/sink/commodity wiring.
    pss = pl.DataFrame({"p": ["u1", "u2"],
                        "source": ["FUEL_n", "FUEL_n"],
                        "sink":   ["n", "n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(d.dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u1", "u2"], "source": ["FUEL_n", "FUEL_n"],
         "sink": ["n", "n"], "c": ["FUEL", "FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["u1", "u2"], "value": [1.0, 1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(0.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss.join(pl.DataFrame({"d": periods}), how="cross")
           .with_columns(value=pl.lit(0.0))
           .select("p", "source", "sink", "d", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").with_columns(value=pl.lit(1.0)))
    # Invest sets for both entities.
    pd_invest_set = pl.DataFrame(
        {"p": ["u1"]*3 + ["u2"]*3, "d": periods + periods})
    ed_invest_set = pd_invest_set.rename({"p": "e"})
    # Cumulative edd_invest_set: invest at d_invest is "alive" at every
    # d ≥ d_invest.  Required for the group-total cap to bind on the
    # full Σ_{d_invest ≤ d} and not just the per-period contribution.
    edd_rows = []
    for e in ("u1", "u2"):
        for d_inv in periods:
            for dd in periods:
                if periods.index(d_inv) <= periods.index(dd):
                    edd_rows.append({"e": e, "d_invest": d_inv, "d": dd})
    edd_invest_set = pl.DataFrame(edd_rows)
    p_entity_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["u1"]*3 + ["u2"]*3,
                      "d": periods + periods, "value": [10.0]*6}))
    ed_invest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u1"]*3 + ["u2"]*3,
                      "d": periods + periods, "value": [10.0]*6}))
    ed_entity_annual_discounted = Param(("e", "d"),
        pl.DataFrame({"e": ["u1"]*3 + ["u2"]*3,
                      "d": periods + periods, "value": [10.0]*6}))
    ed_lifetime_fixed_cost = Param(("e", "d"),
        pl.DataFrame({"e": ["u1"]*3 + ["u2"]*3,
                      "d": periods + periods, "value": [0.0]*6}))
    # Group covers both entities; group total invest cap = 2.
    group_entity = pl.DataFrame({"g": ["g", "g"], "e": ["u1", "u2"]})
    g_invest_total = pl.DataFrame({"g": ["g"]})
    p_group_invest_max_total = Param(("g",),
        pl.DataFrame({"g": ["g"], "value": [2.0]}))
    # High demand each step so total invest=2 binds (not 0).
    nb_dt = compute_nodeBalance_dt(d)
    p_inflow_new = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(-100.0))
            .select("n", "d", "t", "value"))
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize,
        p_flow_upper=p_flow_upper,
        p_flow_upper_existing=p_flow_upper_existing,
        p_slope=p_slope,
        p_inflow=p_inflow_new,
        pd_invest_set=pd_invest_set,
        ed_invest_set=ed_invest_set,
        edd_invest_set=edd_invest_set,
        p_entity_max_units=p_entity_max_units,
        ed_invest_period_set=ed_invest_set.clone(),
        ed_invest_max_period=ed_invest_max_period,
        ed_entity_annual_discounted=ed_entity_annual_discounted,
        ed_lifetime_fixed_cost=ed_lifetime_fixed_cost,
        group_entity=group_entity,
        g_invest_total=g_invest_total,
        p_group_invest_max_total=p_group_invest_max_total,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "maxInvestGroup_entity_total_p" in set(pb.cstr_names())
    # Hand-calc: Σ_{e, d} v_invest ≤ 2; cap binds with high demand.
    v_inv = sol.value("v_invest_p")
    total = float(v_inv["value"].sum())
    assert total == pytest.approx(2.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B15-5 + B15-6 — minInvest floors and no-investment equality.

def test_min_invest_and_no_investment_eq(toy_invest_3d):
    """Covers B15-5 (`minInvest_entity_period`) consolidated with
    B15-6 (`fix_v_invest_no_investment_eq_p`).

    Both are forced-value constraints on v_invest (one floor, one
    equality).  In one fixture:
      * fix_v_invest_no_investment_eq pins v_invest[u, d1] == 0.
      * minInvest_entity_period[d2] = 1 ⇒ v_invest[u, d2] ≥ 1.

    With invest cost = 10/unit and no demand, the LP picks the cheapest
    feasible point:
      v_invest[d1]=0 (pinned), v_invest[d2]=1 (floor), v_invest[d3]=0.
    """
    d = toy_invest_3d
    # Forbid invest at d1 (forced equality).
    forbid = pl.DataFrame({"e": ["u"], "d": ["d1"]})
    # Floor invest at d2 (>=1).
    ed_invest_min_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d2"], "value": [1.0]}))
    data = dataclasses.replace(d,
        ed_invest_forbidden_no_investment=forbid,
        ed_invest_min_period=ed_invest_min_period,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    cstr = set(pb.cstr_names())
    assert "fix_v_invest_no_investment_eq_p" in cstr
    assert "minInvest_entity_period_p" in cstr
    v_inv = sol.value("v_invest_p").sort("d")
    by_d = {row["d"]: row["value"]
            for row in v_inv.iter_rows(named=True)}
    # Hand-calc: d1=0 (pinned), d2=1 (floor binds), d3=0 (no incentive).
    assert by_d["d1"] == pytest.approx(0.0, abs=1e-7)
    assert by_d["d2"] == pytest.approx(1.0, rel=1e-7)
    assert by_d["d3"] == pytest.approx(0.0, abs=1e-7)
