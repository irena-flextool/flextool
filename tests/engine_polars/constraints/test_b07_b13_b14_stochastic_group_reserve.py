"""Surface B.7 / B.13 / B.14 — non-anticipativity (absent flow constraint),
group-level inertia source/sink + non_sync inside-exclusion, and reserve
invest tightening.

* ``test_b07_no_non_anticipativity_flow_constraint`` — covers **B7-4**
  (structural finding): the engine never emits a constraint named
  ``non_anticipativity_flow*``; flow continuity across siblings is
  delegated to ``non_anticipativity_storage_use`` (positive control).

* ``test_b13_inertia_source_and_sink_separately`` — covers **B13-4**:
  with ``inertia_constant`` set on BOTH source and sink of the same
  process, the LHS picks up two independent terms (one per side), giving
  effective coefficient 2·v_flow·unitsize.

* ``test_b13_non_sync_inside_exclusion`` — covers **B13-6**: a process
  in ``process_group_inside_nonSync`` is excluded from the non-sync LHS
  incoming sum (and from the RHS outgoing sum) via a symmetric anti-join.

* ``test_b14_reserve_process_invest_tightening`` — covers **B14-6**:
  with ``v_invest_p`` available, the per-process reserve upper bound
  picks up ``-Σ v_invest_p · max_share`` on the LHS so investing tightens
  (= raises) the reserve cap.  Solver invests to satisfy reserve demand.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool

from .conftest import solver_options


def _solve(data) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# B7-4 — structural finding: no non_anticipativity_flow constraint emitted.

def test_b07_no_non_anticipativity_flow_constraint(toy_2branch_2d):
    """Covers B7-4 — pin: NO ``non_anticipativity_flow*`` constraint name
    appears in ``cstr_names``.  Positive control: with stochastic storage
    on AND a flow touching it, ``non_anticipativity_storage_use`` IS
    emitted (the indirect mechanism that enforces flow continuity).
    """
    d = toy_2branch_2d
    periods = ["d1", "d2", "d2_b1", "d2_b2"]
    # Inject a process whose sink is the storage node so v_flow exists and
    # the storage_use constraint actually fires.
    pss = pl.DataFrame({"p": ["p"], "source": ["FUEL_n"], "sink": ["s"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(d.dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p"], "source": ["FUEL_n"], "sink": ["s"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(10.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").unique()
              .with_columns(value=pl.lit(1.0)))
    p_commodity_price = Param(("c", "d", "t"),
        d.dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
            .select("c", "d", "t", "value"))
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    names = set(pb.cstr_names())
    # Hand-calc: by-design absence — no constraint named non_anticipativity_flow*.
    assert not any(n.startswith("non_anticipativity_flow") for n in names), (
        f"non_anticipativity_flow* should not exist; found "
        f"{[n for n in names if n.startswith('non_anticipativity_flow')]}")
    # Positive control: the storage net-charge variant IS emitted.
    assert "non_anticipativity_storage_use" in names


# ---------------------------------------------------------------------------
# B13-4 — inertia LHS source-side AND sink-side independent contributions.

def test_b13_inertia_source_and_sink_separately(toy_group_reserve):
    """Covers B13-4 — ``_inertia_side('source', ...)`` and ``_inertia_side(
    'sink', ...)`` add SEPARATELY-keyed terms; the same process's source
    AND sink inertia constants both contribute.

    Hand-calc: 1 producer u (FUEL_n → n1), unitsize=100, v_flow=0.30
    (= 30 MW served / 100 unitsize, demand 30 at n1, demand 20 at n2 hits
    VOLL).  inertia_constant = 1.0 on both source and sink, group spans
    {FUEL_n, n1}.
        LHS_inertia = (v_flow·1·100)_source + (v_flow·1·100)_sink + slack·limit
                    = 30 + 30 + slack·100  >=  100
        ⇒ slack = (100 − 60) / 100 = 0.40 at each timestep.
    A single-side accounting would yield slack = 0.70 — the test fails if
    either side is dropped.
    """
    d = toy_group_reserve
    # Add source-side node FUEL_n into group_node so source inertia binds.
    group_node = pl.DataFrame(
        {"g": ["g"]*3, "n": ["n1", "n2", "FUEL_n"]})
    process_sink_inertia = pl.DataFrame({"p": ["u"], "sink": ["n1"]})
    process_source_inertia = pl.DataFrame({"p": ["u"], "source": ["FUEL_n"]})
    p_sink_iner = Param(("p", "sink"),
        pl.DataFrame({"p": ["u"], "sink": ["n1"], "value": [1.0]}))
    p_src_iner = Param(("p", "source"),
        pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "value": [1.0]}))
    groupInertia = pl.DataFrame({"g": ["g"]})
    pdGroup_inertia_limit = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [100.0]}))
    pdGroup_penalty_inertia = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1.0]}))
    data = dataclasses.replace(d,
        group_node=group_node,
        groupInertia=groupInertia,
        process_sink_inertia=process_sink_inertia,
        process_source_inertia=process_source_inertia,
        p_process_sink_inertia_constant=p_sink_iner,
        p_process_source_inertia_constant=p_src_iner,
        pdGroup_inertia_limit=pdGroup_inertia_limit,
        pdGroup_penalty_inertia=pdGroup_penalty_inertia,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "inertia_constraint" in set(pb.cstr_names())
    vq_in = sol.value("vq_inertia").sort(["d", "t"])
    # Hand-calc per (g, d, t):  60 + slack·100 ≥ 100  ⇒  slack = 0.40.
    # If only one side counted: 30 + slack·100 ≥ 100 ⇒ slack = 0.70.
    for v in vq_in["value"].to_list():
        assert v == pytest.approx(0.40, rel=1e-7), (
            f"slack {v} != 0.40 — one inertia side likely dropped")


# ---------------------------------------------------------------------------
# B13-6 — non_sync_constraint excludes processes in process_group_inside_nonSync.

def test_b13_non_sync_inside_exclusion(toy_group_reserve):
    """Covers B13-6 — ``_filter_out_inside`` anti-join on ``(p, g)``.

    Hand-calc: two producers u and u2 BOTH feed n1 (FUEL_n → n1), each
    capped at 1 unit (= 100 MW via unitsize=100).  Demand = 150 MW
    forces BOTH to run (u=100, u2=50 — cheaper unit pinned at cap).
    Both are in process_sink_nonSync; u2 is marked inside-group ``g``.

    FUEL_n is NOT in group_node, so RHS outgoing = 0.  Constraint:
        LHS = Σ_{p NOT inside g} v_flow·unitsize·step − vq·step
        RHS = 0
    With exclusion firing (u2 dropped):
        LHS = u·100·1 = 100  ⇒  vq = 100/1 = 100
    Without exclusion:
        LHS = u·100 + u2·100 (per unitsize) = 100 + 50 = 150  ⇒ vq = 150
    Pin vq = 100 to confirm the (u2, g) anti-join fires.
    """
    d = toy_group_reserve
    # Two producers feeding n1; u2 is "inside" group g.
    pss = pl.DataFrame({"p": ["u", "u2"],
                        "source": ["FUEL_n", "FUEL_n"],
                        "sink": ["n1", "n1"]})
    pss_eff = pss.clone()
    pss_dt = pss.join(d.dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u", "u2"], "source": ["FUEL_n"]*2, "sink": ["n1"]*2,
         "c": ["FUEL"]*2})
    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["u", "u2"], "value": [100.0, 100.0]}))
    # u capped at 1 unit (100 MW), u2 capped at 0.5 unit (50 MW); demand
    # 150 MW forces u=100 and u2=50 — the only feasible split.
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.when(pl.col("p") == "u2")
                                    .then(0.5).otherwise(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").unique()
              .with_columns(value=pl.lit(1.0)))
    # u has 1 existing unit; u2 has 1 existing unit.
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["u", "u2"], "d": ["d1", "d1"], "value": [1.0, 1.0]}))
    # Demand 150 MW at n1, 0 at n2 — forces both u and u2 (each cap 100) to run.
    nb_dt = d.nodeBalance_dt
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("n") == "n1").then(-150.0).otherwise(0.0)
    ).select("n", "d", "t", "value"))
    # NonSync setup: both flagged sink_nonSync; u2 inside group g.
    process_sink_nonSync = pl.DataFrame(
        {"p": ["u", "u2"], "sink": ["n1", "n1"]})
    process_group_inside_nonSync = pl.DataFrame({"p": ["u2"], "g": ["g"]})
    groupNonSync = pl.DataFrame({"g": ["g"]})
    pdGroup_non_synchronous_limit = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [0.5]}))
    pdGroup_penalty_non_synchronous = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1.0]}))
    # Drop reserve to keep this test focused on non_sync.
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope,
        p_process_existing_count=p_process_existing_count,
        p_inflow=p_inflow,
        groupNonSync=groupNonSync,
        process_sink_nonSync=process_sink_nonSync,
        process_group_inside_nonSync=process_group_inside_nonSync,
        pdGroup_non_synchronous_limit=pdGroup_non_synchronous_limit,
        pdGroup_penalty_non_synchronous=pdGroup_penalty_non_synchronous,
        # Disable reserve subsystem.
        reserve_upDown_group=None,
        reserve_upDown_group_method_timeseries=None,
        reserve_upDown_group_method_dynamic=None,
        reserve_upDown_group_method_n_1=None,
        prundt=None,
        process_reserve_upDown_node_active=None,
        pdtReserve_upDown_group_reservation=None,
        p_reserve_upDown_group_penalty_reserve=None,
        p_process_reserve_upDown_node_max_share=None,
        p_process_reserve_upDown_node_reliability=None,
        # Remove capacity_margin so we don't tangle constraints.
        groupCapacityMargin=None,
        pdGroup_capacity_margin=None,
        pdGroup_penalty_capacity_margin=None,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "non_sync_constraint" in set(pb.cstr_names())
    # Hand-calc: only u contributes; LHS=100, RHS=0 → vq=100. Without
    # exclusion u2 would add 50 → vq=150. Pin to 100 to confirm anti-join.
    vq_ns = sol.value("vq_non_synchronous").sort(["d", "t"])
    for v in vq_ns["value"].to_list():
        assert v == pytest.approx(100.0, rel=1e-7), (
            f"slack {v} != 100.0 — inside-group exclusion likely not firing")


# ---------------------------------------------------------------------------
# B14-6 — reserve_process_* invest tightening.

def test_b14_reserve_process_invest_tightening(toy_group_reserve):
    """Covers B14-6 — reserve LHS picks up ``-Σ v_invest_p · max_share``;
    making the reserve provider invest-eligible lets the solver invest to
    satisfy the reserve cap.

    Hand-calc:
      existing_count[u, d1] = 1, max_share = 0.25  →  baseline cap on
      v_reserve = 0.25 · 1 · unitsize = 0.25 (in unitsize=100 ⇒ 25 MW).
      Reserve demand (timeseries) = 10 (in unitsize=1 internal scaling
      via reliability=1, unitsize=100 — the LHS coefficient is unitsize·
      reliability = 100; so v_reserve · 100 ≥ 10 ⇒ v_reserve ≥ 0.1).
    Without invest the per-process bound 0.25 is already loose.  Tighten
    by raising demand to 80 ⇒ v_reserve ≥ 0.8, but baseline cap = 0.25
    binds (vq_reserve absorbs).  Make u investable with max 4 units;
    optimum: invest enough so 0.25·(existing+v_invest) ≥ 0.8
        ⇒  v_invest ≥ 0.8/0.25 − 1 = 2.2  ⇒ v_invest = 2.2.
    Use unitsize=1 for cleaner arithmetic — see fixture changes below.
    """
    d = toy_group_reserve
    # Switch to unitsize=1 for tractable hand-calc on the reserve cap.
    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [1.0]}))
    # Reserve demand high enough that the per-process cap binds.
    pdtReserve = Param(("r", "ud", "g", "d", "t"),
        pl.DataFrame({"r": ["r1"]*2, "ud": ["up"]*2, "g": ["g"]*2,
                      "d": ["d1"]*2, "t": ["t01", "t02"],
                      "value": [0.8, 0.8]}))
    # max_share = 0.25 (so existing alone gives 0.25 · 1 = 0.25 < 0.8).
    p_max_share = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"],
                      "value": [0.25]}))
    # Make u investable.  edd_invest_set: invest at d1 alive at d1.
    pd_invest_set = pl.DataFrame({"p": ["u"], "d": ["d1"]})
    ed_invest_set = pd_invest_set.rename({"p": "e"})
    edd_invest_set = pl.DataFrame(
        {"e": ["u"], "d_invest": ["d1"], "d": ["d1"]})
    p_entity_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [4.0]}))
    ed_invest_period_set = ed_invest_set.clone()
    ed_invest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [4.0]}))
    # Cheap invest cost so solver picks invest over slack.
    ed_entity_annual_discounted = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [1.0]}))
    ed_lifetime_fixed_cost = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["d1"], "value": [0.0]}))
    # Existing-only cap so v_flow has room (we don't drive flow here).
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pl.DataFrame({"p": ["u"], "source": ["FUEL_n"], "sink": ["n1"],
                      "d": ["d1"], "value": [10.0]}))
    # Trim demand so flow penalty doesn't dominate.
    nb_dt = d.nodeBalance_dt
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0))
            .select("n", "d", "t", "value"))
    # Drop capacity_margin to keep the LP focused.
    data = dataclasses.replace(d,
        p_unitsize=p_unitsize,
        pdtReserve_upDown_group_reservation=pdtReserve,
        p_process_reserve_upDown_node_max_share=p_max_share,
        p_inflow=p_inflow,
        pd_invest_set=pd_invest_set,
        ed_invest_set=ed_invest_set,
        edd_invest_set=edd_invest_set,
        p_entity_max_units=p_entity_max_units,
        ed_invest_period_set=ed_invest_period_set,
        ed_invest_max_period=ed_invest_max_period,
        ed_entity_annual_discounted=ed_entity_annual_discounted,
        ed_lifetime_fixed_cost=ed_lifetime_fixed_cost,
        p_flow_upper_existing=p_flow_upper_existing,
        groupCapacityMargin=None,
        pdGroup_capacity_margin=None,
        pdGroup_penalty_capacity_margin=None,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    cstr = set(pb.cstr_names())
    assert "reserve_process_upward" in cstr
    # Hand-calc: cap = 0.25·(1 + v_invest) ≥ 0.8 ⇒ v_invest ≥ 2.2.
    v_inv = sol.value("v_invest_p")
    inv_u = v_inv.filter(pl.col("p") == "u")["value"][0]
    assert inv_u == pytest.approx(2.2, rel=1e-6), (
        f"v_invest[u]={inv_u} != 2.2 — reserve invest tightening missing")
    # And v_reserve hits demand 0.8 with no slack.
    v_res = sol.value("v_reserve").sort(["d", "t"])
    for v in v_res["value"].to_list():
        assert v == pytest.approx(0.8, rel=1e-6)
