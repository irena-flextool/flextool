"""Synthetic minimal-feature toy: reserve subsystem in isolation.

Tests that the reserve subsystem reduces dispatchable capacity (via the
``maxToSink`` LHS coupling) AND that ``vq_reserve`` is correctly priced
when reservation cannot be fully met.

Topology
--------
* 1 nodeBalance node ``elec``, 1 commodity node ``gas`` priced at 10
  €/MWh.
* 1 unit process ``u``: source=``gas``, sink=``elec``, eff partition,
  unitsize 100 MW, slope=1.0, existing capacity = 100 MW (existing_count=1).
* 1 reserve (r, ud, g) = (``r1``, ``up``, ``g``) with method ``timeseries``.
* ``pdtReserve_upDown_group_reservation[r1, up, g, p2020, t] = 50`` MW.
* ``p_process_reserve_upDown_node_max_share[u, r1, up, elec] = 1.0``
  (vacuous; the binding constraint is the maxToSink coupling).
* Demand at ``elec`` = 80 MW each of 2 timesteps (=> v_flow = 0.8 in
  unitsize units; ``maxToSink``'s coupling leaves only v_reserve_up
  ≤ 0.2 = 20 MW).

Closed-form
-----------
maxToSink (per (u, gas, elec, p2020, t)):
    v_flow  +  Σ_r v_reserve_up[u, r, up, elec, p2020, t]  ≤  1.0

Demand → v_flow = 0.8 → Σ v_reserve_up ≤ 0.2 (= 20 MW).

reserveBalance_timeseries_eq (per (r1, up, g, p2020, t)):
    Σ_{(p,n)} v_reserve · unitsize · reliability  +  vq_reserve · reservation
        ≥  reservation
    =>  v_reserve_up · 100 · 1.0  +  vq_reserve · 50  ≥  50

Optimal: v_reserve_up = 0.2 (binding maxToSink), vq_reserve = 0.6
(obj-minimising — the only way to satisfy reserveBalance):
    20 + 0.6 · 50 = 50.

Active obj contributions:
    obj_commodity = 2 · 0.8 · 100 · 1.0 · 10 · 1.0 = 1600
    obj_vq_reserve = Σ_t vq_reserve · reservation · penalty · op_factor
                   = 2 · 0.6 · 50 · 1000 · 1.0 = 60_000

Total = 61_600.0.

Why this toy matters
--------------------
This is the smallest scenario in which the reserve subsystem and the
``maxToSink`` LHS coupling interact non-trivially.  Any breakage in
either the coupling sign / coefficient or in the reserveBalance LHS
(reliability factor, vq scaling by reservation) shows up directly in
the obj.
"""
from __future__ import annotations

import polars as pl
from polar_high import Param
from flextool.engine_polars.input import FlexData


def data() -> FlexData:
    # Time
    dt = pl.DataFrame({"d": ["p2020", "p2020"], "t": ["t01", "t02"]})
    p_step_duration = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"]*2, "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_rp_cost_weight = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"]*2, "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_inflation_op = Param(("d",), pl.DataFrame({"d": ["p2020"], "value": [1.0]}))
    p_period_share = Param(("d",), pl.DataFrame({"d": ["p2020"], "value": [1.0]}))

    nodeBalance = pl.DataFrame({"n": ["elec"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [-80.0, -80.0]}))
    p_penalty_up = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1e9, 1e9]}))
    p_penalty_down = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1e9, 1e9]}))

    # Process topology (eff)
    pss = pl.DataFrame({"p": ["u"], "source": ["gas"], "sink": ["elec"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))

    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u"], "source": ["gas"], "sink": ["elec"], "c": ["GAS_COM"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [100.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pl.DataFrame({"p": ["u"]*2, "source": ["gas"]*2, "sink": ["elec"]*2,
                       "d": ["p2020"]*2, "t": ["t01", "t02"],
                       "value": [1.0, 1.0]}))
    p_slope = Param(("p", "d", "t"),
        pl.DataFrame({"p": ["u"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame({"c": ["GAS_COM"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [10.0, 10.0]}))

    # ── Reserve data ──────────────────────────────────────────────────
    reserve_upDown_group = pl.DataFrame(
        {"r": ["r1"], "ud": ["up"], "g": ["g"]})
    reserve_upDown_group_method_timeseries = pl.DataFrame(
        {"r": ["r1"], "ud": ["up"], "g": ["g"], "method": ["timeseries"]})
    reserve_upDown_group_method_dynamic = pl.DataFrame(
        schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8, "method": pl.Utf8})
    reserve_upDown_group_method_n_1 = pl.DataFrame(
        schema={"r": pl.Utf8, "ud": pl.Utf8, "g": pl.Utf8, "method": pl.Utf8})

    # v_reserve domain — 1 process × 1 reserve × 1 ud × 1 node × 2 timesteps.
    prundt = pl.DataFrame({
        "p": ["u"]*2, "r": ["r1"]*2, "ud": ["up"]*2, "n": ["elec"]*2,
        "d": ["p2020"]*2, "t": ["t01", "t02"],
    })
    process_reserve_upDown_node_active = pl.DataFrame(
        {"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["elec"]})
    group_node = pl.DataFrame({"g": ["g"], "n": ["elec"]})

    p_process_reserve_upDown_node_reliability = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["elec"],
                       "value": [1.0]}))
    pdtReserve_upDown_group_reservation = Param(("r", "ud", "g", "d", "t"),
        pl.DataFrame({"r": ["r1"]*2, "ud": ["up"]*2, "g": ["g"]*2,
                       "d": ["p2020"]*2, "t": ["t01", "t02"],
                       "value": [50.0, 50.0]}))
    p_reserve_upDown_group_penalty_reserve = Param(("r", "ud", "g"),
        pl.DataFrame({"r": ["r1"], "ud": ["up"], "g": ["g"],
                       "value": [1000.0]}))
    p_process_reserve_upDown_node_max_share = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["elec"],
                       "value": [1.0]}))

    # Existing count needed by reserve_process_upward bound (1 unit).
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["u"], "d": ["p2020"], "value": [1.0]}))

    return FlexData(
        dt=dt,
        p_step_duration=p_step_duration,
        p_rp_cost_weight=p_rp_cost_weight,
        p_inflation_op=p_inflation_op,
        p_period_share=p_period_share,
        nodeBalance=nodeBalance,
        nodeBalance_dt=nodeBalance_dt,
        p_inflow=p_inflow,
        p_penalty_up=p_penalty_up,
        p_penalty_down=p_penalty_down,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize,
        p_flow_upper=p_flow_upper,
        p_slope=p_slope,
        p_commodity_price=p_commodity_price,
        p_process_existing_count=p_process_existing_count,
        # Reserve
        reserve_upDown_group=reserve_upDown_group,
        reserve_upDown_group_method_timeseries=reserve_upDown_group_method_timeseries,
        reserve_upDown_group_method_dynamic=reserve_upDown_group_method_dynamic,
        reserve_upDown_group_method_n_1=reserve_upDown_group_method_n_1,
        prundt=prundt,
        process_reserve_upDown_node_active=process_reserve_upDown_node_active,
        group_node=group_node,
        p_process_reserve_upDown_node_reliability=p_process_reserve_upDown_node_reliability,
        pdtReserve_upDown_group_reservation=pdtReserve_upDown_group_reservation,
        p_reserve_upDown_group_penalty_reserve=p_reserve_upDown_group_penalty_reserve,
        p_process_reserve_upDown_node_max_share=p_process_reserve_upDown_node_max_share,
    )


def expected_obj() -> float:
    """obj = obj_commodity + obj_vq_reserve = 1600 + 60_000 = 61_600."""
    return 1600.0 + 60_000.0
