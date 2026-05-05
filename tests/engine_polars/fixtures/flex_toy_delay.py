"""Synthetic minimal-feature toy: delayed-process (hydro chain) in
isolation.

Tests that ``_delay.delayed_input_expr`` correctly time-shifts the
source-side input contribution to the ``conversion_indirect`` LHS, so
demand at ``t`` is supplied by a release at ``t − 1``.

Topology
--------
* 3 nodes: ``water_upstream`` (commodity ``WATER_COM`` priced 1 €/MWh),
  ``p_d`` (the indirect-process self-node, NOT in nodeBalance), and
  ``water_downstream`` (in nodeBalance, demand sink).
* 1 indirect process ``p_d`` (eff partition, unitsize=100 MW, slope=1.0):
  * input edge:  source=``water_upstream``, sink=``p_d``    — delayed
  * output edge: source=``p_d``, sink=``water_downstream``  — undelayed
* 4 timesteps t01..t04.  Demand at ``water_downstream``:
    t01: 0, t02..t04: 10 MW each.
* Delay map: ``dtt__delay_duration`` rows (d=p2020, td=1):
    (t_source=t01, t_sink=t02), (t02, t03), (t03, t04).
* Delay weight: ``p_process_delay_weight[p_d, td=1] = 1.0``.

Closed-form
-----------
At the sink-side ``t``, the conversion_indirect constraint reads:
    LHS_delayed[t]  ==  output_v_flow[t] · unitsize

With ``td=1`` weight 1.0:
    LHS_delayed[t02] = input_v_flow[t01] · unitsize
    LHS_delayed[t03] = input_v_flow[t02] · unitsize
    LHS_delayed[t04] = input_v_flow[t03] · unitsize
    LHS_delayed[t01] = 0  (no entry in dtt with t_sink=t01)

Demand drives output_v_flow = 0.1 at t02,t03,t04 (= 10 MW / 100 MW
unitsize), and = 0 at t01 (no demand).  Therefore:
    input_v_flow[t01] = 0.1, input_v_flow[t02] = 0.1, input_v_flow[t03] = 0.1
    input_v_flow[t04] = 0   (cost-minimising — never used)

Active obj contributions:
    obj_commodity = Σ_t input_v_flow · unitsize · slope · price · op_factor
                  = 3 · 0.1 · 100 · 1.0 · 1.0 · 1.0
                  = 30.0
    obj_slack     = 0  (nodeBalance satisfied exactly at every t)

Total expected obj = 30.0.

Why this toy matters
--------------------
Verifies that ``_delay.delayed_input_expr`` is wired into
``conversion_indirect`` correctly: any wrong sign on the delay term, a
wrong ``t_source`` mapping, or a missing time-shift collapses
this 4-line closed-form.
"""
from __future__ import annotations

import polars as pl
from polar_high import Param
from flextool.engine_polars.input import FlexData


def data() -> FlexData:
    # Time: 4 timesteps.
    ts = ["t01", "t02", "t03", "t04"]
    dt = pl.DataFrame({"d": ["p2020"]*4, "t": ts})
    p_step_duration = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"]*4, "t": ts, "value": [1.0]*4}))
    p_rp_cost_weight = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"]*4, "t": ts, "value": [1.0]*4}))
    p_inflation_op = Param(("d",), pl.DataFrame({"d": ["p2020"], "value": [1.0]}))
    p_period_share = Param(("d",), pl.DataFrame({"d": ["p2020"], "value": [1.0]}))

    # Nodes: water_downstream is in nodeBalance.
    nodeBalance = pl.DataFrame({"n": ["water_downstream"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), pl.DataFrame({
        "n": ["water_downstream"]*4, "d": ["p2020"]*4, "t": ts,
        "value": [0.0, -10.0, -10.0, -10.0],
    }))
    p_penalty_up = Param(("n", "d", "t"), pl.DataFrame({
        "n": ["water_downstream"]*4, "d": ["p2020"]*4, "t": ts,
        "value": [1e9]*4}))
    p_penalty_down = Param(("n", "d", "t"), pl.DataFrame({
        "n": ["water_downstream"]*4, "d": ["p2020"]*4, "t": ts,
        "value": [1e9]*4}))

    # Process topology — both edges in pss (and pss_eff).
    pss = pl.DataFrame({
        "p":      ["p_d", "p_d"],
        "source": ["water_upstream", "p_d"],
        "sink":   ["p_d", "water_downstream"],
    })
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))

    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p_d"], "source": ["water_upstream"], "sink": ["p_d"],
         "c": ["WATER_COM"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p_d"], "value": [100.0]}))

    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pl.DataFrame({
            "p":      ["p_d"]*8,
            "source": ["water_upstream"]*4 + ["p_d"]*4,
            "sink":   ["p_d"]*4 + ["water_downstream"]*4,
            "d":      ["p2020"]*8,
            "t":      ts*2,
            "value":  [1.0]*8,
        }))
    p_slope = Param(("p", "d", "t"),
        pl.DataFrame({"p": ["p_d"]*4, "d": ["p2020"]*4, "t": ts,
                       "value": [1.0]*4}))
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame({"c": ["WATER_COM"]*4, "d": ["p2020"]*4, "t": ts,
                       "value": [1.0]*4}))

    # ── Indirect-conversion ───────────────────────────────────────────
    process_indirect = pl.DataFrame({"p": ["p_d"]})
    process_input_flows = pl.DataFrame(
        {"p": ["p_d"], "source": ["water_upstream"], "sink": ["p_d"]})
    process_output_flows = pl.DataFrame(
        {"p": ["p_d"], "source": ["p_d"], "sink": ["water_downstream"]})
    process_indirect_dt = process_indirect.join(dt, how="cross")

    # ── Delay tables ──────────────────────────────────────────────────
    process_delayed = pl.DataFrame({"p": ["p_d"]})
    process_delayed__duration = pl.DataFrame({
        "p": ["p_d"], "td": [1.0]})
    process_source_delayed = pl.DataFrame({
        "p": ["p_d"], "source": ["water_upstream"]})
    process_source_undelayed = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8})
    process_source_sink_delayed = pl.DataFrame({
        "p": ["p_d"], "source": ["water_upstream"], "sink": ["p_d"]})
    process_source_sink_undelayed = pl.DataFrame(
        {"p": ["p_d"], "source": ["p_d"], "sink": ["water_downstream"]})

    dtt__delay_duration = pl.DataFrame({
        "d":        ["p2020"]*3,
        "t_source": ["t01", "t02", "t03"],
        "t_sink":   ["t02", "t03", "t04"],
        "td":       [1.0, 1.0, 1.0],
    })
    p_process_delay_weight = Param(("p", "td"),
        pl.DataFrame({"p": ["p_d"], "td": [1.0], "value": [1.0]}))

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
        process_indirect=process_indirect,
        process_input_flows=process_input_flows,
        process_output_flows=process_output_flows,
        process_indirect_dt=process_indirect_dt,
        process_delayed=process_delayed,
        process_delayed__duration=process_delayed__duration,
        process_source_delayed=process_source_delayed,
        process_source_undelayed=process_source_undelayed,
        process_source_sink_delayed=process_source_sink_delayed,
        process_source_sink_undelayed=process_source_sink_undelayed,
        dtt__delay_duration=dtt__delay_duration,
        p_process_delay_weight=p_process_delay_weight,
    )


def expected_obj() -> float:
    """obj_commodity = 3 timesteps · 0.1 · 100 · 1.0 · 1.0 · 1.0 = 30.0."""
    return 30.0
