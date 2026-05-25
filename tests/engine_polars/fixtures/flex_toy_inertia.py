"""Synthetic minimal-feature toy: inertia constraint in isolation.

A small dispatch scenario with one unit-type process providing inertia
on the sink side.  The inertia constraint binds with margin so that
``vq_inertia = 0``: any breakage in the inertia LHS coefficients (the
``inertia_constant × unitsize`` factor) or in the constraint emission
(missing rows, wrong domain) shows up either in the obj (slack
spuriously priced) or in the row-count assertion.

Topology
--------
* 1 node ``elec`` (in nodeBalance), 1 commodity node ``gas``
  (commodity ``GAS_COM`` with price 10 €/MWh).
* 1 unit process ``u``: source=``gas``, sink=``elec``, eff partition,
  ``p_unitsize[u] = 100`` MW, ``p_slope = 1.0`` (no eff loss).
* Demand at ``elec`` = 100 MW each of 2 timesteps (=> v_flow = 1.0
  in unitsize units; the unit dispatches at full capacity).
* 1 group ``g``: ``group_node = {(g, elec)}``;
  ``pdGroup_inertia_limit[g, p2020] = 200`` MJ;
  ``pdGroup_penalty_inertia[g, p2020] = 1e6`` €/(MJ·s).
* ``process_sink_inertia = {(u, elec)}``;
  ``p_process_sink_inertia_constant[u, elec] = 5`` MJ/MW.

Closed-form expected obj
------------------------
The inertia constraint per (g, d, t):
    LHS  = v_flow[u, gas, elec, d, t] · inertia_constant · unitsize
         + vq_inertia[g, d, t] · pdGroup_inertia_limit
    RHS  = pdGroup_inertia_limit (=200)

With v_flow forced to 1.0 by the demand and unitsize=100,
inertia_constant=5: LHS_first_term = 1.0 · 5 · 100 = 500 ≥ 200, so
vq_inertia = 0 and the slack term in the obj is zero.

Active obj contributions:
    obj_commodity = Σ_{d,t} v_flow · unitsize · slope · price · op_factor
                  = 2 · 1.0 · 100 · 1.0 · 10 · 1.0 = 2000 €
    obj_slack     = 0  (nodeBalance satisfied exactly)
    obj_inertia   = 0  (vq_inertia = 0)

Total expected obj = 2000.0.

Why this toy matters
--------------------
Verifies that ``inertia_constraint`` is *emitted* (1 row per (g, d, t),
i.e. 1 × 2 = 2 rows here), that its LHS uses the correct
``inertia_constant × unitsize`` coefficient, and that it does *not*
spuriously trigger the slack penalty when the floor is met.
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

    # Nodes
    nodeBalance = pl.DataFrame({"n": ["elec"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [-100.0, -100.0]}))
    p_penalty_up = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1e9, 1e9]}))
    p_penalty_down = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1e9, 1e9]}))

    # Process topology
    pss = pl.DataFrame({"p": ["u"], "source": ["gas"], "sink": ["elec"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))

    # Commodity hookup
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["u"], "source": ["gas"], "sink": ["elec"], "c": ["GAS_COM"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["u"], "value": [100.0]}))
    # p_flow_upper = existing / unitsize.  For 1 unit of capacity 100 / 100 = 1.0.
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

    # ── Inertia data ──────────────────────────────────────────────────
    groupInertia = pl.DataFrame({"g": ["g"]})
    group_node = pl.DataFrame({"g": ["g"], "n": ["elec"]})
    process_sink_inertia = pl.DataFrame({"p": ["u"], "sink": ["elec"]})
    p_process_sink_inertia_constant = Param(("p", "sink"),
        pl.DataFrame({"p": ["u"], "sink": ["elec"], "value": [5.0]}))
    pdGroup_inertia_limit = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["p2020"], "value": [200.0]}))
    pdGroup_penalty_inertia = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["p2020"], "value": [1.0e6]}))

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
        # inertia
        groupInertia=groupInertia,
        group_node=group_node,
        process_sink_inertia=process_sink_inertia,
        p_process_sink_inertia_constant=p_process_sink_inertia_constant,
        pdGroup_inertia_limit=pdGroup_inertia_limit,
        pdGroup_penalty_inertia=pdGroup_penalty_inertia,
    )


def expected_obj() -> float:
    """obj_commodity = 2 timesteps · v_flow=1.0 · 100 · slope=1.0 · 10 · 1.0 = 2000."""
    return 2000.0
