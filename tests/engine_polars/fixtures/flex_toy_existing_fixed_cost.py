"""Synthetic minimal-feature toy: existing-entity fixed cost (§8.1
tripwire) — currently DEFERRED in polar_high.

Probes the constant-offset term

    + Σ_{e ∈ entity, d ∈ period_in_use}
          p_entity_all_existing[e, d] · ed_fixed_cost[e, d]
              · p_inflation_factor_operations_yearly[d] · pd_branch_weight[d]

(``pd_branch_weight ≡ 1.0`` in deterministic single-branch scenarios).

This is a pure constant — the optimiser drops it, but flextool's
reported ``v_obj`` includes it.  polar_high's loaders for
``p_ed_fixed_cost`` and ``p_entity_all_existing`` are populated, but
the objective-side wiring is currently deferred (see
``flextool/model.py:1149`` and ``audit/objective_audit.md`` §8.1).

Topology
--------
* 1 nodeBalance node ``elec``, 1 commodity node ``gas`` priced 10 €/MWh.
* 1 unit process ``u``: source=``gas``, sink=``elec``, eff partition,
  unitsize=100 MW, slope=1.0.
* Demand at ``elec`` = 50 MW each of 2 timesteps (=> v_flow = 0.5).
* ``p_entity_all_existing[u, p2020] = 100`` MW (existing capacity).
* ``p_ed_fixed_cost[u, p2020] = 50`` €/MW/period (non-zero).

Closed-form
-----------
Dispatch obj:
    obj_commodity = 2 · 0.5 · 100 · 1.0 · 10 · 1.0 = 1000.0
    obj_slack     = 0.

§8.1 fixed-cost constant:
    obj_fc = 100 · 50 · 1.0 · 1.0 = 5000.0

Total expected obj (when §8.1 is wired) = 6000.0.

Currently polar_high returns 1000.0 only — the ``xfail(strict=True)`` mark
flips this test from ``passing`` to ``xfailed`` while §8.1 is deferred.
When §8.1 lands, the test auto-passes (XPASS → strict-fail → review),
prompting removal of the xfail mark.
"""
from __future__ import annotations

import polars as pl
from polar_high import Param
from flextool.engine_polars.input import FlexData


def data() -> FlexData:
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
                       "t": ["t01", "t02"], "value": [-50.0, -50.0]}))
    p_penalty_up = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1e9, 1e9]}))
    p_penalty_down = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["elec"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1e9, 1e9]}))

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

    # ── §8.1 existing-entity fixed cost ────────────────────────────────
    p_entity_all_existing = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["p2020"], "value": [100.0]}))
    p_ed_fixed_cost = Param(("e", "d"),
        pl.DataFrame({"e": ["u"], "d": ["p2020"], "value": [50.0]}))

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
        p_entity_all_existing=p_entity_all_existing,
        p_ed_fixed_cost=p_ed_fixed_cost,
    )


def expected_obj() -> float:
    """When §8.1 is wired, expected obj = dispatch + fixed_cost_constant.

    obj_commodity = 1000.0,  obj_§8.1 = 5000.0  =>  total = 6000.0.
    """
    return 1000.0 + 5000.0
