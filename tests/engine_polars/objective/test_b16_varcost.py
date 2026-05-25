"""Surface B.16 — Variable & Commodity Cost (objective contributions).

Closed-form perturbation tests for three currently-untested obj terms:

* B16-3 — eff sink-side O&M  (`+ v_flow * unitsize * p_pdt_varCost_sink * op_factor`)
          model.py:2524-2529
* B16-4 — eff connection process O&M (`+ v_flow * unitsize * p_pdt_varCost_process * op_factor`)
          model.py:2534-2539
* B16-7 — commodity sell revenue (`- v_flow * unitsize * p_commodity_price * op_factor`)
          model.py:2443-2448

Each test perturbs exactly one parameter, holding v_flow fixed by
construction (slack penalty 1e6 dwarfs all cost perturbations / inflow
balance is fully determined), so the obj delta isolates the algebraic
contribution of the term under test.  Pattern mirrors
``tests/engine_polars/perturbation/test_perturb_unitsize_scales_commodity_buy_eff.py``.
"""
from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars._pdt_join import compute_pss_dt
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


# ---------------------------------------------------------------------------
# Helpers (kept inline per AGENT_TEMPLATE "no helper modules").

def _solve(data: FlexData):
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    assert sol.optimal, "LP did not solve to optimality"
    return pb, sol


def _flow_sum(sol) -> float:
    """Σ v_flow over all (p, source, sink, d, t) — every fixture below has
    a single arc whose v_flow ought to be invariant under the perturbation."""
    vf = sol.value("v_flow")
    return float(vf["value"].sum())


# ---------------------------------------------------------------------------
# B16-3 — eff sink-side O&M
#
# Setup: reuse `toy_1n1p_1d2t` (eff process FUEL→n, demand 10/step,
# slack penalty 1e6).  Inject the eff-sink varCost index + param so the
# §5.3 term fires.  v_flow is pinned at 10 per step (slack would cost
# 1e6, FUEL costs 1, so producing 10 dominates).  Doubling the
# sink-side O&M leaves v_flow unchanged ⇒ Δobj = (2-1)·Σ v_flow·unitsize·op.

def test_b16_3_eff_sink_varcost_isolated(toy_1n1p_1d2t):
    base_vc = 1.0
    pert_vc = 2.0

    pss_dt = compute_pss_dt(toy_1n1p_1d2t)   # (p, source, sink, d, t) for the one arc
    # §5.3 expects an eff-unit-sink index keyed (p, source, sink, d, t).
    sink_idx = pss_dt.select("p", "source", "sink", "d", "t")
    # Param keyed (p, sink, d, t) — one row per (sink, d, t).
    sink_param_frame = (sink_idx.select("p", "sink", "d", "t")
                                .with_columns(value=pl.lit(base_vc)))

    base_data = replace(
        toy_1n1p_1d2t,
        pssdt_varCost_eff_unit_sink=sink_idx,
        p_pdt_varCost_sink=Param(("p", "sink", "d", "t"), sink_param_frame),
    )
    _, sol_base = _solve(base_data)
    base_obj = float(sol_base.obj)
    base_flow_sum = _flow_sum(sol_base)

    pert_data = replace(
        base_data,
        p_pdt_varCost_sink=Param(("p", "sink", "d", "t"),
            sink_param_frame.with_columns(value=pl.lit(pert_vc))),
    )
    _, sol_pert = _solve(pert_data)
    pert_obj = float(sol_pert.obj)

    # Sanity: v_flow invariant (cost perturbation << slack penalty 1e6).
    assert _flow_sum(sol_pert) == pytest.approx(base_flow_sum, rel=1e-9)
    # Hand-calc: Δ = (2-1) * Σ v_flow * unitsize(=1) * op_factor(=1*1*1/1=1).
    expected_delta = (pert_vc - base_vc) * base_flow_sum
    assert pert_obj - base_obj == pytest.approx(expected_delta, rel=1e-7)


# ---------------------------------------------------------------------------
# B16-4 — eff connection process O&M
#
# Build a 2-node connection: source node `a` (positive supply +10) →
# sink node `b` (demand 10).  Arc carries v_flow=10 each step, forced
# by nodeBalance.  No commodities, no slope multiplier path.  Inject
# `pssdt_varCost_eff_connection` + `p_pdt_varCost_process` (keyed
# (p, d, t)) and perturb the process O&M.

def _build_eff_connection_data(varcost: float) -> FlexData:
    dt = pl.DataFrame({"d": ["d1", "d1"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))

    nb = pl.DataFrame({"n": ["a", "b"]})
    nb_dt = nb.join(dt, how="cross")
    # `a` supplies +10 (positive inflow), `b` demands -10.
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("n") == "a").then(pl.lit(10.0))
              .otherwise(pl.lit(-10.0))
    ).select("n", "d", "t", "value"))
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    pss = pl.DataFrame({"p": ["link"], "source": ["a"], "sink": ["b"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    # Source-side participation in nodeBalance for node `a` (eff partition,
    # so `flow_from_nodeBalance_eff`).  Slope=1 so the eff branch is just
    # v_flow * unitsize.
    flow_from_nb_eff = pss.clone()

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["link"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("link"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    # Empty commodity wiring (link is not a commodity buy / sell).
    flow_from_commodity_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame(schema={"c": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                              "value": pl.Float64}))

    # §5.4 connection varCost: index over (p, source, sink, d, t),
    # param keyed (p, d, t).
    conn_idx = pss_dt.select("p", "source", "sink", "d", "t")
    conn_param_frame = (dt.with_columns(p=pl.lit("link"),
                                        value=pl.lit(varcost))
                          .select("p", "d", "t", "value"))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen, p_penalty_down=p_pen,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_nodeBalance_eff=flow_from_nb_eff,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper, p_slope=p_slope,
        p_commodity_price=p_commodity_price,
        pssdt_varCost_eff_connection=conn_idx,
        p_pdt_varCost_process=Param(("p", "d", "t"), conn_param_frame),
    )


def test_b16_4_eff_connection_varcost_isolated():
    base_vc, pert_vc = 1.0, 3.0

    _, sol_base = _solve(_build_eff_connection_data(base_vc))
    _, sol_pert = _solve(_build_eff_connection_data(pert_vc))

    base_flow_sum = _flow_sum(sol_base)   # = 10 + 10 = 20 (forced by balance)
    assert _flow_sum(sol_pert) == pytest.approx(base_flow_sum, rel=1e-9)
    # Hand-calc: Δ = (3-1) * Σ v_flow(=20) * unitsize(=1) * op_factor(=1) = 40.
    expected_delta = (pert_vc - base_vc) * base_flow_sum
    assert float(sol_pert.obj) - float(sol_base.obj) == pytest.approx(
        expected_delta, rel=1e-7)


# ---------------------------------------------------------------------------
# B16-7 — commodity sell revenue (negative obj contribution)
#
# One nodeBalance node `n` with positive inflow (+8), one process
# `p_sell` (n → SELL_n) with SELL_n a commodity-priced node.  The only
# outlet for the +8 supply is the sell process, so v_flow=8 is forced
# by balance.  Perturbing the commodity sell price changes only the
# sell-revenue obj term (model.py:2443-2448), with the leading minus
# sign making higher price ⇒ more negative obj.

def _build_commodity_sell_data(price: float) -> FlexData:
    dt = pl.DataFrame({"d": ["d1", "d1"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))

    nb = pl.DataFrame({"n": ["n"]})
    nb_dt = nb.join(dt, how="cross")
    # Positive inflow ⇒ excess must be exported via the sell process.
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(8.0)).select("n", "d", "t", "value"))
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    pss = pl.DataFrame(
        {"p": ["p_sell"], "source": ["n"], "sink": ["SELL_n"]})
    # noEff partition keeps the algebra straight (no slope multiplier).
    pss_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_noEff = pss.clone()
    pss_dt = pss.join(dt, how="cross")
    # SELL_n is NOT in nodeBalance — flow_to_n carries no rows for it.
    flow_to_n = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "n": pl.Utf8})
    # Source-side noEff: subtract v_flow from `n`'s balance.
    flow_from_nb_noEff = pss.clone()

    # flow_to_commodity: (p, source, sink, c) — sink-side commodity sell.
    flow_to_commodity = pl.DataFrame(
        {"p": ["p_sell"], "source": ["n"], "sink": ["SELL_n"], "c": ["EXPORT"]})
    # Empty buy-side commodity frames (term must NOT fire on these).
    flow_from_commodity_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p_sell"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p_sell"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("EXPORT"), value=pl.lit(price))
          .select("c", "d", "t", "value"))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen, p_penalty_down=p_pen,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_nodeBalance_noEff=flow_from_nb_noEff,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        flow_to_commodity=flow_to_commodity,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
    )


def test_b16_7_commodity_sell_revenue_isolated():
    base_price, pert_price = 5.0, 7.0

    _, sol_base = _solve(_build_commodity_sell_data(base_price))
    _, sol_pert = _solve(_build_commodity_sell_data(pert_price))

    # v_flow forced by balance: +8 inflow at `n`, only outlet is p_sell
    # ⇒ v_flow=8 each of 2 steps ⇒ Σ v_flow = 16.
    base_flow_sum = _flow_sum(sol_base)
    assert base_flow_sum == pytest.approx(16.0, rel=1e-9)
    assert _flow_sum(sol_pert) == pytest.approx(base_flow_sum, rel=1e-9)

    # Hand-calc: Δobj = -(price_pert - price_base) * Σ v_flow * unitsize * op
    #                 = -(7-5) * 16 * 1 * 1 = -32.  Leading minus comes from
    # model.py:2446 ``obj = obj - Sum(...)``.
    expected_delta = -(pert_price - base_price) * base_flow_sum
    assert float(sol_pert.obj) - float(sol_base.obj) == pytest.approx(
        expected_delta, rel=1e-7)
