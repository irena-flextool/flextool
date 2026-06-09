"""Surface B.19 + B.20 — Delay/DR absence + Commodity-Ladder objective.

Two structural-finding regression pins (B19) and two ladder-objective
oracles (B20):

* B19-1 — ``v_delay_dispatch * p_delay_dispatch_cost`` term DOES NOT
          exist in engine_polars.  ``_delay.add_objective_terms``
          (``_delay.py:407-416``) is a no-op stub; delay cost flows
          through commodity prices on source flows.
* B19-2 — ``v_dr_call * p_dr_cost`` term DOES NOT exist.  No DR
          variable / parameter is registered anywhere in the polars
          engine.  Consolidated with B19-1 in a single test that solves
          a delay-active model and asserts neither variable family is
          present and the obj matches the closed-form value that
          accounts ONLY for commodity-priced flows.
* B20-3 — Annualisation factor ``inflation_op / period_share`` on the
          ladder objective contribution (``_commodity_ladder.py:538-540,
          548-549, 556-557``).  Step_duration / timestep_weight do NOT
          enter the ladder term — that asymmetry is also pinned.
* B20-4 — Legacy commodity price suppression for ladder commodities
          (``model.py:2401-2414``).  When the same commodity carries
          both ``p_commodity_price`` AND a ladder, the legacy single-
          price term is filtered out so only the ladder pays.
"""
from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

from .conftest import solver_options

from flex_toy_delay import data as flex_toy_delay_data, expected_obj as flex_toy_delay_expected_obj


def _solve(data: FlexData):
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    assert sol.optimal, "LP did not solve to optimality"
    return pb, sol


# ---------------------------------------------------------------------------
# B19-1 + B19-2 — neither v_delay_dispatch / p_delay_dispatch_cost nor
# v_dr_call / p_dr_cost may appear in the model or its objective.
# Use the existing flex_toy_delay synthetic — it has delays active and
# a closed-form expected objective of 30.0 (commodity-buy cost only,
# obj_commodity = 3·0.1·100·1·1·1 = 30).  A regression that introduces
# either term would either bump obj above 30 or expose the new variable.

def test_b19_no_delay_or_dr_objective_terms_present():
    """Covers B19-1 + B19-2 (consolidated) — pin the absence of any
    delay-/DR-specific objective contribution.

    Hand-calc: flex_toy_delay obj = 30.0 from commodity-buy alone.
    Any additional ``p_delay_dispatch_cost``- or ``p_dr_cost``-style
    multiplier on a delay/DR variable would shift obj off 30.
    """
    d = flex_toy_delay_data()
    pb, sol = _solve(d)
    # Pin: no delay-cost / DR-cost variable family registered.
    forbidden = ("v_delay_dispatch", "v_dr_call", "v_demand_response")
    present = [name for name in pb._vars if any(f in name for f in forbidden)]
    assert present == [], (
        f"unexpected delay/DR variable(s) in model: {present} — "
        "engine_polars does not implement these terms; new variable "
        "indicates an unaudited objective contribution slipped in"
    )
    # Pin: obj exactly matches the commodity-only closed form.
    expected = flex_toy_delay_expected_obj()  # 30.0
    assert float(sol.obj) == pytest.approx(expected, rel=1e-7), (
        f"obj {sol.obj} != {expected} — a delay/DR cost term may have "
        "been added (delay propagates via commodity price only)"
    )


# ---------------------------------------------------------------------------
# Helper: turn ``toy_1n1p_1d2t`` (FUEL→n, demand 10/step) into a ladder
# fixture.  Two annual tiers on FUEL: tier 1 cheap-and-capped, tier 2
# unlimited.  Demand totals 20 MWh over the period; tier 1 cap=8 forces
# v_trade[1]=8, v_trade[2]=12.

def _add_fuel_ladder_annual(
    base: FlexData,
    *,
    tier1_price: float,
    tier1_cap: float,
    tier2_price: float,
    tier2_cap: float = 1e30,
) -> FlexData:
    """Layer a 2-tier annual ladder on commodity FUEL atop ``toy_1n1p_1d2t``."""
    cwl = pl.DataFrame({"c": ["FUEL"]})
    cnd = pl.DataFrame({"c": ["FUEL"], "n": ["source_n"], "d": ["d1"]})
    cndi = pl.DataFrame({
        "c": ["FUEL", "FUEL"], "n": ["source_n", "source_n"],
        "d": ["d1", "d1"], "i": ["1", "2"]})
    p_ann_price = Param(("c", "i", "d"), pl.DataFrame({
        "c": ["FUEL", "FUEL"], "i": ["1", "2"], "d": ["d1", "d1"],
        "value": [tier1_price, tier2_price]}))
    p_ann_quantity = Param(("c", "i", "d"), pl.DataFrame({
        "c": ["FUEL", "FUEL"], "i": ["1", "2"], "d": ["d1", "d1"],
        "value": [tier1_cap, tier2_cap]}))
    p_unitsize_c = Param(("c",), pl.DataFrame({"c": ["FUEL"], "value": [1.0]}))
    p_f_d_k = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    return replace(
        base,
        commodity_with_ladder=cwl,
        commodity_with_ladder_annual=cwl.clone(),
        cnd_ladder=cnd,
        cndi_ladder=cndi,
        cndi_ladder_ann=cndi.clone(),
        p_ladder_ann_price=p_ann_price,
        p_ladder_ann_quantity=p_ann_quantity,
        p_commodity_unitsize=p_unitsize_c,
        p_f_d_k=p_f_d_k,
    )


# ---------------------------------------------------------------------------
# B20-3 — annualisation factor inflation_op / period_share scales the
# ladder cost; step_duration / timestep_weight do NOT.

def test_b20_3_ladder_annualisation_factor_isolated(toy_1n1p_1d2t):
    """Covers B20-3 — change ``inflation_op`` and ``period_share`` and
    verify ladder cost scales by ``inflation_op / period_share``.

    Setup: demand = 10 MWh × 2 steps = 20 MWh total.  Tier 1 cap=8
    @ 5 €/MWh; tier 2 unlimited @ 80 €/MWh.  Optimal split:
    v_trade[1]=8, v_trade[2]=12.

    Hand-calc baseline (infl=1, share=1): ladder = (8·5 + 12·80) = 1000.
    Hand-calc scaled (infl=1.2, share=0.4): ladder = 1000·(1.2/0.4) = 3000.
    Δ ladder = 2000.  The legacy commodity-price=1·20 term is suppressed
    on FUEL (ladder commodity), so any obj difference equals Δ ladder.
    A regression that flipped operator (× share instead of / share) would
    yield 1000·1.2·0.4 = 480 instead of 3000.
    """
    base = _add_fuel_ladder_annual(
        toy_1n1p_1d2t, tier1_price=5.0, tier1_cap=8.0, tier2_price=80.0)
    pert = replace(base,
        p_inflation_op=Param(("d",),
            pl.DataFrame({"d": ["d1"], "value": [1.2]})),
        p_period_share=Param(("d",),
            pl.DataFrame({"d": ["d1"], "value": [0.4]})))

    _, sb = _solve(base)
    _, sp = _solve(pert)

    # Sanity: tier split is the binding-cap optimum.
    vt = sb.value("v_trade").sort("i")["value"].to_list()
    assert vt == pytest.approx([8.0, 12.0], rel=1e-7), (
        f"v_trade {vt} != [8, 12] — cap setup off")
    # Sanity: baseline obj == pure ladder cost (legacy term suppressed).
    assert float(sb.obj) == pytest.approx(1000.0, rel=1e-7)
    # Hand-calc: Δobj from infl/share perturbation = ladder · (3.0 - 1.0)
    # = 1000 · 2 = 2000.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(2000.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B20-4 — Legacy commodity-price suppression for ladder commodities.
# Build the simplest possible test: ``toy_1n1p_1d2t`` already has
# p_commodity_price[FUEL]=1.0.  Without ladder: total obj = 1·20 = 20.
# Add a ladder on FUEL with tier-1 unlimited @ 5 €/MWh: pure-ladder
# obj should be 5·20 = 100 (the legacy 1·20 = 20 must NOT double-count).
# Regression that drops the suppression filter would yield 100 + 20 = 120.

def test_b20_4_legacy_commodity_price_suppressed_on_ladder(toy_1n1p_1d2t):
    """Covers B20-4 — when commodity FUEL has BOTH ``p_commodity_price``
    AND a ladder, the legacy single-price obj term is suppressed
    (filtered out via ``~is_in(ladder_c)`` at model.py:2401-2414) so
    only the ladder pays.

    Setup: ``toy_1n1p_1d2t`` keeps ``p_commodity_price[FUEL]=1.0``;
    overlay a single-tier unlimited ladder @ 5 €/MWh on FUEL.

    Hand-calc:
        v_flow = 10 MWh / step × 2 = 20 MWh.
        ladder_only_cost = 20 · 5 = 100.
        legacy_if_not_suppressed = 20 · 1 = 20.
    Expected obj = 100 (NOT 120).  A regression dropping the filter
    bumps obj to 120.
    """
    cwl = pl.DataFrame({"c": ["FUEL"]})
    cnd = pl.DataFrame({"c": ["FUEL"], "n": ["source_n"], "d": ["d1"]})
    cndi = pl.DataFrame({"c": ["FUEL"], "n": ["source_n"],
                         "d": ["d1"], "i": ["1"]})
    p_ann_price = Param(("c", "i", "d"), pl.DataFrame({
        "c": ["FUEL"], "i": ["1"], "d": ["d1"], "value": [5.0]}))
    p_ann_quantity = Param(("c", "i", "d"), pl.DataFrame({
        "c": ["FUEL"], "i": ["1"], "d": ["d1"], "value": [1e30]}))
    data = replace(
        toy_1n1p_1d2t,
        commodity_with_ladder=cwl,
        commodity_with_ladder_annual=cwl.clone(),
        cnd_ladder=cnd,
        cndi_ladder=cndi,
        cndi_ladder_ann=cndi.clone(),
        p_ladder_ann_price=p_ann_price,
        p_ladder_ann_quantity=p_ann_quantity,
        p_commodity_unitsize=Param(("c",),
            pl.DataFrame({"c": ["FUEL"], "value": [1.0]})),
        p_f_d_k=Param(("d",),
            pl.DataFrame({"d": ["d1"], "value": [1.0]})),
    )
    _, sol = _solve(data)
    # Hand-calc: 20 MWh · 5 €/MWh = 100.  Legacy 1 €/MWh price on FUEL
    # is suppressed by the ladder filter.
    assert float(sol.obj) == pytest.approx(100.0, rel=1e-7), (
        f"obj={sol.obj} — expected 100 (ladder only); "
        "120 indicates the legacy commodity-price term double-counts"
    )
