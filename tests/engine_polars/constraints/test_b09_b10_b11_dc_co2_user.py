"""Surface B.9 / B.10 / B.11 — DC power flow back-flow + CO2 cap cardinality
+ user-defined constraint coefficient terms.

* ``test_b09_v_angle_bounds_and_back_flow_capacity`` — covers **B9-3**:
  builds an in-test 2-bus DC island with a single unidirectional arc and
  reverse-direction demand, verifies ``maxToSink_back`` binds at
  ``existing/unitsize`` and that ``v_angle`` honours its loose ±π bounds.

* ``test_b10_co2_max_period_g_d_cardinality`` — covers **B10-4**:
  two groups, two periods; only ``(g1, d1)`` carries a cap.  Asserts
  ``co2_max_period`` emits exactly one row (the (g, d) cross-product is
  filtered through ``group_d_co2_capped``, NOT inflated to 2×2=4).

* ``test_b10_co2_max_total_absent`` — structural finding (xfail-strict):
  pin that no ``co2_max_total*`` constraint name exists in the polars
  engine.  Will flip to PASS — and the xfail must be removed — when
  the v3.32.0 ``co2_max_total`` is ported (see
  ``../flextool-e/flextool.mod:4019-4055``).

* ``test_b11_user_cstr_state_and_prebuilt_coefs`` — covers **B11-3**,
  **B11-4**, **B11-5** in one fixture: storage node with
  ``p_node_constraint_state_coefficient``, a process with
  ``p_process_constraint_prebuilt_capacity_coefficient`` (pre-summed
  constant) AND ``edd_invest_lookback_set`` so the cumulative-invest
  variable contribution is exercised on the lookback period.

* ``test_b11_empty_lhs_short_circuit`` — covers **B11-6**: a fixture with
  ``cdt_le`` populated but ALL coefficient/contribution Params None;
  asserts the engine does NOT emit any ``process_constraint_*`` row
  (silent drop).
"""
from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


def _solve(data: FlexData) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# Local DC PF fixture — 2-bus island, single forward arc.

def _build_dc_2bus(
    *, susceptance: float, sink_inflow: float,
    existing_cap: float, unitsize: float = 1.0,
) -> FlexData:
    """Two buses (n1 ref, n2), one DC line p (n1→n2 only — no reverse arc).
    ``sink_inflow`` is the inflow at n2 (negative = demand pulling power).
    ``existing_cap`` populates ``p_flow_upper_existing``; the back-flow
    bound is ``p_flow_upper_existing / p_unitsize`` already (units of
    capacity).
    """
    dt = pl.DataFrame({"d": ["d1"], "t": ["t01"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp   = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_psh  = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))

    # nodeBalance: both buses, n2 has demand (or surplus).
    nb = pl.DataFrame({"n": ["n1", "n2"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.when(pl.col("n") == "n2")
                                   .then(pl.lit(sink_inflow))
                                   .otherwise(pl.lit(-sink_inflow)))
             .select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # DC arc: forward only, n1→n2.  v_flow ≥ 0 is the forward direction;
    # reverse uses v_flow_back.
    pss = pl.DataFrame({"p": ["line"], "source": ["n1"], "sink": ["n2"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    # Source-side debit: keep the forward arc in flow_from_nodeBalance_noEff
    # so n1's balance sees -v_flow when v_flow > 0.
    flow_from_nodeBalance_noEff = pss.clone()
    flow_from_nodeBalance_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    # Required by the PROCESSES feature gate even when the source-side
    # carries no commodity (DC arc has no fuel).  Empty frames are OK.
    flow_from_commodity_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame(schema={"c": pl.Utf8, "d": pl.Utf8,
                              "t": pl.Utf8, "value": pl.Float64}))

    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["line"], "value": [unitsize]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(existing_cap / unitsize))
              .select("p", "source", "sink", "d", "t", "value"))
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss.join(pl.DataFrame({"d": ["d1"]}), how="cross")
           .with_columns(value=pl.lit(existing_cap / unitsize))
           .select("p", "source", "sink", "d", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("line"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))

    # DC PF wiring.
    node_dc = pl.DataFrame({"n": ["n1", "n2"]})
    conn_dc = pl.DataFrame({"p": ["line"]})
    ref = pl.DataFrame({"n": ["n1"]})
    p_susc = Param(("p",),
        pl.DataFrame({"p": ["line"], "value": [susceptance]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_nodeBalance_eff=flow_from_nodeBalance_eff,
        flow_from_nodeBalance_noEff=flow_from_nodeBalance_noEff,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_flow_upper_existing=p_flow_upper_existing,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        node_dc_power_flow=node_dc,
        connection_dc_power_flow=conn_dc,
        node_reference_angle=ref,
        p_connection_susceptance=p_susc,
    )


# ---------------------------------------------------------------------------
# B9-3 — v_angle loose ±π bounds + back-flow capacity binding (maxToSink_back).

def test_b09_v_angle_bounds_and_back_flow_capacity():
    """Covers B9-3 — reverse-direction demand forces ``v_flow_back`` to bind
    at existing/unitsize; ``v_angle`` honours ±π loose bounds.

    Setup: 2-bus DC island.  susceptance=1.0, unitsize=1.0, existing=2.0
    on the only arc (n1→n2).  ``sink_inflow=+5`` at n2 ⇒ NET surplus at
    n2 (n1 has -5 demand).  Power must flow n2→n1, but the arc only has
    a forward (n1→n2) v_flow ≥ 0; reverse goes via ``v_flow_back``.

    Hand-calc:
      Reverse-flow needed = 5 MW; capacity = existing/unitsize = 2.
      So v_flow_back = 2 (binds maxToSink_back), v_flow = 0.
      Slack absorbs the 3 MW gap at each side (vq_up at n1, vq_down at n2).
      Angle: dc_flow_eq → (0 - 2)*1 = 1*(θ_n1 - θ_n2)
             with θ_n1 = 0 (ref) ⇒ θ_n2 = +2.0  (within ±π).
    """
    d = _build_dc_2bus(
        susceptance=1.0, sink_inflow=5.0, existing_cap=2.0, unitsize=1.0)
    pb, sol = _solve(d)
    assert sol.optimal
    names = set(pb.cstr_names())
    assert "dc_flow_eq" in names
    assert "dc_reference_angle_eq" in names
    assert "maxToSink_back" in names
    # Hand-calc: v_flow_back binds at 2.0; v_flow stays at 0.
    vfb = sol.value("v_flow_back")["value"][0]
    assert vfb == pytest.approx(2.0, rel=1e-7), (
        f"v_flow_back={vfb} != 2.0 — maxToSink_back not binding")
    # Hand-calc: θ_n1=0 (ref pin), θ_n2 = -(0 - 2)*1/1 = 2.0 (within ±π).
    ang = sol.value("v_angle").sort("n")
    ang_n1 = ang.filter(pl.col("n") == "n1")["value"][0]
    ang_n2 = ang.filter(pl.col("n") == "n2")["value"][0]
    assert ang_n1 == pytest.approx(0.0, abs=1e-9), (
        f"reference angle {ang_n1} != 0")
    assert ang_n2 == pytest.approx(2.0, rel=1e-7), (
        f"v_angle[n2]={ang_n2} != 2.0 — dc_flow_eq sign wrong")
    # And the loose Var bound must allow |angle| < π (we sit comfortably below).
    assert abs(ang_n2) < 3.14159265


# ---------------------------------------------------------------------------
# B10-4 — co2_max_period emits exactly one row per (g, d) in
# group_d_co2_capped — NOT a g × d cross-product over all periods.

def _toy_2node_chp_with_co2(d: FlexData, *, two_periods: bool = False,
                              caps: dict[tuple[str, str], float] | None = None
                              ) -> FlexData:
    """Add CO2 cap wiring on top of toy_2node_chp.  ``caps`` maps
    ``(g, d) → cap`` and pins both ``group_d_co2_capped`` and
    ``p_co2_max_period``.
    """
    # Mark FUEL with co2_content=2.0; the input arc fuel→chp goes through
    # the eff partition (slope=1.0, no skew).
    p_co2_content = Param(("c",),
        pl.DataFrame({"c": ["FUEL"], "value": [2.0]}))
    flow_from_co2_capped = pl.DataFrame({
        "p": ["chp"], "source": ["fuel"], "sink": ["chp"],
        "c": ["FUEL"], "g": ["g1"]})
    if caps is None:
        caps = {("g1", "d1"): 4.0}
    g_d = pl.DataFrame({
        "g": [k[0] for k in caps],
        "d": [k[1] for k in caps],
    })
    p_cap = Param(("g", "d"),
        pl.DataFrame({"g": [k[0] for k in caps],
                      "d": [k[1] for k in caps],
                      "value": list(caps.values())}))
    return dataclasses.replace(d,
        p_co2_content=p_co2_content,
        flow_from_co2_capped=flow_from_co2_capped,
        group_d_co2_capped=g_d,
        p_co2_max_period=p_cap,
    )


def test_b10_co2_max_period_g_d_cardinality(toy_2node_chp):
    """Covers B10-4 — ``co2_max_period`` row count matches
    ``group_d_co2_capped``, NOT ``groups × periods``.

    Setup: extend toy_2node_chp with a 2nd period d2 (same wiring) and
    define caps ONLY for ``(g1, d1)`` and ``(g1, d2)`` (g2 has no rows).

    Hand-calc: row count = len(group_d_co2_capped) = 2 (NOT 4 = 2 g × 2 d
    or 1 g × 2 d × |c|).
    """
    d = toy_2node_chp
    # Extend dt to add d2.
    dt2 = pl.DataFrame({"d": ["d1", "d1", "d2", "d2"],
                        "t": ["t01", "t02", "t01", "t02"]})
    p_step = Param(("d", "t"), dt2.with_columns(value=pl.lit(1.0)))
    p_rp   = Param(("d", "t"), dt2.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1", "d2"], "value": [1.0, 1.0]}))
    p_psh  = Param(("d",), pl.DataFrame({"d": ["d1", "d2"], "value": [1.0, 1.0]}))
    nb_dt2 = d.nodeBalance.join(dt2, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt2.with_columns(value=pl.when(pl.col("n") == "heat")
                                     .then(-5.0).otherwise(-10.0))
              .select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt2.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt2.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    pss_dt2 = d.process_source_sink.join(dt2, how="cross")
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt2.with_columns(value=pl.lit(1.0))
               .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt2.with_columns(p=pl.lit("chp"), value=pl.lit(1.0))
           .select("p", "d", "t", "value"))
    process_indirect_dt = d.process_indirect.join(dt2, how="cross")
    d2 = dataclasses.replace(d,
        dt=dt2, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance_dt=nb_dt2, p_inflow=p_inflow,
        p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        pss_dt=pss_dt2, p_flow_upper=p_flow_upper, p_slope=p_slope,
        process_indirect_dt=process_indirect_dt,
    )
    # Caps only on (g1, d1) and (g1, d2) — g2 absent on purpose.
    d2 = _toy_2node_chp_with_co2(d2, caps={("g1", "d1"): 100.0,
                                            ("g1", "d2"): 100.0})
    pb, sol = _solve(d2)
    assert sol.optimal
    # Hand-calc: row count == len(group_d_co2_capped) == 2.
    assert pb.cstr_row_count("co2_max_period") == 2, (
        f"co2_max_period row count {pb.cstr_row_count('co2_max_period')} != 2 "
        f"— (g, d) filter dropped or cross-product inflated")


# ---------------------------------------------------------------------------
# B.10 — structural finding: co2_max_total does NOT exist in polars engine.
# Pin with xfail-strict so the test flips to FAIL (= remove this xfail)
# when the v3.32.0 port lands.

@pytest.mark.xfail(strict=True,
    reason="co2_max_total port pending — see ../flextool-e/flextool.mod:4019-4055")
def test_b10_co2_max_total_absent_xfail(toy_2node_chp):
    """Structural finding — once the lifetime-cap variant is ported, the
    polars engine will emit a ``co2_max_total`` constraint name; this
    xfail will then trip (strict) and the test must be deleted.
    """
    d = _toy_2node_chp_with_co2(toy_2node_chp)
    pb, _ = _solve(d)
    names = set(pb.cstr_names())
    # When implemented, expect at least one constraint whose name matches.
    assert any(n.startswith("co2_max_total") for n in names), (
        f"co2_max_total* missing from {sorted(names)}")


# ---------------------------------------------------------------------------
# B11-3, B11-4, B11-5 — user-cstr state coef + prebuilt const + cumulative
# invest contribution, all in a single 2-period fixture.

def _add_user_constraint_storage_invest(d: FlexData) -> FlexData:
    """Layer a 2-period storage + investable-process user-constraint over
    a base toy_1n1p_1d2t fixture.  Adds:
      * storage node ``s`` with state_coef on user constraint ``c1``
      * process ``p`` with prebuilt-capacity coef AND invest-lookback
      * ``cdt_le`` carrying (c1, d2, t01) — d2 is the lookback target
    """
    # Two periods so edd_invest_lookback_set is non-empty for d2.
    dt = pl.DataFrame({"d": ["d1", "d2"], "t": ["t01", "t01"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp   = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",),
        pl.DataFrame({"d": ["d1", "d2"], "value": [1.0, 1.0]}))
    p_psh  = Param(("d",),
        pl.DataFrame({"d": ["d1", "d2"], "value": [1.0, 1.0]}))

    # Two nodes: ``n`` (regular) and ``s`` (storage).  Inflow = 0 → process
    # output is unconstrained by demand; user constraint is the only driver.
    nb = pl.DataFrame({"n": ["n", "s"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # Process p — investable, has 1 existing unit, unitsize=2.
    pss = pl.DataFrame({"p": ["p"], "source": ["FUEL_n"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p"], "source": ["FUEL_n"], "sink": ["n"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [2.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss.join(pl.DataFrame({"d": ["d1", "d2"]}), how="cross")
           .with_columns(value=pl.lit(1.0))
           .select("p", "source", "sink", "d", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    # Storage on ``s``.
    nodeState = pl.DataFrame({"n": ["s"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first()
        .select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last()
        .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [3.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["s", "s"], "d": ["d1", "d2"], "value": [1.0, 1.0]}))
    p_state_self_discharge = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [0.0]}))
    p_state_existing_capacity = Param(("n", "d"),
        pl.DataFrame({"n": ["s", "s"], "d": ["d1", "d2"], "value": [3.0, 3.0]}))
    dtttdt = pl.DataFrame({
        "d": ["d1", "d2"], "t": ["t01", "t01"],
        "t_previous": ["t01", "t01"],
        "t_previous_within_timeset": ["t01", "t01"],
        "d_previous": ["d1", "d2"],
        "t_previous_within_solve": ["t01", "t01"],
    })

    # Invest setup for p — investable in d1, alive in d2 too.
    pd_invest_set = pl.DataFrame({"p": ["p", "p"], "d": ["d1", "d2"]})
    ed_invest_set = pd_invest_set.rename({"p": "e"})
    edd_invest_set = pl.DataFrame({
        "e": ["p"]*3, "d_invest": ["d1", "d1", "d2"], "d": ["d1", "d2", "d2"]})
    # Lookback: d_invest STRICTLY < d.  So only (p, d1, d2).
    edd_invest_lookback_set = pl.DataFrame({
        "e": ["p"], "d_invest": ["d1"], "d": ["d2"]})
    # Max units = 1 so v_invest_p[p, d1] ∈ [0, 1].  ed_invest_max_period
    # is divided by unitsize (= 2) inside the engine, so set it to 2.0 to
    # keep the per-period cap loose at 1.0.  Coupled with the tight RHS
    # this forces all three LHS pieces (state, prebuilt-const, cum-invest)
    # to contribute their max simultaneously.
    p_entity_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["p", "p"], "d": ["d1", "d2"], "value": [1.0, 1.0]}))
    ed_invest_period_set = ed_invest_set.clone()
    ed_invest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["p", "p"], "d": ["d1", "d2"], "value": [2.0, 2.0]}))
    ed_entity_annual_discounted = Param(("e", "d"),
        pl.DataFrame({"e": ["p", "p"], "d": ["d1", "d2"], "value": [10.0, 10.0]}))
    ed_lifetime_fixed_cost = Param(("e", "d"),
        pl.DataFrame({"e": ["p", "p"], "d": ["d1", "d2"], "value": [0.0, 0.0]}))
    p_process_existing_count = Param(("p", "d"),
        pl.DataFrame({"p": ["p", "p"], "d": ["d1", "d2"], "value": [1.0, 1.0]}))

    # User constraint axes: c1 sense ``<=`` over (c, d, t) — apply at d2/t01
    # only so the lookback cumulative-invest term (d_invest=d1 < d=d2) fires.
    cdt_le = pl.DataFrame({"c": ["c1"], "d": ["d2"], "t": ["t01"]})
    p_constraint_constant = Param(("c",),
        pl.DataFrame({"c": ["c1"], "value": [20.0]}))
    # State coef: 1.0 on (s, c1).  Contributes v_state[s,d2,t01] * 3 * 1.
    p_node_constraint_state_coefficient = Param(("n", "c"),
        pl.DataFrame({"n": ["s"], "c": ["c1"], "value": [1.0]}))
    # Prebuilt process coef: 1.0 on (p, c1).
    # Pre-summed constant = existing_count[p,d2] * unitsize[p] * coef
    #                     = 1 * 2 * 1 = 2  → adds to LHS.
    # Cumulative invest contribution = v_invest_p[p, d_invest=d1] * unitsize * 1
    #                                = v_invest_p[p, d1] * 2.
    p_process_constraint_prebuilt_capacity_coefficient = Param(("p", "c"),
        pl.DataFrame({"p": ["p"], "c": ["c1"], "value": [1.0]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_flow_upper_existing=p_flow_upper_existing,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
        p_process_existing_count=p_process_existing_count,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_self_discharge,
        p_state_existing_capacity=p_state_existing_capacity,
        dtttdt=dtttdt,
        pd_invest_set=pd_invest_set,
        ed_invest_set=ed_invest_set,
        edd_invest_set=edd_invest_set,
        edd_invest_lookback_set=edd_invest_lookback_set,
        p_entity_max_units=p_entity_max_units,
        ed_invest_period_set=ed_invest_period_set,
        ed_invest_max_period=ed_invest_max_period,
        ed_entity_annual_discounted=ed_entity_annual_discounted,
        ed_lifetime_fixed_cost=ed_lifetime_fixed_cost,
        cdt_le=cdt_le,
        p_constraint_constant=p_constraint_constant,
        p_node_constraint_state_coefficient=p_node_constraint_state_coefficient,
        p_process_constraint_prebuilt_capacity_coefficient=
            p_process_constraint_prebuilt_capacity_coefficient,
    )


def test_b11_user_cstr_state_and_prebuilt_coefs():
    """Covers B11-3 + B11-4 + B11-5 in a single fixture.

    LHS of user-constraint c1 at (d2, t01), sense ``>=``:
       state_term:     v_state[s,d2,t01] * state_unitsize(3) * coef(1)
       prebuilt_const: existing_count(1) * unitsize(2) * coef(1) = 2
       cum_invest:     v_invest_p[p, d1] * unitsize(2) * coef(1)
    Bounds: v_state <= state_upper(1), v_invest_p[d1] <= entity_max_units(1).
    Max attainable LHS = 1*3 + 2 + 1*2 = 7.

    Pick RHS = 7 so the constraint is tight: ALL three LHS contributions
    must hit their max simultaneously.  Hand-calc unique solution:
       v_state[s,d2,t01] = 1,  v_invest_p[p,d1] = 1.
    If state_coef were dropped:        max LHS = 0+2+2 = 4 < 7 -> infeasible
    If prebuilt-const were dropped:    max LHS = 3+0+2 = 5 < 7 -> infeasible
    If cum-invest piece were dropped:  max LHS = 3+2+0 = 5 < 7 -> infeasible
    so the test fails (infeasible) if ANY of B11-3/B11-4/B11-5 regresses.
    """
    d = _add_user_constraint_storage_invest(_make_dummy())
    # Switch sense to >= so the constraint forces all LHS terms to bind.
    cdt_ge = d.cdt_le
    d = dataclasses.replace(d, cdt_le=None, cdt_ge=cdt_ge)
    p_const = Param(("c",),
        pl.DataFrame({"c": ["c1"], "value": [7.0]}))
    d = dataclasses.replace(d, p_constraint_constant=p_const)
    pb, sol = _solve(d)
    assert sol.optimal, (
        "infeasible — at least one of state-coef / prebuilt-const / "
        "cum-invest contributions to user-constraint LHS is missing")
    assert "process_constraint_greater_than" in set(pb.cstr_names())
    # Hand-calc: v_state must hit cap 1 to provide state_term = 3 (B11-3).
    v_state = sol.value("v_state").filter(
        (pl.col("n") == "s") & (pl.col("d") == "d2"))["value"][0]
    assert v_state == pytest.approx(1.0, rel=1e-6), (
        f"v_state[s,d2]={v_state} != 1.0 — state-coef path (B11-3) absent")
    # Hand-calc: v_invest_p[p, d1] must hit cap 1 (B11-5; cumulative-invest
    # via edd_invest_lookback_set on (p, d1, d2) — B11-4 prebuilt-const
    # supplies the constant 2 piece without which feasibility is impossible).
    v_inv = sol.value("v_invest_p").filter(
        (pl.col("p") == "p") & (pl.col("d") == "d1"))["value"][0]
    assert v_inv == pytest.approx(1.0, rel=1e-6), (
        f"v_invest_p[p,d1]={v_inv} != 1.0 — cumulative-invest contribution "
        "(B11-5) absent")


def _make_dummy() -> FlexData:
    """Tiny placeholder — _add_user_constraint_storage_invest builds the
    whole FlexData from scratch and ignores its argument; this keeps the
    helper signature uniform with the broader test suite.
    """
    dt = pl.DataFrame({"d": ["d1"], "t": ["t01"]})
    return FlexData(
        dt=dt,
        p_step_duration=Param(("d", "t"), dt.with_columns(value=pl.lit(1.0))),
        p_rp_cost_weight=Param(("d", "t"), dt.with_columns(value=pl.lit(1.0))),
        p_inflation_op=Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]})),
        p_period_share=Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]})),
        nodeBalance=pl.DataFrame({"n": ["x"]}),
        nodeBalance_dt=pl.DataFrame({"n": ["x"], "d": ["d1"], "t": ["t01"]}),
        p_inflow=Param(("n", "d", "t"),
            pl.DataFrame({"n": ["x"], "d": ["d1"], "t": ["t01"], "value": [0.0]})),
        p_penalty_up=Param(("n", "d", "t"),
            pl.DataFrame({"n": ["x"], "d": ["d1"], "t": ["t01"], "value": [1e6]})),
        p_penalty_down=Param(("n", "d", "t"),
            pl.DataFrame({"n": ["x"], "d": ["d1"], "t": ["t01"], "value": [1e6]})),
    )


# ---------------------------------------------------------------------------
# B11-6 — empty-frame short-circuit: cdt_le rows present but ALL LHS
# coefficient Params are None ⇒ no process_constraint_* row emitted.

def test_b11_empty_lhs_short_circuit(toy_1n1p_1d2t):
    """Covers B11-6 — when ``cdt_le`` is non-empty but every coefficient
    source (flow / invest / state / prebuilt) is empty, the engine
    short-circuits the inner ``if lhs_pieces or cstr_const_pieces``
    block and emits NO constraint row, even though ``has_any_cdt`` is
    True.  Pins the silent-drop edge case in model.py:1953-1971.
    """
    d = toy_1n1p_1d2t
    # cdt_le populated, p_constraint_constant populated, but NO contribution
    # Params (no flow_constraint_idx, no invest/state/prebuilt coefs).
    cdt_le = pl.DataFrame({"c": ["c1"], "d": ["d1"], "t": ["t01"]})
    p_const = Param(("c",),
        pl.DataFrame({"c": ["c1"], "value": [0.0]}))
    data = dataclasses.replace(d,
        cdt_le=cdt_le,
        p_constraint_constant=p_const,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    # Hand-calc: even though has_any_cdt is True, no LHS pieces ⇒ NO
    # process_constraint_* row is registered.
    names = set(pb.cstr_names())
    leaked = [n for n in names if n.startswith("process_constraint_")]
    assert leaked == [], (
        f"empty LHS should silently short-circuit; got "
        f"process_constraint_* rows: {leaked}")
