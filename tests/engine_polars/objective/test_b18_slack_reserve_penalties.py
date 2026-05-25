"""Surface B.18 — Slack & Reserve Penalty obj contributions.

Closed-form perturbation tests for six currently-untested terms:

* B18-2 — ``vq_state_down · p_penalty_down · p_node_capacity_for_scaling
          · op_factor``  (model.py:2389-2390)
* B18-3 — ``vq_state_down · p_penalty_down · op_factor`` no-scaling
          fallback branch (model.py:2391-2393)
* B18-4 — ``vq_reserve · pdtReserve_upDown_group_reservation
          · p_reserve_upDown_group_penalty_reserve · op_factor``
          (_reserve.py:581-609)
* B18-5 — ``vq_inertia · pdGroup_inertia_limit · pdGroup_penalty_inertia
          · op_factor``  (_group_slack.py:1158-1164)
* B18-6 — ``vq_non_synchronous · p_group_capacity_for_scaling
          · pdGroup_penalty_non_synchronous · op_factor``
          (_group_slack.py:1166-1172)
* B18-7 — ``vq_capacity_margin · p_group_capacity_for_scaling
          · pdGroup_penalty_capacity_margin · p_inflation_op``
          (_group_slack.py:1174-1182).  CRITICAL: NO step_duration /
          rp_cost_weight / period_share / pdt_branch_weight — only
          ``inflation_op`` applies; the asymmetry is load-bearing.

Pattern: solve baseline, perturb exactly one penalty parameter, solve
again, assert ``Δobj == Δpen · vq · weights`` per the model formulas.
B18-7 additionally pins the asymmetry by perturbing the penalty under
``step_duration=2, rp_cost_weight=2, period_share=0.5`` and verifying
Δobj does NOT scale with the temporal weights.
"""
from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars._pdt_join import compute_nodeBalance_dt
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


def _solve(data: FlexData):
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    assert sol.optimal, "LP did not solve to optimality"
    return pb, sol


# ---------------------------------------------------------------------------
# B18-2 + B18-3 — vq_state_down penalty with / without capacity scaling.
#
# Use ``toy_costs_only_1d2t`` as the spine: process is FUEL→n which can
# only ADD to node n (sink-side).  Setting p_inflow=+5 leaves no outlet,
# so vq_state_down absorbs all 5 each step (LP optimum: vq_state_down=5
# at every (d, t), v_flow=0).  Then perturb p_penalty_down and verify
# Δobj = Δpen · Σ vq_state_down · scale · op_factor.

def _make_drain_data(base: FlexData, *, pen_down: float,
                      ncs: float | None) -> FlexData:
    """Inject positive inflow → vq_state_down active; set penalty / ncs."""
    nb_dt = compute_nodeBalance_dt(base)
    p_inflow_pos = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(5.0))
              .select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(pen_down))
              .select("n", "d", "t", "value"))
    extras: dict = {}
    if ncs is not None:
        extras["p_node_capacity_for_scaling"] = Param(("n", "d"),
            pl.DataFrame({"n": ["n"], "d": ["d1"], "value": [ncs]}))
    return replace(base, p_inflow=p_inflow_pos, p_penalty_down=p_pen_dn,
                   **extras)


def test_b18_2_vq_state_down_with_capacity_scaling(toy_costs_only_1d2t):
    """Covers B18-2 (direct) + B18-3 (consolidated via the ``ncs=None`` arm).

    Hand-calc: vq_state_down=5 at each of 2 steps ⇒ Σ vq=10.
    With ncs=2: Δobj = (1500-1000)·10·2·op_factor(=1) = 10000.
    Without ncs (B18-3): Δobj = (1500-1000)·10·op_factor(=1) = 5000.
    """
    base = _make_drain_data(toy_costs_only_1d2t, pen_down=1000.0, ncs=2.0)
    pert = _make_drain_data(toy_costs_only_1d2t, pen_down=1500.0, ncs=2.0)
    _, sb = _solve(base)
    _, sp = _solve(pert)
    # Sanity: vq_state_down pinned at 5 each step in both runs.
    vqb = sb.value("vq_state_down")["value"].sum()
    vqp = sp.value("vq_state_down")["value"].sum()
    assert vqb == pytest.approx(10.0, rel=1e-9)
    assert vqp == pytest.approx(vqb, rel=1e-9)
    # Hand-calc: Δ = (1500-1000) · 10 · ncs(=2) · op_factor(=1) = 10000.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(10000.0, rel=1e-7)


def test_b18_3_vq_state_down_no_capacity_scaling_fallback(toy_costs_only_1d2t):
    """Covers B18-3 — when ``p_node_capacity_for_scaling is None`` the
    code takes the fallback branch (model.py:2391-2393) which omits the
    ncs factor entirely (NOT a silent ×1).
    """
    base = _make_drain_data(toy_costs_only_1d2t, pen_down=1000.0, ncs=None)
    pert = _make_drain_data(toy_costs_only_1d2t, pen_down=1500.0, ncs=None)
    _, sb = _solve(base)
    _, sp = _solve(pert)
    assert sb.value("vq_state_down")["value"].sum() == pytest.approx(10.0, rel=1e-9)
    # Hand-calc: Δ = (1500-1000) · 10 · op_factor(=1·1·1/1=1) = 5000.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(5000.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B18-4 — vq_reserve penalty.  toy_group_reserve already wires reserve
# (r1, up, g) with reservation=10 and penalty=500.  At baseline, u
# (cap=100, max_share=1.0, reliability=1) supplies all 10 of reserve so
# vq_reserve=0.  Force the slack by raising reservation above what u
# can offer (drop max_share to 0.0 → reserve provider contributes nothing).

def test_b18_4_vq_reserve_penalty_isolated(toy_group_reserve):
    """Covers B18-4 — perturb ``p_reserve_upDown_group_penalty_reserve``;
    Δobj == Δpen · Σ vq_reserve · pdtReserve · op_factor (op=1 here).

    Hand-calc: max_share=0 ⇒ no provider ⇒ reserveBalance forces vq=1
    (the slack is capped at 1 in _reserve.py:273).  reservation=4 each
    step ⇒ Σ over 2 steps of (vq=1 · res=4) = 8.  Δpen=200 ⇒ Δobj=1600.
    """
    d = toy_group_reserve
    # max_share=0 disables the only provider so the slack carries the load.
    p_max_share = Param(("p", "r", "ud", "n"),
        pl.DataFrame({"p": ["u"], "r": ["r1"], "ud": ["up"], "n": ["n1"],
                      "value": [0.0]}))
    # Reservation = 4 each step (cap on vq_reserve is 1, so enters as ·4).
    pdtRes = Param(("r", "ud", "g", "d", "t"),
        pl.DataFrame({"r": ["r1"]*2, "ud": ["up"]*2, "g": ["g"]*2,
                      "d": ["d1"]*2, "t": ["t01", "t02"],
                      "value": [4.0, 4.0]}))
    base = replace(d, p_process_reserve_upDown_node_max_share=p_max_share,
                   pdtReserve_upDown_group_reservation=pdtRes,
                   p_reserve_upDown_group_penalty_reserve=Param(
                       ("r", "ud", "g"),
                       pl.DataFrame({"r": ["r1"], "ud": ["up"], "g": ["g"],
                                     "value": [500.0]})))
    pert = replace(base, p_reserve_upDown_group_penalty_reserve=Param(
        ("r", "ud", "g"),
        pl.DataFrame({"r": ["r1"], "ud": ["up"], "g": ["g"],
                      "value": [700.0]})))
    _, sb = _solve(base)
    _, sp = _solve(pert)
    # Sanity: vq_reserve at upper bound 1 each step.
    vq = sb.value("vq_reserve").sort(["d", "t"])["value"].to_list()
    assert vq == pytest.approx([1.0, 1.0], rel=1e-7)
    # Hand-calc: Δ = (700-500) · Σ(vq·res) = 200 · (1·4 + 1·4) = 1600.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(1600.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B18-5 + B18-6 — inertia & non_sync slack penalties.  Both live on
# group constraints in toy_group_reserve; we add the relevant gates and
# perturb the penalty parameter.  Capacity_margin isolated separately
# in B18-7 (test below).

def test_b18_5_vq_inertia_penalty_isolated(toy_group_reserve):
    """Covers B18-5 — with NO inertia-providing process in the group, the
    inertia constraint LHS is just slack·limit, so vq_inertia is pinned
    to 1.0 each step.  Perturb pdGroup_penalty_inertia and check Δobj.

    Hand-calc: vq=1 each of 2 steps; Δobj = Σ(vq · limit · Δpen · op_factor)
    = 2 · 1 · 100 · (300-200) · 1 = 20000.
    """
    d = toy_group_reserve
    base = replace(d,
        groupInertia=pl.DataFrame({"g": ["g"]}),
        pdGroup_inertia_limit=Param(("g", "d"),
            pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [100.0]})),
        pdGroup_penalty_inertia=Param(("g", "d"),
            pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [200.0]})),
        # Disable capacity_margin / reserve to keep the obj clean of
        # other penalty changes when we resolve.
        groupCapacityMargin=None, pdGroup_capacity_margin=None,
        pdGroup_penalty_capacity_margin=None,
        reserve_upDown_group=None,
        reserve_upDown_group_method_timeseries=None,
        prundt=None, process_reserve_upDown_node_active=None,
        pdtReserve_upDown_group_reservation=None,
        p_reserve_upDown_group_penalty_reserve=None,
        p_process_reserve_upDown_node_max_share=None,
        p_process_reserve_upDown_node_reliability=None,
    )
    pert = replace(base, pdGroup_penalty_inertia=Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [300.0]})))
    _, sb = _solve(base)
    _, sp = _solve(pert)
    vq = sb.value("vq_inertia").sort(["d", "t"])["value"].to_list()
    assert vq == pytest.approx([1.0, 1.0], rel=1e-7)
    # Hand-calc: Δ = 2 · 1 · 100 · (300-200) · 1 = 20000.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(20000.0, rel=1e-7)


def test_b18_6_vq_non_synchronous_penalty_isolated(toy_group_reserve):
    """Covers B18-6 — with a non-sync incoming flow that exceeds the
    limit, vq_non_synchronous absorbs the violation; Δobj scales with
    p_group_capacity_for_scaling and pdGroup_penalty_non_synchronous.

    Set up: u (cap=100·1=100, demand=50 forces v_flow=0.5) is sink_nonSync;
    non_sync_limit=0 ⇒ the entire incoming flow is in violation.
    Constraint (per (g, d, t)): -vq·step_dur + v_flow·unitsize·step_dur·invc
                                ≤ outgoing_RHS · limit · invc = 0
    With invc=1, step_dur=1, v_flow=0.5, unitsize=100: 50 - vq ≤ 0
    ⇒ vq ≥ 50 each step (LP minimises ⇒ vq=50).

    Hand-calc: Δobj = 2 steps · vq(=50) · scale(=1) · Δpen(=300)
                    · op_factor(=1) = 30000.
    """
    d = toy_group_reserve
    # Force demand of 50 at n1 only (so v_flow at u = 0.5 of unitsize=100).
    nb_dt = compute_nodeBalance_dt(d)
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("n") == "n1").then(-50.0).otherwise(0.0)
    ).select("n", "d", "t", "value"))
    base = replace(d,
        p_inflow=p_inflow,
        groupNonSync=pl.DataFrame({"g": ["g"]}),
        process_sink_nonSync=pl.DataFrame({"p": ["u"], "sink": ["n1"]}),
        pdGroup_non_synchronous_limit=Param(("g", "d"),
            pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [0.0]})),
        pdGroup_penalty_non_synchronous=Param(("g", "d"),
            pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [200.0]})),
        # Drop reserve + capacity_margin to keep obj clean.
        groupCapacityMargin=None, pdGroup_capacity_margin=None,
        pdGroup_penalty_capacity_margin=None,
        reserve_upDown_group=None,
        reserve_upDown_group_method_timeseries=None,
        prundt=None, process_reserve_upDown_node_active=None,
        pdtReserve_upDown_group_reservation=None,
        p_reserve_upDown_group_penalty_reserve=None,
        p_process_reserve_upDown_node_max_share=None,
        p_process_reserve_upDown_node_reliability=None,
    )
    pert = replace(base, pdGroup_penalty_non_synchronous=Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [500.0]})))
    _, sb = _solve(base)
    _, sp = _solve(pert)
    vq = sb.value("vq_non_synchronous").sort(["d", "t"])["value"].to_list()
    assert vq == pytest.approx([50.0, 50.0], rel=1e-7)
    # Hand-calc: Δ = 2 · 50 · 1(scale) · (500-200) · 1 = 30000.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(30000.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B18-7 — vq_capacity_margin penalty: NO step_duration / rp_cost_weight /
# period_share scaling — only p_inflation_op.
#
# Strategy: solve under non-trivial temporal weights (step_duration=2,
# rp_cost_weight=2, period_share=0.5) and perturb the capacity_margin
# penalty.  If the asymmetry holds, Δobj == Δpen · vq · scale · infl.
# A regression that re-uses the full op_factor would inflate Δobj by
# step·rpcw/psh = 2·2/0.5 = 8×.

def test_b18_7_vq_capacity_margin_no_step_duration_scaling(toy_group_reserve):
    """Covers B18-7 — capacity_margin penalty must scale ONLY by
    inflation_op, NOT by step_duration / rp_cost_weight / period_share.

    Setup: raise pdGroup_capacity_margin to 200 so producer cap (=100)
    cannot meet demand+margin (50+200=250) ⇒ vq_capacity_margin=150.
    Set step_duration=2, rp_cost_weight=2, period_share=0.5: a regression
    that mistakenly applies these to the slack penalty would 8× Δobj.

    Hand-calc: Δobj = vq(=150) · scale(=1) · Δpen(=400) · infl(=1) = 60000.
    A bug that re-uses op_factor would yield 60000 · (2·2/0.5) = 480000.
    """
    d = toy_group_reserve
    # Heavy temporal weights to expose any leak into the capacity_margin term.
    dt = d.dt
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(2.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(2.0)))
    p_psh = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [0.5]}))
    base = replace(d,
        p_step_duration=p_step, p_rp_cost_weight=p_rp, p_period_share=p_psh,
        pdGroup_capacity_margin=Param(("g", "d"),
            pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [200.0]})),
        pdGroup_penalty_capacity_margin=Param(("g", "d"),
            pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1000.0]})),
        # Drop reserve to avoid cross-perturbation noise.
        reserve_upDown_group=None,
        reserve_upDown_group_method_timeseries=None,
        prundt=None, process_reserve_upDown_node_active=None,
        pdtReserve_upDown_group_reservation=None,
        p_reserve_upDown_group_penalty_reserve=None,
        p_process_reserve_upDown_node_max_share=None,
        p_process_reserve_upDown_node_reliability=None,
    )
    pert = replace(base, pdGroup_penalty_capacity_margin=Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["d1"], "value": [1400.0]})))
    _, sb = _solve(base)
    _, sp = _solve(pert)
    vq = sb.value("vq_capacity_margin").sort(["g", "d"])["value"].to_list()
    assert vq == pytest.approx([150.0], rel=1e-7), (
        f"vq_capacity_margin {vq} != 150 — capacity_margin RHS or LHS off")
    # Hand-calc: Δ = 150 · 1(scale) · (1400-1000) · 1(infl) · 1000 = 60_000_000.
    # The trailing ×1000 is the BUG A4 unit conversion (CUR/kW → CUR/MW)
    # added in commit 3a4b2aa5 to ``_group_slack.py`` and the parity
    # emitter ``calc_slacks.py`` — capacity-margin slack penalties are
    # stored as CUR/kW in the DB and the LP coefficient lifts them to
    # CUR/MW.  The test's original 60_000.0 expectation predates that
    # commit and was bit-rotted by it.
    # A regression that mis-applies op_factor would yield 60_000_000·8
    # = 480_000_000 (the 8× step_duration/rp_cost_weight/period_share
    # composition); we still catch that here at rel=1e-7.
    assert float(sp.obj) - float(sb.obj) == pytest.approx(60_000_000.0, rel=1e-7)
