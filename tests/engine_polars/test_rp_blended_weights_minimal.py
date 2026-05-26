"""Hand-calculable RP-blended-weights minimal test (Phases 7-9 exerciser).

Designed as part of Phase 6.6 (see
``specs/rp_blended_weights_test_design.md``) to give the next agent a
small, closed-form fixture that distinguishes the *with-inter-period*
optimum from the *without-inter-period* optimum by a fixed cost
increment of 2.0 — entirely from hand arithmetic, no black-box solver
required for the golden.

Scenario
--------
* 1 storage node ``bat`` (both nodeBalance and nodeState).
* 1 period ``p_rp`` containing the rep block: 2 timesteps ``t01`` /
  ``t02``.
* 2 base periods ``b1``, ``b2``; chain ``b1 → b2``.
* 1 RP block starting at ``t01``, last step ``t02``.
* Weights: ``p_rp_weight[b1, t01] = p_rp_weight[b2, t01] = 1.0`` (both
  base periods map to the same rep with weight 1).
* Inflow profile: ``+5`` at t01, ``-3`` at t02 → net intra-RP surplus
  ``+2``.  Slack penalty 1.0 each direction.  Capacity generous (100).

Expected LP optima
------------------
* **Phase 6 only** (intra-period state-change only; current Phase
  6.5 HEAD): the +2 surplus accumulates into ``v_state[t02] -
  v_state_rp_start = +2`` at no slack cost.  ``sol.obj = 0.0``.
* **Phase 7 + 8** (``rp_inter_period_balance`` + ``..._cyclic``): the
  sum of cyclic + balance LHS = 0 over the 2-base-period chain forces
  ``v_state[t02] - v_state_rp_start = 0``.  The +2 imbalance must
  spill via slack; cost = ``1.0 × 2 = 2.0`` (penalty × qty).

Phase 9 (``rp_inter_period_max_state``) is *emitted* but inactive
here — ``v_state_inter`` has no coupling beyond the chain, defaults to
0, and the capacity bound (100) is non-binding.  See ``specs/`` for
the surfaced Phase-9 verification gap.

Phases 7-9 are now landed (HEAD `1f445528`), so this test PASSES with
the hand-derived golden ``sol.obj == 2.0``.  Phase 10 stripped the
strict-xfail mark and deleted the Phase-6-only baseline canary.

The full derivation is in
``specs/rp_blended_weights_test_design.md`` §B.1.
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData


def _build_toy_rp_2base_1rep() -> FlexData:
    """Synthesise the minimal RP-blended-weights LP fixture.

    See module docstring for scenario and specs/ for arithmetic.
    """
    dt = pl.DataFrame({"d": ["p_rp", "p_rp"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rpcw = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",),
        pl.DataFrame({"d": ["p_rp"], "value": [1.0]}))
    p_psh = Param(("d",),
        pl.DataFrame({"d": ["p_rp"], "value": [1.0]}))

    nb = pl.DataFrame({"n": ["bat"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), pl.DataFrame({
        "n": ["bat"] * 2, "d": ["p_rp"] * 2,
        "t": ["t01", "t02"], "value": [5.0, -3.0]}))
    p_pup = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0))
             .select("n", "d", "t", "value"))
    p_pdn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0))
             .select("n", "d", "t", "value"))

    nodeState = pl.DataFrame({"n": ["bat"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first()
        .select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last()
        .select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [1.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p_rp"], "value": [100.0]}))
    p_state_sd = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [0.0]}))
    p_state_exi_cap = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p_rp"], "value": [100.0]}))

    # Cyclic-within-timeset over 2 steps in p_rp.
    dtttdt = pl.DataFrame({
        "d": ["p_rp", "p_rp"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["p_rp", "p_rp"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    # RP-blended-weights structures.
    storage_bind_within_solve_blended_weights = pl.DataFrame({"n": ["bat"]})
    nodeState_rp = pl.DataFrame({"n": ["bat"]})
    rp_base_period_set = pl.DataFrame({"b": ["b1", "b2"]})
    # Chain b2 ← b1 (Phase 7 active edge).
    rp_base_chain = pl.DataFrame({"b": ["b2"], "b_prev": ["b1"]})
    rp_base_first = pl.DataFrame({"b": ["b1"]})
    rp_base_last = pl.DataFrame({"b": ["b2"]})
    rp_block_first = pl.DataFrame({"d": ["p_rp"], "t": ["t01"]})
    p_rp_last_step = pl.DataFrame({"r": ["t01"], "last_step": ["t02"]})
    rp_base__rep = Param(("b", "r"), pl.DataFrame({
        "b": ["b1", "b2"], "r": ["t01", "t01"], "value": [1.0, 1.0]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rpcw,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pup, p_penalty_down=p_pdn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_sd,
        p_state_existing_capacity=p_state_exi_cap,
        dtttdt=dtttdt,
        storage_bind_within_solve_blended_weights=storage_bind_within_solve_blended_weights,
        nodeState_rp=nodeState_rp,
        rp_base_period_set=rp_base_period_set,
        rp_base_chain=rp_base_chain,
        rp_base_first=rp_base_first,
        rp_base_last=rp_base_last,
        rp_block_first=rp_block_first,
        p_rp_last_step=p_rp_last_step,
        rp_base__rep=rp_base__rep,
    )


def test_rp_2base_1rep_cyclic_forces_two_unit_spill():
    """Phase 7+8: cyclic forces Δs = 0, slack absorbs the +2 surplus.

    Hand-derivation summary (full version in specs/):

        @ t01 (rp_block_first):
            (v_state_rp_start - v_state[t01]) + vq_up - vq_dn = -5
        @ t02 (interior):
            (v_state[t01]    - v_state[t02])   + vq_up - vq_dn = +3

    Cyclic + balance (b1 ↔ b2, weights 1,1) ⇒
        2 · (v_state[t02] - v_state_rp_start) = 0
        ⇒ v_state[t02] = v_state_rp_start
        ⇒ net slack imbalance = -2
        ⇒ min(vq_up + vq_dn) = 2,  cost = 1.0 × 2 = 2.0.
    """
    d = _build_toy_rp_2base_1rep()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol.optimal, "LP did not solve to optimum"
    assert abs(sol.obj - 2.0) < 1e-9, (
        f"obj = {sol.obj!r}, expected 2.0.  Phase 7+8 should emit "
        f"rp_inter_period_balance + ..._cyclic and force the +2 "
        f"intra-RP surplus to spill via slack (cost = 2.0).  "
        f"Cost 0.0 ⇒ constraints missing (Phase 6 only).  Other "
        f"value ⇒ partial/incorrect constraint emission — inspect "
        f"the LP build."
    )
