"""RP-blended-weights inter-period balance must be var-unit scaled.

Regression guard for a mis-scaling in
``model.py``'s ``rp_inter_period_balance`` / ``rp_inter_period_cyclic``
RHS: the seasonal-drift term ``(v_state_last - v_state_rp_start)`` was
multiplied by ``p_state_unitsize`` even though the LHS
``v_state_inter`` delta — and both drift vars — are already in
var-units (physical = var × unitsize).  The extra factor inflates the
inter-period state drift by ``unitsize``.

Every other blended-weights test (``test_rp_blended_weights_minimal``,
``test_forward_only_blended_weights``, ``test_within_period_blended_weights``)
uses ``p_state_unitsize == 1``, where ``× unitsize`` is a no-op — so the
bug is invisible there.  This fixture sets ``unitsize == 1000`` (the
schema default) and reads out a PHYSICAL observable, so the factor
becomes load-bearing.

Fixture (a scaled clone of the minimal ``forward_only`` fixture)
----------------------------------------------------------------
* 1 storage node ``bat``; ``p_state_unitsize == 1000``.
* 1 period ``p_rp`` with a 2-step rep block ``t01`` / ``t02``, inflow
  ``+5 / -3`` → net intra-RP physical surplus ``+2``.
* 2 base periods chained ``b1 → b2`` (``bind_forward_only`` — only
  ``rp_inter_period_balance`` fires, no cyclic closure).
* ``p_state_upper == 1.0`` (var-units) ⇒ physical capacity
  ``1.0 × 1000 = 1000`` energy units — comfortably above the ``+2``
  surplus, so the store CAN absorb the whole surplus at zero slack.

Hand-derived optima
-------------------
With no slack the intra-RP surplus fixes the var-unit drift
``D = v_state[t02] - v_state_rp_start = +2 / unitsize = 0.002``
(so the physical drift ``D × unitsize == +2``).

``rp_inter_period_balance`` (edge b1→b2, weight 1) reads
``v_state_inter[b2] - v_state_inter[b1] == weight × D`` in var-units, and
``rp_inter_period_max_state`` caps ``v_state_inter ≤ p_state_upper = 1.0``.

* **Correct (var-unit) RHS**: ``v_state_inter[b2] = 0.002`` — far under
  the ``1.0`` cap.  The store absorbs the whole surplus, no slack.
  **obj == 0.0**.
* **Buggy (× unitsize) RHS**: the drift is inflated by ``1000`` to
  ``2.0`` var-units, which overflows the ``1.0`` cap.  The LP must
  spill ``1`` physical unit via slack (penalty 1.0) to keep
  ``v_state_inter[b2] ≤ 1.0``.  **obj == 1.0**.

The ``1.0`` gap is exactly one factor of the (inflated) drift clipped
at capacity — i.e. the bug's signature.  At ``unitsize == 1`` the two
RHS forms coincide, which is why the existing tests never caught it.

See the parent ``test_forward_only_blended_weights`` for the
``unitsize == 1`` sibling of this fixture.
"""
from __future__ import annotations

import polars as pl

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

UNITSIZE = 1000.0
# Var-unit capacity on v_state_inter.  Physical cap = STATE_CAP * UNITSIZE
# = 1000 energy units, far above the +2 physical surplus — so the store
# can hold the whole surplus.  Under the buggy (x unitsize) RHS the
# var-unit drift inflates to 2.0 and overflows this cap.
STATE_CAP = 1.0


def _build_toy_rp_unitsize() -> FlexData:
    """Minimal forward_only RP-blended-weights fixture at unitsize=1000.

    Structurally identical to
    ``test_forward_only_blended_weights._build_toy_rp_2base_1rep(
    method="bind_forward_only_blended_weights")`` except
    ``p_state_unitsize`` resolves to 1000 (not 1) and the state cap is
    tightened to 1.0 var-units so the inter-period drift's scaling is
    observable in the objective.
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
    # Net +2 physical surplus over the rep block.
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
    # The load-bearing knob: unitsize != 1.
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [UNITSIZE]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p_rp"], "value": [STATE_CAP]}))
    p_state_sd = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [0.0]}))
    p_state_exi_cap = Param(("n", "d"),
        pl.DataFrame({"n": ["bat"], "d": ["p_rp"],
                      "value": [STATE_CAP * UNITSIZE]}))

    dtttdt = pl.DataFrame({
        "d": ["p_rp", "p_rp"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["p_rp", "p_rp"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    # forward_only: only rp_inter_period_balance fires (no cyclic).
    bat_only = pl.DataFrame({"n": ["bat"]})
    storage_bind_forward_only_blended_weights = bat_only
    nodeState_rp = bat_only

    rp_base_period_set = pl.DataFrame({"b": ["b1", "b2"]})
    rp_base_chain = pl.DataFrame({"b": ["b2"], "b_prev": ["b1"]})
    rp_base_first = pl.DataFrame({"b": ["b1"]})
    rp_base_last = pl.DataFrame({"b": ["b2"]})
    rp_block_first = pl.DataFrame({"d": ["p_rp"], "t": ["t01"]})
    p_rp_last_step = pl.DataFrame({"r": ["t01"], "last_step": ["t02"]})
    rp_base__rep = Param(("b", "r"), pl.DataFrame({
        "b": ["b1", "b2"], "r": ["t01", "t01"], "value": [1.0, 1.0]}))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_timestep_weight=p_rpcw,
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
        storage_bind_forward_only_blended_weights=storage_bind_forward_only_blended_weights,
        nodeState_rp=nodeState_rp,
        rp_base_period_set=rp_base_period_set,
        rp_base_chain=rp_base_chain,
        rp_base_first=rp_base_first,
        rp_base_last=rp_base_last,
        rp_block_first=rp_block_first,
        p_rp_last_step=p_rp_last_step,
        rp_base__rep=rp_base__rep,
    )


def test_inter_period_drift_is_var_unit_scaled() -> None:
    """The inter-period drift must NOT scale with unitsize.

    Physical capacity (1000 energy units) dwarfs the +2 physical
    surplus, so the correct optimum absorbs the whole surplus at zero
    slack: **obj == 0.0**.

    If ``rp_inter_period_balance`` multiplies the drift RHS by
    ``p_state_unitsize`` (the bug), the var-unit drift is inflated
    1000x to 2.0, overflowing the 1.0 var-unit cap on
    ``v_state_inter``; the LP is forced to spill 1 physical unit via
    slack and **obj == 1.0**.  That 1.0 is exactly the unitsize
    mis-scaling clipped at capacity.
    """
    d = _build_toy_rp_unitsize()
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol.optimal, f"LP did not solve to optimum; status={sol.status!r}"
    assert abs(sol.obj - 0.0) < 1e-9, (
        f"obj = {sol.obj!r}, expected 0.0.  A non-zero objective (≈1.0) "
        f"means rp_inter_period_balance inflated the inter-period drift "
        f"by p_state_unitsize (= {UNITSIZE}), overflowing the var-unit "
        f"state cap and forcing spurious slack spill.  The inter-period "
        f"balance must couple var-unit v_state_inter deltas to var-unit "
        f"(v_state_last - v_state_rp_start) drifts WITHOUT a unitsize "
        f"factor (consistent with rp_inter_period_max_state / "
        f"maxState_rp_start)."
    )
