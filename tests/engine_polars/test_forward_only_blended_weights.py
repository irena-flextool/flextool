"""Phase D — ``bind_forward_only_blended_weights`` constraint test.

This RP-blended-weights variant shares all machinery with
``bind_within_solve_blended_weights`` *except* the
``rp_inter_period_cyclic`` end-to-start closure constraint.  Use case:
an RP investment solve where storage state should chain forward across
base periods but the final state need not equal the initial state.

Fixture
-------

Built from the same closed-form scenario as
:mod:`tests.engine_polars.test_rp_blended_weights_minimal` (Phase 6.6 /
specs/rp_blended_weights_test_design.md) — a 1-storage-node fixture
with a +5/-3 inflow profile across two timesteps in a single RP block,
two base periods (b1 → b2, weights 1.0), and slack penalty 1.0.

For ``bind_within_solve_blended_weights``:
    cyclic + balance ⇒ Δs = 0 ⇒ the +2 surplus must spill via slack
    ⇒ ``sol.obj == 2.0``.

For ``bind_forward_only_blended_weights``:
    only ``rp_inter_period_balance`` fires (no cyclic closure).  The
    LP is free to drive ``v_state_inter[b2] - v_state_inter[b1] = +2``
    via ``v_state[t02] - v_state_rp_start = +2`` at zero slack cost
    ⇒ ``sol.obj == 0.0``.

The cyclic constraint is the load-bearing difference between the two
variants; this fixture exercises exactly that difference.
"""
from __future__ import annotations

import polars as pl

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData


def _build_toy_rp_2base_1rep(*, method: str) -> FlexData:
    """Synthesise the minimal RP-blended-weights LP fixture.

    Mirrors :func:`tests.engine_polars.test_rp_blended_weights_minimal
    ._build_toy_rp_2base_1rep`, parameterised on the binding method so a
    single helper covers both within_solve and forward_only variants.
    The only difference is which ``storage_bind_*_blended_weights``
    partition gets the singleton node — ``nodeState_rp`` is the union
    (Phase D extension) so the shared RP machinery fires identically;
    only ``rp_inter_period_cyclic`` is gated on the within_solve subset.
    """
    if method not in (
            "bind_within_solve_blended_weights",
            "bind_forward_only_blended_weights"):
        raise ValueError(f"unsupported method: {method!r}")

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

    dtttdt = pl.DataFrame({
        "d": ["p_rp", "p_rp"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["p_rp", "p_rp"],
        "t_previous_within_solve": ["t02", "t01"],
    })

    # RP-blended-weights structures.  Route the singleton node to the
    # appropriate partition; nodeState_rp is the union (Phase D).
    bat_only = pl.DataFrame({"n": ["bat"]})
    if method == "bind_within_solve_blended_weights":
        storage_bind_within_solve_blended_weights = bat_only
        storage_bind_forward_only_blended_weights = None
    else:  # bind_forward_only_blended_weights
        storage_bind_within_solve_blended_weights = None
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


def _solve(method: str) -> float:
    """Build the fixture, solve, return the objective."""
    d = _build_toy_rp_2base_1rep(method=method)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol.optimal, (
        f"LP did not solve to optimum for method={method!r}; "
        f"status={sol.status!r}"
    )
    return sol.obj


def test_within_solve_vs_forward_only_objective_differs() -> None:
    """The cyclic-closure constraint binds in this fixture.

    With ``bind_within_solve_blended_weights`` the cyclic + balance
    constraints force ``v_state[t02] - v_state_rp_start = 0``, and the
    +2 intra-RP inflow surplus must spill via slack
    (penalty 1.0 × 2 units = cost 2.0).

    With ``bind_forward_only_blended_weights`` only the balance
    constraint fires; the LP drives ``v_state_inter[b2] - v_state_inter[b1]
    = +2`` (paying nothing) and the slack stays at 0 ⇒ cost 0.0.

    Verifies (a) the forward_only solve succeeds (the Phase C
    not-yet-implemented guard has been lifted for forward_only), and
    (b) the objective is strictly less than the within_solve variant
    by 2.0 — the exact hand-derived spill cost.
    """
    obj_within = _solve("bind_within_solve_blended_weights")
    obj_fwd = _solve("bind_forward_only_blended_weights")

    assert abs(obj_within - 2.0) < 1e-9, (
        f"within_solve obj = {obj_within!r}, expected 2.0 (Phase 7+8 "
        f"cyclic+balance forces the +2 surplus to spill via slack)."
    )
    assert abs(obj_fwd - 0.0) < 1e-9, (
        f"forward_only obj = {obj_fwd!r}, expected 0.0 — only "
        f"rp_inter_period_balance should fire and the LP should drive "
        f"v_state_inter[b2] - v_state_inter[b1] = +2 at zero slack "
        f"cost.  If obj > 0 the cyclic constraint is still firing for "
        f"forward_only (Phase D bind_set_rp gating bug); if obj < 0 "
        f"the balance constraint is broken or absent."
    )
    assert obj_fwd < obj_within, (
        f"forward_only ({obj_fwd!r}) must be strictly cheaper than "
        f"within_solve ({obj_within!r}) on this fixture — the LP has "
        f"strictly more freedom (one fewer equality constraint)."
    )


def test_forward_only_matches_no_inter_period_reference() -> None:
    """Indirect proof that ``rp_inter_period_cyclic`` does not fire
    for the forward_only variant.

    Without inter-period coupling the +2 intra-RP surplus accumulates
    into ``v_state[t02] - v_state_rp_start = +2`` at zero cost.  The
    forward_only variant achieves exactly that objective — so the
    cyclic-closure constraint (which would force a 2.0 slack cost)
    must NOT have been emitted for the forward_only node.  This is
    the assertion gate for the Phase D bind_set_rp filter on
    ``rp_inter_period_cyclic`` (model.py — the one constraint that
    intentionally stays scoped to ``storage_bind_within_solve_blended_weights``
    rather than the ``nodeState_rp`` union).
    """
    obj_fwd = _solve("bind_forward_only_blended_weights")
    # Same closed-form "no cyclic" optimum the Phase-6-only canary
    # validated for within_solve before Phases 7+8 landed.
    assert abs(obj_fwd - 0.0) < 1e-9, (
        f"forward_only obj = {obj_fwd!r}, expected 0.0 — equivalent "
        f"to a Phase-6-only build with no inter-period cyclic-closure "
        f"constraint."
    )
