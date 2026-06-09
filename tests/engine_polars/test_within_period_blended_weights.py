"""Phase E — ``bind_within_period_blended_weights`` constraint test.

This RP-blended-weights variant implements a per-FlexTool-period cyclic
closure: each period's blended-weights chain closes WITHIN ITSELF
rather than across the whole solve.  Use case (the spec's motivating
example): a multi-period investment model where each FlexTool period
represents a future year (e.g. 2025, 2030, 2035) and the underlying
weather time series is contiguous within a year but NOT contiguous
across years (gap-year semantics), so the cross-year state link must
be dropped.

Fixture
-------

Two FlexTool periods ``y2025`` and ``y2030``, each with one base
period and one representative block of two timesteps.  Inflow profile
is engineered so the three variants land on three different objective
values:

* Period ``y2025`` carries a +2 net intra-RP surplus
  (``+5/-3`` over its two steps).
* Period ``y2030`` carries a -1 net intra-RP deficit
  (``-2/+1`` over its two steps).

Each base period maps to its own rep with weight 1.0 (so the "drift"
seen by ``rp_inter_period_balance`` / ``rp_inter_period_cyclic`` at a
base ``b`` is exactly the inflow imbalance over its own period's rep
block).  Slack penalty = 1.0 each direction, capacity generous.

Hand-derived optima
-------------------

Let Δ₂₅ = +2 (period-1 surplus) and Δ₃₀ = -1 (period-2 deficit).
``v_state_inter[b]`` is unbounded above by capacity but the
state-change is bounded by what inflow + slack can absorb.

* ``bind_forward_only_blended_weights`` — no closure constraint of any
  kind.  Each base's state-change can absorb its own period's drift
  for free: ``Δ₂₅ → v_state_inter`` jump of +2, ``Δ₃₀`` → -1.  No
  slack needed.  **obj = 0.0**.

* ``bind_within_solve_blended_weights`` — single across-solve cycle:
  the chain ``b25 → b30`` closes back end-to-start.  Sum of all drifts
  around the closed loop must equal 0, i.e. ``Δ₂₅ + Δ₃₀ + slack = 0``
  ⇒ ``2 + (-1) + slack = 0`` ⇒ ``slack = -1`` (one unit of state_up
  spill in EITHER period — the LP picks whichever is cheapest;
  symmetric penalties → either works at cost 1.0).  **obj = 1.0**.

* ``bind_within_period_blended_weights`` — each period closes its own
  cycle independently.  Period ``y2025``: ``Δ₂₅ + slack₂₅ = 0`` ⇒
  ``slack₂₅ = 2`` (spill down 2 units, cost 2.0).  Period ``y2030``:
  ``Δ₃₀ + slack₃₀ = 0`` ⇒ ``slack₃₀ = 1`` (spill up 1 unit, cost
  1.0).  **obj = 3.0**.

These three numbers are the load-bearing assertions below.  Any drift
indicates a regression in the cyclic-closure topology or the per-
period join semantics added in Phase E to
``model.py``'s ``rp_inter_period_cyclic`` block.

If the orthogonal forward_only sanity check is interesting, see
``test_forward_only_blended_weights.py`` which exercises a closely
related (single-period) variant.
"""
from __future__ import annotations

import polars as pl

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData


def _build_toy_rp_two_periods(*, method: str) -> FlexData:
    """Synthesise the two-period RP-blended-weights LP fixture.

    Two FlexTool periods (``y2025``, ``y2030``) — each with its own
    base period, its own representative block, and a distinct inflow
    profile.  The ``method`` argument selects which
    ``storage_bind_*_blended_weights`` partition gets the singleton
    node ``bat``; ``nodeState_rp`` is the union of all three blended-
    weights variants (Phase E extension) so the shared RP machinery in
    ``model.py`` fires identically across the three variants.  Only
    ``rp_inter_period_cyclic`` further filters back to the
    within_solve ∪ within_period subset and gates its (b_first, b_last)
    pairing on the optional ``d`` column carried on
    ``rp_base_first`` / ``rp_base_last``.
    """
    if method not in (
            "bind_within_solve_blended_weights",
            "bind_within_period_blended_weights",
            "bind_forward_only_blended_weights"):
        raise ValueError(f"unsupported method: {method!r}")

    # Two periods, each with two timesteps.  Step labels are unique
    # across periods (t25_01/t25_02 vs t30_01/t30_02) so the join keys
    # on ``rp_block_first`` / ``p_rp_last_step`` (keyed by ``r``) point
    # to exactly one period's rep block — the cross-period collision
    # mode in the rhs sum is avoided by construction.
    dt = pl.DataFrame({
        "d": ["y2025", "y2025", "y2030", "y2030"],
        "t": ["t25_01", "t25_02", "t30_01", "t30_02"],
    })
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rpcw = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",),
        pl.DataFrame({"d": ["y2025", "y2030"], "value": [1.0, 1.0]}))
    p_psh = Param(("d",),
        pl.DataFrame({"d": ["y2025", "y2030"], "value": [1.0, 1.0]}))

    nb = pl.DataFrame({"n": ["bat"]})
    nb_dt = nb.join(dt, how="cross")
    # +5/-3 in y2025 ⇒ net +2 surplus.  -2/+1 in y2030 ⇒ net -1 deficit.
    p_inflow = Param(("n", "d", "t"), pl.DataFrame({
        "n": ["bat"] * 4,
        "d": ["y2025", "y2025", "y2030", "y2030"],
        "t": ["t25_01", "t25_02", "t30_01", "t30_02"],
        "value": [5.0, -3.0, -2.0, 1.0],
    }))
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
        pl.DataFrame({"n": ["bat", "bat"],
                      "d": ["y2025", "y2030"],
                      "value": [100.0, 100.0]}))
    p_state_sd = Param(("n",),
        pl.DataFrame({"n": ["bat"], "value": [0.0]}))
    p_state_exi_cap = Param(("n", "d"),
        pl.DataFrame({"n": ["bat", "bat"],
                      "d": ["y2025", "y2030"],
                      "value": [100.0, 100.0]}))

    # Cyclic-within-timeset over the 2 steps in each period.
    # Period y2025's wrap: t25_01 ← t25_02 and back; same for y2030.
    dtttdt = pl.DataFrame({
        "d": ["y2025", "y2025", "y2030", "y2030"],
        "t": ["t25_01", "t25_02", "t30_01", "t30_02"],
        "t_previous": ["t25_02", "t25_01", "t30_02", "t30_01"],
        "t_previous_within_timeset": ["t25_02", "t25_01",
                                       "t30_02", "t30_01"],
        "d_previous": ["y2025", "y2025", "y2030", "y2030"],
        "t_previous_within_solve": ["t25_02", "t25_01",
                                     "t30_02", "t30_01"],
    })

    # RP-blended-weights structures.  Route the singleton node to the
    # appropriate partition; nodeState_rp is the union (Phase E).
    bat_only = pl.DataFrame({"n": ["bat"]})
    storage_bind_within_solve_blended_weights = None
    storage_bind_within_period_blended_weights = None
    storage_bind_forward_only_blended_weights = None
    if method == "bind_within_solve_blended_weights":
        storage_bind_within_solve_blended_weights = bat_only
    elif method == "bind_within_period_blended_weights":
        storage_bind_within_period_blended_weights = bat_only
    else:  # bind_forward_only_blended_weights
        storage_bind_forward_only_blended_weights = bat_only
    nodeState_rp = bat_only

    # rp_base_period_set: two bases, one per period.
    rp_base_period_set = pl.DataFrame({"b": ["b25", "b30"]})
    # rp_block_first: each period's representative block starts at t*_01.
    rp_block_first = pl.DataFrame({
        "d": ["y2025", "y2030"],
        "t": ["t25_01", "t30_01"],
    })
    # p_rp_last_step: maps the rep_start to its last step within the
    # SAME period.  Distinct rep_start labels per period mean each row
    # is unambiguous.
    p_rp_last_step = pl.DataFrame({
        "r": ["t25_01", "t30_01"],
        "last_step": ["t25_02", "t30_02"],
    })
    # rp_base__rep: each base maps to its OWN period's rep with weight
    # 1.0.  No cross-period mapping (the within_period semantics fall
    # out naturally from this label discipline).
    rp_base__rep = Param(("b", "r"), pl.DataFrame({
        "b": ["b25", "b30"],
        "r": ["t25_01", "t30_01"],
        "value": [1.0, 1.0],
    }))

    # rp_base_chain / rp_base_first / rp_base_last — TOPOLOGY DEPENDS
    # ON THE VARIANT.  Within-solve concatenates the two bases into one
    # chain (b25 → b30) with a single cycle b25 ↔ b30; within-period
    # gives each period a SINGLE-base trivial chain that closes onto
    # itself via the ``d`` column on the first/last frames.
    if method == "bind_within_period_blended_weights":
        # No predecessor edges (1 base per period).  rp_base_chain stays
        # empty so balance fires nowhere; cyclic fires once per period.
        rp_base_chain = pl.DataFrame(
            {"b": [], "b_prev": [], "d": []},
            schema={"b": pl.Utf8, "b_prev": pl.Utf8, "d": pl.Utf8},
        )
        # Each period's first base == its last base (single-base
        # subchain): cyclic closes v_state_inter[b25] ↔ v_state_inter[b25]
        # ⇒ drift_25 must vanish.
        rp_base_first = pl.DataFrame({
            "b": ["b25", "b30"],
            "d": ["y2025", "y2030"],
        })
        rp_base_last = pl.DataFrame({
            "b": ["b25", "b30"],
            "d": ["y2025", "y2030"],
        })
    else:
        # within_solve / forward_only: one chain b25 → b30.  Singleton
        # first / last (the ``d`` column is absent so the cyclic-emit
        # block falls back to the legacy cross-join — single pair).
        rp_base_chain = pl.DataFrame({"b": ["b30"], "b_prev": ["b25"]})
        rp_base_first = pl.DataFrame({"b": ["b25"]})
        rp_base_last = pl.DataFrame({"b": ["b30"]})

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
        storage_bind_within_solve_blended_weights=storage_bind_within_solve_blended_weights,
        storage_bind_within_period_blended_weights=storage_bind_within_period_blended_weights,
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
    d = _build_toy_rp_two_periods(method=method)
    pb = Problem()
    build_flextool(pb, d)
    sol = pb.solve(options={"random_seed": 42, "parallel": "off"})
    assert sol.optimal, (
        f"LP did not solve to optimum for method={method!r}; "
        f"status={sol.status!r}"
    )
    return sol.obj


def test_three_variants_distinguished_by_objective() -> None:
    """The three RP-blended-weights variants land on three distinct
    hand-derived optima on the two-period fixture.

    See module docstring for the closed-form derivation.  Strict
    equality assertions to 1e-9 because the fixture is constructed so
    every variable except the slack is uniquely determined and the
    optimum is integer-valued.
    """
    obj_fwd = _solve("bind_forward_only_blended_weights")
    obj_within_solve = _solve("bind_within_solve_blended_weights")
    obj_within_period = _solve("bind_within_period_blended_weights")

    assert abs(obj_fwd - 0.0) < 1e-9, (
        f"forward_only obj = {obj_fwd!r}, expected 0.0 — no cyclic "
        f"closure means each base's state-change can absorb its own "
        f"period's drift for free.  If obj > 0 the cyclic constraint "
        f"is still firing for forward_only (Phase D/E bind_set_rp "
        f"gating bug)."
    )
    assert abs(obj_within_solve - 1.0) < 1e-9, (
        f"within_solve obj = {obj_within_solve!r}, expected 1.0 — "
        f"single across-solve cycle forces "
        f"Δ_y2025 + Δ_y2030 = 0; the +2 surplus must net the -1 "
        f"deficit with 1 unit of slack on either side at penalty 1.0."
    )
    assert abs(obj_within_period - 3.0) < 1e-9, (
        f"within_period obj = {obj_within_period!r}, expected 3.0 — "
        f"each period closes its own cycle independently: "
        f"period y2025 must spill 2 (cost 2.0), period y2030 must "
        f"spill 1 (cost 1.0), total 3.0.  If obj == 1.0 the "
        f"per-period (b_first, b_last) pairing in model.py's cyclic-"
        f"emit block is collapsing to the within_solve cross-join "
        f"(Phase E rp_base_first/last d-column join bug); if obj == "
        f"0.0 the cyclic constraint is not firing at all for the "
        f"within_period variant."
    )

    assert obj_fwd < obj_within_solve < obj_within_period, (
        f"expected forward_only < within_solve < within_period "
        f"(progressively tighter closure adds progressively more "
        f"slack cost); got {obj_fwd!r} / {obj_within_solve!r} / "
        f"{obj_within_period!r}."
    )


def test_within_period_cyclic_fires_independently_per_period() -> None:
    """Direct verification that the per-period cyclic constraint is
    forcing each period's drift to zero independently.

    In the within_period variant, period y2025 has a +2 inflow surplus
    that MUST be spilled (cost 2.0) and period y2030 has a -1 deficit
    that MUST be spilled (cost 1.0).  If either period's cyclic
    closure failed to fire, the optimum would drop below 3.0.  Also
    asserts the optimum is strictly greater than the within_solve
    optimum (which is 1.0) — the gap is exactly the slack the within-
    solve cycle saves by netting the two periods against each other.
    """
    obj_within_period = _solve("bind_within_period_blended_weights")
    obj_within_solve = _solve("bind_within_solve_blended_weights")

    assert abs(obj_within_period - 3.0) < 1e-9, (
        f"within_period obj = {obj_within_period!r}, expected 3.0 — "
        f"per-period cyclic closure forces each period's drift to "
        f"vanish."
    )
    assert obj_within_period - obj_within_solve == 2.0 or (
        abs((obj_within_period - obj_within_solve) - 2.0) < 1e-9
    ), (
        f"within_period - within_solve = "
        f"{obj_within_period - obj_within_solve!r}, expected 2.0 — "
        f"the gap is the slack the across-solve cycle saves by "
        f"netting +2 against -1 (within_solve = 1.0 vs. within_period "
        f"= 3.0)."
    )
