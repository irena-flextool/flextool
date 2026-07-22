"""Over-build guard for the adequacy calibrator (C1c) — a PURE decision.

The sizer (C1b) keeps bumping ``energy_margin_adder`` on every node that
still sheds unserved energy.  On a node whose shortfall is genuinely
resource-capped — no more firm capacity, imports, or storage can be built to
serve it — bumping the margin further just injects demand the solve cannot
serve, and (worse) concentrating demand on an unresponsive node drives the
solve toward infeasibility.  This module decides, each correction, which
shedding nodes to KEEP bumping and which to FREEZE and FLAG as
resource-capped so the operator surfaces them for firm capacity / imports /
storage instead of more demand margin.

The observable that IS present
------------------------------
Earlier revisions gated the freeze on margin-induced CURTAILMENT rising AT
the shedding node (``C_k > C_0``).  That is the WRONG observable for the
calibrator's real targets: demand nodes never curtail (VRE spill is upstream),
so ``C_k > C_0`` is permanently FALSE at a demand node and the guard could
NEVER flag it — while it kept concentrating demand there until the solve went
infeasible.  The signature that IS present at a resource-capped demand node is
a residual that stays FLAT despite its adder rising: the margin is buying no
adequacy.  So the freeze keys off the residual RESPONSE to the bump, not
curtailment.

The rule, per node ``n``, comparing the just-solved iteration ``k`` against
the prior iteration ``k-1``:

* ``ΔSlack(n) = S_{k-1}(n) − S_k(n)``  — unserved energy REDUCED (MWh) by the
  bump ``n`` received last round.

Freeze ``n`` (and add it to the flagged set) when the PRECONDITION and the
STALL condition both hold:

* PRECONDITION — ``S_{k-1}(n) > shed_tol``: the node was ALREADY shedding in
  the prior iteration, so it WAS bumped last round.  The freeze rule asks "did
  MY prior bump to this node fail to buy adequacy?", which is only meaningful
  for a node that was actually shedding (and therefore bumped) last round.  A
  node with ``S_{k-1}(n) ≤ shed_tol`` was NOT shedding last round (so the sizer
  proposed no bump for it) and only STARTS shedding this round — because margin
  added to OTHER nodes shifted dispatch onto it.  Such a genuinely short,
  never-yet-bumped node must get its FIRST bump, never be frozen; without this
  precondition its ``ΔSlack < 0`` (gap grew from zero) would trip the stall
  condition (``ΔSlack < stall_fraction · 0 = 0``) and freeze it forever,
  exactly backwards.  ``shed_tol`` is the SAME per-node shedding tolerance the
  sizer uses (:func:`._sizing.sized_increments`), threaded in so the two agree
  on "was this node shedding?".

* STALL — ``ΔSlack(n) < stall_fraction · S_{k-1}(n)``: the residual barely
  moved (or grew) in response to last round's bump, i.e. the margin bought
  essentially no adequacy.  ``stall_fraction`` is the CLI ``--stall-fraction``
  knob (default :data:`_STALLED_GAP_FRACTION` = 0.05): a HIGHER value freezes a
  stalled node SOONER (demands a larger relative residual drop to keep bumping).

The precondition plus the stall condition are required — deliberately
conservative, biased to trip one iteration LATE rather than early (every node
gets at least one bump before it can be flagged, and a little idle capacity
beats leaving real unserved energy on a node margin CAN still help).  A
legitimately resource-capped node still gets frozen: it sheds every round (so
``S_{k-1} > shed_tol`` once it has been bumped) and its residual stalls, so the
condition trips after it has had its bump(s), and it never runs away to
infeasibility.  The guard needs a prior iteration to diff, so it can only act
from the SECOND correction onward; the loop calls it only when a prior record
exists.  Frozen nodes stay frozen for the rest of the run — the loop persists
the flagged set and never bumps a flagged node again.

Curtailment plays NO part in the freeze decision (the loop still reads it for
reporting).  Everything here is pure arithmetic over plain dicts — no I/O — so
the rule is unit-testable in isolation, which is the point of C1c.
"""

from __future__ import annotations

# Default fraction of the prior gap below which ΔSlack counts as "stopped
# moving" (the STALL condition).  Exposed on the CLI as ``--stall-fraction``;
# this constant is the default the loop threads in.
_STALLED_GAP_FRACTION = 0.05

# Default per-node shedding tolerance (MWh) for the freeze PRECONDITION —
# mirrors the sizer's ``_sizing._SHED_TOL_MWH`` so guard and sizer agree on
# "was this node shedding last round?".  The loop threads the sizer's actual
# constant in; this default keeps the pure function standalone-testable.
_DEFAULT_SHED_TOL_MWH = 1e-6


def guard_freeze(
    candidate: dict[str, float],
    *,
    residual: dict[str, float],
    prev_residual: dict[str, float],
    stall_fraction: float = _STALLED_GAP_FRACTION,
    shed_tol: float = _DEFAULT_SHED_TOL_MWH,
) -> tuple[dict[str, float], set[str]]:
    """Freeze resource-capped nodes out of *candidate* increments.

    Applies the freeze rule (module docstring) to each node *candidate*
    proposes to bump, using the current iteration ``k`` residual (*residual*)
    and the prior iteration ``k-1`` residual (*prev_residual*).  A node is
    frozen (and flagged) when it was shedding last round
    (``prev_residual[node] > shed_tol``) AND its residual failed to respond to
    that bump (``ΔSlack < stall_fraction · prev_residual[node]``).
    Curtailment is NOT consulted — a demand node never curtails, so keying the
    freeze on curtailment could never flag it (the bug this replaces).

    Parameters
    ----------
    candidate
        ``{node: increment}`` the sizer proposes for this correction (already
        stripped of persistently-flagged nodes by the caller).
    residual, prev_residual
        This iteration's and the prior iteration's per-node residual unserved
        energy (MWh).  A node absent from either contributes ``0.0``.
    stall_fraction
        Fraction of the prior gap below which ΔSlack counts as stalled
        (``config.stall_fraction`` / ``--stall-fraction``).  HIGHER freezes a
        stalled node sooner.
    shed_tol
        Per-node shedding tolerance (MWh).  A node is eligible for freezing
        only if it was shedding LAST round (``prev_residual[node] > shed_tol``);
        a node at/below it was not bumped last round and must get its first
        bump.  Threaded from the sizer so the two agree (default matches
        :data:`._sizing._SHED_TOL_MWH`).

    Returns
    -------
    tuple[dict[str, float], set[str]]
        ``(kept, newly_flagged)`` — *candidate* with the frozen nodes removed,
        and the set of nodes flagged this round.
    """
    kept: dict[str, float] = {}
    newly_flagged: set[str] = set()
    for node, inc in candidate.items():
        s_k = residual.get(node, 0.0)
        s_prev = prev_residual.get(node, 0.0)
        delta_slack = s_prev - s_k

        # PRECONDITION: only a node that was shedding last round was bumped
        # last round, so the freeze question is only meaningful for it.  A
        # newly-shedding node (s_prev <= shed_tol) must get its first bump.
        cond_was_shedding = s_prev > shed_tol
        # STALL: the residual barely moved despite the bump → margin buys no
        # adequacy at this (resource-capped) node.
        cond_stalled = delta_slack < stall_fraction * s_prev

        if cond_was_shedding and cond_stalled:
            newly_flagged.add(node)
        else:
            kept[node] = inc
    return kept, newly_flagged


__all__ = ["guard_freeze"]
