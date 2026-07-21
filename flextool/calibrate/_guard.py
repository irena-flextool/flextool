"""Over-build guard for the adequacy calibrator (C1c) — a PURE decision.

The sizer (C1b) keeps bumping ``energy_margin_adder`` on every node that
still sheds unserved energy.  On a node whose shortfall is genuinely
resource-capped — no more firm capacity, imports, or storage can be built to
serve it, only *spill* grows — bumping the margin further just injects demand
that the solve curtails again.  This module decides, each correction, which
shedding nodes to KEEP bumping and which to FREEZE and FLAG as
resource-capped so the operator surfaces them for firm capacity / imports /
storage instead of more demand margin.

The rule (spec §8.2), per node ``n``, comparing the just-solved iteration
``k`` against the prior iteration ``k-1`` and the iteration-0 baseline:

* ``ΔSlack(n)  = S_{k-1}(n) − S_k(n)``          unserved energy REDUCED (MWh);
* ``ΔCurtail(n) = max(0, C_k(n) − C_{k-1}(n))``  NEW spill added (MWh);
* ``η(n) = ΔSlack(n) / (ΔCurtail(n) + ε)``       marginal margin efficiency.

Freeze ``n`` (and add it to the flagged set) only when the PRECONDITION and
ALL three conditions hold:

* PRECONDITION — ``S_{k-1}(n) > shed_tol``: the node was ALREADY shedding in
  the prior iteration, so it WAS bumped last round.  The freeze rule asks
  "did MY prior bump to this node fail to buy adequacy?", which is only
  meaningful for a node that was actually shedding (and therefore bumped)
  last round.  A node with ``S_{k-1}(n) ≤ shed_tol`` was NOT shedding last
  round (so the sizer proposed no bump for it) and only STARTS shedding this
  round — because margin added to OTHER nodes shifted dispatch onto it.  Such
  a genuinely short, never-yet-bumped node must get its FIRST bump, never be
  frozen; without this precondition its ``ΔSlack < 0`` (gap grew from zero)
  trips ``η < 0`` and, since ``0.05 · S_{k-1} = 0``, ``ΔSlack < 0`` too — so
  it would be frozen forever, exactly backwards.  ``shed_tol`` is the SAME
  per-node shedding tolerance the sizer uses (:func:`._sizing.sized_increments`),
  threaded in so the two agree on "was this node shedding?".

1. ``η(n) < tightness``               — each new unit of spill is buying almost
   no adequacy (the CLI ``over_build_tightness`` knob; lenient default so
   normal economic over-build never trips);
2. ``ΔSlack(n) < 0.05 · S_{k-1}(n)``  — the remaining gap has essentially
   stopped moving;
3. ``C_k(n) > C_0(n)``                — baseline-reference: only margin-INDUCED
   spill counts.  A node whose curtailment has not risen above its
   iteration-0 baseline is exhibiting inherent/economic curtailment and must
   NOT be flagged.

The precondition plus all three conditions are required — deliberately
conservative, biased to trip one iteration LATE rather than early (better a
little idle capacity than leaving real unserved energy), and guaranteeing
every node gets at least one bump before it can be flagged.  A legitimately
resource-capped node still gets frozen: it sheds EVERY round (so ``S_{k-1} >
shed_tol`` once it has been bumped), its margin-induced spill keeps growing,
and its slack stalls — the three conditions then trip after it has had its
bump(s).  The guard needs a prior iteration to diff, so it can only act from
the SECOND correction onward; the loop calls it only when a prior record
exists.  Frozen nodes stay frozen for the rest of the run — the loop persists
the flagged set and never bumps a flagged node again.

Everything here is pure arithmetic over plain dicts — no I/O — so the rule is
unit-testable in isolation, which is the point of C1c.
"""

from __future__ import annotations

# Fraction of the prior gap below which ΔSlack counts as "stopped moving"
# (condition 2).  Fixed by spec §8.2, not a CLI knob.
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
    curtailment: dict[str, float],
    prev_residual: dict[str, float],
    prev_curtailment: dict[str, float],
    baseline_curtailment: dict[str, float],
    tightness: float,
    shed_tol: float = _DEFAULT_SHED_TOL_MWH,
    eps: float = 1.0,
) -> tuple[dict[str, float], set[str]]:
    """Freeze resource-capped nodes out of *candidate* increments.

    Applies the three-condition freeze rule (module docstring) to each node
    that *candidate* proposes to bump, using the current iteration ``k``
    signals (*residual* / *curtailment*), the prior iteration ``k-1`` signals
    (*prev_residual* / *prev_curtailment*), and the iteration-0
    *baseline_curtailment*.

    Curtailment dicts are keyed by SINK NODE (as
    :func:`flextool.calibrate._readers.read_curtailment_by_sink` returns them);
    a node absent from a curtailment dict contributes ``0.0``.

    Parameters
    ----------
    candidate
        ``{node: increment}`` the sizer proposes for this correction (already
        stripped of persistently-flagged nodes by the caller).
    tightness
        The ``η`` threshold (``config.over_build_tightness``).
    shed_tol
        Per-node shedding tolerance (MWh).  A node is eligible for freezing
        only if it was shedding LAST round (``prev_residual[node] > shed_tol``);
        a node at/below it was not bumped last round and must get its first
        bump.  Threaded from the sizer so the two agree (default matches
        :data:`._sizing._SHED_TOL_MWH`).
    eps
        Small MWh floor in the ``η`` denominator so a node with zero new spill
        never divides by zero (default ``1.0`` MWh).

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
        c_k = curtailment.get(node, 0.0)
        c_prev = prev_curtailment.get(node, 0.0)
        c_0 = baseline_curtailment.get(node, 0.0)

        delta_slack = s_prev - s_k
        delta_curtail = max(0.0, c_k - c_prev)
        eta = delta_slack / (delta_curtail + eps)

        # PRECONDITION: only a node that was shedding last round was bumped
        # last round, so the freeze question is only meaningful for it.  A
        # newly-shedding node (s_prev <= shed_tol) must get its first bump.
        cond_was_shedding = s_prev > shed_tol
        cond_efficiency = eta < tightness
        cond_stalled = delta_slack < _STALLED_GAP_FRACTION * s_prev
        cond_induced_spill = c_k > c_0

        if (
            cond_was_shedding
            and cond_efficiency
            and cond_stalled
            and cond_induced_spill
        ):
            newly_flagged.add(node)
        else:
            kept[node] = inc
    return kept, newly_flagged


__all__ = ["guard_freeze"]
