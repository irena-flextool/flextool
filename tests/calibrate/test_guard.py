"""Tests for the adequacy-calibrator over-build guard (C1c).

The guard rule is the point — exercise it HARD as a pure function.  Per node
``n``, comparing the just-solved iteration ``k`` against the prior iteration
``k-1``, freeze (and flag) ``n`` only when BOTH hold:

1. PRECONDITION ``S_{k-1} > shed_tol``  — the node was shedding last round, so
   it WAS bumped (the freeze question is only meaningful for a bumped node);
2. STALL ``ΔSlack = S_{k-1} − S_k < stall_fraction · S_{k-1}``  — the residual
   barely moved (or grew) in response to that bump, i.e. margin buys it no
   adequacy → it is resource-capped.

Curtailment plays NO part in the freeze — a demand node never curtails, so the
OLD ``C_k > C_0`` gate could never flag the calibrator's real targets (and kept
concentrating demand until the solve went infeasible).  The freeze keys off the
residual RESPONSE instead.  These tests pin: the two-condition AND, the
precondition (a never-bumped newly-shedding node is kept), a flat/worse
residual freezing WITHOUT any curtailment signal, boundary strictness, the
``stall_fraction`` knob, persistence, and the first-correction "nothing to
diff" case.
"""

from __future__ import annotations

from flextool.calibrate._guard import guard_freeze

# The default stall fraction (CLI --stall-fraction default 0.05).
STALL = 0.05


# ---------------------------------------------------------------------------
# FREEZE: was shedding last round AND residual stalled → flagged and dropped.
# ---------------------------------------------------------------------------

def test_freeze_when_shedding_and_stalled_without_curtailment():
    # S: 1000 → 990 (ΔSlack = 10, only 1% of S_prev=1000 → stalled < 5%).
    # No curtailment anywhere: the freeze does NOT depend on it (this is the
    # exact demand-node case the old curtailment gate could never flag).
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 990.0},
        prev_residual={"n": 1000.0},
    )
    assert flagged == {"n"}
    assert kept == {}  # frozen node removed from the increments


def test_freeze_when_residual_flat_despite_bump():
    # The purest "margin buys no adequacy" signature: residual completely flat
    # (ΔSlack = 0) despite the node having been bumped last round → frozen.
    kept, flagged = guard_freeze(
        {"n": 0.5},
        residual={"n": 500.0},
        prev_residual={"n": 500.0},
    )
    assert flagged == {"n"}
    assert kept == {}


def test_freeze_when_residual_grew():
    # Residual WORSENED (ΔSlack < 0) despite the bump — still resource-capped
    # (adding demand there only shifts more shed onto it) → frozen.  The
    # precondition holds (it was shedding last round), so this is not the
    # newly-shedding case below.
    kept, flagged = guard_freeze(
        {"n": 0.5},
        residual={"n": 520.0},
        prev_residual={"n": 500.0},
    )
    assert flagged == {"n"}
    assert kept == {}


def test_freeze_only_the_capped_node_others_kept():
    # "cap" stalled (1% drop); "ok" is still progressing (50% drop) → kept.
    kept, flagged = guard_freeze(
        {"cap": 0.7, "ok": 0.3},
        residual={"cap": 990.0, "ok": 500.0},
        prev_residual={"cap": 1000.0, "ok": 1000.0},
    )
    assert flagged == {"cap"}
    assert kept == {"ok": 0.3}


# ---------------------------------------------------------------------------
# NO-FREEZE: the node is still responding to its bump.
# ---------------------------------------------------------------------------

def test_no_freeze_when_still_progressing():
    # ΔSlack = 500 (50% of S_prev, well above 5%) — adequacy still improving
    # fast → must NOT flag, regardless of any (irrelevant) curtailment.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 500.0},          # 1000 → 500, ΔSlack = 500
        prev_residual={"n": 1000.0},
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


# ---------------------------------------------------------------------------
# PRECONDITION: a node not shedding last round (never bumped) is never frozen.
# ---------------------------------------------------------------------------

def test_no_freeze_newly_shedding_node_never_bumped_last_round():
    # s_prev = 0 (node did NOT shed last round → the sizer proposed no bump →
    # it was never bumped).  This round margin added to OTHER nodes shifted
    # dispatch onto it and it STARTS shedding: s_k = 500.  Without the
    # precondition ΔSlack = -500 < 0.05·0 = 0 would trip the stall condition
    # and freeze it forever — the opposite of correct.  The precondition
    # (s_prev > shed_tol) keeps it, so this genuinely-short node gets its FIRST
    # bump.
    kept, flagged = guard_freeze(
        {"n": 0.5},
        residual={"n": 500.0},          # s_k = 500 (newly shedding)
        prev_residual={"n": 0.0},       # s_prev = 0 → not shedding last round
    )
    assert flagged == set()
    assert kept == {"n": 0.5}


def test_freeze_node_at_shed_tol_boundary_kept():
    # PRECONDITION is strict (s_prev > shed_tol): a node exactly AT the
    # tolerance was not shedding last round → kept even though ΔSlack < 0 would
    # otherwise trip the stall condition.
    kept, flagged = guard_freeze(
        {"n": 0.5},
        residual={"n": 500.0},
        prev_residual={"n": 1e-6},      # == shed_tol → not "shedding"
        shed_tol=1e-6,
    )
    assert flagged == set()
    assert kept == {"n": 0.5}


# ---------------------------------------------------------------------------
# Boundary: the stall condition is a STRICT inequality.
# ---------------------------------------------------------------------------

def test_stalled_boundary_is_strict_less_than():
    # ΔSlack exactly == stall_fraction · S_prev (50 of 1000).  The stall
    # condition is ``ΔSlack < 0.05·S_prev`` (STRICT) → 50 < 50 is False → kept.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 950.0},          # ΔSlack = 50 == 5% of 1000
        prev_residual={"n": 1000.0},
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


# ---------------------------------------------------------------------------
# The stall_fraction knob: HIGHER freezes a stalled node SOONER.
# ---------------------------------------------------------------------------

def test_stall_fraction_knob_controls_sensitivity():
    # ΔSlack = 100 (10% of 1000).  At the default 0.05 the node is still
    # progressing (10% > 5%) → kept; at a stricter 0.2 the same 10% drop counts
    # as stalled (10% < 20%) → frozen.
    args = dict(
        candidate={"n": 0.7},
        residual={"n": 900.0},
        prev_residual={"n": 1000.0},
    )
    kept_lenient, flagged_lenient = guard_freeze(**args, stall_fraction=0.05)
    assert flagged_lenient == set()
    assert kept_lenient == {"n": 0.7}

    kept_strict, flagged_strict = guard_freeze(**args, stall_fraction=0.2)
    assert flagged_strict == {"n"}
    assert kept_strict == {}


# ---------------------------------------------------------------------------
# Only candidate nodes are considered.
# ---------------------------------------------------------------------------

def test_only_candidate_nodes_evaluated():
    # "b" would trip both conditions but is NOT a candidate this round → the
    # guard freezes only among nodes it is asked to keep.
    kept, flagged = guard_freeze(
        {"a": 0.4},  # only "a" is a candidate
        residual={"a": 500.0, "b": 990.0},
        prev_residual={"a": 1000.0, "b": 1000.0},  # "a" still progressing
    )
    assert flagged == set()
    assert kept == {"a": 0.4}


# ---------------------------------------------------------------------------
# Persistence + first-correction: exercised through compute_step / the loop
# state, since the pure guard is stateless by design.
# ---------------------------------------------------------------------------

def _cfg(stall_fraction: float):
    from pathlib import Path

    from flextool.calibrate._loop import CalibConfig

    return CalibConfig(
        iterations=5,
        slack_threshold_mwh=1.0,
        damping_first=1.0,
        damping_remaining=1.0,
        over_build_tightness=0.05,  # retained field, not consulted by the guard
        warm_start_cache_dir=Path("/nonexistent"),
        work_dir=Path("/nonexistent"),
        out_root=Path("/nonexistent"),
        stall_fraction=stall_fraction,
    )


def _record(iteration, residual, curtailment):
    from flextool.calibrate._loop import IterRecord

    return IterRecord(
        iteration=iteration,
        adders={},
        residual=residual,
        curtailment=curtailment,
        penalty_total=0.0,
        penalty_by_node={},
    )


def test_first_correction_flags_nobody():
    # prev_record is None → the guard cannot diff → every shedding node is
    # bumped and nothing is flagged, no matter the signals.
    from flextool.calibrate._loop import compute_step

    W = 8760.0
    increments, newly = compute_step(
        {"n": 1000.0}, {}, None, _cfg(STALL),
        W=W, flagged=set(),
    )
    assert newly == set()
    assert increments == {"n": 1000.0 / W}  # λ=1, overshoot=1 → residual/W


def test_second_correction_can_flag():
    # With a prior record, a stalled residual now DOES flag.
    from flextool.calibrate._loop import compute_step

    W = 8760.0
    prev = _record(1, {"n": 1000.0}, {"n": 200.0})
    increments, newly = compute_step(
        {"n": 990.0}, {}, prev, _cfg(STALL),
        W=W, flagged=set(),
    )
    assert newly == {"n"}
    assert increments == {}  # frozen → no bump


def test_persistently_flagged_node_never_bumped_again():
    # A node already in the persistent ``flagged`` set is dropped BEFORE the
    # guard runs — even if its current signals would look "still progressing",
    # it gets no further increment.  This is the loop's persistence contract.
    from flextool.calibrate._loop import compute_step

    W = 8760.0
    prev = _record(2, {"n": 900.0}, {"n": 300.0})
    increments, newly = compute_step(
        {"n": 500.0},                 # big ΔSlack → would look healthy…
        {}, prev, _cfg(STALL),
        W=W, flagged={"n"},
    )
    assert increments == {}          # …but it is frozen, so no bump
    assert newly == set()


def test_guard_no_op_at_zero_stall_fraction():
    # stall_fraction = 0.0 → the stall condition is ``ΔSlack < 0.0`` — a node
    # whose slack is IMPROVING (ΔSlack > 0) can never trip it, so the guard is
    # a strict no-op (the role the old tightness=0.0 played).
    from flextool.calibrate._loop import compute_step

    W = 8760.0
    prev = _record(1, {"n": 1000.0}, {"n": 200.0})
    increments, newly = compute_step(
        {"n": 800.0}, {}, prev, _cfg(0.0),
        W=W, flagged=set(),
    )
    assert newly == set()
    assert increments == {"n": 800.0 / W}
