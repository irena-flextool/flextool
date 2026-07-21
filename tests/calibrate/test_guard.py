"""Tests for the adequacy-calibrator over-build guard (C1c).

The guard rule (spec §8.2) is the point — exercise it HARD as a pure
function.  Per node ``n``, comparing iteration ``k`` against ``k-1`` and the
iteration-0 baseline, freeze (and flag) ``n`` only when ALL THREE hold:

1. ``η(n) = ΔSlack / (ΔCurtail + ε) < tightness``   (spill buys ~no adequacy);
2. ``ΔSlack < 0.05 · S_{k-1}``                       (the gap stopped moving);
3. ``C_k > C_0``                                     (margin-INDUCED spill).

The three-condition AND is load-bearing: the NO-FREEZE tests below each break
the freeze by relaxing a single condition — two of them isolate that
condition cleanly (the other two still tripped), and the η-healthy case
naturally also un-stalls (a node removing lots of slack IS progressing), so
that test pins the η gate while noting cond 2 relaxes with it.  Plus:
boundary strictness, persistence across rounds, the first-correction "nothing
to diff" case, and the ε zero-division guard.
"""

from __future__ import annotations

from flextool.calibrate._guard import guard_freeze

# A tightness the FREEZE case clears with room to spare; the default CLI knob
# is 0.05 (lenient), and these synthetic η values sit either side of it.
TIGHTNESS = 0.05


# ---------------------------------------------------------------------------
# FREEZE: all three conditions tripped → node flagged and dropped.
# ---------------------------------------------------------------------------

def test_freeze_when_all_three_conditions_hold():
    # S: 1000 → 990 (ΔSlack = 10, only 1% of S_prev=1000 → stalled).
    # C: 200 → 700 (ΔCurtail = 500 new spill; C_k=700 > C_0=50 → induced).
    # η = 10 / (500 + 1) ≈ 0.02 < 0.05.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 990.0},
        curtailment={"n": 700.0},
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},
        baseline_curtailment={"n": 50.0},
        tightness=TIGHTNESS,
    )
    assert flagged == {"n"}
    assert kept == {}  # frozen node removed from the increments


def test_freeze_only_the_capped_node_others_kept():
    # "cap" trips all three; "ok" is still progressing (huge ΔSlack) → kept.
    kept, flagged = guard_freeze(
        {"cap": 0.7, "ok": 0.3},
        residual={"cap": 990.0, "ok": 100.0},
        curtailment={"cap": 700.0, "ok": 700.0},
        prev_residual={"cap": 1000.0, "ok": 1000.0},
        prev_curtailment={"cap": 200.0, "ok": 200.0},
        baseline_curtailment={"cap": 50.0, "ok": 50.0},
        tightness=TIGHTNESS,
    )
    assert flagged == {"cap"}
    assert kept == {"ok": 0.3}


# ---------------------------------------------------------------------------
# NO-FREEZE: exactly one condition turned off (the AND is load-bearing).
# ---------------------------------------------------------------------------

def test_no_freeze_when_still_progressing():
    # Condition 2 OFF: ΔSlack = 500 (50% of S_prev, well above 5%) — adequacy
    # is still improving fast even though spill is high (η small).  This is
    # economic over-build while the gap keeps closing → must NOT flag.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 500.0},          # 1000 → 500, ΔSlack = 500
        curtailment={"n": 20000.0},     # huge spill → η tiny (cond 1 tripped)
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},
        baseline_curtailment={"n": 50.0},
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


def test_no_freeze_when_curtailment_at_or_below_baseline():
    # Condition 3 is the SOLE blocker: conditions 1 & 2 both HOLD, only
    # ``C_k > C_0`` fails.  ΔSlack = 10 (< 5% of 1000 → cond 2 holds);
    # ΔCurtail = 700-200 = 500 → η = 10/501 ≈ 0.02 < 0.05 (cond 1 holds);
    # but C_k = 700 <= C_0 = 800 → no margin-induced spill above baseline →
    # inherent/economic curtailment → must NOT flag.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 990.0},
        curtailment={"n": 700.0},       # C_k = 700
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},
        baseline_curtailment={"n": 800.0},  # C_0 = 800 → C_k <= C_0
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


def test_no_freeze_when_curtailment_strictly_below_baseline():
    # Condition 3 OFF, stronger: C_k = 100 < C_0 = 300.  Curtailment has
    # actually FALLEN below baseline → definitely not margin-induced.  Even
    # with η low and the gap stalled, must not flag.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 995.0},          # ΔSlack = 5 (< 5% → cond 2 holds)
        curtailment={"n": 100.0},
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 90.0},   # ΔCurtail = 10 → η = 5/11 ≈ 0.45
        baseline_curtailment={"n": 300.0},
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


def test_no_freeze_when_efficiency_healthy():
    # Condition 1 OFF: lots of slack removed per unit of new spill → η high.
    # S: 1000 → 900 (ΔSlack = 100), C: 200 → 210 (ΔCurtail = 10) →
    # η = 100 / 11 ≈ 9.1 ≫ 0.05.  Margin is still buying adequacy → keep.
    # (Condition 3 holds: C_k=210 > C_0=50.  Condition 2 would fail too, but
    #  condition 1 alone is enough to keep it — this pins the η gate.)
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 900.0},
        curtailment={"n": 210.0},
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},
        baseline_curtailment={"n": 50.0},
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


# ---------------------------------------------------------------------------
# PRECONDITION: a node not shedding last round (never bumped) is never frozen.
# ---------------------------------------------------------------------------

def test_no_freeze_newly_shedding_node_never_bumped_last_round():
    # The exact review scenario: s_prev = 0 (node did NOT shed last round, so
    # the sizer proposed no bump for it → it was never bumped).  This round,
    # margin added to OTHER nodes shifted dispatch onto it and it STARTS
    # shedding: s_k = 500, with margin-induced spill c_k = 300 > c_0 = 0.
    # Without the precondition all three legacy conditions trip (η<0 →
    # cond_efficiency; ΔSlack = -500 < 0.05·0 = 0 → cond_stalled; c_k>c_0 →
    # cond_induced_spill), so the node would be frozen and NEVER bumped — the
    # opposite of correct.  The precondition (s_prev > shed_tol) keeps it, so
    # this genuinely-short node gets its FIRST bump.
    kept, flagged = guard_freeze(
        {"n": 0.5},
        residual={"n": 500.0},              # s_k = 500 (newly shedding)
        curtailment={"n": 300.0},           # c_k = 300 > c_0 = 0
        prev_residual={"n": 0.0},           # s_prev = 0 → not shedding last round
        prev_curtailment={"n": 0.0},
        baseline_curtailment={"n": 0.0},
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.5}


def test_freeze_node_at_shed_tol_boundary_kept():
    # PRECONDITION is strict (s_prev > shed_tol): a node exactly AT the
    # tolerance was not shedding last round → kept even though the three
    # conditions would otherwise trip.
    kept, flagged = guard_freeze(
        {"n": 0.5},
        residual={"n": 500.0},
        curtailment={"n": 300.0},
        prev_residual={"n": 1e-6},          # == shed_tol → not "shedding"
        prev_curtailment={"n": 0.0},
        baseline_curtailment={"n": 0.0},
        tightness=TIGHTNESS,
        shed_tol=1e-6,
    )
    assert flagged == set()
    assert kept == {"n": 0.5}


# ---------------------------------------------------------------------------
# ε zero-division guard.
# ---------------------------------------------------------------------------

def test_eps_guards_zero_new_curtailment_no_division_error():
    # ΔCurtail = 0 (curtailment unchanged) → η = ΔSlack / (0 + ε), finite.
    # With ε=1 and ΔSlack=10, η = 10 > tightness → not flagged (and no crash).
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 990.0},
        curtailment={"n": 200.0},
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},   # ΔCurtail = 0
        baseline_curtailment={"n": 50.0},
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


def test_node_with_zero_new_curtailment_never_flagged_even_when_stalled():
    # ΔCurtail = 0 AND the gap stalled (ΔSlack tiny) — but η = ΔSlack/ε.  With
    # a tiny ΔSlack = 0.001, η = 0.001 < tightness (cond 1 holds) and cond 2
    # holds; condition 3 (C_k > C_0) is the guard: here C_k == C_0 so NOT
    # flagged.  A node that adds no spill above baseline is never resource-
    # capped by margin, regardless of ε.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 999.999},
        curtailment={"n": 50.0},
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 50.0},
        baseline_curtailment={"n": 50.0},  # C_k == C_0
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


# ---------------------------------------------------------------------------
# Boundary: conditions use strict inequalities exactly as spec'd.
# ---------------------------------------------------------------------------

def test_stalled_boundary_is_strict_less_than():
    # ΔSlack exactly == 5% · S_prev (50 of 1000).  Condition 2 is
    # ``ΔSlack < 0.05·S_prev`` (STRICT) → 50 < 50 is False → NOT flagged.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 950.0},          # ΔSlack = 50 == 5% of 1000
        curtailment={"n": 700.0},
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},
        baseline_curtailment={"n": 50.0},
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


def test_eta_boundary_is_strict_less_than():
    # η exactly == tightness → condition 1 is ``η < tightness`` (STRICT) →
    # not flagged.  Construct η = 0.05 = ΔSlack/(ΔCurtail+ε):
    # ΔSlack = 10, ΔCurtail + ε = 200 → ΔCurtail = 199.
    kept, flagged = guard_freeze(
        {"n": 0.7},
        residual={"n": 990.0},          # ΔSlack = 10 (< 5% → cond 2 holds)
        curtailment={"n": 399.0},       # ΔCurtail = 199 → η = 10/200 = 0.05
        prev_residual={"n": 1000.0},
        prev_curtailment={"n": 200.0},
        baseline_curtailment={"n": 50.0},  # cond 3 holds
        tightness=TIGHTNESS,
    )
    assert flagged == set()
    assert kept == {"n": 0.7}


# ---------------------------------------------------------------------------
# Only candidate nodes are considered.
# ---------------------------------------------------------------------------

def test_only_candidate_nodes_evaluated():
    # A node that trips all three conditions but is NOT a candidate this round
    # (e.g. already non-shedding, so the sizer proposed no bump) is untouched:
    # the guard freezes only among nodes it is asked to keep.
    kept, flagged = guard_freeze(
        {"a": 0.4},  # only "a" is a candidate
        residual={"a": 100.0, "b": 990.0},
        curtailment={"a": 700.0, "b": 700.0},
        prev_residual={"a": 1000.0, "b": 1000.0},  # "a" still progressing
        prev_curtailment={"a": 200.0, "b": 200.0},
        baseline_curtailment={"a": 50.0, "b": 50.0},
        tightness=TIGHTNESS,
    )
    # "b" would trip all three, but it is not a candidate → never flagged.
    assert flagged == set()
    assert kept == {"a": 0.4}


# ---------------------------------------------------------------------------
# Persistence + first-correction: exercised through compute_step / the loop
# state, since the pure guard is stateless by design.
# ---------------------------------------------------------------------------

def _cfg(tightness: float):
    from pathlib import Path

    from flextool.calibrate._loop import CalibConfig

    return CalibConfig(
        iterations=5,
        slack_threshold_mwh=1.0,
        damping_first=1.0,
        damping_remaining=1.0,
        over_build_tightness=tightness,
        warm_start_cache_dir=Path("/nonexistent"),
        work_dir=Path("/nonexistent"),
        out_root=Path("/nonexistent"),
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
        {"n": 1000.0}, {"n": 700.0}, {}, None, _cfg(TIGHTNESS),
        W=W, flagged=set(), baseline_curtailment={"n": 50.0},
    )
    assert newly == set()
    assert increments == {"n": 1000.0 / W}  # λ=1 → increment = residual/W


def test_second_correction_can_flag():
    # With a prior record, the same stalled/high-spill signals now DO flag.
    from flextool.calibrate._loop import compute_step

    W = 8760.0
    prev = _record(1, {"n": 1000.0}, {"n": 200.0})
    increments, newly = compute_step(
        {"n": 990.0}, {"n": 700.0}, {}, prev, _cfg(TIGHTNESS),
        W=W, flagged=set(), baseline_curtailment={"n": 50.0},
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
        {"n": 300.0}, {}, prev, _cfg(TIGHTNESS),
        W=W, flagged={"n"}, baseline_curtailment={"n": 50.0},
    )
    assert increments == {}          # …but it is frozen, so no bump
    assert newly == set()


def test_guard_no_op_below_tightness_floor():
    # With tightness = 0.0 (as the convergence smoke uses), condition 1 is
    # ``η < 0.0``.  A node whose slack is IMPROVING has ΔSlack > 0 → η ≥ 0 →
    # never < 0 → the guard is a strict no-op.  This mirrors the loop's
    # monotone-progress fixture: nothing resource-capped, nothing flagged.
    from flextool.calibrate._loop import compute_step

    W = 8760.0
    prev = _record(1, {"n": 1000.0}, {"n": 200.0})
    increments, newly = compute_step(
        {"n": 800.0}, {"n": 700.0}, {}, prev, _cfg(0.0),
        W=W, flagged=set(), baseline_curtailment={"n": 50.0},
    )
    assert newly == set()
    assert increments == {"n": 800.0 / W}
