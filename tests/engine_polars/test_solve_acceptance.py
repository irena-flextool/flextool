"""Unit tests for the solve accept/reject policy.

Exercises :func:`flextool.engine_polars._solve_acceptance.classify_acceptance`
as a pure function over synthesised solver diagnostics — no DB, no live
solve.  The central regression case is the observed crossover-off roll:
HiGHS status ``kUnknown``, primal feasible, ``primal_dual_objective_error
== 0.00477`` — which must be ACCEPTED as near-optimal, with no misleading
scaling hint.
"""

from __future__ import annotations

import math

import pytest

from polar_high import SolveDiagnostics
from polar_high.autoscale import RangeReport

from flextool.engine_polars._solve_acceptance import classify_acceptance


# --- stubs ------------------------------------------------------------------


class _StubSolution:
    """Minimal stand-in for :class:`polar_high.Solution`.

    Exposes only what ``classify_acceptance`` touches: ``optimal``,
    ``primal_feasibility_tolerance`` and ``solve_diagnostics()``.
    """

    def __init__(
        self,
        *,
        optimal: bool,
        diagnostics: SolveDiagnostics | None,
        primal_feasibility_tolerance: float = 1e-7,
    ) -> None:
        self.optimal = optimal
        self._diag = diagnostics
        self.primal_feasibility_tolerance = primal_feasibility_tolerance

    def solve_diagnostics(self) -> SolveDiagnostics | None:
        return self._diag


def _diag(
    *,
    status_name: str,
    primal_feasible: bool = True,
    dual_feasible: bool = False,
    num_primal_infeasibilities: int = 0,
    max_primal_infeasibility: float = 7.16e-8,
    max_relative_primal_infeasibility: float = 1.89e-11,
    primal_dual_objective_error: float = 0.00477,
) -> SolveDiagnostics:
    """Build a SolveDiagnostics; defaults reproduce the observed roll."""
    return SolveDiagnostics(
        model_status=None,
        model_status_name=status_name,
        primal_feasible=primal_feasible,
        dual_feasible=dual_feasible,
        num_primal_infeasibilities=num_primal_infeasibilities,
        num_dual_infeasibilities=0,
        max_primal_infeasibility=max_primal_infeasibility,
        max_relative_primal_infeasibility=max_relative_primal_infeasibility,
        max_dual_infeasibility=1e-7,
        primal_dual_objective_error=primal_dual_objective_error,
        objective_value=3196.89,
    )


def _range_report(*, trigger: bool) -> RangeReport:
    return RangeReport(
        matrix=(1e-3, 8e4),
        cost=(6e-3, 2e2),
        bound=(math.nan, math.nan),
        rhs=(1e-4, 5e3),
        cross_group_max_ratio=8e7,
        trigger=trigger,
    )


# --- accept paths -----------------------------------------------------------


def test_optimal_accepted_outright() -> None:
    acc = classify_acceptance(
        _StubSolution(optimal=True, diagnostics=None),
        ranges_post=None,
        solve_name="s",
    )
    assert acc.accepted
    assert not acc.near_optimal
    assert acc.message == ""
    assert acc.scaling_hint is None


def test_kunknown_near_optimal_accepted() -> None:
    """The exact observed failure: kUnknown, primal feasible, pd=0.00477."""
    acc = classify_acceptance(
        _StubSolution(optimal=False, diagnostics=_diag(status_name="kUnknown")),
        ranges_post=_range_report(trigger=False),
        solve_name="st_roll",
    )
    assert acc.accepted
    assert acc.near_optimal
    assert "st_roll" in acc.message
    assert "0.477%" in acc.message
    assert "Using the primal solution" in acc.message
    # A well-conditioned accepted solve must never carry a scaling hint.
    assert acc.scaling_hint is None


def test_kunknown_near_optimal_notes_dual_feasibility() -> None:
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(status_name="kUnknown", dual_feasible=True),
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert acc.accepted and acc.near_optimal
    assert "dual solution is also feasible" in acc.message


# --- reject paths -----------------------------------------------------------


def test_kunknown_large_gap_rejected() -> None:
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(
                status_name="kUnknown", primal_dual_objective_error=0.03
            ),
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "objective error" in acc.message
    assert "3.000%" in acc.message


def test_kunknown_primal_infeasible_rejected() -> None:
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(
                status_name="kUnknown",
                primal_feasible=False,
                num_primal_infeasibilities=5,
                max_relative_primal_infeasibility=1e-3,
            ),
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "NOT" in acc.message and "feasible" in acc.message
    assert "5 infeasibilities" in acc.message


def test_kunknown_abs_infeasibility_over_margin_rejected() -> None:
    """Primal passes the relative check but blows the absolute backstop
    (max_primal_infeasibility > tol * margin)."""
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(
                status_name="kUnknown",
                max_primal_infeasibility=1e-4,  # >> 1e-7 * 10
            ),
            primal_feasibility_tolerance=1e-7,
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted


@pytest.mark.parametrize(
    "status_name,needle",
    [
        ("kInfeasible", "infeasible"),
        ("kUnbounded", "unbounded"),
        ("kTimeLimit", "time limit"),
        ("kIterationLimit", "iteration limit"),
        ("kSolveError", "solve stage"),
        ("kPresolveError", "presolve stage"),
        ("kPostsolveError", "postsolve stage"),
    ],
)
def test_failure_status_rejected_with_precise_cause(
    status_name: str, needle: str
) -> None:
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(status_name=status_name, primal_feasible=False),
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert not acc.near_optimal
    assert needle in acc.message
    # The scaling advice must NOT masquerade as the cause of a real failure.
    assert "scaling MAY be" not in acc.message


def test_unrecognised_status_says_unknown_reason() -> None:
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(status_name="kSomethingNew"),
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "could not be determined" in acc.message


def test_no_diagnostics_rejected_honestly() -> None:
    """Subprocess/commercial shim path: no queryable handle → cannot verify
    → reject with an honest 'no diagnostics available' message."""
    acc = classify_acceptance(
        _StubSolution(optimal=False, diagnostics=None),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "no solver diagnostics are available" in acc.message
    assert acc.scaling_hint is None


def test_missing_solve_diagnostics_method_degrades_not_crashes() -> None:
    """A polar-high too old to expose ``solve_diagnostics`` must degrade to the
    honest 'cannot diagnose → reject' path, never AttributeError the cascade."""

    class _OldSolution:
        optimal = False
        primal_feasibility_tolerance = 1e-7
        # deliberately no ``solve_diagnostics`` attribute

    acc = classify_acceptance(
        _OldSolution(),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "no solver diagnostics are available" in acc.message
    assert acc.scaling_hint is None


# --- scaling-hint gating ----------------------------------------------------


def test_reject_scaling_hint_only_when_post_trigger() -> None:
    # post-ranges still ill-conditioned → hint present
    acc = classify_acceptance(
        _StubSolution(
            optimal=False, diagnostics=_diag(status_name="kInfeasible")
        ),
        ranges_post=_range_report(trigger=True),
        solve_name="s",
    )
    assert acc.scaling_hint is not None
    assert "scaling MAY be" in acc.scaling_hint

    # post-ranges well-conditioned → no hint
    acc2 = classify_acceptance(
        _StubSolution(
            optimal=False, diagnostics=_diag(status_name="kInfeasible")
        ),
        ranges_post=_range_report(trigger=False),
        solve_name="s",
    )
    assert acc2.scaling_hint is None

    # no post-ranges computed → no hint
    acc3 = classify_acceptance(
        _StubSolution(
            optimal=False, diagnostics=_diag(status_name="kInfeasible")
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert acc3.scaling_hint is None


# --- env overrides ----------------------------------------------------------


def test_kill_switch_disables_near_optimal(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_ACCEPT_NEAR_OPTIMAL", "0")
    acc = classify_acceptance(
        _StubSolution(optimal=False, diagnostics=_diag(status_name="kUnknown")),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0" in acc.message


def test_pd_gap_env_override_tightens(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_ACCEPT_PD_GAP", "1e-3")
    # 0.00477 now exceeds the tightened 1e-3 gap → reject.
    acc = classify_acceptance(
        _StubSolution(optimal=False, diagnostics=_diag(status_name="kUnknown")),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert "objective error" in acc.message


def test_pd_gap_env_override_loosens(monkeypatch) -> None:
    monkeypatch.setenv("FLEXTOOL_ACCEPT_PD_GAP", "5e-2")
    acc = classify_acceptance(
        _StubSolution(
            optimal=False,
            diagnostics=_diag(
                status_name="kUnknown", primal_dual_objective_error=0.03
            ),
        ),
        ranges_post=None,
        solve_name="s",
    )
    assert acc.accepted and acc.near_optimal
