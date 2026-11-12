"""Unit tests for the solve accept/reject policy.

Exercises :func:`flextool.engine.solve._acceptance.classify_acceptance`
as a pure function over synthesised solver diagnostics — no DB, no live
solve.  The central regression case is the observed crossover-off roll:
HiGHS status ``kUnknown``, primal feasible, ``primal_dual_objective_error
== 0.00477`` — which must be ACCEPTED as near-optimal, with no misleading
scaling hint.
"""

from __future__ import annotations

import dataclasses
import math
import types

import pytest

from polar_high import SolveDiagnostics
from polar_high.autoscale import RangeReport

from flextool.engine.solve._acceptance import classify_acceptance


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


# --- relative ceiling follows the solve's own tolerance ----------------------
#
# Ceiling = max(1e-6, 10 x primal_feasibility_tolerance), unless
# FLEXTOOL_ACCEPT_PRIMAL_REL is set.  Hand derivation:
#   tol 1e-7 (HiGHS default): 10 x 1e-7 = 1e-6, max(1e-6, 1e-6) = 1e-6
#   tol 1e-3:                 10 x 1e-3 = 1e-2, max(1e-6, 1e-2) = 1e-2
#   tol 0 (no live handle):   max(1e-6, 0)                      = 1e-6


@pytest.mark.parametrize(
    ("tolerance", "ceiling"), [(1e-7, 1e-6), (1e-3, 1e-2), (1e-8, 1e-6), (0.0, 1e-6)],
)
def test_relative_ceiling_follows_the_tolerance(monkeypatch, tolerance, ceiling) -> None:
    from flextool.engine.solve._acceptance import primal_rel_ceiling

    monkeypatch.delenv("FLEXTOOL_ACCEPT_PRIMAL_REL", raising=False)
    assert primal_rel_ceiling(tolerance) == pytest.approx(ceiling, rel=1e-12)


def _cyprus_like(tolerance: float) -> _StubSolution:
    """The reported dual-simplex solve: status Unknown with 124 dual
    infeasibilities, 0 primal infeasibilities, max primal infeasibility
    9.93e-4 (relative 5.82e-5), primal-dual objective error 7.4e-13."""
    return _StubSolution(
        optimal=False,
        diagnostics=_diag(
            status_name="kUnknown",
            max_primal_infeasibility=9.93e-4,
            max_relative_primal_infeasibility=5.82e-5,
            primal_dual_objective_error=7.4e-13,
        ),
        primal_feasibility_tolerance=tolerance,
    )


def test_loose_tolerance_solve_accepted_near_optimal(monkeypatch) -> None:
    """tol 1e-3: 5.82e-5 <= ceiling 1e-2; absolute 9.93e-4 <= 10 x 1e-3;
    gap 7.4e-13 <= 1e-2 -> accepted.  5.82e-5 > 1e-6, so it passes only
    because of the tolerance-derived ceiling (the notice flag is set)."""
    monkeypatch.delenv("FLEXTOOL_ACCEPT_PRIMAL_REL", raising=False)
    acc = classify_acceptance(_cyprus_like(1e-3), ranges_post=None, solve_name="s")
    assert acc.accepted and acc.near_optimal, acc.message
    assert acc.above_fixed_ceiling


def test_default_tolerance_rejects_the_same_residual(monkeypatch) -> None:
    """tol 1e-7: ceiling 1e-6 < 5.82e-5 -> rejected as primal infeasible."""
    monkeypatch.delenv("FLEXTOOL_ACCEPT_PRIMAL_REL", raising=False)
    acc = classify_acceptance(_cyprus_like(1e-7), ranges_post=None, solve_name="s")
    assert not acc.accepted
    assert "NOT feasible" in acc.message
    assert "5.82e-05 > 1.0e-06" in acc.message


def test_accept_within_the_fixed_floor_needs_no_notice(monkeypatch) -> None:
    """The observed crossover-off roll (relative 1.89e-11 <= 1e-6) was
    accepted before too: no behaviour-change notice."""
    monkeypatch.delenv("FLEXTOOL_ACCEPT_PRIMAL_REL", raising=False)
    acc = classify_acceptance(
        _StubSolution(optimal=False, diagnostics=_diag(status_name="kUnknown")),
        ranges_post=None,
        solve_name="s",
    )
    assert acc.accepted and not acc.above_fixed_ceiling


def test_explicit_primal_rel_knob_wins_over_the_derived_ceiling(monkeypatch) -> None:
    """A set FLEXTOOL_ACCEPT_PRIMAL_REL replaces the derived ceiling, both
    ways: 1e-6 rejects the tol-1e-3 solve (derived would accept), 1e-4
    accepts 5.82e-5 at tol 1e-7 (derived 1e-6 would reject; the absolute
    backstop is moved out of the way with the margin knob)."""
    from flextool.engine.solve._acceptance import primal_rel_ceiling

    monkeypatch.setenv("FLEXTOOL_ACCEPT_PRIMAL_REL", "1e-6")
    assert primal_rel_ceiling(1e-3) == 1e-6
    acc = classify_acceptance(_cyprus_like(1e-3), ranges_post=None, solve_name="s")
    assert not acc.accepted
    assert "5.82e-05 > 1.0e-06" in acc.message

    monkeypatch.setenv("FLEXTOOL_ACCEPT_PRIMAL_REL", "1e-4")
    monkeypatch.setenv("FLEXTOOL_ACCEPT_PRIMAL_ABS_MARGIN", "1e5")
    acc = classify_acceptance(_cyprus_like(1e-7), ranges_post=None, solve_name="s")
    assert acc.accepted and acc.near_optimal
    assert not acc.above_fixed_ceiling  # the user's own limit, not the new rule


def test_malformed_primal_rel_knob_falls_back_to_the_derived_ceiling(monkeypatch) -> None:
    from flextool.engine.solve._acceptance import primal_rel_ceiling

    for raw in ("abc", "0", "-1e-3"):
        monkeypatch.setenv("FLEXTOOL_ACCEPT_PRIMAL_REL", raw)
        assert primal_rel_ceiling(1e-3) == pytest.approx(1e-2, rel=1e-12)


# --- an LP with rows but no variables (HiGHS: kModelEmpty) -------------------


def _variable_free_highs(row_lower: list[float], row_upper: list[float]):
    """A real HiGHS run of an LP with these rows and no column: HiGHS reports ``kModelEmpty`` and ignores the
    rows."""
    import highspy
    import numpy as np

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lp = highspy.HighsLp()
    lp.num_col_ = 0
    lp.num_row_ = len(row_lower)
    lp.row_lower_ = np.array(row_lower, dtype=np.float64)
    lp.row_upper_ = np.array(row_upper, dtype=np.float64)
    lp.a_matrix_.start_ = np.array([0])
    h.passModel(lp)
    h.run()
    assert h.getModelStatus().name == "kModelEmpty"
    return h


def _variable_free_solution(row_lower, row_upper, row_names):
    sol = _StubSolution(optimal=False, diagnostics=_diag(status_name="kModelEmpty", primal_feasible=False))
    sol.highs = _variable_free_highs(row_lower, row_upper)
    sol.row_names = row_names
    return sol


def test_variable_free_row_that_cannot_hold_is_infeasible() -> None:
    """A hard balance ``0 = 589`` with nothing in it: infeasible, not "no variables or constraints"."""
    acc = classify_acceptance(
        _variable_free_solution([0.0, 589.0, -1.0], [0.0, 589.0, 2.0], ["holds", "balance[west,t1]", "range"]),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert acc.model_status == "kInfeasible"
    assert acc.message == (
        "solve for s is infeasible: the model has no variables, and 1 of its 3 constraints cannot hold with "
        "nothing in them: balance[west,t1] (0 not in [589, 589])")


def test_variable_free_failing_rows_are_capped_in_the_message() -> None:
    n = 7
    acc = classify_acceptance(
        _variable_free_solution([-math.inf] * n, [-1.0] * n, [f"r{i}" for i in range(n)]),
        ranges_post=None,
        solve_name="s",
    )
    assert acc.model_status == "kInfeasible"
    assert acc.message.endswith("r4 (0 not in [-inf, -1]) and 2 more")
    assert "r5" not in acc.message


def test_variable_free_rows_that_hold_say_there_is_nothing_to_solve() -> None:
    acc = classify_acceptance(
        _variable_free_solution([0.0, -math.inf], [0.0, 3.0], ["a", "b"]),
        ranges_post=None,
        solve_name="s",
    )
    assert not acc.accepted
    assert acc.model_status == "kModelEmpty"
    assert acc.message == (
        "solve for s has no variables: its 2 constraints hold with nothing in them, so there is nothing to solve")


def test_model_empty_without_a_readable_lp_keeps_the_plain_message() -> None:
    acc = classify_acceptance(
        _StubSolution(optimal=False, diagnostics=_diag(status_name="kModelEmpty", primal_feasible=False)),
        ranges_post=None,
        solve_name="s",
    )
    assert acc.model_status == "kModelEmpty"
    assert acc.message == "solve for s has no variables or constraints"


# --- time limit: a MIP keeps its incumbent with a warning, an LP stops (owner decision D2, every solver) -------


class _Var:
    def __init__(self, integer: bool) -> None:
        import polars as pl

        self.integer = integer
        self.frame = pl.DataFrame({"col_id": [0]})


def _time_limited(*, mip: bool, diagnostics=None, shim_gap: float | None = None) -> _StubSolution:
    sol = _StubSolution(optimal=False, diagnostics=diagnostics)
    sol._vars = {"x": _Var(integer=False), "u": _Var(integer=mip)}
    if diagnostics is None:  # a commercial / --save-memory result file
        sol.shim_status = "TIME_LIMIT"
        sol.shim_mip_gap = shim_gap
    return sol


@pytest.mark.parametrize("source", ["highs", "result_file"])
def test_time_limited_mip_with_a_solution_is_kept_with_a_warning(source: str) -> None:
    diagnostics = _diag(status_name="kTimeLimit", primal_feasible=True) if source == "highs" else None
    acc = classify_acceptance(
        _time_limited(mip=True, diagnostics=diagnostics, shim_gap=0.010101), ranges_post=None, solve_name="s",
    )
    assert acc.accepted and acc.time_limited and acc.model_status == "kTimeLimit"
    assert "time limit" in acc.message
    if source == "result_file":
        assert "1.0101%" in acc.message


@pytest.mark.parametrize("source", ["highs", "result_file"])
def test_time_limited_lp_stops_the_run(source: str) -> None:
    diagnostics = _diag(status_name="kTimeLimit", primal_feasible=True) if source == "highs" else None
    acc = classify_acceptance(_time_limited(mip=False, diagnostics=diagnostics), ranges_post=None, solve_name="s")
    assert not acc.accepted and not acc.time_limited and acc.model_status == "kTimeLimit"
    assert "time limit" in acc.message


def _highs_time_limit_diag(**mip_gap: float) -> types.SimpleNamespace:
    """kTimeLimit diagnostics with a feasible incumbent, with ``mip_gap`` set to the given value or absent.

    Built from a dict so it works both with a polar-high whose ``SolveDiagnostics`` has the ``mip_gap`` field
    and with the released 3.12.4 that has not."""
    d = dataclasses.asdict(_diag(status_name="kTimeLimit", primal_feasible=True))
    d.pop("mip_gap", None)
    d.update(mip_gap)
    return types.SimpleNamespace(**d)


def test_time_limited_highs_mip_names_the_gap() -> None:
    acc = classify_acceptance(
        _time_limited(mip=True, diagnostics=_highs_time_limit_diag(mip_gap=0.002286)),
        ranges_post=None, solve_name="invest_2030",
    )
    assert acc.accepted and acc.time_limited
    assert "not proven optimal (relative MIP gap 0.2286%)." in acc.message


@pytest.mark.parametrize("gap", ["absent", math.inf, math.nan])
def test_time_limited_highs_mip_without_a_gap_points_to_the_log(gap) -> None:
    diagnostics = _highs_time_limit_diag() if gap == "absent" else _highs_time_limit_diag(mip_gap=gap)
    acc = classify_acceptance(_time_limited(mip=True, diagnostics=diagnostics), ranges_post=None, solve_name="s")
    assert acc.accepted and acc.time_limited
    assert "(the remaining gap is in the solver log above)" in acc.message


def test_time_limited_highs_mip_reads_the_gap_from_the_pre_unscale_snapshot() -> None:
    """The Layer-2 unscale resets the live HiGHS info; the gap comes from the snapshot taken before it."""
    sol = _time_limited(mip=True, diagnostics=_highs_time_limit_diag())   # live handle: no gap
    sol.pre_unscale_diagnostics = _highs_time_limit_diag(mip_gap=0.002286)
    acc = classify_acceptance(sol, ranges_post=None, solve_name="s")
    assert acc.accepted and "relative MIP gap 0.2286%" in acc.message


def test_time_limited_result_file_with_a_non_finite_gap_points_to_the_log() -> None:
    acc = classify_acceptance(_time_limited(mip=True, shim_gap=math.inf), ranges_post=None, solve_name="s")
    assert acc.accepted and "(the remaining gap is in the solver log above)" in acc.message


def test_time_limited_mip_without_a_solution_stops_the_run() -> None:
    acc = classify_acceptance(
        _time_limited(mip=True, diagnostics=_diag(status_name="kTimeLimit", primal_feasible=False)),
        ranges_post=None, solve_name="s",
    )
    assert not acc.accepted and acc.model_status == "kTimeLimit"
