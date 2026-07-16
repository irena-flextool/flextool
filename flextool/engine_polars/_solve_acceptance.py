"""Accept/reject policy for a completed top-level cascade solve.

FlexTool historically treated any HiGHS model status other than
``kOptimal`` as a hard failure: it logged "non-optimal solve", printed a
scaling hint, and aborted the cascade before writing any output.  That is
wrong for one common, legitimate case.

An interior-point solve run **without crossover** (a choice the model
generator makes via the ``run_crossover`` solver option) returns the raw
interior point rather than a basic vertex.  On an aggressive presolve, the
HiGHS post-solve step can leave the *dual* objective slightly inconsistent
even though the *primal* solution is genuinely feasible and — because the
pre-post-solve duality gap was ~0 — in-practice optimal.  HiGHS reports
this as ``kUnknown`` (it cannot *certify* optimality), not as a failure.
FlexTool consumes the primal solution for every output, so such a solve is
usable.

This module decides, from the solver's own diagnostics, whether a
non-``kOptimal`` solve is safe to accept:

* ``kOptimal``                → accept (unchanged).
* a genuine failure status    → reject, naming the precise cause.
* ``kUnknown`` that is primal-feasible with a small primal--dual objective
  gap → accept as *near-optimal* (use the primal solution), else reject.

The predicate is conjunctive and derives its primal-feasibility margin
from the solver's own ``primal_feasibility_tolerance`` rather than a magic
constant, so it cannot silently wave through an infeasible solution.  All
policy and thresholds live here; polar-high only supplies the facts
(:meth:`polar_high.Solution.solve_diagnostics`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from flextool.engine_polars.autoscale._report import (
    format_nonoptimal_hint as _format_nonoptimal_hint,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from polar_high import Solution
    from polar_high.autoscale import RangeReport


# --- thresholds (env-overridable, deterministic defaults) -------------------
#
# Defaults are justified against the observed failure and the tolerances the
# model already declares acceptable; see module docstring and the PR notes.

# Upper bound on the accepted primal--dual objective gap for a near-optimal
# ``kUnknown``.  ``primal_dual_objective_error`` is an upper bound on how far
# the primal could be from optimal if the (inconsistent) duals were taken at
# face value.  The observed crossover-off roll had 0.00477 (0.477%); models
# routinely carry ``mip_rel_gap=0.01`` (1%), i.e. a 1% optimality gap is
# already deemed acceptable.  1e-2 accepts the observed case with headroom
# while rejecting the ~3% stalls seen in decomposition subproblems.
_ACCEPT_PD_GAP_DEFAULT = 1e-2

# Scale-invariant primal-feasibility ceiling: two decades above the default
# ``primal_feasibility_tolerance`` (1e-7) but far below any physically
# meaningful constraint violation.
_ACCEPT_PRIMAL_REL_DEFAULT = 1e-6

# Absolute backstop: HiGHS enforces feasibility on the internally-scaled LP,
# so the unscaled slack can exceed the nominal tolerance.  One decade of
# un-scaling headroom over the solver's own tolerance.
_ACCEPT_PRIMAL_ABS_MARGIN_DEFAULT = 10.0


def _env_float(name: str, default: float) -> float:
    """Read a positive float from ``os.environ[name]`` or return default.

    A malformed or non-positive value falls back to the default rather than
    silently disabling a guard.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return val if val > 0.0 else default


def _near_optimal_enabled() -> bool:
    """Whether the conditional near-optimal accept is active.

    Set ``FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0`` to restore the legacy
    hard-fail-on-non-kOptimal behaviour (e.g. for strict test runs).
    """
    return os.environ.get("FLEXTOOL_ACCEPT_NEAR_OPTIMAL", "1") != "0"


# Model statuses that are never acceptable, mapped to an honest, cause-naming
# message body.  Keyed by ``HighsModelStatus`` enum *name* so the module never
# imports highspy.
_FAILURE_MESSAGES: dict[str, str] = {
    "kInfeasible": (
        "is infeasible: no assignment of the variables satisfies all "
        "constraints"
    ),
    "kUnbounded": (
        "is unbounded: the objective improves without limit, so a cost or "
        "bound is missing"
    ),
    "kUnboundedOrInfeasible": (
        "is unbounded or infeasible (presolve could not distinguish the "
        "two); re-run with presolve off to disambiguate"
    ),
    "kTimeLimit": "hit the time limit before proving optimality",
    "kIterationLimit": "hit the iteration limit before proving optimality",
    "kMemoryLimit": "hit the memory limit before proving optimality",
    "kObjectiveBound": (
        "stopped at an objective bound, not a proven optimum"
    ),
    "kObjectiveTarget": (
        "stopped at an objective target, not a proven optimum"
    ),
    "kSolveError": (
        "failed inside the solver (solve stage); the solution is not usable"
    ),
    "kPresolveError": (
        "failed inside the solver (presolve stage); the solution is not "
        "usable"
    ),
    "kPostsolveError": (
        "failed inside the solver (postsolve stage); the solution is not "
        "usable"
    ),
    "kModelEmpty": "has no variables or constraints",
    "kNotset": "returned no model status",
    "kLoadError": "could not be loaded by the solver",
    "kModelError": "was rejected as malformed by the solver",
    "kInterrupt": "was interrupted before proving optimality",
    "kHighsInterrupt": "was interrupted before proving optimality",
}


@dataclass
class Acceptance:
    """Outcome of :func:`classify_acceptance`.

    ``accepted``      — whether the cascade may consume this solve's solution.
    ``near_optimal``  — accepted despite a non-``kOptimal`` status (log INFO,
                        not silently).
    ``message``       — human-readable line describing the decision.
    ``scaling_hint``  — optional multi-line remediation hint, populated only
                        when scaling is genuinely implicated in a *reject*.
    """

    accepted: bool
    near_optimal: bool
    message: str
    scaling_hint: str | None


def _scaling_hint_for_reject(
    ranges_post: "RangeReport | None",
) -> str | None:
    """Return the scaling remediation hint only when the *actually solved*
    (post-autoscale) LP is still ill-conditioned.

    The historical bug keyed this off the raw, pre-autoscale ranges — always
    wide for FlexTool commodity ladders — so the hint fired on essentially
    every non-optimal solve regardless of cause.  Keying off ``ranges_post``
    (the post-Layer-2 ranges, computed only when the pre-ranges tripped the
    detector) means the hint appears only when the autoscaler could NOT tame
    the range spread, i.e. when scaling is a plausible culprit.
    """
    if ranges_post is None or not ranges_post.trigger:
        return None
    hint = _format_nonoptimal_hint(ranges_post)
    return hint or None


def classify_acceptance(
    sol: "Solution",
    *,
    ranges_post: "RangeReport | None",
    solve_name: str,
) -> Acceptance:
    """Decide whether *sol* is safe for the cascade to consume.

    See the module docstring for the policy.  ``ranges_post`` is the
    post-autoscale :class:`RangeReport` for this solve (``None`` when the raw
    LP never tripped the scaling detector); it gates the reject-path scaling
    hint only.
    """
    # Fast path: HiGHS certified optimality — nothing to decide.
    if sol.optimal:
        return Acceptance(
            accepted=True, near_optimal=False, message="", scaling_hint=None,
        )

    diag = sol.solve_diagnostics()

    # No queryable solver handle (synthesised Solution, or the read-only
    # subprocess/commercial shim): we cannot verify the solution, so we must
    # NOT accept it.  Reject with an honest "cannot diagnose" message.
    if diag is None:
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"non-optimal solve for {solve_name}: the solver did not "
                "certify optimality and no solver diagnostics are available "
                "to assess the solution"
            ),
            scaling_hint=None,
        )

    status = diag.model_status_name

    # A named failure status is never acceptable — report the precise cause.
    if status in _FAILURE_MESSAGES:
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=f"solve for {solve_name} {_FAILURE_MESSAGES[status]}",
            scaling_hint=_scaling_hint_for_reject(ranges_post),
        )

    # Anything that is neither kOptimal (handled above) nor a known failure
    # nor kUnknown is an unrecognised status: reject and say we don't know.
    if status != "kUnknown":
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"non-optimal solve for {solve_name}: the solver returned an "
                f"unrecognised status ({status}); the cause could not be "
                "determined from the available solver diagnostics"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
        )

    # --- kUnknown: the crossover-off / post-solve-uncertified case ----------
    # The kill-switch restores the legacy hard-fail.
    if not _near_optimal_enabled():
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"non-optimal solve for {solve_name}: the solver could not "
                "certify optimality (status Unknown) and near-optimal "
                "acceptance is disabled (FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0)"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
        )

    primal_rel_tol = _env_float(
        "FLEXTOOL_ACCEPT_PRIMAL_REL", _ACCEPT_PRIMAL_REL_DEFAULT
    )
    primal_abs_margin = _env_float(
        "FLEXTOOL_ACCEPT_PRIMAL_ABS_MARGIN", _ACCEPT_PRIMAL_ABS_MARGIN_DEFAULT
    )
    pd_gap_tol = _env_float("FLEXTOOL_ACCEPT_PD_GAP", _ACCEPT_PD_GAP_DEFAULT)

    # Non-negotiable: the primal solution — the thing the cascade consumes —
    # must actually be feasible.  Use HiGHS' own verdict plus a scale-
    # invariant relative check and a tolerance-derived absolute backstop.
    primal_ok = (
        diag.primal_feasible
        and diag.num_primal_infeasibilities == 0
        and diag.max_relative_primal_infeasibility <= primal_rel_tol
        and diag.max_primal_infeasibility
        <= sol.primal_feasibility_tolerance * primal_abs_margin
    )
    if not primal_ok:
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"non-optimal solve for {solve_name}: the solver could not "
                "certify optimality and the returned primal solution is NOT "
                "feasible (max relative primal infeasibility "
                f"{diag.max_relative_primal_infeasibility:.2e} > "
                f"{primal_rel_tol:.0e}; "
                f"{diag.num_primal_infeasibilities} infeasibilities). The "
                "solution cannot be used"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
        )

    # Second requirement: a bounded optimality gap.
    if diag.primal_dual_objective_error > pd_gap_tol:
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"non-optimal solve for {solve_name}: the primal solution is "
                "feasible but the primal-dual objective error "
                f"{diag.primal_dual_objective_error:.3%} exceeds the accepted "
                f"optimality gap {pd_gap_tol:.2%}; optimality cannot be "
                "certified"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
        )

    # Accept as near-optimal: feasible primal, small gap.
    msg = (
        f"Accepted near-optimal solve for {solve_name}: HiGHS could not "
        "certify optimality after postsolve (status Unknown), but the primal "
        "solution is feasible (max relative primal infeasibility "
        f"{diag.max_relative_primal_infeasibility:.2e} <= {primal_rel_tol:.0e}"
        f"; {diag.num_primal_infeasibilities} infeasibilities) and the "
        f"primal-dual objective error {diag.primal_dual_objective_error:.3%} "
        f"is within the accepted optimality gap {pd_gap_tol:.2%}. Using the "
        "primal solution."
    )
    if diag.dual_feasible:
        msg += (
            " The dual solution is also feasible; only the post-solve dual "
            "objective certificate is inconsistent."
        )
    return Acceptance(
        accepted=True, near_optimal=True, message=msg, scaling_hint=None,
    )


__all__ = ["Acceptance", "classify_acceptance"]
