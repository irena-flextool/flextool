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
* ``kModelEmpty`` of an LP with rows but no variables (HiGHS ignores the
  rows of such an LP) → reject, as infeasible when a row excludes 0.
* ``kUnknown`` that is primal-feasible with a small primal--dual objective
  gap → accept as *near-optimal* (use the primal solution), else reject.
* a solve stopped by a **search limit** (time, work, node, leaf, solution-count or iteration limit) or at an
  **objective target or bound** the user set, on every solver (HiGHS ``kTimeLimit`` / ``kSolutionLimit`` /
  ``kIterationLimit`` / ``kObjectiveTarget`` / ``kObjectiveBound``; a commercial or ``--save-memory`` result
  file's ``TIME_LIMIT`` / ``WORK_LIMIT`` / ``NODE_LIMIT`` / ``SOLUTION_LIMIT`` / ``ITERATION_LIMIT`` /
  ``USER_OBJ_LIMIT`` / ``OBJECTIVE_TARGET`` / ``OBJECTIVE_BOUND``) → without a feasible solution, reject,
  naming the stop and its option; an LP, reject (an LP iterate at a limit is not a usable optimum, even when
  primal feasible); a MIP with a feasible solution, accept with a warning naming the stop, its option and the
  gap (the solution is not proven optimal; the gap is named when the solver reported it: a commercial
  solver's result file or console output, the ``--save-memory`` HiGHS child's gap file, or in-process
  HiGHS's ``SolveDiagnostics.mip_gap`` when the installed polar-high has that field).  A memory limit, an
  interrupt and every other uncertified status stay rejected.

``FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0`` rejects every status other than optimal, a limit-stopped MIP included.

The predicate is conjunctive and derives its primal-feasibility limits
from the solver's own ``primal_feasibility_tolerance`` (the option value
HiGHS ran with, read from the live ``Highs`` handle; the Layer-2 unscale's
``setSolution`` resets the status and info counters, not the options):

* relative ceiling ``max(1e-6, 10 x primal_feasibility_tolerance)`` -- a
  solve run with a loose tolerance (e.g. 1e-3) is judged against what it
  was asked to achieve, while the HiGHS default 1e-7 keeps the 1e-6 floor;
* absolute backstop ``10 x primal_feasibility_tolerance``.

The dual side is judged by the primal--dual objective error only (FlexTool
consumes the primal), so there is no dual-infeasibility ceiling to scale.

Precedence: an explicit ``FLEXTOOL_ACCEPT_PRIMAL_REL`` (a positive float)
replaces the derived relative ceiling outright, tighter or looser; a
malformed or non-positive value is ignored and the derived ceiling applies.

All policy and thresholds live here; polar-high only supplies the facts
(:meth:`polar_high.Solution.solve_diagnostics`).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from flextool.engine.autoscale._report import (
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

# Scale-invariant primal-feasibility ceiling: the floor is one decade above
# the HiGHS default ``primal_feasibility_tolerance`` (1e-7) and far below any
# physically meaningful constraint violation; a solve run with a looser
# tolerance gets ``_ACCEPT_PRIMAL_REL_TOL_FACTOR`` times its own tolerance
# (see :func:`primal_rel_ceiling`).
_ACCEPT_PRIMAL_REL_DEFAULT = 1e-6
_ACCEPT_PRIMAL_REL_TOL_FACTOR = 10.0

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


def _env_override(name: str) -> float | None:
    """The positive float set in ``os.environ[name]``, else ``None`` (unset,
    malformed or non-positive)."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0.0 else None


def primal_rel_ceiling(primal_feasibility_tolerance: float) -> float:
    """Largest max relative primal infeasibility of an accepted near-optimal
    solve: ``FLEXTOOL_ACCEPT_PRIMAL_REL`` when set, else
    ``max(1e-6, 10 x primal_feasibility_tolerance)``.

    A tolerance of 0 (no live solver handle) gives the 1e-6 floor.
    """
    override = _env_override("FLEXTOOL_ACCEPT_PRIMAL_REL")
    if override is not None:
        return override
    return max(
        _ACCEPT_PRIMAL_REL_DEFAULT,
        _ACCEPT_PRIMAL_REL_TOL_FACTOR * float(primal_feasibility_tolerance),
    )


def _near_optimal_enabled() -> bool:
    """Whether the conditional near-optimal accept is active.

    Set ``FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0`` to restore the legacy
    hard-fail-on-non-kOptimal behaviour (e.g. for strict test runs): it
    rejects an uncertified ``kUnknown`` and a MIP stopped by a limit alike.
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
    "kSolutionLimit": "hit the node or solution limit before proving optimality",
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
    # (A model with no variables but with rows is judged by its rows: see
    # :func:`_classify_variable_free_model`.)
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
    ``above_fixed_ceiling`` — accepted near-optimal with a max relative primal
                        infeasibility above the former fixed 1e-6 ceiling
                        (accepted only because the ceiling follows the
                        solve's tolerance); the caller prints the
                        behaviour-change notice for it.
    ``limit_stopped`` — accepted MIP solution of a solve stopped by a limit
                        (time, node, solution, work, iteration) or at an
                        objective target or bound: the caller logs
                        :attr:`message` as a WARNING.
    ``model_status``  — the solve's own HiGHS model status name
                        (``kOptimal``, ``kInfeasible``, ...) as the check saw
                        it: from the solve's diagnostics (never the live
                        handle, which reads ``kNotset`` after the autoscale
                        unscale), else from a subprocess / commercial
                        solver's parsed status; ``None`` when unknown.  For
                        the run record.
    """

    accepted: bool
    near_optimal: bool
    message: str
    scaling_hint: str | None
    above_fixed_ceiling: bool = False
    model_status: str | None = None
    limit_stopped: bool = False


#: The status a subprocess / commercial solve's parsed solution file reports
#: (``Solution.shim_status``, set by ``_subprocess_solve``) as a HiGHS model
#: status name.  ``OTHER`` (nothing more is known) maps to ``None``.
_SHIM_MODEL_STATUS = {
    "OPTIMAL": "kOptimal",
    "INFEASIBLE": "kInfeasible",
    "UNBOUNDED": "kUnbounded",
    "INF_OR_UNBD": "kUnboundedOrInfeasible",
    "TIME_LIMIT": "kTimeLimit",
    "ITERATION_LIMIT": "kIterationLimit",
    "SOLUTION_LIMIT": "kSolutionLimit",
    "NODE_LIMIT": "kSolutionLimit",        # HiGHS's own node-limit status
    "WORK_LIMIT": "kTimeLimit",            # Gurobi WorkLimit: a deterministic time limit
    "USER_OBJ_LIMIT": "kObjectiveTarget",  # Gurobi BestObjStop / BestBdStop
    "OBJECTIVE_TARGET": "kObjectiveTarget",
    "OBJECTIVE_BOUND": "kObjectiveBound",
    "OBJECTIVE_LIMIT": "kObjectiveBound",  # CPLEX 12 / 21 / 22: LP objective-limit aborts
    "CUTOFF": "kObjectiveBound",           # Gurobi Cutoff: no solution better than the cutoff
    "MEM_LIMIT": "kMemoryLimit",
    "INTERRUPTED": "kInterrupt",
}

#: Statuses of a solver run as a program that have no HiGHS twin, and why each is not a usable optimum
#: (``_subprocess_solve``).  Rejected like any other non-optimal status, with this reason.
_SHIM_STATUS_REASONS = {
    "NO_STATUS": (
        "did not say how it ended: its result file carries no status and its console log names none (is the "
        "log turned off? Gurobi: LogToConsole 1; COPT: Logging 1 and LogToConsole 1), so a solve stopped by a "
        "limit or an interrupt cannot be told from a finished one"
    ),
    "OPTIMAL_INFEASIBLE": (
        "is optimal only for the solver's scaled model: once unscaled, the solution breaks the feasibility "
        "tolerances (CPLEX: optimal with unscaled infeasibilities)"
    ),
    "OPTIMAL_FACE_UNBOUNDED": (
        "has an unbounded optimal face: the solution is an interior point of it, not a vertex (CPLEX barrier "
        "without crossover; a cost or bound is probably missing)"
    ),
}


def _is_mip(sol: "Solution") -> bool:
    """Whether the solved problem has integer columns (from the Solution's variables)."""
    from flextool.engine.solve._solver_dispatch import problem_is_mip

    return problem_is_mip(getattr(sol, "_vars", None) or getattr(sol, "vars", None))


#: Result-file statuses (``Solution.shim_status``) of a solve stopped by a search limit or at an objective target
#: or bound the user set: judged by :func:`_limit_verdict`.  ``OBJECTIVE_LIMIT`` (CPLEX's LP-only objective-limit
#: aborts) and ``CUTOFF`` (Gurobi: no solution information) are not among them: they are rejected by name.
LIMIT_SHIM_STATUSES = frozenset({
    "TIME_LIMIT", "WORK_LIMIT", "NODE_LIMIT", "SOLUTION_LIMIT", "ITERATION_LIMIT",
    "USER_OBJ_LIMIT", "OBJECTIVE_TARGET", "OBJECTIVE_BOUND",
})

#: HiGHS model statuses judged by :func:`_limit_verdict`.
_LIMIT_MODEL_STATUSES = frozenset({
    "kTimeLimit", "kSolutionLimit", "kIterationLimit", "kObjectiveTarget", "kObjectiveBound",
})

#: What stopped the solve, by its native status (a HiGHS model status, or a result-file status of another solver).
_LIMIT_NOUNS = {
    "kTimeLimit": "time limit", "TIME_LIMIT": "time limit",
    "kSolutionLimit": "node or solution limit",
    "NODE_LIMIT": "node limit", "SOLUTION_LIMIT": "solution limit", "WORK_LIMIT": "work limit",
    "kIterationLimit": "iteration limit", "ITERATION_LIMIT": "iteration limit",
    "kObjectiveTarget": "objective target", "OBJECTIVE_TARGET": "objective target",
    "kObjectiveBound": "objective bound", "OBJECTIVE_BOUND": "objective bound",
    "USER_OBJ_LIMIT": "objective limit",
}

#: Stops at a value the user set (not a search effort): "change or remove" it rather than "raise" it.
_OBJECTIVE_STOPS = frozenset({
    "kObjectiveTarget", "OBJECTIVE_TARGET", "kObjectiveBound", "OBJECTIVE_BOUND", "USER_OBJ_LIMIT",
})

#: The solver options behind each stop, by ``(solver, native status)``.  HiGHS has one status for its node,
#: leaf, stall-node and improving-solution limits, so all four are named.  A stop not listed (an unverified
#: option, e.g. COPT's iteration limit) names "the solver's own ... parameter".
_LIMIT_OPTIONS = {
    ("highs", "kSolutionLimit"): "mip_max_nodes, mip_max_leaves, mip_max_stall_nodes or mip_max_improving_sols",
    ("highs", "kIterationLimit"): "simplex_iteration_limit, ipm_iteration_limit or pdlp_iteration_limit",
    ("highs", "kObjectiveTarget"): "objective_target",
    ("highs", "kObjectiveBound"): "objective_bound",
    ("gurobi", "NODE_LIMIT"): "NodeLimit",
    ("gurobi", "SOLUTION_LIMIT"): "SolutionLimit",
    ("gurobi", "WORK_LIMIT"): "WorkLimit",
    ("gurobi", "ITERATION_LIMIT"): "IterationLimit or BarIterLimit",
    ("gurobi", "USER_OBJ_LIMIT"): "BestObjStop or BestBdStop",
    ("cplex", "NODE_LIMIT"): "mip limits nodes",
    ("cplex", "SOLUTION_LIMIT"): "mip limits solutions",
    ("cplex", "ITERATION_LIMIT"): "simplex limits iterations or barrier limits iteration",
    ("copt", "NODE_LIMIT"): "NodeLimit",
}

_SOLVER_DISPLAY_NAMES = {"highs": "HiGHS", "gurobi": "Gurobi", "cplex": "CPLEX", "copt": "COPT", "xpress": "Xpress"}


def _remedy(solver: str | None, native: str, noun: str) -> str:
    """What to change so the next run is not stopped there (lower case; capitalised at a sentence start)."""
    if noun == "time limit":
        return "raise the time limit (--solver-time-limit, or the solver's own time-limit parameter)"
    action = f"change or remove the {noun}" if native in _OBJECTIVE_STOPS else "raise or remove the limit"
    options = _LIMIT_OPTIONS.get((solver or "", native))
    if options is None:
        return f"{action} (the solver's own {noun} parameter)"
    return (
        f"{action} ({_SOLVER_DISPLAY_NAMES.get(solver or '', solver)} {options}, in the solver's option file or "
        "the solve's solver_arguments)"
    )


def _capitalised(text: str) -> str:
    return text[:1].upper() + text[1:]


def _limit_verdict(
    sol: "Solution",
    solve_name: str,
    *,
    status: str,
    native: str,
    solver: str | None,
    gap: float | None,
    primal_feasible: bool,
    ranges_post: "RangeReport | None",
) -> Acceptance:
    """The rule for a solve stopped by a search limit or at an objective target or bound (owner decision D2,
    extended to every limit; every solver).

    *status* is the HiGHS model status recorded for the run, *native* the solver's own status (the same for
    HiGHS), which names the stop and its option.  Without a feasible solution there is nothing to use.  An LP
    stopped there has no usable solution either (the iterate is not a feasible optimum).  A MIP keeps its
    feasible solution with a warning: it is not proven optimal, which a limit on a MIP asks for.
    ``FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0`` rejects that too.
    """
    noun = _LIMIT_NOUNS.get(native, "limit")
    remedy = _remedy(solver, native, noun)
    objective_stop = native in _OBJECTIVE_STOPS
    if not primal_feasible:
        stop = f"stopped at an {noun}" if objective_stop else f"hit the {noun}"
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"solve for {solve_name} {stop} before finding a feasible solution, so there is no solution to "
                f"use. {_capitalised(remedy)}"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
            model_status=status,
        )
    if not _is_mip(sol):
        stop = (
            f"stopped at an {noun}, not a proven optimum" if objective_stop
            else f"hit the {noun} before proving optimality"
        )
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"solve for {solve_name} {stop}: it is a linear problem, and a linear solve stopped at its "
                f"{noun} has no usable solution. {_capitalised(remedy)}"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
            model_status=status,
        )
    # HiGHS reports ``inf`` (or ``NaN``) when it has no gap to give; that reads as "no gap" here.
    gap_text = (
        f" (relative MIP gap {_format_gap(gap)})" if gap is not None and math.isfinite(gap)
        else " (the remaining gap is in the solver log above)"
    )
    kept = (
        f"Solve {solve_name} stopped at its {noun} with a feasible integer solution that is not proven "
        f"optimal{gap_text}"
    )
    if not _near_optimal_enabled():
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"{kept}, and near-optimal acceptance is disabled (FLEXTOOL_ACCEPT_NEAR_OPTIMAL=0). "
                f"{_capitalised(remedy)}"
            ),
            scaling_hint=None,
            model_status=status,
        )
    return Acceptance(
        accepted=True,
        near_optimal=True,
        message=f"{kept}. FlexTool uses this solution; {remedy} or the MIP gap for a proven one.",
        scaling_hint=None,
        model_status=status,
        limit_stopped=True,
    )


def highs_objective_bound_warning(
    sol: "Solution", solve_name: str, solver_options: "Mapping[str, Any] | None",
) -> str | None:
    """The warning for a HiGHS MIP solved with a finite ``objective_bound`` (from any option layer), else ``None``.

    HiGHS treats the bound as a cutoff on a MIP: it prunes every solution not better than the bound and then
    reports the best one it kept as optimal, so its "optimal" may not be the optimum (measured with HiGHS 1.15.1).
    """
    if not solver_options or "objective_bound" not in solver_options:
        return None
    try:
        bound = float(solver_options["objective_bound"])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bound) or not _is_mip(sol):
        return None
    return (
        f"Solve {solve_name} is a MIP solved by HiGHS with objective_bound {bound:g} (unscaled currency): HiGHS "
        "treats the bound as "
        "a cutoff and reports the best solution within it as optimal, so this solution may not be optimal. "
        "Remove objective_bound for a proven optimum."
    )


def _format_gap(gap: float) -> str:
    """A relative MIP gap as a percentage: four decimals (``0.2286%``), or three significant digits below 0.01 %
    (``3.2e-05%``) so a small nonzero gap does not print as ``0.0000%`` next to "not proven optimal"."""
    if 0.0 < abs(gap) < 1e-4:
        return f"{gap * 100:.3g}%"
    return f"{gap:.4%}"


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


#: How many failing rows a variable-free model's message names.
_VARIABLE_FREE_ROWS_SHOWN = 5


def _classify_variable_free_model(sol: "Solution", solve_name: str) -> Acceptance | None:
    """The verdict on a ``kModelEmpty`` solve whose LP has rows but no columns, else ``None``.

    Every row of such an LP reads ``row_lower <= 0 <= row_upper``.  A row that excludes 0 cannot hold, so the
    model is infeasible (``kInfeasible``, the failing rows named).  When all hold, there is still nothing to
    solve: rejected as ``kModelEmpty``, with a message that says the constraints are there.  ``None`` when the
    live LP cannot be read or has no rows (the plain ``kModelEmpty`` message applies).
    """
    get_lp = getattr(getattr(sol, "highs", None), "getLp", None)
    if not callable(get_lp):
        return None
    lp = get_lp()
    if int(lp.num_col_) != 0 or int(lp.num_row_) == 0:
        return None
    lower = np.asarray(lp.row_lower_, dtype=np.float64)
    upper = np.asarray(lp.row_upper_, dtype=np.float64)
    tol = float(sol.primal_feasibility_tolerance) or 1e-7
    failing = np.flatnonzero((lower > tol) | (upper < -tol))
    n_rows = int(lp.num_row_)
    if failing.size == 0:
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"solve for {solve_name} has no variables: its {n_rows} "
                "constraints hold with nothing in them, so there is nothing "
                "to solve"
            ),
            scaling_hint=None,
            model_status="kModelEmpty",
        )
    names = list(getattr(sol, "row_names", None) or ())
    shown = [
        (names[i] if i < len(names) and names[i] else f"row {i}")
        + f" (0 not in [{lower[i]:g}, {upper[i]:g}])"
        for i in failing[:_VARIABLE_FREE_ROWS_SHOWN].tolist()
    ]
    more = failing.size - len(shown)
    return Acceptance(
        accepted=False,
        near_optimal=False,
        message=(
            f"solve for {solve_name} is infeasible: the model has no "
            f"variables, and {failing.size} of its {n_rows} "
            f"constraints cannot hold with nothing in them: {', '.join(shown)}"
            + (f" and {more} more" if more else "")
        ),
        scaling_hint=None,
        model_status="kInfeasible",
    )


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
            model_status="kOptimal",
        )

    # ``solve_diagnostics`` landed in polar-high 3.7.0 (pyproject pins it).
    # Guard the call so an environment that somehow has an older polar-high
    # degrades to the honest "cannot diagnose → reject" path below instead of
    # crashing the whole cascade with an AttributeError.
    #
    # The autoscale Layer-2 unscale pushes the unscaled solution back into
    # HiGHS with ``setSolution``, which resets the live model status to
    # ``kNotset`` and clears the info counters; it keeps the solve's own
    # diagnostics as ``pre_unscale_diagnostics``, so read those first.
    diag = getattr(sol, "pre_unscale_diagnostics", None)
    if diag is None:
        diag_fn = getattr(sol, "solve_diagnostics", None)
        diag = diag_fn() if callable(diag_fn) else None

    # No queryable solver handle (synthesised Solution, or the read-only
    # subprocess/commercial shim), or a polar-high too old to diagnose: we
    # cannot verify the solution, so we must NOT accept it.  Reject with an
    # honest "cannot diagnose" message.
    if diag is None:
        shim_status = getattr(sol, "shim_status", None)
        if shim_status in LIMIT_SHIM_STATUSES:
            # The result file of a solve stopped by a limit holds its incumbent (a commercial result without one
            # is refused before this check, ``_subprocess_solve._NO_SOLUTION_REASONS``).  The ``--save-memory``
            # HiGHS child's status is named as in-process HiGHS names it.
            solver = getattr(sol, "shim_solver", None)
            model_status = _SHIM_MODEL_STATUS[shim_status]
            return _limit_verdict(
                sol, solve_name,
                status=model_status,
                native=model_status if solver == "highs" else shim_status,
                solver=solver,
                gap=getattr(sol, "shim_mip_gap", None),
                primal_feasible=getattr(sol, "shim_primal_feasible", True),
                ranges_post=ranges_post,
            )
        model_status = _SHIM_MODEL_STATUS.get(shim_status)
        if shim_status in _SHIM_STATUS_REASONS:
            return Acceptance(
                accepted=False,
                near_optimal=False,
                message=(
                    f"non-optimal solve for {solve_name}: the solver {_SHIM_STATUS_REASONS[shim_status]} "
                    f"(solver status {shim_status})"
                ),
                scaling_hint=None,
                model_status=None,
            )
        if model_status in _FAILURE_MESSAGES:
            return Acceptance(
                accepted=False,
                near_optimal=False,
                message=f"solve for {solve_name} {_FAILURE_MESSAGES[model_status]}",
                scaling_hint=None,
                model_status=model_status,
            )
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=(
                f"non-optimal solve for {solve_name}: the solver did not "
                "certify optimality and no solver diagnostics are available "
                "to assess the solution"
                + (f" (solver status {shim_status})" if shim_status not in (None, "OTHER") else "")
            ),
            scaling_hint=None,
            model_status=model_status,
        )

    status = diag.model_status_name

    # HiGHS reports any LP without columns as ``kModelEmpty`` and ignores its
    # rows, so a variable-free row that cannot hold (a hard ``0 = demand``)
    # would read as "no constraints".  Judge such rows here.
    if status == "kModelEmpty":
        variable_free = _classify_variable_free_model(sol, solve_name)
        if variable_free is not None:
            return variable_free

    # A solve stopped by a limit or at an objective target or bound: the D2 rule, every limit.
    if status in _LIMIT_MODEL_STATUSES:
        # ``SolveDiagnostics.mip_gap`` is newer than polar-high 3.12.4; without it the warning points to the log.
        return _limit_verdict(
            sol, solve_name,
            status=status, native=status, solver="highs",
            gap=getattr(diag, "mip_gap", None),
            primal_feasible=bool(diag.primal_feasible),
            ranges_post=ranges_post,
        )

    # A named failure status is never acceptable — report the precise cause.
    if status in _FAILURE_MESSAGES:
        return Acceptance(
            accepted=False,
            near_optimal=False,
            message=f"solve for {solve_name} {_FAILURE_MESSAGES[status]}",
            scaling_hint=_scaling_hint_for_reject(ranges_post),
            model_status=status,
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
            model_status=status,
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
            model_status=status,
        )

    primal_tol = float(sol.primal_feasibility_tolerance)
    primal_rel_tol = primal_rel_ceiling(primal_tol)
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
        <= primal_tol * primal_abs_margin
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
                f"{primal_rel_tol:.1e}; "
                f"{diag.num_primal_infeasibilities} infeasibilities). The "
                "solution cannot be used"
            ),
            scaling_hint=_scaling_hint_for_reject(ranges_post),
            model_status=status,
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
            model_status=status,
        )

    # Accept as near-optimal: feasible primal, small gap.
    msg = (
        f"Accepted near-optimal solve for {solve_name}: HiGHS could not "
        "certify optimality after postsolve (status Unknown), but the primal "
        "solution is feasible (max relative primal infeasibility "
        f"{diag.max_relative_primal_infeasibility:.2e} <= {primal_rel_tol:.1e}"
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
    above_fixed = (
        _env_override("FLEXTOOL_ACCEPT_PRIMAL_REL") is None
        and diag.max_relative_primal_infeasibility > _ACCEPT_PRIMAL_REL_DEFAULT
    )
    return Acceptance(
        accepted=True, near_optimal=True, message=msg, scaling_hint=None,
        above_fixed_ceiling=above_fixed, model_status=status,
    )


__all__ = [
    "Acceptance", "LIMIT_SHIM_STATUSES", "classify_acceptance", "highs_objective_bound_warning",
    "primal_rel_ceiling",
]
