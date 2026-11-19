"""What a HiGHS solve that ended "Optimal", but not cleanly, means for the results: one ``Note:`` line.

HiGHS prints its own ``WARNING:`` lines when an interior-point solve without crossover ends imprecise and
postsolve turns the status into Optimal, or when it calls a solution optimal "since relative violation of
tolerances is no more than X".  Those lines are correct but read as alarms.  After the acceptance verdict
(:func:`flextool.engine.solve._acceptance.classify_acceptance`), the cascade prints one note that says what the
outcome means (codes W-0540..W-0542; texts and the crossover fix in ``run_warnings._engine_specs``).

The facts come from the HiGHS handle in process (``getInfo()`` and the options): they describe the solution
HiGHS returned, after postsolve.  HiGHS's "Before postsolve" lines describe the presolved problem only; a
violation there that postsolve removed gives no note.  The autoscale Layer-2 unscale wipes the handle's info
(``setSolution``), so it keeps a snapshot first (:func:`snapshot_highs_outcome`, stored as
``Solution.pre_unscale_highs_outcome``).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

_LOGGER = logging.getLogger(__name__)

#: ``HighsInfo`` fields the interpretation reads (all present in highspy >= 1.10).
_INFO_FIELDS = (
    "valid", "ipm_iteration_count", "crossover_iteration_count", "simplex_iteration_count", "basis_validity",
    "mip_node_count", "primal_dual_objective_error",
    "num_primal_infeasibilities", "max_primal_infeasibility",
    "num_dual_infeasibilities", "max_dual_infeasibility",
    "num_primal_residual_errors", "max_primal_residual_error",
    "num_dual_residual_errors", "max_dual_residual_error",
    "num_relative_primal_infeasibilities", "max_relative_primal_infeasibility",
    "num_relative_dual_infeasibilities", "max_relative_dual_infeasibility",
    "num_relative_primal_residual_errors", "max_relative_primal_residual_error",
    "num_relative_dual_residual_errors", "max_relative_dual_residual_error",
)
#: Options that set the tolerances the counters are judged against, and the algorithm.
_OPTION_FIELDS = (
    "primal_feasibility_tolerance", "dual_feasibility_tolerance",
    "primal_residual_tolerance", "dual_residual_tolerance", "run_crossover", "solver",
)

#: The four KKT checks HiGHS reports: (info stem, singular stem, tolerance option, the words of its log).
_CHECKS = (
    ("primal_infeasibilities", "primal_infeasibility", "primal_feasibility_tolerance", "primal infeasibilities"),
    ("dual_infeasibilities", "dual_infeasibility", "dual_feasibility_tolerance", "dual infeasibilities"),
    ("primal_residual_errors", "primal_residual_error", "primal_residual_tolerance", "primal residual errors"),
    ("dual_residual_errors", "dual_residual_error", "dual_residual_tolerance", "dual residual errors"),
)

RELATIVE_TOLERANCES = "W-0540"
IPM_RELATIVE_ONLY = "W-0541"
NEAR_OPTIMAL = "W-0542"


def snapshot_highs_outcome(h: Any) -> dict[str, Any] | None:
    """The facts of the solve that just ran on HiGHS handle *h* (model status, info counters, tolerances), or
    ``None`` when the handle cannot tell (no handle, a shim, an info that is not valid)."""
    if h is None or not callable(getattr(h, "getInfo", None)):
        return None
    try:
        info = h.getInfo()
        status = h.getModelStatus()
        facts: dict[str, Any] = {"model_status": getattr(status, "name", str(status))}
        for name in _INFO_FIELDS:
            if hasattr(info, name):
                value = getattr(info, name)
                facts[name] = value if isinstance(value, (bool, int, float)) else int(value)
        for name in _OPTION_FIELDS:
            try:
                result = h.getOptionValue(name)
            except Exception:  # an option this highspy does not have
                continue
            facts[name] = result[1] if isinstance(result, tuple) else result
    except Exception:  # pragma: no cover - a handle that cannot be read is not interpreted
        return None
    if not facts.get("valid", True):
        return None
    return facts


def outcome_facts(sol: Any) -> dict[str, Any] | None:
    """The facts of *sol*'s solve: the snapshot taken before the autoscale unscale, else the live handle."""
    facts = getattr(sol, "pre_unscale_highs_outcome", None)
    if facts is not None:
        return facts
    return snapshot_highs_outcome(getattr(sol, "highs", None))


@dataclass(frozen=True)
class SolverNote:
    code: str
    fmt: dict[str, Any]
    #: The full facts, for the debug log.
    detail: str


def _num(facts: dict[str, Any], name: str) -> int:
    value = facts.get(name)
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def _max(facts: dict[str, Any], name: str) -> float:
    value = facts.get(name)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else 0.0


def interpret_optimal(facts: dict[str, Any] | None, *, is_mip: bool) -> SolverNote | None:
    """The note on an LP solve that HiGHS called optimal but not cleanly, else ``None``.

    * Some *relative* check fails in the returned solution (HiGHS: "Model status is Optimal since relative
      violation of tolerances is no more than X") → :data:`RELATIVE_TOLERANCES`, with X (the largest relative
      violation over its tolerance).
    * Otherwise, after an interior-point solve without crossover, when an absolute check fails (the relative
      ones hold: the "IPM imprecise ... Model status changed from Unknown to Optimal" case) or when the facts
      cannot tell (HiGHS's residual checks not in its info: it recomputes the info without them after removing a
      user scaling, ``user_objective_scale`` / ``user_bound_scale``, which polar-high's autoscale sets) →
      :data:`IPM_RELATIVE_ONLY`.  A solve whose facts are complete and clean gets no note.
    """
    if not facts or is_mip or facts.get("model_status") != "kOptimal":
        return None
    failing: list[tuple[float, str]] = []
    for plural, singular, tolerance, words in _CHECKS:
        num = _num(facts, f"num_relative_{plural}")
        if num:
            top = _max(facts, f"max_relative_{singular}")
            tol = _max(facts, tolerance) or 1e-7
            failing.append((top / tol, f"{num} relative {words} up to {top:.3g} (tolerance {tol:g})"))
    if failing:
        factor = max(f for f, _ in failing)
        return SolverNote(RELATIVE_TOLERANCES, {"factor": f"{factor:.3g}"}, "; ".join(t for _, t in failing))
    ipm_no_crossover = _num(facts, "ipm_iteration_count") > 0 and _num(facts, "crossover_iteration_count") == 0
    if not ipm_no_crossover:
        return None
    absolute = [
        f"{_num(facts, f'num_{plural}')} {words} up to {_max(facts, f'max_{singular}'):.3g}"
        for plural, singular, _tolerance, words in _CHECKS
        if _num(facts, f"num_{plural}")
    ]
    if absolute:
        return SolverNote(IPM_RELATIVE_ONLY, {}, "; ".join(absolute) + " (absolute; the relative ones hold)")
    unknown = [words for plural, _singular, _tolerance, words in _CHECKS
               if not isinstance(facts.get(f"num_{plural}"), (int, float)) or facts[f"num_{plural}"] < 0]
    if unknown:
        return SolverNote(IPM_RELATIVE_ONLY, {}, f"HiGHS's info has no {' / '.join(unknown)} (user-scaled solve)")
    return None


#: Codes noted once per run (not once per solve): the generic note on HiGHS's IPX warnings.
_ONCE_PER_RUN = frozenset({IPM_RELATIVE_ONLY})


def note_solver_outcome(sol: Any, acceptance: Any, *, solve_name: str, solver: str, noted: set) -> None:
    """Print the note on *sol*'s accepted solve (see the module docstring), once per code and complete solve
    (:data:`IPM_RELATIVE_ONLY`, the generic one: once per run).

    *noted* holds the ``(code, solve)`` pairs already printed in this run (``solve`` ``None`` for a once-per-run
    code); a later roll of the same solve with the same outcome only goes to the debug log.  Prints at once
    (flushes an open ``collecting()`` scope), so the note stands next to the solver's output it explains.
    """
    from flextool import run_warnings

    if solver != "highs" or not getattr(acceptance, "accepted", False):
        return
    if getattr(acceptance, "near_optimal", False):
        found: SolverNote | None = SolverNote(NEAR_OPTIMAL, {}, str(getattr(acceptance, "message", "")))
    elif getattr(acceptance, "model_status", None) == "kOptimal":
        from flextool.engine.solve._acceptance import _is_mip

        facts = outcome_facts(sol)
        found = interpret_optimal(facts, is_mip=_is_mip(sol))
    else:
        return
    if found is None:
        return
    complete = run_warnings.complete_solve(solve_name) or solve_name
    key = (found.code, None if found.code in _ONCE_PER_RUN else complete)
    if key in noted:
        _LOGGER.debug("%s also in '%s': %s", found.code, solve_name, found.detail)
        return
    noted.add(key)
    run_warnings.note(found.code, solve=complete, details=f"in '{solve_name}': {found.detail}", **found.fmt)
    run_warnings.flush()


__all__ = [
    "IPM_RELATIVE_ONLY", "NEAR_OPTIMAL", "RELATIVE_TOLERANCES", "SolverNote", "interpret_optimal",
    "note_solver_outcome", "outcome_facts", "snapshot_highs_outcome",
]
