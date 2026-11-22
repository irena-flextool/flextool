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

W-0540 carries HiGHS's own factor X (its ``lpKktCheck``): from HiGHS's info through
``polar_high.tolerance_violation`` on an unscaled solve, and from ``polar_high.kkt_report`` (the check redone in
the space HiGHS solved) on a user-scaled one, whose info HiGHS recomputes in the unscaled space without the
residuals.  Without a report (HiGHS other than 1.15.x, an exception) a user-scaled solve gets no W-0540.  A
``save_memory`` solve (and every cold HiGHS solve) runs in a child process: its ``Solution.highs`` is a shim
without ``getInfo``, so no snapshot and no report.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from collections.abc import Collection
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
    "primal_residual_tolerance", "dual_residual_tolerance", "optimality_tolerance", "kkt_tolerance",
    "user_bound_scale", "user_objective_scale", "run_crossover", "solver",
)
#: The 16 KKT counters and the P-D error: replaced by ``kkt_report``'s on a user-scaled solve.
_KKT_FIELDS = _INFO_FIELDS[6:]
#: HiGHS's default ``kkt_tolerance``: another value replaces all five tolerances (``HighsSolution.cpp:83-89``).
_DEFAULT_KKT_TOLERANCE = 1e-7

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
    if facts["model_status"] == "kOptimal":
        _add_kkt_facts(h, facts)
    return facts


def _add_kkt_facts(h: Any, facts: dict[str, Any]) -> None:
    """Add where HiGHS's factor X comes from (``kkt_source``) to the facts of a kOptimal solve.

    Unscaled: ``"highs-info"``, X from the info counters (:func:`interpret_optimal`).  User-scaled
    (``user_bound_scale`` / ``user_objective_scale`` set): ``polar_high.kkt_report(h)`` on the intact handle; its
    counters replace the unscaled ones of the info, and X, its terms, the basis and the tolerances are kept
    (``"polar-high"``).  No report: ``kkt_source`` ``None`` and ``kkt_unavailable`` (the reason).
    """
    if not (facts.get("user_bound_scale") or facts.get("user_objective_scale")):
        facts["kkt_source"] = "highs-info"
        facts["basis_valid"] = facts.get("basis_validity") == 1
        return
    facts["kkt_source"] = None
    try:
        import polar_high

        rep = polar_high.kkt_report(h)
        if rep is None:
            facts["kkt_unavailable"] = f"no KKT report (HiGHS {h.version()})"
            return
        for name in _KKT_FIELDS:
            facts[name] = getattr(rep, name)
        facts.update(kkt_x=float(rep.max_tolerance_relative_violation), kkt_terms=tuple(rep.terms),
                     basis_valid=bool(rep.basis_valid), kkt_tolerances=dataclasses.asdict(rep.tolerances),
                     kkt_source="polar-high")
    except Exception as exc:  # the snapshot never fails
        _LOGGER.debug("KKT report failed: %s", exc)
        facts["kkt_unavailable"] = f"KKT report failed: {exc}"


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
    #: HiGHS's factor X of a :data:`RELATIVE_TOLERANCES` note (the worst roll's goes on the line).
    value: float | None = None


def _num(facts: dict[str, Any], name: str) -> int:
    value = facts.get(name)
    return int(value) if isinstance(value, (int, float)) and value > 0 else 0


def _tolerances(facts: dict[str, Any]) -> Any:
    """The five tolerances of HiGHS's check, from the options in *facts* (``kkt_tolerance`` other than its default
    replaces all five)."""
    import polar_high

    kkt = facts.get("kkt_tolerance")
    if isinstance(kkt, (int, float)) and kkt != _DEFAULT_KKT_TOLERANCE:
        return polar_high.KktTolerances(*(float(kkt),) * 5)
    names = ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "primal_residual_tolerance",
             "dual_residual_tolerance", "optimality_tolerance")
    return polar_high.KktTolerances(*(float(facts.get(n) or _DEFAULT_KKT_TOLERANCE) for n in names))


def _violation(facts: dict[str, Any]) -> tuple[float, tuple[tuple[str, float], ...], bool] | None:
    """HiGHS's factor X of the solve, its terms (name, measure / tolerance) and whether a basis judged it; ``None``
    when the facts cannot tell (a user-scaled solve without a KKT report)."""
    source = facts.get("kkt_source")
    if source == "polar-high" and isinstance(facts.get("kkt_x"), (int, float)):
        return float(facts["kkt_x"]), tuple(facts.get("kkt_terms") or ()), bool(facts.get("basis_valid"))
    if source != "highs-info":
        _LOGGER.debug("user-scaled solve; %s", facts.get("kkt_unavailable") or "no KKT report")
        return None
    import polar_high

    basis = facts.get("basis_validity") == 1
    try:
        x, terms = polar_high.tolerance_violation(facts, basis_valid=basis, tolerances=_tolerances(facts))
    except (KeyError, ValueError) as exc:
        _LOGGER.debug("no tolerance violation factor: %s", exc)
        return None
    return x, terms, basis


def interpret_optimal(facts: dict[str, Any] | None, *, is_mip: bool) -> list[SolverNote]:
    """The notes on an LP solve that HiGHS called optimal, in print order (empty when there is none).

    * After an interior-point solve without crossover, always → :data:`IPM_RELATIVE_ONLY` (printed once per run:
      it explains HiGHS's IPX warnings, whatever they were).
    * HiGHS's factor X of its KKT check is above 1 (HiGHS: "Model status is Optimal since relative violation of
      tolerances is no more than X") → :data:`RELATIVE_TOLERANCES` with X as HiGHS prints it (``%.3g``).  With a
      basis X takes the absolute infeasibilities and the P-D objective error, without one the relative measures.
      A user-scaled solve without a KKT report has no X and gets no W-0540.
    """
    if not facts or is_mip or facts.get("model_status") != "kOptimal":
        return []
    notes: list[SolverNote] = []
    if _num(facts, "ipm_iteration_count") > 0 and _num(facts, "crossover_iteration_count") == 0:
        notes.append(SolverNote(IPM_RELATIVE_ONLY, {}, "interior point without crossover"
                                + ("" if facts.get("kkt_source") else " (user-scaled; no KKT report)")))
    found = _violation(facts)
    if found is not None and found[0] > 1:
        x, terms, basis = found
        over = [f"{name.removeprefix('max_').replace('_', ' ')} {facts.get(name, float('nan')):.3g} = "
                f"{ratio:.3g}x its tolerance" for name, ratio in terms if ratio > 1]
        kind = "basis: absolute infeasibilities and P-D error" if basis else "no basis: relative measures"
        notes.append(SolverNote(RELATIVE_TOLERANCES, {"factor": f"{x:.3g}", "rolls": ""},
                                f"{'; '.join(over)} ({kind})", value=float(x)))
    return notes


#: Codes noted once per run (not once per solve): the generic note on HiGHS's IPX warnings.
_ONCE_PER_RUN = frozenset({IPM_RELATIVE_ONLY})


def note_solver_outcome(sol: Any, acceptance: Any, *, solve_name: str, solver: str, noted: set,
                        pending: dict | None = None) -> None:
    """Print the notes on *sol*'s accepted solve (see the module docstring), once per code and complete solve
    (:data:`IPM_RELATIVE_ONLY`, the generic one: once per run, printed before W-0540 as it explains the IPX
    warnings above both).

    *noted* holds the ``(code, solve)`` pairs already printed in this run (``solve`` ``None`` for a once-per-run
    code); a later case of the same solve with the same outcome only goes to the debug log.  Prints at once
    (flushes an open ``collecting()`` scope), so the note stands next to the solver's output it explains.

    A roll's :data:`RELATIVE_TOLERANCES` with *pending* (a dict the caller keeps for the run) is gathered instead:
    :func:`flush_solver_outcome_notes` prints one line per complete solve, with the number of its rolls that had
    the outcome and the largest factor among them.
    """
    from flextool import run_warnings

    if solver != "highs" or not getattr(acceptance, "accepted", False):
        return
    if getattr(acceptance, "near_optimal", False):
        found = [SolverNote(NEAR_OPTIMAL, {}, str(getattr(acceptance, "message", "")))]
    elif getattr(acceptance, "model_status", None) == "kOptimal":
        from flextool.engine.solve._acceptance import _is_mip

        found = interpret_optimal(outcome_facts(sol), is_mip=_is_mip(sol))
    else:
        return
    complete = run_warnings.complete_solve(solve_name) or solve_name
    for one in found:
        key = (one.code, None if one.code in _ONCE_PER_RUN else complete)
        if key in noted:
            _LOGGER.debug("%s also in '%s': %s", one.code, solve_name, one.detail)
            continue
        if (one.code == RELATIVE_TOLERANCES and pending is not None and solve_name != complete
                and one.value is not None):
            pending.setdefault(complete, {})[solve_name] = (one.value, one.detail)
            continue
        noted.add(key)
        run_warnings.note(one.code, solve=complete, details=f"in '{solve_name}': {one.detail}", **one.fmt)
        run_warnings.flush()


def flush_solver_outcome_notes(pending: dict, noted: set, *, keep: Collection[str] = ()) -> None:
    """Print the gathered :data:`RELATIVE_TOLERANCES` notes of the rolls in *pending* (``{complete solve: {roll:
    (X, detail)}}``, :func:`note_solver_outcome`): one line per complete solve, ``... at up to <largest X>x ... in
    <n> rolls``, the worst roll first in the debug log.  *keep*: the complete solves whose rolls are still running
    (left in *pending*): the one running now and those it is nested in.  A solve already noted in this run (a later instance of a nested solve) only goes to the
    debug log."""
    from flextool import run_warnings

    for complete in [c for c in pending if c not in keep]:
        rolls = pending.pop(complete)
        if not rolls:
            continue
        ordered = sorted(rolls.items(), key=lambda item: -item[1][0])
        details = "; ".join(f"in '{roll}': {detail}" for roll, (_x, detail) in ordered)
        key = (RELATIVE_TOLERANCES, complete)
        if key in noted:
            _LOGGER.debug("%s also in '%s': %s", RELATIVE_TOLERANCES, complete, details)
            continue
        noted.add(key)
        worst = ordered[0][1][0]
        count = len(ordered)
        run_warnings.note(RELATIVE_TOLERANCES, solve=complete, details=details, factor=f"{worst:.3g}",
                          rolls=f" in {count} roll{'s' if count != 1 else ''}")
        run_warnings.flush()


__all__ = [
    "IPM_RELATIVE_ONLY", "NEAR_OPTIMAL", "RELATIVE_TOLERANCES", "SolverNote", "flush_solver_outcome_notes",
    "interpret_optimal", "note_solver_outcome", "outcome_facts", "snapshot_highs_outcome",
]
