"""The scaling range guard: judge the LP a solver will receive, before the solve.

After Layer 2 (FlexTool's per-quantity rescale) and Layer 3 (HiGHS'
``user_bound_scale`` / ``user_objective_scale``), polar-high's
:func:`~polar_high.autoscale.assess_scaling` judges the bound / RHS, cost and
matrix magnitudes against the solver's tolerance window.  This module holds
FlexTool's policy around that verdict:

* **bound / RHS** — an error when no power-of-two ``user_bound_scale`` puts
  every meaningful limit inside ``[10·tol, 1e18·tol]`` (17 decades: the
  smallest limit is then enforced to 10 %, the largest stays in the range
  where solves were seen to work), a warning when the limits span more than
  13 decades (:data:`WARN_DECADES`).
* **cost** — a warning only (spread above 13 decades, or the smallest scaled
  cost below 10 × the dual tolerance).
* **matrix** — a warning only, and only for coefficient terms HiGHS drops
  (``<= 1e-9``) or refuses (``>= 1e15``); the spread is reported only.
* **commercial solvers** (Gurobi, CPLEX, Xpress, COPT) get no FlexTool
  ``user_*_scale``, so their LP is judged at ``N = 0`` with their own
  feasibility tolerance and every error becomes a warning.

``--scaling-guard error|warn|off`` (``FLEXTOOL_SCALING_GUARD``, default
``error``) decides what an error does: :class:`ScalingRangeError` (a
:class:`FlexToolConfigError`, printed by the CLI without a traceback), a
printed warning, or nothing (no verdict).  ``FLEXTOOL_AUTOSCALE_STRICT=1``
raises even under ``warn`` / ``off`` when Layer 2 turned a model that fitted
the window into one that does not.

The first solve of a shape is judged from the Layer 3 readout
(:func:`run_guard`), with the entries at each end named by
:func:`polar_high.autoscale.locate_extremes` (warning / error only).  Later
rolls that are built again (cache hits) are rechecked from the ranges of the
LP actually built (:class:`RollRecheck`, a polar-high ``pre_run_check``):
when the cached exponent leaves the window and another one fits, that roll
gets it; when none fits, the verdict follows the policy.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from polar_high.autoscale import (
    Finding,
    Layer3Plan,
    RangeReport,
    ScalingConfig,
    ScalingVerdict,
    assess_scaling,
    has_explicit_option,
    locate_extremes,
    recommend_scaling,
    resolve_feasibility_tolerances,
)

if TYPE_CHECKING:
    from flextool.run_options import RunOptions

_logger = logging.getLogger(__name__)

#: ``--scaling-guard`` values.
GUARD_MODES: tuple[str, ...] = ("error", "warn", "off")
DEFAULT_GUARD_MODE = "error"
#: A bound / RHS (and cost) spread above this many decades warns.
WARN_DECADES = 13.0
#: Entries named at each end of a warning / error.
LOCATE_K = 5

#: The solvers that do not receive FlexTool's ``user_*_scale``: their
#: findings warn only.
COMMERCIAL_SOLVERS: tuple[str, ...] = ("gurobi", "cplex", "xpress", "copt")
#: Each commercial solver's primal / dual feasibility parameter (native
#: names, compared as the solver reads them) and its default.
COMMERCIAL_TOLERANCE_PARAMETERS: dict[str, tuple[str, str]] = {
    "gurobi": ("FeasibilityTol", "OptimalityTol"),
    "cplex": ("simplex tolerances feasibility", "simplex tolerances optimality"),
    "xpress": ("FEASTOL", "OPTIMALITYTOL"),
    "copt": ("FeasTol", "DualTol"),
}
COMMERCIAL_DEFAULT_TOLERANCE = 1e-6

#: The override hints name the terminal flag, its environment variable and
#: the GUI setting (Solver options → Scaling guard), so the same text serves
#: every way of starting a run.
_OVERRIDE_ERROR = ("To solve anyway: --scaling-guard warn (FLEXTOOL_SCALING_GUARD=warn; in the GUI: "
                   "Solver options, Scaling guard: Warn).")
_OVERRIDE_WARN = ("The solve continues; --scaling-guard off (FLEXTOOL_SCALING_GUARD=off; in the GUI: "
                  "Solver options, Scaling guard: Off) hides this check.")
#: A pre-Layer-2 bound / RHS spread above this many decades does not fit the
#: tolerance window (17 decades), so ``--scaling basic`` (no Layer 2) stops.
_BASIC_STOP_DECADES = 17.0


def resolve_guard_mode(run_options: "RunOptions | None") -> str:
    """``--scaling-guard`` > ``FLEXTOOL_SCALING_GUARD`` > ``error``."""
    from flextool.run_options import RunOptions

    opts = run_options if run_options is not None else RunOptions.from_env()
    mode = opts.value("scaling_guard") or DEFAULT_GUARD_MODE
    return str(mode)


def _strict() -> bool:
    return os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1"


# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------


def highs_tolerances(problem: Any) -> tuple[float, float, str]:
    """``(primal, dual, source)``: the feasibility tolerances the simplex
    enforces (``kkt_tolerance`` does not replace them here) as set on
    ``problem`` (highs.opt, the solve's solver_arguments, the flags), else
    HiGHS' default 1e-7."""
    tol_p, tol_d = resolve_feasibility_tolerances(problem, kkt_override=False)
    source = ("solver options" if has_explicit_option(problem, "primal_feasibility_tolerance")
              else "HiGHS default")
    return float(tol_p), float(tol_d), source


def commercial_tolerances(
    solver: str, options: Sequence[Any] | None,
) -> tuple[float, float, str]:
    """``(primal, dual, source)`` of a commercial solve: the solver's
    feasibility / optimality parameter in the resolved parameter list
    (``CommercialOption`` with ``name`` / ``value`` / ``layer``), else
    :data:`COMMERCIAL_DEFAULT_TOLERANCE`."""
    from flextool.engine.solve._solver_catalogue import commercial_option_key

    names = COMMERCIAL_TOLERANCE_PARAMETERS.get(solver)
    found: dict[str, tuple[float, str]] = {}
    if names is not None:
        keys = {commercial_option_key(solver, n): which for which, n in zip(("p", "d"), names)}
        for opt in options or ():
            which = keys.get(commercial_option_key(solver, opt.name))
            if which is None:
                continue
            try:
                found[which] = (float(opt.value), str(opt.layer))
            except (TypeError, ValueError):
                continue
    tol_p, src_p = found.get("p", (COMMERCIAL_DEFAULT_TOLERANCE, f"{solver} default"))
    tol_d = found.get("d", (COMMERCIAL_DEFAULT_TOLERANCE, ""))[0]
    return tol_p, tol_d, src_p


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GuardMessage:
    """One printed finding: its group (``bound_rhs`` / ``cost`` / ``matrix``),
    the level after the policy (``warn`` / ``error``) and the text."""

    group: str
    level: str
    text: str
    #: The registry code of the console line (:func:`_code_of`).
    code: str = ""


@dataclass(frozen=True)
class GuardOutcome:
    """What the guard concluded for one solve (region, roll)."""

    mode: str
    solver: str
    #: ``ok`` / ``warn`` / ``error`` after the policy (a commercial error is
    #: ``warn``), ``off`` when the check did not run.
    level: str
    verdict: ScalingVerdict | None
    tol_source: str = ""
    messages: tuple[GuardMessage, ...] = ()
    #: ``group -> side -> entries`` from :func:`locate_extremes`.
    located: Mapping[str, Mapping[str, list]] = field(default_factory=dict)
    #: ``FLEXTOOL_AUTOSCALE_STRICT=1`` and Layer 2 caused a bound / RHS error.
    strict_raise: bool = False
    region: str | None = None
    #: The solve (or roll) the outcome describes.
    solve_name: str = ""
    #: The Layer 3 exponent a roll recheck set (``None``: unchanged).
    adjusted_bound_scale: int | None = None


def _off(mode: str, solver: str, solve_name: str, region: str | None) -> GuardOutcome:
    return GuardOutcome(mode=mode, solver=solver, level="off", verdict=None,
                        region=region, solve_name=solve_name)


def _effective_level(finding: Finding, commercial: bool) -> str:
    if finding.level == "error" and commercial:
        return "warn"
    return finding.level


def run_guard(
    problem: Any,
    *,
    ranges: RangeReport,
    ranges_pre: RangeReport | None,
    plan: Layer3Plan | None,
    solve_name: str,
    mode: str,
    solver: str = "highs",
    commercial_options: Sequence[Any] | None = None,
    region: str | None = None,
    locate: bool = True,
    layer2_applied: bool = False,
) -> GuardOutcome:
    """Judge ``ranges`` (the post-Layer-2 readout of the LP the solver gets)
    under ``plan`` and build the messages.  Pure apart from the
    :func:`locate_extremes` walks (warning / error only, ``locate``).
    ``mode`` ``off`` skips the verdict unless ``FLEXTOOL_AUTOSCALE_STRICT=1``
    (then only a Layer-2-caused error is acted on)."""
    if mode == "off" and not _strict():
        return _off(mode, solver, solve_name, region)
    commercial = solver in COMMERCIAL_SOLVERS
    if commercial:
        tol_p, tol_d, tol_source = commercial_tolerances(solver, commercial_options)
    else:
        tol_p, tol_d, tol_source = highs_tolerances(problem)
    if plan is None:
        plan = Layer3Plan(user_objective_scale=0, user_bound_scale=0, simplex_scale_strategy=2,
                          reasoning="no Layer 3 plan", bound_source="unscaled")
    verdict = assess_scaling(
        ranges, plan, primal_tol=tol_p, dual_tol=tol_d, warn_decades=WARN_DECADES,
        apply_user_scale=not commercial, ranges_pre=ranges_pre,
    )
    return _outcome(problem, verdict, plan=plan, mode=mode, solver=solver, tol_source=tol_source,
                    solve_name=solve_name, region=region, locate=locate,
                    layer2_applied=layer2_applied)


def _outcome(
    problem: Any,
    verdict: ScalingVerdict,
    *,
    plan: Layer3Plan,
    mode: str,
    solver: str,
    tol_source: str,
    solve_name: str,
    region: str | None,
    locate: bool,
    layer2_applied: bool,
    roll_of: str | None = None,
    adjusted_from: int | None = None,
) -> GuardOutcome:
    commercial = solver in COMMERCIAL_SOLVERS
    bound = verdict.finding("bound_rhs")
    strict_raise = bool(
        _strict() and bound is not None and bound.level == "error" and bound.caused_by_prescale
    )
    if mode == "off" and not strict_raise:
        return _off(mode, solver, solve_name, region)
    located: dict[str, dict[str, list]] = {}
    if locate:
        for f in verdict.findings:
            if f.group in ("bound_rhs", "matrix") and f.level in ("warn", "error"):
                located[f.group] = _locate(problem, f)
    messages: list[GuardMessage] = []
    level = "ok"
    for f in verdict.findings:
        if f.group == "info" or f.level not in ("warn", "error"):
            continue
        eff = _effective_level(f, commercial)
        text = format_finding(
            f, verdict, plan=plan, solve_name=solve_name, solver=solver, region=region,
            located=located.get(f.group, {}), level=eff, tol_source=tol_source,
            layer2_applied=layer2_applied, roll_of=roll_of, adjusted_from=adjusted_from,
        )
        messages.append(GuardMessage(f.group, eff, text, _code_of(f.group, f.level)))
        if eff == "error" or (eff == "warn" and level == "ok"):
            level = eff
    return GuardOutcome(
        mode=mode, solver=solver, level=level, verdict=verdict, tol_source=tol_source,
        messages=tuple(messages), located=located, strict_raise=strict_raise, region=region,
        solve_name=solve_name,
    )


def _locate(problem: Any, finding: Finding) -> dict[str, list]:
    """The :data:`LOCATE_K` entries at each end of ``finding`` (``{}`` when the
    problem no longer has its families, e.g. after a release)."""
    out: dict[str, list] = {}
    if getattr(problem, "_released", False) or not hasattr(problem, "_cstrs"):
        return out
    for side, ext in (finding.extremes or {}).items():
        if ext is None or not ext.family or side not in ("min", "max"):
            continue
        group = ext.component
        try:
            out[side] = locate_extremes(problem, ext.family, group, side, k=LOCATE_K)
        except Exception:  # noqa: BLE001 — naming is a courtesy; the verdict stands
            if _strict():
                raise
            _logger.exception("scaling guard: could not name the %s entries of %s", side, ext.family)
    return out


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def _g(x: float) -> str:
    return f"{x:.3g}"


def _dims(entry: Mapping[str, Any]) -> str:
    dims = entry.get("dims") or {}
    return ", ".join(f"{k}={v}" for k, v in dims.items())


def _end_line(label: str, side: str, finding: Finding, located: Mapping[str, list],
              hint: str = "") -> str | None:
    """``"  Largest:  1e+18 (model units) in <family> (<kind>) at <dims>"``.
    When more than one value of the family lies within 10x of the end, the
    count follows with one ``dims`` line for each of those named (at most
    :data:`LOCATE_K`; entries farther than 10x are not listed).  ``hint`` (a
    sentence) closes the single line, or follows the list on its own line."""
    ext = (finding.extremes or {}).get(side)
    entries = located.get(side) or []
    if ext is None and not entries:
        return None
    family = (entries[0]["family"] if entries else ext.family) or "an unnamed family"
    component = (entries[0]["component"] if entries else ext.component) or ""
    kind = {"bound": "variable bound", "rhs": "constraint right-hand side",
            "matrix": "coefficient"}.get(component, component)
    value = entries[0]["model"] if entries else (ext.model if ext is not None else math.nan)
    if not (isinstance(value, float) and math.isfinite(value)) and ext is not None:
        value = ext.post
    head = f"  {label} {_g(abs(value))} (model units) in {family} ({kind})"
    if not entries:
        return f"{head} — {hint}" if hint else head
    count = int(entries[0].get("count_within_factor") or 0)
    if count <= 1:
        single = f"{head} at {_where(entries[0])}"
        return f"{single} — {hint}" if hint else single
    factor = entries[0].get("factor", 10.0)
    lines = [f"{head}, {count} values within {factor:g}x of it, e.g. at:"]
    lines.extend(f"      {_where(e)}" for e in entries[:count])
    if hint:
        lines.append(f"      {hint[:1].upper()}{hint[1:]}")
    return "\n".join(lines)


def _where(entry: Mapping[str, Any]) -> str:
    """The dimensions of one located entry (and of its column, for a
    coefficient)."""
    where = _dims(entry) or "(no dimensions)"
    column = entry.get("column")
    if column:
        cdims = ", ".join(f"{k}={v}" for k, v in (column.get("dims") or {}).items())
        where += f"  (column {column.get('family')}: {cdims})"
    return where


def format_finding(
    finding: Finding,
    verdict: ScalingVerdict,
    *,
    plan: Layer3Plan,
    solve_name: str,
    solver: str,
    region: str | None,
    located: Mapping[str, list],
    level: str,
    tol_source: str,
    layer2_applied: bool,
    roll_of: str | None = None,
    adjusted_from: int | None = None,
) -> str:
    """The printed text of one warning / error finding."""
    commercial = solver in COMMERCIAL_SOLVERS
    who = f"Solve {solve_name}"
    if roll_of:
        who += f" (a roll of {roll_of})"
    if region:
        who += f", Benders region {region}"
    if commercial:
        who += f" ({solver})"
    n = finding.numbers
    tol = verdict.primal_tol
    tol_text = f"primal feasibility tolerance {tol:g}" + (f", {tol_source}" if tol_source else "")
    override = _OVERRIDE_ERROR if level == "error" else _OVERRIDE_WARN
    if finding.group == "bound_rhs":
        return _format_bound_rhs(finding, verdict, plan, who, n, tol, tol_text, located, level,
                                 commercial, solver, layer2_applied, override, adjusted_from)
    if finding.group == "cost":
        lines = [f"{who}: the objective costs span {n.get('decades', 0.0):.1f} decades after scaling "
                 f"({_g(n['s_lo'])} … {_g(n['s_hi'])})."]
        if "small_cost" in finding.kind:
            lines.append(f"  The smallest scaled cost is below {_g(n['small_edge'])} (10 x the dual "
                         f"feasibility tolerance {verdict.dual_tol:g}); the solver may treat it as zero.")
        if "spread" in finding.kind:
            lines.append(f"  More than {WARN_DECADES:g} decades: the cheapest choices may be decided by "
                         f"rounding.")
        for side, label in (("max", "Largest:"), ("min", "Smallest:")):
            ext = (finding.extremes or {}).get(side)
            if ext is not None and ext.family:
                lines.append(f"  {label} in the objective term {ext.family}")
        lines.append("  " + _OVERRIDE_WARN)
        return "\n".join(lines)
    # matrix
    lines = []
    if "refused" in finding.kind:
        lines.append(f"{who}: a coefficient term of {_g(n['s_hi'])} is at or above "
                     f"{_g(n['large_matrix'])}; HiGHS refuses such a model.")
        end = _end_line("Largest:", "max", finding, located)
        if end:
            lines.append(end)
    if "dropped" in finding.kind:
        lines.append(("  " if lines else f"{who}: ") +
                     f"coefficient terms of {_g(n['small_matrix'])} or less (smallest "
                     f"{_g(n['s_lo'])}); HiGHS drops them from the model.")
        end = _end_line("Smallest:", "min", finding, located)
        if end:
            lines.append(end)
    lines.append(f"  The coefficients span {n.get('decades', 0.0):.1f} decades; HiGHS equilibrates the "
                 f"matrix itself.")
    lines.append("  " + _OVERRIDE_WARN)
    return "\n".join(lines)


def _format_bound_rhs(
    f: Finding, verdict: ScalingVerdict, plan: Layer3Plan, who: str, n: Mapping[str, float],
    tol: float, tol_text: str, located: Mapping[str, list], level: str, commercial: bool,
    solver: str, layer2_applied: bool, override: str, adjusted_from: int | None,
) -> str:
    decades = n.get("decades", 0.0)
    span = f"({_g(n['b_lo'])} … {_g(n['b_hi'])})"
    after = "as the solver gets them" if commercial else "after FlexTool's scaling"
    fits = bool(n.get("fits"))
    width = math.log10(verdict.window[1] / verdict.window[0])
    if f.level == "error" and not fits:
        lines = [f"{who} cannot be scaled into a range the solver handles accurately.",
                 f"  Limits and bounds span {decades:.1f} decades {after} {span}; at most "
                 f"{width:.0f} fit ({tol_text})."]
        if f.kind.startswith("scale_option_limit") or "scale_option_limit" in f.kind:
            lines[0] = (f"{who}: its limits and bounds are too large (or too small) for the "
                        f"solver's scale option (user_bound_scale in [-30, 30]).")
            lines[1] = (f"  They span {decades:.1f} decades {after} {span}, which would fit, but only "
                        f"at a scale outside that range ({tol_text}).")
    elif f.level == "error":
        n_used = int(n.get("n_used", 0))
        source = {"manual": "--user-bound-scale / FLEXTOOL_USER_BOUND_SCALE",
                  "external": "user_bound_scale in the solver options",
                  "unscaled": f"{solver} gets no FlexTool scaling"}.get(plan.bound_source if not commercial
                                                                        else "unscaled", "")
        lines = [f"{who}: user_bound_scale {n_used} ({source}) puts the limits at "
                 f"{_g(n['s_lo'])} … {_g(n['s_hi'])}, outside the range the solver handles accurately "
                 f"({_g(n['window_lo'])} … {_g(n['window_hi'])}, {tol_text}).",
                 f"  Limits and bounds span {decades:.1f} decades {after} {span}; an exponent in "
                 f"[{int(n['n_min'])}, {int(n['n_max'])}] fits."]
    else:
        lines = [f"{who}: limits and bounds span {decades:.1f} decades {after} {span}, more than "
                 f"{WARN_DECADES:g} ({tol_text})."]
    ext = (f.extremes or {}).get("max")
    model = located.get("max", [{}])[0].get("model") if located.get("max") else (
        ext.model if ext is not None else math.nan)
    hint = ('a stand-in for "no limit"? Use inf.'
            if isinstance(model, float) and math.isfinite(model) and model >= 1e10 else "")
    big = _end_line("Largest: ", "max", f, located, hint=hint)
    if big:
        lines.append(big)
    small = _end_line("Smallest:", "min", f, located)
    if small:
        lines.append(small)
    if f.level == "error" and not fits:
        lines.append(f"  Fitting the largest value, the smallest limit could be violated by up to "
                     f"{_g(n['fit_large_violation_model'])} (model units), "
                     f"{_g(n['fit_large_violation_ratio'])}x its size.")
        if plan.bound_tag.startswith("unscalable") and not commercial:
            lines.append(f"  FlexTool keeps the smallest limit enforceable (user_bound_scale "
                         f"{int(n.get('n_used', 0))}): the largest values, at {_g(n['s_hi'])} after "
                         f"scaling, are then beyond {_g(n['window_hi'])}.")
    else:
        scale = "" if commercial else f"With user_bound_scale {int(n.get('n_used', 0))}, the"
        lines.append(f"  {scale or 'The'} smallest limit could be violated by up to "
                     f"{_g(n['violation_model'])} (model units); every limit by at most "
                     f"{100 * n['r_lo']:.3g} %.")
    if adjusted_from is not None:
        lines.append(f"  This roll uses user_bound_scale {int(n.get('n_used', 0))} instead of "
                     f"{adjusted_from} (the exponent of the first roll).")
    pre = n.get("decades_pre")
    if layer2_applied and pre is not None and decades > pre + 0.05:
        lines.append(f"  Layer 2 widened the span from {pre:.1f} decades: --scaling basic avoids that.")
    elif (layer2_applied and pre is not None and f.level != "error" and not commercial
          and pre > _BASIC_STOP_DECADES):
        lines.append(f"  Layer 2 narrowed the span from {pre:.1f} decades; under --scaling basic this "
                     f"would stop.")
    if commercial and f.level == "error":
        lines.append(f"  {solver} gets no FlexTool scaling, so this is a warning only.")
        override = _OVERRIDE_WARN
    lines.append("  " + override)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def enforce(outcome: GuardOutcome | None, *, base_solve: str) -> None:
    """Act on ``outcome``: show each warning once per run (a coded line,
    W-0543..W-0546, naming the solve; its report goes to the debug log),
    keyed by base solve, region, group and level (:func:`_warn_key`, so the
    rolls of a solve show it once), raise :class:`ScalingRangeError` for an
    error under ``error`` (or a Layer-2-caused error under
    ``FLEXTOOL_AUTOSCALE_STRICT=1``), show an error as a warning under
    ``warn``."""
    if outcome is None:
        return
    from flextool.engine._behaviour_changes import mark_once
    from flextool.engine.input._solve_state import ScalingRangeError

    errors = [m for m in outcome.messages if m.level == "error"]
    if outcome.strict_raise:
        bound = [m for m in outcome.messages if m.group == "bound_rhs"]
        raise ScalingRangeError("\n".join(m.text for m in bound) + "\n  (raised because "
                                "FLEXTOOL_AUTOSCALE_STRICT=1 and Layer 2 caused it)")
    if errors and outcome.mode == "error":
        raise ScalingRangeError("\n".join(m.text for m in errors))
    if outcome.mode == "off":
        return
    from flextool import run_warnings

    name = outcome.solve_name or base_solve
    if outcome.region:
        name = f"{name}, Benders region {outcome.region}"
    for m in outcome.messages:
        if mark_once(_warn_key(base_solve, outcome.region, m.group, m.level, outcome.adjusted_bound_scale)):
            # The report (the span, where the extremes are, the violation) goes to the debug log.
            run_warnings.warn_grouped(m.code or _code_of(m.group, m.level), [name], details={name: m.text})


def _code_of(group: str, level: str) -> str:
    """The registry code of a guard finding of ``group`` at the guard's ``level`` (before the policy: a commercial
    solver's error is a warning, but still the "cannot be scaled" line)."""
    if group == "bound_rhs":
        return "W-0544" if level == "error" else "W-0543"
    return "W-0545" if group == "cost" else "W-0546"


def _warn_key(base_solve: str, region: str | None, group: str, level: str,
              adjusted: int | None) -> tuple:
    """The once-per-run key of a guard message: a later roll whose finding
    turns into an error, or that gets an exponent of its own (``adjusted``,
    the exponent; ``None`` when the roll kept the cached one), prints
    again."""
    return ("scaling_guard", base_solve, region, group, level, adjusted)


def warn_could_not_run(*, readout_failed: bool) -> bool:
    """Show, once per run, that the guard could not judge a solve (W-0547; the
    monolith and every Benders region show the same line, under one key).
    ``readout_failed``: the Layer 3 range readout was missing; otherwise the
    check itself failed.  Returns True the first time in a run (the caller
    then logs the details)."""
    from flextool import run_warnings
    from flextool.engine._behaviour_changes import mark_once

    if not mark_once(("scaling_guard_failed",)):
        return False
    run_warnings.warn_grouped("W-0547", why=" (the range readout failed)" if readout_failed else "")
    return True


# ---------------------------------------------------------------------------
# Per-roll recheck (polar-high ``pre_run_check``)
# ---------------------------------------------------------------------------


def report_from_hook_ranges(ranges: Mapping[str, Any]) -> RangeReport:
    """A :class:`RangeReport` of the ranges a ``pre_run_check`` receives (no
    extremes: the hook ranges do not carry families)."""
    nan = (math.nan, math.nan)

    def pair(key: str) -> tuple[float, float]:
        v = ranges.get(key)
        return nan if v is None else (float(v[0]), float(v[1]))

    def opt(key: str) -> tuple[float, float] | None:
        v = ranges.get(key)
        return None if v is None else (float(v[0]), float(v[1]))

    cols, rows = opt("col_bound_meaningful"), opt("row_bound_meaningful")
    present = [p for p in (cols, rows) if p is not None]
    meaningful = ((min(p[0] for p in present), max(p[1] for p in present)) if present else None)
    residue = ranges.get("bound_rhs_residue")
    return RangeReport(
        matrix=pair("matrix"), cost=pair("cost"), bound=pair("col_bound"), rhs=pair("row_bound"),
        cross_group_max_ratio=math.nan, trigger=False,
        bound_rhs_meaningful=meaningful,
        bound_rhs_residue=None if residue is None else (int(residue[0]), float(residue[1])),
        bound_rhs_beyond_infinite=ranges.get("bound_rhs_beyond_infinite"),
        integer_bound=opt("integer_bound"),
        matrix_continuous=opt("matrix_continuous"),
        integer_matrix=opt("integer_matrix"),
        cost_continuous=opt("cost_continuous"),
        integer_cost=opt("integer_cost"),
        residue_floor=ranges.get("residue_floor"),
        infinite_bound=ranges.get("infinite_bound"),
    )


def _will_report(verdict: ScalingVerdict, mode: str, base_solve: str, region: str | None,
                 adjusted: int | None = None) -> bool:
    """Whether :func:`enforce` would print or raise something new for
    ``verdict`` (an error under ``error``, or a finding whose warning this run
    has not printed yet, keyed as :func:`enforce` keys it)."""
    from flextool.engine._behaviour_changes import already_warned

    if mode == "off":
        return False
    for f in verdict.findings:
        if f.group == "info" or f.level not in ("warn", "error"):
            continue
        if f.level == "error" and mode == "error":
            return True
        if not already_warned(_warn_key(base_solve, region, f.group, f.level, adjusted)):
            return True
    return False


class RollRecheck:
    """``pre_run_check`` for a roll that replays a cached Layer 3 plan.

    Called by polar-high right before the solver runs (``Problem.solve``,
    the first ``WarmProblem.solve``, ``Problem.write_mps``) with the ranges of
    the LP actually built.  When the cached ``user_bound_scale`` (automatic)
    leaves the tolerance window and another exponent fits, this roll gets
    the exponent Layer 3 picks for its own ranges (logged once per base
    solve); when none fits, or a manual / external exponent leaves the
    window, the verdict follows the policy (``ScalingRangeError`` under
    ``error``, raised out of the solve call before the solver runs).  The
    outcome and the plan the roll used are kept on the instance
    (:attr:`outcome`, :attr:`plan`) for the roll's report."""

    def __init__(
        self,
        problem: Any,
        *,
        plan: Layer3Plan,
        config: ScalingConfig,
        mode: str,
        solve_name: str,
        base_solve: str,
        layer2_applied: bool,
    ) -> None:
        """``solve_name`` is the name the messages use (the orchestrator
        passes ``"<solve>, roll <k>"``); ``base_solve`` keys the
        once-per-run warnings."""
        self.problem = problem
        self.plan = plan
        self.config = config
        self.mode = mode
        self.solve_name = solve_name
        self.base_solve = base_solve
        self.layer2_applied = layer2_applied
        self.outcome: GuardOutcome | None = None
        self.called = False

    def __call__(self, ranges: Mapping[str, Any], highs: Any) -> dict[str, Any] | None:
        from flextool.engine._behaviour_changes import mark_once

        self.called = True
        if self.mode == "off" and not _strict() and self.plan.bound_source != "auto":
            return None
        report = report_from_hook_ranges(ranges)
        tol_p, tol_d, tol_source = highs_tolerances(self.problem)
        plan = self.plan
        verdict = assess_scaling(report, plan, primal_tol=tol_p, dual_tol=tol_d,
                                 warn_decades=WARN_DECADES)
        bound = verdict.finding("bound_rhs")
        adjusted_from = None
        if bound is not None and bound.level == "error" and plan.bound_source == "auto":
            again = recommend_scaling(report, self.config, primal_feasibility_tolerance=tol_p)
            if again.user_bound_scale != plan.user_bound_scale:
                adjusted_from = plan.user_bound_scale
                plan = replace(plan, user_bound_scale=again.user_bound_scale,
                               bound_tag=again.bound_tag, bound_window=again.bound_window,
                               reasoning=f"{plan.reasoning}; roll recheck: user_bound_scale "
                                         f"{adjusted_from} -> {again.user_bound_scale} "
                                         f"({again.bound_tag})")
                verdict = assess_scaling(report, plan, primal_tol=tol_p, dual_tol=tol_d,
                                         warn_decades=WARN_DECADES)
                # Logged once per run and base solve (``start_run`` resets it).
                if mark_once(("scaling_guard_roll", self.base_solve)):
                    _logger.info(
                        "scaling guard [%s]: the cached user_bound_scale %d leaves the tolerance "
                        "window of this roll; using %d (%s)", self.solve_name, adjusted_from,
                        again.user_bound_scale, again.bound_tag,
                    )
        self.plan = plan
        # Name the entries only when something new is printed or raised: a
        # warning the first roll printed is not printed again, and the walk
        # would cost every roll what the shape cache saves.
        name_them = _will_report(verdict, self.mode, self.base_solve, None,
                                 plan.user_bound_scale if adjusted_from is not None else None)
        if name_them:
            report = self._with_extremes(report)
            verdict = assess_scaling(report, plan, primal_tol=tol_p, dual_tol=tol_d,
                                     warn_decades=WARN_DECADES)
        outcome = _outcome(
            self.problem, verdict, plan=plan, mode=self.mode, solver="highs", tol_source=tol_source,
            solve_name=self.solve_name, region=None, locate=name_them,
            layer2_applied=self.layer2_applied,
            adjusted_from=adjusted_from,
        )
        self.outcome = replace(
            outcome,
            adjusted_bound_scale=plan.user_bound_scale if adjusted_from is not None else None,
        )
        enforce(self.outcome, base_solve=self.base_solve)
        if adjusted_from is not None:
            return {"user_bound_scale": int(plan.user_bound_scale)}
        return None

    def _with_extremes(self, report: RangeReport) -> RangeReport:
        """``report`` with the families at each end, from one range walk of
        the built Problem (a warning / error path only)."""
        pb = self.problem
        if getattr(pb, "_released", False):
            return report
        try:
            from polar_high.autoscale import detect_ranges

            full = detect_ranges(pb, self.config)
        except Exception:  # noqa: BLE001 — the verdict stands without names
            if _strict():
                raise
            _logger.exception("scaling guard [%s]: could not read the extremes", self.solve_name)
            return report
        return replace(report, extremes=full.extremes)


__all__ = [
    "COMMERCIAL_DEFAULT_TOLERANCE",
    "COMMERCIAL_SOLVERS",
    "COMMERCIAL_TOLERANCE_PARAMETERS",
    "DEFAULT_GUARD_MODE",
    "GUARD_MODES",
    "GuardMessage",
    "GuardOutcome",
    "LOCATE_K",
    "RollRecheck",
    "WARN_DECADES",
    "commercial_tolerances",
    "enforce",
    "format_finding",
    "highs_tolerances",
    "report_from_hook_ranges",
    "resolve_guard_mode",
    "run_guard",
    "warn_could_not_run",
]
