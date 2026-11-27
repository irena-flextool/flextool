"""The autoscale steps of a sub-solve: the Layer 1 emit, the per-solve report, the scaling guard, the Layer 2 /
Layer 3 pre-solve apply and its shape cache, the console summary and the post-solve unscale."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from flextool.run_options import RunOptions
from flextool.engine.autoscale import (
    GuardOutcome as _AutoscaleGuardOutcome,
    Layer2Plan as _AutoscaleLayer2Plan,
    Layer3Plan as _AutoscaleLayer3Plan,
    RangeReport as _AutoscaleRangeReport,
    RollRecheck as _AutoscaleRollRecheck,
    ScalingMode as _AutoscaleScalingMode,
    apply_layer2 as _autoscale_apply_layer2,
    apply_layer2_with_exponents as _autoscale_apply_layer2_with_exponents,
    apply_scaling as _autoscale_apply_scaling,
    detect_ranges as _autoscale_compute_ranges,
    format_console_summary as _autoscale_format_console_summary,
    mode_enables_layer1 as _autoscale_mode_enables_layer1,
    mode_enables_layer3 as _autoscale_mode_enables_layer3,
    recommend_scaling as _autoscale_recommend_scaling,
    resolve_guard_mode as _autoscale_resolve_guard_mode,
    resolve_scaling_config as _autoscale_resolve_config,
    run_guard as _autoscale_run_guard,
    unscale_solution as _autoscale_unscale_solution,
    write_report as _autoscale_write_report,
)

if TYPE_CHECKING:
    from polar_high import Problem, Solution
    from polar_high.autoscale import ScalingConfig

    from flextool.engine.solve._orchestration._memory import _MemoryRecorder


def _autoscale_emit_layer1(
    sol: "Solution | None",
    *,
    solve_name: str,
    logger: logging.Logger,
    work_folder: str | os.PathLike | None,
    layer2_plan: "_AutoscaleLayer2Plan | None" = None,
    layer3_plan: "_AutoscaleLayer3Plan | None" = None,
    scaling_config: "ScalingConfig | None" = None,
    guard: "_AutoscaleGuardOutcome | None" = None,
) -> None:
    """Layer 1 (detect) post-solve emitter.

    Reads polar-high's already-computed ``Solution.streamed_lp_ranges``
    (no duplicate matrix walk), runs the autoscaler's Layer 1
    detection, and logs the four ranges + trigger flag.  Optionally
    writes a YAML audit report when the config carries a path.

    Phase 1b is detection-only — this function does NOT modify the LP
    or the solve options.  Layer 2 / Layer 3 will hook in alongside
    it in later phases.

    Failures here are non-fatal: a missing ``streamed_lp_ranges`` (the
    subprocess path — polar-high streams ranges during the in-process
    solve, which the subprocess child does not share back) skips the
    layer with a debug-level note rather than breaking the solve.
    """
    # ``scaling_config`` is the sub-solve's resolved config (the run's options over the solve's
    # ``solve.scaling``); ``None`` (a direct call) resolves it from the environment.
    cfg = scaling_config if scaling_config is not None else _autoscale_resolve_config(None)
    if not _autoscale_mode_enables_layer1(cfg.mode):
        return
    if sol is None:
        return
    streamed = getattr(sol, "streamed_lp_ranges", None)
    if not isinstance(streamed, dict):
        logger.debug(
            "autoscale Layer 1 skipped for %s: no streamed_lp_ranges on Solution",
            solve_name,
        )
        return

    try:
        ranges = _autoscale_compute_ranges(sol, cfg)
    except Exception:  # pragma: no cover — guard against future API drift
        logger.exception(
            "autoscale Layer 1 failed for %s; continuing without it", solve_name,
        )
        return

    def _fmt(span: tuple[float, float]) -> str:
        import math as _math
        if _math.isnan(span[0]) or _math.isnan(span[1]):
            return "empty"
        return f"{span[0]:.1e}, {span[1]:.1e}"

    logger.info(
        "autoscale Layer 1 [%s]: Matrix [%s], Cost [%s], Bound [%s], "
        "RHS [%s], cross=%s, trigger=%s",
        solve_name,
        _fmt(ranges.matrix), _fmt(ranges.cost),
        _fmt(ranges.bound), _fmt(ranges.rhs),
        (f"{ranges.cross_group_max_ratio:.1e}"
         if ranges.cross_group_max_ratio == ranges.cross_group_max_ratio
         else "n/a"),
        ranges.trigger,
    )

    # Default report location next to the existing scaling_report file
    # when no explicit path is configured — keeps both diagnostics
    # together for the operator.
    yaml_path = cfg.report_yaml_path
    if yaml_path is None and work_folder is not None:
        yaml_path = Path(work_folder) / "solve_data" / f"autoscale_{solve_name}.yaml"
    if yaml_path is not None:
        try:
            report_tree: dict = {"layer1": ranges}
            if layer2_plan is not None:
                from flextool.engine.autoscale._report import (
                    render_layer2 as _render_l2,
                )
                report_tree["layer2"] = _render_l2(layer2_plan)
            if layer3_plan is not None:
                from flextool.engine.autoscale._report import (
                    render_layer3 as _render_l3,
                )
                report_tree["layer3"] = _render_l3(layer3_plan)
            if guard is not None:
                from flextool.engine.autoscale._report import (
                    render_guard as _render_guard,
                )
                report_tree["guard"] = _render_guard(guard)
            _autoscale_write_report(report_tree, yaml_path)
        except Exception:  # pragma: no cover — non-fatal
            logger.exception(
                "autoscale Layer 1 report write failed (%s)", yaml_path,
            )


def _autoscale_report_path(
    cfg: "ScalingConfig", solve_name: str, work_folder: "str | os.PathLike | None",
) -> "Path | None":
    """Where the per-solve autoscale YAML goes: the configured path, else
    ``<work_folder>/solve_data/autoscale_<solve>.yaml``."""
    if cfg.report_yaml_path is not None:
        return Path(cfg.report_yaml_path)
    if work_folder is None:
        return None
    return Path(work_folder) / "solve_data" / f"autoscale_{solve_name}.yaml"


def _autoscale_write_presolve_report(
    *,
    solve_name: str,
    logger: logging.Logger,
    work_folder: "str | os.PathLike | None",
    ranges: "_AutoscaleRangeReport | None",
    layer2_plan: "_AutoscaleLayer2Plan | None",
    layer3_plan: "_AutoscaleLayer3Plan | None",
    guard: "_AutoscaleGuardOutcome | None",
    scaling_config: "ScalingConfig",
) -> None:
    """Write the autoscale YAML before the solve, at decision time: the
    pre-solve ranges, the Layer 2 / Layer 3 decisions and the scaling-guard
    verdict.  So the file exists also when the solve stops (a guard error, a
    solver failure) or no post-solve readout runs; the post-solve emitter
    (:func:`_autoscale_emit_layer1`) rewrites it with the solved LP's ranges.
    Non-fatal."""
    if not _autoscale_mode_enables_layer1(scaling_config.mode) or ranges is None:
        return
    yaml_path = _autoscale_report_path(scaling_config, solve_name, work_folder)
    if yaml_path is None:
        return
    try:
        from flextool.engine.autoscale._report import (
            render_guard as _render_guard,
            render_layer2 as _render_l2,
            render_layer3 as _render_l3,
        )
        tree: dict = {"layer1": ranges}
        if layer2_plan is not None:
            tree["layer2"] = _render_l2(layer2_plan)
        if layer3_plan is not None:
            tree["layer3"] = _render_l3(layer3_plan)
        if guard is not None:
            tree["guard"] = _render_guard(guard)
        _autoscale_write_report(tree, yaml_path)
    except Exception:  # pragma: no cover — non-fatal
        logger.exception("autoscale pre-solve report write failed (%s)", yaml_path)


def _autoscale_report_recheck_stop(
    hook: "_AutoscaleRollRecheck | None",
    *,
    solve_name: str,
    logger: logging.Logger,
    work_folder: "str | os.PathLike | None",
    ranges: "_AutoscaleRangeReport | None",
    layer2_plan: "_AutoscaleLayer2Plan | None",
    scaling_config: "ScalingConfig",
) -> None:
    """Rewrite the pre-solve autoscale YAML with the verdict of a roll
    recheck that stopped the solve call (``ScalingRangeError`` from the
    ``pre_run_check``, or a later failure in the warm build / ``write_mps``):
    the file written before the call holds the first roll's replayed
    verdict, and no post-solve report follows.  No-op when the hook did not
    reach a verdict."""
    if hook is None or not hook.called or hook.outcome is None:
        return
    _autoscale_write_presolve_report(
        solve_name=solve_name, logger=logger, work_folder=work_folder, ranges=ranges,
        layer2_plan=layer2_plan, layer3_plan=hook.plan, guard=hook.outcome,
        scaling_config=scaling_config,
    )


def _roll_label(complete_solve_name: str, roll_index: "int | str | None") -> str:
    """``"<solve>, roll <k>"`` (1-based) for the scaling guard's messages
    about a roll after the first; the solve name otherwise."""
    if isinstance(roll_index, int) and roll_index > 0:
        return f"{complete_solve_name}, roll {roll_index + 1}"
    return complete_solve_name


def _autoscale_guard_check(
    pb: "Problem",
    *,
    ranges: "_AutoscaleRangeReport | None",
    ranges_pre: "_AutoscaleRangeReport | None",
    layer2_plan: "_AutoscaleLayer2Plan | None",
    layer3_plan: "_AutoscaleLayer3Plan | None",
    solve_name: str,
    run_options: "RunOptions | None",
    scaling_config: "ScalingConfig",
    logger: logging.Logger,
    solver: str = "highs",
    commercial_options: "list | None" = None,
) -> "_AutoscaleGuardOutcome | None":
    """The scaling guard of one solve, from the Layer 3 readout (no extra
    walk; :func:`locate_extremes` only on a warning / error).  ``None`` when
    the autoscaler is off for this solve (``--scaling off|solver_only``: no
    readout, no guard) or the readout is missing.  A failure of the check
    itself is logged once and the solve goes on (re-raised under
    ``FLEXTOOL_AUTOSCALE_STRICT=1``).  The verdict is acted on by
    :func:`_autoscale_enforce_guard`, called outside every ``except``."""
    if not _autoscale_mode_enables_layer3(scaling_config.mode):
        return None
    from flextool.engine.autoscale import warn_guard_could_not_run as _could_not_run
    if ranges is None:
        # The Layer 3 readout failed (logged there; re-raised under strict).
        _could_not_run(readout_failed=True)
        return None
    try:
        return _autoscale_run_guard(
            pb, ranges=ranges, ranges_pre=ranges_pre, plan=layer3_plan,
            solve_name=solve_name, mode=_autoscale_resolve_guard_mode(run_options),
            solver=solver, commercial_options=commercial_options,
            layer2_applied=layer2_plan is not None,
        )
    except Exception:
        if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
            raise
        if _could_not_run(readout_failed=False):
            logger.exception("scaling guard failed for %s", solve_name)
        return None


def _autoscale_apply_layer3_pre_solve(
    pb: "Problem",
    *,
    layer2_plan: "_AutoscaleLayer2Plan | None",
    solve_name: str,
    logger: logging.Logger,
    scaling_config: "ScalingConfig | None" = None,
    ranges_pre: "_AutoscaleRangeReport | None" = None,
    readout: "dict | None" = None,
) -> "_AutoscaleLayer3Plan | None":
    """Layer 3 (HiGHS-native top-up) pre-solve apply.

    Runs unconditionally when the autoscaler is enabled — Layer 3 is
    cheap and sets HiGHS options that take effect only when needed
    (``user_*_scale`` defaults to 0 = no-op).  The recommendation is
    derived from the *post-Layer-2* coefficient ranges so the residual
    spread (after Layer 2's per-type rescale) drives the global
    HiGHS-side scaling.

    Returns ``None`` when the autoscaler is disabled or the readout
    fails; the caller continues without setting any Layer 3 options.

    Layer 2's mutation of the LP arrays is the same the polar-high
    streaming solve sees — Layer 3 just picks exponents from those
    arrays.  Precedence-respect: when the caller (highs.opt file,
    ``set_solver_options`` from elsewhere) has already set
    ``user_bound_scale`` or ``user_objective_scale``, Layer 3 skips that
    axis and the caller's value wins.

    ``ranges_pre`` (the pre-Layer-2 readout) is reused instead of a second
    walk when Layer 2 was not attempted (``layer2_plan is None`` and the
    mode or the trigger skipped it): the LP is then unchanged.  ``readout``,
    when given, receives the report Layer 3 decided on (``"ranges"``), for
    the scaling guard and the console summary.
    """
    cfg = scaling_config if scaling_config is not None else _autoscale_resolve_config(None)
    if not _autoscale_mode_enables_layer3(cfg.mode):
        return None
    l2_attempted = (
        ranges_pre is not None
        and cfg.mode is _AutoscaleScalingMode.FULL
        and ranges_pre.trigger
    )
    if layer2_plan is None and ranges_pre is not None and not l2_attempted:
        ranges_post_l2 = ranges_pre
    else:
        try:
            ranges_post_l2 = _autoscale_compute_ranges(pb, cfg)
        except Exception:  # pragma: no cover — guard against future API drift
            if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
                raise
            logger.exception(
                "autoscale Layer 3 pre-solve range readout failed for %s; "
                "skipping Layer 3 (Layer 1 / Layer 2 still applied)",
                solve_name,
            )
            return None
    if readout is not None:
        readout["ranges"] = ranges_post_l2
    try:
        plan = _autoscale_recommend_scaling(ranges_post_l2, cfg, problem=pb)
    except Exception:  # pragma: no cover
        logger.exception(
            "autoscale Layer 3 recommendation failed for %s; "
            "skipping (HiGHS internal scaling still applies)",
            solve_name,
        )
        return None
    try:
        _autoscale_apply_scaling(pb, plan)
    except Exception:  # pragma: no cover
        logger.exception(
            "autoscale Layer 3 option apply failed for %s; HiGHS "
            "internal scaling will fill in",
            solve_name,
        )
        return plan  # still return for report visibility
    logger.info(
        "autoscale Layer 3 [%s]: user_objective_scale=%d, "
        "user_bound_scale=%d, simplex_scale_strategy=%d (%s)",
        solve_name,
        plan.user_objective_scale,
        plan.user_bound_scale,
        plan.simplex_scale_strategy,
        plan.reasoning,
    )
    return plan


def _autoscale_apply_layer2_pre_solve(
    pb: "Problem",
    *,
    solve_name: str,
    logger: logging.Logger,
    scaling_config: "ScalingConfig | None" = None,
) -> "tuple[_AutoscaleLayer2Plan | None, _AutoscaleRangeReport | None]":
    """Layer 2 (semantic per-type scaling) pre-solve apply.

    Runs only when the autoscaler is enabled AND the pre-solve Layer-1
    detector trips (per ``AutoScaleConfig.threshold_decades``).  When
    triggered, mutates ``pb`` in place and returns the inverse-plan
    that ``_autoscale_unscale_post_solve`` consumes immediately after
    ``pb.solve(...)``.

    Returns ``(plan, ranges_pre)`` where:

    * ``plan`` is ``None`` when Layer 2 was skipped (config off or the
      Layer-1 trigger did not fire).
    * ``ranges_pre`` is the pre-Layer-2 :class:`RangeReport` (always
      present when the autoscaler is enabled and the readout succeeded;
      ``None`` only when disabled or the readout itself failed).  The
      caller threads it into the console summary and the non-optimal
      hint so both reports describe the LP the autoscaler *decided on*
      rather than re-reading after Layer 2 mutated the arrays.

    The detector here uses :func:`_autoscale_compute_ranges` on the
    pre-solve ``Problem`` — :mod:`_ranges` falls back to
    ``Problem._build_lp_arrays`` for that path.  Layer-1 emission
    after solve still happens via the existing post-solve hook so the
    operator-facing log line and YAML report describe the *scaled*
    LP that HiGHS actually saw.
    """
    cfg = scaling_config if scaling_config is not None else _autoscale_resolve_config(None)
    # Layer 2 is FlexTool-side semantic per-quantity scaling: only fires
    # in ``ScalingMode.FULL``.  In ``BASIC`` we still want Layer 1's
    # pre-solve ranges to be available for the console summary, so we
    # compute them when Layer 3 is enabled too.
    if not _autoscale_mode_enables_layer1(cfg.mode):
        return None, None
    try:
        ranges_pre = _autoscale_compute_ranges(pb, cfg)
    except Exception:  # pragma: no cover — guard against future API drift
        if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
            raise
        logger.exception(
            "autoscale Layer 2 pre-solve range readout failed for %s; "
            "skipping Layer 2 (Layer 1 post-solve still fires)",
            solve_name,
        )
        return None, None
    if cfg.mode is not _AutoscaleScalingMode.FULL:
        # BASIC mode: skip Layer 2's LP-array mutation but still report
        # the pre-solve ranges so Layer 3 / the console summary see them.
        return None, ranges_pre
    if not ranges_pre.trigger:
        return None, ranges_pre
    try:
        plan = _autoscale_apply_layer2(pb, cfg)
    except Exception:  # pragma: no cover
        if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
            # Opt-in fail-fast for tests / CI: surface Layer-2 errors
            # (notably autoscale-registry gaps — an unregistered constraint
            # / variable / parameter raising KeyError in
            # bucket_coefficients) loudly instead of silently reverting to
            # an un-scaled LP.  This is the behavioral backstop for dynamic
            # (f-string) constraint names that the static-literal grep in
            # test_registry_coverage cannot see (e.g. ramp_*_constraint).
            # Production runs leave the flag unset and keep degrading
            # gracefully so a registry gap never blocks a user's solve.
            raise
        logger.exception(
            "autoscale Layer 2 apply failed for %s; reverting solve to "
            "un-scaled LP (Layer 1 post-solve still fires)",
            solve_name,
        )
        return None, ranges_pre
    logger.info(
        "autoscale Layer 2 [%s]: exponents=%s, rows=%d, skipped_rows=%d, "
        "integer_cols=%d",
        solve_name,
        {t.value: e for t, e in plan.type_exponents.items()},
        plan.row_factors.shape[0],
        len(plan.skipped_rows),
        len(plan.skipped_integer_cols),
    )
    return plan, ranges_pre


@dataclass
class _AutoscaleShapeCacheEntry:
    """Cached autoscale DECISION for one structural fingerprint.

    Rolling dispatch solves that share the same structural fingerprint
    (``_warm._fingerprint``) emit an LP of identical shape, so the
    autoscaler's per-layer DECISION is invariant across rolls:

    * ``layer2_exponents`` — the per-:class:`QuantityType` power-of-two
      exponents Layer 2 chose on the first solve of this shape.  ``None``
      when Layer 2 did not trigger (or BASIC/OFF mode).  Replaying these
      via :func:`apply_layer2_with_exponents` reinstalls byte-identical
      side vectors on a subsequent roll's freshly-built Problem WITHOUT
      re-walking coefficients.
    * ``layer3_plan`` — the :class:`Layer3Plan` (``user_*_scale`` etc.)
      Layer 3 recommended.  Re-applied verbatim via
      :func:`apply_scaling` (cheap option-set, no range walk).  ``None``
      when Layer 3 was disabled or its readout failed.
    * ``ranges_pre`` / ``ranges_post`` — the pre-/post-Layer-2
      :class:`RangeReport`s, cached so the per-roll Layer-1 YAML emit and
      the (deduped) console summary keep their range context without a
      re-walk.
    * ``guard`` — the scaling guard's outcome (see the field).

    The KEY property: a cache HIT re-applies all of the above WITHOUT a
    single :func:`detect_ranges` / ``bucket_coefficients`` traversal,
    which is where the per-roll multi-GB ``priv_dirty`` spikes came from.
    """

    layer2_exponents: "dict | None"
    layer2_buckets_before: "dict"
    layer2_buckets_after: "dict"
    layer3_plan: "_AutoscaleLayer3Plan | None"
    ranges_pre: "_AutoscaleRangeReport | None"
    ranges_post: "_AutoscaleRangeReport | None"
    #: The scaling guard's outcome on the first solve of this shape
    #: (``None``: no guard ran).  A HIT replays it (warnings print once per
    #: run) with no walk and no ``locate_extremes``; the roll itself is
    #: rechecked by :class:`_AutoscaleRollRecheck` from its built LP.
    guard: "_AutoscaleGuardOutcome | None" = None


def _autoscale_disable_cache() -> bool:
    """True when ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE=1`` — always recompute
    (the pre-cache, per-roll-traversal behaviour)."""
    return os.environ.get("FLEXTOOL_DISABLE_AUTOSCALE_CACHE") == "1"


def _autoscale_apply_layer2_from_cache(
    pb: "Problem",
    entry: "_AutoscaleShapeCacheEntry",
    *,
    solve_name: str,
    logger: logging.Logger,
) -> "_AutoscaleLayer2Plan | None":
    """Cache-HIT Layer-2 re-apply — NO coefficient walk.

    Reinstalls the cached per-type exponents onto THIS roll's Problem via
    :func:`apply_layer2_with_exponents` (O(#families); zero
    ``detect_ranges`` / ``bucket_coefficients``).  Returns the replayed
    :class:`Layer2Plan` (needed by :func:`_autoscale_unscale_post_solve`)
    or ``None`` when Layer 2 did not trigger for this shape.
    """
    if entry.layer2_exponents is None:
        return None
    try:
        plan = _autoscale_apply_layer2_with_exponents(
            pb,
            entry.layer2_exponents,
            type_buckets_before=entry.layer2_buckets_before,
            type_buckets_after=entry.layer2_buckets_after,
        )
    except Exception:  # pragma: no cover — guard against API drift
        if os.environ.get("FLEXTOOL_AUTOSCALE_STRICT") == "1":
            raise
        logger.exception(
            "autoscale Layer 2 cached replay failed for %s; reverting to "
            "un-scaled LP for this roll",
            solve_name,
        )
        return None
    logger.debug(
        "autoscale Layer 2 [%s]: replayed cached exponents=%s (no range walk)",
        solve_name,
        {t.value: e for t, e in plan.type_exponents.items()},
    )
    return plan


def _autoscale_apply_layer3_from_cache(
    pb: "Problem",
    entry: "_AutoscaleShapeCacheEntry",
    *,
    solve_name: str,
    logger: logging.Logger,
) -> "_AutoscaleLayer3Plan | None":
    """Cache-HIT Layer-3 re-apply — NO post-Layer-2 range walk.

    Re-applies the cached :class:`Layer3Plan`'s HiGHS options to THIS
    roll's Problem via :func:`apply_scaling` (a plain ``set_solver_options``
    merge).  Returns the cached plan for report visibility, or ``None``
    when Layer 3 produced no plan on the first solve of this shape.
    """
    plan = entry.layer3_plan
    if plan is None:
        return None
    try:
        _autoscale_apply_scaling(pb, plan)
    except Exception:  # pragma: no cover
        logger.exception(
            "autoscale Layer 3 cached option apply failed for %s; HiGHS "
            "internal scaling will fill in",
            solve_name,
        )
        return plan
    logger.debug(
        "autoscale Layer 3 [%s]: replayed cached user_objective_scale=%d, "
        "user_bound_scale=%d (no range walk)",
        solve_name,
        plan.user_objective_scale,
        plan.user_bound_scale,
    )
    return plan


def _autoscale_lp_shape_signature(pb: "Problem", base_solve_name: str) -> tuple:
    """Structural signature of a BUILT Problem for the autoscale cache key.

    Invariant across rolls of one rolling solve (same matrix shape +
    family layout) but distinct for genuinely different LPs. Cheap —
    O(#var families + #cstr families), NO coefficient walk. Scoped by
    ``base_solve_name`` so only rolls of the SAME named rolling solve can
    share a cached scaling decision (guards against a same-shape /
    different-magnitude collision between unrelated solves).
    """
    var_sig = tuple(sorted(
        (name, int(v.frame.height), bool(v.integer))
        for name, v in pb._vars.items()
    ))
    cstr_sig = tuple(
        (cname, 1 if over is None else int(over.height))
        for cname, _proto, over in pb._cstrs
    )
    return (base_solve_name, int(pb._next_col), var_sig, cstr_sig)


def _autoscale_emit_console_summary(
    *,
    ranges_pre: "_AutoscaleRangeReport | None",
    ranges_post: "_AutoscaleRangeReport | None",
    layer2_plan: "_AutoscaleLayer2Plan | None",
    layer3_plan: "_AutoscaleLayer3Plan | None",
    solve_name: str,
    already_emitted: set[str],
    memrec: "_MemoryRecorder | None" = None,
    logger: logging.Logger | None = None,
    scaling_config: "ScalingConfig | None" = None,
    guard: "_AutoscaleGuardOutcome | None" = None,
) -> None:
    """Emit the one-line user-visible autoscale summary.

    Uses ``print(...)`` rather than ``logger.info`` so the line surfaces
    at the default log level in the same stream where FlexTool's other
    phase-progress lines (``Input: …``, the HiGHS banner) appear.  We
    de-duplicate by solve name so a rolling solve emits the line once
    per base-solve, not once per roll — Layer 1/2/3 decisions are
    identical across rolls of the same base solve when the autoscaler
    is enabled.

    When ``memrec`` is supplied, a ``polar-high scaling`` phase-progress
    row is emitted immediately after the summary so the log carries a
    timestamp/memory checkpoint at the moment scaling finishes and HiGHS
    is about to run (the long, output-silent solve sits right after it).
    The checkpoint is gated by the same dedup as the summary, so it fires
    once per base solve — exactly when the summary text is printed.
    """
    cfg = scaling_config if scaling_config is not None else _autoscale_resolve_config(None)
    if not _autoscale_mode_enables_layer1(cfg.mode):
        return
    if ranges_pre is None:
        return
    if solve_name in already_emitted:
        return
    line = _autoscale_format_console_summary(
        ranges_pre=ranges_pre,
        ranges_post=ranges_post,
        layer2_plan=layer2_plan,
        layer3_plan=layer3_plan,
        threshold_decades=cfg.threshold_decades,
        guard=guard,
    )
    print(line, flush=True)
    if memrec is not None:
        try:
            memrec.checkpoint(
                "polar_high_scaling", logger, user_label="polar-high scaling",
            )
        except Exception:
            pass
    already_emitted.add(solve_name)


def _autoscale_unscale_post_solve(
    sol: "Solution | None",
    plan: "_AutoscaleLayer2Plan | None",
    *,
    solve_name: str,
    logger: logging.Logger,
    problem: object = None,
) -> None:
    """Layer 2 unscale guard — invoke immediately after ``pb.solve(...)``.

    Idempotent when ``plan is None`` or ``sol is None``: the eager
    unscale keeps the rest of the pipeline (output writers, range
    re-readouts) blind to the Layer-2 substitution.

    Also records on ``sol`` the primal feasibility tolerance *problem* was
    solved with (``sol.flextool_primal_tolerance``; with the column
    factors the unscale leaves on ``sol``, what the carried units online
    are rounded by: ``output._coarse_flows.var_zero_tolerance``).
    """
    if sol is None:
        return
    if problem is not None:
        from flextool.engine.autoscale._guard import highs_tolerances
        sol.flextool_primal_tolerance = highs_tolerances(problem)[0]
    if plan is None:
        return
    try:
        _autoscale_unscale_solution(sol, plan)
    except Exception:  # pragma: no cover
        logger.exception(
            "autoscale Layer 2 unscale failed for %s; downstream output "
            "values are still in scaled units — re-run with autoscale off",
            solve_name,
        )
