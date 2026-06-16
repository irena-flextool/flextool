"""Native polar_high orchestrator — master loop conductor.

The orchestrator:

* Combines the foundation modules (``_solve_config`` + ``_timeline`` +
  ``_recursive_solve`` + ``_stochastic``).
* Drives the per-solve preprocessing through the L0-L9 batch +
  ``preprocessing_solve_time`` + ``solve_writers``.
* Runs the actual solve via ``polar_high.Problem.solve`` (HiGHS).
* Captures :class:`SolveHandoff` per solve via the native
  :func:`build_handoff_from_solution`, threads it forward as
  ``prior_handoff``, and routes it into the in-memory handoff slot of
  the runner so the consume side (``preprocessing_solve_time``,
  ``handoff_writers``) reads from it.

Design choices
--------------

* The orchestrator drives ``_native_run_model.native_run_model`` once
  per top-level invocation, with a **polar_high-as-inner-solver** wrapper
  that:
    - Reads the per-solve snapshot via ``load_flextool``.
    - Builds the LP via ``build_flextool``.
    - Solves via ``polar_high`` (HiGHS).
    - Captures handoff via ``build_handoff_from_solution``.
    - Deposits handoff into ``state.handoffs`` for the next iteration's
      preprocessing.

* **Storage-fixing handoff is in-memory by default** when
  ``state.handoffs`` is non-None.  The file-copy path
  (``shutil.copy`` of ``solve_data_<parent>/fix_storage_*.csv``) is
  consulted only when ``state.handoffs is None``.

* **Roll-counter reset semantics**: every top-level
  :func:`run_orchestration` call invokes
  ``state.solve.roll_counter = state.solve.make_roll_counter()`` first
  so test re-use of the same SolveConfig doesn't desync (R-O5).

* **``model_solve`` validation**: enforced loud-and-early.  Empty
  ``model_solve`` or more-than-one model raises
  :class:`FlexToolConfigError`.

* **``run_chain``** is a thin compat shim that always delegates here.
"""
from __future__ import annotations

import logging
import math
import os
import re
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from flextool.engine_polars._solve_handoff import SolveHandoff
from flextool.engine_polars._solve_state import (
    FlexToolConfigError,
    PathConfig,
    RunnerState,
)
from flextool.engine_polars._determinism import (
    DETERMINISM_OPTIONS,
    SIMPLEX_SCALE_STRATEGY_ADVANCED,
)
from flextool.engine_polars.autoscale import (
    Layer2Plan as _AutoscaleLayer2Plan,
    Layer3Plan as _AutoscaleLayer3Plan,
    RangeReport as _AutoscaleRangeReport,
    ScalingMode as _AutoscaleScalingMode,
    USER_BOUND_SCALE_MAX as _USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN as _USER_BOUND_SCALE_MIN,
    apply_layer2 as _autoscale_apply_layer2,
    apply_layer2_with_exponents as _autoscale_apply_layer2_with_exponents,
    apply_scaling as _autoscale_apply_scaling,
    detect_ranges as _autoscale_compute_ranges,
    format_console_summary as _autoscale_format_console_summary,
    format_nonoptimal_hint as _autoscale_format_nonoptimal_hint,
    mode_enables_layer1 as _autoscale_mode_enables_layer1,
    mode_enables_layer3 as _autoscale_mode_enables_layer3,
    recommend_scaling as _autoscale_recommend_scaling,
    resolve_scaling_config as _autoscale_resolve_config,
    resolve_user_bound_scale_override as _resolve_user_bound_scale_override,
    unscale_solution as _autoscale_unscale_solution,
    write_report as _autoscale_write_report,
)


def _wrap_log_prose(text: str, width: int = 100, indent: str = "  ") -> str:
    """Wrap a prose log message at ``width`` chars, continuation indented."""
    return textwrap.fill(
        text, width=width, subsequent_indent=indent,
        break_long_words=False, break_on_hyphens=False,
    )


# Legacy ``scale_the_objective`` default — historically the
# ``flextool_base.dat``'s ``param scale_the_objective default 1e-6`` and the
# pre-autoscale analyser's fallback when the rough-objective heuristic
# returned a non-finite value.  Phase 2b drops the legacy data-driven
# analyser entirely; we keep the build-time cost multiplier so that:
#
#   * The MPS export (post-Layer-2, pre-solve) still carries pre-scaled
#     cost coefficients in the same magnitude as before — handoff to
#     external solvers continues to see a recognisable objective.
#   * The output writers' un-scaling path (``_resolve_inv_scale_the_objective``)
#     continues to find ``solve_data/scale_the_objective.csv`` with a non-1
#     value; missing-file fallback in that helper is also ``1.0 / 1e-6``,
#     so the un-scaling round-trip is byte-stable.
#   * The user can still override per-solve via DB
#     ``solve.scale_the_objective`` — autoscale Layer 3's
#     ``user_objective_scale`` is HiGHS-internal and stacks on top
#     (HiGHS un-scales internally on output, so layering the build-time
#     scale and the HiGHS-side scale is well-defined).
_LEGACY_DEFAULT_OBJECTIVE_SCALE = 1e-6


def _resolve_effective_obj_scale(user_value: object | None) -> float:
    """Coerce a raw DB ``solve.scale_the_objective`` value to a float.

    Returns the user value when finite and strictly positive; otherwise
    falls back to :data:`_LEGACY_DEFAULT_OBJECTIVE_SCALE` (1e-6).  This
    mirrors the defensive contract the retired
    ``scaling.resolve_effective_scaling`` had: any malformed user value
    (None, non-numeric string, 0, NaN, negative) falls back rather than
    crashing the cascade — the failure mode would be HiGHS's own
    division by zero on output un-scaling.
    """
    if user_value is None:
        return _LEGACY_DEFAULT_OBJECTIVE_SCALE
    try:
        candidate = float(user_value)
    except (TypeError, ValueError):
        return _LEGACY_DEFAULT_OBJECTIVE_SCALE
    if not math.isfinite(candidate) or candidate <= 0.0:
        return _LEGACY_DEFAULT_OBJECTIVE_SCALE
    return candidate


def _baseline_highs_options(
    *,
    user_bound_scale_override: int | None = None,
    scaling_mode: "_AutoscaleScalingMode | None" = None,
) -> dict[str, object]:
    """Build the base HiGHS solver-option dict (determinism + matrix scale).

    Replaces the retired ``scaling.recommended_highs_options`` helper.
    Sets:

    * ``simplex_scale_strategy`` — defaults to
      :data:`SIMPLEX_SCALE_STRATEGY_ADVANCED` (Curtis-Reid matrix
      equilibration).  When ``scaling_mode == ScalingMode.OFF`` we force
      ``simplex_scale_strategy=0`` so HiGHS' own equilibration is also
      disabled — that's the only mode in which FlexTool touches the
      HiGHS-internal scaling knob.  Layer 3's :func:`apply_scaling` may
      re-assert this value on the cold path; on warm rebuilds the value
      set here is the authoritative one.
    * :data:`DETERMINISM_OPTIONS` — ``random_seed`` / ``parallel`` /
      ``solver`` / ``presolve`` pins for byte-deterministic LP solutions.
    * ``user_bound_scale`` — only when ``user_bound_scale_override`` is
      a non-zero integer (CLI ``--user-bound-scale N`` / DB
      ``solve.user_bound_scale``).  Clamped to the HiGHS-safe range
      ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``.  When unset, the
      autoscaler's Layer 3 may still emit its own value based on the
      coefficient ranges.

    ``user_cost_scale`` is intentionally NOT set — costs are already
    multiplied by ``scale_the_objective`` inside ``build_flextool``, and
    Layer 3 may add ``user_objective_scale`` on top; layering a third
    cost-side knob would compound confusingly.
    """
    if scaling_mode is _AutoscaleScalingMode.OFF:
        # OFF mode: disable HiGHS' internal equilibration too.  ``parallel``
        # / ``random_seed`` / ``solver`` / ``presolve`` pins from
        # DETERMINISM_OPTIONS still apply — OFF is about scaling, not
        # determinism.
        simplex_scale = 0
    else:
        simplex_scale = SIMPLEX_SCALE_STRATEGY_ADVANCED
    options: dict[str, object] = {
        "simplex_scale_strategy": simplex_scale,
        **DETERMINISM_OPTIONS,
    }
    if user_bound_scale_override is not None and user_bound_scale_override != 0:
        n = int(user_bound_scale_override)
        if n > _USER_BOUND_SCALE_MAX:
            n = _USER_BOUND_SCALE_MAX
        if n < _USER_BOUND_SCALE_MIN:
            n = _USER_BOUND_SCALE_MIN
        options["user_bound_scale"] = n
    return options


def _autoscale_emit_layer1(
    sol: "Solution | None",
    *,
    solve_name: str,
    logger: logging.Logger,
    work_folder: str | os.PathLike | None,
    layer2_plan: "_AutoscaleLayer2Plan | None" = None,
    layer3_plan: "_AutoscaleLayer3Plan | None" = None,
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
    # ``cli_args=None`` is intentional: the CLI surface
    # (``cmd_run_flextool``) mirrors ``--scaling`` /
    # ``--user-bound-scale`` into the ``FLEXTOOL_SCALING`` /
    # ``FLEXTOOL_USER_BOUND_SCALE`` env vars before invoking the
    # orchestrator, matching the existing env-threading convention
    # documented on the ``run_chain_from_db`` call site.  Cascade-
    # internal hops therefore observe operator intent without
    # plumbing the parsed ``args`` namespace through every helper.
    cfg = _autoscale_resolve_config(None)
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
                from flextool.engine_polars.autoscale._report import (
                    render_layer2 as _render_l2,
                )
                report_tree["layer2"] = _render_l2(layer2_plan)
            if layer3_plan is not None:
                from flextool.engine_polars.autoscale._report import (
                    render_layer3 as _render_l3,
                )
                report_tree["layer3"] = _render_l3(layer3_plan)
            _autoscale_write_report(report_tree, yaml_path)
        except Exception:  # pragma: no cover — non-fatal
            logger.exception(
                "autoscale Layer 1 report write failed (%s)", yaml_path,
            )


def _autoscale_apply_layer3_pre_solve(
    pb: "Problem",
    *,
    layer2_plan: "_AutoscaleLayer2Plan | None",
    solve_name: str,
    logger: logging.Logger,
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
    """
    cfg = _autoscale_resolve_config(None)
    if not _autoscale_mode_enables_layer3(cfg.mode):
        return None
    try:
        ranges_post_l2 = _autoscale_compute_ranges(pb, cfg)
    except Exception:  # pragma: no cover — guard against future API drift
        logger.exception(
            "autoscale Layer 3 pre-solve range readout failed for %s; "
            "skipping Layer 3 (Layer 1 / Layer 2 still applied)",
            solve_name,
        )
        return None
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
    cfg = _autoscale_resolve_config(None)
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
    cfg = _autoscale_resolve_config(None)
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


def _autoscale_emit_nonoptimal_hint(
    *,
    ranges_pre: "_AutoscaleRangeReport | None",
    sol: "Solution | None",
) -> None:
    """Emit the scaling-related hint when HiGHS reports non-optimal.

    Only fires when (a) the solve genuinely returned non-optimal AND
    (b) the autoscaler's Layer-1 detector had flagged poor scaling on
    the pre-solve LP.  Printing scaling advice on a well-conditioned
    LP that simply happened to be infeasible would be misleading, so
    we keep the trigger conjunctive.
    """
    if sol is None:
        return
    if ranges_pre is None:
        return
    if sol.optimal:
        return
    hint = _autoscale_format_nonoptimal_hint(ranges_pre)
    if hint:
        print(hint, flush=True)


def _autoscale_unscale_post_solve(
    sol: "Solution | None",
    plan: "_AutoscaleLayer2Plan | None",
    *,
    solve_name: str,
    logger: logging.Logger,
) -> None:
    """Layer 2 unscale guard — invoke immediately after ``pb.solve(...)``.

    Idempotent when ``plan is None`` or ``sol is None``: the eager
    unscale keeps the rest of the pipeline (output writers, range
    re-readouts) blind to the Layer-2 substitution.
    """
    if plan is None or sol is None:
        return
    try:
        _autoscale_unscale_solution(sol, plan)
    except Exception:  # pragma: no cover
        logger.exception(
            "autoscale Layer 2 unscale failed for %s; downstream output "
            "values are still in scaled units — re-run with autoscale off",
            solve_name,
        )


if TYPE_CHECKING:
    from collections.abc import Callable

    import polars as pl
    from polar_high import Problem, Solution

    from flextool.engine_polars.input import FlexData


# ---------------------------------------------------------------------------
# Opt-in memory diagnostics
# ---------------------------------------------------------------------------


# Whitelist of ``user_label`` strings that are emitted to the log in
# regular (non-verbose) mode.  Set ``FLEXTOOL_MEMORY_VERBOSE=1`` to emit
# every checkpoint (the full pre-cleanup trace).
#
# ``Solve start: <name>, k/N`` markers are emitted as plain text lines
# (no checkpoint), so they don't appear here.
_MEM_WHITELIST_LABELS: frozenset[str] = frozenset({
    "Run start",
    "Inputs prepared",
    "FlexData built",
    "Matrix built by polar-high",
    "polar-high scaling",
    "Solver",
    "Outputs written",
    "Solve cleanup",
})


def _is_whitelisted_mem_label(user_label: str | None) -> bool:
    """True when ``user_label`` matches a whitelisted phase label."""
    if not user_label:
        return False
    return user_label in _MEM_WHITELIST_LABELS


class _MemoryRecorder:
    """Opt-in tracemalloc + RSS checkpoint recorder.

    Activated by ``FLEXTOOL_MEMORY_DIAGNOSTICS=1``.  When the env var is
    not set, callers should construct :class:`_NoopMemoryRecorder`
    instead (or simply skip construction); :meth:`checkpoint` here is the
    hot-path no-op fallback only when ``enabled`` is False.

    Each :meth:`checkpoint` call appends one row to
    ``<work_folder>/solve_data/memory_diagnostics.csv`` with schema::

        checkpoint,t_elapsed_s,traced_current_mb,traced_peak_mb,rss_mb

    The file is open/append/closed per row (same atomicity pattern as
    :class:`flextool.cli._timing.TimingRecorder.record`)
    so a crash mid-cascade still leaves a parseable trail.

    A one-liner is also logged at INFO level via the supplied logger so
    progress is visible in stdout even when the GUI buffers.

    ``tracemalloc.start()`` is invoked lazily on first checkpoint to keep
    the cost localised to instrumented runs.
    """

    _HEADER = (
        "checkpoint",
        "t_elapsed_s",
        "traced_current_mb",
        "traced_peak_mb",
        "rss_mb",
    )

    def __init__(self, csv_path: Path | None = None,
                 enabled: bool = True,
                 verbose: bool = True) -> None:
        """Construct a phase-progress recorder.

        Parameters
        ----------
        csv_path
            Where to write the per-checkpoint CSV.  ``None`` skips CSV
            emission (verbose log lines still fire).
        enabled
            Full diagnostic mode — starts tracemalloc on first checkpoint
            so ``traced_peak`` becomes meaningful, and writes the CSV.
            When ``False`` we still emit human-readable log lines with
            RSS + section time + Δrss (RSS reads from ``/proc`` are
            essentially free); ``peak`` shows as ``-`` since tracemalloc
            isn't running.
        verbose
            Emit log lines (one per checkpoint).  Set ``False`` only if
            you want a fully silent recorder (rare; debugging).
        """
        self.enabled = enabled
        self.verbose = verbose
        self.t0 = time.perf_counter()
        self._t_prev = self.t0
        self._rss_prev_mb: float = 0.0
        self._peak_prev_mb: float = 0.0
        self._sys_prev_mb: float = 0.0
        self._swap_prev_mb: float = 0.0
        self._header_emitted: bool = False
        self._path = Path(csv_path) if csv_path is not None else None
        self._started = False
        # FLEXTOOL_PYRAMID_PROFILE=1: previous RSS for per-batch delta.
        self._pyramid_prev_rss_mb: float = 0.0
        if self.enabled and self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            import csv as _csv
            with open(self._path, "w", newline="") as f:
                _csv.writer(f).writerow(self._HEADER)

    @staticmethod
    def _read_rss_mb() -> float:
        """Read committed memory (anon RSS + swap, in MB) from
        ``/proc/self/status``.

        We deliberately don't report ``VmRSS`` (= ``RssAnon`` +
        ``RssFile`` + ``RssShmem``) because file-backed pages are
        evictable cache from the kernel's POV and don't reflect the
        process's true memory commitment.  ``RssAnon`` (anonymous
        resident, i.e. heap + private mappings) plus ``VmSwap`` (the
        same anonymous pages that have been swapped out) gives the
        right picture of "memory this process actually needs" —
        what systemd-oomd's PSI signal effectively responds to, and
        what tracks the system monitor's "Used" number more closely
        than raw ``VmRSS``.

        Returns 0.0 if /proc isn't available (non-Linux) or the
        relevant lines aren't found.  We never want diagnostics to
        raise.
        """
        anon_kb = 0.0
        swap_kb = 0.0
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("RssAnon:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            anon_kb = float(parts[1])
                    elif line.startswith("VmSwap:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            swap_kb = float(parts[1])
        except OSError:
            pass
        return (anon_kb + swap_kb) / 1024.0

    @staticmethod
    def _read_sys_swap_used_mb() -> float:
        """System-level used swap (MB) = ``SwapTotal - SwapFree`` from
        ``/proc/meminfo``.  Returns 0.0 when there's no swap configured
        (``SwapTotal == 0``) or ``/proc`` isn't available.
        """
        total_kb = 0.0
        free_kb = 0.0
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("SwapTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kb = float(parts[1])
                    elif line.startswith("SwapFree:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            free_kb = float(parts[1])
                    if total_kb and free_kb:
                        break
        except OSError:
            pass
        if total_kb <= 0:
            return 0.0
        return max(0.0, (total_kb - free_kb) / 1024.0)

    @staticmethod
    def _read_sys_used_mb() -> float:
        """System-level used memory (MB), matching what most monitors
        ("htop", KSysGuard, GNOME) show as "Used".

        Computed as ``MemTotal - MemAvailable`` from ``/proc/meminfo``.
        ``MemAvailable`` is the kernel's own estimate of how much
        memory could be allocated to a new process without swapping —
        it already accounts for evictable page-cache + reclaimable
        slab + lazily-freed anonymous pages (MADV_FREE).  Subtracting
        from total gives the closest single-number match to what the
        user sees in their desktop's system monitor.

        Includes contributions from every process on the host, not
        just this one — which is the right metric for desktop-crash
        awareness (the desktop crashes when total ``MemAvailable``
        approaches zero, regardless of which process is consuming the
        pages).

        Returns 0.0 if /proc isn't available.
        """
        total_kb = 0.0
        avail_kb = 0.0
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            total_kb = float(parts[1])
                    elif line.startswith("MemAvailable:"):
                        parts = line.split()
                        if len(parts) >= 2:
                            avail_kb = float(parts[1])
                    if total_kb and avail_kb:
                        break
        except OSError:
            pass
        if total_kb <= 0:
            return 0.0
        return (total_kb - avail_kb) / 1024.0

    # Fixed widths used to align the mem output into a table.
    # Label column fits the longest whitelisted label
    # ("Matrix built by polar-high", 26 chars).
    _LABEL_W = 28      # label column (left-aligned)
    # Each data cell renders "<absolute> (<+/-delta>)" right-aligned
    # within these widths.  Sized for normal use; cells widen
    # naturally when values overflow (right-align preserves the gutter).
    _TIME_CELL_W = 17  # fits e.g. "9999.9s (+999.9)"
    _MEM_CELL_W = 19   # fits e.g. "-100.00 GB (+99.99)"

    @staticmethod
    def _pick_size_unit(mb: float) -> tuple[float, str]:
        """Pick GB vs MB display unit for an absolute MB value.
        Returns ``(divisor, label)`` — e.g. ``(1024.0, "GB")``.
        """
        if abs(mb) >= 1024.0:
            return 1024.0, "GB"
        return 1.0, "MB"

    @classmethod
    def _fmt_mem_cell(cls, mb: float | None, delta_mb: float | None) -> str:
        """Render a memory cell as ``"<absolute> (<±delta>)"`` right-
        aligned within ``_MEM_CELL_W``.  Both numbers are rendered in
        the unit chosen for the absolute, so the cell reads as a single
        consistent magnitude (e.g. ``"5.18 GB (+0.21)"`` — both GB).
        When ``mb`` is None, a single dash fills the cell.
        """
        if mb is None:
            return f"{'-':>{cls._MEM_CELL_W}}"
        div, label = cls._pick_size_unit(mb)
        # Two decimals when GB, integer when MB — matches the example.
        if label == "GB":
            val = f"{mb / div:.2f} {label}"
        else:
            val = f"{mb / div:.0f} {label}"
        if delta_mb is None:
            cell = f"{val} (-)"
        else:
            if abs(delta_mb) < (0.005 if label == "GB" else 0.5):
                delta_str = "+0"
            else:
                sign = "+" if delta_mb >= 0 else "-"
                a = abs(delta_mb) / div
                fmt = ".2f" if label == "GB" else ".0f"
                delta_str = f"{sign}{a:{fmt}}"
            cell = f"{val} ({delta_str})"
        return f"{cell:>{cls._MEM_CELL_W}}"

    @classmethod
    def _fmt_time_cell(cls, t_elapsed: float, t_section: float | None) -> str:
        """Render the time cell as ``"<elapsed>s (+<section>)"`` right-
        aligned within ``_TIME_CELL_W``.  First-row ``t_section is None``
        renders ``"<elapsed>s (-)"``.
        """
        if t_section is None:
            cell = f"{t_elapsed:.1f}s (-)"
        else:
            sign = "+" if t_section >= 0 else "-"
            cell = f"{t_elapsed:.1f}s ({sign}{abs(t_section):.1f})"
        return f"{cell:>{cls._TIME_CELL_W}}"

    def _emit_header(self) -> None:
        """Print the column-header line for the phase-progress table."""
        blank_label = " " * self._LABEL_W
        header = (
            f"{blank_label}  "
            f"{'time':^{self._TIME_CELL_W}}  "
            f"|  {'RSS memory':^{self._MEM_CELL_W}}  "
            f"|  {'system memory':^{self._MEM_CELL_W}}  "
            f"|  {'system swap':^{self._MEM_CELL_W}}"
        )
        try:
            print(header, flush=True)
        except OSError:
            pass

    def checkpoint(self, label: str, logger: logging.Logger,
                   user_label: str | None = None) -> None:
        """Record a phase checkpoint.

        ``label`` is the canonical machine-readable identifier persisted
        to the CSV (when full diagnostics is enabled).  ``user_label``
        (optional) is the human-friendly phrasing emitted to the log;
        when absent, ``label`` is used.

        Log lines always emit (RSS read from ``/proc`` is essentially
        free).  When full diagnostics is enabled (env-var
        ``FLEXTOOL_MEMORY_DIAGNOSTICS=1``) the ``traced_peak`` column
        and the CSV emission are populated by tracemalloc; otherwise
        the peak shows as ``-``.
        """
        peak_mb: float | None = None
        current_mb: float | None = None
        if self.enabled:
            import tracemalloc
            if not self._started:
                tracemalloc.start()
                self._started = True
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / (1024.0 * 1024.0)
            peak_mb = peak / (1024.0 * 1024.0)
        rss_mb = self._read_rss_mb()
        sys_mb = self._read_sys_used_mb()
        swap_mb = self._read_sys_swap_used_mb()
        t_elapsed = time.perf_counter() - self.t0
        # Section deltas relative to previous *emitted* checkpoint.  When
        # we suppress an emission we leave ``_t_prev`` /
        # ``_rss_prev_mb`` / ``_sys_prev_mb`` / ``_peak_prev_mb`` alone,
        # so the next emitted line shows the cumulative delta covering
        # all the suppressed activity in between — exactly what the
        # user wants to see in the compact mode.
        t_section = t_elapsed - (self._t_prev - self.t0)
        delta_rss = rss_mb - self._rss_prev_mb
        delta_sys = sys_mb - self._sys_prev_mb
        delta_swap = swap_mb - self._swap_prev_mb
        # CSV row — only when full diagnostics is enabled and a path was
        # configured.  Always written for every checkpoint (independent
        # of the log-line whitelist) so the CSV remains a complete
        # trace.
        if self.enabled and self._path is not None and peak_mb is not None:
            row = (
                str(label),
                f"{t_elapsed:.6f}",
                f"{current_mb:.3f}",
                f"{peak_mb:.3f}",
                f"{rss_mb:.3f}",
            )
            import csv as _csv
            try:
                with open(self._path, "a", newline="") as f:
                    _csv.writer(f).writerow(row)
            except OSError:
                pass
        # FLEXTOOL_PYRAMID_PROFILE=1: emit a tab-separated stderr line
        # for emit_solve_time.* labels (polar-high precedent).  Reuses
        # the already-computed rss_mb / t_elapsed; zero cost when unset.
        if (os.environ.get("FLEXTOOL_PYRAMID_PROFILE") == "1"
                and label.startswith("emit_solve_time.")):
            delta_pyramid = rss_mb - self._pyramid_prev_rss_mb
            rss_gb = rss_mb / 1024.0
            delta_gb = delta_pyramid / 1024.0
            try:
                import sys as _sys
                _sys.stderr.write(
                    f"[pyramid profile]\tphase={label}\t"
                    f"rss_gb={rss_gb:.2f}\tdelta_gb={delta_gb:+.2f}\t"
                    f"t_s={t_elapsed:.1f}\n"
                )
                _sys.stderr.flush()
            except OSError:
                pass
            self._pyramid_prev_rss_mb = rss_mb
        # Decide whether to emit the log line.  Regular mode shows only
        # the whitelisted phase labels; ``FLEXTOOL_MEMORY_VERBOSE=1``
        # restores the full per-checkpoint trace.
        verbose_mode = bool(os.environ.get("FLEXTOOL_MEMORY_VERBOSE"))
        emit = self.verbose and (
            verbose_mode or _is_whitelisted_mem_label(user_label or label)
        )
        if emit:
            display = user_label or label
            label_col = f"{display:<{self._LABEL_W}}"
            is_first = self._t_prev == self.t0
            # Emit the column header once, immediately before the first
            # data line.  Inline labels are dropped from data lines
            # (cleaner, narrower); the header is the legend.
            if not self._header_emitted:
                self._emit_header()
                self._header_emitted = True
            elif display == "Solver":
                # Blank line + header repeat above each Solver row so
                # the dominant solve phase visually separates from the
                # per-group prep block printed above it.
                try:
                    print("", flush=True)
                except OSError:
                    pass
                self._emit_header()
            # First-row convention: time delta renders "(-)" (no prior
            # checkpoint to subtract from); memory deltas equal their
            # absolute (prev=0), so e.g. "+230.1" appears alongside the
            # absolute "230.1 MB" — which is correct: the process has
            # consumed exactly that much since the recorder started.
            time_cell = self._fmt_time_cell(
                t_elapsed, None if is_first else t_section
            )
            rss_cell = self._fmt_mem_cell(rss_mb, delta_rss)
            sys_cell = self._fmt_mem_cell(sys_mb, delta_sys)
            swap_cell = self._fmt_mem_cell(swap_mb, delta_swap)
            line = (
                f"{label_col}  "
                f"{time_cell}  "
                f"|  {rss_cell}  "
                f"|  {sys_cell}  "
                f"|  {swap_cell}"
            )
            try:
                print(line, flush=True)
            except OSError:
                pass
            # Advance prev-section bookkeeping ONLY on actual emission so
            # the next emitted line's delta covers all the suppressed
            # activity since the last visible checkpoint.
            self._t_prev = time.perf_counter()
            self._rss_prev_mb = rss_mb
            self._sys_prev_mb = sys_mb
            self._swap_prev_mb = swap_mb
            if peak_mb is not None:
                self._peak_prev_mb = peak_mb


class _NoopMemoryRecorder:
    """Zero-overhead drop-in when ``FLEXTOOL_MEMORY_DIAGNOSTICS`` is unset.

    Retained for callers that explicitly want a fully silent recorder
    (rare; debugging).  The default code path now uses
    :class:`_MemoryRecorder` with ``enabled=False`` instead — that mode
    still emits user-visible log lines (RSS + section time + Δrss)
    while skipping CSV emission and tracemalloc startup.
    """

    enabled = False

    def checkpoint(self, label: str, logger: logging.Logger,
                   user_label: str | None = None) -> None:  # noqa: D401, ARG002
        return None


# Module-level recorder reference.  ``run_orchestration`` constructs the
# per-run recorder and publishes it here so deeper-stack modules (e.g.
# :mod:`flextool.engine_polars.input`'s ``_apply_db_overrides``) can
# emit phase progress in the unified ``[mem]`` format without each
# carrying a recorder kwarg.  Reset to ``None`` when the run completes
# so a subsequent run starts clean.
_PHASE_RECORDER: "_MemoryRecorder | None" = None


def set_phase_recorder(rec: "_MemoryRecorder | None") -> None:
    """Publish the current run's phase recorder so deeper callers can
    emit checkpoints without explicit plumbing.  Pass ``None`` to clear.
    """
    global _PHASE_RECORDER
    _PHASE_RECORDER = rec


def get_phase_recorder() -> "_MemoryRecorder | None":
    """Return the current run's phase recorder, or ``None`` when none
    is active (e.g. unit tests that bypass ``run_orchestration``).
    """
    return _PHASE_RECORDER


# ---------------------------------------------------------------------------
# Heap release (glibc malloc_trim)
# ---------------------------------------------------------------------------
#
# The polars/Rust allocator routes through glibc malloc, and glibc's main
# arena holds freed pages internally instead of returning them to the OS.
# After a heavy allocation+free cycle (``write_workdir_inputs``,
# ``load_flextool``, the broadcast cascade) we leak hundreds of MB to
# multiple GB of unmapped-but-untrimmed heap.  Direct measurement on
# H2_trade y2050 (2026-05-13):  RSS 3.8 GB → 2.25 GB after a single
# ``malloc_trim(0)`` call (1.6 GB / 41 % drop).  ``pa.default_memory_pool
# ().release_unused()`` and ``gc.collect()`` had zero effect — polars
# does not route through pyarrow's pool, so only the libc-level trim
# releases anything.
#
# The helper is a no-op on non-glibc systems (musl Alpine containers,
# macOS, Windows).  Safe to call freely; cost is ~10-50ms per call.
_libc_malloc_trim = None


def _try_malloc_trim() -> bool:
    """Call ``libc.so.6.malloc_trim(0)`` if available; return True on success.

    Cached lookup after the first call.  Failures (non-glibc systems,
    missing libc, etc.) are logged once at DEBUG level and the helper
    becomes a permanent no-op for the process lifetime.
    """
    global _libc_malloc_trim
    if _libc_malloc_trim is False:
        return False
    if _libc_malloc_trim is None:
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            # malloc_trim(size_t pad) -> int.  pad=0 means trim aggressively.
            _libc_malloc_trim = libc.malloc_trim
        except (OSError, AttributeError):
            _libc_malloc_trim = False
            return False
    try:
        _libc_malloc_trim(0)
        return True
    except Exception:  # noqa: BLE001
        _libc_malloc_trim = False
        return False


def _phase_prof(label: str) -> None:
    """Env-gated (FLEXTOOL_PHASE_PROFILE=1) epoch-stamped RSS print to stderr. No-op otherwise.
    Epoch matches flextool/_mem_sampler.py's `epoch=` field for 1:1 alignment with mem.log."""
    import os
    import sys
    import time
    if os.environ.get("FLEXTOOL_PHASE_PROFILE") != "1":
        return
    try:
        with open("/proc/self/status") as _f:
            for _ln in _f:
                if _ln.startswith("VmRSS:"):
                    sys.stderr.write(f"[phase profile] epoch={time.time():.3f}\tstep={label}\trss_gb={int(_ln.split()[1])/(1024*1024):.3f}\n")
                    sys.stderr.flush()
                    break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Cross-level retention audit (env-gated diagnostic + regression hook)
#
# A prior solve-level's ``Solution.highs`` (and ``flex_data_provider``) must be
# released BEFORE the next solve builds its FlexData + LP — otherwise the two
# levels' footprints coexist (storage + dispatch ≈ 2x peak; the DES 7/9
# near-OOM).  This records any prior step whose level is EXHAUSTED (no upcoming
# solve of that level) yet still holds a live ``solution.highs`` at the instant
# a new solve is about to build.  A non-empty record == the cross-level
# retention bug is present.  Gated by ``FLEXTOOL_LEVEL_RELEASE_AUDIT=1``;
# consumed by tests/engine_polars/test_cross_level_highs_release.py.
# ---------------------------------------------------------------------------
_LEVEL_RELEASE_AUDIT: "list[dict]" = []


def _audit_prior_level_release(*, steps, step_level_keys, all_level_keys,
                               iter_idx, this_level, complete_solve_name):
    """Append a violation record for exhausted-level prior steps still holding
    a live ``solution.highs``.  No-op unless ``FLEXTOOL_LEVEL_RELEASE_AUDIT=1``."""
    if os.environ.get("FLEXTOOL_LEVEL_RELEASE_AUDIT") != "1":
        return
    upcoming = (
        set(all_level_keys[iter_idx + 1:])
        if (iter_idx is not None and all_level_keys) else set()
    )
    violators = []
    for _k, _step in (steps or {}).items():
        _lvl = step_level_keys.get(_k)
        sol = getattr(_step, "solution", None)
        if sol is None or getattr(sol, "highs", None) is None:
            continue
        if _lvl is not None and _lvl != this_level and _lvl not in upcoming:
            violators.append(_k)
    _LEVEL_RELEASE_AUDIT.append({
        "kind": "exhausted_entry",
        "solve": complete_solve_name,
        "iter_idx": iter_idx,
        "violators": violators,
    })


def _audit_cold_rebuild_release(*, steps, complete_solve_name):
    """At a COLD rebuild (warm_used False), no parked step's HiGHS is the
    reuse source, so any prior step still holding a live ``solution.highs``
    is a same-level stacking risk.  Record them (post-release should be
    empty).  No-op unless ``FLEXTOOL_LEVEL_RELEASE_AUDIT=1``."""
    if os.environ.get("FLEXTOOL_LEVEL_RELEASE_AUDIT") != "1":
        return
    violators = [
        _k for _k, _step in (steps or {}).items()
        if getattr(getattr(_step, "solution", None), "highs", None) is not None
    ]
    _LEVEL_RELEASE_AUDIT.append({
        "kind": "cold_rebuild",
        "solve": complete_solve_name,
        "violators": violators,
    })


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SnapshotSolution:
    """Lightweight Solution-API stand-in carrying only the captured
    decision-variable frames a sub-solve needs to survive polar-high's
    internal between-solve release of ``Solution._vars``.

    The cascade captures :attr:`OrchestrationStep.captured_vars` at the
    earliest reliable point in the per-solve loop (immediately before
    the step is deposited in ``self._all_steps``).  Polar-high may
    release the live :class:`polar_high.Solution._vars` dict for
    earlier sub-solves between the deposit and the end-of-cascade
    consumers (e.g. :func:`flextool.process_outputs.read_parameters.
    _entity_all_capacity`), so end-of-cascade writers that look at
    every sub-solve's decision variables fall back to this snapshot.

    The wrapper only satisfies the duck-typed contract that
    ``_entity_all_capacity`` needs:

    * ``solution._vars`` — dict-like with the captured names as keys
      (supports ``"v_invest_p" in solution._vars``).
    * ``solution.value(name)`` — returns the captured polars long-form
      DataFrame.

    All other ``Solution`` attributes (``obj``, ``optimal``, ``highs``,
    ``col_value``, …) are intentionally absent; callers that need
    those should read the live :attr:`OrchestrationStep.solution`
    (last step only by default, or every step under
    ``keep_solutions=True``).
    """

    _vars: "dict[str, pl.DataFrame]"

    def value(self, name: str) -> "pl.DataFrame":
        return self._vars[name]


@dataclass
class OrchestrationStep:
    """Per-solve result of :func:`run_orchestration`.

    Mirrors :class:`flextool.engine_polars.chain.ChainStep` but produced
    by the native orchestrator path.  ``handoff`` is the carrier used to
    seed the *next* solve's preprocessing.

    Attributes
    ----------
    solve_name : str
        The complete (sub-)solve identifier emitted by flextool's
        orchestration loop (e.g. ``"y2025_5week"`` or
        ``"dispatch_fullYear_roll_roll_3"``).
    solution : polar_high.Solution | None
        The HiGHS solution.  By default (``keep_solutions=False`` on
        :func:`run_chain_from_db` / :func:`run_orchestration`) only the
        LAST sub-solve in a cascade retains its ``solution`` — earlier
        steps clear this slot to release the HiGHS instance + variable
        arrays.  Set ``keep_solutions=True`` to retain ``solution`` on
        every step (Phase C.5 — memory discipline).  Also ``None`` on
        the failed-solve path.
    handoff : SolveHandoff
        polar_high-derived handoff carriers, threaded forward.  Always
        populated (kilobyte-sized; safe to retain across the cascade).
    obj : float | None
        Objective value (cached for quick comparison; equal to
        ``solution.obj``).  Always populated when the solve succeeded —
        survives the ``keep_solutions=False`` slim pass, so cascade-
        wide objective sweeps work without ``keep_solutions=True``.
    optimal : bool | None
        Phase C.5 — slim summary mirror of ``solution.optimal`` that
        survives the per-step memory release.  ``None`` only on the
        failed-solve path (where ``solution`` is also ``None``).
        Consumers that only need the optimal/non-optimal status (e.g.
        CLI exit-code branch in ``cmd_run_flextool.py``) should read
        this instead of ``solution.optimal`` so they work without
        ``keep_solutions=True``.
    warm_used : bool
        Δ.12d — True if this solve was produced by warm-updating the
        prior solve's :class:`polar_high.WarmProblem` instance; False
        if it was a cold rebuild.  Always False for the first solve
        and for ``warm=False`` runs.  Always populated (slim summary).
    flex_data : FlexData | None
        Δ.31 — the polars input bundle this sub-solve consumed.  Held
        on the step so downstream :func:`flextool.process_outputs.
        write_outputs` can build the parameter / set namespaces in
        memory instead of re-parsing the workdir CSVs.  Subject to the
        same ``keep_solutions`` gating as ``solution`` (Phase C.5):
        only the LAST step retains ``flex_data`` by default.  ``None``
        on the failed-load path.
    flex_data_provider : FlexDataProvider | None
        The per-sub-solve :class:`FlexDataProvider` populated by the
        cascade's writers.  Subject to the same ``keep_solutions``
        gating as ``solution`` / ``flex_data`` (Phase C.5): only the
        LAST step retains it by default.  Consumed by ``--csv-dump``
        in ``cmd_run_flextool`` to snapshot the cascade's derived
        frames to disk.
    """

    solve_name: str
    solution: "Solution | None"
    handoff: SolveHandoff
    obj: float | None = None
    optimal: bool | None = None
    warm_used: bool = False
    flex_data: "FlexData | None" = None
    flex_data_provider: "object | None" = None
    captured_vars: "dict[str, pl.DataFrame]" = field(default_factory=dict)
    """Per-sub-solve snapshot of the decision-variable frames that
    end-of-cascade writers (``_entity_all_capacity`` and friends) need
    after polar-high has released the live ``Solution._vars`` dict for
    earlier sub-solves.

    Captured at the latest reliable point in the per-solve loop —
    immediately before the step is deposited — so it always reflects
    the sub-solve's own values regardless of how polar-high or
    downstream slimming touches ``self.solution``.  Empty dict when
    the solve failed or when there was no Solution to capture from.

    Access via :attr:`effective_solution` rather than reading directly:
    callers normally want the live Solution when it's still populated
    and the snapshot only as a fallback.
    """

    @property
    def effective_solution(self) -> "Solution | SnapshotSolution | None":
        """Return the live :attr:`solution` when its ``_vars`` is still
        populated; otherwise a :class:`SnapshotSolution` over the
        captured frames.  ``None`` only when both are unavailable
        (failed-solve path, or solution slimmed AND no capture taken).

        End-of-cascade consumers that walk every sub-solve's decision
        variables (e.g. ``read_parameters_multi`` →
        ``_entity_all_capacity``) should read this instead of
        :attr:`solution` so non-last sub-solves remain observable
        after polar-high's between-solve ``_vars`` release.
        """
        sol = self.solution
        live_vars = getattr(sol, "_vars", None) if sol is not None else None
        if live_vars:
            return sol
        if self.captured_vars:
            return SnapshotSolution(_vars=dict(self.captured_vars))
        return sol


# ---------------------------------------------------------------------------
# Scaling-output helper  (shared by cascade & single-solve paths)
# ---------------------------------------------------------------------------


def _write_scale_csv(
    *,
    solve_data_dir: Path,
    solve_name: str,
    effective_obj_scale: float,
    logger: logging.Logger,
) -> None:
    """Emit ``solve_data/scale_the_objective.csv`` for the given solve.

    Required by the downstream parquet / CSV writers — they read it via
    :func:`flextool.process_outputs.read_highs_solution.
    _resolve_inv_scale_the_objective` to un-scale variable values and
    duals back to user-facing units.  Best-effort: a write failure logs
    a warning but does not raise.

    The legacy human-readable ``scaling_report.txt`` diagnostic was
    retired in Phase 2b; the autoscaler's ``solve_data/autoscale_<solve>.yaml``
    (written from :func:`_autoscale_emit_layer1`) is the new
    machine-readable audit.  Callers should write the CSV exactly once
    per base solve — its value is invariant across rolls of the same
    base solve.
    """
    try:
        from flextool.engine_polars._emit_solve_writers import (
            derive_scale_the_objective,
        )
        sd = Path(solve_data_dir)
        sd.mkdir(parents=True, exist_ok=True)
        path = sd / "scale_the_objective.csv"
        derive_scale_the_objective(effective_obj_scale).write_csv(
            path, line_terminator="\r\n",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "scale_the_objective.csv write failed for %s: %s",
            solve_name, exc,
        )


# ---------------------------------------------------------------------------
# Master loop
# ---------------------------------------------------------------------------


def _validate_model_solve(state: RunnerState) -> list[str]:
    """Validate ``state.solve.model_solve`` and return the solve list.

    There must be exactly one model with at least one solve.  Multi-
    model is documented as unsupported.
    """
    if not state.solve.model_solve:
        raise FlexToolConfigError(
            "No model. Make sure the 'model' class defines solves [Array]."
        )
    if len(state.solve.model_solve) > 1:
        raise FlexToolConfigError(
            "Trying to run more than one model — not supported. "
            "model_solve must contain exactly one model."
        )
    solves = next(iter(state.solve.model_solve.values()))
    if not solves:
        raise FlexToolConfigError("No solves in model.")
    return solves


def _bootstrap_dirs(work_folder: Path, logger: logging.Logger) -> None:
    """Create ``solve_data/``, ``output_raw/``, ``output_plots/`` under
    *work_folder* if they don't exist.

    Mirrors lines 63-76 of the flextool reference.
    """
    for sub in ("solve_data", "output_raw", "output_plots"):
        try:
            (work_folder / sub).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.debug(f"{sub} folder existed")


def run_orchestration(
    state: RunnerState,
    work_folder: Path | str,
    *,
    runner_factory=None,
    db_url: str | None = None,
    scenario_name: str | None = None,
    warm: bool = True,
    keep_solutions: bool = False,
    csv_dump: bool = False,
) -> dict[str, OrchestrationStep]:
    """Drive the master loop natively.

    Per-step:

    1. Bootstrap directories (idempotent).
    2. Validate ``state.solve.model_solve`` (exactly one model, ≥1 solve).
    3. Reset ``state.solve.roll_counter`` for repeatable test runs (R-O5).
    4. Drive flextool's ``orchestration.run_model`` with a polar_high
       cascade solver — each per-solve iteration loads the snapshot via
       ``load_flextool``, builds the LP via ``build_flextool``, solves
       via HiGHS, captures the handoff, and deposits it into
       ``state.handoffs`` (which the consume side already reads from).

    Parameters
    ----------
    state : RunnerState
        Native polar_high state carrier.  ``state.solve`` and
        ``state.timeline`` must be populated (call
        :func:`run_chain_from_db` for the canonical end-to-end path that
        sets these up from a DB).  The function may flip
        ``state.handoffs`` from ``None`` to ``{}`` to enable the in-memory
        capture/consume path; this is done unconditionally — the native
        orchestrator always uses in-memory handoff (storage-fixing falls
        through to the file-copy path only when ``state.handoffs`` is
        explicitly set to ``None`` after this returns).
    work_folder : Path | str
        Directory the snapshot tree lives under.  Created if missing.
        Per-solve preprocessing CSVs are emitted under
        ``work_folder/solve_data/``.
    runner_factory : callable | None
        Optional override for constructing the underlying
        :class:`FlexToolRunner` — used by tests that want to short-
        circuit flextool's preprocessing.  Default uses the canonical
        constructor.
    warm : bool, default False
        Δ.12d — when True, attempt warm LP updates between consecutive
        structurally-compatible per-solve iterations using
        :class:`polar_high.WarmProblem`.  Reuses one WarmProblem across
        the cascade, applying ``_apply_warm_updates`` between solves
        and falling back to a cold rebuild whenever the structural
        fingerprint changes or any unmapped Param differs.  Decisions
        are recorded per-step on :attr:`OrchestrationStep.warm_used`.
        Default ``False`` preserves the original cold-rebuild
        behaviour.

    Returns
    -------
    dict[str, OrchestrationStep]
        Mapping ``complete_solve_name → OrchestrationStep`` in solve
        order (Python dict insertion order preserved).

    Raises
    ------
    FlexToolConfigError
        Empty / multi-model ``model_solve``.
    FlexToolSolveError
        Any per-solve LP infeasibility / non-optimal status.
    """
    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    state.paths = PathConfig(work_folder=work_folder)

    logger = state.logger
    _bootstrap_dirs(work_folder, logger)
    solves = _validate_model_solve(state)

    # Reset roll-counter so repeated calls with the same SolveConfig
    # don't desync.  R-O5 in the orchestration risk register.
    state.solve.roll_counter = state.solve.make_roll_counter()

    # Always enable in-memory handoff for the native path.
    if state.handoffs is None:
        state.handoffs = {}

    # Stash the ``csv_dump`` flag on the state so per-iter sites in
    # ``_drive_cascade`` can consult it (gates ``data.dump_csvs``).
    state.csv_dump = bool(csv_dump)  # type: ignore[attr-defined]

    # The cascade solver runs polar_high on each solve and captures the
    # handoff.  We use flextool's orchestration loop driver because it
    # encodes the recursive/rolling/stochastic expansion + per-solve
    # preprocessing chain we still consume.  Our cascade solver is the
    # `solver.run(...)` callback inside that loop.
    return _drive_cascade(state, work_folder, solves, runner_factory,
                          db_url=db_url, scenario_name=scenario_name,
                          warm=warm, keep_solutions=keep_solutions)


def _drive_cascade(
    state: RunnerState,
    work_folder: Path,
    solves: list[str],
    runner_factory,
    *,
    db_url: str | None = None,
    scenario_name: str | None = None,
    keep_solutions: bool = False,
    warm: bool = True,
) -> dict[str, OrchestrationStep]:
    """Drive the flextool master loop with a polar_high cascade solver.

    For every per-solve iteration:

    1. Read the snapshot via ``load_flextool``.
    2. Build the LP via ``build_flextool`` (cold rebuild) OR warm-update
       the prior iteration's :class:`polar_high.WarmProblem`.
    3. Solve via HiGHS.
    4. Build the handoff via ``build_handoff_from_solution``.
    5. Deposit it into ``state.handoffs`` so the next iteration's
       preprocessing picks it up.

    Parameter ``warm`` toggles per-iteration warm-LP updates: when True,
    the cascade reuses one ``WarmProblem`` across consecutive
    structurally-compatible iterations.  See
    :mod:`flextool.engine_polars._warm` for the structural-fingerprint
    + Param-classification machinery.  Cold rebuild (``warm=False``)
    remains the default for backward compatibility with every existing
    caller.

    Emits an :class:`OrchestrationStep` per solve and runs the LAST
    solve too (the polar_high-side bookkeeping is the deliverable,
    not just an intermediate).
    """
    # Late imports — keep the orchestration module's import surface narrow
    # for callers that only need the dataclass.
    from flextool.engine_polars._solver_base import SolverRunner
    from flextool.engine_polars._native_run_model import native_run_model

    from polar_high import Problem, WarmProblem
    from flextool.engine_polars.input import (
        build_handoff_from_solution,
        load_flextool,
    )
    from flextool.engine_polars.model import build_flextool
    from flextool.engine_polars._output_writer import (
        OutputWriterState,
        write_outputs_for_solve,
    )
    from flextool.engine_polars._warm import (
        _IncompatibleUpdate,
        _apply_warm_updates,
        _build_warm_problem,
        _fingerprint,
    )

    results: dict[str, OrchestrationStep] = {}
    # Δ.1: adapter that reuses flextool's process_outputs writers.  The
    # state carrier collects ``periods_already_emitted`` across the
    # cascade so we don't have to round-trip through SolveHandoff.
    writer_state = OutputWriterState()

    # The runner_factory hook lets tests inject a mock; the default uses
    # FlexToolRunner constructed against the same DB the state was
    # loaded from.  Since RunnerState doesn't carry a DB URL
    # by default, callers must supply this via runner_factory or use
    # ``run_chain_from_db`` which constructs the runner explicitly.
    if runner_factory is None:
        raise FlexToolConfigError(
            "run_orchestration requires a runner_factory to construct "
            "the underlying FlexToolRunner.  Use run_chain_from_db for "
            "the canonical end-to-end path that wires this for you."
        )
    _drive_rec = get_phase_recorder()
    _drive_logger = state.logger
    runner = runner_factory()
    if _drive_rec is not None:
        _drive_rec.checkpoint(
            "flextool_runner_constructed", _drive_logger,
            user_label="FlexToolRunner constructed",
        )
    # Push our state's handoff slot onto the runner's state so the
    # cascade and any consume hooks share the same dict.
    runner.state.handoffs = state.handoffs
    # Per-level Provider cache (Design A).  ``native_run_model`` lazily
    # initialises this on first iter, but seeding it here makes the
    # invariant ``state._level_providers is dict`` explicit at every
    # entry point (cascade + fast_load) instead of relying on hasattr
    # probes downstream.
    runner.state._level_providers = {}
    # Step 2.5 — forward the cascade-input Provider seeded in
    # ``run_chain_from_db`` onto runner.state so the per-sub-solve hook
    # at :mod:`flextool.engine_polars._native_run_model` (line 365-370)
    # picks it up.  ``None`` is allowed for entry points that bypass
    # ``run_chain_from_db`` — the hook then builds an empty Provider.
    _cip = getattr(state, "cascade_input_provider", None)
    if _cip is not None:
        runner.state.cascade_input_provider = _cip
    # Phase 5c — forward the engine_polars-side ``override_provider``
    # callable onto ``runner.state`` so the per-sub-solve hook in
    # :mod:`flextool.engine_polars._native_run_model` (Phase 5b) picks
    # it up.  ``None`` keeps the no-override default.
    _op = getattr(state, "override_provider", None)
    if _op is not None:
        runner.state.override_provider = _op
    runner.state.logger.setLevel(logging.ERROR)
    # Forward the opt-in memory recorder (no-op when env var unset) so
    # ``_PolarHighCascadeSolver.run`` can fire the first-iter checkpoints.
    runner.state._memory_recorder = getattr(  # type: ignore[attr-defined]
        state, "_memory_recorder", _NoopMemoryRecorder()
    )

    # Δ.12c — build a SpineDbReader once and reuse it across the cascade.
    # When db_url + scenario_name are supplied (run_chain_from_db wires
    # them), the override chain fires for every per-solve load — covering
    # the seeds the workdir CSV path can't provide once Δ.12-drop /
    # Δ.12c have retired the redundant CSVs.  When the caller didn't
    # supply them, we fall back to load_flextool's per-call autoresolve
    # (which works for fixtures whose work_folder follows the
    # ``work_<scenario>`` convention).
    cascade_db_reader = None
    if db_url is not None and scenario_name is not None:
        from flextool.engine_polars._spinedb_reader import SpineDbReader
        # Phase 4.6 — thread axis_enums + contract from the cascade
        # provider if available so the reader casts on emit.
        _cip_for_reader = getattr(state, "cascade_input_provider", None)
        _cascade_axis_enums = getattr(_cip_for_reader, "axis_enums", None) \
            if _cip_for_reader is not None else None
        _cascade_contract = getattr(_cip_for_reader, "contract", None) \
            if _cip_for_reader is not None else None
        try:
            cascade_db_reader = SpineDbReader(
                db_url, scenario=scenario_name,
                axis_enums=_cascade_axis_enums,
                contract=_cascade_contract,
            )
        except Exception:  # noqa: BLE001
            cascade_db_reader = None
        if _drive_rec is not None:
            _drive_rec.checkpoint(
                "cascade_spinedb_reader_constructed", _drive_logger,
                user_label="Inputs prepared",
            )

        # ``input/p_all_entity_unitsize`` — solve-invariant entity-unitsize
        # cascade (virtual_unitsize OR existing OR 1000.0) over ALL entities
        # (unit ∪ node ∪ connection).  Computed ONCE here against the
        # whole-model (unfiltered-scenario) ``cascade_db_reader`` and seeded
        # into the cascade-input Provider, which every per-sub-solve Provider
        # copies frame-by-frame (see ``_native_run_model`` seed loop).  This
        # (a) avoids the per-roll recompute in ``apply_derived_b`` and
        # (b) gives the output reader a complete carrier covering every
        # entity — including invest candidates absent from the LAST solve's
        # pss/invest sets — so ``read_parameters`` no longer KeyErrors on
        # such entities.  ``_cip`` is the same Provider object forwarded onto
        # ``runner.state.cascade_input_provider`` above and read back by the
        # cascade as ``cascade_input_provider``.
        if cascade_db_reader is not None and _cip is not None:
            try:
                from flextool.engine_polars._derived_params import (
                    _entity_unitsize_lf,
                )
                _all_us_df = (
                    _entity_unitsize_lf(cascade_db_reader)
                    .rename({"us": "value"})
                    .collect()
                )
                if _all_us_df.height > 0:
                    _cip.put("input/p_all_entity_unitsize", _all_us_df)
            except Exception:  # noqa: BLE001
                # Non-fatal: the per-solve ``apply_derived_b`` fallback
                # recomputes ``p_all_entity_unitsize`` from ``source`` and
                # the output reader falls back to its reconstruction path.
                pass

    class _PolarHighCascadeSolver(SolverRunner):
        def __init__(self, runner_state):
            super().__init__(runner_state)
            self._all_steps: dict[str, OrchestrationStep] = results
            # Δ.12d — warm-LP carry-over state.  ``_warm_problem`` holds
            # the live :class:`polar_high.WarmProblem` reused across
            # consecutive structurally-compatible iterations; ``_prior_data``
            # / ``_prior_fp`` snapshot the previous iteration's FlexData +
            # fingerprint for the diff scan in
            # :func:`_apply_warm_updates`.  All three stay None when
            # ``warm=False`` (the existing cold-cascade behaviour) AND
            # are reset to None on every cold rebuild.
            self._warm_problem: "WarmProblem | None" = None
            self._prior_data = None
            self._prior_fp: "tuple | None" = None
            # Per-iter slim of the PRIOR step's parked Solution — see the
            # block just before ``self._all_steps[step_key] = ...`` in
            # :meth:`run`.  Tracks the step_key parked on the previous
            # iter so we can null its heavy ``_vars`` + ``highs`` once the
            # per-iter writers and ``build_handoff_from_solution`` have
            # finished consuming it.  Bounds peak RSS during the cascade
            # — without this, every iter's full ``Var.frame`` dataframe
            # set stays parked until the post-loop slim at the bottom of
            # :func:`_native_run_model`, which on multi-roll runs is too
            # late (storage→dispatch OOMs).
            self._prev_step_key: "str | None" = None
            # Phase 2 — per-step level_key sidecar.  Populated when
            # parking each step so the warm-path "keep one
            # ``Solution.highs`` + one ``flex_data_provider`` per level"
            # slim can iterate prior steps and resolve their level.
            # Keyed by the same ``step_key`` used in ``self._all_steps``.
            self._step_level_keys: "dict[str, tuple]" = {}
            # Per-base-solve gating for the scaling CSV.  The CSV value
            # (effective_obj_scale) is invariant across rolls of the same
            # base solve, so we track which base solve names already have
            # ``scale_the_objective.csv`` written and skip subsequent
            # rolls.  Phase 2b dropped the legacy diagnostic TXT report
            # (``FLEXTOOL_SCALING_REPORT=1``); the autoscaler's per-solve
            # YAML report (``solve_data/autoscale_<solve>.yaml``) is the
            # replacement and is gated inside
            # :func:`_autoscale_emit_layer1`.
            self._scale_csv_written: set[str] = set()
            # Autoscale console summary dedup: one line per base solve.
            # Layer 1/2/3 decisions are identical across rolls of the
            # same base solve, so the operator-facing summary fires once.
            self._autoscale_summary_emitted: set[str] = set()
            # Warm-path autoscale plan cache: Layer 2 writes side
            # vectors on the Problem at first-build; the WarmProblem's
            # canonical matrix bakes them in (and ``_param_cells``
            # caches the scaled factors for tracked Params).  The plan
            # stays valid across subsequent ``_apply_warm_updates``
            # reuses because ``WarmProblem.update_param`` updates HiGHS
            # cells via the cached factors — no re-scaling, no plan
            # re-evaluation.  The plan is needed on every warm solve so
            # :func:`_autoscale_unscale_post_solve` can restore the
            # solution to physical coordinates.  Cleared whenever
            # ``self._warm_problem`` is dropped.
            self._autoscale_warm_layer2_plan: "_AutoscaleLayer2Plan | None" = None
            # Cache of the pre-Layer-2 RangeReport from the warm first
            # build, surfaced as ``Solution.streamed_lp_ranges`` after
            # every warm solve so downstream telemetry (the LP-bound-
            # range smoke in ``test_invest_chain_regression``, the
            # autoscale Layer-1 YAML) sees the same four (min, max)
            # pairs the autoscaler decided on.  Mirrors the cold path's
            # post-``run_one_solve`` ``sol.streamed_lp_ranges = …``
            # assignment (see the longer comment around the cold-path
            # write below); dropped together with the warm problem.
            self._autoscale_warm_ranges_pre: (
                "_AutoscaleRangeReport | None"
            ) = None
            # Per-structural-shape autoscale DECISION cache.  Keyed by the
            # BUILT LP's structural signature
            # (:func:`_autoscale_lp_shape_signature` — matrix shape +
            # per-family layout, scoped by base solve name), which stays
            # invariant across rolls of a rolling solve where
            # ``_fingerprint(data)`` would slide; each
            # value is an :class:`_AutoscaleShapeCacheEntry` carrying the
            # Layer-2 exponents, the Layer-3 plan, and the pre/post
            # RangeReports computed on the FIRST solve of that shape.  On
            # every subsequent same-shape solve (notably the COLD-rebuild-
            # per-roll path where ladder Params force a cold rebuild yet
            # the matrix shape is invariant) the cached decision is
            # re-applied via :func:`_autoscale_apply_layer2_from_cache` /
            # :func:`_autoscale_apply_layer3_from_cache` WITHOUT any
            # ``detect_ranges`` / ``bucket_coefficients`` traversal — that
            # traversal was the source of the per-roll multi-GB transient
            # peaks (the autoscale memory pyramid).  Disable with
            # ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE=1`` (always recompute).
            self._autoscale_shape_cache: (
                "dict[tuple, _AutoscaleShapeCacheEntry]"
            ) = {}
            # v60 per-solve decomposition — complete-solve names that ran
            # under ``decomposition=lagrangian``.  Used by the consume-side
            # guard in :meth:`run` to raise loudly if a downstream solve
            # would try to consume a Lagrangian solve's (absent) handoff —
            # cross-scheme handoff is a deferred follow-up.
            self._lagrangian_solve_names: set[str] = set()

        def _run_lagrangian_solve(
            self,
            complete_solve_name: str,
            base_solve_name: str,
            data,
        ) -> int:
            """Run *complete_solve_name* via the Lagrangian region driver.

            Selected per solve from ``solve.decomposition = lagrangian``.
            Decomposes over the groups whose
            ``group.decomposition_method`` is ``lagrangian_region`` (the
            ``decomp_<REG>`` groups the PLEXOS-to-FlexTool writer emits),
            using the per-solve knobs resolved by
            :meth:`SolveConfig.lagrangian_config_for`.

            First-cut scope (per the decomposition spec): the solve runs
            and reports convergence/objective, deposits an
            :class:`OrchestrationStep`, and is recorded in
            ``self._lagrangian_solve_names`` so the consume-side guard in
            :meth:`run` can fire.  It does NOT deposit a ``SolveHandoff``
            into ``state.handoffs`` — threading a Lagrangian solve's
            results into a downstream monolithic solve is a deferred
            follow-up, guarded loudly rather than silently dropped.
            """
            from flextool.engine_polars._lagrangian import solve_lagrangian
            from flextool.decomposition.region_filter import (
                discover_decomposition_regions_from_db,
            )

            # Regions are discovered from the DB (the same source the
            # group-level decomposition_method lives in).  The DB-driven
            # run path (run_chain_from_db) always supplies ``db_url``; a
            # caller that bypasses it cannot resolve the region groups, so
            # fail with an actionable message rather than guess.
            if db_url is None:
                raise FlexToolConfigError(
                    f"Solve '{base_solve_name}' requests "
                    f"decomposition=lagrangian but the run was started "
                    f"without a database URL (db_url is None). Lagrangian "
                    f"region decomposition is only available on the "
                    f"DB-driven run path (run_chain_from_db)."
                )
            regions = discover_decomposition_regions_from_db(db_url)
            if len(regions) < 2:
                raise FlexToolConfigError(
                    f"Solve '{base_solve_name}' requests "
                    f"decomposition=lagrangian but the model declares "
                    f"{len(regions)} group(s) with "
                    f"decomposition_method='lagrangian_region' "
                    f"({regions or '(none)'}); at least two region groups "
                    f"are required."
                )

            alpha, max_iter, tol = state.solve.lagrangian_config_for(
                base_solve_name
            )
            # Lagrangian decomposition is HiGHS-only; pass the solve's
            # SolverConfig so the driver can raise an actionable error if
            # the user selected a commercial solver.
            solver_cfg = state.solve.solver_configs.get(
                complete_solve_name
            ) or state.solve.solver_configs.get(base_solve_name)

            self.state.logger.warning(
                "Solve '%s': decomposition=lagrangian over %d region "
                "groups %s (alpha=%g, max_iter=%d, tol=%g).",
                base_solve_name, len(regions), regions, alpha, max_iter, tol,
            )
            result = solve_lagrangian(
                data,
                work_dir=self.state.paths.work_folder,
                regions=regions,
                alpha=alpha,
                max_iters=max_iter,
                tol=tol,
                solver_config=solver_cfg,
            )
            _conv_msg = (
                "Solve '%s': Lagrangian decomposition %s after %d "
                "iterations, total_objective=%.6g."
            )
            if result.converged:
                self.state.logger.warning(
                    _conv_msg, base_solve_name, "converged",
                    result.iterations, result.total_objective,
                )
            else:
                # Loud, not silent: non-convergence within max_iters is a
                # result the operator must see, but we don't abort the
                # chain over it (the decomposition still produced a point).
                self.state.logger.error(
                    _conv_msg, base_solve_name, "did NOT converge",
                    result.iterations, result.total_objective,
                )

            # Record so the consume-side guard fires for any downstream
            # solve, and deposit a slim OrchestrationStep (no Solution /
            # SolveHandoff — the Lagrangian result is not yet threaded
            # into the monolithic output / handoff pipeline).
            self._lagrangian_solve_names.add(complete_solve_name)
            self._all_steps[complete_solve_name] = OrchestrationStep(
                solve_name=complete_solve_name,
                solution=None,
                handoff=None,
                obj=result.total_objective,
                optimal=result.converged,
                warm_used=False,
                flex_data=data,
                flex_data_provider=getattr(
                    self.state, "current_provider", None,
                ),
            )
            self._prev_step_key = complete_solve_name
            return 0

        def run(self, complete_solve_name: str) -> int:
            _phase_prof("run_enter")
            # Cross-level eviction — release any EXHAUSTED prior solve-level's
            # live HiGHS instance + flex_data_provider BEFORE this solve builds
            # its FlexData/LP, so two level footprints never coexist (the DES
            # storage+dispatch ≈ 2x peak that drove the 7/9 near-OOM).  This
            # hoists the post-solve slim's exhausted-level branch
            # (``:2632-2647``) ahead of the allocation instead of running it
            # after — the prior level's per-iter writers + handoff already
            # consumed its solution on its own iter, so releasing here is safe.
            # A level is "exhausted" when no upcoming iter shares its level_key
            # and it is not the level THIS solve belongs to; same-level steps
            # are kept for warm reuse.  ``self._warm_problem`` for an exhausted
            # level was already nulled at the level boundary
            # (_native_run_model.py:497-505), so nulling the step's
            # ``solution.highs`` here drops the last reference and frees it.
            # malloc_trim reclaims the HiGHS (glibc) heap; the polars-side
            # provider is freed by dropping the Python ref.
            if not keep_solutions:
                _ilk = getattr(self.state, "_all_level_keys", ())
                _iidx = getattr(self.state, "_current_iter_index", None)
                _tlvl = getattr(self.state, "_current_level_key", None)
                _upcoming = (
                    set(_ilk[_iidx + 1:])
                    if (_iidx is not None and _ilk) else set()
                )
                # Disable knob for A/B peak-memory measurement and the
                # regression test's negative control (mirrors the
                # ``POLAR_HIGH_DISABLE_PRUNE_DOWN`` style escape hatch).
                if os.environ.get("FLEXTOOL_DISABLE_XLEVEL_RELEASE") != "1":
                    _released = False
                    for _k, _step in (getattr(self, "_all_steps", None) or {}).items():
                        _lvl = self._step_level_keys.get(_k)
                        if _lvl is None or _lvl == _tlvl or _lvl in _upcoming:
                            continue  # current level or still-upcoming: keep
                        _sol = getattr(_step, "solution", None)
                        if _sol is not None and getattr(_sol, "highs", None) is not None:
                            _sol.highs = None
                            _released = True
                        if getattr(_step, "flex_data_provider", None) is not None:
                            _step.flex_data_provider = None
                            _released = True
                        # Also evict the exhausted level's entry from the
                        # per-level FlexDataProvider cache — it is keyed by
                        # level_key and only reused by FUTURE same-level rolls,
                        # of which an exhausted level has none.  Without this
                        # the cache (``state._level_providers``) pins the
                        # level's polars FlexData for the whole cascade even
                        # after the step ref above is dropped.
                        _lp = getattr(self.state, "_level_providers", None)
                        if isinstance(_lp, dict) and _lp.pop(_lvl, None) is not None:
                            _released = True
                    if _released:
                        _try_malloc_trim()
                # Cross-level retention audit — runs AFTER the eviction above so
                # a correct release records zero violators.  Gated by the same
                # ``not keep_solutions`` as the eviction: the release invariant
                # only applies when slimming is active (``keep_solutions=True``
                # deliberately retains every level's solution).
                _audit_prior_level_release(
                    steps=getattr(self, "_all_steps", None),
                    step_level_keys=getattr(self, "_step_level_keys", {}),
                    all_level_keys=getattr(self.state, "_all_level_keys", ()),
                    iter_idx=getattr(self.state, "_current_iter_index", None),
                    this_level=getattr(self.state, "_current_level_key", None),
                    complete_solve_name=complete_solve_name,
                )
            # Optional per-iter phase-timing (opt-in via env var).  Emits
            # `per_iter` rows to the workdir's timings.csv covering
            # lp_build / solve / handoff and a warm_used marker.  See
            # specs/warm_start_phase_breakdown_handoff.md.
            _phase_timing = (
                os.environ.get("FLEXTOOL_PHASE_TIMING") == "1"
                and getattr(self.state, "timing_recorder", None) is not None
            )
            _tr = self.state.timing_recorder if _phase_timing else None
            _roll_idx = getattr(self.state, "current_roll_index", "")
            if _roll_idx is None:
                _roll_idx = ""
            _t_build_start = time.perf_counter() if _phase_timing else 0.0
            # Δ.12 — wire ``handoff=`` through ``load_flextool`` so the
            # in-memory carriers from the prior solve flow into this
            # solve's FlexData directly.  Replaces the previous
            # implicit dependency on flextool's per-solve preprocessing
            # rewriting ``solve_data/p_entity_*.csv`` between solves.
            # After Δ.12 the cascade reads these five carrier-derived
            # fields from the in-memory ``SolveHandoff`` rather than the
            # workdir CSVs:
            #
            #   * ``p_entity_invested``
            #   * ``p_entity_divested``
            #   * ``p_entity_previously_invested_capacity``
            #   * ``p_roll_continue_state``
            #   * ``p_fix_storage_quantity``
            prior_for_load = (
                self.state.handoffs.get(self.state.last_captured_solve)
                if self.state.last_captured_solve is not None else None
            )
            _sub_solve_provider = getattr(
                self.state, "current_provider", None,
            )
            data = load_flextool(
                self.state.paths.work_folder,
                handoff=prior_for_load,
                db_reader=cascade_db_reader,
                provider=_sub_solve_provider,
            )
            # Release heap held by the broadcast cascade scratch frames.
            # On H2_trade y2050 this drops RSS ~1.6 GB / 41 %; expected
            # to scale with timeline size.  No-op on non-glibc.
            _try_malloc_trim()
            # Memory checkpoint — fires on level-boundary iters (the
            # last roll of a roll group) so the recorded delta aggregates
            # across all rolls in the group.  ``_native_run_model`` sets
            # the flag before each ``solver.run()`` call.
            _emit_phase = bool(getattr(
                self.state, "emit_phase_checkpoints_this_iter", False,
            ))
            _memrec_local = getattr(self.state, "_memory_recorder", None)
            if _memrec_local is not None and _emit_phase:
                _memrec_local.checkpoint(
                    "load_flextool_end", self.state.logger,
                    user_label="FlexData built",
                )

            # --- LP scaling -------------------------------------------------
            # Phase 2b — the legacy ``scaling.analyze_solve`` /
            # ``ScaleTable`` / ``resolve_effective_scaling`` pipeline has
            # been retired in favour of the autoscale package; the
            # cascade now resolves the per-solve effective objective
            # scale directly from the user's DB override (defaulting to
            # the legacy 1e-6 when absent) and lets autoscale Layer 3
            # handle residual cost / bound magnitudes inside HiGHS.
            base_solve_name = re.sub(r"_roll_\d+$", "", complete_solve_name)

            # v60 per-solve decomposition routing -----------------------
            # Consume-side guard: if the immediately-preceding captured
            # solve ran under decomposition=lagrangian and THIS solve is a
            # different base solve, it would consume the (intentionally
            # absent) Lagrangian handoff.  Cross-scheme handoff
            # (Lagrangian → monolithic dispatch) is a deferred follow-up,
            # so we fail loudly rather than silently load handoff=None.
            _prev_captured = self.state.last_captured_solve
            if (
                _prev_captured is not None
                and _prev_captured in self._lagrangian_solve_names
                and base_solve_name
                != re.sub(r"_roll_\d+$", "", _prev_captured)
            ):
                raise FlexToolConfigError(
                    f"Solve '{base_solve_name}' follows '{_prev_captured}', "
                    f"which ran under decomposition=lagrangian, and would "
                    f"consume its results. Cross-scheme handoff "
                    f"(Lagrangian solve feeding a downstream solve) is not "
                    f"yet supported. Make the downstream solve lagrangian "
                    f"too, or order the chain so the Lagrangian solve is "
                    f"terminal."
                )
            # When this solve resolves to decomposition=lagrangian, run it
            # through the Lagrangian region coordinator instead of building
            # and solving a monolithic LP.
            if state.solve.decomposition_for(base_solve_name) == "lagrangian":
                return self._run_lagrangian_solve(
                    complete_solve_name, base_solve_name, data,
                )

            user_obj_scale = state.solve.scale_the_objective.get(complete_solve_name)
            effective_obj_scale = _resolve_effective_obj_scale(user_obj_scale)
            # ``user_bound_scale`` resolution priority:
            # ``FLEXTOOL_USER_BOUND_SCALE`` env var (set by
            # ``--user-bound-scale`` CLI flag) > DB ``solve.user_bound_scale``
            # > autoscale Layer 3's automatic recommendation > HiGHS'
            # own internal scaling.  HiGHS' "Consider setting the
            # user_bound_scale option to <N>" warning still prints a
            # value if any case slips through Layer 3; pass it via
            # ``--user-bound-scale``.
            _cli_ubs = os.environ.get("FLEXTOOL_USER_BOUND_SCALE")
            user_bound_scale_override = _resolve_user_bound_scale_override(
                _cli_ubs if _cli_ubs is not None
                else state.solve.user_bound_scale.get(complete_solve_name)
            )
            # Resolve the autoscaler's mode once for this sub-solve so the
            # baseline-options builder, the cold/warm LP construction, and
            # the Layer 2 / Layer 3 helpers all see the same value.  Cascade-
            # internal call sites read ``FLEXTOOL_SCALING`` from env.
            _scaling_cfg = _autoscale_resolve_config(None)
            _scaling_mode = _scaling_cfg.mode

            # HiGHS solver options.  ``simplex_scale_strategy`` =
            # advanced (Curtis-Reid) is always-on; ``user_bound_scale``
            # is only emitted when explicitly requested via
            # ``--user-bound-scale`` CLI / DB override (see priority
            # block above).  Default: HiGHS does its own scaling.
            # Cap solve time via env var if the operator requested it.
            _diag_tlim = os.environ.get("FLEXTOOL_HIGHS_TIME_LIMIT")
            # Allow operator to override HiGHS ``mip_rel_gap`` via
            # ``--solver-mip-gap GAP`` CLI flag (env-var-plumbed).  Only
            # bites MIP solves; pure-LP solves ignore it.
            _cli_mip_gap = os.environ.get("FLEXTOOL_HIGHS_MIP_GAP")
            # Allow operator to override HiGHS ``presolve`` via
            # ``--presolve {on,off,choose}`` CLI flag (env-var-plumbed).
            _cli_presolve = os.environ.get("FLEXTOOL_HIGHS_PRESOLVE")
            # Allow operator to override HiGHS ``threads`` via
            # ``--highs-threads N`` CLI flag (env-var-plumbed).  N > 1
            # flips ``parallel`` to ``on`` and trades determinism for
            # wall-clock speedup; N == 1 keeps the DETERMINISM_OPTIONS
            # pinning intact.  Resolving once per ``_finalise_highs_options``
            # call (= once per sub-solve) ensures every Highs instance in
            # the process sees the same value, sidestepping HiGHS'
            # "global scheduler already initialised" rejection path.
            _cli_threads = os.environ.get("FLEXTOOL_HIGHS_THREADS")
            # ``--solver-log-level {silent,normal,verbose}`` CLI flag,
            # env-var-plumbed.  Replaces the v55-era ``solve.solver_log_level``
            # DB knob removed in Batch C.7.  ``silent`` flips HiGHS'
            # ``output_flag`` off; ``verbose`` additionally bumps
            # ``log_dev_level=2`` for per-iteration solver telemetry.
            _cli_log_level = os.environ.get("FLEXTOOL_SOLVER_LOG_LEVEL")

            def _build_cli_overrides() -> dict[str, object]:
                """Translate CLI env-var-plumbed flags into a HiGHS
                options dict that the effective-options resolver layers
                on top of ``solver_arguments`` and ``highs.opt``.
                """
                cli: dict[str, object] = {}
                if _diag_tlim:
                    try:
                        cli["time_limit"] = float(_diag_tlim)
                    except ValueError:
                        pass
                if _cli_mip_gap:
                    try:
                        cli["mip_rel_gap"] = float(_cli_mip_gap)
                    except ValueError:
                        pass
                if _cli_presolve in ("on", "off", "choose"):
                    cli["presolve"] = _cli_presolve
                if _cli_threads is not None:
                    try:
                        n = int(_cli_threads)
                    except ValueError:
                        n = 1
                    if n > 1:
                        cli["threads"] = n
                        # User opted out of the determinism pin; HiGHS
                        # needs ``parallel="on"`` before it will actually
                        # use the threads.
                        cli["parallel"] = "on"
                    # n == 1 (or n <= 0) keeps the deterministic defaults
                    # from DETERMINISM_OPTIONS — no override needed.
                if _cli_log_level == "silent":
                    cli["output_flag"] = False
                elif _cli_log_level == "verbose":
                    cli["output_flag"] = True
                    cli["log_dev_level"] = 2
                elif _cli_log_level == "normal":
                    cli["output_flag"] = True
                # Anything else (None, unknown) leaves HiGHS defaults
                # standing — same as the pre-C.7 behaviour where the
                # DB-side knob fed nothing.
                return cli

            # Per-solve ``solver_arguments`` 1d-map (Batch C.1).  Empty
            # dict when no entry authored on the active solve.
            _solver_args_map = state.solve.solver_settings.arguments.get(
                complete_solve_name, {}
            )
            # ``solver_config/highs.opt`` floor parsed by the resolver.
            # ``state.paths.solver_config_dir`` is None on direct native
            # callers (the file is only present on the CLI path); the
            # resolver treats that as an empty floor.
            _highs_opt_path = (
                state.paths.solver_config_dir / "highs.opt"
                if state.paths.solver_config_dir is not None
                else None
            )

            # --- LP build & solve ------------------------------------------
            # Δ.12d — warm-LP per-iteration decision.  When ``warm`` is
            # True AND the prior iteration left a live WarmProblem whose
            # fingerprint matches this iteration's data, we push the
            # Param diff into the live LP.  Any ``_IncompatibleUpdate``
            # (unmapped Param differs, gate transitions, …) drops back
            # to a cold rebuild.  Cold rebuild also fires on the first
            # iteration and on any structural fingerprint mismatch.
            #
            # Phase 3 — warm-LP is a HiGHS-only design (polar-high's
            # WarmProblem wraps a single live HiGHS instance).  When the
            # active solve picks a commercial solver we disable warm
            # reuse for this iteration, log a one-time warning, and
            # cold-rebuild + dispatch through ``run_one_solve``.
            from flextool.engine_polars._solve_config import (
                SolverConfig as _SolverConfig,
            )
            _active_solver_cfg = state.solve.solver_configs.get(
                complete_solve_name, _SolverConfig()
            )
            _warm_disabled_by_solver = (
                warm and _active_solver_cfg.name != "highs"
            )
            # ``FLEXTOOL_SAVE_MEMORY=1`` opts into polar-high's
            # ``save_memory=True`` solve path, which drops the polar-side
            # LP source and round-trips the HiGHS instance through MPS
            # mid-solve.  The Problem is then in a "released" state and
            # can no longer be warm-reused, so every iteration must
            # cold-rebuild.  Resolve once per sub-solve (cheap) so the
            # knob can be toggled between runs of the same cascade.
            _save_memory = os.environ.get("FLEXTOOL_SAVE_MEMORY") == "1"
            _warm_disabled_by_save_memory = warm and _save_memory
            if _warm_disabled_by_save_memory and not getattr(
                self, "_warm_disabled_by_save_memory_warned", False
            ):
                state.logger.warning(
                    _wrap_log_prose(
                        "FLEXTOOL_SAVE_MEMORY=1: warm-LP reuse disabled; "
                        "every sub-solve will cold-rebuild, write MPS, and "
                        "dispatch to a subprocess HiGHS. Expect ~+30-60 s "
                        "I/O per sub-solve in exchange for HiGHS' "
                        "active-solve memory living outside this Python "
                        "process."
                    ),
                )
                self._warm_disabled_by_save_memory_warned = True
            if _warm_disabled_by_solver and not getattr(
                self, "_warm_disabled_warned", False
            ):
                state.logger.warning(
                    _wrap_log_prose(
                        f"warm-start is unavailable for solver "
                        f"{_active_solver_cfg.name!r}; falling back to cold "
                        f"rebuilds per sub-solve, expect slower per-iter "
                        f"wall-clock."
                    ),
                )
                self._warm_disabled_warned = True
            # HiGHS soft-promote: warm=False on HiGHS without
            # FLEXTOOL_SAVE_MEMORY=1 used to fall through to an in-
            # process cold rebuild that built a fresh ``highspy.Highs``
            # inside this Python process — undoing the entire reason
            # warm reuse exists in the first place (peak RSS).  Retired
            # path: route every HiGHS cold solve through the same
            # ``cmd_solve_mps`` subprocess the save-memory branch uses,
            # bounding peak RSS to ``write_mps``'s footprint.  Mutate
            # only the local ``_save_memory`` — do NOT touch the env
            # var, which would leak to sibling solves.
            if (
                (not warm)
                and _active_solver_cfg.name == "highs"
                and not _save_memory
            ):
                if not getattr(
                    self, "_warm_disabled_softpromote_warned", False,
                ):
                    state.logger.warning(
                        _wrap_log_prose(
                            "Warm reuse disabled (warm=False); HiGHS solve "
                            "will route through the cmd_solve_mps "
                            "subprocess to bound memory footprint.  Set "
                            "FLEXTOOL_SAVE_MEMORY=1 explicitly to silence "
                            "this warning."
                        ),
                    )
                    self._warm_disabled_softpromote_warned = True
                _save_memory = True
            warm_used = False
            warm_active = (
                warm
                and not _warm_disabled_by_solver
                and not _warm_disabled_by_save_memory
            )
            if warm_active:
                fp = _fingerprint(data)
                tried_warm = (
                    self._warm_problem is not None
                    and self._prior_data is not None
                    and self._prior_fp == fp
                )
                if tried_warm:
                    try:
                        _apply_warm_updates(self._warm_problem,
                                            self._prior_data, data)
                        warm_used = True
                    except _IncompatibleUpdate as _warm_exc:
                        # Drop the stale warm problem so the next
                        # branch builds a fresh one.  The cached Layer 2
                        # plan dies with it — the next first-build will
                        # regenerate it from the fresh LP.
                        #
                        # Diagnostic: surface WHICH condition forced the
                        # cold rebuild.  On rolling cascades the ladder
                        # Params (commit 7b5ccb3e) trip this every roll,
                        # which re-runs the full pre-solve autoscale
                        # traversal (the between-solves memory pyramid).
                        # The exception message names the offending
                        # Param / reason; log it at WARNING so a single
                        # run pins the cause without tracemalloc.
                        state.logger.warning(
                            "warm reuse fell back to COLD REBUILD for %s "
                            "(re-runs pre-solve autoscale traversal): %s",
                            complete_solve_name, _warm_exc,
                        )
                        self._warm_problem = None
                        self._autoscale_warm_layer2_plan = None
                        self._autoscale_warm_ranges_pre = None
                if not warm_used:
                    # Cross-level eviction — same-level COLD-rebuild case
                    # (e.g. ladder period switch, 7b5ccb3e).  We are about to
                    # build a fresh LP and are NOT warm-reusing, so no parked
                    # step's HiGHS is a reuse source and ``self._warm_problem``
                    # was just nulled above.  Any prior step still holding a
                    # live ``solution.highs`` — typically the previous
                    # same-level roll; exhausted *other* levels were already
                    # freed at the top of run() — is now the ONLY reference to
                    # that HiGHS.  Release it BEFORE ``_build_warm_problem``
                    # allocates, else two same-level footprints coexist (the
                    # DES 7/9 dispatch-on-dispatch stack).  Per-iter writers +
                    # handoff already consumed each prior step on its own iter,
                    # so this is safe.  ``flex_data_provider`` is NOT dropped
                    # here — same-level rolls reuse the per-level provider
                    # cache (``state._level_providers``).
                    if (
                        not keep_solutions
                        and os.environ.get("FLEXTOOL_DISABLE_XLEVEL_RELEASE") != "1"
                    ):
                        _cr_released = False
                        for _ck, _cstep in (getattr(self, "_all_steps", None) or {}).items():
                            _csol = getattr(_cstep, "solution", None)
                            if _csol is not None and getattr(_csol, "highs", None) is not None:
                                _csol.highs = None
                                _cr_released = True
                        if _cr_released:
                            _try_malloc_trim()
                    if not keep_solutions:
                        _audit_cold_rebuild_release(
                            steps=getattr(self, "_all_steps", None),
                            complete_solve_name=complete_solve_name,
                        )
                    # Build the warm problem first WITHOUT solver
                    # options so we can inspect LP ranges, then push the
                    # finalised HiGHS options through ``set_solver_options``
                    # on the underlying Problem.
                    _phase_prof("build_start")
                    self._warm_problem = _build_warm_problem(
                        data,
                        scale_the_objective=effective_obj_scale,
                        solver_options=None,
                    )
                    _phase_prof("build_done")
                    if _memrec_local is not None and _emit_phase:
                        _memrec_local.checkpoint(
                            "lp_build_end", self.state.logger,
                            user_label="Matrix built by polar-high",
                        )
                    inner_pb = self._warm_problem.problem
                    from flextool.engine_polars._solver_dispatch import (
                        _resolve_effective_highs_options,
                    )
                    highs_options = _resolve_effective_highs_options(
                        solver_arguments_map=_solver_args_map,
                        highs_opt_path=_highs_opt_path,
                        cli_overrides=_build_cli_overrides(),
                        baseline=_baseline_highs_options(
                            user_bound_scale_override=user_bound_scale_override,
                            scaling_mode=_scaling_mode,
                        ),
                    )
                    inner_pb.set_solver_options(highs_options)
                    # Autoscale Layer 2 + Layer 3 on the warm-active
                    # first-build branch.  Same call sequence as the
                    # cold path below — see the longer-form comment
                    # there for the rationale.  Skipping these on warm
                    # solves used to leave HiGHS staring at an unscaled
                    # LP, costing both numerical health and ~tens of GB
                    # of internal simplex working set on
                    # poorly-conditioned LPs.
                    #
                    # First-build only: Layer 2 writes side vectors on
                    # the Problem and ``WarmProblem._initial_build``
                    # bakes them into the canonical matrix (with
                    # ``_param_cells`` caching the scaled factors for
                    # tracked Params).  Subsequent
                    # ``_apply_warm_updates`` Param mutations update
                    # HiGHS coefficients via those cached factors — no
                    # re-canonicalisation, no Layer 2 re-apply.
                    # ``self._autoscale_warm_layer2_plan`` caches the
                    # plan for use by
                    # :func:`_autoscale_unscale_post_solve` after every
                    # warm solve (first build AND reuses).
                    #
                    # Per-shape autoscale DECISION cache: keyed on the
                    # BUILT LP's structural signature (matrix shape +
                    # per-family layout), which is invariant across rolls
                    # of the same rolling solve even though
                    # ``_fingerprint(data)`` slides (a windowed period/dt
                    # field's height tracks the rolling horizon).  On a HIT
                    # we replay the cached Layer-2 exponents + Layer-3 plan
                    # WITHOUT any ``detect_ranges`` / ``bucket_coefficients``
                    # walk (the per-roll multi-GB spike).  A cold rebuild of
                    # an already-seen shape (the ladder-Param
                    # ``_IncompatibleUpdate`` path) therefore skips the
                    # traversals entirely.
                    _shape_key = _autoscale_lp_shape_signature(
                        inner_pb, base_solve_name,
                    )
                    _cache_entry = (
                        None if _autoscale_disable_cache()
                        else self._autoscale_shape_cache.get(_shape_key)
                    )
                    _phase_prof("autoscale_l2_start")
                    if _cache_entry is not None:
                        # HIT — replay decision, no range walk.
                        self._autoscale_warm_layer2_plan = (
                            _autoscale_apply_layer2_from_cache(
                                inner_pb, _cache_entry,
                                solve_name=complete_solve_name,
                                logger=self.state.logger,
                            )
                        )
                        _autoscale_ranges_pre = _cache_entry.ranges_pre
                        self._autoscale_warm_ranges_pre = _autoscale_ranges_pre
                        _phase_prof("autoscale_l3_start")
                        _autoscale_layer3_plan = (
                            _autoscale_apply_layer3_from_cache(
                                inner_pb, _cache_entry,
                                solve_name=complete_solve_name,
                                logger=self.state.logger,
                            )
                        )
                        _autoscale_ranges_post = _cache_entry.ranges_post
                    else:
                        # MISS — run the full traversals, then cache.
                        (
                            self._autoscale_warm_layer2_plan,
                            _autoscale_ranges_pre,
                        ) = _autoscale_apply_layer2_pre_solve(
                            inner_pb,
                            solve_name=complete_solve_name,
                            logger=self.state.logger,
                        )
                        # Cache the pre-Layer-2 RangeReport so subsequent
                        # warm reuses can still attach it as
                        # ``Solution.streamed_lp_ranges`` (the cascade only
                        # builds the LP once, so the four ranges are
                        # invariant across rolls of the same warm problem).
                        self._autoscale_warm_ranges_pre = _autoscale_ranges_pre
                        _phase_prof("autoscale_l3_start")
                        _autoscale_layer3_plan = _autoscale_apply_layer3_pre_solve(
                            inner_pb,
                            layer2_plan=self._autoscale_warm_layer2_plan,
                            solve_name=complete_solve_name,
                            logger=self.state.logger,
                        )
                        _autoscale_ranges_post = None
                        if not _autoscale_disable_cache():
                            _l2p = self._autoscale_warm_layer2_plan
                            self._autoscale_shape_cache[_shape_key] = (
                                _AutoscaleShapeCacheEntry(
                                    layer2_exponents=(
                                        dict(_l2p.type_exponents)
                                        if _l2p is not None else None
                                    ),
                                    layer2_buckets_before=(
                                        dict(_l2p.type_buckets_before)
                                        if _l2p is not None else {}
                                    ),
                                    layer2_buckets_after=(
                                        dict(_l2p.type_buckets_after)
                                        if _l2p is not None else {}
                                    ),
                                    layer3_plan=_autoscale_layer3_plan,
                                    ranges_pre=_autoscale_ranges_pre,
                                    ranges_post=None,
                                )
                            )
                    _phase_prof("autoscale_summary_start")
                    _autoscale_emit_console_summary(
                        ranges_pre=_autoscale_ranges_pre,
                        ranges_post=_autoscale_ranges_post,
                        layer2_plan=self._autoscale_warm_layer2_plan,
                        layer3_plan=_autoscale_layer3_plan,
                        solve_name=base_solve_name,
                        already_emitted=self._autoscale_summary_emitted,
                        memrec=_memrec_local if _emit_phase else None,
                        logger=self.state.logger,
                    )
                    _phase_prof("autoscale_done")
                # ``WarmProblem.solve`` always keeps the HiGHS instance
                # alive on ``Solution.highs`` — that's the whole point
                # of warm reuse — so the output writer adapter
                # (``write_all_variables`` / ``write_all_handoffs``)
                # sees the live solver as it does for cold rebuilds
                # under ``keep_solver=True``.  No extra kwarg required.
                # Blank line so HiGHS' "Running HiGHS …" banner (and the
                # grey solver-output block in the GUI) separates from the
                # scaling/LP-build rows above it.
                print("", flush=True)
                _t_solve_start = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                _phase_prof("solve_start")
                sol = self._warm_problem.solve()
                _t_solve_end = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                _phase_prof("after_solve")
                # Attach the cached pre-Layer-2 RangeReport as
                # ``streamed_lp_ranges`` on the warm Solution so the
                # Layer-1 emit hook and downstream telemetry
                # (e.g. ``test_invest_chain_lp_bound_range_smoke``) see
                # the four (min, max) pairs the autoscaler decided on.
                # ``WarmProblem.solve`` returns a bare Solution with no
                # streamed ranges of its own — the polar-high in-process
                # streaming-solve path that populates that attribute is
                # bypassed when HiGHS' ``Highs.run`` is invoked directly
                # on the live instance, so the warm path has to surface
                # the ranges itself.  Mirrors the cold-path assignment
                # below (search for ``sol.streamed_lp_ranges = {``).
                _warm_ranges = self._autoscale_warm_ranges_pre
                if (
                    _warm_ranges is not None
                    and getattr(sol, "streamed_lp_ranges", None) is None
                ):
                    try:
                        sol.streamed_lp_ranges = {
                            "matrix": _warm_ranges.matrix,
                            "cost": _warm_ranges.cost,
                            "col_bound": _warm_ranges.bound,
                            "row_bound": _warm_ranges.rhs,
                        }
                    except Exception:  # pragma: no cover — Solution may
                        # forbid the assignment in a future polar-high
                        pass
                # Eager unscale on every warm solve (first-build AND
                # reuses) so output writers see physical-coordinate
                # primal / duals / reduced costs.  No-op when the
                # cached plan is None (Layer 2 was off or didn't
                # trigger at first-build).
                _phase_prof("unscale_start")
                if self._autoscale_warm_layer2_plan is not None:
                    _autoscale_unscale_post_solve(
                        sol, self._autoscale_warm_layer2_plan,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                _phase_prof("unscale_done")
                self._prior_data = data
                self._prior_fp = fp
            else:
                pb = Problem()
                _phase_prof("build_start")
                build_flextool(pb, data, scale_the_objective=effective_obj_scale)
                _phase_prof("build_done")
                if _memrec_local is not None and _emit_phase:
                    _memrec_local.checkpoint(
                        "lp_build_end", self.state.logger,
                        user_label="Matrix built by polar-high",
                    )
                from flextool.engine_polars._solver_dispatch import (
                    _resolve_effective_highs_options,
                )
                highs_options = _resolve_effective_highs_options(
                    solver_arguments_map=_solver_args_map,
                    highs_opt_path=_highs_opt_path,
                    cli_overrides=_build_cli_overrides(),
                    baseline=_baseline_highs_options(
                        user_bound_scale_override=user_bound_scale_override,
                        scaling_mode=_scaling_mode,
                    ),
                )
                pb.set_solver_options(highs_options)
                # ── DIAGNOSTIC: per-substep RSS in the pre-write_mps gap ──
                # OOM in this gap (post "Matrix built", pre write_mps) is
                # invisible to both ``_MemoryRecorder`` (single checkpoint
                # for "Matrix built") and ``POLAR_HIGH_WRITE_MPS_PROFILE``
                # (only fires inside write_mps).  This closure samples
                # ``psutil.Process().memory_info().rss`` at each substep
                # below and writes to stderr in the same format as the
                # polar-high profile.  Activate with
                # ``FLEXTOOL_AUTOSCALE_PROFILE=1``; zero overhead when off.
                _autoscale_profile = (
                    os.environ.get("FLEXTOOL_AUTOSCALE_PROFILE") == "1"
                )
                if _autoscale_profile:
                    try:
                        import psutil as _psutil
                        _ap_proc = _psutil.Process()
                        _ap_t0 = time.monotonic()
                        _ap_prev = _ap_proc.memory_info().rss / (1024 ** 3)
                        import sys as _sys
                        def _ap(phase: str, **extras) -> None:
                            nonlocal _ap_prev
                            rss = _ap_proc.memory_info().rss / (1024 ** 3)
                            delta = rss - _ap_prev
                            wall = time.monotonic() - _ap_t0
                            sign = "+" if delta >= 0 else ""
                            extras_str = "\t".join(
                                f"{k}={v}" for k, v in extras.items()
                            )
                            print(
                                f"[autoscale profile]\tphase={phase}\t"
                                f"rss_gb={rss:.2f}\tdelta_gb={sign}{delta:.2f}"
                                f"\twall_s={wall:.2f}"
                                + (f"\t{extras_str}" if extras_str else ""),
                                file=_sys.stderr, flush=True,
                            )
                            _ap_prev = rss
                        _ap("enter")
                    except ImportError:
                        _autoscale_profile = False
                        print(
                            "FLEXTOOL_AUTOSCALE_PROFILE=1 but psutil not "
                            "installed; profiling disabled.",
                            file=__import__("sys").stderr, flush=True,
                        )
                # autoscale Layer 2 (semantic per-type) pre-solve apply.
                # Trigger gate is the same Layer-1 four-range readout —
                # see ``_autoscale_apply_layer2_pre_solve``.  Plan is
                # consumed by ``_autoscale_unscale_post_solve`` once the
                # solve returns so downstream output writers see the
                # un-scaled solution.
                # Per-shape autoscale DECISION cache (cold ``warm=False``
                # / save_memory path).  Keyed on the BUILT LP's structural
                # signature (matrix shape + per-family layout), which is
                # invariant across rolls of the same rolling solve even
                # though ``_fingerprint(data)`` slides as the rolling
                # horizon moves.  Cheap to compute (no coefficient walk) so
                # repeated same-shape cold solves replay the cached decision
                # without the per-roll ``detect_ranges`` /
                # ``bucket_coefficients`` traversal.  Honours
                # ``FLEXTOOL_DISABLE_AUTOSCALE_CACHE=1``.
                _cold_key = (
                    None if _autoscale_disable_cache()
                    else _autoscale_lp_shape_signature(pb, base_solve_name)
                )
                _cache_entry = (
                    self._autoscale_shape_cache.get(_cold_key)
                    if _cold_key is not None else None
                )
                _phase_prof("autoscale_l2_start")
                if _cache_entry is not None:
                    # HIT — replay decision, NO range walk.
                    _autoscale_layer2_plan = _autoscale_apply_layer2_from_cache(
                        pb, _cache_entry,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                    _autoscale_ranges_pre = _cache_entry.ranges_pre
                    if _autoscale_profile:
                        _ap("layer2_applied_cached",
                            n_cstrs=len(pb._cstrs),
                            n_vars=len(pb._vars))
                    _phase_prof("autoscale_l3_start")
                    _autoscale_layer3_plan = _autoscale_apply_layer3_from_cache(
                        pb, _cache_entry,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                    if _autoscale_profile:
                        _ap("layer3_applied_cached")
                    _phase_prof("autoscale_summary_start")
                    _autoscale_ranges_post = _cache_entry.ranges_post
                    if _autoscale_profile:
                        _ap("ranges_post_cached",
                            ranges_post_ran=str(_autoscale_ranges_post is not None))
                else:
                    # MISS — run the full traversals, then cache.
                    (
                        _autoscale_layer2_plan,
                        _autoscale_ranges_pre,
                    ) = _autoscale_apply_layer2_pre_solve(
                        pb,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                    if _autoscale_profile:
                        _ap("layer2_applied",
                            n_cstrs=len(pb._cstrs),
                            n_vars=len(pb._vars))
                    # Layer 3 (HiGHS-native top-up): set user_objective_scale,
                    # user_bound_scale, and simplex_scale_strategy from the
                    # post-Layer-2 ranges so HiGHS sees a final LP that is
                    # already inside its comfort zone.  Layer 3 is HiGHS-
                    # internal (no inverse transform on the solution); the
                    # writeModel MPS export remains unscaled.
                    _phase_prof("autoscale_l3_start")
                    _autoscale_layer3_plan = _autoscale_apply_layer3_pre_solve(
                        pb,
                        layer2_plan=_autoscale_layer2_plan,
                        solve_name=complete_solve_name,
                        logger=self.state.logger,
                    )
                    if _autoscale_profile:
                        _ap("layer3_applied")
                    # Console summary: one user-visible line per base solve
                    # describing the autoscaler's pre/post ranges and the
                    # Layer 2 / Layer 3 decisions.  Read post-Layer-2 ranges
                    # from the (mutated) Problem so the "after" view reflects
                    # what HiGHS will see; Layer 3 acts inside HiGHS so its
                    # values are surfaced separately in the same line.
                    _phase_prof("autoscale_summary_start")
                    _autoscale_ranges_post: "_AutoscaleRangeReport | None" = None
                    if (
                        _autoscale_ranges_pre is not None
                        and _autoscale_ranges_pre.trigger
                    ):
                        try:
                            _autoscale_cfg_for_post = _autoscale_resolve_config(
                                None,
                            )
                            _autoscale_ranges_post = _autoscale_compute_ranges(
                                pb, _autoscale_cfg_for_post,
                            )
                        except Exception:  # pragma: no cover — non-fatal
                            self.state.logger.exception(
                                "autoscale post-Layer-2 range readout failed "
                                "for %s; console summary will omit the "
                                "'ranges post' segment",
                                complete_solve_name,
                            )
                    if _autoscale_profile:
                        _ap("ranges_post_computed",
                            ranges_post_ran=str(_autoscale_ranges_post is not None))
                    if _cold_key is not None:
                        _l2p = _autoscale_layer2_plan
                        self._autoscale_shape_cache[_cold_key] = (
                            _AutoscaleShapeCacheEntry(
                                layer2_exponents=(
                                    dict(_l2p.type_exponents)
                                    if _l2p is not None else None
                                ),
                                layer2_buckets_before=(
                                    dict(_l2p.type_buckets_before)
                                    if _l2p is not None else {}
                                ),
                                layer2_buckets_after=(
                                    dict(_l2p.type_buckets_after)
                                    if _l2p is not None else {}
                                ),
                                layer3_plan=_autoscale_layer3_plan,
                                ranges_pre=_autoscale_ranges_pre,
                                ranges_post=_autoscale_ranges_post,
                            )
                        )
                _autoscale_emit_console_summary(
                    ranges_pre=_autoscale_ranges_pre,
                    ranges_post=_autoscale_ranges_post,
                    layer2_plan=_autoscale_layer2_plan,
                    layer3_plan=_autoscale_layer3_plan,
                    solve_name=base_solve_name,
                    already_emitted=self._autoscale_summary_emitted,
                    memrec=_memrec_local if _emit_phase else None,
                    logger=self.state.logger,
                )
                _phase_prof("autoscale_done")
                if _autoscale_profile:
                    _ap("console_summary_done")
                # Phase 3 — multi-solver dispatch.  ``run_one_solve`` calls
                # ``pb.solve(keep_solver=True)`` for the default HiGHS path
                # (byte-identical to the pre-Phase-3 behaviour); routes to
                # ``solve_via_subprocess`` (HiGHS CLI / commercial CLI) on
                # every cold path, which always returns a real
                # ``polar_high.Solution`` with the HiGHS instance read back
                # from the MPS.  The cascade-level SolverConfig lookup uses
                # the active solve name with the standard
                # default-when-absent fallback.
                from flextool.engine_polars._solver_dispatch import (
                    run_one_solve,
                )
                _t_solve_start = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                # Pre-solve range capture: every cold path now goes
                # subprocess, and the subprocess Solution carries no
                # ``streamed_lp_ranges`` (polar-high populates that
                # during its in-process streaming solve, which the
                # subprocess child doesn't share with us).  Re-use the
                # post-Layer-2 RangeReport already computed above to
                # synthesize the dict :func:`_autoscale_emit_layer1`
                # consumes, so the per-solve ``autoscale_<solve>.yaml``
                # still lands under ``solve_data/``.
                _ranges_for_l1 = _autoscale_ranges_post or _autoscale_ranges_pre
                # Blank line so HiGHS' "Running HiGHS …" banner (and the
                # grey solver-output block in the GUI) separates from the
                # scaling/LP-build rows above it.
                print("", flush=True)
                _phase_prof("solve_start")
                sol = run_one_solve(
                    pb, _active_solver_cfg, logger=state.logger,
                    save_memory=_save_memory,
                    solve_name=complete_solve_name,
                    work_folder=getattr(state, "work_folder", None),
                )
                _t_solve_end = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                _phase_prof("after_solve")
                # Attach the pre-solve ranges as
                # ``streamed_lp_ranges`` on the Solution so the L1
                # emit hook (which expects a dict per polar-high's
                # in-process contract) sees the four (min, max) pairs
                # the solver actually saw — matrix / cost / col_bound /
                # row_bound, matching ``ranges_from_streamed``'s key
                # contract.
                if (
                    _ranges_for_l1 is not None
                    and getattr(sol, "streamed_lp_ranges", None) is None
                ):
                    try:
                        sol.streamed_lp_ranges = {
                            "matrix": _ranges_for_l1.matrix,
                            "cost": _ranges_for_l1.cost,
                            "col_bound": _ranges_for_l1.bound,
                            "row_bound": _ranges_for_l1.rhs,
                        }
                    except Exception:  # pragma: no cover — Solution may
                        # forbid the assignment in a future polar-high
                        pass
                # Eager unscale — restore primal / duals / reduced costs
                # to the un-scaled coordinate so output writers and
                # subsequent rolling iterations see physical values.
                _phase_prof("unscale_start")
                _autoscale_unscale_post_solve(
                    sol, _autoscale_layer2_plan,
                    solve_name=complete_solve_name,
                    logger=self.state.logger,
                )
                _phase_prof("unscale_done")
            # autoscale Layer 1 (detect) — log the four LP coefficient
            # ranges + trigger flag now that ``streamed_lp_ranges`` is
            # populated.  Detection-only in Phase 1b; Layer 2 / Layer 3
            # actions land in later phases.
            _phase_prof("layer1emit_start")
            _autoscale_emit_layer1(
                sol,
                solve_name=complete_solve_name,
                logger=self.state.logger,
                work_folder=self.state.paths.work_folder
                if self.state.paths is not None else None,
                layer2_plan=locals().get("_autoscale_layer2_plan"),
                layer3_plan=locals().get("_autoscale_layer3_plan"),
            )
            _phase_prof("layer1emit_done")
            # Memory checkpoint after the solve completes.  Fires on
            # level-boundary iters only; on within-group rolling iters
            # the suppressed deltas accumulate into the next emitted
            # group's lines.
            if _memrec_local is not None and _emit_phase:
                _memrec_local.checkpoint(
                    "solve_end", self.state.logger,
                    user_label="Solver",
                )
            # Emit per-iter lp_build / solve / warm_used rows now that
            # the solve is done and we have valid timestamps from
            # whichever branch ran.  handoff is recorded at end-of-run.
            if _phase_timing:
                _tr.record(
                    "per_iter",
                    subphase="lp_build",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=_t_solve_start - _t_build_start,
                    t_start=_t_build_start,
                )
                _tr.record(
                    "per_iter",
                    subphase="solve",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=_t_solve_end - _t_solve_start,
                    t_start=_t_solve_start,
                )
                _tr.record(
                    "per_iter",
                    subphase="warm_used",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=1.0 if warm_used else 0.0,
                )
            _t_handoff_start = (
                time.perf_counter() if _phase_timing else 0.0
            )
            # Write the ``scale_the_objective.csv`` (consumed by the
            # output writers' un-scaling path) BEFORE the optimality
            # check, so a non-optimal / time-limited solve still leaves
            # behind a coherent solve_data/ directory.
            #
            # Per-base-solve gating: the CSV value is invariant across
            # rolls of the same base solve, so emit it only on the first
            # roll.  The legacy ``scaling_report.txt`` diagnostic (and
            # its ``FLEXTOOL_SCALING_REPORT=1`` gate) was retired in
            # Phase 2b — the autoscaler's per-solve YAML report (written
            # from ``_autoscale_emit_layer1``) is the replacement.
            _t_scale_start = time.perf_counter() if _phase_timing else 0.0
            _write_csv = base_solve_name not in self._scale_csv_written
            if _write_csv:
                _write_scale_csv(
                    solve_data_dir=self.state.paths.work_folder / "solve_data",
                    solve_name=complete_solve_name,
                    effective_obj_scale=effective_obj_scale,
                    logger=self.state.logger,
                )
                self._scale_csv_written.add(base_solve_name)
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="scale_csv_report",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_scale_start,
                    t_start=_t_scale_start,
                )
            if not sol.optimal:
                self.state.logger.error(
                    f"non-optimal solve for {complete_solve_name}"
                )
                # When poor scaling was detected on the pre-solve LP,
                # surface an actionable hint explaining the suspected
                # cause and three concrete remediation paths (unit
                # conventions, single-thread HiGHS, disable autoscaler).
                _autoscale_emit_nonoptimal_hint(
                    ranges_pre=locals().get("_autoscale_ranges_pre"),
                    sol=sol,
                )
                return 1

            prior = prior_for_load
            # ``--csv-dump``: gate ``data.dump_csvs`` behind the
            # orchestrator-level ``csv_dump`` flag.  In default mode the
            # cascade stays in-memory; ``--csv-dump`` materialises the
            # full FlexData → CSV snapshot for debug.
            _t_dump_start = time.perf_counter() if _phase_timing else 0.0
            try:
                if getattr(self.state, "csv_dump", False):
                    data.dump_csvs(self.state.paths.work_folder)
            except Exception as exc:  # noqa: BLE001
                self.state.logger.warning(
                    f"dump_csvs failed for {complete_solve_name}: {exc}"
                )
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="dump_csvs",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_dump_start,
                    t_start=_t_dump_start,
                )
            # Emit TIER A ``output_raw`` artefacts BEFORE the in-memory
            # handoff is built.  ``write_all_handoffs`` (called by the
            # adapter) refreshes ``solve_data/period_capacity.csv`` and
            # other handoff CSVs; ``build_handoff_from_solution`` then
            # reads those refreshed files for the in-memory handoff.
            _t_wofs_start = time.perf_counter() if _phase_timing else 0.0
            _phase_prof("write_outputs_start")
            try:
                # Phase G — pass in-memory FlexData and the cascade-known
                # ``is_first_solve`` boolean so handoff writers + extractors
                # can short-circuit ~12 + ~10 per-iter CSV reads (audit:
                # specs/in_memory_carriers_audit.md).  CSV fallback paths
                # remain in place for callers that synthesize a Solution
                # without a FlexData (e.g. unit tests).
                _is_first = (
                    self.state.last_captured_solve is None
                    or len(self.state.handoffs or {}) == 0
                )
                write_outputs_for_solve(
                    sol,
                    work_folder=self.state.paths.work_folder,
                    solve_name=complete_solve_name,
                    prior_handoff=prior,
                    writer_state=writer_state,
                    flex_data=data,
                    is_first_solve=_is_first,
                    scale_the_objective=effective_obj_scale,
                    provider=getattr(
                        self.state, "current_provider", None,
                    ),
                    csv_dump=getattr(self.state, "csv_dump", False),
                )
            except Exception as exc:  # noqa: BLE001
                self.state.logger.warning(
                    f"write_outputs_for_solve failed for "
                    f"{complete_solve_name}: {exc}"
                )
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="write_outputs_for_solve",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_wofs_start,
                    t_start=_t_wofs_start,
                )
            _phase_prof("write_outputs_done")

            # Phase 4 (Gap F) — thread the in-memory FlexData + the
            # upper-level parent handoff so the extractors / fix_storage
            # parent overlay skip their workdir CSV reads where the same
            # data is already in scope.  ``parent_handoff`` is the upper
            # nesting parent (used for fix_storage propagation, deposited
            # by ``_native_run_model``); ``prior_handoff`` is the
            # sequence predecessor (used for cumulative accumulators).
            parent_complete = getattr(
                self.state, "current_parent_complete", None
            )
            parent_handoff = (
                self.state.handoffs.get(parent_complete)
                if parent_complete is not None
                and self.state.handoffs is not None else None
            )
            _t_bhf_start = time.perf_counter() if _phase_timing else 0.0
            _phase_prof("build_handoff_start")
            handoff = build_handoff_from_solution(
                sol, self.state.paths.work_folder, complete_solve_name,
                prior_handoff=prior,
                flex_data=data,
                parent_handoff=parent_handoff,
                provider=getattr(self.state, "current_provider", None),
            )
            _phase_prof("build_handoff_done")
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="build_handoff_from_solution",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_bhf_start,
                    t_start=_t_bhf_start,
                )
            # Deposit so the next iteration's translator picks it up
            # AND we have it for the result dict.
            self.state.handoffs[complete_solve_name] = handoff
            # Un-scale the objective value back to user-facing units.
            # ``build_flextool`` multiplied the objective coefficients by
            # ``effective_obj_scale``, so HiGHS reports a scaled value.
            # Overwrite ``sol.obj`` in place so the public
            # ``step.solution.obj`` matches the unscaled ``step.obj`` /
            # the ``v_obj__{solve}.parquet`` value that the legacy
            # writer un-scales via ``_resolve_inv_scale_the_objective``.
            # Without this, callers reading ``step.solution.obj`` see
            # the LP-internal (scaled-by-1e-6) magnitude — a parity
            # break with the legacy flextool objective.
            unscaled_obj = (
                sol.obj / effective_obj_scale if sol.obj is not None else None
            )
            if sol is not None and unscaled_obj is not None:
                sol.obj = unscaled_obj
            # In rolling solves, every iteration's ``complete_solve_name``
            # is the parent solve name (see ``recursive_solves.py:259``:
            # ``complete_solves[roll_name] = complete_solve_name``).  Use
            # the actual per-roll name from ``solve_data/solve_current.csv``
            # — the file flextool rewrites between rolls — so
            # ``_all_steps`` has one entry per roll instead of every roll
            # overwriting the parent key.  ``write_outputs`` keys its
            # union over sub-solves on this dict, and the parquet
            # writers use the same per-roll name (see
            # ``read_highs_solution._actual_solve_name``).
            from flextool.process_outputs.read_highs_solution import (
                _actual_solve_name,
            )
            step_key = _actual_solve_name(
                self.state.paths.work_folder, complete_solve_name,
                provider=getattr(self.state, "current_provider", None),
            )
            # NOTE: the previous "slim PRIOR iter's _vars + highs at
            # the start of iter N" block has been retired.  Both paths
            # now do their slim AFTER ``Outputs written`` below:
            #
            # * Warm path: per-level retention slim (Phase 2) — keeps
            #   one ``Solution.highs`` + one ``flex_data_provider`` per
            #   live level, drops everything else.
            # * Cold path (``_save_memory``): eager prior-iter slim —
            #   nulls everything heavy on the prior step.
            #
            # Both blocks live at the bottom of this method, just after
            # the ``outputs_written_end`` memory checkpoint.
            # Capture per-sub-solve decision-variable frames before the
            # step is deposited.  Polar-high may release ``sol._vars``
            # internally between sub-solves (a memory optimisation on
            # the ``polar-high-opt`` branch), so the LAST iter is the
            # only one whose live ``solution._vars`` survives to
            # end-of-cascade.  End-of-cascade writers
            # (``_entity_all_capacity`` and friends) need EACH sub-
            # solve's own ``v_invest_p`` / ``v_invest_n`` /
            # ``v_divest_p`` / ``v_divest_n``; snapshot those four
            # long-form polars frames here so they survive the
            # between-solve release.  Cost: 4 small DataFrames per
            # sub-solve (typically a few hundred rows each).  See
            # :class:`SnapshotSolution` for the wrapper consumers read.
            captured_vars: "dict[str, pl.DataFrame]" = {}
            _phase_prof("captured_vars_start")
            if sol is not None and getattr(sol, "_vars", None):
                for _vname in (
                    "v_invest_p", "v_invest_n",
                    "v_divest_p", "v_divest_n",
                ):
                    if _vname in sol._vars:
                        try:
                            captured_vars[_vname] = sol.value(_vname)
                        except Exception:  # noqa: BLE001
                            # Best-effort: if a variable can't be
                            # materialised here, the end-of-cascade
                            # writer's empty-fallback branch handles
                            # it (see ``_entity_all_capacity._try_value``).
                            pass
            _phase_prof("captured_vars_done")
            self._all_steps[step_key] = OrchestrationStep(
                solve_name=step_key,
                solution=sol,
                handoff=handoff,
                obj=unscaled_obj,
                optimal=bool(getattr(sol, "optimal", False)) if sol is not None else None,
                warm_used=warm_used,
                flex_data=data,
                flex_data_provider=getattr(
                    self.state, "current_provider", None,
                ),
                captured_vars=captured_vars,
            )
            # Track the just-parked step_key so the next iter can slim
            # THIS iter's Solution (see block above).
            self._prev_step_key = step_key
            # Phase 2 — record this step's level_key for the warm-path
            # per-level slim below.  ``state._current_level_key`` was
            # set by ``_native_run_model`` immediately before this call.
            _this_level_key = getattr(
                self.state, "_current_level_key", None,
            )
            if _this_level_key is not None:
                self._step_level_keys[step_key] = _this_level_key
            if _phase_timing:
                _tr.record(
                    "per_iter",
                    subphase="handoff",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_handoff_start,
                    t_start=_t_handoff_start,
                )
            # End-of-iter heap trim — release per-roll scratch frames so
            # the next iter's load_flextool doesn't compound on stale heap.
            # No-op on non-glibc.  Cost: ~10-50ms per iter.
            _try_malloc_trim()
            # Memory checkpoint after the per-iter writers + handoff
            # capture finish.  Same gating as the other phase checkpoints:
            # level-boundary iters only.
            if _memrec_local is not None and _emit_phase:
                _memrec_local.checkpoint(
                    "outputs_written_end", self.state.logger,
                    user_label="Outputs written",
                )
            # Phase 2 — warm-path per-level retention slim.  Per the
            # user's design:
            #
            # * Keep ONE ``Solution.highs`` (the live ``polar_high.Solution``
            #   wrapping the WarmProblem's HiGHS instance) per level —
            #   the MOST RECENT parked one of each level.
            # * Keep ONE ``flex_data_provider`` per level — same gating.
            # * Drop both as soon as the pipeline has no more upcoming
            #   solves of that level (``state._all_level_keys[i+1:]``).
            # * Drop ``Solution._vars`` after ``Outputs written`` on
            #   warm too (warm-start uses HiGHS' basis, not these
            #   polars frames).
            # * Drop ``flex_data`` after the solve consumes it.
            #
            # ``flex_data`` and ``solution`` (the Python object, minus
            # the heavy ``.highs`` + ``._vars`` slots) on the LAST step
            # overall survive the per-iter slim because cmd_run_flextool
            # passes them to ``write_outputs``.  We never null the
            # ``step.solution`` object itself here — only its
            # ``_vars`` dict and its ``.highs`` reference.
            #
            # Memory-pressure-yielding for kept Highs instances is a
            # future concern, explicitly out of scope per the user.
            #
            # ``keep_solutions=True`` opts out of per-iter slimming
            # entirely — callers like ``tests/test_scenarios.py`` and
            # any other ``solve_steps``-style end-of-cascade walker
            # union per-step ``flex_data`` + ``solution`` after the
            # cascade returns and would crash on the nulled fields
            # otherwise.
            if not _save_memory and not keep_solutions:
                _all_level_keys = getattr(
                    self.state, "_all_level_keys", ()
                )
                _iter_idx = getattr(
                    self.state, "_current_iter_index", None,
                )
                _upcoming_levels: "set" = set()
                if _iter_idx is not None and _all_level_keys:
                    _upcoming_levels = set(
                        _all_level_keys[_iter_idx + 1:]
                    )
                _this_level = self._step_level_keys.get(step_key)
                # Walk every parked step.  For each, decide whether to
                # keep its ``solution.highs`` + ``flex_data_provider``.
                for _k, _step in self._all_steps.items():
                    _step_lvl = self._step_level_keys.get(_k)
                    _is_just_parked = (_k == step_key)
                    # ``solution._vars`` and ``flex_data`` are dropped on
                    # every PRIOR step regardless of level (the per-iter
                    # writers + handoff carrier already consumed them).
                    if not _is_just_parked and _step.solution is not None:
                        try:
                            _step.solution._vars = {}
                        except Exception:  # noqa: BLE001
                            pass
                    if not _is_just_parked:
                        _step.flex_data = None
                    # ``solution.highs`` + ``flex_data_provider`` are
                    # kept only on the MOST RECENT parked step of each
                    # level whose pipeline still has upcoming iters.
                    # Just-parked step's level always has at least one
                    # member (itself) so the "drop entire level" rule
                    # fires only on PRIOR steps whose level is exhausted.
                    if _is_just_parked:
                        # Even the just-parked step drops its highs +
                        # flex_data_provider when its level has no
                        # more upcoming iters.  Saves the level's last
                        # parked Highs for the duration of subsequent
                        # other-level work that would otherwise pin it.
                        if (
                            _step_lvl is not None
                            and _step_lvl not in _upcoming_levels
                            and _this_level != _step_lvl
                        ):
                            # Can't happen: just-parked step's level
                            # IS ``_this_level``.  Defensive no-op.
                            pass
                        continue
                    # Prior step.  Drop its highs / provider when EITHER:
                    #   (a) the level it belongs to is exhausted
                    #       (``_step_lvl not in _upcoming_levels`` and
                    #       ``_step_lvl != _this_level``), OR
                    #   (b) the level it belongs to is the same as the
                    #       just-parked step's level — in which case
                    #       the just-parked step is the new "most recent
                    #       of this level" and this older sibling is
                    #       superseded.
                    _level_exhausted = (
                        _step_lvl is not None
                        and _step_lvl != _this_level
                        and _step_lvl not in _upcoming_levels
                    )
                    _same_level_older = (
                        _step_lvl is not None
                        and _step_lvl == _this_level
                    )
                    if _level_exhausted or _same_level_older:
                        if _step.solution is not None:
                            try:
                                # Null the WHOLE solution, not just
                                # ``.highs``.  A superseded prior step's
                                # ``polar_high.Solution`` retains the
                                # ``col_names``/``row_names`` lists and the
                                # ``col_value``/``row_dual``/``col_dual``
                                # numpy arrays — all O(LP size) (~1.26 GB
                                # per roll on the real DES).  Nulling only
                                # ``.highs`` released the HiGHS handle but
                                # left those arrays parked for the whole
                                # cascade; dropping the whole object is the
                                # per-roll floor-ratchet release.  Warm
                                # reuse is unaffected — it runs off
                                # ``self._warm_problem``, never off parked
                                # step solutions; and ``handoff`` /
                                # ``captured_vars`` (the only per-step state
                                # later consumers need) live on the step,
                                # not inside ``solution``.
                                _step.solution = None
                            except Exception:  # noqa: BLE001
                                pass
                        _step.flex_data_provider = None
                # Trim the libc heap after potentially dropping multiple
                # large Highs instances + polars frames.
                _try_malloc_trim()
            # Cold-path (save-memory) eager slim of the PRIOR iter's
            # parked OrchestrationStep.  We can't slim the JUST-parked
            # step here — the orchestration cli (``cmd_run_flextool``)
            # passes the LAST step's ``flex_data`` + ``solution`` to
            # :func:`write_outputs`, and from inside the per-iter callback
            # we don't yet know which iter is last.  Slimming the PRIOR
            # iter instead leaves one step's heavy state live at any
            # given time (the just-parked one) and guarantees the LAST
            # step survives the cascade intact.
            #
            # By the time we reach here the PRIOR iter's per-iter
            # consumers all ran on its own iter:
            #
            # * ``write_outputs_for_solve`` (the writers that needed
            #   ``sol.highs.allVariableNames()`` / ``getSolution()``
            #   / ``getLp().row_names_``) — done.
            # * ``build_handoff_from_solution`` — done; carrier stored
            #   in ``step.handoff`` survives this slim.
            # * ``captured_vars`` snapshot — done; lives on
            #   ``step.captured_vars`` independently of ``sol._vars``.
            #
            # On cold the cascade rebuilds the LP from scratch every
            # sub-solve (warm reuse is disabled via the
            # ``_warm_disabled_by_save_memory`` branch at the top of
            # this method), so the prior iter has no further consumer.
            # Drop everything heavy on it — that is the root-cause fix
            # for the cross-solve RSS climb on ``--save-memory`` runs.
            #
            # ``flex_data_provider`` is dropped here by default; set
            # ``FLEXTOOL_COLD_KEEP_PROVIDER=1`` to retain it across the
            # cold-path cascade (trades higher RSS for skipping the
            # per-iter Spine DB re-read).  Real-model measurement at
            # the time of writing did not produce a default-changing
            # signal — the knob exists for workloads where the DB
            # re-read dominates wall time.
            # ``keep_solutions=True`` opts out (see the warm-slim
            # block above for the same rationale — callers that union
            # per-step state after the cascade returns crash on nulled
            # fields).
            if _save_memory and not keep_solutions and self._prev_step_key is not None:
                _prev_step = self._all_steps.get(self._prev_step_key)
                if _prev_step is not None and _prev_step is not self._all_steps.get(step_key):
                    _prev_step.flex_data = None
                    if os.environ.get("FLEXTOOL_COLD_KEEP_PROVIDER") != "1":
                        _prev_step.flex_data_provider = None
                    _psol = _prev_step.solution
                    if _psol is not None:
                        try:
                            _psol._vars = {}
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            _psol.highs = None
                        except Exception:  # noqa: BLE001
                            pass
            return 0

    # Drive the cascade via the native ``native_run_model``.  Native
    # emitters thread ``sub_solve_provider`` through every emit_* call.
    native_run_model(runner.state, _PolarHighCascadeSolver(runner.state))
    # Mirror the in-memory handoff dict back onto our state in case
    # callers want to inspect it.
    state.handoffs = runner.state.handoffs

    # Phase C.5 — slim every step except the LAST, releasing the heaviest
    # per-step state (Solution + FlexData + FlexDataProvider) once
    # downstream consumers (handoff extraction, raw-output write) have
    # run for that sub-solve.  ``keep_solutions=True`` opts out — used
    # by tests that need per-step ``solution`` / ``flex_data`` access.
    if not keep_solutions and results:
        last_key = next(reversed(results))
        for k, step in results.items():
            if k == last_key:
                continue
            step.solution = None
            step.flex_data = None
            step.flex_data_provider = None
        # Free the HiGHS heap once the per-step references are gone.
        # Cheap and a no-op on non-glibc.
        _try_malloc_trim()
    return results


# ---------------------------------------------------------------------------
# run_chain_from_db — top-level entry point
# ---------------------------------------------------------------------------


def run_chain_from_db(
    input_db_url: str | Path,
    scenario_name: str | None = None,
    work_folder: Path | str | None = None,
    *,
    flextool_dir: Path | str | None = None,
    solver_config_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
    warm: bool = True,
    keep_solutions: bool = False,
    csv_dump: bool = False,
    override_provider: "Callable[[], dict[str, pl.DataFrame]] | None" = None,
) -> dict[str, OrchestrationStep]:
    """Run a flextool multi-solve scenario end-to-end natively.

    Combines:

    1. :func:`flextool.engine_polars._native_input_writer.write_workdir_inputs`
       populates the cascade-input Provider with every derived frame
       from the Spine DB (pure in-memory; no workdir CSVs are
       written).
    2. ``_orchestration.run_orchestration`` to drive the per-solve loop
       with a polar_high-as-inner-solver wrapper.
    3. Returns one :class:`OrchestrationStep` per per-solve iteration.

    For tests / scripts that want a single function call to go from a
    DB scenario to a dict of (Solution, SolveHandoff) pairs.

    Parameters
    ----------
    input_db_url : str | Path
        Spine SQLite URL or path.  A bare path is upgraded to ``sqlite:///``.
    scenario_name : str, optional
        Scenario filter to apply.  ``None`` picks the first scenario.
    work_folder : Path | str, optional
        Where to materialise the CSVs.  ``None`` uses an auto-cleaned
        tempdir.
    flextool_dir, solver_config_dir : Path, optional
        Override the default flextool install location.  Default:
        ``/home/jkiviluo/sources/flextool/{flextool,bin}``.
    logger : logging.Logger, optional
        Logger to use.  ``None`` constructs one named after the scenario.
    warm : bool, default False
        Δ.12d — when True, reuse one :class:`polar_high.WarmProblem`
        across consecutive structurally-compatible per-solve iterations
        in the cascade, applying ``_apply_warm_updates`` between solves
        rather than cold-rebuilding.  See
        :func:`run_orchestration` for full semantics.
    keep_solutions : bool, default False
        Phase C.5 — when False (default), only the LAST step in the
        returned dict retains ``solution`` / ``flex_data`` /
        ``flex_data_provider``; earlier steps clear those slots to
        release the HiGHS instance + variable arrays + writer-frame
        snapshot for that sub-solve.  All slim fields (``solve_name``,
        ``obj``, ``optimal``, ``warm_used``, ``handoff``) remain
        populated on every step.  Set ``True`` to retain the full
        per-step state — required by tests that need per-step
        ``solution`` / ``flex_data`` access (parity sweeps, warm
        comparisons, etc.).

    Returns
    -------
    dict[str, OrchestrationStep]
        Mapping ``complete_solve_name → OrchestrationStep``.  Iterate
        in insertion order to walk the chain.
    """
    from flextool.engine_polars._db_loader import FlexToolRunner

    if logger is None:
        logger = logging.getLogger(
            f"flextool.engine_polars.run_chain_from_db[{scenario_name}]"
        )

    # v52 multi-solver dispatch (Phase 2 startup hint).  Probe each
    # solver in polar-high's catalog with a trivial 1-var LP so users
    # see at a glance which are licensed on this machine (vs wrapper-
    # installed-but-no-license vs not-installed-at-all) before the
    # solve loop selects one.  See ``specs/flextool-multi-solver-handoff.md``
    # Step 4b.  Cached at module level so repeat cascade runs don't
    # re-probe.
    try:
        from flextool.engine_polars._solver_dispatch import (
            probe_solver_licenses,
        )
        statuses = probe_solver_licenses()
        # HiGHS is an open-source dependency that's always present and
        # always usable -- nothing to opt in to.  Drop it from the
        # "Available solvers" line so it only reports the commercial
        # adapters whose installation status actually varies.
        statuses = {n: s for n, s in (statuses or {}).items() if n != "highs"}
        if statuses:
            formatted = ", ".join(f"{n}={s}" for n, s in statuses.items())
            # Trailing blank line so the mem-checkpoint table that
            # follows visually separates from the header block.
            logger.info("Available solvers: %s\n", formatted)
    except ImportError:  # pragma: no cover — older polar_high without dispatch
        pass

    db_url = str(input_db_url)
    if "://" not in db_url:
        db_url = f"sqlite:///{db_url}"

    if work_folder is None:
        work_folder = Path(tempfile.mkdtemp(prefix="flextool_run_chain_"))
    else:
        work_folder = Path(work_folder)
        work_folder.mkdir(parents=True, exist_ok=True)

    # ``flextool_dir`` defaults to the installed flextool package directory
    # (resolved via importlib.resources so it works both editable and wheel).
    # ``solver_config_dir`` defaults to ``<cwd>/solver_config`` — this is where the user's
    # editable ``highs.opt`` lives; the package's ``highs.opt.template``
    # is only used to seed that file on first run.
    from flextool._resources import package_data_path
    flextool_dir_resolved = (
        Path(flextool_dir) if flextool_dir is not None
        else package_data_path("")
    )
    solver_config_dir_resolved = (
        Path(solver_config_dir) if solver_config_dir is not None else Path.cwd() / "solver_config"
    )

    # Cascade-input Provider population from the Spine DB.  Pure
    # in-memory: ``write_workdir_inputs`` runs the input_derivation
    # pipeline whose emitters populate the Provider directly, so no
    # CSVs hit disk.
    from flextool.engine_polars._native_input_writer import (
        write_workdir_inputs,
    )

    # Phase-progress recorder.  Always emits user-visible log lines
    # (RSS + section time + Δrss) so users following the run can see
    # what each phase is doing.  Setting FLEXTOOL_MEMORY_DIAGNOSTICS=1
    # additionally enables tracemalloc (so ``traced_peak`` is meaningful)
    # and writes the per-checkpoint CSV under ``solve_data/`` for
    # post-hoc analysis.
    _mem_enabled = os.environ.get("FLEXTOOL_MEMORY_DIAGNOSTICS") == "1"
    if _mem_enabled:
        (work_folder / "solve_data").mkdir(parents=True, exist_ok=True)
        _memrec = _MemoryRecorder(
            work_folder / "solve_data" / "memory_diagnostics.csv",
            enabled=True,
        )
    else:
        _memrec = _MemoryRecorder(csv_path=None, enabled=False)
    # Publish so deeper modules (input.py's _apply_db_overrides) emit
    # in the unified [mem] format.
    set_phase_recorder(_memrec)
    _memrec.checkpoint("cascade_start", logger,
                       user_label="Run start")

    # Construct the cascade-input Provider and let
    # ``write_workdir_inputs`` populate it from the Spine DB.  The
    # Provider is then attached to the cascade ``RunnerState`` below
    # so the per-sub-solve hook in
    # :mod:`flextool.engine_polars._native_run_model` picks it up at
    # provider seed time.
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    cascade_input_provider = FlexDataProvider()

    write_workdir_inputs(
        db_url,
        scenario_name,
        work_folder,
        logger=logger,
        provider=cascade_input_provider,
        memory_recorder=_memrec,
    )
    # input_derivation allocates and frees a lot of polars scratch
    # state; glibc's heap retains the freed pages.  Release them
    # before the polars-heavy ``load_flextool`` starts so the heap
    # watermark doesn't compound.
    _try_malloc_trim()
    _memrec.checkpoint("write_workdir_inputs_end", logger,
                       user_label="Input data prepared (after malloc_trim)")

    # Construct the underlying FlexToolRunner — still needed to carry
    # the cross-cutting ``RunnerState`` (timeline, solve config, handoff
    # dict) into ``native_run_model``, which drives the per-solve
    # preprocessing chain (``preprocessing_solve_time``,
    # ``solve_writers``, ``handoff_writers``).  No write_input call.
    def _runner_factory() -> "FlexToolRunner":
        runner = FlexToolRunner(
            input_db_url=db_url,
            scenario_name=scenario_name,
            flextool_dir=flextool_dir_resolved,
            solver_config_dir=solver_config_dir_resolved,
            work_folder=work_folder,
        )
        runner.state.logger.setLevel(logging.ERROR)
        return runner

    # Build a minimal native RunnerState so callers can introspect
    # state.handoffs after the run.  The real per-solve mutation happens
    # on the underlying flextool runner's state (driven inside
    # _drive_cascade); we mirror the handoffs dict back.
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig

    sc = SolveConfig.load_from_db_url(db_url, scenario_name, logger=logger)
    _memrec.checkpoint("solve_config_loaded", logger,
                       user_label="SolveConfig loaded (from DB)")
    tc = TimelineConfig.load_from_db_url(db_url, scenario_name, logger=logger)
    tc.create_assumptive_parts(sc)
    tc.create_timeline_from_timestep_duration(sc)
    _memrec.checkpoint("timeline_constructed", logger,
                       user_label="TimelineConfig constructed (from DB)")

    state = RunnerState(
        paths=PathConfig(work_folder=work_folder),
        solve=sc,
        logger=logger,
        timeline=tc,
        handoffs={},
    )
    # Stash the memory recorder so ``run_orchestration`` →
    # ``_PolarHighCascadeSolver`` can fire the remaining first-iter
    # checkpoints (load / build / solve) without having to plumb it
    # through additional keyword arguments.
    state._memory_recorder = _memrec  # type: ignore[attr-defined]
    # Step 2.5 — seed the cascade-input Provider onto the state so
    # ``_drive_cascade`` can forward it onto ``runner.state`` (which the
    # per-sub-solve Provider hook in
    # :mod:`flextool.engine_polars._native_run_model` consults).
    state.cascade_input_provider = cascade_input_provider  # type: ignore[attr-defined]
    # Phase 5c — attach the optional external override provider onto
    # the cascade ``RunnerState``.  ``_drive_cascade`` forwards it onto
    # the underlying ``runner.state``; the per-sub-solve hook in
    # :mod:`flextool.engine_polars._native_run_model` (Phase 5b) invokes
    # it after the parent-handoff translator at every iteration.
    state.override_provider = override_provider

    return run_orchestration(
        state, work_folder, runner_factory=_runner_factory,
        db_url=db_url, scenario_name=scenario_name, warm=warm,
        keep_solutions=keep_solutions, csv_dump=csv_dump,
    )


__all__ = [
    "OrchestrationStep",
    "run_orchestration",
    "run_chain_from_db",
]
