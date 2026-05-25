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
    USER_BOUND_SCALE_MAX as _USER_BOUND_SCALE_MAX,
    USER_BOUND_SCALE_MIN as _USER_BOUND_SCALE_MIN,
    apply_layer2 as _autoscale_apply_layer2,
    apply_layer3 as _autoscale_apply_layer3,
    compute_ranges as _autoscale_compute_ranges,
    format_console_summary as _autoscale_format_console_summary,
    format_nonoptimal_hint as _autoscale_format_nonoptimal_hint,
    recommend_layer3 as _autoscale_recommend_layer3,
    resolve_auto_scale_config as _autoscale_resolve_config,
    resolve_user_bound_scale_override as _resolve_user_bound_scale_override,
    unscale_solution as _autoscale_unscale_solution,
    write_report as _autoscale_write_report,
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
) -> dict[str, object]:
    """Build the base HiGHS solver-option dict (determinism + matrix scale).

    Replaces the retired ``scaling.recommended_highs_options`` helper.
    Sets:

    * :data:`SIMPLEX_SCALE_STRATEGY_ADVANCED` — Curtis-Reid matrix
      equilibration.  Layer 3's :func:`apply_layer3` will *re-assert* this
      value when it merges its own options on top, so this is the
      authoritative source on the warm-rebuild path (which does not run
      Layer 3) and a redundant-but-consistent pin on the cold path.
    * :data:`DETERMINISM_OPTIONS` — ``random_seed`` / ``parallel`` /
      ``solver`` / ``presolve`` pins for byte-deterministic LP solutions.
    * ``user_bound_scale`` — only when ``user_bound_scale_override`` is
      a non-zero integer (CLI ``--user-bound-scale N`` / DB
      ``solve.user_bound_scale``).  Clamped to the HiGHS-safe range
      ``[USER_BOUND_SCALE_MIN, USER_BOUND_SCALE_MAX]``.  When unset, the
      autoscaler's Layer 3 may still emit its own value based on
      post-Layer-2 ranges; if neither sets it, polar-high's stream-time
      auto-pick (``Problem(auto_user_bound_scale=True)``) takes over.

    ``user_cost_scale`` is intentionally NOT set — costs are already
    multiplied by ``scale_the_objective`` inside ``build_flextool``, and
    Layer 3 may add ``user_objective_scale`` on top; layering a third
    cost-side knob would compound confusingly.
    """
    options: dict[str, object] = {
        "simplex_scale_strategy": SIMPLEX_SCALE_STRATEGY_ADVANCED,
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
    commercial-solver / LiteSolution path) skips the layer with a
    debug-level note rather than breaking the solve.
    """
    # ``cli_args=None`` is intentional: the CLI surface
    # (``cmd_run_flextool``) mirrors ``--auto-scale`` /
    # ``--user-bound-scale`` into the ``FLEXTOOL_AUTO_SCALE`` /
    # ``FLEXTOOL_USER_BOUND_SCALE`` env vars before invoking the
    # orchestrator, matching the existing env-threading convention
    # documented on the ``run_chain_from_db`` call site.  Cascade-
    # internal hops therefore observe operator intent without
    # plumbing the parsed ``args`` namespace through every helper.
    cfg = _autoscale_resolve_config(None)
    if not cfg.enabled:
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
    arrays.  The HiGHS options it sets *override* polar-high's
    stream-time ``auto_user_bound_scale`` heuristic (polar-high's gate
    is "caller has NOT set ``user_bound_scale``").
    """
    cfg = _autoscale_resolve_config(None)
    if not cfg.enabled:
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
        plan = _autoscale_recommend_layer3(ranges_post_l2, cfg)
    except Exception:  # pragma: no cover
        logger.exception(
            "autoscale Layer 3 recommendation failed for %s; "
            "skipping (HiGHS internal scaling still applies)",
            solve_name,
        )
        return None
    try:
        _autoscale_apply_layer3(pb, plan)
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
    if not cfg.enabled:
        return None, None
    try:
        ranges_pre = _autoscale_compute_ranges(pb, cfg)
    except Exception:  # pragma: no cover — guard against future API drift
        logger.exception(
            "autoscale Layer 2 pre-solve range readout failed for %s; "
            "skipping Layer 2 (Layer 1 post-solve still fires)",
            solve_name,
        )
        return None, None
    if not ranges_pre.trigger:
        return None, ranges_pre
    try:
        plan = _autoscale_apply_layer2(pb, cfg)
    except Exception:  # pragma: no cover
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


def _autoscale_emit_console_summary(
    *,
    ranges_pre: "_AutoscaleRangeReport | None",
    ranges_post: "_AutoscaleRangeReport | None",
    layer2_plan: "_AutoscaleLayer2Plan | None",
    layer3_plan: "_AutoscaleLayer3Plan | None",
    solve_name: str,
    already_emitted: set[str],
) -> None:
    """Emit the one-line user-visible autoscale summary.

    Uses ``print(...)`` rather than ``logger.info`` so the line surfaces
    at the default log level in the same stream where FlexTool's other
    phase-progress lines (``Input: …``, the HiGHS banner) appear.  We
    de-duplicate by solve name so a rolling solve emits the line once
    per base-solve, not once per roll — Layer 1/2/3 decisions are
    identical across rolls of the same base solve when the autoscaler
    is enabled.
    """
    cfg = _autoscale_resolve_config(None)
    if not cfg.enabled:
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
    from polar_high import Problem, Solution

    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig
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
    "Solver",
    "Outputs written",
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
        self._header_emitted: bool = False
        self._path = Path(csv_path) if csv_path is not None else None
        self._started = False
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
    _TIME_W = 10       # cumulative column (right-aligned)
    _SIZE_W = 12       # MB/GB column (right-aligned); fits "system total"

    @classmethod
    def _fmt_size(cls, mb: float | None) -> str:
        """Format an MB value as GB when ≥ 1024 MB, otherwise MB.
        Right-aligned within ``_SIZE_W`` so a column of these stacks.
        ``None`` renders as a dash placeholder of the same width.
        """
        if mb is None:
            return f"{'-':>{cls._SIZE_W}}"
        if mb >= 1024.0:
            s = f"{mb / 1024.0:.2f} GB"
        else:
            s = f"{mb:.0f} MB"
        return f"{s:>{cls._SIZE_W}}"

    @classmethod
    def _fmt_delta(cls, delta_mb: float | None) -> str:
        """Format a signed delta in MB / GB, right-aligned to
        ``_SIZE_W``.  ``None`` renders as a dash placeholder of the
        same width.  Near-zero values render as ``+0`` rather than
        signed-rounded noise.
        """
        if delta_mb is None:
            return f"{'-':>{cls._SIZE_W}}"
        if abs(delta_mb) < 0.5:
            return f"{'+0':>{cls._SIZE_W}}"
        sign = "+" if delta_mb >= 0 else "-"
        a = abs(delta_mb)
        if a >= 1024.0:
            s = f"{sign}{a / 1024.0:.2f} GB"
        else:
            s = f"{sign}{a:.0f} MB"
        return f"{s:>{cls._SIZE_W}}"

    def _emit_header(self) -> None:
        """Print the column-header line for the phase-progress table."""
        blank_label = " " * self._LABEL_W
        tw = self._SIZE_W
        tcol = self._TIME_W
        header = (
            f"{blank_label}  {'cumulative':>{tcol}}  "
            f"| {'ΔRSS memory':>{tw}}  {'Δsystem':>{tw}}  "
            f"| {'RSS memory':>{tw}}  {'system total':>{tw}}  "
            f"{'system swap':>{tw}}"
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
        delta_peak = (peak_mb - self._peak_prev_mb) if peak_mb is not None else None
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
            # Section-delta block (ΔRSS memory / Δsystem).
            if is_first:
                d_rss_str = self._fmt_size(None)
                d_sys_str = self._fmt_size(None)
            else:
                d_rss_str = self._fmt_delta(delta_rss)
                d_sys_str = self._fmt_delta(delta_sys)
            # Cumulative block: process RSS + swap, system used,
            # system swap.
            mem_str = self._fmt_size(rss_mb)
            sys_str = self._fmt_size(sys_mb)
            swap_str = self._fmt_size(swap_mb)
            time_col = f"{t_elapsed:.1f}s"
            line = (
                f"{label_col}  "
                f"{time_col:>{self._TIME_W}}  "
                f"| {d_rss_str}  {d_sys_str}  "
                f"| {mem_str}  {sys_str}  {swap_str}"
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
    warm: bool = False,
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
    warm: bool = False,
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
    from flextool.engine_polars._db_loader import FlexToolRunner
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
    runner.state.logger.setLevel(logger_level := logging.ERROR)
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
        def run(self, complete_solve_name: str) -> int:
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
            user_obj_scale = state.solve.scale_the_objective.get(complete_solve_name)
            effective_obj_scale = _resolve_effective_obj_scale(user_obj_scale)
            # ``user_bound_scale`` resolution priority:
            # ``FLEXTOOL_USER_BOUND_SCALE`` env var (set by
            # ``--user-bound-scale`` CLI flag) > DB ``solve.user_bound_scale``
            # > autoscale Layer 3's automatic recommendation (off the
            # post-Layer-2 ranges) > polar-high's stream-time auto-pick
            # (the cold Problem is built with ``auto_user_bound_scale=True``)
            # > HiGHS' own internal scaling.  HiGHS' "Consider setting
            # the user_bound_scale option to <N>" warning still prints a
            # value if any case slips through Layer 3; pass it via
            # ``--user-bound-scale``.
            _cli_ubs = os.environ.get("FLEXTOOL_USER_BOUND_SCALE")
            user_bound_scale_override = _resolve_user_bound_scale_override(
                _cli_ubs if _cli_ubs is not None
                else state.solve.user_bound_scale.get(complete_solve_name)
            )

            # HiGHS solver options.  ``simplex_scale_strategy`` =
            # advanced (Curtis-Reid) is always-on; ``user_bound_scale``
            # is only emitted when explicitly requested via
            # ``--user-bound-scale`` CLI / DB override (see priority
            # block above).  Default: HiGHS does its own scaling.
            # Cap solve time via env var if the operator requested it.
            _diag_tlim = os.environ.get("FLEXTOOL_HIGHS_TIME_LIMIT")
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

            def _finalise_highs_options(opts: dict) -> dict:
                if _diag_tlim:
                    try:
                        opts["time_limit"] = float(_diag_tlim)
                    except ValueError:
                        pass
                if _cli_presolve in ("on", "off", "choose"):
                    opts["presolve"] = _cli_presolve
                if _cli_threads is not None:
                    try:
                        n = int(_cli_threads)
                    except ValueError:
                        n = 1
                    if n > 1:
                        opts["threads"] = n
                        # User opted out of the determinism pin; HiGHS
                        # needs ``parallel="on"`` before it will actually
                        # use the threads.
                        opts["parallel"] = "on"
                    # n == 1 (or n <= 0) keeps the deterministic defaults
                    # from DETERMINISM_OPTIONS — no override needed.
                return opts

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
            if _warm_disabled_by_solver and not getattr(
                self, "_warm_disabled_warned", False
            ):
                state.logger.warning(
                    "warm-start is unavailable for solver %r; falling back "
                    "to cold rebuilds per sub-solve, expect slower per-iter "
                    "wall-clock.",
                    _active_solver_cfg.name,
                )
                self._warm_disabled_warned = True
            warm_used = False
            warm_active = warm and not _warm_disabled_by_solver
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
                    except _IncompatibleUpdate:
                        # Drop the stale warm problem so the next
                        # branch builds a fresh one.
                        self._warm_problem = None
                if not warm_used:
                    # Build the warm problem first WITHOUT solver
                    # options so we can inspect LP ranges, then push the
                    # finalised HiGHS options through ``set_solver_options``
                    # on the underlying Problem.
                    self._warm_problem = _build_warm_problem(
                        data,
                        scale_the_objective=effective_obj_scale,
                        solver_options=None,
                    )
                    if _memrec_local is not None and _emit_phase:
                        _memrec_local.checkpoint(
                            "lp_build_end", self.state.logger,
                            user_label="Matrix built by polar-high",
                        )
                        # Blank line so the next phase (HiGHS' banner +
                        # scaling block) visually separates from the
                        # LP-build summary.
                        print("", flush=True)
                    inner_pb = self._warm_problem.problem
                    highs_options = _finalise_highs_options(
                        _baseline_highs_options(
                            user_bound_scale_override=user_bound_scale_override,
                        )
                    )
                    inner_pb.set_solver_options(highs_options)
                # ``WarmProblem.solve`` always keeps the HiGHS instance
                # alive on ``Solution.highs`` — that's the whole point
                # of warm reuse — so the output writer adapter
                # (``write_all_variables`` / ``write_all_handoffs``)
                # sees the live solver as it does for cold rebuilds
                # under ``keep_solver=True``.  No extra kwarg required.
                _t_solve_start = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                sol = self._warm_problem.solve()
                _t_solve_end = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                self._prior_data = data
                self._prior_fp = fp
            else:
                pb = Problem(auto_user_bound_scale=True)
                build_flextool(pb, data, scale_the_objective=effective_obj_scale)
                if _memrec_local is not None and _emit_phase:
                    _memrec_local.checkpoint(
                        "lp_build_end", self.state.logger,
                        user_label="Matrix built by polar-high",
                    )
                    # Blank line so HiGHS' "Running HiGHS …" banner and
                    # its scaling-warning block visually separate from
                    # the LP-build phase summary above.
                    print("", flush=True)
                highs_options = _finalise_highs_options(
                    _baseline_highs_options(
                        user_bound_scale_override=user_bound_scale_override,
                    )
                )
                pb.set_solver_options(highs_options)
                # autoscale Layer 2 (semantic per-type) pre-solve apply.
                # Trigger gate is the same Layer-1 four-range readout —
                # see ``_autoscale_apply_layer2_pre_solve``.  Plan is
                # consumed by ``_autoscale_unscale_post_solve`` once the
                # solve returns so downstream output writers see the
                # un-scaled solution.
                (
                    _autoscale_layer2_plan,
                    _autoscale_ranges_pre,
                ) = _autoscale_apply_layer2_pre_solve(
                    pb,
                    solve_name=complete_solve_name,
                    logger=self.state.logger,
                )
                # Layer 3 (HiGHS-native top-up): set user_objective_scale,
                # user_bound_scale, and simplex_scale_strategy from the
                # post-Layer-2 ranges so HiGHS sees a final LP that is
                # already inside its comfort zone.  Layer 3 is HiGHS-
                # internal (no inverse transform on the solution); the
                # writeModel MPS export remains unscaled.
                _autoscale_layer3_plan = _autoscale_apply_layer3_pre_solve(
                    pb,
                    layer2_plan=_autoscale_layer2_plan,
                    solve_name=complete_solve_name,
                    logger=self.state.logger,
                )
                # Console summary: one user-visible line per base solve
                # describing the autoscaler's pre/post ranges and the
                # Layer 2 / Layer 3 decisions.  Read post-Layer-2 ranges
                # from the (mutated) Problem so the "after" view reflects
                # what HiGHS will see; Layer 3 acts inside HiGHS so its
                # values are surfaced separately in the same line.
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
                _autoscale_emit_console_summary(
                    ranges_pre=_autoscale_ranges_pre,
                    ranges_post=_autoscale_ranges_post,
                    layer2_plan=_autoscale_layer2_plan,
                    layer3_plan=_autoscale_layer3_plan,
                    solve_name=base_solve_name,
                    already_emitted=self._autoscale_summary_emitted,
                )
                # Phase 3 — multi-solver dispatch.  ``run_one_solve`` calls
                # ``pb.solve(keep_solver=True)`` for the default HiGHS path
                # (byte-identical to the pre-Phase-3 behaviour); routes to
                # ``polar_high.solvers.solve`` + LiteSolution wrapping on
                # the commercial path.  The cascade-level SolverConfig
                # lookup uses the active solve name with the standard
                # default-when-absent fallback.
                from flextool.engine_polars._solver_dispatch import (
                    run_one_solve,
                )
                _t_solve_start = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                sol = run_one_solve(
                    pb, _active_solver_cfg, logger=state.logger,
                )
                _t_solve_end = (
                    time.perf_counter() if _phase_timing else 0.0
                )
                # Eager unscale — restore primal / duals / reduced costs
                # to the un-scaled coordinate so output writers and
                # subsequent rolling iterations see physical values.
                _autoscale_unscale_post_solve(
                    sol, _autoscale_layer2_plan,
                    solve_name=complete_solve_name,
                    logger=self.state.logger,
                )
            # autoscale Layer 1 (detect) — log the four LP coefficient
            # ranges + trigger flag now that ``streamed_lp_ranges`` is
            # populated.  Detection-only in Phase 1b; Layer 2 / Layer 3
            # actions land in later phases.
            _autoscale_emit_layer1(
                sol,
                solve_name=complete_solve_name,
                logger=self.state.logger,
                work_folder=self.state.paths.work_folder
                if self.state.paths is not None else None,
                layer2_plan=locals().get("_autoscale_layer2_plan"),
                layer3_plan=locals().get("_autoscale_layer3_plan"),
            )
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
            handoff = build_handoff_from_solution(
                sol, self.state.paths.work_folder, complete_solve_name,
                prior_handoff=prior,
                flex_data=data,
                parent_handoff=parent_handoff,
                provider=getattr(self.state, "current_provider", None),
            )
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
            # Slim the PRIOR iter's parked Solution before parking this
            # iter's.  The prior iter's per-iter writers
            # (``write_outputs_for_solve``) and
            # ``build_handoff_from_solution`` ran before its
            # ``self._all_steps[...] = OrchestrationStep(...)`` deposit
            # — so by the time we get here on iter N, the iter-(N-1)
            # Solution's heavy ``_vars`` dict (one ``Var.frame``
            # polars DataFrame per LP variable) and its ``highs`` C++
            # instance are no longer needed.  Drop those; keep the
            # cheap 1-D arrays (``col_value``, ``col_dual``,
            # ``row_dual``), the small scalars (``optimal``, ``obj``,
            # ``col_names``, ``row_names``) — leaves the door open for
            # a future level-warm-start optimisation that seeds the
            # next cold-built LP's initial col_value from the prior
            # solution without paying the GB-scale frame cost.  The
            # post-loop slim at the bottom of ``_native_run_model``
            # still runs and nulls the whole ``step.solution`` on
            # non-last steps; this block only bounds the in-cascade
            # peak.  See
            # ``/tmp/highs-memory-investigation/`` HiGHS attribution
            # logs for the per-iter ~5.6 KB * ~400 vars * 80 iter =
            # 172 MB climb this addresses.
            if self._prev_step_key is not None:
                _prev = self._all_steps.get(self._prev_step_key)
                if _prev is not None and _prev.solution is not None:
                    _prev_sol = _prev.solution
                    try:
                        _prev_sol._vars = {}
                    except Exception:  # noqa: BLE001
                        pass
                    try:
                        _prev_sol.highs = None
                    except Exception:  # noqa: BLE001
                        pass
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
    warm: bool = False,
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


def run_single_solve_from_db(
    input_db_url: str | Path,
    scenario_name: str,
    work_folder: Path | str,
    *,
    logger: logging.Logger | None = None,
    emit_output: bool = True,
    csv_dump: bool = False,
) -> "OrchestrationStep":
    """Surgical fast-path single-solve from a Spine DB.

    Bypasses the per-solve CSV input-writer pipeline (the
    ``input_derivation`` + ``engine_polars/_emit_*`` cascade) entirely.
    Reads inputs directly from a
    :class:`flextool.engine_polars._spinedb_reader.SpineDbReader`,
    builds the LP via the override chain + a small native topology
    helper, solves via HiGHS, and emits ``output_raw/`` parquets via
    the existing output writer adapter (with a tiny support-CSV
    bootstrap that replaces the preprocessing pipeline's CSV writes).

    **EXPERIMENTAL / NON-PRODUCTION.**  This is the fast path the user
    flagged for ``test_24h_shipping``-style simple single-solve
    workloads.  No feature detection, no fallback: any helper coverage
    gap raises :class:`flextool.engine_polars._fast_load.FastLoadError`
    with the exact field name.  The slow path
    (:func:`run_chain_from_db`) remains the canonical multi-solve
    driver.

    Parameters
    ----------
    input_db_url : str | Path
        Spine SQLite URL or path.  Bare paths are upgraded to
        ``sqlite:///``.
    scenario_name : str
        Scenario name; required.  The fast path doesn't auto-pick.
    work_folder : Path | str
        Where to materialise ``solve_data/`` and ``output_raw/``.
        Created if absent.  No CSVs are written into ``solve_data/``
        beyond the small support cluster the output writer needs.
    logger : logging.Logger, optional
        Logger.  Defaults to a scenario-named logger.
    emit_output : bool, default True
        When False, skip the output-writer adapter call and the
        support CSV writes.  Useful for benchmarking the LP-build
        path in isolation.

    Returns
    -------
    OrchestrationStep
        With ``solve_name = scenario_name``, the live HiGHS solution,
        a stub :class:`SolveHandoff` (no carriers — single-solve mode
        has no next solve to hand off to), and ``warm_used = False``.

    Raises
    ------
    FastLoadError
        Override-chain helpers couldn't populate a required FlexData
        field.  Message names the field; investigate the helper.
    FlexToolSolveError
        LP infeasible / non-optimal.
    """
    if logger is None:
        logger = logging.getLogger(
            f"flextool.engine_polars.run_single_solve_from_db[{scenario_name}]"
        )

    import time as _time

    db_url = str(input_db_url)
    if "://" not in db_url:
        db_url = f"sqlite:///{db_url}"

    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)

    # 1. Construct the SpineDbReader once.  Phase 4.6: build axis enums
    # against the SpineDBBackend and thread them so the reader casts on
    # emit — matching the activation path in ``load_flextool``.
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.spinedb_backend._axis_enums import (
        build_axis_enums,
        load_axis_contract,
    )
    from flextool.spinedb_backend import SpineDBBackend
    _t0 = _time.perf_counter()
    _se_axis_enums = None
    _se_contract = None
    try:
        _se_contract = load_axis_contract()
        with SpineDBBackend(db_url, None) as _se_ab:
            _se_axis_enums = build_axis_enums(_se_ab, _se_contract)
    except Exception:  # noqa: BLE001
        _se_axis_enums = None
        _se_contract = None
    reader = SpineDbReader(
        db_url, scenario=scenario_name,
        axis_enums=_se_axis_enums, contract=_se_contract,
    )
    print(f"Input: DB reader open: {_time.perf_counter() - _t0:.3f}s")

    # 2. Load SolveConfig + TimelineConfig (Γ.8.A / Γ.8.B).  These
    # populate per-solve config the override chain consumes implicitly
    # (timeline-derived dt, period_timeset cascades).  In single-solve
    # mode we only need them for cross-validation; the fast loader
    # consumes the SpineDbReader directly.
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig
    _t0 = _time.perf_counter()
    sc = SolveConfig.load_from_source(reader, logger=logger)
    tc = TimelineConfig.load_from_source(reader, logger=logger)
    tc.create_assumptive_parts(sc)
    tc.create_timeline_from_timestep_duration(sc)
    print(f"Input: solve/timeline config: {_time.perf_counter() - _t0:.3f}s")

    # 3. Build the FlexData via the source-only loader (Δ.25).
    from flextool.engine_polars._fast_load import load_flextool_source_only
    print("Input: override chain passes:")
    _t0 = _time.perf_counter()
    flex_data = load_flextool_source_only(
        reader, work_folder, logger=logger,
    )
    print(f"Input: total override chain: {_time.perf_counter() - _t0:.3f}s")

    # 4. Build the LP.
    from polar_high import Problem
    from flextool.engine_polars.model import build_flextool

    # --- LP scaling -------------------------------------------------------
    # Phase 2b — the legacy ``scaling.analyze_solve`` /
    # ``resolve_effective_scaling`` pipeline was retired in favour of the
    # autoscale package; the single-solve path now resolves the
    # effective objective scale directly from the user's DB override
    # (defaulting to the legacy 1e-6 when absent) and lets autoscale
    # Layer 2 / Layer 3 (applied below) handle residual cost / bound
    # magnitudes.
    user_obj_scale = sc.scale_the_objective.get(scenario_name)
    effective_obj_scale = _resolve_effective_obj_scale(user_obj_scale)
    # ``FLEXTOOL_USER_BOUND_SCALE`` env var (set by --user-bound-scale CLI
    # flag) takes priority over the DB ``solve.user_bound_scale`` value.
    _cli_ubs = os.environ.get("FLEXTOOL_USER_BOUND_SCALE")
    user_bound_scale_override = _resolve_user_bound_scale_override(
        _cli_ubs if _cli_ubs is not None
        else sc.user_bound_scale.get(scenario_name)
    )

    _t0 = _time.perf_counter()
    problem = Problem(auto_user_bound_scale=True)
    build_flextool(problem, flex_data, scale_the_objective=effective_obj_scale)
    print(f"Input: LP build: {_time.perf_counter() - _t0:.3f}s")

    # Baseline HiGHS solver options — Curtis-Reid simplex scale, the
    # determinism pin, and an explicit ``user_bound_scale`` only when
    # the operator supplied one (CLI / DB).  Layer 3's
    # :func:`apply_layer3` will merge ``user_objective_scale``,
    # ``user_bound_scale`` (when it auto-recommends), and re-assert
    # ``simplex_scale_strategy`` on top of this base.
    highs_options = _baseline_highs_options(
        user_bound_scale_override=user_bound_scale_override,
    )
    # ``FLEXTOOL_HIGHS_PRESOLVE`` env var (set by --presolve CLI flag)
    # overrides DETERMINISM_OPTIONS' baked-in ``presolve = "on"``.
    _cli_presolve = os.environ.get("FLEXTOOL_HIGHS_PRESOLVE")
    if _cli_presolve in ("on", "off", "choose"):
        highs_options["presolve"] = _cli_presolve

    problem.set_solver_options(highs_options)

    # autoscale Layer 2 (semantic per-type) pre-solve apply — see the
    # cascade path's wire-in for the rationale.  None when the
    # autoscaler is disabled or the Layer-1 trigger does not fire.
    _autoscale_layer2_plan = _autoscale_apply_layer2_pre_solve(
        problem, solve_name=scenario_name, logger=logger,
    )
    _autoscale_layer3_plan = _autoscale_apply_layer3_pre_solve(
        problem,
        layer2_plan=_autoscale_layer2_plan,
        solve_name=scenario_name,
        logger=logger,
    )

    # 5. Solve.  Phase 3 — dispatch through ``run_one_solve`` so the
    # commercial-solver path works end-to-end.  The default HiGHS path
    # is byte-identical to the pre-Phase-3 ``problem.solve(keep_solver=True)``
    # call (``run_one_solve`` short-circuits to that exact invocation when
    # ``solver_config.name == 'highs'``).
    from flextool.engine_polars._solver_dispatch import run_one_solve
    from flextool.engine_polars._solve_config import (
        SolverConfig as _SolverConfig,
    )
    solver_cfg = sc.solver_configs.get(scenario_name, _SolverConfig())
    sol = run_one_solve(problem, solver_cfg, logger=logger)
    # Eager Layer-2 unscale before any output writer touches ``sol``.
    _autoscale_unscale_post_solve(
        sol, _autoscale_layer2_plan, solve_name=scenario_name, logger=logger,
    )
    _autoscale_emit_layer1(
        sol, solve_name=scenario_name, logger=logger, work_folder=work_folder,
        layer2_plan=_autoscale_layer2_plan,
        layer3_plan=_autoscale_layer3_plan,
    )
    if not sol.optimal:
        logger.error(
            "fast single-solve: HiGHS returned non-optimal status "
            "(%s) for scenario %s; obj=%r",
            getattr(sol, "status", None), scenario_name,
            getattr(sol, "obj", None),
        )

    # 6. Output emission — materialise the FlexData to flextool's CSV
    # layout (input/, solve_data/) so handoff_writers, read_parameters,
    # and the wide-format CSV writers downstream find their inputs.
    # Then write the small support-CSV cluster the output_raw writer
    # adapter needs and call the adapter.  All steps tolerate partial
    # state — handoff writers log warnings on individual failures.
    if emit_output and sol.optimal:
        from flextool.engine_polars._native_input_writer import (
            write_output_support_csvs,
        )
        from flextool.engine_polars._output_writer import (
            OutputWriterState, write_outputs_for_solve,
        )

        # Δ.30 — wire dump_csvs into the fast path so handoff_writers
        # (input/p_entity_unitsize.csv, input/process_unit.csv, …) and
        # the post-solve wide-format CSV / parquet writers
        # (read_parameters.py: solve_data/p_node.csv, p_process_sink.csv,
        # p_commodity.csv, …) find their inputs.  Without this only
        # output_raw is produced; output_csv / output_parquet / etc. fail.
        #
        # ``run_single_solve_from_db`` always dumps ``flex_data`` to disk
        # — the downstream output writers in this path read CSVs back
        # from ``solve_data/``.  This is independent of ``--csv-dump``;
        # the fast single-solve path does not use the streaming
        # in-memory cascade.
        flex_data.dump_csvs(work_folder)
        write_output_support_csvs(
            flex_data, work_folder, solve_name=scenario_name,
        )
        writer_state = OutputWriterState()
        write_outputs_for_solve(
            sol,
            work_folder=work_folder,
            solve_name=scenario_name,
            prior_handoff=None,
            writer_state=writer_state,
            csv_dump=csv_dump,
        )

    # Always emit ``scale_the_objective.csv`` (even on non-optimal solves —
    # downstream un-scaling needs it for any partial output the writer
    # produced).  The legacy TXT diagnostic was retired in Phase 2b; the
    # autoscaler's YAML report (``solve_data/autoscale_<solve>.yaml``,
    # written from ``_autoscale_emit_layer1`` above) is the replacement.
    _write_scale_csv(
        solve_data_dir=Path(work_folder) / "solve_data",
        solve_name=scenario_name,
        effective_obj_scale=effective_obj_scale,
        logger=logger,
    )

    # 7. Build a stub SolveHandoff (no carriers in single-solve mode).
    from flextool.engine_polars._solve_handoff import SolveHandoff
    handoff = SolveHandoff()

    # Un-scale the objective value back to user-facing units.
    # ``build_flextool`` multiplied the objective coefficients by
    # ``effective_obj_scale``, so HiGHS reports a scaled value.
    # Overwrite ``sol.obj`` in place so the public ``step.solution.obj``
    # matches the unscaled ``step.obj`` / the ``v_obj__{solve}.parquet``
    # value that the legacy writer un-scales via
    # ``_resolve_inv_scale_the_objective``.
    unscaled_obj = (
        sol.obj / effective_obj_scale
        if sol.optimal and sol.obj is not None
        else None
    )
    if sol is not None and unscaled_obj is not None:
        sol.obj = unscaled_obj

    return OrchestrationStep(
        solve_name=scenario_name,
        solution=sol,
        handoff=handoff,
        obj=unscaled_obj,
        optimal=bool(getattr(sol, "optimal", False)) if sol is not None else None,
        warm_used=False,
        flex_data=flex_data,
    )


__all__ = [
    "OrchestrationStep",
    "run_orchestration",
    "run_chain_from_db",
    "run_single_solve_from_db",
]
