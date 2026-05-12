"""Native flexpy orchestrator — Γ.8.D conductor.

This module is the master loop replacement for
``flextool/flextoolrunner/orchestration.py:run_model`` (638 LOC).
The Γ.8.D port lands a flexpy-native driver that:

* Combines the foundation modules (Γ.8.A ``_solve_config`` +
  Γ.8.B ``_timeline`` + Γ.8.C ``_recursive_solve`` + ``_stochastic``).
* Drives the per-solve preprocessing using flextool's existing
  preprocessing modules (the L0-L9 batch + ``preprocessing_solve_time``
  + ``solve_writers``) — those CSV writers stay the source of truth
  while flexpy's `load_flextool` continues to read them.
* Runs the actual solve via ``polar_high.Problem.solve`` (HiGHS)
  instead of glpsol/AMPL.
* Captures :class:`SolveHandoff` per solve via the native
  :func:`build_handoff_from_flexpy`, threads it forward as
  ``prior_handoff``, and routes it into the in-memory handoff slot of
  flextool's runner so the consume side (``preprocessing_solve_time``,
  ``handoff_writers``, ``cumulative_handoffs``) reads from it.

Design choices
--------------

* **CSV writers are still flextool's** — replacing them is a separate
  phase (Γ.7 / Γ.9 in the audit numbering).  Γ.8.D's job is to run the
  master loop natively, not to retire CSVs.  This means the orchestrator
  drives ``flextoolrunner.orchestration.run_model`` once per top-level
  invocation, but with a **flexpy-as-inner-solver** wrapper that:
    - Reads the per-solve snapshot via ``load_flextool``.
    - Builds the LP via ``build_flextool``.
    - Solves via ``polar_high`` (HiGHS).
    - Captures handoff via ``build_handoff_from_flexpy``.
    - Deposits handoff into ``state.handoffs`` for the next iteration's
      preprocessing.

* **Storage-fixing handoff is in-memory by default** when
  ``state.handoffs`` is non-None.  The legacy file-copy path
  (``shutil.copy`` of ``solve_data_<parent>/fix_storage_*.csv``) is
  consulted only when ``state.handoffs is None`` — see
  ``flextool/flextoolrunner/orchestration.py:362-431``.  This is a
  documented behaviour divergence: flexpy's preferred path is in-memory
  and avoids touching disk.

* **Roll-counter reset semantics**: every top-level
  :func:`run_orchestration` call invokes
  ``state.solve.roll_counter = state.solve.make_roll_counter()`` first
  so test re-use of the same SolveConfig doesn't desync (R-O5).

* **`model_solve` validation**: enforced loud-and-early.  Empty
  ``model_solve`` or more-than-one model raises
  :class:`FlexToolConfigError` per the audit's
  ``orchestration.py:634-637``.

* **Δ.12e**: the legacy file-symlink ``run_chain(native=False)`` driver
  retired once the native cascade reached feature parity (warm-LP,
  handoff carriers, output writer, override chain).  ``run_chain`` is
  now a thin compat shim that always delegates here.

Reference: ``flextool/flextoolrunner/orchestration.py``.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from flextool.engine_polars._solve_handoff import SolveHandoff
from flextool.engine_polars._solve_state import (
    FlexToolConfigError,
    PathConfig,
    RunnerState,
)
from flextool.engine_polars import scaling as _scaling

if TYPE_CHECKING:
    from polar_high import Solution

    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._timeline import TimelineConfig
    from flextool.engine_polars.input import FlexData


# Repository pin — flextool's preprocessing modules live in the same
# tree as the engine_polars package, but the legacy ``flextool``
# top-level (separate repo at ``/home/jkiviluo/sources/flextool``) is
# also available for some Spine fixtures.  Mirror the path-shim used
# elsewhere so imports of ``flextool.flextoolrunner.flextoolrunner``
# resolve when running against either tree.
_REPO_ROOT = Path("/home/jkiviluo/sources/flextool")


def _ensure_flextool_importable() -> None:
    if str(_REPO_ROOT) not in sys.path:
        sys.path.append(str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Opt-in memory diagnostics
# ---------------------------------------------------------------------------


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
    :class:`flextool.flextoolrunner.timing_recorder.TimingRecorder.record`)
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

    def __init__(self, csv_path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.t0 = time.perf_counter()
        self._path = Path(csv_path)
        self._started = False
        if not self.enabled:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        import csv as _csv
        with open(self._path, "w", newline="") as f:
            _csv.writer(f).writerow(self._HEADER)

    @staticmethod
    def _read_rss_mb() -> float:
        """Read VmRSS (kB) from /proc/self/status and return MB.

        Returns 0.0 if /proc isn't available (non-Linux) or the line
        isn't found.  We never want diagnostics to raise.
        """
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # "VmRSS:    123456 kB"
                        parts = line.split()
                        if len(parts) >= 2:
                            return float(parts[1]) / 1024.0
        except OSError:
            pass
        return 0.0

    def checkpoint(self, label: str, logger: logging.Logger) -> None:
        if not self.enabled:
            return
        import tracemalloc
        if not self._started:
            tracemalloc.start()
            self._started = True
        current, peak = tracemalloc.get_traced_memory()
        rss_mb = self._read_rss_mb()
        t_elapsed = time.perf_counter() - self.t0
        current_mb = current / (1024.0 * 1024.0)
        peak_mb = peak / (1024.0 * 1024.0)
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
        # Always emit one-liner so the user sees phase progress even when
        # the inner runner's logger has been silenced (run_orchestration
        # sets ``runner.state.logger.setLevel(ERROR)``).  The caller's
        # logger gets the INFO record (cheap when level filters it out);
        # the unconditional stdout write is what surfaces in the GUI.
        logger.info(
            "[mem] %s @ t=%.1fs  rss=%.0f MB  traced_peak=%.0f MB",
            label, t_elapsed, rss_mb, peak_mb,
        )
        try:
            print(
                f"[mem] {label} @ t={t_elapsed:.1f}s  "
                f"rss={rss_mb:.0f} MB  traced_peak={peak_mb:.0f} MB",
                flush=True,
            )
        except OSError:
            pass


class _NoopMemoryRecorder:
    """Zero-overhead drop-in when ``FLEXTOOL_MEMORY_DIAGNOSTICS`` is unset.

    Both attributes accessed by the rest of the module
    (:meth:`checkpoint` and :attr:`enabled`) are present so callers don't
    need to branch on ``is None`` at every site.
    """

    enabled = False

    def checkpoint(self, label: str, logger: logging.Logger) -> None:  # noqa: D401, ARG002
        return None


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
        The HiGHS solution.  ``None`` only on the failed-solve path.
    handoff : SolveHandoff
        Flexpy-derived handoff carriers, threaded forward.
    obj : float | None
        Objective value (cached for quick comparison; equal to
        ``solution.obj``).
    warm_used : bool
        Δ.12d — True if this solve was produced by warm-updating the
        prior solve's :class:`polar_high.WarmProblem` instance; False
        if it was a cold rebuild.  Always False for the first solve
        and for ``warm=False`` runs.
    flex_data : FlexData | None
        Δ.31 — the polars input bundle this sub-solve consumed.  Held
        on the step so downstream :func:`flextool.process_outputs.
        write_outputs` can build the parameter / set namespaces in
        memory instead of re-parsing the workdir CSVs.  ``None`` only
        on the failed-load path (the build-LP step would also have
        failed, so callers usually short-circuit before reading it).
    """

    solve_name: str
    solution: "Solution | None"
    handoff: SolveHandoff
    obj: float | None = None
    warm_used: bool = False
    flex_data: "FlexData | None" = None


# ---------------------------------------------------------------------------
# Scaling-output helper  (shared by cascade & single-solve paths)
# ---------------------------------------------------------------------------


def _write_scale_csv_and_report(
    *,
    solve_data_dir: Path,
    output_raw_dir: Path,
    solve_name: str,
    scale_table: "_scaling.ScaleTable",
    effective_row_scaling: str,
    effective_obj_scale: float,
    user_row_scaling: object | None,
    flex_data: "FlexData",
    solution: "Solution | None",
    logger: logging.Logger,
    write_csv: bool = True,
    write_report: bool = True,
) -> None:
    """Emit ``scale_the_objective.csv`` and ``scaling_report.txt``.

    The CSV is required by the downstream parquet/CSV writers — they read
    it via :func:`flextool.process_outputs.read_highs_solution.
    _resolve_inv_scale_the_objective` to un-scale variable values back to
    user-facing units.  The TXT report is a human-readable diagnostic.

    Both writes are best-effort: failures log a warning but do not raise.

    ``write_csv`` / ``write_report`` allow callers to suppress either
    artifact independently — e.g. the cascade gates the CSV per
    ``base_solve_name`` (its value is invariant across rolls) and gates
    the diagnostic report behind ``FLEXTOOL_SCALING_REPORT=1``.
    """
    if write_csv:
        try:
            from flextool.flextoolrunner.solve_writers import (
                write_scale_the_objective,
            )
            write_scale_the_objective(solve_data_dir, effective_obj_scale)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scale_the_objective.csv write failed for %s: %s",
                solve_name, exc,
            )

    if not write_report:
        return
    try:
        from flextool.engine_polars.scaling_report import write_scaling_report
        write_scaling_report(
            scale_table=scale_table,
            flex_data=flex_data,
            solve_data_dir=solve_data_dir,
            solve_name=solve_name,
            solution=solution,
            output_raw_dir=output_raw_dir,
            applied_row_scaling=effective_row_scaling,
            applied_obj_scale=effective_obj_scale,
            override_source=(
                "user_db_setting"
                if isinstance(user_row_scaling, str)
                and user_row_scaling.strip().lower() in ("yes", "no")
                else None
            ),
            stdout_summary=True,
            logger=logger,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scaling report failed for %s: %s", solve_name, exc)


# ---------------------------------------------------------------------------
# Master loop
# ---------------------------------------------------------------------------


def _validate_model_solve(state: RunnerState) -> list[str]:
    """Validate ``state.solve.model_solve`` and return the solve list.

    Mirrors the guard at ``flextool/flextoolrunner/orchestration.py:78``
    and ``:634-637``: there must be exactly one model with at least one
    solve.  Multi-model is documented as unsupported.
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
) -> dict[str, OrchestrationStep]:
    """Drive the master loop natively.

    Per-step:

    1. Bootstrap directories (idempotent).
    2. Validate ``state.solve.model_solve`` (exactly one model, ≥1 solve).
    3. Reset ``state.solve.roll_counter`` for repeatable test runs (R-O5).
    4. Drive flextool's ``orchestration.run_model`` with a flexpy
       cascade solver — each per-solve iteration loads the snapshot via
       ``load_flextool``, builds the LP via ``build_flextool``, solves
       via HiGHS, captures the handoff, and deposits it into
       ``state.handoffs`` (which the consume side already reads from).

    Parameters
    ----------
    state : RunnerState
        Native flexpy state carrier.  ``state.solve`` and
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
    _ensure_flextool_importable()
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

    # The cascade solver runs flexpy on each solve and captures the
    # handoff.  We use flextool's orchestration loop driver because it
    # encodes the recursive/rolling/stochastic expansion + per-solve
    # preprocessing chain we still consume.  Our cascade solver is the
    # `solver.run(...)` callback inside that loop.
    return _drive_cascade(state, work_folder, solves, runner_factory,
                          db_url=db_url, scenario_name=scenario_name,
                          warm=warm)


def _drive_cascade(
    state: RunnerState,
    work_folder: Path,
    solves: list[str],
    runner_factory,
    *,
    db_url: str | None = None,
    scenario_name: str | None = None,
    warm: bool = False,
) -> dict[str, OrchestrationStep]:
    """Drive the flextool master loop with a flexpy cascade solver.

    For every per-solve iteration:

    1. Read the snapshot via ``load_flextool``.
    2. Build the LP via ``build_flextool`` (cold rebuild) OR warm-update
       the prior iteration's :class:`polar_high.WarmProblem`.
    3. Solve via HiGHS.
    4. Build the handoff via ``build_handoff_from_flexpy``.
    5. Deposit it into ``state.handoffs`` so the next iteration's
       preprocessing picks it up.

    Parameter ``warm`` toggles per-iteration warm-LP updates: when True,
    the cascade reuses one ``WarmProblem`` across consecutive
    structurally-compatible iterations.  See
    :mod:`flextool.engine_polars._warm` for the structural-fingerprint
    + Param-classification machinery.  Cold rebuild (``warm=False``)
    remains the default for backward compatibility with every existing
    caller.

    This mirrors :func:`flextool.engine_polars.input.load_flextool_from_db`'s
    multi-solve cascade but emits an :class:`OrchestrationStep` per solve
    and runs the LAST solve too (the flexpy-side bookkeeping is the
    deliverable, not just an intermediate).
    """
    # Late imports — keep the orchestration module's import surface narrow
    # for callers that only need the dataclass.
    from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
    from flextool.flextoolrunner.solver_runner import SolverRunner
    from flextool.engine_polars._native_run_model import native_run_model

    from polar_high import Problem, WarmProblem
    from flextool.engine_polars.input import (
        build_handoff_from_flexpy,
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
    # loaded from.  Since RunnerState in flexpy doesn't carry a DB URL
    # by default, callers must supply this via runner_factory or use
    # ``run_chain_from_db`` which constructs the runner explicitly.
    if runner_factory is None:
        raise FlexToolConfigError(
            "run_orchestration requires a runner_factory to construct "
            "the underlying FlexToolRunner.  Use run_chain_from_db for "
            "the canonical end-to-end path that wires this for you."
        )
    runner = runner_factory()
    # Push our state's handoff slot onto the runner's state so flextool's
    # consume hooks (preprocessing_solve_time + capture_post_solve) read
    # from the same dict.
    runner.state.handoffs = state.handoffs
    runner.state.logger.setLevel(logger_level := logging.ERROR)
    # Forward the opt-in memory recorder (no-op when env var unset) so
    # ``_FlexpyCascadeSolver.run`` can fire the first-iter checkpoints.
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
        try:
            cascade_db_reader = SpineDbReader(db_url, scenario=scenario_name)
        except Exception:  # noqa: BLE001
            cascade_db_reader = None

    class _FlexpyCascadeSolver(SolverRunner):
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
            # Per-base-solve gating for the scaling CSV + diagnostic report.
            # The CSV value (effective_obj_scale) is invariant across rolls
            # of the same base solve — and the TXT report is diagnostic
            # only.  We track which base solve names we've already written
            # each artifact for, so subsequent rolls skip the work.  The
            # TXT report additionally requires ``FLEXTOOL_SCALING_REPORT=1``
            # to be set, unless the solve is non-optimal (in which case we
            # always emit it once for actionable diagnostics).
            self._scale_csv_written: set[str] = set()
            self._scale_report_written: set[str] = set()
            # Opt-in memory diagnostics — fire the first-iter checkpoints
            # exactly once (subsequent iterations leave the flag set).
            self._mem_first_load_done: bool = False

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
            data = load_flextool(
                self.state.paths.work_folder,
                handoff=prior_for_load,
                db_reader=cascade_db_reader,
            )
            # Release heap held by the broadcast cascade scratch frames.
            # On H2_trade y2050 this drops RSS ~1.6 GB / 41 %; expected
            # to scale with timeline size.  No-op on non-glibc.
            _try_malloc_trim()
            # Memory checkpoint (opt-in, first iteration only).
            _memrec_local = getattr(self.state, "_memory_recorder", None)
            if _memrec_local is not None and not self._mem_first_load_done:
                _memrec_local.checkpoint(
                    "first_load_flextool_end", self.state.logger,
                )

            # --- LP scaling -------------------------------------------------
            # The analyser is keyed on the base solve name (the rolling
            # ``_roll_N`` suffix is stripped) so every iteration of a
            # rolling cascade reuses the first iteration's ScaleTable
            # via ``_scale_cache``.  The user's per-solve DB overrides
            # win when present and well-formed; otherwise we apply the
            # analyser's recommendation.
            base_solve_name = re.sub(r"_roll_\d+$", "", complete_solve_name)
            scale_table = _scaling.analyze_solve(
                solve_name=base_solve_name,
                flex_data=data,
                work_folder=self.state.paths.work_folder,
                logger=self.state.logger,
            )
            user_row_scaling = state.solve.use_row_scaling.get(complete_solve_name)
            user_obj_scale = state.solve.scale_the_objective.get(complete_solve_name)
            effective_row_scaling, effective_obj_scale = (
                _scaling.resolve_effective_scaling(
                    scale_table, user_row_scaling, user_obj_scale,
                )
            )
            # ``user_bound_scale`` resolution: explicit DB override wins
            # (typically the value HiGHS recommends in its "user-scaled
            # problem has some excessively large row bounds" warning);
            # otherwise fall back to the input-data heuristic in
            # ``recommended_highs_options``.
            user_bound_scale_override = _scaling.resolve_user_bound_scale_override(
                state.solve.user_bound_scale.get(complete_solve_name)
            )

            # HiGHS solver options are picked AFTER the LP is built so we
            # can read the actual coefficient ranges off the assembled
            # numpy arrays (via ``Problem.peek_lp_ranges``) instead of
            # guessing from input-data ranges.  ``simplex_scale_strategy``
            # = advanced (Curtis-Reid) is always-on; ``user_bound_scale``
            # comes from the explicit DB override → LP-range
            # recommendation → input-data heuristic (in priority order).
            # Cap solve time via env var if the operator requested it.
            _diag_tlim = os.environ.get("FLEXTOOL_HIGHS_TIME_LIMIT")

            def _finalise_highs_options(opts: dict) -> dict:
                if _diag_tlim:
                    try:
                        opts["time_limit"] = float(_diag_tlim)
                    except ValueError:
                        pass
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
                    if (
                        _memrec_local is not None
                        and not self._mem_first_load_done
                    ):
                        _memrec_local.checkpoint(
                            "first_lp_build_end", self.state.logger,
                        )
                    inner_pb = self._warm_problem.problem
                    lp_ranges = inner_pb.peek_lp_ranges()
                    highs_options = _finalise_highs_options(
                        _scaling.recommended_highs_options(
                            scale_table,
                            user_bound_scale_override=user_bound_scale_override,
                            lp_ranges=lp_ranges,
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
                pb = Problem()
                build_flextool(pb, data, scale_the_objective=effective_obj_scale)
                if (
                    _memrec_local is not None
                    and not self._mem_first_load_done
                ):
                    _memrec_local.checkpoint(
                        "first_lp_build_end", self.state.logger,
                    )
                lp_ranges = pb.peek_lp_ranges()
                highs_options = _finalise_highs_options(
                    _scaling.recommended_highs_options(
                        scale_table,
                        user_bound_scale_override=user_bound_scale_override,
                        lp_ranges=lp_ranges,
                    )
                )
                pb.set_solver_options(highs_options)
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
            # Memory checkpoint after the first solve completes — and
            # latch the first-iter flag so the four first-iter
            # checkpoints don't re-fire on subsequent rolls.
            if _memrec_local is not None and not self._mem_first_load_done:
                _memrec_local.checkpoint(
                    "first_solve_end", self.state.logger,
                )
                self._mem_first_load_done = True
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
            # Write scaling diagnostic (CSV + report incl. section 8.5)
            # BEFORE the optimality check, so a non-optimal / time-limited
            # solve still produces actionable scaling info.
            #
            # Per-base-solve gating: the CSV value is invariant across
            # rolls of the same base solve, so emit it only on the first
            # roll.  The diagnostic TXT is gated behind
            # ``FLEXTOOL_SCALING_REPORT=1`` (also once per base solve),
            # except for non-optimal solves which always force one emit
            # for actionable diagnostics.
            _t_scale_start = time.perf_counter() if _phase_timing else 0.0
            _report_env = os.environ.get("FLEXTOOL_SCALING_REPORT") == "1"
            _force_report = not sol.optimal
            _write_csv = base_solve_name not in self._scale_csv_written
            _write_report = (
                _force_report
                or (
                    _report_env
                    and base_solve_name not in self._scale_report_written
                )
            )
            if _write_csv or _write_report:
                _write_scale_csv_and_report(
                    solve_data_dir=self.state.paths.work_folder / "solve_data",
                    output_raw_dir=self.state.paths.work_folder / "output_raw",
                    solve_name=complete_solve_name,
                    scale_table=scale_table,
                    effective_row_scaling=effective_row_scaling,
                    effective_obj_scale=effective_obj_scale,
                    user_row_scaling=user_row_scaling,
                    flex_data=data,
                    solution=sol,
                    logger=self.state.logger,
                    write_csv=_write_csv,
                    write_report=_write_report,
                )
                if _write_csv:
                    self._scale_csv_written.add(base_solve_name)
                if _write_report:
                    self._scale_report_written.add(base_solve_name)
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
                    f"flexpy non-optimal for {complete_solve_name}"
                )
                return 1

            prior = prior_for_load
            # Δ.30 — materialise FlexData to flextool's CSV layout so
            # handoff_writers + downstream wide-format CSV/parquet
            # writers find their inputs.  Idempotent / overwrites; the
            # legacy preprocessing already wrote most of these but
            # dump_csvs ensures the in-memory canonical wins, keeping
            # FlexData → CSV in sync per solve.
            _t_dump_start = time.perf_counter() if _phase_timing else 0.0
            try:
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
            # Δ.1 — emit TIER A output_raw artefacts BEFORE the in-memory
            # handoff is built.  ``write_all_handoffs`` (called by the
            # adapter) refreshes ``solve_data/period_capacity.csv`` and
            # other handoff CSVs; ``build_handoff_from_flexpy`` then
            # reads those refreshed files for the in-memory handoff.
            _t_wofs_start = time.perf_counter() if _phase_timing else 0.0
            try:
                write_outputs_for_solve(
                    sol,
                    work_folder=self.state.paths.work_folder,
                    solve_name=complete_solve_name,
                    prior_handoff=prior,
                    writer_state=writer_state,
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
            handoff = build_handoff_from_flexpy(
                sol, self.state.paths.work_folder, complete_solve_name,
                prior_handoff=prior,
                flex_data=data,
                parent_handoff=parent_handoff,
            )
            if _phase_timing:
                _tr.record(
                    "handoff_part",
                    subphase="build_handoff_from_flexpy",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_bhf_start,
                    t_start=_t_bhf_start,
                )
            # Deposit so flextool's consume side picks it up on the next
            # iteration's preprocessing AND so we have it for the result
            # dict.
            #
            # IMPORTANT: flextool's ``orchestration.run_model`` also
            # calls ``capture_post_solve(state, complete_solve[solve])``
            # immediately after ``solver.run`` returns.  That helper
            # reads the current ``solve_data/*.csv`` files and OVERWRITES
            # the carriers on the same handoff object — which would
            # silently replace our flexpy-derived ``realized_invest``
            # with whatever the prior-handoff-seeded preprocessing
            # wrote to disk for THIS solve (i.e., the prior solve's
            # state, not our actual flexpy result).  We monkey-patch
            # ``capture_post_solve`` to a no-op for the duration of
            # this loop in ``_drive_cascade`` below — the override
            # restores it on exit.
            self.state.handoffs[complete_solve_name] = handoff
            # Un-scale the objective value back to user-facing units.
            # ``build_flextool`` multiplied the objective coefficients by
            # ``effective_obj_scale``, so HiGHS reports a scaled value.
            unscaled_obj = (
                sol.obj / effective_obj_scale if sol.obj is not None else None
            )
            self._all_steps[complete_solve_name] = OrchestrationStep(
                solve_name=complete_solve_name,
                solution=sol,
                handoff=handoff,
                obj=unscaled_obj,
                warm_used=warm_used,
                flex_data=data,
            )
            if _phase_timing:
                _tr.record(
                    "per_iter",
                    subphase="handoff",
                    solve=complete_solve_name,
                    roll_index=_roll_idx,
                    seconds=time.perf_counter() - _t_handoff_start,
                    t_start=_t_handoff_start,
                )
            return 0

    # Phase 3 — drive the cascade via the native ``native_run_model``.
    # The legacy ``flextoolrunner.orchestration.run_model`` is no longer
    # invoked from the engine_polars cascade.  ``native_run_model``
    # deliberately omits the ``capture_post_solve`` call after each
    # ``solver.run`` — equivalent to the previous monkey-patch-to-no-op
    # but cleaner.  Belt-and-suspenders: also patch the legacy hook in
    # ``flextool.flextoolrunner.orchestration`` so any other consumer
    # that may still reference it via that module keeps the no-op
    # semantic for the duration of the cascade.
    from flextool.flextoolrunner import orchestration as _flx_orch
    from flextool.engine_polars._native_input_writer import _native_leaf_set_override
    _real_capture = _flx_orch.capture_post_solve
    _flx_orch.capture_post_solve = lambda state, solve_name: None
    try:
        with _native_leaf_set_override():
            native_run_model(runner.state, _FlexpyCascadeSolver(runner.state))
    finally:
        _flx_orch.capture_post_solve = _real_capture
    # Mirror the in-memory handoff dict back onto our state in case
    # callers want to inspect it.
    state.handoffs = runner.state.handoffs
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
    bin_dir: Path | str | None = None,
    logger: logging.Logger | None = None,
    warm: bool = False,
) -> dict[str, OrchestrationStep]:
    """Run a flextool multi-solve scenario end-to-end natively.

    Combines:

    1. :func:`flextool.engine_polars._native_input_writer.write_workdir_inputs`
       (Δ.20 — engine_polars-owned) populates the workdir's ``input/``
       and ``solve_data/`` CSVs.  Replaces the legacy
       ``FlexToolRunner.write_input`` call from earlier phases — the
       cascade no longer calls into the FlexToolRunner.write_input
       method directly.
    2. ``_orchestration.run_orchestration`` to drive the per-solve loop
       with a flexpy-as-inner-solver wrapper.
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
    flextool_dir, bin_dir : Path, optional
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

    Returns
    -------
    dict[str, OrchestrationStep]
        Mapping ``complete_solve_name → OrchestrationStep``.  Iterate
        in insertion order to walk the chain.
    """
    _ensure_flextool_importable()
    from flextool.flextoolrunner.flextoolrunner import FlexToolRunner

    if logger is None:
        logger = logging.getLogger(
            f"flexpy.run_chain_from_db[{scenario_name}]"
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
        formatted = ", ".join(f"{n}={s}" for n, s in statuses.items())
        logger.info("Solver license status: %s", formatted)
    except ImportError:  # pragma: no cover — older polar_high without dispatch
        pass

    db_url = str(input_db_url)
    if not db_url.startswith("sqlite:") and not db_url.startswith("postgresql"):
        db_url = f"sqlite:///{db_url}"

    if work_folder is None:
        work_folder = Path(tempfile.mkdtemp(prefix="flexpy_run_chain_"))
    else:
        work_folder = Path(work_folder)
        work_folder.mkdir(parents=True, exist_ok=True)

    flextool_dir_resolved = (
        Path(flextool_dir) if flextool_dir is not None
        else _REPO_ROOT / "flextool"
    )
    bin_dir_resolved = (
        Path(bin_dir) if bin_dir is not None else _REPO_ROOT / "bin"
    )

    # Δ.20 — workdir CSV population is now owned by engine_polars.
    # ``_native_input_writer.write_workdir_inputs`` invokes the input
    # writers from inside engine_polars; the live cascade no longer
    # references ``FlexToolRunner.write_input``.  Future Δ.20+ phases
    # progressively replace the underlying writers with InputSource-
    # driven helpers.
    from flextool.engine_polars._native_input_writer import (
        write_workdir_inputs,
    )

    # Opt-in memory diagnostics (FLEXTOOL_MEMORY_DIAGNOSTICS=1).  Pin the
    # CSV under ``solve_data/`` so it sits next to ``timings.csv`` and the
    # other workdir artefacts.  When the env var is unset we still
    # construct a no-op recorder so downstream sites can call
    # ``.checkpoint(...)`` unconditionally — at zero cost.
    _mem_enabled = os.environ.get("FLEXTOOL_MEMORY_DIAGNOSTICS") == "1"
    if _mem_enabled:
        (work_folder / "solve_data").mkdir(parents=True, exist_ok=True)
        _memrec: _MemoryRecorder | _NoopMemoryRecorder = _MemoryRecorder(
            work_folder / "solve_data" / "memory_diagnostics.csv",
            enabled=True,
        )
    else:
        _memrec = _NoopMemoryRecorder()
    _memrec.checkpoint("cascade_start", logger)

    write_workdir_inputs(
        db_url,
        scenario_name,
        work_folder,
        logger=logger,
    )
    # The legacy CSV writer (2.4 kLOC pure-Python loops) allocates and
    # frees a lot of pandas/dict scratch state; glibc's heap retains the
    # freed pages.  Release them before the polars-heavy ``load_flextool``
    # starts so the heap watermark doesn't compound.
    _try_malloc_trim()
    _memrec.checkpoint("write_workdir_inputs_end", logger)

    # Construct the underlying FlexToolRunner — still needed to carry
    # the cross-cutting ``RunnerState`` (timeline, solve config, handoff
    # dict) through ``flextool.flextoolrunner.orchestration.run_model``,
    # which the native cascade still drives for the per-solve
    # preprocessing chain (``preprocessing_solve_time``,
    # ``solve_writers``, ``handoff_writers``).  No write_input call.
    def _runner_factory() -> "FlexToolRunner":
        runner = FlexToolRunner(
            input_db_url=db_url,
            scenario_name=scenario_name,
            flextool_dir=flextool_dir_resolved,
            bin_dir=bin_dir_resolved,
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
    tc = TimelineConfig.load_from_db_url(db_url, scenario_name, logger=logger)
    tc.create_assumptive_parts(sc)
    tc.create_timeline_from_timestep_duration(sc)

    state = RunnerState(
        paths=PathConfig(work_folder=work_folder),
        solve=sc,
        logger=logger,
        timeline=tc,
        handoffs={},
    )
    # Stash the memory recorder so ``run_orchestration`` →
    # ``_FlexpyCascadeSolver`` can fire the remaining first-iter
    # checkpoints (load / build / solve) without having to plumb it
    # through additional keyword arguments.
    state._memory_recorder = _memrec  # type: ignore[attr-defined]

    return run_orchestration(
        state, work_folder, runner_factory=_runner_factory,
        db_url=db_url, scenario_name=scenario_name, warm=warm,
    )


def run_single_solve_from_db(
    input_db_url: str | Path,
    scenario_name: str,
    work_folder: Path | str,
    *,
    logger: logging.Logger | None = None,
    emit_output: bool = True,
) -> "OrchestrationStep":
    """Δ.25 — Surgical fast-path single-solve from a Spine DB.

    Bypasses :func:`flextool.flextoolrunner.input_writer.write_input`
    entirely.  Reads inputs directly from a
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
            f"flexpy.run_single_solve_from_db[{scenario_name}]"
        )

    import time as _time

    db_url = str(input_db_url)
    if not db_url.startswith("sqlite:") and not db_url.startswith("postgresql"):
        db_url = f"sqlite:///{db_url}"

    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)

    # 1. Construct the SpineDbReader once.
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    _t0 = _time.perf_counter()
    reader = SpineDbReader(db_url, scenario=scenario_name)
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
    # Single-solve has no rolling cascade, so the cache key equals the
    # scenario name.  Resolve user overrides defensively (helper handles
    # malformed / non-finite / non-positive DB values).
    scale_table = _scaling.analyze_solve(
        solve_name=scenario_name,
        flex_data=flex_data,
        work_folder=work_folder,
        logger=logger,
    )
    user_row_scaling = sc.use_row_scaling.get(scenario_name)
    user_obj_scale = sc.scale_the_objective.get(scenario_name)
    effective_row_scaling, effective_obj_scale = (
        _scaling.resolve_effective_scaling(
            scale_table, user_row_scaling, user_obj_scale,
        )
    )
    user_bound_scale_override = _scaling.resolve_user_bound_scale_override(
        sc.user_bound_scale.get(scenario_name)
    )

    _t0 = _time.perf_counter()
    problem = Problem()
    build_flextool(problem, flex_data, scale_the_objective=effective_obj_scale)
    print(f"Input: LP build: {_time.perf_counter() - _t0:.3f}s")

    # HiGHS solver options (always-on advanced row scaling + explicit
    # ``user_bound_scale`` override OR LP-coefficient-range based
    # recommendation from ``problem.peek_lp_ranges()`` — the actual
    # arrays HiGHS will see — falling back to the input-data heuristic
    # when LP introspection isn't available).
    lp_ranges = problem.peek_lp_ranges()
    highs_options = _scaling.recommended_highs_options(
        scale_table,
        user_bound_scale_override=user_bound_scale_override,
        lp_ranges=lp_ranges,
    )

    problem.set_solver_options(highs_options)

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
        )

    # Always emit the scaling CSV + report (even on non-optimal solves —
    # the diagnostic report is most useful when the solve is degenerate).
    _write_scale_csv_and_report(
        solve_data_dir=Path(work_folder) / "solve_data",
        output_raw_dir=Path(work_folder) / "output_raw",
        solve_name=scenario_name,
        scale_table=scale_table,
        effective_row_scaling=effective_row_scaling,
        effective_obj_scale=effective_obj_scale,
        user_row_scaling=user_row_scaling,
        flex_data=flex_data,
        solution=sol,
        logger=logger,
    )

    # 7. Build a stub SolveHandoff (no carriers in single-solve mode).
    from flextool.engine_polars._solve_handoff import SolveHandoff
    handoff = SolveHandoff()

    # Un-scale the objective value back to user-facing units.
    # ``build_flextool`` multiplied the objective coefficients by
    # ``effective_obj_scale``, so HiGHS reports a scaled value.
    unscaled_obj = (
        sol.obj / effective_obj_scale
        if sol.optimal and sol.obj is not None
        else None
    )

    return OrchestrationStep(
        solve_name=scenario_name,
        solution=sol,
        handoff=handoff,
        obj=unscaled_obj,
        warm_used=False,
        flex_data=flex_data,
    )


__all__ = [
    "OrchestrationStep",
    "run_orchestration",
    "run_chain_from_db",
    "run_single_solve_from_db",
]
