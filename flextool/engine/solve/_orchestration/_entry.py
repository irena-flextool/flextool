"""The entry points: :func:`run_orchestration` (validate ``model_solve``, set up the work folder, drive the
cascade) and :func:`run_chain_from_db` (a scenario from a Spine DB, end to end)."""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from flextool import run_record
from flextool.run_options import RunOptions
from flextool.engine.input._solve_state import (
    FlexToolConfigError,
    PathConfig,
    RunnerState,
)
from flextool.sqlite_url import sqlite_url_string

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    import polars as pl
    from flextool._solver_config_source import SolverConfigSource

from flextool.engine.solve._orchestration._cascade import _drive_cascade
from flextool.engine.solve._orchestration._highs_options import _reset_solver_args_echo, _run_options_or_env
from flextool.engine.solve._orchestration._memory import _MemoryRecorder, _try_malloc_trim, set_phase_recorder
from flextool.engine.solve._orchestration._steps import OrchestrationStep
from flextool.engine.solve._orchestration._guards import _dispatch_realization_raw_guard, _invest_branching_raw_guard


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
    # Fail-fast pass: refuse an unrecognised invest_solve_branching_method for the model's top-level solves
    # before any solve runs.  This only sees the top-level ``solves`` array (not solves reached via
    # ``contains_solves`` recursion), so it is a convenience — the authoritative per-solve guard lives at the top
    # of ``run()``.
    for _s in solves:
        _invest_branching_raw_guard(state.solve, _s)
        # W4 Slice 1 Guard (fail-fast pass): reject an unrecognised
        # dispatch_realization_method, or 'separate_dispatch' on a solve
        # shape it does not support, for the model's top-level solves before
        # any solve runs.  Like the branching pass, this only sees the
        # top-level ``solves`` array (not solves reached via
        # ``contains_solves`` recursion / rolling), so it is a convenience —
        # the authoritative per-solve guard lives at the top of ``run()``.
        _dispatch_realization_raw_guard(state.solve, _s)
    return solves


def _bootstrap_dirs(work_folder: Path, logger: logging.Logger) -> None:
    """Create ``solve_data/``, ``output_raw/``, ``output_plots/`` under *work_folder*, clearing
    THIS run's own output set first (item 163 §4, design_163.md).

    A reused work folder's leftover ``output_raw/`` (per-roll parquets, ``_solve_order.txt``,
    ``manifest.json``) and ``solve_data/`` handoff-mirror files must never leak into this run:
    ``union_realized_slice.has_persisted_slices`` would union stale per-roll parquets into the fresh
    run's output, and the mirror writers' own "preserve prior content" checks would mistake a
    leftover file from an EARLIER run for same-cascade state written by THIS run (design_163.md F6).

    Only the run's own output set is cleared:

    * ``output_raw/`` — ``shutil.rmtree`` + recreate.
    * The six ``solve_data/`` basenames in
      :data:`flextool.process_outputs.handoff_writers.MIRROR_FILES` — ``unlink(missing_ok=True)``.
      Most of those six are written only under ``csv_dump=True`` (design_163.md §11 Q1) and key
      their own "first write this cascade" logic off the fresh-per-run
      :class:`~flextool.engine.output._writer.OutputWriterState`, not off disk, so they are
      already self-resetting; ``period_capacity.csv`` (written every run, unconditionally) and
      ``p_roll_continue_state.csv`` (whose "preserve prior content" skip has no in-memory guard of
      its own) are the two that actually depend on this clearing for correctness. Clearing all six
      is simplest and matches what the module already names for this purpose.

    Nothing else is touched: not ``input/``, not any other ``solve_data/*`` file (dump / debug
    output the output cascade never reads back, item 163), not ``solver_config/``,
    ``basis_cache/`` or ``subprocess/``.  This is a no-op for the GUI and Spine Toolbox, which
    already give each job / execution a fresh work folder; it is the fix for the CLI-in-cwd rerun
    case, where a stale ``output_raw/`` from a failed or ``--csv-dump`` run used to corrupt the
    next run's outputs (design_163.md §4).

    A symlinked ``output_raw/`` is refused rather than followed, so the clearing can never reach
    outside the work folder.

    Mirrors lines 63-76 of the flextool reference, plus the item-163 clearing.
    """
    from flextool.process_outputs.handoff_writers import MIRROR_FILES

    output_raw = work_folder / "output_raw"
    if output_raw.is_symlink():
        raise FlexToolConfigError(
            "output_raw/ in the work folder is a symlink; FlexTool clears "
            "it at the start of each run"
        )
    removed = 0
    if output_raw.is_dir():
        removed += sum(1 for p in output_raw.rglob("*") if p.is_file())
        shutil.rmtree(output_raw)
    elif output_raw.exists():
        # Not a directory, not a symlink (checked above) — an odd leftover
        # (e.g. a plain file at that path). Clear it the same way.
        removed += 1
        output_raw.unlink()
    output_raw.mkdir(parents=True, exist_ok=True)

    for sub in ("solve_data", "output_plots"):
        try:
            (work_folder / sub).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            logger.debug(f"{sub} folder existed")

    solve_data = work_folder / "solve_data"
    for name in MIRROR_FILES:
        target = solve_data / name
        if target.exists():
            removed += 1
        target.unlink(missing_ok=True)

    if removed:
        logger.info(
            f"Removed {removed} files an earlier run left in "
            f"{work_folder}/output_raw/ and solve_data/ (this run writes "
            f"them anew)"
        )


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
    warm : bool, default True
        When True, solve in-process through :class:`polar_high.WarmProblem`
        and reuse the built LP for the next sub-solve only when that
        sub-solve's FlexData and solver configuration are identical
        (labels included; see :mod:`flextool.engine.solve._warm`).  Every
        other sub-solve, including every roll of a rolling solve, builds a
        new LP.  Decisions are recorded per-step on
        :attr:`OrchestrationStep.warm_used`.

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
    from flextool.engine.preprocess.params._coarse import reset_warnings as _reset_coarse_warnings
    _reset_coarse_warnings()
    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)
    # Rebuild ``paths`` for this run's work_folder, but CARRY FORWARD the
    # solver option files from the incoming state — every solve reads its
    # ``highs.opt`` / ``<solver>.opt`` layer from them.  ``run_chain_from_db``
    # sets them (read once at run start); a bare ``PathConfig(work_folder=...)``
    # here would silently drop the option-file floor.
    _prior_paths = state.paths
    state.paths = PathConfig(
        work_folder=work_folder,
        solver_config_dir=_prior_paths.solver_config_dir if _prior_paths is not None else None,
        solver_option_files=_prior_paths.solver_option_files if _prior_paths is not None else None,
    )

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


AXIS_ENUMS_ENV = "FLEXTOOL_AXIS_ENUMS"


AXIS_ENUMS_WARNING = (
    "axis-Enum mode (FLEXTOOL_AXIS_ENUMS=1) is EXPERIMENTAL: development/test use only; production runs use "
    "Utf8 axes. Untested in this mode: writing results (process_outputs); region decomposition and the CLI "
    "region export, the staged input loader and csv_reload build their providers without a vocabulary and "
    "stay Utf8."
)


def _activate_axis_enums_from_env(
    provider: "object", db_url: str, scenario_name: str, logger: logging.Logger,
) -> None:
    """Turn on axis-Enum mode for the whole run when ``FLEXTOOL_AXIS_ENUMS=1``.

    Sets the scenario's axis Enum vocabulary and contract on the cascade-input *provider*; the cascade reader
    and every sub-solve provider take them from there.  Any other value, or the variable unset, leaves the
    provider without a vocabulary (all axes Utf8, the production mode).  Logs :data:`AXIS_ENUMS_WARNING` once.
    """
    if os.environ.get(AXIS_ENUMS_ENV) != "1":
        return
    from flextool.spinedb_backend._axis_enums import build_scenario_axis_enums

    provider.axis_enums, provider.contract = build_scenario_axis_enums(db_url, scenario_name)
    logger.warning(AXIS_ENUMS_WARNING)


# ---------------------------------------------------------------------------
# run_chain_from_db — top-level entry point
# ---------------------------------------------------------------------------


def run_chain_from_db(
    input_db_url: str | Path,
    scenario_name: str | None = None,
    work_folder: Path | str | None = None,
    *,
    flextool_dir: Path | str | None = None,
    solver_config_dir: Path | str | Sequence[Path | str] | None = None,
    logger: logging.Logger | None = None,
    warm: bool = True,
    keep_solutions: bool = False,
    csv_dump: bool = False,
    override_provider: "Callable[[], dict[str, pl.DataFrame]] | None" = None,
    run_options: "RunOptions | None" = None,
    solver_config_source: "SolverConfigSource | None" = None,
) -> dict[str, OrchestrationStep]:
    """Run a flextool multi-solve scenario end-to-end natively.

    Combines:

    1. :func:`flextool.engine.preprocess._input_writer.write_workdir_inputs`
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
    scenario_name : str
        The scenario to run.  Required: every sub-solve loads its
        parameters through a SpineDbReader filtered to this scenario.
        ``None`` or ``""`` raises :class:`FlexToolConfigError`.
    work_folder : Path | str, optional
        Where to materialise the CSVs.  ``None`` uses an auto-cleaned
        tempdir.
    flextool_dir : Path, optional
        Override the installed flextool package directory.
    solver_config_dir : Path or list of Path, optional
        The folder (or folders, highest first, read file by file) of the
        solver option files (``highs.opt``, ``gurobi.opt``, ...); wins over
        ``--solver-config-dir`` / ``FLEXTOOL_SOLVER_CONFIG_DIR`` and the
        database-layout search
        (:func:`flextool._solver_config_source.resolve_solver_config_source`).
    logger : logging.Logger, optional
        Logger to use.  ``None`` constructs one named after the scenario.
    warm : bool, default True
        When True, solve in-process through :class:`polar_high.WarmProblem`
        and reuse the built LP only for an identical next sub-solve (same
        FlexData, labels included, and same solver configuration).  See
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
    run_options : RunOptions, optional
        The run's operator options (``cmd_run_flextool`` passes the parsed
        flags over the environment).  ``None`` resolves them from the
        environment variables alone (Spine Toolbox tools, a Python
        console, tests).  Stored on ``RunnerState.run_options``.
    solver_config_source : SolverConfigSource, optional
        The option-file folder already resolved by the caller (the CLI, which
        shows it in its run header); ``None`` resolves it here.

    Returns
    -------
    dict[str, OrchestrationStep]
        Mapping ``complete_solve_name → OrchestrationStep``.  Iterate
        in insertion order to walk the chain.

    Raises
    ------
    FlexToolConfigError
        When ``scenario_name`` is missing, or the input database cannot be
        opened for that scenario.
    """
    if not scenario_name:
        raise FlexToolConfigError(
            f"run_chain_from_db needs scenario_name (got {scenario_name!r}) for the input database "
            f"{input_db_url}: every sub-solve loads its parameters through a SpineDbReader filtered "
            "to that scenario. Pass the scenario to run (CLI: --scenario-name)."
        )
    from flextool.engine.input._db_loader import FlexToolRunner

    # The timeset block warnings of this run, shared by its two TimelineConfig construction sites so each is
    # printed once per run (a new run starts with a new set).
    _timeset_warned: set[tuple] = set()

    if logger is None:
        logger = logging.getLogger(
            f"flextool.engine.solve.run_chain_from_db[{scenario_name}]"
        )
    # The run's operator options: the CLI passes its flags over the environment; a library caller (Spine Toolbox
    # tool, Python console, test) passes none and gets the environment's.
    run_options = _run_options_or_env(run_options)

    # v52 multi-solver dispatch (Phase 2 startup hint).  Probe each
    # solver in polar-high's catalog with a trivial 1-var LP so users
    # see at a glance which are licensed on this machine (vs wrapper-
    # installed-but-no-license vs not-installed-at-all) before the
    # solve loop selects one.  See ``specs/flextool-multi-solver-handoff.md``
    # Step 4b.  Cached at module level so repeat cascade runs don't
    # re-probe.
    try:
        from flextool.engine.solve._solver_dispatch import (
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
            logger.info("Solvers: %s\n", formatted)
    except ImportError:  # pragma: no cover — older polar_high without dispatch
        pass

    db_url = str(input_db_url)
    if "://" not in db_url:
        db_url = sqlite_url_string(db_url)

    # Behaviour-change notices fire only for an upgraded database's markers; each entity once per run.
    from flextool.engine import _behaviour_changes
    _behaviour_changes.start_run(_behaviour_changes.read_markers_from_url(db_url))
    # W-0490: a migration history that an old spinedb_api bug may have hit (TODO(v5.0.0): remove).
    try:
        from flextool.update_flextool.value_list_history import warn_at_run_start
        warn_at_run_start(db_url)
    except Exception:  # noqa: BLE001 - a check of the history never stops a run
        logging.getLogger(__name__).debug("value list history check failed", exc_info=True)
    # W-0491: list defaults an upgrade before 4.6.3 left empty, until the repair tool's record (own try: a W-0490
    # error never hides it; TODO(v5.0.0): remove).
    try:
        from flextool.update_flextool.list_default_restore import note_at_run_start
        note_at_run_start(db_url)
    except Exception:  # noqa: BLE001 - a check of the defaults never stops a run
        logging.getLogger(__name__).debug("list defaults check failed", exc_info=True)
    _reset_solver_args_echo()

    if work_folder is None:
        work_folder = Path(tempfile.mkdtemp(prefix="flextool_run_chain_"))
    else:
        work_folder = Path(work_folder)
        work_folder.mkdir(parents=True, exist_ok=True)

    # ``flextool_dir`` defaults to the installed flextool package directory
    # (resolved via importlib.resources so it works both editable and wheel).
    from flextool._resources import package_data_path
    flextool_dir_resolved = (
        Path(flextool_dir) if flextool_dir is not None
        else package_data_path("")
    )
    # The solver option files: one ladder of folders per run, read file by file
    # (explicit argument > --solver-config-dir > FLEXTOOL_SOLVER_CONFIG_DIR >
    # the database layout > the Toolbox project folder > the current folder >
    # FlexTool's defaults), read ONCE here and kept on ``state.paths.solver_option_files``, so every
    # roll and solve uses the files as they were at run start.  The copy under
    # ``solve_data/solver_config/`` is the record of what the run used; it is
    # written afresh every run and never read back.
    from flextool._solver_config_source import (
        resolve_solver_config_source,
        snapshot_solver_option_files,
    )
    # A caller that passes the resolved folder (the CLI) has shown it in its header already.
    resolved_here = solver_config_source is None or solver_config_dir is not None
    if resolved_here:
        solver_config_source = resolve_solver_config_source(
            explicit=solver_config_dir, run_options=run_options, input_db_url=db_url,
        )
    from flextool.engine.solve._solver_dispatch import reset_run_warnings
    reset_run_warnings()
    solver_option_files = snapshot_solver_option_files(solver_config_source, work_folder, logger,
                                                       announce=resolved_here)
    # Item 234: the run record names the folder and a hash of each option file, so two runs can be compared
    # without their logs.
    run_record.active().note_solver_config(solver_option_files.record())
    solver_config_dir_resolved = (
        solver_option_files.snapshot_dir or solver_config_source.dir
    )

    # Cascade-input Provider population from the Spine DB.  Pure
    # in-memory: ``write_workdir_inputs`` runs the input_derivation
    # pipeline whose emitters populate the Provider directly, so no
    # CSVs hit disk.
    from flextool.engine.preprocess._input_writer import (
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
    # Publish so deeper modules (flex_data/_db_overrides.py's _apply_db_overrides) emit
    # in the unified [mem] format.
    set_phase_recorder(_memrec)
    _memrec.checkpoint("cascade_start", logger,
                       user_label="Run start")

    # Construct the cascade-input Provider and let
    # ``write_workdir_inputs`` populate it from the Spine DB.  The
    # Provider is then attached to the cascade ``RunnerState`` below
    # so the per-sub-solve hook in
    # :mod:`flextool.engine.solve._run_model` picks it up at
    # provider seed time.
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    cascade_input_provider = FlexDataProvider()
    _activate_axis_enums_from_env(cascade_input_provider, db_url, scenario_name, logger)
    # The run's axis vocabulary (None: Utf8, production) is the live one from here to the end of the run, also
    # for the steps before the first load_flextool; restored on exit, so an earlier axis-Enum run in this
    # process (load_flextool leaves its vocabulary live for the model build) does not type this run's frames.
    from flextool.engine.input._axis_enums import reset_global_axis_enums, set_global_axis_enums
    _axis_enums_token = set_global_axis_enums(cascade_input_provider.axis_enums)
    try:
        # ``read_input`` timing row: the DB read, the input preparation and
        # the run set-up (solve and timeline configuration, runner) up to the
        # runner's construction, before ``preprocessing_global``.  The runner
        # built in ``_runner_factory`` below makes the recorder that starts
        # timings.csv afresh, so the row is recorded there, once the runner
        # exists, and only by the first call.
        _t_read_input_start = time.perf_counter()
        _read_input_recorded = False
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
            nonlocal _read_input_recorded
            runner = FlexToolRunner(
                input_db_url=db_url,
                scenario_name=scenario_name,
                flextool_dir=flextool_dir_resolved,
                solver_config_dir=solver_config_dir_resolved,
                work_folder=work_folder,
                run_options=run_options,
                timeset_warned=_timeset_warned,
            )
            runner.state.logger.setLevel(logging.ERROR)
            _recorder = runner.state.timing_recorder
            if _recorder is not None and not _read_input_recorded:
                _read_input_recorded = True
                _recorder.record(
                    "read_input",
                    seconds=time.perf_counter() - _t_read_input_start,
                    t_start=_t_read_input_start,
                )
            return runner

        # Build a minimal native RunnerState so callers can introspect
        # state.handoffs after the run.  The real per-solve mutation happens
        # on the underlying flextool runner's state (driven inside
        # _drive_cascade); we mirror the handoffs dict back.
        from flextool.engine.input._solve_config import SolveConfig
        from flextool.engine.input._timeline import TimelineConfig

        sc = SolveConfig.load_from_db_url(db_url, scenario_name, logger=logger)
        # Item 236 A3: a solve's solver_precommand is never run; say so once per solve in this run.
        from flextool.engine.solve._solver_dispatch import (
            check_run_solver_options,
            check_run_solver_wrappers,
            warn_solver_precommands,
        )
        warn_solver_precommands(sc.solver_settings.precommand, logger)
        # Item 240 B1: the solver_arguments of every solve the run runs, under its solver, before the first solve
        # (a solve the model does not run is not checked: its options never reach a solver; review b1 F7).
        _run_solves = sc.solves_in_run()
        check_run_solver_options(sc.solver_settings, logger, solves=_run_solves, shown_as=sc.database_solve_names())
        # Item 240 B2: this computer's wrapper of every commercial solver the run uses.
        check_run_solver_wrappers(sc.solver_settings, solves=_run_solves)
        _memrec.checkpoint("solve_config_loaded", logger,
                           user_label="SolveConfig loaded (from DB)")
        tc = TimelineConfig.load_from_db_url(db_url, scenario_name, logger=logger)
        tc.create_assumptive_parts(sc)
        tc.validate_timeset_blocks(sc, _timeset_warned)
        tc.create_timeline_from_timestep_duration(sc)
        _memrec.checkpoint("timeline_constructed", logger,
                           user_label="TimelineConfig constructed (from DB)")

        state = RunnerState(
            # The solver option files MUST be carried here: this is the state the
            # per-solve loop reads its ``highs.opt`` / ``<solver>.opt`` layer
            # from.  Omitting them silently drops the option-file floor.
            paths=PathConfig(
                work_folder=work_folder,
                solver_config_dir=solver_config_dir_resolved,
                solver_option_files=solver_option_files,
            ),
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
        # :mod:`flextool.engine.solve._run_model` consults).
        state.cascade_input_provider = cascade_input_provider  # type: ignore[attr-defined]
        # Phase 5c — attach the optional external override provider onto
        # the cascade ``RunnerState``.  ``_drive_cascade`` forwards it onto
        # the underlying ``runner.state``; the per-sub-solve hook in
        # :mod:`flextool.engine.solve._run_model` (Phase 5b) invokes
        # it after the parent-handoff translator at every iteration.
        state.override_provider = override_provider
        state.run_options = run_options

        return run_orchestration(
            state, work_folder, runner_factory=_runner_factory,
            db_url=db_url, scenario_name=scenario_name, warm=warm,
            keep_solutions=keep_solutions, csv_dump=csv_dump,
        )
    finally:
        reset_global_axis_enums(_axis_enums_token)
