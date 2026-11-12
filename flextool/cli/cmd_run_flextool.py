import os
# glibc malloc arena cap — set BEFORE any C-extension import that
# allocates via malloc.  glibc defaults to up to 8 × ncores arenas
# (≈256 on a 32-core workstation); each arena holds its own
# freed-but-not-returned-to-OS pages, so worst-case fragmentation
# scales with core count.  Capping to 4 is a precautionary middle
# ground: ~64× reduction vs the default, while still allowing up to
# four concurrent allocators (HiGHS parallel presolve, Benders
# subproblem runs) without serialising every malloc through one heap.
# No measured benefit on the FlexTool cascade workload as of this
# writing — FlexTool's hot path is essentially single-threaded
# (polars at most 8 threads, HiGHS serial simplex by default).  Kept
# only as a cheap precaution against future workloads where arena
# growth could matter.  ``setdefault`` so the shell wins.
os.environ.setdefault("MALLOC_ARENA_MAX", "4")

import argparse
import sys
import logging
import math
import shutil
import signal
import threading
import traceback
from pathlib import Path
from datetime import datetime
import time

# Leave breadcrumbs if a compiled extension (polars / HiGHS / numpy) crashes
# natively. faulthandler can't stop a segfault, but it dumps the Python
# traceback of every thread to stderr at fault time — so a crash during, say,
# the first polars op prints the offending frame instead of nothing. The GUI
# captures this child's stderr into the job log. See flextool.env_check for
# the out-of-process probe + auto-remediation that prevents the crash.
import faulthandler
try:
    faulthandler.enable()
except (AttributeError, ValueError, OSError):
    # stderr may be unavailable (e.g. detached / redirected to a closed fd).
    pass
from flextool._safe_text import make_printable
from flextool._mem_sampler import start_mem_sampler
from flextool.process_outputs.write_outputs import resolve_output_settings, write_outputs
from flextool import run_record
from flextool.cli._console import run_tool
from flextool.cli._timing import TimingRecorder
from flextool.run_options import RunOptions
from flextool._solver_config_source import input_db_filesystem_path, resolve_solver_config_source
from flextool._thread_budget import header_text, peek_mismatch
from flextool.update_flextool.ensure_settings_db import ensure_settings_db
from spinedb_api.filters.tools import name_from_dict
from spinedb_api import to_database, DateTime
from flextool.update_flextool.db_handle import open_database_mapping
from spinedb_api.exception import NothingToCommit

# Start the memory sampler as the first statement after imports.  The
# few hundred ms of import-cascade RSS that precede this point are not
# captured; the workload-level RSS curve (what the sampler exists to
# measure) is fully captured.  Gated by FLEXTOOL_MEM_SAMPLER=1; no-op
# when the env var is unset.
start_mem_sampler()

class FlushingStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = FlushingStream(sys.stdout)

#return_codes
#0 : Success
#-1: Failure (Defined in the Toolbox)
#1: Infeasible or unbounded problem (not implemented in the toolbox, functionally same as -1. For a possiblity of a graphical depiction)


class RunTerminated(SystemExit):
    """SIGTERM or SIGHUP ended the run: a ``SystemExit`` (code 128 + signal) so ``finally`` blocks still run."""


def _install_termination_handlers():
    """Turn SIGTERM / SIGHUP into :class:`RunTerminated` (POSIX, main thread); return the function that undoes it."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        return lambda: None

    def _terminate(signum, frame):
        raise RunTerminated(128 + signum)

    previous = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            previous[sig] = signal.signal(sig, _terminate)
        except (ValueError, OSError):
            continue

    def restore():
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler if handler is not None else signal.SIG_DFL)
            except (ValueError, OSError):
                pass

    return restore


def _start_run_record(*, args, run_options, scenario_name, output_location, output_subdir, write_methods,
                      work_folder):
    """Start this run's ``_run_status.json`` and its log capture; return the recorder and its cleanup function.

    With ``FLEXTOOL_RUN_LOG_OWNER=1`` (read and removed, so no process this run starts sees it) the parent holds the
    output pipe and writes ``_run.log``: the engine prints the marker that tells it where.  Otherwise the engine
    copies its own Python console output into ``_run.log`` (``log_capture: "python"``).  A folder that cannot be
    written gives one warning and a run without a record; the exit code never depends on the record.
    """
    owner = os.environ.pop(run_record.RUN_LOG_OWNER_ENV, None)
    log_capture = "parent" if owner == "1" else "python"
    parquet_dir = Path(output_location) / "output_parquet" / output_subdir
    try:
        rec = run_record.RunRecorder.start(
            parquet_dir=parquet_dir,
            input={"db_url": args.input_db_url, "scenario": scenario_name, "source_file": args.input_source_file},
            output={
                "location": output_location,
                "subdir": output_subdir,
                "work_folder": str(Path(work_folder).resolve()),
                "write_methods": list(write_methods),
            },
            options=run_options.summary_lines(),
            argv=list(sys.argv),
            log_capture=log_capture,
            launcher=args.run_launcher,
            run_options=run_options,
        )
    except OSError as exc:
        logging.warning("Cannot write the run record in %s (%s); the run continues without it.", parquet_dir, exc)
        return run_record.NULL_RECORDER, lambda: None
    run_record.set_active(rec)
    handler = rec.warning_handler()
    root = logging.getLogger()
    root.addHandler(handler)
    writer = None
    restore_tee = None
    if log_capture == "parent":
        print(run_record.format_marker(parquet_dir, rec.run_id), flush=True)
    else:
        try:
            writer = run_record.RunLogWriter(parquet_dir / run_record.RUN_LOG_FILE)
            restore_tee = run_record.install_tee(writer)
        except OSError as exc:
            logging.warning("Cannot write the run log in %s (%s); the run continues without it.", parquet_dir, exc)
            if writer is not None:
                writer.close()
                writer = None

    def cleanup():
        if restore_tee is not None:
            restore_tee()
        if writer is not None:
            writer.close()
        root.removeHandler(handler)
        rec.close()
        run_record.set_active(None)

    return rec, cleanup


def _scan_failure_error(last_step, scenario_name):
    """The run record's ``error`` for a cascade the final optimality scan rejected."""
    if last_step is None:
        return {"type": "NoSolveSteps", "message": f"the run of scenario {scenario_name!r} produced no solve steps"}
    name = getattr(last_step, "solve_name", None) or scenario_name
    if getattr(last_step, "is_benders", False):
        return {"type": "BendersNoIncumbent",
                "message": f"decomposed solve {name!r} found no feasible solution"}
    return {"type": "SolveNotAccepted", "message": f"solve {name!r} did not solve to optimality"}


def _run_solve(args, scenario_name, work_folder, timing_recorder, run_options=None, solver_config_source=None):
    """Δ.21 — drive the native polar_high cascade.

    Returns a tuple ``(return_code, last_step)`` where ``last_step``
    is the :class:`flextool.engine.OrchestrationStep` of the
    final (or only) sub-solve, used by the caller to thread
    ``flex_data`` + ``solution`` into ``write_outputs`` for the
    in-memory parameter / set namespace path (Δ.31).
    """
    if scenario_name:
        timing_recorder.set_scenario(scenario_name)

    # ``--csv-dump`` is a one-way debug snapshot from the live
    # FlexDataProvider: when on, the cascade dumps both
    # ``flex_data.dump_csvs(work_folder)`` (per-solve) and the Provider's
    # captured derived frames (post-cascade snapshot below).  When off
    # the cascade runs purely in-memory — no CSVs hit
    # ``solve_data/`` from the writer-port modules.
    csv_dump_on = bool(getattr(args, 'csv_dump', False))

    from flextool.engine import run_chain_from_db

    # Drive the native cascade end-to-end.  ``run_chain_from_db``
    # handles flextool's preprocessing (write_input) AND the per-solve
    # LP build+solve+handoff loop in-process.
    t_solve_start = time.perf_counter()
    steps = run_chain_from_db(
        args.input_db_url,
        scenario_name,
        work_folder=work_folder,
        csv_dump=csv_dump_on,
        run_options=run_options,
        solver_config_source=solver_config_source,
    )

    # ``--csv-dump``: snapshot the last sub-solve's Provider to disk.
    # The Provider holds every derived frame the cascade's writers
    # produced; ``snapshot_processed_inputs`` writes them under
    # ``work_folder`` mirroring the cascade's parent-qualified key
    # layout.  This is a debug oracle; the cascade itself reads only
    # from the in-memory Provider.
    if csv_dump_on and steps:
        last_step = next(reversed(list(steps.values())))
        provider = getattr(last_step, "flex_data_provider", None)
        if provider is not None:
            try:
                provider.snapshot_processed_inputs(work_folder)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "--csv-dump: snapshot_processed_inputs failed: %s", exc,
                )

    all_solves_seconds = time.perf_counter() - t_solve_start
    print("--- All Flextool solves time %.4s seconds ---" % all_solves_seconds)
    timing_recorder.record('all_solves', seconds=all_solves_seconds,
                           t_start=t_solve_start)

    if not steps:
        logging.error("Native cascade produced no solve steps; aborting.")
        return 1, None
    return _scan_cascade_optimality(steps)


def _scan_cascade_optimality(steps):
    """Classify the cascade outcome into a ``(exit_code, last_step)`` pair.

    Scans sub-solves for a non-optimal outcome.  Two distinct cases:

    * A solve accepted as *near-optimal* by
      :func:`classify_acceptance` — a crossover-off interior-point solve
      HiGHS would not certify ``kOptimal`` but whose primal is feasible with
      a small primal--dual gap — already wrote a usable solution and returned
      success from the per-solve driver.  Its strict ``optimal`` mirror is
      ``False`` (HiGHS status Unknown), so it is recognised here via
      ``step.near_optimal`` and must NOT be treated as a failure.

    * A Benders-decomposed solve that found a FEASIBLE incumbent but did not
      close the optimality gap to tolerance within its iteration cap is NOT
      infeasible — every subproblem/master LP solved to optimality and the
      written outputs are a valid feasible plan, just not certified optimal.
      Surface a loud warning (with the actual gap vs the required tolerance)
      and let the run SUCCEED (exit 0) so the parquet results are consumed
      downstream.

    * Anything else non-optimal (a Benders solve with no feasible incumbent
      at all, or a step whose solution the cascade never accepted) is a
      genuine failure → exit 1.  A monolithic LP that HiGHS genuinely could
      not solve is rejected earlier at the solve site (``classify_acceptance``
      → ``FlexToolSolveError``, which names the precise cause) and never
      reaches this scan; if one does, it is reported here with the facts the
      slim step carries rather than a guessed "infeasible/unbounded" cause.
    """
    last_step = None
    for name, step in steps.items():
        last_step = step
        # Phase C.5 — intermediate steps no longer hold ``solution``
        # under default (slim) cascade; read the slim ``optimal``
        # summary instead so the non-optimal check works without
        # ``keep_solutions=True``.  ``step.solution`` is only populated
        # for the LAST step (or every step under ``keep_solutions``).
        #
        # ``optimal`` is the STRICT HiGHS verdict; ``near_optimal`` flags a
        # solve ``classify_acceptance`` accepted despite a non-``kOptimal``
        # status (crossover-off interior point).  Both mean "usable
        # solution written" — treat either as success.
        if step.optimal or getattr(step, "near_optimal", False):
            continue
        if step.is_benders and step.obj is not None and math.isfinite(step.obj):
            # Non-convergence with a feasible incumbent — warn loudly, keep
            # the written results, do NOT fail the run.
            logging.error(_benders_nonconvergence_banner(name, step))
            continue
        logging.error(
            "Native cascade: solve %r did not solve to optimality; exit=1. "
            "This solve's solution was not accepted for consumption "
            "(strict-optimal=%r, near-optimal-accepted=%r, benders=%r, "
            "objective=%r). A genuine LP failure names its precise cause "
            "(infeasible / unbounded / limit reached) at the solve site; a "
            "Benders solve reaching here found no feasible incumbent. See the "
            "solve log above for the solver's reported status.",
            name,
            step.optimal,
            getattr(step, "near_optimal", False),
            step.is_benders,
            step.obj,
        )
        return 1, step
    return 0, last_step


def _benders_nonconvergence_banner(name, step) -> str:
    """A loud, plain-English banner for a Benders solve that found a feasible
    incumbent but never met its convergence tolerance.

    Kept in FlexTool class vocabulary (node group / connection / flow) — never
    model-instance terms — mirroring the ``_benders_failure_message`` contract.
    """
    def _pct(x):
        return f"{x * 100:.4g}%" if x is not None and math.isfinite(x) else "unknown"

    gap = getattr(step, "benders_gap", None)
    tol = getattr(step, "benders_tol", None)
    iters = getattr(step, "benders_iterations", None)
    bar = "#" * 76
    return (
        f"\n{bar}\n"
        f"WARNING: decomposed solve {name!r} did NOT meet its convergence "
        f"tolerance.\n"
        f"{bar}\n"
        f"  Relative gap reached : {_pct(gap)}\n"
        f"  Required tolerance   : {_pct(tol)}\n"
        f"  Iterations run       : {iters if iters is not None else 'unknown'}\n"
        f"  Best feasible cost   : {step.obj:.6g}\n"
        f"\n"
        f"  The results ARE feasible and HAVE been written to the outputs "
        f"(parquet /\n"
        f"  results DB), but they are NOT certified optimal: the true optimum "
        f"may be\n"
        f"  up to the gap above cheaper than the reported cost.\n"
        f"\n"
        f"  To close the gap, try any of: raise the decomposition iteration "
        f"limit,\n"
        f"  loosen the convergence tolerance, set the in-out stabilization "
        f"weight\n"
        f"  (~0.3-0.7) to break a stalled plateau, or give any under-supplied "
        f"node\n"
        f"  group a finite fail-safe import price on its boundary nodes.\n"
        f"{bar}"
    )


def resolve_output_path(input_db_url, flextool_location, output_location, cwd,
                        project_folder_file=None):
    """Resolve the TRUE output root for a CLI run (5-tier rule).

    This is the path that outputs actually land under, so it is what
    gets persisted to the "Output info" DB as ``scenario/output_location``
    (Toolbox's comparison / re-create steps read it back to locate each
    scenario's parquet) AND what is passed as ``fallback_output_location``
    to ``write_outputs`` and used for the timings.csv directory.

    The five tiers, in precedence order:

    1. **Explicit ``--output-location``** wins outright.  ``write_outputs``
       already honours an explicit ``output_location`` over the
       ``fallback_output_location`` (see ``_resolve_settings``), so before
       this change an explicit ``--output-location`` steered where files
       were written but was NOT reflected in the persisted Output-info
       record (which used the fallback ``output_path``).  Folding it into
       tier 1 fixes that latent inconsistency: the recorded location now
       matches where the files actually go.

    2. **Project-folder file (``--project-folder-file``).**  This is the
       USER-LOCAL output redirect: the maintainer points Spine Toolbox's
       FlexTool run Tool at a gitignored file (seeded by ``self_update``
       as ``templates/project_folder.txt``) whose CONTENTS name a project
       folder, so a user can redirect every output (``output_parquet/``,
       ``results.sqlite``, plots, the per-project ``plot_settings.yaml``)
       into a per-project directory with ZERO git-committed change.

       The file's first non-empty, non-``#``-comment line is the project
       folder.  If that line is an ABSOLUTE path it is used verbatim; if
       RELATIVE it is resolved against the file's repo anchor —
       ``file.resolve().parent.parent`` — the SAME anchor the legacy
       ``--flextool-location`` walk uses, so a ``templates/
       project_folder.txt`` line of ``projects/Rivendell`` lands the
       output at ``<repo>/projects/Rivendell``.

       **A supplied ``--project-folder-file`` is a COMPLETE replacement
       for the legacy ``--flextool-location`` and therefore NEVER falls
       through to CWD.**  When the file is missing, unreadable, empty, or
       comment-only — i.e. its CONTENTS name no folder — this tier still
       fires, falling back to the FILE'S repo anchor
       (``Path(project_folder_file).resolve().parent.parent``), the same
       FlexTool root the legacy ``--flextool-location`` walk produced.
       This matches the seeded ``templates/project_folder.txt`` whose own
       comment says "Leave blank to use the FlexTool root".  Only when
       ``--project-folder-file`` was NOT supplied at all (None / empty
       arg) does resolution continue to tiers 3-5.

    3. **GUI-project layout.**  When the input DB file sits directly inside
       a directory named ``input_sources`` (the FlexTool GUI project
       layout, ``<project>/input_sources/<db>.sqlite``), the output root is
       that directory's parent — the project folder.  This is
       location-agnostic: it works wherever the project lives on disk.
       The DB filesystem path is recovered from ``input_db_url`` (which may
       be a ``sqlite:///`` URL possibly carrying an appended filter
       query-config) using the same ``sqlite:///`` stripping idiom used
       elsewhere in this file, plus a query-part strip.  If the path can't
       be resolved to an existing file, this tier is skipped (no crash).

    4. **Legacy ``--flextool-location`` bridge.**  ``.parent.parent`` of the
       resolved flextool-location path (the historical Spine Toolbox
       anchor; kept for backward compatibility one release).

    5. **Fallback** to the current working directory.

    Pure path logic — deterministic, no randomness, no side effects.
    """
    # Tier 1 — explicit override wins.
    if output_location:
        return Path(output_location)

    # Tier 2 — project-folder file.  A supplied --project-folder-file is a
    # COMPLETE replacement for --flextool-location: it ALWAYS yields an
    # output root and never falls through to CWD.  Its CONTENTS name the
    # project folder when present; otherwise (blank / comment-only /
    # missing / unreadable) we fall back to the FILE'S repo anchor
    # (.parent.parent), the same FlexTool root the legacy
    # --flextool-location walk produced.  Only an unsupplied (None / empty)
    # arg lets resolution continue to tiers 3-5.
    if project_folder_file:
        project_folder = _read_project_folder_file(project_folder_file)
        if project_folder is not None:
            return project_folder
        # CONTENTS name no folder — anchor at the file's repo root.
        try:
            return Path(project_folder_file).resolve().parent.parent
        except OSError:
            # ``resolve()`` should not raise for a plain path on POSIX even
            # when it doesn't exist, but degrade without crashing if it
            # ever does: anchor at the un-resolved path's .parent.parent.
            return Path(project_folder_file).parent.parent

    # Tier 3 — GUI project layout: <project>/input_sources/<db>.sqlite.
    db_fs_path = _input_db_filesystem_path(input_db_url)
    if db_fs_path is not None:
        try:
            resolved = db_fs_path.resolve()
        except OSError:
            resolved = None
        if resolved is not None and resolved.is_file() \
                and resolved.parent.name == "input_sources":
            return resolved.parent.parent

    # Tier 4 — legacy flextool-location anchor walk.
    if flextool_location:
        return Path(flextool_location).resolve().parent.parent

    # Tier 5 — fallback to CWD.
    return Path(cwd)


def _read_project_folder_file(project_folder_file):
    """Resolve a project-folder redirect from a ``--project-folder-file``.

    The file's CONTENTS name the project folder: the first non-empty,
    non-``#``-comment line is taken (whitespace-stripped).  An ABSOLUTE
    line is returned verbatim; a RELATIVE line is resolved against the
    file's repo anchor (``file.resolve().parent.parent``, the same anchor
    the legacy ``--flextool-location`` walk uses), so a ``templates/
    project_folder.txt`` line of ``projects/Rivendell`` maps to
    ``<repo>/projects/Rivendell``.

    Returns a :class:`~pathlib.Path` when the file's CONTENTS name a
    usable project folder, or ``None`` when the path arg is empty, the
    file is missing / unreadable, or it has no non-comment content.  A
    ``None`` return does NOT mean "fall through to CWD": the caller
    (``resolve_output_path``) treats a supplied-but-content-less
    ``--project-folder-file`` as the FlexTool root by anchoring at the
    file's ``.parent.parent`` — so this tier never reaches CWD once the
    arg is supplied.  Robust: any read error → ``None`` (never raises).
    """
    if not project_folder_file:
        return None
    file_path = Path(project_folder_file)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError:
        # Missing / unreadable / not a file — skip this tier silently.
        return None
    line = None
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        line = stripped
        break
    if line is None:
        # Empty or comment-only — fall through.
        return None
    candidate = Path(line)
    if candidate.is_absolute():
        return candidate
    # Relative → resolve against the file's repo anchor (parent.parent),
    # matching the legacy --flextool-location walk so a templates/-anchored
    # relative line roots at the repo root.
    try:
        anchor = file_path.resolve().parent.parent
    except OSError:
        return None
    return anchor / candidate


def _input_db_filesystem_path(input_db_url):
    """Best-effort filesystem path for a (possibly sqlite) input DB URL.

    Returns a :class:`~pathlib.Path` for ``sqlite:///`` URLs and bare
    filesystem paths, stripping any appended filter query-config
    (``?spinedbfilter=...``); returns ``None`` for non-sqlite URLs (e.g.
    ``mysql://``) or when the value is empty.  Does not touch the
    filesystem — purely string → path.
    """
    return input_db_filesystem_path(input_db_url)


def _run_options_from_args(args):
    """The run's :class:`~flextool.run_options.RunOptions`: the given flags over the environment.

    A ``--highs-threads`` below 1 is ignored with a warning (the environment variable or the default 1 then
    applies); a ``--benders-workers`` of 0 or less is the automatic setting.
    """
    if args.highs_threads is not None and args.highs_threads < 1:
        logging.warning("Ignoring --highs-threads %s: it must be at least 1.", args.highs_threads)
        args.highs_threads = None
    if args.benders_workers is not None and args.benders_workers < 1:
        args.benders_workers = None
    return RunOptions.from_cli(args)


def build_parser() -> argparse.ArgumentParser:
    """The ``cmd_run_flextool`` command line (also used to check what the GUI and child runs send)."""
    parser = argparse.ArgumentParser()
    parser.description = "Run flextool using the specified database URL. Return codes are 0: success, 1: infeasible or unbounded, -1: failure."
    parser.add_argument('input_db_url', help='Database URL to connect to (can be copied from Toolbox workflow db item')
    parser.add_argument('output_db_url', metavar='DB_URL', nargs='?', default=None, help='Save information about result location to database for post-processing')
    parser.add_argument('--settings-db-url', help='Settings for post-processing')
    parser.add_argument('--migrate', action='store_true', default=False,
                        help='Migrate the input database to the current FlexTool version before the run, if it is '
                             'older (the same migration as flextool-migrate-database).  A frozen database is not '
                             'migrated: the run stops.  Parallel runs with --migrate on one database migrate it '
                             'once; the others wait for it.  Without the flag, an older database is refused.')
    parser.add_argument('--scenario-name', help='Name for the scenario in the database that should be executed', nargs='?', default=None)
    parser.add_argument(
        '--debug',
        nargs='?',
        const='basic',
        default='off',
        choices=['off', 'basic', 'full'],
        metavar='LEVEL',
        help='Diagnostic verbosity level (default: off).  '
             '``off``  — quiet; only user-facing INFO and WARNING. '
             '``basic`` — verbose memory checkpoint trace + DEBUG log '
             'level; no tracemalloc, no perf overhead beyond extra '
             'stdout.  Bare ``--debug`` selects this level.  '
             '``full`` — basic plus tracemalloc-backed memory '
             'diagnostics CSV.  Tracemalloc instruments every Python '
             'allocation and typically slows allocation-heavy phases '
             '(input_derivation, cascade rolls) by 2-5×; use only '
             'when investigating Python-side allocation regressions.',
    )
    parser.add_argument(
        '--save-memory', action='store_true', default=None,
        help='Trade wall time for peak memory: build the LP, write it to '
             'a temp MPS file, drop everything Python-side AND the live '
             'HiGHS instance, then spawn a separate subprocess to solve '
             'the MPS in a clean address space. The parent process '
             '(holding ~7-11 GB of polars frames + FlexData) sits idle '
             'while the child does its ~50 GB active-solve work, so the '
             'two never compound in the same process heap. Adds ~+30-60 s '
             'I/O per sub-solve. Also disables warm-LP reuse across '
             'cascade iterations (the Problem is released after MPS '
             'write). Off by default; use when models OOM on the default '
             'in-process path.  Environment variable: '
             'FLEXTOOL_SAVE_MEMORY=1.',
    )
    parser.add_argument(
        '--warm-start', action='store_true', default=None,
        help='Reuse a cached HiGHS basis across solves of the same '
             'structural model, both in-process and on the --save-memory '
             'subprocess path (not for Benders solves); safe cold '
             'fallback on any mismatch.  Environment variable: '
             'FLEXTOOL_WARM_START=1.',
    )
    parser.add_argument(
        '--basis-cache-dir', metavar='PATH', default=None,
        help='Folder for the --warm-start basis cache (default: '
             '<work folder>/basis_cache).  Share one folder between runs '
             'to reuse bases across them.  Environment variable: '
             'FLEXTOOL_BASIS_CACHE_DIR; the flag wins.',
    )
    parser.add_argument(
        '--solver-config-dir', metavar='PATH', default=None,
        help='Folder of the solver option files (highs.opt, gurobi.opt, '
             'cplex.opt, xpress.opt, copt.opt).  Unset: solver_config/ next '
             'to the input database, then in the workspace of a GUI project '
             'database (<workspace>/projects/<project>/input_sources/), then '
             'in the Spine Toolbox project folder (--project-folder-file), '
             'then in the current folder; when none has an option file, '
             'FlexTool\'s defaults are used with a warning.  A solver whose '
             'file is missing in the folder uses FlexTool\'s default.  The run '
             'copies the files it used to <work folder>/solve_data/'
             'solver_config/.  Environment variable: '
             'FLEXTOOL_SOLVER_CONFIG_DIR; the flag wins.',
    )
    parser.add_argument('--output-spreadsheet', metavar='PATH', help='Save results to spreadsheet file')
    parser.add_argument('--write-methods', type=str, nargs='+', default=None,
                        choices=['plot', 'parquet', 'excel', 'csv', 'spinedb'],
                        help='Output methods to use (default: plot parquet)')
    parser.add_argument('--results-db-url', type=str, default=None,
                        help='Target SpineDB URL for the spinedb write-method (default: <output-dir>/results.sqlite)')
    parser.add_argument('--output-config', metavar='PATH',
                        default=None,
                        help='Path to output configuration file (default: the bundled flextool/schemas/default_plots.yaml)')
    parser.add_argument('--active-configs', type=str, nargs='+', default=None,
                        help='Active output configurations to use (default: default)')
    parser.add_argument('--plot-rows', type=int, nargs=2, default=None, metavar=('FIRST', 'LAST'),
                        help='First and last row to plot in time series (default: 0 167)')
    parser.add_argument('--output-location', metavar='PATH', default=None,
                        help='Override output location path')
    parser.add_argument('--output-subdir', metavar='NAME', default=None,
                        help='Subdirectory name under output_parquet/ (and the '
                             'other output dirs). Defaults to the scenario '
                             'name for backward compatibility.')
    parser.add_argument('--flextool-location', nargs='?', default=None, const=None,
                        help='When running in Spine Toolbox, this argument provides the location of FlexTool so outputs can be directed there (instead of work directories). Defaults to the user\'s current working directory. The value may be omitted (Spine Toolbox sometimes passes the bare flag) — in that case the default is used. Legacy bridge: kept for backward compatibility; prefer --project-folder-file.')
    parser.add_argument('--project-folder-file', metavar='PATH', default=None,
                        help='Path to a USER-LOCAL file whose CONTENTS name '
                             'the project folder to direct outputs into '
                             '(output_parquet/, results.sqlite, plots, and '
                             'the per-project plot_settings.yaml).  The '
                             'first non-empty, non-#-comment line is the '
                             'project folder: an absolute path is used as-is; '
                             'a relative path is resolved against the file\'s '
                             'repo anchor (its .parent.parent).  Spine Toolbox '
                             'points this at templates/project_folder.txt '
                             '(gitignored, seeded by flextool-update).  This '
                             'flag is a COMPLETE replacement for '
                             '--flextool-location: a missing / empty / '
                             'comment-only file does NOT fall through to the '
                             'work dir but anchors at the file\'s repo root '
                             '(its .parent.parent), so a supplied '
                             '--project-folder-file never lands outputs in '
                             'the CWD.  Lower precedence than '
                             '--output-location, higher than the '
                             'input_sources/ layout and --flextool-location.')
    parser.add_argument('--input-source-file', metavar='PATH', default=None,
                        help='The file the input database was made from (e.g. the Excel file the GUI converts). '
                             'Recorded in the run record (_run_status.json) so a reader can tell whether the '
                             'results are older than their source.')
    parser.add_argument('--run-launcher', metavar='NAME', default='direct',
                        help='Who started this run (gui, flextool-run, workflow, calibrator, netload; default '
                             'direct), recorded in the run record (_run_status.json).  A launcher that starts '
                             'another adds it only when the command line has none, so the outermost one is kept.')
    parser.add_argument('--work-folder', metavar='PATH', default=None,
                        help='Working directory for intermediate files (default: current directory). '
                             'Enables parallel scenario execution by isolating each run.')
    parser.add_argument('--only-first-file-per-plot', action='store_true', default=False,
                        help='Only produce the first file for each plot (quick overview mode)')
    parser.add_argument('--write-plot-data', action='store_true', default=False,
                        help='Also write the data drawn in each plot as a CSV file next to it (same name, .csv)')
    parser.add_argument('--region', metavar='GROUP_NAME', default=None,
                        help='Write the filtered inputs of one Benders region '
                             'to ``input_region_<GROUP_NAME>/`` and stop; no '
                             'solve runs.  The group must have '
                             '``decomposition_method=benders_regional`` in '
                             'the database.  For inspecting what a region '
                             'subproblem sees: cross-region processes are '
                             'replaced with import/export half-flows and the '
                             'coupling variables are listed in '
                             '``solve_data/region_coupling.csv``.  Benders '
                             'solves themselves are set per solve in the '
                             'database (solve.decomposition) and do not '
                             'need this flag.')
    # Decomposition is DB-driven and per-solve (v62): set
    # ``solve.decomposition = benders`` plus the per-solve
    # ``solve.benders_max_iter`` / ``benders_tolerance`` /
    # ``benders_in_out_weight`` knobs in the database.  The old global ``--decomposition`` / ``--lagrangian-*``
    # CLI flags were removed — the orchestrator reads the scheme per solve
    # so a single chain can mix monolithic and Benders solves.  See
    # docs/dev/decomposition.md.
    parser.add_argument('--threads', type=int, default=None, metavar='N',
                        help='Threads for one FlexTool run: FlexTool\'s data '
                             'processing and the solver.  FlexTool\'s own '
                             'processing (polars) uses at most N and at most 8 '
                             '(POLARS_MAX_THREADS can set another limit below '
                             'N); HiGHS gets threads=N (its parallel simplex '
                             'stays off) and a Benders solve runs at most N '
                             'region solves at once.  Default: the available '
                             'cores, with HiGHS and Benders as before.  When '
                             'several runs execute in parallel, give each about '
                             'cores / runs.  Environment variable: '
                             'FLEXTOOL_THREADS; the flag wins.')
    parser.add_argument('--highs-threads', type=int, default=None, metavar='N',
                        help='Number of HiGHS solver threads (default 1). '
                             'Values > 1 enable HiGHS parallel mode and trade '
                             'determinism for wall-clock speedup; goldens are '
                             'not guaranteed to reproduce across runs in that '
                             'mode.  At most --threads when that is given.  '
                             'Environment variable: '
                             'FLEXTOOL_HIGHS_THREADS; the flag wins.')
    parser.add_argument('--benders-workers', type=int, default=None, metavar='N',
                        help='Threads for the region solves of a Benders '
                             'solve (default, or 0: min(regions, cores - 1)). '
                             'A per-machine choice; the decomposition itself '
                             'is set per solve in the database.  Environment '
                             'variable: FLEXTOOL_BENDERS_WORKERS; the flag '
                             'wins.')
    parser.add_argument(
        '--scaling',
        choices=['off', 'solver_only', 'basic', 'full'],
        default=None,
        help=(
            "Choose FlexTool's autoscaler strategy. HiGHS' internal matrix "
            "equilibration (simplex_scale_strategy) is unaffected by this "
            "flag EXCEPT when --scaling=off, where it is forced to 0. To "
            "tune HiGHS-internal options, use the solver config file.\n"
            "\n"
            "  off          Disable ALL scaling, including HiGHS' internal "
            "matrix equilibration (forces simplex_scale_strategy=0). Use "
            "this if you want raw numerics or to export the truly unscaled "
            "LP. Expect HiGHS warnings.\n"
            "  solver_only  Disable the FlexTool autoscaler. HiGHS still "
            "scales the matrix internally per its own default "
            "(simplex_scale_strategy=2, equilibration). Useful when "
            "exporting MPS for an external solver.\n"
            "  basic        Compute LP ranges (Layer 1) and recommend "
            "power-of-two user_objective_scale + user_bound_scale to HiGHS "
            "(Layer 3). No LP-array mutation; MPS exports reflect the "
            "unscaled model. HiGHS' own matrix equilibration runs per its "
            "default.\n"
            "  full         The full autoscaler: range detection (Layer 1), "
            "semantic per-type column/row/cost scaling of the LP arrays "
            "(Layer 2), and HiGHS user_*_scale recommendation (Layer 3). "
            "Produces the most robust conditioning. Default.\n"
            "\n"
            "Precedence for the mode: this flag > FLEXTOOL_SCALING > the "
            "solve's scaling parameter > full.\n"
            "\n"
            "Precedence for user_objective_scale and user_bound_scale:\n"
            "  1. --user-bound-scale N (or FLEXTOOL_USER_BOUND_SCALE)\n"
            "  2. user_*_scale set via solver config file\n"
            "  3. Layer 3 autoscaler recommendation\n"
            "  4. HiGHS default (0)"
        ),
    )
    parser.add_argument('--user-bound-scale', type=int, default=None,
                        metavar='N',
                        help='HiGHS ``user_bound_scale`` override (power of '
                             'two: multiplies all col bounds and RHS by '
                             '2**N).  When HiGHS prints '
                             '"Consider setting the user_bound_scale option '
                             'to <N>" in its scaling warning, pass that '
                             '<N> here.  Clamped to [-10, 0].  When '
                             'unset, the autoscaler chooses.  Environment '
                             'variable: FLEXTOOL_USER_BOUND_SCALE; the flag '
                             'wins.')
    parser.add_argument('--presolve', choices=['on', 'off', 'choose'],
                        default=None,
                        help='HiGHS ``presolve`` override.  Default '
                             '(unset) keeps the determinism-pinned '
                             '"on" setting from '
                             '``DETERMINISM_OPTIONS``.  ``off`` disables '
                             'presolve entirely (much slower but useful '
                             'for memory or numerical diagnostics).  '
                             'Environment variable: '
                             'FLEXTOOL_HIGHS_PRESOLVE; the flag wins.')
    parser.add_argument('--solver-log-level',
                        choices=['silent', 'normal', 'verbose'],
                        default=None,
                        help='Solver log on the screen and in the run '
                             'log.  ``silent`` hides the log of every '
                             'solver (HiGHS ``output_flag=false``; a '
                             'commercial solver\'s log is still read and '
                             'saved as solve_data/subprocess/<solve>.log).  '
                             '``normal`` (default) shows it.  ``verbose`` '
                             '(HiGHS only) also sets ``log_dev_level=2`` '
                             'for per-iteration solver telemetry.  '
                             'Environment variable: '
                             'FLEXTOOL_SOLVER_LOG_LEVEL; the flag wins.')
    parser.add_argument('--solver-time-limit', type=float, default=None,
                        metavar='SECONDS',
                        help='Time limit of each solve in seconds, on every '
                             'solver (HiGHS time_limit, Gurobi TimeLimit, '
                             'CPLEX timelimit, Xpress MAXTIME, COPT '
                             'TimeLimit).  Unset (default) means no limit; 0 '
                             'or a negative value is ignored with a warning.  '
                             'It overrides a time limit in the solver\'s '
                             'option file (solver_config/<solver>.opt) or a '
                             'solve\'s solver_arguments.  A MIP solve stopped '
                             'at the limit keeps its best solution with a '
                             'warning; an LP, or a solve without a solution, '
                             'stops the run.  Environment variable: '
                             'FLEXTOOL_SOLVER_TIME_LIMIT; the flag wins.  '
                             '(FLEXTOOL_HIGHS_TIME_LIMIT still works for HiGHS '
                             'only.)')
    parser.add_argument('--solver-mip-gap', type=float, default=None,
                        metavar='GAP',
                        help='Relative MIP optimality gap of every solve of the '
                             'run, on every solver (HiGHS mip_rel_gap, Gurobi '
                             'MIPGap, CPLEX mip tolerances mipgap, Xpress '
                             'MIPRELSTOP, COPT RelGap).  Unset (default): each '
                             'solve uses its ``solver_mip_gap`` in the '
                             'database, else the gap in the solver\'s option '
                             'file.  It overrides both, and a gap in a solve\'s '
                             'solver_arguments.  Only affects MIP solves '
                             '(integer investments, unit-commitment / online '
                             'variables); pure-LP solves ignore it.  '
                             'Environment variable: FLEXTOOL_SOLVER_MIP_GAP; '
                             'the flag wins.  (FLEXTOOL_HIGHS_MIP_GAP still '
                             'works for HiGHS only.)')
    parser.add_argument('--csv-dump', action='store_true',
                        default=False,
                        help='Debug visibility for cascade-internal '
                             'artefacts. Default: the cascade keeps '
                             'input/, solve_data/, cross_solve/, and '
                             'output_raw/ off disk in the final work '
                             'folder, leaving only the user-facing '
                             'output_parquet/<scenario>/ tree (plus any '
                             'output_csv/, output_excel/, output_plots/ '
                             'requested via --write-methods). With the '
                             'flag set, every intermediate directory '
                             'survives the run for inspection, and each '
                             'sub-solve dumps its inputs to input/ and '
                             'solve_data/ (one folder: a later sub-solve '
                             'overwrites the files of an earlier one).')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_options = _run_options_from_args(args)

    # Accept either a SQLAlchemy URL ("sqlite:///path") or a bare
    # filesystem path ("path/to.sqlite") for any DB argument. Downstream
    # readers (SpineDbReader) already do this, but DatabaseMapping calls
    # in this file consume the args directly, so normalise once here.
    def _as_db_url(value):
        if value is None:
            return None
        return value if "://" in value else f"sqlite:///{value}"

    args.input_db_url = _as_db_url(args.input_db_url)
    args.output_db_url = _as_db_url(args.output_db_url)
    args.settings_db_url = _as_db_url(args.settings_db_url)

    input_db_url = args.input_db_url
    settings_db_url = args.settings_db_url
    scenario_name = args.scenario_name
    debug_level = args.debug  # 'off' | 'basic' | 'full'
    DEBUG = debug_level != 'off'
    # ``--debug=basic`` widens stdout to include the full per-checkpoint
    # phase-progress trace (every memory recorder event, not just the
    # six whitelisted phase labels).  ``--debug=full`` additionally
    # enables tracemalloc-backed diagnostics that write the
    # per-checkpoint CSV to ``solve_data/memory_diagnostics.csv`` — the
    # tracemalloc instrumentation typically slows allocation-heavy
    # phases by 2-5×, so it is gated to the explicit ``full`` opt-in.
    # ``setdefault`` lets a caller still override either env var.
    if debug_level in ('basic', 'full'):
        os.environ.setdefault('FLEXTOOL_MEMORY_VERBOSE', '1')
    if debug_level == 'full':
        os.environ.setdefault('FLEXTOOL_MEMORY_DIAGNOSTICS', '1')
    # The TRUE output root (where outputs land, and what is persisted to
    # the "Output info" DB as scenario/output_location) is resolved by a
    # 5-tier rule (see ``resolve_output_path`` for the full rationale):
    #   1. ``--output-location``                         (explicit wins),
    #   2. ``--project-folder-file``                     (user-local
    #      CONTENTS name a project folder; when blank/    redirect; a
    #      missing, the file's .parent.parent repo root)  supplied file
    #                                                      NEVER falls
    #                                                      through to CWD),
    #   3. ``<project>`` when the input DB sits in an    (GUI project
    #      ``input_sources/`` dir                         layout),
    #   4. ``--flextool-location``.parent.parent         (legacy bridge),
    #   5. CWD                                            (fallback).
    # Tiers 3-5 are only reached when NO --project-folder-file is supplied.
    output_path = resolve_output_path(
        input_db_url=args.input_db_url,
        flextool_location=args.flextool_location,
        output_location=args.output_location,
        cwd=Path.cwd(),
        project_folder_file=args.project_folder_file,
    )
    work_folder = Path(args.work_folder) if args.work_folder else Path.cwd()
    work_folder.mkdir(parents=True, exist_ok=True)
    wf = work_folder

    # Default formatter strips the ``INFO:<file>:<line>:`` preamble so
    # user-facing INFO lines (license status, solver progress, etc.)
    # read as plain prose.  WARNING / ERROR carry their level via the
    # message body of the ``logging.warning(...)`` calls themselves
    # ("Failed to ...", etc.), so dropping ``%(levelname)s`` here
    # doesn't hide the severity.  ``--debug`` restores the full
    # prefix for diagnosis.
    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format=(
            '%(levelname)s:%(filename)s:%(lineno)d:%(message)s'
            if DEBUG else '%(message)s'
        ),
        # Control characters from names in the input data are shown as \xNN, not sent to the terminal.
        handlers=[make_printable(logging.StreamHandler(sys.stdout))]
    )
    if not DEBUG:
        # Silence routine "wrote …" / "Wrote N output variables" INFO
        # chatter from the per-solve output + handoff writers in regular
        # mode.  These fire on every sub-solve and tell the user
        # nothing they can't infer from "Solver" + the parquet
        # contents.  WARNINGs (failed writes, missing files, etc.)
        # still surface because we only raise the writer-module
        # thresholds to WARNING.  --debug restores the full chatter.
        for _noisy in (
            "flextool.process_outputs.handoff_writers",
            "flextool.process_outputs.read_highs_solution",
        ):
            logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Self-heal missing lightweight settings DBs so fresh clones don't
    # fail opaquely when the user forgot to run `flextool-update`. Only
    # seeds output_info / output_settings / comparison_settings by
    # basename; other paths are left untouched.
    for _candidate in (args.output_db_url, args.settings_db_url):
        try:
            ensure_settings_db(_candidate)
        except Exception as _exc:
            logging.warning("Failed to auto-seed %s: %s", _candidate, _exc)

    # --migrate: bring an older input database to the current version before anything opens it (the engine refuses
    # one that is not current).  A frozen database stops the run, as flextool-migrate-database does.
    if args.migrate:
        _migrate_input_or_exit(input_db_url)

    # Phase-timing recorder: constructed once per CLI invocation, lives
    # on ``runner.state.timing_recorder``, writes a structured timings.csv
    # at <work_folder>/solve_data/timings.csv (one row per phase, atomic
    # append style so a crash mid-run still leaves usable data).
    timing_recorder = TimingRecorder(work_folder=wf, scenario=scenario_name)
    t_total_start = time.perf_counter()

    # --- Regional filter mode ---------------------------------------------
    # ``--region GROUP`` writes ``input_region_<GROUP>/`` (the inputs one
    # Benders region subproblem sees) and exits without invoking the solver.
    if args.region:
        from flextool.decomposition.region_decomposition import (
            write_input_for_region as _write_input_for_region,
        )
        _region_output = wf / f"input_region_{args.region}"
        try:
            result = _write_input_for_region(
                input_db_url=input_db_url,
                scenario_name=scenario_name,
                logger=logging.getLogger("flextool.region_filter"),
                region_group=args.region,
                output_dir=_region_output,
                work_folder=work_folder,
            )
        except Exception as exc:
            logging.error("Regional filter failed: %s", exc, exc_info=True)
            sys.exit(-1)
        print(f"Wrote filtered region inputs to {_region_output}")
        print(
            f"Coupling variables ({len(result['half_flows'])}): "
            f"{[hf.virtual_node for hf in result['half_flows']]}"
        )
        sys.exit(0)

    # Benders decomposition is now DB-driven and per-solve: the
    # orchestrator reads ``solve.decomposition`` for each solve and runs
    # the Benders region driver for the ones set to ``benders``
    # (see engine.solve._orchestration / docs/dev/decomposition.md).  The
    # old global ``--decomposition lagrangian`` standalone path was
    # removed; nothing special happens here — the normal run path below
    # handles every scheme.

    # Resolve scenario_name when omitted: pull it from the DB's
    # active filter.  ``run_chain_from_db`` requires a concrete
    # scenario (it never picks one), so stop here when there is none.
    if not scenario_name:
        with open_database_mapping(input_db_url) as db_map:
            _filters = db_map.get_filter_configs()
            if _filters:
                scenario_name = name_from_dict(_filters[0])
    if not scenario_name:
        logging.error(
            "No scenario to run for %s: pass --scenario-name, or use a database URL "
            "that carries a scenario filter (Spine Toolbox: connect a scenario filter "
            "to the tool).", input_db_url,
        )
        sys.exit(1)

    # Where the outputs land and which write methods run: explicit flags >
    # settings DB > defaults, resolved once here (the same rule
    # write_outputs applies) so the run record, write_outputs, timings.csv
    # and the output-info DB all name the same folder.
    output_subdir = args.output_subdir or scenario_name
    # The scenario name and the output folder name become folder names (output_parquet/<subdir>/, ...): refuse a
    # name that is not one plain folder name before anything is written or cleaned up.
    from flextool.safe_names import UnsafeNameError, check_output_name
    try:
        check_output_name(scenario_name, "scenario name")
        if args.output_subdir:
            check_output_name(args.output_subdir, "--output-subdir name")
    except UnsafeNameError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    output_location, write_methods = resolve_output_settings(
        write_methods=args.write_methods,
        output_location=args.output_location,
        settings_db_url=settings_db_url,
        fallback_output_location=str(output_path),
    )

    # The run record (_run_status.json, _run.log) starts here, before the
    # header block, so the header and every later warning are in it.
    rec, _cleanup_run_record = _start_run_record(
        args=args, run_options=run_options, scenario_name=scenario_name,
        output_location=output_location, output_subdir=output_subdir,
        write_methods=write_methods, work_folder=work_folder,
    )
    _restore_signals = _install_termination_handlers()
    state, exit_code, error = "succeeded", 0, None
    try:
        _run_scenario(
            args=args, run_options=run_options, rec=rec, scenario_name=scenario_name,
            work_folder=work_folder, output_path=output_path, output_location=output_location,
            output_subdir=output_subdir, write_methods=write_methods, timing_recorder=timing_recorder,
            t_total_start=t_total_start,
        )
    except RunTerminated as exc:
        signum = int(exc.code) - 128
        state, exit_code = "interrupted", exc.code
        error = {"type": "RunTerminated", "message": f"stopped by {signal.Signals(signum).name}"}
        raise
    except KeyboardInterrupt:
        # An unhandled KeyboardInterrupt ends the interpreter by SIGINT (Windows: STATUS_CONTROL_C_EXIT).
        state, exit_code = "interrupted", (-signal.SIGINT if os.name == "posix" else run_record.WINDOWS_CTRL_C_EXIT_SIGNED)
        error = {"type": "KeyboardInterrupt", "message": "interrupted by Ctrl-C"}
        raise
    except SystemExit as exc:
        if exc.code in (None, 0):
            state, exit_code = "succeeded", 0
        else:
            state = "failed"
            exit_code = exc.code if isinstance(exc.code, int) else 1
            error = {"type": "SystemExit", "message": f"exit code {exc.code}"}
        raise
    except BaseException as exc:
        state, exit_code = "failed", 1
        error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        rec.finish(state, exit_code=exit_code, error=error)
        _cleanup_run_record()
        _restore_signals()


#: The exit code of a run stopped by a frozen database (``--migrate``), as in ``flextool-migrate-database``.
EXIT_FROZEN = 2


def _migrate_input_or_exit(input_db_url: str) -> None:
    """Migrate the input database for ``--migrate`` (once across parallel runs); exit 2 on a frozen database."""
    from flextool._file_lock import FileLockError
    from flextool.update_flextool.frozen import FrozenDatabaseError
    from flextool.update_flextool.migrate_before_run import migrate_before_run

    try:
        migrate_before_run(input_db_url)
    except FrozenDatabaseError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        sys.exit(EXIT_FROZEN)
    except FileLockError as exc:
        print(f"The input database was not migrated: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


def _run_scenario(*, args, run_options, rec, scenario_name, work_folder, output_path, output_location,
                  output_subdir, write_methods, timing_recorder, t_total_start):
    """The run of one scenario after its run record has started: header, cascade, outputs, output-info DB.

    Exits 1 (``sys.exit``) when the cascade fails; the caller records the run's end state.
    """
    input_db_url = args.input_db_url
    settings_db_url = args.settings_db_url
    wf = work_folder
    # The folder of the solver option files, resolved once: the header shows it and the engine uses it.
    solver_config_source = resolve_solver_config_source(
        run_options=run_options, input_db_url=input_db_url, project_folder_file=args.project_folder_file,
    )

    # Header block — one aligned key/value pair per line, blank lines
    # before and after, so the user has a self-contained summary of
    # what's being run.
    _header_pairs = [
        ("Work dir", str(work_folder)),
        ("DB URL", str(input_db_url)),
        ("Scenario", scenario_name),
        ("Output", str(output_location)),
        ("Options", ", ".join(run_options.summary_lines()) or "defaults"),
        ("Threads", header_text(run_options)),
        ("Solver config", solver_config_source.describe()),
    ]
    _threads_warning = peek_mismatch(run_options)
    if _threads_warning is not None:
        logging.warning(_threads_warning)
    _header_keyw = max(len(k) for k, _ in _header_pairs) + 2
    print("")
    for _k, _v in _header_pairs:
        print(f"{(_k + ':').ljust(_header_keyw)}{_v}")
    # No trailing blank line here -- the "Available solvers:" log emits
    # its own trailing newline so the blank lands AFTER the licence line,
    # not before it.

    try:
        return_code, last_step = _run_solve(
            args, scenario_name, work_folder, timing_recorder, run_options,
            solver_config_source=solver_config_source,
        )
    except Exception as e:
        # FlexToolUserError signals a user-visible configuration problem
        # (unknown solver, missing license, model-level solver error), and
        # FlexToolConfigError one in the input data (no scenario, an empty
        # scenario, an old database version).  The message is already
        # human-readable; logging the traceback on top is just noise.
        # Other exceptions get the full traceback because they're
        # (probably) bugs in flextool.
        try:
            from flextool.engine.input._solve_state import FlexToolConfigError
            from flextool.engine.solve._solver_dispatch import (
                FlexToolUserError,
            )
            user_errors: tuple[type[BaseException], ...] = (FlexToolConfigError, FlexToolUserError)
        except Exception:  # noqa: BLE001
            user_errors = ()
        if isinstance(e, user_errors):
            logging.error(str(e))
        else:
            logging.error(
                f"Native cascade failed: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
        # The record keeps the exception (for a rejected LP, ``finish``
        # replaces its "Error: 1" by the acceptance message).
        rec.finish("failed", exit_code=1, error={"type": type(e).__name__, "message": str(e)})
        sys.exit(1)

    # If successful and requested, write outputs
    if return_code == 0:
        t_write_outputs = time.perf_counter()
        # Δ.31 — pass the last step's flex_data + solution so
        # write_outputs can build par/s in memory.  ``solve_name``
        # is the complete sub-solve identifier (e.g. ``y2025_5week``
        # for a roll, or just the scenario name for a single solve).
        wo_solve_name = (
            last_step.solve_name if last_step else None
        ) or scenario_name
        # A standalone Benders-only final solve carries only a
        # SnapshotSolution invest carrier (not a full Solution), so it
        # cannot yet drive processed outputs (TIER 2, planned follow-up).
        # Emit a clear, targeted notice and SKIP write_outputs entirely
        # rather than letting it fail and degrade to a generic warning.
        # The invest→dispatch chain ends on a real dispatch Solution
        # (is_benders=False) and is unaffected.
        if last_step is not None and getattr(
            last_step, "is_benders", False
        ):
            logging.info(
                "Final solve '%s' ran under decomposition=benders and "
                "does not yet produce processed outputs on its own. The "
                "decomposition objective/region summary was logged above. "
                "To get output files, add a downstream dispatch solve to "
                "the chain (model.solves = [%s, <dispatch solve>]); the "
                "dispatch solve produces the outputs. (Standalone "
                "Benders output processing is a planned follow-up.)",
                wo_solve_name,
                wo_solve_name,
            )
            rec.outputs_skipped(
                "the final solve ran under Benders decomposition, which does not produce outputs on its own"
            )
        else:
            try:
                # Multi-solve (rolling) note: the last step alone would
                # collapse par/s to the final roll's (d,t).  write_outputs
                # detects the per-roll realized slices persisted under
                # ``output_raw/`` (``has_persisted_slices``) and unions them
                # into the full-timeline par/s; ``last_step`` then only
                # supplies the static (solve-invariant) attrs + the per-attr
                # shape template.  We deliberately do NOT pass ``solve_steps``
                # here — the union activates on persisted-parquet presence and
                # carries solve labels via parquet filenames + the
                # ``output_raw/_solve_order.txt`` creation-order manifest.
                wo_flex_data = last_step.flex_data if last_step else None
                wo_solution = last_step.solution if last_step else None
                write_outputs(
                    scenario_name=scenario_name,
                    output_location=output_location,
                    subdir=output_subdir,
                    output_config_path=args.output_config,
                    active_configs=args.active_configs,
                    write_methods=write_methods,
                    plot_rows=(
                        tuple(args.plot_rows) if args.plot_rows else None
                    ),
                    settings_db_url=settings_db_url,
                    fallback_output_location=str(output_path),
                    raw_output_dir=str(wf / 'output_raw'),
                    only_first_file=args.only_first_file_per_plot,
                    write_plot_data=args.write_plot_data,
                    timing_recorder=timing_recorder,
                    flex_data=wo_flex_data,
                    solution=wo_solution,
                    solve_name=wo_solve_name,
                    flex_data_provider=getattr(
                        last_step, "flex_data_provider", None
                    ),
                    results_db_url=args.results_db_url,
                )
                if 'parquet' not in write_methods:
                    rec.outputs_not_requested()
            except FileNotFoundError as exc:
                # The in-memory parameter / set path doesn't read
                # ``solve_data/`` CSVs, but ``read_variables`` still
                # reads ``output_raw/`` parquets.  Catch missing-parquet
                # cases here (rare) and exit cleanly.
                logging.warning(
                    "write_outputs failed (%s).  output_raw/ artefacts "
                    "ARE produced; downstream output_csv/, output_parquet/, "
                    "output_excel/, output_plots/ are skipped on this run.",
                    exc,
                )
                rec.outputs_skipped(f"writing the outputs failed: {exc}")
        timing_recorder.record('write_outputs', subphase='total',
                               seconds=time.perf_counter() - t_write_outputs,
                               t_start=t_write_outputs)

        # output_raw/ is the cascade's intermediate parquet stash for
        # write_outputs to consume. Keep it on disk only when the user
        # opted in via --csv-dump (debug). On a normal run the user only
        # wants the canonical output_parquet/<scenario>/ tree.
        if not args.csv_dump:
            raw_dir = wf / 'output_raw'
            if raw_dir.exists():
                shutil.rmtree(raw_dir, ignore_errors=True)

    full_seconds = time.perf_counter() - t_total_start
    print("\n--- Full execution time %.4s seconds ---------------------------------------" % full_seconds)
    print("--------------------------------------------------------------------------\n")
    timing_recorder.record('total', seconds=full_seconds, t_start=t_total_start)

    # Move timings.csv into the per-scenario output dir alongside
    # summary_solve.csv, under the resolved output location (the one
    # write_outputs used).
    try:
        _final_csv_dir = (
            Path(output_location) / 'output_csv' / output_subdir
            if output_subdir else
            Path(output_location) / 'output_csv'
        )
        timing_recorder.finalize(_final_csv_dir)
    except Exception as _exc:
        logging.warning("Failed to copy timings.csv to output dir: %s", _exc)

    # A failed cascade (a solve the scan did not accept, or no solve at all)
    # wrote no outputs of this run: exit 1 and leave the output-info database
    # alone, so no reader takes this run's ``finish_time`` for finished results.
    if return_code != 0:
        logging.error(
            "Run of scenario %r failed; no outputs were written and the "
            "output-info database was not updated.", scenario_name,
        )
        rec.finish("failed", exit_code=1, error=_scan_failure_error(last_step, scenario_name))
        sys.exit(1)

    # Write scenario information to output database if provided
    if args.output_db_url:
        # Check if database exists
        db_exists = os.path.exists(args.output_db_url.replace('sqlite:///', ''))

        with open_database_mapping(args.output_db_url, create=not db_exists) as output_db:
            # Create/update scenario class if it doesn't exist
            output_db.add_or_update_entity_class(name="scenario")

            # Create/update parameter definition for 'output_location'
            output_db.add_or_update_parameter_definition(
                entity_class_name="scenario",
                name="output_location",
                description="Full path to the working directory"
            )

            # Add/update scenario entity
            output_db.add_or_update_entity(
                entity_class_name="scenario",
                name=scenario_name
            )

            output_db.add_or_update_alternative(name=scenario_name)

            # Convert folder path to database representation: the resolved
            # output location, where this run's outputs actually are.
            value, type_ = to_database(str(output_location))

            # Add/update folder infio
            output_db.add_or_update_parameter_value(
                entity_class_name="scenario",
                entity_byname=(scenario_name,),
                parameter_definition_name="output_location",
                alternative_name=scenario_name,
                value=value,
                type=type_
            )

            output_db.add_or_update_parameter_definition(
                entity_class_name="scenario",
                name="finish_time",
                description="Timestamp when the scenario run finished"
            )

            dt_value = DateTime(datetime.now())
            value, type_ = to_database(dt_value)

            # Add/update execution time
            output_db.add_or_update_parameter_value(
                entity_class_name="scenario",
                entity_byname=(scenario_name,),
                parameter_definition_name="finish_time",
                alternative_name=scenario_name,
                value=value,
                type=type_
            )

            try:
                output_db.commit_session("Added/updated scenario information")
            except NothingToCommit:
                pass



# Debug flag
DEBUG = False  # Set via environment variable or config

if __name__ == '__main__':
    run_tool(main)
