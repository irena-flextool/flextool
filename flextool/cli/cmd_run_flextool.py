import os
# glibc malloc arena cap — set BEFORE any C-extension import that
# allocates via malloc.  glibc defaults to up to 8 × ncores arenas
# (≈256 on a 32-core workstation); each arena holds its own
# freed-but-not-returned-to-OS pages, so worst-case fragmentation
# scales with core count.  Capping to 4 is a precautionary middle
# ground: ~64× reduction vs the default, while still allowing up to
# four concurrent allocators (HiGHS parallel presolve, Lagrangian
# scenario runs) without serialising every malloc through one heap.
# No measured benefit on the FlexTool cascade workload as of this
# writing — FlexTool's hot path is essentially single-threaded
# (polars pinned to 1 thread, --highs-threads typically 1).  Kept
# only as a cheap precaution against future workloads where arena
# growth could matter.  ``setdefault`` so the shell wins.
os.environ.setdefault("MALLOC_ARENA_MAX", "4")

import argparse
import sys
import logging
import shutil
import traceback
from pathlib import Path
from datetime import datetime
import time
from flextool.process_outputs.write_outputs import write_outputs
from flextool.cli._timing import TimingRecorder
from flextool.common_utils.precision import resolve_precision_digits
from flextool.update_flextool.ensure_settings_db import ensure_settings_db
from spinedb_api.filters.tools import name_from_dict
from spinedb_api import DatabaseMapping, to_database, DateTime
from spinedb_api.exception import NothingToCommit

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


def _run_solve(args, scenario_name, work_folder, timing_recorder):
    """Δ.21 — drive the native polar_high cascade.

    Returns a tuple ``(return_code, last_step)`` where ``last_step``
    is the :class:`flextool.engine_polars.OrchestrationStep` of the
    final (or only) sub-solve, used by the caller to thread
    ``flex_data`` + ``solution`` into ``write_outputs`` for the
    in-memory parameter / set namespace path (Δ.31).

    Δ.25: when ``--fast-single-solve`` is passed, dispatch to the
    surgical single-solve path that bypasses
    ``write_input``/``run_chain_from_db`` entirely.  Experimental.
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

    # Δ.25 fast single-solve dispatch.
    if getattr(args, 'fast_single_solve', False):
        from flextool.engine_polars import run_single_solve_from_db
        if not scenario_name:
            logging.error(
                "--fast-single-solve requires --scenario-name "
                "(the fast path doesn't auto-pick scenarios)."
            )
            return 1, None
        t_solve_start = time.perf_counter()
        step = run_single_solve_from_db(
            args.input_db_url,
            scenario_name,
            work_folder=work_folder,
        )
        all_solves_seconds = time.perf_counter() - t_solve_start
        print("--- Fast single-solve time %.4s seconds ---" % all_solves_seconds)
        timing_recorder.record('all_solves', seconds=all_solves_seconds,
                               t_start=t_solve_start)
        if not step.optimal:
            logging.error(
                "Fast single-solve: non-optimal (status=%r).",
                getattr(step.solution, "status", None)
                if step.solution else None,
            )
            return 1, step
        return 0, step

    from flextool.engine_polars import run_chain_from_db

    # Drive the native cascade end-to-end.  ``run_chain_from_db``
    # handles flextool's preprocessing (write_input) AND the per-solve
    # LP build+solve+handoff loop in-process.
    t_solve_start = time.perf_counter()
    steps = run_chain_from_db(
        args.input_db_url,
        scenario_name,
        work_folder=work_folder,
        csv_dump=csv_dump_on,
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
    # Non-optimal in any sub-solve → infeasible/unbounded exit code.
    last_step = None
    for name, step in steps.items():
        last_step = step
        # Phase C.5 — intermediate steps no longer hold ``solution``
        # under default (slim) cascade; read the slim ``optimal``
        # summary instead so the non-optimal check works without
        # ``keep_solutions=True``.  ``step.solution`` is only populated
        # for the LAST step (or every step under ``keep_solutions``).
        if not step.optimal:
            logging.error(
                "Native cascade: solve %r non-optimal (status=%r); "
                "exit=1 (infeasible/unbounded).",
                name,
                getattr(step.solution, "status", None) if step.solution else None,
            )
            return 1, step
    return 0, last_step


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Run flextool using the specified database URL. Return codes are 0: success, 1: infeasible or unbounded, -1: failure."
    parser.add_argument('input_db_url', help='Database URL to connect to (can be copied from Toolbox workflow db item')
    parser.add_argument('output_db_url', metavar='DB_URL', nargs='?', default=None, help='Save information about result location to database for post-processing')
    parser.add_argument('--settings-db-url', help='Settings for post-processing')
    parser.add_argument('--scenario-name', help='Name for the scenario in the database that should be executed', nargs='?', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        '--save-memory', action='store_true',
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
             'in-process path.',
    )
    parser.add_argument('--output-spreadsheet', metavar='PATH', help='Save results to spreadsheet file')
    parser.add_argument('--write-methods', type=str, nargs='+', default=None,
                        choices=['plot', 'parquet', 'excel', 'csv'],
                        help='Output methods to use (default: plot parquet)')
    parser.add_argument('--output-config', metavar='PATH',
                        default=None,
                        help='Path to output configuration file (default: templates/default_plots.yaml)')
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
    parser.add_argument('--flextool-location', default=None,
                        help='When running in Spine Toolbox, this argument provides the location of FlexTool so outputs can be directed there (instead of work directories). Defaults to the user\'s current working directory.')
    parser.add_argument('--work-folder', metavar='PATH', default=None,
                        help='Working directory for intermediate files (default: current directory). '
                             'Enables parallel scenario execution by isolating each run.')
    parser.add_argument('--only-first-file-per-plot', action='store_true', default=False,
                        help='Only produce the first file for each plot (quick overview mode)')
    parser.add_argument('--precision-digits', metavar='N', type=int, default=None,
                        help='Round every numeric input parameter to N significant '
                             'figures before writing CSVs (typical: 10).  '
                             'Collapses accumulated float-noise so HiGHS '
                             'mip_detect_symmetry can aggregate structurally-identical '
                             'coefficients.  0 or unset disables rounding (default).  '
                             'Overrides the FLEXTOOL_PRECISION_DIGITS env var.')
    parser.add_argument('--region', metavar='GROUP_NAME', default=None,
                        help='Produce a filtered per-region input directory '
                             '``input_region_<GROUP_NAME>/`` for Lagrangian '
                             'decomposition (Agent 3.1).  The group must have '
                             '``decomposition_method=lagrangian_region`` in '
                             'the DB.  Cross-region processes are replaced '
                             'with import/export half-flows; the coupling '
                             'variables are listed in '
                             '``solve_data/region_coupling.csv``.  When this '
                             'flag is set, no solve runs — this is the '
                             'filter-only entry point used by the coordinator.')
    parser.add_argument('--decomposition',
                        metavar='SCHEME',
                        choices=['none', 'lagrangian'],
                        default='none',
                        help='Run the full solve via a decomposition scheme '
                             'instead of the monolithic orchestrator.  Only '
                             '``lagrangian`` is currently supported: it drives '
                             'one HiGHS instance per decomposition-region and '
                             'prices the cross-region pipeline flows via a '
                             'damped subgradient on λ until the '
                             'primal-average imbalance is below tolerance '
                             '(Agent 3.2).  Requires at least two groups '
                             'declared with ``decomposition_method='
                             '"lagrangian_region"``.  See docs/decomposition'
                             '.md for the full workflow.')
    parser.add_argument('--lagrangian-alpha', type=float, default=0.1,
                        help='Base step size for the Lagrangian subgradient '
                             'loop (default 0.1).  The actual per-iteration '
                             'step is ``α / √k``.')
    parser.add_argument('--lagrangian-max-iter', type=int, default=80,
                        help='Maximum outer-loop iterations for '
                             '``--decomposition lagrangian`` (default 80).')
    parser.add_argument('--lagrangian-tolerance', type=float, default=1.0,
                        help='Tail-averaged imbalance threshold (primal '
                             'units) for declaring Lagrangian convergence '
                             '(default 1.0).')
    parser.add_argument('--highs-threads', type=int, default=1,
                        help='Number of HiGHS solver threads.  Default 1. '
                             'Values > 1 enable HiGHS parallel mode and trade '
                             'determinism for wall-clock speedup; goldens are '
                             'not guaranteed to reproduce across runs in that '
                             'mode.')
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
            "Precedence for user_objective_scale and user_bound_scale:\n"
            "  1. --user-bound-scale N (CLI override)\n"
            "  2. user_*_scale set via solver config file\n"
            "  3. Layer 3 autoscaler recommendation\n"
            "  4. HiGHS default (0)\n"
            "\n"
            "Env fallback: FLEXTOOL_SCALING."
        ),
    )
    parser.add_argument('--user-bound-scale', type=int, default=None,
                        metavar='N',
                        help='HiGHS ``user_bound_scale`` override (power of '
                             'two: multiplies all col bounds and RHS by '
                             '2**N).  When HiGHS prints '
                             '"Consider setting the user_bound_scale option '
                             'to <N>" in its scaling warning, pass that '
                             '<N> here.  Clamped to [-10, 0].  Overrides '
                             'any DB value; falls through to the '
                             'input-data heuristic when unset.')
    parser.add_argument('--presolve', choices=['on', 'off', 'choose'],
                        default=None,
                        help='HiGHS ``presolve`` override.  Default '
                             '(unset) keeps the determinism-pinned '
                             '"on" setting from '
                             '``DETERMINISM_OPTIONS``.  ``off`` disables '
                             'presolve entirely (much slower but useful '
                             'for memory or numerical diagnostics).')
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
                             'survives the run for inspection.')
    parser.add_argument('--fast-single-solve', action='store_true',
                        default=False,
                        help='(EXPERIMENTAL) — bypass the '
                             'standard ``flextool.input_derivation`` '
                             'pipeline entirely.  Reads inputs '
                             'directly from Spine via SpineDbReader, '
                             'builds the LP via the override chain, '
                             'and emits ``output_raw/`` parquets via '
                             'a tiny support-CSV bootstrap.  '
                             'Single-solve only; no rolling, no '
                             'nested cascade, no warm-LP, no handoff '
                             'plumbing.  Raises loudly on any helper '
                             'coverage gap (no fallback to the slow '
                             'path).  Use for cold-start latency '
                             'benchmarking on simple fixtures; the '
                             'default path remains ``run_chain_from_db``.')

    args = parser.parse_args()
    # --user-bound-scale / --presolve are surfaced through env vars
    # read by ``_orchestration._finalise_highs_options`` and the
    # cascade's user_bound_scale resolution.  Env vars keep the
    # threading shallow: no new kwargs on run_chain_from_db /
    # run_orchestration / _drive_cascade required.
    if args.user_bound_scale is not None:
        os.environ['FLEXTOOL_USER_BOUND_SCALE'] = str(args.user_bound_scale)
    if args.presolve is not None:
        os.environ['FLEXTOOL_HIGHS_PRESOLVE'] = args.presolve
    if args.highs_threads is not None and args.highs_threads >= 1:
        os.environ['FLEXTOOL_HIGHS_THREADS'] = str(args.highs_threads)
    # ``--scaling`` (off/solver_only/basic/full) — CLI > env > default-full.
    # Surfacing via the same ``FLEXTOOL_SCALING`` env var that
    # ``resolve_scaling_config`` already consults keeps the threading
    # shallow (no new kwargs on run_chain_from_db / run_orchestration /
    # _drive_cascade).  When the flag is unset (``args.scaling is None``)
    # the existing env value — if any — survives untouched, preserving
    # the env-fallback contract.
    if args.scaling is not None:
        os.environ['FLEXTOOL_SCALING'] = args.scaling
    # ``--save-memory`` — opt-in peak-RSS reduction at solve time.
    # Plumbed via env var so the orchestrator picks it up without an
    # extra kwarg on ``run_chain_from_db`` / ``_drive_cascade``.
    if args.save_memory:
        os.environ['FLEXTOOL_SAVE_MEMORY'] = '1'

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
    DEBUG = args.debug
    # ``--debug`` widens stdout to also include the full per-checkpoint
    # phase-progress trace (every memory recorder event, not just the
    # six whitelisted phase labels) and writes the per-checkpoint CSV
    # to ``solve_data/memory_diagnostics.csv``.  Both env vars are only
    # set when not already provided so a caller can still override.
    if DEBUG:
        os.environ.setdefault('FLEXTOOL_MEMORY_VERBOSE', '1')
        os.environ.setdefault('FLEXTOOL_MEMORY_DIAGNOSTICS', '1')
    # Legacy: Spine Toolbox passed ``--flextool-location <repo>/template/flextool_location.txt``
    # so the output dir resolved to the repo root via ``.parent.parent``.
    # Now defaults to CWD (the user's project directory) when unset, and
    # still honours the explicit path-walk for old Toolbox profiles.
    if args.flextool_location:
        output_path = Path(args.flextool_location).resolve().parent.parent
    else:
        output_path = Path.cwd()
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
        handlers=[logging.StreamHandler(sys.stdout)]
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
            "flextool.engine_polars.handoff_writers",
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

    # Phase-timing recorder: constructed once per CLI invocation, lives
    # on ``runner.state.timing_recorder``, writes a structured timings.csv
    # at <work_folder>/solve_data/timings.csv (one row per phase, atomic
    # append style so a crash mid-run still leaves usable data).
    timing_recorder = TimingRecorder(work_folder=wf, scenario=scenario_name)
    t_total_start = time.perf_counter()

    # resolve_precision_digits respects FLEXTOOL_PRECISION_DIGITS env override.
    effective_precision = resolve_precision_digits(args.precision_digits)

    # --- Regional filter mode (Agent 3.1) --------------------------------
    # ``--region GROUP`` produces ``input_region_<GROUP>/`` and exits
    # without invoking the solver.  The Lagrangian coordinator (Agent
    # 3.2) then orchestrates multiple region solves itself.
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
                precision_digits=effective_precision,
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

    # --- Lagrangian decomposition mode ----------------------------------
    # ``--decomposition lagrangian`` drives the spatial Lagrangian
    # coordinator instead of the monolithic orchestrator.  Requires the
    # scenario to declare ≥ 2 decomposition-region groups; we bail out
    # with a clear error if that precondition is unmet.  The native
    # coordinator lives in ``engine_polars._lagrangian`` (see
    # specs/lagrangian_port_handoff.md for the Δ.22 rewiring history).
    if args.decomposition == 'lagrangian':
        from flextool.decomposition.region_filter import (
            discover_decomposition_regions_from_db,
        )
        if not scenario_name:
            with DatabaseMapping(input_db_url) as db_map:
                _filters = db_map.get_filter_configs()
                if _filters:
                    scenario_name = name_from_dict(_filters[0])
        if not scenario_name:
            logging.error(
                "--decomposition lagrangian requires --scenario-name (the "
                "group filter needs to know which DB scenario to read)."
            )
            sys.exit(-1)
        regions_detected = discover_decomposition_regions_from_db(input_db_url)
        if len(regions_detected) < 2:
            logging.error(
                "--decomposition lagrangian needs at least two groups with "
                "decomposition_method='lagrangian_region' in the scenario; "
                "found %s.", regions_detected or '(none)',
            )
            sys.exit(-1)
        try:
            from flextool.engine_polars import run_chain_from_db
            from flextool.engine_polars._lagrangian import solve_lagrangian
            from flextool.engine_polars._solve_config import SolveConfig
            # Drive the native cascade to materialise the whole-system
            # ``FlexData`` for the Lagrangian coordinator.  The last
            # ``OrchestrationStep`` always retains ``flex_data`` (the
            # cascade's slim-step optimisation only clears earlier
            # steps); no need to set ``keep_solutions=True``.
            steps = run_chain_from_db(
                input_db_url,
                scenario_name,
                work_folder=wf,
            )
            if not steps:
                raise RuntimeError(
                    "Native cascade produced no solve steps; cannot "
                    "build FlexData for Lagrangian decomposition."
                )
            last_step = next(reversed(list(steps.values())))
            flex_data = last_step.flex_data
            if flex_data is None:
                raise RuntimeError(
                    "Last cascade step has no flex_data; cannot run "
                    "Lagrangian decomposition."
                )
            # Phase 3 — Lagrangian decomposition is HiGHS-only.  Read the
            # active solve's :class:`SolverConfig` and let
            # ``solve_lagrangian`` raise ``FlexToolUserError`` with an
            # actionable hint when the user picked a commercial solver.
            try:
                _sc = SolveConfig.load_from_db_url(
                    input_db_url, scenario_name,
                )
                _active_solve = next(
                    iter(_sc.model_solve.values()), None
                )
                _active_solve_name = (
                    _active_solve[0]
                    if _active_solve else scenario_name
                )
                _solver_cfg = _sc.solver_configs.get(_active_solve_name)
            except Exception:  # noqa: BLE001
                _solver_cfg = None
            # ``run_chain_from_db`` runs the cascade in-memory and does
            # NOT write workdir input CSVs (per its docstring).  The
            # decomposition_method dict can't be loaded from disk, so
            # pass the regions discovered from the DB directly.
            result = solve_lagrangian(
                flex_data,
                work_dir=wf,
                regions=regions_detected,
                alpha=args.lagrangian_alpha,
                max_iters=args.lagrangian_max_iter,
                tol=args.lagrangian_tolerance,
                solver_config=_solver_cfg,
            )
        except Exception as exc:
            logging.error("Lagrangian coordinator failed: %s", exc, exc_info=True)
            sys.exit(1)
        print(
            f"Lagrangian decomposition: converged={result.converged}, "
            f"iterations={result.iterations}, "
            f"total_objective={result.total_objective:.6g}"
        )
        for r, obj in result.region_objectives.items():
            print(f"  region {r}: {obj:.6g}")
        for pipe, lam in result.final_lambdas.items():
            print(f"  λ[{pipe}] = {lam:.6g}")
        sys.exit(0 if result.converged else 1)

    # Resolve scenario_name when omitted: pull it from the DB's
    # active filter.  ``run_chain_from_db`` accepts a None scenario
    # but downstream ``SolveConfig.load_from_db_url`` requires a
    # concrete name, so fix it up here.
    if not scenario_name:
        with DatabaseMapping(input_db_url) as db_map:
            _filters = db_map.get_filter_configs()
            if _filters:
                scenario_name = name_from_dict(_filters[0])

    # Header block — one aligned key/value pair per line, blank lines
    # before and after, so the user has a self-contained summary of
    # what's being run.
    _header_pairs = [
        ("Work dir", str(work_folder)),
        ("DB URL", str(input_db_url)),
        ("Scenario", scenario_name if scenario_name else "(unresolved)"),
        ("Output", str(args.output_location or output_path)),
    ]
    _header_keyw = max(len(k) for k, _ in _header_pairs) + 2
    print("")
    for _k, _v in _header_pairs:
        print(f"{(_k + ':').ljust(_header_keyw)}{_v}")
    # No trailing blank line here -- the "Available solvers:" log emits
    # its own trailing newline so the blank lands AFTER the licence line,
    # not before it.

    try:
        return_code, last_step = _run_solve(
            args, scenario_name, work_folder, timing_recorder,
        )
    except Exception as e:
        # FlexToolUserError signals a user-visible configuration problem
        # (unknown solver, missing license, model-level solver error).
        # The message is already human-readable; logging the traceback
        # on top is just noise.  Other exceptions get the full
        # traceback because they're (probably) bugs in flextool.
        try:
            from flextool.engine_polars._solver_dispatch import (
                FlexToolUserError,
            )
        except Exception:  # noqa: BLE001
            FlexToolUserError = ()  # type: ignore[assignment]
        if isinstance(e, FlexToolUserError):
            logging.error(str(e))
        else:
            logging.error(
                f"Native cascade failed: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
        sys.exit(1)

    # If successful and requested, write outputs
    output_subdir = args.output_subdir or scenario_name
    if return_code == 0:
        t_write_outputs = time.perf_counter()
        try:
            # Δ.31 — pass the last step's flex_data + solution so
            # write_outputs can build par/s in memory.  ``solve_name``
            # is the complete sub-solve identifier (e.g. ``y2025_5week``
            # for a roll, or just the scenario name for a single solve).
            wo_flex_data = last_step.flex_data if last_step else None
            wo_solution = last_step.solution if last_step else None
            wo_solve_name = (
                last_step.solve_name if last_step else None
            ) or scenario_name
            write_outputs(
                scenario_name=scenario_name,
                output_location=args.output_location,
                subdir=output_subdir,
                output_config_path=args.output_config,
                active_configs=args.active_configs,
                write_methods=args.write_methods,
                plot_rows=tuple(args.plot_rows) if args.plot_rows else None,
                settings_db_url=settings_db_url,
                fallback_output_location=str(output_path),
                raw_output_dir=str(wf / 'output_raw'),
                only_first_file=args.only_first_file_per_plot,
                timing_recorder=timing_recorder,
                flex_data=wo_flex_data,
                solution=wo_solution,
                solve_name=wo_solve_name,
            )
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
    # summary_solve.csv.  Mirror write_outputs's resolution of the output
    # location so the file lands in the same parent regardless of whether
    # output_location was supplied via the CLI / env / settings DB.
    try:
        _resolved_output_location = args.output_location or str(output_path) or ''
        _final_csv_dir = (
            Path(_resolved_output_location) / 'output_csv' / output_subdir
            if output_subdir else
            Path(_resolved_output_location) / 'output_csv'
        )
        timing_recorder.finalize(_final_csv_dir)
    except Exception as _exc:
        logging.warning("Failed to copy timings.csv to output dir: %s", _exc)

    # Write scenario information to output database if provided
    if args.output_db_url:
        # Check if database exists
        db_exists = os.path.exists(args.output_db_url.replace('sqlite:///', ''))

        with DatabaseMapping(args.output_db_url, create=not db_exists) as output_db:
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

            # Convert folder path to database representation
            value, type_ = to_database(str(output_path))

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
    main()
