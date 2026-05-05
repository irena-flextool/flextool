import argparse
import sys
import logging
import traceback
from typing import Callable
from functools import wraps
from pathlib import Path
from datetime import datetime
import time
import os
from flextool.process_outputs.result_writer import write_outputs
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.flextoolrunner.timing_recorder import TimingRecorder
from flextool.flextoolrunner.precision import (
    report_near_duplicates,
    resolve_precision_digits,
    resolve_report_near_duplicates,
)
from flextool.flextoolrunner.scaling import resolve_auto_scale
from flextool.flextoolrunner.solver_runner import (
    resolve_ipm,
    resolve_relax_feasibility,
)
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


# Δ.14 — engine-selection helper.  Pulled out of ``main`` for direct
# unit testing (precedence semantics: explicit CLI flag > env var >
# default).  Pure function: no I/O, no side effects.
def _resolve_engine(cli_value, env_value, default='gmpl'):
    """Return ``'native'`` or ``'gmpl'`` per Δ.14 precedence rules.

    Parameters
    ----------
    cli_value : str | None
        The literal value of ``--engine`` from argparse: ``'native'``,
        ``'gmpl'``, or ``None`` (flag absent).
    env_value : str | None
        ``os.environ.get('FLEXPY_USE_NATIVE_ORCHESTRATION')`` — a bare
        ``None`` if unset.  Truthy values (``'1'``, ``'true'``,
        ``'yes'``, ``'on'``, case-insensitive after ``strip()``) select
        ``'native'``; everything else (including the empty string and
        ``'0'``) keeps the default.
    default : str, optional
        Fallback when neither the flag nor the env var force a choice.
        Δ.14 ships with ``'gmpl'`` so existing GUI / Toolbox subprocess
        invocations don't accidentally swap their solver backend on
        upgrade.

    Returns
    -------
    str
        One of ``'native'`` / ``'gmpl'``.

    Notes
    -----
    The env-var truth-table mirrors
    :func:`flextool.engine_polars.chain.run_chain`'s feature flag (Γ.8.D)
    so a single env-var setting drives both the CLI dispatcher and any
    direct ``run_chain`` callers (e.g. test fixtures) consistently.
    """
    if cli_value in ('native', 'gmpl'):
        return cli_value
    if env_value is not None:
        normalised = env_value.strip().lower()
        if normalised in ('1', 'true', 'yes', 'on'):
            return 'native'
    return default


def _warn_dropped_native_flags(args):
    """Δ.14 — log a single warning enumerating GMPL-only flags that
    the user passed alongside ``--engine=native``.

    Pulled out of :func:`_run_native_solve` for direct testing — the
    enumeration is the only behaviour worth pinning, the rest of
    ``_run_native_solve`` is straight delegation.
    """
    gmpl_only_flags = {
        '--use-old-raw-csv': args.use_old_raw_csv,
        '--ipm': args.ipm,
        '--auto-scale': args.auto_scale,
        '--relax-feasibility': args.relax_feasibility is not None,
        '--glpsol-timing': args.glpsol_timing,
        '--highs-threads (>1)': (args.highs_threads or 1) > 1,
        '--precision-digits': args.precision_digits is not None,
        '--report-near-duplicates': args.report_near_duplicates,
    }
    dropped = [name for name, active in gmpl_only_flags.items() if active]
    if dropped:
        logging.warning(
            "engine=native: ignoring GMPL-only flag(s) %s — these "
            "configure the glpsol/HiGHS legacy pipeline and have no "
            "effect on the polar-high-opt cascade.  Re-run with "
            "--engine=gmpl if you need them.",
            ', '.join(dropped),
        )
    return dropped


def _run_native_solve(args, scenario_name, work_folder, timing_recorder):
    """Δ.14 — Native (flexpy / polar-high-opt) cascade.

    Returns ``0`` on success, ``1`` on any non-optimal sub-solve.
    Output-tree emission (``write_outputs``, ``timings.csv`` finalize,
    output-DB updates) is shared with the GMPL path back in
    :func:`main` — this helper only drives the solve loop.

    Several legacy CLI flags are GMPL-pipeline-specific and no-op
    under the native path; :func:`_warn_dropped_native_flags` logs
    their names.  Aligned with the audit's "fail loudly when native
    can't satisfy a CLI flag" policy: a warning is loud enough for
    diagnosis without breaking GUI-driven invocations that pass a
    default-set of flags.
    """
    from flextool.engine_polars import run_chain_from_db

    _warn_dropped_native_flags(args)

    print(f'Scenario: {scenario_name}')
    if scenario_name:
        timing_recorder.set_scenario(scenario_name)

    # Drive the native cascade end-to-end.  ``run_chain_from_db``
    # handles flextool's preprocessing (write_input) AND the per-solve
    # LP build+solve+handoff loop in-process.
    t_solve_start = time.perf_counter()
    steps = run_chain_from_db(
        args.input_db_url,
        scenario_name,
        work_folder=work_folder,
    )
    all_solves_seconds = time.perf_counter() - t_solve_start
    print("--- All Flextool solves time %.4s seconds ---" % all_solves_seconds)
    timing_recorder.record('all_solves', seconds=all_solves_seconds,
                           t_start=t_solve_start)

    if not steps:
        logging.error("Native cascade produced no solve steps; aborting.")
        return 1
    # Non-optimal in any sub-solve → infeasible/unbounded exit code.
    for name, step in steps.items():
        if step.solution is None or not step.solution.optimal:
            logging.error(
                "Native cascade: solve %r non-optimal (status=%r); "
                "exit=1 (infeasible/unbounded).",
                name,
                getattr(step.solution, "status", None) if step.solution else None,
            )
            return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Run flextool using the specified database URL. Return codes are 0: success, 1: infeasible or unbounded, -1: failure."
    parser.add_argument('input_db_url', help='Database URL to connect to (can be copied from Toolbox workflow db item')
    parser.add_argument('output_db_url', metavar='DB_URL', nargs='?', default=None, help='Save information about result location to database for post-processing')
    parser.add_argument('--settings-db-url', help='Settings for post-processing')
    parser.add_argument('--scenario-name', help='Name for the scenario in the database that should be executed', nargs='?', default=None)
    parser.add_argument('--debug', action='store_true')
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
    parser.add_argument('--flextool-location', default='template/flextool_location.txt',
                        help='When running in Spine Toolbox, this argument provides the location of FlexTool so outputs can be directed there (instead of work directories).')
    parser.add_argument('--work-folder', metavar='PATH', default=None,
                        help='Working directory for intermediate files (default: current directory). '
                             'Enables parallel scenario execution by isolating each run.')
    parser.add_argument('--only-first-file-per-plot', action='store_true', default=False,
                        help='Only produce the first file for each plot (quick overview mode)')
    parser.add_argument('--use-old-raw-csv', action='store_true', default=False,
                        help='Keep only the legacy glpsol-driven output_raw/*.csv pathway; '
                             'skip the HiGHS → parquet extractor.')
    parser.add_argument('--highs-threads', metavar='N', type=int, default=1,
                        help='Number of threads HiGHS may use for the MIP / LP solve (default: 1). '
                             'Serial is the reliable default because HiGHS PAMI (parallel dual '
                             'simplex) can stall indefinitely on degenerate LPs that reappear with '
                             'tiny post-optimality residuals — observed across HiGHS 1.11 / 1.12 / '
                             '1.14 on rivendell hydro-cascade and UC scenarios. Raise this only on '
                             'machines with spare cores AND after confirming PAMI is actually '
                             'faster on your specific model; be ready to drop back to 1 if stalls '
                             'resurface.')
    parser.add_argument('--precision-digits', metavar='N', type=int, default=None,
                        help='Round every numeric input parameter to N significant '
                             'figures before writing CSVs (typical: 10).  '
                             'Collapses accumulated float-noise so HiGHS '
                             'mip_detect_symmetry can aggregate structurally-identical '
                             'coefficients.  0 or unset disables rounding (default).  '
                             'Overrides the FLEXTOOL_PRECISION_DIGITS env var.')
    parser.add_argument('--report-near-duplicates', action='store_true', default=False,
                        help='After writing input CSVs, scan each parameter column '
                             'for clusters of nearly-equal values (per-parameter, '
                             'rel_tol=1e-6).  Diagnostic only — never fails the run.  '
                             'Also triggered by FLEXTOOL_REPORT_NEAR_DUPS=1.')
    parser.add_argument('--relax-feasibility', nargs='?', const='default', default=None,
                        metavar='TOL',
                        help='Loosen HiGHS primal + dual feasibility tolerance. '
                             'Without a value, uses 1e-5 (two orders of magnitude looser '
                             'than HiGHS default 1e-7) — enough to absorb sub-tolerance '
                             'residuals on wide-bound models (rivendell S19 dual-simplex '
                             'stall) without being irresponsibly loose.  Pass '
                             '``--relax-feasibility=1e-4`` etc. to set an explicit '
                             'tolerance.  Also triggered by FLEXTOOL_RELAX_FEASIBILITY '
                             '(empty / truthy -> default; numeric -> explicit).')
    parser.add_argument('--ipm', action='store_true', default=False,
                        help='Switch HiGHS to interior-point solver.  IPM has no basis, '
                             'so it cannot stall on Markowitz-pivot degeneracy the way '
                             'dual simplex does on rivendell S19.  For MIPs the LP '
                             'relaxation uses IPM; branch-and-bound still drives the '
                             'integer search.  Also triggered by FLEXTOOL_IPM=1.')
    parser.add_argument('--region', metavar='GROUP_NAME', default=None,
                        help='Produce a filtered per-region input directory '
                             '``input_region_<GROUP_NAME>/`` for Lagrangian '
                             'decomposition (Agent 3.1).  The group must have '
                             '``decomposition_method=lagrangian_region`` in '
                             'the DB.  Cross-region processes are replaced '
                             'with import/export half-flows; the coupling '
                             'variables are listed in '
                             '``solve_data/region_coupling.csv``.  When this '
                             'flag is set, GMPL is NOT invoked — this is the '
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
    parser.add_argument('--glpsol-timing', action='store_true', default=False,
                        help='Record per-constraint matrix-generation time '
                             'from glpsol stdout. Writes '
                             'solve_data/glpsol_constraint_timing.csv with '
                             'columns solve,phase,constraint,elapsed_s. Use '
                             'to identify which constraint families dominate '
                             'MPS generation cost. Diagnostic only.')
    parser.add_argument('--auto-scale', action='store_true', default=False,
                        help='Apply the per-solve ScaleAnalyzer recommendation '
                             'for use_row_scaling (Agent 8, LP-scaling).  Without '
                             'this flag the analyzer still runs and emits '
                             'solve_data/scaling_analysis.json, but the row-scaling '
                             'recommendation is not acted on — an explicit opt-in '
                             'is required.  The objective scalar '
                             '(scale_the_objective) IS auto-applied unconditionally '
                             '(Agent 12); its value comes from the same analyser '
                             'and lives in solve_data/scale_the_objective.csv.  '
                             'The flag is also triggered by FLEXTOOL_AUTO_SCALE=1.  '
                             'A DB-supplied solve.use_row_scaling is always '
                             'respected; the analyzer only fills in missing values.')
    parser.add_argument('--engine',
                        choices=['gmpl', 'native'],
                        default=None,
                        help='Solver-orchestration backend (Δ.14).  '
                             '``gmpl`` (default) runs the legacy '
                             'glpsol/HiGHS pipeline through '
                             '``FlexToolRunner.run_model`` — the path the '
                             'GUI / Toolbox subprocess invocations have '
                             'always used.  ``native`` runs the in-process '
                             'flexpy/polar-high-opt cascade via '
                             '``flextool.engine_polars.run_chain_from_db``: '
                             'flextool''s preprocessing still emits the '
                             'snapshot CSVs, but each per-solve LP is built '
                             'and solved inside this Python process and '
                             'handoff state flows in-memory between sub-'
                             'solves (no glpsol invocation).  Precedence: '
                             '(1) explicit ``--engine=...`` flag, '
                             '(2) ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env '
                             'var (``1`` / ``true`` / ``yes`` / ``on`` → '
                             'native), (3) default ``gmpl``.  Several '
                             'GMPL-only flags no-op under '
                             '``--engine=native`` — see the dispatch close '
                             'stanza for the full list (e.g. ``--ipm``, '
                             '``--auto-scale``, ``--relax-feasibility``, '
                             '``--use-old-raw-csv``, ``--glpsol-timing``).')

    args = parser.parse_args()
    input_db_url = args.input_db_url
    settings_db_url = args.settings_db_url
    scenario_name = args.scenario_name
    DEBUG = args.debug
    output_path = Path(args.flextool_location).resolve().parent.parent
    work_folder = Path(args.work_folder) if args.work_folder else None
    if work_folder is not None:
        work_folder.mkdir(parents=True, exist_ok=True)
    wf = work_folder if work_folder is not None else Path.cwd()

    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Self-heal missing lightweight settings DBs so fresh clones don't
    # fail opaquely when the user forgot to run `flextool-update`. Only
    # seeds output_info / output_settings / comparison_settings by
    # basename; other paths are left untouched.
    _repo_root = Path(__file__).resolve().parent.parent.parent
    for _candidate in (args.output_db_url, args.settings_db_url):
        try:
            ensure_settings_db(_candidate, _repo_root)
        except Exception as _exc:
            logging.warning("Failed to auto-seed %s: %s", _candidate, _exc)

    # Phase-timing recorder: constructed once per CLI invocation, lives
    # on ``runner.state.timing_recorder``, writes a structured timings.csv
    # at <work_folder>/solve_data/timings.csv (one row per phase, atomic
    # append style so a crash mid-run still leaves usable data).
    # Replaces the legacy two ``solve_progress.csv`` files.
    timing_recorder = TimingRecorder(work_folder=wf, scenario=scenario_name)
    t_total_start = time.perf_counter()
    timer = []
    timer.append(time.perf_counter())

    effective_precision = resolve_precision_digits(args.precision_digits)
    run_near_dup_report = resolve_report_near_duplicates(args.report_near_duplicates)
    auto_scale = resolve_auto_scale(args.auto_scale)
    relax_feasibility = resolve_relax_feasibility(args.relax_feasibility)
    use_ipm = resolve_ipm(args.ipm)

    # --- Regional filter mode (Agent 3.1) --------------------------------
    # ``--region GROUP`` produces ``input_region_<GROUP>/`` and exits
    # without invoking the solver.  The Lagrangian coordinator (Agent
    # 3.2) then orchestrates multiple region solves itself.
    if args.region:
        from flextool.flextoolrunner import input_writer as _input_writer
        _region_output = wf / f"input_region_{args.region}"
        try:
            result = _input_writer.write_input_for_region(
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

    # --- Lagrangian decomposition mode (Agent 3.2) ----------------------
    # ``--decomposition lagrangian`` drives the spatial Lagrangian
    # coordinator instead of the monolithic orchestrator.  Requires the
    # scenario to declare ≥ 2 decomposition-region groups; we bail out
    # with a clear error if that precondition is unmet.
    if args.decomposition == 'lagrangian':
        from flextool.flextoolrunner.lagrangian import run_lagrangian
        from flextool.flextoolrunner import region_filter as _region_filter
        if not scenario_name:
            logging.error(
                "--decomposition lagrangian requires --scenario-name (the "
                "group filter needs to know which DB scenario to read)."
            )
            sys.exit(-1)
        regions_detected = _region_filter.discover_decomposition_regions_from_db(input_db_url)
        if len(regions_detected) < 2:
            logging.error(
                "--decomposition lagrangian needs at least two groups with "
                "decomposition_method='lagrangian_region' in the scenario; "
                "found %s.", regions_detected or '(none)',
            )
            sys.exit(-1)
        try:
            lag_logger = logging.getLogger("flextool.lagrangian")
            result = run_lagrangian(
                db_url=input_db_url,
                scenario=scenario_name,
                alpha=args.lagrangian_alpha,
                max_iterations=args.lagrangian_max_iter,
                tolerance=args.lagrangian_tolerance,
                work_folder=work_folder,
                logger=lag_logger,
                precision_digits=effective_precision,
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

    # --- Δ.14: dispatch on --engine ----------------------------------------
    # Resolve the orchestration backend once.  ``_resolve_engine``
    # encodes the precedence (CLI flag > env var > default 'gmpl').
    # ``--engine=native`` skips the legacy ``FlexToolRunner`` /
    # ``run_model`` path entirely and runs the polar-high-opt cascade
    # via ``run_chain_from_db`` instead; ``--engine=gmpl`` (the default)
    # preserves the historical behaviour byte-for-byte for backward
    # compatibility with the Toolbox / GUI subprocess invocations.
    engine = _resolve_engine(
        args.engine,
        os.environ.get('FLEXPY_USE_NATIVE_ORCHESTRATION'),
    )
    native_engine = (engine == 'native')
    if engine == 'native':
        # Resolve scenario_name when omitted: pull it from the DB's
        # active filter (mirrors the GMPL else-branch below).  The
        # native path's ``run_chain_from_db`` accepts a None scenario
        # but downstream ``SolveConfig.load_from_db_url`` requires a
        # concrete name, so fix it up here.
        if not scenario_name:
            with DatabaseMapping(input_db_url) as db_map:
                _filters = db_map.get_filter_configs()
                if _filters:
                    scenario_name = name_from_dict(_filters[0])
        try:
            return_code = _run_native_solve(
                args, scenario_name, work_folder, timing_recorder,
            )
        except Exception as e:
            logging.error(
                f"Native cascade failed: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}"
            )
            sys.exit(1)
    else:
        if scenario_name:
            runner = FlexToolRunner(input_db_url, output_path, scenario_name, work_folder=work_folder, use_old_raw_csv=args.use_old_raw_csv, highs_threads=args.highs_threads, auto_scale=auto_scale, relax_feasibility=relax_feasibility, use_ipm=use_ipm, glpsol_timing=args.glpsol_timing, timing_recorder=timing_recorder)
            timer.insert(0, time.perf_counter())
            init_seconds = timer[0] - timer[1]
            print("--- Init time %.4s seconds ---" % init_seconds)
            timing_recorder.record('cli_init', seconds=init_seconds)
            t_write_input = time.perf_counter()
            runner.write_input(input_db_url, scenario_name, precision_digits=effective_precision)
            timer.insert(0, time.perf_counter())
            write_seconds = timer[0] - timer[1]
            print("--- Write time %.4s seconds ---" % write_seconds)
            timing_recorder.record('write_input', subphase='per_scenario',
                                   seconds=write_seconds, t_start=t_write_input)

        else:
            runner = FlexToolRunner(input_db_url, output_path, work_folder=work_folder, use_old_raw_csv=args.use_old_raw_csv, highs_threads=args.highs_threads, auto_scale=auto_scale, relax_feasibility=relax_feasibility, use_ipm=use_ipm, glpsol_timing=args.glpsol_timing, timing_recorder=timing_recorder)
            timer.insert(0, time.perf_counter())
            init_seconds = timer[0] - timer[1]
            print("--- Init time %.4s seconds ---" % init_seconds)
            timing_recorder.record('cli_init', seconds=init_seconds)
            t_write_input = time.perf_counter()
            runner.write_input(input_db_url, precision_digits=effective_precision)
            timer.insert(0, time.perf_counter())
            write_seconds = timer[0] - timer[1]
            print("--- Write time %.4s seconds ---" % write_seconds)
            timing_recorder.record('write_input', subphase='all',
                                   seconds=write_seconds, t_start=t_write_input)
            with DatabaseMapping(input_db_url) as db_map:
                scenario_name = name_from_dict(db_map.get_filter_configs()[0])
            timing_recorder.set_scenario(scenario_name)

        # Diagnostic: cluster near-duplicate numeric parameter values.  Opt-in
        # via --report-near-duplicates or FLEXTOOL_REPORT_NEAR_DUPS=1; silent
        # by default.  Never fails the run.
        if run_near_dup_report:
            try:
                report_near_duplicates(wf / "input")
            except Exception as exc:
                print(f"[precision] near-duplicate report failed: {exc}")

        print(f'Scenario: {scenario_name}')
        t_solve_start = time.perf_counter()
        try:
            return_code = runner.run_model()
            timer.insert(0, time.perf_counter())
            all_solves_seconds = timer[0] - timer[1]
            print("--- All Flextool solves time %.4s seconds ---" % all_solves_seconds)
            timing_recorder.record('all_solves', seconds=all_solves_seconds,
                                   t_start=t_solve_start)
        except Exception as e:
            logging.error(f"Model run failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            sys.exit(1)

    # If successful and requested, write outputs
    output_subdir = args.output_subdir or scenario_name
    if return_code == 0:
        t_write_outputs = time.perf_counter()
        try:
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
            )
        except FileNotFoundError as exc:
            # Δ.14 — known gap on the native engine.  ``write_outputs``
            # consumes wide-format ``solve_data/p_<entity>.csv`` files
            # (``p_node.csv``, ``p_process_sink.csv``, …) that the GMPL
            # phase-1 dump emits.  The native cascade doesn't run that
            # dump — it produces ``output_raw/`` alone.  Closing this
            # gap properly is Δ.15 scope (port the wide-format dump to
            # ``engine_polars/_dump_csvs.py`` or refactor
            # ``read_parameters`` to read from ``input/`` directly).
            # For now: when running native AND ``write_outputs`` fails
            # on a missing solve_data CSV, log a clear warning and
            # carry on so ``output_raw/`` (the only artefact the
            # cascade currently produces) lands cleanly and the CLI
            # exits 0.  Under GMPL the same exception is fatal — this
            # tolerance is gated on ``native_engine``.
            if native_engine:
                logging.warning(
                    "engine=native: write_outputs failed (%s).  This "
                    "is a known Δ.15 gap — the native cascade does not "
                    "yet emit the wide-format solve_data CSVs that "
                    "process_outputs.read_parameters consumes.  "
                    "output_raw/ artefacts ARE produced; output_csv/, "
                    "output_parquet/, output_excel/, output_plots/ are "
                    "skipped on this run.",
                    exc,
                )
            else:
                raise
        timer.insert(0, time.perf_counter())
        timing_recorder.record('write_outputs', subphase='total',
                               seconds=time.perf_counter() - t_write_outputs,
                               t_start=t_write_outputs)

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

def debug_only(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            return func(*args, **kwargs)

    return wrapper

if __name__ == '__main__':
    main()
