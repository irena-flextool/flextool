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

    if scenario_name:
        runner = FlexToolRunner(input_db_url, output_path, scenario_name, work_folder=work_folder, use_old_raw_csv=args.use_old_raw_csv, highs_threads=args.highs_threads, auto_scale=auto_scale, relax_feasibility=relax_feasibility, use_ipm=use_ipm, glpsol_timing=args.glpsol_timing)
        timer.insert(0, time.perf_counter())
        print("--- Init time %.4s seconds ---" % (timer[0] - timer[1]))
        with open(wf / "solve_data/solve_progress.csv", "w") as solve_progress:
            solve_progress.write('scenario,' + scenario_name + '\n')
            solve_progress.write('Init time,' + str(round(timer[0] - timer[1],4)) + '\n')
        runner.write_input(input_db_url, scenario_name, precision_digits=effective_precision)
        timer.insert(0, time.perf_counter())
        print("--- Write time %.4s seconds ---" % (timer[0] - timer[1]))
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('Write input time,' + str(round(timer[0] - timer[1],4)) + '\n')

    else:
        runner = FlexToolRunner(input_db_url, output_path, work_folder=work_folder, use_old_raw_csv=args.use_old_raw_csv, highs_threads=args.highs_threads, auto_scale=auto_scale, relax_feasibility=relax_feasibility, use_ipm=use_ipm, glpsol_timing=args.glpsol_timing)
        timer.insert(0, time.perf_counter())
        print("--- Init time %.4s seconds ---" % (timer[0] - timer[1]))
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('scenario,unknown\n')
            solve_progress.write('Init time,' + str(round(timer[0] - timer[1],4)) + '\n')
        runner.write_input(input_db_url, precision_digits=effective_precision)
        timer.insert(0, time.perf_counter())
        print("--- Write time %.4s seconds ---" % (timer[0] - timer[1]))
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('Write all input time,' + str(round(timer[0] - timer[1],4)) + '\n')
        with DatabaseMapping(input_db_url) as db_map:
            scenario_name = name_from_dict(db_map.get_filter_configs()[0])

    # Diagnostic: cluster near-duplicate numeric parameter values.  Opt-in
    # via --report-near-duplicates or FLEXTOOL_REPORT_NEAR_DUPS=1; silent
    # by default.  Never fails the run.
    if run_near_dup_report:
        try:
            report_near_duplicates(wf / "input")
        except Exception as exc:
            print(f"[precision] near-duplicate report failed: {exc}")

    print(f'Scenario: {scenario_name}')
    try:
        return_code = runner.run_model()
        timer.insert(0, time.perf_counter())
        print("--- All Flextool solves time %.4s seconds ---" % (timer[0] - timer[1]))
        with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('All Flextool solves,' + str(round(timer[0] - timer[1],4)) + '\n')
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)
    
    # If successful and requested, write outputs
    if return_code == 0:
        output_subdir = args.output_subdir or scenario_name
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
        )
        timer.insert(0, time.perf_counter())
    
    print("\n--- Full execution time %.4s seconds ---------------------------------------" % (timer[0] - timer[-1]))
    print("--------------------------------------------------------------------------\n")
    with open(wf / "solve_data/solve_progress.csv", "a") as solve_progress:
        solve_progress.write('Full execution time,' + str(round(timer[0] - timer[-1],4)) + '\n')

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
