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
from flextool.flextoolrunner.timing_recorder import TimingRecorder
from flextool.flextoolrunner.precision import resolve_precision_digits
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


# Δ.21 — engine-selection helper.  GMPL retired; only ``'native'`` is
# accepted.  ``--engine=gmpl`` is rejected with a clear error message;
# ``FLEXPY_USE_NATIVE_ORCHESTRATION`` is now vestigial (native is the
# only path) and emits a deprecation warning when set.
#
# Pure function: no I/O, no side effects.  Exposed for direct unit
# testing — the CLI calls it from ``main`` to validate any explicit
# ``--engine`` value the user passed.
_ENGINE_RETIRED_GMPL_MESSAGE = (
    "GMPL path retired in Δ.21. Use --engine=native (default)."
)


def _resolve_engine(cli_value, env_value, default='native'):
    """Return ``'native'`` per Δ.21.

    The function preserves the old (cli_value, env_value, default)
    signature for test compatibility but the only valid runtime value
    is ``'native'``.  Any explicit ``--engine=gmpl`` invocation raises
    :class:`SystemExit` with the retirement message.

    The ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env var is now vestigial:
    truthy values (``'1'`` / ``'true'`` / ``'yes'`` / ``'on'``,
    case-insensitive after ``strip()``) emit a deprecation warning
    once but otherwise keep the default behaviour.  Falsy or unset
    values are silent.

    Parameters
    ----------
    cli_value : str | None
        The literal value of ``--engine`` from argparse: ``'native'``
        or ``None`` (flag absent).  ``'gmpl'`` triggers a hard exit.
    env_value : str | None
        ``os.environ.get('FLEXPY_USE_NATIVE_ORCHESTRATION')`` — a bare
        ``None`` if unset.
    default : str, optional
        Reserved kwarg kept for API stability with Δ.14.  Always
        returns ``'native'`` regardless of value (the legacy CLI no
        longer offers any other engine).

    Returns
    -------
    str
        Always ``'native'``.
    """
    if cli_value == 'gmpl':
        # Hard error — the legacy GMPL path has been retired entirely.
        # The CLI calls ``sys.exit(2)`` with the retirement banner so
        # automation / GUI invocations get a readable diagnostic.
        print(
            f"error: --engine=gmpl: {_ENGINE_RETIRED_GMPL_MESSAGE}",
            file=sys.stderr,
        )
        sys.exit(2)
    if cli_value is not None and cli_value != 'native':
        # argparse's ``choices=`` should forbid this branch in normal
        # use.  Defensive guard for direct callers only.
        print(
            f"error: --engine={cli_value!r} is not recognised; "
            "the only supported value is 'native'.",
            file=sys.stderr,
        )
        sys.exit(2)
    if env_value is not None:
        normalised = env_value.strip().lower()
        if normalised in ('1', 'true', 'yes', 'on'):
            logging.warning(
                "FLEXPY_USE_NATIVE_ORCHESTRATION is deprecated as of "
                "Δ.21: native is now the only engine. Unset the env "
                "var to silence this warning."
            )
    # ``default`` is honoured for API stability; the live CLI always
    # passes the new default.
    return default if default == 'native' else 'native'


def _run_native_solve(args, scenario_name, work_folder, timing_recorder):
    """Δ.21 — drive the native (flexpy / polar-high) cascade.

    Returns a tuple ``(return_code, last_step)`` where ``last_step``
    is the :class:`flextool.engine_polars.OrchestrationStep` of the
    final (or only) sub-solve, used by the caller to thread
    ``flex_data`` + ``solution`` into ``write_outputs`` for the
    in-memory parameter / set namespace path (Δ.31).

    Δ.25: when ``--fast-single-solve`` is passed, dispatch to the
    surgical single-solve path that bypasses
    ``write_input``/``run_chain_from_db`` entirely.  Experimental.
    """
    print(f'Scenario: {scenario_name}')
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
    parser.add_argument('--engine',
                        choices=['gmpl', 'native'],
                        default=None,
                        help='Solver-orchestration backend.  Δ.21 retired '
                             'the legacy GMPL/glpsol pipeline; the only '
                             'supported value is ``native`` (the default), '
                             'which runs the in-process flexpy/polar-high '
                             'cascade via '
                             '``flextool.engine_polars.run_chain_from_db``.  '
                             'Passing ``--engine=gmpl`` is rejected with a '
                             'clear error.  '
                             '``FLEXPY_USE_NATIVE_ORCHESTRATION`` is '
                             'vestigial (no-op) but emits a deprecation '
                             'warning when set truthy.')
    parser.add_argument('--highs-threads', type=int, default=1,
                        help='Number of HiGHS solver threads.  Accepted '
                             'for GUI/Toolbox subprocess compatibility; '
                             'currently a no-op on the native path until '
                             'thread-count plumbing reaches '
                             '``polar_high.Problem.solve``.  Default 1.')
    parser.add_argument('--csv-dump', action='store_true',
                        default=False,
                        help='Phase E-c — opt-in debug visibility for '
                             'cascade-internal CSV emission.  When set, '
                             'the cascade emits CSVs for input/, '
                             'solve_data/, and cross_solve/ as in the '
                             'legacy (pre-Phase-E-c) behaviour.  Default '
                             'is off: cascade runs purely in-memory, '
                             'with output_raw/*.parquet and '
                             'output_processed/* (per --write-methods) '
                             'as the only on-disk artefacts.  Use for '
                             'debugging the engine_polars writer port; '
                             'do not rely on these files from '
                             'downstream tooling.')
    parser.add_argument('--fast-single-solve', action='store_true',
                        default=False,
                        help='Δ.25 (EXPERIMENTAL) — bypass '
                             '``flextool.flextoolrunner.input_writer.'
                             'write_input`` entirely.  Reads inputs '
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
    input_db_url = args.input_db_url
    settings_db_url = args.settings_db_url
    scenario_name = args.scenario_name
    DEBUG = args.debug
    output_path = Path(args.flextool_location).resolve().parent.parent
    work_folder = Path(args.work_folder) if args.work_folder else Path.cwd()
    work_folder.mkdir(parents=True, exist_ok=True)
    wf = work_folder

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

    # ``effective_precision`` is the only GMPL-era knob still consumed
    # on the native path — by ``--region`` (input filter).  The
    # GMPL-pipeline-only flags
    # (``--use-old-raw-csv``, ``--ipm``, ``--auto-scale``,
    # ``--relax-feasibility``, ``--glpsol-timing``,
    # ``--report-near-duplicates``, and ``--highs-threads``) were
    # warn-deprecated in Δ.21 and removed entirely in Δ.22.
    effective_precision = resolve_precision_digits(args.precision_digits)

    # --- Regional filter mode (Agent 3.1) --------------------------------
    # ``--region GROUP`` produces ``input_region_<GROUP>/`` and exits
    # without invoking the solver.  The Lagrangian coordinator (Agent
    # 3.2) then orchestrates multiple region solves itself.
    if args.region:
        from flextool.flextoolrunner.region_decomposition import (
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
        from flextool.flextoolrunner.region_filter import (
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

    # --- Δ.21: native is now the only engine -------------------------------
    # ``_resolve_engine`` validates the ``--engine`` value (rejects
    # ``gmpl`` with sys.exit(2)) and emits a deprecation warning when
    # the legacy ``FLEXPY_USE_NATIVE_ORCHESTRATION`` env var is set.
    # The legacy GMPL/glpsol dispatch branch was removed in Δ.21; the
    # CLI always runs the polar-high cascade via
    # ``run_chain_from_db``.  ``--engine=native`` is preserved as a
    # no-op for backward compatibility with one release-cycle of
    # automation; passing nothing has the same effect.
    _resolve_engine(
        args.engine,
        os.environ.get('FLEXPY_USE_NATIVE_ORCHESTRATION'),
    )
    # Resolve scenario_name when omitted: pull it from the DB's
    # active filter.  ``run_chain_from_db`` accepts a None scenario
    # but downstream ``SolveConfig.load_from_db_url`` requires a
    # concrete name, so fix it up here.
    if not scenario_name:
        with DatabaseMapping(input_db_url) as db_map:
            _filters = db_map.get_filter_configs()
            if _filters:
                scenario_name = name_from_dict(_filters[0])
    try:
        return_code, last_step = _run_native_solve(
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
            # Δ.31: the in-memory parameter / set path doesn't read
            # ``solve_data/`` CSVs anymore, but ``read_variables`` still
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
