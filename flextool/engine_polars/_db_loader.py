import logging
import sys
import spinedb_api as api
from spinedb_api import DatabaseMapping
from pathlib import Path

from flextool.engine_polars._db_reader import check_version
from flextool.input_derivation import run as _input_derivation_run
from flextool.engine_polars._solve_config import SolveConfig
from flextool.engine_polars._timeline import TimelineConfig
from flextool.engine_polars._solve_state import PathConfig, RunnerState, FlexToolError, FlexToolConfigError
from flextool.cli._timing import TimingRecorder


class FlexToolRunner:
    """Thin coordinator that builds RunnerState for the native cascade.

    Public API:
        write_input()  — reads DB, writes input/ CSV files (delegates to input_writer)

    See the engine_polars cascade modules for the full input-writing flow.
    """

    def __init__(self, input_db_url=None, output_path=None, scenario_name=None, flextool_dir=None, solver_config_dir=None, root_dir=None, work_folder=None, highs_threads=None, auto_scale=False, timing_recorder: "TimingRecorder | None" = None):
        try:
            logger = logging.getLogger(__name__)
            # Resolve work_folder: default to cwd for backward compatibility
            resolved_work_folder = Path(work_folder) if work_folder is not None else Path.cwd()
            # delete highs.log from previous run
            highs_log = resolved_work_folder / "HiGHS.log"
            if highs_log.exists():
                highs_log.unlink()
            # make a directory for solve data
            (resolved_work_folder / "solve_data").mkdir(exist_ok=True)
            # Build PathConfig — defaults are PyPI-friendly: the
            # ``flextool`` package directory for static data, CWD for
            # outputs and user-editable ``solver_config/highs.opt``.
            from flextool._resources import package_data_path
            _pkg_dir = package_data_path("")
            paths = PathConfig(
                flextool_dir=Path(flextool_dir) if flextool_dir is not None else _pkg_dir,
                solver_config_dir=Path(solver_config_dir) if solver_config_dir is not None else Path.cwd() / "solver_config",
                root_dir=Path(root_dir) if root_dir is not None else Path.cwd(),
                output_path=Path(output_path) if output_path is not None else Path.cwd(),
                work_folder=resolved_work_folder,
            )
            # read the data in
            # open connection to input db
            if scenario_name:
                scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
            with (DatabaseMapping(input_db_url) as db):
                if scenario_name:
                    api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
                else:
                    scen_names = db.get_scenario_items()
                    if len(scen_names) == 0:
                        message = "No scenario found"
                        logger.error(message)
                        raise FlexToolConfigError(message)
                    scenario_name=scen_names[0]['name']
                # The Work dir / DB URL / Scenario / Output header is
                # now printed once by ``cmd_run_flextool.main`` before
                # the cascade starts.  The previous multi-line INFO
                # here duplicated the Scenario line and split related
                # metadata across the startup log; drop it.
                if len(db.get_scenario_alternative_items(scenario_name=scenario_name)) == 0:
                    message = "No alternatives in the scenario, i.e. empty scenario."
                    logger.error(message)
                    raise FlexToolConfigError(message)

                # Pre-warm both entity and parameter_value caches so
                # every find_* call downstream (SolveConfig, TimelineConfig,
                # input_writer) hits memory.  Adding fetch_all("entity")
                # matches the spinedb-api docs' performance guidance and
                # mirrors the engine_polars _solve_config / _timeline
                # load_from_db_url paths.
                db.fetch_all("entity")
                db.fetch_all("parameter_value")
                check_version(db=db, logger=logger)
                # Solve-level fields — delegated to SolveConfig
                solve = SolveConfig.load_from_db(db=db, logger=logger)
                # Timeline-level fields — delegated to TimelineConfig
                timeline = TimelineConfig.load_from_db(db=db, logger=logger)

            # Post-DB initialization of timeline
            timeline.create_assumptive_parts(solve)
            timeline.create_timeline_from_timestep_duration(solve)

            # Assemble RunnerState — the single cross-cutting state container
            self.state = RunnerState(
                paths=paths, solve=solve, timeline=timeline, logger=logger,
            )
            # HiGHS thread count (CLI override; solver_runner defaults to 4 when None).
            self.state.highs_threads = highs_threads
            # Agent 8 (LP-scaling) — opt-in flag that lets the Python
            # ScaleAnalyzer enable row scaling.  Batch C.10 removed
            # the DB-stored ``use_row_scaling`` setting; the per-solve
            # row-scaling toggle is now driven by ``--scaling`` CLI +
            # this auto-scale flag (or the FLEXTOOL_FORCE_ROW_SCALING
            # env-var test hook).  Analysis itself runs unconditionally
            # (writes JSON); this flag only gates auto-application.
            self.state.auto_scale = auto_scale
            # Phase-timing recorder.  The CLI constructs one earlier and
            # passes it in; direct callers (tests) get a fresh recorder
            # bootstrapped here so timings.csv coverage is consistent
            # across both entry points.  The recorder always lives on
            # ``state.timing_recorder``.
            if timing_recorder is not None:
                self.state.timing_recorder = timing_recorder
                # Late-bind scenario when the CLI couldn't pass it in.
                if scenario_name and not timing_recorder.scenario:
                    timing_recorder.set_scenario(scenario_name)
            elif self.state.timing_recorder is None:
                self.state.timing_recorder = TimingRecorder(
                    work_folder=resolved_work_folder,
                    scenario=scenario_name,
                )
        except FlexToolError:
            sys.exit(-1)

    def write_input(
        self,
        input_db_url,
        scenario_name=None,
        precision_digits: int = 0,
        *,
        provider=None,
    ) -> None:
        """Write input/ CSVs to the runner's workdir.

        Used by the regional-decomposition wrapper and a handful of
        debug callers that want a freshly-staged ``input/`` directory.
        When *provider* is None an ephemeral
        :class:`flextool.engine_polars._flex_data_provider.FlexDataProvider`
        is constructed; the cascade itself constructs its own Provider in
        ``_drive_cascade``.
        """
        if provider is None:
            from flextool.engine_polars._flex_data_provider import FlexDataProvider
            provider = FlexDataProvider()
        _input_derivation_run(
            input_db_url,
            provider,
            self.state.logger,
            scenario_name=scenario_name,
            work_folder=self.state.paths.work_folder,
            precision_digits=precision_digits,
        )
        # Persist the cascade-input Provider on ``self.state`` so the
        # native cascade picks up the seeded ``input/<class>`` frames.
        # Mirrors the same handoff that
        # ``engine_polars._orchestration.run_orchestration`` performs.
        self.state.cascade_input_provider = provider
