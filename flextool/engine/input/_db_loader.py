import logging
from contextlib import ExitStack
import spinedb_api as api
from flextool import run_record
from flextool.update_flextool.db_handle import open_database_mapping
from pathlib import Path

from flextool.engine.input._db_reader import check_version, schema_version_message
from flextool.input_derivation import run as _input_derivation_run
from flextool.engine.input._solve_config import SolveConfig
from flextool.engine.input._timeline import TimelineConfig
from flextool.engine.input._solve_state import PathConfig, RunnerState, FlexToolConfigError
from flextool.cli._timing import TimingRecorder
from flextool.run_options import RunOptions


class FlexToolRunner:
    """Thin coordinator that builds RunnerState for the native cascade.

    Public API:
        write_input()  — reads DB, writes input/ CSV files (delegates to input_writer)

    See the engine cascade modules for the full input-writing flow.

    A database that cannot be run (no scenario, an empty scenario, a version older than the tool) raises
    :class:`FlexToolConfigError` with the reason; the CLI logs it and records it as the run's error.
    """

    def __init__(
        self,
        input_db_url=None,
        output_path=None,
        scenario_name=None,
        flextool_dir=None,
        solver_config_dir=None,
        root_dir=None,
        work_folder=None,
        highs_threads=None,
        auto_scale=False,
        timing_recorder: "TimingRecorder | None" = None,
        run_options: "RunOptions | None" = None,
        timeset_warned: "set[tuple] | None" = None,
    ):
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
        # outputs; the solver option folder as a run resolves it
        # (``flextool._solver_config_source``; ``run_chain_from_db`` passes its own).
        from flextool._resources import package_data_path
        from flextool._solver_config_source import read_solver_option_files, resolve_solver_config_source
        _pkg_dir = package_data_path("")
        solver_option_files = None
        if solver_config_dir is None:
            # Read the files here, so a folder found outside a workspace this computer set up keeps the
            # database's rules (item 240 B3; ``run_chain_from_db`` hands its own files to the solves).
            _source = resolve_solver_config_source(run_options=run_options, input_db_url=input_db_url)
            solver_config_dir = _source.dir
            solver_option_files = read_solver_option_files(_source)
        paths = PathConfig(
            flextool_dir=Path(flextool_dir) if flextool_dir is not None else _pkg_dir,
            solver_config_dir=Path(solver_config_dir) if solver_config_dir is not None else None,
            root_dir=Path(root_dir) if root_dir is not None else Path.cwd(),
            output_path=Path(output_path) if output_path is not None else Path.cwd(),
            work_folder=resolved_work_folder,
            solver_option_files=solver_option_files,
        )
        # read the data in
        # open connection to input db
        if scenario_name:
            scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
        with ExitStack() as _db_stack:
            try:
                db = _db_stack.enter_context(open_database_mapping(input_db_url))
            except api.SpineDBVersionError as exc:
                raise FlexToolConfigError(schema_version_message(input_db_url, exc)) from None
            if scenario_name:
                api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
            else:
                scen_names = db.get_scenario_items()
                if len(scen_names) == 0:
                    raise FlexToolConfigError("No scenario found")
                scenario_name=scen_names[0]['name']
            # The Work dir / DB URL / Scenario / Output header is
            # now printed once by ``cmd_run_flextool.main`` before
            # the cascade starts.  The previous multi-line INFO
            # here duplicated the Scenario line and split related
            # metadata across the startup log; drop it.
            _scenario_alternatives = db.get_scenario_alternative_items(scenario_name=scenario_name)
            if len(_scenario_alternatives) == 0:
                raise FlexToolConfigError("No alternatives in the scenario, i.e. empty scenario.")
            # The run record lists the scenario's alternatives in rank order.
            run_record.active().note_input(alternatives=[
                item["alternative_name"]
                for item in sorted(_scenario_alternatives, key=lambda item: item["rank"])
            ])

            # Pre-warm both entity and parameter_value caches so
            # every find_* call downstream (SolveConfig, TimelineConfig,
            # input_writer) hits memory.  Adding fetch_all("entity")
            # matches the spinedb-api docs' performance guidance and
            # mirrors the engine _solve_config / _timeline
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
        # ``timeset_warned``: the warnings this run printed already (see validate_timeset_blocks).
        timeline.validate_timeset_blocks(solve, timeset_warned)
        timeline.create_timeline_from_timestep_duration(solve)

        # Assemble RunnerState — the single cross-cutting state container
        self.state = RunnerState(
            paths=paths, solve=solve, timeline=timeline, logger=logger,
        )
        # The run's operator options (``None`` = read the environment when a reader needs them).
        self.state.run_options = run_options
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

    def write_input(
        self,
        input_db_url,
        scenario_name=None,
        *,
        provider=None,
    ) -> None:
        """Write input/ CSVs to the runner's workdir.

        Used by the regional-decomposition wrapper and a handful of
        debug callers that want a freshly-staged ``input/`` directory.
        When *provider* is None an ephemeral
        :class:`flextool.engine.preprocess._flex_data_provider.FlexDataProvider`
        is constructed; the cascade itself constructs its own Provider in
        ``_drive_cascade``.
        """
        if provider is None:
            from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
            provider = FlexDataProvider()
        _input_derivation_run(
            input_db_url,
            provider,
            self.state.logger,
            scenario_name=scenario_name,
            work_folder=self.state.paths.work_folder,
        )
        # Persist the cascade-input Provider on ``self.state`` so the
        # native cascade picks up the seeded ``input/<class>`` frames.
        # Mirrors the same handoff that
        # ``engine.solve._orchestration.run_orchestration`` performs.
        self.state.cascade_input_provider = provider
