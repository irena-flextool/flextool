from __future__ import annotations

from dataclasses import dataclass, field

# Revision of the GUI solver settings in settings.yaml.  1 (no key): presolve "choose" and a 0.001 MIP gap were
# sent with every run.  2: presolve and scaling are sent only when the user picks them; the GUI sends no MIP gap.
SOLVER_SETTINGS_REVISION = 2

#: How the GUI shows a date and time (finish times of runs, file times): DD.MM.YY hh:mm.
TIMESTAMP_FORMAT = "%d.%m.%y %H:%M"


@dataclass
class PlotSettings:
    """Settings for a set of plots (single scenario or comparison)."""
    start_time: int = 0
    duration: int = 0
    config_file: str = ""
    active_configs: list[str] = field(default_factory=list)
    dispatch_plots: bool = True  # comparison only: include --dispatch-plots
    only_first_file: bool = False  # limit to one file per plot (quick overview)
    write_plot_data: bool = False  # also write each plot's drawn data as CSV next to it
    # Per-variant duration, keyed by variant letter (e.g. {"h": 168, "w": 8760}).
    # Only ints — "all" sentinel from the template is resolved to a concrete
    # integer the first time the variant is seen and stored here.
    variant_durations: dict[str, int] = field(default_factory=dict)


@dataclass
class ViewerSettings:
    """Settings for the result viewer window."""
    last_scenario: str = ""
    last_entry: str = ""      # e.g., "0.0"
    last_variant: str = ""    # e.g., "t"
    last_mode: str = "single" # "single", "comparison", "network"
    # Saved viewer-window placement, keyed by monitor configuration
    # (ui_metrics.monitor_signature): {signature -> "WxH+X+Y"}.
    window_geometry: dict[str, str] = field(default_factory=dict)
    left_pane_width: int = 0  # saved horizontal sash position (0 = use default)
    scenario_pane_height: int = 0  # saved vertical sash in left column (0 = default)
    # cw (TkDefaultFont.measure("0")) at the time window_geometry / sash
    # positions were saved. 0 means unknown — use raw saved values.
    layout_cw: int = 0
    cache_gb: float = 0.5     # plot cache memory limit in GB


@dataclass
class ScenarioRun:
    """Recorded resource usage from a previous successful scenario run.

    Persisted in settings.yaml under ``scenario_resource_history``, keyed
    by output subdir. Used by the execution manager to set a learned
    memory budget for the next run of the same scenario.
    """
    peak_rss_mb: float = 0.0    # high-water mark observed by MemoryWatchdog
    runtime_s: float = 0.0      # wall-clock seconds from start_time to end_time
    last_run: str = ""          # ISO-8601 local timestamp, e.g. "2026-04-25T14:30:15"


@dataclass
class ProjectSettings:
    """Per-project settings stored in settings.yaml."""
    # Auto-generate flags
    auto_generate_scen_plots: bool = True
    auto_generate_scen_excels: bool = False
    auto_generate_scen_csvs: bool = True
    auto_generate_comp_plots: bool = True
    auto_generate_comp_excel: bool = False
    # SpineDB results database. Produced only during the solve (the writer
    # needs the live s/par namespaces), so it has no parquet-based regen
    # path — unlike the other outputs above. One results.sqlite per project,
    # each scenario appended as its own alternative.
    auto_generate_comp_spinedb: bool = False

    # Diagnostic verbosity for scenario execution.  Controlled by the
    # "Debug" radio group in the main window.  Values mirror the CLI
    # ``--debug`` flag in ``flextool/cli/cmd_run_flextool.py``:
    #
    #   "off"   — no extra flags appended.
    #   "basic" — appends ``--debug=basic`` (verbose memory checkpoints
    #             + DEBUG log level; no tracemalloc overhead).
    #   "full"  — appends ``--debug=full --csv-dump`` (basic + tracemalloc
    #             diagnostics CSV + retained intermediate CSVs).  Slow;
    #             reserved for allocation-regression investigations.
    debug_level: str = "off"

    # When True, scenario execution runs with --save-memory: builds the
    # LP, writes MPS, drops everything Python-side, then spawns a HiGHS
    # subprocess to solve. Parent and solver memory no longer compound
    # in the same process address space. Trades ~+30-60 s I/O per
    # sub-solve and warm-LP reuse for substantial peak RSS relief.
    # Controlled by the "Save memory" checkbox in the main window,
    # above "Debug".
    save_memory: bool = False

    # ── Solver options (CLI knobs surfaced via the "Solver options…"
    # dialog launched from the main window side menu).  Values mirror
    # the matching flags in ``flextool/cli/cmd_run_flextool.py`` and
    # are appended by ExecutionManager only when they differ from the
    # defaults below, so the engine command line stays clean on the
    # common path.
    #
    # The previous ``highs_threads`` and ``user_bound_scale`` fields
    # were dropped: the thread count (FlexTool's processing and HiGHS) is the canonical responsibility
    # of ``execution_limits.max_cores_per_job`` (Execution jobs
    # window, "Threads per run"), and ``user_bound_scale`` overlaps with ``scaling`` for
    # the rare advanced-user case where the autoscaler is bypassed in
    # favour of a hand-set bound scale via ``solver_arguments``.

    # Solver log verbosity (``--solver-log-level``; ``silent`` also hides a commercial solver's log).  One of
    # ``"silent"`` | ``"normal"`` | ``"verbose"``.  Default "normal";
    # only appended when non-default.
    solver_log_level: str = "normal"

    # HiGHS wall-clock time limit in whole seconds
    # (``--solver-time-limit``).  0 means "no limit" (the CLI's unset
    # default).  Only appended when > 0.
    solver_time_limit: int = 0

    # FlexTool autoscaler strategy (``--scaling``): ``None`` (the dialog's "Default") sends nothing, so each
    # solve's ``scaling`` in the database governs (else full); ``"off"`` | ``"solver_only"`` | ``"basic"`` |
    # ``"full"`` is sent and wins over the database.
    scaling: str | None = None

    # HiGHS ``presolve`` (``--presolve``): ``None`` (the dialog's "Default (on)") sends nothing, so the GUI
    # solves like a terminal run (presolve on); ``"on"`` | ``"off"`` | ``"choose"`` is sent.
    presolve: str | None = None

    # Revision of the solver settings above.  A settings.yaml without it (or with an older one) holds the
    # earlier GUI defaults, which ``settings_io.load_project_settings`` updates once
    # (see ``SOLVER_SETTINGS_REVISION``).
    solver_settings_revision: int = SOLVER_SETTINGS_REVISION

    # ── Calibrate-investments dialog settings (surfaced via the
    # "Calibrate investments…" dialog).  Representative-period build knobs
    # (calib_rp_*), the set of solves the calibrator iterates over
    # (calib_selected_solves), and the iteration / sizing / damping /
    # stall-guard controls.  Persisted so the dialog reopens with the
    # user's last choices.
    calib_rp_n_rp: int = 20
    calib_rp_period_length: int = 108
    calib_rp_force_sustained: bool = True
    calib_rp_force_peak: bool = False
    calib_rp_force_window: int = 24
    # Disposition of the freshly built RP alternative:
    #   "detached"     — create the alternative but attach it to nothing;
    #   "add"          — append it to each selected scenario's stack (default,
    #                    so its investment solve immediately uses the periods);
    #   "new_scenario" — leave the selected scenarios untouched and clone each
    #                    into a new "<scenario>_<alternative>" scenario that
    #                    carries the same stack plus the new alternative on top.
    calib_rp_scenario_mode: str = "add"  # "detached" | "add" | "new_scenario"
    calib_selected_solves: list[str] = field(default_factory=list)
    calib_max_iterations: int = 8
    calib_sizing: str = "timed"  # "timed" | "uniform"
    calib_overshoot_pct: float = 0.0
    calib_damping_first: float = 1.0
    calib_damping_remaining: float = 0.5
    calib_stall_fraction: float = 0.05
    calib_keep_artifacts: bool = False

    # Plot settings
    single_plot_settings: PlotSettings = field(default_factory=PlotSettings)
    comparison_plot_settings: PlotSettings = field(default_factory=PlotSettings)

    # Input source numbers: source name -> number (live sources only).
    input_source_numbers: dict[str, int] = field(default_factory=dict)

    # Persistent per-number source identity: str(number) -> SourceRecord.
    # Survives file deletion so orphaned results stay attributable and a
    # re-added file reclaims its old number. Garbage-collected once a
    # number has neither a live file nor any results. See SourceRecord.
    source_registry: dict[str, SourceRecord] = field(default_factory=dict)

    # External input references: source name -> POSIX path relative to project root.
    # Files are read in place (not copied into input_sources/).
    external_refs: dict[str, str] = field(default_factory=dict)

    # Bare-name ownership for executed scenario folders.
    # Maps scenario_name -> source_number. Scenarios in this map write to
    # ``output_parquet/<name>/`` (no suffix); other sources with the same
    # scenario name write to ``output_parquet/<name>_<src#>/``.
    bare_output_owners: dict[str, int] = field(default_factory=dict)

    # Ordered list of scenario names for execution
    scenario_order: list[str] = field(default_factory=list)

    # User-ordered list of executed-scenario names for the result viewer
    # tree (drag/Alt+Up/Down reordering). New scenarios are appended at the
    # end of this list when first seen. Names that no longer exist on disk
    # are pruned at next scan.
    executed_scenario_order: list[str] = field(default_factory=list)

    # Scenarios used for the last comparison outputs
    comp_plots_scenarios: list[str] = field(default_factory=list)
    comp_excel_scenarios: list[str] = field(default_factory=list)

    # Scenarios ticked in the result-viewer comparison mode. Distinct from
    # comp_plots_scenarios (which tracks the last run) because the viewer's
    # ticks may evolve independently until a regen/plot is triggered.
    comp_viewer_scenarios: list[str] = field(default_factory=list)

    # Persisted checkbox states
    checked_input_sources: list[str] = field(default_factory=list)    # source names
    checked_available_scenarios: list[str] = field(default_factory=list)  # "source_number|name" keys
    checked_executed_scenarios: list[str] = field(default_factory=list)   # scenario names

    # Result viewer settings
    viewer_settings: ViewerSettings = field(default_factory=ViewerSettings)

    # Per-scenario resource history: output_subdir -> ScenarioRun.
    # Populated after a successful run; consulted at dispatch time so the
    # next run gets a learned memory budget instead of the static auto fallback.
    scenario_resource_history: dict[str, ScenarioRun] = field(default_factory=dict)

    # Per-project execution limits — primary source of truth for the
    # execution manager.  Defaults match the conservative single-job
    # profile the user prefers as a baseline (1 worker, 1 core, auto
    # budget, 0.5 GB reserve, no swap).  GlobalSettings still carries
    # the same fields as a legacy fallback; see ExecutionManager.
    execution_limits: "ExecutionLimits" = field(
        default_factory=lambda: ExecutionLimits()
    )
    # Last chosen "Max. parallel executions" for this project. 0 means
    # "use the GlobalSettings fallback"; otherwise an explicit value
    # honoured by ExecutionManager.
    max_workers: int = 1

    # Transient flag (not persisted): set by execution manager when scenarios
    # finish, cleared by the result viewer when it picks up the changes.
    scenarios_changed: bool = field(default=False, repr=False)


@dataclass
class ExecutionLimits:
    """Per-machine resource limits for FlexTool subprocess execution.

    Primary storage is ``ProjectSettings.execution_limits`` (per-project
    settings.yaml).  ``GlobalSettings.execution_limits`` is retained as a
    legacy fallback for projects whose settings.yaml predates the
    per-project field.  A value of 0 / 0.0 means "auto" (compute at
    dispatch time from system info).
    """
    max_cores_per_job: int = 1            # "Threads per run": passed as --threads to each subprocess
    memory_cap_per_job_gb: float = 0.0    # 0 = auto: (system_total - system_reserve_gb) / max_workers
    system_reserve_gb: float = 0.5        # tier 4: leave at least this much free system RAM
    swap_allowance_gb: float = 0.0        # tier 4: 0 = no swap; >0 allowed but warns user


@dataclass
class GlobalSettings:
    """Global settings stored in projects/projects.yaml."""
    recent_project: str | None = None
    theme: str = "dark"  # Valid values: "dark", "light", "os"
    exec_jobs_sash: int = 0  # saved Jobs/Progress sash position (0 = default)
    # cw at the time the exec-jobs geometry/sash was saved. 0 = unknown.
    exec_jobs_layout_cw: int = 0
    # Saved exec-jobs window placement, keyed by monitor configuration
    # (ui_metrics.monitor_signature): {signature -> "WxH+X+Y"}. Each layout
    # remembers its own placement; an unseen layout falls back to the
    # monitor-aware default. Restored clamped/rescaled on open.
    exec_jobs_geometry: dict[str, str] = field(default_factory=dict)
    # Saved main-window placement, same per-configuration scheme as above.
    main_window_geometry: dict[str, str] = field(default_factory=dict)
    # cw at the time main_window_geometry was saved. 0 = unknown.
    main_window_layout_cw: int = 0
    # Legacy fallback for projects whose settings.yaml predates the
    # per-project ``max_workers`` field.  0 means "not set yet".
    max_workers: int = 0
    # Legacy fallback for projects whose settings.yaml predates the
    # per-project ``execution_limits`` field.  ExecutionManager reads
    # ProjectSettings first and only consults this when the project
    # field is missing / empty.
    execution_limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    font_size_pt: int = 10        # body / menu / heading base size
    code_font_size_pt: int = 0    # TkFixedFont (logs, code views); 0 = auto = body
    # Check PyPI / the git remote for a newer version once at startup.
    check_updates_on_startup: bool = True
    # env_fingerprint() of the environment for which the polars native
    # self-check last passed.  Empty / mismatched => the check re-runs once
    # at startup (so a fresh install or a swapped polars build is verified
    # exactly once).  See flextool.env_check.
    polars_check_fingerprint: str = ""
    # Saved AI assistant chat-window placement, same per-configuration scheme
    # as exec_jobs_geometry ({signature -> "WxH+X+Y"}), and the cw it was
    # saved at (0 = unknown).  Written by the chat window when it closes.
    assistant_window_geometry: dict[str, str] = field(default_factory=dict)
    assistant_window_layout_cw: int = 0


@dataclass
class SourceRecord:
    """Persistent identity of an input source, keyed by its source number.

    Stored in ``ProjectSettings.source_registry`` as ``str(number) ->
    SourceRecord``. Unlike ``input_source_numbers`` (a live name→number
    index), this record **survives deletion of the underlying file** so
    that:

    * executed-scenario folders left on disk (``output_parquet/<name>_N``)
      can still be attributed to a named file (drives the greyed-out
      "ghost" rows in the input-source list), and
    * re-adding the same file reclaims its original number instead of
      drifting to a new one (the renumbering bug).

    A record is garbage-collected from settings.yaml on the next
    ``refresh()`` once **neither** a live file **nor** any executed-
    scenario results reference its number — so empty ghosts never
    accumulate. The "retired" state is derived per refresh from
    ``(has live file?, has results?)``; it is not persisted here.
    """
    name: str = ""
    # Project-root-relative POSIX path (e.g. "input_sources/examples.sqlite"
    # for an internal file, "../data/input.xlsx" for an external one).
    # Empty when the path is unknown (legacy orphan with no recorded owner).
    path: str = ""


@dataclass
class InputSourceInfo:
    """Information about an input source file."""
    name: str
    file_type: str  # "xlsx" or "sqlite"
    number: int
    # "ok", "error", "empty", "editing", "retired", or "frozen_old" (a frozen
    # sqlite older than this FlexTool that the user did not migrate: not runnable)
    status: str
    scenarios: list[str] = field(default_factory=list)
    # Set when the source lives outside the project; stored as POSIX path
    # relative to the project root (e.g. "../data/input.xlsx").
    external_rel_path: str | None = None
    # True for a "ghost" row: the input file is gone but executed-scenario
    # results still reference this number. Such rows are informational
    # (greyed out, not runnable) and disappear once the results are removed.
    retired: bool = False
    # Number of executed-scenario result folders that resolve to this
    # source number (used for the ghost-row tooltip / GC decision).
    result_count: int = 0
    # The label of a frozen sqlite (``flextool-freeze``), or None.
    frozen_label: str | None = None
    # The frozen marker and version for the status tooltip ("" when not frozen).
    frozen_note: str = ""


@dataclass
class ScenarioInfo:
    """Information about an available scenario."""
    name: str
    source_number: int
    source_name: str


@dataclass
class ExecutedScenarioInfo:
    """Information about an executed scenario with results.

    ``status`` is the folder's ``_run_status.json`` (``None`` for a folder from before run records).
    ``run_state`` is its ``state``, with a stale ``running`` replaced by ``abandoned`` / ``unknown``
    (``run_record.effective_state``), for display.  ``checkable`` is ``run_record.results_usable(status)``:
    only then may the row be checked for plots and comparison, and ``unusable_reason`` says why not.
    """
    name: str
    source_number: int
    timestamp: str
    has_plots: bool = False
    has_excel: bool = False
    has_csvs: bool = False
    has_comp_plots: bool = False
    has_comp_excel: bool = False
    status: dict | None = None
    run_state: str | None = None
    parquet_state: str | None = None
    checkable: bool = True
    unusable_reason: str | None = None
