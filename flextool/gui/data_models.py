from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PlotSettings:
    """Settings for a set of plots (single scenario or comparison)."""
    start_time: int = 0
    duration: int = 0
    config_file: str = ""
    active_configs: list[str] = field(default_factory=list)
    dispatch_plots: bool = True  # comparison only: include --dispatch-plots
    only_first_file: bool = False  # limit to one file per plot (quick overview)
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
    window_geometry: str = ""  # saved Tk geometry string
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
    # were dropped: HiGHS thread count is the canonical responsibility
    # of ``execution_limits.max_cores_per_job`` (Execution jobs
    # window), and ``user_bound_scale`` overlaps with ``scaling`` for
    # the rare advanced-user case where the autoscaler is bypassed in
    # favour of a hand-set bound scale via ``solver_arguments``.

    # HiGHS log verbosity (``--solver-log-level``).  One of
    # ``"silent"`` | ``"normal"`` | ``"verbose"``.  Default "normal";
    # only appended when non-default.
    solver_log_level: str = "normal"

    # HiGHS wall-clock time limit in whole seconds
    # (``--solver-time-limit``).  0 means "no limit" (the CLI's unset
    # default).  Only appended when > 0.
    solver_time_limit: int = 0

    # HiGHS MIP relative optimality gap (``--solver-mip-gap``), routed to
    # HiGHS' ``mip_rel_gap`` option.  ``solver_mip_gap_set`` gates whether
    # the value is sent at all: when True the gap is appended (and 0 is a
    # valid value — solve to a proven exact optimum); when False no
    # override is emitted and the solver_config/<solver>.opt baseline (or
    # the solver's built-in default) governs.  Only affects MIP solves
    # (integer investments, unit-commitment / online variables); pure-LP
    # solves ignore it.
    solver_mip_gap_set: bool = True
    solver_mip_gap: float = 0.001

    # On-disk format used when the solver is dispatched via a matrix
    # file (``--matrix-file-format``).  One of ``"mps"`` | ``"lp"``.
    # Default "mps"; only appended when non-default.  The in-process
    # vs. file decision itself is implicit (HiGHS + no --save-memory
    # = direct binding; commercial solvers + HiGHS --save-memory =
    # file write using the chosen format).
    matrix_file_format: str = "mps"

    # FlexTool autoscaler strategy (``--scaling``).  One of
    # ``"off"`` | ``"solver_only"`` | ``"basic"`` | ``"full"``.
    # Default "full"; only appended when non-default.
    scaling: str = "full"

    # HiGHS ``presolve`` override (``--presolve``).  One of
    # ``"on"`` | ``"off"`` | ``"choose"``.  All three are appended to
    # the engine command line: "choose" is HiGHS' native default and
    # lets the solver decide per-problem, overriding the engine's
    # determinism-pinned "on" baseline (that pin only governs the test
    # gate, which does not go through this CLI path).  Default "choose".
    presolve: str = "choose"

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
    max_cores_per_job: int = 1            # passed as --highs-threads to each subprocess
    memory_cap_per_job_gb: float = 0.0    # 0 = auto: (system_total - system_reserve_gb) / max_workers
    system_reserve_gb: float = 0.5        # tier 4: leave at least this much free system RAM
    swap_allowance_gb: float = 0.0        # tier 4: 0 = no swap; >0 allowed but warns user


@dataclass
class GlobalSettings:
    """Global settings stored in projects/projects.yaml."""
    recent_project: str | None = None
    theme: str = "dark"  # Valid values: "dark", "light", "os"
    exec_jobs_sash: int = 0  # saved Jobs/Progress sash position (0 = default)
    # cw at the time exec_jobs_sash was saved. 0 = unknown / use as-is.
    exec_jobs_layout_cw: int = 0
    # Legacy fallback for projects whose settings.yaml predates the
    # per-project ``max_workers`` field.  0 means "not set yet".
    max_workers: int = 0
    # Legacy fallback for projects whose settings.yaml predates the
    # per-project ``execution_limits`` field.  ExecutionManager reads
    # ProjectSettings first and only consults this when the project
    # field is missing / empty.
    execution_limits: ExecutionLimits = field(default_factory=ExecutionLimits)
    font_size_pt: int = 10        # body / menu / heading base size
    code_font_size_pt: int = 0    # TkFixedFont (logs, code views); 0 = auto = body+2
    # Check PyPI / the git remote for a newer version once at startup.
    check_updates_on_startup: bool = True
    # env_fingerprint() of the environment for which the polars native
    # self-check last passed.  Empty / mismatched => the check re-runs once
    # at startup (so a fresh install or a swapped polars build is verified
    # exactly once).  See flextool.env_check.
    polars_check_fingerprint: str = ""


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
    status: str  # "ok", "error", "empty", "editing", or "retired"
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


@dataclass
class ScenarioInfo:
    """Information about an available scenario."""
    name: str
    source_number: int
    source_name: str


@dataclass
class ExecutedScenarioInfo:
    """Information about an executed scenario with results."""
    name: str
    source_number: int
    timestamp: str
    has_plots: bool = False
    has_excel: bool = False
    has_csvs: bool = False
    has_comp_plots: bool = False
    has_comp_excel: bool = False
