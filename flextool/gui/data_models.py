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

    # Plot settings
    single_plot_settings: PlotSettings = field(default_factory=PlotSettings)
    comparison_plot_settings: PlotSettings = field(default_factory=PlotSettings)

    # Input source numbers: source name -> number
    input_source_numbers: dict[str, int] = field(default_factory=dict)

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

    # Transient flag (not persisted): set by execution manager when scenarios
    # finish, cleared by the result viewer when it picks up the changes.
    scenarios_changed: bool = field(default=False, repr=False)


@dataclass
class ExecutionLimits:
    """Per-machine resource limits for FlexTool subprocess execution.

    All fields are persisted in projects/projects.yaml under
    ``execution_limits``. A value of 0 / 0.0 means "auto" (compute at
    dispatch time from system info).
    """
    max_cores_per_job: int = 1            # passed as --highs-threads to each subprocess
    memory_cap_per_job_gb: float = 0.0    # 0 = auto: (system_total - system_reserve_gb) / max_workers
    system_reserve_gb: float = 4.0        # tier 4: leave at least this much free system RAM
    swap_allowance_gb: float = 0.0        # tier 4: 0 = no swap; >0 allowed but warns user


@dataclass
class GlobalSettings:
    """Global settings stored in projects/projects.yaml."""
    recent_project: str | None = None
    theme: str = "dark"  # Valid values: "dark", "light", "os"
    exec_jobs_sash: int = 0  # saved Jobs/Progress sash position (0 = default)
    # cw at the time exec_jobs_sash was saved. 0 = unknown / use as-is.
    exec_jobs_layout_cw: int = 0
    # Last chosen value for "Max. parallel executions" in the execution
    # window. 0 means "not set yet" → use cpu_count() - 1.
    max_workers: int = 0
    execution_limits: ExecutionLimits = field(default_factory=ExecutionLimits)


@dataclass
class InputSourceInfo:
    """Information about an input source file."""
    name: str
    file_type: str  # "xlsx" or "sqlite"
    number: int
    status: str  # "ok", "error", "empty", or "editing"
    scenarios: list[str] = field(default_factory=list)
    # Set when the source lives outside the project; stored as POSIX path
    # relative to the project root (e.g. "../data/input.xlsx").
    external_rel_path: str | None = None


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
