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
    cache_gb: float = 0.5     # plot cache memory limit in GB


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

    # Ordered list of scenario names for execution
    scenario_order: list[str] = field(default_factory=list)

    # Scenarios used for the last comparison outputs
    comp_plots_scenarios: list[str] = field(default_factory=list)
    comp_excel_scenarios: list[str] = field(default_factory=list)

    # Persisted checkbox states
    checked_input_sources: list[str] = field(default_factory=list)    # source names
    checked_available_scenarios: list[str] = field(default_factory=list)  # "source_number|name" keys
    checked_executed_scenarios: list[str] = field(default_factory=list)   # scenario names

    # Result viewer settings
    viewer_settings: ViewerSettings = field(default_factory=ViewerSettings)

    # Transient flag (not persisted): set by execution manager when scenarios
    # finish, cleared by the result viewer when it picks up the changes.
    scenarios_changed: bool = field(default=False, repr=False)


@dataclass
class GlobalSettings:
    """Global settings stored in projects/projects.yaml."""
    recent_project: str | None = None
    theme: str = "dark"  # Valid values: "dark", "light", "os"


@dataclass
class InputSourceInfo:
    """Information about an input source file."""
    name: str
    file_type: str  # "xlsx" or "sqlite"
    number: int
    status: str  # "ok", "error", "empty", or "editing"
    scenarios: list[str] = field(default_factory=list)


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
