from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from flextool.gui.data_models import (
    GlobalSettings,
    PlotSettings,
    ProjectSettings,
)

SETTINGS_FILENAME = "settings.yaml"
GLOBAL_SETTINGS_FILENAME = "projects.yaml"


def load_project_settings(project_path: Path) -> ProjectSettings:
    """Load project settings from settings.yaml in the project directory.

    Returns defaults if the file does not exist or cannot be parsed.
    """
    settings_file = project_path / SETTINGS_FILENAME
    if not settings_file.exists():
        return ProjectSettings()

    try:
        with open(settings_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return ProjectSettings()

    if not isinstance(data, dict):
        return ProjectSettings()

    settings = ProjectSettings()
    settings.auto_generate_scen_plots = data.get(
        "auto_generate_scen_plots", settings.auto_generate_scen_plots
    )
    settings.auto_generate_scen_excels = data.get(
        "auto_generate_scen_excels", settings.auto_generate_scen_excels
    )
    settings.auto_generate_scen_csvs = data.get(
        "auto_generate_scen_csvs", settings.auto_generate_scen_csvs
    )
    settings.auto_generate_comp_plots = data.get(
        "auto_generate_comp_plots", settings.auto_generate_comp_plots
    )
    settings.auto_generate_comp_excel = data.get(
        "auto_generate_comp_excel", settings.auto_generate_comp_excel
    )
    settings.input_source_numbers = data.get(
        "input_source_numbers", settings.input_source_numbers
    )
    settings.scenario_order = data.get("scenario_order", settings.scenario_order)
    settings.comp_plots_scenarios = data.get("comp_plots_scenarios", [])
    settings.comp_excel_scenarios = data.get("comp_excel_scenarios", [])
    settings.checked_input_sources = data.get("checked_input_sources", [])
    settings.checked_available_scenarios = data.get("checked_available_scenarios", [])
    settings.checked_executed_scenarios = data.get("checked_executed_scenarios", [])

    single_plot = data.get("single_plot_settings")
    if isinstance(single_plot, dict):
        settings.single_plot_settings = PlotSettings(
            start_time=single_plot.get("start_time", 0),
            duration=single_plot.get("duration", 0),
            config_file=single_plot.get("config_file", ""),
            active_configs=single_plot.get("active_configs", []),
            only_first_file=single_plot.get("only_first_file", False),
        )

    comp_plot = data.get("comparison_plot_settings")
    if isinstance(comp_plot, dict):
        settings.comparison_plot_settings = PlotSettings(
            start_time=comp_plot.get("start_time", 0),
            duration=comp_plot.get("duration", 0),
            config_file=comp_plot.get("config_file", ""),
            active_configs=comp_plot.get("active_configs", []),
            dispatch_plots=comp_plot.get("dispatch_plots", True),
            only_first_file=comp_plot.get("only_first_file", False),
        )

    return settings


def save_project_settings(project_path: Path, settings: ProjectSettings) -> None:
    """Save project settings to settings.yaml in the project directory."""
    settings_file = project_path / SETTINGS_FILENAME
    project_path.mkdir(parents=True, exist_ok=True)

    data = asdict(settings)
    with open(settings_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_global_settings(projects_dir: Path) -> GlobalSettings:
    """Load global settings from projects/projects.yaml.

    Returns defaults if the file does not exist or cannot be parsed.
    """
    settings_file = projects_dir / GLOBAL_SETTINGS_FILENAME
    if not settings_file.exists():
        return GlobalSettings()

    try:
        with open(settings_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return GlobalSettings()

    if not isinstance(data, dict):
        return GlobalSettings()

    theme = data.get("theme", "dark")
    if theme not in ("dark", "light", "os"):
        theme = "dark"

    return GlobalSettings(
        recent_project=data.get("recent_project"),
        theme=theme,
    )


def save_global_settings(projects_dir: Path, settings: GlobalSettings) -> None:
    """Save global settings to projects/projects.yaml."""
    projects_dir.mkdir(parents=True, exist_ok=True)
    settings_file = projects_dir / GLOBAL_SETTINGS_FILENAME

    data = asdict(settings)
    with open(settings_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
