from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import yaml

from flextool.gui.data_models import (
    ExecutionLimits,
    GlobalSettings,
    PlotSettings,
    ProjectSettings,
    ScenarioRun,
    ViewerSettings,
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
    settings.debug = data.get("debug", settings.debug)
    settings.save_memory = bool(data.get("save_memory", settings.save_memory))
    settings.input_source_numbers = data.get(
        "input_source_numbers", settings.input_source_numbers
    )
    settings.external_refs = data.get("external_refs", settings.external_refs)
    settings.bare_output_owners = data.get(
        "bare_output_owners", settings.bare_output_owners
    )
    settings.scenario_order = data.get("scenario_order", settings.scenario_order)
    settings.executed_scenario_order = data.get(
        "executed_scenario_order", settings.executed_scenario_order
    )
    settings.comp_plots_scenarios = data.get("comp_plots_scenarios", [])
    settings.comp_excel_scenarios = data.get("comp_excel_scenarios", [])
    settings.comp_viewer_scenarios = data.get("comp_viewer_scenarios", [])
    settings.checked_input_sources = data.get("checked_input_sources", [])
    settings.checked_available_scenarios = data.get("checked_available_scenarios", [])
    settings.checked_executed_scenarios = data.get("checked_executed_scenarios", [])

    def _clean_variant_durations(raw: object) -> dict[str, int]:
        if not isinstance(raw, dict):
            return {}
        cleaned: dict[str, int] = {}
        for k, v in raw.items():
            if isinstance(v, bool):
                # bool is an int subclass — exclude explicitly
                continue
            if isinstance(v, (int, float)):
                cleaned[str(k)] = int(v)
        return cleaned

    single_plot = data.get("single_plot_settings")
    if isinstance(single_plot, dict):
        settings.single_plot_settings = PlotSettings(
            start_time=single_plot.get("start_time", 0),
            duration=single_plot.get("duration", 0),
            config_file=single_plot.get("config_file", ""),
            active_configs=single_plot.get("active_configs", []),
            only_first_file=single_plot.get("only_first_file", False),
            variant_durations=_clean_variant_durations(
                single_plot.get("variant_durations", {})
            ),
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
            variant_durations=_clean_variant_durations(
                comp_plot.get("variant_durations", {})
            ),
        )

    viewer = data.get("viewer_settings")
    if isinstance(viewer, dict):
        settings.viewer_settings = ViewerSettings(
            last_scenario=viewer.get("last_scenario", ""),
            last_entry=viewer.get("last_entry", ""),
            last_variant=viewer.get("last_variant", ""),
            last_mode=viewer.get("last_mode", "single"),
            window_geometry=viewer.get("window_geometry", ""),
            left_pane_width=viewer.get("left_pane_width", 0),
            scenario_pane_height=viewer.get("scenario_pane_height", 0),
            layout_cw=viewer.get("layout_cw", 0),
            cache_gb=viewer.get("cache_gb", 0.5),
        )

    history_data = data.get("scenario_resource_history", {})
    if isinstance(history_data, dict):
        history: dict[str, ScenarioRun] = {}
        for subdir, run_data in history_data.items():
            if not isinstance(run_data, dict):
                continue
            history[str(subdir)] = ScenarioRun(
                peak_rss_mb=float(run_data.get("peak_rss_mb", 0.0)),
                runtime_s=float(run_data.get("runtime_s", 0.0)),
                last_run=str(run_data.get("last_run", "")),
            )
        settings.scenario_resource_history = history

    limits_data = data.get("execution_limits")
    if isinstance(limits_data, dict):
        settings.execution_limits = ExecutionLimits(
            max_cores_per_job=int(limits_data.get(
                "max_cores_per_job", settings.execution_limits.max_cores_per_job
            )),
            memory_cap_per_job_gb=float(limits_data.get(
                "memory_cap_per_job_gb", settings.execution_limits.memory_cap_per_job_gb
            )),
            system_reserve_gb=float(limits_data.get(
                "system_reserve_gb", settings.execution_limits.system_reserve_gb
            )),
            swap_allowance_gb=float(limits_data.get(
                "swap_allowance_gb", settings.execution_limits.swap_allowance_gb
            )),
        )
    mw = data.get("max_workers")
    if isinstance(mw, int) and mw > 0:
        settings.max_workers = mw

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

    limits_data = data.get("execution_limits")
    if isinstance(limits_data, dict):
        execution_limits = ExecutionLimits(
            max_cores_per_job=limits_data.get("max_cores_per_job", 1),
            memory_cap_per_job_gb=limits_data.get("memory_cap_per_job_gb", 0.0),
            system_reserve_gb=limits_data.get("system_reserve_gb", 4.0),
            swap_allowance_gb=limits_data.get("swap_allowance_gb", 0.0),
        )
    else:
        execution_limits = ExecutionLimits()

    return GlobalSettings(
        recent_project=data.get("recent_project"),
        theme=theme,
        exec_jobs_sash=data.get("exec_jobs_sash", 0),
        exec_jobs_layout_cw=data.get("exec_jobs_layout_cw", 0),
        max_workers=data.get("max_workers", 0),
        execution_limits=execution_limits,
        font_size_pt=data.get("font_size_pt", 10),
        code_font_size_pt=data.get("code_font_size_pt", 0),
    )


def save_global_settings(projects_dir: Path, settings: GlobalSettings) -> None:
    """Save global settings to projects/projects.yaml."""
    projects_dir.mkdir(parents=True, exist_ok=True)
    settings_file = projects_dir / GLOBAL_SETTINGS_FILENAME

    data = asdict(settings)
    with open(settings_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
