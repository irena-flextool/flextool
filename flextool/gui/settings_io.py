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
    SourceRecord,
    ViewerSettings,
)

SETTINGS_FILENAME = "settings.yaml"
GLOBAL_SETTINGS_FILENAME = "projects.yaml"


def _as_geometry_map(raw: object) -> dict[str, str]:
    """Normalise a saved geometry field into a {signature -> geometry} map.

    Accepts the new dict form, or a legacy plain string (from before
    per-monitor-configuration memory) which is migrated under the
    ``"legacy"`` key — used as a layout-agnostic fallback until the window
    is next closed and re-saved under a concrete signature. Anything else
    yields an empty map.
    """
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items() if isinstance(v, str) and v}
    if isinstance(raw, str) and raw:
        return {"legacy": raw}
    return {}


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
    settings.auto_generate_comp_spinedb = data.get(
        "auto_generate_comp_spinedb", settings.auto_generate_comp_spinedb
    )
    # Legacy compat: pre-tiered ``debug: bool`` settings.yaml entries
    # map True→"full" (preserves their old behaviour: tracemalloc +
    # csv-dump) and False→"off".  New entries use ``debug_level``
    # directly.
    if "debug_level" in data:
        _level = data.get("debug_level", settings.debug_level)
        if _level not in ("off", "basic", "full"):
            _level = "off"
        settings.debug_level = _level
    elif "debug" in data:
        settings.debug_level = "full" if bool(data["debug"]) else "off"
    settings.save_memory = bool(data.get("save_memory", settings.save_memory))

    # Solver options — validate each against its allowed set / type and
    # fall back to the dataclass default on anything malformed so a
    # hand-edited settings.yaml can't break the GUI.
    _sll = data.get("solver_log_level", settings.solver_log_level)
    if _sll in ("silent", "normal", "verbose"):
        settings.solver_log_level = _sll
    _stl = data.get("solver_time_limit", settings.solver_time_limit)
    if isinstance(_stl, int) and not isinstance(_stl, bool) and _stl >= 0:
        settings.solver_time_limit = _stl
    _smg = data.get("solver_mip_gap", settings.solver_mip_gap)
    if isinstance(_smg, (int, float)) and not isinstance(_smg, bool) and _smg >= 0:
        settings.solver_mip_gap = float(_smg)
    _smg_set = data.get("solver_mip_gap_set", settings.solver_mip_gap_set)
    if isinstance(_smg_set, bool):
        settings.solver_mip_gap_set = _smg_set
    _mff = data.get("matrix_file_format", settings.matrix_file_format)
    if _mff in ("mps", "lp"):
        settings.matrix_file_format = _mff
    _scl = data.get("scaling", settings.scaling)
    if _scl in ("off", "solver_only", "basic", "full"):
        settings.scaling = _scl
    _ps = data.get("presolve", settings.presolve)
    if _ps in ("on", "off", "choose"):
        settings.presolve = _ps

    # Calibrate-investments dialog — validate each field against its
    # allowed set / type and fall back to the dataclass default on
    # anything malformed so a hand-edited settings.yaml can't break the
    # GUI.  bool is an int subclass, so ints/floats exclude it explicitly.
    _c_n_rp = data.get("calib_rp_n_rp", settings.calib_rp_n_rp)
    if isinstance(_c_n_rp, int) and not isinstance(_c_n_rp, bool):
        settings.calib_rp_n_rp = _c_n_rp
    _c_plen = data.get("calib_rp_period_length", settings.calib_rp_period_length)
    if isinstance(_c_plen, int) and not isinstance(_c_plen, bool):
        settings.calib_rp_period_length = _c_plen
    _c_sust = data.get("calib_rp_force_sustained", settings.calib_rp_force_sustained)
    if isinstance(_c_sust, bool):
        settings.calib_rp_force_sustained = _c_sust
    _c_peak = data.get("calib_rp_force_peak", settings.calib_rp_force_peak)
    if isinstance(_c_peak, bool):
        settings.calib_rp_force_peak = _c_peak
    _c_win = data.get("calib_rp_force_window", settings.calib_rp_force_window)
    if isinstance(_c_win, int) and not isinstance(_c_win, bool):
        settings.calib_rp_force_window = _c_win
    _c_mode = data.get("calib_rp_scenario_mode")
    if isinstance(_c_mode, str) and _c_mode in ("detached", "add", "new_scenario"):
        settings.calib_rp_scenario_mode = _c_mode
    else:
        # Legacy migration: the pre-3-way boolean "add to scenario" flag
        # (True -> append to the scenario, False -> leave the alt detached).
        _c_add = data.get("calib_rp_add_to_scenario")
        if isinstance(_c_add, bool):
            settings.calib_rp_scenario_mode = "add" if _c_add else "detached"
    _c_solves = data.get("calib_selected_solves", settings.calib_selected_solves)
    if isinstance(_c_solves, list):
        settings.calib_selected_solves = [
            s for s in _c_solves if isinstance(s, str) and s
        ]
    _c_iter = data.get("calib_max_iterations", settings.calib_max_iterations)
    if isinstance(_c_iter, int) and not isinstance(_c_iter, bool):
        settings.calib_max_iterations = _c_iter
    _c_sizing = data.get("calib_sizing", settings.calib_sizing)
    if _c_sizing in ("timed", "uniform"):
        settings.calib_sizing = _c_sizing
    _c_over = data.get("calib_overshoot_pct", settings.calib_overshoot_pct)
    if isinstance(_c_over, (int, float)) and not isinstance(_c_over, bool):
        settings.calib_overshoot_pct = float(_c_over)
    _c_damp1 = data.get("calib_damping_first", settings.calib_damping_first)
    if isinstance(_c_damp1, (int, float)) and not isinstance(_c_damp1, bool):
        settings.calib_damping_first = float(_c_damp1)
    _c_dampr = data.get("calib_damping_remaining", settings.calib_damping_remaining)
    if isinstance(_c_dampr, (int, float)) and not isinstance(_c_dampr, bool):
        settings.calib_damping_remaining = float(_c_dampr)
    _c_stall = data.get("calib_stall_fraction", settings.calib_stall_fraction)
    if isinstance(_c_stall, (int, float)) and not isinstance(_c_stall, bool):
        settings.calib_stall_fraction = float(_c_stall)
    _c_keep = data.get("calib_keep_artifacts", settings.calib_keep_artifacts)
    if isinstance(_c_keep, bool):
        settings.calib_keep_artifacts = _c_keep

    settings.input_source_numbers = data.get(
        "input_source_numbers", settings.input_source_numbers
    )

    # Per-number source identity. Keys are stringified numbers; values are
    # {name, path} dicts. Tolerate a hand-edited / malformed map by
    # validating each entry and dropping the rest.
    registry_data = data.get("source_registry", {})
    if isinstance(registry_data, dict):
        registry: dict[str, SourceRecord] = {}
        for num_key, rec in registry_data.items():
            key = str(num_key)
            if not key.isdigit():
                continue
            if isinstance(rec, dict):
                registry[key] = SourceRecord(
                    name=str(rec.get("name", "")),
                    path=str(rec.get("path", "")),
                )
        settings.source_registry = registry

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
            window_geometry=_as_geometry_map(viewer.get("window_geometry")),
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
        exec_jobs_geometry=_as_geometry_map(data.get("exec_jobs_geometry")),
        main_window_geometry=_as_geometry_map(data.get("main_window_geometry")),
        main_window_layout_cw=data.get("main_window_layout_cw", 0),
        max_workers=data.get("max_workers", 0),
        execution_limits=execution_limits,
        font_size_pt=data.get("font_size_pt", 10),
        code_font_size_pt=data.get("code_font_size_pt", 0),
        check_updates_on_startup=data.get("check_updates_on_startup", True),
        polars_check_fingerprint=data.get("polars_check_fingerprint", ""),
    )


def save_global_settings(projects_dir: Path, settings: GlobalSettings) -> None:
    """Save global settings to projects/projects.yaml."""
    projects_dir.mkdir(parents=True, exist_ok=True)
    settings_file = projects_dir / GLOBAL_SETTINGS_FILENAME

    data = asdict(settings)
    with open(settings_file, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
