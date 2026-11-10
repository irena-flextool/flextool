"""The GUI's "Write plotted data (CSV)" setting: kept in settings.yaml and sent as ``--write-plot-data``.

No Tk: the manager constructor only stores ``project_path`` / ``settings``, and the settings only read and write YAML.
"""
from __future__ import annotations

from pathlib import Path

from flextool.cli.cmd_run_flextool import build_parser
from flextool.gui.data_models import PlotSettings, ProjectSettings
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager
from flextool.gui.settings_io import load_project_settings, save_project_settings


def _run_cmd(tmp_path: Path, settings: ProjectSettings) -> list[str]:
    mgr = ExecutionManager(project_path=tmp_path, settings=settings)
    job = ExecutionJob(job_id=0, scenario_name="scen", output_subdir="scen", input_db_url="sqlite:///input.sqlite")
    return mgr._build_run_command(job, tmp_path / "work")


def test_write_plot_data_round_trips_and_reaches_the_run(tmp_path: Path):
    assert "--write-plot-data" not in _run_cmd(tmp_path, ProjectSettings())

    settings = ProjectSettings()
    settings.single_plot_settings = PlotSettings(write_plot_data=True)
    settings.comparison_plot_settings = PlotSettings(write_plot_data=True)
    save_project_settings(tmp_path, settings)
    loaded = load_project_settings(tmp_path)
    assert loaded.single_plot_settings.write_plot_data
    assert loaded.comparison_plot_settings.write_plot_data

    cmd = _run_cmd(tmp_path, loaded)
    assert cmd[1:3] == ["-m", "flextool.cli.cmd_run_flextool"]
    assert build_parser().parse_args(cmd[3:]).write_plot_data is True
