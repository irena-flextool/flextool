"""The GUI's "Write plotted data (CSV)" setting: kept in settings.yaml and sent as ``--write-plot-data``.

The first test needs no Tk: the manager constructor only stores ``project_path`` / ``settings``, and the settings
only read and write YAML.  The checkbox test opens a withdrawn Tk root (``--run-gui``, under ``xvfb-run -a``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

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
    assert cmd[1:4] == ["-P", "-m", "flextool.cli.cmd_run_flextool"]
    assert build_parser().parse_args(cmd[4:]).write_plot_data is True


@pytest.mark.gui
def test_checkbox_shows_a_plot_settings_yaml_on(tmp_path: Path):
    """``write_plot_data: true`` in plot_settings.yaml: the checkbox is checked and disabled with a hint, and the
    project's own setting (off) is what the dialog saves."""
    import tkinter as tk

    from flextool.gui.dialogs.plot_dialog import PLOT_DATA_ON_IN_YAML_HINT, _PlotSection

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available")
    root.withdraw()
    try:
        def section():
            return _PlotSection(root, label="Plot settings:", settings=PlotSettings(write_plot_data=False),
                                default_config_file="templates/default_plots.yaml", project_path=tmp_path)

        plain = section()
        assert not plain._write_plot_data_var.get()
        assert not plain._write_plot_data_check.instate(["disabled"])

        (tmp_path / "plot_settings.yaml").write_text("write_plot_data: true\n", encoding="utf-8")
        on = section()
        assert on._write_plot_data_var.get()
        assert on._write_plot_data_check.instate(["disabled"])
        assert PLOT_DATA_ON_IN_YAML_HINT in str(on._write_plot_data_check.cget("text"))
        assert on.collect().write_plot_data is False
    finally:
        root.destroy()
