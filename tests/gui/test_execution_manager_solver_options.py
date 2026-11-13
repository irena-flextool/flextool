"""The GUI → ``cmd_run_flextool`` solver flags, and the one-time update of older solver settings.

A GUI run sends a solver flag only for a setting the user changed, so it solves like a terminal run of the same
database: the default settings send no solver flag apart from ``--threads`` (the Execution jobs window's
"Threads per run") and ``--workspace-solver-config-dir`` (the workspace's option files; ``--solver-config-dir``
when the user set ``FLEXTOOL_SOLVER_CONFIG_DIR``).  The GUI has no MIP-gap setting.  A settings.yaml of an earlier GUI held the old defaults
(presolve ``choose``, MIP gap 0.001 sent with every run); loading it updates them once.

No Tk: the manager constructor only stores ``project_path`` / ``settings`` and reads package-bundled plot
configs, and the settings tests only read and write YAML.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from flextool.cli.cmd_run_flextool import build_parser
from flextool.gui.data_models import SOLVER_SETTINGS_REVISION, ProjectSettings
from flextool.gui.execution_manager import ExecutionJob, ExecutionManager
from flextool.gui.settings_io import (
    SETTINGS_FILENAME,
    load_project_settings,
    save_project_settings,
    solver_defaults_dialog_text,
    solver_defaults_notice,
)
from flextool.run_options import OPTIONS, RunOptions

_REGISTRY_FLAGS = {spec.flag for spec in OPTIONS.values() if spec.flag}


def _cmd_for(tmp_path: Path, settings: ProjectSettings | None = None) -> list[str]:
    """The engine command line of a scenario job under *settings*."""
    mgr = ExecutionManager(project_path=tmp_path, settings=settings or ProjectSettings())
    job = ExecutionJob(job_id=0, scenario_name="scen", output_subdir="scen", input_db_url="sqlite:///input.sqlite")
    return mgr._build_run_command(job, tmp_path / "work")


def _registry_flags_in(cmd: list[str]) -> set[str]:
    return {tok.split("=", 1)[0] for tok in cmd if tok.split("=", 1)[0] in _REGISTRY_FLAGS}


def _run_options(cmd: list[str]) -> RunOptions:
    assert cmd[1:4] == ["-P", "-m", "flextool.cli.cmd_run_flextool"]
    return RunOptions.from_cli(build_parser().parse_args(cmd[4:]), environ={})


def test_default_settings_send_only_the_thread_count(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    cmd = _cmd_for(tmp_path)
    assert _registry_flags_in(cmd) == {"--threads", "--workspace-solver-config-dir"}
    options = _run_options(cmd)
    assert {k for k, r in options.resolved.items() if r.source == "cli"} == {"threads", "workspace_solver_config_dir"}


def test_the_workspace_option_folder_is_sent_explicitly(tmp_path: Path, monkeypatch):
    """A GUI run names the workspace's ``solver_config`` (the GUI's current folder) as the workspace's folder
    (trusted only when this computer set the workspace up; item 240 B3), or the folder of
    ``FLEXTOOL_SOLVER_CONFIG_DIR`` as the user's own choice when that is set, so the run never looks for it near
    an external database."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    options = _run_options(_cmd_for(tmp_path))
    assert options.value("workspace_solver_config_dir") == (str(workspace / "solver_config"),)
    assert options.value("solver_config_dir") is None
    monkeypatch.setenv("FLEXTOOL_SOLVER_CONFIG_DIR", str(tmp_path / "chosen"))
    options = _run_options(_cmd_for(tmp_path))
    assert options.value("solver_config_dir") == (str(tmp_path / "chosen"),)
    assert options.value("workspace_solver_config_dir") is None


def test_the_project_option_folder_comes_first(tmp_path: Path, monkeypatch):
    """A project with its own ``solver_config`` gets it ahead of the workspace's, one flag per folder, both as
    folders the window found (trusted only when this computer set them up; item 240 B3);
    ``FLEXTOOL_SOLVER_CONFIG_DIR`` replaces only the workspace's folder (several, separated by os.pathsep, the
    user's own choice) and the run puts the found project folder above them."""
    import os

    from flextool._solver_config_source import resolve_solver_config_source

    workspace = tmp_path / "ws"
    project = workspace / "projects" / "P"
    (project / "solver_config").mkdir(parents=True)
    monkeypatch.chdir(workspace)
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    mgr = ExecutionManager(project_path=project, settings=ProjectSettings())
    job = ExecutionJob(job_id=0, scenario_name="scen", output_subdir="scen", input_db_url="sqlite:///input.sqlite")
    cmd = mgr._build_run_command(job, tmp_path / "work")
    assert _run_options(cmd).value("workspace_solver_config_dir") == (
        str(project / "solver_config"), str(workspace / "solver_config"))
    monkeypatch.setenv("FLEXTOOL_SOLVER_CONFIG_DIR", os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")]))
    cmd = mgr._build_run_command(job, tmp_path / "work")
    assert cmd.count("--solver-config-dir") == 2 and cmd.count("--workspace-solver-config-dir") == 1
    options = _run_options(cmd)
    assert options.value("workspace_solver_config_dir") == (str(project / "solver_config"),)
    assert options.value("solver_config_dir") == (str(tmp_path / "a"), str(tmp_path / "b"))
    ladder = resolve_solver_config_source(run_options=options)
    assert [folder for folder, _reason in ladder.layers] == [
        project / "solver_config", tmp_path / "a", tmp_path / "b"]


@pytest.mark.parametrize(
    ("field", "value", "key", "expected"),
    [
        ("presolve", "on", "presolve", "on"),
        ("presolve", "off", "presolve", "off"),
        ("presolve", "choose", "presolve", "choose"),
        ("scaling", "full", "scaling", "full"),
        ("scaling", "basic", "scaling", "basic"),
        ("scaling", "off", "scaling", "off"),
        ("solver_log_level", "verbose", "solver_log_level", "verbose"),
        ("solver_time_limit", 600, "solver_time_limit", 600.0),
    ],
)
def test_a_changed_setting_sends_exactly_its_flag(tmp_path: Path, field, value, key, expected, monkeypatch):
    """Each changed setting reaches the run as the same option a terminal user would set by hand."""
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    cmd = _cmd_for(tmp_path, ProjectSettings(**{field: value}))
    gui = _run_options(cmd)
    assert {k for k, r in gui.resolved.items() if r.source == "cli"} == {"threads", "workspace_solver_config_dir", key}
    by_hand = RunOptions.from_cli(
        build_parser().parse_args(["in.sqlite", OPTIONS[key].flag, str(value), "--threads", "1"]), environ={},
    )
    assert gui.get(key) == by_hand.get(key)
    assert gui.value(key) == expected


@pytest.mark.parametrize("per_run", [1, 6])
def test_threads_per_run_is_sent_as_the_thread_budget(tmp_path: Path, per_run: int):
    """The Execution jobs window's "Threads per run" (stored as ``max_cores_per_job``) reaches the run as
    ``--threads``, which gives HiGHS ``threads`` without its parallel simplex; ``--highs-threads`` is not sent."""
    from flextool.engine.solve._orchestration import _operator_highs_overrides
    from flextool.gui.data_models import ExecutionLimits

    settings = ProjectSettings()
    settings.execution_limits = ExecutionLimits(max_cores_per_job=per_run)
    cmd = _cmd_for(tmp_path, settings)
    assert "--highs-threads" not in cmd
    options = _run_options(cmd)
    assert (options.value("threads"), options.source("threads")) == (per_run, "cli")
    assert _operator_highs_overrides(options) == {"threads": per_run}


def test_calibration_sends_threads_per_run(tmp_path: Path):
    from flextool.calibrate._cli import build_parser as calibrate_parser
    from flextool.gui.calibrate_commands import build_calibrate_command

    argv = build_calibrate_command(
        "python", "sqlite:///in.sqlite", "scen", iterations=2, sizing="uniform", overshoot=1.0,
        damping_first=1.0, damping_remaining=0.5, stall_fraction=0.05, warm_start_cache_dir=None, work_dir=None,
        output_location=None, debug=False, threads=3, solver_config_dir="/ws/solver_config",
    )
    assert argv[1:4] == ["-P", "-m", "flextool.calibrate"]
    parsed = calibrate_parser().parse_args(argv[4:])
    assert parsed.threads == 3 and parsed.solver_config_dir == [Path("/ws/solver_config")]


def test_calibrator_solves_get_the_option_folder(tmp_path: Path):
    """The calibrator passes its ``--solver-config-dir`` on to each solve it starts."""
    from flextool.calibrate._solve import solve_argv

    argv = solve_argv("in.sqlite", "scen", work_dir=tmp_path / "w", out_root=tmp_path / "o", cache_dir=tmp_path / "c",
                      solver_config_dir=Path("/ws/solver_config"))
    assert argv[1:4] == ["-P", "-m", "flextool_run_supervisor"]
    options = RunOptions.from_cli(build_parser().parse_args(argv[4:]), environ={})
    assert (options.value("solver_config_dir"), options.source("solver_config_dir")) == (
        (str(Path("/ws/solver_config")),), "cli")
    # The FlexTool window's found folders are passed on as found (item 240 B3).
    argv = solve_argv("in.sqlite", "scen", work_dir=tmp_path / "w", out_root=tmp_path / "o", cache_dir=tmp_path / "c",
                      workspace_solver_config_dir=Path("/ws/solver_config"))
    options = RunOptions.from_cli(build_parser().parse_args(argv[4:]), environ={})
    assert options.value("solver_config_dir") is None
    assert options.value("workspace_solver_config_dir") == (str(Path("/ws/solver_config")),)


def test_the_gui_never_sends_a_mip_gap(tmp_path: Path):
    assert not hasattr(ProjectSettings(), "solver_mip_gap")
    assert not hasattr(ProjectSettings(), "solver_mip_gap_set")
    cmd = _cmd_for(tmp_path, ProjectSettings(presolve="off", scaling="basic", solver_time_limit=5))
    assert "--solver-mip-gap" not in _registry_flags_in(cmd)


# --- settings.yaml --------------------------------------------------------


def _write_settings(project: Path, data: dict) -> None:
    project.mkdir(parents=True, exist_ok=True)
    (project / SETTINGS_FILENAME).write_text(yaml.safe_dump(data), encoding="utf-8")


def _read_settings(project: Path) -> dict:
    return yaml.safe_load((project / SETTINGS_FILENAME).read_text(encoding="utf-8"))


def _load_counting(project: Path) -> tuple[ProjectSettings, int]:
    calls = []
    settings = load_project_settings(project, on_solver_defaults_reset=lambda: calls.append(1))
    return settings, len(calls)


def test_old_default_settings_are_updated_once(tmp_path: Path, monkeypatch):
    """An earlier GUI's defaults become "Default" once; after the save, a reload neither resets nor notifies."""
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    project = tmp_path / "proj"
    _write_settings(project, {"presolve": "choose", "scaling": "full",
                              "solver_mip_gap_set": True, "solver_mip_gap": 0.001})

    settings, notices = _load_counting(project)
    assert (settings.presolve, settings.scaling) == (None, None)
    assert notices == 1
    assert _registry_flags_in(_cmd_for(tmp_path, settings)) == {"--threads", "--workspace-solver-config-dir"}

    save_project_settings(project, settings)
    stored = _read_settings(project)
    assert stored["solver_settings_revision"] == SOLVER_SETTINGS_REVISION
    assert "solver_mip_gap" not in stored and "solver_mip_gap_set" not in stored

    # The user now picks presolve "choose" on purpose: it is kept and no notice repeats.
    settings.presolve = "choose"
    save_project_settings(project, settings)
    again, notices = _load_counting(project)
    assert again.presolve == "choose" and notices == 0


@pytest.mark.parametrize("presolve", ["on", "off"])
def test_a_deliberate_old_presolve_is_kept(tmp_path: Path, presolve: str):
    project = tmp_path / "proj"
    _write_settings(project, {"presolve": presolve, "scaling": "basic", "solver_mip_gap_set": False})
    settings, notices = _load_counting(project)
    assert (settings.presolve, settings.scaling) == (presolve, "basic")
    # Nothing the old file sent changes: no notice.
    assert notices == 0


def test_an_old_ticked_mip_gap_alone_gives_the_notice(tmp_path: Path):
    project = tmp_path / "proj"
    _write_settings(project, {"presolve": "on", "solver_mip_gap_set": True, "solver_mip_gap": 0.01})
    settings, notices = _load_counting(project)
    assert settings.presolve == "on" and notices == 1


def test_a_new_project_has_no_notice_and_records_the_revision(tmp_path: Path):
    project = tmp_path / "proj"
    settings, notices = _load_counting(project)
    assert notices == 0 and (settings.presolve, settings.scaling) == (None, None)
    save_project_settings(project, settings)
    assert _load_counting(project)[1] == 0


def test_the_notice_names_both_changes():
    text = solver_defaults_notice("my_project")
    assert "presolve is on" in text
    assert "solver_mip_gap" in text and "highs.opt" in text
    assert "my_project" in text


def test_the_dialog_shows_the_notice_items_on_their_own_lines():
    lines = solver_defaults_dialog_text("my_project").splitlines()
    assert lines[0].startswith("Behaviour changed in ") and "presolve is on" in lines[0]
    assert lines[1] == "Affected projects: my_project."
    assert lines[2].startswith("This notice is shown once.")
    assert " ".join(lines) == solver_defaults_notice("my_project")   # the same content as the printed line


def test_settings_io_tolerates_stale_lagrangian_workers_key(tmp_path: Path):
    # The retired subgradient-workers knob may still linger in old settings.yaml files: ignored on load.
    project = tmp_path / "proj"
    save_project_settings(project, ProjectSettings(solver_time_limit=7))
    data = _read_settings(project)
    data["lagrangian_workers"] = 4
    _write_settings(project, data)
    loaded = load_project_settings(project)
    assert loaded.solver_time_limit == 7
    assert not hasattr(loaded, "lagrangian_workers")


def test_a_stored_matrix_file_format_is_ignored(tmp_path: Path):
    """The removed matrix-file-format option of older files loads without error, is not sent and is not saved."""
    project = tmp_path / "proj"
    _write_settings(project, {"matrix_file_format": "lp", "solver_settings_revision": SOLVER_SETTINGS_REVISION})
    settings = load_project_settings(project)
    assert not hasattr(settings, "matrix_file_format")
    assert not any(tok.startswith("--matrix-file-format") for tok in _cmd_for(tmp_path, settings))
    save_project_settings(project, settings)
    assert "matrix_file_format" not in _read_settings(project)
