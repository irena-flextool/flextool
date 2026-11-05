"""The GUI → ``cmd_run_flextool`` solver flags, and the one-time update of older solver settings.

A GUI run sends a solver flag only for a setting the user changed, so it solves like a terminal run of the same
database: the default settings send no solver flag apart from ``--highs-threads`` (the Execution jobs window's
per-job core budget).  The GUI has no MIP-gap setting.  A settings.yaml of an earlier GUI held the old defaults
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
    assert cmd[1:3] == ["-m", "flextool.cli.cmd_run_flextool"]
    return RunOptions.from_cli(build_parser().parse_args(cmd[3:]), environ={})


def test_default_settings_send_only_the_thread_count(tmp_path: Path):
    cmd = _cmd_for(tmp_path)
    assert _registry_flags_in(cmd) == {"--highs-threads"}
    options = _run_options(cmd)
    assert {k for k, r in options.resolved.items() if r.source == "cli"} == {"highs_threads"}


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
def test_a_changed_setting_sends_exactly_its_flag(tmp_path: Path, field, value, key, expected):
    """Each changed setting reaches the run as the same option a terminal user would set by hand."""
    cmd = _cmd_for(tmp_path, ProjectSettings(**{field: value}))
    gui = _run_options(cmd)
    assert {k for k, r in gui.resolved.items() if r.source == "cli"} == {"highs_threads", key}
    by_hand = RunOptions.from_cli(
        build_parser().parse_args(["in.sqlite", OPTIONS[key].flag, str(value), "--highs-threads", "1"]), environ={},
    )
    assert gui.get(key) == by_hand.get(key)
    assert gui.value(key) == expected


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


def test_old_default_settings_are_updated_once(tmp_path: Path):
    """An earlier GUI's defaults become "Default" once; after the save, a reload neither resets nor notifies."""
    project = tmp_path / "proj"
    _write_settings(project, {"presolve": "choose", "scaling": "full",
                              "solver_mip_gap_set": True, "solver_mip_gap": 0.001})

    settings, notices = _load_counting(project)
    assert (settings.presolve, settings.scaling) == (None, None)
    assert notices == 1
    assert _registry_flags_in(_cmd_for(tmp_path, settings)) == {"--highs-threads"}

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
