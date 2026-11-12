"""Scenario names become result folder names (``flextool.safe_names``): a name that is not one plain folder name is
refused where a run starts, before anything is written or removed, and the folder cleanups stay inside their folder.

The names a FlexTool user has (letters of any alphabet, spaces, punctuation, the GUI's ``<name>_<N>`` folders) pass.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flextool.safe_names import (
    UnsafeNameError,
    UnsafePathError,
    check_output_name,
    output_name_problem,
    require_inside,
    require_strictly_inside,
    safe_file_component,
)

REFUSED = ["", ".", "..", "../x", "a/b", "a\\b", "/tmp/x", "-x", "--settings-db-url=x", "a\nb", "\x1b[31mred",
           "tab\there", "C:x", "CON", "nul", "Aux.txt", "COM1", "lpt9.dat"]
ALLOWED = ["base", "Base_1", "S2 Dry", "Säätö 2030", "風力", "a.b", "x=1", "(test) #2", "a-b", "base..v2", ".hidden",
           "CONSOLE", "auxiliary", "wind+solar", "@home", "S1: base"]


@pytest.mark.parametrize("name", REFUSED)
def test_names_that_are_not_one_plain_folder_name_are_refused(name: str) -> None:
    assert output_name_problem(name, windows=False) is not None
    with pytest.raises(UnsafeNameError, match="cannot be used as a folder name"):
        check_output_name(name)


@pytest.mark.parametrize("name", ALLOWED)
def test_ordinary_names_pass(name: str) -> None:
    assert output_name_problem(name, windows=False) is None


@pytest.mark.parametrize("name", ["S1: base", "a?b", 'a"b', "a|b", "a*b", "a<b", "trailing.", "trailing "])
def test_windows_file_name_rules_apply_on_windows(name: str) -> None:
    assert output_name_problem(name, windows=True) is not None
    assert output_name_problem(name, windows=False) is None


def test_the_message_shows_control_characters_visibly() -> None:
    with pytest.raises(UnsafeNameError) as exc:
        check_output_name("a\x1b[2Jb")
    assert "'a\\x1b[2Jb'" in str(exc.value)  # one backslash, as Python writes it


def test_the_folder_guards(tmp_path: Path) -> None:
    root = tmp_path / "output_parquet"
    (root / "base_1").mkdir(parents=True)
    assert require_inside(root / "base_1", root) == (root / "base_1").resolve()
    assert require_inside(root, root) == root.resolve()
    with pytest.raises(UnsafePathError):
        require_inside(root / ".." / "elsewhere", root)
    with pytest.raises(UnsafePathError):
        require_inside(tmp_path, root)
    with pytest.raises(UnsafePathError):
        require_strictly_inside(root, root)
    if hasattr(os, "symlink"):
        outside = tmp_path / "outside"
        outside.mkdir()
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except OSError:
            return
        with pytest.raises(UnsafePathError):
            require_inside(root / "link", root)


@pytest.mark.parametrize(("text", "expected"), [
    ("../../x", ".._.._x"), ("a/b\\c", "a_b_c"), ("..", "_"), ("", "_"), ("ok name", "ok name"),
    ("bell\x07", "bell_"), ("Flows: unit to node", "Flows: unit to node"),
])
def test_plot_file_name_parts_stay_in_their_folder(text: str, expected: str) -> None:
    assert safe_file_component(text, windows=False) == expected
    assert safe_file_component("Flows: unit to node", windows=True) == "Flows_ unit to node"


@pytest.fixture()
def restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.mark.parametrize("name", ["../escape", "a/b", "-x", "line\nbreak", "ABSOLUTE"])
def test_a_run_with_such_a_name_stops_before_writing_anything(restore_env, monkeypatch, tmp_path: Path,
                                                              name: str) -> None:
    """``flextool-run`` / Spine Toolbox: the run stops with exit code 1 and nothing appears outside the project."""
    import flextool.engine
    from flextool.cli import cmd_run_flextool

    if name == "ABSOLUTE":
        name = str(tmp_path / "absolute_target")

    def must_not_run(*args, **kwargs):
        raise AssertionError("the model must not run")

    monkeypatch.setattr(flextool.engine, "run_chain_from_db", must_not_run)
    project = tmp_path / "project"
    project.mkdir()
    db = project / "input.sqlite"
    db.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", f"sqlite:///{db}", f"--scenario-name={name}", "--work-folder", str(project / "work"),
        "--output-location", str(project)])
    with pytest.raises(SystemExit) as exc:
        cmd_run_flextool.main()
    assert exc.value.code == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["project"]
    assert not (project / "output_parquet").exists()


def test_the_window_does_not_queue_such_a_name(tmp_path: Path) -> None:
    """The FlexTool window: no job (so no ``work/<name>_<N>`` cleanup) for a name that is not a folder name."""
    from flextool.gui.data_models import ProjectSettings, ScenarioInfo
    from flextool.gui.execution_manager import ExecutionManager

    (tmp_path / "input_sources").mkdir()
    mgr = ExecutionManager(project_path=tmp_path, settings=ProjectSettings())
    scenarios = [ScenarioInfo(name=n, source_number=1, source_name="a.sqlite") for n in ("../../x", "base")]
    jobs = mgr.add_jobs(scenarios)
    assert [j.scenario_name for j in jobs] == ["base"]
    assert [j.output_subdir for j in jobs] == ["base_1"]


def test_a_plot_name_from_the_settings_stays_in_the_plot_folder(tmp_path: Path) -> None:
    from flextool.plot_outputs.format_helpers import generate_split_filename

    plot_dir = tmp_path / "output_plots" / "base_1"
    for name, member in (("../../../evil", None), ("ok", "../x"), ("/abs/evil", "a\\b")):
        path = Path(generate_split_filename(name, str(plot_dir), "png", file_member=member))
        assert path.parent == plot_dir
