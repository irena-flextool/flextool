"""The GUI's Spine DB editor launch uses the current interpreter, not PATH.

``flextool-gui`` started from a venv that is not activated
(``.venv/bin/flextool-gui``) has no venv ``bin`` on PATH, so a
``shutil.which("spine-db-editor")`` lookup reported the editor missing
even though Spine Toolbox was installed (and importable) in that venv.
Every launcher now runs ``<sys.executable> -m
spinetoolbox.spine_db_editor.main <url>...`` after checking that
``spinetoolbox`` is importable.

No window is ever opened: ``subprocess.Popen`` is replaced by a recorder,
and Spine Toolbox is a stub package on ``sys.path`` (so the tests also
run where Spine Toolbox is not installed).
"""
from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from flextool.cli import cmd_open_results_db
from flextool.gui import db_editor_integration, platform_utils
from flextool.gui.db_editor_integration import DbEditorManager

MODULE = "spinetoolbox.spine_db_editor.main"


class _FakeProc:
    """Stands in for a running editor process."""

    def __init__(self, argv, **kwargs) -> None:
        self.argv = list(argv)
        self.kwargs = kwargs
        self.returncode = None

    def poll(self):
        return self.returncode


@pytest.fixture
def popen_calls(monkeypatch: pytest.MonkeyPatch) -> list[_FakeProc]:
    calls: list[_FakeProc] = []

    def fake_popen(argv, **kwargs):
        proc = _FakeProc(argv, **kwargs)
        calls.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return calls


@pytest.fixture
def toolbox_stub_no_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Spine Toolbox importable (stub package) while PATH has no
    ``spine-db-editor`` at all — the non-activated-venv situation."""
    pkg = tmp_path / "site" / "spinetoolbox"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    for name in list(sys.modules):
        if name == "spinetoolbox" or name.startswith("spinetoolbox."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    importlib.invalidate_caches()
    empty_bin = tmp_path / "empty_bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", str(empty_bin))


@pytest.fixture
def toolbox_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "spinetoolbox" or name.startswith("spinetoolbox."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_argv_is_current_interpreter_module() -> None:
    assert platform_utils.spine_db_editor_argv("sqlite:///a.sqlite") == [
        sys.executable, "-m", MODULE, "sqlite:///a.sqlite"]
    assert platform_utils.spine_db_editor_argv(
        ["sqlite:///a.sqlite", "sqlite:///b.sqlite"]) == [
        sys.executable, "-m", MODULE, "sqlite:///a.sqlite",
        "sqlite:///b.sqlite"]


def test_cli_launcher_shares_the_gui_argv() -> None:
    urls = ["sqlite:///a.sqlite", "sqlite:///b.sqlite"]
    assert (cmd_open_results_db._build_launch_argv(urls)
            == platform_utils.spine_db_editor_argv(urls))


def test_manager_launches_without_editor_on_path(
        toolbox_stub_no_path, popen_calls, tmp_path) -> None:
    mgr = DbEditorManager()
    try:
        proc = mgr.open_database("sqlite:///input.sqlite", "input.sqlite")
        assert proc is not None, "launch refused although spinetoolbox imports"
        assert [p.argv for p in popen_calls] == [
            [sys.executable, "-m", MODULE, "sqlite:///input.sqlite"]]
        # Process tracking and launch-log capture are kept.
        assert mgr.is_editor_running("input.sqlite")
        assert popen_calls[0].kwargs.get("stderr") == subprocess.STDOUT
        assert proc in mgr._launch_logs
    finally:
        mgr.cleanup()


def test_manager_opens_several_urls_in_one_editor(
        toolbox_stub_no_path, popen_calls) -> None:
    mgr = DbEditorManager()
    try:
        urls = ["sqlite:///a.sqlite", "sqlite:///b.sqlite"]
        assert mgr.open_database(urls, "pair") is not None
        assert popen_calls[0].argv == [sys.executable, "-m", MODULE, *urls]
    finally:
        mgr.cleanup()


def test_manager_warns_when_toolbox_not_importable(
        toolbox_missing, popen_calls, caplog) -> None:
    mgr = DbEditorManager()
    with caplog.at_level("WARNING", logger=db_editor_integration.__name__):
        assert mgr.open_database("sqlite:///x.sqlite", "x.sqlite") is None
    assert popen_calls == []
    assert "spinetoolbox" in caplog.text and sys.executable in caplog.text
    assert not mgr.is_editor_running("x.sqlite")


def test_platform_utils_launcher_matches(
        toolbox_stub_no_path, popen_calls) -> None:
    assert platform_utils.open_spine_db_editor("sqlite:///a.sqlite") is not None
    assert popen_calls[0].argv == [
        sys.executable, "-m", MODULE, "sqlite:///a.sqlite"]


def test_platform_utils_launcher_refuses_without_toolbox(
        toolbox_missing, popen_calls) -> None:
    assert platform_utils.open_spine_db_editor("sqlite:///a.sqlite") is None
    assert popen_calls == []
