"""Resolution + launch-wiring tests for the Open-results-DB launcher.

These tests exercise ONLY the URL-resolution logic and the
launch/missing-file branching of ``flextool.cli.cmd_open_results_db``.
They never open a real Spine DB editor: the subprocess launch is captured
via a fake ``Popen`` (and the public ``main`` path is monkeypatched so no
process is spawned).
"""
import sys

import pytest

from flextool.cli import cmd_open_results_db as mod


def _write_project_folder_file(tmp_path, project_dir):
    """Seed a templates/project_folder.txt pointing at ``project_dir``."""
    templates = tmp_path / "templates"
    templates.mkdir()
    pf = templates / "project_folder.txt"
    pf.write_text(str(project_dir) + "\n", encoding="utf-8")
    return pf


def test_resolve_results_db_path_absolute_project(tmp_path):
    project = tmp_path / "MyProject"
    project.mkdir()
    pf = _write_project_folder_file(tmp_path, project)

    db_path = mod.resolve_results_db_path(str(pf))

    assert db_path == project / "results.sqlite"


def test_resolve_results_db_path_blank_file_anchors_repo_root(tmp_path):
    # A blank/comment-only file anchors at the file's .parent.parent
    # (the repo root) — matching the run Tool's tier-2 fallback.
    templates = tmp_path / "templates"
    templates.mkdir()
    pf = templates / "project_folder.txt"
    pf.write_text("# no project folder here\n", encoding="utf-8")

    db_path = mod.resolve_results_db_path(str(pf))

    assert db_path == tmp_path / "results.sqlite"


def test_to_sqlite_url(tmp_path):
    p = tmp_path / "results.sqlite"
    assert mod._to_sqlite_url(p) == f"sqlite:///{p.as_posix()}"


def test_build_launch_argv_uses_db_editor_module():
    argv = mod._build_launch_argv("sqlite:///x/results.sqlite")
    assert argv == [
        sys.executable,
        "-m",
        "spinetoolbox.spine_db_editor.main",
        "sqlite:///x/results.sqlite",
    ]


def test_launch_db_editor_builds_detached_argv():
    captured = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return "PROC"

    result = mod.launch_db_editor(
        "sqlite:///p/results.sqlite", _popen=fake_popen
    )

    assert result == "PROC"
    assert captured["argv"] == [
        sys.executable,
        "-m",
        "spinetoolbox.spine_db_editor.main",
        "sqlite:///p/results.sqlite",
    ]
    # Detached: must request a new session so the GUI outlives the Tool.
    assert captured["kwargs"].get("start_new_session") is True


def test_main_launches_when_results_db_exists(tmp_path, monkeypatch, capsys):
    project = tmp_path / "MyProject"
    project.mkdir()
    (project / "results.sqlite").write_text("", encoding="utf-8")
    pf = _write_project_folder_file(tmp_path, project)

    launched = {}
    monkeypatch.setattr(
        mod, "launch_db_editor", lambda url: launched.setdefault("url", url)
    )

    rc = mod.main(["--project-folder-file", str(pf)])

    assert rc == 0
    expected_url = mod._to_sqlite_url(project / "results.sqlite")
    assert launched["url"] == expected_url
    assert "Opening Spine DB editor" in capsys.readouterr().out


def test_main_missing_results_db_reports_and_does_not_launch(
    tmp_path, monkeypatch, capsys
):
    project = tmp_path / "MyProject"
    project.mkdir()  # no results.sqlite inside
    pf = _write_project_folder_file(tmp_path, project)

    def fail_launch(_url):  # pragma: no cover - must NOT be called
        pytest.fail("launch_db_editor must not be called for a missing DB")

    monkeypatch.setattr(mod, "launch_db_editor", fail_launch)

    rc = mod.main(["--project-folder-file", str(pf)])

    assert rc == 1
    err = capsys.readouterr().err
    assert "results.sqlite not found" in err
    assert str(project / "results.sqlite") in err


def test_main_explicit_results_db_url_bare_path(
    tmp_path, monkeypatch
):
    db = tmp_path / "custom_results.sqlite"
    db.write_text("", encoding="utf-8")

    launched = {}
    monkeypatch.setattr(
        mod, "launch_db_editor", lambda url: launched.setdefault("url", url)
    )

    rc = mod.main(["--results-db-url", str(db)])

    assert rc == 0
    assert launched["url"] == mod._to_sqlite_url(db)


def test_main_explicit_results_db_url_missing(tmp_path, monkeypatch, capsys):
    db = tmp_path / "nope.sqlite"

    def fail_launch(_url):  # pragma: no cover
        pytest.fail("must not launch when explicit DB path is missing")

    monkeypatch.setattr(mod, "launch_db_editor", fail_launch)

    rc = mod.main(["--results-db-url", str(db)])

    assert rc == 1
    assert "results.sqlite not found" in capsys.readouterr().err
