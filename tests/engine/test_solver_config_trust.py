"""Item 240 B3: option files that a run finds on its own are trusted only in a workspace this computer set up.

A ``solver_config/`` found next to the input database, in the workspace of a project database, in the Spine Toolbox
project folder, in the current folder or named by the FlexTool window (``--workspace-solver-config-dir``) has its
files trusted (file paths, server settings) only when the folder it is in is listed in this computer's
``solver_settings.json`` (``trusted_workspaces``, in the user config folder).  ``ensure_runtime_files`` lists its
workspace only when it creates ``solver_config/`` or finds nothing in it but FlexTool's own default files.
Otherwise each line gets the input database's rules.  A folder the user names (``--solver-config-dir``,
``FLEXTOOL_SOLVER_CONFIG_DIR``) stays trusted.

The expected outcomes follow from those rules by hand: which folder is listed, and which lines the database's rules
refuse (a full path, a server parameter, a name not in FlexTool's list) or pass (a plain file name, a listed name).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from flextool import _local_solver_settings as settings
from flextool._solver_config_source import (
    REASON_CWD,
    REASON_DB_FOLDER,
    REASON_ENV,
    REASON_FLAG,
    REASON_WORKSPACE,
    read_solver_option_files,
    resolve_solver_config_source,
    untrusted_note,
)
from flextool.engine.solve._solver_dispatch import (
    FlexToolUserError,
    _resolve_effective_highs_options,
    reset_run_warnings,
    resolve_commercial_options,
)
from flextool.run_options import Resolved, RunOptions

posix_only = pytest.mark.skipif(os.name == "nt", reason="inodes and file modes as on Linux and macOS")

#: A Gurobi option file with a full path (``NodefileDir``: a file parameter), as someone else's workspace may hold.
EDITED_GUROBI = "Threads 2\nNodefileDir /srv/elsewhere/nodes\n"


@pytest.fixture(autouse=True)
def user_home(tmp_path_factory, monkeypatch) -> Path:
    """This test's own user config folder (so the list of trusted workspaces starts empty), outside the test's
    folder: a ``FLEXTOOL_USER_HOME`` inside the current folder is ignored (review b2b3 L2)."""
    home = tmp_path_factory.mktemp("user_home")
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(home))
    monkeypatch.delenv("FLEXTOOL_SOLVER_CONFIG_DIR", raising=False)
    monkeypatch.delenv("FLEXTOOL_ALLOW_UNLISTED_SOLVER_PARAMETERS", raising=False)
    reset_run_warnings()
    return home


def _options(**flags: str) -> RunOptions:
    return RunOptions(resolved={key: Resolved(value, "cli") for key, value in flags.items()})


def _received_workspace(root: Path, gurobi_text: str = EDITED_GUROBI) -> Path:
    """A workspace that came from elsewhere: FlexTool's workspace marker and an edited ``gurobi.opt``."""
    (root / "templates").mkdir(parents=True)
    (root / "templates" / "project_folder.txt").write_text("")
    (root / "solver_config").mkdir()
    (root / "solver_config" / "gurobi.opt").write_text(gurobi_text)
    return root


def _gurobi_file_layer(source) -> tuple[str, str, str | None]:
    files = read_solver_option_files(source, ("gurobi",))
    return files.text("gurobi"), files.origin("gurobi"), files.untrusted_note("gurobi")


# ---------------------------------------------------------------------------
# Which folders are trusted
# ---------------------------------------------------------------------------


def test_a_received_workspace_with_edited_files_is_not_trusted(tmp_path, monkeypatch) -> None:
    """Opening a received workspace (``ensure_runtime_files`` at the window's start or ``flextool-update``) does not
    list it, so a run from there gives its edited ``gurobi.opt`` the database's rules: the full path is refused,
    naming the file, the line and how to trust the folder; nothing reached the solver."""
    from flextool.update_flextool import ensure_runtime_files

    ws = _received_workspace(tmp_path / "received")
    monkeypatch.chdir(ws)
    ensure_runtime_files()
    assert (ws / "solver_config" / "gurobi.opt").read_text() == EDITED_GUROBI  # never overwritten
    assert settings.trusted_workspaces() == ()
    source = resolve_solver_config_source(run_options=_options(), cwd=ws)
    assert (source.dir, source.reason, source.trusted) == (ws / "solver_config", REASON_CWD, False)
    text, origin, note = _gurobi_file_layer(source)
    with pytest.raises(FlexToolUserError) as err:
        resolve_commercial_options("gurobi", opt_file=origin, opt_text=text, opt_untrusted=note,
                                   solver_config=None, run_options=_options())
    message = str(err.value)
    assert f"{ws / 'solver_config' / 'gurobi.opt'}, line 2" in message
    assert "'/srv/elsewhere/nodes'" in message and "plain file name" in message
    assert "--solver-config-dir" in message and "Trust this workspace's solver option files" in message
    # The GUI names the same folder as its workspace: the same rules.
    source = resolve_solver_config_source(run_options=_options(workspace_solver_config_dir=str(ws / "solver_config")))
    assert (source.reason, source.trusted) == (REASON_WORKSPACE, False)


def test_the_marker_cannot_be_forged_by_files_inside_the_workspace(tmp_path, monkeypatch) -> None:
    """A received workspace that ships FlexTool's own settings file (at the workspace root, under ``config/`` as in
    the user config folder, and in ``solver_config/``) naming itself as trusted is still not trusted: the list is
    read only from this computer's user config folder."""
    from flextool.update_flextool import ensure_runtime_files

    ws = _received_workspace(tmp_path / "received")
    forged = json.dumps({"format": 1, "trusted_workspaces": [str(ws), settings.workspace_key(ws)]})
    for place in (ws, ws / "config", ws / "solver_config"):
        place.mkdir(exist_ok=True)
        (place / "solver_settings.json").write_text(forged)
    monkeypatch.chdir(ws)
    ensure_runtime_files()
    assert settings.trusted_workspaces() == ()
    assert resolve_solver_config_source(run_options=_options(), cwd=ws).trusted is False
    # FLEXTOOL_USER_HOME naming the workspace (the current folder) is ignored with a warning, so the forged list
    # is not read even then (review b2b3 L2); the platform's folder (here XDG_CONFIG_HOME) is used instead.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(ws))
    assert settings.settings_path() == tmp_path / "xdg" / "flextool" / "solver_settings.json"
    assert resolve_solver_config_source(run_options=_options(), cwd=ws).trusted is False


def test_a_workspace_set_up_here_is_listed_once_and_trusted(tmp_path, monkeypatch) -> None:
    """``ensure_runtime_files`` in an empty folder creates ``solver_config/`` and lists the folder (twice called,
    one entry); a later edit by the user stays trusted, so a full path in the file reaches the solver."""
    from flextool.update_flextool import ensure_runtime_files

    ws = tmp_path / "fresh"
    ws.mkdir()
    monkeypatch.chdir(ws)
    ensure_runtime_files()
    ensure_runtime_files()
    assert settings.trusted_workspaces() == (settings.workspace_key(ws),)
    (ws / "solver_config" / "gurobi.opt").write_text(EDITED_GUROBI)
    source = resolve_solver_config_source(run_options=_options(), cwd=ws)
    assert (source.dir, source.trusted, source.trust_note) == (ws / "solver_config", True, None)
    text, origin, note = _gurobi_file_layer(source)
    options = resolve_commercial_options("gurobi", opt_file=origin, opt_text=text, opt_untrusted=note,
                                         solver_config=None, run_options=_options())
    assert [(o.name, o.value) for o in options] == [("Threads", "2"), ("NodefileDir", "/srv/elsewhere/nodes")]


def test_a_workspace_holding_only_flextool_defaults_is_listed(tmp_path, monkeypatch) -> None:
    """A ``solver_config/`` that holds only unedited FlexTool templates (an install of an earlier version) is
    FlexTool's own: listed.  One edited file, or a symbolic link, is not."""
    from flextool._resources import package_data_path
    from flextool.update_flextool import ensure_runtime_files

    template = Path(package_data_path("solver_config/highs.opt.template")).read_bytes()
    own = tmp_path / "own"
    (own / "solver_config").mkdir(parents=True)
    (own / "solver_config" / "highs.opt").write_bytes(template)
    (own / "solver_config" / "notes.txt").write_text("not an option file")
    linked = tmp_path / "linked"
    (linked / "solver_config").mkdir(parents=True)
    (linked / "solver_config" / "highs.opt").symlink_to(own / "solver_config" / "highs.opt")
    for ws in (own, linked):
        monkeypatch.chdir(ws)
        ensure_runtime_files()
    assert settings.trusted_workspaces() == (settings.workspace_key(own),)


def test_named_folders_stay_trusted_and_a_listed_folder_matches_through_a_symlink(tmp_path) -> None:
    """``--solver-config-dir`` and ``FLEXTOOL_SOLVER_CONFIG_DIR`` are the user's choice: trusted, listed or not.
    A database's folder is trusted once the database's folder is listed, also when reached through a symlink."""
    ws = _received_workspace(tmp_path / "ws")
    flag = resolve_solver_config_source(run_options=_options(solver_config_dir=str(ws / "solver_config")))
    assert (flag.reason, flag.trusted) == (REASON_FLAG, True)
    env = resolve_solver_config_source(environ={"FLEXTOOL_SOLVER_CONFIG_DIR": str(ws / "solver_config")})
    assert (env.reason, env.trusted) == (REASON_ENV, True)

    db = ws / "model.sqlite"
    db.write_bytes(b"")
    found = resolve_solver_config_source(run_options=_options(), input_db_url=f"sqlite:///{db}", cwd=tmp_path)
    assert (found.reason, found.trusted) == (REASON_DB_FOLDER, False)
    alias = tmp_path / "alias"
    alias.symlink_to(ws)
    assert settings.add_trusted_workspace(alias) is True
    assert settings.trusted_workspaces() == (settings.workspace_key(ws),)
    found = resolve_solver_config_source(run_options=_options(), input_db_url=f"sqlite:///{db}", cwd=tmp_path)
    assert (found.reason, found.trusted) == (REASON_DB_FOLDER, True)
    settings.remove_trusted_workspace(ws)
    assert resolve_solver_config_source(run_options=_options(), cwd=ws).trusted is False


def test_a_damaged_settings_file_leaves_the_folder_untrusted_and_says_so(tmp_path, user_home) -> None:
    ws = _received_workspace(tmp_path / "ws")
    (user_home / "config").mkdir(parents=True)
    settings.settings_path().write_text("{not json")
    source = resolve_solver_config_source(run_options=_options(), cwd=ws)
    assert source.trusted is False
    assert "is not valid JSON" in source.trust_note


# ---------------------------------------------------------------------------
# What the database's rules do to an untrusted file
# ---------------------------------------------------------------------------

#: An untrusted ``gurobi.opt`` line, and the part of the refusal (``None``: it passes) set by the database's rules.
UNTRUSTED_GUROBI_LINES = [
    ("LogFile run.log\n", None),  # a plain file name: in the solve's work folder, where Gurobi runs
    ("NodefileDir /tmp/x\n", "plain file name"),
    ("CloudAccessID abc\n", "connects the solver to another computer"),
    ("MIPGapp 0.01\n", "not in FlexTool's list of Gurobi parameters"),
    ("LogFile gurobi.env\n", "a solver reads by itself"),
]


@pytest.mark.parametrize(("line", "refused"), UNTRUSTED_GUROBI_LINES)
def test_untrusted_gurobi_lines_get_the_database_rules(line, refused) -> None:
    note = untrusted_note(Path("/ws/solver_config"), REASON_CWD)

    def resolve(untrusted):
        return resolve_commercial_options("gurobi", opt_file="/ws/solver_config/gurobi.opt", opt_text=line,
                                          opt_untrusted=untrusted, solver_config=None, run_options=_options())

    trusted = resolve(None)  # the same line in a trusted file passes (an unlisted name with a warning)
    assert len(trusted) == 1
    if refused is None:
        assert [(o.name, o.value) for o in resolve(note)] == [("LogFile", "run.log")]
        return
    with pytest.raises(FlexToolUserError) as err:
        resolve(note)
    message = str(err.value)
    assert "/ws/solver_config/gurobi.opt, line 1" in message and refused in message
    assert message.endswith(note)


def test_untrusted_highs_file_names_go_to_the_work_folder(tmp_path) -> None:
    """HiGHS: an untrusted ``highs.opt`` with ``log_file=/tmp/x.log`` is refused; ``log_file=run.log`` becomes a
    file in the solve's work folder.  The in-process solve and the ``--save-memory`` child both take this option
    set (``_orchestration._solve_highs_options``)."""
    work = tmp_path / "solve_data" / "subprocess"
    with pytest.raises(FlexToolUserError, match=r"highs.opt, line 1.*plain file name.*NOTE$"):
        _resolve_effective_highs_options(
            solver_arguments_map=None, highs_opt_path=Path("highs.opt"), highs_opt_text="log_file=/tmp/x.log\n",
            cli_overrides=None, workspace=work, highs_opt_untrusted="NOTE")
    got = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=Path("highs.opt"), highs_opt_text="log_file=run.log\n",
        cli_overrides=None, workspace=work, highs_opt_untrusted="NOTE")
    assert got == {"log_file": str(work / "run.log")}
    trusted = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=Path("highs.opt"), highs_opt_text="log_file=/tmp/x.log\n",
        cli_overrides=None, workspace=work)
    assert trusted == {"log_file": "/tmp/x.log"}


def test_the_run_record_says_whether_the_folder_was_trusted(tmp_path) -> None:
    ws = _received_workspace(tmp_path / "ws")
    source = resolve_solver_config_source(run_options=_options(), cwd=ws)
    files = read_solver_option_files(source)
    assert files.record()["trusted"] is False
    assert files.untrusted_note("gurobi") == source.trust_note
    # The folder has no highs.opt: HiGHS takes FlexTool's packaged default, which is trusted.
    assert files.untrusted_note("highs") is None
    settings.add_trusted_workspace(ws)
    source = resolve_solver_config_source(run_options=_options(), cwd=ws)
    assert read_solver_option_files(source).record()["trusted"] is True


# ---------------------------------------------------------------------------
# The file-by-file ladder (item 231): trust per folder, so per file
# ---------------------------------------------------------------------------

#: A project's own gurobi.opt (from someone else, or not yet trusted), with a full path.
PROJECT_GUROBI = "Threads 3\nNodefileDir /srv/received/nodes\n"


def _gui_workspace_with_project(root: Path) -> tuple[Path, Path]:
    """A workspace this computer set up (listed) whose ``gurobi.opt`` names a path and whose ``highs.opt`` sets a
    log file outside the work folder, and a project with its own ``gurobi.opt`` (a path too)."""
    ws = _received_workspace(root / "ws", gurobi_text=EDITED_GUROBI)
    (ws / "solver_config" / "highs.opt").write_text("log_file=/var/log/flextool/highs.log\n")
    project = ws / "projects" / "P"
    (project / "solver_config").mkdir(parents=True)
    (project / "solver_config" / "gurobi.opt").write_text(PROJECT_GUROBI)
    settings.add_trusted_workspace(ws)
    return ws, project


def _window_ladder(project: Path, ws: Path):
    """The ladder a FlexTool window run of *project* resolves: the window's flags, parsed by the run's parser."""
    from flextool._solver_config_source import gui_solver_config_flags
    from flextool.cli.cmd_run_flextool import build_parser

    flags = gui_solver_config_flags(project, environ={}, cwd=ws)
    assert flags == ["--workspace-solver-config-dir", str(project / "solver_config"),
                     "--workspace-solver-config-dir", str(ws / "solver_config")]
    options = RunOptions.from_cli(build_parser().parse_args(["sqlite:///x.sqlite", *flags]), environ={})
    return resolve_solver_config_source(run_options=options, cwd=ws)


def test_an_untrusted_project_file_over_a_trusted_workspace_gets_the_database_rules(tmp_path) -> None:
    """The workspace is trusted, the project is not: the project's ``gurobi.opt`` (which wins file by file) gets
    the database's rules, so its full path is refused with the project's note; the workspace's ``highs.opt`` (no
    project file for HiGHS) keeps the workspace's trust and its full path."""
    ws, project = _gui_workspace_with_project(tmp_path)
    source = _window_ladder(project, ws)
    assert [(folder, reason) for folder, reason in source.layers] == [
        (project / "solver_config", "project of the FlexTool window"),
        (ws / "solver_config", "workspace of the FlexTool window")]
    assert (source.trusted, source.layer_note(1)) == (False, None)
    files = read_solver_option_files(source)
    assert files.origin("gurobi") == str(project / "solver_config" / "gurobi.opt")
    note = files.untrusted_note("gurobi")
    assert note is not None and "Trust this project's solver option files" in note
    with pytest.raises(FlexToolUserError) as err:
        resolve_commercial_options("gurobi", opt_file=files.origin("gurobi"), opt_text=files.text("gurobi"),
                                   opt_untrusted=note, solver_config=None, run_options=_options())
    assert f"{project / 'solver_config' / 'gurobi.opt'}, line 2" in str(err.value)
    assert "'/srv/received/nodes'" in str(err.value)
    assert files.untrusted_note("highs") is None
    got = _resolve_effective_highs_options(
        solver_arguments_map=None, highs_opt_path=Path(files.origin("highs")), highs_opt_text=files.text("highs"),
        cli_overrides=None, highs_opt_untrusted=files.untrusted_note("highs"))
    assert got == {"log_file": "/var/log/flextool/highs.log"}
    record = files.record()
    assert [layer["trusted"] for layer in record["layers"]] == [False, True]
    assert (record["files"]["gurobi.opt"]["trusted"], record["files"]["highs.opt"]["trusted"]) == (False, True)
    assert "not a project this computer set up" in source.describe()
    # The database layout of a project database gives the same ladder and the same trust.
    db = project / "input_sources" / "model.sqlite"
    db.parent.mkdir()
    db.write_bytes(b"")
    by_db = resolve_solver_config_source(run_options=_options(), input_db_url=f"sqlite:///{db}", cwd=tmp_path)
    assert (by_db.trusted, by_db.layer_note(1)) == (False, None)
    assert read_solver_option_files(by_db).untrusted_note("gurobi") is not None
    # Trusting the project (the window's box) lets its own path through.
    settings.add_trusted_workspace(project)
    files = read_solver_option_files(_window_ladder(project, ws))
    assert files.untrusted_note("gurobi") is None


def test_named_folders_stay_trusted_below_an_untrusted_found_project_folder(tmp_path) -> None:
    """``FLEXTOOL_SOLVER_CONFIG_DIR`` set for the window: its folder replaces the workspace's and is trusted (the
    user's choice); the project's folder stays on top as a found folder, judged on its own."""
    from flextool._solver_config_source import gui_solver_config_ladder

    ws, project = _gui_workspace_with_project(tmp_path)
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    (chosen / "cplex.opt").write_text("workdir /scratch/cplex\n")
    found, named = gui_solver_config_ladder(project, environ={"FLEXTOOL_SOLVER_CONFIG_DIR": str(chosen)}, cwd=ws)
    assert (found, named) == ([project / "solver_config"], [chosen])
    options = RunOptions(resolved={"workspace_solver_config_dir": Resolved(tuple(map(str, found)), "cli"),
                                   "solver_config_dir": Resolved(tuple(map(str, named)), "cli")})
    files = read_solver_option_files(resolve_solver_config_source(run_options=options, cwd=ws))
    assert files.untrusted_note("gurobi") is not None and files.untrusted_note("cplex") is None
    assert files.origin("cplex") == str(chosen / "cplex.opt")


def test_a_shipped_template_in_an_untrusted_folder_is_flextools_own(tmp_path) -> None:
    """A file that is byte for byte a shipped template is trusted wherever it lies (it is FlexTool's own)."""
    from flextool._resources import package_data_path

    ws = _received_workspace(tmp_path / "ws")
    template = Path(package_data_path("solver_config/gurobi.opt.template")).read_text(encoding="utf-8")
    (ws / "solver_config" / "gurobi.opt").write_text(template, encoding="utf-8")
    files = read_solver_option_files(resolve_solver_config_source(run_options=_options(), cwd=ws))
    assert files.source.trusted is False and files.untrusted_note("gurobi") is None


def test_a_relative_folder_named_like_the_default_stays_untrusted(tmp_path, monkeypatch) -> None:
    """Whether a file is FlexTool's own is decided by its content (a shipped template's hash), never by the text
    of its origin or a folder's name (review b2b3 M3): a folder whose name starts with "FlexTool default" is
    judged as any other, and an edited text whose origin claims to be the default is not trusted."""
    from dataclasses import replace

    monkeypatch.chdir(tmp_path)
    folder = Path("FlexTool default x") / "solver_config"
    folder.mkdir(parents=True)
    (folder / "gurobi.opt").write_text(EDITED_GUROBI)
    source = resolve_solver_config_source(run_options=_options(workspace_solver_config_dir=str(folder)), cwd=tmp_path)
    files = read_solver_option_files(source)
    assert files.untrusted_note("gurobi") is not None
    forged = replace(files, origins={"gurobi": "FlexTool default (gurobi.opt.template)"})
    assert forged.untrusted_note("gurobi") is not None


# ---------------------------------------------------------------------------
# Projects this computer set up
# ---------------------------------------------------------------------------


def test_a_project_is_noted_only_when_nothing_received_waits_in_it(tmp_path) -> None:
    """The window notes a project it creates or opens when its ``solver_config/`` is missing or holds only
    FlexTool's defaults and no received file waits for review; edited files, a held ``.review`` file or a 4.4
    development build's ``imported_solver_config/`` keep it off the list."""
    from flextool._solver_config_source import forget_trusted_folder, note_project_set_up_here

    projects = tmp_path / "ws" / "projects"
    fresh, edited, held, old = (projects / name for name in ("Fresh", "Edited", "Held", "Old"))
    for project in (fresh, edited, held, old):
        project.mkdir(parents=True)
    (edited / "solver_config").mkdir()
    (edited / "solver_config" / "gurobi.opt").write_text(PROJECT_GUROBI)
    (held / "solver_config").mkdir()
    (held / "solver_config" / "gurobi.opt.review").write_text(PROJECT_GUROBI)
    (old / "imported_solver_config").mkdir()
    assert [note_project_set_up_here(p) for p in (fresh, edited, held, old)] == [True, False, False, False]
    assert note_project_set_up_here(fresh) is False  # already listed
    assert settings.trusted_workspaces() == (settings.workspace_key(fresh),)
    forget_trusted_folder(fresh)
    assert settings.trusted_workspaces() == ()


def test_creating_and_renaming_a_project_keep_its_trust_with_it(tmp_path, monkeypatch) -> None:
    from flextool.gui.project_utils import create_project, rename_project

    monkeypatch.chdir(tmp_path)
    project = create_project("Mine")
    assert settings.is_trusted_workspace(project)
    renamed = rename_project("Mine", "Ours")
    assert settings.is_trusted_workspace(renamed) and not settings.is_trusted_workspace(project)
    # A folder left at a new project's path with received option files is not noted.
    leftover = tmp_path / "projects" / "Left" / "solver_config"
    leftover.mkdir(parents=True)
    (leftover / "gurobi.opt").write_text(PROJECT_GUROBI)
    assert not settings.is_trusted_workspace(create_project("Left"))


# ---------------------------------------------------------------------------
# Review b2b3: the trust is for a folder and its files, not for a path
# ---------------------------------------------------------------------------


def _trusted_workspace_file(root: Path) -> tuple[Path, Path]:
    """A workspace this computer trusted with its edited ``gurobi.opt``; returns it and the file."""
    ws = _received_workspace(root)
    settings.add_trusted_workspace(ws)
    return ws, ws / "solver_config" / "gurobi.opt"


def _gurobi_note(ws: Path) -> str | None:
    return read_solver_option_files(resolve_solver_config_source(run_options=_options(), cwd=ws)).untrusted_note(
        "gurobi")


@posix_only
def test_a_folder_put_at_a_trusted_path_is_not_trusted(tmp_path) -> None:
    """M1: the trusted workspace is deleted and a received copy is unpacked at the same path; the copy is another
    folder (another inode), so its files get the database's rules and the note says the folder was replaced.
    Trusting it again (the box, the command) binds the new folder."""
    import shutil

    ws, _file = _trusted_workspace_file(tmp_path / "ws")
    assert _gurobi_note(ws) is None
    shutil.rmtree(ws)  # ext4 gives the next folder the same inode number: the birth time tells them apart
    _received_workspace(ws, gurobi_text="Threads 2\nResultFile /home/u/.bashrc\n")
    note = _gurobi_note(ws)
    assert note is not None and "is not the one this computer trusted" in note
    assert settings.trust_status(ws).problem is not None
    settings.update_settings_file(lambda data: settings.apply_settings(data, workspace=ws, trust_workspace=True))
    assert _gurobi_note(ws) is None


@posix_only
def test_a_later_edit_is_trusted_only_when_no_other_account_can_change_the_file(tmp_path) -> None:
    """M1: a file changed after the folder was trusted stays trusted when this account owns it and no other
    account can write it (the user's own edit, no new tick); writable by every account it gets the database's
    rules, naming the file, until the folder is trusted again.  A file unchanged since the tick stays trusted
    whatever its mode."""
    ws, path = _trusted_workspace_file(tmp_path / "ws")
    path.chmod(0o646)
    assert _gurobi_note(ws) is None  # as it was when trusted
    path.write_text("Threads 4\nNodefileDir /srv/other\n")
    path.chmod(0o644)
    assert _gurobi_note(ws) is None  # the user's own edit
    path.chmod(0o646)
    note = _gurobi_note(ws)
    assert note is not None and "gurobi.opt changed after this computer trusted the folder" in note
    assert "every account can write it" in note and "flextool-trust-workspace" in note
    settings.add_trusted_workspace(ws)  # already listed, the same folder: no new binding
    assert _gurobi_note(ws) is not None
    settings.update_settings_file(lambda data: settings.apply_settings(data, workspace=ws, trust_workspace=True))
    assert _gurobi_note(ws) is None


def test_a_text_entry_of_a_development_build_is_bound_to_the_folder_now_there(tmp_path, user_home) -> None:
    """Migration: a plain path entry (4.4 development builds) is bound, on its first read, to the folder at that
    path with its current files; a folder put there afterwards is then not trusted."""
    ws = _received_workspace(tmp_path / "ws")
    (user_home / "config").mkdir(parents=True, exist_ok=True)
    settings.settings_path().write_text(json.dumps({"format": 1, "trusted_workspaces": [str(ws)]}))
    assert settings.is_trusted_workspace(ws)
    (entry,) = json.loads(settings.settings_path().read_text())["trusted_workspaces"]
    assert entry["path"] == settings.workspace_key(ws)
    assert entry["files"] == {"gurobi.opt": settings._sha256(EDITED_GUROBI.encode())}
    assert entry["id"] == list(settings.folder_identity(ws))


def test_a_rename_carries_the_entry_without_trusting_a_change(tmp_path) -> None:
    """A renamed project keeps its entry (identity and hashes), so a change the old entry did not cover is still
    not covered at the new path."""
    ws, path = _trusted_workspace_file(tmp_path / "old")
    new = tmp_path / "new"
    ws.rename(new)
    settings.move_trusted_workspace(ws, new)
    assert settings.trusted_workspaces() == (settings.workspace_key(new),)
    assert settings.trust_status(new).file_problem("gurobi.opt", settings._sha256(EDITED_GUROBI.encode()), None) is None


@pytest.mark.parametrize("entry", [".", "~/flextool", "relative/ws"])
def test_a_relative_list_entry_is_refused(tmp_path, user_home, monkeypatch, entry) -> None:
    """L1: "." would trust every folder a run starts from; such an entry makes the list unreadable, with a message
    asking for full folder paths, so nothing is trusted."""
    ws = _received_workspace(tmp_path / "ws")
    monkeypatch.chdir(ws)
    (user_home / "config").mkdir(parents=True, exist_ok=True)
    settings.settings_path().write_text(json.dumps({"format": 1, "trusted_workspaces": [entry]}))
    with pytest.raises(settings.LocalSolverSettingsError, match="not a full folder path"):
        settings.is_trusted_workspace(ws)
    source = resolve_solver_config_source(run_options=_options(), cwd=ws)
    assert source.trusted is False and "Write full folder paths" in source.trust_note


def test_a_relative_or_workspace_user_home_is_ignored(tmp_path, tmp_path_factory, monkeypatch, caplog) -> None:
    """L2: a relative FLEXTOOL_USER_HOME (or XDG_CONFIG_HOME), or one inside the current folder, is ignored with
    a warning; the platform's folder is used.  One that names a folder inside a database's folder judged for
    trust (not the current folder) makes that folder's trust unreadable."""
    from flextool import user_dirs

    # Every fallback lands in this test's folders, never in the real user folders.
    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    xdg = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.chdir(tmp_path)
    for value in ("user_home", str(tmp_path / "inside")):
        monkeypatch.setenv("FLEXTOOL_USER_HOME", value)
        with caplog.at_level("WARNING", logger="flextool.user_dirs"):
            assert user_dirs.user_config_dir() == xdg / "flextool"
        assert f"Ignoring FLEXTOOL_USER_HOME={value}" in caplog.text
    monkeypatch.setenv("XDG_CONFIG_HOME", "xdg")
    monkeypatch.setenv("FLEXTOOL_USER_HOME", "")
    assert user_dirs.user_config_dir() == Path.home() / ".config" / "flextool"
    assert Path.home() != Path(os.path.expanduser("~" + os.environ.get("USER", "")))
    # A user home inside a database's folder (not the current folder) is refused for that folder.
    ws = _received_workspace(tmp_path / "ws")
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(ws / "carried"))
    assert settings.settings_path() == ws / "carried" / "config" / "solver_settings.json"
    settings.settings_path().parent.mkdir(parents=True)
    settings.settings_path().write_text(json.dumps({"format": 1, "trusted_workspaces": [str(ws)]}))
    db = ws / "model.sqlite"
    db.write_bytes(b"")
    found = resolve_solver_config_source(run_options=_options(), input_db_url=f"sqlite:///{db}", cwd=tmp_path)
    assert found.trusted is False and "lies inside" in found.trust_note


def test_a_windows_junction_counts_as_a_link(tmp_path, monkeypatch) -> None:
    """L5: a ``solver_config`` that is a Windows junction (or other reparse point) is never FlexTool's own folder,
    so the workspace it is in is not listed by being opened."""
    from flextool._solver_config_source import holds_only_flextool_defaults

    folder = tmp_path / "ws" / "solver_config"
    folder.mkdir(parents=True)
    assert holds_only_flextool_defaults(folder)
    monkeypatch.setattr(Path, "is_junction", lambda self: self == folder, raising=False)
    assert settings.is_link_like(folder) and not holds_only_flextool_defaults(folder)


def test_the_trust_command_shows_the_files_and_asks(tmp_path, monkeypatch, capsys) -> None:
    """L6: ``flextool-trust-workspace <folder>`` prints the edited option files with their lines and trusts the
    folder only on "y"; ``--remove`` takes it off."""
    from flextool.cli.cmd_trust_workspace import EXIT_OK, EXIT_REFUSED, main

    ws = _received_workspace(tmp_path / "ws")
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main([str(ws)]) == EXIT_REFUSED
    out = capsys.readouterr().out
    assert "gurobi.opt" in out and "NodefileDir /srv/elsewhere/nodes" in out
    assert not settings.is_trusted_workspace(ws)
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert main([str(ws / "solver_config")]) == EXIT_OK
    assert settings.is_trusted_workspace(ws) and _gurobi_note(ws) is None
    assert main(["--remove", str(ws)]) == EXIT_OK
    assert not settings.is_trusted_workspace(ws)
