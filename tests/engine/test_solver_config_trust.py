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

#: A Gurobi option file with a full path (``NodefileDir``: a file parameter), as someone else's workspace may hold.
EDITED_GUROBI = "Threads 2\nNodefileDir /srv/elsewhere/nodes\n"


@pytest.fixture(autouse=True)
def user_home(tmp_path, monkeypatch) -> Path:
    """This test's own user config folder (so the list of trusted workspaces starts empty)."""
    home = tmp_path / "user_home"
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
    # Pointing the user config folder at the workspace is the user's own act (an environment variable), shown
    # here only to pin where the list is read from.
    monkeypatch.setenv("FLEXTOOL_USER_HOME", str(ws))
    assert resolve_solver_config_source(run_options=_options(), cwd=ws).trusted is True


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
