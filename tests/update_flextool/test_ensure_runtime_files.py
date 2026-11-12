"""`ensure_runtime_files` idempotently materializes the per-installation
runtime files that FlexTool needs but does not track in git.

It is the single create-if-missing seeder shared by `update_flextool`
(CLI / Spine Toolbox) and the GUI startup path, so a fresh install — or
an update that adds a new template — gets everything in place without the
user having to run "Update FlexTool…" first.

The function operates relative to the current working directory, so these
tests chdir into a throwaway workspace.
"""
from __future__ import annotations

import sys

import pytest

from flextool.update_flextool import ensure_runtime_files

# Lightweight, dependency-free artifacts we assert on directly.  (The
# function also materializes the heavier example/settings SQLites; those
# are exercised by running the real function below, but we keep the
# assertions focused on the user-editable runtime files.)
SOLVERS = ("highs", "gurobi", "cplex", "xpress", "copt")


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_seeds_full_scope_from_scratch(workspace):
    ensure_runtime_files()
    # Live solver options — HiGHS + the four commercial solvers.
    for s in SOLVERS:
        assert (workspace / "solver_config" / f"{s}.opt").is_file(), s
    # Project-folder guide + the root user DBs + results DB.
    assert (workspace / "templates" / "project_folder.txt").is_file()
    for db in (
        "input_data.sqlite", "output_settings.sqlite", "output_info.sqlite",
        "comparison_settings.sqlite", "results.sqlite",
    ):
        assert (workspace / db).is_file(), db


def test_opt_matches_bundled_template(workspace):
    from flextool._resources import package_data_path

    ensure_runtime_files()
    for s in SOLVERS:
        seeded = (workspace / "solver_config" / f"{s}.opt").read_text()
        template = package_data_path(
            f"solver_config/{s}.opt.template"
        ).read_text()
        assert seeded == template, s


def test_idempotent_preserves_user_edits(workspace):
    ensure_runtime_files()
    gurobi = workspace / "solver_config" / "gurobi.opt"
    gurobi.write_text(gurobi.read_text() + "MyEdit 9\n")
    folder_txt = workspace / "templates" / "project_folder.txt"
    mtime_before = folder_txt.stat().st_mtime

    ensure_runtime_files()  # second run must not overwrite anything

    assert "MyEdit 9" in gurobi.read_text()
    assert folder_txt.stat().st_mtime == mtime_before


def test_missing_opt_is_reseeded_but_others_left_alone(workspace):
    # Models the "update added / user deleted one file" case: only the
    # absent file is recreated; an edited sibling is untouched.
    ensure_runtime_files()
    (workspace / "solver_config" / "copt.opt").unlink()
    highs = workspace / "solver_config" / "highs.opt"
    highs.write_text("custom\n")

    ensure_runtime_files()

    assert (workspace / "solver_config" / "copt.opt").is_file()
    assert highs.read_text() == "custom\n"


def test_resilient_when_a_step_cannot_write(workspace, monkeypatch):
    # A failure in one group (here: solver_config seeding) must be logged
    # and swallowed, leaving the rest of the seeding to complete.  self_update
    # calls ``shutil.copyfile`` by module attribute, so patching it here
    # reaches that call site.
    import shutil

    real_copyfile = shutil.copyfile

    def _flaky_copyfile(src, dst, *a, **k):
        if str(dst).endswith(".opt"):
            raise OSError("simulated read-only solver_config")
        return real_copyfile(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copyfile", _flaky_copyfile)
    # Should not raise despite the .opt copies failing.
    ensure_runtime_files()
    assert not list((workspace / "solver_config").glob("*.opt"))
    # The independent project-folder guide still got written.
    assert (workspace / "templates" / "project_folder.txt").is_file()


# ---------------------------------------------------------------------------
# Item 233: an unedited copy of an EARLIER shipped template is replaced by the current one; an edited file never.
# The "earlier template" is a literal whose hash the test computes itself and puts in the shipped-hash list.
# ---------------------------------------------------------------------------

_OLD = b"# old header\nThreads 0\n"


def _template_bytes(solver: str) -> bytes:
    from flextool._resources import package_data_path

    return package_data_path(f"solver_config/{solver}.opt.template").read_bytes()


@pytest.fixture
def old_gurobi_known(monkeypatch):
    import hashlib

    from flextool import _solver_config_source as scs

    known = dict(scs.SHIPPED_TEMPLATE_SHA256)
    known["gurobi"] = frozenset({hashlib.sha256(_OLD).hexdigest()})
    monkeypatch.setattr(scs, "SHIPPED_TEMPLATE_SHA256", known)


def _refresh(workspace, **kwargs):
    from pathlib import Path

    from flextool.update_flextool.self_update import _refresh_solver_config

    (workspace / "solver_config").mkdir(exist_ok=True)
    _refresh_solver_config(Path("solver_config"), **kwargs)


def test_unedited_earlier_default_is_replaced_at_gui_start(workspace, old_gurobi_known):
    """Through ``ensure_runtime_files`` (what the FlexTool window runs at start)."""
    (workspace / "solver_config").mkdir()
    (workspace / "solver_config" / "gurobi.opt").write_bytes(_OLD)
    ensure_runtime_files()
    assert (workspace / "solver_config" / "gurobi.opt").read_bytes() == _template_bytes("gurobi")
    assert not list((workspace / "solver_config").glob("*.tmp-*"))


def test_crlf_copy_of_an_earlier_default_is_replaced(workspace, old_gurobi_known):
    (workspace / "solver_config").mkdir()
    path = workspace / "solver_config" / "gurobi.opt"
    path.write_bytes(_OLD.replace(b"\n", b"\r\n"))
    path.chmod(0o600)
    _refresh(workspace)
    assert path.read_bytes() == _template_bytes("gurobi")
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o777 == 0o600  # a private file stays private


def test_edited_file_is_kept_byte_for_byte(workspace, old_gurobi_known):
    edited = b"# old header\nThreads 4\n"
    (workspace / "solver_config").mkdir()
    (workspace / "solver_config" / "gurobi.opt").write_bytes(edited)
    _refresh(workspace)
    assert (workspace / "solver_config" / "gurobi.opt").read_bytes() == edited


def test_copy_of_the_current_default_is_not_rewritten(workspace):
    import os

    (workspace / "solver_config").mkdir()
    path = workspace / "solver_config" / "highs.opt"
    path.write_bytes(_template_bytes("highs"))
    os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    _refresh(workspace)
    assert path.stat().st_mtime_ns == 1_000_000_000


def test_failed_replace_keeps_the_file_and_goes_on(workspace, old_gurobi_known, monkeypatch, caplog):
    import logging
    import os

    from flextool.update_flextool import self_update

    (workspace / "solver_config").mkdir()
    (workspace / "solver_config" / "gurobi.opt").write_bytes(_OLD)

    def _locked(src, dst):
        raise PermissionError("file in use")

    monkeypatch.setattr(self_update.os, "replace", _locked)
    with caplog.at_level(logging.INFO):
        _refresh(workspace, warn=True)
    assert (workspace / "solver_config" / "gurobi.opt").read_bytes() == _OLD
    assert not list((workspace / "solver_config").glob("*.tmp-*"))
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "gurobi.opt" in warnings[0].getMessage()
    # The other solvers were still seeded.
    for s in ("highs", "cplex", "xpress", "copt"):
        assert (workspace / "solver_config" / f"{s}.opt").is_file(), s
    # At the window start (no warn) the same failure is an INFO line only.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        _refresh(workspace)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert os.path.exists("solver_config/gurobi.opt")


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need a privilege on Windows")
def test_symlinked_earlier_default_is_left_alone(workspace, old_gurobi_known, tmp_path_factory):
    """A linked option file (shared by a team) stays a link, and its target is not touched."""
    shared = tmp_path_factory.mktemp("shared") / "gurobi.opt"
    shared.write_bytes(_OLD)
    (workspace / "solver_config").mkdir()
    link = workspace / "solver_config" / "gurobi.opt"
    link.symlink_to(shared)
    _refresh(workspace)
    assert link.is_symlink() and link.resolve() == shared.resolve()
    assert shared.read_bytes() == _OLD


def test_seeded_file_is_writable_when_the_template_is_read_only(workspace, monkeypatch, tmp_path_factory):
    import stat

    from flextool.update_flextool import self_update

    src = tmp_path_factory.mktemp("pkg") / "solver_config"
    src.mkdir()
    for s in SOLVERS:
        (src / f"{s}.opt.template").write_bytes(_template_bytes(s))
        (src / f"{s}.opt.template").chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    monkeypatch.setattr(self_update, "package_data_path", lambda rel: src.parent / rel)
    _refresh(workspace)
    seeded = workspace / "solver_config" / "gurobi.opt"
    assert seeded.read_bytes() == _template_bytes("gurobi")
    assert seeded.stat().st_mode & stat.S_IWUSR
