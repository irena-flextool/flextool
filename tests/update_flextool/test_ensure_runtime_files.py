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
    # calls ``shutil.copy`` by module attribute, so patching it here reaches
    # that call site.
    import shutil

    real_copy = shutil.copy

    def _flaky_copy(src, dst, *a, **k):
        if str(dst).endswith(".opt"):
            raise OSError("simulated read-only solver_config")
        return real_copy(src, dst, *a, **k)

    monkeypatch.setattr(shutil, "copy", _flaky_copy)
    # Should not raise despite the .opt copies failing.
    ensure_runtime_files()
    # The independent project-folder guide still got written.
    assert (workspace / "templates" / "project_folder.txt").is_file()
