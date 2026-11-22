"""W-0490: a migration history that the spinedb_api 0.31-0.34 value-list bug may have hit (design D, rev 3).

Each database is built from the current schema, then its ``commit`` table gets the version commits of the case.
The expected findings are hand-derived from the design's rule: a session (consecutive version commits a → N) that
ends in 2024-04-29 .. 2025-07-07 and whose first list-value step in (a, N] appends to an existing list.
"""
from __future__ import annotations

import inspect
import re
import shutil
import sqlite3
from pathlib import Path

import pytest

from flextool import run_warnings
from flextool.update_flextool import db_migration
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool.value_list_history import Finding
from tests._support.coded_warnings import capture_coded

_PREFIX = "Updated Flextool data structure to version "


def _add_commits(path: Path, commits: list[tuple[str, str]]) -> None:
    con = sqlite3.connect(path)
    with con:
        con.executemany('INSERT INTO "commit" (comment, date, user) VALUES (?, ?, ?)',
                        [("Added new parameter value lists" if v == "list" else _PREFIX + v, f"{d} 12:00:00",
                          "spinedb_api") for v, d in commits])
    con.close()


@pytest.fixture
def make(schema_db_url: str, tmp_path: Path):
    template = Path(schema_db_url.removeprefix("sqlite:///"))

    def _make(commits: list[tuple[str, str]], name: str = "model.sqlite", folder: Path = tmp_path) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        shutil.copy2(template, path)
        _add_commits(path, commits)
        return path
    return _make


CYPRUS = [("20", "2024-05-27"), ("21", "2024-07-17"), ("22", "2024-09-19")]


@pytest.mark.parametrize("commits, expected", [
    pytest.param(CYPRUS, [Finding(22, ("invest_methods",), 21, 22, "2024-09-19")], id="a_v21_to_v22"),
    pytest.param([("19", "2024-08-28"), ("22", "2024-08-28")], [], id="b_template_19_to_22"),
    pytest.param([("21", "2024-07-17"), ("22", "2025-08-01")], [], id="c_after_window"),
    pytest.param([], [], id="d_json_built"),
    pytest.param([("21", "2024-07-17"), ("23", "2025-04-14")],
                 [Finding(22, ("invest_methods",), 21, 23, "2025-04-14")], id="e_v21_to_v23"),
    pytest.param([("3.0", "2024-05-01"), ("10", "2024-05-02"), ("10", "2024-05-03"), ("19", "2024-05-04")], [],
                 id="f_duplicates_and_float"),
    pytest.param([("21", "2024-07-17"), ("22", "2025-07-07")],
                 [Finding(22, ("invest_methods",), 21, 22, "2025-07-07")], id="window_end_inclusive"),
    pytest.param([("21", "2024-07-17"), ("22", "2025-07-08")], [], id="day_after_window_end"),
    pytest.param([("21", "2024-04-01"), ("22", "2024-04-28")], [], id="day_before_window_start"),
    pytest.param([("22", "2024-09-19")], [], id="first_commit_has_no_start"),
    pytest.param([("21", "2024-07-17"), ("list", "2024-09-19"), ("list", "2025-08-01"), ("22", "2025-08-01")],
                 [Finding(22, ("invest_methods",), 21, 22, "2024-09-19")], id="failed_in_window_retried_after"),
    pytest.param([("21", "2024-07-17"), ("list", "2025-08-01"), ("22", "2025-08-01")], [],
                 id="list_step_after_window"),
])
def test_findings(make, commits, expected) -> None:
    assert vlh.possible_value_list_overwrite_ro(make(commits)) == expected


def test_an_unreadable_file_gives_no_finding(tmp_path: Path) -> None:
    bad = tmp_path / "x.sqlite"
    bad.write_bytes(b"not a database")
    assert vlh.possible_value_list_overwrite_ro(bad) == []
    assert vlh.possible_value_list_overwrite_ro(tmp_path / "missing.sqlite") == []


def _step_blocks() -> dict[int, str]:
    """``{N: source of step N}``, the helper functions it calls included (three levels deep)."""
    parts = re.split(r"elif next_version == (\d+):", inspect.getsource(db_migration._migrate_open))

    def expand(text: str, depth: int, seen: set[str]) -> str:
        out = text
        for name in sorted(set(re.findall(r"\b(\w+)\(", text)) - seen):
            func = getattr(db_migration, name, None)
            if depth and inspect.isfunction(func) and func.__module__ == db_migration.__name__:
                seen.add(name)
                out += expand(inspect.getsource(func), depth - 1, seen)
        return out
    return {int(parts[i]): expand(parts[i + 1], 3, set()) for i in range(1, len(parts), 2)}


def test_touch_and_append_match_the_migration_steps() -> None:
    """Every step whose code imports or looks up list values is in TOUCH (53 binds a list to a definition, the
    design's survey); every APPEND list is named by its step."""
    marks = ("add_value_list_manual(", "add_new_parameters(", "change_optional_output_type(", '"list_value"',
             "'list_value'", "update_timestructure(")
    blocks = _step_blocks()
    touching = {n for n, text in blocks.items() if any(m in text for m in marks)}
    assert touching - vlh.TOUCH == set()
    assert vlh.TOUCH - touching - set(range(1, 12)) <= {53}
    for step, lists in vlh.APPEND.items():
        assert "add_value_list_manual(" in blocks[step]
        assert all(f'"{name}"' in blocks[step] for name in lists), step


def test_warning_once_per_run(make) -> None:
    path = make(CYPRUS)
    run_warnings.start_run()
    with capture_coded() as got:
        vlh.warn_at_run_start(f"sqlite:///{path}")
        vlh.warn_at_run_start(f"sqlite:///{path}")
    assert got.lines() == ["Warning: a spinedb_api bug may have changed the value list on 2024-09-19 "
                           "('invest_methods') [W-0490]"]
    assert got.detail("W-0490", "invest_methods").startswith("migrated from data version 21 to 22 on 2024-09-19")
    run_warnings.start_run()
    with capture_coded() as got:
        vlh.warn_at_run_start(f"sqlite:///{path}")
    assert len(got.lines()) == 1, "a new run warns again (no silencing)"


def test_no_warning_for_an_unaffected_history(make) -> None:
    with capture_coded() as got:
        vlh.warn_at_run_start(f"sqlite:///{make([('19', '2024-08-28'), ('22', '2024-08-28')])}")
    assert got.calls == []


@pytest.mark.solver
def test_a_run_warns_once_at_its_start(tmp_path_factory) -> None:
    """The engine run (``run_chain_from_db``) gives W-0490 once for its input database."""
    from tests.engine._unit_db import make_db, run

    url, scenario = make_db(tmp_path_factory, alt="w0490", values=[], base_chain=["Base", "lossless_2way"])
    _add_commits(Path(url.removeprefix("sqlite:///")), CYPRUS)
    with capture_coded() as got:
        run(tmp_path_factory, url, scenario)
    assert [c.names for c in got.calls if c.code == "W-0490"] == [["invest_methods"]]
    assert sum("[W-0490]" in line for line in got.lines()) == 1


def test_gui_shows_the_text_once_per_opened_project(make, tmp_path: Path) -> None:
    from flextool.gui.data_models import ProjectSettings
    from flextool.gui.input_sources import InputSourceManager

    project = tmp_path / "project"
    make(CYPRUS, "cyprus.sqlite", project / "input_sources")
    make([("19", "2024-08-28"), ("22", "2024-08-28")], "fine.sqlite", project / "input_sources")
    mgr = InputSourceManager(project, ProjectSettings())
    messages = mgr.plan_db_migrations()[3]
    assert not [m for m in messages if "W-0490" in m], "not with the migration messages (their own box)"
    flagged = mgr.take_check_findings()
    assert [(name, f.step, f.date) for name, f in flagged] == [("cyprus.sqlite", 22, "2024-09-19")]
    assert not mgr.take_check_findings(), "taken once"
    mgr.plan_db_migrations()
    assert not mgr.take_check_findings(), "a refresh does not repeat it"
    again = InputSourceManager(project, ProjectSettings())
    again.plan_db_migrations()
    assert len(again.take_check_findings()) == 1, "opening it again does"


def test_gui_text_for_one_file() -> None:
    text = vlh.gui_dialog_text([("cyprus.sqlite", Finding(22, ("invest_methods",), 21, 22, "2024-09-19"))])
    assert text == (
        "cyprus.sqlite may contain wrong investment settings.\n\n"
        "When it was upgraded on 19.9.2024, a bug in a library FlexTool uses may have changed `not_allowed` into "
        "`cumulative_limits`. Units, connections or nodes that should not invest could then invest without "
        "limit.\n\n"
        "The documentation has a short checklist for finding and fixing this.")


def test_gui_text_lists_every_flagged_file_once() -> None:
    inv = ("invest_methods",)
    text = vlh.gui_dialog_text([
        ("a.sqlite", Finding(22, inv, 21, 22, "2024-09-19")),
        ("b.sqlite", Finding(22, inv, 21, 30, "2025-01-05")),
        ("b.sqlite", Finding(22, inv, 21, 30, "2025-01-05")),
        ("c.sqlite", Finding(27, ("minimum_time_methods",), 26, 30, "2025-03-01")),
    ])
    paragraphs = text.split("\n\n")
    assert paragraphs[0] == ("These databases may contain wrong investment settings:\n"
                             "  a.sqlite (upgraded on 19.9.2024)\n  b.sqlite (upgraded on 5.1.2025)")
    assert paragraphs[1].startswith("When they were upgraded, a bug in a library FlexTool uses may have changed")
    assert paragraphs[2] == "c.sqlite may contain wrong settings in the value list `minimum_time_methods`."
    assert paragraphs[3].startswith("When it was upgraded on 1.3.2025, a bug")
    assert paragraphs[-1] == "The documentation has a short checklist for finding and fixing this."
    assert text.count("checklist") == 1


def test_gui_planning_survives_an_unexpected_history_error(make, tmp_path: Path, monkeypatch) -> None:
    """An unexpected error of the check (implausible data) is logged, and the migration planning goes on."""
    from flextool.gui.data_models import ProjectSettings
    from flextool.gui.input_sources import InputSourceManager

    project = tmp_path / "project"
    make(CYPRUS, "cyprus.sqlite", project / "input_sources")

    def boom(path):
        raise OverflowError("cannot convert float infinity to integer")

    monkeypatch.setattr(vlh, "possible_value_list_overwrite_ro", boom)
    mgr = InputSourceManager(project, ProjectSettings())
    assert len(mgr.plan_db_migrations()) == 4
    assert mgr.take_check_findings() == []


def test_guard_stops_an_import_that_rewrites_a_member(schema_db_url: str, tmp_path: Path, monkeypatch) -> None:
    """The bug's effect (an add that updates the first member in place) stops the migration step."""
    from flextool.update_flextool.db_handle import open_database_mapping

    path = tmp_path / "g.sqlite"
    shutil.copy2(schema_db_url.removeprefix("sqlite:///"), path)

    def buggy_import(db, parameter_value_lists):
        first = db.get_items("list_value", parameter_value_list_name="invest_methods")[0]
        new = parameter_value_lists[0][1]
        assert db.update_item("list_value", id=first["id"], value=f'"{new}"'.encode(), type="str")[0]
        return 1, []

    with open_database_mapping(f"sqlite:///{path}") as db:
        db_migration.add_value_list_manual(db, [["invest_methods", "first_new"]])   # the real import: fine
        monkeypatch.setattr(db_migration, "import_data", buggy_import)
        with pytest.raises(db_migration.FlexToolMigrationError, match="invest_methods: 'not_allowed'"):
            db_migration.add_value_list_manual(db, [["invest_methods", "second_new"]])
