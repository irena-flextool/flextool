"""The GUI "database upgraded" message names the behaviour notices the upgrade added (notices design §1, G1-G7).

No Tk.  Databases are built from the schema JSON (``schema_db_url``) and copied into ``tmp_path``; markers are
written with plain ``sqlite3``.  Expected texts are written out by hand from the design and the registry's console
texts, not taken from the code under test.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from flextool.engine._behaviour_changes import MARKER_PREFIX, added_notices, undecided_names
from flextool.gui import message_sources as ms
from flextool.gui.messages import MESSAGE_SPECS, GuiMessage, MessageCenter, database_key
from flextool.gui.message_sources import UpgradeResult
from flextool.update_flextool import frozen
from tests.gui.test_gui_messages import _check_style

UPGRADED = MESSAGE_SPECS["database_upgraded"]
OLDER = "An older FlexTool may not be able to run it any more."
RESULTS_HEADING = "These FlexTool changes affect the results of this model:"
NEUTRAL_HEADING = "The upgrade added these behaviour notes:"
CLOSING = ("Each run names the affected entities in its Notes; delete a note's marker in the database's Metadata tab "
           "once checked.")
UNDECIDED = ("Since the upgrade did not decide the minimum up/down methods for the following units, they are "
             "currently operated without those limits even though they have minimum up/down time parameter values. "
             "Please choose a method to activate the limits.")
UNDECIDED_NODES = ("Since the upgrade did not decide the storage start and end methods for the following nodes, a run "
                   "that uses them stops with an error: their storage_start_end_method fixes a start or an end without "
                   "its value. Please set storage_state_start or storage_state_end, or choose another method.")
W0600 = "W-0600: storage starts free by default (fix_nothing)"
W0601 = "W-0601: the upgrade rewrote storage_start_end_method"
W0634 = "W-0634: the upgrade wrote minimum_time_method"
W0635 = "W-0635: rolling within_solve realizes the jump in one solve"


def _msg(facts: dict) -> GuiMessage:
    title, paragraphs = ms.upgraded_text(facts)
    return GuiMessage(UPGRADED, "k", "x.sqlite", title, paragraphs, instance="r")


def _db(schema_db_url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(schema_db_url.removeprefix("sqlite:///"), path)
    return path


def _sql(path: Path, *statements: tuple[str, tuple]) -> None:
    con = sqlite3.connect(path)
    try:
        for sql, args in statements:
            con.execute(sql, args)
        con.commit()
    finally:
        con.close()


def _add_marker(path: Path, change_id: str, value: str = '{"since": 83}') -> None:
    _sql(path, ("INSERT INTO metadata (name, value) VALUES (?, ?)", (MARKER_PREFIX + change_id, value)))


# -- G1 ----------------------------------------------------------------------------------------------------------------
BRANCHES = {
    "plain": ({"from": 82, "to": 83}, "This database was upgraded",
              f"FlexTool upgraded it from data version 82 to 83 so that this FlexTool can run it. {OLDER}"),
    "schema": ({"from": 83, "to": 83, "schema": True}, "This database was upgraded",
               f"FlexTool updated the database format of this file so that this FlexTool can run it. {OLDER}"),
    "copy": ({"from": 82, "to": 83, "copy_of": "f.sqlite"}, "An upgraded copy was added",
             "This is an upgraded copy of the frozen database f.sqlite (from data version 82 to 83), added as a new "
             "input source. The frozen file is unchanged."),
}


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_g1_upgraded_text_lists_the_added_notices(branch):
    facts, title, first = BRANCHES[branch]
    # No notices: today's paragraph exactly.
    assert ms.upgraded_text(facts) == (title, (first,))
    # One results-changing and one results-unchanged id (and an unknown one, skipped); the results group in code
    # order (W-0600 before W-0635 whatever the facts' order).
    got = _msg({**facts, "notices": ["rolling_within_solve_one_solve", "no_such_change", "storage_fix_method_rewritten",
                                     "storage_default_fix_nothing"]})
    assert (got.title, got.paragraphs) == (title, (
        first, RESULTS_HEADING, f"{W0600}\n{W0635}", NEUTRAL_HEADING, W0601, CLOSING))
    _check_style(got)
    # Undecided units with only W-0634: the results heading carries the owner's paragraph and the names; W-0634
    # itself is under the neutral heading.  The fixed paragraph adds two sentences (the one exception).
    got = _msg({**facts, "notices": ["minimum_time_method_written"], "undecided": ["coal_1", "gas_2"]})
    assert got.paragraphs == (first, RESULTS_HEADING, UNDECIDED, "coal_1, gas_2 (W-0231)", NEUTRAL_HEADING, W0634,
                              CLOSING)
    _check_style(got, max_sentences=5)
    # Undecided storage nodes with only W-0601: the same, with their own paragraph and the code W-0601; with both
    # lists the units come first.
    got = _msg({**facts, "notices": ["storage_fix_method_rewritten"], "undecided_nodes": ["s1", "s2"]})
    assert got.paragraphs == (first, RESULTS_HEADING, UNDECIDED_NODES, "s1, s2 (W-0601)", NEUTRAL_HEADING, W0601,
                              CLOSING)
    _check_style(got, max_sentences=5)
    got = _msg({**facts, "notices": ["storage_fix_method_rewritten", "minimum_time_method_written"],
                "undecided": ["coal_1"], "undecided_nodes": ["s1"]})
    assert got.paragraphs == (first, RESULTS_HEADING, UNDECIDED, "coal_1 (W-0231)", UNDECIDED_NODES, "s1 (W-0601)",
                              NEUTRAL_HEADING, f"{W0601}\n{W0634}", CLOSING)
    _check_style(got, max_sentences=7)
    # Only unknown ids: nothing added.
    assert ms.upgraded_text({**facts, "notices": ["no_such_change"]}) == (title, (first,))


def test_g1_long_lists_stay_within_the_style_limits():
    from flextool.run_warnings._behaviour_specs import BEHAVIOUR_TEXTS

    ids = list(BEHAVIOUR_TEXTS)[:30]
    assert len(ids) == 30
    _check_style(_msg({"from": 82, "to": 83, "notices": ids}))
    # 12 undecided units: the first 10 named, then "and 2 more".
    units = [f"u{i}" for i in range(1, 13)]
    got = _msg({"from": 82, "to": 83, "notices": ids, "undecided": units})
    assert "u1, u2, u3, u4, u5, u6, u7, u8, u9, u10 and 2 more (W-0231)" in got.paragraphs
    _check_style(got, max_sentences=5)


# -- G2 ----------------------------------------------------------------------------------------------------------------
def test_g2_added_notices_and_undecided_names():
    before = {"storage_default_fix_nothing": "x"}
    after = {"storage_default_fix_nothing": "x", "no_such_change": "y", "rolling_within_solve_one_solve": "{}",
             "minimum_time_method_written": "{}"}
    # Present before → not listed; no texts → not listed; code order (W-0634 before W-0635).
    assert added_notices(before, after) == ("minimum_time_method_written", "rolling_within_solve_one_solve")
    assert added_notices(after, after) == ()
    marker = {"since": 83, "writes": [["a", "Base", "both"]],
              "undecided": [["coal_1", "online_unknown", ["S1"]], ["gas_2", "x", ["S1", "S2"]],
                            ["coal_1", "y", ["S2"]]]}
    mtm = "minimum_time_method_written"
    assert undecided_names({mtm: json.dumps(marker)}, mtm) == ("coal_1", "gas_2")
    assert undecided_names({}, mtm) == ()
    assert undecided_names({mtm: "not json"}, mtm) == ()
    assert undecided_names({mtm: "[1, 2]"}, mtm) == ()
    # The v77 marker: rows (node, alternative, method, scenarios).
    v77 = {"since": 77, "rewrites": [], "undecided": [["s1", "Base", "fix_start", ["S1"]],
                                                     ["s2", "Alt", "fix_end", ["S2"]]]}
    assert undecided_names({"storage_fix_method_rewritten": json.dumps(v77)}, "storage_fix_method_rewritten") == (
        "s1", "s2")


# -- G3 / G4 / G5 --------------------------------------------------------------------------------------------------
def _fake_upgrade(change_id: str):
    def upgrade(path, **_kw):
        _add_marker(Path(path), change_id)
        return True, False, [f"{Path(path).name}: upgraded"]
    return upgrade


def test_g3_upgrade_one_lists_the_marker_the_upgrade_added(schema_db_url, tmp_path, monkeypatch):
    from flextool.gui import main_window

    path = _db(schema_db_url, tmp_path / "in" / "my model #1.sqlite")
    _add_marker(path, "storage_default_fix_nothing")                       # there before: not listed
    monkeypatch.setattr(main_window, "check_and_upgrade_database", _fake_upgrade("rolling_within_solve_one_solve"))
    r = main_window.upgrade_one(path.name, path, "plain", None, progress_callback=None, cancel_check=None)
    assert (r.upgraded, r.failed, r.notices, r.undecided) == (True, False, ("rolling_within_solve_one_solve",), ())

    # W-0634 with undecided units: the names come along.
    marker = json.dumps({"since": 83, "writes": [], "undecided": [["u_a", "r", ["S"]], ["u_b", "r", ["S"]]]})

    def upgrade_min_time(p, **_kw):
        _add_marker(Path(p), "minimum_time_method_written", marker)
        return True, False, []

    other = _db(schema_db_url, tmp_path / "in" / "other.sqlite")
    monkeypatch.setattr(main_window, "check_and_upgrade_database", upgrade_min_time)
    r = main_window.upgrade_one(other.name, other, "plain", None, progress_callback=None, cancel_check=None)
    assert (r.notices, r.undecided, r.undecided_nodes) == (("minimum_time_method_written",), ("u_a", "u_b"), ())

    # W-0601 with undecided storage nodes: the nodes come along, in their own list.
    v77 = json.dumps({"since": 77, "rewrites": [], "undecided": [["s1", "Base", "fix_start", ["S"]]]})

    def upgrade_storage(p, **_kw):
        _add_marker(Path(p), "storage_fix_method_rewritten", v77)
        return True, False, []

    third = _db(schema_db_url, tmp_path / "in" / "third.sqlite")
    monkeypatch.setattr(main_window, "check_and_upgrade_database", upgrade_storage)
    r = main_window.upgrade_one(third.name, third, "plain", None, progress_callback=None, cancel_check=None)
    assert (r.notices, r.undecided, r.undecided_nodes) == (("storage_fix_method_rewritten",), (), ("s1",))


def test_g3_frozen_copy_reads_the_copy_and_leaves_the_original(schema_db_url, tmp_path, monkeypatch):
    from flextool.gui import main_window

    original = _db(schema_db_url, tmp_path / "frozen.sqlite")
    copy_path = tmp_path / "frozen_v83.sqlite"
    data, mtime = original.read_bytes(), os.stat(original).st_mtime_ns

    def migrate_copy(orig, copy, **_kw):
        shutil.copy2(orig, copy)
        _add_marker(Path(copy), "separate_dispatch_realization_looks_ahead")
        return True, False, ["copied"]

    monkeypatch.setattr(main_window, "_migrate_frozen_copy", migrate_copy)
    r = main_window.upgrade_one("frozen.sqlite", original, "frozen_copy", copy_path, progress_callback=None,
                                cancel_check=None)
    assert Path(r.path) == copy_path and r.copy_of == "frozen.sqlite"
    assert r.notices == ("separate_dispatch_realization_looks_ahead",)
    assert original.read_bytes() == data and os.stat(original).st_mtime_ns == mtime
    assert frozen.read_markers_ro(original) == {}


@pytest.mark.parametrize("failing_call", [1, 2])
def test_g4_an_unreadable_marker_read_lists_nothing(schema_db_url, tmp_path, monkeypatch, failing_call):
    from flextool.gui import main_window

    path = _db(schema_db_url, tmp_path / "m.sqlite")
    monkeypatch.setattr(main_window, "check_and_upgrade_database", _fake_upgrade("rolling_within_solve_one_solve"))
    real, calls = frozen.read_markers_ro, []

    def read(p):
        calls.append(Path(p))
        return None if len(calls) == failing_call else real(p)

    monkeypatch.setattr(frozen, "read_markers_ro", read)
    r = main_window.upgrade_one(path.name, path, "plain", None, progress_callback=None, cancel_check=None)
    assert len(calls) == 2
    assert (r.upgraded, r.failed, r.notices, r.undecided) == (True, False, (), ())


def test_g5_an_older_spinedb_revision_reads_its_markers(schema_db_url, tmp_path, monkeypatch):
    """CR-1 B2: a file at an older SpineDB revision (which a DatabaseMapping refuses to open without upgrading)
    reads its markers the same, so its v82 marker is not taken as new."""
    from flextool.gui import main_window

    path = _db(schema_db_url, tmp_path / "old.sqlite")
    _add_marker(path, "end_of_life_closest_seam")
    _sql(path, ("UPDATE alembic_version SET version_num = ?", ("989fccf80441",)))
    assert frozen.read_markers_ro(path) == {"end_of_life_closest_seam": '{"since": 83}'}
    monkeypatch.setattr(main_window, "check_and_upgrade_database", _fake_upgrade("rolling_within_solve_one_solve"))
    r = main_window.upgrade_one(path.name, path, "plain", None, progress_callback=None, cancel_check=None)
    assert r.notices == ("rolling_within_solve_one_solve",)


def test_g5_the_before_read_runs_before_the_migration(schema_db_url, tmp_path, monkeypatch):
    """The markers are read from the file before the migration call and from the result after it: a before read
    after the migration would take every added marker as already there."""
    from flextool.gui import main_window

    path = _db(schema_db_url, tmp_path / "m.sqlite")
    calls: list[str] = []
    real = frozen.read_markers_ro

    def read(p):
        calls.append("read")
        return real(p)

    def upgrade(p, **_kw):
        calls.append("upgrade")
        _add_marker(Path(p), "rolling_within_solve_one_solve")
        return True, False, []

    monkeypatch.setattr(frozen, "read_markers_ro", read)
    monkeypatch.setattr(main_window, "check_and_upgrade_database", upgrade)
    r = main_window.upgrade_one(path.name, path, "plain", None, progress_callback=None, cancel_check=None)
    assert calls == ["read", "upgrade", "read"]
    assert r.notices == ("rolling_within_solve_one_solve",)


# -- G6 ----------------------------------------------------------------------------------------------------------------
def test_g6_read_markers_ro(schema_db_url, tmp_path):
    assert frozen.read_markers_ro(tmp_path / "missing.sqlite") == {}
    no_metadata = tmp_path / "plain.sqlite"
    _sql(no_metadata, ("CREATE TABLE t (x INTEGER)", ()))
    not_db = tmp_path / "notes.sqlite"
    not_db.write_bytes(b"this is not a database file" * 100)
    path = _db(schema_db_url, tmp_path / "a b%#.sqlite")
    _add_marker(path, "storage_default_fix_nothing", "old")
    _add_marker(path, "storage_default_fix_nothing", "new")              # a later row wins
    _sql(path, ("INSERT INTO metadata (name, value) VALUES (?, ?)", ("flextool_behaviour_changeX", "not a marker")))
    stamps = {p: (p.read_bytes(), os.stat(p).st_mtime_ns) for p in (no_metadata, not_db, path)}
    assert frozen.read_markers_ro(no_metadata) == {}
    assert frozen.read_markers_ro(not_db) is None
    assert frozen.read_markers_ro(path) == {"storage_default_fix_nothing": "new"}
    for p, stamp in stamps.items():
        assert (p.read_bytes(), os.stat(p).st_mtime_ns) == stamp


# -- G7 ----------------------------------------------------------------------------------------------------------------
def test_g7_a_pending_event_keeps_its_notices(tmp_path):
    db = tmp_path / "P" / "m.sqlite"
    db.parent.mkdir(parents=True)
    db.write_bytes(b"")
    center = MessageCenter(tmp_path / "s.json", (ms.upgraded_source(lambda: [database_key(db)]),),
                           today=lambda: date(2026, 1, 1))
    first = UpgradeResult(db.name, db, 82, 83, None, True, False, [],
                          ("minimum_time_method_written", "storage_fix_method_rewritten"), ("u1", "u2"), ("s1",))
    ms.post_upgrades(center, [first], "run1")
    second = UpgradeResult(db.name, db, 83, 84, None, True, False, [], ("rolling_within_solve_one_solve",), ("u3",))
    ms.post_upgrades(center, [second], "run2")
    center.recompute("other")
    (msg,) = center.messages
    assert msg.instance == "run2"
    assert msg.paragraphs[1:] == (RESULTS_HEADING, W0635, UNDECIDED, "u1, u2, u3 (W-0231)", UNDECIDED_NODES,
                                  "s1 (W-0601)", NEUTRAL_HEADING, f"{W0601}\n{W0634}", CLOSING)
    # Dismissed: the next upgrade starts a fresh list.
    center.dismiss(msg)
    ms.post_upgrades(center, [second], "run3")
    center.recompute("other")
    (msg,) = center.messages
    assert msg.paragraphs[1:] == (RESULTS_HEADING, W0635, UNDECIDED, "u3 (W-0231)", CLOSING)
