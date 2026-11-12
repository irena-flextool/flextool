"""The input and discovery tools only read: nothing under the project changes, no file appears, no file stays open.

The project path has a space, ``#``, ``%25`` and ``ä``: an unencoded SQLite URI breaks there and makes SQLite create
another file next to the project, which the folder listing would show.  A database is opened through the read-only
URI, fetched and closed (the closed mapping answers every fetched item type); a conversion replaced between two
calls is read again; an exclusive lock gives the busy message within the 5 s timeout; a database older than this
FlexTool gives ``stored: false`` with the note on older parameter names.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from flextool.assistant.context import make_context
from flextool.assistant.errors import MESSAGES, ToolError
from flextool.assistant.inputs import _db
from flextool.assistant.registry import call_tool
from flextool.update_flextool import FLEXTOOL_DB_VERSION

from tests.assistant.conftest import compare_rows, make_project

CALLS = [
    ("get_gui_context", {}),
    ("list_sources", {}),
    ("list_runs", {}),
    ("list_runs", {"source": "1", "scenario": "coal"}),
    ("list_entities", {"entity_class": "unit", "scenario": "coal"}),
    ("list_entities", {"entity_class": "unit__outputNode", "member": "heat"}),
    ("get_entity", {"name": ["west"], "scenario": "coal"}),
    ("get_parameter", {"entity_class": "node", "parameter": "inflow", "entity": ["west"], "scenario": "coal"}),
    ("get_parameter", {"entity_class": "unit", "parameter": "existing", "scenario": "all"}),
    ("describe_parameter", {"entity_class": "unit", "parameter": "existing"}),
    ("compare_inputs", {"scenario_a": "coal", "scenario_b": "coal_chp"}),
]


def _snapshot(root: Path) -> dict[str, tuple]:
    """Every path under *root*: (kind, size, mtime_ns, sha256 of a file)."""
    out: dict[str, tuple] = {}
    for path in sorted(root.rglob("*")):
        st = path.stat()
        if path.is_dir():
            out[path.relative_to(root).as_posix()] = ("dir",)
        else:
            out[path.relative_to(root).as_posix()] = ("file", st.st_size, st.st_mtime_ns,
                                                      hashlib.sha256(path.read_bytes()).hexdigest())
    return out


def _open_files_under(folder: Path) -> list[str]:
    out = []
    for fd in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(fd)
        except OSError:
            continue
        if target.startswith(str(folder)):
            out.append(target)
    return out


_HAS_PROC = Path("/proc/self/fd").is_dir()


@pytest.mark.parametrize("tool,args", CALLS, ids=[f"{t}-{i}" for i, (t, _) in enumerate(CALLS)])
def test_a_tool_call_changes_nothing_and_leaves_nothing_open(assistant_project: Path, tool: str, args: dict) -> None:
    root = assistant_project.parent           # the folder around the project too: a stray file appears there
    before = _snapshot(root)
    out = call_tool(make_context(assistant_project), tool, args)
    assert out.ok, out.message
    assert _snapshot(root) == before
    if _HAS_PROC:
        assert _open_files_under(root) == []


def test_every_fetched_item_type_answers_on_the_closed_mapping(assistant_project: Path) -> None:
    from spinedb_api.exception import SpineDBAPIError

    path = assistant_project / "input_sources" / "tests.sqlite"
    mapping, version, markers = _db.closed_mapping(path)
    for item_type in _db.FETCHED:
        assert list(mapping.find_by_type(item_type)), item_type
    # The fixture database is migrated to this FlexTool's version.
    assert float(version) == FLEXTOOL_DB_VERSION
    assert markers == {}
    # A type outside FETCHED needs the closed session: the markers are therefore read before the mapping closes.
    with pytest.raises(SpineDBAPIError):
        list(mapping.get_items("metadata"))
    if _HAS_PROC:
        assert _open_files_under(assistant_project) == []


def _live_sqlite_connections() -> int:
    """Connections the garbage collector tracks, garbage not yet collected included."""
    return sum(isinstance(obj, sqlite3.Connection) for obj in gc.get_objects())


@pytest.mark.filterwarnings("error::ResourceWarning")
def test_a_file_it_cannot_read_leaves_no_connection_open(tmp_path: Path, tests_db: Path) -> None:
    """``spinedb_api`` connects before it finds the file is no database and drops that connection without closing
    it.  Before: the connection stayed open (Windows locks the file) until a garbage collection in some later test
    finalised it, which Python 3.13+ reports as an unclosed-database ResourceWarning there."""
    project = make_project(tmp_path, tests_db)
    path = project / "input_sources" / "tests.sqlite"
    path.write_bytes(b"not a database" * 100)
    gc.collect()
    before = _live_sqlite_connections()
    with pytest.raises(ToolError) as info:
        _db.closed_mapping(path)
    assert info.value.key == "not_flextool"
    assert _live_sqlite_connections() == before
    if _HAS_PROC:
        assert _open_files_under(project) == []
    gc.collect()                                          # a connection left behind would warn here


def test_uri_is_read_only_and_percent_encoded() -> None:
    assert _db.uri_of_posix("/tmp/G #2 ä%25x/t.sqlite") == (
        "sqlite:///file:/tmp/G%20%232%20%C3%A4%2525x/t.sqlite?mode=ro&uri=true")
    # Windows: a drive path becomes file:/C:/…
    assert _db.uri_of_posix("C:/Users/A B/x#1.sqlite") == "sqlite:///file:/C:/Users/A%20B/x%231.sqlite?mode=ro&uri=true"


def test_a_unc_path_gives_an_empty_authority() -> None:
    from pathlib import PureWindowsPath

    # C3-m3: the server goes into the path part (file:////server/…), never the URI authority; the server and
    # share names are percent-encoded as the rest.
    unc = PureWindowsPath(r"\\server\my share\G #1\x.sqlite").as_posix()
    assert _db.uri_of_posix(unc) == "sqlite:///file:////server/my%20share/G%20%231/x.sqlite?mode=ro&uri=true"
    drive = PureWindowsPath(r"C:\a b\x.sqlite").as_posix()
    assert _db.uri_of_posix(drive) == "sqlite:///file:/C:/a%20b/x.sqlite?mode=ro&uri=true"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX reads //tmp/x as /tmp/x; Windows reads it as a share")
def test_a_double_slash_path_opens_read_only(tmp_path: Path, tests_db: Path) -> None:
    copy = tmp_path / "copy.sqlite"
    shutil.copy(tests_db, copy)
    doubled = Path("/" + str(copy))                  # pathlib keeps the two leading slashes, as of a UNC path
    assert doubled.as_posix().startswith("//")
    assert _db.load_uncached(doubled).version is not None
    uri = _db.db_uri(doubled)
    conn = sqlite3.connect(uri[len("sqlite:///"):], uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="attempt to write a readonly database"):
            conn.execute("CREATE TABLE written_by_the_test (x)")
    finally:
        conn.close()


def test_scenario_views_are_kept_in_a_small_lru(ctx) -> None:
    """C3-m8.  At most two views per database; a hit is refreshed; a comparison after an eviction is unchanged
    (tests.json: commodity coal price 20 in alternative coal only; unit coal_chp existing coal 1200, coal_chp
    1000)."""
    db = _db.load(ctx.project_path / "input_sources" / "tests.sqlite")
    coal = db.view("coal")
    every = db.view("all")
    assert db.view("coal") is coal                       # a hit, now the most recent
    db.view("coal_chp")                                  # evicts "all", the least recent
    assert len(db._views) == 2 and list(db._views) == ["coal", "coal_chp"]
    assert db.view("coal") is coal
    assert db.view("all") is not every
    out = call_tool(ctx, "compare_inputs", {"scenario_a": "coal", "scenario_b": "coal_chp", "max_rows": 500})
    assert out.ok, out.message
    diffs = compare_rows(json.loads(out.text))
    assert (diffs[("commodity", ("coal",), "price")]["a"], diffs[("commodity", ("coal",), "price")]["b"]) == (
        20.0, None)
    assert (diffs[("unit", ("coal_chp",), "existing")]["a"], diffs[("unit", ("coal_chp",), "existing")]["b"]) == (
        1200.0, 1000.0)
    assert len(db._views) == 2


@pytest.mark.skipif(sys.platform != "win32", reason="the drive-letter URI is opened on Windows only")
def test_windows_drive_uri_opens(assistant_project: Path) -> None:
    path = assistant_project / "input_sources" / "tests.sqlite"
    assert _db.db_uri(path).startswith(f"sqlite:///file:/{path.drive}")
    assert _db.load_uncached(path).version is not None


def _write_value(db_path: Path, value: float) -> None:
    """Set unit coal_plant existing of alternative coal to *value* (a synthetic change on a tmp copy)."""
    from spinedb_api import to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{db_path.as_posix()}") as db:
        item = db.get_parameter_value_item(entity_class_name="unit", entity_byname=("coal_plant",),
                                           parameter_definition_name="existing", alternative_name="coal")
        raw, value_type = to_database(value)
        db.update_parameter_value_item(id=item["id"], value=raw, type=value_type)
        db.commit_session("test: change one value")


def test_a_conversion_replaced_between_calls_is_read_again(tmp_path: Path, tests_db: Path) -> None:
    project = tmp_path / "G #2 ä%25x project"
    (project / "input_sources").mkdir(parents=True)
    (project / "intermediate").mkdir()
    sheet = project / "input_sources" / "model.xlsx"
    sheet.write_bytes(b"the tools read the conversion of a spreadsheet, not the spreadsheet")
    (project / "settings.yaml").write_text("input_source_numbers:\n  model.xlsx: 1\n", encoding="utf-8")
    conversion = project / "intermediate" / "model.sqlite"
    shutil.copy(tests_db, conversion)
    now = time.time()
    os.utime(sheet, (now - 100, now - 100))
    os.utime(conversion, (now - 50, now - 50))          # older than 10 s: not being converted
    ctx = make_context(project)
    args = {"entity_class": "unit", "parameter": "existing", "entity": ["coal_plant"], "scenario": "coal",
            "source": "model.xlsx"}
    first = call_tool(ctx, "get_parameter", args)
    assert first.ok, first.message
    # tests.json: unit coal_plant existing in alternative coal is 500.
    assert json.loads(first.text)["value"] == 500.0
    if _HAS_PROC:
        assert _open_files_under(project) == []
    changed = tmp_path / "changed.sqlite"
    shutil.copy(tests_db, changed)
    _write_value(changed, 750.0)
    os.replace(changed, conversion)                      # as the window does: the conversion is a new file
    os.utime(conversion, (now - 40, now - 40))
    second = call_tool(ctx, "get_parameter", args)
    assert second.ok, second.message
    assert json.loads(second.text)["value"] == 750.0
    if _HAS_PROC:
        assert _open_files_under(project) == []


def test_spreadsheet_states(tmp_path: Path, tests_db: Path) -> None:
    project = tmp_path / "p"
    (project / "input_sources").mkdir(parents=True)
    (project / "input_sources" / "model.xlsx").write_bytes(b"x")
    ctx = make_context(project)
    args = {"entity_class": "unit", "parameter": "existing", "entity": ["coal_plant"], "scenario": "coal"}
    out = call_tool(ctx, "get_parameter", args)
    assert not out.ok and out.message == MESSAGES["not_converted"]
    (project / "intermediate").mkdir()
    shutil.copy(tests_db, project / "intermediate" / "model.sqlite")    # just written: being converted
    out = call_tool(ctx, "get_parameter", args)
    assert not out.ok and out.message == MESSAGES["converting"]


def test_an_exclusive_lock_gives_the_busy_message(tmp_path: Path, tests_db: Path, monkeypatch) -> None:
    """A locked database is given up after the assistant's SQLite timeout, not after the database library's
    default of 1 800 s.  The timeout is set to 1 s here, and the bound is generous: SQLite's busy handler counts its
    nominal sleeps, which a loaded macOS or Windows runner oversleeps (9.2 s for the 5 s timeout: CI of PR #362).
    The same call on another copy runs first, so the timed call does not include the worker's first import of the
    database library."""
    args = {"entity_class": "unit", "parameter": "existing", "entity": ["coal_plant"], "scenario": "coal"}
    warm = make_project(tmp_path / "warm", tests_db)
    assert call_tool(make_context(warm), "get_parameter", args).ok
    monkeypatch.setattr(_db, "SQLITE_TIMEOUT_S", 1)
    project = make_project(tmp_path, tests_db)
    db_path = project / "input_sources" / "tests.sqlite"
    holder = sqlite3.connect(db_path)
    try:
        holder.execute("BEGIN EXCLUSIVE")
        started = time.monotonic()
        out = call_tool(make_context(project), "get_parameter", args)
        elapsed = time.monotonic() - started
    finally:
        holder.rollback()
        holder.close()
    assert not out.ok and out.message == MESSAGES["busy"]
    assert elapsed < 4.0


def _make_older(db_path: Path) -> None:
    """model.version 77 and unit.existing renamed, as a database from before a rename (synthetic)."""
    from spinedb_api import to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{db_path.as_posix()}") as db:
        version = db.get_parameter_definition_item(entity_class_name="model", name="version")
        raw, value_type = to_database(77.0)
        db.update_parameter_definition_item(id=version["id"], default_value=raw, default_type=value_type)
        existing = db.get_parameter_definition_item(entity_class_name="unit", name="existing")
        db.update_parameter_definition_item(id=existing["id"], name="existing_before_rename")
        db.add_metadata_item(name="flextool_behaviour_change:storage_default_fix_nothing", value="{}")
        db.commit_session("test: an older database")


def test_an_older_database_never_asserts_a_default(tmp_path: Path, tests_db: Path) -> None:
    older = tmp_path / "older.sqlite"
    shutil.copy(tests_db, older)
    _make_older(older)
    project = make_project(tmp_path, older)
    ctx = make_context(project)
    out = call_tool(ctx, "get_parameter", {"entity_class": "unit", "parameter": "existing",
                                           "entity": ["coal_plant"], "scenario": "coal"})
    assert out.ok, out.message
    rec = json.loads(out.text)
    assert rec["stored"] is False and rec["is_default"] is None and rec["value"] is None
    note = next(n for n in rec["notes"] if n.startswith("This database is older than this FlexTool"))
    # coal_plant's stored parameters in tests.json include existing (now under its old name) and invest_cost.
    assert "existing_before_rename" in note and "invest_cost" in note
    db = _db.load(project / "input_sources" / "tests.sqlite")
    assert db.version == 77 and _db.version_status(db.version) == "needs_migration"
    assert db.behaviour_notices and db.behaviour_notices[0].startswith("Behaviour changed in ")
    described = json.loads(call_tool(ctx, "describe_parameter", {"entity_class": "unit",
                                                                  "parameter": "existing"}).text)
    assert described["schema_note"] == "This is the parameter as this FlexTool defines it."
