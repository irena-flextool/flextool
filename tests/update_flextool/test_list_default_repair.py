"""List references stored as the member's text (item 279) are repaired before a migration opens the file.

spinedb_api 0.31.0-0.32.2 stored a list default as the member's JSON text (``b'"not_allowed"'``) instead of the
``list_value`` id; the current spinedb_api reads it as ``None`` and its next write stores ``NULL``.  Every database
here is made up in ``tmp_path`` (from ``schema_db_url``, the bundled v25 template or the v75 seed) and broken with raw
SQL.  Expected members are the texts written into the broken rows; their ids are looked up in the row's own list.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sqlite3
from pathlib import Path

import pytest
from spinedb_api import from_database, to_database

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool import db_migration as m
from flextool.update_flextool import frozen as fz
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database
from flextool.update_flextool.list_default_repair import list_defaults_repairable_ro, repair_list_defaults
from tests.db_utils import json_to_db

SEED = Path(__file__).resolve().parent / "data" / "v75_seed_stoch_two_period.json"
V25_TEMPLATE = package_data_path("schemas/pre_v26/flextool_template_v25.json")
MARKER = "flextool_behaviour_change:"


def _copy_schema_db(schema_db_url: str, folder: Path) -> Path:
    path = folder / "db.sqlite"
    shutil.copy2(schema_db_url.removeprefix("sqlite:///"), path)
    return path


def _sql(path: Path, statement: str, args=()) -> None:
    with contextlib.closing(sqlite3.connect(path)) as con:
        with con:
            con.execute(statement, args)


def _def_id(path: Path, cls: str, name: str) -> int:
    with contextlib.closing(sqlite3.connect(path)) as con:
        return con.execute("SELECT pd.id FROM parameter_definition pd JOIN entity_class ec ON pd.entity_class_id = "
                           "ec.id WHERE ec.name = ? AND pd.name = ?", (cls, name)).fetchone()[0]


def _break_default(path: Path, cls: str, name: str, payload) -> int:
    def_id = _def_id(path, cls, name)
    _sql(path, "UPDATE parameter_definition SET default_value = ?, default_type = 'list_value_ref' WHERE id = ?",
         (payload, def_id))
    return def_id


def _own_member_id(path: Path, def_id: int, text: str) -> int:
    """The id of the member *text* in the value list of definition *def_id* (raw SQL, decoded in Python)."""
    with contextlib.closing(sqlite3.connect(path)) as con:
        rows = con.execute("SELECT lv.id, lv.value FROM list_value lv JOIN parameter_definition pd ON "
                           "lv.parameter_value_list_id = pd.parameter_value_list_id WHERE pd.id = ?",
                           (def_id,)).fetchall()
    ids = [i for i, v in rows if json.loads(v) == text]
    assert len(ids) == 1
    return ids[0]


def _dump(path: Path) -> dict:
    with contextlib.closing(sqlite3.connect(path)) as con:
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        return {t: {r[0]: r for r in con.execute(f'SELECT * FROM "{t}"')} for t in tables}


def _column(path: Path, table: str, column: str) -> int:
    with contextlib.closing(sqlite3.connect(path)) as con:
        return [r[1] for r in con.execute(f"PRAGMA table_info({table})")].index(column)


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


def _assert_closed(path: Path) -> None:
    if Path("/proc/self/fd").is_dir():
        assert _open_files_under(path.parent) == []
    path.unlink()


def _migrate(url_or_path) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        migrate_database(str(url_or_path))
    return out.getvalue()


def test_current_database_rows_are_repaired_in_their_own_list(schema_db_url: str, tmp_path: Path) -> None:
    path = _copy_schema_db(schema_db_url, tmp_path)
    with open_database_mapping("sqlite:///" + str(path)) as db:
        db.add_entity(entity_class_name="node", name="n1")
        value, typ = to_database("storage")
        db.add_parameter_value(entity_class_name="node", entity_byname=("n1",), parameter_definition_name="node_type",
                               alternative_name="base", value=value, type=typ)
        db.commit_session("node n1")
    # Texts that are not the shipped defaults (reinvest_automatic, no, single_solve, balance); "yes" is also in
    # has_DC, has_invest and optional_output, so only the own list (yes_no) gives the right id.
    lifetime = _break_default(path, "unit", "lifetime_method", b'"no_reinvestment"')
    is_dc = _break_default(path, "connection", "is_DC", b'"yes"')
    _break_default(path, "solve", "solve_mode", b'"single_shot"')  # renamed away: not in its list
    _sql(path, "UPDATE parameter_value SET value = ? WHERE type = 'list_value_ref'", (b'"commodity"',))
    node_type = _def_id(path, "node", "node_type")
    # R2: the probe reads the text payloads without a ValueError.  It parses the default of every definition named in
    # the superseded texts whose description is not the old text, so this definition must stay one of them.
    assert ("unit", "lifetime_method") in m._head_superseded_texts()
    assert not m._head_defaults_stale_ro(path)
    assert list_defaults_repairable_ro(path)

    expected = _dump(path)
    default_col = _column(path, "parameter_definition", "default_value")
    for def_id, text in ((lifetime, "no_reinvestment"), (is_dc, "yes")):
        row = list(expected["parameter_definition"][def_id])
        row[default_col] = str(_own_member_id(path, def_id, text)).encode()
        expected["parameter_definition"][def_id] = tuple(row)
    value_col = _column(path, "parameter_value", "value")
    (value_id, value_row), = expected["parameter_value"].items()
    row = list(value_row)
    row[value_col] = str(_own_member_id(path, node_type, "commodity")).encode()
    expected["parameter_value"][value_id] = tuple(row)

    out = _migrate(path)
    assert "3 repaired, 1 left" in out and "solve.solve_mode" in out and '"single_shot"' in out
    assert _dump(path) == expected  # incl. the commit table and the unmatched row, byte for byte
    assert not list_defaults_repairable_ro(path)
    with open_database_mapping("sqlite:///" + str(path)) as db:
        def default(cls, name):
            pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
            return from_database(pdef["default_value"], pdef["default_type"])

        assert default("unit", "lifetime_method") == "no_reinvestment"
        assert default("connection", "is_DC") == "yes"
        assert default("solve", "solve_mode") is None
        (pv,) = db.get_items("parameter_value")
        assert pv["parsed_value"] == "commodity"
    digest = fz.sha256_of(path)
    _migrate(path)  # idempotent
    assert fz.sha256_of(path) == digest
    _assert_closed(path)


def test_payload_shapes(schema_db_url: str, tmp_path: Path, capsys) -> None:
    path = _copy_schema_db(schema_db_url, tmp_path)
    as_text = _break_default(path, "unit", "conversion_method", '"piecewise_linear"')  # SQLite TEXT
    _break_default(path, "unit", "startup_method", b"not json")
    no_list = _break_default(path, "unit", "invest_method", b'"invest_total"')
    _sql(path, "UPDATE parameter_definition SET parameter_value_list_id = NULL WHERE id = ?", (no_list,))
    digits = _def_id(path, "node", "invest_method")
    with contextlib.closing(sqlite3.connect(path)) as con:
        (digits_payload,) = con.execute("SELECT default_value FROM parameter_definition WHERE id = ?",
                                        (digits,)).fetchone()
    _sql(path, "UPDATE parameter_definition SET default_value = ? WHERE id = ?", (digits_payload.decode(), digits))
    before = _dump(path)

    result = repair_list_defaults(path)

    assert result.repaired == 1
    assert sorted(result.unmatched) == [("unit", "invest_method", '"invest_total"'),
                                        ("unit", "startup_method", "not json")]
    out = capsys.readouterr().out
    assert "1 repaired, 2 left" in out and "unit.startup_method" in out and "unit.invest_method" in out
    after = _dump(path)
    col = _column(path, "parameter_definition", "default_value")
    assert after["parameter_definition"][as_text][col] == str(_own_member_id(path, as_text, "piecewise_linear")).encode()
    after["parameter_definition"][as_text] = before["parameter_definition"][as_text]
    assert after == before  # the others (invalid JSON, no list, a TEXT id) byte-identical
    assert isinstance(after["parameter_definition"][digits][col], str)
    _assert_closed(path)


def test_clean_current_database_is_not_touched(schema_db_url: str, tmp_path: Path) -> None:
    path = _copy_schema_db(schema_db_url, tmp_path)
    assert not list_defaults_repairable_ro(path)
    digest = fz.sha256_of(path)
    _migrate(path)
    assert fz.sha256_of(path) == digest


def test_older_database_default_reads_the_renamed_member(tmp_path: Path) -> None:
    """v56 renames storage_nested_fix_method ``no`` to ``fix_nothing``; ``no`` is also a member of yes_no."""
    path = tmp_path / "db.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        initialize_database(str(V25_TEMPLATE), str(path))
        migrate_database("sqlite:///" + str(path), up_to=55)
    _break_default(path, "node", "storage_nested_fix_method", b'"no"')
    out = _migrate(path)
    assert "1 repaired, 0 left" in out
    with open_database_mapping("sqlite:///" + str(path)) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="node", name="storage_nested_fix_method")
        assert from_database(pdef["default_value"], pdef["default_type"]) == "fix_nothing"
        version = db.get_item("parameter_definition", entity_class_name="model", name="version")
        assert from_database(version["default_value"], version["default_type"]) == FLEXTOOL_DB_VERSION
    _assert_closed(path)


def test_v77_keeps_fix_start_for_a_storage_node_with_its_own_start(tmp_path: Path) -> None:
    """The old default ``fix_start`` stored as text: after the repair the v77 step sees it.  By the rule of
    ``classify_storage_default``: node ``store`` has no stored method and its own ``storage_state_start`` in
    ``base`` (in every scenario of the seed), fix_start fixes only the start, and an explicit fix_start under the
    new default fix_nothing starts it as before, so fix_start is written in ``base``; the default fixed something,
    so ``storage_default_fix_nothing`` is ``"77"``."""
    path = tmp_path / "db.sqlite"
    with contextlib.redirect_stdout(io.StringIO()):
        url = json_to_db(SEED, path)
        migrate_database(url, up_to=76)
    with open_database_mapping(url) as db:
        assert all("base" in [r["alternative_name"] for r in db.get_items("scenario_alternative", scenario_name=s)]
                   for s in [s["name"] for s in db.get_items("scenario")])
        db.add_entity(entity_class_name="node", name="store")
        db.add_entity_alternative(entity_class_name="node", entity_byname=("store",), alternative_name="base",
                                  active=True)
        for name, parsed in (("node_type", "storage"), ("storage_state_start", 0.5)):
            value, typ = to_database(parsed)
            db.add_parameter_value(entity_class_name="node", entity_byname=("store",), parameter_definition_name=name,
                                   alternative_name="base", value=value, type=typ)
        db.commit_session("storage node")
    _break_default(path, "node", "storage_start_end_method", b'"fix_start"')
    out = _migrate(path)
    assert "1 repaired, 0 left" in out
    with open_database_mapping(url) as db:
        methods = {(pv["entity_name"], pv["alternative_name"]): pv["parsed_value"] for pv in
                   db.find_parameter_values(entity_class_name="node", parameter_definition_name="storage_start_end_method")}
        markers = {md["name"]: md["value"] for md in db.get_items("metadata")}
    assert methods == {("store", "base"): "fix_start"}
    assert json.loads(markers[MARKER + "storage_fix_method_rewritten"])["rewrites"] == [
        ["store", "base", "default fix_start", "fix_start"]]
    assert markers[MARKER + "storage_default_fix_nothing"] == "77"
    _assert_closed(path)


def test_frozen_current_database_is_left_unchanged(schema_db_url: str, tmp_path: Path) -> None:
    path = _copy_schema_db(schema_db_url, tmp_path)
    fz.freeze(path, "Reference")
    _break_default(path, "unit", "lifetime_method", b'"no_reinvestment"')
    digest = fz.sha256_of(path)
    _migrate(path)
    assert fz.sha256_of(path) == digest


def test_file_without_value_lists_has_nothing_to_repair(tmp_path: Path) -> None:
    path = tmp_path / "old.sqlite"
    _sql(path, "CREATE TABLE parameter_definition (id INTEGER PRIMARY KEY, name TEXT, default_value BLOB, "
               "default_type TEXT)")
    _sql(path, "INSERT INTO parameter_definition (name, default_value, default_type) VALUES ('x', ?, "
               "'list_value_ref')", (b'"a"',))
    assert not list_defaults_repairable_ro(path)
    assert repair_list_defaults(path).repaired == 0
    _assert_closed(path)


@pytest.mark.parametrize("payload", [b"12", "12"])
def test_digit_payloads_are_not_candidates(tmp_path: Path, payload) -> None:
    path = tmp_path / "db.sqlite"
    for statement in ("CREATE TABLE list_value (id INTEGER PRIMARY KEY, parameter_value_list_id INTEGER, "
                      "value BLOB, type TEXT)",
                      "CREATE TABLE parameter_definition (id INTEGER PRIMARY KEY, parameter_value_list_id INTEGER, "
                      "default_value BLOB, default_type TEXT)",
                      "CREATE TABLE parameter_value (id INTEGER PRIMARY KEY, parameter_definition_id INTEGER, "
                      "value BLOB, type TEXT)"):
        _sql(path, statement)
    _sql(path, "INSERT INTO list_value VALUES (1, 1, ?, 'float')", (b"12",))
    _sql(path, "INSERT INTO parameter_definition VALUES (1, 1, ?, 'list_value_ref')", (payload,))
    assert not list_defaults_repairable_ro(path)
    assert repair_list_defaults(path).repaired == 0
    _assert_closed(path)
