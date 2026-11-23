"""Repair value-list references stored as the member's text (item 279).

spinedb_api 0.31.0-0.32.2 stored a definition's list default (``default_type = 'list_value_ref'``) as the member's
JSON text, e.g. ``b'"not_allowed"'``, instead of the ``list_value`` id.  The current spinedb_api reads such a row as
``None`` without a warning, and its next write of the definition stores ``NULL`` for good.  The repair looks the text
up in the row's own value list with raw ``sqlite3`` and stores the id of the one member with the same value and type.
It adds no commit row and changes nothing else.  The same is done for ``parameter_value`` rows.

:func:`~flextool.update_flextool.db_migration.migrate_database` runs it before the first spinedb_api open, so an
affected database migrates like an unaffected one.  Residual: a current-version file with text defaults (only a
pre-0.33 Spine library can write one) is repaired only by an explicit migration, not by a run or the GUI.
"""
from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

_TIMEOUT_S = 5.0
_JSON_TYPES = {bool: "bool", int: "float", float: "float", str: "str"}


@dataclass
class ListDefaultRepair:
    """What :func:`repair_list_defaults` did: rows repaired, and ``(class, parameter, text)`` of the rows left."""

    repaired: int = 0
    unmatched: list[tuple[str, str, str]] = field(default_factory=list)


def _raw(payload) -> bytes:
    return payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)


def match_list_member(members: list[tuple[int, bytes, str]], payload) -> int | None:
    """The id of the one member ``(id, value, type)`` of *members* whose decoded value and type equal the JSON
    *payload* (bytes or str); ``None`` for no match, several matches or a payload that is not JSON."""
    try:
        wanted = json.loads(_raw(payload))
    except ValueError:
        return None
    kind = _JSON_TYPES.get(type(wanted))
    hits = []
    for member_id, value, typ in members:
        try:
            if typ == kind and json.loads(_raw(value)) == wanted:
                hits.append(member_id)
        except (TypeError, ValueError):
            continue
    return hits[0] if len(hits) == 1 else None


def _has_shape(con: sqlite3.Connection) -> bool:
    """Whether the file has value lists (SpineDB alembic 989fccf80441 and later)."""
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"list_value", "parameter_definition", "parameter_value"} <= tables:
        return False
    return "parameter_value_list_id" in {r[1] for r in con.execute("PRAGMA table_info(parameter_definition)")}


def _scan(con: sqlite3.Connection):
    """``(fixes, left)``: ``(table, column, row id, member id)`` per repairable row and ``(definition id, text)`` per
    row no single member matches."""
    members: dict[int, list] = {}
    for member_id, list_id, value, typ in con.execute("SELECT id, parameter_value_list_id, value, type FROM list_value"):
        members.setdefault(list_id, []).append((member_id, value, typ))
    rows = [("parameter_definition", "default_value", *r) for r in con.execute(
        "SELECT id, id, parameter_value_list_id, default_value FROM parameter_definition "
        "WHERE default_type = 'list_value_ref'")]
    rows += [("parameter_value", "value", *r) for r in con.execute(
        "SELECT pv.id, pd.id, pd.parameter_value_list_id, pv.value FROM parameter_value pv "
        "JOIN parameter_definition pd ON pv.parameter_definition_id = pd.id WHERE pv.type = 'list_value_ref'")]
    fixes, left = [], []
    for table, column, row_id, def_id, list_id, payload in rows:
        if payload is None or _raw(payload).isdigit():
            continue
        member_id = match_list_member(members.get(list_id, []), payload) if list_id is not None else None
        if member_id is None:
            left.append((def_id, _raw(payload).decode("utf-8", "replace")))
        else:
            fixes.append((table, column, row_id, member_id))
    return fixes, left


def list_defaults_repairable_ro(path) -> bool:
    """Whether :func:`repair_list_defaults` would repair a row of *path* (read read-only; ``True`` when it cannot be
    read, so the case is decided under the migration lock)."""
    from flextool.update_flextool.frozen import _connect_ro

    try:
        con = _connect_ro(Path(path))
    except sqlite3.Error:
        return True
    try:
        return _has_shape(con) and bool(_scan(con)[0])
    except sqlite3.Error:
        return True
    finally:
        con.close()


def _names(con: sqlite3.Connection, def_id: int) -> tuple[str, str]:
    try:
        row = con.execute("SELECT ec.name, pd.name FROM parameter_definition pd JOIN entity_class ec "
                          "ON pd.entity_class_id = ec.id WHERE pd.id = ?", (def_id,)).fetchone()
    except sqlite3.Error:
        row = None
    return (row[0], row[1]) if row else ("definition", str(def_id))


def repair_list_defaults(path) -> ListDefaultRepair:
    """Store the member id in every list reference of *path* that holds the text of exactly one member of its own
    list (one transaction, no commit row).  Other text references are left as they are and printed."""
    path = Path(path)
    if not path.is_file():
        return ListDefaultRepair()
    with contextlib.closing(sqlite3.connect(path, timeout=_TIMEOUT_S, isolation_level=None)) as con:
        if not _has_shape(con):
            return ListDefaultRepair()
        con.execute("BEGIN IMMEDIATE")
        try:
            fixes, left = _scan(con)
            for table, column, row_id, member_id in fixes:
                con.execute(f"UPDATE {table} SET {column} = ? WHERE id = ?", (str(member_id).encode(), row_id))
            con.execute("COMMIT")
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        result = ListDefaultRepair(len(fixes), [(*_names(con, def_id), text) for def_id, text in left])
    if result.repaired or result.unmatched:
        print(f"Repairing list defaults stored as text in {path}: {result.repaired} repaired, "
              f"{len(result.unmatched)} left", flush=True)
        for cls, name, text in result.unmatched:
            print(f"  left as it is: {cls}.{name} = {text} (no single member of its value list matches)", flush=True)
    return result
