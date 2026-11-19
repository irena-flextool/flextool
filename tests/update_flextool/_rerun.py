"""Shared helpers for the migration rerun tests: restamp a database, stop a migration after a commit, compare two
databases item by item."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from spinedb_api import DatabaseMapping, from_database, to_database

from flextool.update_flextool.db_migration import ALREADY_APPLIED_PREFIX, migrate_database


class Kill(BaseException):
    """A hard kill right after a commit: nothing after it runs (not an ``Exception``, like a ``KeyboardInterrupt``)."""


def url_of(path: Path) -> str:
    return "sqlite:///" + str(Path(path).resolve())


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def migrate(path: Path, up_to: int | None = None) -> None:
    """Migrate the file *path* (no backup copy: a :class:`Kill` must leave the file as the kill left it)."""
    quiet(migrate_database, url_of(path), up_to=up_to, backup=False)


def copy_db(source: Path, target: Path) -> Path:
    shutil.copyfile(source, target)
    return target


def restamp(db, version: int) -> None:
    """Set ``model.version`` of the open *db* to *version* in its session (the caller commits)."""
    pdef = db.get_item("parameter_definition", entity_class_name="model", name="version")
    val, typ = to_database(float(version))
    db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)


def restamp_file(path: Path, version: int) -> None:
    with DatabaseMapping(url_of(path)) as db:
        restamp(db, version)
        db.commit_session(f"restamp {version}")


def version_of(path: Path) -> float:
    with DatabaseMapping(url_of(path)) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="model", name="version")
        return from_database(pdef["default_value"], pdef["default_type"])


@contextlib.contextmanager
def kill_after_commit(n: int | None) -> Iterator[list[int]]:
    """Count the commits of every ``DatabaseMapping`` while inside; raise :class:`Kill` right after the *n*-th
    (``None``: never).  Yields a one-item list holding the count."""
    real = DatabaseMapping.commit_session
    count = [0]

    def counting(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        count[0] += 1
        if n is not None and count[0] == n:
            raise Kill()
        return result

    DatabaseMapping.commit_session = counting
    try:
        yield count
    finally:
        DatabaseMapping.commit_session = real


@contextlib.contextmanager
def already_applied_records() -> Iterator[list[str]]:
    """Collect the messages of the ``_already_applied`` records logged while inside."""
    messages: list[str] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            message = record.getMessage()
            if message.startswith(ALREADY_APPLIED_PREFIX):
                messages.append(message)

    handler = _Handler(logging.INFO)
    root = logging.getLogger()
    level = root.level
    root.addHandler(handler)
    root.setLevel(min(level, logging.INFO) if level else logging.INFO)
    try:
        yield messages
    finally:
        root.removeHandler(handler)
        root.setLevel(level)


def _value(val, typ) -> str:
    parsed = from_database(val, typ)
    if hasattr(parsed, "to_dict"):
        return json.dumps(parsed.to_dict(), sort_keys=True, default=str)
    return repr(parsed)


def snapshot(path: Path) -> dict:
    """Everything a migration changes, keyed by name (not id), for :func:`diff`."""
    with DatabaseMapping(url_of(path)) as db:
        lists: dict = {}
        for x in db.get_items("list_value"):
            lists.setdefault(x["parameter_value_list_name"], []).append((x["index"], _value(x["value"], x["type"])))
        return {
            "definitions": {(d["entity_class_name"], d["name"]): (
                tuple(d["parameter_type_list"] or ()), _value(d["default_value"], d["default_type"]),
                d["parameter_value_list_name"], d["description"], d["parameter_group_name"])
                for d in db.get_items("parameter_definition")},
            "values": {(v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"],
                        v["alternative_name"]): _value(v["value"], v["type"]) for v in db.get_items("parameter_value")},
            "list_values": {name: tuple(sorted(members)) for name, members in lists.items()},
            "lists": sorted(x["name"] for x in db.get_items("parameter_value_list")),
            "classes": {c["name"]: (tuple(c["dimension_name_list"]), c["description"], c["active_by_default"])
                        for c in db.get_items("entity_class")},
            "entities": sorted((e["entity_class_name"], tuple(e["entity_byname"])) for e in db.get_items("entity")),
            "entity_alternatives": sorted((e["entity_class_name"], tuple(e["entity_byname"]), e["alternative_name"],
                                           e["active"]) for e in db.get_items("entity_alternative")),
            "alternatives": sorted(a["name"] for a in db.get_items("alternative")),
            "scenario_alternatives": sorted((a["scenario_name"], a["alternative_name"], a["rank"])
                                            for a in db.get_items("scenario_alternative")),
            "parameter_groups": sorted((g["name"], g["color"], g["priority"]) for g in db.get_items("parameter_group")),
            "metadata": sorted((m["name"], m["value"]) for m in db.get_items("metadata")),
        }


def diff(expected: dict, actual: dict) -> list[tuple]:
    """``(part, key, expected, actual)`` for every item where the two :func:`snapshot` results differ."""
    out = []
    for part, a in expected.items():
        b = actual[part]
        if a == b:
            continue
        if isinstance(a, dict):
            for key in sorted(set(a) | set(b), key=repr):
                if a.get(key, "<absent>") != b.get(key, "<absent>"):
                    out.append((part, key, a.get(key, "<absent>"), b.get(key, "<absent>")))
        else:
            out += [(part, key, "present", "<absent>") for key in sorted(set(a) - set(b), key=repr)]
            out += [(part, key, "<absent>", "present") for key in sorted(set(b) - set(a), key=repr)]
    return out
