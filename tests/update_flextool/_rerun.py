"""Shared helpers for the migration rerun tests: restamp a database, stop a migration after a commit, compare two
databases item by item, build the v75b seed."""
from __future__ import annotations

import contextlib
import io
import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from spinedb_api import DatabaseMapping, from_database, import_data, to_database

from flextool._resources import package_data_path
from flextool.update_flextool.db_migration import ALREADY_APPLIED_PREFIX, migrate_database
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.fixture_schema import unpack_value
from flextool.update_flextool.initialize_database import initialize_database

#: The bundled pre-v26 templates (the v2 and v25 seeds).
TEMPLATES = package_data_path("schemas/pre_v26")
#: The data of the v75 seed: alternatives, scenarios, entities and values of a two-period stochastic model (its
#: definitions are not those of a migrated v75 database: only the data sections are read).
V75_SEED_DATA = Path(__file__).parent / "data" / "v75_seed_stoch_two_period.json"


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
    with open_database_mapping(url_of(path)) as db:
        restamp(db, version)
        db.commit_session(f"restamp {version}")


def version_of(path: Path) -> float:
    with open_database_mapping(url_of(path)) as db:
        pdef = db.get_item("parameter_definition", entity_class_name="model", name="version")
        return from_database(pdef["default_value"], pdef["default_type"])


@contextlib.contextmanager
def kill_after_commit(n: int | None) -> Iterator[list[str]]:
    """Record the commits of every ``DatabaseMapping`` while inside; raise :class:`Kill` right after the *n*-th
    (``None``: never).  Yields the list of their messages."""
    real = DatabaseMapping.commit_session
    messages: list[str] = []

    def counting(self, *args, **kwargs):
        result = real(self, *args, **kwargs)
        messages.append(args[0] if args else kwargs.get("comment", ""))
        if n is not None and len(messages) == n:
            raise Kill()
        return result

    DatabaseMapping.commit_session = counting
    try:
        yield messages
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


def commit_messages(path: Path, up_to: int) -> list[str]:
    """The messages of the commits migrating a copy of *path* to *up_to* makes (the last one is the version
    commit)."""
    probe = copy_db(path, path.with_name("probe.sqlite"))
    with kill_after_commit(None) as messages:
        migrate(probe, up_to=up_to)
    probe.unlink()
    return messages


def kill_and_rerun(clean: dict, k: int, j: int, work: Path) -> list:
    """Run step k on clean k-1, kill it right after its j-th commit, migrate to k again: the differences from clean
    k (an error as one entry)."""
    path = copy_db(clean[k - 1], work / f"k{k}_j{j}.sqlite")
    try:
        try:
            with kill_after_commit(j):
                migrate(path, up_to=k)
            return [("not killed", j)]
        except Kill:
            pass
        if version_of(path) != k - 1:
            return [("stamp after the kill", version_of(path))]
        migrate(path, up_to=k)
        return diff(snapshot(clean[k]), snapshot(path))
    except Exception as exc:  # noqa: BLE001 - reported as the case's result
        return [("error", f"{type(exc).__name__}: {exc}")]
    finally:
        path.unlink(missing_ok=True)


def import_v75_seed_data(path: Path) -> None:
    """Add the data sections of :data:`V75_SEED_DATA` to the database file *path* (its schema at 75)."""
    with open(V75_SEED_DATA, encoding="utf-8") as f:
        data = json.load(f)
    with open_database_mapping(url_of(path)) as db:
        _, errors = import_data(db, unparse_value=unpack_value, **{key: data[key] for key in (
            "alternatives", "scenarios", "scenario_alternatives", "entities", "entity_alternatives",
            "parameter_values")})
        if errors:
            raise ValueError(f"v75 seed data: {errors[:3]}")
        db.commit_session("v75 seed data")


def build_v75b(path: Path) -> Path:
    """The v75b seed at *path*: the v2 template migrated to 75 (a real v75 schema), an empty ``Base`` (as a new
    SpineDB database has) and the data of the v75 seed."""
    quiet(initialize_database, str(TEMPLATES / "flextool_template_v2.json"), str(path))
    migrate(path, up_to=75)
    with open_database_mapping(url_of(path)) as db:
        db.add_alternative(name="Base")
        db.commit_session("Base")
    import_v75_seed_data(path)
    return path


_V75B: dict[Path, Path] = {}


def v75b_url(tmp_path_factory, target: Path) -> str:
    """The URL of a copy at *target* of the v75b seed, built once per session (about 14 s) in a folder of
    *tmp_path_factory*."""
    base = tmp_path_factory.getbasetemp()
    if base not in _V75B:
        _V75B[base] = build_v75b(tmp_path_factory.mktemp("v75b") / "v75b.sqlite")
    return url_of(copy_db(_V75B[base], target))


def _value(val, typ) -> str:
    parsed = from_database(val, typ)
    if hasattr(parsed, "to_dict"):
        return json.dumps(parsed.to_dict(), sort_keys=True, default=str)
    return repr(parsed)


def snapshot(path: Path) -> dict:
    """Everything a migration changes, keyed by name (not id), for :func:`diff`."""
    with open_database_mapping(url_of(path)) as db:
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
