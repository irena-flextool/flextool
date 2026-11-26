"""The current data version: its rerunnable step and the refresh of a database already at it."""
from typing import Callable

from spinedb_api import from_database

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration.common import _append_list_members, _pair_is, _replace_text_pairs
from flextool.update_flextool.db_migration.v82 import _v82_definitions, _v82_definitions_part, _v82_superseded_texts
from flextool.update_flextool.db_migration.v83 import (
    _replace_release_texts,
    _V83_NEW_MEMBERS,
    _V83_RELEASE_TEXTS,
    _v83_superseded_texts,
)


# --- Version 84: one branching setting of investment solves (item 1024): ``v84_branching`` ------------------------

def _migrate_v84_schema_batch(db) -> None:
    """All schema changes of database version 84 (v83 -> v84), in one step:
    :func:`v84_branching._migrate_v84_schema_batch` (imported here: it uses the helpers of this module)."""
    from flextool.update_flextool.db_migration import v84_branching

    v84_branching._migrate_v84_schema_batch(db)


#: The schema step of the current data version.  The template sync reruns it on the master template, so that schema
#: entries appended to its tables after the version was first stamped reach the shipped schema (a database at the
#: current version is otherwise not migrated again).
RERUNNABLE_HEAD_STEP: dict[int, Callable] = {84: _migrate_v84_schema_batch}


def _head_schema_additions() -> tuple[set[tuple[str, str]], dict[str, tuple[str, ...]]]:
    """``(definitions, list members)`` the schema steps of the current release add: what a database at that
    version must have (on the classes it has).  4.7.0 has two steps, 83 and 84 (v84 inside the release)."""
    if FLEXTOOL_DB_VERSION != 84:
        return set(), {}
    from flextool.update_flextool.db_migration.v84_branching import _V84_LIST, _V84_METHODS, _v84_definitions

    return set(_v82_definitions()) | set(_v84_definitions()), {**_V83_NEW_MEMBERS, _V84_LIST: _V84_METHODS}


def _head_schema_missing_ro(path) -> bool:
    """Whether the database *path* (read read-only) lacks a definition (on a class it has) or a list member (of a
    list it has) that the schema step of the current data version adds (``True`` when it cannot be read)."""
    import sqlite3

    from flextool.update_flextool.frozen import _connect_ro

    definitions, lists = _head_schema_additions()
    if not definitions and not lists:
        return False
    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return True
    try:
        classes = {r[0] for r in con.execute("SELECT name FROM entity_class").fetchall()}
        have = set(con.execute(
            "SELECT ec.name, pd.name FROM parameter_definition pd "
            "JOIN entity_class ec ON pd.entity_class_id = ec.id").fetchall())
        members: dict[str, set] = {r[0]: set() for r in con.execute("SELECT name FROM parameter_value_list")}
        for name, value, typ in con.execute(
                "SELECT pvl.name, lv.value, lv.type FROM list_value lv "
                "JOIN parameter_value_list pvl ON lv.parameter_value_list_id = pvl.id").fetchall():
            members.setdefault(name, set()).add(from_database(value, typ))
    except sqlite3.Error:
        return True
    finally:
        con.close()
    if any(cls in classes and (cls, name) not in have for cls, name in definitions):
        return True
    return any(name in members and not set(wanted) <= members[name] for name, wanted in lists.items())


def _head_superseded() -> dict[tuple[str, str], list[tuple[str | None, str]]]:
    """The superseded shipped ``(default, description)`` pairs that the refresh of the current data version replaces,
    per definition.  None at 83: every database at 82 passed the v81 refresh inside step 82, and the v82 entries are
    description-only (:func:`_head_superseded_texts`)."""
    return {}


def _head_superseded_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the superseded shipped descriptions that the refresh of the current data
    version replaces by text alone, whatever the default (:func:`_v82_superseded_texts`,
    :func:`_v83_superseded_texts` and :func:`v84_branching._v84_dispatch_realization_texts`, applied in that order)."""
    if FLEXTOOL_DB_VERSION != 84:
        return {}
    return {**_v82_superseded_texts(), **_v83_superseded_texts()}


def _is_superseded_pair(cls: str, name: str, default, description: str | None) -> bool:
    """Whether the definition ``(cls, name)`` has exactly a superseded shipped pair of :func:`_head_superseded`:
    its parsed default and its description both equal to what FlexTool shipped.  The test of
    :func:`_head_defaults_stale_ro`; :func:`_replace_superseded_pairs` applies the same one (:func:`_pair_is`)."""
    return any(_pair_is(pair, default, description) for pair in _head_superseded().get((cls, name), ()))


def _head_defaults_stale_ro(path) -> bool:
    """Whether the database *path* (read read-only) has a definition with a superseded shipped default and
    description (:func:`_is_superseded_pair`) or a superseded shipped description (:func:`_head_superseded_texts`
    and :data:`_V83_RELEASE_TEXTS`, by text alone; ``True`` when it cannot be read)."""
    import sqlite3

    from flextool.update_flextool.frozen import _connect_ro

    superseded = _head_superseded()
    texts = {key: old for key, (old, _new) in _head_superseded_texts().items()}
    release = {key: olds for key, (olds, _new) in _V83_RELEASE_TEXTS.items()} if FLEXTOOL_DB_VERSION == 84 else {}
    if FLEXTOOL_DB_VERSION == 84:
        from flextool.update_flextool.db_migration.v84_branching import _v84_dispatch_realization_texts

        texts.update({key: old for key, (old, _new) in _v84_dispatch_realization_texts().items()})
    if not superseded and not texts and not release:
        return False
    names = sorted({name for _cls, name in (*superseded, *texts, *release)})
    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return True
    try:
        rows = con.execute(
            "SELECT ec.name, pd.name, pd.default_value, pd.default_type, pd.description FROM parameter_definition pd "
            "JOIN entity_class ec ON pd.entity_class_id = ec.id WHERE pd.name IN (%s)" % ",".join("?" * len(names)),
            names).fetchall()
        # A default from a value list is stored as a reference to its list member.
        members = {int(i): (v, t) for i, v, t in con.execute("SELECT id, value, type FROM list_value").fetchall()}
    except sqlite3.Error:
        return True
    finally:
        con.close()

    def parsed(value, typ):
        if value is None:
            return None
        if typ == "list_value_ref":
            # A payload that is not an id (a member's text, item 279) reads as None, as in spinedb_api.
            raw = value.encode() if isinstance(value, str) else bytes(value)
            member = members.get(int(raw)) if raw.isdigit() else None
            return from_database(*member) if member is not None else None
        return from_database(value, typ)

    return any(texts.get((cls, name)) == (desc or "") or (desc or "") in release.get((cls, name), ())
               or _is_superseded_pair(cls, name, parsed(value, typ), desc) for cls, name, value, typ, desc in rows)


def _refresh_head_schema(db) -> None:
    """Bring a database at the current data version to the schema of its release's steps: the v82 definitions it
    lacks (:func:`_v82_definitions_part`; an existing definition keeps its fields), the members v83 adds to the lists
    it has (:data:`_V83_NEW_MEMBERS`; no value is written, no member removed, no default changed: a database at 84
    has had steps 83 and 84), the v84 value list and definition it lacks (:func:`v84_branching._v84_definitions_part`) and the current description in place of a superseded shipped one (:func:`_head_superseded_texts`
    and :data:`_V83_RELEASE_TEXTS`, by text alone).  A description the user edited, a default the user set and every value are kept; otherwise a
    description is never rewritten.  Idempotent."""
    if FLEXTOOL_DB_VERSION == 84:
        from flextool.update_flextool.db_migration.v84_branching import _v84_definitions_part, _v84_dispatch_realization_texts

        _v82_definitions_part(db, update=False)
        for name, members in _V83_NEW_MEMBERS.items():
            if db.get_item("parameter_value_list", name=name):
                _append_list_members(db, name, members)
        _v84_definitions_part(db, update=False)
        _replace_text_pairs(db, _head_superseded_texts())
        _replace_release_texts(db)
        _replace_text_pairs(db, _v84_dispatch_realization_texts())
