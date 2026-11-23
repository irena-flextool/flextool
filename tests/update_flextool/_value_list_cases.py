"""Made-up W-0490 databases for the value-list repair tool's tests (``test_value_list_repair_*.py``).

:func:`make_hit` copies the current schema database (``schema_db_url``) and gives it, with raw ``sqlite3``, the
history and the ``invest_methods`` list of a database that step 22 hit (as the design's case F2):

1. "before" values are written with spinedb_api (their commits come first);
2. version commits →20, →21; C = ``"Added new parameter value lists"``; C+1 = ``"Added cumulative investments"``;
   version commit →22 (the session 21→22 ends in the W-0490 window);
3. the shipped ``not_allowed`` row R (index 0) gets the text ``cumulative_limits`` and commit C; the references to
   the shipped ``cumulative_limits`` row are pointed to R and that row is deleted (the bug appended no row);
4. "after" values are written with spinedb_api (``cumulative_limits`` resolves to R, as in a real buggy file).

The siblings keep the schema's commit (2, older than the session).  No ``tests.db_utils`` import, so the polars
string cache stays on.  TODO(v5.0.0): remove with W-0490 (except :func:`make_lost_defaults`, part B).

:func:`make_lost_defaults` gives a version 82 copy of the schema database whose named list defaults are ``NULL``,
as after an upgrade before 4.6.3 (part B of the tool, the run note W-0491).
"""
from __future__ import annotations

import contextlib
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

VERSION_PREFIX = "Updated Flextool data structure to version "
LIST_STEP = "Added new parameter value lists"
STEP_22 = "Added cumulative investments"
SESSION = [("20", "2024-05-27"), ("21", "2024-07-17")]
HIT_DATE = "2024-09-19"
RETRY_DATE = "2025-08-01"  # after the window
LEGACY = "invest_legacy"  # a member FlexTool never shipped


@dataclass(frozen=True)
class HitCase:
    """A made-up hit file and the ids its construction gave (the tests' oracle)."""

    path: Path
    list_id: int  # invest_methods
    r_id: int  # the overwritten row (the schema's not_allowed)
    start_commit: int  # version commit →21
    c: int  # "Added new parameter value lists"
    end_commit: int  # version commit →22
    template_commit: int  # the siblings' commit


def sql(path: Path, statement: str, params: tuple = ()) -> list[tuple]:
    """Run one statement on *path* and commit; returns its rows."""
    with contextlib.closing(sqlite3.connect(path)) as con:
        with con:
            return con.execute(statement, params).fetchall()


def add_commit(path: Path, comment: str, date: str) -> int:
    """Append a ``commit`` row and return its id."""
    with contextlib.closing(sqlite3.connect(path)) as con:
        with con:
            cur = con.execute('INSERT INTO "commit" (comment, date, user) VALUES (?, ?, ?)',
                              (comment, f"{date} 12:00:00", "spinedb_api"))
            return int(cur.lastrowid)


def member_id(path: Path, list_name: str, member: str) -> int | None:
    rows = sql(path, "SELECT lv.id FROM list_value lv JOIN parameter_value_list l ON lv.parameter_value_list_id = "
                     "l.id WHERE l.name = ? AND CAST(lv.value AS TEXT) = ?", (list_name, f'"{member}"'))
    return rows[0][0] if rows else None


def write_values(path: Path, items) -> None:
    """``(class, entity, alternative, member)`` ``invest_method`` values, written with spinedb_api (one commit)."""
    if not items:
        return
    from spinedb_api import to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{path}") as db:
        for cls, entity, alternative, member in items:
            if not db.get_items("entity", entity_class_name=cls, name=entity):
                db.add_entity(entity_class_name=cls, name=entity)
            value, typ = to_database(member)
            db.add_parameter_value(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=
                                   "invest_method", alternative_name=alternative, value=value, type=typ)
        db.commit_session("Updated")


def write_items(path: Path, values=(), scenarios=None) -> None:
    """``(class, entity, parameter, alternative, value)`` values and ``{scenario: [alternative, ...]}`` (lowest rank
    first), written with spinedb_api in one commit; entities (active in the value's alternative) and alternatives
    are added as needed."""
    if not values and not scenarios:
        return
    from spinedb_api import to_database

    from flextool.update_flextool.db_handle import open_database_mapping

    with open_database_mapping(f"sqlite:///{path}") as db:
        alternatives = {a for _, _, _, a, _ in values} | {a for alts in (scenarios or {}).values() for a in alts}
        for name in sorted(alternatives):
            if not db.get_items("alternative", name=name):
                db.add_alternative(name=name)
        for cls, entity, param, alternative, value in values:
            if not db.get_items("entity", entity_class_name=cls, name=entity):
                db.add_entity(entity_class_name=cls, name=entity)
            if not db.get_items("entity_alternative", entity_class_name=cls, entity_byname=(entity,),
                                alternative_name=alternative):
                db.add_entity_alternative(entity_class_name=cls, entity_byname=(entity,),
                                          alternative_name=alternative, active=True)
            val, typ = to_database(value)
            db.add_parameter_value(entity_class_name=cls, entity_byname=(entity,), parameter_definition_name=param,
                                   alternative_name=alternative, value=val, type=typ)
        for scenario, alts in (scenarios or {}).items():
            db.add_scenario(name=scenario)
            for rank, alternative in enumerate(alts, start=1):
                db.add_scenario_alternative(scenario_name=scenario, alternative_name=alternative, rank=rank)
        db.commit_session("Updated")


FINGERPRINT = "Added cumulative investments and parameter value types"
FINGERPRINT_DATE = "2024-08-28"
V77_COMMIT = "v77: storage_start_end_method default fix_nothing"
V78_COMMIT = "v78: descriptions"
UPGRADE_DATE = "2025-10-01"


@dataclass(frozen=True)
class LostCase:
    """A made-up v82 file with lost list defaults and the commit ids its construction gave (the tests' oracle)."""

    path: Path
    emptied_commit: int  # the commit of the emptied definitions (a migration step)
    v77_commit: int | None
    last_version_commit: int


def clear_defaults(path: Path, names, commit: int | None) -> None:
    """Set the ``class.parameter`` definitions *names* to a ``NULL`` default with commit *commit*."""
    for name in names:
        cls, param = name.split(".")
        rows = sql(path, "UPDATE parameter_definition SET default_value = NULL, default_type = NULL, commit_id = ? "
                         "WHERE name = ? AND entity_class_id = (SELECT id FROM entity_class WHERE name = ?) "
                         "RETURNING id", (commit, param, cls))
        assert len(rows) == 1, name


def make_lost_defaults(tmp_path: Path, names, fingerprint: bool = True, v77: bool = True, *, schema_db=None,
                       name: str = "lost.sqlite", before=(), after=(), scenarios=None) -> LostCase:
    """A version 82 copy of the schema database (*schema_db*, built here when ``None``) whose ``class.parameter``
    defaults *names* are ``NULL``.  History by raw ``sqlite3``: *before* values (:func:`write_items`, with
    *scenarios*), the FlexTool 3.x v22 step commit (the fingerprint, dated 2024-08-28) and the version commit
    →22 when *fingerprint*, the v77 step commit when *v77*, the v78 step commit (the emptied definitions'
    commit, as on real files) and the version commit →82; then *after* values."""
    path = tmp_path / name
    if schema_db is None:
        from flextool.update_flextool.initialize_database import initialize_database

        schema_db = tmp_path / f"{name}.schema.sqlite"
        initialize_database(str(Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "spinedb_schema.json"),
                            str(schema_db))
    shutil.copy2(Path(str(schema_db).removeprefix("sqlite:///")), path)
    write_items(path, before, scenarios)
    if fingerprint:
        add_commit(path, FINGERPRINT, FINGERPRINT_DATE)
        add_commit(path, VERSION_PREFIX + "22", FINGERPRINT_DATE)
    v77_commit = add_commit(path, V77_COMMIT, UPGRADE_DATE) if v77 else None
    emptied = add_commit(path, V78_COMMIT, UPGRADE_DATE)
    clear_defaults(path, names, emptied)
    last = add_commit(path, VERSION_PREFIX + "82", UPGRADE_DATE)
    write_items(path, after)
    return LostCase(path, emptied, v77_commit, last)


def _repoint(path: Path, old: int, new: int) -> None:
    """Point every list reference (default or value) to member *old* to member *new*."""
    sql(path, "UPDATE parameter_definition SET default_value = ? WHERE default_type = 'list_value_ref' AND "
              "CAST(default_value AS TEXT) = ?", (str(new).encode(), str(old)))
    sql(path, "UPDATE parameter_value SET value = ? WHERE type = 'list_value_ref' AND CAST(value AS TEXT) = ?",
        (str(new).encode(), str(old)))


def make_archive(path: Path, archive: Path) -> None:
    """Copy *path* to *archive* and make the copy a data version 21 file (``make_hit``'s *archive*)."""
    shutil.copy2(path, archive)
    sql(archive, "UPDATE parameter_definition SET default_value = ?, default_type = 'float' WHERE name = 'version' "
                 "AND entity_class_id = (SELECT id FROM entity_class WHERE name = 'model')", (b"21.0",))
    appended = member_id(archive, "invest_methods", "cumulative_limits")
    assert not sql(archive, "SELECT id FROM parameter_value WHERE type = 'list_value_ref' AND CAST(value AS TEXT) = "
                            "?", (str(appended),))
    sql(archive, "DELETE FROM list_value WHERE id = ?", (appended,))


def make_hit(tmp_path: Path, schema_db: str | Path, *, name: str = "hit.sqlite", before=(), after=(),
             legacy: bool = False, pre_v77_group: bool = False, overwrite: bool = True,
             retried: bool = False, archive: Path | None = None) -> HitCase:
    """A step-22 hit file (module docstring).  *legacy*: R held the legacy member ``LEGACY`` before the session,
    and ``not_allowed`` is another row (index 14, commit NULL) that the references to R had used (every shipped
    member present; "before" values may name either).  *pre_v77_group*: the group
    ``invest_method`` is bound to ``invest_methods`` with the default R (before v77).  ``overwrite=False`` leaves
    the list as shipped (``cumulative_limits`` appended at index 13 with commit C: the bug did not fire).
    *retried*: the migration failed after its value-list commit (in the window; R keeps that commit, C - 1) and was
    retried after the window (C, C+1 and →22 dated ``RETRY_DATE``).  *archive*: a copy taken right before the
    session's first list-value commit (after the "before" values and the version commits →20, →21), made a data
    version 21 file by raw SQL (``model.version`` 21, no ``cumulative_limits`` member; no commit added, so it is
    the same lineage as the hit file)."""
    path = tmp_path / name
    shutil.copy2(Path(str(schema_db).removeprefix("sqlite:///")), path)
    list_id = sql(path, "SELECT id FROM parameter_value_list WHERE name = 'invest_methods'")[0][0]
    r_id, appended = member_id(path, "invest_methods", "not_allowed"), member_id(path, "invest_methods",
                                                                                   "cumulative_limits")
    template_commit = sql(path, "SELECT commit_id FROM list_value WHERE id = ?", (r_id,))[0][0]
    if pre_v77_group:
        sql(path, "UPDATE parameter_definition SET parameter_value_list_id = ?, default_value = ?, default_type = "
                  "'list_value_ref' WHERE name = 'invest_method' AND entity_class_id = (SELECT id FROM entity_class "
                  "WHERE name = 'group')", (list_id, str(r_id).encode()))
    if legacy:
        sql(path, "UPDATE list_value SET value = ? WHERE id = ?", (f'"{LEGACY}"'.encode(), r_id))
        sql(path, 'INSERT INTO list_value (parameter_value_list_id, "index", value, type, commit_id) '
                  "VALUES (?, 14, ?, 'str', NULL)", (list_id, b'"not_allowed"'))
        _repoint(path, r_id, member_id(path, "invest_methods", "not_allowed"))
    write_values(path, before)
    for version, date in SESSION:
        start = add_commit(path, VERSION_PREFIX + version, date)
    if archive is not None:
        make_archive(path, archive)
    failed = add_commit(path, LIST_STEP, HIT_DATE) if retried else None
    date = RETRY_DATE if retried else HIT_DATE
    c = add_commit(path, LIST_STEP, date)
    add_commit(path, STEP_22, date)
    end = add_commit(path, VERSION_PREFIX + "22", date)
    if overwrite:
        sql(path, "UPDATE list_value SET value = ?, commit_id = ? WHERE id = ?",
            (b'"cumulative_limits"', failed or c, r_id))
        _repoint(path, appended, r_id)
        sql(path, "DELETE FROM list_value WHERE id = ?", (appended,))
    else:
        sql(path, "UPDATE list_value SET commit_id = ? WHERE id = ?", (c, appended))
    write_values(path, after)
    return HitCase(path, list_id, r_id, start, c, end, template_commit)
