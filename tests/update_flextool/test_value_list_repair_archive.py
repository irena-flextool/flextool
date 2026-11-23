"""The value-list repair tool's archive mode (W-0490 package P3): which archive is accepted, the same-lineage
same-row rule (critique R4), and the archive's evidence on each suspect.

Every hit file is made by ``_value_list_cases.make_hit``; its *archive* is a copy taken right before the session's
first list-value commit, made a data version 21 file (no ``cumulative_limits`` member) with no commit added, so
it is the same lineage as the hit file.  The oracle is the construction: a "before" value written as
``not_allowed`` references R, which read ``not_allowed`` in the archive; every archive change below is raw SQL that
adds no commit (the lineage stays) unless the test changes a commit row on purpose.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import contextlib
from pathlib import Path

from flextool.update_flextool import value_list_archive as va
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs
from tests.update_flextool._value_list_cases import LEGACY, add_commit, make_hit, member_id, sql, write_values

LATER = "2025-01-10"
_PV = ("SELECT pv.id FROM parameter_value pv JOIN entity e ON e.id = pv.entity_id WHERE e.name = ? AND "
       "pv.parameter_definition_id = (SELECT pd.id FROM parameter_definition pd JOIN entity_class ec ON ec.id = "
       "pd.entity_class_id WHERE pd.name = 'invest_method' AND ec.name = ?)")


def _pv(path: Path, entity: str, cls: str = "unit") -> int:
    return sql(path, _PV, (entity, cls))[0][0]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _hit(tmp_path: Path, schema_db_url, **kw) -> tuple:
    archive = tmp_path / "old.sqlite"
    case = make_hit(tmp_path, schema_db_url, archive=archive, **kw)
    return case, archive


def _evidence(case, archive: Path, tmp_path: Path) -> tuple[vr.Analysis, va.Archive, dict]:
    """Analyse a snapshot of the hit file, read *archive* and apply it; the archive must stay byte-identical."""
    folder = tmp_path / "snap"
    folder.mkdir(parents=True, exist_ok=True)
    a = vr.analyse(vr.snapshot(case.path, folder))
    before = _sha(archive) if archive.is_file() else None
    ar = va.read_archive(archive, a)
    found = va.apply_archive(a, vs.suspects(a), ar)
    if before is not None:
        assert _sha(archive) == before
    return a, ar, {(s.kind, s.cls, s.entity): s for s in found}


def _variant(archive: Path, name: str, *statements) -> Path:
    """A copy of *archive* changed by raw SQL statements ``(sql, params)``."""
    copy = archive.with_name(name)
    shutil.copy2(archive, copy)
    for statement, params in statements:
        sql(copy, statement, params)
    return copy


def test_same_row_changed_missing_and_added_values(tmp_path, schema_db_url) -> None:
    """U1 is the same row in the archive (id, commit, entity, alternative, definition): certain (archive),
    ``not_allowed``.  U2 was saved again after the overwrite (same id, new commit): "changed since", no
    suggestion.  U9's value was removed from the archive: "no value (default not_allowed)", the position proof
    still makes it certain.  U3 (written after, a new entity): "not in the archive".  The archive is
    byte-identical, no handle is left open and it can be deleted."""
    before = [("unit", "U1", "base", "not_allowed"), ("unit", "U2", "base", "not_allowed"),
              ("unit", "U9", "base", "not_allowed")]
    case, archive = _hit(tmp_path, schema_db_url, before=before, after=[("unit", "U3", "base", "cumulative_limits")])
    later = add_commit(case.path, "Updated", LATER)
    sql(case.path, "UPDATE parameter_value SET commit_id = ? WHERE id = ?", (later, _pv(case.path, "U2")))
    sql(archive, "DELETE FROM parameter_value WHERE id = ?", (_pv(archive, "U9"),))
    a, ar, found = _evidence(case, archive, tmp_path)
    assert (a.state, ar.status, ar.same_lineage, ar.version) == (vr.HIT, va.ACCEPTED, True, 21)
    got = {e: (s.label, s.suggestion, s.archive, s.archive_note) for (k, _, e), s in found.items() if k == vs.VALUE}
    assert got == {
        "U1": (vs.CERTAIN_ARCHIVE, "set:not_allowed", "not_allowed",
               "your old copy has this row with the same commit: it was not_allowed"),
        "U2": (vs.AFTER, None, "not_allowed", "was not_allowed, changed since"),
        "U9": (vs.CERTAIN, "set:not_allowed", None, "archive: no value (default not_allowed)"),
        "U3": (vs.AFTER, None, None, "not in the archive (added later)")}
    assert found[("value", "unit", "U1")].certain and not found[("value", "unit", "U2")].certain
    if Path("/proc/self/fd").is_dir():
        assert _open_files_under(tmp_path) == []
    archive.unlink()


def test_renamed_entity_and_reused_name(tmp_path, schema_db_url) -> None:
    """R4.  U1 (``invest_period``) is renamed ``U1_old`` and U2 (``not_allowed``) is renamed ``U1`` after the
    session.  The suspect now called U1 is U2's row: the row id gives ``not_allowed``, never the name's
    ``invest_period``.  A new value on a reused name ``U7`` (old U7 renamed away) is evidence only.  Without the
    same lineage (one archive commit row differs) the same name match is a conflict, never certain (archive)."""
    before = [("unit", "U1", "base", "invest_period"), ("unit", "U2", "base", "not_allowed"),
              ("unit", "U7", "base", "invest_total")]
    case, archive = _hit(tmp_path, schema_db_url, before=before)
    renamed = add_commit(case.path, "Updated", LATER)
    for old, new in (("U1", "U1_old"), ("U2", "U1"), ("U7", "U7_old")):
        sql(case.path, "UPDATE entity SET name = ?, commit_id = ? WHERE name = ?", (new, renamed, old))
    write_values(case.path, [("unit", "U7", "base", "cumulative_limits")])
    _, ar, found = _evidence(case, archive, tmp_path)
    assert ar.same_lineage
    u1, u7 = found[("value", "unit", "U1")], found[("value", "unit", "U7")]
    assert (u1.label, u1.suggestion, u1.archive) == (vs.CERTAIN_ARCHIVE, "set:not_allowed", "not_allowed")
    assert (u7.label, u7.suggestion, u7.archive_note) == (vs.AFTER, None, "was invest_total before the overwrite")

    other = _variant(archive, "other.sqlite", ('UPDATE "commit" SET comment = ? WHERE id = 2', ("Import 3 items",)))
    _, ar, found = _evidence(case, other, tmp_path / "o")
    u1 = found[("value", "unit", "U1")]
    assert (ar.status, ar.same_lineage) == (va.ACCEPTED, False)
    assert (u1.label, u1.suggestion, u1.archive) == (va.CONFLICT, None, "invest_period")

    # Same lineage, but the archive row with U2's id belongs to another entity: no certainty from the id.
    moved = _variant(archive, "moved.sqlite",
                     ("INSERT INTO entity (class_id, name) VALUES ((SELECT id FROM entity_class WHERE name = 'unit'), "
                      "'X')", ()),
                     ("UPDATE parameter_value SET entity_id = (SELECT id FROM entity WHERE name = 'X') WHERE id = ?",
                      (_pv(archive, "U2"),)))
    _, ar, found = _evidence(case, moved, tmp_path / "m")
    u1 = found[("value", "unit", "U1")]
    assert (ar.same_lineage, u1.label, u1.suggestion, u1.archive) == (True, va.CONFLICT, None, "invest_period")


def test_refused_and_unused_archives(tmp_path, schema_db_url) -> None:
    """Refused: data version 22, ``cumulative_limits`` in the list (already affected), the last upgrade not before
    the overwrite, the lost member missing, a layout without ``list_value``, not a FlexTool file, no file.  On a
    file without an overwrite the archive is unused.  A refused archive leaves the suspects as they are."""
    case, archive = _hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")])
    list_id = sql(archive, "SELECT id FROM parameter_value_list WHERE name = 'invest_methods'")[0][0]
    r_id = member_id(archive, "invest_methods", "not_allowed")
    version = ("UPDATE parameter_definition SET default_value = ? WHERE name = 'version'", (b"22.0",))
    affected = ('INSERT INTO list_value (parameter_value_list_id, "index", value, type) VALUES (?, 13, ?, \'str\')',
                (list_id, b'"cumulative_limits"'))
    late = ('UPDATE "commit" SET date = ? WHERE comment = ?',
            ("2024-09-20 00:00:00", "Updated Flextool data structure to version 21"))
    lost = ("UPDATE list_value SET value = ? WHERE id = ?", (f'"{LEGACY}"'.encode(), r_id))
    layout = ("DROP TABLE list_value", ())
    text = tmp_path / "notes.sqlite"
    text.write_text("not a database", encoding="utf-8")
    cases = {
        _variant(archive, "v22.sqlite", version): "data version 22",
        _variant(archive, "affected.sqlite", affected): "already affected",
        _variant(archive, "late.sqlite", late): "its last upgrade (2024-09-20) is not before the overwrite "
                                                "(2024-09-19)",
        _variant(archive, "lost.sqlite", lost): "does not hold 'not_allowed'",
        _variant(archive, "layout.sqlite", layout): "layout",
        text: "not a FlexTool database",
        tmp_path / "missing.sqlite": "does not exist",
    }
    a = vr.analyse(vr.snapshot(case.path, tmp_path))
    plain = vs.suspects(a)
    for path, reason in cases.items():
        ar = va.read_archive(path, a)
        assert ar.refused and reason in ar.reason, (path.name, ar.reason)
        assert va.apply_archive(a, plain, ar) == plain

    not_hit = make_hit(tmp_path, schema_db_url, name="intact.sqlite", overwrite=False)
    b = vr.analyse(vr.snapshot(not_hit.path, tmp_path))
    assert b.state == vr.NOT_HIT and va.read_archive(archive, b).status == va.UNUSED


def test_archive_without_history_is_evidence_only(tmp_path, schema_db_url) -> None:
    """An archive without version commits is accepted with the warning; U1 (same row) stays certain by position
    only, with the archive as information."""
    case, archive = _hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")])
    bare = _variant(archive, "bare.sqlite", ('DELETE FROM "commit" WHERE comment LIKE ?', (vlh._PREFIX + "%",)))
    _, ar, found = _evidence(case, bare, tmp_path)
    u1 = found[("value", "unit", "U1")]
    assert (ar.status, ar.reason, ar.same_lineage) == (va.NO_HISTORY, va.NO_HISTORY_WARNING, False)
    assert (u1.label, u1.suggestion, u1.archive_note) == (vs.CERTAIN, "set:not_allowed",
                                                          "was not_allowed before the overwrite")


def test_same_row_is_certain_when_the_position_proof_fails(tmp_path, schema_db_url) -> None:
    """``hit-unanchored`` (a retried migration) is never certain by position; the same row in a same-lineage
    archive makes U1 certain (archive).  Without the lineage it stays "probably", no suggestion."""
    case, archive = _hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")], retried=True)
    a, ar, found = _evidence(case, archive, tmp_path)
    u1 = found[("value", "unit", "U1")]
    assert (a.state, ar.same_lineage) == (vr.HIT_UNANCHORED, True)
    assert (u1.label, u1.suggestion, u1.certain) == (vs.CERTAIN_ARCHIVE, "set:not_allowed", True)

    other = _variant(archive, "other.sqlite", ('UPDATE "commit" SET user = ? WHERE id = 1', ("someone",)))
    _, ar, found = _evidence(case, other, tmp_path / "o")
    u1 = found[("value", "unit", "U1")]
    assert (ar.same_lineage, u1.label, u1.suggestion, u1.archive) == (False, "probably not_allowed", None,
                                                                       "not_allowed")


def test_conflict_legacy_member_and_default(tmp_path, schema_db_url) -> None:
    """The archive's same row names ``invest_total`` while the position proof says ``not_allowed``: a conflict,
    no suggestion.  The archive's unit default ``invest_period`` is information only: the shipped default stays
    recommended with the reason.  In a legacy-member hit the archive's row reads the legacy member: no
    suggestion."""
    case, archive = _hit(tmp_path, schema_db_url, before=[("unit", "U1", "base", "not_allowed")])
    total = str(member_id(archive, "invest_methods", "invest_total")).encode()
    period = str(member_id(archive, "invest_methods", "invest_period")).encode()
    sql(archive, "UPDATE parameter_value SET value = ? WHERE id = ?", (total, _pv(archive, "U1")))
    sql(archive, "UPDATE parameter_definition SET default_value = ? WHERE name = 'invest_method' AND "
                 "entity_class_id = (SELECT id FROM entity_class WHERE name = 'unit')", (period,))
    _, _, found = _evidence(case, archive, tmp_path)
    u1, default = found[("value", "unit", "U1")], found[("default", "unit", "")]
    assert (u1.label, u1.suggestion, u1.archive, u1.certain) == (va.CONFLICT, None, "invest_total", False)
    assert (default.suggestion, default.reason, default.archive_note) == (
        "set:not_allowed", vs.DEFAULT_REASON, "your old copy had invest_period")

    (tmp_path / "l").mkdir()
    legacy, legacy_archive = _hit(tmp_path / "l", schema_db_url, legacy=True, before=[("unit", "U1", "base", LEGACY)])
    _, ar, found = _evidence(legacy, legacy_archive, tmp_path / "l")
    u1 = found[("value", "unit", "U1")]
    assert (ar.status, u1.label, u1.suggestion, u1.archive) == (va.ACCEPTED, va.ARCHIVE_LEGACY, None, LEGACY)


def test_group_member_translated_at_v77(tmp_path, schema_db_url) -> None:
    """A pre-v77 group value ``not_allowed`` (in ``invest_methods``) that v77 carried by name as
    ``cumulative_limits`` into ``group_invest_methods``: the archive shows it as ``not_active`` (the v77 rename),
    information only.  The rename table is pinned to the v77 list."""
    from flextool.update_flextool.db_migration import _V77_GROUP_INVEST_METHODS

    case, archive = _hit(tmp_path, schema_db_url, pre_v77_group=True, before=[("group", "G1", "base", "not_allowed")])
    v77 = add_commit(case.path, "v77: group invest methods", LATER)
    group_list = sql(case.path, "SELECT id FROM parameter_value_list WHERE name = 'group_invest_methods'")[0][0]
    sql(case.path, "UPDATE parameter_definition SET parameter_value_list_id = ?, default_value = ?, commit_id = ? "
                   "WHERE name = 'invest_method' AND entity_class_id = (SELECT id FROM entity_class WHERE name = "
                   "'group')", (group_list, str(member_id(case.path, "group_invest_methods", "not_active")).encode(),
                                v77))
    sql(case.path, "UPDATE parameter_value SET value = ?, commit_id = ? WHERE id = ?",
        (str(member_id(case.path, "group_invest_methods", "cumulative_limits")).encode(), v77,
         _pv(case.path, "G1", "group")))
    _, _, found = _evidence(case, archive, tmp_path)
    g1 = found[("carried", "group", "G1")]
    assert (g1.label, g1.suggestion, g1.archive, g1.archive_note) == (
        vs.CARRIED_LABEL, None, "not_active", "was not_active before the overwrite (not_allowed then)")
    assert va.translate("not_allowed", "group", 76) == "not_allowed"
    assert va.translate("not_allowed", "unit", 82) == "not_allowed"
    assert set(va.GROUP_RENAMES.values()) <= set(_V77_GROUP_INVEST_METHODS)
    assert not set(va.GROUP_RENAMES) & set(_V77_GROUP_INVEST_METHODS)
    assert set(_V77_GROUP_INVEST_METHODS) - set(va.GROUP_RENAMES.values()) == (
        set(vlh.PRIOR_MEMBERS[22]) - set(va.GROUP_RENAMES)) | {vlh.APPENDED[22]}


_PRE_ENTITY = """
CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL);
CREATE TABLE "commit" (id INTEGER PRIMARY KEY, comment VARCHAR(255) NOT NULL, date DATETIME NOT NULL, user VARCHAR(45));
CREATE TABLE entity_class_type (id INTEGER PRIMARY KEY, name VARCHAR(255));
CREATE TABLE entity_type (id INTEGER PRIMARY KEY, name VARCHAR(255));
CREATE TABLE entity_class (id INTEGER PRIMARY KEY, type_id INTEGER, name VARCHAR(255), commit_id INTEGER);
CREATE TABLE object_class (entity_class_id INTEGER PRIMARY KEY, type_id INTEGER);
CREATE TABLE entity (id INTEGER PRIMARY KEY, type_id INTEGER, class_id INTEGER, name VARCHAR(255), commit_id INTEGER);
CREATE TABLE object (entity_id INTEGER PRIMARY KEY, type_id INTEGER);
CREATE TABLE alternative (id INTEGER PRIMARY KEY, name VARCHAR(255), commit_id INTEGER);
CREATE TABLE parameter_value_list (id INTEGER PRIMARY KEY, name VARCHAR(255), commit_id INTEGER);
CREATE TABLE list_value (id INTEGER PRIMARY KEY, parameter_value_list_id INTEGER, "index" INTEGER, type VARCHAR(255),
                         value BLOB, commit_id INTEGER);
CREATE TABLE parameter_definition (id INTEGER PRIMARY KEY, entity_class_id INTEGER, object_class_id INTEGER,
                                   name VARCHAR(155), default_type VARCHAR(255), default_value BLOB,
                                   parameter_value_list_id INTEGER, commit_id INTEGER);
CREATE TABLE parameter_value (id INTEGER PRIMARY KEY, parameter_definition_id INTEGER, entity_id INTEGER,
                              entity_class_id INTEGER, object_id INTEGER, type VARCHAR(255), value BLOB,
                              alternative_id INTEGER, commit_id INTEGER);
INSERT INTO alembic_version VALUES ('989fccf80441');
INSERT INTO "commit" VALUES (1, 'Create the database', '2023-03-01 10:00:00', 'spinedb_api'),
    (2, 'Updated Flextool data structure to version 21', '2024-07-17 12:00:00', 'spinedb_api');
INSERT INTO entity_class_type VALUES (1, 'object');
INSERT INTO entity_type VALUES (1, 'object');
INSERT INTO entity_class VALUES (1, 1, 'model', 1), (2, 1, 'unit', 1);
INSERT INTO object_class VALUES (1, 1), (2, 1);
INSERT INTO entity VALUES (1, 1, 2, 'U1', 1), (2, 1, 2, 'U5', 1), (3, 1, 2, 'U6', 1);
INSERT INTO object VALUES (1, 1), (2, 1), (3, 1);
INSERT INTO alternative VALUES (1, 'base', 1);
INSERT INTO parameter_value_list VALUES (1, 'invest_methods', 1);
INSERT INTO list_value VALUES (10, 1, 0, 'str', '"not_allowed"', 1), (11, 1, 1, 'str', '"invest_total"', 1);
INSERT INTO parameter_definition VALUES (1, 1, 1, 'version', 'float', '21.0', NULL, 2),
    (2, 2, 2, 'invest_method', 'list_value_ref', '11', 1, 1);
INSERT INTO parameter_value VALUES (1, 2, 1, 2, 1, 'list_value_ref', '10', 1, 1),
    (2, 2, 2, 2, 2, NULL, '"not_allowed"', 1, 1), (3, 2, 3, 2, 3, NULL, '"invest_total"', 1, 1);
"""


def test_pre_entity_layout_archive(tmp_path, schema_db_url) -> None:
    """A SpineDB file from before the entity refactor (``object_class`` / ``object`` tables, made by raw SQL): read
    by name (not the same lineage).  U1 is a list reference, U5 and U6 are plain values from before list
    references (read with ``from_database``).  U6's ``invest_total`` contradicts the position proof: conflict.
    The archive default ``invest_total`` is information."""
    case = make_hit(tmp_path, schema_db_url, before=[("unit", e, "base", "not_allowed") for e in ("U1", "U5", "U6")])
    archive = tmp_path / "pre_entity.sqlite"
    with contextlib.closing(sqlite3.connect(archive)) as con:
        con.executescript(_PRE_ENTITY)
    _, ar, found = _evidence(case, archive, tmp_path)
    assert (ar.status, ar.same_lineage, ar.version) == (va.ACCEPTED, False, 21)
    got = {e: (s.label, s.suggestion, s.archive) for (k, _, e), s in found.items() if k == vs.VALUE}
    assert got == {"U1": (vs.CERTAIN, "set:not_allowed", "not_allowed"),
                   "U5": (vs.CERTAIN, "set:not_allowed", "not_allowed"),
                   "U6": (va.CONFLICT, None, "invest_total")}
    assert found[("default", "unit", "")].archive_note == "your old copy had invest_total"
    archive.unlink()
