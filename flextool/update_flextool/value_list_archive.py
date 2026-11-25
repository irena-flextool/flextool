"""The value-list repair tool's archive mode (``--archive OLD.sqlite``, W-0490): an older copy as evidence.

The archive is read with ``sqlite3`` only (``frozen.ro_uri``, read-only, closed in ``finally``), never with a
``DatabaseMapping``: an old SpineDB schema would want an upgrade, which writes.  Both SpineDB layouts are read: the
current one and the one before the entity refactor (spinedb_api < 0.31, with ``object_class`` / ``object`` tables
next to ``entity_class`` / ``entity``, which already hold the names; ``invest_method`` is only on 1-D classes).  A
``list_value_ref`` is resolved against the archive's own ``list_value``; any other value via ``from_database`` (a
value stored before list references).  A layout without these tables is refused (fail closed).

:func:`read_archive` accepts an archive only if its data version is below 22, its last version commit is older than
the overwrite commit, and its ``invest_methods`` list holds the lost member and not ``cumulative_limits`` (the
member the overwrite wrote).  A file without version commits is accepted with a warning: its date cannot be
checked, so it is never "certain (archive)".

**Same lineage** = the archive's ``commit`` rows equal the database's for every id up to the archive's largest.
Only then can a value be ``certain (archive)``, and only for the **same row** (critique R4): the same
``parameter_value.id`` with an equal ``commit_id`` and equal entity, alternative and definition ids (a renamed
entity whose old name another entity took gets no certainty from the name).  Otherwise the archive is evidence
only, matched by (class, entity name, alternative): "was X before the overwrite", never pre-selected.

:func:`apply_archive` (precedence, plan §1):

- same row, the archive's member is a member of the suspect's own list: ``certain (archive)``, ``set:<member>``;
  if the value was ``certain`` by position and the archive names another member: ``conflict``, no suggestion;
- same row, a member FlexTool no longer has: ``archive: legacy member`` (``conflict`` if certain by position);
- matched by name only, or the same row changed since: information; a ``certain`` value whose archive member
  differs becomes ``conflict``;
- a method parameter default: FlexTool's default stays the recommendation (R13); the archive default is information
  ("your old copy had X");
- a carried v77 group value: information only.

Members are translated forward (step 22 only): a group value ``not_allowed`` is ``not_active`` at data version 77
and later (``db_migration._v77_group_invest_methods``); ``invest_methods`` members are otherwise unchanged.

TODO(v5.0.0): remove with W-0490 (see ``value_list_history``).
"""
from __future__ import annotations

import contextlib
import dataclasses
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs

ACCEPTED = "accepted"
NO_HISTORY = "accepted without history"  # evidence only, never certain (archive)
REFUSED = "refused"
UNUSED = "unused"  # the database shows no overwrite: nothing to compare
CONFLICT = "conflict"
ARCHIVE_LEGACY = "archive: legacy member"
NO_HISTORY_WARNING = ("the archive has no upgrade history, so its date cannot be checked: it is shown as evidence "
                      "only")
_CLASSES = ("unit", "connection", "node", "group")
#: The group member v77 renamed (``db_migration._v77_group_invest_methods``).
GROUP_RENAMES = {"not_allowed": "not_active"}


@dataclass(frozen=True)
class ArchiveValue:
    """An archive ``invest_method`` value; *member* is its text (``None`` when it resolves to no text)."""

    id: int
    commit: int | None
    entity_id: int
    alternative_id: int
    definition_id: int
    cls: str
    entity: str
    alternative: str
    member: str | None


@dataclass(frozen=True)
class Archive:
    """The check result and, when accepted, what the archive holds.  *reason* is the refusal or the warning."""

    path: Path
    status: str
    reason: str = ""
    version: int | None = None
    same_lineage: bool = False
    values: tuple[ArchiveValue, ...] = ()
    defaults: dict[str, str | None] = field(default_factory=dict)  # class → default member
    entities: frozenset[tuple[str, str]] = frozenset()  # (class, name)
    alternatives: frozenset[str] = frozenset()

    @property
    def refused(self) -> bool:
        return self.status == REFUSED

    @property
    def check(self) -> str:
        """One line for the prompt and the ``.txt`` side file."""
        lineage = ", same lineage" if self.same_lineage else ", not the same lineage"
        detail = f": {self.reason}" if self.reason else ""
        return f"{self.path}: {self.status}" + (lineage if self.status == ACCEPTED else "") + detail


def translate(member: str | None, cls: str, version: int) -> str | None:
    """An archive member as the database at data version *version* calls it."""
    if cls == "group" and version >= 77:
        return GROUP_RENAMES.get(member, member)
    return member


def _commits(con: sqlite3.Connection) -> list[tuple]:
    return con.execute('SELECT id, comment, CAST(date AS TEXT), user FROM "commit" ORDER BY id').fetchall()


def _decode(payload, typ, members: dict[int, str | None]) -> str | None:
    if typ == "list_value_ref":
        return members.get(vs._ref(payload))
    if payload is None:
        return None
    try:
        from spinedb_api import from_database

        value = from_database(payload if isinstance(payload, bytes) else str(payload).encode("utf-8"), typ)
    except Exception:
        return None
    return value if isinstance(value, str) else None


def read_archive(path: str | Path, analysis: vr.Analysis) -> Archive:
    """Check the archive *path* against the analysed snapshot and read its ``invest_method`` values and defaults."""
    from flextool.update_flextool.frozen import _connect_ro, read_db_version_ro

    path = Path(path)
    if analysis.state not in (vr.HIT, vr.HIT_UNANCHORED, vr.RENAMED_BACK):
        return Archive(path, UNUSED, "the database shows no value-list overwrite to compare with")

    def refuse(reason: str) -> Archive:
        return Archive(path, REFUSED, reason)

    if not path.is_file():
        return refuse("it does not exist")
    version = read_db_version_ro(path)
    if version is None:
        return refuse("it is not a FlexTool database")
    if version >= 22:
        return refuse(f"it is at data version {version}; an archive must be from before version 22 (the upgrade "
                      "that overwrote the list)")
    with contextlib.closing(_connect_ro(analysis.path)) as con:
        row = con.execute('SELECT CAST(date AS TEXT) FROM "commit" WHERE id = ?',
                          (vs.overwrite_commit(analysis),)).fetchone()
        db_commits = _commits(con)
    try:
        archive_con = _connect_ro(path)
    except sqlite3.Error as exc:
        return refuse(f"it could not be opened: {exc}")
    try:
        commits = _commits(archive_con)
        # Same lineage: the database has every archive commit, unchanged (sibling copies share commit ids).
        lineage = bool(commits) and db_commits[:len(commits)] == commits
        return _read(archive_con, path, analysis, version, commits, lineage, row[0] if row else "")
    except sqlite3.Error as exc:
        return refuse(f"its layout is not one this tool reads ({exc})")
    finally:
        archive_con.close()


def _read(con: sqlite3.Connection, path: Path, a: vr.Analysis, version: int, commits: list[tuple], lineage: bool,
          overwrite_date: str) -> Archive:
    versions = [(cid, str(date)) for cid, comment, date, _ in commits if (comment or "").startswith(vlh._PREFIX)]
    if versions and not versions[-1][1] < overwrite_date:
        return Archive(path, REFUSED, f"its last upgrade ({versions[-1][1][:10]}) is not before the overwrite "
                                      f"({overwrite_date[:10]})")
    members = {i: vr._text(v) for i, v in con.execute("SELECT id, value FROM list_value")}
    row = con.execute("SELECT id FROM parameter_value_list WHERE name = 'invest_methods'").fetchone()
    invest = set() if row is None else {vr._text(v) for (v,) in con.execute(
        "SELECT value FROM list_value WHERE parameter_value_list_id = ?", row)}
    if not invest:
        return Archive(path, REFUSED, "it has no 'invest_methods' list")
    if vlh.APPENDED[22] in invest:
        return Archive(path, REFUSED, f"its 'invest_methods' list already holds '{vlh.APPENDED[22]}' (it is "
                                      "already affected)")
    if a.lost is not None and a.lost not in invest:
        return Archive(path, REFUSED, f"its 'invest_methods' list does not hold '{a.lost}'")
    marks = ",".join("?" * len(_CLASSES))
    values = tuple(ArchiveValue(*row[:8], _decode(row[8], row[9], members)) for row in con.execute(
        "SELECT pv.id, pv.commit_id, pv.entity_id, pv.alternative_id, pv.parameter_definition_id, ec.name, e.name, "
        "al.name, pv.value, pv.type FROM parameter_value pv JOIN parameter_definition pd ON pd.id = "
        "pv.parameter_definition_id JOIN entity_class ec ON ec.id = pd.entity_class_id JOIN entity e ON e.id = "
        "pv.entity_id JOIN alternative al ON al.id = pv.alternative_id WHERE pd.name = 'invest_method' AND "
        f"ec.name IN ({marks})", _CLASSES))
    defaults = {cls: _decode(v, t, members) for cls, v, t in con.execute(
        "SELECT ec.name, pd.default_value, pd.default_type FROM parameter_definition pd JOIN entity_class ec ON "
        f"ec.id = pd.entity_class_id WHERE pd.name = 'invest_method' AND ec.name IN ({marks})", _CLASSES)}
    entities = frozenset(con.execute(
        f"SELECT ec.name, e.name FROM entity e JOIN entity_class ec ON ec.id = e.class_id WHERE ec.name IN ({marks})",
        _CLASSES))
    alternatives = frozenset(n for (n,) in con.execute("SELECT name FROM alternative"))
    status, reason = (ACCEPTED, "") if versions else (NO_HISTORY, NO_HISTORY_WARNING)
    return Archive(path, status, reason, version, lineage and bool(versions), values, defaults, entities,
                   alternatives)


def apply_archive(a: vr.Analysis, suspects, archive: Archive) -> tuple[vs.Suspect, ...]:
    """The suspects with the archive's evidence (plan §1 precedence); unchanged when the archive is not accepted.
    Part B suspects are never matched (an archive before version 22 says nothing about them)."""
    suspects = tuple(suspects)
    if archive.status not in (ACCEPTED, NO_HISTORY):
        return suspects
    from flextool.update_flextool.frozen import _connect_ro

    with contextlib.closing(_connect_ro(a.path)) as con:
        rows = {i: rest for i, *rest in con.execute(
            "SELECT id, commit_id, entity_id, alternative_id, parameter_definition_id FROM parameter_value")}
        own: dict[int, set[str | None]] = {}
        for list_id, value in con.execute("SELECT parameter_value_list_id, value FROM list_value"):
            own.setdefault(list_id, set()).add(vr._text(value))
    own.setdefault(a.list_id, set()).update({*a.restore, a.lost})
    by_id = {v.id: v for v in archive.values} if archive.same_lineage else {}
    by_name = {(v.cls, v.entity, v.alternative): v for v in archive.values}
    return tuple(_one(s, a, archive, rows, own, by_id, by_name) for s in suspects)


def _one(s: vs.Suspect, a: vr.Analysis, archive: Archive, rows, own, by_id, by_name) -> vs.Suspect:
    if s.part_b or s.kind not in (vs.VALUE, vs.DEFAULT, vs.CARRIED):
        return s
    if s.kind == vs.DEFAULT:
        old = translate(archive.defaults.get(s.cls), s.cls, a.version)
        note = f"your old copy had {old}" if old else "your old copy had no default"
        return dataclasses.replace(s, archive=old, archive_note=note)
    pv_id = int(s.item.split(":")[1])
    commit, *ids = rows.get(pv_id, (None, None, None, None))
    same = by_id.get(pv_id) if s.kind == vs.VALUE else None
    if same is not None and [same.entity_id, same.alternative_id, same.definition_id] != ids:
        same = None  # the id is now another entity's value: name evidence only (R4)
    match = same or by_name.get((s.cls, s.entity, s.alternative))
    if match is None:
        if (s.cls, s.entity) not in archive.entities or s.alternative not in archive.alternatives:
            return dataclasses.replace(s, archive_note="not in the archive (added later)")
        default = translate(archive.defaults.get(s.cls), s.cls, a.version) or "none"
        return dataclasses.replace(s, archive_note=f"archive: no value (default {default})")
    member = translate(match.member, s.cls, a.version)
    shown = member or "a value that is not a member"
    current = member is not None and member in own.get(s.list_id, ())
    certain = s.label == vs.CERTAIN
    if same is not None and same.commit == commit:
        note = f"your old copy has this row with the same commit: it was {shown}"
        if certain and member != a.lost:
            return dataclasses.replace(s, label=CONFLICT, suggestion=None, archive=member, archive_note=note,
                                       evidence=f"{s.evidence}; but {note}")
        if current:
            return dataclasses.replace(s, label=vs.CERTAIN_ARCHIVE, suggestion=f"set:{member}", archive=member,
                                       archive_note=note, evidence=f"{note} (same row, same lineage)")
        return dataclasses.replace(s, label=ARCHIVE_LEGACY, suggestion=None, archive=member, archive_note=note,
                                   evidence=f"{s.evidence}; {note}, a member FlexTool no longer has")
    note = f"was {shown}, changed since" if same is not None else f"was {shown} before the overwrite"
    if member != match.member:
        note += f" ({match.member} then)"
    if certain and same is None and member != a.lost:
        return dataclasses.replace(s, label=CONFLICT, suggestion=None, archive=member, archive_note=note,
                                   evidence=f"{s.evidence}; but the archive {note}")
    return dataclasses.replace(s, archive=member, archive_note=note)
