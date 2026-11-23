"""The value-list repair tool's write: apply a plan to the private snapshot, verify it, publish it as the copy.

Only the snapshot (:func:`~flextool.update_flextool.value_list_repair.snapshot`, ``<output>.partial-<pid>`` in the
output's folder) is ever written; the source is never opened for writing, and the repair is never done in place.

1. Apply: one spinedb_api session on the snapshot, opened with ``upgrade=True`` (an old SpineDB revision is upgraded
   on the private copy only, before the state the verify compares against is read), and one commit with the
   message ``value_list_history.REPAIR_COMMIT`` in every run.  It restores the members missing from R's list with
   ``import_data`` (guarded: no existing member may change), writes each ``set:`` answer (``to_database(member)``,
   as the v77 step writes a list default, so a default is stored as a ``list_value_ref``), adds ``fix_start`` for a
   storage node, removes the frozen marker (a repaired copy of a frozen database is not frozen; the label goes to
   the ``.txt`` side file), and writes each record that is due (``RECORD_KEY``, ``PART_B_RECORD_KEY``).
2. Verify, read-only with ``sqlite3``: every ``set:`` reads its member as a ``list_value_ref`` with an ASCII-digit
   payload; every other suspect row and every row no suspect names is unchanged; R's list holds every shipped
   member and ``cumulative_limits``; the empty shipped defaults are the expected ones (the storage default stays
   empty while a storage question has no answer); no text-form default is left; the metadata changed only by the
   records and the frozen marker; at most one commit was added; each written record is valid.
3. Publish: the side files ``<output>.value_list_repair.csv`` (the applied answers, replayable with ``--answers``)
   and ``.txt`` (provenance) are written to temporary names, the snapshot is renamed to the output without ever
   replacing a file (a hard link on POSIX, ``os.rename`` on Windows, retried while Windows holds the file), then the
   side files are renamed.  On any exception (``KeyboardInterrupt`` too) the snapshot, its journal files, the
   temporary side files and an output this run created are deleted.

TODO(v5.0.0): remove with W-0490 (see ``value_list_history``), except the part B rules.
"""
from __future__ import annotations

import contextlib
import datetime
import os
import sqlite3
import time
from dataclasses import dataclass, replace
from pathlib import Path

from flextool.update_flextool import list_default_restore as ldr
from flextool.update_flextool import value_list_answers as va
from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool import value_list_repair as vr
from flextool.update_flextool import value_list_suspects as vs

SIDE_CSV, SIDE_TXT = ".value_list_repair.csv", ".value_list_repair.txt"
_PART_A_STATES = (vr.HIT, vr.HIT_UNANCHORED, vr.RENAMED_BACK, vr.NOT_HIT)
_STORAGE_VALUE = ("SELECT pv.value, pv.type FROM parameter_value pv JOIN parameter_definition pd ON pd.id = "
                  "pv.parameter_definition_id WHERE pv.entity_id = ? AND pv.alternative_id = ? AND pd.name = ?")


def default_output(source: str | Path) -> Path:
    """``<stem>_repaired.sqlite`` next to *source*."""
    source = Path(source)
    return source.with_name(f"{source.stem}_repaired.sqlite")


def check_output(source: str | Path, output: str | Path) -> Path:
    """*output*, or :class:`~value_list_repair.RepairError` when it is the source or exists already."""
    output = Path(output)
    if output.resolve() == Path(source).resolve():
        raise vr.RepairError("The repaired copy must be a new file, not the original.")
    if output.exists():
        raise vr.RepairError(f"{output} exists already; give another --output, or move that file away.")
    return output


def discard(snapshot: str | Path) -> None:
    """Delete a snapshot and its journal files (a quit, a dry run)."""
    vr._remove(Path(snapshot))


def check_schema(path: str | Path) -> None:
    """Refuse a SpineDB schema revision this spinedb_api does not know (written by a newer Spine Toolbox)."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    from flextool.update_flextool.frozen import read_schema_revision_ro

    revision = read_schema_revision_ro(path)
    if revision is None:
        return
    config = Config()
    config.set_main_option("script_location", "spinedb_api:alembic")
    try:
        known = ScriptDirectory.from_config(config).get_revision(revision) is not None
    except Exception:
        known = False
    if not known:
        raise vr.RepairError(f"The SpineDB schema of the database (revision {revision}) is newer than this "
                             "FlexTool's Spine library: update FlexTool.")


@dataclass(frozen=True)
class Previous:
    """What earlier tool runs recorded in the analysed file.  *part_a* / *part_b*: the rerun line when a valid
    record covers that part (``""`` otherwise); *kept*: the part A suspects such a record kept; *suspects*: what is
    left to ask (a covered part dropped; a part B default kept in an earlier repair pre-selected ``keep``)."""

    part_a: str
    part_b: str
    kept: tuple
    suspects: tuple


def previous_repair(a: vr.Analysis, suspects) -> Previous:
    """The records of earlier runs in the analysed snapshot (a rerun on a repaired copy asks nothing that a valid
    record covers).  A part B record is valid only while its ``empty`` hash equals the current empty set's."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.frozen import _connect_ro

    part_a = ""
    if a.finding is not None and vlh.record_ends(a.finding, a.path):
        rec = vlh.read_record(a.path)
        part_a = f"repaired on {rec.date} ({rec.value['set']} set, {rec.value['kept']} kept)"
    with contextlib.closing(_connect_ro(a.path)) as con:
        rec_b = ldr.parse_part_b_record(con)
        same = rec_b is not None and rec_b.empty == ldr.empty_hash(ldr.empty_shipped_defaults(con))
    part_b = (f"defaults restored on {rec_b.date} ({rec_b.set} set, {rec_b.kept} kept)"
              if same and a.version == FLEXTOOL_DB_VERSION else "")
    left, kept = [], []
    for s in suspects:
        if part_a and not s.part_b:
            kept.append(s)
        elif not (part_b and s.part_b):
            if rec_b is not None and s.part_b and s.kind == vs.DEFAULT and (
                    s.commit is None or s.commit < rec_b.commit):
                s = replace(s, suggestion=va.KEEP, reason=f"kept in the repair of {rec_b.date}")
            left.append(s)
    return Previous(part_a, part_b, tuple(kept), tuple(left))


@dataclass(frozen=True)
class Written:
    """A published copy: its path, the side files, the summary lines (every write, the records, what was left)."""

    output: Path
    answers: Path
    report: Path
    lines: tuple[str, ...]
    part_a_record: bool
    part_b_record: bool


def _raw(payload):
    return payload.encode("utf-8") if isinstance(payload, str) else (None if payload is None else bytes(payload))


def _target(con: sqlite3.Connection, item: str):
    """``(payload bytes, type)`` of the row suspect *item* names, ``None`` when there is no row."""
    kind, _, rest = item.partition(":")
    if kind == "pv":
        row = con.execute("SELECT value, type FROM parameter_value WHERE id = ?", (int(rest),)).fetchone()
    elif kind == "def":
        row = con.execute("SELECT default_value, default_type FROM parameter_definition WHERE id = ?",
                          (int(rest),)).fetchone()
    else:
        entity, alternative = map(int, rest.split(":"))
        row = con.execute(_STORAGE_VALUE, (entity, alternative, ldr.STORAGE[1])).fetchone()
    return None if row is None else (_raw(row[0]), row[1])


@dataclass(frozen=True)
class _State:
    targets: dict
    values: dict
    defaults: dict
    members: dict  # list_value id -> (list id, value bytes)
    metadata: tuple
    commits: int
    empty: tuple


def _state(path: Path, items) -> _State:
    from flextool.update_flextool.frozen import _connect_ro

    with contextlib.closing(_connect_ro(path)) as con:
        return _State(
            {i: _target(con, i) for i in items},
            {i: (_raw(v), t) for i, v, t in con.execute("SELECT id, value, type FROM parameter_value")},
            {i: (_raw(v), t) for i, v, t in con.execute(
                "SELECT id, default_value, default_type FROM parameter_definition")},
            {i: (lid, _raw(v)) for i, lid, v in con.execute(
                "SELECT id, parameter_value_list_id, value FROM list_value")},
            tuple(sorted(n for (n,) in con.execute("SELECT name FROM metadata"))),
            con.execute('SELECT count(*) FROM "commit"').fetchone()[0],
            tuple(ldr.empty_shipped_defaults(con)))


def _restore(db, a: vr.Analysis, members) -> str:
    """Add *members* to R's list with ``import_data``; no existing member may change (the 0.31–0.34 bug)."""
    from spinedb_api import from_database, import_data

    from flextool.update_flextool.db_migration import _list_members

    name = db.get_item("parameter_value_list", id=a.list_id)["name"]
    before = _list_members(db, [name])
    _, errors = import_data(db, parameter_value_lists=[(name, m) for m in members])
    after = _list_members(db, [name])
    now = {from_database(lv["value"], lv["type"]) for lv in db.get_items("list_value", parameter_value_list_name=name)}
    if errors or any(after.get(i) != v for i, v in before.items()) or not set(members) <= now:
        raise vr.RepairError(f"Adding {', '.join(members)} to the list {name} did not work as expected "
                             f"({'; '.join(map(str, errors)) or 'an existing member changed'}); nothing was written.")
    return f"restored {', '.join(members)} in the list {name}"


def _apply(a: vr.Analysis, plan: va.Plan, before: _State, previous: Previous | None, today: str,
           frozen) -> tuple[dict, list[str]]:
    from spinedb_api import to_database

    from flextool.sqlite_url import sqlite_url_string
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.frozen import FROZEN_KEY, _replace_rows

    lines: list[str] = []
    records: dict = {}
    if plan.part_a_record and a.finding is not None and not (previous and previous.part_a):
        counts = plan.counts(False)
        records[vlh.RECORD_KEY] = vlh.record_value(a.finding, counts["set"], counts["kept"])
    filled = {f"{d.suspect.cls}.{d.suspect.parameter}" for d in plan.sets if d.suspect.kind == vs.DEFAULT}
    empty = tuple(n for n in before.empty if n not in filled)
    if plan.part_b_record and not (previous and previous.part_b and empty == before.empty):
        counts = plan.counts(True)
        records[ldr.PART_B_RECORD_KEY] = ldr.part_b_record_value(today, counts["set"], counts["kept"],
                                                                  counts["storage"], ldr.empty_hash(empty))
    with open_database_mapping(sqlite_url_string(a.path.resolve().as_posix()), create=False, upgrade=True) as db:
        if plan.restore:
            lines.append(_restore(db, a, plan.restore))
        for d in plan.sets:
            s, (value, typ) = d.suspect, to_database(d.answer.member)
            kind, _, rest = s.item.partition(":")
            if kind == "pv":
                db.update_parameter_value(id=db.get_item("parameter_value", id=int(rest))["id"], value=value,
                                          type=typ)
            elif kind == "def":
                db.update_parameter_definition(id=db.get_item("parameter_definition", id=int(rest))["id"],
                                               default_value=value, default_type=typ)
            else:
                db.add_parameter_value(entity_class_name=s.cls, entity_byname=(s.entity,), parameter_definition_name=
                                       s.parameter, alternative_name=s.alternative, value=value, type=typ)
            where = f" {s.entity} [{s.alternative}]" if s.entity else " (class default)"
            lines.append(f"{s.cls}.{s.parameter}{where}: {s.current or 'empty'} -> {d.answer.member}")
        if frozen is not None:
            _replace_rows(db, FROZEN_KEY, None)
            lines.append(f"the copy is not frozen (the original is frozen as '{frozen.label}')")
        for key, value in records.items():
            _replace_rows(db, key, value)
            lines.append(f"record {key} = {value}")
        committed = bool(lines)
        if committed:
            db.commit_session(vlh.REPAIR_COMMIT)
    if plan.held:
        lines.append(va.STORAGE_HELD)
    return {"records": records, "empty": empty, "commit": committed}, lines


def _verify(a: vr.Analysis, plan: va.Plan, before: _State, done: dict) -> None:
    """Raise :class:`~value_list_repair.RepairError` unless the written snapshot holds exactly the plan (module
    docstring, step 2)."""
    from flextool.update_flextool.frozen import FROZEN_KEY, _connect_ro
    from flextool.update_flextool.list_default_repair import list_defaults_repairable_ro

    applied = {d.suspect.item: d for d in plan.sets}
    after = _state(a.path, before.targets)
    bad = []
    for item, now in after.targets.items():
        if item not in applied:
            if now != before.targets[item]:
                bad.append(f"{item} changed although it was not set")
            continue
        d = applied[item]
        member = after.members.get(int(now[0])) if now and now[1] == "list_value_ref" and now[0].isdigit() else None
        if member is None or member[0] != d.suspect.list_id or vr._text(member[1]) != d.answer.member:
            bad.append(f"{item} does not read {d.answer.member}")
    ids = {k: {int(i.partition(":")[2]) for i in applied if i.startswith(k + ":")} for k in ("pv", "def")}
    if any(after.values.get(i) != v for i, v in before.values.items() if i not in ids["pv"]) or \
            len(after.values) - len(before.values) != sum(i.startswith("sn:") for i in applied):
        bad.append("a parameter value no answer names changed")
    if {i for i in before.defaults.keys() | after.defaults.keys()
            if before.defaults.get(i) != after.defaults.get(i)} - ids["def"]:
        bad.append("a definition no answer names changed")
    new = [m for i, m in after.members.items() if i not in before.members]
    if any(after.members.get(i) != m for i, m in before.members.items()) or sorted(
            vr._text(v) for lid, v in new if lid == a.list_id) != sorted(plan.restore) or len(new) != len(plan.restore):
        bad.append("the value lists changed otherwise than by the restored members")
    if a.finding is not None and a.state in _PART_A_STATES:
        listed = {vr._text(v) for lid, v in after.members.values() if lid == a.list_id}
        if not {*vlh.PRIOR_MEMBERS[a.finding.step], vlh.APPENDED[a.finding.step]} <= listed:
            bad.append("the list is not complete")
    if after.empty != done["empty"]:
        bad.append(f"the empty shipped defaults are {list(after.empty)}, expected {list(done['empty'])}")
    keys = set(done["records"])
    expected = sorted([n for n in before.metadata if n not in keys and n != FROZEN_KEY] + list(keys))
    if list(after.metadata) != expected:
        bad.append(f"the metadata are {list(after.metadata)}, expected {expected}")
    if after.commits - before.commits != int(done["commit"]):
        bad.append(f"{after.commits - before.commits} commits were added, expected {int(done['commit'])}")
    if list_defaults_repairable_ro(a.path):
        bad.append("a list default is stored as text")
    if vlh.RECORD_KEY in keys and not vlh.record_ends(a.finding, a.path):
        bad.append("the W-0490 record does not end the finding")
    if ldr.PART_B_RECORD_KEY in keys:
        with contextlib.closing(_connect_ro(a.path)) as con:
            rec = ldr.parse_part_b_record(con)
        if rec is None or rec.empty != ldr.empty_hash(after.empty):
            bad.append("the part B record is not valid")
    if bad:
        raise vr.RepairError("The repaired copy failed its check, so it was not written: " + "; ".join(bad) + ".")


def _move(src: Path, dst: Path) -> None:
    """Rename *src* to *dst*, never replacing a file at *dst*."""
    appeared = vr.RepairError(f"{dst} appeared while the copy was being made; it was not overwritten.")
    if os.name != "nt":
        try:
            os.link(src, dst)
        except FileExistsError:
            raise appeared from None
        except OSError:  # a file system without hard links
            if dst.exists():
                raise appeared from None
            os.replace(src, dst)
            return
        try:
            os.unlink(src)
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        return
    for attempt in range(20):
        try:
            os.rename(src, dst)
            return
        except FileExistsError:
            raise appeared from None
        except PermissionError:  # a virus scanner or the indexer still holds the file
            if attempt == 19:
                raise
            time.sleep(0.25)


def _report(a: vr.Analysis, plan: va.Plan, source: Path, sha: str, frozen, archive, today: str,
            lines: list[str]) -> str:
    out = [f"FlexTool value list repair, {today}", f"source: {source}", f"snapshot sha256: {sha}"]
    if frozen is not None:
        out.append(f"frozen: {frozen.label} (version {frozen.version}, {frozen.date}); the copy is not frozen")
    if archive is not None:
        out.append(f"archive: {archive.check}")
    out.append(f"state: {a.state}" + (f"; R = list_value {a.r.id} ({a.r.value})" if a.r else "")
               + (f"; C = commit {a.c}" if a.c is not None else "") + (f"; H = commit {a.h}" if a.h else ""))
    out.append(f"list-default pre-pass: {a.pre_pass.repaired} repaired, {len(a.pre_pass.unmatched)} left")
    out += [f"write: {line}" for line in lines] or ["write: nothing"]
    out += [f"not answered: {d.suspect.item}" for d in plan.decisions if d.answer is None]
    out += [f"stale: {item}: {why}" for item, why in plan.stale] + [f"warning: {w}" for w in plan.warnings]
    return "\n".join(out) + "\n"


def write_copy(a: vr.Analysis, plan: va.Plan, output: str | Path, *, source: str | Path, archive=None,
               previous: Previous | None = None, today: str | None = None) -> Written:
    """Apply *plan* to the analysed snapshot ``a.path``, verify it and publish it as *output* with its side files
    (module docstring).  On any failure nothing is left: the snapshot is deleted too."""
    from flextool.update_flextool import frozen as fz

    output, partial, today = Path(output), Path(a.path), today or datetime.date.today().isoformat()
    finals = [output.with_name(output.name + SIDE_CSV), output.with_name(output.name + SIDE_TXT)]
    temps = [f.with_name(f"{f.name}.partial-{os.getpid()}") for f in finals]
    published = False
    try:
        check_schema(partial)
        if fz.schema_outdated_ro(partial):  # upgrade first: the check compares against the upgraded rows
            from flextool.update_flextool.db_handle import open_database_mapping

            from flextool.sqlite_url import sqlite_url_string

            with open_database_mapping(sqlite_url_string(partial.resolve().as_posix()), create=False, upgrade=True):
                pass
        before = _state(partial, [d.suspect.item for d in plan.decisions])
        sha, frozen = fz.sha256_of(partial), fz.read_frozen(partial)
        done, lines = _apply(a, plan, before, previous, today, frozen)
        _verify(a, plan, before, done)
        va.write_answers(temps[0], [d.suspect for d in plan.decisions], plan.key, plan.answers)
        temps[1].write_text(_report(a, plan, Path(source), sha, frozen, archive, today, lines), encoding="utf-8")
        _move(partial, output)
        published = True
        for temp, final in zip(temps, finals):
            os.replace(temp, final)
    except BaseException:
        discard(partial)
        for path in temps + (finals + [output] if published else []):
            path.unlink(missing_ok=True)
        raise
    return Written(output, finals[0], finals[1], tuple(lines), vlh.RECORD_KEY in done["records"],
                   ldr.PART_B_RECORD_KEY in done["records"])
