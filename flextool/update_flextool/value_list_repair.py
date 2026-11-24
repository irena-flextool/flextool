"""The value-list repair tool (``flextool-repair-value-list``, W-0490): a private snapshot and its analysis.

:func:`snapshot` copies the database with the ``sqlite3`` backup API (the source is opened read-only and never
written).  :func:`analyse` works on that copy only: it runs the item-279 repair of text-form list defaults on it
(:func:`~flextool.update_flextool.list_default_repair.repair_list_defaults`, a write), then reads, with ``sqlite3``
only, what the ``invest_methods`` list of the W-0490 session shows:

- R, the row at the lowest index, and C, the session's ``"Added new parameter value lists"`` commit followed by the
  step's own commit (``value_list_history.STEP_COMMIT``), both inside the session;
- the lost member: the one shipped member (``PRIOR_MEMBERS``) missing from the list, or a hand-restored one (a
  shipped member added after the session at an index above every sibling); none = a legacy member;
- the position proof: the siblings sit at their template positions, share one commit older than the session's
  start version commit (or NULL), and the sibling at index k has id ``R.id + k``; the lost member's template
  position is R's index.  It proves that R was created as template position 0 in the same import as its siblings;
- the state (``STATES``).  A NULL ``commit_id`` counts as older than any commit (it predates commit tracking).

Residual: a user who renamed ``not_allowed`` at index 0 before the session cannot be told apart from the shipped
row, so the proof still holds for such a file.

TODO(v5.0.0): remove with W-0490 (see ``value_list_history``).
"""
from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from flextool.update_flextool import value_list_history as vlh
from flextool.update_flextool.list_default_repair import ListDefaultRepair, repair_list_defaults

HIT = "hit"  # R reads the appended member, its commit is C (anchored), a sibling is older than C
HIT_UNANCHORED = "hit-unanchored"  # as hit with a known lost member, but R's commit is not an anchored C
RENAMED_BACK = "renamed-back"  # R reads the lost member again, written after the session; position proof holds
NOT_HIT = "not-hit"  # the list is complete and R never read the appended member
UNCLEAR = "unclear"  # anything else: show rows and commits, ask nothing, write nothing
NO_HISTORY = "no-history"  # no version commits: W-0490 does not apply
NO_FINDING = "no-finding"  # version commits, but no W-0490 session
STATES = (HIT, HIT_UNANCHORED, RENAMED_BACK, NOT_HIT, UNCLEAR, NO_HISTORY, NO_FINDING)
SEVERAL = "several W-0490 sessions"  # the reason of an unclear file with more than one session

_BUSY_TIMEOUT_S = 5.0


class RepairError(Exception):
    """A refusal or a failure; the message says why.  Nothing was written to the source."""


def schema_outdated_message(name: str = "this database") -> str:
    """The refusal for a database whose SpineDB schema is older than its data version expects."""
    return (f"The Spine database schema of {name} is older than its data version expects: run "
            "flextool-migrate-database on it (or open it in FlexTool), then run this tool again.")


@dataclass(frozen=True)
class Member:
    """A ``list_value`` row; *value* is the member text (``None`` when it is not a JSON string)."""

    id: int
    index: int
    value: str | None
    commit: int | None


@dataclass(frozen=True)
class Analysis:
    """What the snapshot shows.  *r* is the list's row at the lowest index, *c* the overwrite commit (C), *h* R's
    commit in ``renamed-back``; *lost* the lost member (``None``: legacy or unknown), *lost_row* the id of a
    hand-restored row; *restore* the members the write must add for the list to be complete."""

    path: Path
    version: int
    pre_pass: ListDefaultRepair
    state: str
    finding: vlh.Finding | None = None
    list_id: int | None = None
    members: tuple[Member, ...] = ()
    r: Member | None = None
    c: int | None = None
    h: int | None = None
    lost: str | None = None
    lost_row: int | None = None
    restore: tuple[str, ...] = ()
    position_proof: bool = False
    reason: str = ""


def _remove(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def snapshot(source: str | Path, directory: str | Path, name: str | None = None) -> Path:
    """Copy *source* to ``<directory>/<name or source name>.partial-<pid>`` with the ``sqlite3`` backup API (a
    consistent copy even while Spine Toolbox has the file open) and return the copy's path."""
    from flextool.update_flextool.frozen import ro_uri

    source = Path(source)
    if not source.is_file():
        raise RepairError(f"{source} does not exist.")
    target = Path(directory) / f"{name or source.name}.partial-{os.getpid()}"
    if target.exists():
        raise RepairError(f"{target} exists already; remove it and try again.")
    try:
        with contextlib.closing(sqlite3.connect(ro_uri(source), uri=True, timeout=_BUSY_TIMEOUT_S)) as src, \
                contextlib.closing(sqlite3.connect(target)) as dst:
            # Hold a read lock for the whole copy: the busy timeout applies here, while the backup itself would
            # retry a locked source forever.
            src.execute("BEGIN")
            src.execute("SELECT count(*) FROM sqlite_master").fetchone()
            src.backup(dst)
            src.execute("ROLLBACK")
            dst.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.Error as exc:
        _remove(target)
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise RepairError(f"{source.name} is in use: close Spine Toolbox / the FlexTool GUI and try again.") \
                from exc
        raise RepairError(f"{source} could not be copied: {exc}") from exc
    except BaseException:
        _remove(target)
        raise
    return target


def analyse(snapshot_path: str | Path) -> Analysis:
    """Run the item-279 pre-pass on the snapshot (it writes: call this on a :func:`snapshot` only, never on the
    source) and analyse its W-0490 session."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.frozen import _connect_ro, read_db_version_ro

    path = Path(snapshot_path)
    version = read_db_version_ro(path)
    if version is None:
        raise RepairError(f"{path} is not a FlexTool database.")
    if version > FLEXTOOL_DB_VERSION:
        raise RepairError(f"The database is at data version {version}, newer than this FlexTool "
                          f"({FLEXTOOL_DB_VERSION}): update FlexTool.")
    base = {"path": path, "version": version, "pre_pass": repair_list_defaults(path)}
    try:
        versions, _ = vlh._version_commits(path)
    except sqlite3.Error as exc:
        raise RepairError(f"The history of {path} could not be read: {exc}") from exc
    findings = vlh.session_findings_ro(path)
    if not versions:
        return Analysis(state=NO_HISTORY, **base)
    if not findings:
        return Analysis(state=NO_FINDING, **base)
    for f in findings:
        if f.step not in vlh.PRIOR_MEMBERS:
            raise RepairError(f"W-0490 names migration step {f.step}; this tool has data only for step "
                              f"{', '.join(map(str, vlh.PRIOR_MEMBERS))}. Please report this.")
    if len(findings) > 1:
        return Analysis(state=UNCLEAR, finding=findings[0], reason=SEVERAL, **base)
    con = _connect_ro(path)
    try:
        return _detect(con, findings[0], base)
    except sqlite3.Error as exc:
        raise RepairError(f"{path} could not be read: {exc}") from exc
    finally:
        con.close()


def _text(value) -> str | None:
    try:
        decoded = json.loads(value.encode("utf-8") if isinstance(value, str) else bytes(value))
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, str) else None


def _older(commit: int | None, ref: int | None) -> bool:
    """Whether *commit* is older than *ref* (a NULL commit is older than any)."""
    return commit is None or (ref is not None and commit < ref)


def _position_proof(r: Member, siblings: list[Member], prior: tuple[str, ...], lost: str, start: int) -> bool:
    """R is template position ``prior.index(lost)``, created in the same import as its *siblings* (the shipped
    members other than R and a hand-restored row): template positions, one shared old commit, consecutive ids."""
    if lost not in prior or r.index != prior.index(lost):
        return False
    if sorted(s.value for s in siblings) != sorted(m for m in prior if m != lost):
        return False
    if len({s.commit for s in siblings}) != 1 or not _older(siblings[0].commit, start):
        return False
    return all(s.index == prior.index(s.value) and s.id - r.id == s.index - r.index for s in siblings)


def _detect(con: sqlite3.Connection, f: vlh.Finding, base: dict) -> Analysis:
    prior, appended, (name,) = vlh.PRIOR_MEMBERS[f.step], vlh.APPENDED[f.step], vlh.APPEND[f.step]
    row = con.execute("SELECT id FROM parameter_value_list WHERE name = ?", (name,)).fetchone()
    members = () if row is None else tuple(Member(i, idx, _text(v), c) for i, idx, v, c in con.execute(
        'SELECT id, "index", value, commit_id FROM list_value WHERE parameter_value_list_id = ? ORDER BY "index", id',
        row))
    base = {**base, "finding": f, "list_id": row and row[0], "members": members}
    if not members:
        return Analysis(state=UNCLEAR, reason=f"no '{name}' list", **base)
    commits = con.execute('SELECT id, comment FROM "commit" WHERE id > ? AND id < ? ORDER BY id',
                          (f.start_commit, f.end_commit)).fetchall()
    anchors = [a for (a, first), (_, then) in zip(commits, commits[1:])
               if first == vlh._LIST_STEP and then == vlh.STEP_COMMIT[f.step]]
    r, others = members[0], members[1:]
    c = r.commit if r.commit in anchors else (anchors[0] if anchors else None)
    base = {**base, "r": r, "c": c}
    values = [m.value for m in members]
    if r.value == appended:
        missing = tuple(m for m in prior if m not in values)
        restored = [m for m in others if m.value in prior and m.commit is not None and m.commit > f.end_commit
                    and all(m.index > o.index for o in others if o is not m)]
        lost, lost_row = (missing[0], None) if len(missing) == 1 else (None, None)
        if not missing and len(restored) == 1:
            lost, lost_row = restored[0].value, restored[0].id
        siblings = [m for m in others if m.value in prior and m.id != lost_row]
        proof = lost is not None and _position_proof(r, siblings, prior, lost, f.start_commit)
        found = {"lost": lost, "lost_row": lost_row, "restore": missing, "position_proof": proof}
        has_older = any(_older(m.commit, c if c is not None else f.end_commit) for m in others)
        if r.commit in anchors and has_older:
            return Analysis(state=HIT, **found, **base)
        if lost is not None and has_older:
            return Analysis(state=HIT_UNANCHORED, **found, **base)
        return Analysis(state=UNCLEAR, reason=f"R reads '{appended}', but neither its commit nor a lost member "
                        "shows the overwrite", **found, **base)
    complete = set(prior) <= set(values)
    if complete and appended in values and (
            _older(r.commit, f.start_commit) or any(m.value == appended and m.commit in anchors for m in others)):
        return Analysis(state=NOT_HIT, **base)
    if complete and r.value in prior and r.commit is not None and r.commit > f.end_commit and anchors:
        siblings = [m for m in others if m.value in prior]
        if _position_proof(r, siblings, prior, r.value, f.start_commit):
            return Analysis(state=RENAMED_BACK, h=r.commit, lost=r.value, position_proof=True,
                            restore=() if appended in values else (appended,), **{**base, "c": anchors[0]})
    return Analysis(state=UNCLEAR, reason="the list matches no known pattern", **base)


def by_hand_refusal(a: Analysis) -> str:
    """Why ``--checked-by-hand`` cannot record a repair for the analysed snapshot (``""``: it can).  It needs an
    ``unclear`` file with one W-0490 session (a record covers one session) whose list holds every member the record
    checks (the checklist's steps 2-3)."""
    if a.state != UNCLEAR:
        return (f"--checked-by-hand is only for a database this tool cannot read (exit code 4); this one is "
                f"'{a.state}': run the tool without --checked-by-hand.")
    if a.reason == SEVERAL:
        return ("this database has several W-0490 sessions, and a repair record covers one session only, so "
                "--checked-by-hand cannot end the warning: please report the file.")
    f = a.finding
    have = {m.value for m in a.members}
    missing = [m for m in (*vlh.PRIOR_MEMBERS[f.step], vlh.APPENDED[f.step]) if m not in have]
    if missing:
        return (f"the list {vlh.APPEND[f.step][0]} still lacks {', '.join(missing)}: add them (steps 2-3 of the "
                "W-0490 checklist), then run the tool again.")
    return ""


def all_suspects(a: Analysis) -> tuple:
    """Every suspect of the analysed snapshot in one list: part A (W-0490, ``value_list_suspects.suspects``), then
    part B (``list_default_restore.part_b_suspects``: empty method parameter defaults and storage nodes).  No
    definition is in both: a part A default references R, a part B one is empty."""
    from flextool.update_flextool.list_default_restore import part_b_suspects
    from flextool.update_flextool.value_list_suspects import suspects

    return suspects(a) + part_b_suspects(a)


def part_b_line(a: Analysis) -> str:
    """The line to print instead of part B for a snapshot older than the current data version (``""`` otherwise)."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.list_default_restore import MIGRATE_FIRST

    return MIGRATE_FIRST if a.version < FLEXTOOL_DB_VERSION else ""
