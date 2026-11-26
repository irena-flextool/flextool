"""Whether a database's migration history shows a session that an old spinedb_api bug could have hit (W-0490).

spinedb_api 0.31.0 to 0.34.x (and its master from 2024-04-29 to 2025-06-26) had an import bug: a member added to a
value list that already had members could overwrite the list's first member, when that addition was the first
list-value call of the open session.  The overwritten row kept its id, so every default and value that referred
to the old member silently read the new one.  Migration step 22 (``invest_methods`` + ``cumulative_limits``) did
that to databases that were at exactly data version 21 when a migration started.  The fixed library is 0.35.0
(2025-07-07); FlexTool now requires a fixed one.

A database records no library version, only its migration sessions (the commits ``Updated Flextool data structure
to version N``, with a date).  This module reads them read-only (``sqlite3`` only, ``mode=ro&immutable=1``) and
names each session in the risk window whose first list-value step appends to an existing list.  It never changes
a database and never looks at the values (design: ``specs/two_stage_fix_D_invest_methods_list.md``, rev 3).
Only SQLite files are checked: a database on a server (not a ``sqlite:///`` URL) is never checked.

TODO(v5.0.0): remove this module, W-0490 (``run_warnings/_input_specs.py``), its call sites
(``engine/solve/_orchestration.py``, ``gui/input_sources.py`` ``value_list_findings``, the
``value_list_check`` message in ``gui/messages.py`` and ``gui/message_sources.py``), the repair tool ``flextool-repair-value-list``
(``update_flextool/value_list_repair.py``, ``value_list_suspects.py``, ``value_list_archive.py``,
``value_list_answers.py``, ``value_list_write.py``,
``cli/cmd_repair_value_list.py``) and
their tests.  The warning is temporary.
"""
from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

CODE = "W-0490"
#: The risk window (session end dates, inclusive): the first buggy spinedb_api (master) to the release of 0.35.0.
WINDOW_START = "2024-04-29"
WINDOW_END = "2025-07-07"

#: Migration steps that append a member to a value list that already has members, and those lists.
APPEND: dict[int, tuple[str, ...]] = {
    22: ("invest_methods",),
    27: ("minimum_time_methods",),
    30: ("storage_binding_methods",),
    31: ("storage_binding_methods",),
    34: ("lifetime_methods",),
    38: ("node_type",),
    47: ("transfer_methods", "transfer_methods_group"),
    66: ("energy_margin_methods",),
}
#: Every step that imports or looks up list values.  The first of them in a session is the one the bug could hit;
#: any earlier one loads the list values and protects the appends after it.
TOUCH: frozenset[int] = frozenset(
    [*range(1, 12), 13, 17, 21, 25, 29, 40, 42, 51, 52, 53, 55, 56, 58, 60, 62, 64, 65, 67, 71, 74, 76, 77,
     79, 80, 81, 82, 83, 84, *APPEND])

# The repair tool's data (value_list_repair).  Only step 22 can give a finding (no session in the window ran a
# later APPEND step), so these tables have exactly the key 22 and the tool refuses a finding for any other step.
#: The shipped members of the step's list before the step, in template order (``pre_v26/flextool_template_v2.json``).
PRIOR_MEMBERS: dict[int, tuple[str, ...]] = {22: (
    "not_allowed", "invest_no_limit", "invest_period", "invest_total", "invest_period_total", "retire_no_limit",
    "retire_period", "retire_total", "retire_period_total", "invest_retire_no_limit", "invest_retire_period",
    "invest_retire_total", "invest_retire_period_total")}
#: The member the step appends (the one the bug wrote over the list's first member).
APPENDED: dict[int, str] = {22: "cumulative_limits"}
#: The step's own commit message (``db_migration.common._commit_step``), the commit right after its value-list commit.
STEP_COMMIT: dict[int, str] = {22: "Added cumulative investments"}

_PREFIX = "Updated Flextool data structure to version "

#: The repair tool's record (metadata key) and the comment of its one commit, in every run (part A, part B or both).
RECORD_KEY = "flextool_value_list_repair"
REPAIR_COMMIT = "FlexTool value list repair"


@dataclass(frozen=True)
class Finding:
    step: int
    lists: tuple[str, ...]
    start: int
    end: int
    date: str
    #: Commit ids, not compared: the last id of the start version commit, the first id of the closing one, and the
    #: session's first value-list step commit (0 when none).
    start_commit: int = field(default=0, compare=False)
    end_commit: int = field(default=0, compare=False)
    list_commit: int = field(default=0, compare=False)


#: The commit of a migration step that adds value list members (``db_migration.common.add_value_list_manual``).
_LIST_STEP = "Added new parameter value lists"


def _version_commits(path: Path) -> tuple[list[tuple[int, str, int, int]], list[tuple[int, str]]]:
    """``(versions, list_steps)``: the version commits as ``(version, date, first id, last id)`` (repeats of a version
    collapsed) and the ``(id, date)`` of the value-list step commits."""
    from flextool.update_flextool.frozen import ro_uri

    con = sqlite3.connect(ro_uri(path) + "&immutable=1", uri=True)
    try:
        rows = con.execute('SELECT id, comment, date FROM "commit" WHERE comment LIKE ? OR comment = ? ORDER BY id',
                           (_PREFIX + "%", _LIST_STEP)).fetchall()
    finally:
        con.close()
    out: list[tuple[int, str, int, int]] = []
    steps: list[tuple[int, str]] = []
    for cid, comment, date in rows:
        if comment == _LIST_STEP:
            steps.append((cid, str(date)[:10]))
            continue
        try:
            version = int(float(comment[len(_PREFIX):]))
        except (ValueError, OverflowError):
            continue
        if out and out[-1][0] == version:
            out[-1] = (*out[-1][:3], cid)
        else:
            out.append((version, str(date)[:10], cid, cid))
    return out, steps


def possible_value_list_overwrite_ro(path: str | Path, *, strict: bool = False) -> list[Finding]:
    """The findings of W-0490 and of the GUI's "Please check this database" message: :func:`session_findings_ro`
    without the findings a valid repair record ends (:func:`record_ends`; any error while reading the record keeps
    the finding).  *strict*: a history that cannot be read raises (the GUI then shows a generic message and reads
    again at the next refresh) instead of giving ``[]``."""
    findings = session_findings_ro(path, strict=strict)
    kept = []
    for f in findings:
        try:
            ended = record_ends(f, path)
        except Exception as exc:  # noqa: BLE001  (a record that cannot be read never hides the warning)
            logger.debug("repair record of %s not checked: %s", path, exc)
            ended = False
        if not ended:
            kept.append(f)
    return kept


def session_findings_ro(path: str | Path, *, strict: bool = False) -> list[Finding]:
    """The migration sessions of *path* that may have overwritten a value list member, repaired or not (the repair
    tool reads these); ``[]`` when none, or when the file cannot be read (a check that cannot read the history does
    not warn; with *strict* the read error is raised instead).

    A session is dated by its earliest value-list step commit, else by its closing version commit: a migration that
    failed after such a step (before the all-or-nothing backup) and was retried after the window kept the step's
    overwrite, but its closing commit is from the retry."""
    try:
        commits, list_steps = _version_commits(Path(path))
    except (sqlite3.Error, OSError) as exc:
        logger.debug("value list history of %s not read: %s", path, exc)
        if strict:
            raise
        return []
    findings = []
    # The first version commit has no known start: the database came from a template or a JSON import.
    for (start, _, _, after), (end, end_date, before, _) in zip(commits, commits[1:]):
        inside = [(cid, d) for cid, d in list_steps if after < cid < before]
        date = min([end_date, *(d for _, d in inside)])
        if not WINDOW_START <= date <= WINDOW_END or end <= start:
            continue
        touched = [s for s in TOUCH if start < s <= end]
        first = min(touched, default=None)
        if first in APPEND:
            findings.append(Finding(first, APPEND[first], start, end, date, after, before,
                                    min((cid for cid, _ in inside), default=0)))
    return findings


def warn_at_run_start(db_url: str) -> None:
    """W-0490 for the run's input database, once per run (``run_warnings`` ``per_run``)."""
    from flextool import run_warnings
    from flextool.update_flextool.frozen import sqlite_path_from

    path = sqlite_path_from(db_url)
    if path is None or not path.is_file():
        return
    for f in possible_value_list_overwrite_ro(path):
        details = {name: f"migrated from data version {f.start} to {f.end} on {f.date}" for name in f.lists}
        run_warnings.warn_grouped(CODE, f.lists, details=details, per_run=True, run_log=True, date=f.date)


def _dmy(date: str) -> str:
    """``2024-09-19`` → ``19.9.2024`` (the date format of the GUI)."""
    year, month, day = date.split("-")
    return f"{int(day)}.{int(month)}.{year}"


#: The title of the GUI's W-0490 message (the message list, ``gui/message_sources.py``).
GUI_TITLE = "Please check this database"


def gui_message(findings: list[Finding]) -> tuple[str, tuple[str, ...]]:
    """Title and paragraphs of the GUI's W-0490 message for one database (*findings*: its findings, as
    :func:`possible_value_list_overwrite_ro` gives them).  Short and plain; the technical detail is on the registry
    page (``docs_url(CODE)``), which the message's "Open documentation" button opens.  Only step 22
    (``invest_methods``) can be flagged, so the text names that list's effect.  The subject line of the message
    names the file."""
    dates: list[str] = []
    for f in findings:
        if _dmy(f.date) not in dates:
            dates.append(_dmy(f.date))
    return GUI_TITLE, (
        f"When this database was upgraded on {' and '.join(dates)}, a bug in a library that FlexTool uses may have "
        "changed 'not_allowed' into 'cumulative_limits'. Units, connections or nodes that should not invest could "
        "then invest without limit.",
        "FlexTool's repair tool writes a repaired copy and leaves this file as it is: the Copy repair command "
        "button copies the command to paste into a terminal, and the documentation shows how to swap the copy in.",
        "This message stays until a repaired file is in its place.",
    )


@dataclass(frozen=True)
class Record:
    """A repair record row: its parsed JSON value, and the id, comment and date of the commit that wrote it."""

    value: dict
    commit: int | None
    comment: str
    date: str


def record_row(con: sqlite3.Connection, key: str) -> tuple[str, int | None, str, str] | None:
    """``(value, commit id, commit comment, commit date)`` of the newest metadata row *key* (several rows: the
    newest by id), or ``None``."""
    row = con.execute('SELECT m.value, m.commit_id, c.comment, c.date FROM metadata m LEFT JOIN "commit" c ON c.id = '
                      "m.commit_id WHERE m.name = ? ORDER BY m.id DESC LIMIT 1", (key,)).fetchone()
    return None if row is None else (str(row[0]), row[1], str(row[2] or ""), str(row[3] or "")[:10])


def record_value(finding: Finding, set_: int, kept: int, *, checked_by_hand: bool = False) -> str:
    """The W-0490 repair record's metadata value (at most 255 characters).  *checked_by_hand*: the user confirmed
    the W-0490 checklist for a list the tool cannot read (``--checked-by-hand``); the record says so."""
    data = {"v": 1, "step": finding.step, "end_commit": finding.end_commit, "end_date": finding.date, "set": set_,
            "kept": kept}
    if checked_by_hand:
        data["checked_by_hand"] = True
    return json.dumps(data, separators=(",", ":"))


def parse_record(value: str) -> dict | None:
    """The record's fields, or ``None`` when *value* is not a version 1 record."""
    try:
        data = json.loads(value)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("v") != 1 or not isinstance(data.get("end_date"), str) or not all(
            isinstance(data.get(k), int) and not isinstance(data.get(k), bool)
            for k in ("step", "end_commit", "set", "kept")) or not isinstance(data.get("checked_by_hand", False), bool):
        return None
    return data


def read_record(path: str | Path) -> Record | None:
    """The newest W-0490 repair record of *path* if it parses and its commit is the tool's (``REPAIR_COMMIT``: a
    record copied in by an import has another comment), else ``None``; read-only."""
    from flextool.update_flextool.frozen import _connect_ro

    with contextlib.closing(_connect_ro(Path(path))) as con:
        row = record_row(con, RECORD_KEY)
    if row is None or row[2] != REPAIR_COMMIT or (value := parse_record(row[0])) is None:
        return None
    return Record(value, row[1], row[2], row[3])


def record_ends(finding: Finding, path: str | Path) -> bool:
    """Whether *path* holds a repair record that ends *finding*: a valid record (:func:`read_record`) for the same
    step, ``end_commit`` and ``end_date``, written by a commit after ``end_commit``, and the step's list holds every
    shipped member and the appended one now.  Any read error: ``False`` (the finding stays)."""
    from flextool.update_flextool.frozen import _connect_ro

    try:
        rec = read_record(path)
        if rec is None or rec.commit is None or rec.commit <= finding.end_commit or (
                rec.value["step"], rec.value["end_commit"], rec.value["end_date"]) != (
                finding.step, finding.end_commit, finding.date) or finding.step not in PRIOR_MEMBERS:
            return False
        with contextlib.closing(_connect_ro(Path(path))) as con:
            rows = con.execute("SELECT lv.value FROM list_value lv JOIN parameter_value_list l ON l.id = "
                               "lv.parameter_value_list_id WHERE l.name = ?", (APPEND[finding.step][0],)).fetchall()
    except (sqlite3.Error, OSError) as exc:
        logger.debug("repair record of %s not read: %s", path, exc)
        return False
    members = set()
    for (value,) in rows:
        try:
            members.add(json.loads(bytes(value) if not isinstance(value, str) else value))
        except (TypeError, ValueError):
            continue
    return {*PRIOR_MEMBERS[finding.step], APPENDED[finding.step]} <= members
