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

TODO(v5.0.0): remove this module, W-0490 (``run_warnings/_input_specs.py``), its call sites
(``engine/solve/_orchestration.py``, ``gui/input_sources.py``) and its tests.  The warning is temporary.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
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
     79, 80, 81, 82, *APPEND])

_PREFIX = "Updated Flextool data structure to version "


@dataclass(frozen=True)
class Finding:
    step: int
    lists: tuple[str, ...]
    start: int
    end: int
    date: str


def _version_commits(path: Path) -> list[tuple[int, str]]:
    from flextool.update_flextool.frozen import ro_uri

    con = sqlite3.connect(ro_uri(path) + "&immutable=1", uri=True)
    try:
        rows = con.execute('SELECT comment, date FROM "commit" WHERE comment LIKE ? ORDER BY id',
                           (_PREFIX + "%",)).fetchall()
    finally:
        con.close()
    out: list[tuple[int, str]] = []
    for comment, date in rows:
        try:
            version = int(float(comment[len(_PREFIX):]))
        except ValueError:
            continue
        if not out or out[-1][0] != version:
            out.append((version, str(date)[:10]))
    return out


def possible_value_list_overwrite_ro(path: str | Path) -> list[Finding]:
    """The migration sessions of *path* that may have overwritten a value list member; ``[]`` when none, or when the
    file cannot be read (a check that cannot read the history does not warn)."""
    try:
        commits = _version_commits(Path(path))
    except (sqlite3.Error, OSError) as exc:
        logger.debug("value list history of %s not read: %s", path, exc)
        return []
    findings = []
    # The first version commit has no known start: the database came from a template or a JSON import.
    for (start, _), (end, date) in zip(commits, commits[1:]):
        if not WINDOW_START <= date <= WINDOW_END or end <= start:
            continue
        touched = [s for s in TOUCH if start < s <= end]
        first = min(touched, default=None)
        if first in APPEND:
            findings.append(Finding(first, APPEND[first], start, end, date))
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


def gui_messages(path: str | Path) -> list[str]:
    """The W-0490 text the GUI shows when it opens *path* (one per flagged session)."""
    from flextool.run_warnings import docs_url

    out = []
    for f in possible_value_list_overwrite_ro(path):
        lists = ", ".join(f"'{name}'" for name in f.lists)
        text = (f"{Path(path).name}: this database may have been affected by a spinedb_api bug (value lists) "
                f"[{CODE}]. It was migrated from data version {f.start} to {f.end} on {f.date}. Migrations in that "
                "period may have run with spinedb_api 0.31-0.34, in which adding a member to a value list could "
                "overwrite the list's first member; every default and value that used that member then changed "
                f"with it. Check the value list {lists}: it should have all its members.")
        if "invest_methods" in f.lists:
            text += (" 'invest_methods' should contain 'not_allowed'. If it is missing, the default of "
                     "invest_method (unit, connection, node) and every value that was not_allowed before that date "
                     "now read 'cumulative_limits', which lets those entities invest and retire without limit.")
        out.append(text + f" If you copied data from this database into another one, check that one too. "
                          f"The checklist: {docs_url(CODE)}")
    return out
