"""Part B of the value-list repair tool: method parameter defaults lost before FlexTool 4.6.3 (item 279).

spinedb_api 0.31.0–0.32.2 stored a list default as the member's text; the next write by a later Spine library stored
``NULL`` for good.  So a database upgraded before 4.6.3 can hold empty defaults where FlexTool ships a member.  This
module finds them, with ``sqlite3`` only (read-only), in a snapshot at the current data version:

- the **reference table** (:func:`reference`): the method parameters (list-valued definitions) of
  ``schemas/spinedb_schema.json`` with a member default (FlexTool's default), ``{(class, parameter): (list name,
  member)}``;
- the **bug-reachable set** (:func:`bug_reachable`): the reference definitions that had a member default in the v25
  template (``schemas/pre_v26/flextool_template_v25.json``), the ones the bug could reach;
- :func:`empty_shipped_defaults` / :func:`empty_hash`: the sorted ``class.parameter`` names of the reference table
  whose default is empty (``NULL``, JSON ``null``, a ``list_value_ref`` that resolves to no member of its own list,
  or text the item-279 pre-pass left), and a short hash of them.  The repair record and the run note W-0491 use
  both;
- :func:`part_b_suspects`: a default suspect for each empty method parameter default bound to its shipped list,
  and a storage suspect for each node the version 77 upgrade would have given ``fix_start`` had the default been
  readable (:func:`storage_suspects`).  A storage suspect has no pre-selected answer of its own; the block's Enter
  keeps the start free and lists the node to check;
- :func:`block_lines` / :func:`block_prompt`: the text of the tool's one question for all method parameter
  defaults and storage nodes (part A ``invest_method`` defaults too), :func:`check_lines` the entities to check;
- :func:`list_defaults_lost_ro` / :func:`note_at_run_start`: the run and GUI note W-0491, until the tool's part B
  record carries the hash of the current empty set.

The bug leaves no trace on the row, so the evidence is per database (the fingerprint commit of the FlexTool 3.x v22
step) and per row (a hand edit after the last version commit, :func:`changed_by_hand`).  Standard library only at
module level: the run note and the GUI import it; the storage rule (``db_migration``, spinedb_api) is imported
inside :func:`storage_suspects`.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import NamedTuple

FINGERPRINT = "Added cumulative investments and parameter value types"  # FlexTool 3.x v22 step, 2024-08-28
FINGERPRINT_SINCE = "2024-08-28"
VERSION_PREFIX = "Updated Flextool data structure to version "
V77_PREFIX = "v77:"
STORAGE = ("node", "storage_start_end_method")
#: The shipped ``node.storage_start_end_method`` default before version 77 (the v77 rule's old default).
OLD_STORAGE_DEFAULT = "fix_start"
STORAGE_KIND = "storage"
#: Part B's repair record (metadata key), separate from the W-0490 record.
PART_B_RECORD_KEY = "flextool_list_default_repair"

PROBABLY = "probably emptied by a Spine library bug (spinedb_api 0.31.0–0.32.2, databases upgraded 2024–2026)"
NO_SIGN = "empty; no sign of the bug"
HAND = "emptied by hand on {date}"
STORAGE_FIXED = "start fixed in an unaffected database"
STORAGE_LATER = "start added after the upgrade"
RESTORE_REASON = ("Recommended: FlexTool's default. A method parameter default that is not FlexTool's default, or "
                  "a missing one, can lead to unwanted behaviour. Restoring it changes no result at this version: "
                  "FlexTool uses this value when none is stored.")
STORAGE_NOTE = ("No answer is recommended: many models bind storage with storage_binding_method instead of fixing "
                "the start.")
MIGRATE_FIRST = ("Empty method parameter defaults: migrate the database first (open it in FlexTool, or "
                 "flextool-migrate-database), then run the tool again to restore them.")
_SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
_COMMENT_CHARS = 60
#: The run / GUI note for a database whose defaults are still empty (:func:`list_defaults_lost_ro`).
NOTE_CODE = "W-0491"
#: Seconds the note's check waits for a writer's lock: a locked file gives no note, not a stalled project open.
NOTE_TIMEOUT_S = 0.5
_STORAGE_NAME = ".".join(STORAGE)

logger = logging.getLogger(__name__)


def _definitions(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    members: dict[str, set] = {}
    for name, value, *_ in data["parameter_value_lists"]:
        members.setdefault(name, set()).add(value)
    return {(cls, name): (lst, default) for cls, name, default, lst, *_ in data["parameter_definitions"]
            if lst and isinstance(default, str) and default in members.get(lst, ())}


@functools.cache
def reference() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, parameter): (list name, shipped member)}`` of the current schema's list defaults."""
    return _definitions(_SCHEMAS / "spinedb_schema.json")


@functools.cache
def bug_reachable() -> frozenset[tuple[str, str]]:
    """The reference definitions that had a member default in the v25 template."""
    return frozenset(reference().keys() & _definitions(_SCHEMAS / "pre_v26" / "flextool_template_v25.json").keys())


def _decode(payload):
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    return raw if isinstance(raw, bytes) else None


def _gaps(con: sqlite3.Connection) -> list[tuple]:
    """``(definition id, class, parameter, class id, list id, list name, member, commit)`` of each reference
    definition bound to its shipped list whose default is empty, sorted by ``class.parameter``."""
    ref = reference()
    members = {i: lid for i, lid in con.execute("SELECT id, parameter_value_list_id FROM list_value")}
    out = []
    for def_id, cls, name, class_id, list_id, list_name, value, typ, commit in con.execute(
            "SELECT pd.id, ec.name, pd.name, ec.id, pd.parameter_value_list_id, l.name, pd.default_value, "
            "pd.default_type, pd.commit_id FROM parameter_definition pd JOIN entity_class ec ON ec.id = "
            "pd.entity_class_id JOIN parameter_value_list l ON l.id = pd.parameter_value_list_id"):
        shipped = ref.get((cls, name))
        if shipped is None or shipped[0] != list_name:
            continue
        raw = _decode(value)
        if raw is None or raw.strip() == b"null":
            empty = True
        elif typ == "list_value_ref":
            empty = not (raw.isdigit() and members.get(int(raw)) == list_id)
        else:
            empty = False  # a stored value of another type: not a lost default
        if empty:
            out.append((def_id, cls, name, class_id, list_id, list_name, shipped[1], commit))
    return sorted(out, key=lambda g: (g[1], g[2]))


def empty_definition_ids(conn: sqlite3.Connection) -> set[int]:
    """The definition ids of the full reference table whose default is empty (:func:`empty_shipped_defaults`)."""
    return {g[0] for g in _gaps(conn)}


def empty_shipped_defaults(conn: sqlite3.Connection) -> list[str]:
    """The sorted ``class.parameter`` names of the full reference table whose default is empty."""
    return [f"{g[1]}.{g[2]}" for g in _gaps(conn)]


def empty_hash(names) -> str:
    """The first 12 hex of sha1 over the newline-joined *names*."""
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


class PartBRecord(NamedTuple):
    """A valid part B repair record and the id and date of the commit that wrote it (a ``NamedTuple``: a
    dataclass needs the module in ``sys.modules``, which a file loaded on its own is not)."""

    date: str
    set: int
    kept: int
    storage: int
    empty: str
    commit: int | None
    commit_date: str


def part_b_record_value(date: str, set_: int, kept: int, storage: int, empty: str) -> str:
    """The part B record's metadata value; *empty* is :func:`empty_hash` of :func:`empty_shipped_defaults` after
    the write (the full reference table)."""
    return json.dumps({"v": 1, "date": date, "set": set_, "kept": kept, "storage": storage, "empty": empty},
                      separators=(",", ":"))


def parse_part_b_record(conn: sqlite3.Connection) -> PartBRecord | None:
    """The newest part B record of the database on *conn* when it parses as version 1 and was written by the
    repair tool's commit (``value_list_history.REPAIR_COMMIT``; a record copied in by an import has another
    comment), else ``None``.  Raises :class:`sqlite3.Error` on a read error."""
    from flextool.update_flextool.value_list_history import REPAIR_COMMIT, record_row

    row = record_row(conn, PART_B_RECORD_KEY)
    if row is None or row[2] != REPAIR_COMMIT:
        return None
    try:
        data = json.loads(row[0])
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get("v") != 1 or not all(
            isinstance(data.get(k), str) for k in ("date", "empty")) or not all(
            isinstance(data.get(k), int) and not isinstance(data.get(k), bool) for k in ("set", "kept", "storage")):
        return None
    return PartBRecord(data["date"], data["set"], data["kept"], data["storage"], data["empty"], row[1], row[3])


class LostDefaults(NamedTuple):
    """What the note W-0491 is about: the sorted bug-reachable empty ``class.parameter`` names, and
    :func:`empty_hash` of the **full** empty set (the hash a part B record carries; the GUI message's instance)."""

    names: tuple[str, ...]
    empty_hash: str


def _data_version(con: sqlite3.Connection) -> int | None:
    row = con.execute("SELECT pd.default_value FROM parameter_definition pd JOIN entity_class ec ON ec.id = "
                      "pd.entity_class_id WHERE ec.name = 'model' AND pd.name = 'version'").fetchone()
    try:
        return int(float(json.loads(_decode(row[0])))) if row else None
    except (TypeError, ValueError, OverflowError):
        return None


def list_defaults_lost_ro(path: str | Path, *, strict: bool = False) -> LostDefaults | None:
    """W-0491's check, cheap and read-only (one ``sqlite3`` connection, :data:`NOTE_TIMEOUT_S`): the empty
    bug-reachable defaults of a database at the current data version that has the fingerprint, unless a valid part
    B record (:func:`parse_part_b_record`) carries the hash of the full empty set.  ``None`` when there is nothing
    to say, and on any read error (a locked or unreadable file gives no note; with *strict* the error is raised,
    so that the GUI does not keep it as "nothing to say")."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.frozen import _connect_ro

    try:
        with contextlib.closing(_connect_ro(Path(path), timeout=NOTE_TIMEOUT_S)) as con:
            if _data_version(con) != FLEXTOOL_DB_VERSION or not any(
                    str(d or "")[:10] >= FINGERPRINT_SINCE
                    for (d,) in con.execute('SELECT date FROM "commit" WHERE comment = ?', (FINGERPRINT,))):
                return None
            empty = empty_shipped_defaults(con)
            names = tuple(n for n in empty if tuple(n.split(".", 1)) in bug_reachable())
            if not names:
                return None
            digest = empty_hash(empty)
            record = parse_part_b_record(con)
            return None if record is not None and record.empty == digest else LostDefaults(names, digest)
    except (sqlite3.Error, OSError) as exc:
        logger.debug("list defaults of %s not checked: %s", path, exc)
        if strict:
            raise
        return None


def note_order(names) -> list[str]:
    """*names* as the note shows them: ``node.storage_start_end_method`` first (the one that can change results),
    the rest sorted."""
    return sorted(names, key=lambda n: (n != _STORAGE_NAME, n))


def note_at_run_start(db_url: str) -> None:
    """W-0491 for the run's input database, once per run; the run log lists every name with its shipped value."""
    from flextool import run_warnings
    from flextool.update_flextool.frozen import sqlite_path_from

    path = sqlite_path_from(db_url)
    if path is None or not path.is_file() or (lost := list_defaults_lost_ro(path)) is None:
        return
    ref = reference()
    details = {n: f"empty method parameter default; FlexTool uses its default {ref[tuple(n.split('.', 1))][1]}"
               for n in note_order(lost.names)}
    run_warnings.warn_grouped(NOTE_CODE, note_order(lost.names), details=details, per_run=True, run_log=True)


def _commits(con: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    return {i: (str(d or "")[:10], c or "") for i, d, c in con.execute('SELECT id, date, comment FROM "commit"')}


def has_fingerprint(commits: dict[int, tuple[str, str]]) -> bool:
    return any(c == FINGERPRINT and d >= FINGERPRINT_SINCE for d, c in commits.values())


@functools.cache
def flextool_commits() -> tuple[str, ...]:
    """The comments of FlexTool's own commits after a version commit: the current-version refresh of
    ``migrate_database`` and the repair tool's.  (``db_migration`` imports spinedb_api: only the repair tool
    calls this.)"""
    from flextool.update_flextool.db_migration import REFRESH_COMMIT
    from flextool.update_flextool.value_list_history import REPAIR_COMMIT

    return (REFRESH_COMMIT, REPAIR_COMMIT)


def changed_by_hand(commit: int | None, commits: dict[int, tuple[str, str]]) -> bool:
    """Whether a definition with *commit* was changed by hand: the commit is later than the last version commit
    and is not one of :func:`flextool_commits`.  The rule of part A's kept ``invest_method`` default and of part
    B's :data:`HAND` label.  ``migrate_database`` ends every session with a version commit, so "after the last
    upgrade" means after the current version's upgrade."""
    versions = [i for i, (_, c) in commits.items() if c.startswith(VERSION_PREFIX)]
    return (commit is not None and bool(versions) and commit > max(versions)
            and commits.get(commit, ("", ""))[1] not in flextool_commits())


def part_b_suspects(analysis) -> tuple:
    """The part B suspects of an analysed snapshot at the current data version (``()`` for an older one: the caller
    prints :data:`MIGRATE_FIRST`): default suspects, then storage suspects."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION
    from flextool.update_flextool.frozen import _connect_ro

    if analysis.version != FLEXTOOL_DB_VERSION:
        return ()
    with contextlib.closing(_connect_ro(analysis.path)) as con:
        gaps, commits = _gaps(con), _commits(con)
        out = [_default_suspect(con, g, commits) for g in gaps]
    if any((g[1], g[2]) == STORAGE for g in gaps) and has_fingerprint(commits):
        out += storage_suspects(analysis.path, commits)
    return tuple(out)


def _when(commits, commit) -> tuple[str, str]:
    date, comment = commits.get(commit, ("", ""))
    return date, comment if len(comment) <= _COMMENT_CHARS else comment[:_COMMENT_CHARS - 1] + "…"


def _default_suspect(con, gap, commits):
    from flextool.update_flextool import value_list_suspects as vs

    def_id, cls, name, class_id, list_id, _list_name, member, commit = gap
    date, comment = _when(commits, commit)
    if changed_by_hand(commit, commits):
        label = HAND.format(date=date) + (f" ({comment})" if comment else "")
    elif has_fingerprint(commits) and (cls, name) in bug_reachable():
        label = PROBABLY
    else:
        label = NO_SIGN
    ctx = vs.Context(relying=len(vs.relying_entities(con, class_id, def_id)), date=date, comment=comment,
                     invest=False)
    return vs.Suspect(f"def:{def_id}", vs.DEFAULT, cls, name, "", "", list_id, "", label,
                      f"the method parameter default is empty ({label}); FlexTool's default is {member}",
                      f"set:{member}", ctx, commit, reason=RESTORE_REASON, keep_text="keep (no default)", part_b=True)


def storage_suspects(path: str | Path, commits: dict[int, tuple[str, str]] | None = None) -> list:
    """One suspect ``sn:<entity id>:<alternative id>`` per ``(node, alternative)`` where the version 77 upgrade
    writes the old default ``fix_start`` (``db_migration.classify_storage_default``, the exact v77 rule, on the
    snapshot opened read-only).  None when the snapshot has no v77 commit.  Raises
    :class:`~value_list_repair.RepairError` for an older SpineDB schema revision, before any spinedb_api open."""
    from flextool.update_flextool.db_migration import v77 as dbm
    from flextool.update_flextool import value_list_suspects as vs
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.frozen import _connect_ro, schema_outdated_ro

    path = Path(path)
    with contextlib.closing(_connect_ro(path)) as con:
        commits = commits if commits is not None else _commits(con)
        v77 = [i for i, (_, c) in commits.items() if c.startswith(V77_PREFIX)]
        if not v77:
            return []
        if schema_outdated_ro(path):  # spinedb_api would refuse it without closing its engine (Windows holds the file)
            from flextool.update_flextool.value_list_repair import RepairError, schema_outdated_message

            raise RepairError(schema_outdated_message())
        with open_database_mapping(f"sqlite:///{path.resolve().as_posix()}", upgrade=False) as db:
            methods, starts, ends, scenario_alts, inactive = dbm._v77_storage_inputs(db)
        writes = dbm.classify_storage_default(methods, starts, ends, scenario_alts, OLD_STORAGE_DEFAULT, inactive)
        list_id = con.execute(
            "SELECT pd.parameter_value_list_id FROM parameter_definition pd JOIN entity_class ec ON ec.id = "
            "pd.entity_class_id WHERE ec.name = ? AND pd.name = ?", STORAGE).fetchone()[0]
        scenarios: dict[int, list[str]] = {}
        for alt, name in con.execute("SELECT sa.alternative_id, s.name FROM scenario_alternative sa JOIN scenario s "
                                     "ON s.id = sa.scenario_id ORDER BY s.name"):
            scenarios.setdefault(alt, []).append(name)
        out = []
        for node, alternative in writes:
            entity_id, alt_id, commit = con.execute(
                "SELECT e.id, a.id, pv.commit_id FROM entity e JOIN entity_class ec ON ec.id = e.class_id "
                "JOIN alternative a ON a.name = ? LEFT JOIN parameter_value pv ON pv.entity_id = e.id AND "
                "pv.alternative_id = a.id AND pv.parameter_definition_id = (SELECT pd.id FROM parameter_definition "
                "pd WHERE pd.entity_class_id = ec.id AND pd.name = 'storage_state_start') "
                "WHERE ec.name = 'node' AND e.name = ?", (alternative, node)).fetchone()
            date, comment = _when(commits, commit)
            if commit is None or commit < min(v77):
                label, evidence = STORAGE_FIXED, ("an unaffected database keeps this node's start fixed (fix_start); "
                                                  "this one leaves it free (results change)")
            else:
                label, evidence = STORAGE_LATER, "start value added after the upgrade; the upgrade never saw it"
            ctx = vs.Context(scenarios=tuple(scenarios.get(alt_id, ())), date=date, comment=comment, invest=False)
            out.append(vs.Suspect(f"sn:{entity_id}:{alt_id}", STORAGE_KIND, *STORAGE, node, alternative, list_id, "",
                                  label, evidence, None, ctx, commit, reason=STORAGE_NOTE,
                                  keep_text="keep (the start stays free)", part_b=True))
    return out


# -- the block question (design ``repair_partB_prompt_design.md`` §1c) --------------------------------------------
O1 = "The default values for all or some methods are not set in this database."
O2 = (" This is probably due to a Spine library bug (spinedb_api 0.31.0–0.32.2, databases upgraded "
      "2024–2026).")
W = ("Some invest_method defaults read cumulative_limits instead of not_allowed: {} Spine library bug overwrote "
     "not_allowed in the invest_methods list (W-0490).")
CHECK_INTRO = " Afterwards, check these entities and give them their own value by hand where needed:"
CHECK_CLOSING = ("  after the restore they neither invest nor retire; give them their own invest_method if they "
                 "should")
KEEP_IT = "If you chose cumulative_limits as a default yourself, type e and keep it."
EACH = "e goes through them here, one by one."
SKIPS = "x skips: the defaults stay {}, which may cause unwanted behaviour later."
#: Names per check-list line in the prompt (the report lists all).
CHECK_NAMES = 8
_KEEP = "keep"


def _part_a(block) -> list:
    return [s for s in block if s.kind == "default" and not s.part_b]


def _sets(block) -> list:
    return [s for s in block if s.kind == "default" and s.suggestion != _KEEP]


def _default_line(s) -> str:
    from flextool.update_flextool import value_list_suspects as vs

    head = f"  {s.cls}.{s.parameter}: {vs.rely(s.context.relying or 0)} on the default"
    if not s.part_b:
        old = f"; {s.archive_note}" if s.archive_note else ""  # --archive: what the old copy had (R13)
        if s.suggestion == _KEEP:
            return f"{head}, which reads {s.current}; changed by hand on {s.context.date}, so it is kept{old}"
        return f"{head}, which reads {s.current} (FlexTool's default: {vs.FLEXTOOL_DEFAULT}){old}"
    out = f"{head} ({reference()[(s.cls, s.parameter)][1]})"
    if s.suggestion == _KEEP:
        return f"{out}; {s.reason}, so it stays unset"
    if s.label == NO_SIGN:
        return f"{out}; no sign of the bug"
    return out if s.label == PROBABLY else f"{out}; {s.label}"


def _names(names, limit: int | None) -> str:
    if limit is None or len(names) <= limit:
        return ", ".join(names)
    return f"{', '.join(names[:limit])} and {len(names) - limit} more (the report lists all)"


def check_lines(defaults, storage, limit: int | None = CHECK_NAMES) -> list[str]:
    """The entities to check by hand after a restore: for each part A ``invest_method`` default in *defaults* (set
    to FlexTool's default), its relying entities split by cumulative cap (at most *limit* names per line; ``None``:
    all), then one closing line; each storage suspect in *storage* (its start stays free)."""
    out = []
    for s in sorted(defaults, key=lambda s: (s.cls, s.parameter)):
        for capped, which, how in ((False, "no cumulative cap", "up to 1e6 MW"),
                                   (True, "with a cumulative cap", "within the cap")):
            names = [n for n, c in s.context.relying_names if c == capped]
            if names:
                out.append(f"  {s.cls}.{s.parameter}, {which}: {_names(names, limit)} could invest and retire {how} "
                           "until now")
    if out:
        out.append(CHECK_CLOSING)
    for s in storage:
        why = ("an unaffected database fixes its start (fix_start); here it stays free" if s.label == STORAGE_FIXED
               else "start added after the upgrade; it stays free")
        out.append(f"  node {s.entity} [{s.alternative}] (scenarios: {', '.join(s.context.scenarios) or 'none'}): "
                   f"{why}")
    return out


def offers_each(block, change: bool) -> bool:
    """``e`` is offered when Enter changes results, or a part A default is in the block (owner answer Q1)."""
    return change or bool(_part_a(block))


def block_lines(block, change: bool) -> list[str]:
    """The block question's lines (unwrapped; the item lines start with two spaces) for the suspects *block*
    (``value_list_suspects.defaults_block``); *change* is ``value_list_suspects.results_change(block)``."""
    defaults = sorted((s for s in block if s.kind == "default"), key=lambda s: (s.cls, s.parameter))
    part_b = [s for s in defaults if s.part_b]
    part_a, sets = _part_a(block), _sets(block)
    out = []
    probably = bool(part_b) and any(s.label == PROBABLY for s in part_b)
    if part_b:
        out.append(O1 + (O2 if probably else ""))
    if part_a:
        out.append(W.format("another" if probably else "a"))
    out += [_default_line(s) for s in defaults]
    hand = any(s.suggestion == _KEEP for s in part_a)
    kept = any(s.suggestion == _KEEP for s in part_b)
    if sets or not (hand or kept):
        enter = "Enter restores FlexTool's method parameter defaults" + (
            ", keeps the ones changed by hand" if hand else "") + (
            " and leaves the ones kept in an earlier repair" if kept else "") + "."
    else:
        enter = "Enter " + " and ".join(
            (["keeps the defaults changed by hand"] if hand else [])
            + ([f"leaves the {'ones' if hand else 'defaults'} kept in an earlier repair unset"] if kept else [])) + "."
    if change:
        out.append(enter + CHECK_INTRO)
        out += check_lines([s for s in part_a if s in sets and (s.context.relying or 0) > 0],
                           [s for s in block if s.kind == STORAGE_KIND])
    else:
        clauses = (["FlexTool already uses these defaults when none is set"] if any(s.part_b for s in sets) else []) \
            + (["no entity relies on the invest_method defaults that Enter changes"] if any(
                not s.part_b for s in sets) else [])
        out.append(enter + " This does not change results" + (": " + "; ".join(clauses) if clauses else "") + ".")
    if any(not s.part_b for s in sets):
        out.append(KEEP_IT)
    if offers_each(block, change):
        out.append(EACH)
    out.append(SKIPS.format("as they are" if part_a else "unset"))
    return out


def block_prompt(block, change: bool) -> str:
    """The block question's key line: "Enter = keep" when Enter sets nothing."""
    keys = [f"Enter = {'restore' if _sets(block) else 'keep'}", "e = one by one" if offers_each(block, change) else "",
            "x = skip", "q = quit"]
    return f"  [{'; '.join(k for k in keys if k)}] > "
