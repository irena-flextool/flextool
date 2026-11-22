"""Part B of the value-list repair tool: list-valued definition defaults lost before FlexTool 4.6.3 (item 279).

spinedb_api 0.31.0–0.32.2 stored a list default as the member's text; the next write by a later Spine library stored
``NULL`` for good.  So a database upgraded before 4.6.3 can hold empty defaults where FlexTool ships a member.  This
module finds them, with ``sqlite3`` only (read-only), in a snapshot at the current data version:

- the **reference table** (:func:`reference`): the list-valued definitions of ``schemas/spinedb_schema.json`` with a
  member default, ``{(class, parameter): (list name, member)}``;
- the **bug-reachable set** (:func:`bug_reachable`): the reference definitions that had a member default in the v25
  template (``schemas/pre_v26/flextool_template_v25.json``), the ones the bug could reach;
- :func:`empty_shipped_defaults` / :func:`empty_hash`: the sorted ``class.parameter`` names of the reference table
  whose default is empty (``NULL``, JSON ``null``, a ``list_value_ref`` that resolves to no member of its own list,
  or text the item-279 pre-pass left), and a short hash of them.  The repair record and the run note W-0491 use
  both;
- :func:`part_b_suspects`: a default suspect for each empty reference default bound to its shipped list, and a
  storage suspect for each node the version 77 upgrade would have given ``fix_start`` had the default been readable
  (:func:`storage_suspects`).  A storage suspect has no pre-selected answer (owner answer QB (b)).

The bug leaves no trace on the row, so the evidence is per database (the fingerprint commit of the FlexTool 3.x v22
step) and per row (a hand edit after the last version commit).  Standard library only at module level: the run note
and the GUI import it; the storage rule (``db_migration``, spinedb_api) is imported inside :func:`storage_suspects`.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import sqlite3
from pathlib import Path

FINGERPRINT = "Added cumulative investments and parameter value types"  # FlexTool 3.x v22 step, 2024-08-28
FINGERPRINT_SINCE = "2024-08-28"
VERSION_PREFIX = "Updated Flextool data structure to version "
V77_PREFIX = "v77:"
STORAGE = ("node", "storage_start_end_method")
#: The shipped ``node.storage_start_end_method`` default before version 77 (the v77 rule's old default).
OLD_STORAGE_DEFAULT = "fix_start"
STORAGE_KIND = "storage"

PROBABLY = "probably emptied by a Spine library bug (spinedb_api 0.31–0.32, 2024–2025)"
NO_SIGN = "empty; no sign of the bug"
HAND = "emptied by hand on {date}"
STORAGE_FIXED = "start fixed in an unaffected database"
STORAGE_LATER = "start added after the upgrade"
RESTORE_REASON = ("Recommended: FlexTool's shipped default. A class default that is not the shipped default, or a "
                  "missing one, can lead to unwanted behaviour. Restoring it changes no result at this version: "
                  "FlexTool uses this value when none is stored.")
STORAGE_NOTE = ("No answer is recommended: many models bind storage with storage_binding_method instead of fixing "
                "the start.")
MIGRATE_FIRST = ("Empty shipped defaults: migrate the database first (open it in FlexTool, or "
                 "flextool-migrate-database), then run the tool again to restore empty defaults.")
_SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"
_COMMENT_CHARS = 60


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


def empty_shipped_defaults(conn: sqlite3.Connection) -> list[str]:
    """The sorted ``class.parameter`` names of the full reference table whose default is empty."""
    return [f"{g[1]}.{g[2]}" for g in _gaps(conn)]


def empty_hash(names) -> str:
    """The first 12 hex of sha1 over the newline-joined *names*."""
    return hashlib.sha1("\n".join(names).encode("utf-8")).hexdigest()[:12]


def _commits(con: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    return {i: (str(d or "")[:10], c or "") for i, d, c in con.execute('SELECT id, date, comment FROM "commit"')}


def has_fingerprint(commits: dict[int, tuple[str, str]]) -> bool:
    return any(c == FINGERPRINT and d >= FINGERPRINT_SINCE for d, c in commits.values())


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
    versions = [i for i, (_, c) in commits.items() if c.startswith(VERSION_PREFIX)]
    date, comment = _when(commits, commit)
    if commit is not None and versions and commit > max(versions):
        label = HAND.format(date=date) + (f" ({comment})" if comment else "")
    elif has_fingerprint(commits) and (cls, name) in bug_reachable():
        label = PROBABLY
    else:
        label = NO_SIGN
    (relying,) = con.execute(
        "SELECT count(*) FROM entity e WHERE class_id = ? AND NOT EXISTS (SELECT 1 FROM parameter_value pv WHERE "
        "pv.entity_id = e.id AND pv.parameter_definition_id = ?)", (class_id, def_id)).fetchone()
    ctx = vs.Context(relying=relying, date=date, comment=comment, invest=False)
    return vs.Suspect(f"def:{def_id}", vs.DEFAULT, cls, name, "", "", list_id, "", label,
                      f"the class default is empty ({label}); the shipped default is {member}", f"set:{member}", ctx,
                      commit, reason=RESTORE_REASON, keep_text="keep (no default)", part_b=True)


def storage_suspects(path: str | Path, commits: dict[int, tuple[str, str]] | None = None) -> list:
    """One suspect ``sn:<entity id>:<alternative id>`` per ``(node, alternative)`` where the version 77 upgrade
    writes the old default ``fix_start`` (``db_migration.classify_storage_default``, the exact v77 rule, on the
    snapshot opened read-only).  None when the snapshot has no v77 commit."""
    from flextool.update_flextool import db_migration as dbm
    from flextool.update_flextool import value_list_suspects as vs
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.frozen import _connect_ro

    path = Path(path)
    with contextlib.closing(_connect_ro(path)) as con:
        commits = commits if commits is not None else _commits(con)
        v77 = [i for i, (_, c) in commits.items() if c.startswith(V77_PREFIX)]
        if not v77:
            return []
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
