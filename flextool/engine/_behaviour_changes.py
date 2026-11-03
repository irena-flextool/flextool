"""Behaviour-change notices.

When a FlexTool release changes what an existing model does, the database upgrade writes a marker into the
upgraded database (metadata ``flextool_behaviour_change:<id>``, visible in the database editor's Metadata tab).
While the marker is present, each run prints one notice per change naming the entities whose results the change
affects.  Deleting the marker stops the notice.  New databases (and the repository's templates, examples and test
fixtures) carry no markers, so they never print notices.

:data:`BEHAVIOUR_CHANGES` is the registry.  ``since`` / ``published`` are the FlexTool version and its publication
date (``D.M.YYYY``, as in the CHANGELOG ``## Release X (D.M.YYYY)`` heading); both stay ``None`` until the release
that ships the change fills them in (``tests/engine/test_behaviour_change_registry.py`` refuses a release that
leaves one empty).

A run calls :func:`start_run` with the markers of its input database (:func:`read_markers`).  :func:`notice`
prints with ``print(..., flush=True)`` and announces each entity of a change once per run, so rolls and repeated
solves of the same run do not repeat it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from typing import Iterable, Mapping

MARKER_PREFIX = "flextool_behaviour_change:"


@dataclass(frozen=True)
class BehaviourChange:
    id: str
    summary: str
    silence: str
    since: str | None = None
    published: str | None = None


_DELETE_MARKER = "Delete the marker {marker} in the database's Metadata tab once checked."

BEHAVIOUR_CHANGES: dict[str, BehaviourChange] = {
    change.id: change for change in (
        BehaviourChange(
            id="storage_default_fix_nothing",
            summary=("The default storage_start_end_method is now fix_nothing: a storage node without its own "
                     "storage_start_end_method starts free. Before, it was fixed at a storage state of 0 at the "
                     "start of the model when another node had a storage_state_start (results change)."),
            silence="Set storage_start_end_method on these nodes (fix_nothing keeps the new behaviour).",
        ),
        BehaviourChange(
            id="storage_fix_method_rewritten",
            summary=("A fix_start / fix_end / fix_start_end without storage_state_start / storage_state_end is now "
                     "an error. The database upgrade rewrote these nodes to what they did before, so results are "
                     "unchanged."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "storage_fix_method_rewritten"),
        ),
        BehaviourChange(
            id="user_constraint_prebuilt_nets_retirements",
            summary=("The pre-built capacity term of a user constraint "
                     "(constraint_cumulative_pre_built_capacity_coeff) now subtracts the capacity retired in "
                     "earlier periods of the same solve too (before: only retirements of earlier solves of a "
                     "chain), so a single solve and a chain of solves agree."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "user_constraint_prebuilt_nets_retirements"),
        ),
        BehaviourChange(
            id="dc_power_flow_lines_beside_dc_links",
            summary=("DC power-flow lines are the connections inside dc_power_flow_with_angles groups (other than "
                     "is_DC connections). Before, a model with any is_DC connection used the is_DC connections "
                     "instead: the group's lines had no angle constraints and the reference node could move "
                     "(results change)."),
            silence=_DELETE_MARKER.format(marker=MARKER_PREFIX + "dc_power_flow_lines_beside_dc_links"),
        ),
    )
}

_markers: dict[str, str] = {}
_announced: set[tuple[str, str]] = set()


def read_markers(db) -> dict[str, str]:
    """``{change id: marker value}`` of the behaviour-change markers of an open ``DatabaseMapping``."""
    out: dict[str, str] = {}
    for md in db.get_items("metadata"):
        name = str(md["name"])
        if name.startswith(MARKER_PREFIX):
            out[name[len(MARKER_PREFIX):]] = str(md["value"])
    return out


def read_markers_from_url(db_url: str) -> dict[str, str]:
    """:func:`read_markers` of the database at *db_url* (a read-only look at its metadata table).

    A database that cannot be opened gives ``{}``: the run reports that itself when it reads the model.
    """
    from spinedb_api import DatabaseMapping, SpineDBAPIError

    try:
        with DatabaseMapping(str(db_url)) as db:
            return read_markers(db)
    except SpineDBAPIError:
        return {}


def start_run(markers: Mapping[str, str] | None) -> None:
    """Begin a run: remember the input database's markers and forget what earlier runs announced."""
    _markers.clear()
    _markers.update(dict(markers or {}))
    _announced.clear()


def has_marker(change_id: str) -> bool:
    return change_id in _markers


def marker_value(change_id: str) -> str | None:
    return _markers.get(change_id)


def marker_json(change_id: str) -> dict:
    """The marker value of *change_id* parsed as a JSON object; ``{}`` when absent or not JSON."""
    raw = _markers.get(change_id)
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _installed_version() -> str:
    try:
        return metadata.version("flextool")
    except metadata.PackageNotFoundError:
        return "unknown"


def version_phrase(change: BehaviourChange) -> str:
    """``FlexTool 4.2.0 (published 1.12.2026)``, or the not-yet-published form for an unreleased change."""
    if change.since is None:
        return f"the FlexTool release after {_installed_version()} (not yet published)"
    if change.published is None:
        return f"FlexTool {change.since}"
    return f"FlexTool {change.since} (published {change.published})"


def notice_text(change_id: str, kind: str, names: Iterable[str], detail: str = "") -> str:
    change = BEHAVIOUR_CHANGES[change_id]
    text = (f"Behaviour changed in {version_phrase(change)}: {change.summary} "
            f"Affected {kind}: {', '.join(names)}.")
    if detail:
        text += f" {detail}"
    return f"{text} {change.silence}"


def notice(change_id: str, kind: str, names: Iterable[str], detail: str = "") -> bool:
    """Print the notice of *change_id* for the *names* not yet announced in this run.

    Fires only while the input database carries the change's marker.  Returns True when a line was printed.
    """
    if change_id not in BEHAVIOUR_CHANGES:
        raise KeyError(f"unknown behaviour change {change_id!r}")
    if change_id not in _markers:
        return False
    fresh = [str(n) for n in dict.fromkeys(str(n) for n in names) if (change_id, str(n)) not in _announced]
    if not fresh:
        return False
    _announced.update((change_id, n) for n in fresh)
    print(notice_text(change_id, kind, fresh, detail), flush=True)
    return True
