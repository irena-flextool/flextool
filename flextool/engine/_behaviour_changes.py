"""Behaviour-change notices.

When a FlexTool release changes what an existing model does, the database upgrade writes a marker into the
upgraded database (metadata ``flextool_behaviour_change:<id>``, visible in the database editor's Metadata tab).
While the marker is present, each run prints one notice per change naming the entities whose results the change
affects.  Deleting the marker stops the notice.  New databases (and the repository's templates, examples and test
fixtures) carry no markers, so they never print notices.

:data:`BEHAVIOUR_CHANGES` is the registry; the text of each database change (its note code ``W-06xx``, console
text, summary and how to silence it) is in :mod:`flextool.run_warnings._behaviour_specs`.  ``since`` / ``published`` are the FlexTool version and its publication
date (``D.M.YYYY``, as in the CHANGELOG ``## Release X (D.M.YYYY)`` heading); both stay ``None`` until the release
that ships the change fills them in (``tests/engine/test_behaviour_change_registry.py`` refuses a release that
leaves one empty).

A run calls :func:`start_run` with the markers of its input database (:func:`read_markers`).  :func:`notice`
prints one coded ``Note:`` line (:mod:`flextool.run_warnings`; the full notice goes to the debug log) and announces
each entity of a change once per run, so rolls and repeated solves of the same run do not repeat it.  :func:`mark_once` / :func:`already_warned` give other run-time checks the same
once-per-run scope.

``gui_solver_defaults`` is not a database change: its marker is the ``solver_settings_revision`` of a GUI
project's settings.yaml, and the GUI shows the notice once when it updates an older file
(``flextool.gui.settings_io``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from typing import Iterable, Mapping

MARKER_PREFIX = "flextool_behaviour_change:"


@dataclass(frozen=True)
class BehaviourChange:
    """A behaviour change.  A change with an entry in :data:`flextool.run_warnings._behaviour_specs.BEHAVIOUR_TEXTS`
    (every database change) takes its notice code, ``summary`` and ``silence`` from there, so the console note,
    the GUI / assistant text and the warnings page share one text."""
    id: str
    summary: str = ""
    silence: str = ""
    since: str | None = None
    published: str | None = None
    #: The ``Note:`` code of the notice (``W-06xx``); None for a change that is not a run notice.
    code: str | None = None

    def __post_init__(self) -> None:
        from flextool.run_warnings._behaviour_specs import BEHAVIOUR_TEXTS
        texts = BEHAVIOUR_TEXTS.get(self.id)
        if texts is None:
            return
        code, _console, summary, silence = texts
        object.__setattr__(self, "code", self.code or code)
        object.__setattr__(self, "summary", self.summary or summary)
        object.__setattr__(self, "silence", self.silence or silence)


_DELETE_MARKER = "Delete the marker {marker} in the database's Metadata tab once checked."

BEHAVIOUR_CHANGES: dict[str, BehaviourChange] = {
    change.id: change for change in (
        BehaviourChange(id="storage_default_fix_nothing", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="storage_fix_method_rewritten", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="user_constraint_prebuilt_nets_retirements", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="dc_power_flow_lines_beside_dc_links", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="ladder_tier_labels_as_written", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="unnamed_maps_read_per_depth", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="price_ladder_periods_from_solve_periods", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="coarse_process_flows", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="construction_lead_time_commissioning", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="chained_solve_costs_from_run_start", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="stochastic_branch_period_years", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="ramp_branch_point_shutdown_once", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="retirement_of_existing_capacity_only", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="acceptance_ceiling_follows_tolerance", since="4.2.0", published="7.11.2026"),
        BehaviourChange(id="solve_step_series_combined", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="energy_margin_adder_per_original_step", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="capacity_margin_counts_by_flow", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="discount_rate_zero_is_zero", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="invest_without_cost_offered", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="stochastic_storage_paths_tied", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="chained_solve_discount_from_run_start", since="4.4.0", published="13.11.2026"),
        BehaviourChange(id="rolling_cyclic_storage_anchored", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="nested_storage_start_borrowed", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="nested_storage_fix_cumulative", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="rp_storage_chain_pinned", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="storage_end_matches_start", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="storage_reference_start", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="annuity_over_payback_time", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="fixed_cost_per_operating_period", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="annuity_timing", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="net_salvage_once", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="built_capacity_retirable", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="automatic_replacement_cost", since="4.6.0", published="19.11.2026"),
        BehaviourChange(id="end_of_life_closest_seam", since="4.6.0", published="19.11.2026"),
        BehaviourChange(
            id="gui_solver_defaults",
            summary=("Runs started from the FlexTool GUI now use the same solver settings as terminal and Spine "
                     "Toolbox runs: presolve is on (before: choose), and the MIP gap of each solve comes from its "
                     "solver_mip_gap, else mip_rel_gap in solver_config/highs.opt (before: 0.001 for every "
                     "solve; the GUI no longer has a MIP-gap setting). Results of MIP solves (integer "
                     "investments, online variables) can change within the gap; an LP keeps its optimal "
                     "objective value."),
            silence=("This notice is shown once. To keep the old behaviour, choose presolve 'Choose' in the "
                     "Solver options dialog and set solver_mip_gap on the solves in the database."),
            since="4.2.0",
            published="7.11.2026",
        ),
    )
}

_markers: dict[str, str] = {}
_announced: set[tuple[str, str]] = set()
_warned: set = set()


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
    from spinedb_api import SpineDBAPIError
    from flextool.update_flextool.db_handle import open_database_mapping

    try:
        with open_database_mapping(str(db_url)) as db:
            return read_markers(db)
    except SpineDBAPIError:
        return {}


def start_run(markers: Mapping[str, str] | None) -> None:
    """Begin a run: remember the input database's markers and forget what earlier runs announced (also the coded
    warnings of :mod:`flextool.run_warnings`)."""
    _markers.clear()
    _markers.update(dict(markers or {}))
    _announced.clear()
    _warned.clear()
    from flextool import run_warnings
    run_warnings.start_run()


def complete_solve_of(solve: str | None) -> str:
    """The complete solve a sub-solve name belongs to: a roll ``<solve>_roll_<k>`` counts as ``<solve>``."""
    from flextool.run_warnings import complete_solve
    return complete_solve(solve) or ""


def storages_named(names: Iterable[str], noun: str = "storage node") -> str:
    """``"1 storage node (A)"`` / ``"12 storage nodes (A, ...)"``: how many and the first only (owner rule for the
    storage messages: one console line per message type and solve, never a line or a list per node)."""
    names = list(dict.fromkeys(str(n) for n in names))
    count = len(names)
    first = names[0] if names else ""
    return f"{count} {noun}{'' if count == 1 else 's'} ({first}{', ...' if count > 1 else ''})"


def mark_once(key) -> bool:
    """Record *key* for this run without printing (a message that goes to the log, once per run).  Returns True
    the first time in a run, False after that; :func:`start_run` forgets it."""
    if key in _warned:
        return False
    _warned.add(key)
    return True


def already_warned(key) -> bool:
    """Whether :func:`mark_once` recorded *key* earlier in this run."""
    return key in _warned


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


def notice_parts(change_id: str, kind: str, names: Iterable[str], detail: str = "") -> list[str]:
    """The notice of *change_id* in its separate items: what changed, the affected entities, the *detail* (when
    given) and how to silence it.  :func:`notice_text` joins them into the printed line."""
    change = BEHAVIOUR_CHANGES[change_id]
    parts = [f"Behaviour changed in {version_phrase(change)}: {change.summary}",
             f"Affected {kind}: {', '.join(names)}."]
    if detail:
        parts.append(detail)
    parts.append(change.silence)
    return parts


def notice_text(change_id: str, kind: str, names: Iterable[str], detail: str = "") -> str:
    return " ".join(notice_parts(change_id, kind, names, detail))


def display_text(notices: Iterable[Iterable[str]]) -> str:
    """Notices (each as :func:`notice_parts`) for a dialog: each notice its own paragraph with its items on
    their own lines, a blank line between notices."""
    return "\n\n".join("\n".join(parts) for parts in notices)


def unannounced(change_id: str, names: Iterable[str]) -> list[str]:
    """The *names* (de-duplicated, in order) that :func:`notice` has not yet announced for *change_id* in this
    run, so that a caller can build its ``detail`` from the fresh names only."""
    return [n for n in dict.fromkeys(str(n) for n in names) if (change_id, n) not in _announced]


def notice(change_id: str, kind: str, names: Iterable[str], detail: str = "", solve: str | None = None) -> bool:
    """Announce the change *change_id* for the *names* not yet announced in this run: one ``Note:`` line with the
    change's code (``changed in <version>: <what> (<N> cases, e.g. '<first>') [W-06xx]``), and the full notice
    (:func:`notice_text`, every name and the *detail*) in the debug log.

    Fires only while the input database carries the change's marker.  Returns True when the note is announced
    (printed now, or at the end of the open :func:`flextool.run_warnings.collecting` scope).  With *solve* (the
    storage notices): once per change and complete solve (the later calls of the solve only log their new names).
    """
    if change_id not in BEHAVIOUR_CHANGES:
        raise KeyError(f"unknown behaviour change {change_id!r}")
    if change_id not in _markers:
        return False
    import logging
    if solve is not None:
        names = list(dict.fromkeys(str(n) for n in names))
        key = ("notice", change_id, complete_solve_of(solve))
        if not names:
            return False
        if key in _warned:
            fresh = [n for n in names if (change_id, n) not in _announced]
            if fresh:
                _announced.update((change_id, n) for n in fresh)
                logging.getLogger("flextool.engine.storage").debug(
                    "Behaviour change %s in solve '%s', also affected %s: %s", change_id,
                    complete_solve_of(solve), kind, ", ".join(fresh))
            return False
        _warned.add(key)
        _announced.update((change_id, n) for n in names)
        return _note(change_id, kind, names, detail, complete_solve_of(solve))
    fresh = unannounced(change_id, names)
    if not fresh:
        return False
    _announced.update((change_id, n) for n in fresh)
    return _note(change_id, kind, fresh, detail, None)


def _short_version(change: BehaviourChange) -> str:
    """``4.2.0``, or ``the next release`` for a change not yet released (the console line; the debug log has the
    full :func:`version_phrase`)."""
    return change.since if change.since is not None else "the next release"


def _note(change_id: str, kind: str, names: list[str], detail: str, solve: str | None) -> bool:
    """The coded note of *change_id* for *names* (:func:`notice`); a change without a code prints the full text."""
    change = BEHAVIOUR_CHANGES[change_id]
    text = notice_text(change_id, kind, names, detail)
    if change.code is None:
        print(text, flush=True)
        return True
    from flextool import run_warnings
    if solve:
        text = f"In solve '{solve}': {text}"
    run_warnings.note(change.code, names, version=_short_version(change), details=text)
    return True
