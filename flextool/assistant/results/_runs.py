"""Input sources and scenario runs as the tools list them, and the rule that a run's results are offered only
when its latest run succeeded (O1).

A run is a folder ``output_parquet/<scenario>_<source number>/`` with its run record ``_run_status.json`` (P-1).
Whether its results may be read is the run record's own rule: ``run_record.results_usable`` and
``run_record.unusable_code`` on the stored status, with the shown state from ``run_record.effective_state`` (a
``running`` record whose process is gone is shown as abandoned).  Nothing here restates that rule.  A folder
without a run record (from an older FlexTool) is listed when it holds result files.  Statuses are cached by the
file's ``(mtime_ns, size)``.

:func:`resolve_run` is how a result tool gets a folder: only a folder that :func:`run_folders` lists, refused with
the message of its ``unusable_code`` when its results are not usable, with notes on a changed input, another
FlexTool version or a missing run record.  A record that exists but cannot be read now (``unreadable``) or is
damaged (``malformed``) is never usable and is not cached as a missing one.  :func:`comparison_state` says whether the comparison files are built
from the current successful runs (P0 §5.6).
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from functools import cached_property
from datetime import datetime
from pathlib import Path
from typing import Any

from flextool import run_record
from flextool.assistant import project
from flextool.assistant.context import NO_RECORD_STATE, AssistantContext
from flextool.assistant.errors import ToolError, close_names, message, name_list

RESULTS_DIR = "output_parquet"
COMPARISON_DIR = "output_parquet_comparison"
COMPARISON_METADATA = "_metadata.json"
#: The ``format`` of ``_metadata.json`` whose members can be fresh (``db_reader.COMPARISON_METADATA_FORMAT``).
COMPARISON_FORMAT = 2
SCENARIOS_MAX = 100
ERROR_MAX = 300
#: :data:`NO_RECORD_STATE` (from ``context``) is the shown state of a folder without a run record (it holds
#: result files written by an older FlexTool).
#: What every result read from a folder without a run record says (agent feedback 2026-11-09, item 8).
NO_RECORD_NOTE = ("No run record: the run's success is not known; results may be from a failed or older run. Say "
                  "so when you use them.")
#: The states list_runs shows, in this order: the run record's own (``run_record.RUN_STATES``), then
#: :data:`NO_RECORD_STATE`.  A state that neither lists (a record of a newer FlexTool) is shown after them.
RUN_STATES = (*run_record.RUN_STATES, NO_RECORD_STATE)


def state_order(state: str) -> int:
    """Where *state* goes in a listing of states: its place in :data:`RUN_STATES`, a state not listed last."""
    return RUN_STATES.index(state) if state in RUN_STATES else len(RUN_STATES)

_status_cache: dict[str, tuple[tuple[int, int], dict | None]] = {}
_status_lock = threading.Lock()


def read_status(folder: Path) -> dict | None:
    """The run record status of *folder*, read strictly (``None`` without one; ``{"state": "unreadable"}`` /
    ``{"state": "malformed"}`` for one that cannot be read now / is damaged), read again only when the file
    changed.  An unreadable answer is never cached: the next call reads again."""
    path = folder / run_record.RUN_STATUS_FILE
    try:
        st = path.stat()
    except FileNotFoundError:
        return None
    except OSError:
        return {"state": run_record.UNREADABLE_STATE}
    stamp = (st.st_mtime_ns, st.st_size)
    with _status_lock:
        hit = _status_cache.get(str(path))
        if hit is not None and hit[0] == stamp:
            return hit[1]
    status = run_record.read_status_file(folder, strict=True)
    if status is not None and status.get("state") == run_record.UNREADABLE_STATE:
        return status
    with _status_lock:
        _status_cache[str(path)] = (stamp, status)
    return status


def clear_cache() -> None:
    with _status_lock:
        _status_cache.clear()
    _summaries.clear()


def _iso(stamp: float) -> str:
    return datetime.fromtimestamp(stamp).astimezone().isoformat(timespec="seconds")


def _flextool_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("flextool")
    except PackageNotFoundError:
        return None


# ---------------------------------------------------------------------------------------------------------
# Input sources

_summaries: dict[Any, dict] = {}


def _summary(ctx: AssistantContext, source: project.Source) -> dict:
    """``{scenarios, database_version, version_status, behaviour_notices, problem}`` of one source."""
    from flextool.assistant.inputs import _db

    out: dict[str, Any] = {"scenarios": None, "database_version": None, "version_status": None,
                           "behaviour_notices": [], "problem": None}
    try:
        db, notes = project.open_source(ctx, source)
    except ToolError as exc:
        status = {"old_spine_format": "old_spine_format", "busy": "unreadable", "unreadable": "unreadable",
                  "not_flextool": "unreadable"}.get(exc.key)
        out.update(version_status=status, problem=exc.render(ctx.redact))
        if source.kind == "xlsx" and exc.key in ("not_converted", "converting") and source.path.is_file():
            from flextool.gui.input_sources import InputSourceManager

            out["scenarios"] = InputSourceManager(ctx.project_path, project.settings(ctx)).read_scenarios_xlsx(
                source.path)
        return out
    cached = _summaries.get(db.key)
    if cached is None:
        cached = {"scenarios": db.scenario_names(), "database_version": db.version,
                  "version_status": _db.version_status(db.version), "behaviour_notices": list(db.behaviour_notices)}
        _summaries[db.key] = cached
    out.update(cached)
    if notes:
        out["problem"] = " ".join(notes)
    return out


def list_sources(ctx: AssistantContext, scenario: str | None = None) -> list[dict]:
    """The input sources; with *scenario*, only those that have it, each with the scenario's alternatives highest
    rank first and ``base`` (the lowest) in place of the scenario list, and those whose scenarios cannot be read
    now (``scenarios`` null, with their ``problem``): they may have it."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    rows = []
    known: list[str] = []
    for source in project.sources(ctx):
        summary = _summary(ctx, source)
        scenarios = summary["scenarios"]
        if scenario is not None:
            known += scenarios or []
            if scenarios is not None and scenario not in scenarios:
                continue                 # a source whose scenarios cannot be read stays, with its problem
        try:
            modified = _iso(source.path.stat().st_mtime)
        except OSError:
            modified = None
        row: dict[str, Any] = {
            "name": source.name,
            "number": source.number,
            "kind": source.kind,
            "path": project.relative(ctx, source.path),
            "modified": modified,
        }
        if scenario is None or scenarios is None:
            row["scenarios"] = scenarios[:SCENARIOS_MAX] if scenarios is not None else None
            row["scenarios_note"] = (f"showing {SCENARIOS_MAX} of {len(scenarios)} scenarios"
                                     if scenarios is not None and len(scenarios) > SCENARIOS_MAX else None)
        else:
            row["scenario"] = scenario
            ranked = _ranked_alternatives(ctx, source, scenario)
            if ranked is not None:
                row["alternatives"] = ranked
                row["base"] = ranked[-1] if ranked else None
        row.update({
            "database_version": summary["database_version"],
            "version_status": summary["version_status"],
            "conversion": project.conversion_state(ctx, source),
            "behaviour_notices": summary["behaviour_notices"],
        })
        if summary["version_status"] == "needs_migration":
            row["version_note"] = message("needs_migration", db=summary["database_version"],
                                          tool=FLEXTOOL_DB_VERSION)
        elif summary["version_status"] == "newer":
            row["version_note"] = message("newer_database", db=summary["database_version"],
                                          tool=FLEXTOOL_DB_VERSION)
        if summary["problem"]:
            row["problem"] = summary["problem"]
        rows.append(row)
    if scenario is not None and not rows:
        names = list(dict.fromkeys(known))
        raise ToolError("scenario_in_no_source", scenario=scenario,
                        scenarios=name_list(close_names(scenario, names) or names))
    return rows


def _ranked_alternatives(ctx: AssistantContext, source: project.Source, scenario: str) -> list[str] | None:
    """The alternatives of *scenario*, highest rank first (the one whose value counts first), or ``None`` when
    the source cannot be read now."""
    try:
        db, _ = project.open_source(ctx, source)
    except ToolError:
        return None
    ranked = db.scenarios.get(scenario)
    return list(reversed(ranked)) if ranked is not None else None


# ---------------------------------------------------------------------------------------------------------
# Runs


def results_dir(ctx: AssistantContext) -> Path:
    return ctx.project_path / RESULTS_DIR


def run_folders(ctx: AssistantContext) -> list[Path]:
    """The run folders of the project: folders of ``output_parquet`` with a run record or result files."""
    base = results_dir(ctx)
    if not base.is_dir():
        return []
    out = []
    for entry in sorted(base.iterdir(), key=lambda p: p.name):
        if not entry.is_dir() or entry.name.startswith("_"):
            continue
        if _has_record(entry) or any(entry.glob("*.parquet")):
            out.append(entry)
    return out


def _has_record(folder: Path) -> bool:
    """Whether *folder* has a run record file; one that cannot even be looked at (a file pending deletion on
    Windows) counts, so that :func:`read_status` reports it as unreadable."""
    try:
        return (folder / run_record.RUN_STATUS_FILE).is_file()
    except OSError:
        return True


def _source_of(ctx: AssistantContext, folder: str, settings, sources: list[project.Source]
               ) -> tuple[int | None, str | None, str]:
    """``(source number, source name, scenario name from the folder name)``."""
    from flextool.gui.scenario_key import LEGACY_SOURCE_NUMBER, resolve_source_number

    number, scenario = resolve_source_number(folder, dict(settings.bare_output_owners))
    if number == LEGACY_SOURCE_NUMBER:
        return None, None, scenario
    name = next((s.name for s in sources if s.number == number), None)
    if name is None:
        record = settings.source_registry.get(str(number))
        name = record.name if record is not None and record.name else None
    return number, name, scenario


def _input_changed(ctx: AssistantContext, status: dict, source: project.Source | None) -> bool | None:
    """Whether the run's input file changed after the run read it (``None`` when it cannot be told)."""
    recorded = status.get("input") or {}
    if source is None:
        return None
    if recorded.get("source_file"):
        then, path = recorded.get("source_file_mtime"), source.path
    else:
        then, path = recorded.get("db_mtime"), source.database_path(ctx.project_path)
    stamp = run_record.parse_iso(then)
    if stamp is None:
        return None
    try:
        now = path.stat().st_mtime
    except OSError:
        return None
    return int(now) > int(stamp.timestamp())


#: States of a record that exists but whose times cannot be read (a strict read's answers).
_TIMELESS_STATES = frozenset({run_record.UNREADABLE_STATE, run_record.MALFORMED_STATE})


@dataclass
class Run:
    """One run folder as one scan of the project found it.  Its times are computed once (Cycle 2 review n4): a
    listing sorts, pages and notes by them, and a new scan (:func:`runs`) makes new objects."""

    folder: Path
    status: dict | None
    entry: dict
    source: project.Source | None = field(default=None, repr=False)
    #: The scenario part of the folder name.
    folder_scenario: str | None = None

    @property
    def verified(self) -> bool:
        """Whether the folder has a run record: without one, the run's success is not known."""
        return self.status is not None

    @cached_property
    def record_time(self) -> float | None:
        """When the run finished by its readable run record (``finished_at``; ``started_at`` for one that has not
        finished), or ``None``: no record, an unreadable or damaged one, or one without times."""
        if self.status is None or self.entry["state"] in _TIMELESS_STATES:
            return None
        for key in ("finished_at", "started_at"):
            stamp = run_record.parse_iso(self.status.get(key))
            if stamp is not None:
                return stamp.timestamp()
        return None

    @cached_property
    def files_time(self) -> float:
        """The newest modification time of the folder's result files and run record file (0 when none can be
        read).  A copied project gets the copy's times, so this orders runs only among themselves (critique M2)."""
        newest = 0.0
        for path in (*self.folder.glob("*.parquet"), self.folder / run_record.RUN_STATUS_FILE):
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                continue
        return newest

    @cached_property
    def finished(self) -> str | None:
        """The ``finished`` a listing shows, local ``YYYY-MM-DD HH:MM``: the record's ``finished_at`` (``None``
        for a readable record that has not finished); for a run without a readable record when its files were
        last written."""
        if self.record_time is not None:
            stamp = run_record.parse_iso(self.status.get("finished_at")) if self.status else None
            return stamp.astimezone().strftime("%Y-%m-%d %H:%M") if stamp is not None else None
        newest = self.files_time
        return datetime.fromtimestamp(newest).strftime("%Y-%m-%d %H:%M") if newest else None


def newest_first(found: list[Run]) -> list[Run]:
    """*found* newest first (Cycle 2, C1-OD4, critique M2): the runs with a readable run record by its time (at
    full resolution), then the others by when their files were last written (their order is uncertain: a copy of
    the project changes file times); equal times by folder name."""
    def key(run: Run) -> tuple[int, float, str]:
        stamp = run.record_time
        if stamp is not None:
            return 0, -stamp, run.folder.name
        return 1, -run.files_time, run.folder.name
    return sorted(found, key=key)


def _run(ctx: AssistantContext, folder: Path, settings, sources: list[project.Source],
         installed: str | None) -> Run:
    status = read_status(folder)
    number, source_name, folder_scenario = _source_of(ctx, folder.name, settings, sources)
    source = next((s for s in sources if number is not None and s.number == number), None)
    effective = run_record.effective_state(status)
    usable = run_record.results_usable(status)
    code = None if usable else run_record.unusable_code(status, effective)
    entry: dict[str, Any] = {
        "folder": folder.name,
        "source_number": number,
        "source_name": source_name,
        "db_scenario": ((status.get("input") or {}).get("scenario") if status else None) or folder_scenario,
        "state": effective if status is not None else NO_RECORD_STATE,
        "verdict": status.get("verdict") if status else None,
        "started_at": status.get("started_at") if status else None,
        "finished_at": status.get("finished_at") if status else None,
        "flextool_version": ((status.get("versions") or {}).get("flextool") if status else None),
        "other_flextool_version": None,
        "stale_vs_input": _input_changed(ctx, status, source) if status else None,
        "results_usable": usable,
        "unusable_code": code,
        "error": None,
    }
    if entry["flextool_version"] and installed:
        entry["other_flextool_version"] = entry["flextool_version"] != installed
    error = (status.get("error") or {}) if status else {}
    if isinstance(error, dict) and error.get("message"):
        text = str(error["message"])
        entry["error"] = text if len(text) <= ERROR_MAX else text[: ERROR_MAX - 1] + "…"
    if status is None:
        entry["verified"] = False
        entry["note"] = NO_RECORD_NOTE
    return Run(folder, status, entry, source, folder_scenario)


def runs(ctx: AssistantContext) -> list[Run]:
    settings = project.settings(ctx)
    sources = project.sources(ctx)
    installed = _flextool_version()
    return [_run(ctx, folder, settings, sources, installed) for folder in run_folders(ctx)]


def select_runs(ctx: AssistantContext, *, source: str | None = None, scenario: str | None = None,
                state: str | None = None, results_usable: bool | None = None,
                folders: list[str] | None = None) -> list[Run]:
    """The runs that pass every filter given (AND), newest first (:func:`newest_first`: a page offset counts in
    that order).  *folders* holds names and ``*`` patterns; an exact name that
    is no run folder, or a pattern that matches none, is refused with the close names; a *state* that is neither
    a known state nor one a run has, with the states there are."""
    from flextool.assistant.results._query import _matching

    all_runs = runs(ctx)
    selected = all_runs
    if source is not None:
        wanted = project.resolve_source(ctx, source)
        selected = [r for r in selected if r.entry["source_number"] == wanted.number]
    if scenario is not None:
        selected = [r for r in selected if r.entry["db_scenario"] == scenario]
    if state is not None:
        present = sorted({r.entry["state"] for r in all_runs}, key=lambda s: (state_order(s), s))
        if state not in RUN_STATES and state not in present:
            raise ToolError("unknown_run_state", value=state, values=present or list(RUN_STATES))
        selected = [r for r in selected if r.entry["state"] == state]
    if results_usable is not None:
        selected = [r for r in selected if r.entry["results_usable"] is results_usable]
    if folders:
        names = [r.folder.name for r in all_runs]
        for name in folders:
            pattern = "*" in name
            if (name not in names) if not pattern else not _matching(names, [name]):
                if not names:
                    raise ToolError("no_runs")
                close = close_names(name.replace("*", ""), names) if name.replace("*", "") else []
                raise ToolError("no_run_matches" if pattern else "unknown_run", run=name,
                                runs=name_list(close or names))
        keep = set(_matching([r.folder.name for r in selected], [str(f) for f in folders]))
        selected = [r for r in selected if r.folder.name in keep]
    return newest_first(selected)


@dataclass
class ResolvedRun:
    folder: Path
    status: dict | None
    entry: dict
    run_notes: list[str]
    run: Run = field(repr=False)

    @property
    def verified(self) -> bool:
        """Whether the folder has a run record (:attr:`Run.verified`)."""
        return self.run.verified


def find_run(ctx: AssistantContext, run: str) -> Run:
    """The run folder *run*, a name :func:`list_runs` gives, whatever its latest run did."""
    if not run or any(part in run for part in ("/", "\\", "..")):
        raise ToolError("bad_run_name", run=run)
    all_runs = runs(ctx)
    found = next((r for r in all_runs if r.folder.name == run), None)
    if found is None:
        names = [r.folder.name for r in all_runs]
        if not names:
            raise ToolError("no_runs")
        raise ToolError("unknown_run", run=run, runs=name_list(close_names(run, names) or names))
    return found


def resolve_run(ctx: AssistantContext, run: str) -> ResolvedRun:
    """The run folder *run* (a name from :func:`list_runs`) when its results may be read; else the message of its
    ``unusable_code`` (O1: never an earlier run's files)."""
    found = find_run(ctx, run)
    status = found.status
    if not found.entry["results_usable"]:
        code = found.entry["unusable_code"] or "failed"
        if code == "running":
            raise ToolError("run_running", run=run)
        if code == "abandoned":
            raise ToolError("run_abandoned", run=run)
        if code == "unreadable":
            raise ToolError("run_unreadable", run=run)
        if code == "malformed":
            raise ToolError("run_malformed", run=run)
        if code == "no_parquet":
            methods = ((status or {}).get("output") or {}).get("write_methods") or []
            raise ToolError("run_no_parquet", run=run, methods=list(methods) or ["none"])
        error = found.entry["error"] or run_record.unusable_reason(status) or "no reason recorded"
        raise ToolError("run_failed", run=run, error=error)
    return ResolvedRun(found.folder, status, found.entry, run_notes(found), found)


def run_notes(found: Run) -> list[str]:
    """The notes a result read from *found* carries: no run record (:data:`NO_RECORD_NOTE`, always first), an
    input changed after the run, another FlexTool version."""
    notes: list[str] = []
    if found.entry.get("note"):
        notes.append(found.entry["note"])
    if found.entry["stale_vs_input"]:
        notes.append("The input data changed after this run; the results are from the earlier data.")
    if found.entry["other_flextool_version"]:
        notes.append(f"This run was made with FlexTool {found.entry['flextool_version']}; this is FlexTool "
                     f"{_flextool_version()}.")
    return notes


def member_notes(ctx: AssistantContext, folders: list[str]) -> dict[str, list[str]]:
    """:func:`run_notes` of each run folder in *folders* from one scan of the runs (``[]`` for a folder the scan
    does not list)."""
    by_name = {r.folder.name: r for r in runs(ctx)}
    return {name: run_notes(by_name[name]) if name in by_name else [] for name in folders}


# ---------------------------------------------------------------------------------------------------------
# Comparison files


def comparison_state(ctx: AssistantContext) -> dict | None:
    """``{members: [{folder, run_id, started_at, finished_at, fresh}], usable, written_at, write_id}`` of the
    comparison files, or ``None`` when the project has none.

    Usable only when ``_metadata.json`` is of the current ``format`` (2) and every member is fresh: its folder is
    this project's ``output_parquet/<folder>``, and its results are usable with the run it was built from (the
    same ``run_id`` and ``outputs.written_at``; the writer records a run only when it was usable before and after
    the files were read).  A comparison table is further checked when it is read: its scenario level, and the
    ``write_id`` in its footer equal to the metadata's (P0 §5.6).  Metadata with only ``scenarios`` (the window's
    metadata-only update, released FlexTool) or of an older format is never usable: the tools then read each
    scenario's own folder.
    """
    path = ctx.project_path / COMPARISON_DIR / COMPARISON_METADATA
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    base = results_dir(ctx)
    members = meta.get("members")
    written_at = meta.get("written_at")
    if not isinstance(members, list) or not members or not written_at:
        scenarios = [str(s) for s in meta.get("scenarios") or []]
        return {
            "members": [{"folder": s, "run_id": None, "started_at": None, "finished_at": None, "fresh": False}
                        for s in scenarios],
            "usable": False,
            "written_at": None,
            "write_id": None,
            "note": ("The comparison files do not record the runs they were built from; the tools use each "
                     "scenario's own results instead."),
        }
    current = meta.get("format") == COMPARISON_FORMAT and isinstance(meta.get("write_id"), str)
    out_members = []
    outside = False
    unrecorded: list[str] = []
    for member in members:
        entry = member if isinstance(member, dict) else {"folder": member}
        folder = str(entry.get("folder"))
        run_id = entry.get("run_id")
        status = read_status(base / folder) if "/" not in folder and "\\" not in folder else None
        if status is None and "/" not in folder and "\\" not in folder and (base / folder).is_dir():
            unrecorded.append(folder)
        fresh = (current and status is not None and run_record.results_usable(status) and run_id is not None
                 and status.get("run_id") == run_id
                 and (status.get("outputs") or {}).get("written_at") == entry.get("outputs_written_at"))
        if fresh and not _same_folder(entry.get("path"), base / folder):
            fresh, outside = False, True
        out_members.append({"folder": folder, "run_id": run_id,
                            "started_at": status.get("started_at") if status else None,
                            "finished_at": status.get("finished_at") if status else None, "fresh": fresh})
    usable = all(m["fresh"] for m in out_members)
    state: dict[str, Any] = {"members": out_members, "usable": usable, "written_at": written_at,
                             "write_id": meta.get("write_id") if current else None}
    if not current:
        state["note"] = ("The comparison files were written by an earlier FlexTool and cannot be matched to the "
                         "current scenario runs; the tools use each scenario's own results instead.")
    elif outside:
        state["note"] = ("The comparison files were built from scenario folders outside this project; the tools "
                         "use each scenario's own results instead.")
    elif unrecorded:
        state["note"] = (f"The comparison files cannot be matched to their runs: {name_list(unrecorded, 10)} "
                         f"{'has' if len(unrecorded) == 1 else 'have'} no run record. The tools use each "
                         "scenario's own results instead.")
    elif not usable:
        state["note"] = ("The comparison files do not match the current scenario runs; the tools use each "
                         "scenario's own results instead.")
    return state


def _same_folder(recorded: Any, folder: Path) -> bool:
    """Whether *recorded* (a member's resolved ``path``) is *folder*."""
    if not isinstance(recorded, str) or not recorded:
        return False
    try:
        here = folder.resolve()
    except OSError:
        return False
    return os.path.normcase(recorded) == os.path.normcase(str(here))
