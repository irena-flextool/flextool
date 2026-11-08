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
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from flextool import run_record
from flextool.assistant import project
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, close_names, message, name_list

RESULTS_DIR = "output_parquet"
COMPARISON_DIR = "output_parquet_comparison"
COMPARISON_METADATA = "_metadata.json"
SCENARIOS_MAX = 100
RUNS_MAX = 200
ERROR_MAX = 300

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
        out.update(version_status=status, problem=str(exc))
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


def list_sources(ctx: AssistantContext) -> list[dict]:
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    rows = []
    for source in project.sources(ctx):
        summary = _summary(ctx, source)
        scenarios = summary["scenarios"]
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
            "scenarios": scenarios[:SCENARIOS_MAX] if scenarios is not None else None,
            "scenarios_note": (f"showing {SCENARIOS_MAX} of {len(scenarios)} scenarios"
                               if scenarios is not None and len(scenarios) > SCENARIOS_MAX else None),
            "database_version": summary["database_version"],
            "version_status": summary["version_status"],
            "conversion": project.conversion_state(ctx, source),
            "behaviour_notices": summary["behaviour_notices"],
        }
        if summary["version_status"] == "needs_migration":
            row["version_note"] = message("needs_migration", db=summary["database_version"],
                                          tool=FLEXTOOL_DB_VERSION)
        elif summary["version_status"] == "newer":
            row["version_note"] = message("newer_database", db=summary["database_version"],
                                          tool=FLEXTOOL_DB_VERSION)
        if summary["problem"]:
            row["problem"] = summary["problem"]
        rows.append(row)
    return rows


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


@dataclass
class Run:
    folder: Path
    status: dict | None
    entry: dict
    source: project.Source | None = field(default=None, repr=False)


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
        "state": effective,
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
        entry["note"] = ("This folder has no run record (it ran before FlexTool kept run records); FlexTool cannot "
                         "say whether its run succeeded.")
    return Run(folder, status, entry, source)


def runs(ctx: AssistantContext) -> list[Run]:
    settings = project.settings(ctx)
    sources = project.sources(ctx)
    installed = _flextool_version()
    return [_run(ctx, folder, settings, sources, installed) for folder in run_folders(ctx)]


def list_runs(ctx: AssistantContext, source: str | None = None, scenario: str | None = None) -> dict:
    all_runs = runs(ctx)
    if source is not None:
        wanted = project.resolve_source(ctx, source)
        all_runs = [r for r in all_runs if r.entry["source_number"] == wanted.number]
    if scenario is not None:
        all_runs = [r for r in all_runs if r.entry["db_scenario"] == scenario]
    entries = [r.entry for r in all_runs]
    return {
        "runs": entries[:RUNS_MAX],
        "total": len(entries),
        "truncated": (f"showing {RUNS_MAX} of {len(entries)} runs — name a source or a scenario"
                      if len(entries) > RUNS_MAX else None),
        "comparison": comparison_state(ctx),
    }


@dataclass
class ResolvedRun:
    folder: Path
    status: dict | None
    entry: dict
    run_notes: list[str]
    run: Run | None = field(default=None, repr=False)


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
            raise ToolError("run_no_parquet", run=run, methods=", ".join(methods) or "none")
        error = found.entry["error"] or run_record.unusable_reason(status) or "no reason recorded"
        raise ToolError("run_failed", run=run, error=error)
    notes: list[str] = []
    if found.entry.get("note"):
        notes.append(found.entry["note"])
    if found.entry["stale_vs_input"]:
        notes.append("The input data changed after this run; the results are from the earlier data.")
    if found.entry["other_flextool_version"]:
        notes.append(f"This run was made with FlexTool {found.entry['flextool_version']}; this is FlexTool "
                     f"{_flextool_version()}.")
    return ResolvedRun(found.folder, status, found.entry, notes, found)


# ---------------------------------------------------------------------------------------------------------
# Comparison files


def comparison_state(ctx: AssistantContext) -> dict | None:
    """``{members: [{folder, run_id, started_at, finished_at, fresh}], usable, written_at}`` of the comparison
    files, or ``None`` when the project has none.

    Usable only when ``_metadata.json`` records its ``members`` and ``written_at`` and every member's results are
    usable with the run it was built from (the same ``run_id``).  A comparison table is further checked when it
    is read (its own scenario level and time, P0 §5.6).  Metadata with only ``scenarios`` (older writers, the
    window's metadata-only update) is never usable: the tools then read each scenario's own folder.
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
            "note": ("The comparison files do not record the runs they were built from; the tools use each "
                     "scenario's own results instead."),
        }
    out_members = []
    for member in members:
        folder = str(member.get("folder")) if isinstance(member, dict) else str(member)
        run_id = member.get("run_id") if isinstance(member, dict) else None
        status = read_status(base / folder) if "/" not in folder and "\\" not in folder else None
        fresh = (status is not None and run_record.results_usable(status) and run_id is not None
                 and status.get("run_id") == run_id)
        out_members.append({"folder": folder, "run_id": run_id,
                            "started_at": status.get("started_at") if status else None,
                            "finished_at": status.get("finished_at") if status else None, "fresh": fresh})
    usable = all(m["fresh"] for m in out_members)
    state: dict[str, Any] = {"members": out_members, "usable": usable, "written_at": written_at}
    if not usable:
        state["note"] = ("The comparison files do not match the current scenario runs; the tools use each "
                         "scenario's own results instead.")
    return state
