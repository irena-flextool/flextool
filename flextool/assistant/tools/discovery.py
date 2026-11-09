"""Discovery tools: the project's input sources, and its scenario runs with whether their results may be read.

list_runs is compact by default (agent feedback 2026-11-09): counts by state, then one ``[folder, state,
results_usable]`` row per run; the run record's details (verdict, times, version, error) only for named folders, a
filtered set of at most :data:`FULL_DETAIL_MAX` runs, or ``detail: "full"``.  The handler pages the rows itself
(``offset``), measured on the text that is sent (:func:`~flextool.assistant.caps.fit_rows`), so the page note is
never contradicted by a cut of ``fit_json``.
"""
from __future__ import annotations

from typing import Any

from flextool.assistant.caps import fit_rows
from flextool.assistant.context import AssistantContext
from flextool.assistant.registry import Fit, make_tool
from flextool.assistant.results import _runs

#: A filtered set of at most this many runs is listed with full detail.
FULL_DETAIL_MAX = 5
#: The least number of rows for which a field shared by every row is stated once (``same_in_every_row``).
HOIST_MIN_ROWS = 3
#: The fields of a full row, in this order; ``verified`` only on a row without a run record.
_FULL_FIELDS = ("folder", "source_number", "source_name", "db_scenario", "state", "verdict", "started_at",
                "finished_at", "flextool_version", "other_flextool_version", "stale_vs_input", "results_usable",
                "unusable_code", "error", "verified")
#: The fields of a run record: never on a row without one.
_RECORD_FIELDS = frozenset({"verdict", "started_at", "finished_at", "flextool_version", "other_flextool_version",
                            "stale_vs_input", "unusable_code", "error"})
COMPACT_COLUMNS = ["folder", "state", "results_usable"]
_MISSING = object()
PAGE_HINT = "filter by state, results_usable or folders"


def list_sources(ctx: AssistantContext, args: dict) -> Fit:
    scenario = args.get("scenario")
    rows = _runs.list_sources(ctx, scenario=scenario)
    payload: dict[str, Any] = {"sources": rows, "total": len(rows)}
    if scenario is not None:
        payload["notes"] = ["alternatives are highest rank first: a value set in several alternatives comes from "
                            "the first listed; base is the lowest, which the others override."]
    return Fit(payload, rows_key="sources", what="sources", hint="name one input source in the other tools")


def _full_row(run: _runs.Run) -> dict:
    """One run with the details of its run record: null fields left out, ``db_scenario`` only when it is not the
    folder's scenario part; a row without a run record never carries a record field."""
    entry = run.entry
    row: dict[str, Any] = {}
    for key in _FULL_FIELDS:
        value = entry.get(key)
        if value is None:
            continue
        if not run.verified and key in _RECORD_FIELDS:
            continue
        if key == "db_scenario" and value == run.folder_scenario:
            continue
        row[key] = value
    return row


def _same_in_every_row(rows: list[dict]) -> dict:
    """The fields (but ``folder``) that every row has with one value, when there are at least
    :data:`HOIST_MIN_ROWS` rows: a field is stated once only when it holds for each row, so a row never seems to
    have a value it does not (Cycle 1 critique B1)."""
    if len(rows) < HOIST_MIN_ROWS:
        return {}
    common: dict[str, Any] = {}
    for key in _FULL_FIELDS:
        if key == "folder":
            continue
        values = [row.get(key, _MISSING) for row in rows]
        if values[0] is not _MISSING and all(v is not _MISSING and v == values[0] and type(v) is type(values[0])
                                             for v in values):
            common[key] = values[0]
    return common


def list_runs(ctx: AssistantContext, args: dict) -> Fit:
    folders = args.get("folders")
    selected = _runs.select_runs(ctx, source=args.get("source"), scenario=args.get("scenario"),
                                 state=args.get("state"), results_usable=args.get("results_usable"),
                                 folders=folders)
    n = len(selected)
    offset = max(int(args.get("offset") or 0), 0)
    full = (args.get("detail") == "full" or n <= FULL_DETAIL_MAX
            or bool(folders) and all("*" not in str(f) for f in folders))
    counts: dict[str, int] = {}
    for run in selected:
        counts[run.entry["state"]] = counts.get(run.entry["state"], 0) + 1
    payload: dict[str, Any] = {
        "total": n,
        "counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], _runs.state_order(kv[0]), kv[0]))),
        "results_usable": sum(1 for r in selected if r.entry["results_usable"]),
    }
    changed = sum(1 for r in selected if r.entry["stale_vs_input"])
    if changed:
        payload["input_changed_since"] = changed
    other = sum(1 for r in selected if r.entry["other_flextool_version"])
    if other:
        payload["other_flextool_version"] = other
    unrecorded = sum(1 for r in selected if not r.verified)
    if unrecorded:
        payload["notes"] = [f"{unrecorded} {'run has' if unrecorded == 1 else 'runs have'} no run record (state "
                            f"no_record, verified false): the run's success is not known; results may be from a "
                            "failed or older run. Say so when you use them."]
    if full:
        rows = [_full_row(r) for r in selected]
        common = _same_in_every_row(rows)
        if common:
            payload["same_in_every_row"] = common
            rows = [{k: v for k, v in row.items() if k not in common} for row in rows]
    else:
        payload["columns"] = COMPACT_COLUMNS
        rows = [[r.folder.name, r.entry["state"], r.entry["results_usable"]] for r in selected]
    comparison = _runs.comparison_state(ctx)
    if comparison is not None and not full:
        compact = {"usable": comparison["usable"], "members": len(comparison["members"])}
        if comparison.get("note"):
            compact["note"] = comparison["note"]
        comparison = compact
    payload["runs"] = rows[offset:]
    if comparison is not None:
        payload["comparison"] = comparison
    if offset and offset >= n:
        payload["shown"] = f"no runs from offset {offset}; there are {n}"
        return Fit(payload, rows_key="runs", what="runs", hint=f"pass offset or {PAGE_HINT}")

    def note(k: int, _rest: int) -> str | None:
        if offset == 0 and k == n:
            return None
        if k == 0:
            return f"no run fits in one result from offset {offset}; {PAGE_HINT}"
        text = f"runs {offset + 1}–{offset + k} of {n}"
        if offset + k < n:
            text += (f"; pass offset {offset + k} for the next (with the same filters; a new run folder shifts "
                     f"the list), or {PAGE_HINT}")
        return text

    fitted = fit_rows(payload, "runs", ctx.caps.max_chars, ctx.redact, note_key="shown", note=note)
    return Fit(fitted, rows_key="runs", what="runs", hint=f"pass offset or {PAGE_HINT}")


def _prepare_runs(args: dict) -> dict:
    """One folder name where the schema has a list."""
    if isinstance(args.get("folders"), str):
        args = dict(args)
        args["folders"] = [args["folders"]]
    return args


TOOLS = [
    make_tool(
        name="list_sources", title="Input sources of the project",
        description=("The input sources (spreadsheets, databases): number (suffix of their result folders), "
                     "scenarios, database version, conversion state, behaviour notices. scenario: its "
                     "alternatives, highest rank first."),
        properties={"scenario": {"type": "string"}}, handler=list_sources, area="discovery", small_tier=True),
    make_tool(
        name="list_runs", title="Scenario runs of the project",
        description=("Result folders (scenario + source number, as S2_Dry_1): counts by state; folder, state, "
                     "results_usable (no_record: success unknown). folders: names or * patterns. Named folders, up "
                     "to 5 runs or detail full: also verdict, times, version, error. offset pages."),
        properties={"source": {"type": "string"}, "scenario": {"type": "string"}, "state": {"type": "string"},
                    "results_usable": {"type": "boolean"},
                    "folders": {"type": "array", "items": {"type": "string"}},
                    "detail": {"type": "string", "enum": ["full"]}, "offset": {"type": "integer"}},
        handler=list_runs, area="discovery", small_tier=True, prepare=_prepare_runs),
]
