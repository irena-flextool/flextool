"""Discovery tools: the project's input sources, and its scenario runs with whether their results may be read.

list_runs is compact by default (agent feedback 2026-11-09): counts by state, then one ``[folder, state,
results_usable]`` row per run; the run record's details (verdict, times, version, error) only for named folders, a
filtered set of at most :data:`FULL_DETAIL_MAX` runs, or ``detail: "full"``.  The handler pages the rows itself
(``offset``), measured on the text that is sent (:func:`~flextool.assistant.caps.fit_rows`), so the page note is
never contradicted by a cut of ``fit_json``.
"""
from __future__ import annotations

from typing import Any

from flextool.assistant.caps import fit_page
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, name_list
from flextool.assistant.registry import BasicForm, Fit, make_tool
from flextool.assistant.results import _overview as _overview_mod
from flextool.assistant.results import _query, _runs

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
COMPACT_COLUMNS = ["folder", "state", "finished", "results_usable"]
#: The compact columns with the costs of ``detail: "scan"``.
SCAN_COLUMNS = [*COMPACT_COLUMNS, "total_cost", "penalty_cost", "largest_cost"]
#: The most folder names a scan header lists (then "and N more").
NAMES_MAX = 10
#: The latest folders an overview refusal names.
OVERVIEW_LATEST = 5
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


def _state_counts(found: list[_runs.Run]) -> dict[str, int]:
    """Counts by state, largest first (then the order of :data:`~flextool.assistant.results._runs.RUN_STATES`)."""
    counts: dict[str, int] = {}
    for run in found:
        counts[run.entry["state"]] = counts.get(run.entry["state"], 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], _runs.state_order(kv[0]), kv[0])))


def _order_note(found: list[_runs.Run]) -> str | None:
    """The note on the runs without a readable run record, which come last in an uncertain order (critique M2)."""
    m = sum(1 for r in found if r.record_time is None)
    if not m:
        return None
    one = m == 1
    return (f"{m} {'run' if one else 'runs'} without a readable run record {'is' if one else 'are'} listed after "
            f"the others, by when {'its' if one else 'their'} files were last written (finished), which copying a "
            "project changes: their order is uncertain.")


def _names(names: list[str]) -> str:
    return str(name_list(names, NAMES_MAX))


def list_runs(ctx: AssistantContext, args: dict) -> Fit:
    detail = args.get("detail")
    if detail == "overview":
        return _overview(ctx, args)
    folders = args.get("folders")
    selected = _runs.select_runs(ctx, source=args.get("source"), scenario=args.get("scenario"),
                                 state=args.get("state"), results_usable=args.get("results_usable"),
                                 folders=folders)
    n = len(selected)
    offset = max(int(args.get("offset") or 0), 0)
    filtered = any(args.get(key) is not None for key in ("source", "scenario", "state", "results_usable")) or \
        bool(folders)
    latest = not filtered and not args.get("all")
    # A given detail wins; without one, a short or named list gets full detail (critique M6).
    if detail is None:
        full = n <= FULL_DETAIL_MAX or bool(folders) and all("*" not in str(f) for f in folders)
    else:
        full = detail == "full"
    scan = detail == "scan"
    header: dict[str, Any] = {"total": n, "counts": _state_counts(selected),
                              "results_usable": sum(1 for r in selected if r.entry["results_usable"])}
    changed = sum(1 for r in selected if r.entry["stale_vs_input"])
    if changed:
        header["input_changed_since"] = changed
    other = sum(1 for r in selected if r.entry["other_flextool_version"])
    if other:
        header["other_flextool_version"] = other
    notes = []
    unrecorded = sum(1 for r in selected if not r.verified)
    if unrecorded:
        notes.append(f"{unrecorded} {'run has' if unrecorded == 1 else 'runs have'} no run record (state "
                     "no_record, verified false): the run's success is not known; results may be from a failed or "
                     "older run. Say so when you use them.")
    order = _order_note(selected)
    if order:
        notes.append(order)
    if notes:
        header["notes"] = notes
    comparison = _runs.comparison_state(ctx)
    if comparison is not None and not full:
        compact = {"usable": comparison["usable"], "members": len(comparison["members"])}
        if comparison.get("note"):
            compact["note"] = comparison["note"]
        comparison = compact
    if offset and offset >= n:
        payload = {**header, "runs": []}
        if comparison is not None:
            payload["comparison"] = comparison
        payload["shown"] = f"no runs from offset {offset}; there are {n}"
        return Fit(payload, rows_key="runs", what="runs", hint=f"pass offset or {PAGE_HINT}")
    page = selected[offset:]
    if latest:
        page = page[:ctx.caps.max_runs]
    if full:
        rows_all = [_full_row(r) for r in selected]
        common = _same_in_every_row(rows_all)
        if common:
            header["same_in_every_row"] = common
            rows_all = [{k: v for k, v in row.items() if k not in common} for row in rows_all]
        full_rows = rows_all[offset:offset + len(page)]
    costs: dict[str, _overview_mod.Costs | None] = {}

    def cost_of(run: _runs.Run) -> _overview_mod.Costs | None:
        if run.folder.name not in costs:
            costs[run.folder.name] = _overview_mod.read_costs(run.folder)
        return costs[run.folder.name]

    def compact_row(run: _runs.Run) -> list:
        row = [run.folder.name, run.entry["state"], run.finished, run.entry["results_usable"]]
        if scan:
            found = cost_of(run) if run.entry["results_usable"] else None
            row += ([_query._number(found.total), _query._number(found.penalty), found.largest]
                    if found is not None else [None, None, None])
        return row

    def note(k: int) -> str | None:
        if offset == 0 and k == n:
            return None
        if k == 0:
            return f"no run fits in one result from offset {offset}; {PAGE_HINT}"
        if latest:
            text = (f"the latest {k} of {n} runs, newest first" if offset == 0 else
                    f"runs {offset + 1}–{offset + k} of {n}, newest first")
            if offset + k < n:
                text += (f"; pass offset {offset + k} for the next (a run that starts or finishes moves to the "
                         "top), or all true for every run")
            return text
        text = f"runs {offset + 1}–{offset + k} of {n}, newest first"
        if offset + k < n:
            text += (f"; pass offset {offset + k} for the next (with the same filters; a run that starts or "
                     f"finishes moves to the top), or {PAGE_HINT}")
        return text

    def build(k: int) -> dict:
        sent = page[:k]
        payload = dict(header)
        older = _state_counts(selected[offset + k:])
        if older:
            payload["older"] = older
        if full:
            payload["runs"] = full_rows[:k]
        else:
            payload["columns"] = SCAN_COLUMNS if scan else COMPACT_COLUMNS
            payload["runs"] = [compact_row(r) for r in sent]
        if scan:
            usable = [r for r in sent if r.entry["results_usable"]]
            with_costs = [r for r in usable if cost_of(r) is not None]
            units = list(dict.fromkeys(cost_of(r).unit for r in with_costs))
            if units:
                payload["cost_unit"] = _overview_mod.cost_unit_text(units[0], scan=True)
            if len(units) > 1:
                payload["cost_units_differ"] = _names([u or "not recorded" for u in units])
            missing = [r.folder.name for r in usable if cost_of(r) is None]
            if missing:
                payload["no_cost_table"] = _names(missing)
            stale = [r.folder.name for r in with_costs if r.entry["stale_vs_input"]]
            if stale:
                payload["input_changed_in"] = _names(stale)
            versions = [r.folder.name for r in with_costs if r.entry["other_flextool_version"]]
            if versions:
                payload["other_flextool_version_in"] = _names(versions)
        if comparison is not None:
            payload["comparison"] = comparison
        text = note(k)
        if text is not None:
            payload["shown"] = text
        return payload

    fitted = fit_page(build, len(page), ctx.caps.max_chars, ctx.redact)
    return Fit(fitted, rows_key="runs", what="runs", hint=f"pass offset or {PAGE_HINT}")


def _overview(ctx: AssistantContext, args: dict) -> Fit:
    """One run's overview (``detail: "overview"``): ``folders`` with one exact name, or ``scenario`` alone (its
    newest run folder)."""
    folders = [str(f) for f in args.get("folders") or []]
    scenario = args.get("scenario")
    one = len(folders) == 1 and "*" not in folders[0]
    if not one and not (not folders and scenario is not None):
        _refuse_overview(ctx)
    selected = _runs.select_runs(ctx, source=args.get("source"), scenario=scenario, state=args.get("state"),
                                 results_usable=args.get("results_usable"), folders=folders or None)
    if not selected:
        _refuse_overview(ctx)
    run = selected[0]
    entry = run.entry
    row = _full_row(run)
    row.pop("folder", None)
    if not entry["results_usable"]:
        return Fit({"run": run.folder.name, **row, "numbers": _overview_mod.NOT_USABLE})
    head: dict[str, Any] = {"run": run.folder.name}
    for key in ("state", "verdict", "finished_at", "results_usable", "verified"):
        if key in row:
            head[key] = row[key]
    run_notes = _runs.run_notes(run)
    if run_notes:
        head["run_notes"] = run_notes
    if not one and len(selected) > 1:
        head["notes"] = [f"The scenario also has the result folders {_names([r.folder.name for r in selected[1:]])}; "
                         "this is the newest. Name one in folders for its overview."]
    payload = _overview_mod.overview_numbers(run.folder, ctx.caps.max_results, ctx.caps.max_chars, head, ctx.redact)
    return Fit(payload)


def _refuse_overview(ctx: AssistantContext) -> None:
    latest = [r.folder.name for r in _runs.newest_first(_runs.runs(ctx))][:OVERVIEW_LATEST]
    if not latest:
        raise ToolError("no_runs")
    raise ToolError("overview_one_run", runs=latest)


def _prepare_runs(args: dict) -> dict:
    """One folder name where the schema has a list."""
    if isinstance(args.get("folders"), str):
        args = dict(args)
        args["folders"] = [args["folders"]]
    return args


TOOLS = [
    make_tool(
        name="list_sources", title="Input sources of the project",
        description=("The input sources of the project (spreadsheets and databases) with their number (the "
                     "suffix of their result folders), scenarios, database version and whether it is current, the "
                     "state of a spreadsheet's conversion, and notices of FlexTool behaviour changes. scenario: its "
                     "alternatives, highest rank first."),
        properties={"scenario": {"type": "string"}}, handler=list_sources, area="discovery", small_tier=True,
        basic=BasicForm(
            description=("The input sources (spreadsheets, databases): number (suffix of their result folders), "
                         "scenarios, database version, conversion state, behaviour notices. scenario: its "
                         "alternatives, highest rank first."),
            properties=("scenario",))),
    make_tool(
        name="list_runs", title="Scenario runs of the project",
        description=("Result folders (scenario + source number, as S2_Dry_1), newest first. Without filters: the "
                     "latest runs, with counts of the older ones by state; all true: every run. Rows: folder, "
                     "state, finished, results_usable (no_record: success unknown). detail scan adds total_cost "
                     "(discounted, what the model minimises), penalty_cost and the largest cost: which runs look "
                     "wrong. detail overview (one folder, or scenario): verdict, costs by category, capacity per "
                     "entity (existing, invested, total), slacks. detail full (also named folders, up to 5 runs): "
                     "verdict, times, version, error. Filters: source, scenario, state, results_usable, folders "
                     "(names, * patterns). offset pages."),
        properties={"source": {"type": "string"}, "scenario": {"type": "string"}, "state": {"type": "string"},
                    "results_usable": {"type": "boolean"},
                    "folders": {"type": "array", "items": {"type": "string"}},
                    "detail": {"type": "string", "enum": ["scan", "overview", "full"]},
                    "all": {"type": "boolean"}, "offset": {"type": "integer"}},
        handler=list_runs, area="discovery", small_tier=True, prepare=_prepare_runs,
        basic=BasicForm(
            description=("Result folders (scenario + source number, as S2_Dry_1), newest first: the latest runs, "
                         "with counts of the older ones; all true: every run. folder, state, finished, "
                         "results_usable (no_record: success unknown). detail scan: total cost, penalty cost and "
                         "largest cost of each; overview (one folder): costs, capacity, slacks; full: verdict, "
                         "times, error. folders: names or * patterns. offset pages."),
            properties=("scenario", "folders", "detail", "all", "offset"))),
]
