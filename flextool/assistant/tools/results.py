"""Result tools: which tables there are, what one holds, its values filtered and aggregated, and the same values
side by side for several runs.

Results are read only from a run whose latest run succeeded (``_runs.resolve_run``, O1).  Tables are read with
the wide-first query of ``results._query``; every unit is worded by ``unit_wording`` (O2).
A comparison of the runs of the comparison files reads those files only while they were built from the current
successful runs (``_runs.comparison_state``, and the file itself: its scenario level is the member set and it is
not older than the comparison); otherwise every member's own folder (P0 §5.6).
"""
from __future__ import annotations

from typing import Any

from flextool.assistant.caps import CapArg, dumps
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError
from flextool.assistant.registry import Fit, make_tool
from flextool.assistant.results import _catalog, _query, _runs
from flextool.assistant.results._query import AGGS, SCENARIO_LEVEL, Query
from flextool.assistant.unit_wording import UNITS_CONVENTION, add_session_notes, is_energy_unit, shown_unit

QUERY_HINT = "add filters, group_by or top_n"
COMPARISON = "comparison"
COMPARISON_MISMATCH = ("The comparison files do not match the current scenario runs; using each scenario's own "
                       "results instead.")
SUMMARY_NOTE = ("One summary per series and period. To see the values at each time step, keep time in group_by "
                "(and narrow with filters or time_range).")


# ---------------------------------------------------------------------------------------------------------
# Shared helpers


def _shown(unit: str | None) -> str | None:
    return shown_unit(unit) if unit else None


def _unit_payload(units: dict[str, str]) -> Any:
    """One unit, or ``{column label: unit}`` for a table of several units."""
    if "*" in units:
        return _shown(units["*"])
    return {label: _shown(u) for label, u in units.items()}


def _semantics_payload(semantics: set[str]) -> Any:
    return next(iter(semantics)) if len(semantics) == 1 else sorted(semantics)


def _with_units_note(ctx: AssistantContext, payload: dict) -> dict:
    return add_session_notes(payload, [UNITS_CONVENTION], ctx.notes_sent, ctx.notes_in_flight)


def _record_units(rec: dict) -> tuple[Any, Any]:
    cols = rec["columns"]
    if not cols:
        return None, None
    if list(cols) == ["*"]:
        return _shown(cols["*"]["unit"]), cols["*"]["semantics"]
    return ({c: _shown(v["unit"]) for c, v in cols.items()}, {c: v["semantics"] for c, v in cols.items()})


def _verified(payload: dict, rr: _runs.ResolvedRun) -> dict:
    """*payload* with ``verified: false`` when the run has no run record (only then)."""
    if not rr.verified:
        payload["verified"] = False
    return payload


def _top_n(args: dict, ceiling: int) -> dict:
    if args.get("top_n") is not None:
        args = dict(args)
        args["top_n"] = max(1, min(int(args["top_n"]), ceiling))
    return args


def _applied(q: Query, result: _query.Result) -> dict:
    return {
        "mode": result.mode,
        "filters": q.filters or None,
        "time_range": list(q.time_range) if q.time_range else None,
        "group_by": result.kept if result.mode == "aggregate" else None,
        "agg": result.agg,
        "combined_over": result.over or None,
        "how": result.how,
        "sort": {"by": q.sort_by, "descending": q.descending} if q.sort_by else None,
        "top_n": q.top_n,
    }


def _result_notes(result: _query.Result) -> list[str]:
    notes = list(result.notes)
    if result.mode == "summary":
        notes.insert(0, SUMMARY_NOTE)
    if result.empty_dropped:
        what = "series without a value" if result.mode == "summary" else "empty cells (no value in the run)"
        notes.append(f"{result.empty_dropped} {what} are left out.")
    return notes


# ---------------------------------------------------------------------------------------------------------
# find_result_tables, list_result_tables, describe_result_table


def find_result_tables(ctx: AssistantContext, args: dict) -> Fit:
    hits = []
    for rec in _catalog.search(str(args["query"]), int(args["max_results"])):
        unit, semantics = _record_units(rec)
        hits.append({"table": rec["stem"], "topic": rec["topic"], "description": rec["description"], "unit": unit,
                     "semantics": semantics, "dims": rec["dims"], "counterpart": rec["counterpart"]})
    payload = {"tables": hits}
    if not hits:
        payload["note"] = "No result table matches; try other words, or list_result_tables for a run."
    return Fit(_with_units_note(ctx, payload), rows_key="tables", what="tables", hint="ask more precisely")


def _table_unit(table: _query.Table) -> Any:
    units = {m.unit for m in table.measures}
    if not units:
        return None
    if len(units) == 1:
        return _shown(units.pop())
    return sorted(_shown(u) or "no unit" for u in units)


def list_result_tables(ctx: AssistantContext, args: dict) -> dict:
    rr = _runs.resolve_run(ctx, str(args["run"]))
    by_topic: dict[str, list[dict]] = {}
    for stem in _query.table_stems(rr.folder):
        topic = _catalog.topic_of(stem)
        try:
            table = _query.open_table(rr.folder, stem, where=rr.folder.name)
            entry = {"table": stem, "rows": table.layout.num_rows, "unit": _table_unit(table)}
        except ToolError:
            entry = {"table": stem, "rows": None, "unit": None, "problem": "being written or damaged"}
        by_topic.setdefault(topic, []).append(entry)
    order = list(dict.fromkeys([r["topic"] for r in _catalog.records()] + [_catalog.OTHER_TOPIC]))
    topics = sorted(by_topic, key=lambda t: order.index(t) if t in order else len(order))
    wanted = args.get("topic")
    if wanted is not None:
        match = [t for t in topics if t.lower() == str(wanted).lower()]
        if not match:
            raise ToolError("unknown_topic", topic=wanted, run=rr.folder.name, topics=list(topics))
        topics = match
    total = sum(len(by_topic[t]) for t in topics)
    payload: dict[str, Any] = {"run": rr.folder.name, "topics": {}, "total": total, "truncated": None,
                               "run_notes": rr.run_notes}
    budget = int(args["max_chars"]) - 400
    shown = 0
    for t in topics:
        trial = dict(payload["topics"])
        trial[t] = by_topic[t]
        if len(dumps({**payload, "topics": trial})) > budget and payload["topics"]:
            break
        payload["topics"] = trial
        shown += len(by_topic[t])
    if shown < total:
        payload["truncated"] = f"showing {shown} of {total} tables — name a topic"
    return _with_units_note(ctx, _verified(payload, rr))


def describe_result_table(ctx: AssistantContext, args: dict) -> dict:
    rr = _runs.resolve_run(ctx, str(args["run"]))
    table = _query.open_table(rr.folder, str(args["table"]), where=rr.folder.name)
    limit = int(args["max_values"])
    layout = table.layout
    index = _query.read_wide(table, [])
    levels: dict[str, dict] = {}
    for level in table.row_levels:
        values = index[level].unique(maintain_order=True).to_list()
        levels[level] = {"axis": "row", "distinct": len(values), "first": values[:limit]}
    for level in table.column_levels:
        values = layout.level_values(level)
        levels[level] = {"axis": "column", "distinct": len(values), "first": values[:limit]}
    time = None
    if "time" in levels:
        times = index["time"].unique(maintain_order=True).to_list()
        time = {"steps": len(times), "first": times[0] if times else None, "last": times[-1] if times else None}
    periods = levels["period"]["first"] if "period" in levels else None
    units = _query._units(table, list(layout.physical)) if layout.physical else {}
    dims = _catalog.record(table.stem)
    payload: dict[str, Any] = {
        "run": rr.folder.name,
        "table": table.stem,
        "description": _catalog.description(table.stem),
        "topic": _catalog.topic_of(table.stem),
        "dims": dims["dims"] if dims else {"rows": table.row_levels, "columns": table.column_levels},
        "levels": levels,
        "time": time,
        "periods": periods,
        "unit": _unit_payload(units) if units else None,
        "semantics": _semantics_payload({m.semantics for m in table.measures}) if table.measures else None,
        "rows": layout.num_rows,
        "data_columns": len(layout.physical),
        "counterpart": _catalog.counterpart(table.stem),
        "run_notes": rr.run_notes,
    }
    if table.membership:
        payload["note"] = "This table is a set of names with no values."
    return _with_units_note(ctx, _verified(payload, rr))


# ---------------------------------------------------------------------------------------------------------
# query_result


def query_result(ctx: AssistantContext, args: dict) -> Fit:
    args = _top_n(args, int(args["max_rows"]))
    rr = _runs.resolve_run(ctx, str(args["run"]))
    table = _query.open_table(rr.folder, str(args["table"]), where=rr.folder.name)
    q = _query.parse_query(args)
    result = _query.run_query(table, q)
    frame = _query.ordered(result, q)
    max_rows = int(args["max_rows"])
    total = frame.height
    payload: dict[str, Any] = {
        "run": rr.folder.name,
        "table": table.stem,
        "columns": frame.columns,
        "rows": _query.rows_of(frame, max_rows),
        "unit": _unit_payload(result.units) if result.units else None,
        "semantics": _semantics_payload(result.semantics) if result.semantics else None,
        "applied": _applied(q, result),
        "total_rows": total,
        "truncated": f"showing {max_rows} of {total} rows — {QUERY_HINT}" if total > max_rows else None,
        "notes": _result_notes(result),
        "run_notes": rr.run_notes,
    }
    _verified(payload, rr)
    if result.units:
        _with_units_note(ctx, payload)
    return Fit(payload, rows_key="rows", hint=QUERY_HINT)


# ---------------------------------------------------------------------------------------------------------
# compare_runs


def _comparison_source(ctx: AssistantContext, state: dict, stem: str) -> tuple[_query.Table | None, str | None]:
    """The comparison file of *stem* when it may be used (P0 §5.6), else ``(None, why not)``: the metadata says
    every member is fresh, and the table has the members as its scenario level and the metadata's ``write_id``
    in its footer (it was written by the write the metadata describes)."""
    if not state["usable"]:
        return None, state.get("note") or COMPARISON_MISMATCH
    folder = ctx.project_path / _runs.COMPARISON_DIR
    if not (folder / f"{stem}.parquet").is_file():
        return None, (f"The comparison files have no table {stem}; using each scenario's own results "
                      "instead.")
    table = _query.open_table(folder, stem, where="the comparison files")
    members = {m["folder"] for m in state["members"]}
    if SCENARIO_LEVEL not in table.layout.col or set(table.layout.level_values(SCENARIO_LEVEL)) != members:
        return None, COMPARISON_MISMATCH
    if table.layout.write_id is None or table.layout.write_id != state.get("write_id"):
        return None, COMPARISON_MISMATCH
    return table, None


def _preloaded(table: _query.Table, q: _query.Query, write_id: str) -> _query.Table | None:
    """*table* read once for all members (:func:`_query.preload`); ``None`` when the file was rewritten while it
    was read (its footer no longer carries *write_id*)."""
    from flextool.lean_parquet import read_lean_parquet_write_id

    loaded = _query.preload(table, q)
    try:
        same = read_lean_parquet_write_id(table.path) == write_id
    except Exception:  # noqa: BLE001 — OSError, or pyarrow's error for a file being written
        same = False
    return loaded if same else None


#: Entity levels whose values are nodes (or carriers): power and energy are never added up over them in the
#: overview, since the nodes of a model can hold different carriers (Cycle 1 review M3).
NODE_LEVELS = frozenset({"node", "sink", "source", "commodity"})
#: The refusals of an aggregation the overview falls back from (the others, such as a filter that matches nothing,
#: are the caller's to see).
_AGGREGATION_REFUSALS = frozenset({"mixed_units", "mean_of_parts", "combine_quantities", "sum_over_groups"})


def _aggregation_refusal(exc: ToolError) -> bool:
    return exc.key in _AGGREGATION_REFUSALS or exc.key.startswith("sum_")


def _overview(table: _query.Table, q: Query) -> tuple[Query, list[str]]:
    """The query of compare_runs' overview (no group_by, agg, sort, top_n and by_difference): every period,
    category, quantity and group level kept; time combined; entity levels combined where a sum over entities is
    meaningful for every kept column (prices and ratios keep them); periods combined only where a sum over periods
    is (discounted costs).  Power and energy (:func:`~flextool.assistant.unit_wording.is_energy_unit`) keep every
    node level (:data:`NODE_LEVELS`): the nodes may hold different carriers; their units and connections are
    added up into each node, or into each group, and a table with neither keeps its units and connections too.
    agg: on a table with time steps the sum when every column's default statistic is the sum, else the mean (the
    mean of the entities' total at each step); else the sum.  Returns the query and the notes (Cycle 1 design §3,
    critique m6, review M3)."""
    import dataclasses

    stem = table.stem
    keep = _query._selected_columns(table, q, None)
    measure_of = table.measure_of()
    semantics = {measure_of[p].semantics for p in keep}
    energy = any(is_energy_unit(measure_of[p].unit) for p in keep)
    kinds = {level: _query.level_kind(stem, level) for level in table.levels}
    anchored = any(level in NODE_LEVELS or kind == "overlapping" for level, kind in kinds.items())
    kept: list[str] = []
    notes: list[str] = []
    by_carrier: list[str] = []
    for level in table.levels:
        kind = kinds[level]
        if kind == "time":
            continue
        if kind == "period" and all(_query.sum_problem(stem, sem, {"period"}) is None for sem in semantics):
            continue
        if kind == "entity":
            if not all(_query.sum_problem(stem, sem, {"entity"}) is None for sem in semantics):
                notes.append(f"The values are not added up over {level} (they are prices or ratios); each {level} "
                             "is shown.")
            elif energy and (level in NODE_LEVELS or not anchored):
                by_carrier.append(level)
            else:
                continue
        kept.append(level)
    if by_carrier:
        levels = ", ".join(by_carrier)
        notes.append(f"Energy and power are not added up over {levels}: a model's nodes may hold different "
                     f"carriers (electricity, heat, hydrogen); each {levels} is shown.")
    if table.has_time:
        agg = "sum" if all(_query.default_agg(sem) == "sum" for sem in semantics) else "mean"
    else:
        agg = "sum"
    return dataclasses.replace(q, group_by=tuple(kept), agg=agg), notes


def _overview_note(stem: str, levels: list[str], keys: list[str], result: _query.Result) -> str:
    """What the overview computed, in the words of the aggregation (Cycle 1 review M2): its ``how`` when entities
    were added up at each time step before the statistic over time, else the sum over the entity and category
    levels and the statistic over the others."""
    head = f"Totals per {', '.join(keys) or 'run'}"
    tail = " by_difference true gives each series, largest difference first."
    if result.how:
        return f"{head} ({result.how}).{tail}"
    combined = [lv for lv in levels if lv not in keys]
    additive = [lv for lv in combined if _query.level_kind(stem, lv) in ("entity", "part")]
    other = [lv for lv in combined if lv not in additive]
    parts = []
    if additive:
        parts.append(f"sum over {', '.join(additive)}")
    if other:
        parts.append(f"{result.agg} over {', '.join(other)}")
    return f"{head} ({'; '.join(parts) or 'nothing combined'}).{tail}"


#: Room left beside the header and the totals for the rows' ``shown`` note.
_ROWS_NOTE_RESERVE = 200


def _fit_totals(payload: dict, max_chars: int, redact, total: int) -> dict:
    """*payload* with as many leading ``totals`` rows as fit *max_chars* beside the header (the rows are fitted
    after), and ``totals.shown`` when some of the *total* rows are left out (Cycle 1 review m4)."""
    from flextool.assistant.caps import sent_payload

    totals = payload["totals"]
    rows = totals["rows"]

    def trial(k: int) -> dict:
        t = {"columns": totals["columns"], "rows": rows[:k]}
        if k < total:
            t["shown"] = f"totals rows 1–{k} of {total}"
        return {**payload, "totals": t}

    def fits(k: int) -> bool:
        return len(dumps(sent_payload({**trial(k), "rows": []}, redact))) <= max_chars - _ROWS_NOTE_RESERVE

    if fits(len(rows)):
        return trial(len(rows))
    lo, hi, best = 0, len(rows) - 1, 0
    while lo <= hi:
        k = (lo + hi) // 2
        if fits(k):
            best, lo = k, k + 1
        else:
            hi = k - 1
    return trial(best)


def _side_by_side(present: list[tuple], q: Query, *, value_only: bool):
    """The runs' results of *q* joined on their keys, with ``difference_vs_first`` columns: ``(frame, keys,
    run names, difference columns, the first run's result)``."""
    import polars as pl

    frames: list[tuple[str, Any]] = []
    first_result: _query.Result | None = None
    for name, table, scenario, _ in present:
        result = _query.run_query(table, q, scenario=scenario, value_only=value_only)
        if first_result is None:
            first_result = result
        frames.append((name, result.frame.select([*result.keys, pl.col("value").alias(name)])))
    assert first_result is not None
    keys = first_result.keys
    joined = frames[0][1]
    for _, frame in frames[1:]:
        joined = (joined.join(frame, on=keys, how="full", coalesce=True, maintain_order="left_right")
                  if keys else pl.concat([joined, frame], how="horizontal"))
    run_names = [name for name, _ in frames]
    first = run_names[0]
    diff_names: list[str] = []
    for name in run_names[1:]:
        col = "difference_vs_first" if len(run_names) == 2 else f"difference_vs_first:{name}"
        joined = joined.with_columns((pl.col(name) - pl.col(first)).alias(col))
        diff_names.append(col)
    return joined, keys, run_names, diff_names, first_result


def compare_runs(ctx: AssistantContext, args: dict) -> Fit:
    import dataclasses

    import polars as pl

    from flextool.assistant.caps import fit_rows

    args = _top_n(args, int(args["max_rows"]))
    stem = str(args["table"])
    names = [str(r) for r in args["runs"]]
    q = _query.parse_query(args)
    notes: list[str] = []
    source = "run_folders"
    built_from = None
    unverified: list[str] = []
    sources: list[tuple[str, _query.Table | None, str | None, list[str]]] = []
    if names == [COMPARISON]:
        state = _runs.comparison_state(ctx)
        if state is None:
            raise ToolError("no_comparison")
        members = [m["folder"] for m in state["members"]]
        table, why = _comparison_source(ctx, state, stem)
        if table is not None and not table.membership:
            table = _preloaded(table, q, state["write_id"])
            if table is None:
                why = COMPARISON_MISMATCH
        if table is not None:
            source = "comparison_files"
            built_from = [{k: m[k] for k in ("folder", "run_id", "started_at", "finished_at")}
                          for m in state["members"]]
            member_notes = _runs.member_notes(ctx, members)
            sources = [(m, table, m, member_notes[m]) for m in members]
        else:
            notes.append(why)
            names = members
    if not sources:
        if len(dict.fromkeys(names)) < 2:
            raise ToolError("compare_needs_runs")
        for name in dict.fromkeys(names):
            rr = _runs.resolve_run(ctx, name)
            if not rr.verified:
                unverified.append(name)
            path = rr.folder / f"{stem}.parquet"
            table = _query.open_table(rr.folder, stem, where=name) if path.is_file() else None
            sources.append((name, table, None, rr.run_notes))
    present = [s for s in sources if s[1] is not None]
    if not present:
        raise ToolError("table_in_no_run", table=stem, runs=[s[0] for s in sources])
    if present[0][1].membership:
        raise ToolError("membership_table", table=stem)

    overview = (q.group_by is None and q.agg is None and q.sort_by is None and q.top_n is None
                and not args.get("by_difference"))
    totals = None
    totals_n = 0
    if overview:
        oq, overview_notes = _overview(present[0][1], q)
        try:
            joined, keys, run_names, diff_names, first_result = _side_by_side(present, oq, value_only=False)
        except ToolError as exc:
            # A combination the guard refuses (mixed units, the mean of parts): keep the entity levels too, and
            # if that is refused as well, give each series and say why (never fail the call without arguments).
            if not _aggregation_refusal(exc):
                raise
            entity = [lv for lv in present[0][1].levels if _query.level_kind(stem, lv) == "entity"]
            kept = [lv for lv in present[0][1].levels if lv in set(oq.group_by or ()) | set(entity)]
            oq = dataclasses.replace(oq, group_by=tuple(kept))
            try:
                joined, keys, run_names, diff_names, first_result = _side_by_side(present, oq, value_only=False)
            except ToolError as again:
                if not _aggregation_refusal(again):
                    raise
                overview = False
                notes.append(f"Totals are not meaningful for this table ({again.render(ctx.redact)}); each series "
                             "is shown.")
        if overview:
            q_used = oq
            notes.append(_overview_note(stem, list(present[0][1].levels), keys, first_result))
            notes += overview_notes
            parts = [lv for lv in keys if _query.level_kind(stem, lv) == "part"]
            if parts:
                tq = dataclasses.replace(oq, group_by=tuple(lv for lv in keys if lv not in parts))
                try:
                    tj, tkeys, _, _, _ = _side_by_side(present, tq, value_only=False)
                    totals_n = tj.height
                    totals = {"columns": tj.columns, "rows": _query.rows_of(tj, int(args["max_rows"]))}
                except ToolError as exc:
                    if not _aggregation_refusal(exc):
                        raise
                    totals = None
    if not overview:
        q_used = q
        joined, keys, run_names, diff_names, first_result = _side_by_side(present, q, value_only=True)
        if len(first_result.value_columns) > 1:
            notes.append("compare_runs shows the values only; query_result of one run gives the time of each "
                         "maximum.")
    missing: dict[str, Any] = {s[0]: "no such table" for s in sources if s[1] is None}
    for name in run_names:
        n = joined.filter(pl.col(name).is_null()).height
        if n:
            missing[name] = f"{n} keys without a value"
    result = _query.Result(joined, keys, run_names + diff_names, first_result.mode, first_result.agg,
                           first_result.kept, first_result.over, first_result.notes, first_result.semantics,
                           first_result.units, how=first_result.how)
    frame = _query.ordered(result, q_used, default_by=None if overview else (diff_names[0] if diff_names else None))
    max_rows = int(args["max_rows"])
    total = frame.height
    applied = _applied(q_used, first_result)
    if overview:
        applied["mode"] = "overview"
    payload: dict[str, Any] = {
        "table": stem,
        "runs": [s[0] for s in sources],
        "columns": frame.columns,
        "rows": _query.rows_of(frame, max_rows),
        "unit": _unit_payload(first_result.units) if first_result.units else None,
        "semantics": _semantics_payload(first_result.semantics) if first_result.semantics else None,
        "applied": applied,
        "missing": missing or None,
        "source": source,
        "built_from": built_from,
        "total_rows": total,
        "notes": notes + list(first_result.notes),
        "run_notes": {s[0]: s[3] for s in sources if s[3]} or None,
    }
    if totals is not None:
        payload["totals"] = totals
    if unverified:
        payload["unverified"] = unverified
    if first_result.units:
        _with_units_note(ctx, payload)

    def note(k: int, _n: int) -> str | None:
        return None if k == total else f"rows 1–{k} of {total}; {QUERY_HINT}"

    if totals is not None:
        payload = _fit_totals(payload, int(args["max_chars"]), ctx.redact, totals_n)
    fitted = fit_rows(payload, "rows", int(args["max_chars"]), ctx.redact, note_key="shown", note=note)
    return Fit(fitted, rows_key="rows", hint=QUERY_HINT)


# ---------------------------------------------------------------------------------------------------------
# Registry entries

_RUN_TEXT = "run is a result folder name from list_runs (for example S2_Dry_1)."
_FILTERS = {"type": "object", "additionalProperties": {"type": "array", "items": {"type": "string"}}}
_TIME_RANGE = {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
               "required": ["from", "to"], "additionalProperties": False}
_SORT = {"type": "object", "properties": {"by": {"type": "string"}, "descending": {"type": "boolean"}},
         "required": ["by"], "additionalProperties": False}
_QUERY_PROPERTIES = {
    "filters": _FILTERS, "time_range": _TIME_RANGE, "group_by": {"type": "array", "items": {"type": "string"}},
    "agg": {"type": "string", "enum": list(AGGS)}, "sort": _SORT, "top_n": {"type": "integer"},
    "max_rows": {"type": "integer"}, "max_chars": {"type": "integer"},
}
_QUERY_HEAD = (
    "filters keeps values of levels, {\"node\": [\"west\", \"north*\"]} (* matches any text); time_range "
    "{\"from\": …, \"to\": …} keeps time steps. group_by lists the levels to keep; the others are combined with agg "
    "(sum, mean, min, max, count, or time_of_max: the maximum and when it occurs); without group_by, agg combines "
    "over time only. ")
#: query_result's own rule for a call without group_by and agg; compare_runs has its own (critique M2).
_QUERY_SUMMARY = ("Without group_by and agg, a table with time steps gives one summary per series and period "
                  "(mean, min, max, count, time of the maximum, and the sum where it is meaningful). ")
_QUERY_TAIL = (
    "A sum that means nothing "
    "is refused with the reason, for example power summed over time (use the annual table). sort {\"by\": "
    "column, \"descending\": true} and top_n keep the largest (top_n alone sorts by value). max_rows is up to "
    "5000.")
_QUERY_TEXT = _QUERY_HEAD + _QUERY_SUMMARY + _QUERY_TAIL

TOOLS = [
    make_tool(
        name="find_result_tables", title="Which result table answers a question",
        description=("The result tables that best match a question (for example 'unit generation per hour', "
                     "'curtailment', 'costs'): their name, topic, what they hold, unit, the kind of quantity "
                     "(a rate at each time step, an annual total, a level, a price), their levels and the same "
                     "table at the other time resolution. max_results is up to 20 (default 5)."),
        properties={"query": {"type": "string"}, "max_results": {"type": "integer"}},
        required=("query",), handler=find_result_tables, area="results", small_tier=True,
        caps={"max_results": CapArg("max_results", default=5)}),
    make_tool(
        name="list_result_tables", title="Result tables of a run",
        description=("The result tables a run wrote, by topic, with their number of rows and unit. topic keeps "
                     "one topic. " + _RUN_TEXT),
        properties={"run": {"type": "string"}, "topic": {"type": "string"}, "max_chars": {"type": "integer"}},
        required=("run",), handler=list_result_tables, area="results", small_tier=False,
        caps={"max_chars": CapArg("max_chars")}),
    make_tool(
        name="describe_result_table", title="What a result table holds",
        description=("One result table of a run: what it holds, its levels (rows: period, time …; columns: unit, "
                     "node …) with how many values each has and the first ones, its time steps and periods, unit "
                     "and kind of quantity. max_values (default 10, up to 100) is how many values of each level "
                     "are shown. " + _RUN_TEXT),
        properties={"run": {"type": "string"}, "table": {"type": "string"}, "max_values": {"type": "integer"}},
        required=("run", "table"), handler=describe_result_table, area="results", small_tier=False,
        caps={"max_values": CapArg("max_items", maximum=100, default=10)}),
    make_tool(
        name="query_result", title="Values of a result table",
        description=("Values of one result table of a run, filtered and aggregated. " + _QUERY_TEXT + " "
                     + _RUN_TEXT),
        properties={"run": {"type": "string"}, "table": {"type": "string"}, **_QUERY_PROPERTIES},
        required=("run", "table"), handler=query_result, area="results", small_tier=True,
        caps={"max_rows": CapArg("max_rows"), "max_chars": CapArg("max_chars")}),
    make_tool(
        name="compare_runs", title="One result table for several runs",
        description=("The same query of one result table for several runs side by side, with difference_vs_first "
                     "(one per later run). Without group_by, agg, sort and top_n (filters and time_range narrow): "
                     "each run's totals per period and category (entities added up where meaningful); "
                     "by_difference true: each series (with time steps: its mean or sum per period). With "
                     "by_difference, group_by or agg: largest difference first. runs lists folders from "
                     "list_runs, or [\"comparison\"] for the comparison files' runs (each run's own files when "
                     "they do not match). " + _QUERY_HEAD + _QUERY_TAIL),
        properties={"table": {"type": "string"}, "runs": {"type": "array", "items": {"type": "string"}},
                    "by_difference": {"type": "boolean"}, **_QUERY_PROPERTIES},
        required=("table", "runs"), handler=compare_runs, area="results", small_tier=False,
        caps={"max_rows": CapArg("max_rows"), "max_chars": CapArg("max_chars")}),
]
