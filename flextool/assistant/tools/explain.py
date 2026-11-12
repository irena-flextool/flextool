"""explain_entity: one unit, connection or node of one run in one call, its inputs beside its results with the
hints the two give together (item 209; the readers are ``results._entity``).

The run is named (``run``, refused as query_result refuses it) or is the newest run of ``scenario``
(``_runs.select_runs``, then ``_runs.resolve_run`` on it: a scenario whose latest run did not succeed is refused,
never answered from an earlier run).  The inputs are those of the run's own input source and scenario
(``_runs.run_inputs``).  Standard and advanced levels only.
"""
from __future__ import annotations

from typing import Any

from flextool.assistant import project
from flextool.assistant.caps import CapArg, dumps
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, close_names, name_list, render, similar_text
from flextool.assistant.inputs import _values
from flextool.assistant.registry import Fit, make_tool
from flextool.assistant.results import _entity, _runs
from flextool.assistant.tools.inputs import entity_value, match_names, unit_of
from flextool.assistant.unit_wording import (
    STORED_VALUES,
    UNITS_CONVENTION,
    add_session_notes,
    delivered_notes,
    is_convention_unit,
)

#: The latest folders a refusal without run and scenario names.
LATEST = 5
NAMES_MAX = 10
INPUT_COLUMNS = ["parameter", "value", "unit", "from_alternative"]
#: The advice of a result call_tool still has to cut (a max_chars below the last cut step's text).
FIT_HINT = "ask with a larger max_chars"
#: The input parameters the hints read (the scenario's value, or the default).
HINT_PARAMETERS = ("invest_method", "invest_max_total", "invest_max_period", "invest_min_total", "invest_min_period",
                   "retire_max_total", "retire_max_period", "cumulative_max_capacity")


# ---------------------------------------------------------------------------------------------------------
# The run


def _select_run(ctx: AssistantContext, args: dict) -> tuple[_runs.ResolvedRun, list[str]]:
    run, scenario, source = args.get("run"), args.get("scenario"), args.get("source")
    every = _runs.runs(ctx)             # one scan of the run folders for the whole call
    if run is not None:
        rr = _runs.resolve_run(ctx, str(run), every)
        own = rr.entry["db_scenario"]
        if scenario is not None and own != scenario:
            raise ToolError("explain_other_scenario", run=rr.folder.name, value=str(own), scenario=scenario)
        if source is not None:
            wanted = project.resolve_source(ctx, str(source))
            if rr.entry["source_number"] != wanted.number:
                raise ToolError("explain_other_source", run=rr.folder.name, source=wanted.name)
        return rr, []
    if scenario is None:
        latest = [r.folder.name for r in _runs.newest_first(every)][:LATEST]
        if not latest:
            raise ToolError("no_runs")
        raise ToolError("explain_run_or_scenario", runs=latest)
    selected = _runs.select_runs(ctx, source=source, scenario=str(scenario), all_runs=every)
    if not selected:
        known = sorted({r.entry["db_scenario"] for r in every if r.entry["db_scenario"] is not None})
        if scenario not in known:
            raise ToolError("overview_no_scenario_run", scenario=scenario,
                            scenarios=name_list(close_names(str(scenario), known) or known, NAMES_MAX))
        folders = [r.folder.name for r in _runs.newest_first(every) if r.entry["db_scenario"] == scenario]
        raise ToolError("explain_no_source_run", scenario=scenario, source=project.resolve_source(ctx, source).name,
                        runs=name_list(folders, NAMES_MAX))
    # The newest folder of the scenario, refused word for word as query_result refuses it (O1: never an older run).
    rr = _runs.resolve_run(ctx, selected[0].folder.name, every)
    notes = []
    if len(selected) > 1:
        others = name_list([r.folder.name for r in selected[1:]], NAMES_MAX)
        notes.append(_entity.TEXTS["several_runs"].format(folders=others))
    return rr, notes


# ---------------------------------------------------------------------------------------------------------
# The entity


def _entity_class(rr: _runs.ResolvedRun, db, source: str, name: str, wanted: str | None) -> str:
    """The entity's class: the one given, else the only one of unit, connection and node that has the name (in
    the input data, or in the run's results when the input data cannot be read)."""
    if db is not None:
        found = [cls for cls in _entity.CLASSES if db.entity_by_name(cls, name) is not None]
        if wanted is not None:
            if wanted not in found:
                candidates = sorted(e.name for e in db.entities.get(wanted, {}).values())
                raise ToolError("unknown_entity", entity_class=wanted, entity=name, source=source,
                                similar=similar_text(name, candidates))
            return wanted
        if len(found) == 1:
            return found[0]
        if found:
            raise ToolError("explain_several_classes", entity=name, names=found, source=source)
        if rr.entry.get("stale_vs_input"):
            # The input data changed after the run and no longer has the entity: the run's results tell its
            # class (the "input data changed" note of run_inputs says the inputs are the current data's).
            in_results = _entity.classes_in_results(rr.folder, name)
            if len(in_results) == 1:
                return in_results[0]
            if in_results:
                raise ToolError("explain_several_classes", entity=name, names=in_results, source=source)
        other = sorted({e.entity_class for e in match_names(db, name)})
        if other:
            raise ToolError("explain_not_explained", entity=name, names=other, source=source)
        everything = sorted({e.name for cls in _entity.CLASSES for e in db.entities.get(cls, {}).values()})
        raise ToolError("entity_not_found", entity=name, source=source, similar=similar_text(name, everything))
    if wanted is not None:
        return wanted
    found = _entity.classes_in_results(rr.folder, name)
    if len(found) == 1:
        return found[0]
    if found:
        raise ToolError("explain_several_classes", entity=name, names=found, source=source)
    raise ToolError("explain_not_in_results", run=rr.folder.name, entity=name)


def _value_rows(db, view, cls: str, byname: tuple[str, ...]) -> list[tuple[str, Any, str | None, str]]:
    """``(parameter, parsed value, unit, from)`` of every parameter the scenario sets for the entity, by name; for
    the entity itself also :data:`~flextool.assistant.results._entity.ALWAYS_SHOWN` (``from``: ``"default"`` or
    ``"not set"`` when no alternative sets them)."""
    rows = []
    for parameter in db.parameters_of(cls, byname):
        item = view.value(cls, byname, parameter)
        if item is not None:
            rows.append((parameter, item.parsed(), unit_of(view, cls, parameter, byname)[0], item.alternative))
    if len(byname) == 1:
        set_here = {r[0] for r in rows}
        for parameter in _entity.ALWAYS_SHOWN:
            definition = db.definition(cls, parameter)
            if parameter in set_here or definition is None:
                continue
            default = definition.default()
            rows.append((parameter, default, unit_of(view, cls, parameter, byname)[0],
                         _entity.DEFAULT if default is not None else _entity.NOT_SET))
    return sorted(rows, key=lambda r: r[0])


def _hint_inputs(db, view, cls: str, byname: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for parameter in HINT_PARAMETERS:
        item = view.value(cls, byname, parameter)
        if item is not None:
            out[parameter] = item.parsed()
        else:
            definition = db.definition(cls, parameter)
            out[parameter] = definition.default() if definition is not None else None
    return out


def _shown_value(value: Any, summarised: bool) -> Any:
    return _values.compact(value) if summarised else entity_value(value)


def _investment_relevant(parameter: str) -> bool:
    return parameter.startswith(_entity.INVESTMENT_PREFIXES)


class _Inputs:
    """The entity's inputs, read once and shown at each cut step."""

    def __init__(self, db, view, cls: str, name: str) -> None:
        self.byname = (name,)
        self.rows = _value_rows(db, view, cls, self.byname)
        self.hint_inputs = _hint_inputs(db, view, cls, self.byname)
        self.links: list[tuple[str, tuple[str, ...], list]] = []
        self.memberships: dict[str, list[list[str]]] = {}
        self.groups: list[str] = []
        self.nodes: list[str] = []
        inputs: list[str] = []
        for rel in sorted(db.relationships_of(cls, name), key=lambda r: (r.entity_class, r.byname)):
            if not view.active(rel.entity_class, rel.byname)[0]:
                continue
            if cls == "node" and not rel.entity_class.startswith(_entity.NODE_LINK_PREFIXES):
                continue            # a node's flow links are its feeders': not listed (orchestrator Q5)
            if rel.entity_class in _entity.LINK_CLASSES:
                self.links.append((rel.entity_class, rel.byname, _value_rows(db, view, rel.entity_class,
                                                                             rel.byname)))
            else:
                self.memberships.setdefault(rel.entity_class, []).append(list(rel.byname))
            if rel.entity_class == f"group__{cls}":
                self.groups.append(rel.byname[0])
            if rel.entity_class == "unit__outputNode":
                self.nodes.append(rel.byname[1])
            elif rel.entity_class == "unit__inputNode":
                inputs.append(rel.byname[1])
            elif rel.entity_class == "connection__node__node":
                self.nodes += list(rel.byname[1:])
        # The nodes whose prices are shown: a unit's output nodes, then its input nodes.
        self.nodes = list(dict.fromkeys([*self.nodes, *inputs]))

    @property
    def units(self) -> list[str]:
        return [r[2] for r in self.rows if r[2]] + [r[2] for _, _, rows in self.links for r in rows if r[2]]

    def payload(self, cuts: set[str]) -> dict[str, Any]:
        summarised = "maps" in cuts
        relevant_only = "inputs" in cuts
        rows = [[p, _shown_value(v, summarised), u, f] for p, v, u, f in self.rows
                if not relevant_only or _investment_relevant(p)]
        out: dict[str, Any] = {"inputs": {"columns": INPUT_COLUMNS, "rows": rows}}
        if self.links:
            out["links"] = []
            for cls, byname, rows in self.links:
                link: dict[str, Any] = {"entity_class": cls, "entity": list(byname)}
                if rows and not relevant_only:
                    link["parameters"] = [[p, _shown_value(v, summarised), u, f] for p, v, u, f in rows]
                out["links"].append(link)
        if self.memberships:
            out["memberships"] = self.memberships
        return out


# ---------------------------------------------------------------------------------------------------------
# The tool


def explain_entity(ctx: AssistantContext, args: dict) -> Fit:
    rr, notes = _select_run(ctx, args)
    db, view, input_note = _runs.run_inputs(ctx, rr, not_shown="its inputs are not shown",
                                            current="the inputs are those of the current data")
    if input_note:
        notes.append(input_note)
    source = rr.run.source
    source_name = source.name if source is not None else rr.entry["source_name"]
    name = str(args["entity"])
    cls = _entity_class(rr, db, source_name or rr.folder.name, name, args.get("entity_class"))
    inputs = None
    description = None
    if view is not None and db.entity_by_name(cls, name) is not None:
        item = db.entity_by_name(cls, name)
        description = item.description.strip() if item.description and item.description.strip() else None
        active, reason = view.active(cls, item.byname)
        if not active:
            notes.append(render("inactive_entity", {"entity": name, "scenario": view.scenario, "reason": reason},
                                ctx.redact))
        inputs = _Inputs(db, view, cls, name)

    blocks, missing, large = _entity.read_results(rr.folder, cls, name)
    nodes = list(inputs.nodes) if inputs is not None else []
    if cls == "node":
        nodes = [name]
    for key in ("output_flows", "input_flows", "flow_to_first_node", "flow_to_second_node"):
        if key in blocks:
            nodes += [row[1] for row in blocks[key]["rows"]]
    prices = _entity.read_prices(rr.folder, list(dict.fromkeys(nodes)))
    if isinstance(prices, _entity.TooLarge):
        large.append(prices)
    elif prices is not None:
        blocks["prices"] = prices
    elif nodes:
        missing.append(_entity.PRICES_TABLE)
    if "costs" in blocks:
        blocks["costs"]["note"] = _entity.TEXTS["costs_note"].format(table=_entity.ANNUALIZED_COSTS[cls])
    hints = _entity.hints(inputs.hint_inputs if inputs is not None else {}, blocks,
                          inputs.groups if inputs is not None else [])
    if any(key in blocks for key in _entity.ANNUALIZED_BLOCKS):
        notes.append(_entity.TEXTS["annualized_note"])
    if missing:
        notes.append(_entity.TEXTS["no_values"].format(entity=name))
    too_large = [t.table for t in large]
    if too_large:
        notes.append(_entity.TEXTS["too_large"])
    related = _entity.related_tables(rr.folder, cls)

    units = [b.get("unit") for b in blocks.values() if b.get("unit")]
    units += [row[-1] for b in blocks.values() if "unit" in b.get("columns", []) for row in b["rows"] if row[-1]]
    units += inputs.units if inputs is not None else []
    keys = ([UNITS_CONVENTION] if any(is_convention_unit(u) for u in units) else [])
    keys += [STORED_VALUES] if inputs is not None else []
    # Worded against a copy of the notes in flight: only the payload that is sent records its notes (below).
    session_notes = add_session_notes({}, keys, ctx.notes_sent, set(ctx.notes_in_flight))["notes"]

    head: dict[str, Any] = {"run": rr.folder.name, "entity": name, "entity_class": cls, "source": source_name,
                            "scenario": rr.entry["db_scenario"]}
    if description:
        head["description"] = description
    if not rr.verified:
        head["verified"] = False
    max_rows = int(args["max_rows"])
    left_out = (["inputs and links"] if inputs is not None else []) + [
        f"{key} ({block['table']})" for key, block in blocks.items() if key != "capacity"]
    left_out += [f"too_large ({', '.join(too_large)})"] if too_large else []
    left_out += [f"not_in_run ({', '.join(missing)})"] if missing else []

    def build(step: str, keep: int | None) -> dict:
        cuts = _entity.steps_from(step)
        out = dict(head)
        shown = _entity.cut_blocks(blocks, cuts, max_rows)
        step_notes = list(notes)
        if "periods" in cuts:
            step_notes.append(_entity.TEXTS["periods_cut"])
        if step in _entity.FINAL_STEPS:
            if step == "final":
                step_notes.append(_entity.TEXTS["final_cut"].format(blocks=", ".join(left_out) or "nothing"))
                if "capacity" in shown:
                    out["capacity"] = shown["capacity"]
                out["hints"] = hints
                out["notes"] = step_notes + session_notes
                if rr.run_notes:
                    out["run_notes"] = rr.run_notes
                return out
            # The later final steps send only the note on what was left out (the other notes named in it).
            parts = list(left_out)
            if step_notes or session_notes or rr.run_notes:
                parts.append("the other notes")
            if step in ("final_total", "final_hints"):
                if "capacity" in shown:
                    out["capacity"] = _entity.capacity_total(shown["capacity"])
                out["hints"] = _entity.capped_hints(hints, keep)
            else:
                parts += ([f"capacity ({blocks['capacity']['table']})"] if "capacity" in blocks else []) + (
                    ["hints"] if hints else [])
            out["notes"] = [_entity.TEXTS["final_cut_short"].format(n=len(parts)) if step == "final_count"
                            else _entity.TEXTS["final_cut"].format(blocks=", ".join(parts) or "nothing")]
            return out
        out["hints"] = hints
        if inputs is not None:
            out.update(inputs.payload(cuts))
            if "inputs" in cuts:
                step_notes.append(_entity.TEXTS["inputs_cut"])
        out.update(shown)
        if missing:
            out["not_in_run"] = missing
        if too_large:
            out["too_large"] = too_large
        if related and "related" not in cuts:
            out["related_tables"] = related
        out["notes"] = step_notes + session_notes
        if rr.run_notes:
            out["run_notes"] = rr.run_notes
        return out

    payload, _step = _entity.fitted(build, len(hints), int(args["max_chars"]), ctx.redact)
    # The session notes the sent payload carries in full are in flight for the rest of the request.
    ctx.notes_in_flight |= delivered_notes([dumps(payload)], ctx.redact) & set(keys)
    # The payload fits (the final steps shrink until it does); a limit below even the last step's is cut by
    # call_tool, whose note then says to ask for more characters (never query_result's filters advice).
    return Fit(payload, hint=FIT_HINT)


TOOLS = [
    make_tool(
        name="explain_entity", title="One entity's inputs and results in one run",
        description=("One unit, connection or node in one run: its inputs in the run's scenario, and per period "
                     "with table and unit its capacity (existing, invested, divested, total), flows per connected "
                     "node, capacity factor, curtailment, storage state, prices at its nodes (mean, min, max), "
                     "non-zero discounted costs and investment value; hints name the input a result is consistent "
                     "with (an investment limit). run: a result folder from list_runs; or scenario: its newest run "
                     "(source picks the input source). entity_class only for a name of several classes. max_rows "
                     "(default 20) caps each part's rows."),
        properties={"entity": {"type": "string"},
                    "entity_class": {"type": "string", "enum": list(_entity.CLASSES)},
                    "run": {"type": "string"}, "scenario": {"type": "string"}, "source": {"type": "string"},
                    "max_rows": {"type": "integer"}, "max_chars": {"type": "integer"}},
        required=("entity",), handler=explain_entity, area="results", small_tier=False,
        caps={"max_rows": CapArg("max_rows", default=20), "max_chars": CapArg("max_chars")}),
]
