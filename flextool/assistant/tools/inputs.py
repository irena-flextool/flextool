"""Input-data tools: entities, parameter values with where they come from, parameter descriptions, and the input
differences of two scenarios.  Every value is read from the input database as stored (one unfiltered load per
file, ``flextool.assistant.inputs._db``); the scenario decides which alternative's value counts."""
from __future__ import annotations

from typing import Any

from flextool.assistant import project
from flextool.assistant.caps import DESCRIPTION_MAX, NAME_MAX, CapArg, cap_text
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, message, name_list, similar_text
from flextool.assistant.inputs import _db, _values
from flextool.assistant.inputs._notes import notes_for
from flextool.assistant.inputs._schema import ranks_in_words, shipped_schema
from flextool.assistant.inputs._units import parameter_unit
from flextool.assistant.registry import Fit, make_tool
from flextool.assistant.unit_wording import (
    STORED_VALUES,
    UNITS_CONVENTION,
    add_session_notes,
    is_convention_unit,
    shown_unit,
)

LIST_HINT = "use name_contains or member to narrow the list"
VALUE_HINT = "name an entity, or narrow with entity_class"


# ---------------------------------------------------------------------------------------------------------
# Shared helpers


def _open(ctx: AssistantContext, args: dict) -> tuple[project.Source, _db.InputDB, list[str]]:
    source = project.resolve_source(ctx, args.get("source"))
    db, notes = project.open_source(ctx, source)
    return source, db, list(notes)


def _scenario(ctx: AssistantContext, source: project.Source, db: _db.InputDB, args: dict,
              key: str = "scenario") -> str:
    return project.resolve_scenario(ctx, source, db, args.get(key))


def _check_class(db: _db.InputDB, entity_class: str) -> None:
    if entity_class not in db.classes:
        raise ToolError("unknown_class", entity_class=entity_class, similar=similar_text(entity_class, list(db.classes)))


def _names(byname: tuple[str, ...]) -> list[str]:
    """An entity's names for a result, each cut to the length of a name (they are user strings)."""
    return [cap_text(n, NAME_MAX) for n in byname]


def _description(text: str | None) -> str | None:
    """An entity's description for a result (a user string, cut to :data:`DESCRIPTION_MAX`); ``None`` when it has
    none (P3 §2.4).  The payload is redacted when it is sent."""
    return cap_text(text, DESCRIPTION_MAX) if text and text.strip() else None


def _entity_text(byname: tuple[str, ...]) -> str:
    return "'" + "', '".join(byname) + "'" if len(byname) > 1 else f"'{byname[0]}'"


def _byname(db: _db.InputDB, source: project.Source, entity_class: str, entity: Any) -> tuple[str, ...]:
    """The entity's names per dimension; a string names a 0-dimensional entity, or a multi-dimensional one by its
    own name (``coal_chp__heat``)."""
    dims = db.classes[entity_class].dimensions
    names = [entity] if isinstance(entity, str) else [str(e) for e in entity]
    if len(names) == 1 and len(dims) > 1:
        found = db.entity_by_name(entity_class, names[0])
        if found is None:
            raise ToolError("entity_dimensions", entity_class=entity_class, n=len(dims), dimensions=", ".join(dims))
        return found.byname
    byname = tuple(names)
    if db.entity(entity_class, byname) is None:
        candidates = [", ".join(b) for b in db.entities.get(entity_class, {})]
        raise ToolError("unknown_entity", entity_class=entity_class, entity=_entity_text(byname), source=source.name,
                        similar=similar_text(", ".join(byname), candidates))
    return byname


def _value_of(view: _db.ScenarioView, entity_class: str, byname: tuple[str, ...]):
    """The entity's parsed value of a parameter in the scenario (``None`` when no alternative sets it)."""
    def value_of(parameter: str) -> Any:
        item = view.value(entity_class, byname, parameter)
        return item.parsed() if item is not None else None
    return value_of


def _group_members(view: _db.ScenarioView, group: str) -> set[str]:
    """The entity classes of a group's members: the second dimension of every ``group__X`` entity of *group*."""
    db = view.db
    out: set[str] = set()
    for cls, item in db.classes.items():
        if len(item.dimensions) == 2 and item.dimensions[0] == "group":
            for e in db.entities.get(cls, {}).values():
                if e.elements and e.elements[0] == group and view.active(cls, e.byname)[0]:
                    out.add(item.dimensions[1])
    return out


def _unit(view: _db.ScenarioView | None, entity_class: str, parameter: str,
          byname: tuple[str, ...] | None) -> tuple[str | None, str | None]:
    if view is None or byname is None:
        unit, reason = parameter_unit(entity_class, parameter)
    else:
        members = _group_members(view, byname[0]) if entity_class == "group" else None
        unit, reason = parameter_unit(entity_class, parameter, value_of=_value_of(view, entity_class, byname),
                                      members=members)
    return shown_unit(unit), reason


def _session_notes(ctx: AssistantContext, payload: dict, *, units: bool, stored: bool) -> dict:
    keys = ([UNITS_CONVENTION] if units else []) + ([STORED_VALUES] if stored else [])
    return add_session_notes(payload, keys, ctx.notes_sent) if keys else payload


def _version_relation(db: _db.InputDB) -> str | None:
    status = _db.version_status(db.version)
    return {"needs_migration": "older", "newer": "newer"}.get(status)


def _check_parameter(db: _db.InputDB, entity_class: str, parameter: str) -> None:
    if db.definition(entity_class, parameter) is not None:
        return
    if _version_relation(db) is not None and (entity_class, parameter) in shipped_schema().parameters:
        return            # defined by this FlexTool, not (or not under this name) by the older / newer database
    names = sorted(set(db.class_parameters(entity_class)) | {p.name for p in
                                                             shipped_schema().class_parameters(entity_class)})
    raise ToolError("unknown_parameter", entity_class=entity_class, parameter=parameter,
                    similar=similar_text(parameter, names))


# ---------------------------------------------------------------------------------------------------------
# Value records


def value_record(ctx: AssistantContext, db: _db.InputDB, view: _db.ScenarioView, entity_class: str,
                 byname: tuple[str, ...], parameter: str, *, max_rows: int,
                 time_range: tuple[str, str] | None = None) -> dict:
    """The value record of one (entity, parameter) in the scenario (P0 §4.3)."""
    record: dict[str, Any] = {"entity": _names(byname), "parameter": parameter, "scenario": view.scenario}
    notes: list[str] = []
    active, reason = view.active(entity_class, byname)
    if not active:
        record["active"] = False
        notes.append(message("inactive_entity", entity=", ".join(byname), scenario=view.scenario, reason=reason))
    winner = view.value(entity_class, byname, parameter)
    relation = _version_relation(db)
    value: Any = None
    if winner is not None:
        value = winner.parsed()
        record.update(stored=True, is_default=False, **_values.summarize(value, max_rows=max_rows,
                                                                          time_range=time_range))
        record["provenance"] = {
            "from_alternative": winner.alternative,
            "rank_order": view.ranked,
            "overridden": [{"alternative": v.alternative, "value": _values.compact(v.parsed())}
                           for v in view.overridden(entity_class, byname, parameter)],
        }
        outside = view.outside(entity_class, byname, parameter)
        if outside:
            record["provenance"]["also_set_outside_the_scenario"] = outside
    elif relation is not None:
        record.update(stored=False, is_default=None, value=None)
        stored = db.stored_parameters(entity_class, byname)
        notes.append(message("older_database", relation=relation, names=name_list(stored, 40)))
    else:
        definition = db.definition(entity_class, parameter)
        default = definition.default() if definition is not None else None
        if default is not None:
            value = default
            record.update(stored=False, is_default=True, **_values.summarize(default, max_rows=max_rows,
                                                                              time_range=time_range))
        else:
            record.update(stored=False, is_default=False, value=None)
            notes.append(message("no_value"))
        record["provenance"] = {"from_alternative": None, "rank_order": view.ranked, "overridden": []}
        outside = view.outside(entity_class, byname, parameter)
        if outside:
            record["provenance"]["also_set_outside_the_scenario"] = outside
    if value is not None:
        shape, shape_note = _values.shape_of(entity_class, parameter, ", ".join(byname), value)
        record["shape"] = shape
        record["shape_note"] = shape_note
    unit, unit_reason = _unit(view, entity_class, parameter, byname)
    record["unit"] = unit
    record["unit_reason"] = unit_reason
    notes += notes_for(entity_class, parameter, value_of=_value_of(view, entity_class, byname))
    record["notes"] = notes
    return _session_notes(ctx, record, units=bool(unit) and is_convention_unit(unit), stored=True)


# ---------------------------------------------------------------------------------------------------------
# Tools


def list_entities(ctx: AssistantContext, args: dict) -> Fit:
    source, db, notes = _open(ctx, args)
    entity_class = str(args["entity_class"])
    _check_class(db, entity_class)
    scenario = None
    if args.get("scenario") is not None or ctx.default_scenario is not None:
        scenario = _scenario(ctx, source, db, args)
    contains = (args.get("name_contains") or "").lower()
    member = args.get("member")
    view = db.view(scenario) if scenario is not None else None
    active_names: list[dict[str, Any]] = []
    inactive: list[list[str]] = []
    for e in sorted(db.entities.get(entity_class, {}).values(), key=lambda e: e.byname):
        if contains and not any(contains in n.lower() for n in [*e.byname, e.name]):
            continue
        if member is not None and member not in e.byname:
            continue
        if view is not None and not view.active(entity_class, e.byname)[0]:
            inactive.append(_names(e.byname))
        else:
            active_names.append({"entity": _names(e.byname), "description": _description(e.description)})
    limit = args["max_results"]
    payload = {
        "source": source.name,
        "scenario": scenario,
        "entity_class": entity_class,
        "dimensions": list(db.classes[entity_class].dimensions) or [entity_class],
        "entities": active_names[:limit],
        "total": len(active_names),
        "inactive_in_scenario": inactive[:limit] if view is not None else None,
        "truncated": (f"showing {limit} of {len(active_names)} entities — {LIST_HINT}"
                      if len(active_names) > limit else None),
        "notes": notes,
    }
    return Fit(payload, rows_key="entities", what="entities", hint=LIST_HINT)


def _match_names(db: _db.InputDB, name: Any) -> list[_db.EntityItem]:
    names = [name] if isinstance(name, str) else [str(n) for n in name]
    out: list[_db.EntityItem] = []
    for cls in db.classes:
        if len(names) == 1:
            found = db.entity_by_name(cls, names[0])
        else:
            found = db.entity(cls, tuple(names))
        if found is not None:
            out.append(found)
    return out


def get_entity(ctx: AssistantContext, args: dict) -> Fit:
    source, db, notes = _open(ctx, args)
    scenario = _scenario(ctx, source, db, args)
    view = db.view(scenario)
    matches = _match_names(db, args["name"])
    if not matches:
        name = args["name"]
        names = [name] if isinstance(name, str) else list(name)
        everything = sorted({e.name for ents in db.entities.values() for e in ents.values()})
        raise ToolError("entity_not_found", entity=_entity_text(tuple(names)), source=source.name,
                        similar=similar_text(", ".join(names), everything))
    max_rows = args["max_rows"]
    out_matches = []
    any_convention = False
    for e in matches:
        active, reason = view.active(e.entity_class, e.byname)
        parameters = []
        for (cls, byname, param), item in sorted(view.winners().items(), key=lambda kv: kv[0][2]):
            if cls != e.entity_class or byname != e.byname:
                continue
            unit, _ = _unit(view, cls, param, byname)
            any_convention = any_convention or (bool(unit) and is_convention_unit(unit))
            parameters.append({"parameter": param, "summary": _values.compact(item.parsed()), "unit": unit,
                               "from_alternative": item.alternative})
        relationships = []
        if len(e.byname) == 1:
            for rel in sorted(db.relationships_of(e.entity_class, e.name), key=lambda r: (r.entity_class, r.byname)):
                relationships.append({"entity_class": rel.entity_class, "entity": _names(rel.byname),
                                      "active": view.active(rel.entity_class, rel.byname)[0]})
        match: dict[str, Any] = {
            "entity_class": e.entity_class,
            "entity": _names(e.byname),
            "description": _description(e.description),
            "active": active,
            "inactive_reason": reason,
            "parameters": parameters[:max_rows],
            "relationships": relationships[:max_rows],
        }
        cut = [k for k, rows in (("parameters", parameters), ("relationships", relationships)) if len(rows) > max_rows]
        if cut:
            match["truncated"] = (f"showing at most {max_rows} {' and '.join(cut)} — use get_parameter for one "
                                  "parameter, or list_entities with member for the relationships")
        out_matches.append(match)
    payload = {"source": source.name, "scenario": scenario, "matches": out_matches, "notes": notes}
    return Fit(_session_notes(ctx, payload, units=any_convention, stored=True), rows_key="matches",
               what="matches", hint="name the entity with all its dimensions")


def get_parameter(ctx: AssistantContext, args: dict) -> Fit | dict:
    source, db, notes = _open(ctx, args)
    entity_class = str(args["entity_class"])
    parameter = str(args["parameter"])
    _check_class(db, entity_class)
    _check_parameter(db, entity_class, parameter)
    scenario = _scenario(ctx, source, db, args)
    view = db.view(scenario)
    time_range = _values.parse_time_range(args.get("time_range"))
    max_rows = args["max_rows"]
    if args.get("entity") is not None:
        byname = _byname(db, source, entity_class, args["entity"])
        record = value_record(ctx, db, view, entity_class, byname, parameter, max_rows=max_rows,
                              time_range=time_range)
        record["source"] = source.name
        record["notes"] = notes + record["notes"]
        return record
    rows = []
    inactive = 0
    definition = db.definition(entity_class, parameter)
    default = definition.default() if definition is not None else None
    for e in sorted(db.entities.get(entity_class, {}).values(), key=lambda e: e.byname):
        if not view.active(entity_class, e.byname)[0]:
            inactive += 1
            continue
        item = view.value(entity_class, e.byname, parameter)
        if item is not None:
            rows.append({"entity": _names(e.byname), "stored": True,
                         "value": _values.compact(item.parsed()), "from_alternative": item.alternative})
        else:
            rows.append({"entity": _names(e.byname), "stored": False,
                         "value": _values.compact(default) if default is not None else None,
                         "is_default": default is not None})
    unit, unit_reason = _unit(None, entity_class, parameter, None)
    payload = {
        "source": source.name, "scenario": scenario, "entity_class": entity_class, "parameter": parameter,
        "unit": unit, "unit_reason": unit_reason, "entities": rows, "total": len(rows),
        "inactive_in_scenario": inactive, "notes": notes + notes_for(entity_class, parameter),
    }
    payload = _session_notes(ctx, payload, units=bool(unit) and is_convention_unit(unit), stored=True)
    return Fit(payload, rows_key="entities", what="entities", hint=VALUE_HINT)


def describe_parameter(ctx: AssistantContext, args: dict) -> Fit | dict:
    from flextool.engine.input._param_shapes import PARAM_ALLOWED_SHAPES, allowed_shapes_in_words

    schema = shipped_schema()
    entity_class = str(args["entity_class"])
    if entity_class not in schema.classes:
        raise ToolError("unknown_class", entity_class=entity_class, similar=similar_text(entity_class,
                                                                                         list(schema.classes)))
    parameter = args.get("parameter")
    if parameter is None:
        cls = schema.classes[entity_class]
        rows = []
        for p in sorted(schema.class_parameters(entity_class), key=lambda p: p.name):
            unit, _ = _unit(None, entity_class, p.name, None)
            rows.append({"parameter": p.name, "unit": unit, "group": p.group, "description": cap_text(p.description, 160)})
        payload = {"entity_class": entity_class, "description": cls.description,
                   "dimensions": list(cls.dimensions) or [entity_class], "parameters": rows,
                   "schema_note": "These are the parameters as this FlexTool defines them."}
        return Fit(payload, rows_key="parameters", what="parameters", hint="ask for one parameter")
    parameter = str(parameter)
    definition = schema.parameters.get((entity_class, parameter))
    if definition is None:
        raise ToolError("unknown_parameter", entity_class=entity_class, parameter=parameter,
                        similar=similar_text(parameter, [p.name for p in schema.class_parameters(entity_class)]))
    allowed = PARAM_ALLOWED_SHAPES.get((entity_class, parameter))
    shapes = (allowed_shapes_in_words(allowed) if allowed is not None
              else ranks_in_words(schema.value_types.get((entity_class, parameter), ())))
    unit, unit_reason = _unit(None, entity_class, parameter, None)
    payload = {
        "entity_class": entity_class,
        "parameter": parameter,
        "description": cap_text(definition.description, DESCRIPTION_MAX),
        "default": definition.default,
        "value_list": list(schema.value_lists.get(definition.value_list, ())) if definition.value_list else None,
        "allowed_shapes": shapes,
        "unit": unit,
        "unit_reason": unit_reason,
        "group": definition.group,
        "notes": notes_for(entity_class, parameter),
        "docs": _reference_sections(parameter),
        "schema_note": "This is the parameter as this FlexTool defines it.",
    }
    return _session_notes(ctx, payload, units=bool(unit) and is_convention_unit(unit), stored=False)


def _reference_sections(parameter: str) -> list[dict]:
    """The reference.md sections that best match the parameter name (none when the documentation is not
    available)."""
    from flextool.assistant import docs_store

    try:
        store = docs_store.load()
    except docs_store.DocsUnavailable:
        return []
    hits = [h for h in store.search(parameter, kind="docs", max_results=8) if h["doc"] == "reference.md"]
    return [{"doc": h["doc"], "section": h["section"], "anchor": h["anchor"]} for h in hits[:2]]


def compare_inputs(ctx: AssistantContext, args: dict) -> Fit:
    source, db, notes = _open(ctx, args)
    a = project.resolve_scenario(ctx, source, db, args["scenario_a"])
    b = project.resolve_scenario(ctx, source, db, args["scenario_b"])
    only_class = args.get("entity_class")
    if only_class is not None:
        _check_class(db, only_class)
    va, vb = db.view(a), db.view(b)
    wa, wb = va.winners(), vb.winners()
    differences = []
    for key in sorted(set(wa) | set(wb)):
        cls, byname, param = key
        if only_class is not None and cls != only_class:
            continue
        if not (va.active(cls, byname)[0] or vb.active(cls, byname)[0]):
            continue
        ia, ib = wa.get(key), wb.get(key)
        pa = ia.parsed() if ia is not None else None
        pb = ib.parsed() if ib is not None else None
        if ia is not None and ib is not None and (ia is ib or _values.same_value(pa, pb)):
            continue
        differences.append({
            "entity_class": cls, "entity": _names(byname), "parameter": param,
            "a": _values.compact(pa) if ia is not None else None,
            "b": _values.compact(pb) if ib is not None else None,
            "from_alternative_a": ia.alternative if ia is not None else None,
            "from_alternative_b": ib.alternative if ib is not None else None,
        })
    only_a, only_b = [], []
    for cls, ents in sorted(db.entities.items()):
        if only_class is not None and cls != only_class:
            continue
        for byname in sorted(ents):
            in_a, in_b = va.active(cls, byname)[0], vb.active(cls, byname)[0]
            if in_a and not in_b:
                only_a.append({"entity_class": cls, "entity": _names(byname)})
            elif in_b and not in_a:
                only_b.append({"entity_class": cls, "entity": _names(byname)})
    max_rows = args["max_rows"]
    payload = {
        "source": source.name,
        "scenario_a": a,
        "scenario_b": b,
        "alternatives_only_a": [alt for alt in va.ranked if alt not in vb.rank],
        "alternatives_only_b": [alt for alt in vb.ranked if alt not in va.rank],
        "differences": differences[:max_rows],
        "entities_active_only_in": {"a": only_a[:max_rows], "b": only_b[:max_rows]},
        "total": len(differences),
        "truncated": (f"showing {max_rows} of {len(differences)} differences — narrow with entity_class"
                      if len(differences) > max_rows else None),
        "notes": notes + ["a and b are the stored values each scenario uses; a missing value (null) means no "
                          "alternative of that scenario sets it (FlexTool then uses the default)."],
    }
    return Fit(payload, rows_key="differences", what="differences", hint="narrow with entity_class")


# ---------------------------------------------------------------------------------------------------------
# Registry entries

_ENTITY = {"type": "array", "items": {"type": "string"}}     # P0-D7: one name per dimension
_SOURCE = {"type": "string"}
_SCENARIO = {"type": "string"}
_TIME_RANGE = {"type": "object", "properties": {"from": {"type": "string"}, "to": {"type": "string"}},
               "required": ["from", "to"], "additionalProperties": False}
_SOURCE_TEXT = ("source is the input source (file name or number; default: the connection's, or the only one); "
                "scenario the scenario (default: the connection's, or the only one).")

TOOLS = [
    make_tool(
        name="list_entities", title="Entities of a class",
        description=("The entities of an entity class (for example unit, node, unit__outputNode), each with its "
                     "description. With a scenario, the active ones, and the inactive ones apart. name_contains filters by part of a name in any "
                     "dimension (any case); member keeps entities that have that exact name in some dimension. "
                     "max_results is up to 500 (default 100). " + _SOURCE_TEXT),
        properties={"entity_class": {"type": "string"}, "scenario": _SCENARIO, "source": _SOURCE,
                    "name_contains": {"type": "string"}, "member": {"type": "string"},
                    "max_results": {"type": "integer"}},
        required=("entity_class",), handler=list_entities, area="inputs", small_tier=False,
        caps={"max_results": CapArg("max_items", maximum=500, default=100)}),
    make_tool(
        name="get_entity", title="An entity and its data",
        description=("Everything about one entity in a scenario: every class it appears in, whether it is active, "
                     "its description, the values its alternatives set (with units and the alternative each comes "
                     "from) and the multi-dimensional entities that name it (for example the unit__outputNode of a "
                     "unit). name lists the entity's names, one per dimension, for example [\"coal_plant\"] or "
                     "[\"coal_plant\", \"west\"]. "
                     + _SOURCE_TEXT),
        properties={"name": _ENTITY, "scenario": _SCENARIO, "source": _SOURCE, "max_rows": {"type": "integer"}},
        required=("name",), handler=get_entity, area="inputs", small_tier=True,
        caps={"max_rows": CapArg("max_rows")}),
    make_tool(
        name="get_parameter", title="A parameter value and where it comes from",
        description=("The value of a parameter in a scenario as stored in the input data: the value or a summary "
                     "of a series, its shape, unit, the alternative it comes from and the alternatives it overrides, "
                     "or the default when none sets it. Without entity: one line per active entity of the class. "
                     "entity lists its names, one per dimension. time_range "
                     '{"from": …, "to": …} keeps part of a series. ' + _SOURCE_TEXT),
        properties={"entity_class": {"type": "string"}, "parameter": {"type": "string"}, "entity": _ENTITY,
                    "scenario": _SCENARIO, "source": _SOURCE, "time_range": _TIME_RANGE,
                    "max_rows": {"type": "integer"}},
        required=("entity_class", "parameter"), handler=get_parameter, area="inputs", small_tier=True,
        caps={"max_rows": CapArg("max_rows")}),
    make_tool(
        name="describe_parameter", title="What a parameter means",
        description=("The description of a parameter as this FlexTool defines it: meaning, default, allowed "
                     "choices, the shapes it takes (constant, period or time maps), unit, notes and the reference "
                     "section. Without parameter: the parameters of the entity class."),
        properties={"entity_class": {"type": "string"}, "parameter": {"type": "string"}},
        required=("entity_class",), handler=describe_parameter, area="inputs", small_tier=False),
    make_tool(
        name="compare_inputs", title="Input differences of two scenarios",
        description=("The input differences between two scenarios of one input source: the alternatives only one "
                     "of them has, every value that differs (with the alternative it comes from in each), and the "
                     "entities active in only one. entity_class narrows to one class."),
        properties={"scenario_a": _SCENARIO, "scenario_b": _SCENARIO, "source": _SOURCE,
                    "entity_class": {"type": "string"}, "max_rows": {"type": "integer"}},
        required=("scenario_a", "scenario_b"), handler=compare_inputs, area="inputs", small_tier=False,
        caps={"max_rows": CapArg("max_rows")}),
]
