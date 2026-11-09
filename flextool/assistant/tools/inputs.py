"""Input-data tools: entities, parameter values with where they come from, parameter descriptions, and the input
differences of two scenarios.  Every value is read from the input database as stored (one unfiltered load per
file, ``flextool.assistant.inputs._db``); the scenario decides which alternative's value counts."""
from __future__ import annotations

from collections import Counter
from typing import Any

from flextool.assistant import project
from flextool.assistant.caps import DESCRIPTION_MAX, NAME_MAX, CapArg, cap_text
from flextool.assistant.context import AssistantContext
from flextool.assistant.errors import ToolError, close_names, closest_names, did_you_mean, message, name_list, similar_text
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
#: get_entity shows a map of at most this many rows as its table, a longer one as a summary (P3 §4.7).
ENTITY_TABLE_ROWS = 6
VALUE_HINT = "name an entity, or narrow with entity_class"
#: One get_parameter call reads at most this many parameters and entities (P0 §4.3, batching 2026-11-08).
MAX_PARAMETERS = 10
MAX_ENTITIES = 20
CELLS_HINT = "name fewer parameters or entities, or split the call"


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


def _find(db: _db.InputDB, entity_class: str, names: list[str]) -> tuple[str, ...] | None:
    """The entity's names per dimension, or ``None`` when the class has no such entity; one name names a
    0-dimensional entity, or a multi-dimensional one by its own name (``coal_chp__heat``)."""
    if len(names) == 1 and len(db.classes[entity_class].dimensions) > 1:
        found = db.entity_by_name(entity_class, names[0])
        return found.byname if found is not None else None
    byname = tuple(names)
    return byname if db.entity(entity_class, byname) is not None else None


def _byname(db: _db.InputDB, source: project.Source, entity_class: str, entity: Any) -> tuple[str, ...]:
    """The entity's names per dimension (:func:`_find`), or the error that says why there is none."""
    dims = db.classes[entity_class].dimensions
    names = [entity] if isinstance(entity, str) else [str(e) for e in entity]
    found = _find(db, entity_class, names)
    if found is not None:
        return found
    if len(names) == 1 and len(dims) > 1:
        raise ToolError("entity_dimensions", entity_class=entity_class, n=len(dims), dimensions=", ".join(dims))
    byname = tuple(names)
    if db.entity(entity_class, byname) is None:
        candidates = [", ".join(b) for b in db.entities.get(entity_class, {})]
        raise ToolError("unknown_entity", entity_class=entity_class, entity=_entity_text(byname), source=source.name,
                        similar=similar_text(", ".join(byname), candidates))
    return byname


def _entity_value(value: Any) -> Any:
    """A value as get_entity lists it: a constant itself, a map of at most :data:`ENTITY_TABLE_ROWS` rows with its
    table, a longer one as its summary (kind, index names, rows, minimum, mean, maximum)."""
    if _values.is_indexed(value) and len(_values.rows_of(value)) <= ENTITY_TABLE_ROWS:
        return _values.summarize(value, max_rows=ENTITY_TABLE_ROWS)
    return _values.compact(value)


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


def _parameter_names(db: _db.InputDB, entity_class: str, parameter: str) -> list[str] | None:
    """``None`` when *parameter* is a parameter of the class, else every parameter name of the class (of the
    database and of this FlexTool) to suggest from."""
    if db.definition(entity_class, parameter) is not None:
        return None
    if _version_relation(db) is not None and (entity_class, parameter) in shipped_schema().parameters:
        return None       # defined by this FlexTool, not (or not under this name) by the older / newer database
    return sorted(set(db.class_parameters(entity_class)) | {p.name for p in
                                                            shipped_schema().class_parameters(entity_class)})


def _check_parameter(db: _db.InputDB, entity_class: str, parameter: str) -> None:
    names = _parameter_names(db, entity_class, parameter)
    if names is not None:
        raise ToolError("unknown_parameter", entity_class=entity_class, parameter=parameter,
                        similar=did_you_mean(parameter, names))


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
        cls, byname = e.entity_class, e.byname
        for param in db.parameters_of(cls, byname):
            item = view.value(cls, byname, param)
            if item is None:
                continue
            unit, _ = _unit(view, cls, param, byname)
            any_convention = any_convention or (bool(unit) and is_convention_unit(unit))
            parameters.append({"parameter": param, "value": _entity_value(item.parsed()), "unit": unit,
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
            "parameters_total": len(parameters),
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
    if args.get("parameters") is not None or args.get("entities") is not None:
        return _parameter_table(ctx, args)
    if args.get("parameter") is None:
        raise ToolError("no_parameter_named")
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


# ---------------------------------------------------------------------------------------------------------
# get_parameter with lists: one table of (entity, parameter) cells


def _unique(items: list) -> list:
    out: list = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _as_entity(item: Any) -> list[str]:
    return [item] if isinstance(item, str) else [str(n) for n in item]


def prepare_parameter_args(args: dict) -> dict:
    """The forgiving forms of get_parameter's arguments in the schema's: ``parameters`` or ``parameter`` as one name
    or a list, ``entities`` as one name, a list of names or a list of name lists, ``entity`` as one name."""
    out = dict(args)
    if isinstance(out.get("parameter"), list):
        out["parameters"] = [*out.pop("parameter"), *(_as_list(out.get("parameters")))]
    if isinstance(out.get("parameters"), str):
        out["parameters"] = [out["parameters"]]
    if isinstance(out.get("entity"), str):
        out["entity"] = [out["entity"]]
    entities = out.get("entities")
    if isinstance(entities, str):
        out["entities"] = [[entities]]
    elif isinstance(entities, list):
        out["entities"] = [[e] if isinstance(e, str) else e for e in entities]
    return out


def _as_list(value: Any) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


def _requested_entities(db: _db.InputDB, entity_class: str, given: list[list[str]]) -> list[list[str]]:
    """The requested entities; one name each that together name one entity of a multi-dimensional class (the
    names of its dimensions given as a flat list) are that entity."""
    dims = db.classes[entity_class].dimensions
    if (len(dims) > 1 and len(given) == len(dims) and all(len(e) == 1 for e in given)
            and not any(_find(db, entity_class, e) for e in given)
            and db.entity(entity_class, tuple(e[0] for e in given)) is not None):
        return [[e[0] for e in given]]
    return given


def _cell_value(value: Any, row_limit: int, time_range: tuple[str, str] | None) -> Any:
    """A cell's value: a constant itself, a map of at most *row_limit* rows as its table, a longer one as a
    summary (kind, index names, rows, minimum, mean, maximum)."""
    if not _values.is_indexed(value):
        return _values.compact(value)
    summary = _values.summarize(value, max_rows=row_limit, time_range=time_range)
    if "table" in summary:
        return summary
    keys = ("kind", "index_names", "time_range", "n", "min", "mean", "max", "non_numeric")
    return {k: summary[k] for k in keys if k in summary}


def _cell(db: _db.InputDB, view: _db.ScenarioView, entity_class: str, byname: tuple[str, ...], parameter: str,
          row_limit: int, time_range: tuple[str, str] | None) -> dict:
    """One (entity, parameter) cell: value, shape, alternative, default, unit and notes (as value_record)."""
    winner = view.value(entity_class, byname, parameter)
    notes: list[str] = []
    value: Any = None
    alternative = None
    if winner is not None:
        value, alternative, is_default = winner.parsed(), winner.alternative, False
    elif _version_relation(db) is not None:
        is_default = None
        notes.append(message("older_database", relation=_version_relation(db),
                             names=name_list(db.stored_parameters(entity_class, byname), 40)))
    else:
        definition = db.definition(entity_class, parameter)
        value = definition.default() if definition is not None else None
        is_default = value is not None
        if value is None:
            notes.append(message("no_value"))
    shape = None
    if value is not None:
        shape, shape_note = _values.shape_of(entity_class, parameter, ", ".join(byname), value)
        if shape_note:
            notes.append(shape_note)
    unit, unit_reason = _unit(view, entity_class, parameter, byname)
    if unit_reason:
        notes.append(unit_reason)
    notes += notes_for(entity_class, parameter, value_of=_value_of(view, entity_class, byname))
    shown = None
    if value is not None:
        try:
            shown = _cell_value(value, row_limit, time_range)
        except ToolError as exc:           # a time_range bound this value lacks: no numbers, the reason in notes
            if exc.key != "bad_input_time_step":
                raise
            notes.append(str(exc))
    return {"value": shown, "shape": shape, "from_alternative": alternative, "is_default": is_default,
            "unit": unit, "notes": notes}


def _parameter_table(ctx: AssistantContext, args: dict) -> Fit:
    """get_parameter with lists (P0 §4.3, batching 2026-11-08): a row per (entity, parameter), what the rows
    share in the header."""
    source, db, notes = _open(ctx, args)
    entity_class = str(args["entity_class"])
    _check_class(db, entity_class)
    parameters = _unique([str(p) for p in ([args["parameter"]] if args.get("parameter") is not None else [])
                          + list(args.get("parameters") or [])])
    given = _unique(([_as_entity(args["entity"])] if args.get("entity") is not None else [])
                    + [_as_entity(e) for e in (args.get("entities") or [])])
    if not parameters:
        raise ToolError("no_parameter_named")
    if len(parameters) > MAX_PARAMETERS or len(given) > MAX_ENTITIES:
        raise ToolError("too_many_names", max_parameters=MAX_PARAMETERS, max_entities=MAX_ENTITIES,
                        parameters=len(parameters), entities=len(given))
    scenario = _scenario(ctx, source, db, args)
    view = db.view(scenario)
    time_range = _values.parse_time_range(args.get("time_range"))

    known: list[str] = []
    unknown_parameters: list[dict] = []
    for p in parameters:
        names = _parameter_names(db, entity_class, p)
        if names is None:
            known.append(p)
        else:
            unknown_parameters.append({"parameter": p, "did_you_mean": closest_names(p, names)})

    entities: list[tuple[str, ...]] = []
    unknown_entities: list[dict] = []
    inactive: list[list[str]] = []
    hidden = 0
    if given:
        dims = db.classes[entity_class].dimensions
        candidates = sorted({e.name for e in db.entities.get(entity_class, {}).values()})
        for names in _requested_entities(db, entity_class, given):
            byname = _find(db, entity_class, names)
            if byname is None:
                entry: dict[str, Any] = {"entity": _names(tuple(names)),
                                         "similar": close_names(", ".join(names) if len(names) > 1 else names[0],
                                                                candidates)}
                if len(names) == 1 and len(dims) > 1:
                    entry["note"] = message("entity_dimensions", entity_class=entity_class, n=len(dims),
                                            dimensions=", ".join(dims))
                unknown_entities.append(entry)
                continue
            if byname in entities:
                continue
            entities.append(byname)
            if not view.active(entity_class, byname)[0]:
                inactive.append(_names(byname))
    else:
        for e in sorted(db.entities.get(entity_class, {}).values(), key=lambda e: e.byname):
            if view.active(entity_class, e.byname)[0]:
                entities.append(e.byname)
            else:
                hidden += 1

    n_cells = len(entities) * len(known)
    row_limit = max(1, min(args["max_rows"], max(ENTITY_TABLE_ROWS, args["max_rows"] // max(n_cells, 1))))
    cells = [(byname, p, _cell(db, view, entity_class, byname, p, row_limit, time_range))
             for byname in entities for p in known]

    # What the cells of a parameter share goes in the header: its unit (the most common) and its notes.
    units: dict[str, str | None] = {}
    shared_notes: dict[str, list[str]] = {}
    for p in known:
        mine = [c for _, q, c in cells if q == p]
        if not mine:
            continue
        units[p] = Counter(c["unit"] for c in mine).most_common(1)[0][0]
        common = [n for n in mine[0]["notes"] if all(n in c["notes"] for c in mine[1:])]
        if common:
            shared_notes[p] = common
    unit_column = any(c["unit"] != units.get(p) for _, p, c in cells)
    columns = ["entity", "parameter", "value", "shape", "from_alternative", "is_default"]
    columns += (["unit"] if unit_column else []) + ["notes"]
    rows = []
    for byname, p, c in cells:
        row = [_names(byname), p, c["value"], c["shape"], c["from_alternative"], c["is_default"]]
        if unit_column:
            row.append(c["unit"] if c["unit"] != units[p] else None)
        row.append([n for n in c["notes"] if n not in shared_notes.get(p, [])])
        rows.append(row)

    payload: dict[str, Any] = {"source": source.name, "scenario": scenario, "entity_class": entity_class,
                               "units": units, "columns": columns, "rows": rows, "total": len(rows)}
    if shared_notes:
        payload["notes_by_parameter"] = shared_notes
    if inactive:
        payload["inactive_in_scenario"] = inactive
    if hidden:
        payload["inactive_entities_not_shown"] = hidden
    if unknown_parameters:
        payload["unknown_parameters"] = unknown_parameters
    if unknown_entities:
        payload["unknown_entities"] = unknown_entities
    payload["notes"] = notes
    any_convention = any(u and is_convention_unit(u) for u in units.values())
    payload = _session_notes(ctx, payload, units=any_convention, stored=True)
    return Fit(_shorten_notes(payload, ctx.caps.max_chars), rows_key="rows", what="cells", hint=CELLS_HINT)


#: Note lengths tried, longest first, when a table does not fit its result cap with its notes in full.
NOTE_LENGTHS = (160, 80)
NOTES_LEFT_OUT = ("The notes of each parameter were left out so that the rows fit; get_parameter with one parameter "
                  "and one entity gives them.")


def _shorten_notes(payload: dict, max_chars: int) -> dict:
    """*payload* with its parameter and row notes shortened, then left out, until it fits *max_chars*: the rows
    are the answer, so they go last (``fit_json`` drops trailing rows only when the rows alone do not fit)."""
    from flextool.assistant.caps import dumps

    if len(dumps(payload)) <= max_chars:
        return payload
    for limit in NOTE_LENGTHS:
        trial = dict(payload)
        if "notes_by_parameter" in payload:
            trial["notes_by_parameter"] = {p: [cap_text(n, limit) for n in ns]
                                           for p, ns in payload["notes_by_parameter"].items()}
        trial["rows"] = [[*row[:-1], [cap_text(n, limit) for n in row[-1]]] for row in payload["rows"]]
        if len(dumps(trial)) <= max_chars:
            return trial
    trial = {k: v for k, v in payload.items() if k != "notes_by_parameter"}
    trial["rows"] = [[*row[:-1], []] for row in payload["rows"]]
    trial["notes"] = [*payload["notes"], NOTES_LEFT_OUT]
    return trial


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
                        similar=did_you_mean(parameter, [p.name for p in schema.class_parameters(entity_class)]))
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
_NAMES = {"type": "array", "items": {"type": "string"}}
_ENTITIES = {"type": "array", "items": _ENTITY}
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
                     "its description, the value of each parameter the scenario sets (a constant itself, a short "
                     "map as its table, a longer one summarised; with units and the alternative each comes from) "
                     "and the multi-dimensional entities that name it (for example the unit__outputNode of a "
                     "unit). name lists the entity's names, one per dimension, for example [\"coal_plant\"] or "
                     "[\"coal_plant\", \"west\"]. "
                     + _SOURCE_TEXT),
        properties={"name": _ENTITY, "scenario": _SCENARIO, "source": _SOURCE, "max_rows": {"type": "integer"}},
        required=("name",), handler=get_entity, area="inputs", small_tier=True,
        caps={"max_rows": CapArg("max_rows")}),
    make_tool(
        name="get_parameter", title="Parameter values and where they come from",
        description=("Parameter values of an entity class in a scenario as stored in the input data. parameters "
                     f"(up to {MAX_PARAMETERS}) and entities (up to {MAX_ENTITIES}) take lists, so one call answers "
                     "several lookups: one row per entity and parameter with the value (a long series "
                     "summarised), shape, the alternative it comes from or the default; units and shared notes "
                     "once per parameter. Without entities: every active entity. An entity is its names, one per "
                     "dimension. parameter and entity (one each) give one value in full, with the alternatives it "
                     'overrides. time_range {"from": …, "to": …} keeps the steps from one to the other, both included, '
                     "in the series' order; they are index values (time step names such as t0001, or datetimes "
                     "such as 2020-01-01T00:00). " + _SOURCE_TEXT),
        properties={"entity_class": {"type": "string"}, "parameters": _NAMES, "entities": _ENTITIES,
                    "parameter": {"type": "string"}, "entity": _ENTITY,
                    "scenario": _SCENARIO, "source": _SOURCE, "time_range": _TIME_RANGE,
                    "max_rows": {"type": "integer"}},
        required=("entity_class",), handler=get_parameter, area="inputs", small_tier=True,
        caps={"max_rows": CapArg("max_rows")}, prepare=prepare_parameter_args),
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
