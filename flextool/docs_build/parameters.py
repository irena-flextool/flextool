"""The parameter catalogue: every input parameter of this FlexTool, as a docs page and as an Excel data dictionary.

Built from the shipped schema (``schemas/spinedb_schema.json``: descriptions, defaults, value lists, parameter
groups and the ``parameter_types`` map ranks) and the engine's shape allow-list (``PARAM_ALLOWED_SHAPES``), through
the same helpers as the assistant's ``describe_parameter`` (``shipped_schema``, ``parameter_unit``,
``allowed_shapes_in_words``, ``ranks_in_words``), so the two cannot disagree; a test pins that.

The allowed shapes of a parameter, in order of preference, all in the allow-list's words ("a constant, or a map
over periods (1 level)"): the engine's allow-list (plus the stochastic-time map of ``STOCHASTIC_TIME_PARAMS``);
the schema's map ranks when they say exactly what is allowed (a constant, a list of values); the description's own
shape sentence ("Constant, period or time.") translated when its tokens agree with the map ranks, else verbatim;
the description's index sentence ("Index: constraint name, value: coefficient", "(a -> b)"); the map ranks.

``python -m flextool.docs_build --xlsx FILE [--input-db DB [--scenario NAME]]`` writes the Excel data dictionary
(:func:`write_xlsx`); with ``--input-db`` (a file path or a URL) it marks the parameters a database uses
(:func:`database_usage`), read from the parameter-value keys only (no value is parsed, the file is opened
read-only).
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from flextool.assistant.inputs._schema import bracket, ranks_in_words, shipped_schema
from flextool.assistant.inputs._units import parameter_unit
from flextool.sqlite_url import read_only_sqlite_url_string

PAGE_TITLE = "Parameter catalogue"

#: The description's own shape sentence: "Constant, period or time." / "Constant or period.".
_SHAPE_SENTENCE_RE = re.compile(r"(?<![\w-])(Constant[^.]*\.)")
#: Its tokens: (the allow-list shape name, map rank).  ``stochastic time`` has no allow-list shape.
_SENTENCE_TOKENS: dict[str, tuple[str | None, int]] = {
    "constant": ("SCALAR", 0), "period": ("MAP_PERIOD", 1), "time": ("MAP_TIME", 1),
    "period+time": ("MAP_PERIOD_TIME", 2), "stochastic time": (None, 3)}
#: The stochastic-time shape (``_param_shapes.STOCHASTIC_TIME_PARAMS``: branch -> analysis_time -> time).
STOCHASTIC_TIME_WORDS = "a stochastic-time map (branch, analysis_time, time; 3 levels)"
#: The description's index sentence: "Index: period name, value: timeset name." ...
_INDEX_SENTENCE_RE = re.compile(r"Index: ([^.]+?), value: ([^.()]+)")
#: ... or its arrow form: "(option name -> value)", "(map: period, branch, analysis_time, realized → weight)".
_ARROW_RE = re.compile(r"\((?:[\w -]*?map:\s*)?([^()]*?(?:->|→)[^()]*)\)", re.IGNORECASE)
_ARROW_SPLIT_RE = re.compile(r"\s*(?:->|→)\s*")

#: The web page of the catalogue (the Read me sheet links it).
WEB_PAGE_URL = "https://irena-flextool.github.io/flextool/parameter_catalogue/"

#: The sheets of the Excel data dictionary and the columns of the parameters sheet.
README_SHEET = "Read me"
SHEET = "parameters"
CLASSES_SHEET = "classes"
COLUMNS: tuple[str, ...] = ("class", "parameter", "definition", "unit", "quantity type", "allowed shapes",
                            "default", "allowed values", "parameter group")
DB_COLUMNS: tuple[str, ...] = ("used in DB", "entities", "alternatives")
TEAM_COLUMNS: tuple[str, ...] = ("Source", "How to replace", "Notes")
NOT_IN_THIS_VERSION = "not in this FlexTool version"


@dataclass(frozen=True)
class ParameterRecord:
    """One input parameter as this FlexTool defines it."""

    entity_class: str
    parameter: str
    description: str                  # the schema description in full, unit bracket included
    definition: str                   # the description without its unit bracket
    unit: str | None                  # parameter_unit's unit (the tools show it with shown_unit)
    unit_reason: str | None
    default: object                   # as stored in the schema file (a number, a string or None)
    value_list: tuple[str, ...] | None
    declared_shapes: str              # the allow-list in words, else the map ranks in words (describe_parameter)
    allowed_shapes: str               # the catalogue's wording (see the module docstring)
    group: str | None
    quantity_type: str | None         # PARAMETER_TYPES in words ("price per energy"), None when not registered


#: What each entity class is, in plain words for the people who gather a model's data (the schema's own class
#: descriptions are mostly empty).  :func:`class_description` falls back to the schema for a class not listed.
CLASS_HELP: dict[str, str] = {
    "commodity": "A fuel or other product bought from, or sold to, outside the modelled system at a price "
                 "(natural gas, coal, hydrogen for export).",
    "connection": "A link that moves energy between two nodes: a transmission line, an interconnector or a "
                  "pipeline, with its capacity, losses and costs.",
    "constraint": "A constraint the user writes: its terms (flows, capacities, storage states, units online) "
                  "come from the constraint_..._coeff parameters of the other classes; here its sense and its "
                  "constant. A constraint with only capacity terms holds once per period, otherwise once per time "
                  "step.",
    "flowGroup": "A set of flows (a unit or a connection into or out of a node) that share limits, such as a "
                 "maximum instant flow or a cumulative flow over a period.",
    "group": "A set of units, connections and nodes that share limits or targets (investment limits, CO2 "
             "limits, capacity margin, inertia) or that are reported together.",
    "model": "Settings for the whole model: which solves run and in which order, inflation, and numerical "
             "settings.",
    "node": "A point where energy is balanced: demand, supply and transfers meet there. A node can also be a "
            "storage (its state is the energy stored) or a price-setting market.",
    "profile": "A time series of shares of capacity (for example wind or solar capacity factors), used as an "
               "upper limit, a lower limit or a fixed value for a flow or a storage state.",
    "reserve": "A kind of operating reserve (for example primary or secondary), for which capacity is held back.",
    "solve": "One optimisation run inside the model: its periods and time steps, whether it decides "
             "investments or dispatch, rolling or stochastic settings, and solver options.",
    "timeline": "The time steps of the input data and the duration of each.",
    "timeset": "A selection of time steps of a timeline that a solve uses (all of them, or for example "
               "representative weeks).",
    "unit": "A plant or device that turns inputs into outputs: a power plant, a wind or solar farm, an "
            "electrolyser, a heat pump, a battery inverter.",
    "upDown": "The direction of a reserve: up (more output) or down (less output).",
    "commodity__node": "Ties a commodity to a node: the node buys (or sells) that commodity.",
    "connection__node": "The flow between a connection and one of its nodes (for user constraints).",
    "connection__profile": "Applies a profile as a limit on the flow of a connection.",
    "group__connection": "Makes a connection a member of a group.",
    "group__node": "Makes a node a member of a group.",
    "group__unit": "Makes a unit a member of a group.",
    "node__profile": "Applies a profile as a limit on the storage state of a node.",
    "unit__inputNode": "An input of a unit: the node it takes energy or fuel from, and the properties of that "
                       "flow (costs, coefficients, ramping).",
    "unit__outputNode": "An output of a unit: the node it delivers to, and the properties of that flow (costs, "
                        "coefficients, ramping).",
    "connection__node__node": "Which two nodes a connection links (the first and the second node).",
    "flowGroup__connection__node": "Makes the flow of a connection to or from a node a member of a flowGroup.",
    "flowGroup__unit__node": "Makes the flow of a unit to or from a node a member of a flowGroup.",
    "reserve__upDown__group": "The reserve requirement of a group of nodes, for one reserve and direction.",
    "unit__node__profile": "Applies a profile as a limit on the flow of a unit to or from a node.",
    "reserve__upDown__connection__node": "Lets a connection provide a reserve, in one direction, to a node.",
    "reserve__upDown__unit__node": "Lets a unit provide a reserve, in one direction, to a node.",
}


def class_description(entity_class: str) -> str:
    """What *entity_class* is (:data:`CLASS_HELP`, else the schema's class description, else ``""``)."""
    if entity_class in CLASS_HELP:
        return CLASS_HELP[entity_class]
    definition = shipped_schema().classes.get(entity_class)
    return " ".join((definition.description or "").split()) if definition is not None else ""


def _definition(description: str) -> str:
    mark = bracket(description)
    text = description[len(mark):] if mark else description
    return " ".join(text.split())


def _exact_rank_words(value_types: tuple[tuple[str, int], ...]) -> bool:
    """Whether the schema's types say exactly what is allowed: constants and lists of values, no map."""
    return bool(value_types) and all(t in ("float", "str", "bool", "array") for t, _ in value_types)


def _schema_ranks(value_types: tuple[tuple[str, int], ...]) -> set[int]:
    """The map ranks of the schema's types, with 0 for a constant."""
    return {0 if t in ("float", "str", "bool") else r for t, r in value_types if t in ("float", "str", "bool", "map")}


def _sentence_shapes(sentence: str, value_types: tuple[tuple[str, int], ...]) -> str:
    """The description's shape sentence in the allow-list's words, when every token is known and the tokens'
    ranks are the schema's; else the sentence verbatim."""
    from flextool.engine.input._param_shapes import Shape, allowed_shapes_in_words

    tokens = [t.strip().lower() for t in re.split(r",|\bor\b", sentence.rstrip(".")) if t.strip()]
    if not tokens or any(t not in _SENTENCE_TOKENS for t in tokens):
        return sentence
    if {_SENTENCE_TOKENS[t][1] for t in tokens} != _schema_ranks(value_types):
        return sentence
    shapes = {Shape[name] for t in tokens if (name := _SENTENCE_TOKENS[t][0]) is not None}
    words = allowed_shapes_in_words(shapes) if shapes else ""
    if "stochastic time" in tokens:
        words = f"{words}, or {STOCHASTIC_TIME_WORDS}" if words else STOCHASTIC_TIME_WORDS
    return words


def _index_shapes(text: str, value_types: tuple[tuple[str, int], ...]) -> str | None:
    """The description's index sentence as a shape ("a map: constraint name → coefficient (1 level)"), when the
    schema allows one map rank and the sentence names that many index levels; else ``None``."""
    ranks = _schema_ranks(value_types)
    map_ranks = ranks - {0}
    if len(map_ranks) != 1 or any(t not in ("float", "str", "bool", "map") for t, _ in value_types):
        return None
    rank = next(iter(map_ranks))
    if hit := _INDEX_SENTENCE_RE.search(text):
        index, value = [hit.group(1).strip()], hit.group(2).strip()
    elif hit := _ARROW_RE.search(text):
        parts = _ARROW_SPLIT_RE.split(hit.group(1).strip())
        value = parts[-1]
        index = [i.strip() for p in parts[:-1] for i in p.split(",")]
    else:
        return None
    if len(index) != rank:
        return None
    words = f"a map: {', '.join(index)} → {value} ({rank} level{'' if rank == 1 else 's'})"
    return f"a constant, or {words}" if 0 in ranks else words


def _quantity_type(entity_class: str, parameter: str) -> str | None:
    from flextool.engine.autoscale._quantity_types import PARAMETER_TYPES, QuantityType

    kind = PARAMETER_TYPES.get((parameter, entity_class))
    return None if kind is None or kind is QuantityType.UNKNOWN else kind.value.replace("_", " ")


def parameter_record(entity_class: str, parameter: str) -> ParameterRecord:
    """The record of one parameter; ``KeyError`` when this FlexTool does not define it."""
    from flextool.engine.input._param_shapes import (
        PARAM_ALLOWED_SHAPES,
        STOCHASTIC_TIME_PARAMS,
        allowed_shapes_in_words,
    )

    schema = shipped_schema()
    definition = schema.parameters[(entity_class, parameter)]
    value_types = schema.value_types.get((entity_class, parameter), ())
    allowed = PARAM_ALLOWED_SHAPES.get((entity_class, parameter))
    rank_words = ranks_in_words(value_types)
    declared = allowed_shapes_in_words(allowed) if allowed is not None else rank_words
    text = _definition(definition.description)
    if allowed is not None:
        shapes = declared
        if (entity_class, parameter) in STOCHASTIC_TIME_PARAMS:
            shapes = f"{shapes}, or {STOCHASTIC_TIME_WORDS}"
    elif _exact_rank_words(value_types):
        shapes = declared
    elif sentence := _SHAPE_SENTENCE_RE.search(text):
        shapes = _sentence_shapes(sentence.group(1), value_types)
    else:
        shapes = _index_shapes(text, value_types) or rank_words
    unit, unit_reason = parameter_unit(entity_class, parameter)
    return ParameterRecord(
        entity_class=entity_class,
        parameter=parameter,
        description=definition.description,
        definition=text,
        unit=unit,
        unit_reason=unit_reason,
        default=definition.default,
        value_list=schema.value_lists.get(definition.value_list, ()) if definition.value_list else None,
        declared_shapes=declared,
        allowed_shapes=shapes,
        group=definition.group,
        quantity_type=_quantity_type(entity_class, parameter),
    )


def parameter_records() -> list[ParameterRecord]:
    """Every parameter of the shipped schema: by entity class in the schema's class order, then in the schema's
    parameter order."""
    schema = shipped_schema()
    return [parameter_record(cls, p.name) for cls in schema.classes for p in schema.class_parameters(cls)]


@functools.cache
def group_colours() -> dict[str, str]:
    """``{parameter group: hex colour}`` of the shipped schema's ``parameter_groups``."""
    from flextool._resources import package_data_path

    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        data = json.load(f)
    return {row[0]: str(row[1]) for row in data.get("parameter_groups", [])}


def default_text(value: object) -> str:
    """The schema default as a cell shows it (``""`` for none; ``1`` for ``1.0``)."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def default_cell(value: object) -> object:
    """The schema default as an Excel cell: a number stays a number (``1`` for ``1.0``), else text."""
    if isinstance(value, bool) or value is None or not isinstance(value, (int, float)):
        return default_text(value)
    return int(value) if isinstance(value, float) and value.is_integer() else value


# ---------------------------------------------------------------------------
# The docs page


def escape_cell(text: str) -> str:
    """*text* for one Markdown table cell: ``|`` escaped, line breaks as spaces."""
    return " ".join(text.split()).replace("|", r"\|")


def render_parameter_catalogue(markdown_root: Path | None = None) -> str:
    """The parameter catalogue page (Markdown).  *markdown_root* is not read: the page comes from the code."""
    del markdown_root
    recs = parameter_records()
    out = [
        f"# {PAGE_TITLE}",
        "",
        "Every input parameter of this FlexTool version, generated from its database schema: one table per",
        "entity class. The topic pages of [Model parameters](reference/index.md) explain what the parameters do",
        "and how they work together; the [parameter groups](reference/parameter_groups.md) page explains the",
        "group labels.",
        "",
        "- *Unit* is the unit of the default convention (MW, MWh, CUR for the model currency, tonnes of CO2).",
        "  FlexTool does not check units: a model may use others consistently. *dimensionless* is a ratio or a",
        "  share; *MW or count* is a number of units when `virtual_unitsize` is above 0.",
        "- *Default* is the default of the schema. When FlexTool runs it may derive or default other values;",
        "  the results show what the model used.",
        "- *Allowed shapes* says what a value may be: a constant, or a map (an indexed table) over periods, over",
        "  time, or over both (one or two index levels); a stochastic-time map has three levels (branch,",
        "  analysis_time, time). A map whose index is something else names it: *a map: constraint name →",
        "  coefficient*. Where FlexTool checks the shape, this is what it accepts; otherwise it comes from the",
        "  description or the value types of the schema (*a map with n index levels*).",
        "- The same table as an Excel data dictionary, with the parameters a database uses:",
        "  `python -m flextool.docs_build --xlsx catalogue.xlsx --input-db input_data.sqlite`",
        "  (see [Make a data dictionary for your model](how_to/outputs.md#how-to-make-a-data-dictionary-for-your-model)).",
        "",
    ]
    by_class: dict[str, list[ParameterRecord]] = {}
    for r in recs:
        by_class.setdefault(r.entity_class, []).append(r)
    for cls, rows in by_class.items():
        out += [f"## {cls}", ""]
        description = class_description(cls)
        if description:
            out += [escape_cell(description), ""]
        out += ["| parameter | unit | allowed shapes | default | allowed values | group | definition |",
                "|---|---|---|---|---|---|---|"]
        for r in rows:
            cells = [f"`{r.parameter}`", r.unit or "", r.allowed_shapes, default_text(r.default),
                     ", ".join(r.value_list or ()), r.group or "", r.definition]
            out.append("| " + " | ".join(escape_cell(c) for c in cells) + " |")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Database usage (keys only)


class CatalogueInputError(ValueError):
    """The input database or the scenario of ``--input-db`` / ``--scenario`` cannot be used."""


@dataclass(frozen=True)
class Usage:
    entities: int
    alternatives: tuple[str, ...]


@dataclass(frozen=True)
class DatabaseUsage:
    """The parameters a database gives values to: ``{(class, parameter): Usage}`` and the database version."""

    used: dict[tuple[str, str], Usage]
    version: object


def _read_only_url(path: Path) -> str:
    """The SQLite URL that opens *path* read-only (an SQLite URI: the file is never created or written)."""
    return read_only_sqlite_url_string(path)


def input_db_url(value: str) -> str:
    """The URL :func:`database_usage` opens for ``--input-db`` *value*: a file path, or a URL.  A path or a
    ``sqlite:`` URL must name an existing file (``CatalogueInputError`` "no such file") and is opened read-only;
    another URL (a database server) is used as it is."""
    if "://" not in value:
        path = Path(value).expanduser().resolve()
    elif value.startswith("sqlite:"):
        from sqlalchemy.engine.url import make_url

        try:
            database = make_url(value).database
        except Exception:  # noqa: BLE001 — sqlalchemy's ArgumentError for a malformed URL
            raise CatalogueInputError(f"cannot parse the database URL {value!r}") from None
        if not database or database.startswith("file:"):
            return value
        path = Path(database).expanduser().resolve()
    else:
        return value
    if not path.is_file():
        raise CatalogueInputError(f"no such file: {path}")
    return _read_only_url(path)


def database_usage(url: str, scenario: str | None = None) -> DatabaseUsage:
    """Which parameters the database at *url* (a file path or a URL, see :func:`input_db_url`) gives a value, how
    many entities and in which alternatives (the alternatives of *scenario* only, when given).  Reads the keys of
    the parameter values, never a value; an SQLite file is opened read-only (no schema upgrade, no migration)."""
    from spinedb_api import from_database

    from flextool.update_flextool.db_handle import database_mapping

    shown = url
    url = input_db_url(url)
    try:
        db = database_mapping(url, create=False, upgrade=False)
    except Exception as exc:  # noqa: BLE001 — SpineDBAPIError, or SQLAlchemy errors of a damaged file
        text = str(exc).strip().splitlines()
        raise CatalogueInputError(f"cannot open {shown}: {text[0] if text else type(exc).__name__}") from None
    with db:
        classes = {row.id: row.name for row in db.query(db.entity_class_sq.c.id, db.entity_class_sq.c.name)}
        definitions = {row.id: row.name for row in db.query(db.parameter_definition_sq.c.id,
                                                             db.parameter_definition_sq.c.name)}
        alternatives = {row.id: row.name for row in db.query(db.alternative_sq.c.id, db.alternative_sq.c.name)}
        selected: set[int] | None = None
        if scenario is not None:
            scenarios = {row.name: row.id for row in db.query(db.scenario_sq.c.id, db.scenario_sq.c.name)}
            if scenario not in scenarios:
                names = ", ".join(sorted(scenarios)) or "none"
                raise CatalogueInputError(f"the database has no scenario {scenario!r} (its scenarios: {names})")
            sa = db.scenario_alternative_sq
            selected = {row.alternative_id for row in
                        db.query(sa.c.alternative_id).filter(sa.c.scenario_id == scenarios[scenario])}
        pv = db.parameter_value_sq
        entities: dict[tuple[str, str], set[int]] = {}
        alts: dict[tuple[str, str], set[str]] = {}
        for row in db.query(pv.c.entity_class_id, pv.c.parameter_definition_id, pv.c.entity_id,
                            pv.c.alternative_id).distinct():
            if selected is not None and row.alternative_id not in selected:
                continue
            key = (classes[row.entity_class_id], definitions[row.parameter_definition_id])
            entities.setdefault(key, set()).add(row.entity_id)
            alts.setdefault(key, set()).add(alternatives[row.alternative_id])
        version = None
        pd = db.parameter_definition_sq
        ec = db.entity_class_sq
        hit = (db.query(pd.c.default_value, pd.c.default_type).join(ec, ec.c.id == pd.c.entity_class_id)
               .filter(ec.c.name == "model", pd.c.name == "version").first())
        if hit is not None and hit.default_value is not None:
            version = from_database(hit.default_value, hit.default_type)
    db.close()
    used = {key: Usage(len(ids), tuple(sorted(alts[key]))) for key, ids in entities.items()}
    return DatabaseUsage(used=used, version=version)


def version_mismatch(version: object) -> str | None:
    """A one-line note when the database version is not this FlexTool's, else ``None``."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    if version is None:
        return "the database has no model.version: it may not be a FlexTool input database"
    try:
        number = float(version)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"the database version {version!r} is not a number"
    if number < FLEXTOOL_DB_VERSION:
        return (f"the database is at version {default_text(number)} and this FlexTool at {FLEXTOOL_DB_VERSION}: "
                "renamed parameters show under their old names as 'not in this FlexTool version'; "
                "flextool-migrate-database <file> brings the database up to date")
    if number > FLEXTOOL_DB_VERSION:
        return (f"the database is at version {default_text(number)}, newer than this FlexTool "
                f"({FLEXTOOL_DB_VERSION}): update FlexTool to see its parameters")
    return None


# ---------------------------------------------------------------------------
# The Excel data dictionary


def catalogue_rows(usage: DatabaseUsage | None = None) -> list[list[object]]:
    """The rows of the data dictionary (header excluded), in schema order; with *usage*, the database columns,
    and the parameters the database has but this FlexTool does not define last."""
    rows: list[list[object]] = []
    for r in parameter_records():
        row: list[object] = [r.entity_class, r.parameter, r.definition, r.unit or "", r.quantity_type or "",
                             r.allowed_shapes, default_cell(r.default), ", ".join(r.value_list or ()), r.group or ""]
        if usage is not None:
            u = usage.used.get((r.entity_class, r.parameter))
            row += ["yes", u.entities, ", ".join(u.alternatives)] if u else ["", "", ""]
        rows.append(row + [""] * len(TEAM_COLUMNS))
    if usage is not None:
        known = shipped_schema().parameters
        for (cls, param), u in sorted(usage.used.items()):
            if (cls, param) not in known:
                rows.append([cls, param, NOT_IN_THIS_VERSION] + [""] * (len(COLUMNS) - 3)
                            + ["yes", u.entities, ", ".join(u.alternatives)] + [""] * len(TEAM_COLUMNS))
    return rows


def header(with_db: bool) -> list[str]:
    return [*COLUMNS, *(DB_COLUMNS if with_db else ()), *TEAM_COLUMNS]


#: Column widths (characters) of the data dictionary.
_WIDTHS = {"class": 18, "parameter": 30, "definition": 70, "unit": 14, "quantity type": 16, "allowed shapes": 30,
           "default": 12, "allowed values": 30, "parameter group": 16, "used in DB": 11, "entities": 9,
           "alternatives": 24, "Source": 28, "How to replace": 28, "Notes": 28}


#: What each column of the parameters sheet holds, for the Read me sheet.
COLUMN_HELP: dict[str, str] = {
    "class": "The kind of thing the parameter belongs to (unit, node, connection, ...). A name with __ joins "
             "classes: unit__outputNode is a unit and the node it feeds. The classes sheet explains each one.",
    "parameter": "The parameter name, exactly as in the FlexTool input database.",
    "definition": "What the parameter means and how FlexTool uses it (the schema description).",
    "unit": "The unit of measure of the default convention: MW, MWh, CUR (the model currency), tonnes of CO2. "
            "FlexTool does not check units: a model may use others consistently. 'dimensionless' is a ratio or a "
            "share; 'MW or count' is a number of units when virtual_unitsize is above 0.",
    "quantity type": "The kind of quantity (power, energy, price per energy, fraction, ...), the same for every "
                     "parameter measured alike. Useful to filter, for example, all prices.",
    "allowed shapes": "What a value may be: a constant, or a map (an indexed table) over periods, over time, or "
                      "both (1 or 2 index levels). A stochastic-time map has 3 levels (branch, analysis_time, "
                      "time). A map with another index names it, e.g. 'a map: constraint name -> coefficient'.",
    "default": "The value FlexTool uses when the database has none. When FlexTool runs it may derive other "
               "values; the results show what the model used.",
    "allowed values": "For a choice parameter (a method), the values it accepts.",
    "parameter group": "The group the parameter belongs to (the colour as in the FlexTool interface).",
    "used in DB": "'yes' when the input database gives this parameter a value for at least one entity.",
    "entities": "How many entities (units, nodes, ...) have a value in the input database.",
    "alternatives": "The alternatives of the input database that hold those values.",
    "Source": "For your team: where your value comes from (a study, a statistics office, a utility, an "
              "assumption), with the year and a reference you can find again.",
    "How to replace": "For your team: how the value is updated when better data comes (who provides it, which "
                      "file or method, how often), so the next person can repeat it.",
    "Notes": "For your team: anything else, such as the quality of the data or open questions.",
}


def _readme_rows(columns: list[str], usage: DatabaseUsage | None, input_db: str | None,
                 scenario: str | None) -> list[tuple[str, str]]:
    from flextool.engine._behaviour_changes import _installed_version
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    rows: list[tuple[str, str]] = [
        ("FlexTool parameter catalogue", ""),
        ("", ""),
        ("What this file is", "A data dictionary: every input parameter of this FlexTool version, one row each, "
                              "with its meaning, unit, allowed shapes and default. It was written from FlexTool's "
                              "own database schema, so it matches the installed version. It is not an input "
                              "template: FlexTool does not read this file."),
        ("How to use it", "Filter the parameters sheet (the header has filters; the class and parameter columns "
                          "stay visible when you scroll right). Fill in the Source, How to replace and Notes "
                          "columns for the parameters your model uses, and keep the file with the model."),
        ("Web version", WEB_PAGE_URL),
        ("FlexTool version", _installed_version()),
        ("Database version", str(FLEXTOOL_DB_VERSION)),
        ("Made with", "python -m flextool.docs_build --xlsx FILE [--input-db DATABASE [--scenario NAME]]"),
    ]
    if input_db is not None:
        rows.append(("Input database", input_db))
        if scenario is not None:
            rows.append(("Scenario", f"{scenario} (only the values in this scenario's alternatives are counted; "
                                     "an entity that the scenario switches off still counts)"))
        if usage is not None:
            rows.append(("Its version", default_text(usage.version) if usage.version is not None else "none"))
            note = version_mismatch(usage.version)
            if note:
                rows.append(("Note", note))
    rows += [("", ""), ("Columns of the parameters sheet", "")]
    rows += [(name, COLUMN_HELP[name]) for name in columns]
    return rows


def write_xlsx(path: Path, usage: DatabaseUsage | None = None, *, input_db: str | None = None,
               scenario: str | None = None) -> Path:
    """Write the data dictionary to *path*: a Read me sheet (what the file is, the versions, the input database
    of *usage*, one line per column), the parameters sheet (a formatted header row, frozen with the class and
    parameter columns, an autofilter, wrapped text, the group colour as the fill of the group cell and empty
    project columns Source, How to replace, Notes) and the classes sheet (the description of each class)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    columns = header(usage is not None)
    wb = Workbook()
    wrap = Alignment(wrap_text=True, vertical="top")
    bold = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="DDDDDD")

    readme = wb.active
    readme.title = README_SHEET
    for label, text in _readme_rows(columns, usage, input_db, scenario):
        readme.append([label, text])
        readme.cell(row=readme.max_row, column=1).font = bold
        readme.cell(row=readme.max_row, column=2).alignment = wrap
    readme["A1"].font = Font(bold=True, size=14)
    readme.column_dimensions["A"].width = 30
    readme.column_dimensions["B"].width = 100

    ws = wb.create_sheet(SHEET)
    ws.append(columns)
    for cell in ws[1]:
        cell.font = bold
        cell.fill = head_fill
        cell.alignment = wrap
    colours = group_colours()
    group_col = columns.index("parameter group") + 1
    for values in catalogue_rows(usage):
        ws.append(values)
        row = ws.max_row
        for cell in ws[row]:
            cell.alignment = wrap
        colour = colours.get(str(values[group_col - 1]))
        if colour:
            ws.cell(row=row, column=group_col).fill = PatternFill("solid", fgColor=colour.lstrip("#").upper())
    for i, name in enumerate(columns, start=1):
        ws.column_dimensions[get_column_letter(i)].width = _WIDTHS.get(name, 16)
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(columns))}{ws.max_row}"

    classes = wb.create_sheet(CLASSES_SHEET)
    classes.append(["class", "description", "parameters"])
    for cell in classes[1]:
        cell.font = bold
        cell.fill = head_fill
    schema = shipped_schema()
    for cls in schema.classes:
        classes.append([cls, class_description(cls), len(schema.class_parameters(cls))])
        classes.cell(row=classes.max_row, column=2).alignment = wrap
    for letter, width in (("A", 24), ("B", 100), ("C", 11)):
        classes.column_dimensions[letter].width = width
    classes.freeze_panes = "A2"

    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
