"""The parameter catalogue: every input parameter of this FlexTool, as a docs page and as an Excel data dictionary.

Built from the shipped schema (``schemas/spinedb_schema.json``: descriptions, defaults, value lists, parameter
groups and the ``parameter_types`` map ranks) and the engine's shape allow-list (``PARAM_ALLOWED_SHAPES``), through
the same helpers as the assistant's ``describe_parameter`` (``shipped_schema``, ``parameter_unit``,
``allowed_shapes_in_words``, ``ranks_in_words``), so the two cannot disagree; a test pins that.

The allowed shapes of a parameter, in order of preference: the engine's allow-list in words; the schema's map
ranks when they say exactly what is allowed (a constant, a list of values); the description's own shape sentence
("Constant, period or time."); the map ranks in words.

``python -m flextool.docs_build --xlsx FILE [--input-db URL [--scenario NAME]]`` writes the Excel data dictionary
(:func:`write_xlsx`); with ``--input-db`` it marks the parameters a database uses (:func:`database_usage`), read
from the parameter-value keys only (no value is parsed).
"""
from __future__ import annotations

import functools
import json
import re
from dataclasses import dataclass
from pathlib import Path

from flextool.assistant.inputs._schema import bracket, ranks_in_words, shipped_schema
from flextool.assistant.inputs._units import parameter_unit

PAGE_TITLE = "Parameter catalogue"

#: The description's own shape sentence: "Constant, period or time." / "Constant or period.".
_SHAPE_SENTENCE_RE = re.compile(r"(?<![\w-])(Constant[^.]*\.)")

#: The sheet of the Excel data dictionary and its columns.
SHEET = "parameters"
COLUMNS: tuple[str, ...] = ("class", "parameter", "definition", "unit", "allowed shapes", "default",
                            "value list", "parameter group")
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


def _definition(description: str) -> str:
    mark = bracket(description)
    text = description[len(mark):] if mark else description
    return " ".join(text.split())


def _exact_rank_words(value_types: tuple[tuple[str, int], ...]) -> bool:
    """Whether the schema's types say exactly what is allowed: constants and lists of values, no map."""
    return bool(value_types) and all(t in ("float", "str", "bool", "array") for t, _ in value_types)


def parameter_record(entity_class: str, parameter: str) -> ParameterRecord:
    """The record of one parameter; ``KeyError`` when this FlexTool does not define it."""
    from flextool.engine.input._param_shapes import PARAM_ALLOWED_SHAPES, allowed_shapes_in_words

    schema = shipped_schema()
    definition = schema.parameters[(entity_class, parameter)]
    value_types = schema.value_types.get((entity_class, parameter), ())
    allowed = PARAM_ALLOWED_SHAPES.get((entity_class, parameter))
    rank_words = ranks_in_words(value_types)
    declared = allowed_shapes_in_words(allowed) if allowed is not None else rank_words
    text = _definition(definition.description)
    if allowed is not None or _exact_rank_words(value_types):
        shapes = declared
    elif sentence := _SHAPE_SENTENCE_RE.search(text):
        shapes = sentence.group(1)
    else:
        shapes = rank_words
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


# ---------------------------------------------------------------------------
# The docs page


def escape_cell(text: str) -> str:
    """*text* for one Markdown table cell: ``|`` escaped, line breaks as spaces."""
    return " ".join(text.split()).replace("|", r"\|")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "-", text.lower()).strip("-")


def render_parameter_catalogue(markdown_root: Path | None = None) -> str:
    """The parameter catalogue page (Markdown).  *markdown_root* is not read: the page comes from the code."""
    del markdown_root
    schema = shipped_schema()
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
        "- *Allowed shapes* is what the engine accepts where it checks the shape; otherwise the shape the",
        "  description gives, or the value types of the schema (a map with *n* index levels).",
        "- The same table as an Excel data dictionary, with the parameters a database uses:",
        "  `python -m flextool.docs_build --xlsx catalogue.xlsx --input-db sqlite:///input_data.sqlite`",
        "  (see [Make a data dictionary for your model](how_to/outputs.md#how-to-make-a-data-dictionary-for-your-model)).",
        "",
    ]
    by_class: dict[str, list[ParameterRecord]] = {}
    for r in recs:
        by_class.setdefault(r.entity_class, []).append(r)
    for cls, rows in by_class.items():
        out += [f"## {cls}", ""]
        description = schema.classes[cls].description
        if description:
            out += [escape_cell(description), ""]
        out += ["| parameter | unit | allowed shapes | default | choices | group | definition |",
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


def database_usage(url: str, scenario: str | None = None) -> DatabaseUsage:
    """Which parameters the database at *url* gives a value, how many entities and in which alternatives (the
    alternatives of *scenario* only, when given).  Reads the keys of the parameter values, never a value; the
    database is opened read-only (no schema upgrade, no migration)."""
    from spinedb_api import DatabaseMapping, from_database

    try:
        db = DatabaseMapping(url, create=False, upgrade=False)
    except Exception as exc:  # noqa: BLE001 — SpineDBAPIError, or SQLAlchemy errors of a damaged file
        text = str(exc).strip().splitlines()
        raise CatalogueInputError(f"cannot open {url}: {text[0] if text else type(exc).__name__}") from None
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
        row: list[object] = [r.entity_class, r.parameter, r.definition, r.unit or "", r.allowed_shapes,
                             default_text(r.default), ", ".join(r.value_list or ()), r.group or ""]
        if usage is not None:
            u = usage.used.get((r.entity_class, r.parameter))
            row += ["yes", u.entities, ", ".join(u.alternatives)] if u else ["", "", ""]
        rows.append(row + [""] * len(TEAM_COLUMNS))
    if usage is not None:
        known = shipped_schema().parameters
        for (cls, param), u in sorted(usage.used.items()):
            if (cls, param) not in known:
                rows.append([cls, param, NOT_IN_THIS_VERSION, "", "", "", "", "", "yes", u.entities,
                             ", ".join(u.alternatives)] + [""] * len(TEAM_COLUMNS))
    return rows


def header(with_db: bool) -> list[str]:
    return [*COLUMNS, *(DB_COLUMNS if with_db else ()), *TEAM_COLUMNS]


#: Column widths (characters) of the data dictionary.
_WIDTHS = {"class": 18, "parameter": 30, "definition": 70, "unit": 14, "allowed shapes": 30, "default": 12,
           "value list": 30, "parameter group": 16, "used in DB": 11, "entities": 9, "alternatives": 24,
           "Source": 28, "How to replace": 28, "Notes": 28}


def write_xlsx(path: Path, usage: DatabaseUsage | None = None) -> Path:
    """Write the data dictionary to *path*: one sheet, a formatted and frozen header row, an autofilter, wrapped
    text, the group colour as the fill of the group cell and empty project columns (Source, How to replace,
    Notes)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    columns = header(usage is not None)
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET
    ws.append(columns)
    head_fill = PatternFill("solid", fgColor="DDDDDD")
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = head_fill
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    colours = group_colours()
    group_col = columns.index("parameter group") + 1
    wrap = Alignment(wrap_text=True, vertical="top")
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
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
