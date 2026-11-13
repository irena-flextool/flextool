"""Schema sections of the repository's fixture and canonical JSONs, generated from the schema JSON.

Every test fixture (``tests/fixtures/*.json``) and canonical database
(``flextool/schemas/canonical_databases/*.json``) carries its own copy of
the five schema sections in :data:`SCHEMA_SECTIONS`.  A migration only
adds what is absent, so a copy that skipped a change never catches up.
This module makes ``flextool/schemas/spinedb_schema.json`` the single
source: :func:`schema_sections` builds the sections from it through a real
database (so the value encoding is exactly what a database export writes),
and :func:`sync_schema_sections` replaces a fixture's copy with them.

Two value formats are used in the repository:

``packed``
    ``tests/fixtures/*.json``: ``[base64(bytes), type]`` pairs
    (:func:`pack_value` / :func:`unpack_value`).
``decoded``
    canonical databases: ``[json_text, type]`` pairs
    (:func:`flextool.update_flextool.export_database.keep_serialized_parse`).

In the ``packed`` format "no default" is written as SQL NULL
(``[None, None]``), never as the JSON ``null`` blob: migrations add
definitions without a default as SQL NULL, and tests read
``default_value is None`` on them.

The module imports nothing from the ``tests`` package, so it works without
the repository root on ``sys.path``.
"""
from __future__ import annotations

import base64
import contextlib
import functools
import io
import json
import tempfile
from pathlib import Path
from typing import Any, Literal

from spinedb_api import export_data

from flextool._resources import package_data_path
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.export_database import keep_serialized_parse
from flextool.update_flextool.initialize_database import initialize_database
from flextool.sqlite_url import sqlite_url_string

Format = Literal["packed", "decoded"]

#: The sections a fixture copies from the schema; every other section is data.
SCHEMA_SECTIONS: tuple[str, ...] = (
    "entity_classes",
    "parameter_value_lists",
    "parameter_groups",
    "parameter_definitions",
    "parameter_types",
)

#: Normal-form marker of a definition without a default (SQL NULL).
NO_DEFAULT = "<no default>"


class FixtureSchemaError(ValueError):
    """A fixture cannot take the schema sections without losing or misreading data."""


# ---------------------------------------------------------------------------
# Value encodings
# ---------------------------------------------------------------------------


def pack_value(raw_bytes: bytes | None, value_type: str | None) -> list[Any]:
    """Serialize a raw database value to the fixtures' ``[base64, type]`` pair.

    ``raw_bytes`` is ``None`` for a definition without a default (SQL NULL).
    """
    if raw_bytes is None:
        return [None, value_type]
    return [base64.b64encode(raw_bytes).decode("ascii"), value_type]


def unpack_value(packed: Any) -> tuple[bytes | None, str | None]:
    """Reconstruct a raw database value from a ``[base64, type]`` pair."""
    b64, value_type = packed
    if b64 is None:
        return None, value_type
    return base64.b64decode(b64), value_type


def dump_fixture_json(data: dict, path: Path) -> None:
    """Write a ``packed`` fixture JSON in the layout ``tests.db_utils.db_to_json`` writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _decode(value: Any, fmt: Format | Literal["raw"]) -> Any:
    """Decode one stored value to Python; SQL NULL becomes :data:`NO_DEFAULT`.

    ``raw`` is the hand-written schema JSON, where values are plain scalars
    and ``None`` means no default.
    """
    if fmt == "raw":
        return NO_DEFAULT if value is None else value
    if value is None:
        return NO_DEFAULT
    text, _type = value
    if text is None:
        return NO_DEFAULT
    if fmt == "packed":
        return json.loads(base64.b64decode(text))
    return json.loads(text)


# ---------------------------------------------------------------------------
# Schema sections
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _schema_sections_text(fmt: Format) -> str:
    schema_json = Path(package_data_path("schemas/spinedb_schema.json"))
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "schema.sqlite")
        with contextlib.redirect_stdout(io.StringIO()):
            initialize_database(str(schema_json), db_path)
        # Closes the engine too, so Windows can delete the temporary directory.
        with open_database_mapping(sqlite_url_string(db_path)) as db:
            parse = pack_value if fmt == "packed" else keep_serialized_parse
            data = export_data(db, parse_value=parse)
    sections = json.loads(json.dumps({key: data.get(key, []) for key in SCHEMA_SECTIONS}))
    if fmt == "packed":
        for row in sections["parameter_definitions"]:
            if row[2] is not None and _decode(row[2], "packed") is None:
                row[2] = [None, None]
    return json.dumps(sections)


def schema_sections(fmt: Format) -> dict[str, list]:
    """The five schema sections of ``spinedb_schema.json`` in the given value format.

    Built once per process through a temporary database; every call returns
    a fresh copy the caller may modify.
    """
    if fmt not in ("packed", "decoded"):
        raise ValueError(f"unknown fixture value format {fmt!r}")
    return json.loads(_schema_sections_text(fmt))


# ---------------------------------------------------------------------------
# Normal form and differences
# ---------------------------------------------------------------------------


def normalise_sections(data: dict, fmt: Format | Literal["raw"]) -> dict[str, dict]:
    """Order-free, encoding-free view of the five schema sections of ``data``.

    Definition defaults and value-list values are decoded; a definition
    without a default is :data:`NO_DEFAULT` only when stored as SQL NULL, so
    a JSON ``null`` default in a ``packed`` fixture shows up as a difference.
    """
    classes = {row[0]: (tuple(row[1]), *row[2:]) for row in data.get("entity_classes", [])}
    lists: dict[str, list] = {}
    for name, value in data.get("parameter_value_lists", []):
        lists.setdefault(name, []).append(_decode(value, fmt))
    groups = {row[0]: tuple(row[1:]) for row in data.get("parameter_groups", [])}
    definitions = {}
    for row in data.get("parameter_definitions", []):
        row = list(row) + [None] * (6 - len(row))
        definitions[(row[0], row[1])] = {
            "default": _decode(row[2], fmt),
            "value_list": row[3],
            "description": row[4],
            "group": row[5],
        }
    types: dict[tuple, list] = {}
    for row in data.get("parameter_types", []):
        types.setdefault((row[0], row[1]), []).append(tuple(row[2:]))
    return {
        "entity_classes": classes,
        "parameter_value_lists": lists,
        "parameter_groups": groups,
        "parameter_definitions": definitions,
        "parameter_types": {key: sorted(value) for key, value in types.items()},
    }


def _diff_normal(actual: dict, expected: dict) -> list[str]:
    lines = []
    for section in SCHEMA_SECTIONS:
        a, e = actual[section], expected[section]
        for key in sorted(set(a) | set(e), key=repr):
            label = "/".join(key) if isinstance(key, tuple) else key
            if key not in e:
                lines.append(f"{section}: {label}: not in the schema")
            elif key not in a:
                lines.append(f"{section}: {label}: missing")
            elif section == "parameter_definitions":
                for field in ("default", "value_list", "description", "group"):
                    if a[key][field] != e[key][field]:
                        lines.append(f"{section}: {label}: {field}: fixture={a[key][field]!r} "
                                     f"schema={e[key][field]!r}")
            elif a[key] != e[key]:
                lines.append(f"{section}: {label}: fixture={a[key]!r} schema={e[key]!r}")
    return lines


def section_diffs(data: dict, fmt: Format) -> list[str]:
    """Every difference between the schema sections of ``data`` and the schema, one line each.

    Lines read ``<section>: <class/parameter or name>: [<field>: ]...``.
    Empty when the fixture carries exactly the schema's sections.
    """
    return _diff_normal(normalise_sections(data, fmt), normalise_sections(schema_sections(fmt), fmt))


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


def fixture_version(data: dict, fmt: Format) -> int | None:
    """``model.version`` default of a fixture, or ``None`` when it has no version definition."""
    for row in data.get("parameter_definitions", []):
        if row[0] == "model" and row[1] == "version":
            value = _decode(row[2], fmt)
            return None if value in (None, NO_DEFAULT) else int(value)
    return None


def sync_schema_sections(data: dict, fmt: Format) -> dict:
    """Return ``data`` with its five schema sections replaced by the schema's.

    Rules:

    a. A fixture with a ``model.version`` definition must be at
       :data:`FLEXTOOL_DB_VERSION` (migrate it first).  A fixture without
       one is taken as written against the current schema (the sync-only
       path); it gains the version definition from the schema.
    b. A class or definition only the fixture has is dropped when nothing
       uses it; an entity of such a class or a value of such a definition
       raises (a migration is missing).
    c. A value of a definition with a value list must be in the schema's
       list, otherwise raise.
    d. Every other section is returned unchanged, the same objects in the
       same key order; a missing schema section is appended.

    Raises :class:`FixtureSchemaError` for a, b and c.
    """
    version = fixture_version(data, fmt)
    if version is not None and version != FLEXTOOL_DB_VERSION:
        raise FixtureSchemaError(
            f"fixture is at schema version {version}, not {FLEXTOOL_DB_VERSION}: migrate it before the sync")
    sections = schema_sections(fmt)
    classes = {row[0] for row in sections["entity_classes"]}
    definitions = {(row[0], row[1]): row[3] for row in sections["parameter_definitions"]}
    list_values: dict[str, list] = {}
    for name, value in sections["parameter_value_lists"]:
        list_values.setdefault(name, []).append(_decode(value, fmt))

    problems = []
    for row in data.get("entities", []):
        if row[0] not in classes:
            problems.append(f"entity {row[0]} {row[1]!r}: class not in the schema")
    for row in data.get("parameter_values", []):
        cls, byname, name, value, alternative = row[:5]
        where = f"{cls} {byname!r} {name} [{alternative}]"
        if (cls, name) not in definitions:
            problems.append(f"value {where}: parameter not in the schema")
            continue
        value_list = definitions[(cls, name)]
        if value_list is not None:
            parsed = _decode(value, fmt)
            if parsed not in list_values.get(value_list, []):
                problems.append(f"value {where} = {parsed!r}: not in the value list {value_list}")
    if problems:
        raise FixtureSchemaError("fixture data does not fit the schema:\n  " + "\n  ".join(problems))

    out = {key: (sections[key] if key in sections else value) for key, value in data.items()}
    for key in SCHEMA_SECTIONS:
        out.setdefault(key, sections[key])
    return out



def _hashable(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_hashable(item) for item in value)
    return value


def normalise_data(data: dict, fmt: Format) -> dict[str, set]:
    """Order-free view of every non-schema section of ``data``, values decoded.

    A database round trip may reorder rows, so two fixtures hold the same data
    when these sets are equal.
    """
    out = {}
    for key, rows in data.items():
        if key in SCHEMA_SECTIONS:
            continue
        if key == "parameter_values":
            out[key] = {(r[0], _hashable(r[1]), r[2], json.dumps(_decode(r[3], fmt), sort_keys=True),
                         r[3][1] if r[3] else None, *map(_hashable, r[4:])) for r in rows}
        else:
            out[key] = {_hashable(row) for row in rows}
    return out
