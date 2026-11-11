"""The parameters as this FlexTool defines them: the shipped ``schemas/spinedb_schema.json`` (P0-D9).

``describe_parameter``, the units and the notes read the shipped schema, not the user's database, so they describe
the parameters of the installed FlexTool even for a database that is older or newer.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaParameter:
    entity_class: str
    name: str
    default: object                  # as stored in the schema file (a number, a string or None)
    value_list: str | None
    description: str
    group: str | None


@dataclass(frozen=True)
class SchemaClass:
    name: str
    dimensions: tuple[str, ...]
    description: str | None
    active_by_default: bool


@dataclass(frozen=True)
class ShippedSchema:
    classes: dict[str, SchemaClass]
    parameters: dict[tuple[str, str], SchemaParameter]           # (class, parameter)
    value_types: dict[tuple[str, str], tuple[tuple[str, int], ...]]  # (class, parameter) -> ((type, map rank), ...)
    value_lists: dict[str, tuple[str, ...]]

    def class_parameters(self, entity_class: str) -> list[SchemaParameter]:
        return [p for (c, _), p in self.parameters.items() if c == entity_class]


@functools.cache
def shipped_schema() -> ShippedSchema:
    from flextool._resources import package_data_path

    with open(package_data_path("schemas/spinedb_schema.json"), encoding="utf-8") as f:
        data = json.load(f)
    classes = {row[0]: SchemaClass(row[0], tuple(row[1] or ()), row[2], bool(row[4]))
               for row in data["entity_classes"]}
    parameters = {(row[0], row[1]): SchemaParameter(row[0], row[1], row[2], row[3], row[4] or "", row[5])
                  for row in data["parameter_definitions"]}
    types: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for cls, name, type_name, rank in data["parameter_types"]:
        types.setdefault((cls, name), []).append((type_name, int(rank)))
    lists: dict[str, list[str]] = {}
    for name, value in data["parameter_value_lists"]:
        lists.setdefault(name, []).append(value)
    return ShippedSchema(
        classes=classes,
        parameters=parameters,
        value_types={k: tuple(v) for k, v in types.items()},
        value_lists={k: tuple(v) for k, v in lists.items()},
    )


def bracket(description: str | None) -> str | None:
    """The unit bracket a description starts with (``"[MW]"``), or ``None``."""
    if not description or not description.startswith("[") or "]" not in description:
        return None
    return description[: description.index("]") + 1]


def ranks_in_words(value_types: tuple[tuple[str, int], ...]) -> str:
    """What the schema's ``parameter_types`` allow, for a user: ``a constant, or a map with 1 or 3 index levels``."""
    parts: list[str] = []
    names = {t for t, _ in value_types}
    if names & {"float", "str", "bool"}:
        parts.append("a constant")
    if "array" in names:
        parts.append("a list of values (array)")
    if "time_series" in names:
        parts.append("a time series")
    map_ranks = sorted({r for t, r in value_types if t == "map"})
    if map_ranks:
        levels = " or ".join(str(r) for r in map_ranks)
        parts.append(f"a map with {levels} index level{'' if map_ranks == [1] else 's'}")
    return ", or ".join(parts) if parts else "any value"
