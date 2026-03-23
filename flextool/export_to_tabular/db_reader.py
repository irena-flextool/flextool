"""Read all data from a FlexTool Spine DB SQLite database into a DatabaseContents dataclass."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from spinedb_api import DatabaseMapping, from_database


@dataclass
class DatabaseContents:
    """All data extracted from a FlexTool Spine DB."""

    entity_classes: list[dict] = field(default_factory=list)
    """Each has 'name', 'dimension_name_list'."""

    entities: dict[str, list[dict]] = field(default_factory=dict)
    """Keyed by entity_class_name -> list of entity dicts."""

    parameter_definitions: dict[str, list] = field(default_factory=dict)
    """Keyed by entity_class_name -> list of param def dicts."""

    parameter_values: dict[tuple, Any] = field(default_factory=dict)
    """(class_name, entity_byname, param_name, alt_name) -> parsed Python value."""

    alternatives: list[str] = field(default_factory=list)
    """Alternative names."""

    scenarios: list[dict] = field(default_factory=list)
    """Each: {'name': str, 'alternatives': [(alt_name, rank), ...]}."""

    entity_alternatives: dict[tuple, bool] = field(default_factory=dict)
    """(class_name, entity_byname, alt_name) -> active (bool)."""

    parameter_groups: dict[str, dict] = field(default_factory=dict)
    """group_name -> {'priority': int, 'color': str}."""

    param_to_group: dict[tuple, str] = field(default_factory=dict)
    """(class_name, param_name) -> group_name."""

    version: float | None = None
    """FlexTool DB version from model.version default."""

    list_values: dict[str, list] = field(default_factory=dict)
    """value_list_name -> [allowed_values]."""


def _convert_numpy(value: Any) -> Any:
    """Convert numpy scalar types to native Python types for openpyxl compatibility."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.str_):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def read_database(db_url: str) -> DatabaseContents:
    """Read a FlexTool Spine DB and return all contents as a DatabaseContents dataclass.

    Args:
        db_url: SQLAlchemy-style database URL, e.g. ``sqlite:///path/to/db.sqlite``.

    Returns:
        A fully populated :class:`DatabaseContents` instance.
    """
    dc = DatabaseContents()
    db = DatabaseMapping(db_url, create=False)
    try:
        db.fetch_all()
        _read_entity_classes(db, dc)
        _read_entities(db, dc)
        _read_parameter_definitions(db, dc)
        _read_parameter_values(db, dc)
        _read_alternatives(db, dc)
        _read_scenarios(db, dc)
        _read_entity_alternatives(db, dc)
        _read_parameter_groups(db, dc)
        _read_param_to_group(dc)
        _read_version(dc)
        _read_list_values(db, dc)
    finally:
        db.close()
    return dc


def _read_entity_classes(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read entity classes: name and dimension_name_list."""
    for item in db.get_entity_class_items():
        dc.entity_classes.append(
            {
                "name": item["name"],
                "dimension_name_list": item["dimension_name_list"],
            }
        )


def _read_entities(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read entities grouped by entity_class_name."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in db.get_entity_items():
        entity_byname = item["entity_byname"]  # always a tuple
        groups[item["entity_class_name"]].append(
            {
                "name": item["name"],
                "entity_byname": entity_byname,
            }
        )
    dc.entities = dict(groups)


def _read_parameter_definitions(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read parameter definitions grouped by entity_class_name."""
    groups: dict[str, list] = defaultdict(list)
    for item in db.get_parameter_definition_items():
        ext = item.extended()
        default_value = None
        if item["default_value"] is not None:
            default_value = from_database(item["default_value"], item["default_type"])
            default_value = _convert_numpy(default_value)
        groups[ext["entity_class_name"]].append(
            {
                "name": item["name"],
                "description": item["description"],
                "parameter_type_list": ext.get("parameter_type_list"),
                "parameter_value_list_name": ext.get("parameter_value_list_name"),
                "default_value": default_value,
                "default_type": item["default_type"],
                "parameter_group_name": ext.get("parameter_group_name"),
            }
        )
    dc.parameter_definitions = dict(groups)


def _read_parameter_values(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read parameter values keyed by (class, entity_byname, param, alt)."""
    for pv in db.get_parameter_value_items():
        value = from_database(pv["value"], pv["type"])
        value = _convert_numpy(value)
        key = (
            pv["entity_class_name"],
            pv["entity_byname"],
            pv["parameter_name"],
            pv["alternative_name"],
        )
        dc.parameter_values[key] = value


def _read_alternatives(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read alternative names."""
    dc.alternatives = [item["name"] for item in db.get_alternative_items()]


def _read_scenarios(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read scenarios with their ranked alternatives."""
    scenario_alts: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for sa in db.get_scenario_alternative_items():
        scenario_alts[sa["scenario_name"]].append(
            (sa["alternative_name"], sa["rank"])
        )

    for sc in db.get_scenario_items():
        name = sc["name"]
        alts = sorted(scenario_alts.get(name, []), key=lambda x: x[1])
        dc.scenarios.append({"name": name, "alternatives": alts})


def _read_entity_alternatives(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read entity alternatives: (class, entity_byname, alt) -> active."""
    for ea in db.get_entity_alternative_items():
        key = (
            ea["entity_class_name"],
            ea["entity_byname"],
            ea["alternative_name"],
        )
        dc.entity_alternatives[key] = bool(ea["active"])


def _read_parameter_groups(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read parameter groups: group_name -> {priority, color}."""
    for pg in db.get_parameter_group_items():
        d = pg._asdict()
        dc.parameter_groups[d["name"]] = {
            "priority": d.get("priority"),
            "color": d.get("color"),
        }


def _read_param_to_group(dc: DatabaseContents) -> None:
    """Build param_to_group from parameter definitions that have parameter_group_name."""
    for class_name, pdefs in dc.parameter_definitions.items():
        for pdef in pdefs:
            group_name = pdef.get("parameter_group_name")
            if group_name:
                dc.param_to_group[(class_name, pdef["name"])] = group_name


def _read_version(dc: DatabaseContents) -> None:
    """Extract FlexTool DB version from model.version default_value."""
    model_defs = dc.parameter_definitions.get("model", [])
    for pdef in model_defs:
        if pdef["name"] == "version":
            dv = pdef["default_value"]
            if dv is not None:
                dc.version = float(dv)
            return


def _read_list_values(db: DatabaseMapping, dc: DatabaseContents) -> None:
    """Read parameter value list contents: list_name -> [values]."""
    groups: dict[str, list] = defaultdict(list)
    for lv in db.get_list_value_items():
        val = from_database(lv["value"], lv["type"])
        val = _convert_numpy(val)
        groups[lv["parameter_value_list_name"]].append(val)
    dc.list_values = dict(groups)
