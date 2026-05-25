"""Helpers for building parameter values for FlexTool model data.

Uses spinedb_api Map/Array objects for complex values. Provides two output modes:
- Raw Python objects for direct import via spinedb_api.import_data()
- Base64-encoded JSON for writing importable JSON files
"""

import base64
import json
from typing import Any

from spinedb_api import Map, Array, to_database


def make_map(
    keys: list[str], values: list[float | str], index_name: str | None = None
) -> Map:
    """Build a spinedb_api Map (rank-1)."""
    return Map(keys, values, index_name=index_name)


def make_array(values: list[str], index_name: str | None = None) -> Array:
    """Build a spinedb_api Array."""
    return Array(values, index_name=index_name)


def make_time_series(
    timestep_values: list[float], num_timesteps: int | None = None
) -> Map:
    """Build a time series Map with t0001..tNNNN keys."""
    n = num_timesteps or len(timestep_values)
    keys = [f"t{i+1:04d}" for i in range(n)]
    return Map(keys, list(timestep_values[:n]))


def param_value(
    entity_class: str,
    entity_name: str | list[str],
    parameter: str,
    value: Any,
    alternative: str,
) -> list:
    """Build a single parameter_values entry with raw Python value."""
    return [entity_class, entity_name, parameter, value, alternative]


def encode_for_json(value: Any) -> list:
    """Encode a value to [base64_string, type_string] for JSON file output."""
    db_value, db_type = to_database(value)
    b64 = base64.b64encode(db_value).decode("utf-8")
    return [b64, db_type]


def model_data_to_json_serializable(data: dict) -> dict:
    """Convert model data with raw Python values to JSON-serializable format.

    Encodes all parameter values from raw objects (Map, Array, float, str)
    to [base64, type] tuples suitable for JSON output.
    """
    result = dict(data)
    if "parameter_values" in result:
        encoded_pvs = []
        for pv in result["parameter_values"]:
            entity_class, entity_name, param, value, alt = pv
            encoded = encode_for_json(value)
            encoded_pvs.append([entity_class, entity_name, param, encoded, alt])
        result["parameter_values"] = encoded_pvs
    return result
