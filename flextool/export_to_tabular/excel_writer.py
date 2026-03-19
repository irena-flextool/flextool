"""Write FlexTool data to Excel sheets based on SheetSpec configurations.

Each ``write_*`` function populates one openpyxl worksheet according to
its layout type (constant, periodic, timeseries, link, scenario, version,
navigate).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from openpyxl.styles import Font, PatternFill
from openpyxl.styles.colors import Color
from openpyxl.worksheet.worksheet import Worksheet

from spinedb_api import Map, Array

from flextool.export_to_tabular.db_reader import DatabaseContents
from openpyxl.styles import Protection

from flextool.export_to_tabular.formatting import (
    add_navigate_link,
    auto_column_width,
    format_constant_sheet,
    format_constant_sheet_v2,
    format_link_sheet,
    format_link_sheet_v2,
    format_periodic_sheet,
    format_periodic_sheet_v2,
    format_timeseries_sheet,
    format_timeseries_sheet_v2,
    FILL_ENTITY_HEADER,
    FILL_DESC_ROW,
    FILL_PARAM_HEADER,
    FONT_DESC_ROW,
    FONT_NAVIGATE_LINK,
)
from flextool.export_to_tabular.sheet_config import SheetSpec


def _to_native(value: Any) -> Any:
    """Convert numpy scalar types to native Python types for openpyxl."""
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.str_, np.bytes_)):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _array_to_str(arr: Array) -> str:
    """Convert an Array value to a comma-separated string."""
    return ", ".join(str(_to_native(v)) for v in arr.values)


def _is_map(value: Any) -> bool:
    """Check if a value is a Map."""
    return isinstance(value, Map)


def _is_array(value: Any) -> bool:
    """Check if a value is an Array."""
    return isinstance(value, Array)


def _is_scalar(value: Any) -> bool:
    """Check if a value is a scalar (float, int, str) — not Map or Array."""
    return isinstance(value, (int, float, str, np.floating, np.integer, np.str_))


class IndexClassifier:
    """Classifies Map index values as time-indexed or period-indexed.

    Built once from DatabaseContents by collecting known timestep names
    (from timeline entities) and period names (from solve entities).
    """

    def __init__(self, db_contents: DatabaseContents) -> None:
        self.time_indexes: set[str] = set()
        self.period_indexes: set[str] = set()

        for (cls, _byname, pname, _alt), val in db_contents.parameter_values.items():
            if not isinstance(val, Map):
                continue
            # Timeline timestep_duration maps contain all timestep names as indexes
            if cls == "timeline" and pname == "timestep_duration":
                self.time_indexes.update(str(_to_native(idx)) for idx in val.indexes)
            # Solve parameters with Map values use period names as indexes
            if cls == "solve" and pname != "solver_arguments":
                self.period_indexes.update(str(_to_native(idx)) for idx in val.indexes)

    def is_time_indexed(self, value: Any) -> bool:
        """Check if a Map value contains time-series data (should go on _t sheet).

        Classification order:
        1. Explicit index_name == "time" → True
        2. Explicit index_name == "period" → False
        3. Cross-reference indexes with known timesteps/periods from the DB
        4. Fallback: large index count (>100) suggests time-series
        """
        if not isinstance(value, Map) or len(value.indexes) == 0:
            return False
        iname = getattr(value, "index_name", "")
        if iname == "time":
            return True
        if iname == "period":
            return False

        # Cross-reference with known indexes
        map_idxs = {str(_to_native(idx)) for idx in value.indexes}
        has_time_overlap = bool(map_idxs & self.time_indexes)
        has_period_overlap = bool(map_idxs & self.period_indexes)

        if has_time_overlap and not has_period_overlap:
            return True
        if has_period_overlap and not has_time_overlap:
            return False

        # Both or neither overlap — use count as last resort
        return len(map_idxs) > 100


# Module-level classifier instance, set by export_to_excel before writing sheets
_index_classifier: IndexClassifier | None = None


def _is_time_indexed_map(value: Any) -> bool:
    """Check if a Map value contains time-series data (should go on _t sheet, not _p)."""
    if _index_classifier is not None:
        return _index_classifier.is_time_indexed(value)
    # Fallback if classifier not initialized
    if not isinstance(value, Map) or len(value.indexes) == 0:
        return False
    return getattr(value, "index_name", "") == "time"


def _get_extra_columns_for_entity(
    entity_byname: tuple,
    entity_class_name: str,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> list[str | None]:
    """Look up extra entity column values for a given entity.

    For example, for connection_c with extra_entity_class='connection__node__node',
    finds the matching multi-dim entity and extracts the extra dimension values.

    Returns a list of values for each extra_entity_column, or Nones if not found.
    """
    if not spec.extra_entity_columns or not spec.extra_entity_class:
        return []

    extra_entities = db_contents.entities.get(spec.extra_entity_class, [])
    # The main entity name is the first element (e.g. 'ConnBat' for connection)
    main_entity_name = entity_byname[0] if entity_byname else None

    # Find the extra entity that has this main entity as its first element
    for extra_ent in extra_entities:
        extra_byname = extra_ent["entity_byname"]
        if extra_byname[0] == main_entity_name:
            # Return the extra dimensions (skip the first which is the main entity)
            extra_vals = [str(_to_native(v)) for v in extra_byname[1:]]
            # Pad or truncate to match extra_entity_columns length
            while len(extra_vals) < len(spec.extra_entity_columns):
                extra_vals.append(None)
            return extra_vals[: len(spec.extra_entity_columns)]

    return [None] * len(spec.extra_entity_columns)


def _find_alternatives_for_entity(
    entity_class: str,
    entity_byname: tuple,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> list[str]:
    """Find all alternatives that have parameter values or entity_alternatives for an entity."""
    alts: set[str] = set()

    # From parameter values
    for (cls, byname, _param, alt), _val in db_contents.parameter_values.items():
        if cls == entity_class and byname == entity_byname:
            if _param in spec.parameter_names:
                alts.add(alt)

    # From entity alternatives
    for (cls, byname, alt), _active in db_contents.entity_alternatives.items():
        if cls == entity_class and byname == entity_byname:
            alts.add(alt)

    return sorted(alts)


def _get_direction(entity_class: str, spec: SheetSpec) -> str | None:
    """Get the direction value for an entity class from the spec's direction_map."""
    if spec.direction_column and spec.direction_map:
        return spec.direction_map.get(entity_class)
    return None


# ---------------------------------------------------------------------------
# Constant sheet
# ---------------------------------------------------------------------------


def write_constant_sheet(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a constant-layout sheet."""
    # Build header columns
    headers: list[str] = ["alternative"]
    headers.extend(spec.entity_columns)
    headers.extend(spec.extra_entity_columns)
    # Pre-EA params (e.g. has_balance for node_c) go before Entity Alternative
    if spec.pre_ea_params:
        headers.extend(spec.pre_ea_params)
    if spec.has_entity_alternative:
        headers.append("Entity Alternative")
    if spec.direction_column:
        headers.append(spec.direction_column)
    if spec.unpack_index_column:
        headers.append(spec.unpack_index_column)

    # Parameter columns (pre_ea_params already in headers, not repeated here)
    param_headers: list[str] = list(spec.parameter_names)

    all_headers = headers + param_headers

    # --- Row 1: descriptions ---
    ws.cell(row=1, column=1, value="navigate")
    n_fixed = len(headers)
    for col_idx, hdr in enumerate(all_headers, start=1):
        if hdr == "Entity Alternative":
            ws.cell(row=1, column=col_idx, value="Whether the entity is active in this alternative.")
        elif hdr in spec.descriptions:
            ws.cell(row=1, column=col_idx, value=spec.descriptions[hdr])

    # --- Row 2: headers ---
    for col_idx, hdr in enumerate(all_headers, start=1):
        ws.cell(row=2, column=col_idx, value=hdr)

    # --- Collect data rows ---
    data_rows: list[list[Any]] = []

    if spec.unpack_index_column:
        _collect_unpack_rows(data_rows, spec, db_contents, headers, param_headers)
    else:
        _collect_constant_rows(data_rows, spec, db_contents, headers, param_headers)

    # --- Sort rows ---
    data_rows.sort(key=lambda r: tuple(str(v) if v is not None else "" for v in r))

    # --- Write data rows ---
    for row_idx, row_data in enumerate(data_rows, start=3):
        for col_idx, value in enumerate(row_data, start=1):
            if value is not None:
                ws.cell(row=row_idx, column=col_idx, value=value)

    # --- Formatting ---
    n_entity_cols = len(spec.entity_columns)
    n_extra = len(spec.extra_entity_columns) + len(spec.pre_ea_params)
    if spec.has_entity_alternative:
        n_extra += 1
    if spec.direction_column:
        n_extra += 1
    if spec.unpack_index_column:
        n_extra += 1

    format_constant_sheet(ws, n_entity_cols, n_extra, has_entity_alt=False)
    add_navigate_link(ws)
    auto_column_width(ws)


def _collect_constant_rows(
    data_rows: list[list[Any]],
    spec: SheetSpec,
    db_contents: DatabaseContents,
    headers: list[str],
    param_headers: list[str],
) -> None:
    """Collect rows for a regular (non-unpack) constant sheet."""
    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])
        direction = _get_direction(entity_class, spec)

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                row: list[Any] = [alt]

                # Entity elements
                for elem in entity_byname:
                    row.append(str(_to_native(elem)))

                # Extra entity columns
                if spec.extra_entity_columns:
                    extras = _get_extra_columns_for_entity(
                        entity_byname, entity_class, spec, db_contents
                    )
                    row.extend(extras)

                # Pre-EA params (e.g. has_balance for node_c)
                for pname in spec.pre_ea_params:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_scalar(value):
                        row.append(_to_native(value))
                    else:
                        row.append(None)

                # Entity Alternative (after entity cols + pre-EA, before direction)
                if spec.has_entity_alternative:
                    ea_key = (entity_class, entity_byname, alt)
                    ea_val = db_contents.entity_alternatives.get(ea_key)
                    row.append(ea_val if ea_val is not None else None)

                # Direction
                if spec.direction_column:
                    row.append(direction)

                # Parameter values — check for Arrays that need expansion
                param_values: list[Any] = []
                has_array = False
                array_param_idx: int | None = None
                array_elements: list[Any] = []

                for pi, pname in enumerate(spec.parameter_names):
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)

                    if value is None:
                        param_values.append(None)
                    elif _is_map(value):
                        # Maps go on _p or _t sheets, not constant
                        param_values.append(None)
                    elif _is_array(value):
                        has_array = True
                        array_param_idx = pi
                        array_elements = [_to_native(v) for v in value.values]
                        param_values.append(None)  # placeholder
                    elif _is_scalar(value):
                        param_values.append(_to_native(value))
                    else:
                        param_values.append(None)

                if has_array and array_elements:
                    # Expand: one row per array element
                    for elem in array_elements:
                        expanded_row = list(row)
                        expanded_params = list(param_values)
                        expanded_params[array_param_idx] = elem
                        expanded_row.extend(expanded_params)
                        data_rows.append(expanded_row)
                else:
                    row.extend(param_values)
                    data_rows.append(row)


def _collect_unpack_rows(
    data_rows: list[list[Any]],
    spec: SheetSpec,
    db_contents: DatabaseContents,
    headers: list[str],
    param_headers: list[str],
) -> None:
    """Collect rows for an unpacked-Map constant sheet (e.g. constraint coefficients)."""
    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])
        direction = _get_direction(entity_class, spec)

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                # Collect all map indexes across all unpack params
                all_indexes: set[str] = set()
                param_maps: dict[str, Map] = {}

                for pname in spec.parameter_names:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        param_maps[pname] = value
                        for idx in value.indexes:
                            all_indexes.add(str(_to_native(idx)))

                if not all_indexes:
                    continue

                for idx_val in sorted(all_indexes):
                    row: list[Any] = [alt]

                    # Entity elements
                    for elem in entity_byname:
                        row.append(str(_to_native(elem)))

                    # Extra entity columns
                    if spec.extra_entity_columns:
                        extras = _get_extra_columns_for_entity(
                            entity_byname, entity_class, spec, db_contents
                        )
                        row.extend(extras)

                    # Pre-EA params
                    for pre_pname in spec.pre_ea_params:
                        pre_key = (entity_class, entity_byname, pre_pname, alt)
                        pre_val = db_contents.parameter_values.get(pre_key)
                        if pre_val is not None and _is_scalar(pre_val):
                            row.append(_to_native(pre_val))
                        else:
                            row.append(None)

                    # Entity Alternative (after entity cols + pre-EA, before direction)
                    if spec.has_entity_alternative:
                        ea_key = (entity_class, entity_byname, alt)
                        ea_val = db_contents.entity_alternatives.get(ea_key)
                        row.append(ea_val if ea_val is not None else None)

                    # Direction
                    if spec.direction_column:
                        row.append(direction)

                    # Unpack index column
                    row.append(idx_val)

                    # Parameter values
                    for pname in spec.parameter_names:
                        m = param_maps.get(pname)
                        if m is not None:
                            # Find the value at this index
                            found = False
                            for mi, mv in zip(m.indexes, m.values):
                                if str(_to_native(mi)) == idx_val:
                                    row.append(_to_native(mv))
                                    found = True
                                    break
                            if not found:
                                row.append(None)
                        else:
                            row.append(None)

                    data_rows.append(row)


# ---------------------------------------------------------------------------
# Periodic sheet
# ---------------------------------------------------------------------------


def write_periodic_sheet(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a periodic-layout sheet."""
    # Determine index column name from Map values
    index_col_name = _find_index_column_name(spec, db_contents, default="period")

    # Build header columns
    headers: list[str] = ["alternative"]
    headers.extend(spec.entity_columns)
    if spec.direction_column:
        headers.append(spec.direction_column)
    headers.append(index_col_name)

    # Parameter columns (no Entity Alternative on periodic sheets)
    all_headers = headers + list(spec.parameter_names)

    # --- Row 1: descriptions ---
    ws.cell(row=1, column=1, value="navigate")
    n_fixed = len(headers)
    for i, pname in enumerate(spec.parameter_names):
        col_idx = n_fixed + 1 + i
        if pname in spec.descriptions:
            ws.cell(row=1, column=col_idx, value=spec.descriptions[pname])

    # --- Row 2: headers ---
    for col_idx, hdr in enumerate(all_headers, start=1):
        ws.cell(row=2, column=col_idx, value=hdr)

    # --- Collect data rows ---
    data_rows: list[list[Any]] = []
    index_col_pos = len(headers) - 1  # 0-based position of index column

    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])
        direction = _get_direction(entity_class, spec)

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                # Collect all index values and map/array data across params
                all_indexes: set[str] = set()
                param_maps: dict[str, Map] = {}
                param_arrays: dict[str, list[str]] = {}

                for pname in spec.parameter_names:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        # Check for nested Map (Map whose values are Maps)
                        is_nested = any(_is_map(mv) for mv in value.values)
                        if is_nested:
                            # Skip nested Maps — they go to solve_period_period
                            continue
                        # Skip time-indexed Maps — they belong on _t sheets
                        if _is_time_indexed_map(value):
                            continue
                        param_maps[pname] = value
                        for idx in value.indexes:
                            all_indexes.add(str(_to_native(idx)))
                    elif value is not None and _is_array(value):
                        # Array = boolean indicator (period names)
                        period_names = [str(_to_native(v)) for v in value.values]
                        param_arrays[pname] = period_names

                if not all_indexes:
                    continue

                for idx_val in sorted(all_indexes):
                    row: list[Any] = [alt]

                    # Entity elements
                    for elem in entity_byname:
                        row.append(str(_to_native(elem)))

                    # Direction
                    if spec.direction_column:
                        row.append(direction)

                    # Index column value
                    row.append(idx_val)

                    # Parameter values
                    for pname in spec.parameter_names:
                        m = param_maps.get(pname)
                        if m is not None:
                            found = False
                            for mi, mv in zip(m.indexes, m.values):
                                if str(_to_native(mi)) == idx_val:
                                    row.append(_to_native(mv))
                                    found = True
                                    break
                            if not found:
                                row.append(None)
                        elif pname in param_arrays:
                            # Boolean: check if current period is in the array
                            if idx_val in param_arrays[pname]:
                                row.append("yes")
                            else:
                                row.append(None)
                        else:
                            row.append(None)

                    data_rows.append(row)

    # --- Sort rows ---
    data_rows.sort(key=lambda r: tuple(str(v) if v is not None else "" for v in r))

    # --- Write data rows ---
    for row_idx, row_data in enumerate(data_rows, start=3):
        for col_idx, value in enumerate(row_data, start=1):
            if value is not None:
                ws.cell(row=row_idx, column=col_idx, value=value)

    # --- Formatting ---
    n_entity_cols = len(spec.entity_columns)
    n_extra = 0
    if spec.direction_column:
        n_extra += 1
    n_extra += 1  # index column counts as extra

    format_periodic_sheet(ws, n_entity_cols, n_extra)
    add_navigate_link(ws)
    auto_column_width(ws)


def _find_index_column_name(
    spec: SheetSpec,
    db_contents: DatabaseContents,
    default: str = "period",
) -> str:
    """Determine the index column name by scanning Map values for index_name.

    Ignores the placeholder 'x' from the DB and falls back to the spec's
    ``index_name_default`` (from export_settings.yaml) or *default*.
    """
    fallback = spec.index_name_default or default
    name_counts: dict[str, int] = {}

    for entity_class in spec.entity_classes:
        for (cls, _byname, pname, _alt), value in db_contents.parameter_values.items():
            if cls == entity_class and pname in spec.parameter_names and _is_map(value):
                iname = value.index_name
                if iname and iname != "x":
                    name_counts[iname] = name_counts.get(iname, 0) + 1

    if name_counts:
        return max(name_counts, key=name_counts.get)  # type: ignore[arg-type]
    return fallback


# ---------------------------------------------------------------------------
# Nested periodic sheet (solve_period_period)
# ---------------------------------------------------------------------------


def write_nested_periodic_sheet(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a nested-periodic-layout sheet for nested Map parameters.

    Handles parameters whose values are Map-of-Map (outer index =
    current_solve_period, inner Map = periods_included -> 'yes').
    """
    # Build header columns
    headers: list[str] = ["alternative"]
    headers.extend(spec.entity_columns)
    headers.append("current_solve_period")
    headers.append("periods_included")

    # Only include params that actually have nested Map values
    nested_params: list[str] = []
    for pname in spec.parameter_names:
        for (cls, _byname, p, _alt), value in db_contents.parameter_values.items():
            if cls in spec.entity_classes and p == pname and _is_map(value):
                if any(_is_map(mv) for mv in value.values):
                    nested_params.append(pname)
                    break

    if not nested_params:
        # No nested data — write minimal headers
        ws.cell(row=1, column=1, value="navigate")
        all_headers = headers + list(spec.parameter_names)
        for col_idx, hdr in enumerate(all_headers, start=1):
            ws.cell(row=2, column=col_idx, value=hdr)
        add_navigate_link(ws)
        auto_column_width(ws)
        return

    all_headers = headers + nested_params

    # --- Row 1: descriptions ---
    ws.cell(row=1, column=1, value="navigate")
    n_fixed = len(headers)
    for i, pname in enumerate(nested_params):
        col_idx = n_fixed + 1 + i
        if pname in spec.descriptions:
            ws.cell(row=1, column=col_idx, value=spec.descriptions[pname])

    # --- Row 2: headers ---
    for col_idx, hdr in enumerate(all_headers, start=1):
        ws.cell(row=2, column=col_idx, value=hdr)

    # --- Collect data rows ---
    data_rows: list[list[Any]] = []

    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                # Collect nested Map data for this entity+alt
                nested_data: dict[str, Map] = {}
                all_outer_indexes: set[str] = set()
                all_inner_indexes: dict[str, set[str]] = {}

                for pname in nested_params:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        if any(_is_map(mv) for mv in value.values):
                            nested_data[pname] = value
                            for oi in value.indexes:
                                outer_str = str(_to_native(oi))
                                all_outer_indexes.add(outer_str)

                if not nested_data:
                    continue

                # Collect all (outer, inner) combos across all params
                row_combos: set[tuple[str, str]] = set()
                for pname, outer_map in nested_data.items():
                    for oi, inner_val in zip(outer_map.indexes, outer_map.values):
                        outer_str = str(_to_native(oi))
                        if _is_map(inner_val):
                            for ii in inner_val.indexes:
                                inner_str = str(_to_native(ii))
                                row_combos.add((outer_str, inner_str))

                for outer_idx, inner_idx in sorted(row_combos):
                    row: list[Any] = [alt]

                    # Entity elements
                    for elem in entity_byname:
                        row.append(str(_to_native(elem)))

                    # current_solve_period and periods_included
                    row.append(outer_idx)
                    row.append(inner_idx)

                    # Parameter values
                    for pname in nested_params:
                        outer_map = nested_data.get(pname)
                        if outer_map is not None:
                            # Find the inner map at this outer index
                            found = False
                            for oi, inner_val in zip(outer_map.indexes, outer_map.values):
                                if str(_to_native(oi)) == outer_idx and _is_map(inner_val):
                                    # Find value at inner index
                                    for ii, iv in zip(inner_val.indexes, inner_val.values):
                                        if str(_to_native(ii)) == inner_idx:
                                            row.append(_to_native(iv))
                                            found = True
                                            break
                                    break
                            if not found:
                                row.append(None)
                        else:
                            row.append(None)

                    data_rows.append(row)

    # --- Sort rows ---
    data_rows.sort(key=lambda r: tuple(str(v) if v is not None else "" for v in r))

    # --- Write data rows ---
    for row_idx, row_data in enumerate(data_rows, start=3):
        for col_idx, value in enumerate(row_data, start=1):
            if value is not None:
                ws.cell(row=row_idx, column=col_idx, value=value)

    # --- Formatting ---
    n_entity_cols = len(spec.entity_columns)
    n_extra = 2  # current_solve_period + periods_included

    format_periodic_sheet(ws, n_entity_cols, n_extra)
    add_navigate_link(ws)
    auto_column_width(ws)


# ---------------------------------------------------------------------------
# Timeseries sheet
# ---------------------------------------------------------------------------


def write_timeseries_sheet(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a transposed timeseries-layout sheet."""
    # Collect all data columns: each is (entity_class, entity_byname, param, alt, Map)
    columns: list[tuple[str, tuple, str, str, Map]] = []
    all_time_indexes: set[str] = set()

    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])

        for entity in entities:
            entity_byname = entity["entity_byname"]

            for pname in spec.parameter_names:
                for alt in db_contents.alternatives:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        # Only include time-indexed Maps on timeseries sheets
                        if not _is_time_indexed_map(value):
                            continue
                        columns.append(
                            (entity_class, entity_byname, pname, alt, value)
                        )
                        for idx in value.indexes:
                            all_time_indexes.add(str(_to_native(idx)))

    if not columns:
        # Write minimal headers even if no data
        ws.cell(row=1, column=1, value="navigate")
        ws.cell(row=1, column=2, value="alternative")
        ws.cell(row=2, column=2, value="parameter")
        add_navigate_link(ws)
        auto_column_width(ws)
        return

    # Sort columns by (alt, entity_byname, param)
    columns.sort(key=lambda c: (c[3], c[1], c[2]))

    # Sort time indexes
    sorted_times = sorted(all_time_indexes)

    # Determine header structure
    n_entity_dims = len(spec.entity_columns)
    has_direction = spec.direction_column is not None

    # n_header_rows = 2 (alt, param) + n_entity_dims + (1 if direction)
    n_header_rows = 2 + n_entity_dims
    if has_direction:
        n_header_rows += 1

    # --- Write header rows ---
    # Row 1: 'navigate' | 'alternative' | alt values...
    ws.cell(row=1, column=1, value="navigate")
    ws.cell(row=1, column=2, value="alternative")
    for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
        ws.cell(row=1, column=col_idx, value=alt)

    # Row 2: '' | 'parameter' | param names...
    ws.cell(row=2, column=2, value="parameter")
    for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
        ws.cell(row=2, column=col_idx, value=pname)

    # Entity dimension rows
    for dim_idx in range(n_entity_dims):
        row = 3 + dim_idx
        dim_label = spec.entity_columns[dim_idx]
        ws.cell(row=row, column=2, value=dim_label)
        for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
            if dim_idx < len(byname):
                ws.cell(row=row, column=col_idx, value=str(_to_native(byname[dim_idx])))

    # Direction row (if applicable)
    if has_direction:
        dir_row = 2 + n_entity_dims + 1
        ws.cell(row=dir_row, column=2, value=spec.direction_column)
        for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
            direction = _get_direction(ec, spec)
            if direction:
                ws.cell(row=dir_row, column=col_idx, value=direction)

    # 'time' label on the last header row, column A
    ws.cell(row=n_header_rows, column=1, value="time")

    # --- Write time data rows ---
    # Build index lookup for each column's Map for fast access
    col_index_maps: list[dict[str, Any]] = []
    for ec, byname, pname, alt, m in columns:
        idx_map: dict[str, Any] = {}
        for mi, mv in zip(m.indexes, m.values):
            idx_map[str(_to_native(mi))] = _to_native(mv)
        col_index_maps.append(idx_map)

    for time_row_idx, time_val in enumerate(sorted_times):
        row = n_header_rows + 1 + time_row_idx
        ws.cell(row=row, column=1, value=time_val)
        for col_idx, idx_map in enumerate(col_index_maps):
            value = idx_map.get(time_val)
            if value is not None:
                ws.cell(row=row, column=col_idx + 3, value=value)

    # --- Formatting ---
    format_timeseries_sheet(ws, n_header_rows)
    add_navigate_link(ws)
    auto_column_width(ws)


# ---------------------------------------------------------------------------
# Link sheet
# ---------------------------------------------------------------------------


def write_link_sheet(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a link-only (relationship) sheet."""
    # Row 1: headers = entity dimension names
    for col_idx, col_name in enumerate(spec.entity_columns, start=1):
        ws.cell(row=1, column=col_idx, value=col_name)

    # Collect and sort entity rows
    entity_class = spec.entity_classes[0] if spec.entity_classes else None
    if not entity_class:
        format_link_sheet(ws)
        auto_column_width(ws)
        return

    entities = db_contents.entities.get(entity_class, [])
    rows: list[tuple] = []
    for entity in entities:
        byname = entity["entity_byname"]
        rows.append(tuple(str(_to_native(v)) for v in byname))

    rows.sort()

    for row_idx, row_data in enumerate(rows, start=2):
        for col_idx, value in enumerate(row_data, start=1):
            ws.cell(row=row_idx, column=col_idx, value=value)

    format_link_sheet(ws)
    auto_column_width(ws)


# ---------------------------------------------------------------------------
# Scenario sheet
# ---------------------------------------------------------------------------


def write_scenario_sheet(
    ws: Worksheet,
    db_contents: DatabaseContents,
    include_stochastics: bool = False,
) -> None:
    """Write the scenario sheet with formatting.

    Args:
        ws: Target worksheet.
        db_contents: Database contents.
        include_stochastics: If True, add a 'stochastic' row after alternatives.
    """
    scenarios = db_contents.scenarios
    font_dark = Font(color=Color(theme=1, tint=0.0))

    if not scenarios:
        ws.cell(row=1, column=1, value="navigate")
        ws.cell(row=1, column=2, value="Scenario names")
        add_navigate_link(ws)
        auto_column_width(ws)
        return

    # Find max number of alternatives
    max_alts = max(len(sc["alternatives"]) for sc in scenarios) if scenarios else 0

    # Minimum 15 alternative rows (base_alternative + alternative_1..15)
    min_alt_rows = 16  # base_alternative + 15 alternative_N rows
    total_alt_rows = max(min_alt_rows, max_alts)

    # Row 1: 'navigate' | 'Scenario names'
    ws.cell(row=1, column=1, value="navigate")
    cell_b1 = ws.cell(row=1, column=2, value="Scenario names")
    cell_b1.font = font_dark

    # Row 2: '' | scenario names
    for col_idx, sc in enumerate(scenarios, start=2):
        cell = ws.cell(row=2, column=col_idx, value=sc["name"])
        cell.fill = FILL_ENTITY_HEADER
        cell.font = font_dark

    # Row 3: 'base_alternative' | alt names (rank 0)
    cell_a3 = ws.cell(row=3, column=1, value="base_alternative")
    cell_a3.fill = FILL_PARAM_HEADER
    cell_a3.font = font_dark
    for col_idx, sc in enumerate(scenarios, start=2):
        alts = sc["alternatives"]
        if alts:
            cell = ws.cell(row=3, column=col_idx, value=alts[0][0])
            cell.font = font_dark

    # Rows 4+: 'alternative_N' | alt names — always write at least 15 rows
    for alt_idx in range(1, total_alt_rows):
        row = 3 + alt_idx
        cell_label = ws.cell(row=row, column=1, value=f"alternative_{alt_idx}")
        cell_label.fill = FILL_PARAM_HEADER
        cell_label.font = font_dark
        for col_idx, sc in enumerate(scenarios, start=2):
            alts = sc["alternatives"]
            if alt_idx < len(alts):
                cell = ws.cell(row=row, column=col_idx, value=alts[alt_idx][0])
                cell.font = font_dark

    # Stochastic row (after alternatives)
    if include_stochastics:
        stochastic_row = 3 + total_alt_rows
        cell_stoch = ws.cell(row=stochastic_row, column=1, value="stochastic")
        cell_stoch.fill = FILL_PARAM_HEADER
        cell_stoch.font = font_dark

    add_navigate_link(ws)
    auto_column_width(ws)


# ---------------------------------------------------------------------------
# Version sheet
# ---------------------------------------------------------------------------


def write_version_sheet(
    ws: Worksheet,
    version: float | None,
) -> None:
    """Write the version sheet."""
    if version is not None:
        version_int = int(version) if version == int(version) else version
        ws.cell(row=1, column=1, value=f"Generated from FlexTool sqlite version: {version_int}")
    else:
        ws.cell(row=1, column=1, value="Generated from FlexTool sqlite")


# ---------------------------------------------------------------------------
# Navigate sheet
# ---------------------------------------------------------------------------


def write_navigate_sheet(
    ws: Worksheet,
    all_specs: list[SheetSpec],
    navigate_groups: list[dict[str, Any]] | None = None,
) -> None:
    """Write the navigate sheet with grouped, colour-coded hyperlinks.

    Args:
        ws: Target worksheet.
        all_specs: All sheet specifications (used for fallback if no groups).
        navigate_groups: Navigate group configuration from export_settings.yaml.
    """
    # Build set of sheet names that actually exist in the workbook specs
    existing_sheets: set[str] = {spec.sheet_name for spec in all_specs if spec.layout != "navigate"}

    if not navigate_groups:
        # Fallback: flat 3-column list (original behaviour)
        _write_navigate_flat(ws, all_specs, existing_sheets)
        return

    # --- Row 1: column headers ---
    ws.cell(row=1, column=1, value="Constants")
    ws.cell(row=1, column=2, value="Periodic")
    ws.cell(row=1, column=3, value="Timeseries")

    # --- Column E: help text ---
    help_lines = [
        "FlexTool can take parameter data in three ways:",
        "- constants (sheet ends with '_c') - Almost any value can be a constant or can be set as a time series.",
        "- period series (sheets ends with '_p') - These parameters have a period index (e.g. year).",
        "- time series (sheets ends with '_t') - These parameters can have values for each timestep.",
        "",
        "Some sheets establish only relationships between entities without additional parameters.",
        "",
        "Parameters with valid types (drop-down lists) show allowed values when you select the cell.",
    ]
    for i, text in enumerate(help_lines):
        if text:
            ws.cell(row=1 + i, column=5, value=text)

    # --- Write grouped rows ---
    current_row = 2  # start after header row

    for group in navigate_groups:
        group_color = group["color"]
        group_fill = PatternFill(patternType="solid", fgColor=group_color)
        rows_in_group = group.get("rows", [])

        for sheet_row in rows_in_group:
            # Place each sheet name into columns A, B, C in order
            for col_offset, sheet_name in enumerate(sheet_row):
                if sheet_name in existing_sheets:
                    col = col_offset + 1  # 1-based
                    cell = ws.cell(row=current_row, column=col, value=sheet_name)
                    cell.hyperlink = f"#{sheet_name}!A1"
                    cell.font = FONT_NAVIGATE_LINK

            # Apply group fill to columns A, B, C for this row
            for col in range(1, 4):
                ws.cell(row=current_row, column=col).fill = group_fill

            current_row += 1

        # Blank separator row after each group
        current_row += 1

    auto_column_width(ws)


def _write_navigate_flat(
    ws: Worksheet,
    all_specs: list[SheetSpec],
    existing_sheets: set[str],
) -> None:
    """Fallback: write a flat 3-column navigate sheet (original behaviour)."""
    constants: list[str] = []
    periodic: list[str] = []
    timeseries: list[str] = []

    for spec in all_specs:
        if spec.layout == "navigate":
            continue
        elif spec.layout in ("periodic", "nested_periodic"):
            periodic.append(spec.sheet_name)
        elif spec.layout == "timeseries":
            timeseries.append(spec.sheet_name)
        else:
            constants.append(spec.sheet_name)

    ws.cell(row=1, column=1, value="Constants")
    ws.cell(row=1, column=2, value="Periodic")
    ws.cell(row=1, column=3, value="Timeseries")

    for row_idx, name in enumerate(constants, start=2):
        cell = ws.cell(row=row_idx, column=1, value=name)
        cell.hyperlink = f"#{name}!A1"
        cell.font = FONT_NAVIGATE_LINK

    for row_idx, name in enumerate(periodic, start=2):
        cell = ws.cell(row=row_idx, column=2, value=name)
        cell.hyperlink = f"#{name}!A1"
        cell.font = FONT_NAVIGATE_LINK

    for row_idx, name in enumerate(timeseries, start=2):
        cell = ws.cell(row=row_idx, column=3, value=name)
        cell.hyperlink = f"#{name}!A1"
        cell.font = FONT_NAVIGATE_LINK

    auto_column_width(ws)


# ===========================================================================
# v2 format writers — self-describing sheets with embedded metadata
# ===========================================================================


def _get_param_data_type(
    param_name: str,
    entity_classes: list[str],
    db_contents: DatabaseContents,
) -> str:
    """Determine the data type label for a parameter ('string' or 'float').

    Checks the parameter_type_list from the DB: if it contains 'str', returns
    'string'; otherwise returns 'float'.
    """
    for entity_class in entity_classes:
        for pdef in db_contents.parameter_definitions.get(entity_class, []):
            if pdef["name"] == param_name:
                type_list = pdef.get("parameter_type_list")
                if type_list and "str" in type_list:
                    return "string"
                return "float"
    return "float"


def _get_param_description(
    param_name: str,
    entity_classes: list[str],
    db_contents: DatabaseContents,
) -> str | None:
    """Get the description for a parameter from DB definitions."""
    for entity_class in entity_classes:
        for pdef in db_contents.parameter_definitions.get(entity_class, []):
            if pdef["name"] == param_name:
                return pdef.get("description") or None
    return None


def _build_entity_def_label(spec: SheetSpec) -> str:
    """Build the 'entity: ...' definition label for the definition row.

    For single-class single-dim: 'entity: node'
    For single-class multi-dim:  'entity: commodity, node'
    For merged classes (direction_column): 'entity: (cls1: (dim1, dim2), cls2: (dim1, dim2))'
    """
    if spec.direction_column and len(spec.entity_classes) > 1:
        # Merged classes with direction — use explicit class mapping
        parts: list[str] = []
        for cls_name in spec.entity_classes:
            dims = spec.entity_columns
            if len(dims) == 1:
                parts.append(f"{cls_name}: {dims[0]}")
            else:
                dim_str = ", ".join(dims)
                parts.append(f"{cls_name}: ({dim_str})")
        return f"entity: ({', '.join(parts)})"
    else:
        # Simple: single class or multi-dim
        return f"entity: {', '.join(spec.entity_columns)}"


def _build_filter_label(spec: SheetSpec) -> str | None:
    """Build the 'filter: ...' label for merged classes with direction.

    Returns 'filter: (cls1: ^input$, cls2: ^output$)' or None.
    """
    if not spec.direction_column or not spec.direction_map:
        return None
    parts: list[str] = []
    for cls_name, direction_val in spec.direction_map.items():
        parts.append(f"{cls_name}: ^{direction_val}$")
    return f"filter: ({', '.join(parts)})"


def _get_all_param_defs_for_class(
    entity_classes: list[str],
    db_contents: DatabaseContents,
) -> list[dict]:
    """Get all parameter definitions across entity classes (deduplicated by name)."""
    seen: set[str] = set()
    result: list[dict] = []
    for entity_class in entity_classes:
        for pdef in db_contents.parameter_definitions.get(entity_class, []):
            if pdef["name"] not in seen:
                seen.add(pdef["name"])
                result.append(pdef)
    result.sort(key=lambda p: p["name"])
    return result


def _write_param_reference(
    ws: Worksheet,
    start_col: int,
    spec: SheetSpec,
    db_contents: DatabaseContents,
    header_row: int = 3,
) -> None:
    """Write the parameter reference section after an empty separator column.

    Places a 3-row header (Parameter / Description / Data type) followed by
    one row per available parameter for the entity class, starting at
    ``start_col + 1`` (the +1 is the empty separator).

    Args:
        ws: Target worksheet.
        start_col: The column after the last data column (separator column).
        spec: Sheet specification.
        db_contents: Database contents.
        header_row: Row number for the definition/header row (3 for v2).
    """
    all_pdefs = _get_all_param_defs_for_class(spec.entity_classes, db_contents)
    if not all_pdefs:
        return

    # The reference section starts after the empty separator column
    ref_col = start_col + 1

    # Write reference header in the description row (row 1 for v2)
    desc_row = header_row - 2  # row 1
    ws.cell(row=desc_row, column=ref_col, value="Parameter")
    ws.cell(row=desc_row, column=ref_col + 1, value="Data type")
    ws.cell(row=desc_row, column=ref_col + 2, value="Description")

    # Apply description row formatting to the header
    for c in range(ref_col, ref_col + 3):
        cell = ws.cell(row=desc_row, column=c)
        cell.fill = FILL_DESC_ROW
        cell.font = FONT_DESC_ROW

    # Write each parameter starting from the data type row (row 2 for v2)
    dtype_row = header_row - 1  # row 2
    for i, pdef in enumerate(all_pdefs):
        row = dtype_row + i
        ws.cell(row=row, column=ref_col, value=pdef["name"])
        type_list = pdef.get("parameter_type_list")
        dtype = "string" if (type_list and "str" in type_list) else "float"
        ws.cell(row=row, column=ref_col + 1, value=dtype)
        desc = pdef.get("description")
        if desc:
            ws.cell(row=row, column=ref_col + 2, value=desc)


def _lock_metadata_cells(
    ws: Worksheet,
    meta_rows: int,
    meta_cols: int,
    data_start_row: int,
    data_start_col: int,
    max_data_rows: int = 500,
) -> None:
    """Apply sheet protection, locking metadata cells and unlocking data cells.

    Uses column-level unlocking for efficiency: sets entire data columns as
    unlocked rather than individual cells.  For the metadata columns in the
    data region, only the metadata rows stay locked (the default).

    Skips protection entirely when the data region exceeds *max_data_rows*
    to avoid bloating the file with per-cell styles.

    Args:
        ws: Target worksheet.
        meta_rows: Number of metadata rows (rows 1..meta_rows are locked).
        meta_cols: Number of metadata columns (columns 1..meta_cols are locked in data rows).
        data_start_row: First row of the data region.
        data_start_col: First column of the data region.
        max_data_rows: Skip protection if data exceeds this many rows.
    """
    # TODO: Re-enable sheet protection once LibreOffice compatibility is verified.
    # The current openpyxl set_password("") creates a hash that can crash
    # LibreOffice on some systems.  Disabled for now.
    return


# ---------------------------------------------------------------------------
# v2 Constant sheet
# ---------------------------------------------------------------------------


def write_constant_sheet_v2(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a constant-layout sheet in the v2 self-describing format.

    Layout:
        Row 1: navigate | (empty) | description | desc/param descriptions...
        Row 2: (empty)  | (empty) | data type   | string/float...
        Row 3: alternative | entity: X | [filter:] [index:] parameter | [entity existence] | param names...
        Row 4+: data
    """
    # Determine the definition column position
    n_entity_cols = len(spec.entity_columns)
    n_extra_entity = len(spec.extra_entity_columns)
    n_dims = n_entity_cols

    # Build left-side column labels for row 3
    left_cols: list[str] = ["alternative"]
    entity_label = _build_entity_def_label(spec)
    left_cols.append(entity_label)

    # For multi-dim entities, add ALL individual dim name columns after the
    # entity label (the entity label column has no data in data rows)
    if n_dims > 1:
        for dim_name in spec.entity_columns:
            left_cols.append(dim_name)

    # Extra entity columns (e.g. left_node, right_node for connection)
    for extra_col in spec.extra_entity_columns:
        left_cols.append(extra_col)

    # Filter column for direction
    filter_label = _build_filter_label(spec)
    if filter_label:
        left_cols.append(filter_label)

    # Unpack index column
    index_col_positions: set[int] = set()
    if spec.unpack_index_column:
        left_cols.append(f"index: {spec.unpack_index_column}")
        index_col_positions.add(len(left_cols))  # 1-based

    def_col = len(left_cols) + 1  # 1-based position of definition column

    # Build right-side columns (from def_col onward)
    right_cols: list[str] = ["parameter"]
    if spec.has_entity_alternative:
        right_cols.append("entity existence")
    # All parameters (pre_ea_params + parameter_names)
    all_params = list(spec.pre_ea_params) + list(spec.parameter_names)
    right_cols.extend(all_params)

    # --- Row 1: navigate + descriptions (from def_col onward) ---
    ws.cell(row=1, column=1, value="navigate")
    ws.cell(row=1, column=def_col, value="description")
    ea_offset = 1 if spec.has_entity_alternative else 0
    if spec.has_entity_alternative:
        ws.cell(row=1, column=def_col + 1, value="Entity existence")
    for i, pname in enumerate(all_params):
        col = def_col + 1 + ea_offset + i
        desc = spec.descriptions.get(pname)
        if not desc:
            desc = _get_param_description(pname, spec.entity_classes, db_contents)
        if desc:
            ws.cell(row=1, column=col, value=desc)

    # --- Row 2: data types (from def_col onward) ---
    ws.cell(row=2, column=def_col, value="data type")
    if spec.has_entity_alternative:
        ws.cell(row=2, column=def_col + 1, value="string")
    for i, pname in enumerate(all_params):
        col = def_col + 1 + ea_offset + i
        dtype = _get_param_data_type(pname, spec.entity_classes, db_contents)
        ws.cell(row=2, column=col, value=dtype)

    # --- Row 3: definition row ---
    for col_idx, label in enumerate(left_cols, start=1):
        ws.cell(row=3, column=col_idx, value=label)
    for col_idx, label in enumerate(right_cols, start=def_col):
        ws.cell(row=3, column=col_idx, value=label)

    # --- Map column positions for data writing ---
    col_alt = 1

    if n_dims <= 1:
        entity_data_cols = [2]
        next_col_after_entity = 3
    else:
        # Multi-dim: entity label in col 2 (no data in data rows),
        # dims in cols 3..2+n_dims
        entity_data_cols = list(range(3, 3 + n_dims))
        next_col_after_entity = 3 + n_dims

    extra_entity_data_cols = list(
        range(next_col_after_entity, next_col_after_entity + n_extra_entity)
    )
    next_col = next_col_after_entity + n_extra_entity

    filter_data_col: int | None = None
    if filter_label:
        filter_data_col = next_col
        next_col += 1

    unpack_data_col: int | None = None
    if spec.unpack_index_column:
        unpack_data_col = next_col
        next_col += 1

    # def_col should match
    assert next_col == def_col, f"Column mismatch: expected def_col={def_col}, got {next_col}"

    ea_data_col: int | None = None
    param_data_start = def_col + 1
    if spec.has_entity_alternative:
        ea_data_col = def_col + 1
        param_data_start = def_col + 2

    param_col_map: dict[str, int] = {}
    for i, pname in enumerate(all_params):
        param_col_map[pname] = param_data_start + i

    # --- Collect data rows ---
    data_rows: list[list[tuple[int, Any]]] = []

    if spec.unpack_index_column:
        _collect_unpack_rows_v2(
            data_rows, spec, db_contents,
            col_alt=col_alt,
            entity_data_cols=entity_data_cols,
            extra_entity_data_cols=extra_entity_data_cols,
            filter_data_col=filter_data_col,
            unpack_data_col=unpack_data_col,
            ea_data_col=ea_data_col,
            param_col_map=param_col_map,
        )
    else:
        _collect_constant_rows_v2(
            data_rows, spec, db_contents,
            col_alt=col_alt,
            entity_data_cols=entity_data_cols,
            extra_entity_data_cols=extra_entity_data_cols,
            filter_data_col=filter_data_col,
            ea_data_col=ea_data_col,
            param_col_map=param_col_map,
        )

    # --- Sort rows ---
    data_rows.sort(key=lambda r: tuple(
        str(v) if v is not None else "" for _, v in r
    ))

    # --- Write data rows (starting at row 4) ---
    for row_idx, row_cells in enumerate(data_rows, start=4):
        for col, value in row_cells:
            if value is not None:
                ws.cell(row=row_idx, column=col, value=value)

    # --- Write parameter reference section ---
    last_data_col = def_col + len(right_cols) - 1
    _write_param_reference(ws, last_data_col + 1, spec, db_contents, header_row=3)

    # --- Formatting ---
    n_extra = (
        n_extra_entity
        + (1 if filter_label else 0)
        + (1 if spec.unpack_index_column else 0)
    )
    format_constant_sheet_v2(ws, n_entity_cols, n_extra, def_col, index_col_positions)
    add_navigate_link(ws)
    auto_column_width(ws)

    # --- Lock metadata ---
    _lock_metadata_cells(
        ws,
        meta_rows=3,
        meta_cols=def_col,
        data_start_row=4,
        data_start_col=1,
    )


def _collect_constant_rows_v2(
    data_rows: list[list[tuple[int, Any]]],
    spec: SheetSpec,
    db_contents: DatabaseContents,
    *,
    col_alt: int,
    entity_data_cols: list[int],
    extra_entity_data_cols: list[int],
    filter_data_col: int | None,
    ea_data_col: int | None,
    param_col_map: dict[str, int],
) -> None:
    """Collect rows for a v2 constant sheet (non-unpack)."""
    all_params = list(spec.pre_ea_params) + list(spec.parameter_names)

    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])
        direction = _get_direction(entity_class, spec)

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                cells: list[tuple[int, Any]] = []

                # Alternative
                cells.append((col_alt, alt))

                # Entity dimension values
                for i, col in enumerate(entity_data_cols):
                    if i < len(entity_byname):
                        cells.append((col, str(_to_native(entity_byname[i]))))

                # Extra entity columns
                if spec.extra_entity_columns:
                    extras = _get_extra_columns_for_entity(
                        entity_byname, entity_class, spec, db_contents
                    )
                    for i, col in enumerate(extra_entity_data_cols):
                        if i < len(extras):
                            cells.append((col, extras[i]))

                # Direction/filter value
                if filter_data_col is not None and direction is not None:
                    cells.append((filter_data_col, direction))

                # Entity existence
                if ea_data_col is not None:
                    ea_key = (entity_class, entity_byname, alt)
                    ea_val = db_contents.entity_alternatives.get(ea_key)
                    if ea_val is not None:
                        cells.append((ea_data_col, ea_val))

                # Parameter values — check for Arrays
                param_values: dict[str, Any] = {}
                has_array = False
                array_param: str | None = None
                array_elements: list[Any] = []

                for pname in all_params:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)

                    if value is None:
                        param_values[pname] = None
                    elif _is_map(value):
                        param_values[pname] = None
                    elif _is_array(value):
                        has_array = True
                        array_param = pname
                        array_elements = [_to_native(v) for v in value.values]
                        param_values[pname] = None
                    elif _is_scalar(value):
                        param_values[pname] = _to_native(value)
                    else:
                        param_values[pname] = None

                if has_array and array_elements and array_param is not None:
                    for elem in array_elements:
                        expanded_cells = list(cells)
                        for pname in all_params:
                            col = param_col_map[pname]
                            if pname == array_param:
                                expanded_cells.append((col, elem))
                            elif param_values[pname] is not None:
                                expanded_cells.append((col, param_values[pname]))
                        data_rows.append(expanded_cells)
                else:
                    for pname in all_params:
                        if param_values[pname] is not None:
                            cells.append((param_col_map[pname], param_values[pname]))
                    data_rows.append(cells)


def _collect_unpack_rows_v2(
    data_rows: list[list[tuple[int, Any]]],
    spec: SheetSpec,
    db_contents: DatabaseContents,
    *,
    col_alt: int,
    entity_data_cols: list[int],
    extra_entity_data_cols: list[int],
    filter_data_col: int | None,
    unpack_data_col: int | None,
    ea_data_col: int | None,
    param_col_map: dict[str, int],
) -> None:
    """Collect rows for a v2 unpacked-Map constant sheet."""
    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])
        direction = _get_direction(entity_class, spec)

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                # Collect all map indexes across all unpack params
                all_indexes: set[str] = set()
                param_maps: dict[str, Map] = {}

                for pname in spec.parameter_names:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        param_maps[pname] = value
                        for idx in value.indexes:
                            all_indexes.add(str(_to_native(idx)))

                if not all_indexes:
                    continue

                for idx_val in sorted(all_indexes):
                    cells: list[tuple[int, Any]] = []

                    # Alternative
                    cells.append((col_alt, alt))

                    # Entity dimension values
                    for i, col in enumerate(entity_data_cols):
                        if i < len(entity_byname):
                            cells.append((col, str(_to_native(entity_byname[i]))))

                    # Extra entity columns
                    if spec.extra_entity_columns:
                        extras = _get_extra_columns_for_entity(
                            entity_byname, entity_class, spec, db_contents
                        )
                        for i, col in enumerate(extra_entity_data_cols):
                            if i < len(extras):
                                cells.append((col, extras[i]))

                    # Direction/filter value
                    if filter_data_col is not None and direction is not None:
                        cells.append((filter_data_col, direction))

                    # Unpack index value
                    if unpack_data_col is not None:
                        cells.append((unpack_data_col, idx_val))

                    # Entity existence
                    if ea_data_col is not None:
                        ea_key = (entity_class, entity_byname, alt)
                        ea_val = db_contents.entity_alternatives.get(ea_key)
                        if ea_val is not None:
                            cells.append((ea_data_col, ea_val))

                    # Parameter values from maps
                    for pname in spec.parameter_names:
                        m = param_maps.get(pname)
                        if m is not None:
                            for mi, mv in zip(m.indexes, m.values):
                                if str(_to_native(mi)) == idx_val:
                                    cells.append((param_col_map[pname], _to_native(mv)))
                                    break

                    # Pre-EA scalar params (these are not maps)
                    for pname in spec.pre_ea_params:
                        key = (entity_class, entity_byname, pname, alt)
                        value = db_contents.parameter_values.get(key)
                        if value is not None and _is_scalar(value):
                            cells.append((param_col_map[pname], _to_native(value)))

                    data_rows.append(cells)


# ---------------------------------------------------------------------------
# v2 Periodic sheet
# ---------------------------------------------------------------------------


def write_periodic_sheet_v2(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a periodic-layout sheet in the v2 self-describing format.

    Layout:
        Row 1: navigate | | | description | param descriptions...
        Row 2: | | | data type | float...
        Row 3: alternative | entity: X | [filter:] index: period | parameter | param names...
        Row 4+: data
    """
    # Determine index column name
    index_col_name = _find_index_column_name(spec, db_contents, default="period")

    n_entity_cols = len(spec.entity_columns)
    n_dims = n_entity_cols

    # Build left-side columns for row 3
    left_cols: list[str] = ["alternative"]
    entity_label = _build_entity_def_label(spec)
    left_cols.append(entity_label)

    # For multi-dim, add ALL individual dim name columns after entity label
    if n_dims > 1:
        for dim_name in spec.entity_columns:
            left_cols.append(dim_name)

    # Direction/filter column
    filter_label = _build_filter_label(spec)
    if filter_label:
        left_cols.append(filter_label)

    # Index column (period)
    left_cols.append(f"index: {index_col_name}")
    index_col_pos = len(left_cols)  # 1-based

    def_col = len(left_cols) + 1

    # Right-side columns
    right_cols: list[str] = ["parameter"]
    right_cols.extend(spec.parameter_names)

    # --- Row 1: navigate + descriptions ---
    ws.cell(row=1, column=1, value="navigate")
    ws.cell(row=1, column=def_col, value="description")
    for i, pname in enumerate(spec.parameter_names):
        col = def_col + 1 + i
        desc = spec.descriptions.get(pname)
        if not desc:
            desc = _get_param_description(pname, spec.entity_classes, db_contents)
        if desc:
            ws.cell(row=1, column=col, value=desc)

    # --- Row 2: data types ---
    ws.cell(row=2, column=def_col, value="data type")
    for i, pname in enumerate(spec.parameter_names):
        col = def_col + 1 + i
        dtype = _get_param_data_type(pname, spec.entity_classes, db_contents)
        ws.cell(row=2, column=col, value=dtype)

    # --- Row 3: definition row ---
    for col_idx, label in enumerate(left_cols, start=1):
        ws.cell(row=3, column=col_idx, value=label)
    for col_idx, label in enumerate(right_cols, start=def_col):
        ws.cell(row=3, column=col_idx, value=label)

    # --- Map column positions ---
    col_alt = 1

    if n_dims <= 1:
        entity_data_cols = [2]
        next_col = 3
    else:
        entity_data_cols = list(range(3, 3 + n_dims))
        next_col = 3 + n_dims

    filter_data_col: int | None = None
    if filter_label:
        filter_data_col = next_col
        next_col += 1

    index_data_col = next_col
    next_col += 1

    assert next_col == def_col, f"Column mismatch: expected def_col={def_col}, got {next_col}"

    param_data_start = def_col + 1
    param_col_map: dict[str, int] = {}
    for i, pname in enumerate(spec.parameter_names):
        param_col_map[pname] = param_data_start + i

    # --- Collect data rows ---
    data_rows: list[list[tuple[int, Any]]] = []

    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])
        direction = _get_direction(entity_class, spec)

        for entity in entities:
            entity_byname = entity["entity_byname"]
            alts = _find_alternatives_for_entity(
                entity_class, entity_byname, spec, db_contents
            )

            for alt in alts:
                all_indexes: set[str] = set()
                param_maps: dict[str, Map] = {}
                param_arrays: dict[str, list[str]] = {}

                for pname in spec.parameter_names:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        is_nested = any(_is_map(mv) for mv in value.values)
                        if is_nested:
                            continue
                        if _is_time_indexed_map(value):
                            continue
                        param_maps[pname] = value
                        for idx in value.indexes:
                            all_indexes.add(str(_to_native(idx)))
                    elif value is not None and _is_array(value):
                        period_names = [str(_to_native(v)) for v in value.values]
                        param_arrays[pname] = period_names

                if not all_indexes:
                    continue

                for idx_val in sorted(all_indexes):
                    cells: list[tuple[int, Any]] = []

                    cells.append((col_alt, alt))

                    for i, col in enumerate(entity_data_cols):
                        if i < len(entity_byname):
                            cells.append((col, str(_to_native(entity_byname[i]))))

                    if filter_data_col is not None and direction is not None:
                        cells.append((filter_data_col, direction))

                    cells.append((index_data_col, idx_val))

                    for pname in spec.parameter_names:
                        m = param_maps.get(pname)
                        if m is not None:
                            for mi, mv in zip(m.indexes, m.values):
                                if str(_to_native(mi)) == idx_val:
                                    cells.append((param_col_map[pname], _to_native(mv)))
                                    break
                        elif pname in param_arrays:
                            if idx_val in param_arrays[pname]:
                                cells.append((param_col_map[pname], "yes"))

                    data_rows.append(cells)

    # --- Sort rows ---
    data_rows.sort(key=lambda r: tuple(
        str(v) if v is not None else "" for _, v in r
    ))

    # --- Write data rows ---
    for row_idx, row_cells in enumerate(data_rows, start=4):
        for col, value in row_cells:
            if value is not None:
                ws.cell(row=row_idx, column=col, value=value)

    # --- Write parameter reference section ---
    last_data_col = def_col + len(right_cols) - 1
    _write_param_reference(ws, last_data_col + 1, spec, db_contents, header_row=3)

    # --- Formatting ---
    n_extra = (
        (1 if filter_label else 0)
        + 1  # index column
    )
    format_periodic_sheet_v2(ws, n_entity_cols, n_extra, def_col, {index_col_pos})
    add_navigate_link(ws)
    auto_column_width(ws)

    # --- Lock metadata ---
    _lock_metadata_cells(
        ws,
        meta_rows=3,
        meta_cols=def_col,
        data_start_row=4,
        data_start_col=1,
    )


# ---------------------------------------------------------------------------
# v2 Timeseries sheet
# ---------------------------------------------------------------------------


def write_timeseries_sheet_v2(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a transposed timeseries-layout sheet in the v2 self-describing format.

    Layout:
        Row 1: navigate     | alternative      | Base     | Base     | ...
        Row 2:              | parameter: X     |          |          | ...
        Row 3: index: time  | entity: profile  | Wind1    | Battery  | ...
        Row 4+: t0001       |                  | 1.0      | 0.5      | ...

    For multi-param timeseries, 'parameter' is the row 2 label and actual
    param names appear in row 2 data columns. For single-param (like
    profile_t), 'parameter: profile' is used as the label.
    """
    # Collect all data columns
    columns: list[tuple[str, tuple, str, str, Map]] = []
    all_time_indexes: set[str] = set()

    for entity_class in spec.entity_classes:
        entities = db_contents.entities.get(entity_class, [])

        for entity in entities:
            entity_byname = entity["entity_byname"]

            for pname in spec.parameter_names:
                for alt in db_contents.alternatives:
                    key = (entity_class, entity_byname, pname, alt)
                    value = db_contents.parameter_values.get(key)
                    if value is not None and _is_map(value):
                        if not _is_time_indexed_map(value):
                            continue
                        columns.append(
                            (entity_class, entity_byname, pname, alt, value)
                        )
                        for idx in value.indexes:
                            all_time_indexes.add(str(_to_native(idx)))

    if not columns:
        ws.cell(row=1, column=1, value="navigate")
        ws.cell(row=1, column=2, value="alternative")
        ws.cell(row=2, column=2, value="parameter")
        add_navigate_link(ws)
        auto_column_width(ws)
        return

    # Sort columns
    columns.sort(key=lambda c: (c[3], c[1], c[2]))
    sorted_times = sorted(all_time_indexes)

    n_entity_dims = len(spec.entity_columns)
    has_direction = spec.direction_column is not None

    # Determine if single-parameter sheet
    unique_params = set(c[2] for c in columns)
    single_param = len(unique_params) == 1
    default_param = next(iter(unique_params)) if single_param else None

    # n_header_rows: alt row + param row + entity dim row(s) + optional direction
    n_header_rows = 2 + max(1, n_entity_dims)
    if has_direction:
        n_header_rows += 1

    # --- Row 1: 'navigate' | 'alternative' | alt values... ---
    ws.cell(row=1, column=1, value="navigate")
    ws.cell(row=1, column=2, value="alternative")
    for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
        ws.cell(row=1, column=col_idx, value=alt)

    # --- Row 2: '' | 'parameter[: X]' | param names... ---
    if single_param:
        ws.cell(row=2, column=2, value=f"parameter: {default_param}")
    else:
        ws.cell(row=2, column=2, value="parameter")
    for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
        ws.cell(row=2, column=col_idx, value=pname)

    # --- Entity dimension rows ---
    if n_entity_dims <= 1:
        # Single entity dim: row 3 is the entity/time row
        entity_label = _build_entity_def_label(spec)
        ws.cell(row=3, column=1, value="index: time")
        ws.cell(row=3, column=2, value=entity_label)
        for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
            if byname:
                ws.cell(row=3, column=col_idx, value=str(_to_native(byname[0])))
    else:
        # Multiple entity dims
        for dim_idx in range(n_entity_dims):
            row = 3 + dim_idx
            if dim_idx == 0:
                entity_label = _build_entity_def_label(spec)
                ws.cell(row=row, column=2, value=entity_label)
            else:
                ws.cell(row=row, column=2, value=spec.entity_columns[dim_idx])
            for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
                if dim_idx < len(byname):
                    ws.cell(row=row, column=col_idx, value=str(_to_native(byname[dim_idx])))
        ws.cell(row=2 + n_entity_dims, column=1, value="index: time")

    # Direction row
    if has_direction:
        dir_row = 3 + max(1, n_entity_dims)
        ws.cell(row=dir_row, column=2, value=spec.direction_column)
        for col_idx, (ec, byname, pname, alt, _m) in enumerate(columns, start=3):
            direction = _get_direction(ec, spec)
            if direction:
                ws.cell(row=dir_row, column=col_idx, value=direction)

    # Adjust n_header_rows for single dim case
    if n_entity_dims <= 1 and not has_direction:
        n_header_rows = 3

    # --- Write time data rows ---
    col_index_maps: list[dict[str, Any]] = []
    for ec, byname, pname, alt, m in columns:
        idx_map: dict[str, Any] = {}
        for mi, mv in zip(m.indexes, m.values):
            idx_map[str(_to_native(mi))] = _to_native(mv)
        col_index_maps.append(idx_map)

    for time_row_idx, time_val in enumerate(sorted_times):
        row = n_header_rows + 1 + time_row_idx
        ws.cell(row=row, column=1, value=time_val)
        for col_idx, idx_map in enumerate(col_index_maps):
            value = idx_map.get(time_val)
            if value is not None:
                ws.cell(row=row, column=col_idx + 3, value=value)

    # --- Formatting ---
    format_timeseries_sheet_v2(ws, n_header_rows)
    add_navigate_link(ws)
    auto_column_width(ws)

    # --- Lock metadata ---
    _lock_metadata_cells(
        ws,
        meta_rows=n_header_rows,
        meta_cols=2,
        data_start_row=n_header_rows + 1,
        data_start_col=1,
    )


# ---------------------------------------------------------------------------
# v2 Link sheet
# ---------------------------------------------------------------------------


def write_link_sheet_v2(
    ws: Worksheet,
    spec: SheetSpec,
    db_contents: DatabaseContents,
) -> None:
    """Write a link-only (relationship) sheet in the v2 self-describing format.

    Layout:
        Row 1: navigate
        Row 2: 'entity: X, Y' | dim1_name | dim2_name
        Row 3+: (empty)       | dim1_val  | dim2_val
    """
    entity_class = spec.entity_classes[0] if spec.entity_classes else None

    # Row 1: navigate
    ws.cell(row=1, column=1, value="navigate")

    # Row 2: entity definition + dimension headers
    entity_label = _build_entity_def_label(spec)
    ws.cell(row=2, column=1, value=entity_label)
    for col_idx, col_name in enumerate(spec.entity_columns, start=2):
        ws.cell(row=2, column=col_idx, value=col_name)

    if not entity_class:
        format_link_sheet_v2(ws)
        add_navigate_link(ws)
        auto_column_width(ws)
        return

    # Collect and sort entity rows
    entities = db_contents.entities.get(entity_class, [])
    rows: list[tuple] = []
    for entity in entities:
        byname = entity["entity_byname"]
        rows.append(tuple(str(_to_native(v)) for v in byname))

    rows.sort()

    for row_idx, row_data in enumerate(rows, start=3):
        for col_idx, value in enumerate(row_data, start=2):
            ws.cell(row=row_idx, column=col_idx, value=value)

    format_link_sheet_v2(ws)
    add_navigate_link(ws)
    auto_column_width(ws)

    # --- Lock metadata ---
    _lock_metadata_cells(
        ws,
        meta_rows=2,
        meta_cols=1,
        data_start_row=3,
        data_start_col=2,
    )
