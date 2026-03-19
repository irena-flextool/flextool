"""Build SheetSpec objects describing every sheet to generate in the Excel output.

Takes DatabaseContents (from db_reader) and YAML settings to produce an ordered
list of SheetSpec, each encoding the layout, entity columns, parameter columns,
and special rules for one Excel sheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from flextool.export_to_tabular.db_reader import DatabaseContents


@dataclass
class SheetSpec:
    """Description of a single Excel sheet to generate."""

    sheet_name: str
    layout: str  # 'constant', 'periodic', 'timeseries', 'link', 'scenario', 'navigate', 'version'
    entity_classes: list[str]  # DB entity classes that contribute data
    entity_columns: list[str]  # Column header names for entity dimensions
    parameter_names: list[str] = field(default_factory=list)
    direction_column: str | None = None  # e.g. 'input_output' for merged unit__inputNode/outputNode
    direction_map: dict[str, str] = field(default_factory=dict)  # entity_class -> direction value
    extra_entity_columns: list[str] = field(default_factory=list)  # e.g. ['left_node', 'right_node']
    extra_entity_class: str | None = None  # class to get extra elements from
    unpack_index_column: str | None = None  # e.g. 'constraint' for unpacked Map params
    has_entity_alternative: bool = True  # whether to include Entity Alternative column
    pre_ea_params: list[str] = field(default_factory=list)  # params before Entity Alternative column
    index_name_default: str | None = None  # fallback index column name for this layout
    descriptions: dict[str, str] = field(default_factory=dict)  # param_name -> description text


def load_settings() -> dict:
    """Load export_settings.yaml from the same directory as this module."""
    settings_path = Path(__file__).parent / "export_settings.yaml"
    with open(settings_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def classify_param_types(param_type_list: tuple | None) -> set[str]:
    """Given a parameter's type_list tuple, return which sheet layouts it can appear on.

    Returns a set that may contain 'constant', 'periodic', 'timeseries', 'stochastic'.
    """
    if param_type_list is None:
        return {"constant"}

    result: set[str] = set()
    types = set(param_type_list)

    if "float" in types or "str" in types:
        result.add("constant")
    if "array" in types:
        result.add("constant")
    if "2d_map" in types:
        result.add("constant")

    if "1d_map" in types:
        # If 1d_map coexists with 3d_map but has no float/str constant fallback,
        # the 1d_map variant is a simpler time-series (not period-indexed).
        # Only add 'periodic' when there's also a constant type or no 3d_map.
        if "3d_map" in types and not (types & {"float", "str"}):
            # Pure time-series param (e.g. profile: ('1d_map', '3d_map'))
            result.add("timeseries")
        else:
            result.add("periodic")
            if "3d_map" in types:
                result.add("timeseries")

    if "3d_map" in types:
        result.add("stochastic")
        result.add("timeseries")

    if "4d_map" in types:
        result.add("stochastic")

    return result


def _has_time_indexed_values(
    db_contents: DatabaseContents,
    entity_class: str,
    param_name: str,
) -> bool:
    """Check whether any actual parameter value for this param has index_name='time'."""
    for (cls, _byname, pname, _alt), value in db_contents.parameter_values.items():
        if cls == entity_class and pname == param_name:
            if hasattr(value, "index_name") and value.index_name == "time":
                return True
    return False


def _is_string_param(pname: str, entity_class: str, db_contents: DatabaseContents) -> bool:
    """Check if a parameter is string-valued (type_list contains 'str')."""
    for pdef in db_contents.parameter_definitions.get(entity_class, []):
        if pdef["name"] == pname:
            type_list = pdef.get("parameter_type_list")
            if type_list and "str" in type_list:
                return True
            return False
    return False


def order_parameters(
    param_names: list[str],
    entity_class: str,
    db_contents: DatabaseContents,
) -> list[str]:
    """Sort parameters: string-typed first, then numeric, each by group priority and name.

    This ensures the string/float column-type boundary in the import spec
    lines up correctly for roundtrip xlsx compatibility.
    """
    pdefs_dict = {
        p["name"]: p for p in db_contents.parameter_definitions.get(entity_class, [])
    }

    string_grouped: list[tuple[int, str]] = []
    string_ungrouped: list[str] = []
    float_grouped: list[tuple[int, str]] = []
    float_ungrouped: list[str] = []

    for pname in param_names:
        is_str = _is_string_param(pname, entity_class, db_contents)

        group_name = db_contents.param_to_group.get((entity_class, pname))
        if group_name and group_name in db_contents.parameter_groups:
            priority = db_contents.parameter_groups[group_name].get("priority") or 999
            if is_str:
                string_grouped.append((priority, pname))
            else:
                float_grouped.append((priority, pname))
        else:
            if is_str:
                string_ungrouped.append(pname)
            else:
                float_ungrouped.append(pname)

    string_grouped.sort(key=lambda x: (x[0], x[1]))
    string_ungrouped.sort()
    float_grouped.sort(key=lambda x: (x[0], x[1]))
    float_ungrouped.sort()

    return (
        [p[1] for p in string_grouped] + string_ungrouped
        + [p[1] for p in float_grouped] + float_ungrouped
    )


def _get_param_defs_dict(
    db_contents: DatabaseContents, entity_class: str
) -> dict[str, dict]:
    """Return {param_name: pdef_dict} for a given entity class."""
    return {
        pdef["name"]: pdef
        for pdef in db_contents.parameter_definitions.get(entity_class, [])
    }


def _get_dimension_names(db_contents: DatabaseContents, class_name: str) -> tuple:
    """Get dimension_name_list for a class."""
    for ec in db_contents.entity_classes:
        if ec["name"] == class_name:
            return ec["dimension_name_list"]
    return ()


def _make_base_name(class_name: str, drop_dims: list[str]) -> str:
    """Convert a DB entity class name to a sheet base name.

    Replaces ``__`` with ``_`` and drops any dimension names in *drop_dims*.
    """
    parts = class_name.split("__")
    filtered = [p for p in parts if p not in drop_dims]
    return "_".join(filtered)


def _sheet_name_in_order(name: str, sheet_order: list[str]) -> bool:
    """Check if a sheet name is present in the YAML sheet_order list."""
    return name in sheet_order


def build_sheet_specs(
    db_contents: DatabaseContents,
    settings: dict | None = None,
) -> list[SheetSpec]:
    """Build the ordered list of SheetSpec objects for all Excel sheets.

    Args:
        db_contents: Database contents from :func:`db_reader.read_database`.
        settings: Parsed YAML settings dict, or *None* to load from disk.

    Returns:
        Ordered list of :class:`SheetSpec` objects.
    """
    if settings is None:
        settings = load_settings()

    # ---- Build lookup structures from settings ----
    merge_rules: dict[str, dict] = settings.get("merge_classes", {})
    split_rules: dict[str, dict] = settings.get("split_params", {})
    unpack_rules: dict[str, dict] = settings.get("unpack_map_params", {})
    element_rules: dict[str, dict] = settings.get("add_entity_elements", {})
    sheet_order: list[str] = settings.get("sheet_order", [])
    drop_dims: list[str] = settings.get("drop_dimensions", [])
    rename_sheets: dict[str, str] = settings.get("rename_sheets", {})
    ea_classes: set[str] = set(settings.get("entity_alternative_classes", []))
    pre_ea_rules: dict[str, list[str]] = settings.get("pre_entity_alternative_params", {})
    index_name_defaults: dict[str, str] = settings.get("index_name_defaults", {})

    # Classes that are used as extra entity element sources (skip in generic path)
    element_source_classes: set[str] = set()
    for _cls, erule in element_rules.items():
        element_source_classes.add(erule["from_class"])

    # Sets of classes handled by merge/split/unpack so the generic path skips them
    merged_classes: set[str] = set()
    for _group_name, rule in merge_rules.items():
        for cls_name in rule["classes"]:
            merged_classes.add(cls_name)

    split_classes: set[str] = set(split_rules.keys())

    # Map from source class -> list of unpack rules
    unpack_source_classes: dict[str, list[dict[str, Any]]] = {}
    # Track which params are unpacked per class
    unpacked_params_per_class: dict[str, set[str]] = {}
    for group_name, rule in unpack_rules.items():
        for src_cls in rule["source_classes"]:
            unpack_source_classes.setdefault(src_cls, []).append(
                {"group_name": group_name, **rule}
            )
            unpacked_params_per_class.setdefault(src_cls, set()).update(rule["params"])

    handled_classes: set[str] = set()
    specs: list[SheetSpec] = []

    # ---- 1. Process merge_classes rules ----
    for group_name, rule in merge_rules.items():
        classes_map: dict[str, str] = rule["classes"]  # class_name -> direction_value
        direction_col: str = rule["direction_column"]
        class_names = list(classes_map.keys())

        # Collect ALL param defs from the merged classes (they share the same params)
        all_param_defs: dict[str, dict] = {}
        for cls_name in class_names:
            for pdef in db_contents.parameter_definitions.get(cls_name, []):
                all_param_defs[pdef["name"]] = pdef

        # Filter out unpacked params
        unpacked: set[str] = set()
        for cls_name in class_names:
            unpacked.update(unpacked_params_per_class.get(cls_name, set()))
        remaining_params = {
            k: v for k, v in all_param_defs.items() if k not in unpacked
        }

        # Classify params by eligible layouts
        layout_params: dict[str, list[str]] = {
            "constant": [],
            "periodic": [],
            "timeseries": [],
        }
        for pname, pdef in remaining_params.items():
            types = classify_param_types(pdef.get("parameter_type_list"))
            if "constant" in types:
                layout_params["constant"].append(pname)
            if "periodic" in types:
                layout_params["periodic"].append(pname)
            if "timeseries" in types:
                layout_params["timeseries"].append(pname)
            # Check for actual time-indexed values across merged classes
            if "periodic" in types and "timeseries" not in types:
                for cls_name in class_names:
                    if _has_time_indexed_values(db_contents, cls_name, pname):
                        layout_params["timeseries"].append(pname)
                        break

        # Determine entity_columns from first class (all merged should match)
        dims = _get_dimension_names(db_contents, class_names[0])
        entity_cols = list(dims) if dims else [class_names[0]]

        # Determine suffix rules
        has_layouts = [k for k in ("constant", "periodic", "timeseries") if layout_params[k]]
        use_suffix = len(has_layouts) > 1

        direction_map = dict(classes_map)
        layout_to_suffix = {"constant": "_c", "periodic": "_p", "timeseries": "_t"}

        for layout_key in ("constant", "periodic", "timeseries"):
            params = layout_params[layout_key]
            if not params:
                continue

            suffix = layout_to_suffix[layout_key] if use_suffix else ""
            if not use_suffix and layout_key == "constant":
                if _sheet_name_in_order(f"{group_name}_c", sheet_order):
                    suffix = "_c"

            sheet_nm = f"{group_name}{suffix}"
            has_ea = layout_key == "constant" and any(
                c in ea_classes for c in class_names
            )

            spec = SheetSpec(
                sheet_name=sheet_nm,
                layout=layout_key,
                entity_classes=class_names,
                entity_columns=entity_cols,
                direction_column=direction_col,
                direction_map=direction_map,
                has_entity_alternative=has_ea,
            )
            spec._raw_params = params  # type: ignore[attr-defined]
            spec._primary_class = class_names[0]  # type: ignore[attr-defined]
            spec._all_param_defs = all_param_defs  # type: ignore[attr-defined]
            specs.append(spec)

        for cls_name in class_names:
            handled_classes.add(cls_name)

    # ---- 2. Process split_params rules ----
    for entity_class, sub_groups in split_rules.items():
        class_pdefs = _get_param_defs_dict(db_contents, entity_class)
        dims = _get_dimension_names(db_contents, entity_class)
        entity_cols = list(dims) if dims else [entity_class]

        for sub_name, sub_rule in sub_groups.items():
            sub_params = sub_rule["params"]
            # Determine layout from param types in the sub-group
            has_periodic = False
            for pname in sub_params:
                pdef = class_pdefs.get(pname)
                if pdef:
                    type_list = pdef.get("parameter_type_list")
                    if type_list and ("1d_map" in type_list or "2d_map" in type_list):
                        has_periodic = True

            layout = "periodic" if has_periodic else "constant"

            spec = SheetSpec(
                sheet_name=sub_name,
                layout=layout,
                entity_classes=[entity_class],
                entity_columns=entity_cols,
                has_entity_alternative=(layout == "constant" and entity_class in ea_classes),
            )
            # Only include params that actually exist in the DB schema
            existing_params = [p for p in sub_params if p in class_pdefs]
            spec._raw_params = existing_params  # type: ignore[attr-defined]
            spec._primary_class = entity_class  # type: ignore[attr-defined]
            spec._all_param_defs = class_pdefs  # type: ignore[attr-defined]
            specs.append(spec)

        handled_classes.add(entity_class)

    # ---- 2b. Create solve_period_period sheet for nested Map params ----
    # Look for a spec named 'solve_period' and clone it for nested Maps
    for spec in list(specs):
        if spec.sheet_name == "solve_period":
            nested_spec = SheetSpec(
                sheet_name="solve_period_period",
                layout="nested_periodic",
                entity_classes=list(spec.entity_classes),
                entity_columns=list(spec.entity_columns),
                has_entity_alternative=False,
            )
            raw = getattr(spec, "_raw_params", None)
            if raw is not None:
                nested_spec._raw_params = list(raw)  # type: ignore[attr-defined]
            pclass = getattr(spec, "_primary_class", None)
            if pclass is not None:
                nested_spec._primary_class = pclass  # type: ignore[attr-defined]
            all_pdefs = getattr(spec, "_all_param_defs", None)
            if all_pdefs is not None:
                nested_spec._all_param_defs = dict(all_pdefs)  # type: ignore[attr-defined]
            specs.append(nested_spec)
            break

    # ---- 3. Process unpack_map_params rules ----
    for group_name, rule in unpack_rules.items():
        source_classes: list[str] = rule["source_classes"]
        unpack_params: list[str] = rule["params"]
        index_column: str = rule["index_column"]
        direction_col_name: str | None = rule.get("direction_column")

        # Get entity_columns from first source class
        dims = _get_dimension_names(db_contents, source_classes[0])
        entity_cols = list(dims) if dims else [source_classes[0]]

        # Build direction_map if source classes are merged
        dir_map: dict[str, str] = {}
        if direction_col_name:
            for _mg_name, mg_rule in merge_rules.items():
                for cls_name, dir_val in mg_rule["classes"].items():
                    if cls_name in source_classes:
                        dir_map[cls_name] = dir_val

        # Collect param defs for descriptions
        all_pdefs: dict[str, dict] = {}
        for src_cls in source_classes:
            for pdef in db_contents.parameter_definitions.get(src_cls, []):
                if pdef["name"] in unpack_params:
                    all_pdefs[pdef["name"]] = pdef

        sheet_nm = f"{group_name}_c"
        spec = SheetSpec(
            sheet_name=sheet_nm,
            layout="constant",
            entity_classes=source_classes,
            entity_columns=entity_cols,
            parameter_names=unpack_params,
            direction_column=direction_col_name,
            direction_map=dir_map,
            unpack_index_column=index_column,
            has_entity_alternative=False,
        )
        spec._all_param_defs = all_pdefs  # type: ignore[attr-defined]
        specs.append(spec)

    # ---- 4. Process remaining entity classes (generic path) ----
    for ec in db_contents.entity_classes:
        cls_name = ec["name"]
        if cls_name in handled_classes:
            continue
        if cls_name in merged_classes:
            continue
        if cls_name in split_classes:
            continue

        dims = ec["dimension_name_list"]
        entity_cols = list(dims) if dims else [cls_name]
        pdefs = db_contents.parameter_definitions.get(cls_name, [])

        # Filter out params that went to unpack rules
        unpacked = unpacked_params_per_class.get(cls_name, set())
        remaining_pdefs = [p for p in pdefs if p["name"] not in unpacked]
        remaining_param_names = [p["name"] for p in remaining_pdefs]
        remaining_param_defs = {p["name"]: p for p in remaining_pdefs}

        base_name = _make_base_name(cls_name, drop_dims)

        if not remaining_param_names:
            # No params remaining: either link sheet or skip
            if not dims:
                # Zero-dim class with no params — skip (upDown, reserve, etc.)
                continue
            # Skip classes used solely as extra entity element sources
            if cls_name in element_source_classes:
                continue
            # Multi-dim relationship class -> link sheet
            spec = SheetSpec(
                sheet_name=base_name,
                layout="link",
                entity_classes=[cls_name],
                entity_columns=entity_cols,
                has_entity_alternative=False,
            )
            specs.append(spec)
            continue

        # Has parameters: classify by layout
        layout_params: dict[str, list[str]] = {
            "constant": [],
            "periodic": [],
            "timeseries": [],
        }
        for pdef in remaining_pdefs:
            pname = pdef["name"]
            types = classify_param_types(pdef.get("parameter_type_list"))
            if "constant" in types:
                layout_params["constant"].append(pname)
            if "periodic" in types:
                layout_params["periodic"].append(pname)
            if "timeseries" in types:
                layout_params["timeseries"].append(pname)
            # For params with periodic capability but no 3d_map: check actual DB values
            if "periodic" in types and "timeseries" not in types:
                if _has_time_indexed_values(db_contents, cls_name, pname):
                    layout_params["timeseries"].append(pname)

        has_layouts = [k for k in ("constant", "periodic", "timeseries") if layout_params[k]]

        if not has_layouts:
            # Fallback: treat as constant
            has_layouts = ["constant"]
            layout_params["constant"] = remaining_param_names

        # Apply rename_sheets
        display_name = rename_sheets.get(base_name, base_name)

        use_suffix = len(has_layouts) > 1
        layout_to_suffix = {"constant": "_c", "periodic": "_p", "timeseries": "_t"}

        for layout_key in ("constant", "periodic", "timeseries"):
            params = layout_params[layout_key]
            if not params:
                continue

            suffix = layout_to_suffix[layout_key] if use_suffix else ""
            # When only one layout, check sheet_order for suffix convention
            if not use_suffix:
                candidate = f"{display_name}{layout_to_suffix[layout_key]}"
                if _sheet_name_in_order(candidate, sheet_order):
                    suffix = layout_to_suffix[layout_key]

            sheet_nm = f"{display_name}{suffix}"
            has_ea = layout_key == "constant" and cls_name in ea_classes

            spec = SheetSpec(
                sheet_name=sheet_nm,
                layout=layout_key,
                entity_classes=[cls_name],
                entity_columns=entity_cols,
                has_entity_alternative=has_ea,
            )

            # Apply add_entity_elements rules (only to constant sheets)
            if cls_name in element_rules and layout_key == "constant":
                erule = element_rules[cls_name]
                spec.extra_entity_columns = erule["columns"]
                spec.extra_entity_class = erule["from_class"]

            spec._raw_params = params  # type: ignore[attr-defined]
            spec._primary_class = cls_name  # type: ignore[attr-defined]
            spec._all_param_defs = remaining_param_defs  # type: ignore[attr-defined]
            specs.append(spec)

    # ---- 5. Add special sheets ----
    specs.append(SheetSpec(
        sheet_name="scenario",
        layout="scenario",
        entity_classes=[],
        entity_columns=[],
        has_entity_alternative=False,
    ))
    specs.append(SheetSpec(
        sheet_name="navigate",
        layout="navigate",
        entity_classes=[],
        entity_columns=[],
        has_entity_alternative=False,
    ))
    specs.append(SheetSpec(
        sheet_name="version",
        layout="version",
        entity_classes=[],
        entity_columns=[],
        has_entity_alternative=False,
    ))

    # ---- 6. Populate parameter_names using order_parameters ----
    for spec in specs:
        raw_params: list[str] | None = getattr(spec, "_raw_params", None)
        if raw_params is None:
            continue
        primary_class: str = getattr(spec, "_primary_class", "")
        spec.parameter_names = order_parameters(raw_params, primary_class, db_contents)

        # Move pre-EA params out of parameter_names into pre_ea_params
        if spec.has_entity_alternative and spec.layout == "constant":
            for cls in spec.entity_classes:
                pre_ea = pre_ea_rules.get(cls, [])
                if pre_ea:
                    spec.pre_ea_params = [p for p in pre_ea if p in spec.parameter_names]
                    spec.parameter_names = [
                        p for p in spec.parameter_names if p not in spec.pre_ea_params
                    ]
                    break

    # ---- 7. Populate descriptions ----
    for spec in specs:
        all_pdefs_attr: dict[str, dict] | None = getattr(spec, "_all_param_defs", None)
        if all_pdefs_attr is None:
            continue
        for pname in list(spec.parameter_names) + spec.pre_ea_params:
            pdef = all_pdefs_attr.get(pname)
            if pdef and pdef.get("description"):
                spec.descriptions[pname] = pdef["description"]

    # ---- 7b. Set index_name_default from layout type ----
    for spec in specs:
        default = index_name_defaults.get(spec.layout)
        if default:
            spec.index_name_default = default

    # ---- 8. Clean up temporary attributes ----
    for spec in specs:
        for attr in ("_raw_params", "_primary_class", "_all_param_defs"):
            if hasattr(spec, attr):
                delattr(spec, attr)

    # ---- 9. Sort by sheet_order ----
    order_map = {name: i for i, name in enumerate(sheet_order)}
    max_order = len(sheet_order)

    def sort_key(spec: SheetSpec) -> tuple[int, str]:
        if spec.sheet_name in order_map:
            return (order_map[spec.sheet_name], spec.sheet_name)
        return (max_order, spec.sheet_name)

    specs.sort(key=sort_key)

    return specs
