"""Top-level orchestrator for exporting a FlexTool Spine DB to an Excel file."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import openpyxl.cell._writer as _openpyxl_cell_writer
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from flextool.export_to_tabular.db_reader import DatabaseContents, read_database
from flextool.export_to_tabular.sheet_config import SheetSpec, build_sheet_specs, load_settings
from flextool.export_to_tabular.excel_writer import (
    IndexClassifier,
    write_array_transposed_sheet_v2,
    write_constant_sheet,
    write_constant_sheet_v2,
    write_link_sheet,
    write_link_sheet_v2,
    write_navigate_sheet,
    write_nested_periodic_sheet,
    write_nested_periodic_sheet_v2,
    write_periodic_sheet,
    write_periodic_sheet_v2,
    write_scenario_sheet,
    write_ladder_sheet_v2,
    write_stochastic_sheet_v2,
    write_timeseries_sheet,
    write_timeseries_sheet_v2,
    write_version_sheet,
)
import flextool.export_to_tabular.excel_writer as _ew


def _build_tab_color_map(navigate_groups: list[dict[str, Any]]) -> dict[str, str]:
    """Build a mapping from sheet name to tab color (hex RGB string) from navigate groups."""
    color_map: dict[str, str] = {}
    for group in navigate_groups:
        color = group["color"]
        for row in group.get("rows", []):
            for sheet_name in row:
                color_map[sheet_name] = color
    return color_map


def _write_parameter_sheet(
    wb: Workbook,
    ws: Worksheet,
    writer: Callable[[Worksheet, SheetSpec, DatabaseContents], list[str]],
    spec: SheetSpec,
    db_contents: DatabaseContents,
    omit_empty_columns: bool,
) -> Worksheet:
    """Write a constant or periodic v2 sheet; with *omit_empty_columns*, write it again without its empty columns.

    The writer itself decides which values land on the sheet, so the columns it left empty are known only after
    writing.  A sheet whose parameter columns are all empty keeps them: without any parameter column the reader
    would take it for a link sheet.  Returns the sheet that holds the result.
    """
    empty = writer(ws, spec, db_contents)
    all_params = list(spec.pre_ea_params) + list(spec.parameter_names)
    if not omit_empty_columns or not empty or len(empty) == len(all_params):
        return ws
    index = wb.sheetnames.index(ws.title)
    wb.remove(ws)
    ws = wb.create_sheet(spec.sheet_name, index)
    dropped = set(empty)
    writer(ws, replace(
        spec,
        parameter_names=[p for p in spec.parameter_names if p not in dropped],
        pre_ea_params=[p for p in spec.pre_ea_params if p not in dropped],
        omitted_params=empty,
    ), db_contents)
    return ws


_openpyxl_safe_string = _openpyxl_cell_writer.safe_string


def _exact_number_string(value: Any) -> str:
    """openpyxl's cell-value formatter, but a finite float is written with ``repr`` (shortest exact form).

    openpyxl writes numbers with ``"%.16g"``, which loses the 17th significant digit (0.1 + 0.2 comes back as
    0.3).  ``repr`` gives the shortest string that reads back to the same float; Excel and LibreOffice read it.
    """
    if isinstance(value, float) and math.isfinite(value):
        return repr(float(value))  # float() also turns a numpy float into a plain one (numpy's repr differs)
    return _openpyxl_safe_string(value)


@contextmanager
def _exact_floats() -> Iterator[None]:
    """Write floats exactly while saving: swap the formatter openpyxl's cell writer looks up for each cell."""
    _openpyxl_cell_writer.safe_string = _exact_number_string
    try:
        yield
    finally:
        _openpyxl_cell_writer.safe_string = _openpyxl_safe_string


def _save_workbook(wb: Workbook, path: str | Path) -> None:
    """Save *wb* with every float written exactly (see :func:`_exact_number_string`)."""
    with _exact_floats():
        wb.save(path)


def _cells_as_text(wb: Workbook) -> None:
    """Keep every string a string: openpyxl stores a string that starts with ``=`` as a formula.

    The export writes no formulas of its own (its links are cell hyperlinks), and names and descriptions come
    from the input data, so a formula cell would run in Excel on opening, and read back with ``data_only=True``
    it would have no value (the name would be lost).  The value stays exactly as it was.
    """
    for ws in wb.worksheets:
        # ``_cells`` holds only the cells that were written (iter_rows would create the empty ones).
        for cell in ws._cells.values():
            if cell.data_type == "f" and isinstance(cell.value, str):
                cell.data_type = "s"


def export_to_excel(
    db_url: str,
    output_path: str,
    include_advanced: bool = False,
    use_new_format: bool = True,
    include_stochastics: bool = False,  # deprecated alias for include_advanced
    include_groups: list[str] | None = None,
    description_row: bool = True,
    omit_empty_columns: bool = False,
) -> None:
    """Export a FlexTool Spine DB to an Excel (.xlsx) file.

    Args:
        db_url: Spine DB URL (e.g., 'sqlite:///path/to/db.sqlite')
        output_path: Output .xlsx file path
        include_advanced: Whether to include advanced sheets (solve sequences,
            periods_available, stochastic data).
        use_new_format: If True, use the v2 self-describing format with embedded
            metadata.  If False, use the original v1 format.
        include_stochastics: Deprecated alias for include_advanced.
        include_groups: When given, restrict output to parameters whose
            ``parameter_group_name`` is in the list (e.g. ``['basics',
            'model', 'timeline', 'solve_basics']``).  Parameters with no
            group are dropped.  Sheets that lose all their columns are
            dropped, unless listed under a selected group in
            ``always_include_with_groups`` in ``export_settings.yaml``.
        description_row: When False, the standard sheets start with the data-type row; each parameter's
            description stays in the note on its name.  Only for the v2 format.
        omit_empty_columns: When True, a parameter column with no value on its constant or periodic sheet is
            left out (the reference block to the right still lists it).  A sheet whose parameter columns are all
            empty keeps them.  Only for the v2 format.
    """
    if include_stochastics:
        include_advanced = True
    if not use_new_format and not description_row:
        raise ValueError("Leaving out the description row is only possible in the v2 (self-describing) format.")
    if not use_new_format and omit_empty_columns:
        raise ValueError("Leaving out empty columns is only possible in the v2 (self-describing) format.")

    # 1. Read all data from the database
    print(f"Reading database: {db_url}")
    db_contents: DatabaseContents = read_database(db_url)

    # 1b. Build index classifier and entity alternatives index
    _ew._index_classifier = IndexClassifier(db_contents)
    _ew._build_entity_alts_index(db_contents)

    # 2. Load settings and build ordered sheet specifications
    print("Building sheet specifications...")
    settings = load_settings()

    # 2b. Load data type overrides and width settings
    _ew._data_type_overrides = settings.get("data_type_overrides", {})
    _ew._description_row = description_row
    _ew._period_only_params = settings.get("period_only_params", {})
    _ew._stochastic_index_names = settings.get("stochastic_index_names",
                                                ["forecast", "branch_time", "time", "is_realized"])
    _ew._time_structure_classes = set(settings.get("time_structure_classes", []))
    specs: list[SheetSpec] = build_sheet_specs(
        db_contents,
        settings,
        include_groups=set(include_groups) if include_groups else None,
        group_first=use_new_format,
    )
    navigate_groups: list[dict[str, Any]] = settings.get("navigate_groups", [])
    tab_color_map = _build_tab_color_map(navigate_groups) if navigate_groups else {}

    # Filter out advanced sheets unless requested OR they have data
    advanced_sheets: set[str] = set(settings.get("advanced_sheets", []))
    if not include_advanced:
        # Pre-build a set of (class, param) pairs that have data for fast lookup
        _data_keys: set[tuple[str, str]] = set()
        for (cls, _bn, pn, _alt) in db_contents.parameter_values:
            _data_keys.add((cls, pn))

        def _has_data(spec: SheetSpec) -> bool:
            for cls in spec.entity_classes:
                for pn in spec.parameter_names:
                    if (cls, pn) in _data_keys:
                        return True
            return False

        specs = [s for s in specs
                 if s.sheet_name not in advanced_sheets or _has_data(s)]

    # Load info rows setting
    _ew._info_rows = settings.get("info_rows", {})

    # 3. Create the workbook
    wb = Workbook()

    # 4. Delete the default 'Sheet' worksheet
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]

    # Track counts by layout for the summary
    layout_counts: dict[str, int] = {}

    # 5. Write each sheet
    for spec in specs:
        ws = wb.create_sheet(spec.sheet_name)
        layout = spec.layout

        if layout == "constant":
            if use_new_format:
                # Check if this is a pure-array sheet (model_solve_sequence etc.)
                _is_array_sheet = (
                    len(spec.parameter_names) == 1
                    and _ew._get_param_data_type(
                        spec.parameter_names[0], spec.entity_classes,
                        db_contents, layout="constant",
                    ) == "boolean (array)"
                )
                if _is_array_sheet:
                    write_array_transposed_sheet_v2(ws, spec, db_contents)
                else:
                    ws = _write_parameter_sheet(wb, ws, write_constant_sheet_v2, spec, db_contents,
                                                omit_empty_columns)
            else:
                write_constant_sheet(ws, spec, db_contents)
        elif layout == "periodic":
            if use_new_format:
                ws = _write_parameter_sheet(wb, ws, write_periodic_sheet_v2, spec, db_contents,
                                            omit_empty_columns)
            else:
                write_periodic_sheet(ws, spec, db_contents)
        elif layout == "nested_periodic":
            if use_new_format:
                write_nested_periodic_sheet_v2(ws, spec, db_contents)
            else:
                write_nested_periodic_sheet(ws, spec, db_contents)
        elif layout == "timeseries":
            if use_new_format:
                write_timeseries_sheet_v2(ws, spec, db_contents)
            else:
                write_timeseries_sheet(ws, spec, db_contents)
        elif layout == "stochastic":
            if use_new_format:
                write_stochastic_sheet_v2(ws, spec, db_contents)
            else:
                # v1 has no stochastic writer — skip
                del wb[spec.sheet_name]
                continue
        elif layout == "ladder":
            if use_new_format:
                write_ladder_sheet_v2(ws, spec, db_contents)
            else:
                # v1 has no ladder writer — skip
                del wb[spec.sheet_name]
                continue
        elif layout == "link":
            if use_new_format:
                write_link_sheet_v2(ws, spec, db_contents)
            else:
                write_link_sheet(ws, spec, db_contents)
        elif layout == "scenario":
            write_scenario_sheet(ws, db_contents, include_stochastics=include_advanced)
        elif layout == "version":
            if use_new_format:
                # In v2, version info is on the navigate sheet — skip separate sheet
                del wb[spec.sheet_name]
                continue
            write_version_sheet(ws, db_contents.version, db_contents.behaviour_markers)
        elif layout == "navigate":
            write_navigate_sheet(
                ws, specs, navigate_groups=navigate_groups,
                version=db_contents.version if use_new_format else None,
                behaviour_markers=db_contents.behaviour_markers if use_new_format else None,
            )
        else:
            print(f"  Warning: unknown layout '{layout}' for sheet '{spec.sheet_name}', skipping.")
            continue

        # Set tab color from navigate group config
        if spec.sheet_name in tab_color_map:
            ws.sheet_properties.tabColor = tab_color_map[spec.sheet_name]

        layout_counts[layout] = layout_counts.get(layout, 0) + 1

    # 6. Ensure the output directory exists and save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    _cells_as_text(wb)
    _save_workbook(wb, output_path)

    # 7. Print summary
    format_label = "v2 (self-describing)" if use_new_format else "v1 (original)"
    total = len(specs)
    print(f"\nExport complete: {output_path} [{format_label}]")
    print(f"  Total sheets: {total}")
    for layout_type, count in sorted(layout_counts.items()):
        print(f"    {layout_type}: {count}")
    if db_contents.version is not None:
        version_display = int(db_contents.version) if db_contents.version == int(db_contents.version) else db_contents.version
        print(f"  DB version: {version_display}")
