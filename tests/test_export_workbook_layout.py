"""Layout of the exported input workbook (v2 self-describing format): freeze panes, column shading, column widths,
description row and notes, and that none of it changes what the reader reads back.

The database is built from ``templates_examples.json``.  Expected widths are worked out by hand from the strings in
that fixture: a column is ``clamp(longest + 2, lo, hi)`` over its definition-row label, default and data cells, with
(lo, hi) = (8, 32) for key columns, (10, 16) for the definition column and (8, 30) for data columns.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import openpyxl
import pytest
from openpyxl.utils import get_column_letter, range_boundaries
from spinedb_api import Array, DatabaseMapping, Map, from_database, to_database

from flextool._resources import package_data_path
from flextool.export_to_tabular.export_to_excel import export_to_excel
from flextool.process_inputs.read_self_describing_excel import read_self_describing_excel
from flextool.process_inputs.write_self_describing_to_db import write_sheet_data_to_db
from flextool.update_flextool.initialize_database import initialize_database

EXAMPLES_JSON = package_data_path("schemas/canonical_databases/templates_examples.json")
SCHEMA_JSON = package_data_path("schemas/spinedb_schema.json")


@pytest.fixture(scope="module")
def examples_db(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("layout_db") / "examples_layout.sqlite"
    initialize_database(str(EXAMPLES_JSON), str(path))
    return path


@pytest.fixture(scope="module")
def workbook_path(examples_db: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("layout_xlsx") / "examples.xlsx"
    export_to_excel(f"sqlite:///{examples_db}", str(out), include_advanced=True)
    return out


@pytest.fixture(scope="module")
def wb(workbook_path: Path) -> openpyxl.Workbook:
    book = openpyxl.load_workbook(workbook_path)
    yield book
    book.close()


def _column_of(ws, row: int, label: str) -> int:
    for col in range(1, ws.max_column + 1):
        if ws.cell(row=row, column=col).value == label:
            return col
    raise AssertionError(f"{ws.title}: no {label!r} in row {row}")


def _width(ws, col: int) -> float:
    return ws.column_dimensions[get_column_letter(col)].width


def _cf_colours(ws, row: int, col: int) -> set[str]:
    """Fill colours of the conditional-format rules covering one cell."""
    colours = set()
    for cf in ws.conditional_formatting:
        for rng in str(cf.sqref).split():
            c0, r0, c1, r1 = range_boundaries(rng)
            if c0 <= col <= c1 and r0 <= row <= r1:
                for rule in cf.rules:
                    fill = rule.dxf.fill if rule.dxf else None
                    if fill is None or fill.fgColor is None:
                        continue
                    if fill.fgColor.type == "rgb":
                        colours.add(fill.fgColor.rgb[-6:])
                    elif fill.fgColor.type == "theme":
                        colours.add(f"theme{fill.fgColor.theme}{fill.fgColor.tint:+.2f}")
    return colours


# Header rows and key columns stay visible.  Standard sheets: rows 1..definition row and the columns left of the
# definition column (unit_c: description, data type, default rows + definition row 4; alternative, unit -> C5).
# Transposed and stochastic sheets also keep the label column frozen (node_t: 5 header rows, A index, B labels).
@pytest.mark.parametrize(("sheet", "freeze"), [
    ("unit_c", "C5"), ("unit_node_c", "F5"), ("node_p", "D5"), ("price_ladder_annual", "E4"),
    ("node_t", "C6"), ("model_solve_sequence", "C5"), ("unit_s_availability", "E4"),
    ("connection_node", "A3"), ("scenario", "B3"), ("navigate", None),
])
def test_freeze_panes(wb: openpyxl.Workbook, sheet: str, freeze: str | None) -> None:
    assert wb[sheet].freeze_panes == freeze


def test_parameter_columns_alternate_shade(wb: openpyxl.Workbook) -> None:
    # unit_c: definition column C, entity existence D is the first data column and keeps the base grey
    # (white darkened 5 %), the next is shaded.
    ws = wb["unit_c"]
    assert _cf_colours(ws, 5, 4) == {"theme0-0.05"}
    assert "E4E9F1" in _cf_colours(ws, 5, 5)
    assert _cf_colours(ws, 5, 6) == {"theme0-0.05"}


def test_transposed_data_columns_alternate_shade(wb: openpyxl.Workbook) -> None:
    # node_t: data from row 6; C stays white, D is shaded.
    ws = wb["node_t"]
    assert _cf_colours(ws, 6, 3) == set()
    assert _cf_colours(ws, 6, 4) == {"EEF1F6"}


def test_column_widths_fit_content(wb: openpyxl.Workbook) -> None:
    unit_c = wb["unit_c"]
    assert _width(unit_c, 1) == 29  # alternative invest_solveSequence_5weeks (27)
    assert _width(unit_c, 2) == 20  # unit dr_decrease_demand (18)
    assert _width(unit_c, _column_of(unit_c, 4, "conversion_method")) == 21  # default constant_efficiency (19)
    assert _width(unit_c, _column_of(unit_c, 4, "existing")) == 10  # name existing (8), values up to 4
    unit_node_c = wb["unit_node_c"]
    assert _width(unit_node_c, 2) == 8  # entity: (...) reader label, no data
    assert _width(unit_node_c, 5) == 8  # filter column: input / output (6); the filter: {...} label does not count
    node_t = wb["node_t"]
    assert _width(node_t, 1) == 13  # index: time (11)
    assert _width(node_t, 2) == 14  # entity: node (12); the INFO text does not count
    assert _width(node_t, 3) == 21  # alternative desal_water_storage (19)
    assert _width(node_t, 4) == 12  # entity ev_battery (10); the data type row does not count


def test_description_row_and_notes(wb: openpyxl.Workbook, examples_db: Path) -> None:
    with DatabaseMapping(f"sqlite:///{examples_db}") as db:
        description = db.get_parameter_definition_item(
            entity_class_name="unit", name="conversion_method")["description"]
    ws = wb["unit_c"]
    col = _column_of(ws, 4, "conversion_method")
    assert ws.row_dimensions[1].height == 60
    assert ws.cell(row=1, column=col).value == description
    assert ws.cell(row=1, column=col).alignment.wrap_text
    assert ws.cell(row=1, column=col).fill.fgColor.rgb[-6:] == "FFF2CC"
    assert ws.cell(row=1, column=3).fill.fgColor.rgb[-6:] == "FFE699"
    assert ws.cell(row=2, column=col).fill.fgColor.type == "theme"  # data type row stays grey (theme colour)
    note = ws.cell(row=4, column=col).comment
    assert note is not None and description in note.text


def _plain(value: object) -> object:
    """Map / Array as nested tuples of indexes and values: the workbook does not carry index names (separate item)."""
    if isinstance(value, Map):
        return ("map", tuple(value.indexes), tuple(_plain(v) for v in value.values))
    if isinstance(value, Array):
        return ("array", tuple(value.values))
    return value


def _scenario_ranks(url: str) -> dict[tuple[str, str], int]:
    with DatabaseMapping(url) as db:
        return {(s["scenario_name"], s["alternative_name"]): s["rank"] for s in db.get_scenario_alternative_items()}


def _snapshot(url: str) -> tuple[dict, set, dict]:
    with DatabaseMapping(url) as db:
        values = {}
        for p in db.get_parameter_value_items():
            value = _plain(from_database(p["value"], p["type"]))
            values[(p["entity_class_name"], tuple(p["entity_byname"]), p["parameter_definition_name"],
                    p["alternative_name"])] = value
        entities = {(e["entity_class_name"], tuple(e["entity_byname"])) for e in db.get_entity_items()}
        # Only node / unit / connection carry an entity existence column (separate item for the others).
        activity = {(e["entity_class_name"], tuple(e["entity_byname"]), e["alternative_name"]): e["active"]
                    for e in db.get_entity_alternative_items()
                    if e["entity_class_name"] in ("node", "unit", "connection")}
    return values, entities, activity


# Values that do not survive the workbook today, for reasons outside the layout (reported, not fixed here):
# - a multi-parameter time-series sheet with a ``default`` row: the reader takes that row for a time step
#   (an extra ``index: time`` step carrying the default 1.0);
# - a Map with float indexes comes back with string indexes ('2.0').
# The test fails when one of them is fixed (remove it here) or when a new difference appears.
KNOWN_ROUND_TRIP_DEFECTS = {
    ("unit", ("coal_plant",), "availability", "availability"),
    ("unit", ("water_pump",), "delay", "water_pump_delayed"),
}


def test_round_trip_reads_back_the_database(
    examples_db: Path, workbook_path: Path, tmp_path: Path,
) -> None:
    target = tmp_path / "round_trip.sqlite"
    initialize_database(str(SCHEMA_JSON), str(target))
    sheets = read_self_describing_excel(str(workbook_path), skip_sheets={"navigate", "version"})
    write_sheet_data_to_db(sheets, f"sqlite:///{target}", purge_first=True, keep_entities=True)
    values, entities, activity = _snapshot(f"sqlite:///{target}")
    src_values, src_entities, src_activity = _snapshot(f"sqlite:///{examples_db}")
    assert entities == src_entities
    assert activity == src_activity
    assert values.keys() == src_values.keys()
    differing = {k for k in src_values if values[k] != src_values[k]}
    assert differing == KNOWN_ROUND_TRIP_DEFECTS
    # Scenario alternatives come back with the database's ranks (numbered from 1).
    assert _scenario_ranks(f"sqlite:///{target}") == _scenario_ranks(f"sqlite:///{examples_db}")


def _reader_records(path: Path) -> dict[str, list[str]]:
    """The reader's records per sheet, as sorted reprs (formatting and column order must not change them)."""
    out = {}
    for sheet in read_self_describing_excel(str(path), skip_sheets={"navigate", "version"}):
        rows = [repr(sorted((k, repr(v)) for k, v in rec.items())) for rec in sheet.records]
        out[sheet.sheet_name] = sorted(rows) + sorted(repr(e) for e in sheet.link_entities)
    return out


def test_without_description_row_reads_back_the_same(
    examples_db: Path, workbook_path: Path, tmp_path: Path,
) -> None:
    out = tmp_path / "no_description.xlsx"
    export_to_excel(f"sqlite:///{examples_db}", str(out), include_advanced=True, description_row=False)
    book = openpyxl.load_workbook(out)
    try:
        ws = book["unit_c"]
        # Rows: data type, default, definition row 3; the key columns stay frozen.
        assert [ws.cell(row=r, column=3).value for r in (1, 2, 3)] == ["data type", "default", "parameter"]
        assert ws.freeze_panes == "C4"
        assert ws.cell(row=1, column=1).value == "navigate"
        col = _column_of(ws, 3, "conversion_method")
        assert ws.cell(row=3, column=col).comment is not None  # the description is still in the note
    finally:
        book.close()
    assert _reader_records(out) == _reader_records(workbook_path)


def test_without_description_row_needs_the_v2_format(examples_db: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v2"):
        export_to_excel(f"sqlite:///{examples_db}", str(tmp_path / "v1.xlsx"), use_new_format=False,
                        description_row=False)


RAMP_JSON = package_data_path("schemas/canonical_databases/howto_ramp_and_start_up.json")


def test_empty_parameter_groups_start_collapsed(tmp_path: Path) -> None:
    # howto_ramp_and_start_up: units have values only in the basics and tech_advanced groups.  On unit_c the
    # columns are ordered by group (basics, investment, retirement, tech_advanced); every group after the first is
    # an outline group, collapsed (hidden, with ``collapsed`` on the column to its left) when it has no value.
    db = tmp_path / "ramp.sqlite"
    initialize_database(str(RAMP_JSON), str(db))
    out = tmp_path / "ramp.xlsx"
    export_to_excel(f"sqlite:///{db}", str(out))
    with DatabaseMapping(f"sqlite:///{db}") as mapping:
        group_of = {d["name"]: d["parameter_group_name"]
                    for d in mapping.get_parameter_definition_items(entity_class_name="unit")}
    book = openpyxl.load_workbook(out)
    try:
        ws = book["unit_c"]
        groups: dict[str, list[int]] = {}
        for col in range(5, ws.max_column + 1):  # D is entity existence
            name = ws.cell(row=4, column=col).value
            if not name:
                break
            groups.setdefault(group_of[name], []).append(col)
        assert list(groups) == ["basics", "investment", "retirement", "tech_advanced"]
        for group, cols in groups.items():
            assert cols == list(range(cols[0], cols[-1] + 1)), group  # one block per group
            dims = [ws.column_dimensions[get_column_letter(c)] for c in cols]
            if group == "basics":
                assert all(d.outlineLevel == 0 and not d.hidden for d in dims)
            elif group == "tech_advanced":
                assert all(d.outlineLevel == 1 and not d.hidden for d in dims)
            else:
                assert all(d.outlineLevel == 1 and d.hidden for d in dims)
                assert ws.column_dimensions[get_column_letter(cols[0] - 1)].collapsed
        assert ws.sheet_properties.outlinePr.summaryRight is False
    finally:
        book.close()
    with zipfile.ZipFile(out) as archive:
        # openpyxl's own reader drops outlineLevelCol, so look at the sheet XML (sheets are written in order).
        sheet_xml = archive.read(f"xl/worksheets/sheet{book.sheetnames.index('unit_c') + 1}.xml").decode("utf-8")
        assert 'outlineLevelCol="1"' in sheet_xml


def _parameter_columns(ws, def_row: int) -> dict[str, int]:
    """Parameter name -> column, right of the definition column up to the blank separator."""
    def_col = _column_of(ws, def_row, "parameter")
    cols = {}
    for col in range(def_col + 1, ws.max_column + 1):
        name = ws.cell(row=def_row, column=col).value
        if not name:
            break
        cols[name] = col
    return cols


def _columns_with_values(ws, def_row: int) -> set[str]:
    return {name for name, col in _parameter_columns(ws, def_row).items()
            if any(ws.cell(row=r, column=col).value is not None for r in range(def_row + 1, ws.max_row + 1))}


def test_omit_empty_columns(examples_db: Path, wb: openpyxl.Workbook, workbook_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "compact.xlsx"
    export_to_excel(f"sqlite:///{examples_db}", str(out), include_advanced=True, omit_empty_columns=True)
    compact = openpyxl.load_workbook(out)
    try:
        for sheet in ("node_c", "unit_c", "node_p", "solve"):
            full_ws, compact_ws = wb[sheet], compact[sheet]
            # The definition row is the last frozen row (a sheet loses its default row when no kept column has one).
            full_row, compact_row = (int(w.freeze_panes.lstrip("ABCDEFGHIJ")) - 1 for w in (full_ws, compact_ws))
            full_params = _parameter_columns(full_ws, full_row)
            # The compact sheet keeps exactly the columns of the full export that hold a value.
            assert set(_parameter_columns(compact_ws, compact_row)) == _columns_with_values(full_ws, full_row), sheet
            assert len(_parameter_columns(compact_ws, compact_row)) < len(full_params), sheet
            # The reference block still names every parameter of the sheet.
            reference = {compact_ws.cell(row=r, column=c).value
                         for r in range(1, compact_ws.max_row + 1) for c in range(1, compact_ws.max_column + 1)}
            assert set(full_params) - {"entity existence"} <= reference, sheet
    finally:
        compact.close()
    assert _reader_records(out) == _reader_records(workbook_path)


def test_omit_empty_columns_needs_the_v2_format(examples_db: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v2"):
        export_to_excel(f"sqlite:///{examples_db}", str(tmp_path / "v1.xlsx"), use_new_format=False,
                        omit_empty_columns=True)


def test_floats_survive_the_workbook_exactly(tmp_path: Path) -> None:
    # openpyxl writes "%.16g": 0.1 + 0.2 (0.30000000000000004) would come back as 0.3.  The export writes repr().
    exact = 0.1 + 0.2
    third = 1 / 7  # 0.14285714285714285: 17 significant digits
    assert exact != 0.3 and float("%.16g" % third) != third
    db = tmp_path / "floats.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    with DatabaseMapping(f"sqlite:///{db}") as mapping:
        mapping.add_entity(entity_class_name="unit", entity_byname=("u",))
        for name, value in (("existing", exact), ("efficiency", third)):
            v, t = to_database(value)
            mapping.add_parameter_value(entity_class_name="unit", entity_byname=("u",),
                                        parameter_definition_name=name, alternative_name="base", value=v, type=t)
        mapping.commit_session("floats")
    out = tmp_path / "floats.xlsx"
    export_to_excel(f"sqlite:///{db}", str(out))
    values = {rec["param_name"]: rec["value"]
              for sheet in read_self_describing_excel(str(out)) if sheet.sheet_name == "unit_c"
              for rec in sheet.records if rec.get("param_name") in ("existing", "efficiency")}
    assert values == {"existing": exact, "efficiency": third}


def test_array_parameters_are_labelled_string_array(wb: openpyxl.Workbook) -> None:
    # solve.contains_solves and model.solves are Arrays of solve names (schema type list: array only).
    ws = wb["solve"]
    def_row = int(ws.freeze_panes.lstrip("ABCDEFGHIJ")) - 1
    def_col = _column_of(ws, def_row, "parameter")
    type_row = next(r for r in range(1, def_row) if ws.cell(row=r, column=def_col).value == "data type")
    assert ws.cell(row=type_row, column=_column_of(ws, def_row, "contains_solves")).value == "string (array)"
    assert "data type: string (array)" in wb["model_solve_sequence"]["B2"].value
    # No sheet offers a TRUE / FALSE dropdown on an array column.
    for sheet in wb.worksheets:
        for dv in sheet.data_validations.dataValidation:
            if "TRUE" in (dv.formula1 or ""):
                first = str(dv.sqref).split()[0]
                c0, r0, _c1, _r1 = range_boundaries(first)
                assert sheet.cell(row=r0 - 1, column=c0).value == "entity existence", (sheet.title, first)
