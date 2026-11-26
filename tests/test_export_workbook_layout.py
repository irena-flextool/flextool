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


def test_text_width_in_zero_widths() -> None:
    from flextool.export_to_tabular.formatting import text_width
    # Calibri / Carlito advance widths in widths of "0" (CHAR_WIDTHS): e .982 x .855 i .453 s .772 t .661 n 1.037
    # g .929 -> existing = .982 + .855 + .453 + .772 + .661 + .453 + 1.037 + .929 = 6.142
    assert text_width("existing") == pytest.approx(6.142)
    assert text_width("p2020") == pytest.approx(1.037 + 4 * 1.0)
    assert text_width("é") == pytest.approx(1.0)  # outside the table: one "0"


def test_column_widths_fit_content(wb: openpyxl.Workbook) -> None:
    # width = clamp(ceil(widest text in widths of "0" + 1.0), lo, hi); text widths from CHAR_WIDTHS.
    unit_c = wb["unit_c"]
    assert _width(unit_c, 1) == 26  # alternative invest_solveSequence_5weeks: 24.741 -> ceil(25.741)
    assert _width(unit_c, 2) == 19  # unit dr_decrease_demand: 17.527 -> 19
    # conversion_method (16.090) under its default constant_efficiency (15.642) -> 18
    assert _width(unit_c, _column_of(unit_c, 4, "conversion_method")) == 18
    assert _width(unit_c, _column_of(unit_c, 4, "existing")) == 8  # existing 6.142 -> 8, the lower bound
    unit_node_c = wb["unit_node_c"]
    assert _width(unit_node_c, 2) == 8  # entity: (...) reader label, no data
    assert _width(unit_node_c, 5) == 8  # filter column: output 5.473 -> 7, lower bound 8
    node_t = wb["node_t"]
    assert _width(node_t, 1) == 11  # index: time 9.010 -> 11
    assert _width(node_t, 2) == 11  # entity: node 9.757 -> 11; the INFO text does not count
    assert _width(node_t, 3) == 18  # alternative desal_water_storage 16.858 -> 18
    assert _width(node_t, 4) == 10  # entity ev_battery 8.723 -> 10; the data type row does not count


def test_columns_never_narrower_than_their_names(wb: openpyxl.Workbook) -> None:
    from flextool.export_to_tabular.formatting import WIDTH_BOUNDS, text_width
    # Excel draws a cell's text inside a margin of 5 pixels (about 0.7 of a "0"); below the upper bound a column
    # leaves room for its parameter name and that margin.
    hi = WIDTH_BOUNDS["data"][1]
    for sheet in ("unit_c", "node_c", "connection_c", "unit_node_c"):
        ws = wb[sheet]
        def_row = _definition_row(ws)
        for name, col in _parameter_columns(ws, def_row).items():
            if text_width(name) + 0.7 < hi:
                assert _width(ws, col) >= text_width(name) + 0.7, (sheet, name)


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
        activity = {(e["entity_class_name"], tuple(e["entity_byname"]), e["alternative_name"]): e["active"]
                    for e in db.get_entity_alternative_items()}
    return values, entities, activity


def test_time_series_default_row_is_not_a_time_step(wb: openpyxl.Workbook, workbook_path: Path) -> None:
    # unit_t carries several parameters, so it has a parameter row and, because availability has a default (1.0),
    # a ``default`` row under it with ``index: time`` in column A.  In templates_examples.json coal_plant's
    # availability in alternative ``availability`` is a map over t0001..t0072 (72 steps); the default row must not
    # add a 73rd step.
    ws = wb["unit_t"]
    labels = [ws.cell(row=r, column=2).value for r in range(1, 8)]
    assert "default" in labels
    steps = [rec["index_value"]
             for sheet in read_self_describing_excel(str(workbook_path)) if sheet.sheet_name == "unit_t"
             for rec in sheet.records
             if rec["entity_byname"] == ("coal_plant",) and rec["alternative"] == "availability"]
    assert steps == [f"t{i:04d}" for i in range(1, 73)]


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
    # Every value comes back (water_pump's delay included: its float indexes stay numbers).
    assert values == src_values
    # Scenario alternatives come back with the database's ranks (numbered from 1).
    assert _scenario_ranks(f"sqlite:///{target}") == _scenario_ranks(f"sqlite:///{examples_db}")


def _reader_records(path: Path) -> dict[str, list[str]]:
    """The reader's records per sheet, as sorted reprs (formatting and column order must not change them)."""
    out = {}
    for sheet in read_self_describing_excel(str(path), skip_sheets={"navigate", "version"}):
        rows = [repr(sorted((k, repr(v)) for k, v in rec.items())) for rec in sheet.records]
        out[sheet.sheet_name] = sorted(rows) + sorted(repr(e) for e in sheet.link_entities)
    return out


def test_units_row_reads_back_the_same(examples_db: Path, workbook_path: Path, tmp_path: Path) -> None:
    out = tmp_path / "units_row.xlsx"
    export_to_excel(f"sqlite:///{examples_db}", str(out), include_advanced=True, units_row=True)
    book = openpyxl.load_workbook(out)
    try:
        ws = book["unit_c"]
        # Same rows as the default export (unit, data type, default, definition row 4), the first labelled unit.
        assert [ws.cell(row=r, column=3).value for r in (1, 2, 3, 4)] == ["unit", "data type", "default", "parameter"]
        assert ws.freeze_panes == "C5"
        # The unit is the bracket the schema description starts with, without the brackets:
        # existing "[MW or count] ...", fixed_cost "[CUR/kW/a] ...", availability "[-] ...";
        # conversion_method "[choice] ..." is no unit, so its cell stays empty.
        expected = {"existing": "MW or count", "fixed_cost": "CUR/kW/a", "availability": "-",
                    "conversion_method": None}
        for name, unit in expected.items():
            col = _column_of(ws, 4, name)
            assert ws.cell(row=1, column=col).value == unit, name
            assert ws.cell(row=4, column=col).comment is not None  # the description is in the note
        col = _column_of(ws, 4, "existing")
        assert ws.cell(row=1, column=col).fill.fgColor.rgb[-6:] == "FFF2CC"
        assert not ws.cell(row=1, column=col).alignment.wrap_text
        assert ws.row_dimensions[1].height is None  # one line
    finally:
        book.close()
    assert _reader_records(out) == _reader_records(workbook_path)


def test_units_row_needs_the_v2_format(examples_db: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="v2"):
        export_to_excel(f"sqlite:///{examples_db}", str(tmp_path / "v1.xlsx"), use_new_format=False,
                        units_row=True)


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


@pytest.fixture(scope="module")
def compact_path(examples_db: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("compact_xlsx") / "compact.xlsx"
    export_to_excel(f"sqlite:///{examples_db}", str(out), include_advanced=True, omit_empty_columns=True)
    return out


def _definition_row(ws) -> int:
    """The definition row is the last frozen row (a sheet loses its default row when no kept column has one)."""
    return int(ws.freeze_panes.lstrip("ABCDEFGHIJ")) - 1


def test_omit_empty_columns(compact_path: Path, wb: openpyxl.Workbook, workbook_path: Path) -> None:
    out = compact_path
    compact = openpyxl.load_workbook(out)
    try:
        for sheet in ("node_c", "unit_c", "node_p", "solve"):
            full_ws, compact_ws = wb[sheet], compact[sheet]
            full_row, compact_row = _definition_row(full_ws), _definition_row(compact_ws)
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


def test_omit_empty_columns_keeps_a_sheet_without_values_whole(compact_path: Path, wb: openpyxl.Workbook) -> None:
    # In templates_examples no commodity has a period-indexed price, so commodity_p has no value at all.  Without
    # any parameter column the reader would take the sheet for a link sheet, so it keeps all its columns.
    full_ws = wb["commodity_p"]
    assert _columns_with_values(full_ws, _definition_row(full_ws)) == set()
    compact = openpyxl.load_workbook(compact_path)
    try:
        compact_ws = compact["commodity_p"]
        assert (_parameter_columns(compact_ws, _definition_row(compact_ws))
                == _parameter_columns(full_ws, _definition_row(full_ws)) != {})
    finally:
        compact.close()


def test_exact_floats_falls_back_when_openpyxl_moves_its_formatter(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    # A future openpyxl without openpyxl.cell._writer.safe_string: the save goes ahead unpatched, with a warning.
    import openpyxl.cell._writer as cell_writer

    from flextool.export_to_tabular.export_to_excel import _exact_floats
    monkeypatch.delattr(cell_writer, "safe_string")
    with caplog.at_level("WARNING"), _exact_floats():
        assert not hasattr(cell_writer, "safe_string")
    assert not hasattr(cell_writer, "safe_string")
    assert "16 significant digits" in caplog.text


def test_exact_floats_restores_the_formatter_in_place_on_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    import openpyxl.cell._writer as cell_writer

    from flextool.export_to_tabular.export_to_excel import _exact_floats

    def formatter_in_place(value: object) -> str:
        return str(value)

    monkeypatch.setattr(cell_writer, "safe_string", formatter_in_place)
    with _exact_floats():
        assert cell_writer.safe_string is not formatter_in_place
        assert cell_writer.safe_string(0.1 + 0.2) == "0.30000000000000004"
        assert cell_writer.safe_string("text") == "text"  # anything else goes to the formatter in place
    assert cell_writer.safe_string is formatter_in_place


def _index_name_path(value: object) -> tuple[str, ...]:
    if isinstance(value, Array):
        return (value.index_name,)
    if isinstance(value, Map):
        first = value.values[0] if len(value.values) else None
        return (value.index_name,) + (_index_name_path(first) if isinstance(first, Map) else ())
    return ()


def test_index_names_survive_the_workbook(tmp_path: Path) -> None:
    # Spine's default index name (x for a Map, i for an Array) comes back as the parameter's meaningful name; a
    # name of its own stays; a profile map keeps x.  Expected names, by hand:
    #   unit.invest_cost  {p2020, p2025}, x       -> period   (its shape: a map over periods)
    #   unit.delay        {2, 3}, named "lag"     -> lag      (own name; unit_p gets an index name row)
    #   timeset.timeset_duration {t0001}, x       -> duration
    #   solve.solver_arguments {presolve}, x      -> argument
    #   model.solves      Array, i                -> sequence_index
    #   profile.profile   {t0001, t0002}, x       -> x        (a time series: the timeline names t0001, t0002)
    #   unit.piecewise_linear_efficiency {p2020: {0.25, 1.0}}, x / x -> period / load (its per-period curve shape)
    db = tmp_path / "index_names.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    url = f"sqlite:///{db}"
    values = {
        ("unit", ("u",), "invest_cost"): Map(["p2020", "p2025"], [100.0, 90.0]),
        ("unit", ("u",), "delay"): Map([2.0, 3.0], [0.5, 0.5], index_name="lag"),
        ("timeline", ("tl",), "timestep_duration"): Map(["t0001", "t0002"], [1.0, 1.0], index_name="timestep"),
        ("timeset", ("ts",), "timeset_duration"): Map(["t0001"], [2.0]),
        ("solve", ("s",), "solver_arguments"): Map(["presolve"], ["on"]),
        ("model", ("m",), "solves"): Array(["s"]),
        ("profile", ("pr",), "profile"): Map(["t0001", "t0002"], [0.5, 0.7]),
        ("unit", ("u",), "piecewise_linear_efficiency"): Map(["p2020"], [Map([0.25, 1.0], [0.3, 0.4])]),
    }
    expected = {
        ("unit", ("u",), "invest_cost"): ("period",),
        ("unit", ("u",), "delay"): ("lag",),
        ("timeline", ("tl",), "timestep_duration"): ("timestep",),
        ("timeset", ("ts",), "timeset_duration"): ("duration",),
        ("solve", ("s",), "solver_arguments"): ("argument",),
        ("model", ("m",), "solves"): ("sequence_index",),
        ("profile", ("pr",), "profile"): ("x",),
        ("unit", ("u",), "piecewise_linear_efficiency"): ("period", "load"),
    }
    with DatabaseMapping(url) as mapping:
        for cls, byname in {(c, b) for c, b, _p in values}:
            mapping.add_entity(entity_class_name=cls, entity_byname=byname)
        for (cls, byname, name), value in values.items():
            v, t = to_database(value)
            mapping.add_parameter_value(entity_class_name=cls, entity_byname=byname, parameter_definition_name=name,
                                        alternative_name="base", value=v, type=t)
        mapping.commit_session("index names")
    out = tmp_path / "index_names.xlsx"
    export_to_excel(url, str(out), include_advanced=True)
    target = tmp_path / "round_trip.sqlite"
    initialize_database(str(SCHEMA_JSON), str(target))
    write_sheet_data_to_db(read_self_describing_excel(str(out)), f"sqlite:///{target}", purge_first=True,
                           keep_entities=True)
    with DatabaseMapping(f"sqlite:///{target}") as mapping:
        parsed = {(p["entity_class_name"], tuple(p["entity_byname"]), p["parameter_definition_name"]):
                  from_database(p["value"], p["type"]) for p in mapping.get_parameter_value_items()}
    got = {k: _index_name_path(v) for k, v in parsed.items()}
    delay_indexes = list(parsed[("unit", ("u",), "delay")].indexes)
    curve = parsed[("unit", ("u",), "piecewise_linear_efficiency")]
    curve_loads = list(curve.values[0].indexes)
    assert {k: got.get(k) for k in expected} == expected
    # The delay's indexes are numbers (hours), not text.
    assert delay_indexes == [2.0, 3.0] and all(isinstance(i, float) for i in delay_indexes)
    # So are the load shares of the per-period part-load curve, with their values.
    assert list(curve.indexes) == ["p2020"] and curve_loads == [0.25, 1.0]
    assert all(isinstance(i, float) for i in curve_loads) and list(curve.values[0].values) == [0.3, 0.4]
    book = openpyxl.load_workbook(out)
    try:
        ws = book["unit_p"]
        def_row = _definition_row(ws)
        def_col = _column_of(ws, def_row, "parameter")
        labels = [ws.cell(row=r, column=def_col).value for r in range(1, def_row)]
        assert "index name" in labels  # invest_cost (period) and delay (lag) share the sheet
    finally:
        book.close()


def test_entity_activity_of_every_class_survives_the_workbook(tmp_path: Path) -> None:
    # Activity rows (entity_alternative) of a reserve, a constraint (classes with a constant sheet: entity existence
    # column there) and a group__node link (a link-only class: entity existence column on the link sheet), all
    # switched off in alternative "off"; a second group__node link has no activity row.  All come back as written.
    db = tmp_path / "activity.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    url = f"sqlite:///{db}"
    rows = {
        ("reserve__upDown__group", ("r", "up", "g")): False,
        ("constraint", ("c1",)): False,
        ("group__node", ("g", "n")): False,
    }
    with DatabaseMapping(url) as mapping:
        mapping.add_alternative(name="off")
        for cls, byname in (("reserve", ("r",)), ("group", ("g",)), ("node", ("n",)), ("node", ("m",)),
                            ("constraint", ("c1",)), ("reserve__upDown__group", ("r", "up", "g")),
                            ("group__node", ("g", "n")), ("group__node", ("g", "m"))):
            mapping.add_entity(entity_class_name=cls, entity_byname=byname)
        for (cls, byname), active in rows.items():
            mapping.add_entity_alternative(entity_class_name=cls, entity_byname=byname, alternative_name="off",
                                           active=active)
        mapping.commit_session("activity")
    out = tmp_path / "activity.xlsx"
    export_to_excel(url, str(out), include_advanced=True)
    target = tmp_path / "round_trip.sqlite"
    initialize_database(str(SCHEMA_JSON), str(target))
    write_sheet_data_to_db(read_self_describing_excel(str(out)), f"sqlite:///{target}", purge_first=True,
                           keep_entities=True)
    with DatabaseMapping(f"sqlite:///{target}") as mapping:
        got = {(e["entity_class_name"], tuple(e["entity_byname"])): e["active"]
               for e in mapping.get_entity_alternative_items() if e["alternative_name"] == "off"}
        links = {tuple(e["entity_byname"]) for e in mapping.get_entity_items(entity_class_name="group__node")}
    assert got == rows
    assert links == {("g", "n"), ("g", "m")}


def test_sheet_names_fit_excel(tmp_path: Path) -> None:
    # Two stochastic unit parameters give one sheet each, named unit_s_<parameter>: unit_s_availability (19) and
    # unit_s_piecewise_linear_efficiency (34 characters; Excel allows 31) -> unit_s_piecewise_linear_efficie.
    db = tmp_path / "long_names.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db))
    url = f"sqlite:///{db}"
    values = {
        "piecewise_linear_efficiency": Map(["p2020"], [Map([0.25, 1.0], [0.3, 0.4])]),
        "availability": Map(["f1"], [Map(["t0001"], [Map(["t0001"], [0.9])])]),
    }
    with DatabaseMapping(url) as mapping:
        mapping.add_entity(entity_class_name="unit", entity_byname=("u",))
        for name, value in values.items():
            v, t = to_database(value)
            mapping.add_parameter_value(entity_class_name="unit", entity_byname=("u",),
                                        parameter_definition_name=name, alternative_name="base", value=v, type=t)
        mapping.commit_session("long names")
    out = tmp_path / "long_names.xlsx"
    export_to_excel(url, str(out), include_advanced=True)
    book = openpyxl.load_workbook(out)
    try:
        names = book.sheetnames
    finally:
        book.close()
    assert max(len(n) for n in names) <= 31
    assert "unit_s_piecewise_linear_efficie" in names and "unit_s_availability" in names
    target = tmp_path / "round_trip.sqlite"
    initialize_database(str(SCHEMA_JSON), str(target))
    write_sheet_data_to_db(read_self_describing_excel(str(out)), f"sqlite:///{target}", purge_first=True,
                           keep_entities=True)
    with DatabaseMapping(f"sqlite:///{target}") as mapping:
        got = {p["parameter_definition_name"]: _plain(from_database(p["value"], p["type"]))
               for p in mapping.get_parameter_value_items(entity_class_name="unit")}
    assert got == {name: _plain(value) for name, value in values.items()}


def test_fit_sheet_names_keeps_every_candidate_unique_and_short() -> None:
    import json

    from flextool.export_to_tabular.sheet_config import fit_sheet_names
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    # Every <class>_s_<parameter> a stochastic sheet could be named, plus two that cut to the same 31 characters.
    candidates = sorted({f"{cls.replace('__', '_')}_s_{name}" for cls, name, *_ in schema["parameter_types"]})
    candidates += ["x" * 31 + "_a", "x" * 31 + "_b"]
    fitted = fit_sheet_names(candidates)
    assert max(len(n) for n in fitted) <= 31
    assert len(set(fitted)) == len(fitted)
    assert [f for c, f in zip(candidates, fitted) if len(c) <= 31] == [c for c in candidates if len(c) <= 31]
    assert fitted[-2:] == ["x" * 31, "x" * 29 + "~2"]
