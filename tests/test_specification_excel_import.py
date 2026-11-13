"""Import of FlexTool 3.x specification-format Excel files (``import_old_excel_input.json``).

The importer locates every column by its header name, so a workbook in the
current template layout and one whose columns sit in an older template order
must import to the same values.  Both workbooks here are synthetic and hold
the same data; every expected value below is read off the synthetic inputs by
hand.

Covered behaviours:
* activity column found by header (``Entity Alternative`` / legacy ``is_active``),
  'no' read as inactive;
* parameter cells typed by their header, not by column position (a number in
  a column the current template reserves for a text choice stays a number);
* a period series ('_p') overrides the constant ('_c'); periods the series
  leaves blank keep the constant;
* a blank ``*_min_period`` period is written as 0; other blank periods are
  reported, not invented;
* legacy ``solve_mode`` values;
* missing optional columns and an empty time-series sheet are quiet;
* the same map written by two mappings (reserve_group_t) is a quiet no-op;
* an unknown column header is reported and the final status is honest.
"""
from __future__ import annotations

import json
from pathlib import Path

import openpyxl
import pytest
from spinedb_api import DatabaseMapping, Map
from spinedb_api.parameter_value import Array

from flextool.process_inputs.read_tabular_with_specification import TabularReader
from flextool.process_inputs.write_to_input_db import write_to_flextool_input_db
from flextool.update_flextool.initialize_database import initialize_database

_PKG = Path(__file__).resolve().parent.parent / "flextool"
_SPEC = _PKG / "process_inputs" / "import_old_excel_input.json"
_V25_TEMPLATE = _PKG / "schemas" / "pre_v26" / "flextool_template_v25.json"


def _current_columns(sheet: str) -> list[str]:
    """Header row of a sheet in the current template layout."""
    return json.loads(_SPEC.read_text())["tables"][sheet]["layout"]["columns"]


def _row(headers: list[str], **values) -> list:
    """A data row for ``headers`` with ``values`` placed under their header."""
    unknown = set(values) - set(headers)
    assert not unknown, unknown
    return [values.get(h) for h in headers]


def _write_column_sheet(wb: openpyxl.Workbook, name: str, headers: list[str],
                        rows: list[dict]) -> None:
    """Row 1: navigate text, row 2: headers, then data (the template's column sheets)."""
    ws = wb.create_sheet(name)
    ws.append(["navigate"])
    ws.append(headers)
    for values in rows:
        ws.append(_row(headers, **values))


# The model data, shared by both workbooks.  Keys are header names.
_SOLVE_PERIOD_ROWS = [
    dict(alternative="Base", solve="s1", period=p, period_timeset="ts1",
         years_represented=1, realize_dispatch="yes", can_invest="yes")
    for p in ("p1", "p2", "p3")
]
_SOLVE_SEQUENCE_ROWS = [dict(alternative="Base", solve="s1", solve_mode="invest", solver="highs")]
_NODE_C_ROWS = [
    {"alternative": "Base", "node": "n1", "Entity Alternative": "yes", "has_balance": "yes",
     "inflow": -10, "fixed_cost": 5},
    {"alternative": "Base", "node": "n2", "Entity Alternative": "no", "has_balance": "yes"},
]
_NODE_P_ROWS = [
    dict(alternative="Base", node="n1", period="p1", fixed_cost=7, invest_min_period=3,
         annual_flow=1000),
    dict(alternative="Base", node="n1", period="p2"),
    dict(alternative="Base", node="n1", period="p3", fixed_cost=9),
]
_UNIT_C_ROWS = [
    {"alternative": "Base", "unit": "u1", "Entity Alternative": "yes",
     "conversion_method": "constant_efficiency", "existing": 100, "invest_cost": 1000},
]
_UNIT_P_ROWS = [dict(alternative="Base", unit="u1", period=p, invest_cost=800)
                for p in ("p1", "p2", "p3")]
_CONSTRAINT_ROWS = [
    {"alternative": "Base", "constraint": "c1", "Entity Alternative": "yes",
     "sense": "greater_than", "constant": 2},
]
_GROUP_C_ROWS = [dict(alternative="Base", group="g1", capacity_margin=0.05)]
_RESERVE_GROUP_C_ROWS = [dict(alternative="Base", reserve="r1", upDown="up", group="g1",
                              reserve_method="timeseries_only")]


def _write_reserve_group_t(wb: openpyxl.Workbook) -> None:
    """Time-series sheet: header labels in column B, time in column A."""
    ws = wb.create_sheet("reserve_group_t")
    ws.append(["navigate", "alternative", "Base"])
    ws.append([None, "reserve", "r1"])
    ws.append([None, "upDown", "up"])
    ws.append([None, "group", "g1"])
    ws.append(["time", "parameter", "reservation"])
    ws.append(["t1", None, 40])
    ws.append(["t2", None, 50])


def _write_empty_unit_node_t(wb: openpyxl.Workbook) -> None:
    """A time-series sheet with header labels only (no data)."""
    ws = wb.create_sheet("unit_node_t")
    for label in ("alternative", "parameter", "unit", "node"):
        ws.append([None, label])
    ws.append(["time", "input_output"])
    ws.append(["t1"])


def _rename(rows: list[dict], old: str, new: str) -> list[dict]:
    return [{(new if k == old else k): v for k, v in r.items()} for r in rows]


def _current_layout_workbook(path: Path) -> Path:
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    _write_column_sheet(wb, "solve_period", _current_columns("solve_period"), _SOLVE_PERIOD_ROWS)
    _write_column_sheet(wb, "solve_sequence", _current_columns("solve_sequence"), _SOLVE_SEQUENCE_ROWS)
    _write_column_sheet(wb, "node_c", _current_columns("node_c"), _NODE_C_ROWS)
    _write_column_sheet(wb, "node_p", _current_columns("node_p"), _NODE_P_ROWS)
    _write_column_sheet(wb, "unit_c", _current_columns("unit_c"), _UNIT_C_ROWS)
    _write_column_sheet(wb, "unit_p", _current_columns("unit_p"), _UNIT_P_ROWS)
    _write_column_sheet(wb, "constraint_sense_c", _current_columns("constraint_sense_c"),
                        _CONSTRAINT_ROWS)
    _write_column_sheet(wb, "group_c", _current_columns("group_c"), _GROUP_C_ROWS)
    _write_column_sheet(wb, "reserve_group_c", _current_columns("reserve_group_c"),
                        _RESERVE_GROUP_C_ROWS)
    _write_reserve_group_t(wb)
    _write_empty_unit_node_t(wb)
    wb.save(path)
    return path


def _older_layout_workbook(path: Path, extra_node_column: str | None = None) -> Path:
    """Same data in an older template's column order and names.

    * node_c: ``is_active`` before ``has_balance`` (current: after it).
    * unit_c: ``existing`` in column 8, which the current template gives to
      the text choice ``lifetime_method``.
    * constraint_sense_c: ``sense`` before ``is_active`` (current: after).
    * solve_period: legacy ``period_timeblockSet`` / ``is_realized`` headers and
      no ``realized_invest_periods`` / ``fix_storage_periods`` columns.
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    sp_headers = ["alternative", "solve", "period", "period_timeblockSet", "years_represented",
                  "is_realized", "can_invest"]
    sp_rows = _rename(_rename(_SOLVE_PERIOD_ROWS, "period_timeset", "period_timeblockSet"),
                      "realize_dispatch", "is_realized")
    _write_column_sheet(wb, "solve_period", sp_headers, sp_rows)
    _write_column_sheet(wb, "solve_sequence",
                        ["alternative", "solve", "solve_mode", "solver"], _SOLVE_SEQUENCE_ROWS)
    node_headers = ["alternative", "node", "is_active", "has_balance", "inflow", "fixed_cost"]
    node_rows = _rename(_NODE_C_ROWS, "Entity Alternative", "is_active")
    if extra_node_column:
        node_headers.append(extra_node_column)
        node_rows = [dict(r, **{extra_node_column: 1}) for r in node_rows]
    _write_column_sheet(wb, "node_c", node_headers, node_rows)
    _write_column_sheet(wb, "node_p", ["alternative", "node", "period", "annual_flow",
                                       "invest_min_period", "fixed_cost"], _NODE_P_ROWS)
    unit_headers = ["alternative", "unit", "is_active", "conversion_method", "startup_method",
                    "minimum_time_method", "invest_method", "existing", "invest_cost"]
    assert unit_headers.index("existing") == _current_columns("unit_c").index("lifetime_method")
    _write_column_sheet(wb, "unit_c", unit_headers,
                        _rename(_UNIT_C_ROWS, "Entity Alternative", "is_active"))
    _write_column_sheet(wb, "unit_p", ["alternative", "unit", "period", "invest_cost"], _UNIT_P_ROWS)
    _write_column_sheet(wb, "constraint_sense_c",
                        ["alternative", "constraint", "sense", "is_active", "constant"],
                        _rename(_CONSTRAINT_ROWS, "Entity Alternative", "is_active"))
    _write_column_sheet(wb, "group_c", ["alternative", "group", "output_results",
                                        "has_capacity_margin", "capacity_margin"], _GROUP_C_ROWS)
    _write_column_sheet(wb, "reserve_group_c", _current_columns("reserve_group_c")[:5],
                        _RESERVE_GROUP_C_ROWS)
    _write_reserve_group_t(wb)
    _write_empty_unit_node_t(wb)
    wb.save(path)
    return path


def _import(xlsx: Path, db: Path) -> TabularReader:
    initialize_database(str(_V25_TEMPLATE), str(db))
    reader = TabularReader(str(_SPEC))
    write_to_flextool_input_db(str(xlsx), reader, f"sqlite:///{db}", "excel",
                               migration_follows=True)
    return reader


def _values(db: Path) -> dict:
    """{(class, entity, parameter, alternative): parsed value} and entity activities."""
    out = {}
    with DatabaseMapping(f"sqlite:///{db}") as m:
        for pv in m.find_parameter_values():
            key = (pv["entity_class_name"], "__".join(pv["entity_byname"]),
                   pv["parameter_definition_name"], pv["alternative_name"])
            out[key] = pv["parsed_value"]
        for ea in m.find_entity_alternatives():
            key = (ea["entity_class_name"], "__".join(ea["entity_byname"]), "<active>",
                   ea["alternative_name"])
            out[key] = ea["active"]
    return out


def _as_dict(value) -> dict:
    assert isinstance(value, Map), value
    return dict(zip(value.indexes, value.values))


@pytest.fixture(scope="module")
def imported(tmp_path_factory: pytest.TempPathFactory):
    work = tmp_path_factory.mktemp("spec_import")
    results = {}
    for name, build in (("current", _current_layout_workbook), ("older", _older_layout_workbook)):
        reader = _import(build(work / f"{name}.xlsx"), work / f"{name}.sqlite")
        results[name] = (reader, _values(work / f"{name}.sqlite"))
    return results


@pytest.mark.parametrize("layout", ["current", "older"])
def test_values_read_by_header(imported, layout):
    reader, v = imported[layout]
    assert reader.error_count == 0
    # Activity column located by header; 'no' is inactive.
    assert v[("node", "n1", "<active>", "Base")] is True
    assert v[("node", "n2", "<active>", "Base")] is False
    assert v[("unit", "u1", "<active>", "Base")] is True
    assert v[("constraint", "c1", "<active>", "Base")] is True
    # has_balance / sense are not consumed as the activity column.
    assert v[("node", "n1", "has_balance", "Base")] == "yes"
    assert v[("node", "n2", "has_balance", "Base")] == "yes"
    assert v[("constraint", "c1", "sense", "Base")] == "greater_than"
    assert v[("constraint", "c1", "constant", "Base")] == 2.0
    # Numbers stay numbers wherever their column sits.
    assert v[("unit", "u1", "existing", "Base")] == 100.0
    assert v[("group", "g1", "capacity_margin", "Base")] == 0.05
    assert v[("node", "n1", "inflow", "Base")] == -10.0
    assert v[("unit", "u1", "conversion_method", "Base")] == "constant_efficiency"


@pytest.mark.parametrize("layout", ["current", "older"])
def test_period_series_and_blank_periods(imported, layout):
    _, v = imported[layout]
    # _p overrides _c; the blank p2 keeps the constant 5.
    assert _as_dict(v[("node", "n1", "fixed_cost", "Base")]) == {"p1": 7.0, "p2": 5.0, "p3": 9.0}
    assert _as_dict(v[("unit", "u1", "invest_cost", "Base")]) == {"p1": 800.0, "p2": 800.0,
                                                                  "p3": 800.0}
    # A blank minimum is no minimum: 0.
    assert _as_dict(v[("node", "n1", "invest_min_period", "Base")]) == {"p1": 3.0, "p2": 0.0,
                                                                        "p3": 0.0}
    # Any other blank period is left out (and reported), never invented.
    assert _as_dict(v[("node", "n1", "annual_flow", "Base")]) == {"p1": 1000.0}


@pytest.mark.parametrize("layout", ["current", "older"])
def test_solve_settings(imported, layout):
    _, v = imported[layout]
    assert v[("solve", "s1", "solve_mode", "Base")] == "single_solve"
    for param in ("realized_periods", "invest_periods"):
        value = v[("solve", "s1", param, "Base")]
        assert isinstance(value, Array)
        assert list(value.values) == ["p1", "p2", "p3"]
    assert _as_dict(v[("solve", "s1", "period_timeset", "Base")]) == {
        "p1": "ts1", "p2": "ts1", "p3": "ts1"}
    assert ("solve", "s1", "realized_invest_periods", "Base") not in v
    # reserve_group_t: two mappings write the same map — one value, no error.
    assert _as_dict(v[("reserve__upDown__group", "r1__up__g1", "reservation", "Base")]) == {
        "t1": 40.0, "t2": 50.0}


def test_both_layouts_import_identically(imported):
    assert imported["current"][1] == imported["older"][1]


def test_blank_period_warning_and_status(imported):
    reader, _ = imported["current"]
    # The only warning: annual_flow of n1 has no value for p2 and p3.
    assert reader.warning_count == 1


def test_unknown_column_is_reported(tmp_path, capsys):
    xlsx = _older_layout_workbook(tmp_path / "unknown.xlsx", extra_node_column="no_such_column")
    reader = _import(xlsx, tmp_path / "unknown.sqlite")
    out = capsys.readouterr()
    text = out.out + out.err
    assert "Sheet 'node_c': column 'no_such_column' is not a column of this sheet" in text
    assert reader.warning_count == 2  # the unknown column + the blank annual_flow periods
    assert reader.error_count >= 1    # its values have no parameter to land in
    assert "All data processed successfully!" not in text
    assert "Import finished with" in text
