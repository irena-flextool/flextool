"""Units of measure of the result columns (#306): the column→unit resolver and the Excel workbook layout.

The expected units are written by hand from the transforms in ``_output_meta`` (RATE = MW, MONEY_ANNUAL =
M CUR/a, ENERGY_ANNUAL = MWh/a, AVERAGE_MW = MW, RESERVE_REQUIREMENT_AVG / _MAX = MW), not copied from the
code's output.  The workbook tests run with warnings as errors: xlsxwriter only warns about an invalid Table
(a name that reads as a cell reference, a repeated header) and Excel then drops the Table.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from flextool.process_outputs._excel_writer import (
    UNIT_LEVEL,
    _measures,
    reserved_sheet_names,
    table_headers,
    table_name,
    units_sheet_row,
    write_excel_with_metadata,
)
from flextool.process_outputs._output_meta import (
    DIMENSION_TABLE,
    OUTPUT_DIMS,
    OUTPUT_TRANSFORM,
    UnitContext,
    column_units,
    derive_column_meta,
    output_metadata_rows,
    resolve_columns,
    result_key_summary,
    row_units,
    summarise_units,
)
from flextool.process_outputs._sheet_names import excel_sheet_name

RESERVE_COLS = pd.MultiIndex.from_tuples(
    [("primary", "up", "south", "requirement_avg"), ("primary", "up", "south", "requirement_max"),
     ("primary", "up", "south", "shortfall_avg")],
    names=["reserve", "updown", "node_group", "quantity"])


@pytest.fixture(autouse=True)
def _warnings_are_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


# ── resolver ────────────────────────────────────────────────────────────────

def test_uniform_table_has_one_unit_for_every_column():
    cols = pd.MultiIndex.from_tuples([("coal", "east"), ("wind", "east")], names=["unit", "node"])
    assert column_units("unit_outputNode_dt_ee", cols) == ["MW", "MW"]
    assert column_units("annualized_costs_d_p", ["commodity_cost", "co2_cost"]) == ["M CUR/a", "M CUR/a"]


def test_reserve_requirement_table_finds_its_units_on_the_quantity_level():
    """The map labels are on the 4th column level; matching on the text of the whole tuple found none."""
    assert column_units("nodeGroup_reserve_d_eegp", RESERVE_COLS) == ["MW", "MW", "MW"]
    meta = derive_column_meta("nodeGroup_reserve_d_eegp", RESERVE_COLS)
    # The parquet-footer keys stay ``str`` of the column label (the assistant rebuilds them).
    key = str(("primary", "up", "south", "requirement_max"))
    assert meta[key].semantics.value == "level"
    assert meta[str(("primary", "up", "south", "requirement_avg"))].semantics.value == "average"
    rows = output_metadata_rows("nodeGroup_reserve_d_eegp", RESERVE_COLS)
    assert [r["unit"] for r in rows] == ["MW", "MW", "MW"]


def test_a_group_named_like_a_map_label_is_not_read_as_one():
    cols = pd.MultiIndex.from_tuples([("primary", "up", "requirement_max", "requirement_avg")],
                                     names=RESERVE_COLS.names)
    [m] = resolve_columns("nodeGroup_reserve_d_eegp", cols)
    assert m.semantics.value == "average"


def test_comparison_scenario_level_is_skipped_with_or_without_names():
    named = pd.MultiIndex.from_tuples([("base", *c) for c in RESERVE_COLS], names=["scenario", *RESERVE_COLS.names])
    assert column_units("nodeGroup_reserve_d_eegp", named) == ["MW", "MW", "MW"]
    unnamed = [("base", "cumulative_flow"), ("base", "average_flow")]
    assert column_units("flowGroup_gd_p", unnamed) == ["MWh/a", "MW"]


def test_comparison_single_scenario_level_gets_the_table_unit():
    cols = pd.Index(["base", "high_wind"], name="scenario")
    assert column_units("node_inflow__dt", cols) == ["MWh", "MWh"]


def test_mixed_single_level_table_resolves_per_column():
    assert column_units("flowGroup_gd_p", pd.Index(["cumulative_flow", "average_flow"], name="parameter")) \
        == ["MWh/a", "MW"]


def test_empty_context_gives_the_default_units():
    default = resolve_columns("nodeGroup_reserve_d_eegp", RESERVE_COLS)
    assert resolve_columns("nodeGroup_reserve_d_eegp", RESERVE_COLS, ctx=UnitContext()) == default
    assert resolve_columns("nodeGroup_reserve_d_eegp", RESERVE_COLS, ctx={"base": UnitContext()}) == default
    assert row_units("node_capacity_ed_p", pd.Index(["east"]), UnitContext()) is None


def test_summarise_units_is_the_reducer_of_the_summaries():
    assert summarise_units(["MW", "MW", ""]) == "MW"
    assert summarise_units(["MW", "MWh/a", "MW"]) == "MW; MWh/a"
    assert summarise_units([]) == ""
    assert result_key_summary("flowGroup_gd_p")[0] == "MW; MWh/a"
    # A mixed table with unitless columns: the same text in the database / GUI / plot summary and on the
    # workbook's Units sheet (which adds only its "(by column)" note).
    assert result_key_summary("nodeGroup_gd_p")[0] == "MWh/a; per unit"
    for key, spec in OUTPUT_TRANSFORM.items():
        if not isinstance(spec, dict) or "*" in spec or len(OUTPUT_DIMS[key].columns) != 1:
            continue
        df = pd.DataFrame([[0.0] * len(spec)], columns=pd.Index(list(spec), name=OUTPUT_DIMS[key].columns[0]))
        row = units_sheet_row("s", key, df, _measures(key, df))
        summary = result_key_summary(key)
        expected = "per unit" if summary is None or summary[0] == "ratio" else summary[0]
        assert row[UNIT_LEVEL].removesuffix(" (by column)") == expected, key


def test_every_mixed_table_declares_its_label_level():
    for key, spec in OUTPUT_TRANSFORM.items():
        if isinstance(spec, dict):
            assert OUTPUT_DIMS[key].columns, key


def test_assistant_measures_agree_with_the_resolver(tmp_path: Path):
    """The assistant reads the units of a table from its own catalogue; both must give the same unit."""
    from flextool.assistant.results._catalog import measures
    from flextool.lean_parquet import lean_parquet_layout
    from tests.assistant.results_tables import write_result

    for key, spec in OUTPUT_TRANSFORM.items():
        if not isinstance(spec, dict):
            continue
        dims = OUTPUT_DIMS[key]
        labels = [label for label in spec if label != "*"]
        tuples = [tuple(f"{lvl}_x" for lvl in dims.columns[:-1]) + (label,) for label in labels]
        cols = (pd.MultiIndex.from_tuples(tuples, names=list(dims.columns)) if len(dims.columns) > 1
                else pd.Index(labels, name=dims.columns[0]))
        frame = pd.DataFrame([[1.0] * len(labels)], columns=cols,
                             index=pd.MultiIndex.from_tuples([tuple("r" for _ in dims.rows)], names=list(dims.rows))
                             if len(dims.rows) > 1 else pd.Index(["r"], name=dims.rows[0]))
        write_result(tmp_path, key, frame, "base")
        layout = lean_parquet_layout(tmp_path / f"{key}.parquet")
        assert [m.unit for m in measures(key, layout)] == column_units(key, cols), key


# ── Excel workbook ──────────────────────────────────────────────────────────

def _cost_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([("invest", "p2020"), ("invest", "p2025"), ("dispatch", "p2030")],
                                      names=["solve", "period"])
    return pd.DataFrame({"commodity_cost": [1.0, 2.0, 3.0], "co2_cost": [0.5, 0.5, 0.5]}, index=index)


def _flow_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([("p2020", "t01"), ("p2020", "t02")], names=["period", "time"])
    cols = pd.MultiIndex.from_tuples([("coal", "east"), ("wind", "east")], names=["unit", "node"])
    return pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=index, columns=cols)


def _flowgroup_frame() -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples([("g1", "p2020"), ("g1", "p2025")], names=["group", "period"])
    return pd.DataFrame({"cumulative_flow": [10.0, 20.0], "average_flow": [1.0, 2.0]},
                        index=index).rename_axis(columns="parameter")


def _side_block(ws, first_col: int) -> list[tuple]:
    rows = []
    for r in range(1, ws.max_row + 1):
        line = tuple(ws.cell(row=r, column=first_col + j).value for j in range(3))
        if any(v is not None for v in line):
            rows.append(line)
    return rows


def test_workbook_layout(tmp_path: Path):
    xlsx = tmp_path / "out.xlsx"
    write_excel_with_metadata(str(xlsx), [
        ("annualized_costs__d", "annualized_costs_d_p", _cost_frame()),
        ("CO2", "flowGroup_gd_p", _flowgroup_frame()),
        ("unit_outputNode__dt", "unit_outputNode_dt_ee", _flow_frame()),
    ])
    wb = openpyxl.load_workbook(xlsx)
    assert wb.sheetnames[:2] == ["Units", "_output_metadata"]

    # Units sheet: one row per data sheet, with a link to it.
    units = wb["Units"]
    assert [c.value for c in units[1]] == ["sheet", "table", UNIT_LEVEL, "semantics", "description"]
    body = [[c.value for c in row] for row in units.iter_rows(min_row=2)]
    assert [r[0] for r in body] == ["annualized_costs__d", "CO2", "unit_outputNode__dt"]
    assert body[0][2] == "M CUR/a"
    assert body[1][2] == "MW; MWh/a (by column)"
    assert body[2][2] == "MW"
    assert units.cell(row=3, column=1).hyperlink.location == "'CO2'!A1"

    # Single-header sheet: header unchanged at A1, a Table over exactly the data, repeated index values,
    # and a one-line unit block for a uniform table.
    ws = wb["annualized_costs__d"]
    assert [c.value for c in ws[1]][:4] == ["solve", "period", "commodity_cost", "co2_cost"]
    assert [ws.cell(row=r, column=1).value for r in (2, 3)] == ["invest", "invest"]
    assert not ws.merged_cells.ranges
    assert list(ws.tables) == ["tbl_annualized_costs__d"]
    assert ws.tables["tbl_annualized_costs__d"].ref == "A1:D4"
    assert _side_block(ws, 6) == [("column", UNIT_LEVEL, "semantics"),
                                  ("all value columns", "M CUR/a", "annualized")]
    # No header comments: they break a LibreOffice save; the Units sheet and the unit block carry the units.
    assert ws.cell(row=1, column=3).comment is None
    assert wb["Units"].freeze_panes == "A2" and wb["Units"].column_dimensions["A"].width > 20

    # A mixed table lists its units per column; the sheet name ``CO2`` gives a Table name that is not a cell.
    ws = wb["CO2"]
    assert list(ws.tables) == ["tbl_CO2"]
    assert _side_block(ws, 6)[1:] == [("cumulative_flow", "MWh/a", "annualized"),
                                      ("average_flow", "MW", "average")]

    # Wide sheet: the unit of measure is the last header row and the sheet reads back with one more header row.
    back = pd.read_excel(xlsx, sheet_name="unit_outputNode__dt", header=[0, 1, 2], index_col=[0, 1])
    assert list(back.columns.names) == ["unit", "node", UNIT_LEVEL]
    assert list(back.columns) == [("coal", "east", "MW"), ("wind", "east", "MW")]
    assert back.to_numpy().tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_comparison_workbook_adds_the_unit_level_below_the_scenario_level(tmp_path: Path):
    combined = pd.concat({"base": _flowgroup_frame(), "high": _flowgroup_frame()}, axis=1, names=["scenario"])
    xlsx = tmp_path / "compare.xlsx"
    write_excel_with_metadata(str(xlsx), [("flowGroup__gd", "flowGroup_gd_p", combined)])
    assert openpyxl.load_workbook(xlsx).sheetnames == ["Units", "_output_metadata", "flowGroup__gd"]
    back = pd.read_excel(xlsx, sheet_name="flowGroup__gd", header=[0, 1, 2], index_col=[0, 1])
    assert list(back.columns.names) == ["scenario", "parameter", UNIT_LEVEL]
    assert [c[2] for c in back.columns] == ["MWh/a", "MW", "MWh/a", "MW"]
    # _output_metadata gives the scenario of a column in its own cell.
    meta = pd.read_excel(xlsx, sheet_name="_output_metadata")
    assert list(meta.columns[:4]) == ["sheet", "output", "scenario", "column"]
    assert list(zip(meta["scenario"], meta["column"], meta["unit"])) == [
        ("base", "cumulative_flow", "MWh/a"), ("base", "average_flow", "MW"),
        ("high", "cumulative_flow", "MWh/a"), ("high", "average_flow", "MW")]


def test_reserved_and_invalid_names():
    assert excel_sheet_name("Units", reserved_sheet_names()) != "Units"
    used: set[str] = set()
    assert table_name("CO2", used) == "tbl_CO2"
    assert table_name("co2", used) == "tbl_co2_2"
    assert table_name("a b-(c)", used) == "tbl_a_b__c_"
    assert table_headers(["period", "Period", None, ""]) == ["period", "Period_2", "column_3", "column_4"]


def _synthetic(key: str) -> pd.DataFrame:
    """A two-row frame with the declared row and column levels of *key* (mixed tables use their map labels)."""
    dims = OUTPUT_DIMS[key]
    spec = OUTPUT_TRANSFORM.get(key)
    rows = [tuple(f"{lvl}{i}" for lvl in dims.rows) for i in (1, 2)]
    if spec is DIMENSION_TABLE or not dims.columns:
        return pd.MultiIndex.from_tuples(rows, names=list(dims.rows)).to_frame(index=False)
    index = (pd.MultiIndex.from_tuples(rows, names=list(dims.rows)) if len(dims.rows) > 1
             else pd.Index([r[0] for r in rows], name=dims.rows[0]))
    labels = [k for k in spec if k != "*"] if isinstance(spec, dict) else ["a", "b"]
    tuples = [tuple(f"{lvl}_x" for lvl in dims.columns[:-1]) + (label,) for label in labels]
    cols = (pd.MultiIndex.from_tuples(tuples, names=list(dims.columns)) if len(dims.columns) > 1
            else pd.Index(labels, name=dims.columns[0]))
    return pd.DataFrame(1.0, index=index, columns=cols)


def test_every_declared_table_shape_writes_a_valid_workbook(tmp_path: Path):
    used = reserved_sheet_names()
    sheets = [(excel_sheet_name(key, used), key, _synthetic(key)) for key in OUTPUT_DIMS]
    xlsx = tmp_path / "all.xlsx"
    write_excel_with_metadata(str(xlsx), sheets)
    wb = openpyxl.load_workbook(xlsx)
    units = wb["Units"]
    listed = [units.cell(row=r, column=1).value for r in range(2, units.max_row + 1)]
    assert listed == [s for s, _, _ in sheets]
    for sheet, key, df in sheets:
        ws = wb[sheet]
        if df.columns.nlevels == 1:
            assert len(ws.tables) == 1, sheet
        else:
            # The last header row names the unit level (in the column left of the data).
            assert ws.cell(row=df.columns.nlevels + 1, column=df.index.nlevels).value == UNIT_LEVEL, sheet
