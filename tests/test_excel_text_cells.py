"""Names from the input data are written to Excel workbooks as text, never as formulas or links.

A name such as ``=1+1`` stays the string ``=1+1``: in the result workbooks (xlsxwriter) and in the input export
(openpyxl), and it comes back unchanged when the exported workbook is read in again.
"""
from __future__ import annotations

from pathlib import Path

import openpyxl
import pandas as pd
from spinedb_api import DatabaseMapping

from flextool._resources import package_data_path
from flextool.export_to_tabular.export_to_excel import export_to_excel
from flextool.process_outputs._excel_writer import write_excel_with_metadata
from flextool.process_inputs.read_self_describing_excel import read_self_describing_excel
from flextool.process_inputs.write_self_describing_to_db import write_sheet_data_to_db
from flextool.update_flextool.initialize_database import initialize_database

NAMES = ["=1+1", "+A1", "@SUM(A1)", '=HYPERLINK("http://example.invalid","x")', "http://example.invalid"]


def _cells(path: Path) -> list:
    wb = openpyxl.load_workbook(path)
    return [cell for ws in wb.worksheets for row in ws.iter_rows() for cell in row if cell.value is not None]


def test_result_workbook_writes_names_as_text(tmp_path: Path) -> None:
    df = pd.DataFrame({"value": [1.0] * len(NAMES)}, index=pd.Index(NAMES, name="unit"))
    path = tmp_path / "out.xlsx"
    write_excel_with_metadata(path, [("unit_flow", "unit_flow_d", df)])
    cells = _cells(path)
    assert not [c.coordinate for c in cells if c.data_type == "f"]
    values = {c.value for c in cells}
    assert set(NAMES) <= values
    wb = openpyxl.load_workbook(path)
    assert not any(c.hyperlink for c in wb["unit_flow"]._cells.values())


def test_input_export_keeps_names_starting_with_equals(tmp_path: Path) -> None:
    template = package_data_path("schemas/spinedb_schema.json")
    src = tmp_path / "src.sqlite"
    initialize_database(str(template), str(src))
    with DatabaseMapping(f"sqlite:///{src}") as db:
        for name in NAMES[:3]:
            db.add_entity(entity_class_name="node", name=name, description=f"{name} description")
        db.commit_session("names")

    xlsx = tmp_path / "rt.xlsx"
    export_to_excel(f"sqlite:///{src}", str(xlsx), include_advanced=True)
    assert not [(c.parent.title, c.coordinate) for c in _cells(xlsx) if c.data_type == "f"]

    rt = tmp_path / "rt.sqlite"
    initialize_database(str(template), str(rt))
    sheets = read_self_describing_excel(str(xlsx), skip_sheets={"navigate", "version"})
    write_sheet_data_to_db(sheets, f"sqlite:///{rt}", purge_first=True, keep_entities=True)
    with DatabaseMapping(f"sqlite:///{rt}") as db:
        nodes = {e["name"] for e in db.get_items("entity", entity_class_name="node")}
    assert set(NAMES[:3]) <= nodes
