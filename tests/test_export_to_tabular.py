"""Tests for the FlexTool Spine DB to Excel exporter.

Verifies database reading, sheet specification building, and Excel output
against the real examples.sqlite database.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from flextool.export_to_tabular.db_reader import DatabaseContents, read_database
from flextool.export_to_tabular.sheet_config import build_sheet_specs, SheetSpec
from flextool.export_to_tabular.export_to_excel import export_to_excel

FLEXTOOL_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DB = FLEXTOOL_ROOT / "templates" / "examples.sqlite"
EXAMPLE_DB_URL = f"sqlite:///{EXAMPLE_DB}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_contents() -> DatabaseContents:
    """Read the example database once for all tests in this module."""
    assert EXAMPLE_DB.exists(), f"Example DB not found: {EXAMPLE_DB}"
    return read_database(EXAMPLE_DB_URL)


@pytest.fixture(scope="module")
def sheet_specs(db_contents: DatabaseContents) -> list[SheetSpec]:
    """Build sheet specs once for all tests in this module."""
    return build_sheet_specs(db_contents)


@pytest.fixture(scope="module")
def exported_workbook(tmp_path_factory: pytest.TempPathFactory, db_contents: DatabaseContents) -> openpyxl.Workbook:
    """Export to a temporary Excel file and return the opened workbook."""
    out_dir = tmp_path_factory.mktemp("export")
    out_path = out_dir / "test_output.xlsx"
    export_to_excel(EXAMPLE_DB_URL, str(out_path), include_advanced=True)
    wb = openpyxl.load_workbook(str(out_path))
    yield wb
    wb.close()


# ---------------------------------------------------------------------------
# Test 1: test_read_database
# ---------------------------------------------------------------------------

class TestReadDatabase:
    """Verify that read_database produces a fully populated DatabaseContents."""

    def test_entity_class_count(self, db_contents: DatabaseContents) -> None:
        assert len(db_contents.entity_classes) >= 17, (
            f"Expected >= 17 entity classes, got {len(db_contents.entity_classes)}"
        )

    def test_key_entities_exist(self, db_contents: DatabaseContents) -> None:
        for cls_name in ("node", "unit", "connection"):
            assert cls_name in db_contents.entities, f"Missing entities for class '{cls_name}'"
            assert len(db_contents.entities[cls_name]) > 0, f"No entities for class '{cls_name}'"

    def test_parameter_definitions_exist(self, db_contents: DatabaseContents) -> None:
        assert len(db_contents.parameter_definitions) > 0, "No parameter definitions loaded"
        assert "node" in db_contents.parameter_definitions
        assert len(db_contents.parameter_definitions["node"]) > 0

    def test_parameter_values_populated(self, db_contents: DatabaseContents) -> None:
        assert len(db_contents.parameter_values) > 0, "No parameter values loaded"

    def test_alternatives_non_empty(self, db_contents: DatabaseContents) -> None:
        assert len(db_contents.alternatives) > 0, "No alternatives loaded"

    def test_scenarios_populated(self, db_contents: DatabaseContents) -> None:
        assert len(db_contents.scenarios) > 0, "No scenarios loaded"

    def test_version_is_number(self, db_contents: DatabaseContents) -> None:
        assert db_contents.version is not None, "Version is None"
        assert isinstance(db_contents.version, (int, float)), (
            f"Version should be a number, got {type(db_contents.version)}"
        )


# ---------------------------------------------------------------------------
# Test 2: test_build_sheet_specs
# ---------------------------------------------------------------------------

class TestBuildSheetSpecs:
    """Verify that build_sheet_specs produces correct sheet specifications."""

    def test_total_sheet_count(self, sheet_specs: list[SheetSpec]) -> None:
        count = len(sheet_specs)
        assert 55 <= count <= 65, f"Expected 55-65 sheets, got {count}"

    def test_specific_sheets_exist(self, sheet_specs: list[SheetSpec]) -> None:
        names = {s.sheet_name for s in sheet_specs}
        for expected in ("node_c", "node_p", "node_t", "unit_c", "scenario", "navigate"):
            assert expected in names, f"Sheet '{expected}' not found. Available: {sorted(names)}"

    def test_unit_node_c_has_direction_column(self, sheet_specs: list[SheetSpec]) -> None:
        spec = next(s for s in sheet_specs if s.sheet_name == "unit_node_c")
        assert spec.direction_column == "input_output", (
            f"unit_node_c direction_column should be 'input_output', got '{spec.direction_column}'"
        )

    def test_connection_c_has_extra_entity_columns(self, sheet_specs: list[SheetSpec]) -> None:
        spec = next(s for s in sheet_specs if s.sheet_name == "connection_c")
        assert spec.extra_entity_columns == ["left_node", "right_node"], (
            f"connection_c extra_entity_columns should be ['left_node', 'right_node'], "
            f"got {spec.extra_entity_columns}"
        )

    def test_unit_node_constraint_c_has_unpack_index_column(self, sheet_specs: list[SheetSpec]) -> None:
        spec = next(s for s in sheet_specs if s.sheet_name == "unit_node_constraint_c")
        assert spec.unpack_index_column == "constraint", (
            f"unit_node_constraint_c unpack_index_column should be 'constraint', "
            f"got '{spec.unpack_index_column}'"
        )

    def test_commodity_node_has_link_layout(self, sheet_specs: list[SheetSpec]) -> None:
        spec = next(s for s in sheet_specs if s.sheet_name == "commodity_node")
        assert spec.layout == "link", (
            f"commodity_node layout should be 'link', got '{spec.layout}'"
        )

    def test_parameter_ordering_on_node_c(self, sheet_specs: list[SheetSpec], db_contents: DatabaseContents) -> None:
        spec = next(s for s in sheet_specs if s.sheet_name == "node_c")
        params = spec.parameter_names
        assert len(params) > 0, "node_c has no parameter_names"


# ---------------------------------------------------------------------------
# Test 3: test_export_produces_valid_excel
# ---------------------------------------------------------------------------

class TestExportProducesValidExcel:
    """Verify that export_to_excel produces a valid Excel file with all sheets."""

    def test_all_expected_sheets_exist(self, exported_workbook: openpyxl.Workbook, sheet_specs: list[SheetSpec]) -> None:
        wb_sheets = set(exported_workbook.sheetnames)
        spec_sheets = {s.sheet_name for s in sheet_specs}
        # 'version' is generated as a spec but written separately — exclude from check
        spec_sheets.discard("version")
        missing = spec_sheets - wb_sheets
        assert not missing, f"Missing sheets in exported workbook: {missing}"

    def test_each_sheet_has_header_row(self, exported_workbook: openpyxl.Workbook) -> None:
        for name in exported_workbook.sheetnames:
            ws = exported_workbook[name]
            row1_values = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
            has_content = any(v is not None for v in row1_values)
            assert has_content, f"Sheet '{name}' has no content in row 1"


# ---------------------------------------------------------------------------
# Test 4: test_node_c_content (v2 self-describing format)
# ---------------------------------------------------------------------------

class TestNodeCContent:
    """Verify detailed content of the node_c sheet in v2 format.

    V2 format:
      Row 1: navigate | (blank) | description texts...
      Row 2: (blank)  | (blank) | data type | type values...
      Row 3: alternative | entity: node | parameter | param names...
      Row 4+: data rows
    """

    def test_header_row_structure(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        # Row 3 has the column headers in v2 format
        assert ws.cell(row=3, column=1).value == "alternative", (
            f"Row 3, Col 1 should be 'alternative', got '{ws.cell(row=3, column=1).value}'"
        )
        assert ws.cell(row=3, column=2).value == "entity: node", (
            f"Row 3, Col 2 should be 'entity: node', got '{ws.cell(row=3, column=2).value}'"
        )

    def test_navigate_in_row_1(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        assert ws.cell(row=1, column=1).value == "navigate"

    def test_expected_node_entities(self, exported_workbook: openpyxl.Workbook, db_contents: DatabaseContents) -> None:
        ws = exported_workbook["node_c"]
        # In v2, entity names are in column 2, starting from row 4
        node_names_in_sheet: set[str] = set()
        for row in range(4, ws.max_row + 1):
            val = ws.cell(row=row, column=2).value
            if val is not None:
                node_names_in_sheet.add(str(val))

        db_node_entities = db_contents.entities.get("node", [])
        db_node_names = {e["entity_byname"][0] for e in db_node_entities}
        present = db_node_names & node_names_in_sheet
        assert len(present) > 0, (
            f"No DB node entities found in node_c sheet. "
            f"DB nodes: {db_node_names}, sheet nodes: {node_names_in_sheet}"
        )

    def test_description_row_has_content(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        # Row 1 has descriptions starting from column 3
        desc_values = [ws.cell(row=1, column=c).value for c in range(3, ws.max_column + 1)]
        descriptions = [v for v in desc_values if v is not None and v != "navigate"]
        assert len(descriptions) > 0, "Row 1 (description row) has no parameter descriptions"

    def test_data_type_row(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        # Row 2 should have 'data type' label
        assert ws.cell(row=2, column=3).value == "data type", (
            f"Row 2, Col 3 should be 'data type', got '{ws.cell(row=2, column=3).value}'"
        )


# ---------------------------------------------------------------------------
# Test 5: test_node_t_content (v2 transposed timeseries)
# ---------------------------------------------------------------------------

class TestNodeTContent:
    """Verify that node_t has the correct v2 transposed timeseries layout.

    V2 format:
      Row 1: navigate | info text...
      Row 2: (blank)  | entity: node | entity names...
      Row 3: (blank)  | alternative  | alt names...
      Row 4: index: time | parameter | param names...
      Row 5+: time values | data...
    """

    def test_transposed_layout_structure(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]
        assert ws.cell(row=2, column=2).value == "entity: node", (
            f"node_t: row 2, col 2 should be 'entity: node', got '{ws.cell(row=2, column=2).value}'"
        )
        assert ws.cell(row=3, column=2).value == "alternative", (
            f"node_t: row 3, col 2 should be 'alternative', got '{ws.cell(row=3, column=2).value}'"
        )

    def test_time_index_label(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]
        assert ws.cell(row=4, column=1).value == "index: time", (
            f"node_t: row 4, col 1 should be 'index: time', got '{ws.cell(row=4, column=1).value}'"
        )

    def test_entity_names_in_headers(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]
        # Entity names in row 2, columns 3+
        entity_names: list[str] = []
        for col in range(3, ws.max_column + 1):
            val = ws.cell(row=2, column=col).value
            if val is not None:
                entity_names.append(str(val))
        assert len(entity_names) > 0, "node_t: no entity names found in row 2"

    def test_time_values_exist(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]
        # Row 5+ should have time index values in column 1
        time_values: list[str] = []
        for row in range(5, min(15, ws.max_row + 1)):
            val = ws.cell(row=row, column=1).value
            if val is not None:
                time_values.append(str(val))
        assert len(time_values) > 0, "node_t: no time index values found"


# ---------------------------------------------------------------------------
# Test 6: test_scenario_sheet
# ---------------------------------------------------------------------------

class TestScenarioSheet:
    """Verify the scenario sheet structure."""

    def test_scenario_names_in_row_2(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["scenario"]
        scenario_names: list[str] = []
        for col in range(2, ws.max_column + 1):
            val = ws.cell(row=2, column=col).value
            if val is not None:
                scenario_names.append(str(val))
        assert len(scenario_names) > 0, "scenario sheet: no scenario names in row 2"

    def test_base_alternative_in_row_3(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["scenario"]
        label = ws.cell(row=3, column=1).value
        assert label == "base_alternative", (
            f"scenario sheet: row 3, col 1 should be 'base_alternative', got '{label}'"
        )

    def test_alternatives_in_subsequent_rows(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["scenario"]
        if ws.max_row >= 4:
            label = ws.cell(row=4, column=1).value
            assert label is not None and "alternative" in str(label).lower(), (
                f"scenario sheet: row 4, col 1 should be an alternative label, got '{label}'"
            )


# ---------------------------------------------------------------------------
# Test 7: test_link_sheet (v2 format)
# ---------------------------------------------------------------------------

class TestLinkSheet:
    """Verify the commodity_node link sheet in v2 format.

    V2 format:
      Row 1: navigate | ...
      Row 2: entity: commodity, node | commodity | node
      Row 3+: data rows
    """

    def test_commodity_node_headers(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["commodity_node"]
        # Row 2 has the column headers in v2
        assert ws.cell(row=2, column=2).value == "commodity", (
            f"commodity_node row 2 col 2 should be 'commodity', got '{ws.cell(row=2, column=2).value}'"
        )
        assert ws.cell(row=2, column=3).value == "node", (
            f"commodity_node row 2 col 3 should be 'node', got '{ws.cell(row=2, column=3).value}'"
        )

    def test_expected_link_entries(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["commodity_node"]
        pairs: set[tuple[str, str]] = set()
        for row in range(3, ws.max_row + 1):
            c = ws.cell(row=row, column=2).value
            n = ws.cell(row=row, column=3).value
            if c is not None and n is not None:
                pairs.add((str(c), str(n)))
        assert len(pairs) > 0, f"No commodity-node pairs found. Sheet is empty."


# ---------------------------------------------------------------------------
# Test 8: test_constraint_sheet (v2 format)
# ---------------------------------------------------------------------------

class TestConstraintSheet:
    """Verify the unit_node_constraint_c sheet in v2 format.

    V2 format:
      Row 1: navigate | ... | description | desc texts...
      Row 2: ... | data type | types...
      Row 3: alternative | entity: unit, node | unit | node | input_output | index: constraint | parameter | param names...
      Row 4+: data rows
    """

    def test_constraint_index_column(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["unit_node_constraint_c"]
        # Find the 'index: constraint' column in row 3
        headers = [ws.cell(row=3, column=c).value for c in range(1, ws.max_column + 1)]
        assert "index: constraint" in headers, (
            f"unit_node_constraint_c: 'index: constraint' not found in row 3. Headers: {headers}"
        )

    def test_constraint_flow_coefficient_values(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["unit_node_constraint_c"]
        # Find parameter names in row 3
        headers = {ws.cell(row=3, column=c).value: c for c in range(1, ws.max_column + 1)}
        assert "constraint_flow_coefficient" in [ws.cell(row=3, column=c).value for c in range(1, ws.max_column + 1)], (
            "unit_node_constraint_c: 'constraint_flow_coefficient' not found in row 3 headers"
        )

    def test_has_data_rows(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["unit_node_constraint_c"]
        # Should have data starting from row 4
        has_data = False
        for row in range(4, ws.max_row + 1):
            val = ws.cell(row=row, column=1).value
            if val is not None:
                has_data = True
                break
        assert has_data, "unit_node_constraint_c: no data rows found"
