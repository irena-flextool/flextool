"""Tests for the FlexTool Spine DB to Excel exporter.

Verifies database reading, sheet specification building, and Excel output
against the real example_input_template.sqlite database.
"""

from __future__ import annotations

from pathlib import Path

import openpyxl
import pytest

from flextool.export_to_tabular.db_reader import DatabaseContents, read_database
from flextool.export_to_tabular.sheet_config import build_sheet_specs, SheetSpec
from flextool.export_to_tabular.export_to_excel import export_to_excel

FLEXTOOL_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DB = FLEXTOOL_ROOT / "projects" / "africa2" / "input_sources" / "example_input_template.sqlite"
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
    export_to_excel(EXAMPLE_DB_URL, str(out_path))
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
        # node class should have parameter definitions
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

    def test_parameter_groups_loaded(self, db_contents: DatabaseContents) -> None:
        assert len(db_contents.parameter_groups) > 0, "No parameter groups loaded"
        group_names = set(db_contents.parameter_groups.keys())
        assert "inflow" in group_names, f"Missing 'inflow' group. Groups: {group_names}"
        assert "methods" in group_names, f"Missing 'methods' group. Groups: {group_names}"


# ---------------------------------------------------------------------------
# Test 2: test_build_sheet_specs
# ---------------------------------------------------------------------------

class TestBuildSheetSpecs:
    """Verify that build_sheet_specs produces correct sheet specifications."""

    def test_total_sheet_count(self, sheet_specs: list[SheetSpec]) -> None:
        count = len(sheet_specs)
        assert 40 <= count <= 55, f"Expected 40-55 sheets, got {count}"

    def test_specific_sheets_exist(self, sheet_specs: list[SheetSpec]) -> None:
        names = {s.sheet_name for s in sheet_specs}
        for expected in ("node_c", "node_p", "node_t", "unit_c", "scenario", "navigate", "version"):
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

        # Classify each param
        methods_indices: list[int] = []
        inflow_indices: list[int] = []
        ungrouped_indices: list[int] = []

        for i, pname in enumerate(params):
            group = db_contents.param_to_group.get(("node", pname))
            if group == "methods":
                methods_indices.append(i)
            elif group == "inflow":
                inflow_indices.append(i)
            else:
                ungrouped_indices.append(i)

        # methods (priority 1) should come before inflow (priority 2)
        if methods_indices and inflow_indices:
            assert max(methods_indices) < min(inflow_indices), (
                f"methods params should come before inflow params. "
                f"methods at {methods_indices}, inflow at {inflow_indices}"
            )

        # inflow should come before ungrouped
        if inflow_indices and ungrouped_indices:
            assert max(inflow_indices) < min(ungrouped_indices), (
                f"inflow params should come before ungrouped params. "
                f"inflow at {inflow_indices}, ungrouped at {ungrouped_indices}"
            )


# ---------------------------------------------------------------------------
# Test 3: test_export_produces_valid_excel
# ---------------------------------------------------------------------------

class TestExportProducesValidExcel:
    """Verify that export_to_excel produces a valid Excel file with all sheets."""

    def test_all_expected_sheets_exist(self, exported_workbook: openpyxl.Workbook, sheet_specs: list[SheetSpec]) -> None:
        wb_sheets = set(exported_workbook.sheetnames)
        spec_sheets = {s.sheet_name for s in sheet_specs}
        missing = spec_sheets - wb_sheets
        assert not missing, f"Missing sheets in exported workbook: {missing}"

    def test_each_sheet_has_header_row(self, exported_workbook: openpyxl.Workbook) -> None:
        for name in exported_workbook.sheetnames:
            ws = exported_workbook[name]
            # Every sheet should have at least row 1 with some content
            row1_values = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
            has_content = any(v is not None for v in row1_values)
            assert has_content, f"Sheet '{name}' has no content in row 1"


# ---------------------------------------------------------------------------
# Test 4: test_node_c_content
# ---------------------------------------------------------------------------

class TestNodeCContent:
    """Verify detailed content of the node_c sheet."""

    def test_header_row_structure(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        assert ws.cell(row=2, column=1).value == "alternative", (
            f"Row 2, Col 1 should be 'alternative', got '{ws.cell(row=2, column=1).value}'"
        )
        assert ws.cell(row=2, column=2).value == "node", (
            f"Row 2, Col 2 should be 'node', got '{ws.cell(row=2, column=2).value}'"
        )

    def test_expected_node_entities(self, exported_workbook: openpyxl.Workbook, db_contents: DatabaseContents) -> None:
        ws = exported_workbook["node_c"]
        # Get all node names from column 2 (rows 3+)
        node_col = 2
        node_names_in_sheet: set[str] = set()
        for row in range(3, ws.max_row + 1):
            val = ws.cell(row=row, column=node_col).value
            if val is not None:
                node_names_in_sheet.add(str(val))

        # All node entities from the DB should appear
        db_node_entities = db_contents.entities.get("node", [])
        db_node_names = {e["entity_byname"][0] for e in db_node_entities}

        # At least some DB nodes should be present in the sheet
        present = db_node_names & node_names_in_sheet
        assert len(present) > 0, (
            f"No DB node entities found in node_c sheet. "
            f"DB nodes: {db_node_names}, sheet nodes: {node_names_in_sheet}"
        )

    def test_entity_alternative_column_exists(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        headers = [ws.cell(row=2, column=c).value for c in range(1, ws.max_column + 1)]
        assert "Entity Alternative" in headers, (
            f"'Entity Alternative' column not found in node_c headers: {headers}"
        )

    def test_description_row_has_content(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_c"]
        # Row 1 should have descriptions for some parameter columns
        desc_values = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        # Filter out None and 'navigate'
        descriptions = [v for v in desc_values if v is not None and v != "navigate"]
        assert len(descriptions) > 0, "Row 1 (description row) has no parameter descriptions"

    def test_parameter_values_match_db(self, exported_workbook: openpyxl.Workbook, db_contents: DatabaseContents) -> None:
        ws = exported_workbook["node_c"]
        # Build column index map from row 2 headers
        headers: dict[str, int] = {}
        for c in range(1, ws.max_column + 1):
            val = ws.cell(row=2, column=c).value
            if val is not None:
                headers[val] = c

        # Spot-check: find a scalar parameter value in the DB and verify it in the sheet
        checked = 0
        for (cls, byname, pname, alt), value in db_contents.parameter_values.items():
            if cls != "node":
                continue
            if not isinstance(value, (int, float, str)):
                continue
            if pname not in headers:
                continue

            node_name = byname[0]
            param_col = headers[pname]
            alt_col = headers.get("alternative", 1)
            node_col = headers.get("node", 2)

            # Search for matching row
            for row in range(3, ws.max_row + 1):
                row_alt = ws.cell(row=row, column=alt_col).value
                row_node = ws.cell(row=row, column=node_col).value
                if str(row_alt) == str(alt) and str(row_node) == str(node_name):
                    cell_val = ws.cell(row=row, column=param_col).value
                    if cell_val is not None:
                        if isinstance(value, float):
                            assert abs(float(cell_val) - value) < 1e-9, (
                                f"node_c: {node_name}/{alt}/{pname} = {cell_val}, expected {value}"
                            )
                        else:
                            assert str(cell_val) == str(value), (
                                f"node_c: {node_name}/{alt}/{pname} = {cell_val}, expected {value}"
                            )
                        checked += 1
                    break

            if checked >= 5:
                break

        assert checked > 0, "Could not spot-check any parameter values in node_c"


# ---------------------------------------------------------------------------
# Test 5: test_node_t_content
# ---------------------------------------------------------------------------

class TestNodeTContent:
    """Verify that node_t has the correct transposed timeseries layout."""

    def test_transposed_layout_structure(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]

        # Row 1 col 2 should be 'alternative'
        assert ws.cell(row=1, column=2).value == "alternative", (
            f"node_t: row 1, col 2 should be 'alternative', got '{ws.cell(row=1, column=2).value}'"
        )

        # Row 2 col 2 should be 'parameter'
        assert ws.cell(row=2, column=2).value == "parameter", (
            f"node_t: row 2, col 2 should be 'parameter', got '{ws.cell(row=2, column=2).value}'"
        )

    def test_time_indices_in_column_a(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]
        # Find the 'time' label row
        time_row = None
        for row in range(1, ws.max_row + 1):
            if ws.cell(row=row, column=1).value == "time":
                time_row = row
                break

        assert time_row is not None, "node_t: 'time' label not found in column A"

        # Rows below 'time' should have time index values
        time_values: list[str] = []
        for row in range(time_row + 1, min(time_row + 10, ws.max_row + 1)):
            val = ws.cell(row=row, column=1).value
            if val is not None:
                time_values.append(str(val))

        assert len(time_values) > 0, "node_t: no time index values found below 'time' label"
        # Check for t0001-style or similar time indexes
        has_time_like = any("t" in v.lower() or v.startswith("2") for v in time_values)
        assert has_time_like, f"node_t: time values don't look like time indexes: {time_values}"

    def test_entity_names_in_headers(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["node_t"]
        # Entity names should appear in header rows (row 3+) in columns C+
        # Row 3 should have 'node' label in col B
        node_row_label = ws.cell(row=3, column=2).value
        assert node_row_label == "node", (
            f"node_t: row 3, col 2 should be 'node', got '{node_row_label}'"
        )

        # Entity names in columns 3+ of that row
        entity_names: list[str] = []
        for col in range(3, ws.max_column + 1):
            val = ws.cell(row=3, column=col).value
            if val is not None:
                entity_names.append(str(val))

        assert len(entity_names) > 0, "node_t: no entity names found in header rows"


# ---------------------------------------------------------------------------
# Test 6: test_scenario_sheet
# ---------------------------------------------------------------------------

class TestScenarioSheet:
    """Verify the scenario sheet structure."""

    def test_scenario_names_in_row_2(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["scenario"]
        # Row 2 should have scenario names starting from column 2
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

        # At least one alternative name in row 3
        alt_values: list[str] = []
        for col in range(2, ws.max_column + 1):
            val = ws.cell(row=3, column=col).value
            if val is not None:
                alt_values.append(str(val))

        assert len(alt_values) > 0, "scenario sheet: no base alternatives in row 3"

    def test_alternatives_in_subsequent_rows(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["scenario"]
        # Rows 4+ should have alternative_N labels
        if ws.max_row >= 4:
            label = ws.cell(row=4, column=1).value
            assert label is not None and "alternative" in str(label).lower(), (
                f"scenario sheet: row 4, col 1 should be an alternative label, got '{label}'"
            )


# ---------------------------------------------------------------------------
# Test 7: test_link_sheet
# ---------------------------------------------------------------------------

class TestLinkSheet:
    """Verify the commodity_node link sheet."""

    def test_commodity_node_columns(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["commodity_node"]
        # Row 1 should have 'commodity' and 'node' headers
        col1 = ws.cell(row=1, column=1).value
        col2 = ws.cell(row=1, column=2).value
        assert col1 == "commodity", f"commodity_node col 1 header should be 'commodity', got '{col1}'"
        assert col2 == "node", f"commodity_node col 2 header should be 'node', got '{col2}'"

    def test_expected_link_entries(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["commodity_node"]
        # Collect all (commodity, node) pairs
        pairs: set[tuple[str, str]] = set()
        for row in range(2, ws.max_row + 1):
            c = ws.cell(row=row, column=1).value
            n = ws.cell(row=row, column=2).value
            if c is not None and n is not None:
                pairs.add((str(c), str(n)))

        # Verify specific entries
        assert ("Coal", "Coal_node") in pairs, (
            f"Expected ('Coal', 'Coal_node') in commodity_node. Got: {pairs}"
        )
        assert ("Gas", "Gas_node") in pairs, (
            f"Expected ('Gas', 'Gas_node') in commodity_node. Got: {pairs}"
        )


# ---------------------------------------------------------------------------
# Test 8: test_parameter_group_ordering
# ---------------------------------------------------------------------------

class TestParameterGroupOrdering:
    """Verify parameter ordering by group priority on node_c."""

    def test_group_ordering_in_header(self, exported_workbook: openpyxl.Workbook, db_contents: DatabaseContents) -> None:
        ws = exported_workbook["node_c"]

        # Get parameter names from row 2 (skip entity/direction columns)
        all_headers: list[str] = []
        for c in range(1, ws.max_column + 1):
            val = ws.cell(row=2, column=c).value
            if val is not None:
                all_headers.append(str(val))

        # Identify parameter headers (those that are not structural)
        structural = {"alternative", "node", "Entity Alternative", "input_output",
                      "left_node", "right_node", "constraint"}
        param_headers = [h for h in all_headers if h not in structural]

        # Classify parameters
        methods_params: list[str] = []
        inflow_params: list[str] = []
        ungrouped_params: list[str] = []

        for pname in param_headers:
            group = db_contents.param_to_group.get(("node", pname))
            if group == "methods":
                methods_params.append(pname)
            elif group == "inflow":
                inflow_params.append(pname)
            else:
                ungrouped_params.append(pname)

        # Find first and last positions of each group
        if methods_params and inflow_params:
            last_methods_pos = max(param_headers.index(p) for p in methods_params)
            first_inflow_pos = min(param_headers.index(p) for p in inflow_params)
            assert last_methods_pos < first_inflow_pos, (
                f"methods params should come before inflow params. "
                f"Last methods at {last_methods_pos}, first inflow at {first_inflow_pos}"
            )

        if inflow_params and ungrouped_params:
            last_inflow_pos = max(param_headers.index(p) for p in inflow_params)
            first_ungrouped_pos = min(param_headers.index(p) for p in ungrouped_params)
            assert last_inflow_pos < first_ungrouped_pos, (
                f"inflow params should come before ungrouped params. "
                f"Last inflow at {last_inflow_pos}, first ungrouped at {first_ungrouped_pos}"
            )


# ---------------------------------------------------------------------------
# Test 9: test_constraint_sheet
# ---------------------------------------------------------------------------

class TestConstraintSheet:
    """Verify the unit_node_constraint_c sheet."""

    def test_constraint_column_exists(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["unit_node_constraint_c"]
        headers = [ws.cell(row=2, column=c).value for c in range(1, ws.max_column + 1)]
        assert "constraint" in headers, (
            f"unit_node_constraint_c: 'constraint' column not found. Headers: {headers}"
        )

    def test_constraint_flow_coefficient_values(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["unit_node_constraint_c"]
        headers = {ws.cell(row=2, column=c).value: c for c in range(1, ws.max_column + 1)}
        coeff_col = headers.get("constraint_flow_coefficient")
        assert coeff_col is not None, (
            "unit_node_constraint_c: 'constraint_flow_coefficient' column not found"
        )

        # Check that there are actual coefficient values
        values: list[float] = []
        for row in range(3, ws.max_row + 1):
            val = ws.cell(row=row, column=coeff_col).value
            if val is not None:
                values.append(float(val))

        assert len(values) > 0, "unit_node_constraint_c: no constraint_flow_coefficient values found"

    def test_specific_constraint_entries(self, exported_workbook: openpyxl.Workbook) -> None:
        ws = exported_workbook["unit_node_constraint_c"]
        headers = {ws.cell(row=2, column=c).value: c for c in range(1, ws.max_column + 1)}
        constraint_col = headers.get("constraint")
        assert constraint_col is not None, "No 'constraint' column"

        # Collect all constraint names
        constraint_names: set[str] = set()
        for row in range(3, ws.max_row + 1):
            val = ws.cell(row=row, column=constraint_col).value
            if val is not None:
                constraint_names.add(str(val))

        assert len(constraint_names) > 0, (
            "unit_node_constraint_c: no constraint entries found"
        )
