"""Tests for the self-describing Excel reader using the user's example sheets."""

import pytest
import openpyxl

from flextool.process_inputs.read_self_describing_excel import (
    find_crossing_point,
    parse_sheet_metadata,
    parse_link_sheet_metadata,
    parse_transposed_sheet_metadata,
    detect_and_parse_sheet,
    extract_sheet_data,
    _parse_entity_def,
    _parse_filter_def,
    ENTITY_EXISTENCE,
)

EXAMPLE_FILE = "projects/africa2/converted/example_input_template.xlsx"


@pytest.fixture(scope="module")
def workbook():
    wb = openpyxl.load_workbook(EXAMPLE_FILE, data_only=True)
    yield wb
    wb.close()


# ---------------------------------------------------------------------------
# Unit tests for parsing helpers
# ---------------------------------------------------------------------------


class TestParseEntityDef:
    def test_single_dim(self):
        result = _parse_entity_def("entity: node")
        assert len(result) == 1
        assert result[0].class_name == "node"
        assert result[0].dimensions == ["node"]

    def test_multi_dim(self):
        result = _parse_entity_def("entity: commodity, node")
        assert len(result) == 1
        assert result[0].class_name == "commodity__node"
        assert result[0].dimensions == ["commodity", "node"]

    def test_multi_class(self):
        text = "entity: (unit__inputNode: (unit, node), unit__outputNode: (unit, node))"
        result = _parse_entity_def(text)
        assert len(result) == 2
        assert result[0].class_name == "unit__inputNode"
        assert result[0].dimensions == ["unit", "node"]
        assert result[1].class_name == "unit__outputNode"
        assert result[1].dimensions == ["unit", "node"]


class TestParseFilterDef:
    def test_basic(self):
        text = "filter: {unit__inputNode: ^input$, unit__outputNode: ^output$}"
        result = _parse_filter_def(text)
        assert result["unit__inputNode"] == "^input$"
        assert result["unit__outputNode"] == "^output$"


# ---------------------------------------------------------------------------
# Crossing point detection
# ---------------------------------------------------------------------------


class TestCrossingPoint:
    def test_node_c(self, workbook):
        ws = workbook["node_c"]
        def_row, def_col = find_crossing_point(ws)
        # node_c: row 1 has "description" at col C (index 2)
        # Scanning col C down: C1=description, C2=data type, C3=parameter, C4=empty
        # → def_row = 2 (0-based, which is Excel row 3)
        assert def_col == 2, f"Expected def_col=2, got {def_col}"
        assert def_row == 2, f"Expected def_row=2, got {def_row}"

    def test_unit_node_constraint_c(self, workbook):
        ws = workbook["unit_node_constraint_c"]
        def_row, def_col = find_crossing_point(ws)
        # Row 1 has "description" at col G (index 6)
        # Col G: G1=description, G2=data type, G3=parameter, G4=empty
        assert def_col == 6, f"Expected def_col=6, got {def_col}"
        assert def_row == 2, f"Expected def_row=2, got {def_row}"


# ---------------------------------------------------------------------------
# Sheet metadata parsing
# ---------------------------------------------------------------------------


class TestParseMetadata:
    def test_node_c_metadata(self, workbook):
        ws = workbook["node_c"]
        meta = parse_sheet_metadata(ws)
        assert meta is not None
        assert meta.sheet_name == "node_c"

        # Column definitions
        assert meta.alt_col == 0  # "alternative" at col A
        assert meta.entity_classes[0].class_name == "node"
        assert meta.entity_classes[0].dimensions == ["node"]

        # Entity existence column (recognised from "Entity Alternative" alias)
        assert meta.entity_existence_col is not None

        # Parameter columns should include has_balance, inflow, etc.
        param_names = set(meta.param_cols.values())
        assert "has_balance" in param_names
        assert "inflow" in param_names
        assert "existing" in param_names
        # entity existence should be in params (mapped from Entity Alternative)
        assert "entity existence" in param_names

        # Data types
        assert any(dt == "string" for dt in meta.data_types.values())
        assert any(dt == "float" for dt in meta.data_types.values())

        # Data starts at row 3 (0-based)
        assert meta.data_start_row == 3

    def test_unit_node_constraint_c_metadata(self, workbook):
        ws = workbook["unit_node_constraint_c"]
        meta = parse_sheet_metadata(ws)
        assert meta is not None

        # Should have two entity classes
        assert len(meta.entity_classes) == 2
        class_names = {ec.class_name for ec in meta.entity_classes}
        assert "unit__inputNode" in class_names
        assert "unit__outputNode" in class_names

        # Should have filter
        assert meta.filter_col is not None
        assert "unit__inputNode" in meta.filter_map
        assert meta.filter_map["unit__inputNode"] == "^input$"

        # Should have index column
        assert meta.index_col is not None
        assert meta.index_name == "constraint"


# ---------------------------------------------------------------------------
# Link sheet
# ---------------------------------------------------------------------------


class TestLinkSheet:
    def test_commodity_node(self, workbook):
        ws = workbook["commodity_node"]
        meta, _is_scenario = detect_and_parse_sheet(ws)
        assert meta is not None
        assert meta.entity_classes[0].class_name == "commodity__node"
        assert meta.entity_classes[0].dimensions == ["commodity", "node"]

    def test_commodity_node_data(self, workbook):
        ws = workbook["commodity_node"]
        meta, _is_scenario = detect_and_parse_sheet(ws)
        data = extract_sheet_data(ws, meta)
        assert len(data.link_entities) >= 2
        # Should contain Coal→Coal_node and Gas→Gas_node
        entity_set = set(data.link_entities)
        assert ("Coal", "Coal_node") in entity_set
        assert ("Gas", "Gas_node") in entity_set


# ---------------------------------------------------------------------------
# Transposed (timeseries) sheet
# ---------------------------------------------------------------------------


class TestTransposedSheet:
    def test_profile_t(self, workbook):
        ws = workbook["profile_t"]
        meta, _is_scenario = detect_and_parse_sheet(ws)
        assert meta is not None
        assert meta.is_transposed is True

        # Entity class should be profile
        assert meta.entity_classes[0].class_name == "profile"

        # Index should be time
        assert meta.index_name == "time"

        # Alternative and entity rows should be found
        assert meta.alt_row is not None
        assert meta.entity_row is not None

    def test_profile_t_data(self, workbook):
        ws = workbook["profile_t"]
        meta, _is_scenario = detect_and_parse_sheet(ws)
        data = extract_sheet_data(ws, meta)
        assert len(data.records) > 0

        # Check that we got records for different profiles
        entities = {r["entity_byname"] for r in data.records}
        assert ("Wind1",) in entities
        assert ("Battery_profile",) in entities

        # All records should have time index
        for r in data.records:
            assert r["index_value"] is not None
            assert r["index_name"] == "time"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


class TestDataExtraction:
    def test_node_c_data(self, workbook):
        ws = workbook["node_c"]
        meta, _is_scenario = detect_and_parse_sheet(ws)
        data = extract_sheet_data(ws, meta)
        assert len(data.records) > 0

        # Check we got some expected data
        alternatives = {r["alternative"] for r in data.records}
        assert "Base" in alternatives

        entities = {r["entity_byname"] for r in data.records}
        assert ("Coal_node",) in entities or ("NodeA",) in entities

        # Check entity existence records
        ee_records = [r for r in data.records if r["param_name"] == ENTITY_EXISTENCE]
        assert len(ee_records) > 0

    def test_unit_node_constraint_c_data(self, workbook):
        ws = workbook["unit_node_constraint_c"]
        meta, _is_scenario = detect_and_parse_sheet(ws)
        data = extract_sheet_data(ws, meta)
        assert len(data.records) > 0

        # Should have records from both entity classes
        classes = {r["entity_class"] for r in data.records}
        assert "unit__inputNode" in classes or "unit__outputNode" in classes

        # Check index values (constraint names)
        index_vals = {r["index_value"] for r in data.records if r["index_value"]}
        assert "c01" in index_vals or "gas_export" in index_vals
