"""Tests for DC power flow data pipeline in input_writer.

These tests verify:
- Group-level transfer_method override logic
- DC power flow connection/node identification
- Susceptance computation (base_MVA / reactance)
- Reference bus auto-detection (BFS + largest existing capacity)
- CSV file output (node_dc_power_flow, connection_dc_power_flow,
  node_reference_angle, p_connection_susceptance)
- Process method override via ct_method_overrides
- Integration test: full FlexTool run on PGLib case14 IEEE (DC-OPF)
"""

import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, import_data, to_database

from flextool.flextoolrunner.input_writer import (
    METHODS_MAPPING,
    _write_dc_power_flow_data,
    _write_process_method,
)

logger = logging.getLogger("test_dc_power_flow")


# ---------------------------------------------------------------------------
# Helper: create a minimal Spine DB with FlexTool-like schema
# ---------------------------------------------------------------------------

def _create_test_db(db_path: str) -> str:
    """Create a Spine DB with FlexTool entity classes + parameter defs.

    Returns the sqlite:/// URL.
    """
    url = f"sqlite:///{db_path}"
    with DatabaseMapping(url, create=True) as db:
        entity_classes = [
            ("node", ()),
            ("unit", ()),
            ("connection", ()),
            ("group", ()),
            ("group__node", ("group", "node")),
            ("group__connection", ("group", "connection")),
            ("unit__inputNode", ("unit", "node")),
            ("unit__outputNode", ("unit", "node")),
            ("connection__node__node", ("connection", "node", "node")),
        ]
        parameter_definitions = [
            ("connection", "transfer_method"),
            ("connection", "startup_method"),
            ("connection", "reactance"),
            ("connection", "existing"),
            ("connection", "is_DC"),
            ("unit", "conversion_method"),
            ("unit", "startup_method"),
            ("group", "transfer_method"),
            ("group", "base_MVA"),
            ("group", "reference_node"),
            ("group", "candidate_precapacity_to_avoid_big_m"),
            ("node", "existing"),
        ]
        count, errors = import_data(
            db,
            entity_classes=entity_classes,
            parameter_definitions=parameter_definitions,
        )
        assert not errors, f"DB init errors: {errors}"
        db.commit_session("init schema")
    return url


def _add_data(url: str, **kwargs) -> None:
    """Open the DB and import additional data (entities, parameter_values, etc.)."""
    with DatabaseMapping(url) as db:
        count, errors = import_data(db, **kwargs)
        assert not errors, f"import_data errors: {errors}"
        db.commit_session("add test data")


# ---------------------------------------------------------------------------
# Helper: read CSV produced by _write_dc_power_flow_data / _write_process_method
# ---------------------------------------------------------------------------

def _read_csv(filepath: Path) -> list[dict[str, str]]:
    """Read a CSV file and return list of row-dicts."""
    with open(filepath) as f:
        return list(csv.DictReader(f))


def _read_csv_column(filepath: Path, col: str) -> list[str]:
    """Read one column from a CSV file."""
    return [row[col] for row in _read_csv(filepath)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def work_dir(tmp_path: Path) -> Path:
    """Create work directory with input/ subdirectory."""
    (tmp_path / "input").mkdir()
    return tmp_path


@pytest.fixture()
def triangle_db(tmp_path: Path) -> str:
    """3-bus triangle DB for DC power flow tests.

    Topology::

        bus_A ---conn_AB--- bus_B
          \\                  /
         conn_AC          conn_BC
            \\              /
              --- bus_C ---

    All connections have reactance=0.1, existing=100.
    Group ``ac_network`` has transfer_method=dc_power_flow_with_angles, base_MVA=100.
    """
    db_path = str(tmp_path / "triangle.sqlite")
    url = _create_test_db(db_path)

    entities = [
        ("node", "bus_A"),
        ("node", "bus_B"),
        ("node", "bus_C"),
        ("connection", "conn_AB"),
        ("connection", "conn_BC"),
        ("connection", "conn_AC"),
        ("group", "ac_network"),
        # group__node memberships
        ("group__node", ("ac_network", "bus_A")),
        ("group__node", ("ac_network", "bus_B")),
        ("group__node", ("ac_network", "bus_C")),
        # connection topology: connection__node__node (conn, from, to)
        ("connection__node__node", ("conn_AB", "bus_A", "bus_B")),
        ("connection__node__node", ("conn_BC", "bus_B", "bus_C")),
        ("connection__node__node", ("conn_AC", "bus_A", "bus_C")),
    ]

    parameter_values = [
        # Connection parameters
        ("connection", "conn_AB", "transfer_method", "regular"),
        ("connection", "conn_BC", "transfer_method", "regular"),
        ("connection", "conn_AC", "transfer_method", "regular"),
        ("connection", "conn_AB", "reactance", 0.1),
        ("connection", "conn_BC", "reactance", 0.1),
        ("connection", "conn_AC", "reactance", 0.1),
        ("connection", "conn_AB", "existing", 100.0),
        ("connection", "conn_BC", "existing", 100.0),
        ("connection", "conn_AC", "existing", 100.0),
        # Group parameters
        ("group", "ac_network", "transfer_method", "dc_power_flow_with_angles"),
        ("group", "ac_network", "base_MVA", 100.0),
        # Node existing capacity (for reference bus auto-detection)
        ("node", "bus_A", "existing", 200.0),
        ("node", "bus_B", "existing", 0.0),
        ("node", "bus_C", "existing", 50.0),
    ]

    _add_data(url, entities=entities, parameter_values=parameter_values)
    return url


# ===================================================================
# Test 1: DC PF data pipeline — 3-bus triangle
# ===================================================================

class TestDCPowerFlowDataPipeline:
    """Tests for _write_dc_power_flow_data using the triangle fixture."""

    def test_dc_pf_nodes_csv(self, triangle_db: str, work_dir: Path) -> None:
        """All 3 nodes in the DC PF group appear in node_dc_power_flow.csv."""
        with DatabaseMapping(triangle_db) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, work_dir, logger)

        nodes = _read_csv_column(work_dir / "input" / "node_dc_power_flow.csv", "node")
        assert sorted(nodes) == ["bus_A", "bus_B", "bus_C"]

    def test_dc_pf_connections_csv(self, triangle_db: str, work_dir: Path) -> None:
        """All 3 connections appear in connection_dc_power_flow.csv."""
        with DatabaseMapping(triangle_db) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, work_dir, logger)

        conns = _read_csv_column(work_dir / "input" / "connection_dc_power_flow.csv", "process")
        assert sorted(conns) == ["conn_AB", "conn_AC", "conn_BC"]

    def test_susceptance_computation(self, triangle_db: str, work_dir: Path) -> None:
        """Susceptance = base_MVA / reactance = 100 / 0.1 = 1000 for each line."""
        with DatabaseMapping(triangle_db) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, work_dir, logger)

        rows = _read_csv(work_dir / "input" / "p_connection_susceptance.csv")
        susceptances = {r["process"]: float(r["p_connection_susceptance"]) for r in rows}
        assert susceptances == pytest.approx(
            {"conn_AB": 1000.0, "conn_AC": 1000.0, "conn_BC": 1000.0}
        )

    def test_reference_node_auto_detection(self, triangle_db: str, work_dir: Path) -> None:
        """Auto-selected reference node is bus_A (largest existing=200)."""
        with DatabaseMapping(triangle_db) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, work_dir, logger)

        ref_nodes = _read_csv_column(work_dir / "input" / "node_reference_angle.csv", "node")
        assert ref_nodes == ["bus_A"]

    def test_ct_method_overrides_returned(self, triangle_db: str, work_dir: Path) -> None:
        """DC PF connections get ct_method override to no_losses_no_variable_cost."""
        with DatabaseMapping(triangle_db) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, work_dir, logger)

        assert overrides == {
            "conn_AB": "no_losses_no_variable_cost",
            "conn_BC": "no_losses_no_variable_cost",
            "conn_AC": "no_losses_no_variable_cost",
        }


# ===================================================================
# Test 2: Reference bus — explicit vs auto-detection
# ===================================================================

class TestReferenceNodeSelection:

    def test_explicit_reference_node(self, tmp_path: Path) -> None:
        """When reference_node is set on the group, it should be used."""
        db_path = str(tmp_path / "ref_explicit.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "c12"),
                ("group", "g1"),
                ("group__node", ("g1", "n1")),
                ("group__node", ("g1", "n2")),
                ("connection__node__node", ("c12", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "c12", "reactance", 0.2),
                ("connection", "c12", "existing", 50.0),
                ("group", "g1", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "g1", "base_MVA", 100.0),
                ("group", "g1", "reference_node", "n2"),
                # n1 has larger capacity, but explicit ref overrides
                ("node", "n1", "existing", 999.0),
                ("node", "n2", "existing", 1.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, wf, logger)

        ref_nodes = _read_csv_column(wf / "input" / "node_reference_angle.csv", "node")
        assert ref_nodes == ["n2"]

    def test_auto_select_largest_capacity(self, tmp_path: Path) -> None:
        """Without reference_node, the node with largest existing capacity is chosen."""
        db_path = str(tmp_path / "ref_auto.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "alpha"), ("node", "beta"), ("node", "gamma"),
                ("connection", "c_ab"), ("connection", "c_bg"),
                ("group", "net"),
                ("group__node", ("net", "alpha")),
                ("group__node", ("net", "beta")),
                ("group__node", ("net", "gamma")),
                ("connection__node__node", ("c_ab", "alpha", "beta")),
                ("connection__node__node", ("c_bg", "beta", "gamma")),
            ],
            parameter_values=[
                ("connection", "c_ab", "reactance", 0.5),
                ("connection", "c_bg", "reactance", 0.5),
                ("connection", "c_ab", "existing", 10.0),
                ("connection", "c_bg", "existing", 10.0),
                ("group", "net", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "net", "base_MVA", 100.0),
                ("node", "alpha", "existing", 10.0),
                ("node", "beta", "existing", 500.0),  # largest
                ("node", "gamma", "existing", 100.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, wf, logger)

        ref_nodes = _read_csv_column(wf / "input" / "node_reference_angle.csv", "node")
        assert ref_nodes == ["beta"]


# ===================================================================
# Test 3: Susceptance with non-default base_MVA
# ===================================================================

class TestSusceptanceComputation:

    def test_custom_base_mva(self, tmp_path: Path) -> None:
        """Susceptance = base_MVA / reactance with non-default base_MVA."""
        db_path = str(tmp_path / "susc.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "line"),
                ("group", "g"),
                ("group__node", ("g", "n1")),
                ("group__node", ("g", "n2")),
                ("connection__node__node", ("line", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "line", "reactance", 0.05),
                ("connection", "line", "existing", 200.0),
                ("group", "g", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "g", "base_MVA", 200.0),  # non-default
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, wf, logger)

        rows = _read_csv(wf / "input" / "p_connection_susceptance.csv")
        assert len(rows) == 1
        assert rows[0]["process"] == "line"
        assert float(rows[0]["p_connection_susceptance"]) == pytest.approx(4000.0)  # 200/0.05

    def test_zero_reactance_skipped(self, tmp_path: Path) -> None:
        """Connection with zero reactance is skipped (no susceptance row)."""
        db_path = str(tmp_path / "zero_react.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "line"),
                ("group", "g"),
                ("group__node", ("g", "n1")),
                ("group__node", ("g", "n2")),
                ("connection__node__node", ("line", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "line", "reactance", 0.0),
                ("connection", "line", "existing", 100.0),
                ("group", "g", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "g", "base_MVA", 100.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, wf, logger)

        rows = _read_csv(wf / "input" / "p_connection_susceptance.csv")
        assert len(rows) == 0  # zero reactance -> no susceptance

    def test_missing_reactance_skipped(self, tmp_path: Path) -> None:
        """Connection without reactance parameter has no susceptance row."""
        db_path = str(tmp_path / "no_react.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "line"),
                ("group", "g"),
                ("group__node", ("g", "n1")),
                ("group__node", ("g", "n2")),
                ("connection__node__node", ("line", "n1", "n2")),
            ],
            parameter_values=[
                # No reactance set on 'line'
                ("connection", "line", "existing", 100.0),
                ("group", "g", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "g", "base_MVA", 100.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_dc_power_flow_data(db, wf, logger)

        rows = _read_csv(wf / "input" / "p_connection_susceptance.csv")
        assert len(rows) == 0


# ===================================================================
# Test 4: is_DC connections are excluded from DC PF
# ===================================================================

class TestIsDCExclusion:

    def test_is_dc_connection_excluded(self, tmp_path: Path) -> None:
        """Connection with is_DC=yes is not included in DC PF sets."""
        db_path = str(tmp_path / "is_dc.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "ac_line"), ("connection", "dc_link"),
                ("group", "g"),
                ("group__node", ("g", "n1")),
                ("group__node", ("g", "n2")),
                ("connection__node__node", ("ac_line", "n1", "n2")),
                ("connection__node__node", ("dc_link", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "ac_line", "reactance", 0.1),
                ("connection", "ac_line", "existing", 100.0),
                ("connection", "dc_link", "reactance", 0.1),
                ("connection", "dc_link", "existing", 100.0),
                ("connection", "dc_link", "is_DC", "yes"),
                ("group", "g", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "g", "base_MVA", 100.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, wf, logger)

        # Only ac_line should be in DC PF
        conns = _read_csv_column(wf / "input" / "connection_dc_power_flow.csv", "process")
        assert conns == ["ac_line"]

        # dc_link should NOT have a ct_method override
        assert "dc_link" not in overrides
        assert overrides == {"ac_line": "no_losses_no_variable_cost"}


# ===================================================================
# Test 5: Group transfer_method override (non-DC methods)
# ===================================================================

class TestGroupTransferMethodOverride:

    def test_regular_override(self, tmp_path: Path) -> None:
        """Group with transfer_method=regular overrides connection methods."""
        db_path = str(tmp_path / "reg_override.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "c12"),
                ("group", "g"),
                ("group__node", ("g", "n1")),
                ("group__node", ("g", "n2")),
                ("connection__node__node", ("c12", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "c12", "transfer_method", "no_losses_no_variable_cost"),
                ("group", "g", "transfer_method", "regular"),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, wf, logger)

        # Override should be 'regular' (from group), not DC PF
        assert overrides == {"c12": "regular"}

        # No DC PF nodes or connections (since method is 'regular', not DC PF)
        nodes = _read_csv_column(wf / "input" / "node_dc_power_flow.csv", "node")
        assert nodes == []
        conns = _read_csv_column(wf / "input" / "connection_dc_power_flow.csv", "process")
        assert conns == []

    def test_use_connection_transfer_methods_no_override(self, tmp_path: Path) -> None:
        """Group with use_connection_transfer_methods does not override anything."""
        db_path = str(tmp_path / "no_override.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "c12"),
                ("group", "g"),
                ("group__node", ("g", "n1")),
                ("group__node", ("g", "n2")),
                ("connection__node__node", ("c12", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "c12", "transfer_method", "regular"),
                ("group", "g", "transfer_method", "use_connection_transfer_methods"),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, wf, logger)

        assert overrides == {}


# ===================================================================
# Test 6: _write_process_method with DC PF overrides
# ===================================================================

class TestProcessMethodWithOverrides:

    def test_dc_pf_override_changes_method(self, tmp_path: Path) -> None:
        """DC PF override changes connection method to method_2way_1var_off.

        no_losses_no_variable_cost + no_startup + fork_no -> method_2way_1var_off
        """
        db_path = str(tmp_path / "pm.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "line1"),
                ("connection__node__node", ("line1", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "line1", "transfer_method", "regular"),
            ],
        )

        ct_overrides = {"line1": "no_losses_no_variable_cost"}

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_process_method(db, wf, logger, ct_method_overrides=ct_overrides)

        rows = _read_csv(wf / "input" / "process_method.csv")
        method_map = {r["process"]: r["method"] for r in rows}
        assert method_map["line1"] == "method_2way_1var_off"

    def test_regular_connection_without_override(self, tmp_path: Path) -> None:
        """Connection with transfer_method=regular (no override) -> method_2way_2var_exclude."""
        db_path = str(tmp_path / "pm_regular.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "c12"),
                ("connection__node__node", ("c12", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "c12", "transfer_method", "regular"),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_process_method(db, wf, logger, ct_method_overrides={})

        rows = _read_csv(wf / "input" / "process_method.csv")
        method_map = {r["process"]: r["method"] for r in rows}
        assert method_map["c12"] == "method_2way_2var_exclude"

    def test_unit_method_not_affected_by_ct_overrides(self, tmp_path: Path) -> None:
        """Unit methods are not affected by connection ct_method overrides."""
        db_path = str(tmp_path / "pm_unit.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"),
                ("unit", "gen"),
                ("unit__outputNode", ("gen", "n1")),
            ],
            parameter_values=[
                ("unit", "gen", "conversion_method", "constant_efficiency"),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_process_method(db, wf, logger, ct_method_overrides={})

        rows = _read_csv(wf / "input" / "process_method.csv")
        method_map = {r["process"]: r["method"] for r in rows}
        assert method_map["gen"] == "method_1way_1var_off"

    def test_fork_yes_with_multiple_sinks(self, tmp_path: Path) -> None:
        """Connection with >1 sink gets fork_yes, changing method variant."""
        db_path = str(tmp_path / "pm_fork.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        # connection__node__node creates both a source and a sink per row
        # Two rows => 2 sources + 2 sinks => fork_yes
        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"), ("node", "n3"),
                ("connection", "c123"),
                ("connection__node__node", ("c123", "n1", "n2")),
                ("connection__node__node", ("c123", "n1", "n3")),
            ],
            parameter_values=[
                ("connection", "c123", "transfer_method", "no_losses_no_variable_cost"),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            _write_process_method(db, wf, logger, ct_method_overrides={})

        rows = _read_csv(wf / "input" / "process_method.csv")
        method_map = {r["process"]: r["method"] for r in rows}
        # no_losses_no_variable_cost + no_startup + fork_yes -> method_2way_nvar_off
        assert method_map["c123"] == "method_2way_nvar_off"


# ===================================================================
# Test 7: Connection outside group not affected
# ===================================================================

class TestConnectionOutsideGroup:

    def test_connection_partially_outside_group(self, tmp_path: Path) -> None:
        """Connection with only one endpoint in the group is NOT overridden."""
        db_path = str(tmp_path / "partial.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "in_group"), ("node", "outside"),
                ("connection", "cross_border"),
                ("group", "g"),
                ("group__node", ("g", "in_group")),
                # 'outside' is NOT in the group
                ("connection__node__node", ("cross_border", "in_group", "outside")),
            ],
            parameter_values=[
                ("connection", "cross_border", "reactance", 0.1),
                ("connection", "cross_border", "existing", 100.0),
                ("group", "g", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "g", "base_MVA", 100.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, wf, logger)

        assert overrides == {}
        conns = _read_csv_column(wf / "input" / "connection_dc_power_flow.csv", "process")
        assert conns == []


# ===================================================================
# Test 8: Empty DC PF (no groups with dc_power_flow_with_angles)
# ===================================================================

class TestNoDCPowerFlow:

    def test_no_dc_pf_groups(self, tmp_path: Path) -> None:
        """When no DC PF groups exist, CSV files are empty (header only)."""
        db_path = str(tmp_path / "empty_dc.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "n1"), ("node", "n2"),
                ("connection", "c12"),
                ("connection__node__node", ("c12", "n1", "n2")),
            ],
            parameter_values=[
                ("connection", "c12", "transfer_method", "regular"),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, wf, logger)

        assert overrides == {}
        nodes = _read_csv_column(wf / "input" / "node_dc_power_flow.csv", "node")
        assert nodes == []
        conns = _read_csv_column(wf / "input" / "connection_dc_power_flow.csv", "process")
        assert conns == []
        refs = _read_csv_column(wf / "input" / "node_reference_angle.csv", "node")
        assert refs == []
        susc = _read_csv(wf / "input" / "p_connection_susceptance.csv")
        assert susc == []


# ===================================================================
# Test 9: METHODS_MAPPING sanity checks
# ===================================================================

class TestMethodsMapping:

    def test_no_losses_no_startup_fork_no(self) -> None:
        assert METHODS_MAPPING[("no_losses_no_variable_cost", "no_startup", "fork_no")] == "method_2way_1var_off"

    def test_regular_no_startup_fork_no(self) -> None:
        assert METHODS_MAPPING[("regular", "no_startup", "fork_no")] == "method_2way_2var_exclude"

    def test_constant_efficiency_no_startup_fork_no(self) -> None:
        assert METHODS_MAPPING[("constant_efficiency", "no_startup", "fork_no")] == "method_1way_1var_off"

    def test_all_dc_pf_relevant_mappings_exist(self) -> None:
        """All ct_methods that DC PF might use must have valid mappings."""
        for ct in ["no_losses_no_variable_cost", "regular", "exact", "variable_cost_only"]:
            key = (ct, "no_startup", "fork_no")
            assert key in METHODS_MAPPING, f"Missing mapping for {key}"


# ===================================================================
# Test 10: Multiple DC PF groups
# ===================================================================

class TestMultipleDCPFGroups:

    def test_two_independent_groups(self, tmp_path: Path) -> None:
        """Two separate DC PF groups produce correct combined outputs."""
        db_path = str(tmp_path / "multi_grp.sqlite")
        url = _create_test_db(db_path)
        wf = tmp_path / "work"
        (wf / "input").mkdir(parents=True)

        _add_data(
            url,
            entities=[
                ("node", "a1"), ("node", "a2"),
                ("node", "b1"), ("node", "b2"),
                ("connection", "line_a"), ("connection", "line_b"),
                ("group", "grp_a"), ("group", "grp_b"),
                ("group__node", ("grp_a", "a1")),
                ("group__node", ("grp_a", "a2")),
                ("group__node", ("grp_b", "b1")),
                ("group__node", ("grp_b", "b2")),
                ("connection__node__node", ("line_a", "a1", "a2")),
                ("connection__node__node", ("line_b", "b1", "b2")),
            ],
            parameter_values=[
                ("connection", "line_a", "reactance", 0.1),
                ("connection", "line_a", "existing", 100.0),
                ("connection", "line_b", "reactance", 0.2),
                ("connection", "line_b", "existing", 100.0),
                ("group", "grp_a", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "grp_a", "base_MVA", 100.0),
                ("group", "grp_b", "transfer_method", "dc_power_flow_with_angles"),
                ("group", "grp_b", "base_MVA", 200.0),
                # Different existing capacities for each group's auto-ref detection
                ("node", "a1", "existing", 50.0),
                ("node", "a2", "existing", 100.0),
                ("node", "b1", "existing", 300.0),
                ("node", "b2", "existing", 10.0),
            ],
        )

        with DatabaseMapping(url) as db:
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            overrides = _write_dc_power_flow_data(db, wf, logger)

        # All 4 nodes present
        nodes = sorted(_read_csv_column(wf / "input" / "node_dc_power_flow.csv", "node"))
        assert nodes == ["a1", "a2", "b1", "b2"]

        # Both connections present
        conns = sorted(_read_csv_column(wf / "input" / "connection_dc_power_flow.csv", "process"))
        assert conns == ["line_a", "line_b"]

        # Susceptance: line_a = 100/0.1 = 1000, line_b = 200/0.2 = 1000
        rows = _read_csv(wf / "input" / "p_connection_susceptance.csv")
        susc = {r["process"]: float(r["p_connection_susceptance"]) for r in rows}
        assert susc == pytest.approx({"line_a": 1000.0, "line_b": 1000.0})

        # Reference nodes: a2 (existing=100 > 50) and b1 (existing=300 > 10)
        ref_nodes = sorted(_read_csv_column(wf / "input" / "node_reference_angle.csv", "node"))
        assert ref_nodes == sorted(["a2", "b1"])

        # Both connections overridden
        assert overrides == {
            "line_a": "no_losses_no_variable_cost",
            "line_b": "no_losses_no_variable_cost",
        }


# ===================================================================
# Test 11: PGLib case14 IEEE integration test (full FlexTool run)
# ===================================================================

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CASE14_PATH = _PROJECT_ROOT / "tests" / "data" / "pglib_opf_case14_ieee.m"


@pytest.mark.slow
class TestPGLibCase14Integration:
    """Integration test: run FlexTool on PGLib case14 IEEE and validate DC-OPF results.

    The test parses the MATPOWER file, creates a FlexTool Spine DB,
    runs the full model, and checks the objective value and reference
    bus angle against known DC-OPF results.

    For case14 with linear costs only (c2=0 for all generators):
    - Only 2 generators have Pmax > 0: bus 1 (340 MW, $7.92/MWh) and
      bus 2 (59 MW, $23.27/MWh)
    - Total demand = 259 MW, all supplied by the cheapest generator
    - Expected hourly cost = 259 * 7.920951 = $2,051.53/h
    - FlexTool annualizes: v_obj = hourly_cost * 8760
    """

    @pytest.fixture()
    def case14_run(self, tmp_path: Path) -> dict[str, object]:
        """Parse case14, create DB, run FlexTool, return results dict."""
        from flextool.process_inputs.read_matpower import (
            create_flextool_db_from_matpower,
            read_matpower,
        )

        # Parse MATPOWER file
        case = read_matpower(str(_CASE14_PATH))
        assert case.name == "pglib_opf_case14_ieee"
        assert len(case.buses) == 14
        assert len(case.generators) == 5
        assert len(case.branches) == 20

        # Create FlexTool DB
        db_path = str(tmp_path / "case14.sqlite")
        url = create_flextool_db_from_matpower(case, db_path)

        # Run FlexTool with --work-folder to isolate from other runs
        work_dir = tmp_path / "work"
        work_dir.mkdir()

        result = subprocess.run(
            [
                sys.executable, "run_flextool.py",
                url,
                "--scenario-name", "dc_opf_test",
                "--work-folder", str(work_dir),
                "--write-methods", "parquet",
            ],
            capture_output=True,
            text=True,
            cwd=str(_PROJECT_ROOT),
            timeout=120,
        )
        assert result.returncode == 0, (
            f"FlexTool failed (rc={result.returncode}).\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )

        # Read v_obj.csv from work directory
        v_obj_path = work_dir / "output_raw" / "v_obj.csv"
        assert v_obj_path.exists(), f"v_obj.csv not found at {v_obj_path}"
        with open(v_obj_path) as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            obj_value = float(row[1])

        # Read v_angle.csv
        v_angle_path = work_dir / "output_raw" / "v_angle.csv"
        assert v_angle_path.exists(), f"v_angle.csv not found at {v_angle_path}"
        with open(v_angle_path) as f:
            reader = csv.DictReader(f)
            angle_row = next(reader)

        # Read v_flow.csv
        v_flow_path = work_dir / "output_raw" / "v_flow.csv"
        assert v_flow_path.exists(), f"v_flow.csv not found at {v_flow_path}"

        return {
            "case": case,
            "obj_value": obj_value,
            "angle_row": angle_row,
            "stdout": result.stdout,
            "work_dir": work_dir,
        }

    def test_objective_value(self, case14_run: dict) -> None:
        """Total annualized cost matches expected DC-OPF result within 1%."""
        obj_value = case14_run["obj_value"]

        # Expected: all 259 MW demand served by gen at bus 1 ($7.920951/MWh)
        # Total demand = sum of Pd for all buses
        total_demand = sum(b.pd for b in case14_run["case"].buses)
        assert total_demand == pytest.approx(259.0, abs=0.1)

        # Hourly cost = total_demand * cheapest_gen_cost
        expected_hourly_cost = total_demand * 7.920951  # $2,051.53/h
        # FlexTool annualizes: objective = hourly_cost * 8760
        expected_annual = expected_hourly_cost * 8760

        assert obj_value == pytest.approx(expected_annual, rel=0.01), (
            f"Objective mismatch: got {obj_value:.2f}, "
            f"expected {expected_annual:.2f} "
            f"(hourly: {obj_value / 8760:.2f} vs {expected_hourly_cost:.2f})"
        )

    def test_reference_bus_angle_zero(self, case14_run: dict) -> None:
        """Reference bus (bus 1, type=3 slack) has angle = 0."""
        angle_row = case14_run["angle_row"]
        assert float(angle_row["bus_1"]) == 0.0

    def test_nonreference_buses_have_nonzero_angles(self, case14_run: dict) -> None:
        """Non-reference buses have non-zero voltage angles."""
        angle_row = case14_run["angle_row"]
        for bus_col in ["bus_2", "bus_3", "bus_5", "bus_9", "bus_14"]:
            assert float(angle_row[bus_col]) != 0.0, (
                f"{bus_col} should have non-zero angle"
            )
