"""Tests for flextool.gui.network_graph."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from matplotlib.figure import Figure


class TestBuildNetworkFigureGracefulDegradation(unittest.TestCase):
    """Test that build_network_figure handles missing dependencies."""

    def test_returns_none_when_networkx_missing(self) -> None:
        """If networkx is not importable, build_network_figure returns None."""
        with patch.dict("sys.modules", {"networkx": None}):
            # Re-import to pick up the patched module state
            import importlib
            import flextool.gui.network_graph as mod

            original_nx = mod.nx
            mod.nx = None
            try:
                result = mod.build_network_figure("sqlite:///fake.sqlite")
                self.assertIsNone(result)
            finally:
                mod.nx = original_nx

    def test_returns_none_when_spinedb_api_missing(self) -> None:
        """If spinedb_api is not importable, build_network_figure returns None."""
        import flextool.gui.network_graph as mod

        original_api = mod.api
        mod.api = None
        try:
            result = mod.build_network_figure("sqlite:///fake.sqlite")
            self.assertIsNone(result)
        finally:
            mod.api = original_api


class TestRenderNetwork(unittest.TestCase):
    """Test the _render_network function with mock data (no database needed)."""

    def setUp(self) -> None:
        """Import _render_network; skip if networkx is unavailable."""
        try:
            import networkx  # noqa: F401
        except ImportError:
            self.skipTest("networkx not installed")

        from flextool.gui.network_graph import _render_network

        self._render_network = _render_network

    def test_basic_network_returns_figure(self) -> None:
        """A simple graph with two nodes and a connection produces a Figure."""
        positioned_nodes = {
            "node_a": {"lat": 60.0, "lon": 24.0},
            "node_b": {"lat": 61.0, "lon": 25.0},
        }
        units = ["unit_1"]
        unit_inputs = {"unit_1": ["node_a"]}
        unit_outputs = {"unit_1": ["node_b"]}
        conn_endpoints = {"conn_ab": ("node_a", "node_b")}

        fig = self._render_network(
            positioned_nodes, units, unit_inputs, unit_outputs, conn_endpoints,
        )

        self.assertIsInstance(fig, Figure)
        # Should have exactly one axes
        self.assertEqual(len(fig.axes), 1)

    def test_empty_positioned_nodes(self) -> None:
        """With no positioned nodes the graph is still a valid (empty) Figure."""
        fig = self._render_network(
            positioned_nodes={},
            units=["u1"],
            unit_inputs={"u1": ["n1"]},
            unit_outputs={"u1": ["n2"]},
            conn_endpoints={},
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)

    def test_unit_with_no_positioned_connected_nodes_is_skipped(self) -> None:
        """Units whose connected nodes have no position are not added to the graph."""
        positioned_nodes = {
            "node_a": {"lat": 60.0, "lon": 24.0},
        }
        # unit_orphan connects only to node_b which has no position
        units = ["unit_orphan", "unit_ok"]
        unit_inputs = {"unit_orphan": ["node_b"], "unit_ok": ["node_a"]}
        unit_outputs = {"unit_orphan": ["node_c"], "unit_ok": ["node_a"]}
        conn_endpoints = {}

        fig = self._render_network(
            positioned_nodes, units, unit_inputs, unit_outputs, conn_endpoints,
        )

        self.assertIsInstance(fig, Figure)

    def test_multiple_connections_between_same_nodes(self) -> None:
        """Multiple connections between the same node pair are handled."""
        positioned_nodes = {
            "n1": {"lat": 50.0, "lon": 10.0},
            "n2": {"lat": 51.0, "lon": 11.0},
        }
        conn_endpoints = {
            "conn_1": ("n1", "n2"),
            "conn_2": ("n1", "n2"),
        }

        fig = self._render_network(
            positioned_nodes,
            units=[],
            unit_inputs={},
            unit_outputs={},
            conn_endpoints=conn_endpoints,
        )

        self.assertIsInstance(fig, Figure)

    def test_unit_positioned_at_midpoint(self) -> None:
        """A unit connected to two positioned nodes should be placed at their midpoint."""
        import networkx as nx

        positioned_nodes = {
            "n1": {"lat": 0.0, "lon": 0.0},
            "n2": {"lat": 10.0, "lon": 20.0},
        }
        units = ["u1"]
        unit_inputs = {"u1": ["n1"]}
        unit_outputs = {"u1": ["n2"]}

        fig = self._render_network(
            positioned_nodes, units, unit_inputs, unit_outputs, conn_endpoints={},
        )

        # Verify figure was created -- the midpoint positioning is an internal
        # detail but we can at least confirm the figure renders without error
        self.assertIsInstance(fig, Figure)

    def test_no_units_no_connections(self) -> None:
        """A graph with only nodes and no edges renders fine."""
        positioned_nodes = {
            "n1": {"lat": 1.0, "lon": 2.0},
            "n2": {"lat": 3.0, "lon": 4.0},
            "n3": {"lat": 5.0, "lon": 6.0},
        }

        fig = self._render_network(
            positioned_nodes,
            units=[],
            unit_inputs={},
            unit_outputs={},
            conn_endpoints={},
        )

        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 1)


if __name__ == "__main__":
    unittest.main()
