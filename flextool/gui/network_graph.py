"""NetworkGraphBuilder -- reads Spine DB entities and renders a network graph."""

from __future__ import annotations

import logging
from typing import Any

from matplotlib.figure import Figure

try:
    import networkx as nx
except ImportError:
    nx = None  # type: ignore

try:
    import spinedb_api as api
except ImportError:
    api = None  # type: ignore

logger = logging.getLogger(__name__)


def build_network_figure(db_url: str) -> Figure | None:
    """Build a network graph *Figure* from the Spine database at *db_url*.

    Returns ``None`` if no lat/lon data is available or if required
    dependencies (``networkx``, ``spinedb_api``) are missing.
    """
    if nx is None:
        logger.warning("networkx is not installed -- cannot render network graph")
        return None
    if api is None:
        logger.warning("spinedb_api is not installed -- cannot render network graph")
        return None

    try:
        db = api.DatabaseMapping(db_url)
    except Exception:
        logger.exception("Failed to open database: %s", db_url)
        return None

    try:
        # 1. Get nodes with lat/lon
        nodes: dict[str, dict[str, float | None]] = {}
        for entity in db.find_entities(entity_class_name="node"):
            name = entity["entity_byname"][0]
            nodes[name] = {"lat": None, "lon": None}

        for pv in db.find_parameter_values(entity_class_name="node"):
            name = pv["entity_byname"][0]
            param = pv["parameter_definition_name"]
            value = api.from_database(pv["value"], pv["type"])
            if param in ("latitude", "lat") and name in nodes:
                nodes[name]["lat"] = float(value)
            elif param in ("longitude", "lon") and name in nodes:
                nodes[name]["lon"] = float(value)

        # Check if we have coordinates -- abort if not
        if not any(
            n["lat"] is not None and n["lon"] is not None for n in nodes.values()
        ):
            return None

        # Filter to nodes that have coordinates
        positioned_nodes = {
            k: v
            for k, v in nodes.items()
            if v["lat"] is not None and v["lon"] is not None
        }

        # 2. Get units
        units = [
            e["entity_byname"][0]
            for e in db.find_entities(entity_class_name="unit")
        ]

        # 3. Get relationships
        unit_inputs: dict[str, list[str]] = {}
        for e in db.find_entities(entity_class_name="unit__inputNode"):
            unit, node = e["entity_byname"]
            unit_inputs.setdefault(unit, []).append(node)

        unit_outputs: dict[str, list[str]] = {}
        for e in db.find_entities(entity_class_name="unit__outputNode"):
            unit, node = e["entity_byname"]
            unit_outputs.setdefault(unit, []).append(node)

        conn_endpoints: dict[str, tuple[str, str]] = {}
        for e in db.find_entities(entity_class_name="connection__node__node"):
            conn, n1, n2 = e["entity_byname"]
            conn_endpoints[conn] = (n1, n2)

        # 4. Build networkx graph and render
        return _render_network(
            positioned_nodes, units, unit_inputs, unit_outputs, conn_endpoints
        )
    except Exception:
        logger.exception("Error building network graph from %s", db_url)
        return None
    finally:
        db.close()


def _render_network(
    positioned_nodes: dict[str, dict[str, Any]],
    units: list[str],
    unit_inputs: dict[str, list[str]],
    unit_outputs: dict[str, list[str]],
    conn_endpoints: dict[str, tuple[str, str]],
) -> Figure:
    """Build a networkx graph and render it to a matplotlib *Figure*.

    This function is separated from the database layer so it can be
    tested independently with mock data.

    Parameters
    ----------
    positioned_nodes
        ``{node_name: {"lat": float, "lon": float}}``
    units
        List of unit entity names.
    unit_inputs
        ``{unit_name: [input_node, ...]}``
    unit_outputs
        ``{unit_name: [output_node, ...]}``
    conn_endpoints
        ``{connection_name: (node1, node2)}``
    """
    G = nx.Graph()

    # Add nodes at lat/lon positions
    pos: dict[str, tuple[float, float]] = {}
    for name, coords in positioned_nodes.items():
        G.add_node(name, node_type="node")
        pos[name] = (coords["lon"], coords["lat"])  # x=lon, y=lat

    # Add units -- position at midpoint of connected nodes
    for unit in units:
        connected = unit_inputs.get(unit, []) + unit_outputs.get(unit, [])
        connected_positioned = [n for n in connected if n in pos]
        if not connected_positioned:
            continue  # skip units with no positioned nodes
        avg_lon = sum(pos[n][0] for n in connected_positioned) / len(
            connected_positioned
        )
        avg_lat = sum(pos[n][1] for n in connected_positioned) / len(
            connected_positioned
        )
        G.add_node(unit, node_type="unit")
        pos[unit] = (avg_lon, avg_lat)

    # Add connection edges (node -- node arcs)
    for conn, (n1, n2) in conn_endpoints.items():
        if n1 in pos and n2 in pos:
            G.add_edge(n1, n2, edge_type="connection", label=conn)

    # Add unit edges (unit -- node arcs)
    for unit, input_nodes in unit_inputs.items():
        for node in input_nodes:
            if unit in pos and node in pos:
                G.add_edge(node, unit, edge_type="unit_input")

    for unit, output_nodes in unit_outputs.items():
        for node in output_nodes:
            if unit in pos and node in pos:
                G.add_edge(unit, node, edge_type="unit_output")

    # Render to Figure
    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111)

    # Draw nodes by type with different styles
    node_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "node"]
    unit_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "unit"]

    nx.draw_networkx_nodes(
        G, pos, nodelist=node_nodes, ax=ax,
        node_color="steelblue", node_size=300, node_shape="o",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=unit_nodes, ax=ax,
        node_color="coral", node_size=200, node_shape="s",
    )

    # Draw edges by type
    conn_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") == "connection"
    ]
    unit_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("edge_type") in ("unit_input", "unit_output")
    ]

    nx.draw_networkx_edges(
        G, pos, edgelist=conn_edges, ax=ax,
        width=2, edge_color="gray",
        connectionstyle="arc3,rad=0.1",
        arrows=True,
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=unit_edges, ax=ax,
        width=1, edge_color="lightgray", style="dashed",
    )

    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")
    fig.tight_layout()

    return fig
