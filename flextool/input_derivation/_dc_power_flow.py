"""DC power flow input-derivation — BFS reference-node + susceptance.

Ports the legacy ``_write_dc_power_flow_data`` from
:mod:`flextool.flextoolrunner.input_writer` (lines 966-1250) into the
:mod:`flextool.input_derivation` tier.  The derivation reads raw input
through :class:`flextool.spinedb_backend.SpineDBBackend`, performs the
BFS reference-node selection + susceptance computation, and emits four
canonical frames into the :class:`FlexDataProvider`:

* ``input/node_dc_power_flow``        single-column ``node``
* ``input/connection_dc_power_flow``  single-column ``process``
* ``input/node_reference_angle``      single-column ``node``
* ``input/p_connection_susceptance``  ``process, p_connection_susceptance``

Plus a small carrier under ``derived/ct_method_overrides`` consumed by
the next derivation (:mod:`flextool.input_derivation._process_method`,
Phase C) — maps connection_name → overridden ct_method (necessary for
the group-level ``transfer_method`` mechanism that DC PF activates).

This module replaces the on-disk emission and the
``input_writer._write_dc_power_flow_data`` call site.  No CSV touches
disk here; the cascade consumer (:mod:`flextool.engine_polars._dc_power_flow`)
reads the four frames from the Provider.

See ``specs/step_2_5_audit.md`` Section 7 item 9 for the migration
plan and Section 1 for the input/output contract.
"""
from __future__ import annotations

import logging

import polars as pl


def derive_dc_power_flow(
    backend,
    provider,
    logger: logging.Logger,
) -> dict[str, str]:
    """Run the DC power flow derivation.

    Parameters
    ----------
    backend : SpineDBBackend
        The opened backend.  Used for raw EAV reads via
        :meth:`SpineDBBackend.find_entities` and
        :meth:`SpineDBBackend.find_parameter_values`.
    provider : FlexDataProvider
        The cascade-input Provider; receives the four output frames
        plus the ``derived/ct_method_overrides`` carrier.
    logger : logging.Logger
        Routed to the same logger as the legacy helper (DC PF warnings
        about missing reactance, zero existing capacity, and the
        auto-selected reference-node info messages).

    Returns
    -------
    dict[str, str]
        ``ct_method_overrides`` — connection_name → overridden
        ct_method.  Returned for the immediate-caller orchestration
        convenience (``write_input`` passes it to
        :func:`_write_process_method` today), but the canonical
        cross-derivation channel is the Provider's
        ``derived/ct_method_overrides`` key.

    Mirrors :func:`flextool.flextoolrunner.input_writer._write_dc_power_flow_data`
    (input_writer.py:966-1250) line-for-line.  Differences:

    * Reads via ``backend.find_entities`` / ``backend.find_parameter_values``
      instead of a passed-in :class:`DatabaseMapping`.
    * Emits to the Provider via ``provider.put(key, frame)`` instead of
      writing to ``input/<file>.csv``.
    * The four output frames use the canonical CSV column headers as
      their polars columns so downstream byte-parity is preserved when
      the Provider is snapshotted to disk.
    """
    ct_method_overrides: dict[str, str] = {}

    # --- Read group transfer_method parameters ---
    group_transfer_methods: dict[str, str] = {}
    for pv in backend.find_parameter_values(
        entity_class_name="group",
        parameter_definition_name="transfer_method",
    ):
        if pv["type"] is None:
            continue
        group_name = pv["entity_byname"][0]
        group_transfer_methods[group_name] = str(pv["parsed_value"])

    # --- Build group -> member_nodes mapping ---
    group_nodes: dict[str, set[str]] = {}
    for entity in backend.find_entities(entity_class_name="group__node"):
        group_name = entity["entity_byname"][0]
        node_name = entity["entity_byname"][1]
        group_nodes.setdefault(group_name, set()).add(node_name)

    # --- Build connection -> (node1, node2) mapping from connection__node__node ---
    connection_endpoints: dict[str, tuple[str, str]] = {}
    for entity in backend.find_entities(entity_class_name="connection__node__node"):
        conn_name = entity["entity_byname"][0]
        node1 = entity["entity_byname"][1]
        node2 = entity["entity_byname"][2]
        connection_endpoints[conn_name] = (node1, node2)

    # --- Read is_DC parameter for connections ---
    is_dc_connections: set[str] = set()
    for pv in backend.find_parameter_values(
        entity_class_name="connection",
        parameter_definition_name="is_DC",
    ):
        if pv["type"] is None:
            continue
        if str(pv["parsed_value"]) == "yes":
            is_dc_connections.add(pv["entity_byname"][0])

    # --- Apply group transfer_method overrides for ALL groups ---
    dc_pf_groups: list[str] = []
    for group_name, method in group_transfer_methods.items():
        if method == "use_connection_transfer_methods":
            continue
        nodes_in_group = group_nodes.get(group_name, set())
        if not nodes_in_group:
            continue

        is_dc_pf = method == "dc_power_flow_with_angles"
        if is_dc_pf:
            dc_pf_groups.append(group_name)

        # Find connections where BOTH endpoints are in the group's node set
        for conn_name, (n1, n2) in connection_endpoints.items():
            if n1 in nodes_in_group and n2 in nodes_in_group:
                # For dc_power_flow_with_angles: exclude is_DC=yes connections
                if is_dc_pf and conn_name in is_dc_connections:
                    continue
                if is_dc_pf:
                    # DC PF connections use no_losses_no_variable_cost internally
                    ct_method_overrides[conn_name] = "no_losses_no_variable_cost"
                else:
                    ct_method_overrides[conn_name] = method

    # --- DC power flow specific processing ---
    dc_pf_nodes: set[str] = set()
    dc_pf_connections: set[str] = set()
    reference_nodes: list[str] = []
    susceptance_map: dict[str, float] = {}

    for group_name in dc_pf_groups:
        nodes_in_group = group_nodes.get(group_name, set())
        dc_pf_nodes.update(nodes_in_group)

        # Collect DC PF connections for this group
        group_dc_connections: set[str] = set()
        for conn_name, (n1, n2) in connection_endpoints.items():
            if n1 in nodes_in_group and n2 in nodes_in_group:
                if conn_name not in is_dc_connections:
                    group_dc_connections.add(conn_name)
        dc_pf_connections.update(group_dc_connections)

        # --- Reference bus detection ---
        # Check if reference_node parameter is set on this group
        ref_node: str | None = None
        for pv in backend.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="reference_node",
        ):
            if pv["type"] is None:
                continue
            if pv["entity_byname"][0] == group_name:
                ref_node = str(pv["parsed_value"])
                break

        if ref_node is not None:
            if ref_node in nodes_in_group:
                reference_nodes.append(ref_node)
            else:
                logger.warning(
                    "DC power flow: reference_node '%s' for group '%s' is not a member node. "
                    "Falling back to automatic selection.",
                    ref_node, group_name,
                )
                ref_node = None

        if ref_node is None:
            # Automatic selection: build graph, find connected components via BFS,
            # pick node with largest existing capacity in each component.
            adjacency: dict[str, set[str]] = {}
            for conn_name in group_dc_connections:
                n1, n2 = connection_endpoints[conn_name]
                adjacency.setdefault(n1, set()).add(n2)
                adjacency.setdefault(n2, set()).add(n1)

            # Also include isolated nodes from the group
            for node in nodes_in_group:
                adjacency.setdefault(node, set())

            # Read existing capacity for nodes
            node_existing: dict[str, float] = {}
            for pv in backend.find_parameter_values(
                entity_class_name="node",
                parameter_definition_name="existing",
            ):
                if pv["type"] is None:
                    continue
                node_name = pv["entity_byname"][0]
                if node_name in nodes_in_group:
                    val = pv["parsed_value"]
                    if isinstance(val, (int, float)):
                        node_existing[node_name] = float(val)

            # BFS to find connected components
            visited: set[str] = set()
            for start_node in sorted(adjacency.keys()):
                if start_node in visited:
                    continue
                # BFS
                component: list[str] = []
                queue = [start_node]
                visited.add(start_node)
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    for neighbor in adjacency.get(current, set()):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                # Pick node with largest existing capacity in this component
                best_node = component[0]
                best_cap = node_existing.get(best_node, 0.0)
                for node in component[1:]:
                    cap = node_existing.get(node, 0.0)
                    if cap > best_cap:
                        best_cap = cap
                        best_node = node
                reference_nodes.append(best_node)
                logger.info(
                    "DC power flow: auto-selected reference node '%s' (existing=%.1f) "
                    "for connected component in group '%s'.",
                    best_node, best_cap, group_name,
                )

        # --- Susceptance computation ---
        base_mva = 100.0
        for pv in backend.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="base_MVA",
        ):
            if pv["type"] is None:
                continue
            if pv["entity_byname"][0] == group_name:
                base_mva = float(pv["parsed_value"])
                break

        # Read reactance for DC PF connections
        reactance_map: dict[str, float] = {}
        for pv in backend.find_parameter_values(
            entity_class_name="connection",
            parameter_definition_name="reactance",
        ):
            if pv["type"] is None:
                continue
            conn_name = pv["entity_byname"][0]
            if conn_name in group_dc_connections:
                reactance_map[conn_name] = float(pv["parsed_value"])

        # Compute susceptance (accumulated across groups)
        for conn_name in group_dc_connections:
            if conn_name in reactance_map:
                reactance = reactance_map[conn_name]
                if reactance != 0.0:
                    susceptance_map[conn_name] = base_mva / reactance
                else:
                    logger.warning(
                        "DC power flow: connection '%s' has zero reactance — skipping susceptance computation.",
                        conn_name,
                    )
            else:
                logger.warning(
                    "DC power flow: connection '%s' in group '%s' has no reactance parameter set.",
                    conn_name, group_name,
                )

        # --- Candidate pre-capacity warning ---
        precapacity = 0.1
        for pv in backend.find_parameter_values(
            entity_class_name="group",
            parameter_definition_name="candidate_precapacity_to_avoid_big_m",
        ):
            if pv["type"] is None:
                continue
            if pv["entity_byname"][0] == group_name:
                precapacity = float(pv["parsed_value"])
                break

        # Read existing capacity for connections
        conn_existing: dict[str, float] = {}
        for pv in backend.find_parameter_values(
            entity_class_name="connection",
            parameter_definition_name="existing",
        ):
            if pv["type"] is None:
                continue
            conn_name = pv["entity_byname"][0]
            if conn_name in group_dc_connections:
                val = pv["parsed_value"]
                if isinstance(val, (int, float)):
                    conn_existing[conn_name] = float(val)

        for conn_name in group_dc_connections:
            existing = conn_existing.get(conn_name, 0.0)
            if existing == 0.0:
                logger.warning(
                    "DC power flow: connection '%s' in group '%s' has zero existing capacity. "
                    "For DC power flow to work without big-M constraints, set a small existing "
                    "capacity (e.g. %.3f MW). The candidate_precapacity_to_avoid_big_m "
                    "parameter is %.3f MW.",
                    conn_name, group_name, precapacity, precapacity,
                )

    # --- Build canonical frames and place them on the Provider ---
    # node_dc_power_flow — single-column ``node``, sorted (matches the
    # legacy CSV emission order).
    node_dc_frame = pl.DataFrame(
        {"node": sorted(dc_pf_nodes)},
        schema={"node": pl.Utf8},
    )
    provider.put("input/node_dc_power_flow", node_dc_frame)

    # connection_dc_power_flow — single-column ``process``, sorted.
    # Note: the canonical CSV column is ``process`` (matches GMPL's
    # connection_dc_power_flow set declaration).
    conn_dc_frame = pl.DataFrame(
        {"process": sorted(dc_pf_connections)},
        schema={"process": pl.Utf8},
    )
    provider.put("input/connection_dc_power_flow", conn_dc_frame)

    # node_reference_angle — single-column ``node``, preserves
    # reference-selection order (manual entries first, BFS auto-selection
    # second, per the original logic).
    ref_angle_frame = pl.DataFrame(
        {"node": list(reference_nodes)},
        schema={"node": pl.Utf8},
    )
    provider.put("input/node_reference_angle", ref_angle_frame)

    # p_connection_susceptance — two columns ``process,
    # p_connection_susceptance``, sorted by process.  The legacy
    # ``open(..., "w").write(f"{conn},{susceptance_map[conn]}\n")`` emits
    # float values as their Python str repr; preserve byte-parity by
    # casting to Utf8 with the same str() coercion (avoids polars'
    # default float-formatting which adds a trailing decimal).
    sorted_conns = sorted(susceptance_map.keys())
    susceptance_frame = pl.DataFrame(
        {
            "process": sorted_conns,
            "p_connection_susceptance": [
                str(susceptance_map[c]) for c in sorted_conns
            ],
        },
        schema={
            "process": pl.Utf8,
            "p_connection_susceptance": pl.Utf8,
        },
    )
    provider.put("input/p_connection_susceptance", susceptance_frame)

    # derived/ct_method_overrides — cross-derivation carrier for
    # :func:`flextool.input_derivation._process_method.derive_process_method`.
    # Stored as a two-column (process, ct_method) frame so the Provider
    # holds it under the same in-memory contract as every other frame.
    overrides_frame = pl.DataFrame(
        {
            "process": list(ct_method_overrides.keys()),
            "ct_method": list(ct_method_overrides.values()),
        },
        schema={"process": pl.Utf8, "ct_method": pl.Utf8},
    )
    provider.put("derived/ct_method_overrides", overrides_frame)

    return ct_method_overrides


__all__ = ["derive_dc_power_flow"]
