"""Storage-node output heuristic — a storage node is surfaced in the node
balance/price outputs only when it is wired to more than
``MULTI_USE_STORAGE_PROCESS_THRESHOLD`` distinct processes.

Batteries (one inverter, or separate charge + discharge units ⇒ ≤2 processes)
stay hidden; a multi-use hub (e.g. a hydrogen node feeding an electrolyser, a
turbine and several conversion routes) is surfaced automatically.  The helper
is a pure, arc-level function of ``s.node_state`` / ``s.process_source`` /
``s.process_sink`` — no solve required.
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from flextool.process_outputs.out_node import (
    MULTI_USE_STORAGE_PROCESS_THRESHOLD,
    _multi_use_storage_nodes,
)


def _s(node_state, source_arcs, sink_arcs):
    """Build a minimal ``s`` namespace for the helper.

    ``source_arcs`` / ``sink_arcs`` are lists of ``(process, node)`` tuples.
    """
    return SimpleNamespace(
        node_state=pd.Index(node_state, name="node"),
        process_source=pd.MultiIndex.from_tuples(
            source_arcs, names=["process", "source"],
        ) if source_arcs else pd.MultiIndex.from_tuples(
            [], names=["process", "source"],
        ),
        process_sink=pd.MultiIndex.from_tuples(
            sink_arcs, names=["process", "sink"],
        ) if sink_arcs else pd.MultiIndex.from_tuples(
            [], names=["process", "sink"],
        ),
    )


def test_threshold_is_two():
    """Guard the constant: separate charge+discharge units (=2) must hide."""
    assert MULTI_USE_STORAGE_PROCESS_THRESHOLD == 2


def test_single_inverter_battery_hidden():
    # battery: one bidirectional inverter, present on both source and sink.
    s = _s(
        node_state=["battery"],
        source_arcs=[("inverter", "battery")],
        sink_arcs=[("inverter", "battery")],
    )
    assert list(_multi_use_storage_nodes(s)) == []


def test_split_charge_discharge_battery_hidden():
    # battery with separate charger + discharger = 2 distinct processes ⇒ hidden.
    s = _s(
        node_state=["battery"],
        source_arcs=[("discharger", "battery")],
        sink_arcs=[("charger", "battery")],
    )
    assert list(_multi_use_storage_nodes(s)) == []


def test_multi_use_hub_surfaced():
    # H2 hub: electrolyser (in) + turbine, pipe, fuelcell (out) = 4 processes.
    s = _s(
        node_state=["h2"],
        source_arcs=[("turbine", "h2"), ("pipe", "h2"), ("fuelcell", "h2")],
        sink_arcs=[("electrolyser", "h2")],
    )
    assert list(_multi_use_storage_nodes(s)) == ["h2"]


def test_three_processes_surfaced_boundary():
    # exactly 3 distinct processes (>2) ⇒ surfaced (liquefier + two pipes).
    s = _s(
        node_state=["lh2_B"],
        source_arcs=[("pipe_1", "lh2_B"), ("pipe_2", "lh2_B")],
        sink_arcs=[("liquefier", "lh2_B")],
    )
    assert list(_multi_use_storage_nodes(s)) == ["lh2_B"]


def test_bidirectional_process_counted_once():
    # a process on both source and sink sides counts once, not twice — so a
    # node with one bidirectional + one other process = 2 ⇒ still hidden.
    s = _s(
        node_state=["nd"],
        source_arcs=[("bidir", "nd"), ("other", "nd")],
        sink_arcs=[("bidir", "nd")],
    )
    assert list(_multi_use_storage_nodes(s)) == []


def test_only_storage_nodes_considered():
    # a plain balance node with many processes is not storage ⇒ never returned.
    s = _s(
        node_state=["battery"],
        source_arcs=[("g1", "elec"), ("g2", "elec"), ("g3", "elec"),
                     ("inverter", "battery")],
        sink_arcs=[("load", "elec"), ("inverter", "battery")],
    )
    assert list(_multi_use_storage_nodes(s)) == []


def test_mixed_batteries_and_hub():
    s = _s(
        node_state=["battery", "h2"],
        source_arcs=[("inverter", "battery"),
                     ("turbine", "h2"), ("pipe", "h2"), ("fuelcell", "h2")],
        sink_arcs=[("inverter", "battery"), ("electrolyser", "h2")],
    )
    assert list(_multi_use_storage_nodes(s)) == ["h2"]


def test_no_storage_nodes():
    s = _s(node_state=[], source_arcs=[], sink_arcs=[])
    assert list(_multi_use_storage_nodes(s)) == []


def test_storage_node_with_no_arcs_hidden():
    s = _s(node_state=["orphan"], source_arcs=[], sink_arcs=[])
    assert list(_multi_use_storage_nodes(s)) == []


@pytest.mark.parametrize("n_processes,expected_surfaced", [
    (1, False), (2, False), (3, True), (4, True), (10, True),
])
def test_threshold_boundary_parametrized(n_processes, expected_surfaced):
    s = _s(
        node_state=["nd"],
        source_arcs=[(f"p{i}", "nd") for i in range(n_processes)],
        sink_arcs=[],
    )
    surfaced = list(_multi_use_storage_nodes(s)) == ["nd"]
    assert surfaced == expected_surfaced
