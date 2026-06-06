"""Integration test for the hierarchical dispatch tree (Stage 4.4).

Drives the real ``ResultViewer._populate_dispatch_tree`` method body against
on-disk parquet fixtures and a real ``ttk.Treeview``, using a lightweight
stand-in for the rest of the (heavy) viewer.  Verifies:

* each dispatchable nodeGroup becomes a parent (``dispatch_<ng>``);
* its member nodes become children (``dispatchnode_<node>``);
* a catch-all "Ungrouped nodes" parent holds nodes under no displayed group;
* node iids are distinct from group iids.

Must run under ``xvfb-run -a`` (creates a real Tk root).
"""

import tkinter as tk
from tkinter import ttk

import pandas as pd
import pytest

from flextool.gui.result_viewer import (
    _DISPATCH_GROUP_PREFIX,
    _DISPATCH_NODE_PREFIX,
    ResultViewer,
)
from flextool.scenario_comparison.data_models import TimeSeriesResults


def _write_parquet(path, df):
    df.to_parquet(path)


def _make_node_d_ep(nodes):
    cols = pd.MultiIndex.from_tuples(
        [("scen", n, "From units") for n in nodes],
        names=["scenario", "node", "measure"],
    )
    return pd.DataFrame([[0.0] * len(nodes)], columns=cols)


class _FakeViewer:
    """Minimal stand-in exposing what ``_populate_dispatch_tree`` touches."""

    def __init__(self, root, project_path, scenario, results):
        self._plot_tree = ttk.Treeview(root)
        self._tree_entry_map = {}
        self._project_path = project_path
        self._scenario = scenario
        self._dispatch_results = results
        self._dispatch_scenario = ""
        self._dispatch_node_iids = {}

    def _get_selected_scenarios(self):
        return [self._scenario]

    def _load_dispatch_data(self, scenario):
        # Results are pre-loaded in the fixture; node discovery reads
        # ``self._dispatch_results.node_d_ep`` directly.
        self._dispatch_scenario = scenario
        return True


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"No display available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def _children_of(tree, iid):
    return [tree.item(c, "text") for c in tree.get_children(iid)]


def test_hierarchical_tree_with_ungrouped(tk_root, tmp_path):
    scenario = "scenA"
    pdir = tmp_path / "output_parquet" / scenario
    pdir.mkdir(parents=True)

    # G1 + G2 flagged for dispatch; G3 has no members and must be dropped.
    _write_parquet(
        pdir / "nodeGroupDispatch.parquet",
        pd.DataFrame({"group": ["G1", "G2", "G3"]}),
    )
    _write_parquet(
        pdir / "group_node.parquet",
        pd.DataFrame(
            {"group": ["G1", "G1", "G2"], "node": ["n1", "n2", "n3"]}
        ),
    )

    # n4 belongs to no displayed group → "Ungrouped nodes".
    results = TimeSeriesResults()
    results.node_d_ep = _make_node_d_ep(["n1", "n2", "n3", "n4"])

    fake = _FakeViewer(tk_root, tmp_path, scenario, results)
    ResultViewer._populate_dispatch_tree(fake)

    tree = fake._plot_tree
    parents = list(tree.get_children(""))
    parent_labels = [tree.item(p, "text") for p in parents]

    assert parent_labels == ["G1", "G2", "Ungrouped nodes"]

    # Parent iids use the group prefix.
    assert parents == [
        f"{_DISPATCH_GROUP_PREFIX}G1",
        f"{_DISPATCH_GROUP_PREFIX}G2",
        f"{_DISPATCH_GROUP_PREFIX}Ungrouped nodes",
    ]

    # Children = member nodes (tree text), with node-prefixed iids.
    assert _children_of(tree, parents[0]) == ["n1", "n2"]
    assert _children_of(tree, parents[1]) == ["n3"]
    assert _children_of(tree, parents[2]) == ["n4"]

    g1_children = list(tree.get_children(parents[0]))
    assert all(c.startswith(_DISPATCH_NODE_PREFIX) for c in g1_children)
    # The iid→node map recovers the node name for the renderer.
    assert [fake._dispatch_node_iids[c] for c in g1_children] == ["n1", "n2"]

    # Node iids are distinct from group iids (disjoint prefixes).
    assert all(not c.startswith(_DISPATCH_NODE_PREFIX) for c in parents)
    assert all(
        not c.startswith(_DISPATCH_GROUP_PREFIX) for c in g1_children
    )

    # First parent auto-selected.
    assert tree.selection() == (f"{_DISPATCH_GROUP_PREFIX}G1",)


def test_node_in_multiple_groups(tk_root, tmp_path):
    scenario = "scenB"
    pdir = tmp_path / "output_parquet" / scenario
    pdir.mkdir(parents=True)

    _write_parquet(
        pdir / "nodeGroupDispatch.parquet",
        pd.DataFrame({"group": ["G1", "G2"]}),
    )
    _write_parquet(
        pdir / "group_node.parquet",
        pd.DataFrame({"group": ["G1", "G2"], "node": ["n1", "n1"]}),
    )

    results = TimeSeriesResults()
    results.node_d_ep = _make_node_d_ep(["n1"])

    fake = _FakeViewer(tk_root, tmp_path, scenario, results)
    ResultViewer._populate_dispatch_tree(fake)

    tree = fake._plot_tree
    parents = list(tree.get_children(""))
    # No "Ungrouped nodes" — n1 is grouped under both G1 and G2.
    assert [tree.item(p, "text") for p in parents] == ["G1", "G2"]
    assert _children_of(tree, parents[0]) == ["n1"]
    assert _children_of(tree, parents[1]) == ["n1"]

    # The same node under two groups gets two distinct (unique) child iids
    # that both map back to "n1".
    c0 = list(tree.get_children(parents[0]))
    c1 = list(tree.get_children(parents[1]))
    assert c0 != c1
    assert fake._dispatch_node_iids[c0[0]] == "n1"
    assert fake._dispatch_node_iids[c1[0]] == "n1"
