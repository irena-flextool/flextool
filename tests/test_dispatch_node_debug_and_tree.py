"""Stage 4.4 — node-level dispatch handling.

Covers two behaviours:

1. ``build_dispatch_tree`` / ``available_dispatch_nodes`` build the
   nodeGroup→nodes hierarchy (incl. the "Ungrouped nodes" catch-all) used
   by the viewer dispatch tree.
2. ``create_dispatch_plots`` generates per-node dispatch plots ONLY when
   ``debug=True`` (enumerating ALL available nodes from the data), and never
   consumes the legacy ``config['nodes']`` curation list.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from flextool.scenario_comparison.data_models import (
    DispatchMappings,
    TimeSeriesResults,
)
from flextool.scenario_comparison.dispatch_data import (
    available_dispatch_nodes,
    build_dispatch_tree,
)


def _results_with_nodes(nodes: list[str]) -> TimeSeriesResults:
    """A TimeSeriesResults whose ``node_d_ep`` exposes *nodes*.

    Mirrors the real (scenario, node, measure) MultiIndex column layout so
    ``available_dispatch_nodes`` reads the ``node`` level the same way
    ``config_builder``'s ``available_nodes`` discovery did.
    """
    cols = pd.MultiIndex.from_tuples(
        [("scen", n, "From units") for n in nodes],
        names=["scenario", "node", "measure"],
    )
    df = pd.DataFrame([[0.0] * len(nodes)], columns=cols)
    res = TimeSeriesResults()
    res.node_d_ep = df
    return res


class TestAvailableDispatchNodes(unittest.TestCase):
    def test_reads_node_level(self) -> None:
        res = _results_with_nodes(["n3", "n1", "n2"])
        self.assertEqual(available_dispatch_nodes(res), ["n1", "n2", "n3"])

    def test_none_node_d_ep(self) -> None:
        self.assertEqual(available_dispatch_nodes(TimeSeriesResults()), [])


class TestBuildDispatchTree(unittest.TestCase):
    def test_hierarchy_with_ungrouped(self) -> None:
        gn = pd.DataFrame(
            {"group": ["G1", "G1", "G2"], "node": ["n1", "n2", "n3"]}
        )
        tree = build_dispatch_tree(gn, ["G1", "G2"], ["n1", "n2", "n3", "n4"])
        self.assertEqual(
            tree,
            [
                ("G1", ["n1", "n2"]),
                ("G2", ["n3"]),
                ("Ungrouped nodes", ["n4"]),
            ],
        )

    def test_node_in_multiple_groups(self) -> None:
        gn = pd.DataFrame({"group": ["G1", "G2"], "node": ["n1", "n1"]})
        tree = build_dispatch_tree(gn, ["G1", "G2"], ["n1", "n5"])
        self.assertEqual(
            tree,
            [("G1", ["n1"]), ("G2", ["n1"]), ("Ungrouped nodes", ["n5"])],
        )

    def test_no_ungrouped_when_all_grouped(self) -> None:
        gn = pd.DataFrame({"group": ["G1", "G1"], "node": ["n1", "n2"]})
        tree = build_dispatch_tree(gn, ["G1"], ["n1", "n2"])
        self.assertEqual(tree, [("G1", ["n1", "n2"])])

    def test_all_ungrouped_when_no_groups(self) -> None:
        tree = build_dispatch_tree(None, [], ["n1", "n2"])
        self.assertEqual(tree, [("Ungrouped nodes", ["n1", "n2"])])

    def test_membership_restricted_to_available(self) -> None:
        # ``n_missing`` is a member of G1 but has no dispatch data; it must
        # NOT appear as a child and must NOT count as "grouped".
        gn = pd.DataFrame(
            {"group": ["G1", "G1"], "node": ["n1", "n_missing"]}
        )
        tree = build_dispatch_tree(gn, ["G1"], ["n1", "n2"])
        self.assertEqual(
            tree, [("G1", ["n1"]), ("Ungrouped nodes", ["n2"])]
        )

    def test_node_iid_distinct_from_group_iid(self) -> None:
        # The viewer derives child iids ``dispatchnode_<node>`` and parent
        # iids ``dispatch_<group>``.  The two prefixes are DISJOINT so a
        # nodeGroup and a node of the same bare name never collide, and
        # ``_is_dispatch_iid`` recognises either.
        from flextool.gui.result_viewer import (
            _DISPATCH_GROUP_PREFIX,
            _DISPATCH_NODE_PREFIX,
            _is_dispatch_iid,
        )

        group_iid = f"{_DISPATCH_GROUP_PREFIX}G1"
        node_iid = f"{_DISPATCH_NODE_PREFIX}G1"
        self.assertNotEqual(group_iid, node_iid)
        # Disjoint: neither prefix is a prefix of the other's iid.
        self.assertFalse(node_iid.startswith(_DISPATCH_GROUP_PREFIX))
        self.assertFalse(group_iid.startswith(_DISPATCH_NODE_PREFIX))
        # Both are recognised as dispatch iids.
        self.assertTrue(_is_dispatch_iid(group_iid))
        self.assertTrue(_is_dispatch_iid(node_iid))
        # The node-vs-group decision keys off the node prefix.
        self.assertTrue(node_iid.startswith(_DISPATCH_NODE_PREFIX))


class TestCreateDispatchPlotsDebugGating(unittest.TestCase):
    """``create_dispatch_plots`` per-node behaviour vs the ``debug`` flag."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.plot_dir = self.tmp / "output_plot_comparisons"
        self.plot_dir.mkdir()
        # A nodeGroup with data so the group path is exercised regardless.
        self.mappings = DispatchMappings()
        self.mappings.dispatch_groups = pd.DataFrame({"group": ["G1"]})
        self.results = _results_with_nodes(["n1", "n2"])

    def _run(self, debug: bool):
        """Run create_dispatch_plots with rendering + group prep stubbed.

        Returns the set of node names passed to ``prepare_node_dispatch_data``.
        """
        from flextool.scenario_comparison import dispatch_plots

        node_calls: list[str] = []

        def _fake_node(results, scenario, node):
            node_calls.append(node)
            return None, None

        def _fake_group(results, mappings, scenario, ng, **kwargs):
            return None, None

        with (
            patch.object(
                dispatch_plots, "prepare_node_dispatch_data",
                side_effect=_fake_node,
            ),
            patch.object(
                dispatch_plots, "prepare_dispatch_data",
                side_effect=_fake_group,
            ),
            patch.object(dispatch_plots, "plot_dispatch_area"),
        ):
            dispatch_plots.create_dispatch_plots(
                self.results,
                self.mappings,
                self.plot_dir,
                scenarios=["scen"],
                debug=debug,
            )
        return node_calls

    def test_no_node_plots_when_debug_off(self) -> None:
        node_calls = self._run(debug=False)
        self.assertEqual(node_calls, [])

    def test_all_nodes_when_debug_on(self) -> None:
        node_calls = self._run(debug=True)
        # Two passes (ylim + plot) over both discovered nodes; the SET of
        # nodes is what matters and must equal the data universe (there is no
        # longer any curated node selection to override it).
        self.assertEqual(set(node_calls), {"n1", "n2"})


if __name__ == "__main__":
    unittest.main()
