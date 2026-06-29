"""Regression tests for dispatch-plot sign / double-count bugs.

Three independent bugs in ``flextool/scenario_comparison/dispatch_data.py``
that surfaced on the H2_trade model (electrolysers + battery storage):

1. Group dispatch rendered consumers (electrolysers) as PRODUCERS — the
   not-in-aggregate node->unit path negated ``unit_inputNode_dt_ee``, which is
   already stored negated, flipping consumption to positive.
2. Group dispatch dropped ALL non-aggregated connections (batteries,
   transport links) when a model had no aggregated connections, because the
   connection frames were only loaded inside the ``processGroup_Connection``
   guard.
3. Once the connection frames load, the redundant ``not_in_aggregate_connection``
   "total" path double-counted every connection already drawn by the
   directional paths.

Plus the matching per-node bug: ``prepare_node_dispatch_data`` negated the
already-negated input and then clipped, silently dropping unit consumption.

The fixtures are built in-memory (no checked-in DB / no project output).
"""

import unittest

import pandas as pd

from flextool.scenario_comparison.data_models import (
    DispatchMappings,
    TimeSeriesResults,
)
from flextool.scenario_comparison.dispatch_data import (
    prepare_dispatch_data,
    prepare_node_dispatch_data,
)

SCEN = "s"
NODE = "elec_n"
GROUP = "eg"


def _dt_index() -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("p", "t0"), ("p", "t1")], names=["period", "time"]
    )


def _col(*entities: str) -> pd.MultiIndex:
    # (scenario, e0, e1) column with a 'scenario' level so _slice_scenario_df
    # can xs() it, matching the real combined-results layout.
    return pd.MultiIndex.from_tuples(
        [(SCEN, *entities)], names=["scenario", "e0", "e1"]
    )


def _frame(entities: tuple[str, str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {(SCEN, *entities): values}, index=_dt_index()
    ).set_axis(_col(*entities), axis=1)


def _build_results() -> TimeSeriesResults:
    res = TimeSeriesResults()
    # Producer (PV) -> elec node: positive output, also seeds _get_time_index.
    res.unit_outputNode_dt_ee = _frame(("pv", NODE), [5.0, 5.0])
    # Consumer (electrolyser) <- elec node: stored ALREADY NEGATED.
    res.unit_inputNode_dt_ee = _frame(("elc", NODE), [-2.0, -3.0])
    # Battery connection at the elec end: mixed sign (discharge +, charge -).
    res.connection_leftward_dt_eee = _frame(("bc", NODE), [1.0, -2.0])
    # Rightward is keyed on the battery end (not the elec node).
    res.connection_rightward_dt_eee = _frame(("bc", "batt_n"), [-0.9, 1.8])
    return res


def _scen_indexed(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["scenario"] = SCEN
    return df.set_index("scenario")


def _build_mappings() -> DispatchMappings:
    m = DispatchMappings()
    m.dispatch_groups = _scen_indexed([{"group": GROUP}])
    m.group_node = _scen_indexed([{"group": GROUP, "node": NODE}])
    # processGroup_Connection intentionally EMPTY (the bug-2 trigger).
    m.processGroup_Connection = None
    m.not_in_aggregate_unit_to_node = _scen_indexed(
        [{"group": GROUP, "process": "pv", "unit": "pv", "node": NODE}]
    )
    m.not_in_aggregate_node_to_unit = _scen_indexed(
        [{"group": GROUP, "process": "elc", "node": NODE, "unit": "elc"}]
    )
    m.not_in_aggregate_connection_to_node = _scen_indexed(
        [{"group": GROUP, "process": "bc", "connection": "bc", "node": NODE}]
    )
    m.not_in_aggregate_node_to_connection = _scen_indexed(
        [{"group": GROUP, "process": "bc", "node": NODE, "connection": "bc"}]
    )
    m.not_in_aggregate_connection = _scen_indexed(
        [{"group": GROUP, "connection": "bc"}]
    )
    return m


class TestGroupDispatchSigns(unittest.TestCase):
    def setUp(self) -> None:
        self.df, _ = prepare_dispatch_data(
            _build_results(), _build_mappings(), SCEN, GROUP
        )
        self.cols = [str(c) for c in self.df.columns]

    def test_consumer_is_negative_not_positive(self) -> None:
        # Bug 1: electrolyser consumption must stack DOWN (negative), never
        # appear as production.
        elc = [c for c in self.df.columns if "elc" in str(c)]
        self.assertTrue(elc, f"electrolyser column missing: {self.cols}")
        self.assertTrue(
            (self.df[elc[0]] <= 0).all(),
            f"consumer rendered positive: {self.df[elc[0]].tolist()}",
        )

    def test_producer_stays_positive(self) -> None:
        pv = [c for c in self.df.columns if "pv" in str(c)]
        self.assertTrue(pv)
        self.assertTrue((self.df[pv[0]] >= 0).all())

    def test_battery_connection_present_and_split(self) -> None:
        # Bug 2: the battery connection must appear at all; mixed sign so it
        # splits into discharge (_pos) and charge (_neg).
        batt = [c for c in self.cols if c.startswith("(bc, ")]
        self.assertTrue(
            any(c.endswith("_pos") for c in batt),
            f"battery discharge (_pos) missing: {self.cols}",
        )
        self.assertTrue(
            any(c.endswith("_neg") for c in batt),
            f"battery charge (_neg) missing: {self.cols}",
        )

    def test_no_connection_total_double_count(self) -> None:
        # Bug 3: the bare "(bc)" total column must NOT be emitted alongside the
        # directional "(bc, elec_n)" column.
        self.assertNotIn("(bc)", self.cols)
        self.assertFalse(
            any(c.startswith("(bc)") for c in self.cols),
            f"double-count total column present: {self.cols}",
        )

    def test_battery_energy_counted_once(self) -> None:
        # The connection's elec-side flow appears exactly once across the
        # split columns: pos sum + neg sum == raw leftward sum (1.0 + -2.0).
        batt = [c for c in self.df.columns if str(c).startswith("(bc, ")]
        total = sum(self.df[c].sum() for c in batt)
        self.assertAlmostEqual(total, -1.0, places=9)


class TestNodeDispatchConsumer(unittest.TestCase):
    def test_per_node_consumer_not_dropped(self) -> None:
        # Per-node bug: consumer must appear as a negative ``*_in`` column.
        df, _ = prepare_node_dispatch_data(_build_results(), SCEN, NODE)
        cols = [str(c) for c in df.columns]
        in_cols = [c for c in df.columns if str(c).endswith("_in")]
        self.assertTrue(in_cols, f"consumer dropped from per-node: {cols}")
        self.assertTrue((df[in_cols[0]] <= 0).all())


if __name__ == "__main__":
    unittest.main()
