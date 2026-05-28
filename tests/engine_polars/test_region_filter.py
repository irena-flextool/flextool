"""Tests for the regional filter (gap A4+A5).

Exercises ``flextool._region_filter.split`` on the LH2 three-region
fixture plus a few synthetic cases (single-region no-op, all-pairs
topology, missing-region defaults).  Asserts the output FlexDatas have:

* the right node / process membership per region;
* exactly the expected half-flow records, paired correctly;
* no original cross-region pipe rows leaking into a region's process
  frames;
* virtual half-flow connections present in pss_dt with the right
  capacity / unitsize / slope inheritance.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars._pdt_join import compute_pss_dt
from flextool.engine_polars._region_filter import (
    discover_regions,
    load_decomposition_method,
    load_region_membership,
    split,
)


@pytest.fixture(scope="module")
def lh2_workdir(scenario_workdir):
    return scenario_workdir("lh2_three_region", db_fixture="lh2")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_load_decomposition_method(self, lh2_workdir) -> None:
        meth = load_decomposition_method(lh2_workdir)
        assert meth == {
            "region_A": "lagrangian_region",
            "region_B": "lagrangian_region",
            "region_C": "lagrangian_region",
        }

    def test_discover_regions(self, lh2_workdir) -> None:
        regions = discover_regions(lh2_workdir)
        assert regions == ["region_A", "region_B", "region_C"]

    def test_load_decomposition_method_missing_file(self, tmp_path: Path) -> None:
        # Empty work_dir — no input/p_group_decomposition.csv.
        (tmp_path / "input").mkdir()
        meth = load_decomposition_method(tmp_path)
        assert meth == {}

    def test_region_membership_lh2(self, lh2_workdir) -> None:
        data = load_flextool(lh2_workdir)
        mem = load_region_membership(
            data, ["region_A", "region_B", "region_C"]
        )
        assert {"elec_A", "h2_A", "lh2_A", "battery_A"} <= mem["region_A"]["nodes"]
        assert "elec_B" not in mem["region_A"]["nodes"]
        assert {"wind_A", "coal_A", "liquefier_A"} <= mem["region_A"]["processes"]
        assert "wind_B" not in mem["region_A"]["processes"]


# ---------------------------------------------------------------------------
# LH2 fixture: three regions, four cross-region arcs (pipe_AB×2 + pipe_BC×2)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lh2_data(lh2_workdir):
    return load_flextool(lh2_workdir)


@pytest.fixture(scope="module")
def lh2_splits(lh2_data):
    return split(lh2_data, regions=["region_A", "region_B", "region_C"])


class TestLH2RegionA:
    def test_region_a_nodes(self, lh2_splits) -> None:
        sA = lh2_splits[0]
        nodes = set(sA.data.nodeBalance["n"].to_list())
        for required in ("elec_A", "h2_A", "lh2_A", "battery_A"):
            assert required in nodes
        # other regions' nodes excluded
        for forbidden in ("elec_B", "lh2_C", "h2_B"):
            assert forbidden not in nodes
        # No virtual nodes leak into the standard nodeBalance set —
        # half-flow virtual nodes sit OUTSIDE the balance equation.
        for n in nodes:
            assert not n.startswith("hf_")
            assert "__export__" not in n
            assert "__import__" not in n

    def test_region_a_processes(self, lh2_splits) -> None:
        sA = lh2_splits[0]
        procs = set(sA.data.process_source_sink["p"].unique().to_list())
        # local entities present
        assert "wind_A" in procs
        assert "coal_A" in procs
        assert "liquefier_A" in procs
        # other regions' processes excluded
        assert "wind_B" not in procs
        assert "coal_C" not in procs

    def test_region_a_cross_region_pipes_replaced(self, lh2_splits) -> None:
        sA = lh2_splits[0]
        # pipe_AB(lh2_A, lh2_B) original arc must NOT appear.
        cross = sA.data.process_source_sink.filter(
            (pl.col("p") == "pipe_AB")
            & (pl.col("source") == "lh2_A")
            & (pl.col("sink") == "lh2_B")
        )
        assert cross.height == 0
        # Virtual half-flow connection must appear — exactly one for the
        # export direction (A→B), one for the import direction (B→A).
        virtual_export = next(
            hf for hf in sA.half_flows
            if hf.original_p == "pipe_AB" and hf.side == "export"
        )
        virtual_import = next(
            hf for hf in sA.half_flows
            if hf.original_p == "pipe_AB" and hf.side == "import"
        )
        assert virtual_export.in_region_node == "lh2_A"
        assert virtual_import.in_region_node == "lh2_A"
        # Half-flow virtual connections present in pss_dt (computed on
        # demand via the helper — see Phase E.3).
        assert compute_pss_dt(sA.data).filter(
            pl.col("p") == virtual_export.virtual_p
        ).height == 168  # 168 timesteps in fixture

    def test_region_a_no_pipe_bc(self, lh2_splits) -> None:
        sA = lh2_splits[0]
        # pipe_BC has nothing to do with region_A.
        procs = set(sA.data.process_source_sink["p"].unique().to_list())
        assert "pipe_BC" not in procs
        # No half-flows for pipe_BC in region_A.
        for hf in sA.half_flows:
            assert hf.original_p != "pipe_BC"


class TestLH2RegionB:
    def test_region_b_has_four_half_flows(self, lh2_splits) -> None:
        sB = lh2_splits[1]
        # pipe_AB has two directions × pipe_BC has two directions, so B
        # touches all four cross-region arcs.
        assert len(sB.half_flows) == 4
        keys = {(hf.original_p, hf.side) for hf in sB.half_flows}
        assert keys == {
            ("pipe_AB", "import"),
            ("pipe_AB", "export"),
            ("pipe_BC", "export"),
            ("pipe_BC", "import"),
        }


class TestLH2RegionC:
    def test_region_c_pipe_bc_only(self, lh2_splits) -> None:
        sC = lh2_splits[2]
        for hf in sC.half_flows:
            assert hf.original_p == "pipe_BC"
        # One export (lh2_C → lh2_B back-flow) and one import
        # (lh2_B → lh2_C original direction).
        assert {hf.side for hf in sC.half_flows} == {"export", "import"}


class TestCouplingPairs:
    """Each cross-region (p, source, sink) arc has exactly one export
    half-flow and one import half-flow across the regions."""

    def test_each_arc_paired(self, lh2_splits) -> None:
        export_keys: dict[tuple, list[str]] = {}
        import_keys: dict[tuple, list[str]] = {}
        for s in lh2_splits:
            for hf in s.half_flows:
                k = (hf.original_p, hf.original_source, hf.original_sink)
                if hf.side == "export":
                    export_keys.setdefault(k, []).append(hf.region)
                else:
                    import_keys.setdefault(k, []).append(hf.region)
        # Same key set on both sides.
        assert set(export_keys.keys()) == set(import_keys.keys())
        for k in export_keys:
            assert len(export_keys[k]) == 1, f"{k}: multiple exports"
            assert len(import_keys[k]) == 1, f"{k}: multiple imports"
            # Different region on each side.
            assert export_keys[k][0] != import_keys[k][0]


class TestVirtualEntityWiring:
    def test_half_flow_capacities_inherit(self, lh2_splits, lh2_data) -> None:
        """Virtual half-flow's p_flow_upper rows match the original arc."""
        sA = lh2_splits[0]
        export_hf = next(
            hf for hf in sA.half_flows
            if hf.original_p == "pipe_AB" and hf.side == "export"
        )
        # Original p_flow_upper for pipe_AB
        orig = lh2_data.p_flow_upper.frame.filter(
            (pl.col("p") == "pipe_AB")
            & (pl.col("source") == "lh2_A")
            & (pl.col("sink") == "lh2_B")
        ).sort("d", "t")["value"].to_list()
        # Virtual p_flow_upper for the half-flow
        new = sA.data.p_flow_upper.frame.filter(
            pl.col("p") == export_hf.virtual_p
        ).sort("d", "t")["value"].to_list()
        assert new == orig

    def test_half_flow_unitsize_inherits(self, lh2_splits, lh2_data) -> None:
        sA = lh2_splits[0]
        export_hf = next(
            hf for hf in sA.half_flows
            if hf.original_p == "pipe_AB" and hf.side == "export"
        )
        orig_us = float(lh2_data.p_unitsize.frame.filter(
            pl.col("p") == "pipe_AB")["value"][0])
        new_us = float(sA.data.p_unitsize.frame.filter(
            pl.col("p") == export_hf.virtual_p)["value"][0])
        assert new_us == orig_us

    def test_export_halfflow_in_flow_from_n(self, lh2_splits) -> None:
        """Export half-flow must contribute to the in-region node's
        source-side balance (flow_from_n)."""
        sA = lh2_splits[0]
        export_hf = next(
            hf for hf in sA.half_flows
            if hf.original_p == "pipe_AB" and hf.side == "export"
        )
        rows = sA.data.flow_from_n.filter(
            (pl.col("p") == export_hf.virtual_p)
            & (pl.col("n") == "lh2_A")
        )
        assert rows.height == 1
        # And NOT in flow_to_n (the virtual node is not in nodeBalance).
        rows_to = sA.data.flow_to_n.filter(
            pl.col("p") == export_hf.virtual_p
        )
        assert rows_to.height == 0

    def test_import_halfflow_in_flow_to_n(self, lh2_splits) -> None:
        sB = lh2_splits[1]
        import_hf = next(
            hf for hf in sB.half_flows
            if hf.original_p == "pipe_AB" and hf.side == "import"
        )
        rows = sB.data.flow_to_n.filter(
            (pl.col("p") == import_hf.virtual_p)
            & (pl.col("n") == "lh2_B")
        )
        assert rows.height == 1
        rows_from = sB.data.flow_from_n.filter(
            pl.col("p") == import_hf.virtual_p
        )
        assert rows_from.height == 0


class TestSplitPreservesOriginalParityFields:
    """Filtering must not corrupt scalar/dt fields shared across regions."""

    def test_dt_unchanged(self, lh2_splits, lh2_data) -> None:
        for s in lh2_splits:
            assert s.data.dt.height == lh2_data.dt.height

    def test_step_duration_unchanged(self, lh2_splits, lh2_data) -> None:
        for s in lh2_splits:
            assert s.data.p_step_duration.frame.height \
                   == lh2_data.p_step_duration.frame.height


# ---------------------------------------------------------------------------
# Negative / edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_region_no_couplings(self, lh2_data) -> None:
        """When only one region is requested, the filter is a no-op
        for cross-region purposes — no half-flows are emitted."""
        splits = split(lh2_data, regions=["region_A"])
        assert len(splits) == 1
        assert splits[0].region == "region_A"
        # The "rest of the world" is treated as shared.  Cross-region
        # pipes WILL still be classified as cross-region (their endpoints
        # straddle region_A and a "no region" → no, the filter only flags
        # arcs where both endpoints are in some region's set).  With
        # only region_A in the membership map, lh2_B/lh2_C nodes go into
        # "shared" so pipe_AB / pipe_BC stay as ordinary (kept) arcs.
        assert splits[0].half_flows == []

    def test_empty_regions_returns_empty(self, lh2_data) -> None:
        assert split(lh2_data, regions=[]) == []

    def test_two_region_subset_yields_couplings(self, lh2_data) -> None:
        """Restricting the membership to A+B only — pipe_AB still
        crosses regions, but pipe_BC's lh2_C is now shared, so pipe_BC
        stays whole inside whichever region's frame it lives."""
        # When region C isn't in `regions`, lh2_C is considered shared.
        splits = split(lh2_data, regions=["region_A", "region_B"])
        # We should see pipe_AB couplings (both directions), not pipe_BC.
        flat_pipes = {(hf.original_p, hf.side) for s in splits
                      for hf in s.half_flows}
        assert ("pipe_AB", "export") in flat_pipes
        assert ("pipe_AB", "import") in flat_pipes
        # pipe_BC: source lh2_B is in region B; sink lh2_C is shared.
        # Per the classifier, "shared" doesn't classify the arc as
        # cross-region — so pipe_BC stays in B's frames intact.
        assert ("pipe_BC", "export") not in flat_pipes
        assert ("pipe_BC", "import") not in flat_pipes
