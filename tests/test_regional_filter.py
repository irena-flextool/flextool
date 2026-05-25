"""Tests for the regional filter in ``input_writer`` / ``region_filter``
(Agent 3.1).

The filter reads the full ``input/`` directory produced by
:func:`write_input` and copies it into ``input_region_<region>/``, keeping
only entities that belong to the region or are shared (not assigned to
any decomposition region), and replacing cross-region process_connection
entities with import/export half-flows whose flow is the Lagrangian
coupling variable.

These tests use the LH2 three-region fixture:

* region_A has nodes elec_A, h2_A, lh2_A, battery_A
* region_B has nodes elec_B, h2_B, lh2_B, battery_B
* region_C has nodes elec_C, h2_C, lh2_C, battery_C
* coal_market is a shared commodity node (not in any region)
* pipe_AB: lh2_A → lh2_B (source side: region_A, sink side: region_B)
* pipe_BC: lh2_B → lh2_C (source side: region_B, sink side: region_C)

So for the filter:

* region_A: pipe_AB (in-region source) → EXPORT half-flow
* region_B: pipe_AB (in-region sink)   → IMPORT half-flow
            pipe_BC (in-region source) → EXPORT half-flow
* region_C: pipe_BC (in-region sink)   → IMPORT half-flow
"""
from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))
if str(TEST_DIR / "fixtures") not in sys.path:
    sys.path.insert(0, str(TEST_DIR / "fixtures"))

from build_lh2_three_region import SCENARIO  # noqa: E402
from db_utils import json_to_db  # noqa: E402

from flextool.input_derivation import run as _input_derivation_run  # noqa: E402
from flextool.decomposition.region_decomposition import (  # noqa: E402
    write_input_for_region,
)
from flextool.decomposition.region_filter import (  # noqa: E402
    HalfFlow,
    build_region_provider,
    classify_half_flows_from_provider,
    discover_decomposition_regions_from_db,
    discover_region_membership_from_provider,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lh2_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    db_path = tmp_path_factory.mktemp("lh2_db") / "lh2_three_region.sqlite"
    return json_to_db(FIXTURES_DIR / "lh2_three_region.json", db_path)


@pytest.fixture(scope="module")
def staged_provider(
    lh2_db_url: str, tmp_path_factory: pytest.TempPathFactory
):
    """Produce the cascade-input :class:`FlexDataProvider` once per module.

    Step 2.6 — region decomposition is fully in-memory.  The Provider
    carries every ``input/<name>`` frame; per-region derivations are
    Provider-in / Provider-out via
    :func:`region_filter.build_region_provider`.  No disk staging.
    """
    workdir = tmp_path_factory.mktemp("lh2_stage")
    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        from flextool.engine_polars._flex_data_provider import (
            FlexDataProvider,
        )
        provider = FlexDataProvider()
        _input_derivation_run(
            lh2_db_url,
            provider,
            logging.getLogger("test_regional_filter"),
            scenario_name=SCENARIO,
            work_folder=workdir,
        )
    finally:
        os.chdir(prev_cwd)
    assert provider.has("input/node"), (
        "cascade-input provider missing input/node — input_derivation regression"
    )
    return provider


def _region_dir_from_provider(
    staged_provider, region: str, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict]:
    """Build the region-scoped Provider in-memory, then snapshot it to a
    temp dir so the assertions below can read ``<region>/<file>.csv``
    unchanged.  This mirrors the CLI deliverable contract enforced by
    :func:`write_input_for_region`.
    """
    region_provider, result = build_region_provider(
        staged_provider,
        region=region,
        all_regions=["region_A", "region_B", "region_C"],
    )
    out = tmp_path_factory.mktemp(f"input_region_{region}")
    # Strip ``input/`` prefix so the snapshot lays files at <out>/<f>.csv
    # to match the historical disk layout.
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    flat = FlexDataProvider()
    for key, frame in region_provider.items():
        if key.startswith("input/"):
            flat.put(key.split("/", 1)[1], frame)
        else:
            flat.put(key, frame)
    flat.snapshot_processed_inputs(out)
    return out, result


@pytest.fixture(scope="module")
def region_a_dir(
    staged_provider, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict]:
    return _region_dir_from_provider(staged_provider, "region_A", tmp_path_factory)


@pytest.fixture(scope="module")
def region_b_dir(
    staged_provider, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict]:
    return _region_dir_from_provider(staged_provider, "region_B", tmp_path_factory)


@pytest.fixture(scope="module")
def region_c_dir(
    staged_provider, tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, dict]:
    return _region_dir_from_provider(staged_provider, "region_C", tmp_path_factory)


# ---------------------------------------------------------------------------
# Helper readers
# ---------------------------------------------------------------------------


def _read_rows(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [r for r in csv.reader(fh) if r]


def _first_col_set(path: Path) -> set[str]:
    rows = _read_rows(path)
    if not rows:
        return set()
    # Skip header.
    return {r[0] for r in rows[1:] if r}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegionDiscovery:
    def test_decomposition_regions_from_db(self, lh2_db_url: str) -> None:
        regions = set(discover_decomposition_regions_from_db(lh2_db_url))
        assert regions == {"region_A", "region_B", "region_C"}

    def test_region_membership_parses(self, staged_provider) -> None:
        mem = discover_region_membership_from_provider(
            staged_provider, "region_A",
        )
        assert {"elec_A", "h2_A", "lh2_A", "battery_A"} <= mem.nodes
        # Other regions' nodes should not be in region_A's set.
        assert "elec_B" not in mem.nodes
        assert "elec_C" not in mem.nodes

    def test_region_provider_has_input_node(self, staged_provider) -> None:
        """Provider-level assertion (replaces the legacy
        ``(workdir/"input"/"node.csv").exists()`` disk check): the
        cascade-input Provider must carry ``input/node`` after
        :func:`input_derivation.run`."""
        assert staged_provider.has("input/node")
        df = staged_provider.get("input/node")
        assert df is not None and df.height > 0, (
            "input/node frame is empty — fixture build regression"
        )


class TestRegionAMembership:
    def test_region_a_nodes(
        self, region_a_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_a_dir
        nodes = _first_col_set(out / "node.csv")
        for required in ("elec_A", "h2_A", "lh2_A", "battery_A"):
            assert required in nodes, f"{required} missing from region_A node.csv"
        # shared commodity
        assert "coal_market" in nodes, "coal_market (shared) should be kept"
        # other regions' nodes must not appear
        for forbidden in ("elec_B", "h2_B", "lh2_B", "battery_B",
                          "elec_C", "h2_C", "lh2_C", "battery_C"):
            assert forbidden not in nodes, (
                f"{forbidden} should NOT appear in region_A node.csv"
            )

    def test_region_a_units(
        self, region_a_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_a_dir
        units = _first_col_set(out / "process_unit.csv")
        for required in (
            "wind_A", "coal_A", "battery_charge_A", "battery_discharge_A",
            "liquefier_A",
        ):
            assert required in units, f"{required} missing from region_A process_unit.csv"
        for forbidden in ("wind_B", "coal_B", "liquefier_B", "wind_C"):
            assert forbidden not in units

    def test_region_a_pipelines_replaced(
        self, region_a_dir: tuple[Path, dict]
    ) -> None:
        out, result = region_a_dir
        process_conn = _first_col_set(out / "process_connection.csv")
        # Original pipe_AB removed from region_A's input.
        assert "pipe_AB" not in process_conn, (
            "pipe_AB should be removed from region_A process_connection.csv "
            "(replaced by half-flow)"
        )
        # Virtual half-flow connection present.  Naming convention
        # (Agent 3.2): virtual CONNECTION has an ``hf_`` prefix so it
        # is distinct from the virtual NODE (which keeps the
        # ``<pipe>__<side>__<region>`` stem) — entity.csv carries
        # both, and GMPL flags a duplicate tuple otherwise.
        virtual_node = "pipe_AB__export__region_A"
        virtual_connection = "hf_pipe_AB__export__region_A"
        assert virtual_connection in process_conn, (
            f"{virtual_connection!r} (export half-flow) should be in "
            f"region_A process_connection.csv"
        )
        # Virtual node present in node.csv.
        nodes = _first_col_set(out / "node.csv")
        assert virtual_node in nodes, (
            f"virtual export node {virtual_node!r} should appear in "
            f"region_A node.csv"
        )
        # pipe_BC is not in region_A at all.
        assert "pipe_BC" not in process_conn
        # HalfFlow record matches.
        hfs = result["half_flows"]
        assert len(hfs) == 1, f"expected 1 half-flow for region_A, got {len(hfs)}"
        hf = hfs[0]
        assert hf.original_connection == "pipe_AB"
        assert hf.side == "export"
        assert hf.in_region_node == "lh2_A"
        assert hf.virtual_node == virtual_node
        assert hf.virtual_connection == virtual_connection

    def test_region_a_process_source_sink(
        self, region_a_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_a_dir
        # process__source.csv contains (process, source)
        src_rows = _read_rows(out / "process__source.csv")[1:]
        src_map = {r[0]: r[1] for r in src_rows if len(r) >= 2}
        # The virtual export connection (hf_-prefixed) has source = lh2_A
        # (the in-region endpoint) and sink = the virtual node (un-prefixed).
        virtual_node = "pipe_AB__export__region_A"
        virtual_connection = "hf_pipe_AB__export__region_A"
        assert src_map.get(virtual_connection) == "lh2_A", (
            f"expected {virtual_connection} source to be lh2_A, "
            f"got {src_map.get(virtual_connection)!r}"
        )
        snk_rows = _read_rows(out / "process__sink.csv")[1:]
        snk_map = {r[0]: r[1] for r in snk_rows if len(r) >= 2}
        assert snk_map.get(virtual_connection) == virtual_node, (
            f"expected {virtual_connection} sink to be {virtual_node}"
        )


class TestRegionBHasImportAndExport:
    def test_region_b_two_half_flows(
        self, region_b_dir: tuple[Path, dict]
    ) -> None:
        _, result = region_b_dir
        hfs: list[HalfFlow] = result["half_flows"]
        assert len(hfs) == 2, (
            f"expected 2 half-flows for region_B (pipe_AB + pipe_BC), "
            f"got {len(hfs)}: {[(h.original_connection, h.side) for h in hfs]}"
        )
        by_conn = {hf.original_connection: hf for hf in hfs}
        # pipe_AB flows INTO B (B is sink) → import
        assert "pipe_AB" in by_conn
        assert by_conn["pipe_AB"].side == "import"
        assert by_conn["pipe_AB"].in_region_node == "lh2_B"
        # pipe_BC flows OUT OF B (B is source) → export
        assert "pipe_BC" in by_conn
        assert by_conn["pipe_BC"].side == "export"
        assert by_conn["pipe_BC"].in_region_node == "lh2_B"

    def test_region_b_virtual_nodes_present(
        self, region_b_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_b_dir
        nodes = _first_col_set(out / "node.csv")
        assert "pipe_AB__import__region_B" in nodes
        assert "pipe_BC__export__region_B" in nodes
        # Original cross-region pipe nodes that belong to *other* regions
        # must not bleed through.
        assert "lh2_A" not in nodes
        assert "lh2_C" not in nodes

    def test_region_b_process_connection(
        self, region_b_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_b_dir
        pc = _first_col_set(out / "process_connection.csv")
        # Originals removed, virtuals present.  Virtual CONNECTIONs use
        # the ``hf_`` prefix (Agent 3.2 naming fix) so they're distinct
        # from virtual NODEs in entity.csv.
        assert "pipe_AB" not in pc
        assert "pipe_BC" not in pc
        assert "hf_pipe_AB__import__region_B" in pc
        assert "hf_pipe_BC__export__region_B" in pc

    def test_region_b_source_sink_wiring(
        self, region_b_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_b_dir
        src_rows = _read_rows(out / "process__source.csv")[1:]
        src_map = {r[0]: r[1] for r in src_rows if len(r) >= 2}
        snk_rows = _read_rows(out / "process__sink.csv")[1:]
        snk_map = {r[0]: r[1] for r in snk_rows if len(r) >= 2}
        # Import half-flow: virtual CONNECTION is ``hf_<pipe>__import__
        # <region>``; source = virtual NODE (un-prefixed stem), sink =
        # in-region node.
        v_in_conn = "hf_pipe_AB__import__region_B"
        v_in_node = "pipe_AB__import__region_B"
        assert src_map.get(v_in_conn) == v_in_node
        assert snk_map.get(v_in_conn) == "lh2_B"
        # Export half-flow: source = in-region node, sink = virtual node
        v_out_conn = "hf_pipe_BC__export__region_B"
        v_out_node = "pipe_BC__export__region_B"
        assert src_map.get(v_out_conn) == "lh2_B"
        assert snk_map.get(v_out_conn) == v_out_node


class TestRegionCMembership:
    def test_region_c_import_only(
        self, region_c_dir: tuple[Path, dict]
    ) -> None:
        _, result = region_c_dir
        hfs: list[HalfFlow] = result["half_flows"]
        assert len(hfs) == 1
        hf = hfs[0]
        assert hf.original_connection == "pipe_BC"
        assert hf.side == "import"
        assert hf.in_region_node == "lh2_C"

    def test_region_c_no_pipes(
        self, region_c_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_c_dir
        pc = _first_col_set(out / "process_connection.csv")
        assert "pipe_AB" not in pc
        assert "pipe_BC" not in pc


class TestRegionCouplingManifest:
    def test_manifest_present_after_write_input_for_region(
        self, lh2_db_url: str, tmp_path: Path
    ) -> None:
        """Invoking ``write_input_for_region`` writes
        ``solve_data/region_coupling.csv`` with the right rows.
        """
        prev_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = write_input_for_region(
                input_db_url=lh2_db_url,
                scenario_name=SCENARIO,
                logger=logging.getLogger("test_region_manifest"),
                region_group="region_B",
                output_dir=tmp_path / "input_region_region_B",
                work_folder=tmp_path,
            )
        finally:
            os.chdir(prev_cwd)
        manifest = tmp_path / "solve_data" / "region_coupling.csv"
        assert manifest.exists(), "region_coupling.csv was not written"
        rows = _read_rows(manifest)
        assert rows[0] == ["region", "process", "side", "virtual_node"]
        data = rows[1:]
        # Two coupling variables for region_B.
        assert len(data) == 2
        recs = {(r[0], r[1], r[2], r[3]) for r in data if len(r) >= 4}
        assert (
            "region_B", "pipe_AB", "import", "pipe_AB__import__region_B",
        ) in recs
        assert (
            "region_B", "pipe_BC", "export", "pipe_BC__export__region_B",
        ) in recs
        # Double-check the result dict.
        assert len(result["half_flows"]) == 2


class TestFilterPreservesSharedEntities:
    def test_coal_market_and_coal_commodity_kept(
        self, region_a_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_a_dir
        nodes = _first_col_set(out / "node.csv")
        assert "coal_market" in nodes, (
            "shared commodity node coal_market must be kept in every region"
        )
        commodities = _first_col_set(out / "commodity.csv")
        assert "coal" in commodities, "coal commodity must be kept"

    def test_profiles_kept_globally(
        self, region_a_dir: tuple[Path, dict]
    ) -> None:
        out, _ = region_a_dir
        profiles = _first_col_set(out / "profile.csv")
        # Wind profiles for all regions live in the profile namespace;
        # the filter does not prune profiles by region (profiles are
        # shared identifier objects, not spatial).  We only check that
        # at least the in-region wind profile is present.
        assert "wind_profile_A" in profiles
