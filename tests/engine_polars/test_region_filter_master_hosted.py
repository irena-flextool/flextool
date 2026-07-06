"""Master-hosted node mode for the regional splitter (Benders C6 + C7).

Master-hosted nodes are balance/state nodes in NO region group: they
live in the Benders MASTER (:func:`compute_master_hosted_nodes`), and
``split(..., master_hosted_nodes=...)`` re-partitions around them:

* master nodes are excluded from the shared-replicate set — regions
  never carry them;
* arcs classify 4-way (local / cross-region / region↔master /
  master-local): region↔master coupling arcs get exactly ONE half-flow
  on the region side, master-local arcs are dropped from every region
  and NOT half-flowed;
* processes ALL of whose arcs are master-local are subtracted from the
  regions' keep-sets — regions carry no invest/cost rows for them (F3);
* authored data that cannot be partitioned raises a HARD error, never
  a silent degrade (D-a): a unit straddling the boundary (aggregated
  per process across ALL its arcs) and a user constraint referencing
  both sides.

C7 extends ``master_network_data`` with the same mode: instead of
emptying every node-keyed frame it KEEPS the master-side model — the
master-hosted nodes' balance / inflow / penalties / storage / invest,
the master-local arcs (connections AND units) with their costs, and
the all-master user constraints / CO2 caps / feature groups (straddling
ones hard-error).  The build-through tests run ``build_flextool`` over
the reduced master FlexData and compare the emitted rows frame-for-frame
against the monolith build filtered to the master-side entities —
including a master-local INDIRECT unit whose post-cf08c082 ``maxFlow``
output-arc RHS must match (plan F4, the shipping-composition case).

With the default empty set both APIs are byte-identical to today.

Built on the lh2 fixture family (JSON-fixture DBs per CLAUDE.md
invariant #3): membership is manipulated in-memory by removing nodes
from their region group, which is exactly what the driver-side
membership rule produces for un-grouped nodes.
"""
from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from polar_high import Param, Problem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._pdt_join import compute_pss_dt
from flextool.engine_polars._region_filter import (
    _BENDERS_UNCAP_SENTINEL,
    _GROUP_FEATURE_PARAM_FIELDS,
    _GROUP_FEATURE_SET_FIELDS,
    _MASTER_CN_KEYED_FIELDS,
    _MASTER_CO2_ARC_FIELDS,
    _MASTER_CO2_GROUP_FIELDS,
    _MASTER_CO2_LOOKUP_FIELDS,
    _MASTER_GROUP_PROC_FIELDS,
    _MASTER_N_KEYED_FIELDS,
    _MASTER_NODE_FIELDS,
    compute_master_hosted_nodes,
    load_region_membership,
    master_network_data,
    split,
)


REGIONS = ["region_A", "region_B", "region_C"]

#: Big master set: both carrier chains of regions A and B move to the
#: master.  Under it (lh2 topology): the electrolyser connections
#: become region↔master coupling arcs; liquefier_A / liquefier_B
#: become master-local UNITS (every arc h2→lh2 is master-side);
#: pipe_AB becomes a master-local CONNECTION arc pair; pipe_BC becomes
#: a region↔master coupling pair with region_C.  No unit straddles.
MASTER_BIG = frozenset({"h2_A", "lh2_A", "h2_B", "lh2_B"})

#: Small master set: only region_B's carrier chain moves to the
#: master.  liquefier_B is master-local; electrolyser_B, pipe_AB and
#: pipe_BC are all region↔master coupling arcs.
MASTER_B = frozenset({"h2_B", "lh2_B"})

#: Every carrier chain moves to the master: all three liquefiers and
#: both pipes are master-local, all three electrolysers are coupling
#: arcs, and ``daily_group`` (the h2/lh2 stepduration group) becomes an
#: ALL-master group — the vehicle for the all-master group keep tests.
MASTER_ALL6 = frozenset(
    {"h2_A", "lh2_A", "h2_B", "lh2_B", "h2_C", "lh2_C"})


# ---------------------------------------------------------------------------
# Fixtures.  Both cascade workdirs are built BEFORE any load_flextool
# call (the ``_workdirs`` indirection): interleaving a load with a later
# same-process cascade build is a known pre-existing global-axis-enum
# hazard (documented at ``_region_filter.py`` "Base the widening ...").
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _workdirs(scenario_workdir):
    lh2 = scenario_workdir("lh2_three_region", db_fixture="lh2")
    ti = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_trade_invest"
    )
    mb = scenario_workdir(
        "lh2_three_region_trade_invest", db_fixture="lh2_master_build"
    )
    return lh2, ti, mb


@pytest.fixture(scope="module")
def lh2_data(_workdirs):
    return load_flextool(_workdirs[0])


@pytest.fixture(scope="module")
def ti_data(_workdirs):
    return load_flextool(_workdirs[1])


@pytest.fixture(scope="module")
def mb_data(_workdirs):
    """Build-through fixture data: trade-invest sibling with node invest
    on ``lh2_B``, unit invest on ``liquefier_B`` and a second
    ``liquefier_B`` output arc (⇒ INDIRECT nvar unit).  See
    ``lh2_master_build_db_url`` in ``tests/conftest.py``."""
    return load_flextool(_workdirs[2])


def _membership_without(data, master_nodes: frozenset[str]):
    """Region membership with *master_nodes* removed from their groups —
    what the driver's membership rule yields for un-grouped nodes."""
    mem = load_region_membership(data, REGIONS)
    for m in mem.values():
        m["nodes"] = m["nodes"] - set(master_nodes)
    return mem


def _copy_with(data, **field_overrides):
    """Shallow FlexData copy with field overrides; re-attaches the
    post-init ``_axis_enums`` attribute that ``dataclasses.replace``
    drops."""
    new = dataclasses.replace(data)
    for k, v in field_overrides.items():
        setattr(new, k, v)
    new._axis_enums = getattr(data, "_axis_enums", None)
    return new


def _iter_frames(data):
    """Yield ``(field_name, DataFrame)`` for every populated frame /
    Param frame on a FlexData."""
    for f in dataclasses.fields(data):
        v = getattr(data, f.name)
        if v is None:
            continue
        if isinstance(v, Param):
            yield f.name, v.frame
        elif isinstance(v, pl.DataFrame):
            yield f.name, v


def _col_values(frame: pl.DataFrame, col: str) -> set[str]:
    return set(frame[col].cast(pl.Utf8).to_list())


@pytest.fixture(scope="module")
def big_splits(lh2_data):
    return split(
        lh2_data,
        regions=REGIONS,
        region_membership=_membership_without(lh2_data, MASTER_BIG),
        master_hosted_nodes=MASTER_BIG,
    )


# ---------------------------------------------------------------------------
# compute_master_hosted_nodes
# ---------------------------------------------------------------------------


class TestComputeMasterHostedNodes:
    def test_full_membership_yields_empty(self, lh2_data) -> None:
        mem = load_region_membership(lh2_data, REGIONS)
        assert compute_master_hosted_nodes(lh2_data, mem) == set()

    def test_ungrouped_balance_state_nodes_detected(self, lh2_data) -> None:
        mem = _membership_without(lh2_data, MASTER_BIG)
        assert compute_master_hosted_nodes(lh2_data, mem) == set(MASTER_BIG)

    def test_no_balance_or_state_node_stays_shared(self, lh2_data) -> None:
        """coal_market has neither a balance nor a state row: it is in
        no region group yet must NOT become master-hosted (it keeps
        today's shared-replicate semantics)."""
        mem = _membership_without(lh2_data, MASTER_BIG)
        hosted = compute_master_hosted_nodes(lh2_data, mem)
        assert "coal_market" not in hosted


# ---------------------------------------------------------------------------
# Default parameter ⇒ byte-identical
# ---------------------------------------------------------------------------


def _canon(frame: pl.DataFrame) -> pl.DataFrame:
    """Canonical form for content comparison: Utf8-cast the categorical
    dims and sort by every column.  Needed because ``split`` output row
    order is not deterministic for a few Param frames even between two
    IDENTICAL calls (a polars join inside ``promote_param_to_dt`` does
    not maintain order) — a pre-existing property this test must not
    trip over; the mode must not change the CONTENT."""
    casts = [pl.col(c).cast(pl.Utf8) for c, dt in frame.schema.items()
             if isinstance(dt, (pl.Enum, pl.Categorical))]
    if casts:
        frame = frame.with_columns(casts)
    return frame.sort(frame.columns) if frame.width else frame


def _assert_flexdata_equal(a, b, region: str) -> None:
    for f in dataclasses.fields(a):
        va = getattr(a, f.name)
        vb = getattr(b, f.name)
        assert (va is None) == (vb is None), f"{region}: {f.name} None-ness"
        if va is None:
            continue
        if isinstance(va, Param):
            assert isinstance(vb, Param), f"{region}: {f.name} type"
            assert va.dims == vb.dims, f"{region}: {f.name} dims"
            assert _canon(va.frame).equals(_canon(vb.frame)), \
                f"{region}: {f.name} frame"
        elif isinstance(va, pl.DataFrame):
            assert _canon(va).equals(_canon(vb)), f"{region}: {f.name} frame"


def test_default_parameter_is_byte_identical(lh2_data) -> None:
    """``split`` WITHOUT the parameter equals ``split`` WITH the
    parameter present-but-empty, frame for frame."""
    base = split(lh2_data, regions=REGIONS)
    with_param = split(
        lh2_data, regions=REGIONS, master_hosted_nodes=frozenset()
    )
    assert len(base) == len(with_param)
    for sa, sb in zip(base, with_param):
        assert sa.region == sb.region
        assert sa.half_flows == sb.half_flows
        _assert_flexdata_equal(sa.data, sb.data, sa.region)


# ---------------------------------------------------------------------------
# Master-hosted split: region frames, half-flows, master-local drops
# ---------------------------------------------------------------------------


class TestMasterNodesExcludedFromRegions:
    def test_node_sets(self, big_splits) -> None:
        for s in big_splits:
            balance = set(s.data.nodeBalance["n"].cast(pl.Utf8).to_list())
            state = set(s.data.nodeState["n"].cast(pl.Utf8).to_list())
            assert not balance & MASTER_BIG, s.region
            assert not state & MASTER_BIG, s.region

    def test_no_master_node_in_any_n_keyed_frame(self, big_splits) -> None:
        for s in big_splits:
            for name, frame in _iter_frames(s.data):
                if "n" not in frame.columns:
                    continue
                leaked = _col_values(frame, "n") & MASTER_BIG
                assert not leaked, f"{s.region}: {name} carries {leaked}"


class TestRegionMasterCoupling:
    #: (original_p, original_source, original_sink) → (region, side,
    #: in_region_node) — the single region-side half-flow per arc.
    EXPECTED = {
        ("electrolyser_A", "elec_A", "h2_A"): ("region_A", "export", "elec_A"),
        ("electrolyser_A", "h2_A", "elec_A"): ("region_A", "import", "elec_A"),
        ("electrolyser_B", "elec_B", "h2_B"): ("region_B", "export", "elec_B"),
        ("electrolyser_B", "h2_B", "elec_B"): ("region_B", "import", "elec_B"),
        ("pipe_BC", "lh2_B", "lh2_C"): ("region_C", "import", "lh2_C"),
        ("pipe_BC", "lh2_C", "lh2_B"): ("region_C", "export", "lh2_C"),
    }

    def test_exactly_one_half_flow_per_coupling_arc(self, big_splits) -> None:
        seen: dict[tuple, list[tuple]] = {}
        for s in big_splits:
            for hf in s.half_flows:
                key = (hf.original_p, hf.original_source, hf.original_sink)
                seen.setdefault(key, []).append(
                    (s.region, hf.side, hf.in_region_node))
        assert set(seen) == set(self.EXPECTED)
        for key, occurrences in seen.items():
            assert len(occurrences) == 1, f"{key}: {occurrences}"
            assert occurrences[0] == self.EXPECTED[key]

    def test_naming_reuses_region_side_stem(self, big_splits) -> None:
        for s in big_splits:
            for hf in s.half_flows:
                stem = (f"{hf.original_p}__{hf.original_source}__"
                        f"{hf.original_sink}__{hf.side}__{hf.region}")
                assert hf.virtual_p == f"hf_{stem}"
                assert hf.virtual_node == stem

    def test_originals_dropped_and_virtuals_wired(self, big_splits) -> None:
        for s in big_splits:
            pss = s.data.process_source_sink
            for hf in s.half_flows:
                orig = pss.filter(
                    (pl.col("p").cast(pl.Utf8) == hf.original_p)
                    & (pl.col("source").cast(pl.Utf8) == hf.original_source)
                    & (pl.col("sink").cast(pl.Utf8) == hf.original_sink)
                )
                assert orig.height == 0, (s.region, hf.original_p)
                if hf.side == "export":
                    rows = s.data.flow_from_n.filter(
                        (pl.col("p").cast(pl.Utf8) == hf.virtual_p)
                        & (pl.col("n").cast(pl.Utf8) == hf.in_region_node)
                    )
                else:
                    rows = s.data.flow_to_n.filter(
                        (pl.col("p").cast(pl.Utf8) == hf.virtual_p)
                        & (pl.col("n").cast(pl.Utf8) == hf.in_region_node)
                    )
                assert rows.height == 1, (s.region, hf.virtual_p)

    def test_half_flow_on_full_dt_grid(self, big_splits) -> None:
        sC = big_splits[2]
        hf = next(h for h in sC.half_flows if h.side == "import")
        pss_dt = compute_pss_dt(sC.data)
        assert pss_dt.filter(
            pl.col("p").cast(pl.Utf8) == hf.virtual_p
        ).height == 168  # 168 timesteps in the fixture


class TestMasterLocalArcsAndProcs:
    MASTER_LOCAL_PROCS = {"pipe_AB", "liquefier_A", "liquefier_B"}

    def test_master_local_arcs_dropped_not_half_flowed(
        self, big_splits
    ) -> None:
        for s in big_splits:
            procs = _col_values(s.data.process_source_sink, "p")
            assert not procs & self.MASTER_LOCAL_PROCS, s.region
            for hf in s.half_flows:
                assert hf.original_p not in self.MASTER_LOCAL_PROCS

    def test_master_local_procs_in_no_region_frame(self, big_splits) -> None:
        """F3: regions carry NO rows for master-local procs on any
        process- or entity-keyed frame (arcs, unitsize, invest, cost)."""
        for s in big_splits:
            for name, frame in _iter_frames(s.data):
                for col in ("p", "e"):
                    if col not in frame.columns:
                        continue
                    leaked = (_col_values(frame, col)
                              & self.MASTER_LOCAL_PROCS)
                    assert not leaked, \
                        f"{s.region}: {name}[{col}] carries {leaked}"

    def test_master_local_unit_invest_rows_excluded(self, ti_data) -> None:
        """Trade-invest fixture: pipe_AB (master-local under MASTER_BIG)
        must contribute no invest rows to any region, while pipe_BC (a
        region↔master coupling arc) keeps today's shared-carry
        semantics."""
        splits = split(
            ti_data,
            regions=REGIONS,
            region_membership=_membership_without(ti_data, MASTER_BIG),
            master_hosted_nodes=MASTER_BIG,
        )
        for s in splits:
            for name, frame in (
                ("pd_invest_set", s.data.pd_invest_set),
                ("ed_invest_set", s.data.ed_invest_set),
                ("ed_entity_annual_discounted",
                 s.data.ed_entity_annual_discounted.frame),
            ):
                col = "p" if "p" in frame.columns else "e"
                vals = _col_values(frame, col)
                assert "pipe_AB" not in vals, f"{s.region}: {name}"
                assert "pipe_BC" in vals, f"{s.region}: {name}"


# ---------------------------------------------------------------------------
# Uncap sentinel on region↔master coupling arcs (D-e)
# ---------------------------------------------------------------------------


def _pipe_half_flows(splits):
    for s in splits:
        for hf in s.half_flows:
            if hf.original_p.startswith("pipe_"):
                yield s, hf


def _hf_existing_cap(data, virtual_p: str) -> list[float]:
    return (data.p_flow_upper_existing.frame
            .filter(pl.col("p").cast(pl.Utf8) == virtual_p)
            .sort("d")["value"].to_list())


class TestUncapSentinel:
    def _split_b(self, data, uncap: bool):
        return split(
            data,
            regions=REGIONS,
            region_membership=_membership_without(data, MASTER_B),
            master_hosted_nodes=MASTER_B,
            benders_uncap_cross_region=uncap,
        )

    def test_existing_capacity_coupling_gets_sentinel(self, lh2_data) -> None:
        """lh2 pipes carry authored existing capacity 1.0 — the region
        half-flow swaps it for the sentinel; the authored value stays
        available (untouched) for the master build."""
        splits = self._split_b(lh2_data, uncap=True)
        n = 0
        for s, hf in _pipe_half_flows(splits):
            caps = _hf_existing_cap(s.data, hf.virtual_p)
            assert caps, (s.region, hf.virtual_p)
            assert all(c == _BENDERS_UNCAP_SENTINEL for c in caps)
            n += 1
        assert n == 4  # pipe_AB ×2 in region_A, pipe_BC ×2 in region_C
        authored = (lh2_data.p_flow_upper_existing.frame
                    .filter(pl.col("p").cast(pl.Utf8).str.starts_with("pipe_"))
                    ["value"].to_list())
        assert authored == [1.0, 1.0, 1.0, 1.0]

    def test_greenfield_coupling_gets_sentinel(self, ti_data) -> None:
        splits = self._split_b(ti_data, uncap=True)
        n = 0
        for s, hf in _pipe_half_flows(splits):
            caps = _hf_existing_cap(s.data, hf.virtual_p)
            assert caps, (s.region, hf.virtual_p)
            assert all(c == _BENDERS_UNCAP_SENTINEL for c in caps)
            n += 1
        assert n == 4
        authored = (ti_data.p_flow_upper_existing.frame
                    .filter(pl.col("p").cast(pl.Utf8).str.starts_with("pipe_"))
                    ["value"].to_list())
        assert authored == [0.0, 0.0, 0.0, 0.0]

    def test_default_inherits_authored_capacity(
        self, lh2_data, ti_data
    ) -> None:
        for data, expected in ((lh2_data, 1.0), (ti_data, 0.0)):
            splits = self._split_b(data, uncap=False)
            for s, hf in _pipe_half_flows(splits):
                caps = _hf_existing_cap(s.data, hf.virtual_p)
                assert caps
                assert all(c == expected for c in caps)


# ---------------------------------------------------------------------------
# Hard validation: straddling units, mixed user constraints (D-a)
# ---------------------------------------------------------------------------


class TestStraddleValidation:
    def test_directly_straddling_unit_raises(self, lh2_data) -> None:
        """Master-hosting lh2_B alone leaves liquefier_B with an arc
        from a region node (h2_B) to a master node (lh2_B)."""
        with pytest.raises(
            RuntimeError,
            match=r"liquefier_B.*straddles(.|\n)*handover",
        ):
            split(
                lh2_data,
                regions=REGIONS,
                region_membership=_membership_without(
                    lh2_data, frozenset({"lh2_B"})),
                master_hosted_nodes=frozenset({"lh2_B"}),
            )

    def test_mixed_shape_unit_raises_via_entity_aggregation(
        self, lh2_data
    ) -> None:
        """A unit with one master-local arc plus one purely in-region
        arc has NO individually straddling arc yet straddles as an
        entity — the per-process aggregation must catch it."""
        pss = lh2_data.process_source_sink
        extra = pl.DataFrame(
            [{"p": "wind_C", "source": "lh2_A", "sink": "lh2_B"}],
            schema=dict(pss.schema),
        )
        data2 = _copy_with(
            lh2_data, process_source_sink=pl.concat([pss, extra])
        )
        # Precondition (the shape this test exists for): neither of
        # wind_C's arcs straddles on its own.
        assert not ({"wind_C", "elec_C"} & MASTER_BIG)      # in-region arc
        assert {"lh2_A", "lh2_B"} <= MASTER_BIG             # master-local arc
        with pytest.raises(
            RuntimeError,
            match=r"wind_C.*straddles(.|\n)*handover",
        ):
            split(
                data2,
                regions=REGIONS,
                region_membership=_membership_without(data2, MASTER_BIG),
                master_hosted_nodes=MASTER_BIG,
            )

    def test_mixed_user_constraint_raises(self, lh2_data) -> None:
        """A user constraint referencing a master-hosted node AND a
        region node cannot live whole on either side."""
        coeff = Param(
            ("n", "cn"),
            pl.DataFrame({
                "n": ["lh2_B", "elec_B"],
                "cn": ["mixed_cstr", "mixed_cstr"],
                "value": [1.0, 1.0],
            }),
            name="p_node_constraint_state_coeff",
        )
        data2 = _copy_with(
            lh2_data, p_node_constraint_state_coeff=coeff
        )
        with pytest.raises(
            RuntimeError,
            match=r"mixed_cstr.*master-side and region-side",
        ):
            split(
                data2,
                regions=REGIONS,
                region_membership=_membership_without(data2, MASTER_BIG),
                master_hosted_nodes=MASTER_BIG,
            )

    def test_master_node_in_region_membership_raises(self, lh2_data) -> None:
        """A node cannot be both master-hosted and region-grouped."""
        with pytest.raises(RuntimeError, match=r"overlap region membership"):
            split(
                lh2_data,
                regions=REGIONS,
                region_membership=load_region_membership(lh2_data, REGIONS),
                master_hosted_nodes=frozenset({"lh2_B"}),
            )


# ---------------------------------------------------------------------------
# C7 — master_network_data: empty-set byte-identity + master-side keeps
# ---------------------------------------------------------------------------


def _master(data, master_nodes: frozenset[str]):
    return master_network_data(
        data,
        REGIONS,
        region_membership=_membership_without(data, master_nodes),
        master_hosted_nodes=master_nodes,
    )


def test_master_node_fields_partition_is_complete() -> None:
    """Every ``_MASTER_NODE_FIELDS`` member is routed through exactly
    one master-mode keep-filter group — a schema addition to the legacy
    tuple cannot silently skip the master-hosted path."""
    groups = (
        _MASTER_N_KEYED_FIELDS,
        _MASTER_CN_KEYED_FIELDS,
        _MASTER_CO2_ARC_FIELDS,
        _MASTER_CO2_GROUP_FIELDS,
        _MASTER_CO2_LOOKUP_FIELDS,
        _MASTER_GROUP_PROC_FIELDS,
        _GROUP_FEATURE_SET_FIELDS,
        _GROUP_FEATURE_PARAM_FIELDS,
    )
    union: set[str] = set()
    total = 0
    for g in groups:
        union |= set(g)
        total += len(g)
    assert union == set(_MASTER_NODE_FIELDS), (
        f"unrouted: {set(_MASTER_NODE_FIELDS) - union}; "
        f"extra: {union - set(_MASTER_NODE_FIELDS)}"
    )
    assert total == len(set(_MASTER_NODE_FIELDS)), "groups overlap"


def test_master_network_data_empty_set_byte_identical(ti_data) -> None:
    """``master_network_data`` WITHOUT the parameter equals the call
    WITH the parameter present-but-empty, frame for frame."""
    base = master_network_data(ti_data, REGIONS)
    with_param = master_network_data(
        ti_data, REGIONS, master_hosted_nodes=frozenset()
    )
    _assert_flexdata_equal(base, with_param, "master")


class TestMasterNetworkDataContent:
    """Non-empty master set: the reduced data contains exactly the
    master-side model."""

    @pytest.fixture(scope="class")
    def md_big(self, lh2_data):
        return _master(lh2_data, MASTER_BIG)

    #: Arc keep-set under MASTER_BIG (lh2 topology): the region↔master
    #: coupling arcs ∪ the master-local arcs — and nothing else.
    EXPECTED_ARCS = {
        # coupling (single region-side endpoint)
        ("electrolyser_A", "elec_A", "h2_A"),
        ("electrolyser_A", "h2_A", "elec_A"),
        ("electrolyser_B", "elec_B", "h2_B"),
        ("electrolyser_B", "h2_B", "elec_B"),
        ("pipe_BC", "lh2_B", "lh2_C"),
        ("pipe_BC", "lh2_C", "lh2_B"),
        # master-local (no region endpoint)
        ("pipe_AB", "lh2_A", "lh2_B"),
        ("pipe_AB", "lh2_B", "lh2_A"),
        ("liquefier_A", "h2_A", "lh2_A"),
        ("liquefier_B", "h2_B", "lh2_B"),
    }

    def test_arc_keep_set(self, md_big) -> None:
        got = {
            (r["p"], r["source"], r["sink"])
            for r in md_big.process_source_sink.iter_rows(named=True)
        }
        assert got == self.EXPECTED_ARCS

    def test_balance_keeps_master_omits_region_endpoints(
        self, md_big
    ) -> None:
        nb = _col_values(md_big.nodeBalance, "n")
        # Master-hosted endpoints stay balanced.
        assert MASTER_BIG <= nb
        # Region-side endpoints of coupling arcs are omitted (elec_A /
        # elec_B for the electrolysers, lh2_C for pipe_BC).
        assert not nb & {"elec_A", "elec_B", "lh2_C"}
        # Non-terminal region nodes stay as structurally inert balance
        # rows (legacy behaviour, satisfies the build precondition).
        assert "battery_A" in nb and "elec_C" in nb

    def test_node_keyed_frames_filtered_to_master(self, md_big) -> None:
        for fld in _MASTER_N_KEYED_FIELDS:
            v = getattr(md_big, fld, None)
            if v is None:
                continue
            frame = v.frame if isinstance(v, Param) else v
            if "n" not in frame.columns:
                continue
            leaked = _col_values(frame, "n") - MASTER_BIG
            assert not leaked, f"{fld} carries non-master nodes {leaked}"

    def test_master_state_and_inflow_kept(self, md_big, lh2_data) -> None:
        assert _col_values(md_big.nodeState, "n") == {"lh2_A", "lh2_B"}
        inflow_nodes = _col_values(md_big.p_inflow.frame, "n")
        assert inflow_nodes == (
            _col_values(lh2_data.p_inflow.frame, "n") & MASTER_BIG
        )
        # Master rows byte-equal the source rows.
        src = lh2_data.p_inflow.frame.filter(
            pl.col("n").cast(pl.Utf8).is_in(sorted(MASTER_BIG)))
        assert _canon(md_big.p_inflow.frame).equals(_canon(src))

    def test_master_penalties_present(self, md_big) -> None:
        for p in (md_big.p_penalty_up, md_big.p_penalty_down):
            assert MASTER_BIG <= _col_values(p.frame, "n")

    def test_invest_frames_keep_coupling_and_master_procs(
        self, ti_data
    ) -> None:
        """pipe_AB (master-local) AND pipe_BC (coupling) invest lives in
        the master."""
        md = _master(ti_data, MASTER_BIG)
        assert _col_values(md.pd_invest_set, "p") == {"pipe_AB", "pipe_BC"}
        assert _col_values(md.ed_invest_set, "e") == {"pipe_AB", "pipe_BC"}
        assert _col_values(
            md.ed_entity_annual_discounted.frame, "e"
        ) == {"pipe_AB", "pipe_BC"}

    def test_group_membership_filtered_silently(self, md_big) -> None:
        """Bare ``group_node`` rows keep only master members (the
        straddling ``daily_group`` carries no feature ⇒ no raise)."""
        assert _col_values(md_big.group_node, "n") <= MASTER_BIG

    def test_coupling_arc_only_master_accepted(self, lh2_data) -> None:
        """MASTER_B yields ZERO region↔region cross arcs (pipe_AB and
        pipe_BC both become coupling arcs) — the coupling-arc-only case
        is load-bearing (audit 0.D) and must build, not raise."""
        md = _master(lh2_data, MASTER_B)
        got = {
            (r["p"], r["source"], r["sink"])
            for r in md.process_source_sink.iter_rows(named=True)
        }
        assert ("liquefier_B", "h2_B", "lh2_B") in got  # master-local
        assert ("pipe_AB", "lh2_A", "lh2_B") in got     # coupling
        assert ("liquefier_A", "h2_A", "lh2_A") not in got  # region-local


class TestMasterNetworkDataValidation:
    def test_overlap_raises(self, lh2_data) -> None:
        with pytest.raises(RuntimeError, match=r"overlap region membership"):
            master_network_data(
                lh2_data, REGIONS,
                region_membership=load_region_membership(lh2_data, REGIONS),
                master_hosted_nodes=frozenset({"lh2_B"}),
            )

    def test_mixed_user_constraint_raises(self, lh2_data) -> None:
        """The master path raises the SAME error as ``split`` for a
        constraint referencing both sides."""
        coeff = Param(
            ("n", "cn"),
            pl.DataFrame({
                "n": ["lh2_B", "elec_B"],
                "cn": ["mixed_cstr", "mixed_cstr"],
                "value": [1.0, 1.0],
            }),
            name="p_node_constraint_state_coeff",
        )
        data2 = _copy_with(
            lh2_data, p_node_constraint_state_coeff=coeff
        )
        with pytest.raises(
            RuntimeError,
            match=r"mixed_cstr.*master-side and region-side",
        ):
            _master(data2, MASTER_BIG)

    def test_straddling_feature_group_raises(self, lh2_data) -> None:
        """``daily_group`` straddles under MASTER_BIG (h2_C / lh2_C stay
        in region_C); giving it a feature must be a hard error."""
        data2 = _copy_with(
            lh2_data,
            groupCapacityMargin=pl.DataFrame({"g": ["daily_group"]}),
        )
        with pytest.raises(
            RuntimeError,
            match=r"daily_group.*straddles",
        ):
            _master(data2, MASTER_BIG)

    def test_co2_priced_coupling_arc_raises(self, lh2_data) -> None:
        """A CO2-priced flow on a region↔master coupling arc cannot be
        carried whole on either side."""
        data2 = _copy_with(
            lh2_data,
            flow_from_co2_priced=pl.DataFrame({
                "p": ["electrolyser_B"], "source": ["elec_B"],
                "sink": ["h2_B"], "c": ["co2"], "g": ["daily_group"],
            }),
        )
        with pytest.raises(RuntimeError, match=r"CO2-priced"):
            _master(data2, MASTER_BIG)

    def test_co2_capped_group_straddle_raises(self, lh2_data) -> None:
        """A capped group mixing a master-local flow and a region-side
        flow cannot be enforced whole on either side."""
        data2 = _copy_with(
            lh2_data,
            flow_from_co2_capped=pl.DataFrame({
                "p": ["pipe_AB", "wind_C"],
                "source": ["lh2_A", "wind_C"],
                "sink": ["lh2_B", "elec_C"],
                "c": ["co2", "co2"],
                "g": ["daily_group", "daily_group"],
            }),
            group_co2_max_period=pl.DataFrame({"g": ["daily_group"]}),
        )
        with pytest.raises(RuntimeError, match=r"CO2-capped group"):
            _master(data2, MASTER_BIG)


class TestAllMasterContentPartition:
    """All-master user constraints / feature groups / CO2 caps live in
    the MASTER and vanish from every region (the C6 residual)."""

    def _cstr_data(self, lh2_data):
        """lh2 data + an all-master user constraint (both referenced
        nodes in MASTER_BIG) spanning coefficient, constant and cdt
        frames.  Widens the (empty) constraint-axis vocabulary so the
        Enum filters run the production path."""
        d0 = lh2_data.dt.row(0, named=True)
        data2 = _copy_with(
            lh2_data,
            p_node_constraint_state_coeff=Param(
                ("n", "cn"),
                pl.DataFrame({
                    "n": ["lh2_B", "h2_B"],
                    "cn": ["master_cstr", "master_cstr"],
                    "value": [1.0, -1.0],
                }),
                name="p_node_constraint_state_coeff",
            ),
            p_constraint_constant=Param(
                ("cn",),
                pl.DataFrame({"cn": ["master_cstr"], "value": [0.0]}),
                name="p_constraint_constant",
            ),
            cdt_eq=pl.DataFrame({
                "cn": ["master_cstr"],
                "d": [d0["d"]], "t": [d0["t"]],
            }),
        )
        enums = dict(getattr(lh2_data, "_axis_enums", None) or {})
        if "constraint" in enums:
            enums["constraint"] = pl.Enum(
                list(enums["constraint"].categories) + ["master_cstr"])
            data2._axis_enums = enums
        return data2

    def test_all_master_constraint_kept_in_master(self, lh2_data) -> None:
        data2 = self._cstr_data(lh2_data)
        md = _master(data2, MASTER_BIG)
        assert _col_values(
            md.p_node_constraint_state_coeff.frame, "cn"
        ) == {"master_cstr"}
        assert _col_values(
            md.p_constraint_constant.frame, "cn"
        ) == {"master_cstr"}
        assert _col_values(md.cdt_eq, "cn") == {"master_cstr"}

    def test_all_master_constraint_dropped_from_regions(
        self, lh2_data
    ) -> None:
        """The C6 residual: the cn-only frames (constant, cdt) must not
        leave a degenerate ``0 == constant`` copy in any region."""
        data2 = self._cstr_data(lh2_data)
        splits = split(
            data2,
            regions=REGIONS,
            region_membership=_membership_without(data2, MASTER_BIG),
            master_hosted_nodes=MASTER_BIG,
        )
        for s in splits:
            for name, frame in _iter_frames(s.data):
                if "cn" not in frame.columns:
                    continue
                leaked = _col_values(frame, "cn") & {"master_cstr"}
                assert not leaked, f"{s.region}: {name} carries {leaked}"

    def test_all_master_feature_group_partition(self, lh2_data) -> None:
        """Under MASTER_ALL6 ``daily_group`` is all-master: the master
        keeps its feature row, every region drops it."""
        data2 = _copy_with(
            lh2_data,
            groupCapacityMargin=pl.DataFrame({"g": ["daily_group"]}),
        )
        md = _master(data2, MASTER_ALL6)
        assert _col_values(md.groupCapacityMargin, "g") == {"daily_group"}
        splits = split(
            data2,
            regions=REGIONS,
            region_membership=_membership_without(data2, MASTER_ALL6),
            master_hosted_nodes=MASTER_ALL6,
        )
        for s in splits:
            gcm = s.data.groupCapacityMargin
            assert gcm is None or "daily_group" not in _col_values(gcm, "g"), \
                s.region

    def test_all_master_co2_cap_partition(self, lh2_data) -> None:
        """Under MASTER_ALL6 a cap over the pipe flows is all-master:
        the master keeps the cap rows, every region drops them."""
        capped = pl.DataFrame({
            "p": ["pipe_AB", "pipe_BC"],
            "source": ["lh2_A", "lh2_B"],
            "sink": ["lh2_B", "lh2_C"],
            "c": ["co2", "co2"],
            "g": ["daily_group", "daily_group"],
        })
        data2 = _copy_with(
            lh2_data,
            flow_from_co2_capped=capped,
            group_co2_max_period=pl.DataFrame({"g": ["daily_group"]}),
        )
        md = _master(data2, MASTER_ALL6)
        assert _col_values(md.group_co2_max_period, "g") == {"daily_group"}
        assert md.flow_from_co2_capped.height == 2
        splits = split(
            data2,
            regions=REGIONS,
            region_membership=_membership_without(data2, MASTER_ALL6),
            master_hosted_nodes=MASTER_ALL6,
        )
        for s in splits:
            gm = s.data.group_co2_max_period
            assert gm is None or "daily_group" not in _col_values(gm, "g"), \
                s.region
            assert s.data.flow_from_co2_capped.height == 0, s.region


# ---------------------------------------------------------------------------
# C7 — build-through: build_flextool over the reduced master FlexData
# emits the master-side model identically to the monolith (F4).
# ---------------------------------------------------------------------------


def _cstr_rec(pb: Problem, name: str):
    return next(r for r in pb.cstrs_named(name) if r.name == name)


def _filter_val(frame: pl.DataFrame, col: str, value: str) -> pl.DataFrame:
    return frame.filter(pl.col(col).cast(pl.Utf8) == value)


@pytest.fixture(scope="module")
def mb_builds(mb_data):
    """(reduced master FlexData, master Problem, monolith Problem) over
    the build-through fixture under MASTER_BIG."""
    md = master_network_data(
        mb_data,
        REGIONS,
        region_membership=_membership_without(mb_data, MASTER_BIG),
        master_hosted_nodes=MASTER_BIG,
    )
    pb = Problem()
    build_flextool(pb, md)
    mono = Problem()
    build_flextool(mono, mb_data)
    return md, pb, mono


class TestMasterBuildThrough:
    def test_indirect_unit_precondition(self, mb_data) -> None:
        """The fixture's liquefier_B must classify INDIRECT (nvar) —
        otherwise the post-cf08c082 output-arc RHS path is not
        exercised and these tests silently under-cover."""
        assert mb_data.process_indirect is not None
        assert "liquefier_B" in set(
            mb_data.process_indirect["p"].cast(pl.Utf8).to_list())

    def test_master_solves(self, mb_builds) -> None:
        _, pb, _ = mb_builds
        sol = pb.solve()
        assert sol.optimal

    def test_master_storage_node_state_and_invest(self, mb_builds) -> None:
        """v_state, maxState and v_invest_n for the master-hosted
        storage node equal the monolith build filtered to the master
        nodes."""
        _, pb, mono = mb_builds
        in_master = pl.col("n").cast(pl.Utf8).is_in(sorted(MASTER_BIG))
        sm = _canon(pb._vars["v_state"].frame.drop("col_id"))
        so = _canon(
            mono._vars["v_state"].frame.filter(in_master).drop("col_id"))
        assert sm.height > 0 and sm.equals(so)

        im = _canon(pb._vars["v_invest_n"].frame.drop("col_id"))
        io = _canon(
            mono._vars["v_invest_n"].frame.filter(in_master).drop("col_id"))
        assert im.height == 1 and im.equals(io)
        assert set(im["n"].to_list()) == {"lh2_B"}

        ms_m = _cstr_rec(pb, "maxState")
        ms_o = _cstr_rec(mono, "maxState")
        assert _canon(ms_m.over).equals(
            _canon(ms_o.over.filter(in_master)))

    def test_master_balance_rows_and_inflow(self, mb_builds) -> None:
        """The master nodes' balance rows (block nodes ⇒
        ``nodeBalanceBlock_eq``) AND their inflow RHS equal the
        monolith's."""
        _, pb, mono = mb_builds
        bm = _cstr_rec(pb, "nodeBalanceBlock_eq")
        bo = _cstr_rec(mono, "nodeBalanceBlock_eq")
        for n in sorted(MASTER_BIG):
            om = _canon(_filter_val(bm.over, "n", n))
            oo = _canon(_filter_val(bo.over, "n", n))
            assert om.height > 0 and om.equals(oo), n
            rm = _canon(_filter_val(bm.proto.rhs.frame, "n", n))
            ro = _canon(_filter_val(bo.proto.rhs.frame, "n", n))
            assert rm.equals(ro), f"{n}: inflow/RHS mismatch"

    def test_master_penalty_slack_vars(self, mb_builds) -> None:
        _, pb, _ = mb_builds
        for v in ("vq_state_up", "vq_state_down"):
            nodes = set(
                pb._vars[v].frame["n"].cast(pl.Utf8).to_list())
            assert MASTER_BIG <= nodes, v

    def test_master_local_connection_flow_and_cap(self, mb_builds) -> None:
        """pipe_AB (master-local connection): v_flow domain and the
        maxFlow rows (incl. RHS) equal the monolith's."""
        _, pb, mono = mb_builds
        fm = _canon(_filter_val(
            pb._vars["v_flow"].frame, "p", "pipe_AB").drop("col_id"))
        fo = _canon(_filter_val(
            mono._vars["v_flow"].frame, "p", "pipe_AB").drop("col_id"))
        assert fm.height > 0 and fm.equals(fo)
        mf_m = _cstr_rec(pb, "maxFlow")
        mf_o = _cstr_rec(mono, "maxFlow")
        assert _canon(_filter_val(mf_m.over, "p", "pipe_AB")).equals(
            _canon(_filter_val(mf_o.over, "p", "pipe_AB")))
        assert _canon(_filter_val(mf_m.proto.rhs.frame, "p", "pipe_AB")).equals(
            _canon(_filter_val(mf_o.proto.rhs.frame, "p", "pipe_AB")))

    def test_master_local_unit_conversion_cap_and_invest(
        self, mb_builds
    ) -> None:
        """liquefier_B (master-local INDIRECT unit, the F4/shipping
        case): conversion rows, the post-cf08c082 maxFlow output-arc
        RHS and the unit invest all equal the monolith build filtered
        to the unit."""
        _, pb, mono = mb_builds
        ci_m = _cstr_rec(pb, "conversion_indirect")
        ci_o = _cstr_rec(mono, "conversion_indirect")
        cm = _canon(_filter_val(ci_m.over, "p", "liquefier_B"))
        co = _canon(_filter_val(ci_o.over, "p", "liquefier_B"))
        assert cm.height > 0 and cm.equals(co)

        mf_m = _cstr_rec(pb, "maxFlow")
        mf_o = _cstr_rec(mono, "maxFlow")
        om = _canon(_filter_val(mf_m.over, "p", "liquefier_B"))
        oo = _canon(_filter_val(mf_o.over, "p", "liquefier_B"))
        assert om.height > 0 and om.equals(oo)
        # RHS frame-for-frame: covers the indirect OUTPUT-arc rows'
        # existing-only bound alongside the input arcs' loose bound.
        rm = _canon(_filter_val(mf_m.proto.rhs.frame, "p", "liquefier_B"))
        ro = _canon(_filter_val(mf_o.proto.rhs.frame, "p", "liquefier_B"))
        assert rm.equals(ro)

        inv = set(pb._vars["v_invest_p"].frame["p"].cast(pl.Utf8).to_list())
        assert inv == {"liquefier_B", "pipe_AB", "pipe_BC"}

    def test_master_flow_domain_is_exactly_master_side(
        self, mb_builds
    ) -> None:
        """No region-local flow leaks into the master LP."""
        _, pb, _ = mb_builds
        procs = set(pb._vars["v_flow"].frame["p"].cast(pl.Utf8).to_list())
        assert procs == {
            "electrolyser_A", "electrolyser_B",
            "liquefier_A", "liquefier_B", "pipe_AB", "pipe_BC",
        }
