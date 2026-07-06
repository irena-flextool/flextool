"""Master-hosted node mode for the regional splitter (Benders C6).

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

With the default empty set the split is byte-identical to today's
shared-replicate behaviour.

Built on the lh2 fixture family (JSON-fixture DBs per CLAUDE.md
invariant #3): membership is manipulated in-memory by removing nodes
from their region group, which is exactly what the driver-side
membership rule produces for un-grouped nodes.
"""
from __future__ import annotations

import dataclasses

import polars as pl
import pytest

from polar_high import Param

from flextool.engine_polars import load_flextool
from flextool.engine_polars._pdt_join import compute_pss_dt
from flextool.engine_polars._region_filter import (
    _BENDERS_UNCAP_SENTINEL,
    compute_master_hosted_nodes,
    load_region_membership,
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
    return lh2, ti


@pytest.fixture(scope="module")
def lh2_data(_workdirs):
    return load_flextool(_workdirs[0])


@pytest.fixture(scope="module")
def ti_data(_workdirs):
    return load_flextool(_workdirs[1])


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
