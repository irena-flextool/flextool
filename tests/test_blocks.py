"""Unit tests for ``flextool.flextoolrunner.blocks`` (Agent 1.1).

The module's public surface: ``derive_blocks``, ``derive_overlap_set``,
``validate_group_membership``, ``write_block_data`` and the end-to-end
``write_block_data_for_solve`` helper.  These tests cover the pure
Python layer only — integration with the full solve loop is exercised
by the regression suite (which should remain bit-identical because the
emitted CSVs are inert in Agent 1.1).
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from flextool.flextoolrunner.blocks import (
    BlockAssignments,
    BlockBoundaries,
    BlockPredecessors,
    BlockTimelines,
    DEFAULT_BLOCK,
    OverlapSet,
    derive_block_boundaries,
    derive_block_predecessors,
    derive_blocks,
    derive_overlap_set,
    validate_group_membership,
    write_block_data,
)
from flextool.flextoolrunner.runner_state import FlexToolConfigError


# ---------------------------------------------------------------------------
# derive_blocks
# ---------------------------------------------------------------------------


class TestDeriveBlocksDefaultCase:
    """When no group carries ``new_stepduration``, every entity lands
    in ``"default"`` and the step duration matches the solve's
    timeline."""

    def test_all_entities_in_default_block(self) -> None:
        ba = derive_blocks(
            solve="invest",
            solve_config=None,
            timeline_config=None,
            nodes=["electricity", "hydrogen"],
            units=["ocgt", "electrolyser"],
            connections=["transmission"],
            resolution_groups={},
            group_unit=[("co2_cap", "ocgt")],
            group_connection=[],
            group_node=[("co2_cap", "electricity")],
            process_source_sink=[
                ("ocgt", "gas", "electricity"),
                ("electrolyser", "electricity", "hydrogen"),
                ("transmission", "electricity", "electricity_B"),
            ],
            process_ct_method={
                "ocgt": "constant_efficiency",
                "electrolyser": "regular",
                "transmission": "regular",
            },
        )

        assert ba.node_block == {
            "electricity": DEFAULT_BLOCK,
            "hydrogen": DEFAULT_BLOCK,
        }
        assert ba.process_block_in == {
            "ocgt": DEFAULT_BLOCK,
            "electrolyser": DEFAULT_BLOCK,
            "transmission": DEFAULT_BLOCK,
        }
        assert ba.process_block_out == {
            "ocgt": DEFAULT_BLOCK,
            "electrolyser": DEFAULT_BLOCK,
            "transmission": DEFAULT_BLOCK,
        }
        # Only the default block in the duration table.
        assert set(ba.block_step_duration) == {DEFAULT_BLOCK}


class TestDeriveBlocksTwoBlocks:
    """Hand-crafted fixture: one hourly group (block=hourly),
    one daily group (block=daily), one unit crossing them."""

    def test_direct_unit_picks_finer_side(self) -> None:
        """A direct (1var) unit with hourly source + daily sink must
        end up on the hourly side for both process_block_in and _out —
        one flow variable can only live at one resolution."""
        ba = derive_blocks(
            solve="invest",
            solve_config=None,
            timeline_config=None,
            nodes=["elec_node", "h2_node"],
            units=["electrolyser"],
            connections=[],
            resolution_groups={"hourly": 1.0, "daily": 24.0},
            group_unit=[],
            group_connection=[],
            group_node=[
                ("hourly", "elec_node"),
                ("daily", "h2_node"),
            ],
            process_source_sink=[("electrolyser", "elec_node", "h2_node")],
            process_ct_method={"electrolyser": "constant_efficiency"},
        )
        assert ba.node_block["elec_node"] == "hourly"
        assert ba.node_block["h2_node"] == "daily"
        # Direct method: both sides end up on the finer (hourly) block.
        assert ba.process_block_in["electrolyser"] == "hourly"
        assert ba.process_block_out["electrolyser"] == "hourly"
        assert ba.block_step_duration == {
            DEFAULT_BLOCK: 1.0,
            "hourly": 1.0,
            "daily": 24.0,
        }

    def test_indirect_unit_splits_by_side(self) -> None:
        """An indirect (nvar) unit with hourly source + daily sink
        ends up with per-side blocks: in=hourly, out=daily."""
        ba = derive_blocks(
            solve="invest",
            solve_config=None,
            timeline_config=None,
            nodes=["elec_node", "h2_node"],
            units=["electrolyser"],
            connections=[],
            resolution_groups={"hourly": 1.0, "daily": 24.0},
            group_unit=[],
            group_connection=[],
            group_node=[
                ("hourly", "elec_node"),
                ("daily", "h2_node"),
            ],
            process_source_sink=[("electrolyser", "elec_node", "h2_node")],
            process_ct_method={"electrolyser": "regular"},
        )
        assert ba.process_block_in["electrolyser"] == "hourly"
        assert ba.process_block_out["electrolyser"] == "daily"

    def test_explicit_process_membership_wins(self) -> None:
        """When a unit is explicitly in a resolution-group, it
        overrides the per-side node-adjacent logic."""
        ba = derive_blocks(
            solve="invest",
            solve_config=None,
            timeline_config=None,
            nodes=["elec_node", "h2_node"],
            units=["electrolyser"],
            connections=[],
            resolution_groups={"hourly": 1.0, "daily": 24.0},
            group_unit=[("daily", "electrolyser")],
            group_connection=[],
            group_node=[("hourly", "elec_node")],
            process_source_sink=[("electrolyser", "elec_node", "h2_node")],
            process_ct_method={"electrolyser": "regular"},
        )
        # Explicit unit membership in 'daily' overrides everything.
        assert ba.process_block_in["electrolyser"] == "daily"
        assert ba.process_block_out["electrolyser"] == "daily"


# ---------------------------------------------------------------------------
# derive_overlap_set
# ---------------------------------------------------------------------------


class TestOverlapSetAlignedSubsets:
    """24h fine + various coarse nestings."""

    def _two_block_assignments(
        self, coarse_hours: float
    ) -> tuple[BlockAssignments, BlockTimelines]:
        ba = BlockAssignments(
            node_block={"coarse_node": "coarse", "fine_node": DEFAULT_BLOCK},
            process_block_in={},
            process_block_out={},
            block_step_duration={DEFAULT_BLOCK: 1.0, "coarse": coarse_hours},
        )
        # 24 fine hourly steps in period "p".
        fine_rows = [(f"t{i:02d}", 1.0) for i in range(24)]
        # Aggregate to coarse.
        coarse_rows = []
        n_steps = int(24 / coarse_hours)
        for k in range(n_steps):
            coarse_rows.append((f"t{int(k * coarse_hours):02d}", coarse_hours))
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": fine_rows},
            "coarse": {"p": coarse_rows},
        })
        return ba, bt

    def test_single_coarse_row_covers_whole_day(self) -> None:
        ba, bt = self._two_block_assignments(24.0)
        overlap = derive_overlap_set(
            solve="invest",
            block_assignments=ba,
            block_timelines=bt,
        )
        # Identity (default↔default): 24 rows.  Coarse↔fine: 24 rows.
        # Symmetric fine↔coarse: 24 rows.
        default_identity = [r for r in overlap.rows if r[1] == DEFAULT_BLOCK and r[3] == DEFAULT_BLOCK]
        coarse_to_fine = [r for r in overlap.rows if r[1] == "coarse" and r[3] == DEFAULT_BLOCK]
        fine_to_coarse = [r for r in overlap.rows if r[1] == DEFAULT_BLOCK and r[3] == "coarse"]
        assert len(default_identity) == 24
        assert len(coarse_to_fine) == 24
        assert len(fine_to_coarse) == 24
        # Every coarse-to-fine row must point at the single coarse
        # timestep "t00".
        assert all(r[2] == "t00" for r in coarse_to_fine)
        assert all(r[5] == 1.0 for r in coarse_to_fine)

    def test_4_block_coarse_6to1(self) -> None:
        """Four 6-hour coarse rows, each covering six 1h fine rows."""
        ba, bt = self._two_block_assignments(6.0)
        overlap = derive_overlap_set(
            solve="invest",
            block_assignments=ba,
            block_timelines=bt,
        )
        coarse_to_fine = [r for r in overlap.rows if r[1] == "coarse" and r[3] == DEFAULT_BLOCK]
        assert len(coarse_to_fine) == 24
        # Each coarse row covers exactly 6 fine rows.
        coarse_counts: dict[str, int] = {}
        for r in coarse_to_fine:
            coarse_counts[r[2]] = coarse_counts.get(r[2], 0) + 1
        assert coarse_counts == {"t00": 6, "t06": 6, "t12": 6, "t18": 6}

    def test_degenerate_identity_rows(self) -> None:
        """No resolution-group blocks → only identity rows."""
        ba = BlockAssignments(
            node_block={"n1": DEFAULT_BLOCK},
            process_block_in={},
            process_block_out={},
            block_step_duration={DEFAULT_BLOCK: 1.0},
        )
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": [("t00", 1.0), ("t01", 1.0), ("t02", 1.0)]},
        })
        overlap = derive_overlap_set(
            solve="invest",
            block_assignments=ba,
            block_timelines=bt,
        )
        assert len(overlap.rows) == 3
        for row in overlap.rows:
            period, bc, sc, bf, sf, frac = row
            assert bc == DEFAULT_BLOCK
            assert bf == DEFAULT_BLOCK
            assert sc == sf
            assert frac == 1.0


# ---------------------------------------------------------------------------
# validate_group_membership
# ---------------------------------------------------------------------------


class TestValidateGroupMembership:
    """Membership rules — at most one resolution-group and at most one
    decomposition-group per entity."""

    def test_double_resolution_raises(self) -> None:
        with pytest.raises(FlexToolConfigError) as excinfo:
            validate_group_membership(
                group_unit=[("hourly", "u1"), ("daily", "u1")],
                group_connection=[],
                group_node=[],
                resolution_groups={"hourly": 1.0, "daily": 24.0},
                decomposition_groups={},
            )
        assert "u1" in str(excinfo.value)
        assert "hourly" in str(excinfo.value)
        assert "daily" in str(excinfo.value)

    def test_double_decomposition_raises(self) -> None:
        with pytest.raises(FlexToolConfigError) as excinfo:
            validate_group_membership(
                group_unit=[],
                group_connection=[],
                group_node=[
                    ("r1", "n1"),
                    ("r2", "n1"),
                ],
                resolution_groups={},
                decomposition_groups={
                    "r1": "lagrangian_region",
                    "r2": "lagrangian_region",
                },
            )
        assert "n1" in str(excinfo.value)

    def test_regular_groups_unconstrained(self) -> None:
        """Membership in many regular (non-resolution, non-decomp)
        groups is always fine — no raise."""
        validate_group_membership(
            group_unit=[("co2_cap", "u1"), ("reserve_up", "u1"), ("inertia", "u1")],
            group_connection=[],
            group_node=[("co2_cap", "n1"), ("reserve_up", "n1")],
            resolution_groups={},
            decomposition_groups={"r1": "none"},
        )  # no raise

    def test_one_resolution_plus_one_decomposition_ok(self) -> None:
        """An entity may be in one resolution-group and one
        decomposition-group simultaneously."""
        validate_group_membership(
            group_unit=[("hourly", "u1")],
            group_connection=[],
            group_node=[("region_a", "u1")],
            resolution_groups={"hourly": 1.0},
            decomposition_groups={"region_a": "lagrangian_region"},
        )  # no raise


# ---------------------------------------------------------------------------
# derive_block_predecessors (Agent 1.4)
# ---------------------------------------------------------------------------


class TestBlockPredecessors:
    """Per-block predecessor rows mirror ``dtttdt`` tagged by block."""

    def test_block_predecessors_default_case(self) -> None:
        """When only the default block exists and a real jump_list is
        passed, the emitted rows equal that jump_list with
        ``DEFAULT_BLOCK`` prepended (minus the trailing ``jump`` column,
        which is consumed separately via ``dt_jump``).  This is the
        bit-identical degeneracy contract agent 1.4 relies on.
        """
        # Mimic make_step_jump output: 3-step period, cyclic within.
        # Each entry is (period, step, prev, prev_within_ts, prev_period,
        # prev_within_solve, jump).
        jump_list = [
            ("p", "t1", "t3", "t3", "p", "t3", -2),
            ("p", "t2", "t1", "t1", "p", "t1", 1),
            ("p", "t3", "t2", "t2", "p", "t2", 1),
        ]
        ba = BlockAssignments(
            node_block={"n1": DEFAULT_BLOCK},
            process_block_in={},
            process_block_out={},
            block_step_duration={DEFAULT_BLOCK: 1.0},
        )
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": [("t1", 1.0), ("t2", 1.0), ("t3", 1.0)]},
        })
        bp = derive_block_predecessors(
            solve="s",
            block_assignments=ba,
            block_timelines=bt,
            default_jump_list=jump_list,
        )
        expected = [
            (DEFAULT_BLOCK, "p", "t1", "t3", "t3", "p", "t3"),
            (DEFAULT_BLOCK, "p", "t2", "t1", "t1", "p", "t1"),
            (DEFAULT_BLOCK, "p", "t3", "t2", "t2", "p", "t2"),
        ]
        assert bp.rows == expected

    def test_block_predecessors_non_default_cyclic(self) -> None:
        """Non-default block with a 4-step timeline cycles within its
        own rows — first step wraps to last, interior steps point at
        the prior row.  The default block (when present without a
        jump_list) follows the same cyclic pattern."""
        ba = BlockAssignments(
            node_block={"n1": "daily"},
            process_block_in={},
            process_block_out={},
            block_step_duration={DEFAULT_BLOCK: 1.0, "daily": 24.0},
        )
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": []},
            "daily": {"p": [
                ("t00", 24.0),
                ("t24", 24.0),
                ("t48", 24.0),
                ("t72", 24.0),
            ]},
        })
        bp = derive_block_predecessors(
            solve="s",
            block_assignments=ba,
            block_timelines=bt,
        )
        daily_rows = [r for r in bp.rows if r[0] == "daily"]
        assert daily_rows == [
            ("daily", "p", "t00", "t72", "t72", "p", "t72"),
            ("daily", "p", "t24", "t00", "t00", "p", "t00"),
            ("daily", "p", "t48", "t24", "t24", "p", "t24"),
            ("daily", "p", "t72", "t48", "t48", "p", "t48"),
        ]


class TestBlockBoundaries:
    """Per-block first / last step of each period."""

    def test_default_only(self) -> None:
        ba = BlockAssignments(
            node_block={"n1": DEFAULT_BLOCK},
            process_block_in={},
            process_block_out={},
            block_step_duration={DEFAULT_BLOCK: 1.0},
        )
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": [("t1", 1.0), ("t2", 1.0), ("t3", 1.0)]},
        })
        bb = derive_block_boundaries(ba, bt)
        assert bb.first == [(DEFAULT_BLOCK, "p", "t1")]
        assert bb.last == [(DEFAULT_BLOCK, "p", "t3")]

    def test_two_blocks(self) -> None:
        ba = BlockAssignments(
            node_block={"n1": "daily"},
            process_block_in={},
            process_block_out={},
            block_step_duration={DEFAULT_BLOCK: 1.0, "daily": 24.0},
        )
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": [("t00", 1.0), ("t01", 1.0), ("t02", 1.0)]},
            "daily": {"p": [("t00", 24.0), ("t24", 24.0)]},
        })
        bb = derive_block_boundaries(ba, bt)
        assert (DEFAULT_BLOCK, "p", "t00") in bb.first
        assert (DEFAULT_BLOCK, "p", "t02") in bb.last
        assert ("daily", "p", "t00") in bb.first
        assert ("daily", "p", "t24") in bb.last


# ---------------------------------------------------------------------------
# write_block_data — CSV shape
# ---------------------------------------------------------------------------


class TestWriteBlockData:
    def test_emits_all_csvs(self, tmp_path: Path) -> None:
        ba = BlockAssignments(
            node_block={"n1": "coarse", "n2": DEFAULT_BLOCK},
            process_block_in={"u1": "coarse"},
            process_block_out={"u1": DEFAULT_BLOCK},
            block_step_duration={DEFAULT_BLOCK: 1.0, "coarse": 24.0},
        )
        bt = BlockTimelines(per_block={
            DEFAULT_BLOCK: {"p": [("t00", 1.0), ("t01", 1.0)]},
            "coarse": {"p": [("t00", 24.0)]},
        })
        overlap = OverlapSet(rows=[
            ("p", DEFAULT_BLOCK, "t00", DEFAULT_BLOCK, "t00", 1.0),
            ("p", "coarse", "t00", DEFAULT_BLOCK, "t00", 1.0),
        ])
        bp = BlockPredecessors(rows=[
            (DEFAULT_BLOCK, "p", "t00", "t01", "t01", "p", "t01"),
            (DEFAULT_BLOCK, "p", "t01", "t00", "t00", "p", "t00"),
            ("coarse", "p", "t00", "t00", "t00", "p", "t00"),
        ])
        bb = BlockBoundaries(
            first=[(DEFAULT_BLOCK, "p", "t00"), ("coarse", "p", "t00")],
            last=[(DEFAULT_BLOCK, "p", "t01"), ("coarse", "p", "t00")],
        )

        write_block_data(
            block_assignments=ba,
            overlap_set=overlap,
            block_timelines=bt,
            solve_data_dir=tmp_path,
            block_predecessors=bp,
            block_boundaries=bb,
        )

        for fname in [
            "entity_block.csv",
            "process_side_block.csv",
            "block_step_duration.csv",
            "overlap_set.csv",
            "block_step_previous.csv",
            "block_period_time_first.csv",
            "block_period_time_last.csv",
        ]:
            assert (tmp_path / fname).exists(), fname

        # process_side_block emits two rows per process.
        with open(tmp_path / "process_side_block.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["process", "side", "block"]
        assert ["u1", "source", "coarse"] in rows
        assert ["u1", "sink", DEFAULT_BLOCK] in rows

        with open(tmp_path / "block_step_previous.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == [
            "block", "period", "step", "step_previous",
            "step_previous_within_timeset", "period_previous",
            "step_previous_within_solve",
        ]
        assert [DEFAULT_BLOCK, "p", "t00", "t01", "t01", "p", "t01"] in rows

        with open(tmp_path / "block_period_time_first.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["block", "period", "step"]
        assert [DEFAULT_BLOCK, "p", "t00"] in rows
