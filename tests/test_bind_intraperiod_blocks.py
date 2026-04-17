"""Tests for the bind_intraperiod_blocks storage binding method.

The method adds two pieces of Python logic to the runner:
  * `make_period_block()` — derives `period_block_time` and
    `period_block_succ` from the active-time structure.
  * `write_period_block()` — serialises those to CSVs consumed by
    the model.

Block boundaries follow the same rule the rest of FlexTool uses
(`make_step_jump`): a gap in timeline indices (jump > 1) starts a
new block. These tests cover that algorithmic contract directly and
verify it stays consistent with `make_step_jump`.

End-to-end verification of the model constraints themselves
(`stateConstantWithinBlock_eq`, `nodeBalanceBlock_eq`) is done via
the Rivendell S05_phs_chrono scenario during development; a solver-
level pytest would require a self-contained fixture DB — tracked as
a follow-up.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from flextool.flextoolrunner.runner_state import ActiveTimeEntry
from flextool.flextoolrunner.solve_writers import write_period_block
from flextool.flextoolrunner.timeline_config import make_period_block, make_step_jump


def _active(period_map: dict[str, list[tuple[str, int]]]) -> dict[str, list[ActiveTimeEntry]]:
    """Build an active_time_list from compact (timestep, timeline_index) pairs."""
    return {
        period: [ActiveTimeEntry(timestep=ts, index=idx, duration=1.0) for ts, idx in entries]
        for period, entries in period_map.items()
    }


# ---------------------------------------------------------------------------
# make_period_block — block detection and cyclic successor
# ---------------------------------------------------------------------------


class TestMakePeriodBlock:
    def test_single_block_contiguous(self):
        """Contiguous timeline indices → one block, self-cyclic successor."""
        active = _active({"p": [("t1", 0), ("t2", 1), ("t3", 2), ("t4", 3)]})
        pbt, pbs = make_period_block(active)
        assert pbt == [("p", "t1", "t1"), ("p", "t1", "t2"), ("p", "t1", "t3"), ("p", "t1", "t4")]
        assert pbs == [("p", "t1", "t1")]

    def test_two_blocks_with_gap(self):
        """A jump > 1 in timeline indices starts a new block."""
        # t1..t4 at idx 0..3, then a gap, then t5..t8 at idx 5..8
        active = _active(
            {"p": [("t1", 0), ("t2", 1), ("t3", 2), ("t4", 3), ("t5", 5), ("t6", 6), ("t7", 7), ("t8", 8)]}
        )
        pbt, pbs = make_period_block(active)
        block_first_of = {row[2]: row[1] for row in pbt}
        # Block 1 covers t1..t4, block 2 covers t5..t8
        assert [block_first_of[f"t{i}"] for i in range(1, 5)] == ["t1"] * 4
        assert [block_first_of[f"t{i}"] for i in range(5, 9)] == ["t5"] * 4
        # Cyclic successor: t1→t5, t5→t1 within period p
        assert sorted(pbs) == [("p", "t1", "t5"), ("p", "t5", "t1")]

    def test_three_blocks(self):
        """Three blocks, cyclic wrap at period end."""
        active = _active(
            {"p": [("a1", 0), ("a2", 1), ("b1", 10), ("b2", 11), ("c1", 20), ("c2", 21)]}
        )
        pbt, pbs = make_period_block(active)
        block_first_of = {row[2]: row[1] for row in pbt}
        assert block_first_of["a1"] == "a1" and block_first_of["a2"] == "a1"
        assert block_first_of["b1"] == "b1" and block_first_of["b2"] == "b1"
        assert block_first_of["c1"] == "c1" and block_first_of["c2"] == "c1"
        # Forward chain a1→b1→c1, cyclic c1→a1
        assert sorted(pbs) == [("p", "a1", "b1"), ("p", "b1", "c1"), ("p", "c1", "a1")]

    def test_multiple_periods_are_independent(self):
        """Each period has its own block structure; successor never crosses periods."""
        active = _active({
            "p1": [("t1", 0), ("t2", 1), ("t3", 5), ("t4", 6)],
            "p2": [("t1", 0), ("t2", 1), ("t3", 2)],  # single block
        })
        pbt, pbs = make_period_block(active)
        p1_succ = sorted(s for s in pbs if s[0] == "p1")
        p2_succ = sorted(s for s in pbs if s[0] == "p2")
        assert p1_succ == [("p1", "t1", "t3"), ("p1", "t3", "t1")]
        # p2 has a single block → self-cyclic
        assert p2_succ == [("p2", "t1", "t1")]
        # Per-period row counts in period_block_time match active-time lengths
        assert sum(1 for row in pbt if row[0] == "p1") == 4
        assert sum(1 for row in pbt if row[0] == "p2") == 3

    def test_empty_active_time_for_period_is_skipped(self):
        """Empty period list must not emit block rows and must not crash."""
        active = {"p_empty": [], "p_real": [ActiveTimeEntry("t1", 0, 1.0), ActiveTimeEntry("t2", 1, 1.0)]}
        pbt, pbs = make_period_block(active)
        assert all(row[0] == "p_real" for row in pbt)
        assert all(row[0] == "p_real" for row in pbs)

    def test_empty_input(self):
        """No periods at all → empty outputs, no exception."""
        pbt, pbs = make_period_block({})
        assert pbt == []
        assert pbs == []

    def test_period_block_time_covers_every_active_step_exactly_once(self):
        """Every active step appears on exactly one period_block_time row."""
        active = _active(
            {"p": [("t1", 0), ("t2", 1), ("t3", 5), ("t4", 6), ("t5", 7), ("t6", 30)]}
        )
        pbt, _ = make_period_block(active)
        steps_seen = [row[2] for row in pbt]
        assert sorted(steps_seen) == ["t1", "t2", "t3", "t4", "t5", "t6"]
        # No duplicates
        assert len(set(steps_seen)) == len(steps_seen)


# ---------------------------------------------------------------------------
# make_period_block vs make_step_jump — block boundaries must agree
# ---------------------------------------------------------------------------


class TestConsistencyWithStepJump:
    """The two functions must agree on where blocks start and end.

    make_step_jump encodes block-first rows as `previous != previous_within_timeset`
    (the wraparound). make_period_block emits the same information directly.
    A regression in either function should show up here.
    """

    @pytest.mark.parametrize(
        "active_spec",
        [
            # Single block
            {"p": [("t1", 0), ("t2", 1), ("t3", 2), ("t4", 3)]},
            # Two blocks
            {"p": [("t1", 0), ("t2", 1), ("t3", 5), ("t4", 6)]},
            # Three blocks
            {"p": [("a1", 0), ("a2", 1), ("b1", 10), ("b2", 11), ("c1", 20)]},
            # Two periods, independent structures
            {
                "p1": [("t1", 0), ("t2", 1), ("t3", 5), ("t4", 6)],
                "p2": [("t1", 0), ("t2", 1), ("t3", 2)],
            },
        ],
    )
    def test_block_boundaries_agree(self, active_spec):
        active = _active(active_spec)
        pbt, _ = make_period_block(active)

        # make_step_jump marks every block-first row with
        # previous != previous_within_timeset when the period has >1 block.
        # In a single-block period the two fields collapse and no row is
        # marked. Restrict both sides of the comparison to periods with
        # more than one block.
        period__branch = [(p, p) for p in active.keys()]
        jumps = make_step_jump(active, period__branch=period__branch, solve_branch__time_branch_list=[])
        step_jump_block_firsts = {
            (row[0], row[1]) for row in jumps if row[2] != row[3]
        }

        blocks_per_period: dict[str, set[str]] = {}
        for period, block_first, _step in pbt:
            blocks_per_period.setdefault(period, set()).add(block_first)
        multi_block = {p for p, bs in blocks_per_period.items() if len(bs) > 1}

        period_block_firsts = {
            (row[0], row[2])
            for row in pbt
            if row[1] == row[2] and row[0] in multi_block
        }

        assert step_jump_block_firsts == period_block_firsts


# ---------------------------------------------------------------------------
# write_period_block — CSV format
# ---------------------------------------------------------------------------


class TestWritePeriodBlock:
    def test_files_and_headers(self, tmp_path: Path):
        (tmp_path / "solve_data").mkdir()
        pbt = [("p", "t1", "t1"), ("p", "t1", "t2"), ("p", "t3", "t3")]
        pbs = [("p", "t1", "t3"), ("p", "t3", "t1")]
        write_period_block(pbt, pbs, work_folder=tmp_path)

        with open(tmp_path / "solve_data" / "period_block_time.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["period", "block_first", "step"]
        assert rows[1:] == [list(t) for t in pbt]

        with open(tmp_path / "solve_data" / "period_block_succ.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == ["period", "block_first", "block_first_next"]
        assert rows[1:] == [list(t) for t in pbs]

    def test_empty_inputs_produce_header_only(self, tmp_path: Path):
        (tmp_path / "solve_data").mkdir()
        write_period_block([], [], work_folder=tmp_path)
        assert (tmp_path / "solve_data" / "period_block_time.csv").read_text().strip() == "period,block_first,step"
        assert (
            (tmp_path / "solve_data" / "period_block_succ.csv").read_text().strip()
            == "period,block_first,block_first_next"
        )
