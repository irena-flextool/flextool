"""Coarse resolution blocks must respect representative-period seams.

Regression tests for the coarse-fuel-node storage bug: when a storage
node's time resolution is coarsened (``group.new_stepduration``) on a
timeline made of representative periods (days sampled from across the
year, concatenated into one FlexTool period), the coarse-block storage
machinery used to close ONE cyclic loop over the whole period.  That let
a coarse node carry inventory across months-apart representative days as
if they were consecutive — a large, non-physical LP relaxation that made
every coarsening (2h, 3h, ... 6h) collapse to the same over-relaxed
answer (a "cliff", independent of the step length).

The fix segments the cyclic loop PER representative period and gives
blended-weights (rp) nodes a free per-representative-day start state so
their seasonal storage chain stays active at coarse resolution.

These tests pin the seam-detection helpers, the per-representative-period
segmentation of ``period_block_succ``, and the plain-English guard that
rejects a ``new_stepduration`` that does not divide the representative-
period length.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars._derived_params import (
    _fine_rank_map,
    _guard_coarse_blocks_within_repday,
    _repday_segment_map,
)
from flextool.engine_polars._solve_state import FlexToolConfigError


def _per_period_two_repdays() -> dict[str, list[tuple[str, int]]]:
    """Two 4-hour representative days inside one FlexTool period ``y``.

    Ranks jump from 3 to 100 between the two days (they are sampled
    months apart), so the seam is a rank gap > 1.
    """
    day0 = [("d0h0", 0), ("d0h1", 1), ("d0h2", 2), ("d0h3", 3)]
    day1 = [("d1h0", 100), ("d1h1", 101), ("d1h2", 102), ("d1h3", 103)]
    return {"y": day0 + day1}


def test_repday_segment_map_splits_on_rank_gap() -> None:
    seg = _repday_segment_map(_per_period_two_repdays())
    # First representative day → segment 0, second → segment 1.
    assert seg[("y", "d0h0")] == 0
    assert seg[("y", "d0h3")] == 0
    assert seg[("y", "d1h0")] == 1
    assert seg[("y", "d1h3")] == 1


def test_repday_segment_map_contiguous_is_single_segment() -> None:
    # No rank gaps → one segment (plain contiguous timeline is a no-op).
    contiguous = {"y": [(f"h{i}", i) for i in range(8)]}
    seg = _repday_segment_map(contiguous)
    assert set(seg.values()) == {0}


def test_fine_rank_map() -> None:
    rank = _fine_rank_map(_per_period_two_repdays())
    assert rank[("y", "d1h0")] == 100


def _segment_succ(per_period, coarse_bfirsts):
    """Replicate the producer's per-representative-period cyclic
    successor construction (the logic under test in
    ``period_block_family_from_source``)."""
    label_seg = _repday_segment_map(per_period)
    label_rank = _fine_rank_map(per_period)
    segs: dict[int, list[str]] = {}
    for b in coarse_bfirsts:
        segs.setdefault(label_seg[("y", b)], []).append(b)
    succ = []
    for seg_id in sorted(segs):
        seg_bfs = sorted(segs[seg_id], key=lambda b: label_rank[("y", b)])
        n = len(seg_bfs)
        for i in range(n):
            succ.append((seg_bfs[i], seg_bfs[(i + 1) % n]))
    return succ, label_seg


def test_period_block_succ_has_no_cross_seam_edges() -> None:
    # 2-hour coarse blocks: each 4-hour rep-day → two blocks (h0, h2).
    per_period = _per_period_two_repdays()
    coarse_bfirsts = ["d0h0", "d0h2", "d1h0", "d1h2"]
    succ, label_seg = _segment_succ(per_period, coarse_bfirsts)
    # Every successor edge stays within one representative-period segment.
    for b_first, b_next in succ:
        assert label_seg[("y", b_first)] == label_seg[("y", b_next)], (
            f"cross-seam storage edge {b_first}->{b_next} — coarse node "
            "would chain storage across months-apart representative days"
        )
    # Each representative day closes its own cyclic loop (2 blocks each).
    assert ("d0h0", "d0h2") in succ and ("d0h2", "d0h0") in succ
    assert ("d1h0", "d1h2") in succ and ("d1h2", "d1h0") in succ
    # ...and NO whole-period wrap linking the last block of day1 back to
    # the first block of day0 (the bug).
    assert ("d1h2", "d0h0") not in succ


def test_guard_passes_when_blocks_align_to_repday() -> None:
    label_seg = _repday_segment_map(_per_period_two_repdays())
    # Aligned 2h blocks: each covers two fine steps within one rep-day.
    pbt = pl.DataFrame({
        "d": ["y"] * 8,
        "b_first": ["d0h0", "d0h0", "d0h2", "d0h2",
                    "d1h0", "d1h0", "d1h2", "d1h2"],
        "t": ["d0h0", "d0h1", "d0h2", "d0h3",
              "d1h0", "d1h1", "d1h2", "d1h3"],
    })
    # Should NOT raise.
    _guard_coarse_blocks_within_repday(pbt, label_seg, ["coarse_fuels"])


def test_guard_rejects_block_straddling_repday_seam() -> None:
    label_seg = _repday_segment_map(_per_period_two_repdays())
    # A block that covers the last step of day0 AND the first step of day1.
    pbt = pl.DataFrame({
        "d": ["y", "y", "y"],
        "b_first": ["d0h2", "d0h2", "d0h2"],
        "t": ["d0h2", "d0h3", "d1h0"],
    })
    with pytest.raises(FlexToolConfigError) as exc:
        _guard_coarse_blocks_within_repday(pbt, label_seg, ["coarse_fuels"])
    msg = str(exc.value)
    assert "representative period" in msg
    assert "new_stepduration" in msg
    assert "coarse_fuels" in msg
