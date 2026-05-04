"""Block-aware preprocessing parity test (gap B5).

flextool's ``test_blocks.py`` exercises the *producer* side of block
machinery — the ``derive_blocks`` / ``derive_overlap_set`` /
``write_block_data`` pure-Python helpers that emit the per-solve block
CSVs (``entity_block.csv``, ``process_side_block.csv``,
``block_step_duration.csv``, ``overlap_set.csv``,
``block_period_time_first.csv``, ``block_period_time_last.csv``).
flexpy_spike is a *consumer* of those CSVs (preprocessing happens
upstream in flextool), so the equivalent invariants in flexpy are
on-disk shape contracts the input/model layer relies on.

This test pins the two block-aware paths flexpy actually walks:

* **Single-block degeneracy** (``work_base`` — the most common case):
  one ``default`` block per period, identity-only overlap, empty
  ``process_side_block``.  Mirrors flextool's
  ``TestDeriveBlocksDefaultCase`` and
  ``TestOverlapSetAlignedSubsets::test_degenerate_identity_rows``.

* **Multi-block aggregation** (``work_lh2_three_region`` — the
  3-block hourly/daily/default case): per-block step-duration sums
  match the period length; daily-coarse rows partition the
  hourly-fine rows exactly (each coarse step covers 24 fine steps);
  mixed-block arcs land where the .mod expects (electrolyser:
  source=hourly_group, sink=daily_group; pipes: daily on both sides;
  wind: hourly on both sides); per-block first/last steps match the
  block's own timeline.  Mirrors flextool's
  ``TestDeriveBlocksTwoBlocks`` (per-side block assignment),
  ``TestOverlapSetAlignedSubsets`` (24:1 coarse-to-fine partition),
  and ``TestBlockBoundaries::test_two_blocks``.

Both fixtures already pass under their respective parity tests
(``test_flex_base``, ``test_flex_lh2_three_region``); this test
locks down the *structural* contract those parity tests rely on so
an upstream regression in the block-CSV writer (or in flexpy's
single-block tolerance) shows up loudly.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

DATA = Path(__file__).resolve().parent / "data"
WORK_SINGLE = DATA / "work_base"
WORK_MULTI = DATA / "work_lh2_three_region"

DEFAULT_BLOCK = "default"


def _check_single_block_degeneracy(sd: Path) -> None:
    """work_base: one period, one block.  Asserts entity_block,
    block_set, block_step_duration, process_side_block (empty), and
    overlap_set (identity-only) all collapse to default."""
    # block_set: only default.
    bs = pl.read_csv(sd / "block_set.csv")
    assert bs["block"].to_list() == [DEFAULT_BLOCK]

    # entity_block: every entity → default (TestDeriveBlocksDefaultCase).
    eb = pl.read_csv(sd / "entity_block.csv")
    assert eb.height >= 1
    assert set(eb["block"].to_list()) == {DEFAULT_BLOCK}

    # process_side_block: empty (no multi-block arcs needed).
    psb = pl.read_csv(sd / "process_side_block.csv")
    assert psb.height == 0

    # block_step_duration: only default, all unit-1h steps.
    bsd = pl.read_csv(sd / "block_step_duration.csv")
    assert set(bsd["block"].unique().to_list()) == {DEFAULT_BLOCK}
    assert (bsd["step_duration"] == 1.0).all()

    # overlap_set: identity-only — sc==sf, frac==1.0, both blocks default
    # (TestOverlapSetAlignedSubsets::test_degenerate_identity_rows).
    ov = pl.read_csv(sd / "overlap_set.csv")
    assert ov.height > 0
    assert (ov["block_coarse"] == DEFAULT_BLOCK).all()
    assert (ov["block_fine"] == DEFAULT_BLOCK).all()
    assert (ov["step_coarse"] == ov["step_fine"]).all()
    assert (ov["fraction"] == 1.0).all()
    # No duplicates: one identity row per (period, fine step).
    assert ov.select("period", "step_coarse").unique().height == ov.height
    # Identity overlap height matches block_step_duration height.
    assert ov.height == bsd.height


def _check_multi_block_aggregation(sd: Path) -> None:
    """work_lh2_three_region: 3 blocks (default, hourly_group,
    daily_group); 168h horizon; 7 daily coarse rows × 24h."""
    PERIOD = "y2030"
    HOURLY = "hourly_group"
    DAILY = "daily_group"
    HORIZON = 168.0  # one ISO week

    # block_set: three blocks present.
    bs = pl.read_csv(sd / "block_set.csv")
    assert set(bs["block"].to_list()) == {DEFAULT_BLOCK, HOURLY, DAILY}

    # block_step_duration: per-block durations & row counts.
    bsd = pl.read_csv(sd / "block_step_duration.csv")
    per_block = {r["block"]: r for r in (bsd
        .group_by("block")
        .agg(pl.col("step_duration").unique().alias("durs"),
             pl.len().alias("n_rows"))
        .to_dicts())}
    for b in (DEFAULT_BLOCK, HOURLY):
        assert per_block[b]["durs"] == [1.0]
        assert per_block[b]["n_rows"] == int(HORIZON)
    assert per_block[DAILY]["durs"] == [24.0]
    assert per_block[DAILY]["n_rows"] == 7

    # Closed-form invariant: Σ step_duration per (block, period) = HORIZON.
    sums = (bsd
        .group_by("block", "period")
        .agg(pl.col("step_duration").sum().alias("hours"))
        .to_dicts())
    for r in sums:
        assert r["hours"] == HORIZON, r

    # entity_block per-side assignment (TestDeriveBlocksTwoBlocks):
    # hourly nodes on hourly_group; lh2 / h2 nodes on daily_group.
    eb = pl.read_csv(sd / "entity_block.csv")
    nb = dict(zip(eb["entity"].to_list(), eb["block"].to_list()))
    for n in ("elec_A", "elec_B", "elec_C",
              "battery_A", "battery_B", "battery_C", "coal_market"):
        assert nb[n] == HOURLY, (n, nb.get(n))
    for n in ("h2_A", "h2_B", "h2_C", "lh2_A", "lh2_B", "lh2_C"):
        assert nb[n] == DAILY, (n, nb.get(n))

    # process_side_block: indirect electrolyser splits per side
    # (test_indirect_unit_splits_by_side); wind is direct hourly→hourly
    # (test_direct_unit_picks_finer_side); pipes are daily↔daily.
    psb = pl.read_csv(sd / "process_side_block.csv")
    elec_sides = (psb
        .filter(pl.col("process").str.starts_with("electrolyser_"))
        .pivot(values="block", index="process", on="side")
        .to_dicts())
    assert len(elec_sides) >= 1
    for r in elec_sides:
        assert r["source"] == HOURLY, r
        assert r["sink"] == DAILY, r
    for r in psb.filter(pl.col("process").str.starts_with("wind_")).to_dicts():
        assert r["block"] == HOURLY, r
    pipe_rows = psb.filter(pl.col("process").str.starts_with("pipe_")).to_dicts()
    assert len(pipe_rows) >= 2
    for r in pipe_rows:
        assert r["block"] == DAILY, r

    # overlap_set: daily↔hourly partition (TestOverlapSetAlignedSubsets
    # ::test_4_block_coarse_6to1, scaled to 24:1).
    ov = pl.read_csv(sd / "overlap_set.csv")
    assert (ov["fraction"] == 1.0).all()  # exact nesting in this fixture.
    d2h = ov.filter(
        (pl.col("block_coarse") == DAILY) & (pl.col("block_fine") == HOURLY)
    )
    assert d2h.height == 168  # 7 coarse × 24 fine.
    per_coarse_h = (d2h
        .group_by("step_coarse").agg(pl.len().alias("n")).to_dicts())
    assert len(per_coarse_h) == 7
    for r in per_coarse_h:
        assert r["n"] == 24, r
    # daily↔default the same way (the path nodeBalance_eq walks for
    # default-grid nodes connected to a daily arc).
    d2d = ov.filter(
        (pl.col("block_coarse") == DAILY) & (pl.col("block_fine") == DEFAULT_BLOCK)
    )
    assert d2d.height == 168
    for r in (d2d.group_by("step_coarse").agg(pl.len().alias("n"))
                .to_dicts()):
        assert r["n"] == 24, r

    # Self-overlap is identity for every block.
    for b in (DEFAULT_BLOCK, HOURLY, DAILY):
        self_rows = ov.filter(
            (pl.col("block_coarse") == b) & (pl.col("block_fine") == b))
        assert self_rows.height > 0, b
        assert (self_rows["step_coarse"] == self_rows["step_fine"]).all()
        assert (self_rows["fraction"] == 1.0).all()

    # Generalised partition: per (b_c, b_f) row count = duration ratio
    # (or 1 in the reverse-label direction where coarse-label is finer).
    dur = {(r["block"], r["period"], r["step"]): r["step_duration"]
           for r in bsd.to_dicts()}
    per_coarse_all = (ov
        .group_by("period", "block_coarse", "step_coarse", "block_fine")
        .agg(pl.len().alias("n_fine")).to_dicts())
    for r in per_coarse_all:
        d_c = dur[(r["block_coarse"], r["period"], r["step_coarse"])]
        f_durs = (bsd
            .filter((pl.col("block") == r["block_fine"])
                    & (pl.col("period") == r["period"]))
            ["step_duration"].unique().to_list())
        assert len(f_durs) == 1
        d_f = f_durs[0]
        if d_c < d_f:
            assert r["n_fine"] == 1, (r, d_c, d_f)
        else:
            assert r["n_fine"] == int(round(d_c / d_f)), (r, d_c, d_f)

    # block_period_time_first / _last per-block boundaries
    # (TestBlockBoundaries::test_two_blocks).
    first = pl.read_csv(sd / "block_period_time_first.csv")
    last = pl.read_csv(sd / "block_period_time_last.csv")
    for b in (DEFAULT_BLOCK, HOURLY, DAILY):
        block_steps = (bsd
            .filter((pl.col("block") == b) & (pl.col("period") == PERIOD))
            .sort("step"))
        assert block_steps.height > 0, b
        f_row = first.filter(
            (pl.col("block") == b) & (pl.col("period") == PERIOD))
        l_row = last.filter(
            (pl.col("block") == b) & (pl.col("period") == PERIOD))
        assert f_row.height == 1 and f_row["step"][0] == block_steps["step"][0], b
        assert l_row.height == 1 and l_row["step"][0] == block_steps["step"][-1], b


def test_block_preprocessing_invariants():
    """Block-aware preprocessing parity — single-block degeneracy and
    multi-block aggregation, both pinned against on-disk CSVs.
    """
    _check_single_block_degeneracy(WORK_SINGLE / "solve_data")
    _check_multi_block_aggregation(WORK_MULTI / "solve_data")
