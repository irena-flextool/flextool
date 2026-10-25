"""Which step a ramp limit compares each step with.

A ramp row exists for a step only when the step has a predecessor that is adjacent on the timeline in the
un-rolled solve (``make_step_jump``'s ``jump == 1``): no cyclic wrap at the first step of a solve or of a
period that does not continue the previous one, no row across a gap in the timeline, and none at the start of a
representative period (two representative blocks that happen to be adjacent represent different base periods).
On a continuation roll of a rolling solve, the first step of a period ramps from the flow the previous roll
realized at the step just before it.

Both frames are plain Utf8 (``d``, ``t``, ``d_previous``, ``t_previous``); consumers cast to their own dtypes.
"""
from __future__ import annotations

import polars as pl

_LINK_COLS = ("d", "t", "d_previous", "t_previous")


def ramp_link_from_jump(steps: "pl.DataFrame | None",
                        rp_block_first: "pl.DataFrame | None" = None) -> "pl.DataFrame | None":
    """``(d, t, d_previous, t_previous)`` of the steps whose predecessor is adjacent on the timeline.

    ``steps`` carries ``d``, ``t``, ``d_previous``, ``t_previous_within_solve`` and ``jump`` (the
    ``step_previous`` columns renamed as in ``dtttdt``); the predecessor of a step is
    ``(d_previous, t_previous_within_solve)``.  ``rp_block_first`` ``(d, t)``: the first steps of representative
    blocks, which never ramp from the step before them.  ``None`` when ``steps`` is ``None``; an empty frame
    when no step has an adjacent predecessor.
    """
    if steps is None:
        return None
    jump = pl.col("jump").cast(pl.Utf8).cast(pl.Float64, strict=False)
    out = (steps.lazy()
           .filter(jump == 1.0)
           .select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                   pl.col("d_previous").cast(pl.Utf8),
                   pl.col("t_previous_within_solve").cast(pl.Utf8).alias("t_previous")))
    if rp_block_first is not None and rp_block_first.height > 0:
        first = rp_block_first.lazy().select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
        out = out.join(first, on=["d", "t"], how="anti")
    return out.collect()


def ramp_roll_carry(dt: "pl.DataFrame | None",
                    carry: "pl.DataFrame | None",
                    first_step_link: "pl.DataFrame | None",
                    period_branch: "pl.DataFrame | None" = None,
                    ) -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """The first steps of this roll that ramp from the previous roll's realized flow.

    ``dt`` ``(d, t)`` in solve order.  ``carry`` ``[process, source, sink, period, step, duration, value]``: the
    previous roll's ``v_flow`` at its last realized step (one step).  ``first_step_link`` ``[period, time,
    previous_period, previous_within_solve, jump]``: the complete (un-rolled) solve's ``step_previous`` rows of
    the first step of each period of this roll.  ``period_branch`` ``[period, branch]``: a forecast branch
    period that starts at the same step as its anchor period continues from the same realized flow.

    A period's first step ``(d, t1)`` takes the carry when the complete solve's predecessor of ``t1`` is
    adjacent (``jump == 1``) and is the carried step.  Returns ``(dt_ramp_carry, flows)``:
    ``dt_ramp_carry`` = ``(d, t, d_previous, t_previous, duration_previous)`` and ``flows`` =
    ``(p, source, sink, d, t, value)``, all keys Utf8; ``(None, None)`` when no step takes the carry (first
    roll, a single solve, a carried step that is not adjacent, e.g. before a gap in the timeline).
    """
    if (dt is None or dt.height == 0 or carry is None or carry.height == 0
            or first_step_link is None or first_step_link.height == 0):
        return None, None
    carried = (carry.select(pl.col("period").cast(pl.Utf8), pl.col("step").cast(pl.Utf8),
                            pl.col("duration").cast(pl.Float64, strict=False))
               .unique())
    if carried.height != 1:
        raise ValueError(
            "ramp_roll_carry: the carried flow must come from one step of the previous roll, got "
            f"{carried.rows()}")
    c_d, c_t, c_dur = carried.row(0)
    firsts = (dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
              .with_row_index("__i")
              .group_by("d", maintain_order=True).agg(pl.col("t").sort_by("__i").first()))
    link = first_step_link.select(
        pl.col("period").cast(pl.Utf8).alias("d"), pl.col("time").cast(pl.Utf8).alias("t"),
        pl.col("previous_period").cast(pl.Utf8).alias("d_previous"),
        pl.col("previous_within_solve").cast(pl.Utf8).alias("t_previous"),
        pl.col("jump").cast(pl.Utf8).cast(pl.Float64, strict=False).alias("jump"))
    hit = (firsts.join(link, on=["d", "t"], how="inner")
           .filter((pl.col("jump") == 1.0) & (pl.col("d_previous") == c_d) & (pl.col("t_previous") == c_t))
           .select("d", "t"))
    if period_branch is not None and period_branch.height > 0 and hit.height > 0:
        pb = (period_branch.select(pl.col("period").cast(pl.Utf8).alias("d"),
                                   pl.col("branch").cast(pl.Utf8).alias("b"))
              .filter(pl.col("d") != pl.col("b")))
        branches = (hit.join(pb, on="d", how="inner")
                    .join(firsts.rename({"d": "b", "t": "t_b"}), on="b", how="inner")
                    .filter(pl.col("t_b") == pl.col("t"))
                    .select(pl.col("b").alias("d"), "t"))
        hit = pl.concat([hit, branches]).unique(maintain_order=True)
    if hit.height == 0:
        return None, None
    link_out = hit.with_columns(d_previous=pl.lit(c_d), t_previous=pl.lit(c_t),
                                duration_previous=pl.lit(c_dur, dtype=pl.Float64))
    flows = (carry.select(pl.col("process").cast(pl.Utf8).alias("p"), pl.col("source").cast(pl.Utf8),
                          pl.col("sink").cast(pl.Utf8),
                          pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
             .join(hit, how="cross")
             .select("p", "source", "sink", "d", "t", "value"))
    return link_out, flows
