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


def online_stretch_wrap(dt: "pl.DataFrame | None", link: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """``(d, t, d_previous, t_previous)`` Utf8: the online predecessor of every step of ``dt`` that has no ramp
    predecessor (is not a target of ``link`` = ``dt_ramp_link``): the last step of its own stretch.

    A stretch is a run of ``dt`` steps (in solve order) of one period ``d`` that follow each other along ``link``;
    it starts at a step that is not the target of a same-period link (the first step of a solve or of a period, the
    step after a gap in the timeline, a representative-block start) and ends before the next such step.  The
    commitment at a stretch start therefore cycles within its stretch, as a single block of one period does: it
    never reaches across a gap or into another period (a period that continues the previous one is a ``link``
    target and has no wrap).  ``None`` when ``dt`` or ``link`` is ``None``.  Raises ``ValueError`` when a
    same-period link joins steps that are not consecutive steps of that period in ``dt``.
    """
    if dt is None or link is None:
        return None
    steps = (dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
             .with_row_index("__i")
             .with_columns(pl.col("t").shift(1).over("d", order_by="__i").alias("__t_before")))
    lk = link.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                     pl.col("d_previous").cast(pl.Utf8), pl.col("t_previous").cast(pl.Utf8))
    same = lk.filter(pl.col("d") == pl.col("d_previous"))
    checked = steps.join(same, on=["d", "t"], how="inner")
    bad = checked.filter(pl.col("__t_before").is_null() | (pl.col("__t_before") != pl.col("t_previous")))
    if bad.height > 0:
        raise ValueError(
            "online_stretch_wrap: a same-period ramp link joins steps that are not consecutive in the solve: "
            f"{bad.select('d', 't', 't_previous').rows()[:5]}")
    in_run = pl.col("__linked").not_().cum_sum().over("d", order_by="__i")
    runs = (steps.join(same.select("d", "t").with_columns(pl.lit(True).alias("__linked")), on=["d", "t"],
                       how="left")
            .with_columns(pl.col("__linked").fill_null(False))
            .sort("__i")
            .with_columns(in_run.alias("__run")))
    last = (runs.group_by("d", "__run").agg(pl.col("t").sort_by("__i").last().alias("t_previous")))
    starts = runs.join(lk.select("d", "t"), on=["d", "t"], how="anti")
    return (starts.join(last, on=["d", "__run"], how="inner")
            .sort("__i")
            .select("d", "t", pl.col("d").alias("d_previous"), "t_previous"))


def ramp_link_fields(dt: "pl.DataFrame | None", link: "pl.DataFrame | None") -> dict:
    """The FlexData fields derived from the ramp predecessor links: ``dt_ramp_link`` (*link*) and
    ``dt_online_wrap`` (:func:`online_stretch_wrap` of ``dt`` and *link*).  Every setter of ``dt_ramp_link`` sets
    both through this function, so the two never disagree."""
    return {"dt_ramp_link": link, "dt_online_wrap": online_stretch_wrap(dt, link)}


def _with_fan_members(hit: pl.DataFrame, anchors: pl.DataFrame, firsts: pl.DataFrame,
                      period_branch: "pl.DataFrame | None") -> pl.DataFrame:
    """*hit* (the first steps of this roll's periods that take a carry) plus the forecast branch periods of the
    carried *anchors* that start at the same step.  An anchor may be a period label, which has no steps of its
    own (it is not in *firsts*): its branches continue from the carried step all the same."""
    if period_branch is None or period_branch.height == 0 or anchors.height == 0:
        return hit
    pb = (period_branch.select(pl.col("period").cast(pl.Utf8).alias("d"),
                               pl.col("branch").cast(pl.Utf8).alias("b"))
          .filter(pl.col("d") != pl.col("b")))
    branches = (anchors.join(pb, on="d", how="inner")
                .join(firsts.select("d", "t").rename({"d": "b", "t": "t_b"}), on="b", how="inner")
                .filter(pl.col("t_b") == pl.col("t"))
                .select(pl.col("b").alias("d"), "t"))
    return pl.concat([hit, branches]).unique(maintain_order=True)


_END_SUFFIXES = ("rank", "root", "index", "branch")


def _has_end_position(frame: pl.DataFrame, prefix: str) -> bool:
    return all(f"{prefix}{c}" in frame.columns for c in _END_SUFFIXES)


def _end_position(prefix: str) -> "list[pl.Expr]":
    """Select expressions of the base-timeline end position columns ``<prefix>rank / root / index / branch``
    (``solve._carry_routing``) as ``__rank`` (Int64), ``__root`` (Utf8), ``__index`` (Int64), ``__branch`` (Utf8);
    an empty string (a step that cannot be placed) is null."""
    def txt(c):
        v = pl.col(f"{prefix}{c}").cast(pl.Utf8)
        return pl.when(v == "").then(None).otherwise(v)

    return [txt("rank").cast(pl.Int64, strict=False).alias("__rank"), txt("root").alias("__root"),
            txt("index").cast(pl.Int64, strict=False).alias("__index"), txt("branch").alias("__branch")]


def ramp_roll_carry(dt: "pl.DataFrame | None",
                    carried_step: "pl.DataFrame | None",
                    first_step_link: "pl.DataFrame | None",
                    period_branch: "pl.DataFrame | None" = None,
                    flows: "pl.DataFrame | None" = None,
                    rp_block_first: "pl.DataFrame | None" = None,
                    ) -> "tuple[pl.DataFrame | None, pl.DataFrame | None]":
    """The first steps of this roll that ramp from the previous roll's last realized step.

    ``dt`` ``(d, t)`` in solve order.  ``carried_step`` ``[period, step, duration]``: the previous roll's last
    realized step (one row), optionally with ``end_rank, end_root, end_index, end_branch`` (where it ends on the
    base timeline, ``solve._carry_routing.with_end_position``).  ``first_step_link`` ``[period, time,
    previous_period, previous_within_solve, jump]``: the complete (un-rolled) solve's ``step_previous`` rows of the
    first step of each period of this roll, optionally with ``previous_end_rank, previous_end_root,
    previous_end_index, previous_end_branch`` (where that predecessor ends).
    ``period_branch`` ``[period, branch]``: a forecast branch period that starts at the same step as its anchor
    period continues from the same realized step.  ``flows`` ``[process, source, sink, period, step, duration,
    value]``: the previous roll's ``v_flow`` of the ramp-limited arcs at that step (``None`` without ramp limits).
    ``rp_block_first`` ``(d, t)``: representative-block starts never ramp from the step before them.

    A period's first step ``(d, t1)`` takes the carry when the complete solve's predecessor of ``t1`` is
    adjacent (``jump == 1``) and either is the carried step or ends where the carried step ends: the same period
    rank (so the same period, or a branch of it), the same base timeline and the same base-timeline index after
    the step, and, when both are forecast branch periods, the same branch.  So a solve with longer steps continues
    the finer dispatch realized before it (a 6-h step whose predecessor ends at hour 24 continues the realized
    1-h step that ends at hour 24); a hole or an overlap between the two gives no carry.  Without the end
    position columns only the name match applies.  Returns ``(dt_ramp_carry, flows_out)``:
    ``dt_ramp_carry`` = ``(d, t, d_previous, t_previous, duration_previous)`` (also used by the ramp output) and
    ``flows_out`` = ``(p, source, sink, d, t, value, duration)`` (``None`` without ``flows``; ``duration`` the
    arc's own carried step length: a coarse process carries its last realized coarse step), all keys Utf8;
    ``(None, None)`` when no step takes the carry (first roll, a single solve, a carried step that is not
    adjacent, e.g. before a gap in the timeline).
    """
    if (dt is None or dt.height == 0 or carried_step is None or carried_step.height == 0
            or first_step_link is None or first_step_link.height == 0):
        return None, None
    carried = (carried_step.select(pl.col("period").cast(pl.Utf8), pl.col("step").cast(pl.Utf8),
                                   pl.col("duration").cast(pl.Utf8).cast(pl.Float64, strict=False))
               .unique())
    if carried.height != 1:
        raise ValueError(
            "ramp_roll_carry: the carry must come from one step of the previous roll, got "
            f"{carried.rows()}")
    c_d, c_t, c_dur = carried.row(0)
    firsts = (dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
              .with_row_index("__i")
              .group_by("d", maintain_order=True).agg(pl.col("t").sort_by("__i").first()))
    link = first_step_link.select(
        pl.col("period").cast(pl.Utf8).alias("d"), pl.col("time").cast(pl.Utf8).alias("t"),
        pl.col("previous_period").cast(pl.Utf8).alias("d_previous"),
        pl.col("previous_within_solve").cast(pl.Utf8).alias("t_previous"),
        pl.col("jump").cast(pl.Utf8).cast(pl.Float64, strict=False).alias("jump"),
        *(_end_position("previous_end_") if _has_end_position(first_step_link, "previous_end_") else []))
    same = (pl.col("d_previous") == c_d) & (pl.col("t_previous") == c_t)
    c_end = (carried_step.select(_end_position("end_")).unique().row(0)
             if _has_end_position(carried_step, "end_") else None)
    if c_end is not None and None not in c_end[:3] and "__rank" in link.columns:
        c_rank, c_root, c_index, c_branch = c_end
        at_end = ((pl.col("__rank") == c_rank) & (pl.col("__root") == c_root) & (pl.col("__index") == c_index))
        if c_branch is not None:
            at_end = at_end & (pl.col("__branch").is_null() | (pl.col("__branch") == c_branch))
        same = same | at_end.fill_null(False)
    anchors = link.filter((pl.col("jump") == 1.0) & same).select("d", "t")
    hit = _with_fan_members(firsts.join(anchors, on=["d", "t"], how="inner").select("d", "t"),
                            anchors, firsts, period_branch)
    if rp_block_first is not None and rp_block_first.height > 0:
        hit = hit.join(rp_block_first.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique(),
                       on=["d", "t"], how="anti")
    if hit.height == 0:
        return None, None
    link_out = hit.with_columns(d_previous=pl.lit(c_d), t_previous=pl.lit(c_t),
                                duration_previous=pl.lit(c_dur, dtype=pl.Float64))
    if flows is None or flows.height == 0:
        return link_out, None
    f_steps = flows.select(pl.col("period").cast(pl.Utf8), pl.col("step").cast(pl.Utf8)).unique()
    if f_steps.rows() != [(c_d, c_t)]:
        raise ValueError(
            f"ramp_roll_carry: the carried flows are of step(s) {f_steps.rows()}, the carried step is {(c_d, c_t)}")
    duration = (pl.col("duration").cast(pl.Utf8).cast(pl.Float64, strict=False) if "duration" in flows.columns
                else pl.lit(c_dur, dtype=pl.Float64))
    flows_out = (flows.select(pl.col("process").cast(pl.Utf8).alias("p"), pl.col("source").cast(pl.Utf8),
                              pl.col("sink").cast(pl.Utf8),
                              pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0),
                              duration.fill_null(c_dur).alias("duration"))
                 .join(hit, how="cross")
                 .select("p", "source", "sink", "d", "t", "value", "duration"))
    return link_out, flows_out


def check_online_ramp_carry_consistency(dt_online_carry: "pl.DataFrame | None",
                                        p_roll_continue_online_p: "pl.DataFrame | None",
                                        dt_ramp_carry: "pl.DataFrame | None",
                                        p_roll_continue_flow_p: "pl.DataFrame | None",
                                        process_online: "pl.DataFrame | None",
                                        dt_ramp_link: "pl.DataFrame | None") -> None:
    """Raise a clear error when the online carry and the ramp carry / ramp link disagree, instead of silently
    falling back to a predecessor of 0.

    1. Every online process with a ramp-carry flow (``p_roll_continue_flow_p``) must also have an online carry
       (``p_roll_continue_online_p``).  The ramp-row state split reads ``v_startup`` / ``v_shutdown`` relative to
       the predecessor the ramp carry uses; a missing online carry would read a false "0 units online before" at
       a carried step.  The online carry lands on the ramp carry's own steps (``dt_online_carry`` is
       ``dt_ramp_carry``), so the steps agree by construction and only the processes are checked.
    2. A step that takes the online carry must not also be a target of ``dt_ramp_link``: the predecessor is in
       this LP's ``dt`` there, so it cannot also come from the previous solve.

    All frames plain ``(p,)`` / ``(d, t)`` Utf8 (``p_roll_continue_online_p`` / ``p_roll_continue_flow_p``:
    just the ``p`` column of the respective ``Param``'s frame).  No return value; raises ``ValueError`` on a
    violation, otherwise does nothing.
    """
    if dt_ramp_carry is not None and dt_ramp_carry.height > 0 and p_roll_continue_flow_p is not None:
        flow_p = p_roll_continue_flow_p.select(pl.col("p").cast(pl.Utf8)).unique()
        online_p = (process_online.select(pl.col("p").cast(pl.Utf8)).unique()
                    if process_online is not None and process_online.height > 0 else flow_p.head(0))
        at_risk = flow_p.join(online_p, on="p", how="semi")
        rco_p = (p_roll_continue_online_p.select(pl.col("p").cast(pl.Utf8)).unique()
                 if p_roll_continue_online_p is not None else at_risk.head(0))
        missing_p = at_risk.join(rco_p, on="p", how="anti")
        if missing_p.height > 0:
            raise ValueError(
                "check_online_ramp_carry_consistency: process(es) "
                f"{missing_p['p'].to_list()} have a ramp-carry flow but no online carry "
                "(p_roll_continue_online) at the carried step — the ramp split would read a false "
                "online-before-history of 0 there.")
    if (dt_online_carry is not None and dt_online_carry.height > 0
            and dt_ramp_link is not None and dt_ramp_link.height > 0):
        link_targets = dt_ramp_link.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
        oc = dt_online_carry.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
        overlap = oc.join(link_targets, on=["d", "t"], how="semi")
        if overlap.height > 0:
            raise ValueError(
                "check_online_ramp_carry_consistency: step(s) "
                f"{overlap.rows()} are both in dt_online_carry and a dt_ramp_link target — "
                "the two online-predecessor rules must be disjoint.")
