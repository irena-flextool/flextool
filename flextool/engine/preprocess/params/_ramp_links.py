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
    ``flows_out`` = ``(p, source, sink, d, t, value)`` (``None`` without ``flows``), all keys Utf8;
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
    flows_out = (flows.select(pl.col("process").cast(pl.Utf8).alias("p"), pl.col("source").cast(pl.Utf8),
                              pl.col("sink").cast(pl.Utf8),
                              pl.col("value").cast(pl.Float64, strict=False).fill_null(0.0))
                 .join(hit, how="cross")
                 .select("p", "source", "sink", "d", "t", "value"))
    return link_out, flows_out


def online_roll_carry_steps(dt: "pl.DataFrame | None",
                            carried_step: "pl.DataFrame | None",
                            first_step_link: "pl.DataFrame | None",
                            period_branch: "pl.DataFrame | None" = None) -> "pl.DataFrame | None":
    """The step(s) of this roll that take the online-commitment carry (``p_roll_continue_online``) as their
    online predecessor.

    Same inputs as :func:`ramp_roll_carry` (``dt`` in solve order, ``carried_step`` = the previous roll's last
    realized step, ``first_step_link`` = the complete solve's ``step_previous`` rows of the first step of each
    period of this roll, ``period_branch``), but the match does not require ``jump == 1``: a period's first
    step ``(d, t1)`` takes the carry when the complete solve's predecessor of ``t1`` **is** the carried step,
    whatever its jump (a new year still continues the commitment, as today) — plus the forecast-branch periods
    that start at the same step as such a period.

    If no period's complete-solve predecessor matches the carried step (e.g. a nested child solve whose
    complete solve does not contain it), the carry instead lands on the roll's own first step in **timeline
    order** — ``dt``'s row order, which is the solve's chronological order, not a ``(d, t)`` name sort (the bug
    this fixes, N1: a period name that sorts after another, e.g. ``a2025`` after ``p2020``, no longer steals
    the carry from the period that is actually first).

    Returns ``(d, t)`` Utf8, one row per step; ``None`` when there is no carry (first roll, a single solve, or
    ``dt`` is empty).
    """
    if dt is None or dt.height == 0 or carried_step is None or carried_step.height == 0:
        return None
    carried = (carried_step.select(pl.col("period").cast(pl.Utf8), pl.col("step").cast(pl.Utf8)).unique())
    if carried.height != 1:
        raise ValueError(
            "online_roll_carry_steps: the carry must come from one step of the previous roll, got "
            f"{carried.rows()}")
    c_d, c_t = carried.row(0)
    firsts = (dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
              .with_row_index("__i")
              .group_by("d", maintain_order=True)
              .agg(pl.col("t").sort_by("__i").first(), pl.col("__i").sort_by("__i").first()))
    hit = firsts.head(0).select("d", "t")
    if first_step_link is not None and first_step_link.height > 0:
        link = first_step_link.select(
            pl.col("period").cast(pl.Utf8).alias("d"), pl.col("time").cast(pl.Utf8).alias("t"),
            pl.col("previous_period").cast(pl.Utf8).alias("d_previous"),
            pl.col("previous_within_solve").cast(pl.Utf8).alias("t_previous"))
        anchors = (link.filter((pl.col("d_previous") == c_d) & (pl.col("t_previous") == c_t))
                   .select("d", "t"))
        hit = _with_fan_members(firsts.join(anchors, on=["d", "t"], how="inner").select("d", "t"),
                                anchors, firsts.select("d", "t"), period_branch)
    if hit.height == 0:
        # Fallback: the roll's own first step in timeline order (dt's row order), not a (d, t) name sort.
        hit = firsts.sort("__i").head(1).select("d", "t")
    return hit if hit.height > 0 else None


def check_online_ramp_carry_consistency(dt_online_carry: "pl.DataFrame | None",
                                        p_roll_continue_online_p: "pl.DataFrame | None",
                                        dt_ramp_carry: "pl.DataFrame | None",
                                        p_roll_continue_flow_p: "pl.DataFrame | None",
                                        process_online: "pl.DataFrame | None",
                                        dt_ramp_link: "pl.DataFrame | None") -> None:
    """Raise a clear error when the online carry (rule 2, part A) and the ramp carry / ramp link (rule 1,
    part A) disagree, instead of silently falling back to a predecessor of 0 (critique item 2).

    1. Every online process with a ramp-carry flow row at a ``dt_ramp_carry`` step must also have an online
       carry there: the process in ``p_roll_continue_online_p`` and the step in ``dt_online_carry``.  The
       ramp-row state split (part C) reads ``v_startup`` / ``v_shutdown`` relative to the SAME predecessor the
       ramp carry uses; a missing online carry would read a false "0 units online before" at a history-carried
       step instead of raising.
    2. A step that takes the online carry (rule 2) must not also be a target of ``dt_ramp_link`` (rule 1): the
       two rules are defined on disjoint domains (rule 1 only where the predecessor is in this LP's ``dt``,
       rule 2 only where it is not).

    All frames plain ``(p,)`` / ``(d, t)`` Utf8 (``p_roll_continue_online_p`` / ``p_roll_continue_flow_p``:
    just the ``p`` column of the respective ``Param``'s frame).  No return value; raises ``ValueError`` on a
    violation, otherwise does nothing.
    """
    if dt_ramp_carry is not None and dt_ramp_carry.height > 0 and p_roll_continue_flow_p is not None:
        carry_steps = dt_ramp_carry.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
        flow_p = p_roll_continue_flow_p.select(pl.col("p").cast(pl.Utf8)).unique()
        online_p = (process_online.select(pl.col("p").cast(pl.Utf8)).unique()
                    if process_online is not None and process_online.height > 0 else flow_p.head(0))
        at_risk = flow_p.join(online_p, on="p", how="semi")
        if at_risk.height > 0:
            rco_p = (p_roll_continue_online_p.select(pl.col("p").cast(pl.Utf8)).unique()
                     if p_roll_continue_online_p is not None else at_risk.head(0))
            missing_p = at_risk.join(rco_p, on="p", how="anti")
            if missing_p.height > 0:
                raise ValueError(
                    "check_online_ramp_carry_consistency: process(es) "
                    f"{missing_p['p'].to_list()} have a ramp-carry flow but no online carry "
                    "(p_roll_continue_online) at the carried step — the ramp split would read a false "
                    "online-before-history of 0 there.")
            oc_steps = (dt_online_carry.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
                        if dt_online_carry is not None and dt_online_carry.height > 0 else carry_steps.head(0))
            missing_steps = carry_steps.join(oc_steps, on=["d", "t"], how="anti")
            if missing_steps.height > 0:
                raise ValueError(
                    "check_online_ramp_carry_consistency: ramp-carry step(s) "
                    f"{missing_steps.rows()} have no matching online-carry step (dt_online_carry) although an "
                    "online process has a ramp-carry flow there.")
    if (dt_online_carry is not None and dt_online_carry.height > 0
            and dt_ramp_link is not None and dt_ramp_link.height > 0):
        link_targets = dt_ramp_link.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
        oc = dt_online_carry.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
        overlap = oc.join(link_targets, on=["d", "t"], how="semi")
        if overlap.height > 0:
            raise ValueError(
                "check_online_ramp_carry_consistency: step(s) "
                f"{overlap.rows()} are both in dt_online_carry (rule 2) and a dt_ramp_link target (rule 1) — "
                "the two online-predecessor rules must be disjoint.")
