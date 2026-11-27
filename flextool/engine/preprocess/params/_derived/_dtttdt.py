"""§3.9.1, §3.9.3 — step lags: ``dtttdt`` and its jump, period blocks, and the frames derived from
``dtttdt`` (forward only, within solve, block interior and lag)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import _provider_has_key, _provider_read, _try_param
from flextool.engine.preprocess.params._derived._active_steps import _dt_period_active_steps


# ---------------------------------------------------------------------------
# §3.9.1 — dtttdt
# ---------------------------------------------------------------------------


def _roll_first_step_link(workdir: "Path | None", provider: "object | None") -> "pl.DataFrame | None":
    """``solve_data/roll_first_step_link`` (the complete solve's ``step_previous`` of each period's first step);
    ``None`` when absent."""
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "roll_first_step_link.csv"
    if not _provider_has_key(provider, path):
        return None
    return _provider_read(provider, path)


def _rp_block_first_utf8(workdir: "Path | None", provider: "object | None") -> "pl.DataFrame | None":
    """``solve_data/rp_block_first`` as Utf8 ``(d, t)``; ``None`` when absent or empty."""
    if workdir is None:
        return None
    path = Path(workdir) / "solve_data" / "rp_block_first.csv"
    if not _provider_has_key(provider, path):
        return None
    try:
        df = _provider_read(provider, path)
    except Exception:
        return None
    if df.height == 0:
        return None
    return df.select(pl.col("period").cast(pl.Utf8).alias("d"), pl.col("step").cast(pl.Utf8).alias("t"))


def dtttdt_from_source(source: "InputSource",
                          active_solve: str | None,
                          workdir: Path | None = None,
                          *,
                          ctx: "SolveContext | None" = None,
                          provider: "object | None" = None,
                          ) -> pl.DataFrame | None:
    """The ``dtttdt`` lag frame of :func:`dtttdt_with_jump_from_source` without its ``jump`` column."""
    df = dtttdt_with_jump_from_source(source, active_solve, workdir, ctx=ctx, provider=provider)
    return None if df is None else df.drop("jump")


def _timeline_start_hours(source: "InputSource") -> "dict[tuple[str, str], float]":
    """``{(timeline, t): start hour}`` of every raw timeline step (cumulative ``timestep_duration`` in the rank
    order of :func:`_dt_period_active_steps`, i.e. sorted by step name within the timeline)."""
    tl_dur = _try_param(source, "timeline", "timestep_duration")
    if tl_dur is None:
        return {}
    step_col = next((c for c in ("t", "step", "timestep", "x") if c in tl_dur.columns and c not in ("name", "value")),
                    None)
    if step_col is None:
        return {}
    df = (tl_dur.select(pl.col("name").cast(pl.Utf8).alias("tl"), pl.col(step_col).cast(pl.Utf8).alias("t"),
                        pl.col("value").cast(pl.Float64).alias("dur"))
          .sort("tl", "t")
          .with_columns(start=(pl.col("dur").cum_sum().over("tl") - pl.col("dur"))))
    return {(tl, t): h for tl, t, h in df.select("tl", "t", "start").iter_rows()}


def dtttdt_with_jump_from_source(source: "InputSource",
                                 active_solve: str | None,
                                 workdir: Path | None = None,
                                 *,
                                 ctx: "SolveContext | None" = None,
                                 provider: "object | None" = None,
                                 step_duration: "Param | None" = None,
                                 ) -> pl.DataFrame | None:
    """Build the canonical ``(d, t, t_previous, t_previous_within_timeset,
    d_previous, t_previous_within_solve)`` dispatch-step lag frame.

    Mirrors ``timeline_config.make_step_jump`` (default-block path):

      * Cyclic within solve: first step of the first realized period
        wraps to the last step of the last realized period.
      * Cyclic within timeset: the ``t_previous_within_timeset`` column
        is the block's last step for first-of-period rows, else
        identical to ``t_previous`` for jump=1 interior steps; for
        jump>1 transitions (block boundaries) it pins on the prior
        block's last step.
      * Within-period predecessor: previous step in the same period
        cyclically (first wraps to block_last of the period).

    The extra ``jump`` column is ``make_step_jump``'s: the rank difference to the predecessor
    ``(d_previous, t_previous_within_solve)`` in the solve's own timeline (1 for adjacent steps, also aggregated
    ``new_stepduration`` steps; ≤ 0 at a cyclic wrap, > 1 across a gap).  When the solve's timeline is not
    available (raw ranks), 1 where the predecessor ends where the step starts on the timeline (with
    ``step_duration``, the solve's step lengths), otherwise the raw rank difference.
    """
    info = _dt_period_active_steps(source, active_solve, workdir, ctx=ctx, provider=provider)
    if info is None:
        return None
    per_period = info["per_period"]
    period_order = info["period_order"]
    if not period_order:
        return None
    timeline_of = info.get("timeline_for_period") or {}
    hours = _timeline_start_hours(source) if step_duration is not None else {}
    durs: dict[tuple[str, str], float] = {}
    if hours and step_duration is not None:
        durs = {(str(d), str(t)): float(v) for d, t, v in
                step_duration.frame.select("d", "t", "value").iter_rows()}

    step_space = bool(info.get("step_space"))

    def _jump(d_cur: str, t_cur: str, rk_cur: int, d_prev: str, t_prev: str, rk_prev: int) -> int:
        raw = rk_cur - rk_prev
        if step_space or not durs:
            return raw
        tl_c, tl_p = timeline_of.get(d_cur), timeline_of.get(d_prev)
        h_c, h_p = hours.get((tl_c, t_cur)), hours.get((tl_p, t_prev))
        dur_p = durs.get((d_prev, t_prev))
        if tl_c is None or tl_c != tl_p or h_c is None or h_p is None or dur_p is None:
            return raw
        if abs(h_c - (h_p + dur_p)) < 1e-9:
            return 1
        return raw if raw != 1 else 2

    # Per-period anchor lookup: branches whose anchor != self use a
    # per-branch self-loop for the period-wrap predecessor.  Read
    # period__branch.csv (workdir-aware) to discover which periods are
    # branches off a non-self anchor.
    branch_anchor: dict[str, str] = {}
    anchors_with_self: set[str] = set()
    anchor_order: list[str] = []
    pbf: pl.DataFrame | None = None
    period_col, branch_col = "period", "branch"
    # Path B Cat B: prefer ctx.period_branch (schema [d_anchor, b]); it is
    # null-free (under axis Enums its loader raises on a label outside the
    # ``d_anchor`` / ``branch`` vocabulary).  Without a ctx frame, read the
    # provider/workdir CSV.
    if ctx is not None:
        pb_ctx = ctx.period_branch
        if pb_ctx.height > 0:
            pbf = pb_ctx
            period_col, branch_col = "d_anchor", "b"
    if pbf is None and workdir is not None:
        pb_path = Path(workdir) / "solve_data" / "period__branch.csv"
        if _provider_has_key(provider, pb_path):  # Phase E-j — seed-aware
            try:
                pbf = _provider_read(provider, pb_path)
            except Exception:
                pbf = None
    if pbf is not None and pbf.height > 0:
        pb_rows = list(zip(pbf[period_col].to_list(),
                              pbf[branch_col].to_list()))
        for a, b in pb_rows:
            if a == b:
                anchors_with_self.add(a)
                anchor_order.append(a)
        for a, b in pb_rows:
            if b != a and a in anchors_with_self:
                branch_anchor[b] = a
    # Continuation fan-out: the time-branch map (period -> time_branch,
    # from solve_branch__time_branch.csv) lets a continuation branch
    # period wrap to the nearest earlier period of the SAME time-branch
    # instead of self-wrapping, mirroring ``make_step_jump``.  Ctx-first
    # (typed accessor), provider/workdir fallback.
    time_branch_of: dict[str, str] = {}
    if ctx is not None:
        sbtb_ctx = ctx.solve_branch_time_branch
        if sbtb_ctx.height > 0:
            time_branch_of = dict(zip(sbtb_ctx["d"].to_list(),
                                        sbtb_ctx["time_branch"].to_list()))
    if not time_branch_of and workdir is not None:
        sbtb_path = (Path(workdir) / "solve_data"
                     / "solve_branch__time_branch.csv")
        if _provider_has_key(provider, sbtb_path):  # Phase E-j — seed-aware
            try:
                sbtb_df = _provider_read(provider, sbtb_path)
            except Exception:
                sbtb_df = None
            if (sbtb_df is not None and sbtb_df.height > 0
                    and "period" in sbtb_df.columns
                    and "branch" in sbtb_df.columns):
                time_branch_of = dict(zip(sbtb_df["period"].to_list(),
                                            sbtb_df["branch"].to_list()))
    if branch_anchor and not time_branch_of:
        # Failure semantics (do NOT fall back to self-wrap): a
        # period__branch with branch rows but no time-branch map means
        # the per-solve preprocessing artefacts are inconsistent.  A
        # silent self-wrap fallback would sever continuation branch
        # periods from their own branch chain — the exact silent
        # truncation bug this walk repairs — so raise instead.
        raise ValueError(
            "dtttdt_from_source: period__branch declares branch periods "
            f"({sorted(branch_anchor)}) but solve_branch__time_branch is "
            "missing or empty — cannot resolve cross-period wrap "
            "predecessors for branch periods.  The per-solve emitter "
            "(emit_branch_weights_and_map) writes this frame for every "
            "solve; its absence indicates an inconsistent workdir/"
            "Provider state."
        )
    out_rows: list[tuple[str, str, str, str, str, str]] = []
    len(period_order)
    # Find within-solve predecessor for first-of-period rows.
    for pi, period in enumerate(period_order):
        steps = per_period.get(period) or []
        if not steps:
            continue
        # Determine the predecessor period for the j=0 wrap row.
        # Mirrors ``timeline_config.make_step_jump``:
        #   * branch (non-anchor) period → nearest earlier period with
        #     the SAME time-branch (continuation linkage); self-wrap
        #     when none exists (the branching period's own fan);
        #   * first anchor period (== first_period_name in flextool's
        #     reversed walk) → wrap to last_period_name (last in
        #     active_time_list iteration order, == period_order[-1] —
        #     retained quirk);
        #   * non-first anchor period → nearest earlier ANCHOR period
        #     (skips branch periods so a realized continuation period
        #     wraps to the previous realized period); positional
        #     ``period_order[pi - 1]`` fallback when there is no
        #     period__branch data (deterministic path, where
        #     positional ≡ previous-anchor anyway).
        if period in branch_anchor:
            wrap_period = period
            tb = time_branch_of.get(period)
            if tb is not None:
                for qi in range(pi - 1, -1, -1):
                    if time_branch_of.get(period_order[qi]) == tb:
                        wrap_period = period_order[qi]
                        break
            if wrap_period == period:
                # No earlier same-time-branch period exists.  MID-HORIZON
                # reveal: this branch copy's own anchor period is not the
                # first period, so link its first step to the nearest
                # anchor STRICTLY BEFORE the copy's own anchor — the copy
                # inherits the shared pre-reveal trunk end-state (two-stage
                # storage-state non-anticipativity, Slice E design §5.2).
                # FIRST-PERIOD fan: the copy's anchor IS the first period
                # → empty range → wrap_period stays == period (self-wrap,
                # byte-parity with the pre-Slice-E behaviour).  This mirror
                # of ``make_step_jump`` is pinned by the Step-0.5 parity
                # guard.
                p_anchor = branch_anchor.get(period)
                anchor_steps = per_period.get(p_anchor) or [] if p_anchor is not None else []
                if p_anchor in period_order and anchor_steps and anchor_steps[-1][1] < steps[0][1]:
                    # The anchor is a period label whose steps end before the copy's (a split realization roll:
                    # it holds the jump): the copy continues from its last step.
                    wrap_period = p_anchor
                elif p_anchor is not None and p_anchor in period_order:
                    pos_anchor = period_order.index(p_anchor)
                    for qi in range(pos_anchor - 1, -1, -1):
                        if period_order[qi] in anchors_with_self:
                            wrap_period = period_order[qi]
                            break
                elif p_anchor is not None and p_anchor in anchor_order:
                    # The anchor is a period label (no steps, not in period_order): walk the anchors in
                    # period__branch order back from it to the nearest one with steps.
                    for a in reversed(anchor_order[:anchor_order.index(p_anchor)]):
                        if a in period_order:
                            wrap_period = a
                            break
        else:
            if pi == 0:
                wrap_period = period_order[-1]
            else:
                wrap_period = period_order[pi - 1]
                for qi in range(pi - 1, -1, -1):
                    if period_order[qi] in anchors_with_self:
                        wrap_period = period_order[qi]
                        break
        prev_solve_steps = per_period.get(wrap_period) or []
        prev_solve_last_t = prev_solve_steps[-1][0] if prev_solve_steps \
                            else steps[-1][0]
        # Within-period: cyclic over current period.
        steps[-1][0]
        n = len(steps)
        # Track running block_last_t as we walk forward from j=0 to j=n-1.
        # In flextool's reversed loop, block_last initialises to len-1 and
        # decreases on jump>1 transitions.  Equivalent forward pass: the
        # block_last for index j is the index of the next jump>1 boundary
        # to the right (or the last index n-1 if none).
        # Pre-compute for each j, the block_last index (forward sweep).
        block_last_idx_for: list[int] = [n - 1] * n
        # Find boundaries: indices where ranks[j] - ranks[j-1] > 1.
        # block_last_idx_for[k] = block_last for indices in current block.
        # We iterate from the right, tracking the current block's last index.
        cur_last = n - 1
        for k in range(n - 1, 0, -1):
            t_k, rk_k = steps[k]
            t_prev, rk_prev = steps[k - 1]
            block_last_idx_for[k] = cur_last
            if rk_k - rk_prev > 1:
                # Boundary — next k's block_last is k-1.
                cur_last = k - 1
        block_last_idx_for[0] = cur_last
        for j in range(n):
            t_cur, rk_cur = steps[j]
            if j == 0:
                # First step of the period — cross-period wrap.  Use
                # wrap_period computed above (handles branch self-wrap
                # vs anchor cyclic-chain).
                prev_p = wrap_period
                prev_p_steps = per_period.get(prev_p) or []
                prev_t = prev_p_steps[-1][0] if prev_p_steps else t_cur
                # Block's last step in current period — specifically the
                # last step of the block CONTAINING the current step
                # (block_last_idx_for[0]).
                bli0 = block_last_idx_for[0]
                prev_within_ts = steps[bli0][0]
                prev_within_solve = prev_solve_last_t
                d_prev = prev_p
                jump = (_jump(period, t_cur, rk_cur, prev_p, prev_p_steps[-1][0], prev_p_steps[-1][1])
                        if prev_p_steps else 0)
            else:
                t_prev, rk_prev = steps[j - 1]
                prev_t = t_prev
                jump = rk_cur - rk_prev
                if jump > 1:
                    # Block boundary — pin within_ts to the last step of
                    # the block CONTAINING the current step (j), per
                    # ``make_step_jump``'s reverse-walk semantics.  The
                    # outgoing-block last index would be (j-1) but the
                    # canonical algorithm pins it to active_time[block_last]
                    # tracked in reverse-iteration order which equals the
                    # last index of the block containing j.
                    bli = block_last_idx_for[j]
                    prev_within_ts = steps[bli][0]
                else:
                    prev_within_ts = t_prev
                d_prev = period
                prev_within_solve = t_prev
                jump = _jump(period, t_cur, rk_cur, period, t_prev, rk_prev)
            out_rows.append((period, t_cur, prev_t, prev_within_ts,
                              d_prev, prev_within_solve, jump))
    if not out_rows:
        return None
    df = pl.DataFrame(
        out_rows,
        schema=["d", "t", "t_previous", "t_previous_within_timeset",
                  "d_previous", "t_previous_within_solve", "jump"],
        orient="row",
    )
    return df.sort("d", "t")


# ---------------------------------------------------------------------------
# §3.9.3 — period_block / period_block_succ / period_block_time
# ---------------------------------------------------------------------------


def period_block_family_from_source(source: "InputSource",
                                       active_solve: str | None,
                                       workdir: Path | None = None,
                                       *,
                                       ctx: "SolveContext | None" = None,
                                       provider: "object | None" = None,
                                       realized: "pl.DataFrame | None" = None,
                                       ) -> dict | None:
    """``period_block`` ``(d, b_first)``, ``period_block_succ`` ``(d, b_first, b_next)`` and
    ``period_block_time`` ``(d, b_first, t)``: the timeset blocks of each period (mirrors
    ``timeline_config.make_period_block``) for the ``bind_intraperiod_blocks`` nodes.  Per period the active
    steps are taken in rank order; a new block starts where the rank gap to the previous step exceeds 1.

    *realized* ``(d, t)`` (a rolling solve's realized steps): a block also ends at the last realized step of a
    period that continues past it, so the state the roll hands over is the end state of a block (the same cut as
    the coarse steps, ``params._coarse.coarse_node_frames``).

    Returns the three frames, or None when the source is insufficient.
    """
    info = _dt_period_active_steps(source, active_solve, workdir, ctx=ctx, provider=provider)
    if info is None:
        return None
    per_period = info["per_period"]
    period_order = info["period_order"]
    realized_set: set[tuple[str, str]] = set()
    if realized is not None and realized.height > 0:
        realized_set = set(realized.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).iter_rows())
    pb_rows: list[tuple[str, str]] = []         # (d, b_first)
    pbs_rows: list[tuple[str, str, str]] = []   # (d, b_first, b_next)
    pbt_rows: list[tuple[str, str, str]] = []   # (d, b_first, t)
    for period in period_order:
        steps = per_period.get(period) or []
        if not steps:
            continue
        # Index of the period's last realized step when the period continues past it (rolling solves only).
        cut_after = None
        if realized_set:
            last = max((j for j, (t, _rk) in enumerate(steps) if (str(period), str(t)) in realized_set), default=None)
            if last is not None and last < len(steps) - 1:
                cut_after = last
        block_firsts: list[str] = [steps[0][0]]
        cur_b = steps[0][0]
        for j, (t, rk) in enumerate(steps):
            if j > 0 and (rk - steps[j - 1][1] > 1 or j - 1 == cut_after):
                cur_b = t
                block_firsts.append(cur_b)
            pbt_rows.append((period, cur_b, t))
        for i, b in enumerate(block_firsts):
            pb_rows.append((period, b))
            nxt = block_firsts[(i + 1) % len(block_firsts)]
            pbs_rows.append((period, b, nxt))
    period_block = (pl.DataFrame(pb_rows, schema=["d", "b_first"],
                                    orient="row").unique(maintain_order=True)
                    if pb_rows else None)
    period_block_succ = (pl.DataFrame(pbs_rows,
                                          schema=["d", "b_first", "b_next"],
                                          orient="row")
                          if pbs_rows else None)
    period_block_time = (pl.DataFrame(pbt_rows,
                                          schema=["d", "b_first", "t"],
                                          orient="row").unique(maintain_order=True)
                          if pbt_rows else None)

    return dict(
        period_block=period_block,
        period_block_succ=period_block_succ,
        period_block_time=period_block_time,
    )


# ---------------------------------------------------------------------------
# §3.9 helper compose: dtttdt-derived auxiliary frames
# ---------------------------------------------------------------------------


def dtttdt_forward_only_from_dtttdt(dtttdt: pl.DataFrame | None,
                                    first_periods: "list[str] | None" = None,
                                    ) -> pl.DataFrame | None:
    """``dtttdt`` sorted by (d, t), .slice(1) — drops the first wrap row
    so bind_forward_only nodes don't carry a state-change term across
    the very first step of the first period (mod:2188).

    *first_periods* are the periods of ``period_first_of_solve``.  When a
    stochastic fan opens in the solve's first period there are several
    (its forecast branch periods; the period label has no steps): every
    one of them starts the solve, so the first step of each is dropped.
    """
    if dtttdt is None or dtttdt.height == 0:
        return None
    out = dtttdt.sort("d", "t").slice(1)
    if first_periods is not None and len(first_periods) > 1:
        firsts = (dtttdt.filter(pl.col("d").cast(pl.Utf8).is_in([str(p) for p in first_periods]))
                  .group_by("d").agg(pl.col("t").min()))
        out = out.join(firsts, on=["d", "t"], how="anti")
    return out if out.height > 0 else None


def dtttdt_within_solve_from_dtttdt(dtttdt: pl.DataFrame | None,
                                    first_periods: "list[str] | None" = None,
                                    ) -> pl.DataFrame | None:
    """``dtttdt`` with the bind_within_solve wrap of a stochastic fan in the solve's first period; None when the
    plain ``dtttdt`` wrap is already the bind_within_solve one.

    *first_periods* are the periods of ``period_first_of_solve``: with a fan in the solve's first period, its
    forecast branch periods (the roots; the period label has no steps).  ``dtttdt`` (``make_step_jump``) wraps the
    first step of each root to the root's own last step.  bind_within_solve closes one loop per path instead: the
    first step of each root continues from the last step of the last period on its own path (the leaf the root's
    continuation branch periods lead to; with several, the last in name order).  A root that is its own leaf (a
    single-period solve) keeps its wrap, so only a fan in the first period of a multi-period solve gives a frame.
    The tree is read from the first-step rows of ``dtttdt``: a period whose first step continues from another
    period is its child.  A period's first and last steps are its smallest and largest time steps (as
    :func:`dtttdt_forward_only_from_dtttdt` takes the first).
    """
    if dtttdt is None or dtttdt.height == 0 or first_periods is None or len(first_periods) < 2:
        return None
    roots = [str(p) for p in first_periods]
    u = dtttdt.select(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8),
                      _dp=pl.col("d_previous").cast(pl.Utf8))
    span = u.group_by("_d").agg(_first=pl.col("_t").min(), _last=pl.col("_t").max())
    last = dict(span.select("_d", "_last").iter_rows())
    first = dict(span.select("_d", "_first").iter_rows())
    children: dict[str, list[str]] = {}
    for d_name, d_prev in (u.join(span.select("_d", _t=pl.col("_first")), on=["_d", "_t"], how="semi")
                           .select("_d", "_dp").unique().iter_rows()):
        if d_prev != d_name and d_name not in roots:
            children.setdefault(d_prev, []).append(d_name)

    def leaf_of(root: str) -> str:
        leaves, seen, stack = [], set(), [root]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            kids = children.get(node, [])
            if kids:
                stack.extend(kids)
            else:
                leaves.append(node)
        # Convention: with several leaves, the last in sorted name order (as core._storage_state._eq_storage_within_solve_path_end
        # picks d*).
        return max(leaves)

    relink = {r: leaf_of(r) for r in roots if r in last}
    relink = {r: leaf for r, leaf in relink.items() if leaf != r}
    if not relink:
        return None
    wrap = pl.DataFrame({"_d": list(relink), "_t": [first[r] for r in relink],
                         "_dp_new": list(relink.values()), "_tp_new": [last[leaf] for leaf in relink.values()]},
                        schema={"_d": pl.Utf8, "_t": pl.Utf8, "_dp_new": pl.Utf8, "_tp_new": pl.Utf8})
    dp_dtype, tp_dtype = dtttdt.schema["d_previous"], dtttdt.schema["t_previous_within_solve"]
    return (dtttdt.with_columns(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
            .join(wrap, on=["_d", "_t"], how="left")
            .with_columns(
                d_previous=pl.coalesce(pl.col("_dp_new"), pl.col("d_previous").cast(pl.Utf8)).cast(dp_dtype),
                t_previous_within_solve=pl.coalesce(pl.col("_tp_new"),
                                                    pl.col("t_previous_within_solve").cast(pl.Utf8)).cast(tp_dtype))
            .select(dtttdt.columns))


def read_period_first_of_solve(sd: Path, *, provider: "object | None" = None) -> "list[str] | None":
    """The periods of ``solve_data/period_first_of_solve.csv``; None when absent."""
    p = Path(sd) / "period_first_of_solve.csv"
    if not _provider_has_key(provider, p):
        return None
    df = _provider_read(provider, p)
    if df.height == 0 or "period" not in df.columns:
        return None
    return [str(x) for x in df["period"].to_list()]


def dtttdt_block_interior_from_dtttdt(dtttdt: pl.DataFrame | None,
                                          period_block_time: pl.DataFrame
                                              | None = None,
                                          ) -> pl.DataFrame | None:
    """Interior-of-block dtttdt rows of the ``bind_intraperiod_blocks`` timeset blocks.

      1. **One block per period** (or no *period_block_time*): keep dtttdt rows where
         ``t_previous_within_timeset == t_previous`` (jump=1 interior, not a boundary).
      2. **Several timeset blocks per period**: for each (d, b_first) of *period_block_time*, consecutive
         sorted t's give (d, t, t_previous) — the same set as branch 1, built from the blocks.
    """
    if dtttdt is None or dtttdt.height == 0:
        return None
    multi_res = False
    if period_block_time is not None and period_block_time.height > 0:
        # Several timeset blocks in a period: more than one distinct b_first per d.
        nb = (period_block_time
              .group_by("d")
              .agg(pl.col("b_first").n_unique().alias("nb"))
              ["nb"].max())
        if nb is not None and nb > 1:
            multi_res = True
    if multi_res and period_block_time is not None:
        rows: list[tuple[str, str, str]] = []
        pbt_sorted = period_block_time.sort("d", "b_first", "t")
        for (dval, bf), grp in pbt_sorted.group_by(
                ["d", "b_first"], maintain_order=True):
            ts = grp["t"].to_list()
            for i in range(1, len(ts)):
                rows.append((dval, ts[i], ts[i - 1]))
        if not rows:
            return None
        return (pl.DataFrame(rows, schema=["d", "t", "t_previous"],
                                orient="row").unique(maintain_order=True))
    if "t_previous_within_timeset" not in dtttdt.columns:
        return None
    out = (dtttdt
            .filter(pl.col("t_previous_within_timeset")
                    == pl.col("t_previous"))
            .select("d", "t", "t_previous"))
    return out if out.height > 0 else None


def dtttdt_block_lag_from_dtttdt(dtttdt: pl.DataFrame | None,
                                 period_block_time: pl.DataFrame | None,
                                 chained: pl.DataFrame | None = None,
                                 period_branch: pl.DataFrame | None = None,
                                 ) -> pl.DataFrame | None:
    """``(d, t, d_previous, t_previous_within_solve)``: the step each step of a ``bind_intraperiod_blocks`` block
    continues from (the state change of ``nodeBalanceBlock_eq``).

    The blocks of a period form a loop: inside the period the previous step (across block boundaries too), and the
    period's first step continues from the period's own last step (the within-period lag of
    ``flex_data._readers._within_period_lag``).  The periods of *chained* ``(d)`` (in a rolling solve, the periods the roll
    holds only in part) chain forward instead: their rows are the bind_forward_only lag
    (:func:`dtttdt_forward_only_from_dtttdt`), so a period's first step continues from the previous step of the
    solve and the solve's first step has no row (its start is the fix_start state or the roll carry).  Restricted to
    the steps of *period_block_time*.  *period_branch* ``(d, b)``: in a split realization roll a looping period is
    one loop per path (the label's jump and one branch period's tail), as the within-period lag gives it.
    """
    if dtttdt is None or dtttdt.height == 0 or period_block_time is None or period_block_time.height == 0:
        return None
    from flextool.engine.preprocess.flex_data._readers import _within_period_lag

    lag = _within_period_lag(dtttdt, period_branch)
    if lag is None:
        return None
    loop = lag.select("d", "t", "d_previous", pl.col("t_previous_within_period").alias("t_previous_within_solve"))
    if chained is not None and chained.height > 0:
        names = [str(x) for x in chained["d"].to_list()]
        loop = loop.filter(~pl.col("d").cast(pl.Utf8).is_in(names))
        fwd = dtttdt_forward_only_from_dtttdt(dtttdt)
        if fwd is not None:
            chain = (fwd.filter(pl.col("d").cast(pl.Utf8).is_in(names))
                     .select([pl.col(c).cast(loop.schema[c], strict=False) for c in loop.columns]))
            loop = pl.concat([loop, chain], how="vertical")
    steps = period_block_time.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
    out = (loop.with_columns(_d=pl.col("d").cast(pl.Utf8), _t=pl.col("t").cast(pl.Utf8))
           .join(steps.rename({"d": "_d", "t": "_t"}), on=["_d", "_t"], how="semi")
           .select("d", "t", "d_previous", "t_previous_within_solve")
           .sort("d", "t"))
    return out if out.height > 0 else None


def _rolling_realized_steps(flex_data: object, workdir: Path | None,
                            provider: "object | None") -> pl.DataFrame | None:
    """``(d, t)`` the solve realizes when it is a rolling sub-solve (``flex_data.solve_is_rolling``), else None:
    where the bind_intraperiod_blocks blocks are cut (:func:`period_block_family_from_source`)."""
    if not getattr(flex_data, "solve_is_rolling", False) or workdir is None:
        return None
    from flextool.engine.preprocess.flex_data._coarse import _realized_steps

    return _realized_steps(Path(workdir) / "solve_data", flex_data, provider=provider)
