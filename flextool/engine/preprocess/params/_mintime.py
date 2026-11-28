"""Minimum up/down-time look-back: which earlier steps a step's minimum-time constraint sums over.

The look-back of a step ``(d, t)`` is the step itself plus its ancestors along the ramp predecessor links
(``dt_ramp_link``: the predecessor that is adjacent on the timeline), as long as the time elapsed from the start of
the ancestor to the start of ``(d, t)`` stays below the minimum time.  The chain stops at a step without such a
predecessor: the first step of a solve, a period that does not continue the previous one, a gap in the timeline, the
start of a representative period and the first step of a forecast branch that starts the solve.  There is no
cyclic wrap.  Where the chain stops, the units online start free (the first step of a solve or of a period that
does not continue the previous one: no start-up or shut-down there) or cycle within their stretch (after a gap, at a
representative-block start); a unit online at such a step may then shut down at the next step — an end effect, as
at the end of the horizon.

The walk is vectorized over the depth of the chain (iterations = the number of steps inside the longest minimum
time), on integer step indices; the step names are attached once at the output.
"""
from __future__ import annotations

import logging

import polars as pl

logger = logging.getLogger(__name__)

LOOKBACK_COLS = ("p", "d", "t", "d_back", "t_back")


def _step_durations(dt: pl.DataFrame, step_duration: "object | None") -> pl.DataFrame:
    """``(i, d, t, dur)`` of the distinct steps of *dt* (Utf8 names, solve order); a step without a duration
    counts 1 h."""
    steps = dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
    if "step_duration" in dt.columns:
        steps = steps.with_columns(dur=dt["step_duration"].cast(pl.Float64))
    else:
        sd = step_duration.frame if hasattr(step_duration, "frame") else step_duration
        if sd is None:
            steps = steps.with_columns(dur=pl.lit(1.0, dtype=pl.Float64))
        else:
            sd_u = (sd.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                              pl.col("value").cast(pl.Float64).alias("dur"))
                    .unique(subset=["d", "t"], keep="first"))
            steps = steps.join(sd_u, on=["d", "t"], how="left", maintain_order="left")
    return (steps.with_columns(pl.col("dur").fill_null(1.0))
            .unique(subset=["d", "t"], keep="first", maintain_order=True)
            .with_row_index("i"))


def _edges(steps: pl.DataFrame, link: "pl.DataFrame | None") -> pl.DataFrame:
    """``(i, j, dur_j)``: step ``j`` is the ramp predecessor of step ``i`` (both steps of *steps*; a predecessor that
    is not one of them is dropped), ``dur_j`` its duration."""
    if link is None or link.height == 0:
        return pl.DataFrame(schema={"i": pl.UInt32, "j": pl.UInt32, "dur_j": pl.Float64})
    idx = steps.select("i", "d", "t")
    prev = steps.select(pl.col("i").alias("j"), pl.col("d").alias("d_previous"), pl.col("t").alias("t_previous"),
                        pl.col("dur").alias("dur_j"))
    return (link.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                        pl.col("d_previous").cast(pl.Utf8), pl.col("t_previous").cast(pl.Utf8))
            .unique()
            .join(idx, on=["d", "t"], how="inner")
            .join(prev, on=["d_previous", "t_previous"], how="inner")
            .select("i", "j", "dur_j"))


def _walk(start: pl.DataFrame, edges: pl.DataFrame, limit: float, n_steps: int, *, forward: bool = False
          ) -> pl.DataFrame:
    """``start`` ``(i, back, elapsed)`` plus every row reached from it along *edges* while ``elapsed < limit``.

    Backwards (default): ``back`` moves to its predecessor ``j`` and ``elapsed`` grows by ``dur_j``.  Forwards:
    ``i`` moves to a successor (a step whose predecessor ``i`` is) and ``elapsed`` grows by the duration of the
    step left.  One iteration per step of depth; more than ``n_steps`` iterations means the links form a cycle."""
    pairs = [start]
    if forward:
        step = edges.select(pl.col("j").alias("__from"), pl.col("i").alias("__to"), "dur_j")
        key = "i"
    else:
        step = edges.select(pl.col("i").alias("__from"), pl.col("j").alias("__to"), "dur_j")
        key = "back"
    frontier = start
    depth = 0
    while True:
        frontier = (frontier.join(step, left_on=key, right_on="__from", how="inner")
                    .with_columns(pl.col("__to").alias(key), (pl.col("elapsed") + pl.col("dur_j")).alias("elapsed"))
                    .select("i", "back", "elapsed")
                    .filter(pl.col("elapsed") < limit))
        if frontier.height == 0:
            break
        depth += 1
        if depth > n_steps:
            raise ValueError("mintime: the ramp predecessor links (dt_ramp_link) form a cycle")
        pairs.append(frontier)
    return pl.concat(pairs)


def _solve_steps(dt: pl.DataFrame, step_duration: "object | None", link: "pl.DataFrame | None",
                 caller: str) -> "tuple[pl.DataFrame, pl.DataFrame]":
    """``(steps, edges)`` of a solve (:func:`_step_durations`, :func:`_edges`); raises ``ValueError`` when *link*
    is ``None`` although the solve has several steps (the predecessor links were not built)."""
    steps = _step_durations(dt, step_duration)
    if link is None and steps.height > 1:
        raise ValueError(
            f"{caller}: the solve has several time steps but no ramp predecessor links "
            "(dt_ramp_link); the minimum up/down-time look-back cannot be built")
    return steps, _edges(steps, link)


def mintime_lookback(dt: "pl.DataFrame | None", step_duration: "object | None",
                     link: "pl.DataFrame | None", procs: "list[tuple[str, float]]") -> "pl.DataFrame | None":
    """The ``(p, d, t, d_back, t_back)`` look-back rows (Utf8, sorted) of the processes *procs* ``[(p, L)]``.

    ``dt`` ``(d, t)``: the steps of this solve.  ``step_duration``: a ``Param`` or frame ``(d, t, value)`` (hours),
    or ``None`` (a ``step_duration`` column of *dt* wins).  ``link`` ``(d, t, d_previous, t_previous)``: the ramp
    predecessor of each step that has one (``dt_ramp_link``).  A predecessor that is not a step of this solve ends
    the chain.

    Rows: ``(p, d, t, d, t)`` for every step, and ``(p, d, t, d′, t′)`` for every ancestor ``(d′, t′)`` with
    ``elapsed(d′, t′ → d, t) < L``, where ``elapsed`` sums the durations of the chain from ``(d′, t′)`` up to, not
    including, ``(d, t)``.  ``None`` when no process has a positive minimum time or *dt* is empty.  Raises
    ``ValueError`` when *link* is ``None`` although the solve has several steps (the predecessor links were not
    built), or when the links form a cycle.
    """
    procs = [(str(p), float(L)) for p, L in procs if float(L) > 0]
    if dt is None or dt.height == 0 or not procs:
        return None
    steps, edges = _solve_steps(dt, step_duration, link, "mintime_lookback")
    l_max = max(L for _, L in procs)
    start = steps.select("i", pl.col("i").alias("back"), pl.lit(0.0, dtype=pl.Float64).alias("elapsed"))
    all_pairs = _walk(start, edges, l_max, steps.height)
    by_l: dict[float, list[str]] = {}
    for p, L in procs:
        by_l.setdefault(L, []).append(p)
    parts = []
    for L, names in by_l.items():
        within = all_pairs.filter(pl.col("elapsed") < L).select("i", "back")
        parts.append(pl.DataFrame({"p": sorted(set(names))}, schema={"p": pl.Utf8}).join(within, how="cross"))
    rows = pl.concat(parts)
    if rows.height == 0:
        return None
    idx = steps.select("i", "d", "t")
    back = idx.select(pl.col("i").alias("back"), pl.col("d").alias("d_back"), pl.col("t").alias("t_back"))
    return (rows.join(idx, on="i", how="inner").join(back, on="back", how="inner")
            .select(*LOOKBACK_COLS)
            .sort(*LOOKBACK_COLS))


# ---------------------------------------------------------------------------------------------------------------
# Roll carry: the realized start-ups / shut-downs of the previous solve that a minimum time still covers
# ---------------------------------------------------------------------------------------------------------------

HISTORY_COLS = ("p", "d", "t", "startup", "shutdown", "duration", "hours_to_end")


def history_hours_to_end(cells: pl.DataFrame, link: "pl.DataFrame | None", last: "tuple[str, str]",
                         l_max: float) -> "pl.DataFrame | None":
    """``(d, t, hours_to_end)`` Utf8: the realized step *last* and its ancestors along *link* among the realized
    *cells* ``(d, t, dur)``, each with the hours from its start to the end of *last*, while those stay below
    *l_max*.

    ``hours_to_end(last) = dur(last)``; an ancestor adds the durations of the chain from it up to *last*.  The chain
    stops where :func:`mintime_lookback` stops (a step without a predecessor among *cells*).  ``None`` when *last*
    is not a cell or ``dur(last) ≥ l_max`` (no minimum time reaches past the end).  Raises ``ValueError`` when
    *link* is ``None`` although there are several realized cells (the predecessor links were not built, so the
    history cannot be walked; the consumer raises in the same case, :func:`mintime_lookback`).

    The cells are the solve's fine steps, also where a coarse process has one start-up / shut-down per coarse step:
    its readers (``engine.output._coarse_flows``, mode ``first``) put the event at the step's first fine step and 0
    at the others.  The event happens at the start of its coarse step, so the fine ``hours_to_end`` of that first
    step is exact, and the last realized step (an interior step of a coarse step) keeps its row."""
    steps = (cells.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8), pl.col("dur").cast(pl.Float64))
             .unique(subset=["d", "t"], keep="first", maintain_order=True)
             .with_row_index("i"))
    if link is None and steps.height > 1:
        raise ValueError(
            "history_hours_to_end: the solve realized several time steps but has no ramp predecessor links "
            "(dt_ramp_link); the minimum up/down-time history cannot be built")
    hit = steps.filter((pl.col("d") == last[0]) & (pl.col("t") == last[1]))
    if hit.height == 0:
        return None
    start = (hit.select("i", pl.col("i").alias("back"), pl.col("dur").alias("elapsed"))
             .filter(pl.col("elapsed") < l_max))
    if start.height == 0:
        return None
    pairs = _walk(start, _edges(steps, link), l_max, steps.height)
    back = steps.select(pl.col("i").alias("back"), "d", "t")
    return (pairs.join(back, on="back", how="inner")
            .select("d", "t", pl.col("elapsed").alias("hours_to_end")))


def history_reference_step(history: pl.DataFrame) -> "tuple[str, str]":
    """The step the ``hours_to_end`` of *history* ``(…, d, t, duration, hours_to_end)`` count to: the row whose
    ``hours_to_end`` is its own duration.  Raises ``ValueError`` when there is not exactly one such step."""
    ref = (history.filter((pl.col("hours_to_end") - pl.col("duration")).abs() <= 1e-9)
           .select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique())
    if ref.height != 1:
        raise ValueError(
            "mintime roll carry: the on-line history must end at one realized step (the row whose hours_to_end is "
            f"its own duration), got {ref.rows()}")
    return ref.row(0)


def check_history_continues_carried_step(history: pl.DataFrame, carried: "tuple[str, str]",
                                         period_branch: "pl.DataFrame | None" = None) -> None:
    """Raise ``ValueError`` when *history* does not end at the *carried* step ``(d, t)`` (the previous solve's last
    realized step, from which this solve's first steps continue: ``dt_ramp_carry``'s ``d_previous, t_previous``).

    A forecasts-only fan writes its history under the forecast that holds the tied values and its last step under
    the period label; *period_branch* ``(d, b)`` (period → forecast branch) accepts that pair.  A mismatch means the
    history and the carried step were handed over from different solves."""
    ref_d, ref_t = history_reference_step(history)
    c_d, c_t = carried
    if ref_t == c_t and ref_d == c_d:
        return
    if ref_t == c_t and period_branch is not None and period_branch.height > 0:
        pb = period_branch.select(pl.col(period_branch.columns[0]).cast(pl.Utf8).alias("a"),
                                  pl.col(period_branch.columns[1]).cast(pl.Utf8).alias("b"))
        if pb.filter((pl.col("a") == c_d) & (pl.col("b") == ref_d)).height > 0:
            return
    raise ValueError(
        f"mintime roll carry: the realized on-line history ends at step {(ref_d, ref_t)}, but this solve continues "
        f"from step {(c_d, c_t)}: the history and the carried step come from different solves")


def mintime_roll_carry(dt: "pl.DataFrame | None", step_duration: "object | None", link: "pl.DataFrame | None",
                       carry: "pl.DataFrame | None", history: "pl.DataFrame | None",
                       procs: "list[tuple[str, float]]", metric: str,
                       period_branch: "pl.DataFrame | None" = None) -> "pl.DataFrame | None":
    """``(p, d, t, value)`` Utf8: the realized *metric* (``startup`` / ``shutdown``) of the previous solve that the
    minimum time ``L`` of *procs* ``[(p, L)]`` still covers at the steps of this solve.

    ``carry`` ``(d, t, d_previous, t_previous)`` (``dt_ramp_carry``): the roots — the first steps of this solve that
    continue from the previous solve's last realized step on the timeline.  ``history`` ``(p, d, t, startup,
    shutdown, duration, hours_to_end)``: the previous solve's realized start-ups / shut-downs, ``hours_to_end`` from
    the start of the step to the end of the carried step.  A step ``(d, t)`` reached from a root ``r`` along
    ``link`` (``r`` itself, elapsed 0, or a later step with ``elapsed(r → d, t)`` the summed durations from ``r`` up
    to, not including, ``(d, t)``) carries ``Σ_h metric[p, h]`` over ``elapsed + hours_to_end[h] < L``.  ``None``
    when nothing is carried (no roots: first roll, single solve, a carried step that is not adjacent).  Raises
    ``ValueError`` when the history does not end at the carried step
    (:func:`check_history_continues_carried_step`)."""
    procs = [(str(p), float(L)) for p, L in procs if float(L) > 0]
    if (dt is None or dt.height == 0 or carry is None or carry.height == 0 or history is None
            or history.height == 0 or not procs):
        return None
    carried = carry.select(pl.col("d_previous").cast(pl.Utf8), pl.col("t_previous").cast(pl.Utf8)).unique()
    if carried.height != 1:
        raise ValueError(f"mintime_roll_carry: the carry must come from one step, got {carried.rows()}")
    check_history_continues_carried_step(history, carried.row(0), period_branch)
    steps, edges = _solve_steps(dt, step_duration, link, "mintime_roll_carry")
    roots = (steps.join(carry.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique(),
                        on=["d", "t"], how="semi")
             .select("i", pl.col("i").alias("back"), pl.lit(0.0, dtype=pl.Float64).alias("elapsed")))
    if roots.height == 0:
        return None
    reach = _walk(roots, edges, max(L for _, L in procs), steps.height, forward=True)
    lmap = pl.DataFrame({"p": [p for p, _ in procs], "L": [L for _, L in procs]},
                        schema={"p": pl.Utf8, "L": pl.Float64})
    hist = (history.select(pl.col("p").cast(pl.Utf8), pl.col("hours_to_end").cast(pl.Float64),
                           pl.col(metric).cast(pl.Float64).fill_null(0.0).alias("value"))
            .filter(pl.col("value") != 0.0)
            .join(lmap, on="p", how="inner"))
    if hist.height == 0:
        return None
    out = (reach.select("i", "elapsed").join(hist, how="cross")
           .filter(pl.col("elapsed") + pl.col("hours_to_end") < pl.col("L"))
           .group_by("p", "i").agg(pl.col("value").sum())
           .filter(pl.col("value") != 0.0)
           .join(steps.select("i", "d", "t"), on="i", how="inner")
           .select("p", "d", "t", "value")
           .sort("p", "d", "t"))
    return out if out.height > 0 else None


def online_limit_cells(d: object) -> "pl.DataFrame | None":
    """``(p, d, t, cap)`` (Utf8 keys): the most units online ``cap`` of each process with online variables
    (``process_online_linear`` / ``process_online_integer``) at the steps where it is below the units built —
    the ceiling an obligation carried in from an earlier solve meets:

        cap[p,d,t] = 0                                   min_load[p,d,t] > 0 and κ[p] < min_load[p,d,t]
                   = existing_count[p,d] · availability[p,d,t]   otherwise (maxOnline)

    κ[p] = Σ capacity_max_coeff over the unit's output measure A_p (``model._unit_floors.unit_measured_arcs``; 1 by
    default), for an indirect unit min(Σ, 1): where κ < min_load, ``minFlow_minload`` (Σ_{A_p} flow ≥ min_load ·
    v_online), ``maxFlow_online`` (≤ capacity_max_coeff · v_online on each arc of A_p) and, for an indirect unit,
    ``maxOutputSum_online`` (Σ_{A_p} flow ≤ v_online) leave only v_online = 0.  Elsewhere ``maxOnline`` keeps the
    units online at or below the units available.  existing_count is the solve's existing units (the investments
    of earlier solves included); a unit this solve invests in may get more.  Rows only where availability < 1 or
    κ < min_load; availability 0 gives cap 0, also without min_load.

    A coarse process (``processCoarse``, ``process_coarse_step``) is judged as its rows are: once per coarse step,
    at the step's first fine step, on the Δt-weighted means of availability and min_load over the step.  ``None``
    when no step is limited.  Reads the solve's shaped ``p_min_load`` / ``p_process_availability`` /
    ``p_arc_max_cap_coef`` / ``p_process_existing_count``."""
    from flextool.engine.input._param_shapes import promote_param_to_dt
    from flextool.engine.model._common import _availability_factor, _coef_factor
    from flextool.engine.model._unit_floors import unit_measured_arcs

    dt = getattr(d, "dt", None)
    online = [f.select(pl.col("p").cast(pl.Utf8)) for f in (getattr(d, "process_online_linear", None),
                                                           getattr(d, "process_online_integer", None))
              if f is not None and f.height > 0]
    if not online or dt is None or dt.height == 0:
        return None
    procs = pl.concat(online).unique().sort("p")
    keys = ["p", "d", "t"]
    if getattr(d, "p_process_availability", None) is not None:
        cells = _availability_factor(d, procs).frame.select(
            *(pl.col(k).cast(pl.Utf8) for k in keys), pl.col("value").cast(pl.Float64).alias("av"))
    else:
        cells = procs.join(dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)), how="cross").with_columns(
            av=pl.lit(1.0))
    ml = getattr(d, "p_min_load", None)
    arcs = unit_measured_arcs(d)
    if ml is not None and ml.frame.height > 0 and arcs is not None:
        coef = getattr(d, "p_arc_max_cap_coef", None)
        arc_keys = ["p", "source", "sink"]
        kappa = (_coef_factor(arcs, coef, arc_keys).frame if coef is not None
                 else arcs.select(arc_keys).unique().with_columns(value=pl.lit(1.0)))
        kappa = kappa.group_by(pl.col("p").cast(pl.Utf8)).agg(pl.col("value").sum().alias("kappa"))
        indirect = getattr(d, "process_indirect", None)
        if indirect is not None and indirect.height > 0:
            ind = indirect.select(pl.col("p").cast(pl.Utf8)).unique()["p"].implode()
            kappa = kappa.with_columns(pl.when(pl.col("p").is_in(ind)).then(pl.min_horizontal("kappa", pl.lit(1.0)))
                                       .otherwise(pl.col("kappa")).alias("kappa"))
        mls = (promote_param_to_dt(ml, dt)
               .select(*(pl.col(k).cast(pl.Utf8) for k in keys), pl.col("value").cast(pl.Float64).alias("ml"))
               .collect())
        cells = cells.join(mls, on=keys, how="left").join(kappa, on="p", how="left")
    else:
        cells = cells.with_columns(ml=pl.lit(None, pl.Float64), kappa=pl.lit(None, pl.Float64))
    cells = (cells.with_columns(pl.col("ml").fill_null(0.0), pl.col("kappa").fill_null(1.0))
             .select("p", "d", "t", "ml", "kappa", "av"))
    cells = _at_coarse_steps(cells, getattr(d, "process_coarse_step", None))
    ec = getattr(d, "p_process_existing_count", None)
    if ec is not None and ec.frame.height > 0:
        count = (promote_param_to_dt(ec, dt)
                 .select(*(pl.col(k).cast(pl.Utf8) for k in keys), pl.col("value").cast(pl.Float64).alias("count"))
                 .collect())
        cells = cells.join(count, on=keys, how="left")
    else:
        cells = cells.with_columns(count=pl.lit(None, pl.Float64))
    blocked = (pl.col("ml") > 0.0) & (pl.col("kappa") < pl.col("ml"))
    out = (cells.filter(blocked | (pl.col("av") < 1.0))
           .select("p", "d", "t", cap=pl.when(blocked).then(0.0)
                   .otherwise(pl.col("count").fill_null(0.0) * pl.col("av")))
           .sort("p", "d", "t"))
    return out if out.height > 0 else None


def cells_over_limit(frame: "pl.DataFrame | None", limit: "pl.DataFrame | None",
                     tol: float = 1e-9) -> "pl.DataFrame | None":
    """``(p, d, t)`` (Utf8) of the rows of *frame* ``(p, d, t, value, …)`` — units online an earlier solve obliges
    (the carried minimum up-time, a realization pin) — whose value exceeds the most units online there, *limit*
    ``(p, d, t, cap)`` (:func:`online_limit_cells`) by more than *tol*; ``None`` when there is none."""
    if frame is None or frame.height == 0 or limit is None or limit.height == 0:
        return None
    keys = ["p", "d", "t"]
    out = (frame.select(*(pl.col(k).cast(pl.Utf8) for k in keys), pl.col("value").cast(pl.Float64))
           .join(limit, on=keys, how="inner")
           .filter(pl.col("value") > pl.col("cap") + tol)
           .select(keys).unique().sort(keys))
    return out if out.height > 0 else None


def _at_coarse_steps(cells: pl.DataFrame, coarse_step: "pl.DataFrame | None") -> pl.DataFrame:
    """*cells* ``(p, d, t, ml, kappa, av)`` Utf8 with the rows of each coarse process (*coarse_step* ``(p, d,
    b_first, t, w)``) replaced by one row per coarse step at its first fine step: ``ml`` and ``av`` their
    w-weighted means Σ w·x / Σ w (the Δt-weighted means the coarse rows read; dividing by Σ w keeps a constant
    exact)."""
    if coarse_step is None or coarse_step.height == 0:
        return cells
    steps = coarse_step.select(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                               pl.col("b_first").cast(pl.Utf8), pl.col("w").cast(pl.Float64))
    with_b = cells.join(steps, on=["p", "d", "t"], how="left")
    coarse = (with_b.filter(pl.col("b_first").is_not_null())
              .group_by("p", "d", "b_first", maintain_order=True)
              .agg((pl.col("ml") * pl.col("w")).sum() / pl.col("w").sum(), pl.col("kappa").first(),
                   (pl.col("av") * pl.col("w")).sum() / pl.col("w").sum())
              .select("p", "d", pl.col("b_first").alias("t"), "ml", "kappa", "av"))
    return pl.concat([with_b.filter(pl.col("b_first").is_null()).select(cells.columns), coarse])


def note_carried_obligation_cut(cases: "dict[str, str]") -> None:
    """Note W-0592 (one grouped line per run) for the units whose obligation carried in from an earlier solve met
    a step where they cannot run; *cases* ``{unit: detail}`` (solve and step) go to the debug log."""
    if cases:
        from flextool import run_warnings
        run_warnings.note("W-0592", sorted(cases), details=cases, per_run=True)


def cut_at_blocked(frame: "pl.DataFrame | None", blocked: "pl.DataFrame | None", dt: "pl.DataFrame | None",
                   step_duration: "object | None", link: "pl.DataFrame | None", l_max: float,
                   cut: "list | None" = None, reached: "list | None" = None) -> "pl.DataFrame | None":
    """*frame* ``(p, d, t, …)`` — the up-time roll carry (:func:`mintime_roll_carry`) or the realization pins —
    without what an outage voids: the rows of a process end at the first step of *blocked* ``(p, d, t)``
    (:func:`cells_over_limit`) that holds one of its rows, and do not resume after it — every
    step reached along *link* from such a step (the step itself included, within *l_max* hours) loses its rows.
    ``cut`` (a list) gets the ``(p, d, t)`` of those first blocked steps, ``reached`` (a list) the reached cells as
    one ``(p, d, t)`` Utf8 frame.  *frame* unchanged when nothing is cut; ``None`` when nothing is left."""
    if frame is None or frame.height == 0 or blocked is None or blocked.height == 0 or dt is None:
        return frame
    steps, edges = _solve_steps(dt, step_duration, link, "cut_at_blocked")
    keys = ["p", "d", "t"]
    utf8 = frame.with_columns(*(pl.col(k).cast(pl.Utf8).alias(f"__{k}") for k in keys))
    out = utf8.join(steps.select("i", pl.col("d").alias("__d"), pl.col("t").alias("__t")), on=["__d", "__t"],
                    how="inner", maintain_order="left")
    hit = (blocked.select(pl.col("p").cast(pl.Utf8), pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8))
           .join(steps.select("i", "d", "t"), on=["d", "t"], how="inner")
           .join(out.select(pl.col("__p").alias("p"), "i"), on=["p", "i"], how="semi"))
    if hit.height == 0:
        return frame
    procs = hit.select("p").unique().sort("p").with_row_index("back")
    start = (hit.join(procs, on="p", how="inner")
             .select("i", "back", pl.lit(0.0, dtype=pl.Float64).alias("elapsed")))
    walk = _walk(start, edges, l_max, steps.height, forward=True).join(procs, on="back", how="inner")
    if cut is not None:
        later = walk.filter(pl.col("elapsed") > 0.0).select("p", "i")
        cut.extend(hit.join(later, on=["p", "i"], how="anti").sort("p", "d", "t").select("p", "d", "t").rows())
    cells = walk.select("p", "i").unique()
    if reached is not None:
        reached.append(cells.join(steps.select("i", "d", "t"), on="i", how="inner").select("p", "d", "t"))
    out = (out.join(cells.select(pl.col("p").alias("__p"), "i"), on=["__p", "i"], how="anti", maintain_order="left")
           .select(frame.columns))
    return out if out.height > 0 else None


def cut_short_lookback_warnings(min_times: "dict[str, list[tuple[str, float]]]",
                                dt: "pl.DataFrame | None",
                                first_step_link: "pl.DataFrame | None",
                                carry_roots: "pl.DataFrame | None",
                                solve_name: str) -> "list[str]":
    """Warnings for the steps whose minimum up/down-time look-back the solve boundary cuts short (the texts of
    :func:`_cut_short_lookback_cases`)."""
    return [f"Process '{p}' has {param}={L:g} hours; the look-back from time step ({d}, {t}) of solve "
            f"'{solve_name}' covers 0 hours"
            for param, p, L, d, t in _cut_short_lookback_cases(min_times, dt, first_step_link, carry_roots)]


def _cut_short_lookback_cases(min_times: "dict[str, list[tuple[str, float]]]",
                              dt: "pl.DataFrame | None",
                              first_step_link: "pl.DataFrame | None",
                              carry_roots: "pl.DataFrame | None") -> list[tuple]:
    """``[(parameter, process, L, d, t)]``: the steps whose minimum up/down-time look-back the solve boundary cuts
    short.

    A first step of a period of this solve is cut short when the complete solve's predecessor of it is adjacent on
    the timeline (``jump == 1``) but is not a step of this solve, and the step is not one of the *carry_roots*
    ``(d, t)`` (``dt_ramp_carry``: the steps that continue the previous solve's last realized step, from which the
    roll carry of the realized on-line history starts).  Such a step continues a step that no history handed to this
    solve ends at: a hole or an overlap in time between the two solves (a solve with longer steps than the dispatch
    realized before it continues that dispatch when the two steps end at the same time).
    ``min_times`` maps ``"min_uptime"`` / ``"min_downtime"`` to ``[(p, L)]``; ``first_step_link`` ``[period, time,
    previous_period, previous_within_solve, jump]``.  One case per process and parameter, naming the first such
    step.
    """
    if dt is None or dt.height == 0 or first_step_link is None or first_step_link.height == 0:
        return []
    if not any(L > 0 for procs in min_times.values() for _, L in procs):
        return []
    steps = dt.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique()
    cut = (first_step_link.select(
        pl.col("period").cast(pl.Utf8).alias("d"), pl.col("time").cast(pl.Utf8).alias("t"),
        pl.col("previous_period").cast(pl.Utf8).alias("d_previous"),
        pl.col("previous_within_solve").cast(pl.Utf8).alias("t_previous"),
        pl.col("jump").cast(pl.Utf8).cast(pl.Float64, strict=False).alias("jump"))
        .filter(pl.col("jump") == 1.0)
        .join(steps.rename({"d": "d_previous", "t": "t_previous"}), on=["d_previous", "t_previous"], how="anti")
        .join(steps, on=["d", "t"], how="semi"))
    if carry_roots is not None and carry_roots.height > 0:
        cut = cut.join(carry_roots.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8)).unique(),
                       on=["d", "t"], how="anti")
    if cut.height == 0:
        return []
    out = []
    d, t = cut.row(0)[0], cut.row(0)[1]
    for param, procs in min_times.items():
        for p, L in sorted(procs):
            if L > 0:
                out.append((param, p, L, d, t))
    return out


def report_cut_short_lookback(min_times: "dict[str, list[tuple[str, float]]]",
                              dt: "pl.DataFrame | None",
                              first_step_link: "pl.DataFrame | None",
                              carry_roots: "pl.DataFrame | None",
                              solve_name: str) -> "list[str]":
    """Warn (W-0504, one line per parameter and solve) about the :func:`cut_short_lookback_warnings`, which go to
    the debug log; returns them."""
    from flextool import run_warnings
    cases = _cut_short_lookback_cases(min_times, dt, first_step_link, carry_roots)
    messages = cut_short_lookback_warnings(min_times, dt, first_step_link, carry_roots, solve_name)
    by_param: dict[str, dict[str, str]] = {}
    for (param, process, *_rest), message in zip(cases, messages):
        by_param.setdefault(param, {})[process] = message
    for param, found in by_param.items():
        run_warnings.warn_grouped("W-0504", list(found), solve=solve_name, parameter=param.removeprefix("min_"),
                                  details=found)
    return messages
