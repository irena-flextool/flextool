"""Minimum up/down-time look-back: which earlier steps a step's minimum-time constraint sums over.

The look-back of a step ``(d, t)`` is the step itself plus its ancestors along the ramp predecessor links
(``dt_ramp_link``: the predecessor that is adjacent on the timeline), as long as the time elapsed from the start of
the ancestor to the start of ``(d, t)`` stays below the minimum time.  The chain stops at a step without such a
predecessor: the first step of a solve, a period that does not continue the previous one, a gap in the timeline, the
start of a representative period and the first step of a forecast branch that starts the solve.  There is no
cyclic wrap.  Where the chain stops, the on-line state may still be linked to an earlier step (``online_change``
wraps or lags there); a unit on-line at such a step may then shut down at the next step — an end effect, as at the
end of the horizon.

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
    steps = _step_durations(dt, step_duration)
    if link is None:
        if steps.height > 1:
            raise ValueError(
                "mintime_lookback: the solve has several time steps but no ramp predecessor links "
                "(dt_ramp_link); the minimum up/down-time look-back cannot be built")
        link = pl.DataFrame(schema={"d": pl.Utf8, "t": pl.Utf8, "d_previous": pl.Utf8, "t_previous": pl.Utf8})
    idx = steps.select("i", "d", "t")
    prev = steps.select(pl.col("i").alias("j"), pl.col("d").alias("d_previous"), pl.col("t").alias("t_previous"),
                        pl.col("dur").alias("dur_j"))
    edges = (link.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                         pl.col("d_previous").cast(pl.Utf8), pl.col("t_previous").cast(pl.Utf8))
             .unique()
             .join(idx, on=["d", "t"], how="inner")
             .join(prev, on=["d_previous", "t_previous"], how="inner")
             .select("i", "j", "dur_j"))
    l_max = max(L for _, L in procs)
    pairs = [steps.select("i", pl.col("i").alias("back"), pl.lit(0.0, dtype=pl.Float64).alias("elapsed"))]
    frontier = (edges.filter(pl.col("dur_j") < l_max)
                .select("i", pl.col("j").alias("back"), pl.col("dur_j").alias("elapsed")))
    depth = 0
    while frontier.height > 0:
        depth += 1
        if depth > steps.height:
            raise ValueError("mintime_lookback: the ramp predecessor links (dt_ramp_link) form a cycle")
        pairs.append(frontier)
        frontier = (frontier.join(edges, left_on="back", right_on="i", how="inner")
                    .select("i", pl.col("j").alias("back"), (pl.col("elapsed") + pl.col("dur_j")).alias("elapsed"))
                    .filter(pl.col("elapsed") < l_max))
    all_pairs = pl.concat(pairs)
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
    back = idx.select(pl.col("i").alias("back"), pl.col("d").alias("d_back"), pl.col("t").alias("t_back"))
    return (rows.join(idx, on="i", how="inner").join(back, on="back", how="inner")
            .select(*LOOKBACK_COLS)
            .sort(*LOOKBACK_COLS))


def cut_short_lookback_warnings(min_times: "dict[str, list[tuple[str, float]]]",
                                dt: "pl.DataFrame | None",
                                first_step_link: "pl.DataFrame | None",
                                history: "pl.DataFrame | None",
                                solve_name: str) -> "list[str]":
    """Warnings for the steps whose minimum up/down-time look-back the solve boundary cuts short.

    A first step of a period of this solve is cut short when the complete solve's predecessor of it is adjacent on
    the timeline (``jump == 1``) but is not a step of this solve, and the realized on-line history handed over to
    this solve (``history`` ``(p, …)``, from which the roll carry is built) has no row of the process.  ``min_times`` maps
    ``"min_uptime"`` / ``"min_downtime"`` to ``[(p, L)]``; ``first_step_link`` ``[period, time, previous_period,
    previous_within_solve, jump]``.  One message per process and parameter, naming the first such step.
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
        .join(steps.rename({"d": "d_previous", "t": "t_previous"}), on=["d_previous", "t_previous"], how="anti"))
    if cut.height == 0:
        return []
    with_history: set[str] = set()
    if history is not None and history.height > 0:
        with_history = set(history["p"].cast(pl.Utf8).unique().to_list())
    messages = []
    for param, procs in min_times.items():
        for p, L in sorted(procs):
            if L <= 0:
                continue
            if p in with_history:
                continue
            for d, t, _, _, _ in cut.rows():
                messages.append(
                    f"Process '{p}' has {param}={L:g} hours but the look-back from time step ({d}, {t}) of solve "
                    f"'{solve_name}' covers only 0 hours: the time steps before it belong to an earlier solve and no "
                    f"realized on-line history of the process reaches this solve, so its earlier start-ups and "
                    f"shut-downs do not constrain it.")
                break
    return messages


def report_cut_short_lookback(min_times: "dict[str, list[tuple[str, float]]]",
                              dt: "pl.DataFrame | None",
                              first_step_link: "pl.DataFrame | None",
                              history: "pl.DataFrame | None",
                              solve_name: str) -> "list[str]":
    """Log and print (the console of a solve shows only printed lines) the :func:`cut_short_lookback_warnings`;
    returns them."""
    messages = cut_short_lookback_warnings(min_times, dt, first_step_link, history, solve_name)
    for message in messages:
        logger.warning(message)
        print(f"WARNING: {message}", flush=True)
    return messages
