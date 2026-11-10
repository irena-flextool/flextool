"""Peak demand of the capacity-margin groups over each period's full time series (``share_of_peak``).

    P[g,d] = max(0, max_{t ∈ T(d)} Σ_{n ∈ N(g)} −inflow_real[n,d,t] / Δ[t])
    ∀ g ∈ groupCapacityMargin, d ∈ the solve's investment periods and their stochastic branch periods
    T(d) = every time step of the period's original timeline (input/timeline.csv): before representative-period
      selection, before new_stepduration and independent of a roll's window;  Δ[t] = its duration
    inflow_real = the solve's pdtNodeInflow formula (inflow scaling method and its period multipliers), evaluated on
      the original time steps with the raw per-step inflow (input/pt_node_inflow, else the node's constant inflow)
      and the raw stochastic branch inflow (input/pbt_node_inflow); no energy margin
    N(g) = the group's nodes in nodeBalance without a storage state (the nodes whose demand the row covers)

A branch period reads its anchor period's timeline and its own branch inflow.  ``capacity_margin_share`` times P
is the margin of a ``share_of_peak`` group (:func:`flextool.engine.model._group_slack.resolve_capacity_margin`).
The frame is ``solve_data/pdGroup_peak_demand.csv`` (``group, period, value``, MW); it is empty for a solve
without capacity-margin groups or investment periods.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.preprocess._provider_io import _emit, _provider_key
from flextool.engine.preprocess.emit._period_params import derive_pdtNodeInflow_vectorized

PEAK_DEMAND = "solve_data/pdGroup_peak_demand.csv"


def _frame(provider, path: Path) -> pl.DataFrame | None:
    df = provider.get(_provider_key(path)) if provider is not None else None
    return df if df is not None and df.height > 0 else None


def _column(df: pl.DataFrame | None, index: int = 0) -> list[str]:
    if df is None:
        return []
    return [str(v) for v in df.get_column(df.columns[index]).to_list() if v is not None and str(v) != ""]


def _period_timelines(input_dir: Path, solve_data_dir: Path, *, provider) -> dict[str, list[str]]:
    """``{period: its original timelines}``: ``solve_data/period__timeline_set`` (the current solve's periods), else
    the timesets the periods use in any solve (``input/timesets_in_use`` → ``input/timesets__timeline``'s original
    timeline).  The roll of a rolling solve is not a solve of ``timesets_in_use``, so its ``period__timeline_set`` is
    empty and the second source gives its periods' timelines."""
    out: dict[str, list[str]] = {}
    ptl = _frame(provider, solve_data_dir / "period__timeline_set.csv")
    if ptl is not None:
        for d, tl in ptl.select(ptl.columns[:2]).iter_rows():
            if d is not None and tl is not None:
                out.setdefault(str(d), []).append(str(tl))
    if out:
        return out
    tt = _frame(provider, input_dir / "timesets__timeline.csv")
    spt = _frame(provider, input_dir / "timesets_in_use.csv")
    if tt is None or spt is None:
        return out
    col = "original_timeline" if "original_timeline" in tt.columns else tt.columns[1]
    tl_of: dict[str, list[str]] = {}
    for ts, tl in tt.select(tt.columns[0], col).iter_rows():
        if ts is not None and tl is not None:
            tl_of.setdefault(str(ts), []).append(str(tl))
    for _s, d, ts in spt.select(spt.columns[:3]).iter_rows():
        if d is not None and ts is not None:
            for tl in tl_of.get(str(ts), ()):
                if tl not in out.setdefault(str(d), []):
                    out[str(d)].append(tl)
    return out


def derive_capacity_margin_peak(input_dir: Path, solve_data_dir: Path, *, provider) -> pl.DataFrame | None:
    """``(group, period, value)`` Utf8 / Float64: P[g,d] of the module docstring; None when there is none."""
    groups = _column(_frame(provider, input_dir / "groupCapacityMargin.csv"))
    anchors = _column(_frame(provider, solve_data_dir / "invest_periods_of_current_solve.csv"))
    if not groups or not anchors:
        return None
    from flextool.engine.preprocess.params._branch import period_branch_full_lf

    parent: dict[str, str] = {d: d for d in anchors}
    for d, b in (period_branch_full_lf(solve_data_dir.parent, provider=provider).collect()
                 .select(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8)).iter_rows()):
        if d in anchors and b not in parent:
            parent[b] = d
    timelines = _period_timelines(input_dir, solve_data_dir, provider=provider)
    tl_df = _frame(provider, input_dir / "timeline.csv")
    if tl_df is None:
        return None
    tl_df = tl_df.select(pl.col(tl_df.columns[0]).cast(pl.Utf8).alias("timeline"),
                         pl.col(tl_df.columns[1]).cast(pl.Utf8).alias("time"),
                         pl.col(tl_df.columns[2]).cast(pl.Float64, strict=False).alias("duration"))
    steps_of: dict[str, list[str]] = {}
    for tl, t in tl_df.select("timeline", "time").iter_rows():
        steps_of.setdefault(tl, []).append(t)
    steps: list[tuple[str, str]] = []
    period_timeline: list[tuple[str, str]] = []
    for d in sorted(parent):
        for tl in dict.fromkeys(timelines.get(d) or timelines.get(parent[d], [])):
            period_timeline.append((d, tl))
            steps.extend((d, t) for t in steps_of.get(tl, ()))
    if not steps:
        return None

    # The raw per-step inflow: the explicit value of a (node, step), else the node's constant inflow.
    nodes = _column(_frame(provider, input_dir / "node.csv"))
    constant: dict[str, float] = {}
    p_node = _frame(provider, input_dir / "p_node.csv")
    if p_node is not None:
        for n, par, v in p_node.select(p_node.columns[:3]).iter_rows():
            if par == "inflow" and n is not None:
                try:
                    constant[str(n)] = float(v)
                except (TypeError, ValueError):
                    continue
    explicit: dict[tuple[str, str], float] = {}
    pti = _frame(provider, input_dir / "pt_node_inflow.csv")
    if pti is not None:
        for n, t, v in pti.select(pti.columns[:3]).iter_rows():
            if n is not None and t is not None:
                try:
                    explicit[(str(n), str(t))] = float(v)
                except (TypeError, ValueError):
                    continue
    times = sorted({t for _d, t in steps})
    pt_inflow = {(n, t): explicit.get((n, t), constant.get(n, 0.0)) for n in nodes for t in times}

    inflow = derive_pdtNodeInflow_vectorized(
        input_dir, solve_data_dir, provider=provider, steps=steps, pt_inflow=pt_inflow,
        pbt_path=input_dir / "pbt_node_inflow.csv", per_original_step=True)
    if inflow is None or inflow.height == 0:
        return None

    balance = set(_column(_frame(provider, solve_data_dir / "nodeBalance.csv")))
    state = set(_column(_frame(provider, solve_data_dir / "nodeState.csv")))
    gn = _frame(provider, input_dir / "group__node.csv")
    if gn is None:
        return None
    members = (gn.select(pl.col(gn.columns[0]).cast(pl.Utf8).alias("group"),
                         pl.col(gn.columns[1]).cast(pl.Utf8).alias("node"))
               .filter(pl.col("group").is_in(groups) & pl.col("node").is_in(sorted(balance - state)))
               .unique())
    durations = (pl.DataFrame(period_timeline, schema={"period": pl.Utf8, "timeline": pl.Utf8}, orient="row")
                 .join(tl_df, on="timeline", how="inner")
                 .group_by("period", "time").agg(pl.col("duration").first()))
    demand = (inflow.select(pl.col("node").cast(pl.Utf8), pl.col("period").cast(pl.Utf8),
                            pl.col("time").cast(pl.Utf8), pl.col("value").cast(pl.Float64))
              .join(members, on="node", how="inner")
              .join(durations, on=["period", "time"], how="inner")
              .filter(pl.col("duration") > 0.0)
              .group_by("group", "period", "time")
              .agg((-pl.col("value") / pl.col("duration")).sum().alias("demand"))
              .group_by("group", "period")
              .agg(pl.col("demand").max().clip(lower_bound=0.0).alias("value"))
              .sort("group", "period"))
    return demand if demand.height > 0 else None


def emit_capacity_margin_peak(input_dir: Path, solve_data_dir: Path, *, provider) -> None:
    """Emit :data:`PEAK_DEMAND`: an empty frame without capacity-margin groups or investment periods, so no
    earlier solve's peak stays on the provider."""
    frame = derive_capacity_margin_peak(input_dir, solve_data_dir, provider=provider)
    if frame is None:
        frame = pl.DataFrame(schema={"group": pl.Utf8, "period": pl.Utf8, "value": pl.Float64})
    _emit(provider, PEAK_DEMAND, frame)
