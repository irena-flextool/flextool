"""§3.12 — groups, row scaling and inflow: ``process_group_inside_nonSync``, the group and node
capacity for scaling, and the positive / negative / per-step inflow."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import alias_to_axis, align_join_dtypes, axis_lazyframe

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import _try_entities, _try_param
from flextool.engine.preprocess.params._derived._periods import _expand_branch_periods, _period_in_use_set


# ---------------------------------------------------------------------------
# §3.12.1 — process_group_inside_nonSync
# ---------------------------------------------------------------------------


def process_group_inside_nonSync_from_flex(fd: object) -> pl.DataFrame | None:
    """``(p, g)``: the processes whose flows stay inside the non-synchronous group g (internal transfers).

    inside = {(p,g) : g ∈ groupNonSync, ∃ s ∈ src(p) ∩ N(g), k ∈ snk(p) ∩ N(g),
                      not (|src(p) ∩ N(g)| = |snk(p) ∩ N(g)| = 1 ∧ s = k)}
      src(p) = sources of the arcs of p in process_source_sink (other than p itself),
      snk(p) = sinks of the arcs of p (other than p itself),  N(g) = nodes of g (group_node)
    (flextool.mod l.4065/4075, ``emit/_mid_sets.derive_process__group_inside_group_nonSync``).  A process
    with a source and a different sink node in g moves energy between the group's nodes: the non-synchronous
    limit counts neither its infeed nor its draw.  A regular connection with one end in g has the same single
    node on both sides, so it is not inside.  One rule for the single-solve and the synthetic-solve paths.
    Sorted by (p, g); None when no process is inside a group.
    """
    pss = getattr(fd, "process_source_sink", None)
    gn = getattr(fd, "group_node", None)
    gns = getattr(fd, "groupNonSync", None)
    if pss is None or gn is None or gns is None or pss.height == 0 or gn.height == 0 or gns.height == 0:
        return None
    gns_g, gn_g = align_join_dtypes(gns.select("g"), gn.select("g", "n"), ("g",))
    gn_ns = gn_g.join(gns_g, on="g", how="semi")
    if gn_ns.height == 0:
        return None
    ends = []
    for side in ("source", "sink"):
        arcs = (pss.filter(pl.col(side).cast(pl.Utf8) != pl.col("p").cast(pl.Utf8))
                .select("p", pl.col(side).alias("n")).unique())
        arcs, gnj = align_join_dtypes(arcs, gn_ns, ("n",))
        ends.append(arcs.join(gnj, on="n", how="inner")
                    .group_by("p", "g")
                    .agg(pl.col("n").n_unique().alias(f"{side}_count"),
                         pl.col("n").cast(pl.Utf8).min().alias(f"{side}_node")))
    both = ends[0].join(ends[1], on=["p", "g"], how="inner")
    single_node = ((pl.col("source_count") == 1) & (pl.col("sink_count") == 1)
                   & (pl.col("source_node") == pl.col("sink_node")))
    out = both.filter(~single_node).select("p", "g").sort("p", "g")
    return out if out.height > 0 else None


# ---------------------------------------------------------------------------
# §3.12.2 — p_inv_group_cap / p_group_capacity_for_scaling
# ---------------------------------------------------------------------------


def p_group_capacity_for_scaling_from_source(source: "InputSource",
                                                  active_solve: str | None,
                                                  workdir: Path | None = None,
                                                  *,
                                                  ctx: "SolveContext | None" = None,
                                                  provider: "object | None" = None,
                                                  ) -> "Param | None":
    """Per-(g, d) row scaling factor.  Algorithm (audit §3.12.2,
    ``lp_scaling_params.py:91-244``): pow10-clamped per-group capacity
    proxy when scaling is active, else 1.0 across all (g, d).

    For Γ.3.C we cover the **scaling-inactive default**: emit 1.0 across
    all (group, period) pairs (including stochastic-branch siblings when
    workdir-provided ``period_in_use_set.csv`` is read).  Active scaling
    (rare; activated via ``solve.use_row_scaling``) returns None and the
    CSV value survives.
    """
    period_in_use = _period_in_use_set(source, active_solve, workdir,
                                        ctx=ctx, provider=provider)
    if not period_in_use:
        return None
    # A4-followup: broadcast to stochastic-branch siblings — same
    # rationale as p_node_capacity_for_scaling.
    period_in_use = _expand_branch_periods(period_in_use, workdir,
                                            ctx=ctx, provider=provider)
    groups_df = _try_entities(source, "group")
    if groups_df is None or groups_df.height == 0:
        return None
    # Detect scaling active: solve.use_row_scaling truthy.
    urs = _try_param(source, "solve", "use_row_scaling")
    scaling_active = False
    if urs is not None and active_solve is not None:
        sub = urs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            try:
                scaling_active = float(sub["value"][0]) >= 0.5
            except (TypeError, ValueError):
                pass
    if scaling_active:
        return None  # defer; CSV path handles pow10 cascade
    out = (groups_df.lazy().select(alias_to_axis("name", "g"))
              .join(axis_lazyframe({"d": period_in_use}), how="cross")
              .with_columns(value=pl.lit(1.0))
              .sort("g", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("g", "d"), out)


def p_inv_group_cap_from_source(source: "InputSource",
                                    active_solve: str | None,
                                    workdir: Path | None = None,
                                    *,
                                    ctx: "SolveContext | None" = None,
                                    provider: "object | None" = None,
                                    ) -> "Param | None":
    """Reciprocal of ``p_group_capacity_for_scaling``.  When scaling is
    inactive (default), this is 1.0 per (g, d).
    """
    gcs = p_group_capacity_for_scaling_from_source(
        source, active_solve, workdir, ctx=ctx, provider=provider)
    if gcs is None:
        return None
    out = (gcs.frame.lazy()
              .with_columns(value=1.0 / pl.col("value"))
              .sort("g", "d")
              .collect())
    return Param(("g", "d"), out)


def p_node_capacity_for_scaling_from_source(source: "InputSource",
                                                  active_solve: str | None,
                                                  workdir: Path | None = None,
                                                  *,
                                                  ctx: "SolveContext | None" = None,
                                                  provider: "object | None" = None,
                                                  ) -> "Param | None":
    """Per-(n, d) row scaling factor for slack-penalty terms in the
    objective (``vq_up``/``vq_down`` × ``p_penalty`` × this factor).

    Δ.29 — tactical default-1.0 path matching
    :func:`p_group_capacity_for_scaling_from_source`.  Algorithm
    (audit §3.12.2, ``lp_scaling_params.py:91-244``): pow10-clamped
    per-node capacity proxy when ``solve.use_row_scaling`` is active,
    else 1.0 across all (nodeBalance, period_in_use) pairs.

    For Δ.29 we cover the **scaling-inactive default**: emit 1.0 across
    nodeBalance × period_in_use.  Active scaling (rare; gated on
    ``solve.use_row_scaling``) returns None and the CSV value survives.

    The slow path's :func:`_load_node_capacity_for_scaling` reads
    ``solve_data/node_capacity_for_scaling.csv`` and filters to
    nodeBalance via ``df.join(nb, on='n', how='inner')``; we replicate
    that filter inline here.

    TODO(future): port the full pow10 cascade in
    ``preprocessing/lp_scaling_params.py`` for fixtures with
    ``solve.use_row_scaling=1``.  For now the helper bails (returns
    None) when scaling is active so the seed CSV survives.
    """
    period_in_use = _period_in_use_set(source, active_solve, workdir,
                                        ctx=ctx, provider=provider)
    if not period_in_use:
        return None
    # Detect scaling active: solve.use_row_scaling truthy.
    urs = _try_param(source, "solve", "use_row_scaling")
    scaling_active = False
    if urs is not None and active_solve is not None:
        sub = urs.filter(pl.col("name") == active_solve)
        if sub.height > 0:
            try:
                scaling_active = float(sub["value"][0]) >= 0.5
            except (TypeError, ValueError):
                pass
    if scaling_active:
        return None  # defer; CSV path handles pow10 cascade
    # nodeBalance projection — same set as the slow path's filter.  Slack
    # vars span ``nodeBalance ∪ nodeBalancePeriod`` (period nodes also
    # carry ``vq_state_up/down``), so union the period nodes in before the
    # cross-join.  Dtype-safe; when the period set is empty the union
    # collapses to exactly ``nb`` → byte-identical legacy behavior.
    from flextool.engine.preprocess.params._projection import (
        nodeBalance as _nodeBalance_proj,
        nodeBalancePeriod as _nodeBalancePeriod_proj,
    )
    nb = _nodeBalance_proj(source)
    if nb is None or nb.height == 0:
        return None
    nbp = _nodeBalancePeriod_proj(source)
    if nbp is not None and nbp.height > 0:
        nb = (pl.concat([nb.select("n"), nbp.select("n")])
                .unique()
                .select("n"))
    out = (nb.lazy()
              .join(axis_lazyframe({"d": period_in_use}), how="cross")
              .with_columns(value=pl.lit(1.0))
              .sort("n", "d")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d"), out)


# ---------------------------------------------------------------------------
# §3.12.3 — p_positive_inflow / p_negative_inflow
# ---------------------------------------------------------------------------


def p_positive_inflow_from_inflow(p_inflow: object | None) -> "Param | None":
    """Positive component (clip-low at 0) of ``p_inflow``."""
    if p_inflow is None:
        return None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    if fr is None or fr.height == 0:
        return None
    out = (fr.lazy()
            .with_columns(value=pl.when(pl.col("value") > 0.0)
                                  .then(pl.col("value"))
                                  .otherwise(0.0))
            .sort("n", "d", "t")
            .collect())
    return Param(("n", "d", "t"), out)


def p_negative_inflow_from_inflow(p_inflow: object | None) -> "Param | None":
    """Negative component (clip-high at 0) of ``p_inflow``."""
    if p_inflow is None:
        return None
    fr = p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow
    if fr is None or fr.height == 0:
        return None
    out = (fr.lazy()
            .with_columns(value=pl.when(pl.col("value") < 0.0)
                                  .then(pl.col("value"))
                                  .otherwise(0.0))
            .sort("n", "d", "t")
            .collect())
    return Param(("n", "d", "t"), out)


def real_inflow(p_inflow: object | None, p_inflow_unmargined: object | None) -> object | None:
    """The node inflow before the invest-solve energy margin: the real demand.

    ``p_inflow_unmargined`` is ``solve_data/pdtNodeInflow_unmargined``, written by the energy-margin
    emitters whenever a solve on the Provider applies a margin and refreshed by every later solve; without
    it no solve has a margin and the real inflow is ``p_inflow`` itself.  The group rows (non-synchronous
    limit, capacity margin, the group demand of the dynamic reserve requirement) read the real demand: the
    energy margin is an adequacy device and not load.
    """
    return p_inflow if p_inflow_unmargined is None else p_inflow_unmargined


# ---------------------------------------------------------------------------
# §3.12.4 — pdtNodeInflow_per_step
# ---------------------------------------------------------------------------


def pdtNodeInflow_per_step_from_inflow(p_inflow: object | None,
                                          p_step_duration: object | None,
                                          ) -> "Param | None":
    """``p_inflow / p_step_duration`` per (n, d, t).  Used by
    capacity_margin RHS.  Mirrors the .mod's
    ``pdtNodeInflow / step_duration`` divider.

    Note: ``p_inflow`` here is flextool's ``pdtNodeInflow`` shape — a
    long-format (n, d, t, value) frame already integrated over the step.
    Dividing by ``step_duration`` returns the per-hour rate.
    """
    if p_inflow is None or p_step_duration is None:
        return None
    inflow_fr = (p_inflow.frame if hasattr(p_inflow, "frame") else p_inflow)
    sd_fr = (p_step_duration.frame if hasattr(p_step_duration, "frame")
              else p_step_duration)
    if (inflow_fr is None or inflow_fr.height == 0
            or sd_fr is None or sd_fr.height == 0):
        return None
    sd_lf = sd_fr.lazy().select(
        "d", "t", pl.col("value").cast(pl.Float64).alias("dur"))
    out = (inflow_fr.lazy()
              .join(sd_lf, on=["d", "t"], how="inner")
              .with_columns(value=pl.col("value") / pl.col("dur"))
              .select("n", "d", "t", "value")
              .sort("n", "d", "t")
              .collect())
    if out.height == 0:
        return None
    return Param(("n", "d", "t"), out)
