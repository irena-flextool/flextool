"""Shared post-process branch-probability weighting (option A).

The LP objective weights every per-(d, t) operational and penalty cost
term by ``pdt_branch_weight[d, t]`` (``flextool.engine.model.core`` folds
it into ``op_factor`` / ``startup_factor``), so the LP objective is the
probability-weighted expectation over the forecast fan.  The post-process
cost reconstruction must apply the SAME per-(d, t) weight so that both the
per-step detail (``costs__dt.csv``) and the ``Total cost (calculated)``
summary reconcile to the LP objective.

Scope (mirrors the LP exactly):

* Weighted — every per-(d, t) operational cost term (commodity buy/sell,
  per-process commodity, CO2 cost, other operational, startup) and every
  per-(d, t) penalty term (node-state up/down, inertia, non-synchronous,
  reserve up/down).
* NOT weighted — the capacity-margin penalty (a planning-level slack the
  LP intentionally leaves un-weighted; see ``_group_slack.py`` §9.3) and
  every raw physical diagnostic (``*_not_annualized`` slack-event sums,
  physical CO2 tonnes, reserve MW held).

Deterministic / single-branch solves have ``pdt_branch_weight`` absent
(``None``) or a dense all-``1.0`` series → every weighting is an exact
``× 1.0`` no-op, so all non-stochastic outputs are byte-identical.
"""

import pandas as pd


def weight_dt_by_branch(frame, pdt_weight):
    """Multiply a ``(solve, period, time)``-indexed cost frame/Series by
    the per-(d, t) branch weight.

    ``frame`` — a per-(d, t) cost ``DataFrame`` (columns = entities / cost
    categories) or ``Series``.  ``pdt_weight`` — the ``par.pdt_branch_weight``
    ``Series`` indexed by ``(solve, period, time)`` (or ``None`` / empty on
    deterministic runs).

    Missing ``(d, t)`` rows default to weight ``1.0`` (the LP's dense
    ``pdt_branch_weight`` coalesce default).  ``None`` / empty weight or an
    empty ``frame`` returns the input unchanged (byte-parity).
    """
    if frame is None or pdt_weight is None:
        return frame
    if getattr(pdt_weight, "empty", False) or getattr(frame, "empty", False):
        return frame
    factors = pdt_weight.reindex(frame.index).fillna(1.0)
    if getattr(frame, "ndim", 2) == 1:
        return frame.mul(factors)
    return frame.mul(factors, axis=0)


def unweight_period_by_branch(frame, pd_weight):
    """Divide a ``period``-indexed frame/Series by that period's branch
    weight (item 120): the per-period CO2 cap's row isn't branch-weighted
    while the objective's CO2-price term is, so the raw dual is the
    path's own price TIMES its branch probability -- the same shrinkage
    the node-balance dual had before c15c's fix (``write_v_dual_node_balance``).

    ``pd_weight`` is ``par.pd_branch_weight`` -- a ``(solve, period)``-
    indexed Series that keeps its ``solve`` level (``drop_levels`` does not
    touch it; see ``calc_costs.py``'s commodity-ladder pricing for the same
    pattern).  Reduced here to a plain ``{str(period): weight}`` map (last
    solve wins on a period repeated across rolls -- tolerated the same way
    elsewhere); any period missing a row (deterministic / non-forecast
    periods, and the pre-reveal trunk) defaults to ``1.0`` → a no-op.
    """
    if (frame is None or pd_weight is None
            or getattr(pd_weight, "empty", False) or getattr(frame, "empty", False)):
        return frame
    bw_map: dict[str, float] = {}
    for idx, val in pd_weight.items():
        period = idx[-1] if isinstance(idx, tuple) else idx
        bw_map[str(period)] = float(val)
    bw = pd.Series(
        [bw_map.get(str(p), 1.0) for p in frame.index], index=frame.index,
    )
    if getattr(frame, "ndim", 2) == 1:
        return frame.div(bw)
    return frame.div(bw, axis=0)
