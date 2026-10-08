"""Shared post-process branch-probability weighting (option A).

The LP objective weights every per-(d, t) operational and penalty cost
term by ``pdt_branch_weight[d, t]`` (``flextool.engine_polars.model`` folds
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
