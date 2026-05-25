"""Phase E.3 — on-demand cross-join helpers for the persistent scratch
frames ``pss_dt`` / ``nodeBalance_dt`` / ``nodeState_dt`` /
``nodeState_first_dt`` / ``process_indirect_dt`` (plus the
RP-blended-weights ``nodeState_rp_dt`` / ``nodeState_rp_block_first_dt``
helpers added by Phase 4 of the RP-blended-weights restoration).

Pre-E.3, :class:`FlexData` carried these as eager DataFrames built
up-front in :mod:`flextool.engine_polars._fast_load` (fast path) and in
:mod:`flextool.engine_polars.input` (slow path).  On y2050-scale
fixtures the largest of them — ``pss_dt`` = ``process_source_sink × dt``
— is ~1.75 GB and lives for the whole LP build alongside ``v_flow.frame``
(which already carries the same ``(p, source, sink, d, t)`` key set after
:func:`polar_high.Problem.add_var`).  Phase E.3 stops materialising them
eagerly; consumers call the helpers below at the point of use.

Each helper accepts a :class:`FlexData` instance and returns either an
eager DataFrame (suitable as ``index=`` for ``add_var`` / ``over=`` for
``add_cstr`` / ``Sum``) or ``None`` if any constituent set is empty / not
populated for the current scenario.

Consumers that reference any of these multiple times within the same
function MUST cache the result in a local variable — calling the helper
twice rebuilds the cross-join.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from flextool.engine_polars.input import FlexData


__all__ = (
    "compute_pss_dt",
    "compute_nodeBalance_dt",
    "compute_nodeState_dt",
    "compute_nodeState_first_dt",
    "compute_nodeState_rp_dt",
    "compute_nodeState_rp_block_first_dt",
    "compute_process_indirect_dt",
)


def compute_pss_dt(flex_data: "FlexData") -> pl.DataFrame | None:
    """Lazy-build the ``(p, source, sink, d, t)`` cross-product from
    ``flex_data.process_source_sink`` × ``flex_data.dt``.

    Returns ``None`` when either constituent is missing / empty.
    """
    pss = getattr(flex_data, "process_source_sink", None)
    dt = getattr(flex_data, "dt", None)
    if pss is None or dt is None or pss.height == 0 or dt.height == 0:
        return None
    return pss.lazy().join(dt.lazy(), how="cross").collect()


def compute_nodeBalance_dt(flex_data: "FlexData") -> pl.DataFrame | None:
    """Lazy-build the ``(n, d, t)`` cross-product from
    ``flex_data.nodeBalance`` × ``flex_data.dt``.

    Returns ``None`` when either constituent is missing / empty.
    """
    nb = getattr(flex_data, "nodeBalance", None)
    dt = getattr(flex_data, "dt", None)
    if nb is None or dt is None or nb.height == 0 or dt.height == 0:
        return None
    return nb.lazy().join(dt.lazy(), how="cross").collect()


def compute_nodeState_dt(flex_data: "FlexData") -> pl.DataFrame | None:
    """Lazy-build the ``(n, d, t)`` cross-product from
    ``flex_data.nodeState`` × ``flex_data.dt``.

    Returns ``None`` when either constituent is missing / empty.
    """
    ns = getattr(flex_data, "nodeState", None)
    dt = getattr(flex_data, "dt", None)
    if ns is None or dt is None or ns.height == 0 or dt.height == 0:
        return None
    return ns.lazy().join(dt.lazy(), how="cross").collect()


def compute_nodeState_first_dt(flex_data: "FlexData") -> pl.DataFrame | None:
    """Lazy-build the first-(d, t)-per-node slice used by the storage
    ``state_start`` / ``roll_continue`` boundary constraints.

    Mirrors the pre-E.3 eager build in
    :func:`flextool.engine_polars.input._load_storage` (which sometimes
    sourced ``first_period`` from
    ``solve_data/period_first_of_solve.csv`` / ``period_first.csv``).
    Here we take the lexicographically smallest ``d`` in ``dt`` — the
    same fallback those callsites use when no explicit first-period CSV
    is present.  Tests:
    ``tests/engine_polars/test_storage_first_period_fallback.py`` covers
    the agreement.

    Returns ``None`` when ``nodeState`` / ``dt`` are missing or empty.
    """
    ns_dt = compute_nodeState_dt(flex_data)
    if ns_dt is None or ns_dt.height == 0:
        return None
    dt = flex_data.dt
    first_period = dt.lazy().select("d").unique().sort("d").head(1)
    return (
        ns_dt.lazy()
        .join(first_period, on="d", how="inner")
        .group_by("n", "d")
        .agg(pl.col("t").min().alias("t"))
        .select("n", "d", "t")
        .collect()
    )


def compute_nodeState_rp_dt(flex_data: "FlexData") -> pl.DataFrame | None:
    """Lazy-build the ``(n, d, t)`` cross-product from
    ``flex_data.nodeState_rp`` × ``flex_data.dt``.

    Index frame for the RP-blended-weights ``v_state_inter``-family
    variables (Phase 5+) and the intra-period state-change branch
    (Phase 6).  Mirrors :func:`compute_nodeState_dt` but restricted to
    nodes participating in ``bind_using_blended_weights``.

    Returns ``None`` when either constituent is missing / empty.
    """
    nsrp = getattr(flex_data, "nodeState_rp", None)
    dt = getattr(flex_data, "dt", None)
    if nsrp is None or dt is None or nsrp.height == 0 or dt.height == 0:
        return None
    return nsrp.lazy().join(dt.lazy(), how="cross").collect()


def compute_nodeState_rp_block_first_dt(
    flex_data: "FlexData",
) -> pl.DataFrame | None:
    """Lazy-build the ``(n, d, t)`` cross-product from
    ``flex_data.nodeState_rp`` × ``flex_data.rp_block_first``.

    Index frame for the RP-blended-weights ``v_state_rp_start``
    variable (Phase 5+): one variable per (n, d, t) with t restricted
    to the first step of each RP block.  ``rp_block_first`` already
    carries the ``(d, t)`` schema (Phase 2), so a straight cross-join
    with the ``(n,)`` ``nodeState_rp`` set yields the desired index.

    Returns ``None`` when ``nodeState_rp`` or ``rp_block_first`` is
    missing / empty.
    """
    nsrp = getattr(flex_data, "nodeState_rp", None)
    rpbf = getattr(flex_data, "rp_block_first", None)
    if (nsrp is None or rpbf is None
            or nsrp.height == 0 or rpbf.height == 0):
        return None
    return nsrp.lazy().join(rpbf.lazy(), how="cross").collect()


def compute_process_indirect_dt(flex_data: "FlexData") -> pl.DataFrame | None:
    """Lazy-build the ``(p, d, t)`` cross-product from
    ``flex_data.process_indirect`` × ``flex_data.dt``.

    Returns ``None`` when either constituent is missing / empty.
    """
    pi = getattr(flex_data, "process_indirect", None)
    dt = getattr(flex_data, "dt", None)
    if pi is None or dt is None or pi.height == 0 or dt.height == 0:
        return None
    return pi.lazy().join(dt.lazy(), how="cross").collect()
