"""Per-(entity, period) values of the period-valued entity parameters, read through the shape resolver.

``lifetime``, ``invest_cost``, ``salvage_value`` and ``discount_rate`` of units, nodes and connections are a
constant or a period map (``PARAM_ALLOWED_SHAPES``).  Every reader of them goes through
:func:`_ed_period_param_lf`, so a constant, a ``period`` map and a map with the unnamed ``x`` index read alike, a
map with another index is an error naming the entity, and a stochastic branch period inherits its anchor's value.

A leaf module: the readers in ``_derived``, ``_existing`` and ``_npv`` import it at module level, so it imports
``_direct`` (which imports ``_existing``) only inside its functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.input._axis_enums import align_join_dtypes, axis_lazyframe
from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs, inherit_active_branch_periods

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource

ENTITY_CLASSES = ("unit", "node", "connection")


def _ed_period_param_lf(source: "InputSource",
                        parameter_name: str,
                        periods: list[str]) -> pl.LazyFrame | None:
    """``unit/node/connection.<parameter_name>`` as ``(e, d, value)`` over ``periods``.

    Read through the shape resolver (:func:`._direct._e_period_param_union` →
    ``resolve_param_shape`` → ``broadcast_to_period``), one entity class at a time: a constant is broadcast over
    the periods, a period map keeps its own per-period values (also when authored with the silent ``x`` index,
    and also next to a constant on another entity of the same class), and map keys outside ``periods`` are
    dropped.  Every anchor period's explicit value is inherited by its stochastic branch periods
    (:func:`._branch.inherit_active_branch_periods` — unconditionally, not only under recourse, #63) before the
    final restriction to ``periods``.

    Zero values are kept (``filter_zero=False``) so that a branch-native explicit 0 wins over its anchor's
    non-zero value in the expansion (the value only: whether an investment or retirement variable exists does
    not depend on the cost).  Anchor periods are added to the period filter so that their explicit rows survive
    the resolver's inner join until the expansion has used them.  ``None`` when no entity authors the parameter.
    """
    from flextool.engine.preprocess.params._direct import _e_period_param_union

    pairs = active_branch_anchor_pairs()
    anchors = ([] if pairs is None
               else [a for a in pairs["anchor"].unique(maintain_order=True).to_list()
                     if a not in periods])
    period_filter = axis_lazyframe({"d": list(periods) + anchors}).collect()
    parts = []
    for cls in ENTITY_CLASSES:
        param = _e_period_param_union(source, parameter_name, classes=(cls,), period_filter=period_filter)
        if param is not None:
            frame = param.frame if isinstance(param.frame, pl.LazyFrame) else param.frame.lazy()
            parts.append(frame.select("e", "d", pl.col("value").cast(pl.Float64)))
    if not parts:
        return None
    lf = pl.concat(parts).collect()
    lf = inherit_active_branch_periods(lf, group_keys=("e",), period_col="d").lazy()
    # Restrict to ``periods``: semi-join in the frame's own d dtype.
    periods_d = axis_lazyframe({"d": list(periods)})
    lf, periods_d = align_join_dtypes(lf, periods_d, ["d"])
    return lf.join(periods_d, on="d", how="semi")


def _ed_period_value_map(source: "InputSource", parameter_name: str,
                         periods: list[str]) -> dict[tuple[str, str], float]:
    """:func:`_ed_period_param_lf` as a ``{(e, d): value}`` dict (``{}`` when not authored)."""
    lf = _ed_period_param_lf(source, parameter_name, periods)
    if lf is None:
        return {}
    return {(e, d): v for e, d, v in lf.collect().iter_rows()}


def ed_values_on(source: "InputSource", parameter_name: str, ed_lf: pl.LazyFrame,
                 fill: float | None) -> pl.LazyFrame:
    """``ed_lf`` (columns ``[e, d, ...]``) with a Float64 ``value`` column: ``parameter_name[e, d]`` read by
    :func:`_ed_period_param_lf` over the periods of ``ed_lf``, ``fill`` where the entity has no value in that
    period (not authored, or a period map without that period).

    ``fill=None`` leaves those rows null, for a caller that must tell a missing value from a given one (the
    lifetime of an annuity: a missing lifetime is an input error, not a default).  ``ed_lf`` keeps its rows and
    its columns; the join aligns the axis dtypes (:func:`align_join_dtypes`).
    """
    periods = [d for d in ed_lf.select("d").unique(maintain_order=True).collect()["d"].to_list() if d is not None]
    values = _ed_period_param_lf(source, parameter_name, periods) if periods else None
    if values is None:
        return ed_lf.with_columns(value=pl.lit(fill, dtype=pl.Float64))
    left, right = align_join_dtypes(ed_lf, values, ["e", "d"])
    out = left.join(right, on=["e", "d"], how="left", maintain_order="left")
    return out if fill is None else out.with_columns(pl.col("value").fill_null(fill))
