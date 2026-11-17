"""Per-(entity, period) values of the period-valued entity parameters, read through the shape resolver.

The lifetimes, ``invest_cost``, the net salvage value and ``discount_rate`` of units, nodes and connections are a
constant or a period map (``PARAM_ALLOWED_SHAPES``).  The payback time is asked for by its role (``payback_time``):
each entity is read under the name it gives the role, ``payback_time`` or else ``lifetime_technical``
(:mod:`flextool.engine.input._lifetime_roles`).  Every reader of them goes through
:func:`_ed_period_param_lf`, so a constant, a ``period`` map and a map with the unnamed ``x`` index read alike, a
map with another index is an error naming the entity, and a stochastic branch period inherits its anchor's value.

A leaf module: the readers in ``_derived``, ``_existing`` and ``_npv`` import it at module level, so it imports
``_direct`` (which imports ``_existing``) only inside its functions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import polars as pl

from flextool.engine.input._axis_enums import align_join_dtypes, axis_lazyframe
from flextool.engine.input._lifetime_roles import ENTITY_CLASSES, ROLE_CHAIN, source_role_sources
from flextool.engine.preprocess.params._branch import active_branch_anchor_pairs, inherit_active_branch_periods

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource


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
        for name, entities in _names_of(source, cls, parameter_name):
            param = _e_period_param_union(source, name, classes=(cls,), period_filter=period_filter)
            if param is None:
                continue
            frame = param.frame if isinstance(param.frame, pl.LazyFrame) else param.frame.lazy()
            frame = frame.select("e", "d", pl.col("value").cast(pl.Float64))
            if entities is not None:
                frame, keep = align_join_dtypes(frame, pl.LazyFrame({"e": entities}), ["e"])
                frame = frame.join(keep, on="e", how="semi")
            parts.append(frame)
    if not parts:
        return None
    lf = pl.concat(parts).collect()
    lf = inherit_active_branch_periods(lf, group_keys=("e",), period_col="d").lazy()
    # Restrict to ``periods``: semi-join in the frame's own d dtype.
    periods_d = axis_lazyframe({"d": list(periods)})
    lf, periods_d = align_join_dtypes(lf, periods_d, ["d"])
    return lf.join(periods_d, on="d", how="semi")


def _names_of(source: "InputSource", entity_class: str,
              parameter_name: str) -> list[tuple[str, "list[str] | None"]]:
    """``[(name, entities)]`` to read ``parameter_name`` of ``entity_class`` from: the parameter itself (entities
    None: all), or for a role of :data:`ROLE_CHAIN` the names its entities take it from
    (:func:`source_role_sources`), each with its entities when the class reads more than one name."""
    if parameter_name not in ROLE_CHAIN:
        return [(parameter_name, None)]
    sources = source_role_sources(source, entity_class, parameter_name)
    by_name: dict[str, list[str]] = {}
    for e, name in sources.items():
        by_name.setdefault(name, []).append(e)
    if len(by_name) <= 1:
        return [(name, None) for name in by_name]
    return [(name, sorted(by_name[name])) for name in ROLE_CHAIN[parameter_name] if name in by_name]


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


def at_online_period(ed_lf: pl.LazyFrame, compute: "Callable[[pl.LazyFrame], pl.LazyFrame]",
                     commission: "pl.DataFrame | pl.LazyFrame | None", col: str) -> pl.LazyFrame:
    """``compute(ed_lf)`` (``ed_lf``'s rows, ``col`` the value of each decision ``(e, d)``) with every decision of
    ``commission`` (``[e, d, d_c]``, the decisions with a construction lead time) valued at the period ``d_c`` its
    capacity comes online instead of its decision period ``d``::

        value[e, d] = compute[e, c(d)],   c(d) = d_c of (e, d) in ``commission``, else d

    The one lookup of the period a plant is built to: the period maps of ``lifetime_technical``, ``payback_time``,
    ``invest_cost`` and ``discount_rate`` describe the plant as constructed, so a decision reads them all at
    ``c(d)`` (the cohort's end of life and years in service, its price per year in service, the reported costs and
    the notice of an investment without a cost).  ``fixed_cost`` is not one of them: new capacity pays the fixed
    cost of each period it operates in; the lifetime fixed cost reads it at ``c(d)`` only to sum the window of a
    fixed cost that does not change.  A decision whose ``d_c`` is null (it never comes online) keeps the value of ``d``; a value
    missing at ``d_c`` stays missing (``compute``'s own fill), it is not taken from ``d``.  The columns of
    ``compute(ed_lf)`` and their order are kept; without ``commission`` it is ``compute(ed_lf)`` itself."""
    base = compute(ed_lf)
    if commission is None:
        return base
    cmap = (commission.lazy() if isinstance(commission, pl.DataFrame) else commission).select("e", "d", "d_c")
    if cmap.select(pl.len()).collect().item() == 0:
        return base
    cols = base.collect_schema().names()
    # Labels read back from a csv may be numbers: as text they join the axis labels.
    cmap = cmap.with_columns(pl.col(c).cast(pl.Utf8) for c, t in cmap.collect_schema().items()
                             if t.is_numeric())
    base, cmap = align_join_dtypes(base, cmap, ["e", "d"])
    base_d = base.collect_schema().get("d")
    cmap = cmap.filter(pl.col("d_c").is_not_null())
    if cmap.collect_schema().get("d_c") != base_d:
        cmap = cmap.with_columns(pl.col("d_c").cast(base_d, strict=False))
    at_on = (compute(cmap.select("e", pl.col("d_c").alias("d")).unique(maintain_order=True))
               .select("e", pl.col("d").alias("d_c"), pl.col(col).alias("_at_online")))
    cmap, at_on = align_join_dtypes(cmap, at_on, ["e", "d_c"])
    base, cmap = align_join_dtypes(base, cmap, ["e", "d"])
    return (base
              .join(cmap, on=["e", "d"], how="left", maintain_order="left")
              .join(at_on, on=["e", "d_c"], how="left", maintain_order="left")
              .with_columns(pl.when(pl.col("d_c").is_not_null()).then(pl.col("_at_online"))
                            .otherwise(pl.col(col)).alias(col))
              .select(cols))


def ed_values_online(source: "InputSource", parameter_name: str, ed_lf: pl.LazyFrame, fill: float | None,
                     commission: "pl.DataFrame | pl.LazyFrame | None") -> pl.LazyFrame:
    """:func:`ed_values_on` of each decision ``(e, d)`` of ``ed_lf`` at the period its capacity comes online
    (:func:`at_online_period`: ``d_c`` of ``commission`` for a decision with a construction lead time, else
    ``d``)."""
    return at_online_period(ed_lf, lambda lf: ed_values_on(source, parameter_name, lf, fill=fill), commission,
                            "value")
