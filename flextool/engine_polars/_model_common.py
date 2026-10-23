"""Small helpers shared by ``model.py`` and the model feature modules.

Frame utilities (row checks, Utf8 join keys, renamed variable views), the
arc classes of the capacity constraints, and the densified per-entity
coefficient factors.  Nothing here emits rows or columns.
"""

import polars as pl
from polar_high import Param
from polar_high.engine import Var

from ._axis_enums import rename_to_axis
from ._param_shapes import promote_param_to_dt


def _has_rows(frame) -> bool:
    """True when ``frame`` is populated and non-empty."""
    return frame is not None and frame.height > 0


def _var_view(v: Var, suffix: str, rename: dict, *, to_axis: bool = False) -> Var:
    """``v`` with renamed dims and the same columns.

    Used to index one variable twice in a term (e.g. ``d → d_invest`` to sum
    investments of earlier periods, ``t → last_step`` to pick a step).
    ``to_axis`` renames through :func:`rename_to_axis` (axis-typed column).
    """
    frame = (v.frame.pipe(rename_to_axis, rename) if to_axis
             else v.frame.rename(rename))
    return Var(name=v.name + suffix,
               dims=tuple(rename.get(x, x) for x in v.dims),
               frame=frame, lower=v.lower, upper=v.upper)


def _entity_rows(frame: pl.DataFrame, ent_col: str, members: pl.DataFrame,
                 axis: str) -> pl.DataFrame:
    """Rows of ``frame`` whose ``ent_col`` is in ``members[axis]``, with
    ``ent_col`` renamed to ``axis``.

    ``frame`` is keyed on the entity-union (``e``) or process (``p``) axis
    while ``members`` uses the process or node axis.  Their Enum
    vocabularies differ and polars cannot compare them, so ``members`` is
    cast to ``frame``'s dtype first (every member is in the larger
    vocabulary).
    """
    dtype = frame.schema[ent_col]
    keys = (members.select(pl.col(axis).cast(dtype, strict=False).alias(ent_col))
            .unique())
    return (frame.join(keys, on=ent_col, how="semi")
            .pipe(rename_to_axis, {ent_col: axis}))


def _utf8_keys(lf: "pl.LazyFrame", keys) -> "pl.LazyFrame":
    """Add ``_<k>`` Utf8 copies of ``keys`` — join keys that bridge Enum
    vocabulary edges between frames built by different producers."""
    return lf.with_columns([pl.col(k).cast(pl.Utf8).alias(f"_{k}")
                            for k in keys])


def _indirect_sum_arcs(d) -> "pl.DataFrame | None":
    """Output arcs ``(p, source, sink)`` summed by ``maxOutputSum``.

    ``process_output_flows`` (``source == p``, ``p ∈ process_indirect``,
    ``conversion_flow_coeff ≠ 0`` — zero-coefficient outputs are removed
    from every capacity constraint) restricted to arcs that actually
    carry a ``v_flow`` column (``process_source_sink``) and to processes
    still in ``process_indirect`` — region-filtered / Benders-master
    FlexData narrow ``process_indirect`` and ``process_source_sink`` but
    not necessarily ``process_output_flows``.  ``None`` when empty.
    """
    out_arcs = getattr(d, "process_output_flows", None)
    indirect = getattr(d, "process_indirect", None)
    pss = getattr(d, "process_source_sink", None)
    if (out_arcs is None or out_arcs.height == 0 or indirect is None
            or indirect.height == 0 or pss is None or pss.height == 0):
        return None
    arcs = (out_arcs.select("p", "source", "sink")
            .join(indirect.select("p").unique(), on="p", how="semi")
            .join(pss.select("p", "source", "sink"),
                  on=["p", "source", "sink"], how="semi")
            .unique(maintain_order=True))
    return arcs if arcs.height > 0 else None


def _capped_arcs(d, arcs: "pl.DataFrame") -> "pl.DataFrame":
    """``arcs`` (``p, source, sink``) restricted to the CAPPED class of
    ``model._max_flow_rhs``: drops indirect-process arcs that are not output
    arcs (fuel / input arcs) and ``conversion_flow_coeff = 0`` (uncapped)
    edges."""
    keys = ["p", "source", "sink"]
    out = arcs
    indirect = getattr(d, "process_indirect", None)
    if indirect is not None and indirect.height > 0:
        out_arcs = getattr(d, "process_output_flows", None)
        indir_arcs = out.join(indirect.select("p").unique(), on="p",
                              how="semi")
        if out_arcs is not None and out_arcs.height > 0:
            indir_arcs = indir_arcs.join(out_arcs.select(keys), on=keys,
                                         how="anti")
        out = out.join(indir_arcs.select(keys), on=keys, how="anti")
    return _drop_uncapped(d, out)


def _drop_uncapped(d, arcs: "pl.DataFrame") -> "pl.DataFrame":
    """``arcs`` (any frame carrying ``p, source, sink``) without the
    ``conversion_flow_coeff = 0`` edges (``process_source_sink_uncapped``),
    which are outside every per-edge capacity / ramp / min-load
    constraint.  Columns and dtypes of ``arcs`` are preserved."""
    unc = getattr(d, "process_source_sink_uncapped", None)
    if unc is None or unc.height == 0 or arcs.height == 0:
        return arcs
    keys = ["p", "source", "sink"]
    return (_utf8_keys(arcs.lazy(), keys)
            .join(_utf8_keys(unc.lazy(), keys)
                  .select([f"_{k}" for k in keys]),
                  on=[f"_{k}" for k in keys], how="anti")
            .select(arcs.columns).collect())


def _coef_factor(pairs: "pl.DataFrame", coef: "Param", keys: list,
                 *, default: float = 1.0) -> "Param":
    """Densify a sparse per-``keys`` coefficient ``coef`` over the key-tuples
    present in ``pairs``, filling absent tuples with ``default``.

    Folding a sparse coefficient into a constraint via ``Param * Param`` (an
    inner join) would DROP any tuple missing from ``coef`` — collapsing the
    term to 0 instead of multiplying by the schema default.  This is the same
    hazard as :func:`_availability_factor`; here it covers static per-arc
    coefficients (e.g. ``capacity_min_coeff`` keyed ``(p, sink)``).  The join
    keys are compared as ``Utf8`` to bridge Enum-vocabulary edges between the
    consumer set and the coefficient Param.
    """
    base = pairs.select(keys).unique()
    base_lf = base.lazy().with_columns(
        [pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in keys])
    coef_lf = (coef.frame.lazy()
               .with_columns([pl.col(k).cast(pl.Utf8).alias(f"_{k}") for k in keys])
               .drop(keys))
    join_keys = [f"_{k}" for k in keys]
    merged = (base_lf
              .join(coef_lf, on=join_keys, how="left")
              .with_columns(pl.col("value").fill_null(default))
              .select(*keys, "value")
              .collect())
    return Param(tuple(keys), merged)


def _availability_factor(d, members: "pl.DataFrame", *,
                         axis: str = "p",
                         avail: "Param | None" = None) -> "Param":
    """``availability`` as a Param keyed ``(axis, d, t)`` that COVERS every
    entity in ``members`` — the DB-authored value where one exists, else the
    schema default ``1.0``.  ``axis`` is ``"p"`` for the process factor
    (``p_process_availability``, the default) or ``"n"`` for the node-state
    factor (pass ``avail=d.p_node_availability``).

    The availability Params are SPARSE: their ``*_from_source`` producers emit
    only DB-authored rows, expecting absent entities to fall back to the
    ``availability`` schema default (1.0).  But the LP folds the factor into
    constraint RHSs via ``Param * Param`` / ``Var * Param``, which are INNER
    joins (polar_high contract) — so an entity absent from the sparse Param would
    have its RHS row DROPPED rather than multiplied by 1.0, collapsing the
    bound to 0 and forcing the variable (``v_flow`` / ``v_state``) to 0 once
    SOME other entity authors an availability.  Densifying against the
    consumer's own entity set (left-join + ``fill_null(1.0)``) makes the
    multiply a multiplicative overlay.  Callers guard on ``avail is not None``
    (an all-1.0 factor is a no-op).
    """
    avail = avail if avail is not None else d.p_process_availability
    base = members.select(axis).unique().join(d.dt, how="cross")  # (axis, d, t)
    # availability may be authored scalar / 1d_map[period] / 1d_map[time] /
    # 2d_map, so promote to (axis, d, t) before the left-join.
    avail_lf = promote_param_to_dt(avail, d.dt)
    # The entity axis can carry a different Enum vocabulary on the consumer set
    # vs the availability Param (e.g. node_profile ``n`` vs p_node_availability
    # ``n``): compare on a Utf8 key so value-equal members match.  The
    # original-dtype axis is carried through from ``base`` for the downstream
    # multiply.
    base_lf = base.lazy().with_columns(pl.col(axis).cast(pl.Utf8).alias("_k"))
    avail_lf = (avail_lf.with_columns(pl.col(axis).cast(pl.Utf8).alias("_k"))
                        .drop(axis))
    merged = (base_lf
              .join(avail_lf, on=["_k", "d", "t"], how="left")
              .with_columns(pl.col("value").fill_null(1.0))
              .select(axis, "d", "t", "value")
              .collect())
    return Param((axis, "d", "t"), merged)
