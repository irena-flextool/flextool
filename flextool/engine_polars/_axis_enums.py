"""Canonical axis :class:`pl.Enum` dtypes for FlexData dimension columns.

Phase 2 of the ``pl.Enum`` dtype refactor (see
``specs/enum_dtype_refactor_handoff.md``).  This module is pure — no
side effects on import, no globals mutated, no string-cache enablement.

The vocabulary builder (`build_axis_enums`) was Path-A of the original
refactor plan and never landed any live callers; it has been removed
along with the workdir-CSV vocabulary seed helpers.  The remaining
public surface — :func:`cast_frame_axes` / :func:`cast_value_axes` /
:func:`cast_flexdata_axes` / :func:`schema_dtype` / :func:`cast_dim` /
:func:`empty_like` / :func:`align_join_dtypes` — operates on an
``enums`` dict supplied by the caller (today this is `{}` from
:mod:`._fast_load`; the casters short-circuit to no-ops).

Public API
----------

* :func:`cast_frame_axes` — cast a frame's dim columns in-place.
* :func:`cast_value_axes` — same for arbitrary Param/Value containers.
* :func:`cast_flexdata_axes` — same for the FlexData container.
* :func:`schema_dtype` — column-name → canonical dtype lookup.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    # Avoid a circular import at module-load time — FlexData lives in
    # ``input.py`` which itself transitively imports this module.
    from flextool.engine_polars.input import FlexData


# ---------------------------------------------------------------------------
# Axis families
#
# Every column name on the LHS gets the Enum dtype of the column name on
# the RHS.  Keeps the public mapping concise: callers see one dtype per
# logical axis, and synonym columns (``d_invest``, ``t_previous``, …)
# pick up the matching dtype automatically.

_AXIS_SYNONYMS: dict[str, str] = {
    # Period synonyms
    "d_invest": "d",
    "d_divest": "d",
    "d_previous": "d",
    "d_upper": "d",
    "d_back": "d",
    # Time-step synonyms
    "t_previous": "t",
    "t_previous_within_block": "t",
    "t_previous_within_timeset": "t",
    "t_previous_within_solve": "t",
    "t_upper": "t",
    "t_back": "t",
    "t_source": "t",
    "t_sink": "t",
    # Friendly long names that occasionally leak from helper code.
    "node": "n",
    "process": "p",
    "commodity": "c",
    "group": "g",
    "entity": "e",
}


def cast_frame_axes(
    frame: "pl.DataFrame | pl.LazyFrame",
    enums: dict[str, pl.Enum],
    *,
    strict: bool = False,
) -> "pl.DataFrame | pl.LazyFrame":
    """Cast every dim column in ``frame`` whose name is in ``enums`` to
    the matching Enum dtype.  Columns absent from ``enums`` (value
    columns, unmapped axes, etc.) are left untouched.

    Parameters
    ----------
    frame
        Eager ``pl.DataFrame`` or lazy ``pl.LazyFrame``.  The return
        type matches the input type — no eager materialisation of
        LazyFrames.
    enums
        The mapping returned by :func:`build_axis_enums`.
    strict
        If ``False`` (default), unknown values silently become null
        on cast — useful during the Phase 3 rollout so a missing
        vocabulary entry surfaces as nulls rather than an exception.
        Flip to ``True`` once the loader cascade is clean.

    Notes
    -----
    Skips columns that are already in the target Enum dtype, so the
    cast is idempotent.
    """
    if isinstance(frame, pl.LazyFrame):
        schema = frame.collect_schema()
        columns = list(schema.keys())
    else:
        schema = frame.schema
        columns = frame.columns

    exprs = []
    for col in columns:
        target = enums.get(col)
        if target is None:
            continue
        if schema[col] == target:
            continue
        exprs.append(pl.col(col).cast(target, strict=strict))
    if not exprs:
        return frame
    return frame.with_columns(exprs)


def align_join_dtypes(left: "pl.LazyFrame | pl.DataFrame",
                        right: "pl.LazyFrame | pl.DataFrame",
                        cols: list[str] | tuple[str, ...]):
    """Make the dtypes of ``cols`` on ``left`` and ``right`` match.

    Strategy: prefer the Enum dtype if either side has one (Enum joins
    are zero-copy when both sides match exactly); fall back to
    ``pl.Utf8`` otherwise.  Returns the (possibly modified) (left,
    right) pair — no schema rebuild when the dtypes already agree.

    Used to bridge the cascade-helper boundary where one side comes
    from a fresh CSV read (String) and the other from an Enum-cast
    FlexData field.
    """
    if isinstance(left, pl.LazyFrame):
        lschema = left.collect_schema()
    else:
        lschema = left.schema
    if isinstance(right, pl.LazyFrame):
        rschema = right.collect_schema()
    else:
        rschema = right.schema

    left_casts = []
    right_casts = []
    for c in cols:
        ld = lschema.get(c)
        rd = rschema.get(c)
        if ld is None or rd is None or ld == rd:
            continue
        # Prefer Enum side; otherwise fall back to String coercion.
        if isinstance(ld, pl.Enum):
            target = ld
            right_casts.append(pl.col(c).cast(target, strict=False))
        elif isinstance(rd, pl.Enum):
            target = rd
            left_casts.append(pl.col(c).cast(target, strict=False))
        else:
            # Two non-Enum non-matching dtypes — cast both to String.
            left_casts.append(pl.col(c).cast(pl.Utf8, strict=False))
            right_casts.append(pl.col(c).cast(pl.Utf8, strict=False))
    if left_casts:
        left = left.with_columns(left_casts)
    if right_casts:
        right = right.with_columns(right_casts)
    return left, right


def empty_like(frame: "pl.DataFrame | pl.LazyFrame",
                 columns: list[str] | tuple[str, ...],
                 extra: dict[str, "pl.DataType"] | None = None,
                 *,
                 lazy: bool = False) -> "pl.DataFrame | pl.LazyFrame":
    """Build an empty frame whose dim-column dtypes match ``frame``.

    Designed for cascade scratch frames that were previously declared
    as ``pl.LazyFrame(schema={"e": pl.Utf8, "d": pl.Utf8, ...})`` —
    those break the moment ``frame`` is cast to Enum.  This helper
    inspects ``frame``'s schema for each requested column and reuses
    its dtype, falling back to ``pl.Utf8`` only when the column is
    absent from ``frame``.  ``extra`` lets the caller add value-typed
    columns (typically ``"value": pl.Float64``) that aren't in
    ``frame``.

    Usage::

        bounded_walk = empty_like(anchor_lf, ["e", "d", "d_all"], lazy=True)

    ``frame`` may itself be eager or lazy.
    """
    if isinstance(frame, pl.LazyFrame):
        schema = frame.collect_schema()
    else:
        schema = frame.schema
    out_schema: dict[str, "pl.DataType"] = {}
    for c in columns:
        out_schema[c] = schema.get(c, pl.Utf8)
    if extra:
        out_schema.update(extra)
    if lazy:
        return pl.LazyFrame(schema=out_schema)
    return pl.DataFrame(schema=out_schema)


def cast_dim(col_expr: "pl.Expr",
              enums: "dict[str, pl.Enum] | None",
              axis: str) -> "pl.Expr":
    """Align a populated-frame ``pl.Expr`` with its empty-frame
    :func:`schema_dtype` counterpart.

    Used at every cascade-helper site that pairs an empty-frame branch
    declared with ``schema_dtype(_enums, axis)`` against a populated
    branch that emits dim columns from raw String CSV reads.  When
    ``enums`` is populated, casts ``col_expr`` to the canonical Enum
    dtype for ``axis``; otherwise (and for axes absent from the
    mapping) returns the expression unchanged — preserving current
    behaviour when ``_enums`` is ``None``.

    The non-strict cast nulls out values not in the Enum vocabulary
    (consistent with :func:`cast_frame_axes` defaults).
    """
    if enums is None:
        return col_expr
    dt = enums.get(axis)
    if dt is None:
        return col_expr
    return col_expr.cast(dt, strict=False)


def schema_dtype(enums: "dict[str, pl.Enum] | None",
                   axis: str) -> "pl.DataType":
    """Return the Enum dtype for ``axis`` if ``enums`` is populated.

    Otherwise fall back to :class:`pl.Utf8` (the current default).

    Designed for scratch-frame schema declarations in the broadcast
    cascade and adjacent helpers.  Each site that previously hard-coded
    ``schema={"e": pl.Utf8, "d": pl.Utf8}`` becomes::

        _enums = getattr(flex_data, "_axis_enums", None)
        return pl.DataFrame(schema={
            "e": schema_dtype(_enums, "e"),
            "d": schema_dtype(_enums, "d"),
        })

    With ``flex_data._axis_enums is None`` (the current default during
    the cascade) the lookup returns ``pl.Utf8`` and behavior is
    identical to the hard-coded form.  When a future dispatch sets
    ``flex_data._axis_enums`` before the cascade runs, the scratch
    frames pick up the canonical Enum dtype automatically.
    """
    if enums is None:
        return pl.Utf8
    return enums.get(axis, pl.Utf8)


def cast_value_axes(value, enums: dict[str, pl.Enum], *, strict: bool = False):
    """Recursively cast a value's dim columns to the canonical Enums.

    Handles:
      * :class:`polar_high.Param` — rebuilds with cast frame.
      * :class:`pl.DataFrame` / :class:`pl.LazyFrame` — applies
        :func:`cast_frame_axes`.
      * ``dict`` / ``tuple`` / ``list`` — recursively walks and rebuilds
        the same container shape with cast children.
      * Everything else — returned unchanged.

    Used by :func:`load_flextool` to wrap the return values of each
    ``_load_*`` function in a single shot at the call site, so the
    interior of every loader doesn't need an individual cast injection.
    """
    # Late import to avoid circular dependency.
    from polar_high import Param

    if value is None:
        return value
    if isinstance(value, Param):
        # Operate on the lazy form to avoid forcing a collect() — every
        # Param keeps an internal ``lazy`` LazyFrame regardless of whether
        # ``.frame`` has been materialised yet.
        try:
            cast_lazy = cast_frame_axes(value.lazy, enums, strict=strict)
        except Exception:
            return value
        if cast_lazy is value.lazy:
            return value
        return Param(value.dims, cast_lazy,
                      name=getattr(value, "name", None),
                      _sources=getattr(value, "_sources", None))
    if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
        try:
            return cast_frame_axes(value, enums, strict=strict)
        except Exception:
            return value
    if isinstance(value, dict):
        return {k: cast_value_axes(v, enums, strict=strict)
                for k, v in value.items()}
    if isinstance(value, list):
        return [cast_value_axes(v, enums, strict=strict) for v in value]
    if isinstance(value, tuple):
        return tuple(cast_value_axes(v, enums, strict=strict) for v in value)
    return value


def cast_flexdata_axes(flex_data: "FlexData",
                        enums: dict[str, pl.Enum],
                        *, strict: bool = False) -> "FlexData":
    """Walk every Param / DataFrame / LazyFrame field on ``flex_data``
    and cast dim columns to the canonical Enums in place.  Returns the
    same FlexData (mutated) for fluent use.

    Skips ``block_layout`` and other non-frame fields.  Skips fields
    whose value is ``None``.
    """
    from polar_high import Param

    for f in dataclasses.fields(flex_data):
        val = getattr(flex_data, f.name, None)
        if val is None:
            continue
        if isinstance(val, (Param, pl.DataFrame, pl.LazyFrame)):
            new_val = cast_value_axes(val, enums, strict=strict)
            if new_val is not val:
                setattr(flex_data, f.name, new_val)
    return flex_data


__all__ = [
    "cast_dim",
    "cast_frame_axes",
    "cast_value_axes",
    "cast_flexdata_axes",
    "empty_like",
    "align_join_dtypes",
    "schema_dtype",
]
