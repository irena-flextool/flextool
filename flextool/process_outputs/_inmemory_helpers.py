"""Δ.31 — In-memory FlexData → pandas helpers for read_parameters / read_sets.

These helpers translate the polars long-form FlexData fields into the
pandas wide-format shapes that the legacy CSV-based ``read_parameters``
and ``read_sets`` produced.  They are stateless utilities; the
per-attribute mapping lives in
:mod:`flextool.process_outputs.read_parameters` and
:mod:`flextool.process_outputs.read_sets`.

Naming convention: dim columns in FlexData are short (``n``, ``p``,
``d``, ``t``, ``g``, ``c``, ``e``, …).  The legacy pandas namespace
spells them out in full (``node``, ``process``, ``period``, ``time``,
``group``, ``commodity``, ``entity``, …).  :data:`DIM_NAMES` is the
canonical translation table.

Failure mode: every helper raises (``KeyError``, ``ValueError``)
loudly when a FlexData field is absent or has an unexpected schema.
The legacy CSV path was tolerant of empty / missing files — the
in-memory replacement is strict so authoring bugs surface at the
call-site instead of producing silently-empty outputs.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd
import polars as pl


# ---------------------------------------------------------------------------
# Canonical dim-name translation table.
#
# FlexData uses short, model-internal names (``n``, ``p``, ``d``, ``t``).
# Legacy pandas output namespace spells them out so plot / write_outputs
# / out_*.py modules read them as ``period`` / ``time`` / ``node`` /
# ``process`` etc.  Keep this map in sync with the per-field comments in
# :class:`flextool.engine_polars.input.FlexData`.
DIM_NAMES: dict[str, str] = {
    "n": "node",
    "p": "process",
    "g": "group",
    "c": "commodity",
    "f": "profile",
    "d": "period",
    "t": "time",
    "e": "entity",
    "i": "tier",
    "ud": "upDown",
    "r": "reserve",
    "source": "source",
    "sink": "sink",
    "method": "method",
    "param": "param",
    # Storage / time-helper extensions
    "td": "td",
    # Block / branch / lookback names that already match
    "b_first": "b_first",
}


def long_dim(d: str) -> str:
    """Return the long pandas dim-name for a FlexData short dim."""
    if d in DIM_NAMES:
        return DIM_NAMES[d]
    return d


# ---------------------------------------------------------------------------
# Frame → pandas helpers
# ---------------------------------------------------------------------------


def _to_pandas(frame_pl: "pl.DataFrame") -> pd.DataFrame:
    """``polars.DataFrame.to_pandas()`` with a small empty-frame guard."""
    if frame_pl is None:
        raise ValueError("expected a polars DataFrame, got None")
    return frame_pl.to_pandas()


def wide_per_entity(
    frame_pl: "pl.DataFrame",
    *,
    row_dims: Sequence[str],
    col_dim: str,
    value: str = "value",
    row_names: Sequence[str] | None = None,
    col_name: str | None = None,
) -> pd.DataFrame:
    """Pivot a polars long-form frame to a single-level-column pandas
    DataFrame.

    Parameters
    ----------
    frame_pl : polars.DataFrame
        Source frame; must carry ``row_dims + (col_dim, value)``.
    row_dims : sequence of str
        Dim columns that become the (multi-)row-index.
    col_dim : str
        Dim column that becomes the column header.
    value : str, default "value"
        Column carrying the numeric values.
    row_names : sequence of str, optional
        If supplied, the row index level names are set to these.
        Otherwise the source dim names are used (translated via
        :data:`DIM_NAMES`).
    col_name : str, optional
        Column index name (e.g. ``"node"``, ``"entity"``).  Defaults
        to the translated ``col_dim``.
    """
    pdf = _to_pandas(frame_pl)
    pivoted = pdf.pivot(index=list(row_dims), columns=col_dim, values=value)
    pivoted = pivoted.astype(float)
    if row_names is not None:
        pivoted.index.names = list(row_names)
    else:
        pivoted.index.names = [long_dim(d) for d in row_dims]
    pivoted.columns.name = col_name if col_name is not None else long_dim(col_dim)
    return pivoted


def wide_multi_col(
    frame_pl: "pl.DataFrame",
    *,
    row_dims: Sequence[str],
    col_dims: Sequence[str],
    value: str = "value",
    row_names: Sequence[str] | None = None,
    col_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Pivot a polars long-form frame to a wide-column pandas DataFrame
    where the columns are a MultiIndex of ``col_dims``.

    Used for ``flow_min`` / ``flow_max`` / ``process_source_sink_varCost`` /
    ``reserve_upDown_group_reservation`` (column MultiIndex of 3 levels).
    """
    pdf = _to_pandas(frame_pl)
    pivoted = pdf.pivot(index=list(row_dims), columns=list(col_dims), values=value)
    pivoted = pivoted.astype(float)
    if row_names is not None:
        pivoted.index.names = list(row_names)
    else:
        pivoted.index.names = [long_dim(d) for d in row_dims]
    if col_names is not None:
        pivoted.columns.names = list(col_names)
    else:
        pivoted.columns.names = [long_dim(d) for d in col_dims]
    return pivoted


def series_with_index(
    frame_pl: "pl.DataFrame",
    *,
    dim: str,
    value: str = "value",
    name: str | None = None,
) -> pd.Series:
    """Polars long-form (single-dim) → pandas Series with a named index."""
    pdf = _to_pandas(frame_pl)
    s = pdf.set_index(dim)[value].astype(float)
    s.index.name = name if name is not None else long_dim(dim)
    return s


def series_from_two_dim(
    frame_pl: "pl.DataFrame",
    *,
    dims: Sequence[str],
    value: str = "value",
    names: Sequence[str] | None = None,
) -> pd.Series:
    """Polars long-form (multi-dim) → pandas Series with MultiIndex."""
    pdf = _to_pandas(frame_pl)
    s = pdf.set_index(list(dims))[value].astype(float)
    if names is not None:
        s.index.names = list(names)
    else:
        s.index.names = [long_dim(d) for d in dims]
    return s


def series_with_multi_index(
    frame_pl: "pl.DataFrame",
    *,
    dims: Sequence[str],
    value: str = "value",
    names: Sequence[str] | None = None,
) -> pd.Series:
    """polars long-form → pandas Series whose index is a MultiIndex.

    The legacy CSV path produced these for parameters with
    ``header=[0, 1, 2], index_col=0`` and a single ``value`` row that
    pandas exposed as a Series — e.g. ``process_sink_conversion_flow_coeff``,
    ``reserve_upDown_group_penalty``.  The Series's index carries the
    multi-key tuple per cell; the in-memory equivalent is a Series with
    a MultiIndex.
    """
    return series_from_two_dim(frame_pl, dims=dims, value=value, names=names)


# ---------------------------------------------------------------------------
# Set helpers (read_sets)
# ---------------------------------------------------------------------------


def to_index(frame_pl: "pl.DataFrame", *, dim: str, name: str | None = None) -> pd.Index:
    """Polars (single-dim) → pandas Index with a named axis."""
    pdf = _to_pandas(frame_pl.select(dim))
    return pd.Index(pdf[dim].tolist(), name=name if name is not None else long_dim(dim))


def to_multi_index(
    frame_pl: "pl.DataFrame",
    *,
    dims: Sequence[str],
    names: Sequence[str] | None = None,
) -> pd.MultiIndex:
    """Polars long-form → pandas MultiIndex over the supplied dims.

    The frame is restricted to the ``dims`` columns (de-duplicated /
    not — left to the caller; we don't introduce extra ordering).
    """
    pdf = _to_pandas(frame_pl.select(list(dims)))
    if names is None:
        names = [long_dim(d) for d in dims]
    return pd.MultiIndex.from_frame(pdf, names=list(names))


def empty_index(name: str | None = None, dtype: str = "object") -> pd.Index:
    """Empty :class:`pd.Index` with a stable ``name`` / ``dtype``."""
    return pd.Index([], dtype=dtype, name=name)


def empty_multi_index(names: Sequence[str]) -> pd.MultiIndex:
    """Empty :class:`pd.MultiIndex` with ``len(names)`` levels."""
    return pd.MultiIndex.from_arrays([[]] * len(names), names=list(names))


# ---------------------------------------------------------------------------
# Solve-name injection
# ---------------------------------------------------------------------------


def with_solve_column(frame_pl: "pl.DataFrame", solve_name: str) -> "pl.DataFrame":
    """Return ``frame_pl`` with a leading ``solve`` column = ``solve_name``.

    FlexData fields drop the ``solve`` column at load time (see
    ``input.py:_read_long`` family).  The legacy pandas namespace
    keeps it as the leftmost index level.  Inject it back in so the
    pivoted DataFrame's row MultiIndex starts with ``solve``.
    """
    if frame_pl is None:
        return None
    return frame_pl.with_columns(pl.lit(solve_name).alias("solve"))


def add_solve_to_pandas(df: "pd.DataFrame | pd.Series", solve_name: str) -> "pd.DataFrame | pd.Series":
    """Prepend a constant ``solve`` level to the row index.

    Used when the frame already carries other dims and we want to
    inject the solve level *after* pivoting (so wide-column pivots
    don't wear the constant column unnecessarily).
    """
    if isinstance(df.index, pd.MultiIndex):
        new = pd.MultiIndex.from_arrays(
            [[solve_name] * len(df.index)] + [df.index.get_level_values(i) for i in range(df.index.nlevels)],
            names=["solve"] + list(df.index.names),
        )
    else:
        new = pd.MultiIndex.from_arrays(
            [[solve_name] * len(df.index), list(df.index)],
            names=["solve", df.index.name],
        )
    out = df.copy()
    out.index = new
    return out


__all__ = [
    "DIM_NAMES",
    "long_dim",
    "wide_per_entity",
    "wide_multi_col",
    "series_with_index",
    "series_from_two_dim",
    "series_with_multi_index",
    "to_index",
    "to_multi_index",
    "empty_index",
    "empty_multi_index",
    "with_solve_column",
    "add_solve_to_pandas",
]
