"""Row filters for polars frames and :class:`Param` objects by an entity set (the region / master keep-filters)."""
from __future__ import annotations

import polars as pl

from polar_high import Param
from flextool.engine.input._axis_enums import cast_dim


# ---------------------------------------------------------------------------
# Helpers for filtering polars frames / Params
# ---------------------------------------------------------------------------


def _is_in_keep(col: str, keep: set[str]) -> pl.Expr:
    """Membership test for ``pl.col(col)`` against *keep* that tolerates
    keep elements outside the column's *original* Enum vocabulary by
    upcasting the column to the live (widened) Enum first.

    ``keep`` is built by the region splitter and may include synthetic
    virtual-entity tokens (``hf_pipe_*`` / ``*__export__*`` /
    ``*__import__*``) that the Spine-DB-derived axis_enums don't
    contain.  ``split()`` widens the global axis_enums ContextVar to
    include those virtual tokens before the filter runs; here we
    upcast the column to that widened vocabulary via
    :func:`cast_dim` (``enums=None`` reads the live ContextVar).  The
    upcast is a strict superset operation (every original value is in
    the wider Enum), the ``is_in`` then succeeds natively, and the
    filter's output retains the widened Enum dtype.
    """
    return cast_dim(pl.col(col), None, col).is_in(list(keep))


def _filter_frame(df: pl.DataFrame | None, col: str,
                  keep: set[str]) -> pl.DataFrame | None:
    if df is None:
        return None
    if col not in df.columns:
        return df
    return df.filter(_is_in_keep(col, keep))


def _filter_frame_multi(df: pl.DataFrame | None,
                        cond_cols: list[tuple[str, set[str]]]) -> pl.DataFrame | None:
    if df is None:
        return None
    out = df
    for col, keep in cond_cols:
        if col in out.columns:
            out = out.filter(_is_in_keep(col, keep))
    return out


def _filter_param(p: Param | None, col: str,
                  keep: set[str]) -> Param | None:
    if p is None:
        return None
    if col not in p.dims:
        return p
    new_frame = p.frame.filter(_is_in_keep(col, keep))
    return Param(p.dims, new_frame, name=p.name)
