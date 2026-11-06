"""Field-by-field comparison of two FlexData values (a rebuild against the cascade's own FlexData).

A rebuild through the production loader call (:func:`tests._support.load.load_scenario`) must give the
FlexData the cascade built and solved (``ScenarioRun.last_flex_data``, the cascade step's own
``flex_data`` object, never a second load).  Set-like frames come back in a different row order
between loads, so frames are compared as row sets: same columns, same dtypes, equal after sorting on
every column.  The comparison is exact (no float tolerance): both sides are built from the same
provider frames and the same DB by the same code.
"""
from __future__ import annotations

import polars as pl

__all__ = ["field_diff", "field_frame"]


def field_frame(value: object) -> object:
    """The frame of a ``Param`` (its ``.frame``); any other value unchanged."""
    if value is not None and hasattr(value, "frame") and isinstance(value.frame, pl.DataFrame):
        return value.frame
    return value


def _sorted(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    df = df.select(columns)
    return df.sort(columns, nulls_last=True) if columns and df.height else df


def field_diff(rebuilt: object, cascade: object) -> str | None:
    """``None`` when the two field values are equal; otherwise a short description of the difference.

    ``Param`` values compare by frame; frames compare as row sets (see the module docstring); any other
    value (sets, scalars, flags) compares with ``==``.
    """
    a, b = field_frame(rebuilt), field_frame(cascade)
    if a is None and b is None:
        return None
    if a is None or b is None:
        side = "rebuilt" if a is None else "cascade"
        return f"{side} is None, the other is {type(b if a is None else a).__name__}"
    if isinstance(a, pl.DataFrame) != isinstance(b, pl.DataFrame):
        return f"types differ: rebuilt {type(a).__name__}, cascade {type(b).__name__}"
    if not isinstance(a, pl.DataFrame):
        return None if a == b else f"values differ: rebuilt {a!r}, cascade {b!r}"
    if set(a.columns) != set(b.columns):
        return f"columns differ: rebuilt {a.columns}, cascade {b.columns}"
    cols = list(a.columns)
    if a.select(cols).schema != b.select(cols).schema:
        return f"dtypes differ: rebuilt {dict(a.schema)}, cascade {dict(b.schema)}"
    if a.height != b.height:
        return f"row counts differ: rebuilt {a.height}, cascade {b.height}"
    sa, sb = _sorted(a, cols), _sorted(b, cols)
    if sa.equals(sb):
        return None
    return f"rows differ (sorted):\n  rebuilt:\n{sa}\n  cascade:\n{sb}"
