"""
Shared utilities for subplot grid layout and data extraction.

Functions
---------
_calculate_grid_layout  Grid dimensions (n_rows, n_cols) from n_subs and subplots_per_row.
_get_unique_levels      Unique subplot values from DataFrame column index (integer indices).
_extract_subplot_data   Unified .xs() extraction returning always a DataFrame, never a Series.
"""

import pandas as pd


def _calculate_grid_layout(n_subs: int, subplots_per_row: int) -> tuple[int, int]:
    """Return (n_rows, n_cols) for a grid of n_subs subplots.

    Args:
        n_subs: Total number of subplots.
        subplots_per_row: Maximum columns per row.

    Returns:
        (n_rows, n_cols) as integers.
    """
    n_cols = min(subplots_per_row, n_subs)
    n_rows = (n_subs + n_cols - 1) // n_cols  # Ceiling division
    return n_rows, n_cols


def _get_unique_levels(df_columns: pd.Index, level_indices: list) -> list:
    """Return unique subplot values from DataFrame column index.

    Args:
        df_columns: The .columns attribute of a DataFrame.
        level_indices: Integer position(s) of the level(s) to extract.
            Empty list → returns [None] (single plot, no subplotting).
            Single element → returns list of scalars.
            Multiple elements → returns list of tuples.

    Returns:
        List of subplot values (scalars, tuples, or [None]).
    """
    if not level_indices:
        return [None]
    if len(level_indices) == 1:
        return df_columns.get_level_values(level_indices[0]).unique().tolist()
    sub_df = df_columns.to_frame().iloc[:, level_indices].drop_duplicates()
    return [tuple(row) for row in sub_df.values]


def _extract_subplot_data(df: pd.DataFrame, sub, sub_levels: list) -> pd.DataFrame:
    """Extract data for one subplot value, always returning a DataFrame.

    Handles MultiIndex and single-level columns uniformly. Converts any
    Series result of .xs() back to a single-column DataFrame.

    Args:
        df: Source DataFrame with MultiIndex or single-level columns.
        sub: Subplot value (scalar or tuple) to select, or None for the
            whole frame (when sub_levels is empty / subs == [None]).
        sub_levels: Integer level indices that identify this subplot level.

    Returns:
        DataFrame slice for the requested subplot value.
    """
    if sub is None:
        return df
    if len(sub_levels) == 1 and not isinstance(df.columns, pd.MultiIndex):
        result = df[sub]
    elif len(sub_levels) == 1:
        result = df.xs(sub, level=sub_levels[0], axis=1)
    else:
        result = df.xs(sub, level=sub_levels, axis=1)
    if isinstance(result, pd.Series):
        result = result.to_frame()
    return result
