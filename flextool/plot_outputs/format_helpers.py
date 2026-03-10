"""Value formatting and filename utilities for plot_outputs."""
import logging

import numpy as np
import pandas as pd
from decimal import Decimal, InvalidOperation
from matplotlib.ticker import FuncFormatter


def _sig_figs_fmt(x, pos, n: int = 5) -> str:
    """Format x with n significant figures, plain notation, no trailing zeros."""
    if x == 0:
        return '0'
    try:
        d = Decimal(str(x))
        rounded = d.quantize(Decimal(10) ** (d.adjusted() - n + 1))
        result = f'{rounded:f}'
        if '.' in result:
            result = result.rstrip('0').rstrip('.')
        return result
    except (InvalidOperation, ValueError):
        return str(x)


def _get_value_formatter(axis_tick_format, idx: int):
    """Return a tick formatter for subplot idx.

    axis_tick_format can be:
      None           → sig-figs FuncFormatter (default, 5 sig figs, plain notation)
      ',.0f'         → StrMethodFormatter applied to all subplots
      [',.0f', '.2%'] → per-subplot StrMethodFormatter; sig-figs default beyond list length
    The format spec is a standard Python format spec (without braces), e.g. ',.0f', '.2%'.
    """
    if axis_tick_format is None:
        return FuncFormatter(_sig_figs_fmt)
    if isinstance(axis_tick_format, str):
        spec = axis_tick_format
    elif isinstance(axis_tick_format, list):
        entry = axis_tick_format[idx] if idx < len(axis_tick_format) else None
        if entry is None:
            return FuncFormatter(_sig_figs_fmt)
        spec = entry
    else:
        return FuncFormatter(_sig_figs_fmt)
    def _fmt_with_spec(x, pos, _spec=str(spec)):
        try:
            return format(x, _spec)
        except (ValueError, TypeError) as e:
            logging.error(f"axis_tick_format: cannot format value {x!r} with spec {_spec!r}: {e}")
            return str(x)
    return FuncFormatter(_fmt_with_spec)


def generate_split_filename(
    base_name: str,
    plot_dir: str,
    extension: str,
    file_idx: int | None = None,
    needs_split: bool = False,
    file_member: str | None = None,
) -> str:
    """Generate filename with appropriate suffix based on splitting needs.

    - No splitting: base_name.extension
    - With file_member only: base_name_member.extension
    - With splitting: base_name_01.extension, base_name_02.extension, ...
    - With both: base_name_member_01.extension, base_name_member_02.extension, ...

    File index uses leading zeros for numbers < 10 (e.g., _01, _02, ..., _09, _10).
    """
    name = base_name
    if file_member is not None:
        name = f'{name}_{file_member}'
    if not needs_split:
        return f'{plot_dir}/{name}.{extension}'
    else:
        idx_str = f'{file_idx:02d}'
        return f'{plot_dir}/{name}_{idx_str}.{extension}'


def split_into_chunks(items, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _chunk_average_df(df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    """Chunk-average a DataFrame along its (simple) index.

    Divides the index into consecutive blocks of `chunk_size` rows,
    averages each block, and labels the result with the first original
    index label of each chunk.
    """
    chunk_ids = np.arange(len(df)) // chunk_size
    first_labels = df.index[::chunk_size]
    averaged = df.groupby(chunk_ids).mean()
    averaged.index = first_labels[:len(averaged)]
    averaged.index.name = df.index.name
    return averaged
