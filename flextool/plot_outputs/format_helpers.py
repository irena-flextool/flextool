"""Value formatting and filename utilities for plot_outputs."""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import pandas as pd
from decimal import Decimal, InvalidOperation
from matplotlib.ticker import Formatter, FuncFormatter

from flextool.lean_parquet import read_lean_parquet


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


class DynamicFormatter(Formatter):
    """Value-axis formatter that adapts to the data range.

    Determines precision from the maximum tick value (2 significant digits):
    - max >= 1,000: commas, no decimals  (e.g. "1,200,000")
    - max >= 10:    no decimals           (e.g. "45")
    - max >= 1:     1 decimal             (e.g. "4.7")
    - max >= 0.1:   2 decimals            (e.g. "0.46")
    - smaller:      more decimals as needed

    Switches to engineering notation (exponent multiple of 3) when
    |max value| > 990,000,000  (e.g. "1.5e9").
    """

    # ── superscript digits for nice exponent display ──
    _SUPERSCRIPTS = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')

    def format_ticks(self, values):
        if len(values) == 0:
            return []

        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return [str(v) for v in values]

        max_abs = max(abs(v) for v in finite)
        if max_abs == 0:
            return ['0' for _ in values]

        # Engineering notation for very large values
        if max_abs > 990_000_000:
            return self._eng_format(values, max_abs)

        # Number of decimals: show ~2 significant digits
        # decimals = max(0, 1 - floor(log10(max_abs)))
        log_max = math.floor(math.log10(max_abs))
        decimals = max(0, 1 - log_max)

        # Commas for thousands and above
        if max_abs >= 1_000:
            fmt_spec = f',.{decimals}f'
        else:
            fmt_spec = f'.{decimals}f'

        return [format(v, fmt_spec) for v in values]

    def _eng_format(self, values, max_abs):
        """Format with engineering notation (exponent as multiple of 3)."""
        exp = 3 * (math.floor(math.log10(max_abs)) // 3)
        divisor = 10 ** exp

        # Determine decimals for the divided values (2 sig digits)
        max_divided = max_abs / divisor
        log_div = math.floor(math.log10(max_divided)) if max_divided > 0 else 0
        decimals = max(0, 1 - log_div)

        fmt_spec = f',.{decimals}f' if max_divided >= 1_000 else f'.{decimals}f'
        exp_str = str(exp).translate(self._SUPERSCRIPTS)

        result = []
        for v in values:
            if not math.isfinite(v):
                result.append(str(v))
            elif v == 0:
                result.append('0')
            else:
                divided = v / divisor
                formatted = format(divided, fmt_spec)
                result.append(f'{formatted}×10{exp_str}')
        return result

    def __call__(self, x, pos=None):
        """Fallback for individual tick formatting (cursor display, width estimates)."""
        if not math.isfinite(x):
            return str(x)
        if x == 0:
            return '0'
        max_abs = abs(x)
        if max_abs > 990_000_000:
            exp = 3 * (math.floor(math.log10(max_abs)) // 3)
            divisor = 10 ** exp
            divided = x / divisor
            log_div = math.floor(math.log10(abs(divided))) if divided != 0 else 0
            dec = max(0, 1 - log_div)
            exp_str = str(exp).translate(self._SUPERSCRIPTS)
            return f'{format(divided, f".{dec}f")}×10{exp_str}'
        log_max = math.floor(math.log10(max_abs))
        decimals = max(0, 1 - log_max)
        if max_abs >= 1_000:
            return format(x, f',.{decimals}f')
        return format(x, f'.{decimals}f')


def _get_value_formatter(axis_tick_format, idx: int):
    """Return a tick formatter for subplot idx.

    axis_tick_format can be:
      None           → sig-figs FuncFormatter (default, 5 sig figs, plain notation)
      'dynamic'      → DynamicFormatter (adapts to data range)
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
    if spec == 'dynamic':
        return DynamicFormatter()
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


# ---------------------------------------------------------------------------
#  Timeline breaks
# ---------------------------------------------------------------------------

def load_timeline_breaks(*parquet_dirs: str | os.PathLike) -> set[str]:
    """Load timeline break timesteps from one or more parquet directories.

    Reads ``timeline_breaks.parquet`` from each directory and returns the
    union of all break timestep identifiers (e.g. ``{'t1336', 't2848'}``).
    Returns an empty set if no files are found.
    """
    break_times: set[str] = set()
    for d in parquet_dirs:
        path = os.path.join(str(d), 'timeline_breaks.parquet')
        if os.path.exists(path):
            try:
                df = read_lean_parquet(path)
                if 'time' in df.columns:
                    break_times.update(df['time'].astype(str))
            except Exception:
                pass
    return break_times


def split_at_timeline_breaks(
    df: pd.DataFrame,
    break_times: set[str],
) -> list[pd.DataFrame]:
    """Split a DataFrame into contiguous segments at timeline discontinuities.

    Returns a list of DataFrames, one per contiguous block.  Plotting each
    segment separately produces clean gaps (lines/areas stop and restart)
    rather than plunging to zero at NaN values.

    Works with both simple and MultiIndex row indices.
    """
    if not break_times or df.empty:
        return [df]

    # Find the time level in the index
    if isinstance(df.index, pd.MultiIndex):
        time_level = df.index.nlevels - 1  # default: last level
        for i, name in enumerate(df.index.names):
            if name and str(name).lower() in ('time', 't'):
                time_level = i
                break
        time_vals = df.index.get_level_values(time_level).astype(str)
    else:
        time_vals = df.index.astype(str)

    # Find integer positions where breaks occur
    break_positions = [i for i, t in enumerate(time_vals) if t in break_times]
    if not break_positions:
        return [df]

    # Split into contiguous segments
    segments: list[pd.DataFrame] = []
    prev = 0
    for pos in break_positions:
        if prev < pos:
            segments.append(df.iloc[prev:pos])
        prev = pos
    if prev < len(df):
        segments.append(df.iloc[prev:])

    return segments


def insert_timeline_breaks(
    df: pd.DataFrame,
    break_times: set[str],
    gap_rows: int = 3,
) -> pd.DataFrame:
    """Insert NaN rows before timeline discontinuities for visual gaps.

    Inserts *gap_rows* NaN rows (default 3) between contiguous blocks
    so the gap is clearly visible in plots.
    """
    if not break_times or df.empty:
        return df

    segments = split_at_timeline_breaks(df, break_times)
    if len(segments) <= 1:
        return df

    # Insert NaN rows between segments
    parts: list[pd.DataFrame] = []
    for i, seg in enumerate(segments):
        if i > 0:
            prev_seg = parts[-1]
            gap_index = prev_seg.index[-1:].repeat(gap_rows)
            nan_block = pd.DataFrame(
                np.nan,
                index=gap_index,
                columns=df.columns,
            )
            parts.append(nan_block)
        parts.append(seg)

    return pd.concat(parts)
