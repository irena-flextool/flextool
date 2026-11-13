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
from flextool.process_outputs._timeline_breaks import BREAK_LEVELS as _BREAK_LEVELS, empty_breaks
from flextool.safe_names import safe_file_component


# ── superscript digits for nice exponent display (single source) ──
_SUPERSCRIPTS = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


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
    # Alias to the module-level single source so tick formatting and the
    # value-label formatter share one table.
    _SUPERSCRIPTS = _SUPERSCRIPTS

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


# Module-level instance reused only to delegate the small-value (< 1e6)
# comma-formatting path of format_value_label, so small value labels render
# exactly as the dynamic tick path would. This does NOT affect tick labels.
_VALUE_LABEL_DELEGATE = DynamicFormatter()


def format_value_label(x) -> str:
    """Format a bar value label (the ``'dynamic'`` value-label path only).

    For ``abs(x) >= 1_000_000`` produce compact engineering notation in the
    ×10⁶ / ×10⁹ / ×10¹² style with 5 significant figures on the mantissa
    (trailing zeros and a trailing '.' stripped), e.g. ``16_701_871`` →
    ``"16.701×10⁶"``. Below 1e6 (and for non-finite / zero values) delegate
    to the existing dynamic comma formatting so small labels are unchanged.

    Note: this is for *value labels on bars only*. Axis tick labels keep going
    through DynamicFormatter / _get_value_formatter and must not call this.
    """
    if not math.isfinite(x) or x == 0:
        return _VALUE_LABEL_DELEGATE(x, None)
    abs_x = abs(x)
    if abs_x < 1_000_000:
        return _VALUE_LABEL_DELEGATE(x, None)

    # Engineering exponent (multiple of 3); for abs_x >= 1e6 this is >= 6.
    exp = 3 * (math.floor(math.log10(abs_x)) // 3)
    mantissa = x / 10 ** exp
    # 5 significant figures: |mantissa| lies in [1, 1000).
    decimals = max(0, 4 - math.floor(math.log10(abs(mantissa))))
    mantissa_str = format(mantissa, f'.{decimals}f')
    if '.' in mantissa_str:
        mantissa_str = mantissa_str.rstrip('0').rstrip('.')
    exp_str = str(exp).translate(_SUPERSCRIPTS)
    return f'{mantissa_str}×10{exp_str}'


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
    The name parts come from plot settings and the input data: a ``/``, ``\\``, ``:`` or control character in
    them becomes ``_`` (:func:`flextool.safe_names.safe_file_component`; ``:`` only on Windows), so the file stays
    in *plot_dir*.
    """
    name = safe_file_component(base_name)
    if file_member is not None:
        name = f'{name}_{safe_file_component(file_member)}'
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

TIMELINE_BREAKS_FILE = 'timeline_breaks.parquet'
_PERIOD_NAMES = ('period', 'd')
_TIME_NAMES = ('time', 't')

logger = logging.getLogger(__name__)


def write_timeline_breaks(parquet_dir: str | os.PathLike, breaks: pd.MultiIndex) -> None:
    """Write the ``(period, time)`` break steps to ``<parquet_dir>/timeline_breaks.parquet``.

    Two Utf8 columns ``period`` and ``time``; written also when there are no breaks, so a replay knows the
    scenario has none.
    """
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {lv: pa.array([str(v) for v in breaks.get_level_values(lv)], type=pa.string()) for lv in _BREAK_LEVELS})
    table = table.replace_schema_metadata({b'flextool': json.dumps({}).encode()})
    pq.write_table(table, os.path.join(str(parquet_dir), TIMELINE_BREAKS_FILE))


def load_timeline_breaks(*parquet_dirs: str | os.PathLike) -> pd.MultiIndex:
    """The union of the ``(period, time)`` break steps in ``timeline_breaks.parquet`` of each directory.

    A missing file, or a file of an earlier format (without both a ``period`` and a ``time`` column), contributes
    no breaks: such results are drawn without gaps until the scenario is run again.  Returns an empty
    ``(period, time)`` index when nothing is found.
    """
    frames = []
    for d in parquet_dirs:
        path = os.path.join(str(d), TIMELINE_BREAKS_FILE)
        if not os.path.exists(path):
            continue
        try:
            df = read_lean_parquet(path)
        except Exception:  # noqa: BLE001
            logger.debug("Could not read timeline breaks %s", path, exc_info=True)
            continue
        if not set(_BREAK_LEVELS) <= set(df.columns):
            logger.debug("Timeline breaks %s are in an earlier format (no period and time columns); ignored", path)
            continue
        frames.append(df[_BREAK_LEVELS].astype(str))
    if not frames:
        return empty_breaks()
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return pd.MultiIndex.from_frame(out, names=_BREAK_LEVELS)


def has_timeline_breaks(breaks: pd.MultiIndex | None) -> bool:
    """True when ``breaks`` holds at least one break step."""
    return breaks is not None and len(breaks) > 0


def _level_position(names, candidates: tuple[str, ...]) -> int | None:
    for i, name in enumerate(names):
        if name is not None and str(name).lower() in candidates:
            return i
    return None


def _break_positions(
    df: pd.DataFrame,
    breaks: pd.MultiIndex,
    period: object = None,
) -> np.ndarray:
    """Row positions of ``df`` at which a break starts a new line (sorted, never 0).

    A row index with a period and a time level matches the ``(period, time)`` pairs.  A row index with only a time
    level matches on time, restricted to the period of the frame: ``period`` (the file member the frame was cut
    to), else the periods of a ``period`` column level (a subplot or line per period).  Time matching applies only
    when those periods share one set of break times (when the period is unknown, when the breaks hold exactly one
    period); otherwise no breaks are applied, since the rows are shared.
    """
    none = np.empty(0, dtype=np.intp)
    names = list(df.index.names)
    t_lvl = _level_position(names, _TIME_NAMES)
    if t_lvl is None:
        logger.debug("Timeline breaks not applied: the rows have no time level (%s)", names)
        return none
    times = df.index.get_level_values(t_lvl).astype(str)
    p_lvl = _level_position(names, _PERIOD_NAMES)
    if p_lvl is not None:
        rows = pd.MultiIndex.from_arrays([df.index.get_level_values(p_lvl).astype(str), times], names=_BREAK_LEVELS)
        pos = np.flatnonzero(rows.isin(breaks))
    else:
        b_period = breaks.get_level_values('period').astype(str)
        b_time = breaks.get_level_values('time').astype(str)
        if period is not None:
            periods = [str(period)]
        else:
            c_lvl = _level_position(df.columns.names, _PERIOD_NAMES)
            periods = (None if c_lvl is None
                       else df.columns.get_level_values(c_lvl).astype(str).unique().tolist())
        if periods is None:
            if b_period.nunique() != 1:
                logger.debug("Timeline breaks not applied: time-only rows and breaks in %d periods",
                             b_period.nunique())
                return none
            sel = b_time
        else:
            per = pd.DataFrame({'period': b_period, 'time': b_time})
            per = per[per['period'].isin(periods)]
            time_sets = {p: frozenset(per.loc[per['period'] == p, 'time']) for p in periods}
            if len(set(time_sets.values())) != 1:
                logger.debug("Timeline breaks not applied: periods %s on shared time rows break at "
                             "different steps", periods)
                return none
            sel = pd.Index(sorted(next(iter(time_sets.values()))))
        pos = np.flatnonzero(times.isin(sel))
    return pos[pos > 0]


def split_at_timeline_breaks(
    df: pd.DataFrame,
    breaks: pd.MultiIndex | None,
    period: object = None,
) -> list[pd.DataFrame]:
    """Split a DataFrame into contiguous segments at timeline discontinuities.

    Returns a list of DataFrames, one per contiguous block.  ``breaks`` holds the ``(period, time)`` steps at
    which a new line starts; how a frame's rows are matched to them is described in :func:`_break_positions`.
    """
    if not has_timeline_breaks(breaks) or df.empty:
        return [df]
    pos = _break_positions(df, breaks, period)
    if len(pos) == 0:
        return [df]
    bounds = np.concatenate(([0], pos, [len(df)]))
    return [df.iloc[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


def insert_timeline_breaks(
    df: pd.DataFrame,
    breaks: pd.MultiIndex | None,
    gap_rows: int = 3,
    period: object = None,
) -> pd.DataFrame:
    """Insert NaN rows before timeline discontinuities for visual gaps.

    Inserts *gap_rows* NaN rows (default 3) before each break row so the gap is clearly visible in plots; the gap
    rows carry the index label of the row before them.  ``period``: the period the frame was cut to (a file
    member), for frames whose rows have no period level.
    """
    if not has_timeline_breaks(breaks) or df.empty:
        return df
    pos = _break_positions(df, breaks, period)
    if len(pos) == 0:
        return df
    n = len(df)
    at = np.repeat(pos, gap_rows)
    take = np.insert(np.arange(n), at, np.repeat(pos - 1, gap_rows))
    is_gap = np.insert(np.zeros(n, dtype=bool), at, True)
    out = df.iloc[take]
    labels = out.index
    out = out.reset_index(drop=True).where(np.broadcast_to(~is_gap[:, None], out.shape))
    out.index = labels
    return out


def file_member_period(file_member, file_level_names: list) -> object:
    """The period of a file member when the frame was split into files by period, else ``None``."""
    if file_member is None:
        return None
    members = file_member if isinstance(file_member, tuple) else (file_member,)
    pos = _level_position(file_level_names, _PERIOD_NAMES)
    return None if pos is None or pos >= len(members) else members[pos]


def unique_series_names(columns: list[str], taken: set[str]) -> list[str]:
    """*columns* with each name that is already in *taken* (an index column, ``subplot``) or earlier in the list
    renamed to ``<name> (series)``, then ``<name> (series) 2``, ...; so the CSV header never repeats a name."""
    used = set(taken)
    out = []
    for name in columns:
        unique, n = name, 1
        while unique in used:
            unique = f'{name} (series)' if n == 1 else f'{name} (series) {n}'
            n += 1
        used.add(unique)
        out.append(unique)
    return out


def plot_data_frame(batch: list[tuple[str | None, pd.DataFrame]]) -> pd.DataFrame:
    """The data drawn in one plot file, as one flat table for its CSV.

    *batch* is the ``(subplot_title, frame)`` list a builder drew into one figure. Rows are the x axis (time steps
    or bar categories, from the frame index), columns the plotted series: a column MultiIndex is flattened into one
    header with the level values joined by ``" | "``.  A leading ``subplot`` column names the subplot when the file
    has more than one.  A series whose name is already an index column (an entity called ``time``) or ``subplot``
    is written as ``<name> (series)``.  The gap rows that :func:`insert_timeline_breaks` adds (all empty, repeating
    the previous row's label) are dropped; a row that is empty in the data itself stays.  Empty cells stay empty.
    """
    parts = []
    for title, frame in batch:
        gap = frame.index.duplicated(keep='first') & frame.isna().all(axis=1).to_numpy()
        if gap.any():
            frame = frame.loc[~gap]
        if isinstance(frame.columns, pd.MultiIndex):
            columns = [' | '.join(str(v) for v in col) for col in frame.columns]
        else:
            columns = [str(c) for c in frame.columns]
        index_part = pd.DataFrame(index=frame.index).reset_index()
        taken = {str(c) for c in index_part.columns}
        if len(batch) > 1:
            taken.add('subplot')
        values = frame.set_axis(unique_series_names(columns, taken), axis=1).reset_index(drop=True)
        part = pd.concat([index_part, values], axis=1)
        if len(batch) > 1:
            part.insert(0, 'subplot', '' if title is None else str(title))
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    return parts[0] if len(parts) == 1 else pd.concat(parts, ignore_index=True)


def write_plot_data_csv(fig, image_path: str | os.PathLike) -> None:
    """Write the data drawn in *fig* next to its image file: same stem, ``.csv``.

    Builders store the drawn ``(subplot_title, frame)`` list as ``fig.flextool_plot_batch`` only when asked to keep
    the plot data; a figure without it writes nothing.  A failure is logged as a warning naming the file and
    does not stop the plotting.
    """
    batch = getattr(fig, 'flextool_plot_batch', None)
    if batch is None:
        return
    from pathlib import Path
    csv_path = Path(image_path).with_suffix('.csv')
    try:
        plot_data_frame(batch).to_csv(csv_path, index=False, float_format='%.8g', encoding='utf-8')
    except Exception as exc:  # an opt-in side output must never stop the plotting
        logger.warning("Plot data CSV %s not written: %s", csv_path, exc)
