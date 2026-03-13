"""Axis formatting utilities shared across all plot types."""
import re
import pandas as pd
from matplotlib.ticker import MaxNLocator


def _is_datetime_format(s: str) -> bool:
    """Check if a string matches ISO datetime pattern like 2023-01-01T00:00:00."""
    return bool(re.match(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', str(s)))


def _normalize_axis_bounds(raw) -> list | str | None:
    """Convert axis_bounds setting to a list of (min, max) | None entries, or a string keyword.

    Accepts:
      'shared'                 → return 'shared' (resolved later with actual data)
      'independent' or None    → return None (each subplot auto-scales independently)
      [min, max]               → single pair applied to all subplots
      [[min, max], [], [0, 1]] → per-subplot; empty list means auto-scale
    Returns None if raw is falsy or 'independent'.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        lower = raw.strip().lower()
        if lower == 'shared':
            return 'shared'
        if lower == 'independent':
            return None
        return None
    if not raw:
        return None
    if isinstance(raw[0], (int, float)):
        return [(raw[0], raw[1])]
    result = []
    for item in raw:
        result.append((item[0], item[1]) if item else None)
    return result


# Backward-compatible aliases
_normalize_axis_scale = _normalize_axis_bounds


def _subplot_axis_bounds(axis_bounds: list | None, idx: int) -> tuple | None:
    """Return the (min, max) bounds for subplot idx, or None for auto."""
    if not axis_bounds:
        return None
    if len(axis_bounds) == 1:
        return axis_bounds[0]
    return axis_bounds[idx] if idx < len(axis_bounds) else None


# Backward-compatible alias
_subplot_axis_scale = _subplot_axis_bounds


def _apply_subplot_label(ax, xlabel, ylabel, idx: int, row: int, col: int, n_rows: int,
                         expand_label_pad: float = 0) -> None:
    """Apply xlabel/ylabel to ax, supporting both str (positional) and list (per-subplot).

    Parameters
    ----------
    expand_label_pad : float
        Extra labelpad (in points) to push the ylabel further left when
        expand-axis secondary axes occupy space to the left of the main axes.
    """
    ylabel_pad = expand_label_pad if expand_label_pad else None
    if isinstance(ylabel, list):
        val = ylabel[idx] if idx < len(ylabel) else None
        if val:
            ax.set_ylabel(val, labelpad=ylabel_pad)
    elif ylabel and col == 0:
        ax.set_ylabel(ylabel, labelpad=ylabel_pad)

    if isinstance(xlabel, list):
        val = xlabel[idx] if idx < len(xlabel) else None
        if val:
            ax.set_xlabel(val, labelpad=2)
    elif xlabel and row == n_rows - 1:
        ax.set_xlabel(xlabel)


def _set_calendar_xticks(ax, time_index, plot_width_inches: float) -> None:
    """Set x-tick labels for non-datetime (calendar-like) string indices."""
    max_label_len = max(len(str(s)) for s in time_index)
    label_width_inches = max_label_len * 0.08 + 0.3  # ~0.08in per char + gap
    max_labels = max(2, int(plot_width_inches / label_width_inches))

    # Minimum interval needed between ticks (in data points)
    min_interval = max(1, len(time_index) // max_labels)

    # Round up to next "nice" calendar-like interval
    nice_intervals = [1, 2, 4, 6, 12, 24, 48, 168, 336, 672, 1344, 2688, 8760]
    interval = nice_intervals[-1]
    for ni in nice_intervals:
        if ni >= min_interval:
            interval = ni
            break

    tick_positions = list(range(0, len(time_index), interval))
    if not tick_positions:
        tick_positions = [0]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([time_index[i] for i in tick_positions], rotation=0, ha='left')


def _set_datetime_xticks(ax, time_index, plot_width_inches: float) -> None:
    """Set x-tick labels for datetime string indices, with smart spacing and minor ticks."""
    dt = pd.to_datetime(time_index)
    formatted = dt.strftime('%m-%dT%H:%M')

    # Estimate how many labels fit
    min_spacing_inches = 1.1  # label width (~0.8in) + gap
    max_labels = max(2, int(plot_width_inches / min_spacing_inches))

    # Calculate data resolution in hours from first two points
    if len(dt) >= 2:
        resolution_hours = (dt[1] - dt[0]).total_seconds() / 3600
    else:
        resolution_hours = 1.0

    # Minimum interval needed between ticks (in hours)
    total_hours = len(time_index) * resolution_hours
    min_interval_hours = total_hours / max_labels

    # Round up to next "nice" interval
    nice_intervals = [1, 2, 3, 4, 6, 8, 12, 24, 48, 72, 168, 336, 720]
    interval_hours = nice_intervals[-1]
    for ni in nice_intervals:
        if ni >= min_interval_hours:
            interval_hours = ni
            break

    # Convert interval from hours to number of data points
    interval_points = max(1, round(interval_hours / resolution_hours))

    # Find aligned starting position
    if interval_hours >= 24:
        # Align to midnight
        start = next((i for i, t in enumerate(dt) if t.hour == 0 and t.minute == 0), 0)
    else:
        # Align to even hour boundaries
        start = next(
            (i for i, t in enumerate(dt) if t.hour % interval_hours == 0 and t.minute == 0),
            0,
        )

    positions = list(range(start, len(time_index), interval_points))
    if not positions:
        positions = [0]

    ax.set_xticks(positions)
    ax.set_xticklabels([formatted[i] for i in positions], rotation=0, ha='left')

    # When label interval is a multiple of 24h and > 24h, add minor ticks every 24h
    if interval_hours > 24 and interval_hours % 24 == 0:
        daily_points = max(1, round(24 / resolution_hours))
        minor_start = next(
            (i for i, t in enumerate(dt) if t.hour == 0 and t.minute == 0), 0
        )
        minor_positions = [i for i in range(minor_start, len(time_index), daily_points)
                           if i not in positions]
        ax.set_xticks(minor_positions, minor=True)
        ax.grid(True, which='minor', alpha=0.15)


_VALUE_TICK_CHAR_WIDTH = 0.08   # Approximate width per character (inches) at tick font size
_VALUE_TICK_LABEL_HEIGHT = 0.35  # Minimum comfortable spacing between y-axis tick labels (inches)


def _estimate_value_nbins(
    data_min: float, data_max: float,
    axis_length_inches: float,
    formatter,
    is_horizontal_axis: bool = True,
    min_nbins: int = 3,
    max_nbins: int = 8,
) -> int:
    """Estimate how many value-axis ticks fit without overlapping labels.

    For horizontal value axes (labels spaced horizontally): estimates label
    width from formatted sample values.
    For vertical value axes (labels spaced vertically): uses a fixed label
    height estimate.
    """
    if axis_length_inches <= 0:
        return min_nbins

    if is_horizontal_axis:
        sample_values = [data_min, data_max]
        mid = (data_min + data_max) / 2
        if mid != data_min and mid != data_max:
            sample_values.append(mid)
        max_chars = max(len(formatter(v, 0)) for v in sample_values)
        label_size = max_chars * _VALUE_TICK_CHAR_WIDTH + 0.15
    else:
        label_size = _VALUE_TICK_LABEL_HEIGHT

    nbins = int(axis_length_inches / label_size)
    return max(min_nbins, min(max_nbins, nbins))


def set_smart_xticks(ax, time_index, plot_width_inches: float) -> None:
    """Set x-tick labels smartly based on whether the index contains datetime strings.

    For datetime strings: parse, shorten labels, and space ticks based on plot width.
    For non-datetime strings: estimate label width and choose spacing from calendar-like
    intervals (24, 168, 336, 720) based on how many labels fit.
    """
    if len(time_index) == 0:
        return
    if _is_datetime_format(time_index[0]):
        _set_datetime_xticks(ax, time_index, plot_width_inches)
    else:
        _set_calendar_xticks(ax, time_index, plot_width_inches)
