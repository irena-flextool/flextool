import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from decimal import Decimal, InvalidOperation
import logging
import time
from contextlib import contextmanager

logging.getLogger('matplotlib.category').disabled = True
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

# Field names for plot settings
PLOT_FIELD_NAMES = {
    'plot_name', 'map_dimensions_for_plots', 'subplots_per_row', 'legend',
    'bar_orientation', 'base_length', 'max_subplots_per_file', 'max_items_per_file',
    'time_average_duration', 'xlabel', 'ylabel', 'value_label', 'axis_scale_min_max',
    'axis_tick_format', 'always_include_zero'
}


def _is_single_config(d: dict) -> bool:
    """Check if a dict is a single config (has field names) vs multi-config (has config names)."""
    return any(k in PLOT_FIELD_NAMES for k in d)


def _is_datetime_format(s: str) -> bool:
    """Check if a string matches ISO datetime pattern like 2023-01-01T00:00:00."""
    return bool(re.match(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}', str(s)))


def _normalize_axis_scale(raw) -> list | None:
    """Convert axis_scale_min_max setting to a list of (min, max) | None entries.

    Accepts:
      [min, max]               → single pair applied to all subplots
      [[min, max], [], [0, 1]] → per-subplot; empty list means auto-scale
    Returns None if raw is falsy.
    """
    if not raw:
        return None
    if isinstance(raw[0], (int, float)):
        return [(raw[0], raw[1])]
    result = []
    for item in raw:
        result.append((item[0], item[1]) if item else None)
    return result


def _subplot_axis_scale(axis_scale_min_max: list | None, idx: int) -> tuple | None:
    """Return the (min, max) scale for subplot idx, or None for auto."""
    if not axis_scale_min_max:
        return None
    if len(axis_scale_min_max) == 1:
        return axis_scale_min_max[0]
    return axis_scale_min_max[idx] if idx < len(axis_scale_min_max) else None


def _apply_subplot_label(ax, xlabel, ylabel, idx: int, row: int, col: int, n_rows: int) -> None:
    """Apply xlabel/ylabel to ax, supporting both str (positional) and list (per-subplot)."""
    if isinstance(ylabel, list):
        val = ylabel[idx] if idx < len(ylabel) else None
        if val:
            ax.set_ylabel(val)
    elif ylabel and col == 0:
        ax.set_ylabel(ylabel)

    if isinstance(xlabel, list):
        val = xlabel[idx] if idx < len(xlabel) else None
        if val:
            ax.set_xlabel(val, labelpad=2)
    elif xlabel and row == n_rows - 1:
        ax.set_xlabel(xlabel)


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


def set_smart_xticks(ax, time_index, plot_width_inches: float) -> None:
    """Set x-tick labels smartly based on whether the index contains datetime strings.

    For datetime strings: parse, shorten labels, and space ticks based on plot width.
    For non-datetime strings: estimate label width and choose spacing from calendar-like
    intervals (24, 168, 336, 720) based on how many labels fit.
    """
    if len(time_index) == 0:
        return

    if not _is_datetime_format(time_index[0]):
        # Estimate label width from the longest string in the index
        max_label_len = max(len(str(s)) for s in time_index)
        label_width_inches = max_label_len * 0.08 + 0.3  # ~0.08in per char + gap
        effective_width = plot_width_inches * 0.85
        max_labels = max(2, int(effective_width / label_width_inches))

        # Minimum interval needed between ticks (in data points)
        min_interval = max(1, len(time_index) // max_labels)

        # Round up to next "nice" calendar-like interval
        nice_intervals = [1, 2, 4, 6, 12, 24, 48, 168, 336, 720]
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
        return

    # Parse datetimes
    dt = pd.to_datetime(time_index)
    formatted = dt.strftime('%m-%dT%H:%M')

    # Estimate how many labels fit
    min_spacing_inches = 1.1  # label width (~0.8in) + gap
    effective_width = plot_width_inches * 0.85
    max_labels = max(2, int(effective_width / min_spacing_inches))

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
    positions = []
    if interval_hours >= 24:
        # Align to midnight
        start = None
        for i, t in enumerate(dt):
            if t.hour == 0 and t.minute == 0:
                start = i
                break
        if start is None:
            start = 0
    else:
        # Align to even hour boundaries
        start = None
        for i, t in enumerate(dt):
            if t.hour % interval_hours == 0 and t.minute == 0:
                start = i
                break
        if start is None:
            start = 0

    positions = list(range(start, len(time_index), interval_points))
    if not positions:
        positions = [0]

    ax.set_xticks(positions)
    ax.set_xticklabels([formatted[i] for i in positions], rotation=0, ha='left')

    # When label interval is a multiple of 24h and > 24h, add minor ticks every 24h
    if interval_hours > 24 and interval_hours % 24 == 0:
        daily_points = max(1, round(24 / resolution_hours))
        # Find first midnight for minor tick alignment
        minor_start = None
        for i, t in enumerate(dt):
            if t.hour == 0 and t.minute == 0:
                minor_start = i
                break
        if minor_start is None:
            minor_start = 0
        minor_positions = [i for i in range(minor_start, len(time_index), daily_points)
                           if i not in positions]
        ax.set_xticks(minor_positions, minor=True)
        ax.grid(True, which='minor', alpha=0.15)

# Performance tracking
PERF_STATS = {}

@contextmanager
def time_block(name, verbose=False):
    """Context manager to time a block of code and accumulate stats"""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name not in PERF_STATS:
            PERF_STATS[name] = {'total': 0}
        PERF_STATS[name]['total'] += elapsed
        if verbose:
            print(f"  [{name}] {elapsed:.4f}s")

def print_perf_summary():
    """Print summary of performance statistics"""
    if not PERF_STATS:
        return

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Operation':<50}{'Total':>10}")
    print("-"*80)

    for name, stats in sorted(PERF_STATS.items(), key=lambda x: x[1]['total'], reverse=True):
        print(f"{name:<50} {stats['total']:>10.4f}s")

    print("="*80 + "\n")


def split_into_chunks(items, chunk_size):
    """Split a list into chunks of specified size."""
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def generate_split_filename(base_name, plot_dir, extension, file_idx=None, needs_split=False):
    """
    Generate filename with appropriate suffix based on splitting needs.

    - No splitting: base_name.extension
    - With splitting: base_name_01.extension, base_name_02.extension, ...

    File index uses leading zeros for numbers < 10 (e.g., _01, _02, ..., _09, _10).
    """
    if not needs_split:
        return f'{plot_dir}/{base_name}.{extension}'
    else:
        # Format with leading zero for numbers < 10
        idx_str = f'{file_idx:02d}'
        return f'{plot_dir}/{base_name}_{idx_str}.{extension}'


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


def plot_dict_of_dataframes(results_dict, plot_dir, plot_settings,
        active_settings=['default'], plot_rows=(0,167), delete_existing_plots=True):
    """
    Plot dataframes from a dictionary according to key suffixes.

    Args:
        results_dict: Dictionary of pandas DataFrames
        plot_dir: Directory to save PNG files
        result_set_map: Dictionary mapping result keys to (filename, plot_flag) tuples
        delete_existing_plots: If True, delete all existing plots in plot_dir before creating new ones (default: True)
    """

    # Empty plot dir if requested
    if delete_existing_plots:
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            if filename == 'config.yaml':
                continue
            if os.path.isfile(file_path):
                os.remove(file_path)

    for key, df_orig in results_dict.items():
        # Filter plot settings that are active based on chosen_settings (or default to default)
        if key not in plot_settings:
            continue
        if df_orig.empty:
            continue
        chosen_settings = []
        entry = plot_settings[key]
        if _is_single_config(entry):
            # Single config (has field names like plot_name, map_dimensions_for_plots)
            if 'default' in active_settings:
                chosen_settings.append(entry)
        else:
            # Multi-config (keys are config names like 'default', 'sum_periods')
            for setting_name, setting in entry.items():
                if setting_name in active_settings:
                    chosen_settings.append(setting)

        # Loop through all active settings for this dataframe
        for setting in chosen_settings:
            map_dims = setting.get('map_dimensions_for_plots')
            if not map_dims or len(map_dims) < 2:
                continue
            index_types, rules = map_dims[0], map_dims[1]
            if not rules:
                continue
            rules = rules.replace('_', '')

            if index_types is None:
                continue
            if len(index_types.split('_')) == 2:
                df_index_levels, df_columns_levels = index_types.split('_')
            else:
                raise ValueError(f'plot setting {key}: map_dimensions_for_plots first element should contain one underscore to separate row and column index indicators')

            # Extract settings and apply defaults if needed
            key_name = setting.get('plot_name', key)
            plot_name = key_name
            subplots_per_row = setting.get('subplots_per_row', 3)
            legend_position = setting.get('legend', 'right')
            bar_orientation = setting.get('bar_orientation', 'horizontal')
            base_length = setting.get('base_length', 4)
            max_subplots_per_file = setting.get('max_subplots_per_file', 9)
            max_items_per_file = setting.get('max_items_per_file')
            time_avg_duration = setting.get('time_average_duration')
            xlabel = setting.get('xlabel')
            ylabel = setting.get('ylabel')
            value_label = setting.get('value_label', False)
            axis_scale_min_max = _normalize_axis_scale(setting.get('axis_scale_min_max'))
            axis_tick_format = setting.get('axis_tick_format')
            always_include_zero = setting.get('always_include_zero', True)

            if 't' in rules and 'i' not in rules:
                chart_type = 'time'
                df = df_orig.iloc[plot_rows[0]:plot_rows[1]].copy()
            elif 'i' in rules:
                chart_type = 'time'
                df = df_orig.copy()
            else:
                chart_type = 'bar'
                df = df_orig.copy()
            
            # Check the df dimensions match the number of row and column levels
            nr_row_levels = df.index.nlevels
            if len(rules) != nr_row_levels + df.columns.nlevels:
                raise ValueError(f"Number of plot_type rules different from the number of index + column levels in the dataframe. {key_name}")

            levels_to_sort = [i for i, c in enumerate(df_columns_levels) if c in ('e', 'g')]
            if levels_to_sort:
                df = df.sort_index(axis=1, level=levels_to_sort, sort_remaining=False)

            # Sum sum_levels for row index
            sum_row_levels = [i for i, char in enumerate(rules[:nr_row_levels]) if char == 'm']
            if sum_row_levels:
                keep_levels = [i for i in range(nr_row_levels) if i not in sum_row_levels]
                if len(keep_levels) > 0:
                    df = df.groupby(level=keep_levels).sum()
                    for i in sum_row_levels:
                        rules = rules[:i] + rules[i + 1:]
                else:
                    df = df.sum(axis=0).to_frame().T
                    df.index = ['sum']
                    df.index.name = 'sum'

            # Sum sum_levels for column index
            nr_column_levels = df.columns.nlevels
            sum_column_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'm']
            if sum_column_levels:
                keep_levels = [i for i in range(nr_column_levels) if i not in sum_column_levels]
                if len(keep_levels) > 0:
                    df = df.T.groupby(level=keep_levels).sum().T
                    for i in sum_column_levels:
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]
                else:
                    df = df.sum(axis=1).to_frame()
                    df.columns = ['sum']
                    df.columns.name = 'sum'

            # Update number of row and column levels after sums may have removed levels
            nr_row_levels = df.index.nlevels
            nr_column_levels = df.columns.nlevels

            # Average mean_levels for row index
            mean_row_levels = [i for i, char in enumerate(rules[:nr_row_levels]) if char == 'a']
            if mean_row_levels:
                keep_levels = [i for i in range(nr_row_levels) if i not in mean_row_levels]
                if len(keep_levels) > 1:
                    df = df.groupby(level=keep_levels).mean()
                    for i in mean_row_levels:
                        rules = rules[:i] + rules[i + 1:]
                else:
                    df = df.mean(axis=0).to_frame().T
                    df.index = ['mean']
                    df.index.name = 'mean'

            # Average mean_levels for column index
            mean_column_levels = [i for i, char in enumerate(rules[nr_row_levels:]) if char == 'a']
            if mean_column_levels:
                keep_levels = [i for i in range(nr_column_levels) if i not in mean_column_levels]
                if len(keep_levels) > 1:
                    df = df.T.groupby(level=keep_levels).mean().T
                    for i in mean_column_levels:
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]
                else:
                    df = df.mean(axis=1).to_frame()
                    df.columns = ['mean']
                    df.columns.name = 'mean'

            nr_row_levels = df.index.nlevels
            nr_column_levels = df.columns.nlevels

            # Chunk-average the 'i' row level if time_avg_duration is set
            i_positions = [pos for pos, char in enumerate(rules[:nr_row_levels]) if char == 'i']
            if i_positions and time_avg_duration:
                i_pos = i_positions[0]
                chunk_size = int(time_avg_duration)
                other_levels = [lv for lv in range(nr_row_levels) if lv != i_pos]
                if other_levels and nr_row_levels > 1:
                    # MultiIndex: split by other levels, chunk-average each group, recombine
                    parts = []
                    group_level = other_levels[0] if len(other_levels) == 1 else other_levels
                    for group_key, group_df in df.groupby(level=group_level):
                        flat = group_df.droplevel(other_levels)
                        averaged = _chunk_average_df(flat, chunk_size)
                        # Rebuild MultiIndex with original level structure
                        if not isinstance(group_key, tuple):
                            group_key = (group_key,)
                        new_tuples = []
                        for idx_val in averaged.index:
                            row = [None] * nr_row_levels
                            row[i_pos] = idx_val
                            for j, lv in enumerate(other_levels):
                                row[lv] = group_key[j]
                            new_tuples.append(tuple(row))
                        averaged.index = pd.MultiIndex.from_tuples(
                            new_tuples, names=df.index.names
                        )
                        parts.append(averaged)
                    df = pd.concat(parts)
                else:
                    # Simple index: chunk-average directly
                    df = _chunk_average_df(df, chunk_size)

                nr_row_levels = df.index.nlevels
                nr_column_levels = df.columns.nlevels

            # Decide how to plot
            if (not df.empty) & (len(df) > 0):
                # print('')
                # Track level names before rearrangement so we can rebuild rules
                # to match the actual level order after stacking/unstacking
                level_names_before = list(df.index.names) + list(df.columns.names)
                name_to_rule = dict(zip(level_names_before, rules))

                bar_line_levels = [i for i, c in enumerate(rules) if c == "b" or c == 't' or c == 'i']
                pre_stack_nr_row_levels = df.index.nlevels  # Save before stacking changes nlevels
                for i, bar_line_level in enumerate(reversed(bar_line_levels)):
                    # Move bar_line_levels from columns to index (not including period (0), which is there already)
                    if bar_line_level >= pre_stack_nr_row_levels:
                        df = df.stack(bar_line_level - pre_stack_nr_row_levels, future_stack=True)
                        if isinstance(df, pd.Series):
                            df = df.to_frame()

                # Move column-type levels from row index to columns
                # (symmetric to stacking bar/line levels from columns to index above)
                column_type_levels = [i for i, c in enumerate(rules[:df.index.nlevels])
                                      if c in ('u', 'g', 's', 'l', 'e')]
                for col_level in reversed(column_type_levels):
                    if col_level < df.index.nlevels:
                        df = df.unstack(col_level)
                        if isinstance(df, pd.Series):
                            df = df.to_frame()

                # Rebuild rules to match actual level order after stacking/unstacking
                # (stack/unstack place moved levels at the innermost position, which
                # shifts the correspondence between rules characters and DataFrame levels)
                level_names_after = list(df.index.names) + list(df.columns.names)
                if (len(set(level_names_before)) == len(level_names_before)
                        and all(n is not None for n in level_names_before)):
                    try:
                        rules = ''.join(name_to_rule[n] for n in level_names_after)
                    except KeyError:
                        pass  # names changed unexpectedly, keep original rules

                sum_mean_row_levels = [i for i, char in enumerate(rules[:df.index.nlevels]) if char == 'm' or char == 'a']
                if df.index.nlevels - len(sum_mean_row_levels) > 0:
                    for i in reversed(sum_mean_row_levels):
                        df = df.droplevel(i)
                        rules = rules[:i] + rules[i + 1:]
                
                sum_mean_column_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'm' or char == 'a']
                if df.columns.nlevels - len(sum_mean_column_levels) > 0:
                    for i in reversed(sum_mean_column_levels):
                        df = df.droplevel(i, axis=1)
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]
                
                # Get level locations for different types of operations
                grouped_bar_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'g']
                stack_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 's']
                expand_axis_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'e']
                subplot_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'u']
                line_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'l']

                # Determine original subplots
                if not subplot_levels:
                    all_subs = [None]
                elif len(subplot_levels) == 1:
                    all_subs = df.columns.get_level_values(subplot_levels[0]).unique().tolist()
                else:
                    sub_df = df.columns.to_frame().iloc[:, subplot_levels].drop_duplicates()
                    all_subs = [tuple(row) for row in sub_df.values]

                # Determine items for splitting
                if chart_type == 'time':
                    # For time plots, items are lines (from line_levels or stack_levels)
                    default_max_items = 9
                    if max_items_per_file is None:
                        max_items_per_file = default_max_items
                    item_levels = line_levels if line_levels else stack_levels
                    if not item_levels:
                        all_items = [None]
                    elif len(item_levels) == 1:
                        all_items = df.columns.get_level_values(item_levels[0]).unique().tolist()
                    else:
                        item_level_names = [df.columns.names[i] for i in item_levels]
                        item_df = df.columns.to_frame()[item_level_names].drop_duplicates()
                        all_items = [tuple(row) for row in item_df.values]
                else:  # bar chart
                    # For bar plots, items are bars (from df.index)
                    default_max_items = 20
                    if max_items_per_file is None:
                        max_items_per_file = default_max_items
                    all_items = df.index.tolist()

                n_items = len(all_items)
                needs_item_split = n_items > max_items_per_file

                # Two different splitting strategies based on whether subplot_levels exist
                if subplot_levels:
                    # STRATEGY 1: When subplot_levels exist
                    # Items that exceed limit create additional subplots (not files)
                    # Files are only split when total effective subplots exceed max_subplots_per_file

                    # Create item chunks
                    if needs_item_split:
                        item_chunks = list(split_into_chunks(all_items, max_items_per_file))
                    else:
                        item_chunks = [all_items]

                    # Build list of effective subplots: (original_sub, item_chunk)
                    # Each combination of original subplot and item chunk becomes one effective subplot
                    effective_subplots = []
                    for sub in all_subs:
                        for item_chunk in item_chunks:
                            effective_subplots.append((sub, item_chunk))

                    # Now split effective subplots into files
                    n_effective_subs = len(effective_subplots)
                    needs_file_split = n_effective_subs > max_subplots_per_file

                    if needs_file_split:
                        file_chunks = list(split_into_chunks(effective_subplots, max_subplots_per_file))
                    else:
                        file_chunks = [effective_subplots]

                    # Plot each file
                    for file_idx, file_chunk in enumerate(file_chunks, start=1):
                        # Collect unique original subs and item chunks for this file
                        subs_in_file = []
                        items_in_file = set()
                        for sub, item_chunk in file_chunk:
                            if sub not in subs_in_file:
                                subs_in_file.append(sub)
                            items_in_file.update(item_chunk if item_chunk[0] is not None else [])

                        # Filter dataframe for this file
                        df_chunk = df.copy()

                        # Filter by subplots in this file (filter columns)
                        if len(subs_in_file) < len(all_subs):
                            if len(subplot_levels) == 1:
                                level_name = df.columns.names[subplot_levels[0]]
                                mask = df_chunk.columns.get_level_values(level_name).isin(subs_in_file)
                                df_chunk = df_chunk.loc[:, mask]
                            else:
                                col_tuples = df_chunk.columns.to_frame().iloc[:, subplot_levels]
                                mask = col_tuples.apply(tuple, axis=1).isin(subs_in_file)
                                df_chunk = df_chunk.loc[:, mask.values]

                        # Filter by items in this file
                        if items_in_file and needs_item_split:
                            if chart_type == 'bar':
                                df_chunk = df_chunk.loc[df_chunk.index.isin(items_in_file)]
                            else:
                                if item_levels and len(item_levels) == 1:
                                    level_name = df.columns.names[item_levels[0]]
                                    mask = df_chunk.columns.get_level_values(level_name).isin(items_in_file)
                                    df_chunk = df_chunk.loc[:, mask]
                                elif item_levels:
                                    item_level_names = [df.columns.names[i] for i in item_levels]
                                    col_tuples = df_chunk.columns.to_frame()[item_level_names]
                                    mask = col_tuples.apply(tuple, axis=1).isin(items_in_file)
                                    df_chunk = df_chunk.loc[:, mask.values]

                        if df_chunk.empty:
                            continue

                        # Generate filename
                        base_filename = f'{plot_name}'

                        filepath = generate_split_filename(
                            base_filename, plot_dir, 'svg',
                            file_idx=file_idx, needs_split=needs_file_split
                        )

                        # Plot the chunk
                        if chart_type == 'time':
                            if stack_levels:
                                plot_dt_stack_sub(df_chunk, plot_name, plot_dir, stack_levels, subplot_levels,
                                    rows=plot_rows, subplots_per_row=subplots_per_row, legend_position=legend_position,
                                    xlabel=xlabel, ylabel=ylabel,
                                    base_width_per_col=6, subplot_height=base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    axis_tick_format=axis_tick_format,
                                    always_include_zero=always_include_zero,
                                    output_filepath=filepath)
                            else:
                                plot_dt_sub_lines(df_chunk, plot_name, plot_dir, subplot_levels, line_levels,
                                    rows=plot_rows, subplots_per_row=subplots_per_row, legend_position=legend_position,
                                    xlabel=xlabel, ylabel=ylabel,
                                    base_width_per_col=6, subplot_height=base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    axis_tick_format=axis_tick_format,
                                    always_include_zero=always_include_zero,
                                    output_filepath=filepath)
                        elif chart_type == 'bar':
                            plot_rowbars_stack_groupbars(df_chunk, plot_name, plot_dir,
                                stack_levels, expand_axis_levels, subplot_levels, grouped_bar_levels,
                                subplots_per_row=subplots_per_row, legend_position=legend_position,
                                xlabel=xlabel, ylabel=ylabel,
                                bar_orientation=bar_orientation, base_bar_length=base_length,
                                value_label=value_label, axis_scale_min_max=axis_scale_min_max,
                                axis_tick_format=axis_tick_format,
                                always_include_zero=always_include_zero,
                                output_filepath=filepath)
                        else:
                            raise ValueError(f'Could not interpret plot rule for {key}')

                else:
                    # STRATEGY 2: When no subplot_levels exist
                    # Items split directly into files

                    if needs_item_split:
                        item_chunks = list(split_into_chunks(all_items, max_items_per_file))
                    else:
                        item_chunks = [all_items]

                    for file_idx, item_chunk in enumerate(item_chunks, start=1):
                        # Filter dataframe for this chunk
                        df_chunk = df.copy()

                        # Filter by item chunk
                        if needs_item_split:
                            if chart_type == 'bar':
                                df_chunk = df_chunk.loc[df_chunk.index.isin(item_chunk)]
                            else:
                                if item_levels and item_chunk[0] is not None:
                                    if len(item_levels) == 1:
                                        level_name = df.columns.names[item_levels[0]]
                                        mask = df_chunk.columns.get_level_values(level_name).isin(item_chunk)
                                        df_chunk = df_chunk.loc[:, mask]
                                    else:
                                        item_level_names = [df.columns.names[i] for i in item_levels]
                                        col_tuples = df_chunk.columns.to_frame()[item_level_names]
                                        mask = col_tuples.apply(tuple, axis=1).isin(item_chunk)
                                        df_chunk = df_chunk.loc[:, mask.values]

                        if df_chunk.empty:
                            continue

                        # Generate filename
                        if chart_type == 'time':
                            base_filename = f'{plot_name}'
                        else:
                            base_filename = f'{plot_name}'

                        filepath = generate_split_filename(
                            base_filename, plot_dir, 'svg',
                            file_idx=file_idx, needs_split=needs_item_split
                        )

                        # Plot the chunk
                        if chart_type == 'time':
                            if stack_levels:
                                plot_dt_stack_sub(df_chunk, plot_name, plot_dir, stack_levels, subplot_levels,
                                    rows=plot_rows, legend_position=legend_position,
                                    xlabel=xlabel, ylabel=ylabel,
                                    base_width_per_col=6, subplot_height=base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    always_include_zero=always_include_zero,
                                    output_filepath=filepath)
                            else:
                                plot_dt_sub_lines(df_chunk, plot_name, plot_dir, subplot_levels, line_levels,
                                    rows=plot_rows, legend_position=legend_position,
                                    xlabel=xlabel, ylabel=ylabel,
                                    base_width_per_col=6, subplot_height=base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    always_include_zero=always_include_zero,
                                    output_filepath=filepath)
                        elif chart_type == 'bar':
                            plot_rowbars_stack_groupbars(df_chunk, plot_name, plot_dir,
                                stack_levels, expand_axis_levels, subplot_levels, grouped_bar_levels,
                                subplots_per_row=subplots_per_row, legend_position=legend_position,
                                xlabel=xlabel, ylabel=ylabel,
                                bar_orientation=bar_orientation, base_bar_length=base_length,
                                value_label=value_label, axis_scale_min_max=axis_scale_min_max,
                                always_include_zero=always_include_zero,
                                output_filepath=filepath)
                        else:
                            raise ValueError(f'Could not interpret plot rule for {key}')
            # else:
                # print('   ...no data')

            plt.close('all')  # Clean up

    # Print summary after all plots
    # print_perf_summary()


def estimate_legend_width(labels, title='', base_width=1.5):
    """
    Estimate the width in inches needed for a legend based on label content.

    Args:
        labels: List of label strings
        title: Legend title string
        base_width: Minimum width in inches (default: 1.5)

    Returns:
        Estimated width in inches
    """
    if not labels:
        return base_width

    # Calculate max label length
    max_label_len = max(len(str(label)) for label in labels)
    title_len = len(str(title)) if title else 0

    # Estimate width: ~0.09 inches per character + base padding
    # This accounts for typical matplotlib font sizes (8-10pt)
    char_width = 0.09
    label_width = max_label_len * char_width
    title_width = title_len * char_width

    # Use the larger of label or title width, plus padding
    estimated_width = max(label_width, title_width, base_width) + 0.8

    return estimated_width


def plot_dt_sub_lines(df_plot, plot_name, plot_dir, sub_levels, line_levels,
    rows=(0,167), subplots_per_row=3, legend_position='right',
    xlabel=None, ylabel=None, base_width_per_col=6, subplot_height=4,
    axis_scale_min_max=None, axis_tick_format=None, always_include_zero=True,
    output_filepath=None):

    # Convert level indices to level names for later use after xs operations
    if isinstance(df_plot.columns, pd.MultiIndex):
        line_level_names = [df_plot.columns.names[i] for i in line_levels]
    else:
        # Single level index - use indices directly
        line_level_names = line_levels

    # Handle empty sub_levels (single plot, no subplotting)
    if not sub_levels:
        subs = [None]
    elif len(sub_levels) == 1:
        subs = df_plot.columns.get_level_values(sub_levels[0]).unique().tolist()
    else:
        # Join multiple levels as tuples
        sub_df = df_plot.columns.to_frame().iloc[:, sub_levels].drop_duplicates()
        subs = [tuple(row) for row in sub_df.values]

    # Calculate subplot grid
    n_subs = len(subs)
    n_cols = min(subplots_per_row, n_subs)
    n_rows = (n_subs + n_cols - 1) // n_cols  # Ceiling division

    # Calculate legend width if needed
    legend_width = 0
    if legend_position == 'all' and n_cols > 1:
        max_legend_width = 0

        for sub in subs:
            # Extract data for this subplot
            if sub is None:
                df_sub_temp = df_plot
            elif len(sub_levels) == 1:
                df_sub_temp = df_plot.xs(sub, level=sub_levels[0], axis=1)
            else:
                df_sub_temp = df_plot.xs(sub, level=sub_levels, axis=1)
            if isinstance(df_sub_temp, pd.Series):
                df_sub_temp = df_sub_temp.to_frame()

            # Get line labels
            is_multiindex = isinstance(df_sub_temp.columns, pd.MultiIndex)
            if is_multiindex:
                if len(line_level_names) == 1:
                    lines_temp = df_sub_temp.columns.get_level_values(line_level_names[0]).unique().tolist()
                else:
                    line_df = df_sub_temp.columns.to_frame()[line_level_names].drop_duplicates()
                    lines_temp = [tuple(row) for row in line_df.values]
            else:
                lines_temp = df_sub_temp.columns.unique().tolist()

            # Format labels
            legend_labels = [str(line) for line in lines_temp]

            # Estimate width
            width = estimate_legend_width(legend_labels)
            max_legend_width = max(max_legend_width, width)

        legend_width = max_legend_width

    if legend_width > 0 and n_cols > 1:
        total_width = base_width_per_col * n_cols + legend_width * (n_cols - 1)
    else:
        total_width = base_width_per_col * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, subplot_height * n_rows))
    if n_subs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Adjust subplot spacing to accommodate legends
    if legend_width > 0 and n_cols > 1:
        # wspace is the width of spacing as a fraction of average axes width
        wspace = legend_width / base_width_per_col
        # Calculate vertical spacing to prevent row overlap
        # Add space for ~1.5 rows of text, normalized to subplot height
        if xlabel:
            hspace = 0.25 / subplot_height
        else:
            hspace = 0.225 / subplot_height
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Get x-axis index (use last level if MultiIndex, otherwise use the index itself)
    if isinstance(df_plot.index, pd.MultiIndex):
        time_index = df_plot.index.get_level_values(-1).astype(str)
    else:
        time_index = df_plot.index.astype(str)

    for idx, sub in enumerate(subs):
        ax = axes[idx]

        if not sub:
            df_sub = df_plot
        elif isinstance(df_plot.columns, pd.MultiIndex):
            if len(sub_levels) > 1:
                df_sub = df_plot.xs(sub, level=sub_levels, axis=1)
            else:
                df_sub = df_plot.xs(sub, level=sub_levels[0], axis=1)
            if isinstance(df_sub, pd.Series):
                df_sub = df_sub.to_frame()
        else:
            df_sub = df_plot.loc[:,sub].to_frame()

        # Get line combinations from line_levels
        if isinstance(df_sub, pd.Series):
            # Only one line to plot
            ax.plot(time_index, df_sub.values, label=str(sub))
        else:
            # Check if columns are MultiIndex
            is_multiindex = isinstance(df_sub.columns, pd.MultiIndex)

            if is_multiindex:
                if len(line_level_names) == 1:
                    lines = df_sub.columns.get_level_values(line_level_names[0]).unique().tolist()
                else:
                    # Join multiple levels as tuples (use names since sub_levels may have been dropped)
                    line_df = df_sub.columns.to_frame()[line_level_names].drop_duplicates()
                    lines = [tuple(row) for row in line_df.values]
            else:
                # Single level index, just get unique column values
                lines = df_sub.columns.unique().tolist()

            # Plot each line
            for line in lines:
                if is_multiindex:
                    if len(line_level_names) == 1:
                        y_data = df_sub.xs(line, level=line_level_names[0], axis=1)
                    else:
                        # For multiple line_levels, apply xs for all levels at once
                        y_data = df_sub.xs(line, level=line_level_names, axis=1)
                else:
                    # Direct column selection for non-MultiIndex
                    y_data = df_sub[line]

                # Sum if there are still multiple columns remaining
                if isinstance(y_data, pd.DataFrame):
                    y_data = y_data.sum(axis=1)

                ax.plot(time_index, y_data.values, label=str(line))

        # Subplot formatting
        if sub is not None:
            ax.set_title(str(sub), pad=2)

        # Determine if legend should be shown
        show_legend = False
        if legend_position == 'all':
            show_legend = True
        elif legend_position == 'right':
            if not sub_levels:
                show_legend = True
            else:
                col = idx % n_cols
                if col == n_cols - 1 or idx == n_subs - 1:
                    show_legend = True

        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, borderaxespad=0.1)

        ax.grid(True, alpha=0.3)

        # Axis scale, formatter, and labels
        row = idx // n_cols
        col = idx % n_cols
        if always_include_zero:
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 0), max(hi, 0))
        scale = _subplot_axis_scale(axis_scale_min_max, idx)
        if scale:
            ax.set_ylim(scale[0], scale[1])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
        ax.yaxis.set_major_formatter(_get_value_formatter(axis_tick_format, idx))
        _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows)

        set_smart_xticks(ax, time_index, base_width_per_col)

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(plot_name)

    # Use provided filepath or generate default
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    else:
        plt.savefig(f'{plot_dir}/{plot_name}.svg', bbox_inches='tight')
    plt.close(fig)


def plot_dt_stack_sub(df_plot, plot_name, plot_dir, stack_levels, sub_levels,
        rows=(0,167), stack_element_to_split=None, subplots_per_row=3,
        legend_position='right',
        xlabel=None, ylabel=None, base_width_per_col=6, subplot_height=4,
        axis_scale_min_max=None, axis_tick_format=None, always_include_zero=True,
        output_filepath=None):

    # Convert level indices to level names for later use after xs operations
    if isinstance(df_plot.columns, pd.MultiIndex):
        stack_level_names = [df_plot.columns.names[i] for i in stack_levels]
    else:
        # Single level index - use indices directly
        stack_level_names = stack_levels

    # Handle empty sub_levels (single plot, no subplotting)
    if not sub_levels:
        subs = [None]
    elif len(sub_levels) == 1:
        subs = df_plot.columns.get_level_values(sub_levels[0]).unique().tolist()
    else:
        # Join multiple levels as tuples
        sub_df = df_plot.columns.to_frame().iloc[:, sub_levels].drop_duplicates()
        subs = [tuple(row) for row in sub_df.values]

    # Calculate subplot grid
    n_subs = len(subs)
    n_cols = min(subplots_per_row, n_subs)
    n_rows = (n_subs + n_cols - 1) // n_cols  # Ceiling division

    # Calculate legend width if needed
    legend_width = 0
    if legend_position == 'all' and n_cols > 1:
        max_legend_width = 0

        for sub in subs:
            # Extract data for this subplot
            if sub is None:
                df_sub_temp = df_plot
            elif len(sub_levels) == 1:
                df_sub_temp = df_plot.xs(sub, level=sub_levels[0], axis=1)
            else:
                df_sub_temp = df_plot.xs(sub, level=sub_levels, axis=1)
            if isinstance(df_sub_temp, pd.Series):
                df_sub_temp = df_sub_temp.to_frame()

            # Get stack labels
            is_multiindex = isinstance(df_sub_temp.columns, pd.MultiIndex)
            if is_multiindex:
                if len(stack_level_names) == 1:
                    stacks_temp = df_sub_temp.columns.get_level_values(stack_level_names[0]).unique().tolist()
                else:
                    stack_df = df_sub_temp.columns.to_frame()[stack_level_names].drop_duplicates()
                    stacks_temp = [tuple(row) for row in stack_df.values]
            else:
                stacks_temp = df_sub_temp.columns.unique().tolist()

            # Format labels
            legend_labels = [str(stack) for stack in stacks_temp]

            # Estimate width
            width = estimate_legend_width(legend_labels)
            max_legend_width = max(max_legend_width, width)

        legend_width = max_legend_width

    if legend_width > 0 and n_cols > 1:
        total_width = base_width_per_col * n_cols + legend_width * (n_cols - 1)
    else:
        total_width = base_width_per_col * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, subplot_height * n_rows))
    if n_subs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Calculate vertical spacing to prevent row overlap
    # Add space for ~1.5 rows of text, normalized to subplot height
    if xlabel:
        hspace = 1 / subplot_height
    else:
        hspace = 0.7 / subplot_height

    # Adjust subplot spacing to accommodate legends
    if legend_width > 0 and n_cols > 1:
        # wspace is the width of spacing as a fraction of average axes width
        wspace = legend_width / base_width_per_col
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
    else:
        fig.subplots_adjust(hspace=hspace)

    # Get x-axis index (use last level if MultiIndex, otherwise use the index itself)
    if isinstance(df_plot.index, pd.MultiIndex):
        time_index = df_plot.index.get_level_values(-1).astype(str)
    else:
        time_index = df_plot.index.astype(str)

    for idx, sub in enumerate(subs):
        ax = axes[idx]

        # Extract data for this subplot using xs
        if sub is None:
            # No sub_levels - use full dataframe
            df_sub = df_plot
        elif len(sub_levels) == 1:
            df_sub = df_plot.xs(sub, level=sub_levels[0], axis=1)
        else:
            # For multiple sub_levels, apply xs for all levels at once
            df_sub = df_plot.xs(sub, level=sub_levels, axis=1)

        # Get stack combinations from stack_levels
        if isinstance(df_sub, pd.Series):
            # Only one series to plot
            df_to_plot = df_sub.to_frame()
        else:
            # Check if columns are MultiIndex
            is_multiindex = isinstance(df_sub.columns, pd.MultiIndex)

            if is_multiindex:
                if len(stack_level_names) == 1:
                    stacks = df_sub.columns.get_level_values(stack_level_names[0]).unique().tolist()
                else:
                    # Join multiple levels as tuples (use names since levels may have been dropped)
                    stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
                    stacks = [tuple(row) for row in stack_df.values]
            else:
                # Single level index, just get unique column values
                stacks = df_sub.columns.unique().tolist()

            # Build DataFrame with columns for each stack element
            data_dict = {}
            for stack in stacks:
                if is_multiindex:
                    if len(stack_level_names) == 1:
                        y_data = df_sub.xs(stack, level=stack_level_names[0], axis=1)
                    else:
                        # For multiple stack_levels, apply xs for all levels at once
                        y_data = df_sub.xs(stack, level=stack_level_names, axis=1)
                else:
                    # Direct column selection for non-MultiIndex
                    y_data = df_sub[stack]

                # Sum if there are still multiple columns remaining
                if isinstance(y_data, pd.DataFrame):
                    y_data = y_data.sum(axis=1)

                data_dict[str(stack)] = y_data

            df_to_plot = pd.DataFrame(data_dict, index=df_sub.index)

        # Reset index to use time only (drop period)
        df_to_plot.index = time_index

        # Split columns with both positive and negative values if requested
        if stack_element_to_split:
            for col_name in stack_element_to_split:
                if col_name in df_to_plot.columns:
                    # Create positive and negative columns using clip
                    df_to_plot[f'{col_name}_pos'] = df_to_plot[col_name].clip(lower=0)
                    df_to_plot[f'{col_name}_neg'] = df_to_plot[col_name].clip(upper=0)
                    # Drop the original column
                    df_to_plot = df_to_plot.drop(columns=[col_name])

        # Create stacked area plot using pandas (handles pos/neg correctly)
        n_columns = len(df_to_plot.columns)
        colors = plt.colormaps['tab10'].colors[:n_columns]
        if n_columns > 10:
            colors = plt.colormaps['tab20'].colors[:n_columns]
        df_to_plot.plot.area(stacked=True, ax=ax, alpha=1.0, legend=False, linewidth=0, color=colors, xlabel="")

        # Subplot formatting
        if sub is not None:
            ax.set_title(str(sub), pad=2)

        # Determine if legend should be shown
        show_legend = False
        if legend_position == 'all':
            show_legend = True
        elif legend_position == 'right':
            if not sub_levels:
                show_legend = True
            else:
                col = idx % n_cols
                if col == n_cols - 1 or idx == n_subs - 1:
                    show_legend = True

        if show_legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, borderaxespad=0.1)

        ax.grid(True, alpha=0.3)

        # Axis scale, formatter, and labels
        row = idx // n_cols
        col = idx % n_cols
        if always_include_zero:
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 0), max(hi, 0))
        scale = _subplot_axis_scale(axis_scale_min_max, idx)
        if scale:
            ax.set_ylim(scale[0], scale[1])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
        ax.yaxis.set_major_formatter(_get_value_formatter(axis_tick_format, idx))
        _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows)

        set_smart_xticks(ax, time_index, base_width_per_col)

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(plot_name)

    # Use provided filepath or generate default
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    else:
        plt.savefig(f'{plot_dir}/{plot_name}.svg', bbox_inches='tight')
    plt.close(fig)


def plot_rowbars_stack_groupbars(df, key_name, plot_dir, stack_levels, expand_axis_levels,
        sub_levels=[], grouped_bar_levels=None,
        legend_position='right', subplots_per_row=3,
        xlabel=None, ylabel=None, bar_orientation='horizontal', base_bar_length=4,
        value_label=False, axis_scale_min_max=None, axis_tick_format=None,
        always_include_zero=True, output_filepath=None):
    """
    Create horizontal stacked and grouped bar plot.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'period' row index and MultiIndex columns
    key_name : str
        Name for the plot (used in title and filename)
    plot_dir : str
        Directory to save the plot
    stack_levels : list of int or None
        Column level indices that create colored segments within each bar (stacked horizontally).
        Mutually exclusive with grouped_bar_levels.
    expand_axis_levels : list of int
        Column level indices that create groups of bars with two-level y-axis (bars + group labels).
        Previously named 'expand_axis_levels'. Can be combined with either stack_levels or grouped_bar_levels.
    sub_levels : list of int, optional
        Column level indices that create separate subplots (default: [])
    grouped_bar_levels : list of int or None, optional
        Column level indices that create grouped bars side-by-side (like pandas grouped bars).
        Each combination creates a separate bar at the same y-position with different colors.
        Mutually exclusive with stack_levels (default: None).
    legend_position : str, optional
        Legend placement: 'all' shows legend on all subplots,
        'right' shows only on rightmost column (default: 'right').

        Note: When legend_position='all' and multiple columns are used,
        the figure width is automatically increased based on legend content
        to prevent overlap with adjacent subplots.
    subplots_per_row : int, optional
        Number of subplots per row (default: 3)
    sum_levels : list of int, optional
        Level indices to collapse by summing (default: [])
    mean_levels : list of int, optional
        Level indices to collapse by averaging (default: [])
    xlabel : str, optional
        Label for x-axis. Applied only to bottom-most subplots (default: None)
    ylabel : str, optional
        Label for y-axis. Applied only to leftmost subplots (default: None)
    bar_orientation : str, optional
        Bar orientation: 'horizontal' for horizontal bars (barh),
        'vertical' for vertical bars (bar) (default: 'horizontal')
    base_bar_length : float, optional
        Base length in inches for the bar extension dimension.
        For horizontal bars: controls width. For vertical bars: controls height (default: 4)
    """
    if key_name == 'unit_outputNode':
        pass

    # Validate mutual exclusivity
    if stack_levels and grouped_bar_levels:
        raise ValueError(
            "Cannot use both 'stack_levels' and 'grouped_bar_levels' simultaneously. "
            "stack_levels creates horizontal stacks within each bar, while "
            "grouped_bar_levels creates separate bars side-by-side. "
            "Choose one approach or use neither for simple bars."
        )

    # Resolve value_label to a format string or None
    if value_label is True or value_label == 'true':
        value_fmt = '%.3g'
    elif value_label:
        value_fmt = str(value_label)
    else:
        value_fmt = None

    # Normalize None to empty list for consistent checking
    if stack_levels is None:
        stack_levels = []
    if grouped_bar_levels is None:
        grouped_bar_levels = []

    # Convert level indices to names for stability
    if isinstance(df.columns, pd.MultiIndex):
        stack_level_names = [df.columns.names[i] for i in stack_levels] if stack_levels else []
        expand_axis_level_names = [df.columns.names[i] for i in expand_axis_levels] if expand_axis_levels else []
        grouped_bar_level_names = [df.columns.names[i] for i in grouped_bar_levels] if grouped_bar_levels else []
    else:
        # Single level index - use indices directly for data access
        stack_level_names = stack_levels
        expand_axis_level_names = [df.columns.name] if expand_axis_levels else []
        grouped_bar_level_names = [df.columns.name] if grouped_bar_levels else []

    # Handle empty sub_levels (single plot, no subplotting)
    if not sub_levels:
        subs = [None]
    elif len(sub_levels) == 1:
        subs = df.columns.get_level_values(sub_levels[0]).unique().tolist()
    else:
        # Join multiple levels as tuples
        sub_df = df.columns.to_frame().iloc[:, sub_levels].drop_duplicates()
        subs = [tuple(row) for row in sub_df.values]

    # Calculate subplot grid
    n_subs = len(subs)
    n_cols = min(subplots_per_row, n_subs)
    n_rows = (n_subs + n_cols - 1) // n_cols  # Ceiling division

    # First pass: calculate number of bars for each subplot to determine heights
    bar_counts = []
    for sub in subs:
        # Extract data for this subplot using xs
        if sub is None:
            df_sub_temp = df
        elif len(sub_levels) == 1 and isinstance(df.columns, pd.MultiIndex):
            df_sub_temp = df.xs(sub, level=sub_levels[0], axis=1)
        elif len(sub_levels) == 1:
            df_sub_temp = df[sub]
        else:
            df_sub_temp = df.xs(sub, level=sub_levels, axis=1)
        if isinstance(df_sub_temp, pd.Series):
            df_sub_temp = df_sub_temp.to_frame()

        # Get unique group combinations from df_sub_temp
        if not expand_axis_levels:
            groups_temp = [None]
        elif len(expand_axis_level_names) == 1:
            groups_temp = df_sub_temp.columns.get_level_values(expand_axis_level_names[0]).unique().tolist()
        else:
            groups_temp = []
            for group_level_name in expand_axis_level_names:
                groups_temp.append(df_sub_temp.columns.get_level_values(group_level_name).tolist())
            groups_temp = list(zip(*groups_temp))

        periods_temp = df_sub_temp.index.tolist()

        # Calculate number of bars
        if not expand_axis_levels:
            n_bars = len(periods_temp)
        else:
            n_bars = len(groups_temp) * len(periods_temp)

        bar_counts.append(n_bars)

    # Calculate dimensions perpendicular to bar extension (based on number of bars)
    if bar_orientation == 'horizontal':
        # For horizontal bars: calculate height for each row
        row_heights = []
        for row in range(n_rows):
            start_idx = row * n_cols
            end_idx = min(start_idx + n_cols, n_subs)
            max_bars_in_row = max(bar_counts[start_idx:end_idx])
            row_height = 0.3 * max_bars_in_row + 0.8
            row_heights.append(row_height)
        col_widths = None  # Not used for horizontal orientation
    else:  # vertical
        # For vertical bars: calculate width for each column
        col_widths = []
        for col in range(n_cols):
            # Get all subplot indices in this column
            col_indices = [col + row * n_cols for row in range(n_rows) if col + row * n_cols < n_subs]
            if col_indices:
                max_bars_in_col = max(bar_counts[idx] for idx in col_indices)
                col_width = 0.3 * max_bars_in_col + 0.8
            else:
                col_width = 1  # Default if no subplots in this column
            col_widths.append(col_width)
        row_heights = None  # Not used for vertical orientation

    # Calculate legend width if needed
    legend_width = 0
    if legend_position == 'all' and n_cols > 1 and (stack_levels or grouped_bar_levels):
        # Determine what the legend labels will be by analyzing stack or grouped bar levels
        max_legend_width = 0

        for sub in subs:
            # Extract data for this subplot
            if sub is None:
                df_sub_temp = df
            elif len(sub_levels) == 1:
                df_sub_temp = df.xs(sub, level=sub_levels[0], axis=1)
            else:
                df_sub_temp = df.xs(sub, level=sub_levels, axis=1)
            if isinstance(df_sub_temp, pd.Series):
                df_sub_temp = df_sub_temp.to_frame()

            # BRANCH: Get legend items based on mode
            if grouped_bar_levels:
                # Grouped bar legend items
                if len(grouped_bar_level_names) == 1:
                    items_temp = df_sub_temp.columns.get_level_values(grouped_bar_level_names[0]).unique().astype(str).tolist()
                else:
                    item_df = df_sub_temp.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
                    items_temp = [tuple(str(row)) for row in item_df.values]

                # Generate legend title
                if isinstance(df_sub_temp.columns, pd.MultiIndex):
                    legend_title = ' | '.join([str(n) for n in grouped_bar_level_names])
                else:
                    legend_title = str(df_sub_temp.columns.name) if df_sub_temp.columns.name else 'group'
            else:
                # Stack legend items
                if len(stack_level_names) == 1:
                    items_temp = df_sub_temp.columns.get_level_values(stack_level_names[0]).unique().astype(str).tolist()
                else:
                    stack_df = df_sub_temp.columns.to_frame()[stack_level_names].drop_duplicates()
                    items_temp = [tuple(str(row)) for row in stack_df.values]

                # Generate legend title
                if isinstance(df_sub_temp.columns, pd.MultiIndex):
                    legend_title = ' | '.join([str(n) for n in stack_level_names])
                else:
                    legend_title = str(df_sub_temp.columns.name) if df_sub_temp.columns.name else 'stack'

            # Format labels as they will appear in legend
            legend_labels = []
            for item in items_temp:
                if isinstance(item, (tuple, list)):
                    label = ' | '.join(str(v) for v in item)
                else:
                    label = str(item)
                legend_labels.append(label)

            # Estimate width for this subplot's legend
            width = estimate_legend_width(legend_labels, legend_title)
            max_legend_width = max(max_legend_width, width)

        legend_width = max_legend_width

    if isinstance(df.index, pd.MultiIndex):
        bar_labels = df.index.map(lambda x: ' | '.join(map(str, x))).to_list() 
    else:
        bar_labels = df.index.astype(str).tolist()
    # ~0.1 inches per character (font size 9)
    max_bar_label_len = max(len(label) for label in bar_labels) if bar_labels else 0
    # Estimate margins for each label type
    bar_label_width = max_bar_label_len * 0.09
    left_margin = bar_label_width

    # Calculate left margin width needed for group labels
    if expand_axis_levels:
        # Get all unique values at expand_axis_levels from the full df
        not_expand_axis_levels = list(set(range(len(df.columns.names))) - set(expand_axis_levels))
        expand_axis_names = df.columns.droplevel(not_expand_axis_levels)
        joined_expand_axis_names = expand_axis_names.map(' | '.join).tolist() if isinstance(expand_axis_names, pd.MultiIndex) else expand_axis_names.tolist()
        
        # Calculate group label width (~0.08 inches per character for font size 10)
        group_label_width = max(len(s) for s in joined_expand_axis_names) * 0.09
        left_margin = bar_label_width + group_label_width
    else:
        group_label_width = 0

    # Create figure and axes
    if n_subs == 1:
        # Single plot - bbox_inches='tight' handles legend and margins automatically
        fig = None
        axes = [None]
    else:
        # Use GridSpec for variable row/column dimensions based on orientation
        if bar_orientation == 'horizontal':
            # Horizontal bars: height varies by row, width controlled by base_bar_length
            total_height = sum(row_heights)
            total_width = (base_bar_length + left_margin) * n_cols

            # Add extra width for legends on non-rightmost columns
            if legend_width > 0 and n_cols > 1:
                # Rightmost column doesn't need extra space (bbox_inches='tight' handles it)
                total_width += legend_width * (n_cols - 1)

            # Add extra width for left margins when we have expand_axis_levels
            if expand_axis_levels and group_label_width > 0:
                # With expand_axis, ALL columns have secondary axis, so ALL need the group label width
                # The bar_label_width is handled by matplotlib automatically
                total_width += (left_margin + 0.2) * n_cols + legend_width * (n_cols - 1)

            fig = plt.figure(figsize=(total_width, total_height))

            # Calculate horizontal spacing
            wspace = left_margin / base_bar_length
            if legend_width > 0 and n_cols > 1:
                # Add spacing for legends between non-rightmost columns
                wspace += legend_width / base_bar_length

            # Calculate vertical spacing to prevent row overlap
            # Add space for ~1.5 rows of text (title + labels)
            # Assuming 0.15 inches per text row, 1.5 rows = 0.225 inches
            # hspace is relative to average subplot height
            avg_subplot_height = total_height / n_rows if n_rows > 0 else 1
            if xlabel:
                hspace = 1.6 / avg_subplot_height
            else:
                hspace = 1.4 / avg_subplot_height

            gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=row_heights, wspace=wspace, hspace=hspace)

        else:  # vertical
            # Vertical bars: width varies by column, height controlled by base_bar_length
            total_width = sum(col_widths)
            total_height = (base_bar_length + left_margin) * n_rows

            # For vertical orientation, legends still appear on right, so add width if needed
            # (Similar to horizontal case, but may need different handling for legends on bottom rows)
            if legend_width > 0 and n_cols > 1:
                total_width += legend_width * (n_cols - 1)

            fig = plt.figure(figsize=(total_width, total_height))

            # Calculate horizontal spacing
            wspace_val = 0.2  # Default matplotlib spacing
            if legend_width > 0 and n_cols > 1:
                avg_subplot_width = total_width / n_cols
                if ylabel:
                    wspace_val = (legend_width + 0.5) / avg_subplot_width
                else:
                    wspace_val = legend_width / avg_subplot_width

            # Calculate vertical spacing
            hspace = left_margin / base_bar_length

            gs = GridSpec(n_rows, n_cols, figure=fig, width_ratios=col_widths, wspace=wspace_val, hspace=hspace)
        axes = []
        for i in range(n_subs):
            row = i // n_cols
            col = i % n_cols
            axes.append(fig.add_subplot(gs[row, col]))

    for idx, sub in enumerate(subs):
        # Extract data for this subplot using xs
        if sub is None:
            # No sub_levels - use full dataframe
            df_sub = df
        elif len(sub_levels) == 1 and isinstance(df.columns, pd.MultiIndex):
            df_sub = df.xs(sub, level=sub_levels[0], axis=1)
        elif len(sub_levels) == 1:
            df_sub = df[sub]
        else:
            # For multiple sub_levels, apply xs for all levels at once
            df_sub = df.xs(sub, level=sub_levels, axis=1)
        if isinstance(df_sub, pd.Series):
            df_sub = df_sub.to_frame()

        # Get unique group combinations from df_sub
        if not expand_axis_levels:
            groups = [None]
        elif len(expand_axis_level_names) == 1:
            groups = df_sub.columns.get_level_values(expand_axis_level_names[0]).unique().tolist()
        else:
            groups = []
            for group_level_name in expand_axis_level_names:
                groups.append(df_sub.columns.get_level_values(group_level_name).tolist())
            groups = list(zip(*groups))

        # Reverse groups order
        if expand_axis_levels:
            groups = groups[::-1]
            # Format group labels for display (join tuples into strings)
            group_labels = [' | '.join(str(v) for v in g) if isinstance(g, tuple) else str(g) for g in groups]
        else:
            group_labels = []

        # Get bar labels from this subplot's index (not the global df)
        if isinstance(df_sub.index, pd.MultiIndex):
            subplot_bar_labels = df_sub.index.map(lambda x: ' | '.join(map(str, x))).to_list()  
        else:
            subplot_bar_labels = df_sub.index.astype(str).tolist()
        # Reverse to match the reversed groups order
        subplot_bar_labels = subplot_bar_labels[::-1]

        # Build list of all bars (for y-axis positioning)
        all_bars = []
        if not expand_axis_levels:
            # No groups - just one bar per period (use original index values, not labels)
            for idx_val in df_sub.index[::-1]:
                all_bars.append([None, idx_val])
        else:
            # Each group has one bar per period (use original values for data lookup)
            for group in groups:
                for idx_val in df_sub.index[::-1]:
                    all_bars.append([group, idx_val])

        # Create figure for single plot (now that we know bar count)
        if n_subs == 1 and fig is None:
            if bar_orientation == 'horizontal':
                figsize = (base_bar_length, 0.3 * len(all_bars) + 0.8)
            else:  # vertical
                figsize = (0.3 * len(all_bars) + 0.8, base_bar_length)
            fig, ax = plt.subplots(figsize=figsize)
            axes[0] = ax
        else:
            ax = axes[idx]

        # Determine plotting mode and execute appropriate logic
        if grouped_bar_levels:
            # ==================== NEW: GROUPED BARS MODE ====================
            # Get grouped bar combinations
            if len(grouped_bar_level_names) == 1:
                grouped_bars = df_sub.columns.get_level_values(grouped_bar_level_names[0]).unique().tolist()
            else:
                grouped_bar_df = df_sub.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
                grouped_bars = [tuple(row) for row in grouped_bar_df.values]

            # Colors for grouped bars
            n_grouped = len(grouped_bars)
            colors = plt.colormaps['tab10'].colors[:n_grouped]
            if n_grouped > 10:
                colors = plt.colormaps['tab20'].colors[:n_grouped]

            # Calculate bar width and offsets for side-by-side positioning
            total_bar_width = 0.8
            bar_width = total_bar_width / n_grouped
            bar_offsets = np.linspace(-total_bar_width/2 + bar_width/2,
                                      total_bar_width/2 - bar_width/2,
                                      n_grouped)

            # Track which grouped bars have been labeled
            labeled_grouped_bars = set()

            # Plot grouped bars
            for bar_idx, (group, period) in enumerate(all_bars):
                # Get data for this expand_axis group
                if group is None:
                    df_bar = df_sub
                elif len(expand_axis_level_names) == 1 and isinstance(df_sub.columns, pd.MultiIndex):
                    df_bar = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
                elif len(expand_axis_level_names) == 1:
                    df_bar = df_sub[group]
                else:
                    df_bar = df_sub.xs(group, level=expand_axis_level_names, axis=1)
                if isinstance(df_bar, pd.Series):
                    df_bar = df_bar.to_frame()

                # Plot each grouped bar at this position
                for grouped_idx, grouped_bar in enumerate(grouped_bars):
                    # Extract value for this grouped bar
                    if isinstance(df_bar, pd.Series):
                        value = df_bar.loc[period] if period in df_bar.index else 0
                    else:
                        if isinstance(df_bar.columns, pd.MultiIndex):
                            if len(grouped_bar_level_names) == 1:
                                try:
                                    df_grouped = df_bar.xs(grouped_bar, level=grouped_bar_level_names[0], axis=1)
                                except KeyError:
                                    value = 0
                                    df_grouped = None
                            else:
                                try:
                                    df_grouped = df_bar.xs(grouped_bar, level=grouped_bar_level_names, axis=1)
                                except KeyError:
                                    value = 0
                                    df_grouped = None
                        else:
                            if grouped_bar in df_bar.columns:
                                df_grouped = df_bar[grouped_bar]
                            else:
                                value = 0
                                df_grouped = None

                        if df_grouped is not None:
                            if isinstance(df_grouped, pd.DataFrame):
                                df_grouped = df_grouped.sum(axis=1)
                            value = df_grouped.loc[period] if period in df_grouped.index else 0

                    # Create label (only once per grouped bar)
                    if grouped_idx not in labeled_grouped_bars:
                        if isinstance(grouped_bar, (tuple, list)):
                            label = ' | '.join(str(v) for v in grouped_bar)
                        else:
                            label = str(grouped_bar)
                        labeled_grouped_bars.add(grouped_idx)
                    else:
                        label = ''

                    # Plot bar with offset
                    bar_position = bar_idx + bar_offsets[grouped_idx]
                    if bar_orientation == 'horizontal':
                        container = ax.barh(bar_position, value, height=bar_width,
                               label=label,
                               color=colors[grouped_idx % len(colors)])
                    else:  # vertical
                        container = ax.bar(bar_position, value, width=bar_width,
                              label=label,
                              color=colors[grouped_idx % len(colors)])
                    if value_fmt:
                        ax.bar_label(container, fmt=value_fmt, padding=3)

            # Add invisible bars for zero-value grouped bars (for legend completeness)
            for grouped_idx in range(len(grouped_bars)):
                if grouped_idx not in labeled_grouped_bars:
                    grouped_bar = grouped_bars[grouped_idx]
                    if isinstance(grouped_bar, (tuple, list)):
                        label = ' | '.join(str(v) for v in grouped_bar)
                    else:
                        label = str(grouped_bar)
                    if bar_orientation == 'horizontal':
                        ax.barh(0, 0, height=bar_width, left=0,
                               label=label,
                               color=colors[grouped_idx % len(colors)])
                    else:  # vertical
                        ax.bar(0, 0, width=bar_width, bottom=0,
                              label=label,
                              color=colors[grouped_idx % len(colors)])

        elif stack_levels:
            # ==================== EXISTING: STACKED BARS MODE ====================
            # Get stack combinations (for colors and legend)
            if len(stack_level_names) == 1:
                stacks = df_sub.columns.get_level_values(stack_level_names[0]).unique().tolist()
            else:
                stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
                stacks = [tuple(row) for row in stack_df.values]

            # Colors for stacking
            n_stack = len(stacks)
            colors = plt.colormaps['tab10'].colors[:n_stack]
            if n_stack > 10:
                colors = plt.colormaps['tab20'].colors[:n_stack]

            # Track which stacks have been labeled (for legend)
            labeled_stacks = set()

            # Plot bars
            for bar_idx, (group, period) in enumerate(all_bars):
                # Get data for this group
                if group is None:
                    # No groups - use full dataframe
                    df_bar = df_sub
                elif len(expand_axis_level_names) == 1 and isinstance(df_sub.columns, pd.MultiIndex):
                    df_bar = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
                elif len(expand_axis_level_names) == 1:
                    df_bar = df_sub[group]
                else:
                    # For multiple expand_axis_levels, apply xs for all levels at once
                    df_bar = df_sub.xs(group, level=expand_axis_level_names, axis=1)
                if isinstance(df_bar, pd.Series):
                    df_bar = df_bar.to_frame()

                # Collect all values for this bar
                values = []
                for stack_idx, stack in enumerate(stacks):
                    # Get value for this stack segment
                    if isinstance(df_bar, pd.Series):
                        value = df_bar.loc[period] if period in df_bar.index else 0
                    else:
                        if isinstance(df_bar.columns, pd.MultiIndex):
                            if len(stack_level_names) == 1:
                                try:
                                    df_stack = df_bar.xs(stack, level=stack_level_names[0], axis=1)
                                except KeyError:
                                    value = 0
                                    df_stack = None
                            else:
                                try:
                                    # For multiple stack_levels, apply xs for all levels at once
                                    df_stack = df_bar.xs(stack, level=stack_level_names, axis=1)
                                except KeyError:
                                    value = 0
                                    df_stack = None
                        else:
                            # Single column remaining
                            if stack in df_bar.columns:
                                df_stack = df_bar[stack]
                            else:
                                value = 0
                                df_stack = None

                        if df_stack is not None:
                            if isinstance(df_stack, pd.DataFrame):
                                df_stack = df_stack.sum(axis=1)
                            value = df_stack.loc[period] if period in df_stack.index else 0
                        else:
                            value = 0

                    values.append(value)

                # Stack positive values to the right from 0
                left_pos = 0
                for stack_idx, value in enumerate(values):
                    if value > 0:
                        # Add label only if not yet labeled
                        if stack_idx not in labeled_stacks:
                            stack_value = stacks[stack_idx]
                            if isinstance(stack_value, (tuple, list)):
                                label = ' | '.join(str(v) for v in stack_value)
                            else:
                                label = str(stack_value)
                            labeled_stacks.add(stack_idx)
                        else:
                            label = ''
                        if bar_orientation == 'horizontal':
                            ax.barh(bar_idx, value, left=left_pos,
                                   label=label,
                                   color=colors[stack_idx % len(colors)])
                        else:  # vertical
                            ax.bar(bar_idx, value, bottom=left_pos,
                                  label=label,
                                  color=colors[stack_idx % len(colors)])
                        left_pos += value

                # Stack zero values
                for stack_idx, value in enumerate(values):
                    if value == 0.0:
                        if stack_idx not in labeled_stacks:
                            stack_value = stacks[stack_idx]
                            if isinstance(stack_value, (tuple, list)):
                                label = ' | '.join(str(v) for v in stack_value)
                            else:
                                label = str(stack_value)
                            labeled_stacks.add(stack_idx)
                        else:
                            label = ''
                        if bar_orientation == 'horizontal':
                            ax.barh(0, 0, left=0,
                                label=label,
                                color=colors[stack_idx % len(colors)])
                        else:  # vertical
                            ax.bar(0, 0, bottom=0,
                                label=label,
                                color=colors[stack_idx % len(colors)])

                # Stack negative values to the left from 0
                left_neg = 0
                for stack_idx, value in enumerate(values):
                    if value < 0:
                        # Add label only if not yet labeled
                        if stack_idx not in labeled_stacks:
                            stack_value = stacks[stack_idx]
                            if isinstance(stack_value, (tuple, list)):
                                label = ' | '.join(str(v) for v in stack_value)
                            else:
                                label = str(stack_value)
                            labeled_stacks.add(stack_idx)
                        else:
                            label = ''
                        if bar_orientation == 'horizontal':
                            ax.barh(bar_idx, value, left=left_neg,
                                   label=label,
                                   color=colors[stack_idx % len(colors)])
                        else:  # vertical
                            ax.bar(bar_idx, value, bottom=left_neg,
                                  label=label,
                                  color=colors[stack_idx % len(colors)])
                        left_neg += value

        else:
            # ==================== SIMPLE BARS MODE (no stacking, no grouping) ====================
            for bar_idx, (group, period) in enumerate(all_bars):
                # Get data for this expand_axis group
                if group is None:
                    df_bar = df_sub
                elif len(expand_axis_level_names) == 1 and isinstance(df_sub.columns, pd.MultiIndex):
                    df_bar = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
                elif len(expand_axis_level_names) == 1:
                    df_bar = df_sub[group]
                else:
                    df_bar = df_sub.xs(group, level=expand_axis_level_names, axis=1)

                # Sum all values for this period
                if isinstance(df_bar, pd.Series):
                    value = df_bar.loc[period] if period in df_bar.index else 0
                else:
                    value = df_bar.loc[period].sum() if period in df_bar.index else 0

                # Plot single-color bar (no label, no legend)
                if bar_orientation == 'horizontal':
                    container = ax.barh(bar_idx, value, color='steelblue')
                else:  # vertical
                    container = ax.bar(bar_idx, value, color='steelblue')
                if value_fmt:
                    ax.bar_label(container, fmt=value_fmt, padding=3)

        # Set up axis with groups and bars
        # Build bar labels for display (matching all_bars structure)
        if not expand_axis_levels:
            display_bar_labels = subplot_bar_labels
        else:
            # For expand_axis: bar labels repeat for each group
            display_bar_labels = []
            for _ in groups:
                display_bar_labels.extend(subplot_bar_labels)

        # Set main axis for individual bars
        if bar_orientation == 'horizontal':
            ax.set_yticks(range(len(all_bars)), labels=display_bar_labels)
            ax.tick_params('y', length=0)
            ax.set_ylim(-0.5, len(all_bars) - 0.5)
            ax.tick_params(labelsize=9)
        else:  # vertical
            ax.set_xticks(range(len(all_bars)), labels=display_bar_labels)
            ax.tick_params('x', length=0)
            ax.set_xlim(-0.5, len(all_bars) - 0.5)
            ax.tick_params(labelsize=9)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='right')

        if expand_axis_levels:
            # Multiple groups - add two-level axis
            # Calculate group centers and separators
            group_centers = []
            group_lefts = []
            bar_idx = 0
            for group in groups:
                # Count bars in this group (one per period)
                n_bars_in_group = len(subplot_bar_labels)
                group_center = bar_idx + (n_bars_in_group - 1) / 2
                group_centers.append(group_center)
                group_lefts.append(bar_idx - 0.5)
                bar_idx += n_bars_in_group
            group_lefts.append(bar_idx - 0.5)

            # Calculate padding using pre-calculated margin values
            if bar_orientation == 'horizontal':
                # Use pre-calculated margins (convert from inches to points: multiply by 72)
                if bar_label_width > 0 and group_label_width > 0:
                    # Bar separators: bar label margin + small offset
                    bar_tick_length = bar_label_width * 72
                    # Group separators: both label margins + padding
                    group_tick_length = left_margin * 72
                    # Group label padding: bar label margin
                    group_label_pad = (bar_label_width + 0.1) * 72
                else:
                    # Fallback for single subplot or no expand_axis_levels
                    bar_tick_length = 5
                    group_tick_length = 30
                    group_label_pad = 10

                # Add separators between individual bars
                bar_sep_ax = ax.secondary_yaxis(location=0)
                bar_sep_ax.set_yticks([x - 0.5 for x in range(len(all_bars) + 1)], [''] * (len(all_bars) + 1))
                bar_sep_ax.tick_params('y', length=bar_tick_length)

                # Add secondary y-axis for groups
                group_ax = ax.secondary_yaxis(location=0)
                group_ax.set_yticks(group_centers, labels=group_labels)
                group_ax.tick_params('y', length=0, pad=group_label_pad)
                group_ax.tick_params(labelsize=10)

                # Separators for groups
                group_sep_ax = ax.secondary_yaxis(location=0)
                group_sep_ax.set_yticks(group_lefts, [''] * (len(group_labels) + 1))
                group_sep_ax.tick_params('y', length=group_tick_length)
            else:  # vertical
                # Use pre-calculated margins (convert from inches to points: multiply by 72)
                if bar_label_width > 0 and group_label_width > 0:
                    # Bar separators: bar label margin + small offset
                    bar_tick_length = bar_label_width * 72
                    # Group separators: both label margins + padding
                    group_tick_length = left_margin * 72
                    # Group label padding: bar label margin
                    group_label_pad = bar_label_width * 72
                else:
                    # Fallback for single subplot or no expand_axis_levels
                    bar_tick_length = 5
                    group_tick_length = 30
                    group_label_pad = 10

                # Add separators between individual bars
                bar_sep_ax = ax.secondary_xaxis(location=0)
                bar_sep_ax.set_xticks([x - 0.5 for x in range(len(all_bars) + 1)], [''] * (len(all_bars) + 1))
                bar_sep_ax.tick_params('x', length=bar_tick_length)

                # Add secondary x-axis for groups
                group_ax = ax.secondary_xaxis(location=0)
                group_ax.set_xticks(group_centers, labels=group_labels)
                group_ax.tick_params('x', length=0, pad=group_label_pad)
                group_ax.tick_params(labelsize=10)
                plt.setp(group_ax.get_xticklabels(), rotation=90, ha='right')

                # Separators for groups
                group_sep_ax = ax.secondary_xaxis(location=0)
                group_sep_ax.set_xticks(group_lefts, [''] * (len(group_labels) + 1))
                group_sep_ax.tick_params('x', length=group_tick_length)

        # Subplot title
        if sub is not None:
            if isinstance(sub, tuple):
                title = ' | '.join(str(v) for v in sub)
            else:
                title = str(sub)
            ax.set_title(f'{title}', pad=2)
        else:
            ax.set_title(key_name, pad=2)

        # Legend
        if stack_levels or grouped_bar_levels:
            handles, labels_leg = ax.get_legend_handles_labels()

            # Generate legend title based on mode
            if grouped_bar_levels:
                # Legend for grouped bars
                if isinstance(df_sub.columns, pd.MultiIndex):
                    legend_title = ' | '.join([str(n) for n in grouped_bar_level_names])
                else:
                    legend_title = str(df_sub.columns.name) if df_sub.columns.name else 'group'
            else:
                # Legend for stacked bars
                if isinstance(df_sub.columns, pd.MultiIndex):
                    legend_title = ' | '.join([str(n) for n in stack_level_names])
                else:
                    legend_title = str(df_sub.columns.name) if df_sub.columns.name else 'stack'

            # Determine if legend should be shown
            show_legend = False
            if legend_position == 'all':
                show_legend = True
            elif legend_position == 'right':
                if not sub_levels:
                    show_legend = True
                else:
                    col = idx % n_cols
                    if col == n_cols - 1 or idx == n_subs - 1:
                        show_legend = True

            if show_legend:
                ax.legend(handles[::-1], labels_leg[::-1], title=legend_title,
                        bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.1)

        # Value axis: pad for bar labels → apply explicit scale → formatter → labels
        row = idx // n_cols
        col = idx % n_cols
        if value_fmt:
            if bar_orientation == 'horizontal':
                xmin, xmax = ax.get_xlim()
                pad = (xmax - xmin) * 0.12
                ax.set_xlim(xmin - pad, xmax + pad)
            else:
                ymin, ymax = ax.get_ylim()
                pad = (ymax - ymin) * 0.12
                ax.set_ylim(ymin - pad, ymax + pad)
        if always_include_zero:
            if bar_orientation == 'horizontal':
                lo, hi = ax.get_xlim()
                ax.set_xlim(min(lo, 0), max(hi, 0))
            else:
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, 0), max(hi, 0))
        scale = _subplot_axis_scale(axis_scale_min_max, idx)
        if scale:
            if bar_orientation == 'horizontal':
                ax.set_xlim(scale[0], scale[1])
            else:
                ax.set_ylim(scale[0], scale[1])
        _fmt = _get_value_formatter(axis_tick_format, idx)
        if bar_orientation == 'horizontal':
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
            ax.xaxis.set_major_formatter(_fmt)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='upper'))
            ax.yaxis.set_major_formatter(_fmt)
        _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows)

        # Add a dotted zero line on the value axis
        if bar_orientation == 'horizontal':
            ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
        else:
            ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Use provided filepath or generate default
    if output_filepath:
        plt.savefig(output_filepath, bbox_inches='tight')
    else:
        plt.savefig(f'{plot_dir}/{key_name}_d.svg', bbox_inches='tight')
    plt.close(fig)