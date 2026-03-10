"""
Plot Outputs Orchestrator
=========================

Entry point for the plot_outputs module.

plot_dict_of_dataframes() — public entry point
  - Parses settings into PlotConfig
  - Applies dimension rules (sum, average, chunk-average, stack, unstack)
  - Splits data into per-file chunks via _plan_file_splits()
  - Dispatches to plot_lines or plot_bars
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging
from flextool.plot_outputs.format_helpers import (
    generate_split_filename, split_into_chunks, _chunk_average_df,
)
from flextool.plot_outputs.config import PlotConfig, PLOT_FIELD_NAMES, _is_single_config
from flextool.plot_outputs.axis_helpers import _normalize_axis_scale
from flextool.plot_outputs.plot_bars import plot_rowbars_stack_groupbars
from flextool.plot_outputs.plot_lines import plot_dt_sub_lines, plot_dt_stack_sub

logging.getLogger('matplotlib.category').disabled = True
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def _plan_file_splits(
    all_subs: list,
    all_items: list,
    subplot_levels: list[int],
    max_items_per_file: int,
    max_subplots_per_file: int,
) -> list:
    """Plan how to split data into files.

    With subplot_levels: returns a list of file_chunks, each a list of
    (sub, item_chunk) pairs (one pair per effective subplot in that file).

    Without subplot_levels: returns a list of item_chunks, one per file.
    """
    needs_item_split = len(all_items) > max_items_per_file
    item_chunks = (
        list(split_into_chunks(all_items, max_items_per_file))
        if needs_item_split else [all_items]
    )

    if subplot_levels:
        # Combine all (sub, item_chunk) pairs, then split into files
        effective = [(sub, ic) for sub in all_subs for ic in item_chunks]
        if len(effective) > max_subplots_per_file:
            return list(split_into_chunks(effective, max_subplots_per_file))
        return [effective]
    else:
        # Each item_chunk becomes its own file
        return item_chunks


def plot_dict_of_dataframes(results_dict, plot_dir, plot_settings,
        active_settings=['default'], plot_rows=(0, 167), delete_existing_plots=True,
        plot_file_format='png'):
    """
    Plot dataframes from a dictionary according to key suffixes.

    Args:
        results_dict: Dictionary of pandas DataFrames
        plot_dir: Directory to save PNG files
        plot_settings: Dict mapping result keys to plot configuration
        active_settings: List of named configs to activate (default ['default'])
        plot_rows: Row slice to apply for time-series plots
        delete_existing_plots: If True, delete all existing plots in plot_dir (default True)
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
        if key not in plot_settings:
            continue
        if df_orig.empty:
            continue

        # Collect active configs for this key
        chosen_settings = []
        entry = plot_settings[key]
        if _is_single_config(entry):
            if 'default' in active_settings:
                chosen_settings.append(entry)
        else:
            for setting_name, setting in entry.items():
                if setting_name in active_settings:
                    chosen_settings.append(setting)

        for setting in chosen_settings:
            # Parse raw dict into typed PlotConfig; unknown keys are silently ignored
            cfg = PlotConfig(**{k: v for k, v in setting.items() if k in PLOT_FIELD_NAMES})

            map_dims = cfg.map_dimensions_for_plots
            if not map_dims or len(map_dims) < 2:
                continue
            index_types, rules = map_dims[0], map_dims[1]
            if not rules:
                continue
            rules = rules.replace('_', '')

            if index_types is None:
                continue
            parts = index_types.split('_')
            if len(parts) == 2:
                df_index_levels, df_columns_levels = parts
            else:
                raise ValueError(
                    f'plot setting {key}: map_dimensions_for_plots first element should '
                    f'contain one underscore to separate row and column index indicators'
                )

            plot_name = cfg.plot_name or key
            axis_scale_min_max = _normalize_axis_scale(cfg.axis_scale_min_max)

            if 't' in rules and 'i' not in rules:
                chart_type = 'time'
                df = df_orig.iloc[plot_rows[0]:plot_rows[1]].copy()
            elif 'i' in rules:
                chart_type = 'time'
                df = df_orig.copy()
            else:
                chart_type = 'bar'
                df = df_orig.copy()

            # Normalise to always-MultiIndex at entry to simplify downstream handling
            if not isinstance(df.index, pd.MultiIndex):
                df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_arrays([df.columns], names=[df.columns.name])

            nr_row_levels = df.index.nlevels
            if len(rules) != nr_row_levels + df.columns.nlevels:
                raise ValueError(
                    f"Number of plot_type rules different from the number of index + "
                    f"column levels in the dataframe. {plot_name}"
                )

            levels_to_sort = [i for i, c in enumerate(df_columns_levels) if c in ('e', 'g')]
            if levels_to_sort:
                df = df.sort_index(axis=1, level=levels_to_sort, sort_remaining=False)

            # Sum row levels marked 'm'
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

            # Sum column levels marked 'm'
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

            nr_row_levels = df.index.nlevels
            nr_column_levels = df.columns.nlevels

            # Average row levels marked 'a'
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

            # Average column levels marked 'a'
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

            # Chunk-average 'i' row levels if time_average_duration is set
            i_positions = [pos for pos, char in enumerate(rules[:nr_row_levels]) if char == 'i']
            if i_positions and cfg.time_average_duration:
                i_pos = i_positions[0]
                chunk_size = int(cfg.time_average_duration)
                other_levels = [lv for lv in range(nr_row_levels) if lv != i_pos]
                if other_levels and nr_row_levels > 1:
                    parts = []
                    group_level = other_levels[0] if len(other_levels) == 1 else other_levels
                    for group_key, group_df in df.groupby(level=group_level):
                        flat = group_df.droplevel(other_levels)
                        averaged = _chunk_average_df(flat, chunk_size)
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
                    df = _chunk_average_df(df, chunk_size)

                nr_row_levels = df.index.nlevels
                nr_column_levels = df.columns.nlevels

            if (not df.empty) and (len(df) > 0):
                # Track level names before rearrangement to rebuild rules after stack/unstack
                level_names_before = list(df.index.names) + list(df.columns.names)
                name_to_rule = dict(zip(level_names_before, rules))

                # Move bar/line levels from columns to index (batch to avoid Cartesian product)
                nr_row = df.index.nlevels
                col_levels_to_stack = [i - nr_row for i, c in enumerate(rules)
                                       if c in ('b', 't', 'i') and i >= nr_row]
                if col_levels_to_stack:
                    df = df.stack(col_levels_to_stack, future_stack=True)
                    if isinstance(df, pd.Series):
                        df = df.to_frame()

                # Rebuild rules to match level order after stacking
                level_names_mid = list(df.index.names) + list(df.columns.names)
                if (len(set(level_names_before)) == len(level_names_before)
                        and all(n is not None for n in level_names_before)):
                    try:
                        rules = ''.join(name_to_rule[n] for n in level_names_mid)
                    except KeyError:
                        pass

                # Move column-type levels from row index to columns (batch to avoid Cartesian product)
                row_levels_to_unstack = [i for i, c in enumerate(rules[:df.index.nlevels])
                                         if c in ('u', 'g', 's', 'l', 'e')]
                if row_levels_to_unstack:
                    df = df.unstack(row_levels_to_unstack)
                    if isinstance(df, pd.Series):
                        df = df.to_frame()

                # Rebuild rules to match actual level order after stack/unstack
                level_names_after = list(df.index.names) + list(df.columns.names)
                if (len(set(level_names_before)) == len(level_names_before)
                        and all(n is not None for n in level_names_before)):
                    try:
                        rules = ''.join(name_to_rule[n] for n in level_names_after)
                    except KeyError:
                        pass  # names changed unexpectedly, keep original rules

                # Drop any remaining 'm'/'a' row levels
                sum_mean_row_levels = [i for i, char in enumerate(rules[:df.index.nlevels])
                                       if char in ('m', 'a')]
                if df.index.nlevels - len(sum_mean_row_levels) > 0:
                    for i in reversed(sum_mean_row_levels):
                        df = df.droplevel(i)
                        rules = rules[:i] + rules[i + 1:]

                # Drop any remaining 'm'/'a' column levels
                sum_mean_column_levels = [i for i, char in enumerate(rules[df.index.nlevels:])
                                          if char in ('m', 'a')]
                if df.columns.nlevels - len(sum_mean_column_levels) > 0:
                    for i in reversed(sum_mean_column_levels):
                        df = df.droplevel(i, axis=1)
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]

                # Level indices for each plot role
                grouped_bar_levels = [i for i, c in enumerate(rules[df.index.nlevels:]) if c == 'g']
                stack_levels = [i for i, c in enumerate(rules[df.index.nlevels:]) if c == 's']
                expand_axis_levels = [i for i, c in enumerate(rules[df.index.nlevels:]) if c == 'e']
                subplot_levels = [i for i, c in enumerate(rules[df.index.nlevels:]) if c == 'u']
                line_levels = [i for i, c in enumerate(rules[df.index.nlevels:]) if c == 'l']

                # Determine original subplots
                if not subplot_levels:
                    all_subs = [None]
                elif len(subplot_levels) == 1:
                    all_subs = df.columns.get_level_values(subplot_levels[0]).unique().tolist()
                else:
                    sub_df = df.columns.to_frame().iloc[:, subplot_levels].drop_duplicates()
                    all_subs = [tuple(row) for row in sub_df.values]

                # Determine items and per-file limits
                if chart_type == 'time':
                    default_max_items = 9
                    max_items = cfg.max_items_per_file if cfg.max_items_per_file is not None else default_max_items
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
                    default_max_items = 20
                    max_items = cfg.max_items_per_file if cfg.max_items_per_file is not None else default_max_items
                    all_items = df.index.tolist()

                needs_item_split = len(all_items) > max_items
                file_chunks = _plan_file_splits(
                    all_subs, all_items, subplot_levels, max_items, cfg.max_subplots_per_file
                )
                needs_file_split = len(file_chunks) > 1

                if subplot_levels:
                    # STRATEGY 1: file_chunks are lists of (sub, item_chunk) pairs
                    for file_idx, file_chunk in enumerate(file_chunks, start=1):
                        subs_in_file = []
                        items_in_file = set()
                        for sub, item_chunk in file_chunk:
                            if sub not in subs_in_file:
                                subs_in_file.append(sub)
                            items_in_file.update(item_chunk if item_chunk[0] is not None else [])

                        df_chunk = df.copy()

                        # Filter by subplots in this file
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

                        filepath = generate_split_filename(
                            plot_name, plot_dir, plot_file_format,
                            file_idx=file_idx, needs_split=needs_file_split
                        )

                        if chart_type == 'time':
                            if stack_levels:
                                plot_dt_stack_sub(
                                    df_chunk, plot_name, plot_dir, stack_levels, subplot_levels,
                                    rows=plot_rows, subplots_per_row=cfg.subplots_per_row,
                                    legend_position=cfg.legend, xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                    base_width_per_col=6, subplot_height=cfg.base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    axis_tick_format=cfg.axis_tick_format,
                                    always_include_zero=cfg.always_include_zero,
                                    output_filepath=filepath)
                            else:
                                plot_dt_sub_lines(
                                    df_chunk, plot_name, plot_dir, subplot_levels, line_levels,
                                    rows=plot_rows, subplots_per_row=cfg.subplots_per_row,
                                    legend_position=cfg.legend, xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                    base_width_per_col=6, subplot_height=cfg.base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    axis_tick_format=cfg.axis_tick_format,
                                    always_include_zero=cfg.always_include_zero,
                                    output_filepath=filepath)
                        elif chart_type == 'bar':
                            plot_rowbars_stack_groupbars(
                                df_chunk, plot_name, plot_dir,
                                stack_levels, expand_axis_levels, subplot_levels, grouped_bar_levels,
                                subplots_per_row=cfg.subplots_per_row, legend_position=cfg.legend,
                                xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                bar_orientation=cfg.bar_orientation, base_bar_length=cfg.base_length,
                                value_label=cfg.value_label, axis_scale_min_max=axis_scale_min_max,
                                axis_tick_format=cfg.axis_tick_format,
                                always_include_zero=cfg.always_include_zero,
                                output_filepath=filepath)
                        else:
                            raise ValueError(f'Could not interpret plot rule for {key}')

                else:
                    # STRATEGY 2: file_chunks are item lists, one per file
                    for file_idx, item_chunk in enumerate(file_chunks, start=1):
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

                        filepath = generate_split_filename(
                            plot_name, plot_dir, plot_file_format,
                            file_idx=file_idx, needs_split=needs_item_split
                        )

                        if chart_type == 'time':
                            if stack_levels:
                                plot_dt_stack_sub(
                                    df_chunk, plot_name, plot_dir, stack_levels, subplot_levels,
                                    rows=plot_rows, legend_position=cfg.legend,
                                    xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                    base_width_per_col=6, subplot_height=cfg.base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    always_include_zero=cfg.always_include_zero,
                                    output_filepath=filepath)
                            else:
                                plot_dt_sub_lines(
                                    df_chunk, plot_name, plot_dir, subplot_levels, line_levels,
                                    rows=plot_rows, legend_position=cfg.legend,
                                    xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                    base_width_per_col=6, subplot_height=cfg.base_length,
                                    axis_scale_min_max=axis_scale_min_max,
                                    always_include_zero=cfg.always_include_zero,
                                    output_filepath=filepath)
                        elif chart_type == 'bar':
                            plot_rowbars_stack_groupbars(
                                df_chunk, plot_name, plot_dir,
                                stack_levels, expand_axis_levels, subplot_levels, grouped_bar_levels,
                                subplots_per_row=cfg.subplots_per_row, legend_position=cfg.legend,
                                xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                bar_orientation=cfg.bar_orientation, base_bar_length=cfg.base_length,
                                value_label=cfg.value_label, axis_scale_min_max=axis_scale_min_max,
                                always_include_zero=cfg.always_include_zero,
                                output_filepath=filepath)
                        else:
                            raise ValueError(f'Could not interpret plot rule for {key}')

            plt.close('all')  # Clean up
