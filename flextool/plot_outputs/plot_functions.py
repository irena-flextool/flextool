import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
from flextool.plot_outputs.perf import PERF_STATS, time_block, print_perf_summary
from flextool.plot_outputs.format_helpers import (
    _get_value_formatter,
    generate_split_filename, split_into_chunks, _chunk_average_df,
)
from flextool.plot_outputs.config import PLOT_FIELD_NAMES, _is_single_config
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, _format_legend_labels, _should_show_legend,
)
from flextool.plot_outputs.axis_helpers import (
    _normalize_axis_scale, _subplot_axis_scale,
    _apply_subplot_label, set_smart_xticks,
)
from flextool.plot_outputs.subplot_helpers import (
    _calculate_grid_layout, _get_unique_levels, _extract_subplot_data,
)
from flextool.plot_outputs.plot_bars import plot_rowbars_stack_groupbars

logging.getLogger('matplotlib.category').disabled = True
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False



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
    subs = _get_unique_levels(df_plot.columns, sub_levels)

    # Calculate subplot grid
    n_subs = len(subs)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # Calculate legend width if needed
    legend_width = 0
    if legend_position == 'all' and n_cols > 1:
        max_legend_width = 0

        for sub in subs:
            # Extract data for this subplot
            df_sub_temp = _extract_subplot_data(df_plot, sub, sub_levels)

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
            legend_labels = _format_legend_labels(lines_temp)

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

        df_sub = _extract_subplot_data(df_plot, sub, sub_levels)

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

        if _should_show_legend(legend_position, sub_levels, idx, n_cols, n_subs):
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
    subs = _get_unique_levels(df_plot.columns, sub_levels)

    # Calculate subplot grid
    n_subs = len(subs)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # Calculate legend width if needed
    legend_width = 0
    if legend_position == 'all' and n_cols > 1:
        max_legend_width = 0

        for sub in subs:
            # Extract data for this subplot
            df_sub_temp = _extract_subplot_data(df_plot, sub, sub_levels)

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
            legend_labels = _format_legend_labels(stacks_temp)

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
        df_sub = _extract_subplot_data(df_plot, sub, sub_levels)

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

        if _should_show_legend(legend_position, sub_levels, idx, n_cols, n_subs):
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

