import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_dict_of_dataframes(results_dict, plot_dir, settings, plot_rows=(0,167)):
    """
    Plot dataframes from a dictionary according to key suffixes.

    Args:
        results_dict: Dictionary of pandas DataFrames
        plot_dir: Directory to save PNG files
        result_set_map: Dictionary mapping result keys to (filename, plot_flag) tuples
    """
    # Empty csv dir
    for filename in os.listdir(plot_dir):
        file_path = os.path.join(plot_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for key, df in results_dict.items():
        # print(f"Processing {key}...")
        # Do not create a plot if result_set_map has 'False'
        if key not in settings:
            continue
        rules = settings[key][2]
        if not rules:
            continue

        # Process the key (name, row index levels and column index levels)
        split_key = key.split('_')
        key_name = settings[key][1]
        if settings[key][3]:
            subplots_per_row = settings[key][3]
        else:
            subplots_per_row = 3
        if settings[key][4]:
            legend_position = settings[key][4]
        else:
            legend_position = 'right'

        # Decide how to plot
        if (not df.empty) & (len(df) > 0):
            if 'b' in rules[0]:
                chart_type = 'bar'
                rules = rules[1:]
            elif 'x' in rules[0]:
                chart_type = 'time'
                rules = rules[1:]
            else:
                print(f'Plot chart type not defined for {key}')
            if 'g' in split_key[-2]:
                df = df.unstack('group')
            if 'e' in split_key[-2]:
                df = df.unstack(0)
            g_locs = [i for i, char in enumerate(rules) if char == 'g']
            l_locs = [i for i, char in enumerate(rules) if char == 'l']
            s_locs = [i for i, char in enumerate(rules) if char == 's']
            u_locs = [i for i, char in enumerate(rules) if char == 'u']
            if chart_type == 'time':
                if 'nodeGroup' == key_name:
                    # nodeGroup_gdt_p
                    plot_dt_sub_lines(df, key_name, plot_dir, [0], [1])
                    # no others yet
                elif s_locs:
                    plot_dt_stack_sub(df, key_name, plot_dir, s_locs, u_locs, rows=plot_rows)
                elif l_locs:
                    plot_dt_sub_lines(df, key_name, plot_dir, u_locs, l_locs, rows=plot_rows)
            elif chart_type == 'bar':
                plot_rowbars_stack_groupbars(df, key_name, plot_dir, s_locs, g_locs, u_locs, subplots_per_row=subplots_per_row, legend_position=legend_position)

            else:
                print(f'Could not interpret plot rule for {key}')

        plt.close('all')  # Clean up


def plot_dt_sub_lines(df, plot_name, plot_dir, sub_levels, line_levels, rows=(0,167), subplots_per_row=3):
    # Take plotted time
    df_plot = df.iloc[rows[0]:rows[1]]

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

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_subs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Get time index (drop period level)
    time_index = df_plot.index.get_level_values('time')

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
            ax.set_title(str(sub))

        # Only add legend to rightmost column (or always if single plot)
        if not sub_levels:
            # Single plot - always show legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Multiple subplots - only rightmost column
            col = idx % n_cols
            if col == n_cols - 1 or idx == n_subs - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)

        # Set xticks for every 24th time point
        tick_positions = range(0, len(time_index), 24)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_index[i] for i in tick_positions], rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(plot_name)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{plot_name}_dt.svg', bbox_inches='tight')
    plt.close(fig)

def plot_dt_stack_sub(df, plot_name, plot_dir, stack_levels, sub_levels, rows=(0,167), stack_element_to_split=None, subplots_per_row=3):
    # Take plotted time
    df_plot = df.iloc[rows[0]:rows[1]]

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

    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_subs == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Get time index (drop period level)
    time_index = df_plot.index.get_level_values('time')

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
        df_to_plot.plot.area(stacked=True, ax=ax, alpha=0.7, legend=False, linewidth=0)

        # Subplot formatting
        if sub is not None:
            ax.set_title(str(sub))

        # Only add legend to rightmost column (or always if single plot)
        if not sub_levels:
            # Single plot - always show legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        else:
            # Multiple subplots - only rightmost column
            col = idx % n_cols
            if col == n_cols - 1 or idx == n_subs - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)

        # Set xticks for every 24th time point
        tick_positions = range(0, len(time_index), 24)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_index[i] for i in tick_positions], rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(plot_name)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{plot_name}_dt.svg', bbox_inches='tight')
    plt.close(fig)

def plot_rowbars_stack_groupbars(df, key_name, plot_dir, stack_levels, group_levels, sub_levels=[], legend_position='right', subplots_per_row=3):
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
    stack_levels : list of int
        Column level indices that create colored segments within each bar
    group_levels : list of int
        Column level indices that create groups of bars
    sub_levels : list of int, optional
        Column level indices that create separate subplots (default: [])
    legend : str, optional
        Legend placement: 'all' shows legend on all subplots, 'right' shows only on rightmost column (default: 'right')
    subplots_per_row : int, optional
        Number of subplots per row (default: 3)
    """
    if key_name == 'unit_outputNode':
        pass
    # Convert level indices to names for stability
    if isinstance(df.columns, pd.MultiIndex):
        stack_level_names = [df.columns.names[i] for i in stack_levels]
        group_level_names = [df.columns.names[i] for i in group_levels] if group_levels else []
    else:
        # Single level index - use indices directly for data access
        stack_level_names = stack_levels
        group_level_names = [df.columns.name] if group_levels else []

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
        elif len(sub_levels) == 1:
            df_sub_temp = df.xs(sub, level=sub_levels[0], axis=1)
        else:
            df_sub_temp = df.xs(sub, level=sub_levels, axis=1)

        # Get unique group combinations from df_sub_temp
        if not group_levels:
            groups_temp = [None]
        elif len(group_level_names) == 1:
            groups_temp = df_sub_temp.columns.get_level_values(group_level_names[0]).unique().tolist()
        else:
            groups_temp = []
            for group_level_name in group_level_names:
                groups_temp.append(df_sub_temp.columns.get_level_values(group_level_name).tolist())
            groups_temp = list(zip(*groups_temp))

        periods_temp = df_sub_temp.index.tolist()

        # Calculate number of bars
        if not group_levels:
            n_bars = len(periods_temp)
        else:
            n_bars = len(groups_temp) * len(periods_temp)

        bar_counts.append(n_bars)

    # Calculate height for each row (max bars in that row)
    row_heights = []
    for row in range(n_rows):
        start_idx = row * n_cols
        end_idx = min(start_idx + n_cols, n_subs)
        max_bars_in_row = max(bar_counts[start_idx:end_idx])
        row_height = 0.3 * max_bars_in_row + 0.8
        row_heights.append(row_height)

    # Create figure and axes
    if n_subs == 1:
        # Will be created inside the loop with dynamic sizing
        fig = None
        axes = [None]
    else:
        # Use GridSpec for variable row heights
        total_height = sum(row_heights)
        fig = plt.figure(figsize=(6*n_cols, total_height))
        gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=row_heights)
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
        elif len(sub_levels) == 1:
            df_sub = df.xs(sub, level=sub_levels[0], axis=1)
        else:
            # For multiple sub_levels, apply xs for all levels at once
            df_sub = df.xs(sub, level=sub_levels, axis=1)

        # Get unique group combinations from df_sub
        if not group_levels:
            groups = [None]
        elif len(group_level_names) == 1:
            groups = df_sub.columns.get_level_values(group_level_names[0]).unique().tolist()
        else:
            groups = []
            for group_level_name in group_level_names:
                groups.append(df_sub.columns.get_level_values(group_level_name).tolist())
            groups = list(zip(*groups))

        # Reverse groups order
        if group_levels:
            groups = groups[::-1]

        # Get periods from row index and reverse
        periods = df_sub.index.tolist()[::-1]

        # Build list of all bars (for y-axis positioning)
        all_bars = []
        if not group_levels:
            # No groups - just one bar per period
            for period in periods:
                all_bars.append([None, period])
        else:
            # Each group has one bar per period
            for group in groups:
                for period in periods:
                    all_bars.append([group, period])

        # Create figure for single plot (now that we know bar count)
        if n_subs == 1 and fig is None:
            fig, ax = plt.subplots(figsize=(6, 0.3 * len(all_bars) + 0.8))
            axes[0] = ax
        else:
            ax = axes[idx]

        # Get stack combinations (for colors and legend)
        if not stack_levels or len(stack_level_names) == 0:
            # No stacking - treat all data as one stack
            stacks = [None]
        elif len(stack_level_names) == 1:
            stacks = df_sub.columns.get_level_values(stack_level_names[0]).unique().tolist()
        else:
            stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
            stacks = [tuple(row) for row in stack_df.values]

        # Colors for stacking
        n_stack = len(stacks)
        colors = plt.colormaps['tab10'](np.linspace(0, 1, min(n_stack, 10)))
        if n_stack > 10:
            colors = plt.colormaps['tab20'](np.linspace(0, 1, n_stack))

        # Track which stacks have been labeled (for legend)
        labeled_stacks = set()

        # Plot bars
        for bar_idx, (group, period) in enumerate(all_bars):
            # Get data for this group
            if group is None:
                # No groups - use full dataframe
                df_bar = df_sub
            elif len(group_level_names) == 1 and isinstance(df_sub.columns, pd.MultiIndex):
                df_bar = df_sub.xs(group, level=group_level_names[0], axis=1)
            elif len(group_level_names) == 1:
                df_bar = df_sub[group]
            else:
                # For multiple group_levels, apply xs for all levels at once
                df_bar = df_sub.xs(group, level=group_level_names, axis=1)

            # Collect all values for this bar
            values = []
            for stack_idx, stack in enumerate(stacks):
                # Get value for this stack segment
                if stack is None:
                    # No stacking - sum all columns for this period
                    if isinstance(df_bar, pd.Series):
                        value = df_bar.loc[period] if period in df_bar.index else 0
                    else:
                        value = df_bar.loc[period].sum() if period in df_bar.index else 0
                elif isinstance(df_bar, pd.Series):
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
                        if stack_value is None:
                            label = ''
                        elif isinstance(stack_value, (tuple, list)):
                            label = ' | '.join(str(v) for v in stack_value)
                        else:
                            label = str(stack_value)
                    else:
                        label = ''
                    if stack_idx not in labeled_stacks:
                        labeled_stacks.add(stack_idx)
                    ax.barh(bar_idx, value, left=left_pos,
                           label=label,
                           color=colors[stack_idx % len(colors)])
                    left_pos += value

            # Stack negative values to the left from 0
            left_neg = 0
            for stack_idx, value in enumerate(values):
                if value < 0:
                    # Add label only if not yet labeled
                    if stack_idx not in labeled_stacks:
                        stack_value = stacks[stack_idx]
                        if stack_value is None:
                            label = ''
                        elif isinstance(stack_value, (tuple, list)):
                            label = ' | '.join(str(v) for v in stack_value)
                        else:
                            label = str(stack_value)
                    else:
                        label = ''
                    if stack_idx not in labeled_stacks:
                        labeled_stacks.add(stack_idx)
                    ax.barh(bar_idx, value, left=left_neg,
                           label=label,
                           color=colors[stack_idx % len(colors)])
                    left_neg += value

        # Add invisible bars for any stacks that were never labeled (all zero values)
        for stack_idx in range(len(stacks)):
            if stack_idx not in labeled_stacks:
                stack_value = stacks[stack_idx]
                if stack_value is None:
                    label = ''
                elif isinstance(stack_value, (tuple, list)):
                    label = ' | '.join(str(v) for v in stack_value)
                else:
                    label = str(stack_value)
                ax.barh(0, 0, left=0,
                       label=label,
                       color=colors[stack_idx % len(colors)])

        # Set up y-axis with groups and bars
        # Extract bar labels
        if len(all_bars[0]) > 1:
            if isinstance(all_bars[0][1], tuple):
                for i, bar in enumerate(all_bars):
                    all_bars[i][1] = ' | '.join(bar[1])
        bar_labels = [str(bar) for _, bar in all_bars]

        # Set main y-axis for individual bars
        ax.set_yticks(range(len(all_bars)), labels=bar_labels)
        ax.tick_params('y', length=0)
        ax.set_ylim(-0.5, len(all_bars) - 0.5)
        ax.tick_params(labelsize=9)

        if group_levels:
            # Multiple groups - add two-level y-axis
            # Calculate group centers
            if isinstance(groups[0], tuple):
                for i, group in enumerate(groups):
                    groups[i] = ' | '.join(group)
            group_centers = []
            group_lefts = []
            bar_idx = 0
            for group in groups:
                # Count bars in this group (one per period)
                n_bars_in_group = len(periods)
                group_center = bar_idx + (n_bars_in_group - 1) / 2
                group_centers.append(group_center)
                group_lefts.append(bar_idx - 0.5)
                bar_idx += n_bars_in_group
            group_lefts.append(bar_idx - 0.5)

            # Calculate padding
            max_label_length_bars = max(len(str(label)) for label in bar_labels)
            max_label_length_groups = max(len(str(label)) for label in groups)
            pad_value_bars = max_label_length_bars * 6.5
            renderer = fig.canvas.get_renderer()
            left_margin = ax.yaxis.get_tightbbox(renderer).x1 - ax.yaxis.get_tightbbox(renderer).x0
            pad_value_groups = left_margin + max_label_length_groups * 5.0

            # Extra padding for group labels to position them further left
            #extra_group_pad = max_label_length_groups * 6

            # Add separators between individual bars
            bar_sep_ax = ax.secondary_yaxis(location=0)
            bar_sep_ax.set_yticks([x - 0.5 for x in range(len(all_bars) + 1)], [''] * (len(all_bars) + 1))
            bar_sep_ax.tick_params('y', length=left_margin - 5)

            # Add secondary y-axis for groups
            group_ax = ax.secondary_yaxis(location=0)
            group_ax.set_yticks(group_centers, labels=groups)
            group_ax.tick_params('y', length=0, pad=left_margin)
            group_ax.tick_params(labelsize=10)

            # Separators for groups
            group_sep_ax = ax.secondary_yaxis(location=0)
            group_sep_ax.set_yticks(group_lefts, [''] * (len(groups) + 1))
            group_sep_ax.tick_params('y', length=pad_value_groups)

        # Subplot title
        if sub is not None:
            if isinstance(sub, tuple):
                title = ' | '.join(str(v) for v in sub)
            else:
                title = str(sub)
            ax.set_title(f'{key_name} - {title}')
        else:
            ax.set_title(key_name)

        # Legend
        if stack_levels:
            handles, labels_leg = ax.get_legend_handles_labels()

            # Generate legend title - use actual names for single-level Index
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
                        bbox_to_anchor=(1.01, 1), loc='upper left')

        # Labels
        ax.set_xlabel('Value')

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{plot_dir}/{key_name}_d.svg', bbox_inches='tight')
    plt.close(fig)