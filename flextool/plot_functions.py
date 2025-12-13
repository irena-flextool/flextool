import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging
import time
from contextlib import contextmanager

logging.getLogger('matplotlib.category').disabled = True

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
            PERF_STATS[name] = {'count': 0, 'total': 0, 'min': float('inf'), 'max': 0}
        PERF_STATS[name]['count'] += 1
        PERF_STATS[name]['total'] += elapsed
        PERF_STATS[name]['min'] = min(PERF_STATS[name]['min'], elapsed)
        PERF_STATS[name]['max'] = max(PERF_STATS[name]['max'], elapsed)
        if verbose:
            print(f"  [{name}] {elapsed:.4f}s")

def print_perf_summary():
    """Print summary of performance statistics"""
    if not PERF_STATS:
        return

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Operation':<40} {'Count':>8} {'Total':>10} {'Avg':>10} {'Min':>10} {'Max':>10}")
    print("-"*80)

    for name, stats in sorted(PERF_STATS.items(), key=lambda x: x[1]['total'], reverse=True):
        avg = stats['total'] / stats['count']
        print(f"{name:<40} {stats['count']:>8} {stats['total']:>10.4f}s {avg:>10.4f}s {stats['min']:>10.4f}s {stats['max']:>10.4f}s")

    print("="*80 + "\n")


def plot_dict_of_dataframes(results_dict, plot_dir, settings, plot_rows=(0,167), delete_existing_plots=True):
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
            if os.path.isfile(file_path):
                os.remove(file_path)

    for key, df in results_dict.items():
        #print(f"\nProcessing: {key}")
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
            y_locs = [i for i, char in enumerate(rules) if char == 'y']
            if chart_type == 'time':
                if 'nodeGroup' == key_name:
                    # nodeGroup_gdt_p
                    with time_block(f"{key} - plot"):
                        plot_dt_sub_lines(df, key_name, plot_dir, [0], [1], legend_position=legend_position)
                    # no others yet
                elif s_locs:
                    with time_block(f"{key} - plot"):
                        plot_dt_stack_sub(df, key_name, plot_dir, s_locs, u_locs, rows=plot_rows, legend_position=legend_position)
                elif l_locs:
                    with time_block(f"{key} - plot"):
                        plot_dt_sub_lines(df, key_name, plot_dir, u_locs, l_locs, rows=plot_rows, legend_position=legend_position)
            elif chart_type == 'bar':
                with time_block(f"{key} - plot"):
                    plot_rowbars_stack_groupbars(df, key_name, plot_dir, s_locs, g_locs, u_locs, y_locs, subplots_per_row=subplots_per_row, legend_position=legend_position)

            else:
                print(f'Could not interpret plot rule for {key}')

        plt.close('all')  # Clean up

    # Print summary after all plots
    print_perf_summary()


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


def apply_aggregation_and_adjust_levels(df, sum_levels=[], mean_levels=[],
                                       stack_levels=None, sub_levels=None,
                                       line_levels=None, expand_axis_levels=None,
                                       grouped_bar_levels=None):
    """
    Apply sum/mean aggregations and adjust level indices accordingly.

    Args:
        df: DataFrame to aggregate
        sum_levels: List of level indices to collapse by summing (negative for row index)
        mean_levels: List of level indices to collapse by averaging (negative for row index)
        stack_levels, sub_levels, line_levels, expand_axis_levels, grouped_bar_levels:
            Level index lists that need to be adjusted after aggregation

    Returns:
        tuple: (aggregated_df, adjusted_stack_levels, adjusted_sub_levels,
                adjusted_line_levels, adjusted_expand_axis_levels, adjusted_grouped_bar_levels)
    """
    if not sum_levels:
        sum_levels = []
    if not mean_levels:
        mean_levels = []

    # Validate no overlap between sum_levels and mean_levels
    overlap = set(sum_levels) & set(mean_levels)
    if overlap:
        raise ValueError(f"sum_levels and mean_levels cannot contain the same levels. Overlapping levels: {overlap}")

    # Separate row (negative) and column (non-negative) operations
    sum_levels_row = [lvl for lvl in sum_levels if lvl < 0]
    sum_levels_col = sorted([lvl for lvl in sum_levels if lvl >= 0], reverse=True)
    mean_levels_row = [lvl for lvl in mean_levels if lvl < 0]
    mean_levels_col = sorted([lvl for lvl in mean_levels if lvl >= 0], reverse=True)

    # Convert negative row indices to positive indices BEFORE any operations
    # This ensures they refer to the same actual levels even after stacking
    n_row_levels_original = df.index.nlevels if isinstance(df.index, pd.MultiIndex) else 1

    sum_levels_row_positive = []
    for lvl in sum_levels_row:
        pos_level = lvl % n_row_levels_original
        if pos_level >= n_row_levels_original:
            raise ValueError(f"Row level {lvl} (absolute: {pos_level}) out of range (max: {n_row_levels_original-1})")
        sum_levels_row_positive.append(pos_level)

    mean_levels_row_positive = []
    for lvl in mean_levels_row:
        pos_level = lvl % n_row_levels_original
        if pos_level >= n_row_levels_original:
            raise ValueError(f"Row level {lvl} (absolute: {pos_level}) out of range (max: {n_row_levels_original-1})")
        mean_levels_row_positive.append(pos_level)

    # Check if we need to rescue the row index before row aggregations
    # (move a level from expand_axis_levels to row index to keep at least one row level)
    if sum_levels_row_positive or mean_levels_row_positive:
        # Determine which row levels would be removed
        all_row_agg_positive = sorted(set(sum_levels_row_positive + mean_levels_row_positive), reverse=True)
        levels_to_remove = set(all_row_agg_positive)

        remaining_row_levels = n_row_levels_original - len(levels_to_remove)

        # If we would remove all row levels, rescue from expand_axis_levels
        if remaining_row_levels == 0:
            if not expand_axis_levels:
                raise ValueError(
                    "Cannot remove all row index levels: no expand_axis_levels available to rescue. "
                    "At least one row index level must remain after aggregation."
                )

            # Find first available level in expand_axis_levels
            # (not marked for column aggregation)
            rescued_level = None
            rescued_level_idx_in_expand = None
            for level_idx, level in enumerate(expand_axis_levels):
                if level not in sum_levels_col and level not in mean_levels_col:
                    rescued_level = level
                    rescued_level_idx_in_expand = level_idx
                    break

            if rescued_level is None:
                raise ValueError(
                    "Cannot remove all row index levels: all expand_axis_levels are also marked for "
                    "aggregation. At least one level must remain available to move to row index."
                )

            # Move this column level to row index using stack
            if not isinstance(df.columns, pd.MultiIndex):
                raise ValueError("Cannot rescue row index: columns are not a MultiIndex")

            df = df.stack(level=rescued_level)

            # Update expand_axis_levels - remove the rescued level (by position)
            expand_axis_levels = [lvl for i, lvl in enumerate(expand_axis_levels) if i != rescued_level_idx_in_expand]

            # Adjust column level indices for all levels
            # The rescued level is removed from columns, so levels above it shift down
            def adjust_for_rescue(lvl):
                if lvl > rescued_level:
                    return lvl - 1
                elif lvl == rescued_level:
                    raise ValueError(f"Column level {lvl} was moved to row index but is still referenced")
                return lvl

            def adjust_levels_for_rescue(levels):
                if levels is None or not levels:
                    return levels
                return [adjust_for_rescue(lvl) for lvl in levels]

            # Adjust all level lists
            expand_axis_levels = adjust_levels_for_rescue(expand_axis_levels)
            stack_levels = adjust_levels_for_rescue(stack_levels)
            sub_levels = adjust_levels_for_rescue(sub_levels)
            line_levels = adjust_levels_for_rescue(line_levels)
            grouped_bar_levels = adjust_levels_for_rescue(grouped_bar_levels)

            # Adjust column aggregation level indices
            sum_levels_col = [adjust_for_rescue(lvl) for lvl in sum_levels_col]
            mean_levels_col = [adjust_for_rescue(lvl) for lvl in mean_levels_col]

    # Apply row aggregations (axis=0)
    # Process in reverse order to avoid index shifting issues
    # Use positive indices that were converted earlier
    removed_row_levels = []
    if sum_levels_row_positive or mean_levels_row_positive:
        n_row_levels = df.index.nlevels if isinstance(df.index, pd.MultiIndex) else 1

        # Process in reverse order (highest index first)
        for pos_level in all_row_agg_positive:
            # pos_level is already a positive index
            if pos_level >= n_row_levels:
                raise ValueError(f"Row level {pos_level} out of range (max: {n_row_levels-1})")

            # Keep all levels except the one being aggregated
            keep_levels = [i for i in range(n_row_levels) if i != pos_level]
            if not keep_levels:
                raise ValueError(
                    "Internal error: Cannot remove all row index levels. "
                    "This should have been caught by the rescue logic."
                )

            try:
                if pos_level in sum_levels_row_positive:
                    df = df.groupby(level=keep_levels).sum()
                else:  # mean
                    df = df.groupby(level=keep_levels).mean()
            except Exception as e:
                raise ValueError(f"Error aggregating row level {pos_level}: {str(e)}")

            removed_row_levels.append(pos_level)
            n_row_levels -= 1

    # Apply column aggregations (axis=1) - non-negative indices
    # Process in reverse order to avoid index shifting issues
    removed_col_levels = []
    if sum_levels_col or mean_levels_col:
        if not isinstance(df.columns, pd.MultiIndex):
            if sum_levels_col or mean_levels_col:
                raise ValueError("Cannot aggregate column levels: columns are not a MultiIndex")
        else:
            n_col_levels = df.columns.nlevels

            all_col_agg = sorted(set(sum_levels_col + mean_levels_col), reverse=True)
            for agg_level in all_col_agg:
                if agg_level >= n_col_levels:
                    raise ValueError(f"Column level {agg_level} out of range (max: {n_col_levels-1})")

                # Keep all levels except the one being aggregated
                keep_levels = [i for i in range(n_col_levels) if i != agg_level]
                if not keep_levels:
                    raise ValueError("Cannot remove all column levels")

                try:
                    if agg_level in sum_levels_col:
                        df = df.T.groupby(level=keep_levels).sum().T
                    else:  # mean
                        df = df.T.groupby(level=keep_levels).mean().T
                except Exception as e:
                    raise ValueError(f"Error aggregating column level {agg_level}: {str(e)}")

                removed_col_levels.append(agg_level)
                n_col_levels -= 1

    # Adjust column level indices
    # For each level index in the various *_levels parameters,
    # we need to adjust it based on how many levels were removed before it
    removed_col_levels_sorted = sorted(removed_col_levels)

    def adjust_col_level(lvl):
        """Adjust a column level index after aggregations."""
        if lvl in removed_col_levels:
            raise ValueError(f"Column level {lvl} was removed by aggregation but is still referenced")
        # Count how many removed levels are below this level
        adjustment = sum(1 for r in removed_col_levels_sorted if r < lvl)
        return lvl - adjustment

    def adjust_col_levels(levels):
        """Adjust a list of column level indices."""
        if levels is None:
            return None
        return [adjust_col_level(lvl) for lvl in levels]

    # Adjust all the level parameters
    adjusted_stack_levels = adjust_col_levels(stack_levels) if stack_levels else stack_levels
    adjusted_sub_levels = adjust_col_levels(sub_levels) if sub_levels else sub_levels
    adjusted_line_levels = adjust_col_levels(line_levels) if line_levels else line_levels
    adjusted_expand_axis_levels = adjust_col_levels(expand_axis_levels) if expand_axis_levels else expand_axis_levels
    adjusted_grouped_bar_levels = adjust_col_levels(grouped_bar_levels) if grouped_bar_levels else grouped_bar_levels

    return (df, adjusted_stack_levels, adjusted_sub_levels, adjusted_line_levels,
            adjusted_expand_axis_levels, adjusted_grouped_bar_levels)


def plot_dt_sub_lines(df, plot_name, plot_dir, sub_levels, line_levels, rows=(0,167), subplots_per_row=3, legend_position='right', sum_levels=[], mean_levels=[], xlabel=None, ylabel=None, base_width_per_col=6):
    # Take plotted time
    df_plot = df.iloc[rows[0]:rows[1]]

    # Apply aggregations and adjust level indices
    (df_plot, _, sub_levels, line_levels, _, _) = apply_aggregation_and_adjust_levels(
        df_plot, sum_levels, mean_levels,
        stack_levels=None, sub_levels=sub_levels, line_levels=line_levels,
        expand_axis_levels=None, grouped_bar_levels=None
    )

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

    # Create figure and axes
    BASE_HEIGHT_PER_ROW = 4

    if legend_width > 0 and n_cols > 1:
        total_width = base_width_per_col * n_cols + legend_width * (n_cols - 1)
    else:
        total_width = base_width_per_col * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, BASE_HEIGHT_PER_ROW * n_rows))
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
        hspace = 0.225 / BASE_HEIGHT_PER_ROW if BASE_HEIGHT_PER_ROW > 0 else 0.15
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Get x-axis index (use last level if MultiIndex, otherwise use the index itself)
    if isinstance(df_plot.index, pd.MultiIndex):
        time_index = df_plot.index.get_level_values(-1)
    else:
        time_index = df_plot.index

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
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)

        # Apply axis labels conditionally based on subplot position
        row = idx // n_cols
        col = idx % n_cols
        if ylabel and col == 0:
            ax.set_ylabel(ylabel)
        if xlabel and row == n_rows - 1:
            ax.set_xlabel(xlabel)

        # Set xticks for every 24th time point
        tick_positions = range(0, len(time_index), 24)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_index[i] for i in tick_positions], rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(plot_name)

    plt.savefig(f'{plot_dir}/{plot_name}_dt.svg', bbox_inches='tight')
    plt.close(fig)

def plot_dt_stack_sub(df, plot_name, plot_dir, stack_levels, sub_levels, rows=(0,167), stack_element_to_split=None, subplots_per_row=3, legend_position='right', sum_levels=[], mean_levels=[], xlabel=None, ylabel=None, base_width_per_col=6):
    # Take plotted time
    df_plot = df.iloc[rows[0]:rows[1]]

    # Apply aggregations and adjust level indices
    (df_plot, stack_levels, sub_levels, _, _, _) = apply_aggregation_and_adjust_levels(
        df_plot, sum_levels, mean_levels,
        stack_levels=stack_levels, sub_levels=sub_levels, line_levels=None,
        expand_axis_levels=None, grouped_bar_levels=None
    )

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

    # Create figure and axes
    BASE_HEIGHT_PER_ROW = 4

    if legend_width > 0 and n_cols > 1:
        total_width = base_width_per_col * n_cols + legend_width * (n_cols - 1)
    else:
        total_width = base_width_per_col * n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(total_width, BASE_HEIGHT_PER_ROW * n_rows))
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
        hspace = 0.225 / BASE_HEIGHT_PER_ROW if BASE_HEIGHT_PER_ROW > 0 else 0.15
        fig.subplots_adjust(wspace=wspace, hspace=hspace)

    # Get x-axis index (use last level if MultiIndex, otherwise use the index itself)
    if isinstance(df_plot.index, pd.MultiIndex):
        time_index = df_plot.index.get_level_values(-1)
    else:
        time_index = df_plot.index

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
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        ax.grid(True, alpha=0.3)

        # Apply axis labels conditionally based on subplot position
        row = idx // n_cols
        col = idx % n_cols
        if ylabel and col == 0:
            ax.set_ylabel(ylabel)
        if xlabel and row == n_rows - 1:
            ax.set_xlabel(xlabel)

        # Set xticks for every 24th time point
        tick_positions = range(0, len(time_index), 24)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([time_index[i] for i in tick_positions], rotation=45, ha='right')

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(plot_name)

    plt.savefig(f'{plot_dir}/{plot_name}_dt.svg', bbox_inches='tight')
    plt.close(fig)

def plot_rowbars_stack_groupbars(df, key_name, plot_dir, stack_levels, expand_axis_levels, sub_levels=[], grouped_bar_levels=None, legend_position='right', subplots_per_row=3, sum_levels=[], mean_levels=[], xlabel=None, ylabel=None, base_width_per_col=6):
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
    base_width_per_col : float, optional
        Base width in inches for each subplot column (default: 6)
    """
    # Apply aggregations and adjust level indices
    (df, stack_levels, sub_levels, _, expand_axis_levels, grouped_bar_levels) = apply_aggregation_and_adjust_levels(
        df, sum_levels, mean_levels,
        stack_levels=stack_levels, sub_levels=sub_levels, line_levels=None,
        expand_axis_levels=expand_axis_levels, grouped_bar_levels=grouped_bar_levels
    )

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
        elif len(sub_levels) == 1:
            df_sub_temp = df.xs(sub, level=sub_levels[0], axis=1)
        else:
            df_sub_temp = df.xs(sub, level=sub_levels, axis=1)

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

    # Calculate height for each row (max bars in that row)
    row_heights = []
    for row in range(n_rows):
        start_idx = row * n_cols
        end_idx = min(start_idx + n_cols, n_subs)
        max_bars_in_row = max(bar_counts[start_idx:end_idx])
        row_height = 0.3 * max_bars_in_row + 0.8
        row_heights.append(row_height)

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

            # BRANCH: Get legend items based on mode
            if grouped_bar_levels:
                # Grouped bar legend items
                if len(grouped_bar_level_names) == 1:
                    items_temp = df_sub_temp.columns.get_level_values(grouped_bar_level_names[0]).unique().tolist()
                else:
                    item_df = df_sub_temp.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
                    items_temp = [tuple(row) for row in item_df.values]

                # Generate legend title
                if isinstance(df_sub_temp.columns, pd.MultiIndex):
                    legend_title = ' | '.join([str(n) for n in grouped_bar_level_names])
                else:
                    legend_title = str(df_sub_temp.columns.name) if df_sub_temp.columns.name else 'group'
            else:
                # Stack legend items
                if len(stack_level_names) == 1:
                    items_temp = df_sub_temp.columns.get_level_values(stack_level_names[0]).unique().tolist()
                else:
                    stack_df = df_sub_temp.columns.to_frame()[stack_level_names].drop_duplicates()
                    items_temp = [tuple(row) for row in stack_df.values]

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

    # Calculate left margin width needed for group labels
    left_margin_width = 0
    if expand_axis_levels and n_cols > 1:
        max_left_margin = 0

        for sub in subs:
            # Extract data for this subplot
            if sub is None:
                df_sub_temp = df
            elif len(sub_levels) == 1:
                df_sub_temp = df.xs(sub, level=sub_levels[0], axis=1)
            else:
                df_sub_temp = df.xs(sub, level=sub_levels, axis=1)

            # Get group combinations
            if len(expand_axis_level_names) == 1:
                groups_temp = df_sub_temp.columns.get_level_values(expand_axis_level_names[0]).unique().tolist()
            else:
                groups_temp = []
                for group_level_name in expand_axis_level_names:
                    groups_temp.append(df_sub_temp.columns.get_level_values(group_level_name).tolist())
                groups_temp = list(zip(*groups_temp))

            periods_temp = df_sub_temp.index.tolist()

            # Format group labels
            group_labels = []
            for group in groups_temp:
                if isinstance(group, (tuple, list)):
                    label = ' | '.join(str(v) for v in group)
                else:
                    label = str(group)
                group_labels.append(label)

            # Format period labels
            period_labels = [str(p) if not isinstance(p, tuple) else ' | '.join(str(v) for v in p) for p in periods_temp]

            # Estimate width needed for bar labels and group labels
            # Bar labels: ~0.055 inches per character (smaller font, size 9)
            # Group labels: ~0.065 inches per character (size 10)
            max_bar_label_len = max(len(label) for label in period_labels) if period_labels else 0
            max_group_label_len = max(len(label) for label in group_labels) if group_labels else 0

            # Estimate left margin: bar labels width + group labels width + padding
            bar_label_width = max_bar_label_len * 0.055
            group_label_width = max_group_label_len * 0.065
            left_margin = bar_label_width + group_label_width + 0.3  # 0.3 inches padding

            max_left_margin = max(max_left_margin, left_margin)

        left_margin_width = max_left_margin

    # Create figure and axes
    if n_subs == 1:
        # Single plot - bbox_inches='tight' handles legend and margins automatically
        fig = None
        axes = [None]
    else:
        # Use GridSpec for variable row heights
        total_height = sum(row_heights)

        # Calculate width with legend and left margin accommodation
        total_width = base_width_per_col * n_cols

        # Add extra width for legends on non-rightmost columns
        if legend_width > 0 and n_cols > 1:
            # Rightmost column doesn't need extra space (bbox_inches='tight' handles it)
            total_width += legend_width * (n_cols - 1)

        # Add extra width for left margins on all columns with groups
        if left_margin_width > 0 and n_cols > 1:
            # All columns need left margin space, but the leftmost is handled by tight_layout
            # Only add for additional columns beyond the first
            total_width += left_margin_width * (n_cols - 1)

        fig = plt.figure(figsize=(total_width, total_height))

        # Calculate horizontal spacing
        wspace = 0.2  # Default matplotlib spacing
        if legend_width > 0 and n_cols > 1:
            # Add spacing for legends between non-rightmost columns
            wspace = legend_width / base_width_per_col
        if left_margin_width > 0 and n_cols > 1:
            # Add spacing for left margins (accumulated with legend spacing)
            wspace += left_margin_width / base_width_per_col

        # Calculate vertical spacing to prevent row overlap
        # Add space for ~1.5 rows of text (title + labels)
        # Assuming 0.15 inches per text row, 1.5 rows = 0.225 inches
        # hspace is relative to average subplot height
        avg_subplot_height = total_height / n_rows if n_rows > 0 else 1
        hspace = 1.4 / avg_subplot_height if avg_subplot_height > 0 else 0.15

        gs = GridSpec(n_rows, n_cols, figure=fig, height_ratios=row_heights, wspace=wspace, hspace=hspace)
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

        # Get periods from row index and reverse
        periods = df_sub.index.tolist()[::-1]

        # Build list of all bars (for y-axis positioning)
        all_bars = []
        if not expand_axis_levels:
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
            colors = plt.colormaps['tab10'](np.linspace(0, 1, min(n_grouped, 10)))
            if n_grouped > 10:
                colors = plt.colormaps['tab20'](np.linspace(0, 1, n_grouped))

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
                    y_position = bar_idx + bar_offsets[grouped_idx]
                    ax.barh(y_position, value, height=bar_width,
                           label=label,
                           color=colors[grouped_idx % len(colors)])

            # Add invisible bars for zero-value grouped bars (for legend completeness)
            for grouped_idx in range(len(grouped_bars)):
                if grouped_idx not in labeled_grouped_bars:
                    grouped_bar = grouped_bars[grouped_idx]
                    if isinstance(grouped_bar, (tuple, list)):
                        label = ' | '.join(str(v) for v in grouped_bar)
                    else:
                        label = str(grouped_bar)
                    ax.barh(0, 0, height=bar_width, left=0,
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
                elif len(expand_axis_level_names) == 1 and isinstance(df_sub.columns, pd.MultiIndex):
                    df_bar = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
                elif len(expand_axis_level_names) == 1:
                    df_bar = df_sub[group]
                else:
                    # For multiple expand_axis_levels, apply xs for all levels at once
                    df_bar = df_sub.xs(group, level=expand_axis_level_names, axis=1)

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
                            if isinstance(stack_value, (tuple, list)):
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
                    if isinstance(stack_value, (tuple, list)):
                        label = ' | '.join(str(v) for v in stack_value)
                    else:
                        label = str(stack_value)
                    ax.barh(0, 0, left=0,
                           label=label,
                           color=colors[stack_idx % len(colors)])

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
                ax.barh(bar_idx, value, color='steelblue')

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

        if expand_axis_levels:
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
                        bbox_to_anchor=(1.01, 1), loc='upper left')

        # Apply axis labels conditionally based on subplot position
        row = idx // n_cols
        col = idx % n_cols
        if xlabel and row == n_rows - 1:
            ax.set_xlabel(xlabel)
        elif row == n_rows - 1:
            # Default label for bottom row if no custom label provided
            ax.set_xlabel('Value')
        if ylabel and col == 0:
            ax.set_ylabel(ylabel)

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    plt.savefig(f'{plot_dir}/{key_name}_d.svg', bbox_inches='tight')
    plt.close(fig)