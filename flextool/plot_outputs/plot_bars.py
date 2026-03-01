import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from flextool.plot_outputs.format_helpers import _get_value_formatter
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, _format_legend_labels, _should_show_legend,
)
from flextool.plot_outputs.axis_helpers import _subplot_axis_scale, _apply_subplot_label
from flextool.plot_outputs.subplot_helpers import (
    _calculate_grid_layout, _get_unique_levels, _extract_subplot_data,
)
from flextool.plot_outputs.plot_bars_detail import (
    _plot_grouped_bars, _plot_stacked_bars, _plot_simple_bars,
)


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
    subs = _get_unique_levels(df.columns, sub_levels)

    # Calculate subplot grid
    n_subs = len(subs)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # First pass: calculate number of bars for each subplot to determine heights
    bar_counts = []
    for sub in subs:
        # Extract data for this subplot using xs
        df_sub_temp = _extract_subplot_data(df, sub, sub_levels)

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
            df_sub_temp = _extract_subplot_data(df, sub, sub_levels)

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
            legend_labels = _format_legend_labels(items_temp)

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
        df_sub = _extract_subplot_data(df, sub, sub_levels)

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
            _plot_grouped_bars(ax, df_sub, all_bars, expand_axis_level_names,
                               grouped_bar_level_names, bar_orientation, value_fmt)
        elif stack_levels:
            _plot_stacked_bars(ax, df_sub, all_bars, expand_axis_level_names,
                               stack_level_names, bar_orientation)
        else:
            _plot_simple_bars(ax, df_sub, all_bars, expand_axis_level_names,
                              bar_orientation, value_fmt)

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

            if _should_show_legend(legend_position, sub_levels, idx, n_cols, n_subs):
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
