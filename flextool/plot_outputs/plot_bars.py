import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from flextool.plot_outputs.format_helpers import _get_value_formatter
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, estimate_legend_height,
    _format_legend_labels, _should_show_legend,
    build_shared_color_map,
)
from flextool.plot_outputs.axis_helpers import (
    _subplot_axis_bounds, _apply_subplot_label, _estimate_value_nbins,
)
from flextool.plot_outputs.subplot_helpers import (
    BarLayoutParams, _calculate_grid_layout, _get_unique_levels, _extract_subplot_data,
)
from flextool.plot_outputs.plot_bars_detail import (
    _plot_grouped_bars, _plot_stacked_bars, _plot_simple_bars,
)

# ── Layout constants (inches) ──
CHAR_WIDTH = 0.081          # Approximate width per character at font-size 9
LEFT_PAD = 0.25             # Left edge padding
RIGHT_PAD = 0.35            # Right margin for value-axis tick labels
BAR_HEIGHT = 0.24           # Height (or width) per bar including gap
SUBPLOT_VPAD = 0.3          # Space above axes for subplot title
INTER_COL_GAP = 0.4         # Horizontal gap between subplot columns
INTER_ROW_GAP = 0.6         # Vertical gap between subplot rows
VALUE_LABEL_MARGIN = 0.2    # Extra right margin when value labels are shown
YLABEL_WIDTH = 0.4          # Space reserved for y-axis label text
XLABEL_HEIGHT = 0.2         # Space reserved for x-axis label text
LEGEND_GAP = 0.15           # Gap between drawing area and legend box
TITLE_PAD = 0.3             # Top margin for subplot title above axes
BOTTOM_PAD = 0.35           # Bottom margin for x-axis ticks/labels below axes


def _compute_bar_layout(
    effective_plots: list[tuple[str | None, pd.DataFrame]],
    df: pd.DataFrame,
    expand_axis_levels: list[int],
    expand_axis_level_names: list[str],
    stack_levels: list[int],
    stack_level_names: list[str],
    grouped_bar_levels: list[int],
    grouped_bar_level_names: list[str],
    legend_position: str,
    subplots_per_row: int,
    base_bar_length: float,
) -> BarLayoutParams:
    """Compute layout parameters that must be consistent across file batches.

    Examines ALL effective_plots (not just the current batch) so that every
    file produced by plot_rowbars_stack_groupbars uses identical margins.
    """
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # ── bar-label width (max across ALL subplots) ──
    max_bar_label_chars = 0
    for _, df_sub in effective_plots:
        df_sub_clean = df_sub.dropna(how='all')
        if df_sub_clean.empty:
            continue
        if isinstance(df_sub_clean.index, pd.MultiIndex):
            labels = df_sub_clean.index.map(lambda x: ' | '.join(map(str, x)))
        else:
            labels = df_sub_clean.index.astype(str)
        longest = max(len(s) for s in labels) if len(labels) else 0
        max_bar_label_chars = max(max_bar_label_chars, longest)
    bar_label_width = max_bar_label_chars * CHAR_WIDTH

    # ── group-label width ──
    if expand_axis_levels:
        not_expand = list(set(range(len(df.columns.names))) - set(expand_axis_levels))
        expand_names = df.columns.droplevel(not_expand)
        if isinstance(expand_names, pd.MultiIndex):
            joined = expand_names.map(' | '.join).tolist()
        else:
            joined = expand_names.astype(str).tolist()
        group_label_width = max(len(s) for s in joined) * CHAR_WIDTH
    else:
        group_label_width = 0.0

    total_label_width = bar_label_width + group_label_width

    # ── legend width and height (max across ALL subplots) ──
    legend_width = 0.0
    max_legend_entries = 0
    legend_has_title = False
    if stack_levels or grouped_bar_levels:
        if legend_position == 'shared':
            # Union of all labels across all subplots
            all_items_union: list = []
            leg_title = ''
            for _, df_sub in effective_plots:
                if grouped_bar_levels:
                    if len(grouped_bar_level_names) == 1:
                        items = df_sub.columns.get_level_values(
                            grouped_bar_level_names[0]).unique().astype(str).tolist()
                    else:
                        item_df = df_sub.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
                        items = [tuple(str(v) for v in row) for row in item_df.values]
                    if isinstance(df_sub.columns, pd.MultiIndex):
                        leg_title = ' | '.join(str(n) for n in grouped_bar_level_names)
                    else:
                        leg_title = str(df_sub.columns.name) if df_sub.columns.name else 'group'
                else:
                    if len(stack_level_names) == 1:
                        if isinstance(df_sub.columns, pd.MultiIndex):
                            items = df_sub.columns.get_level_values(
                                stack_level_names[0]).unique().astype(str).tolist()
                        else:
                            items = [str(c) for c in df_sub.columns]
                    else:
                        stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
                        items = [tuple(str(v) for v in row) for row in stack_df.values]
                    if isinstance(df_sub.columns, pd.MultiIndex):
                        leg_title = ' | '.join(str(n) for n in stack_level_names)
                    else:
                        leg_title = str(df_sub.columns.name) if df_sub.columns.name else 'stack'
                for item in items:
                    if item not in all_items_union:
                        all_items_union.append(item)
            legend_labels = _format_legend_labels(all_items_union)
            legend_width = estimate_legend_width(legend_labels, leg_title)
            max_legend_entries = len(all_items_union)
            legend_has_title = True
        else:
            for _, df_sub in effective_plots:
                if grouped_bar_levels:
                    if len(grouped_bar_level_names) == 1:
                        items = df_sub.columns.get_level_values(
                            grouped_bar_level_names[0]).unique().astype(str).tolist()
                    else:
                        item_df = df_sub.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
                        items = [tuple(str(v) for v in row) for row in item_df.values]
                    if isinstance(df_sub.columns, pd.MultiIndex):
                        leg_title = ' | '.join(str(n) for n in grouped_bar_level_names)
                    else:
                        leg_title = str(df_sub.columns.name) if df_sub.columns.name else 'group'
                else:
                    if len(stack_level_names) == 1:
                        if isinstance(df_sub.columns, pd.MultiIndex):
                            all_stacks = df_sub.columns.get_level_values(
                                stack_level_names[0]).unique().tolist()
                            # Only count stacks with non-zero values (matching actual legend)
                            items = []
                            for s in all_stacks:
                                try:
                                    vals = df_sub.xs(s, level=stack_level_names[0], axis=1)
                                except (KeyError, TypeError):
                                    continue
                                if (vals.abs() > 1e-6).any().any():
                                    items.append(str(s))
                            if not items:
                                items = [str(s) for s in all_stacks]
                        else:
                            # Single-level columns after subplot extraction
                            items = [str(c) for c in df_sub.columns if (df_sub[c].abs() > 1e-6).any()]
                            if not items:
                                items = [str(c) for c in df_sub.columns]
                    else:
                        stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
                        items = [tuple(str(v) for v in row) for row in stack_df.values]
                    if isinstance(df_sub.columns, pd.MultiIndex):
                        leg_title = ' | '.join(str(n) for n in stack_level_names)
                    else:
                        leg_title = str(df_sub.columns.name) if df_sub.columns.name else 'stack'
                legend_labels = _format_legend_labels(items)
                w = estimate_legend_width(legend_labels, leg_title)
                legend_width = max(legend_width, w)
                max_legend_entries = max(max_legend_entries, len(items))
                legend_has_title = True  # all legend paths set a title

    legend_height = estimate_legend_height(max_legend_entries, legend_has_title) if max_legend_entries > 0 else 0.0

    # ── value-axis tick label width (for vertical bars) ──
    # Estimate from the extreme data values across all subplots
    max_val_chars = 1
    for _, df_sub in effective_plots:
        numeric = df_sub.select_dtypes(include='number')
        if numeric.empty:
            continue
        extremes = [numeric.min().min(), numeric.max().max()]
        for val in extremes:
            if pd.notna(val):
                formatted = f'{val:,.6g}'
                max_val_chars = max(max_val_chars, len(formatted))
    value_axis_width = max(0.5, max_val_chars * CHAR_WIDTH + 0.15)

    return BarLayoutParams(
        bar_label_width=bar_label_width,
        group_label_width=group_label_width,
        total_label_width=total_label_width,
        legend_width=legend_width,
        legend_height=legend_height,
        base_bar_length=base_bar_length,
        value_axis_width=value_axis_width,
    )


def _render_bar_figure(
    effective_plots, df, key_name, plot_dir,
    stack_levels, stack_level_names,
    expand_axis_levels, expand_axis_level_names,
    sub_levels,
    grouped_bar_levels, grouped_bar_level_names,
    legend_position, subplots_per_row,
    xlabel, ylabel, bar_orientation, base_bar_length,
    value_fmt, axis_bounds, axis_tick_format,
    always_include_zero_in_axis, output_filepath,
    layout: BarLayoutParams | None = None,
    shared_color_map: dict[str, tuple] | None = None,
):
    """Render one file's worth of bar subplots to a single figure/file."""
    # Calculate subplot grid
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # Add extra bottom space when xlabel is present
    bottom_pad = BOTTOM_PAD + (XLABEL_HEIGHT if xlabel else 0)
    # Left edge padding only for expand-axis plots (secondary y-axis ticks touch edge)
    left_edge_pad = LEFT_PAD + (YLABEL_WIDTH if ylabel else 0)

    # First pass: calculate number of bars for each subplot to determine heights
    bar_counts = []
    for title, df_sub_temp in effective_plots:
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
        if not expand_axis_levels:
            n_bars = len(periods_temp)
        else:
            n_bars = len(groups_temp) * len(periods_temp)
        bar_counts.append(n_bars)

    # Per-subplot sizes (inches) based on bar counts
    subplot_sizes: list[float] = [
        BAR_HEIGHT * bc + SUBPLOT_VPAD for bc in bar_counts
    ]

    # Create figure and axes using layout-derived margins
    if n_subs == 1:
        # Single plot — deferred until bar count is known (inside render loop)
        fig = None
        axes = [None]
        # Precompute consistent width for single-subplot figures
        _single_width = layout.base_bar_length + layout.total_label_width + left_edge_pad + RIGHT_PAD
        if layout.legend_width > 0:
            _single_width += layout.legend_width + LEGEND_GAP
    else:
        if bar_orientation == 'horizontal':
            # --- Horizontal bars: height varies per subplot, width uniform ---
            # Cell width excludes RIGHT_PAD; it is added once at the figure edge
            cell_width = layout.base_bar_length + layout.total_label_width
            # Per-cell legend space only for 'all' with multiple columns
            if layout.legend_width > 0 and legend_position == 'all' and n_cols > 1:
                cell_width += layout.legend_width + LEGEND_GAP

            # Organize subplots into rows for top-alignment
            row_data: list[list[tuple[int, float]]] = [[] for _ in range(n_rows)]
            for i in range(n_subs):
                r = i // n_cols
                row_data[r].append((i, subplot_sizes[i]))

            # Row height = tallest subplot in that row
            row_heights: list[float] = [
                max(h for _, h in row) if row else 0.0 for row in row_data
            ]

            content_height = sum(row_heights) + INTER_ROW_GAP * max(0, n_rows - 1)
            total_height = content_height + TITLE_PAD + bottom_pad
            # Add legend height excess
            if layout and layout.legend_height > 0:
                min_axes_h = min(subplot_sizes)  # smallest subplot
                legend_excess = max(0, layout.legend_height - min_axes_h + SUBPLOT_VPAD)
                total_height += legend_excess
            total_width = cell_width * n_cols + INTER_COL_GAP * max(0, n_cols - 1) + left_edge_pad + RIGHT_PAD
            # For 'right' legend (or single column), add legend space once
            if layout.legend_width > 0 and not (legend_position == 'all' and n_cols > 1):
                total_width += layout.legend_width + LEGEND_GAP

            fig = plt.figure(figsize=(total_width, total_height))

            axes = [None] * n_subs
            # Start below TITLE_PAD, end above BOTTOM_PAD
            y_cursor = total_height - TITLE_PAD
            for r, row in enumerate(row_data):
                row_top = y_cursor
                for sub_idx, sub_h in row:
                    c = sub_idx % n_cols
                    x_left = (c * (cell_width + INTER_COL_GAP) + layout.total_label_width + left_edge_pad) / total_width
                    ax_width = layout.base_bar_length / total_width
                    # Axes height = bars only; SUBPLOT_VPAD stays above for title/spacing
                    bars_h = sub_h - SUBPLOT_VPAD
                    y_bottom = (row_top - sub_h) / total_height
                    ax_height = bars_h / total_height
                    axes[sub_idx] = fig.add_axes([x_left, y_bottom, ax_width, ax_height])
                y_cursor -= row_heights[r] + INTER_ROW_GAP

        else:  # vertical
            # --- Vertical bars: width varies per subplot, height uniform ---
            # For vertical: x-axis = bar labels (rotated), y-axis = values
            # Each subplot cell: value_axis_labels + bars + right_pad
            vert_subplot_widths = [
                layout.value_axis_width + BAR_HEIGHT * bc + RIGHT_PAD
                for bc in bar_counts
            ]
            cell_height = layout.base_bar_length + layout.total_label_width + SUBPLOT_VPAD

            row_stacks: list[list[tuple[int, float]]] = [[] for _ in range(n_rows)]
            for i in range(n_subs):
                r = i // n_cols
                row_stacks[r].append((i, vert_subplot_widths[i]))

            row_total_widths: list[float] = []
            for stack in row_stacks:
                if stack:
                    w = sum(s for _, s in stack) + INTER_COL_GAP * (len(stack) - 1)
                    if layout.legend_width > 0 and legend_position == 'all' and len(stack) > 1:
                        w += (layout.legend_width + LEGEND_GAP) * len(stack)
                    elif layout.legend_width > 0:
                        w += layout.legend_width + LEGEND_GAP
                else:
                    w = 0.0
                row_total_widths.append(w)

            total_width = max(row_total_widths) if row_total_widths else 1.0
            total_width += left_edge_pad
            content_height = cell_height * n_rows + INTER_ROW_GAP * max(0, n_rows - 1)
            total_height = content_height + TITLE_PAD + bottom_pad
            # Add legend height excess
            if layout and layout.legend_height > 0:
                axes_h = layout.base_bar_length
                legend_excess = max(0, layout.legend_height - axes_h)
                total_height += legend_excess

            fig = plt.figure(figsize=(total_width, total_height))

            axes = [None] * n_subs
            for r, stack in enumerate(row_stacks):
                y_top = total_height - TITLE_PAD - r * (cell_height + INTER_ROW_GAP)
                y_bottom = (y_top - cell_height + layout.total_label_width) / total_height
                ax_height = layout.base_bar_length / total_height

                x_cursor = left_edge_pad
                for sub_idx, sub_w in stack:
                    bars_w = sub_w - layout.value_axis_width - RIGHT_PAD
                    x_left = (x_cursor + layout.value_axis_width) / total_width
                    ax_width = bars_w / total_width
                    axes[sub_idx] = fig.add_axes([x_left, y_bottom, ax_width, ax_height])
                    x_cursor += sub_w + INTER_COL_GAP
                    if layout.legend_width > 0 and legend_position == 'all' and len(stack) > 1:
                        x_cursor += layout.legend_width + LEGEND_GAP

    for idx, (eff_title, df_sub) in enumerate(effective_plots):
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
            subplot_h = BAR_HEIGHT * len(all_bars) + SUBPLOT_VPAD
            bars_only_h = BAR_HEIGHT * len(all_bars)
            if bar_orientation == 'horizontal':
                fig_w = _single_width
                fig_h = subplot_h + TITLE_PAD + bottom_pad
                legend_excess = 0.0
                if layout and layout.legend_height > 0:
                    legend_excess = max(0.0, layout.legend_height - bars_only_h)
                    fig_h += legend_excess
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_axes([
                    (layout.total_label_width + left_edge_pad) / fig_w,
                    (bottom_pad + legend_excess) / fig_h,
                    layout.base_bar_length / fig_w,
                    bars_only_h / fig_h,
                ])
            else:  # vertical
                fig_w = layout.value_axis_width + bars_only_h + RIGHT_PAD
                if layout.legend_width > 0:
                    fig_w += layout.legend_width + LEGEND_GAP
                fig_h = layout.base_bar_length + layout.total_label_width + SUBPLOT_VPAD + TITLE_PAD + bottom_pad
                legend_excess = 0.0
                if layout and layout.legend_height > 0:
                    legend_excess = max(0.0, layout.legend_height - layout.base_bar_length)
                    fig_h += legend_excess
                fig = plt.figure(figsize=(fig_w, fig_h))
                ax = fig.add_axes([
                    layout.value_axis_width / fig_w,
                    (bottom_pad + layout.total_label_width + legend_excess) / fig_h,
                    bars_only_h / fig_w,
                    layout.base_bar_length / fig_h,
                ])
            axes[0] = ax
        else:
            ax = axes[idx]

        # Determine plotting mode and execute appropriate logic
        if grouped_bar_levels:
            _plot_grouped_bars(ax, df_sub, all_bars, expand_axis_level_names,
                               grouped_bar_level_names, bar_orientation, value_fmt,
                               shared_color_map=shared_color_map)
        elif stack_levels:
            _plot_stacked_bars(ax, df_sub, all_bars, expand_axis_level_names,
                               stack_level_names, bar_orientation,
                               shared_color_map=shared_color_map)
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
            ax.tick_params(labelsize=10)
        else:  # vertical
            ax.set_xticks(range(len(all_bars)), labels=display_bar_labels)
            ax.tick_params('x', length=0)
            ax.set_xlim(-0.5, len(all_bars) - 0.5)
            ax.tick_params(labelsize=10)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

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
                if layout.bar_label_width > 0 and layout.group_label_width > 0:
                    # Bar separators: bar label margin + small offset
                    bar_tick_length = layout.bar_label_width * 72
                    # Group separators: both label margins + padding
                    group_tick_length = layout.total_label_width * 72
                    # Group label padding: bar label margin
                    group_label_pad = layout.bar_label_width * 72 + 10
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
                if layout.bar_label_width > 0 and layout.group_label_width > 0:
                    # Bar separators: bar label margin + small offset
                    bar_tick_length = layout.bar_label_width * 72
                    # Group separators: both label margins + padding
                    group_tick_length = layout.total_label_width * 72
                    # Group label padding: bar label margin
                    group_label_pad = layout.bar_label_width * 72 + 10
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
                plt.setp(group_ax.get_xticklabels(), rotation=90, ha='center')

                # Separators for groups
                group_sep_ax = ax.secondary_xaxis(location=0)
                group_sep_ax.set_xticks(group_lefts, [''] * (len(group_labels) + 1))
                group_sep_ax.tick_params('x', length=group_tick_length)

        # Subplot title (only for actual subplot dimensions, not the figure title)
        if eff_title is not None:
            ax.set_title(eff_title, pad=2)

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
                # Axes width depends on orientation: horizontal uses base_bar_length,
                # vertical uses bar area width (BAR_HEIGHT * n_bars)
                if bar_orientation == 'horizontal':
                    axes_width = layout.base_bar_length
                else:
                    axes_width = BAR_HEIGHT * len(all_bars)
                legend_x = 1 + LEGEND_GAP / axes_width
                # For stacked bars, _plot_stacked_bars already builds legend entries
                # in the desired order; for grouped bars, reverse to match visual order
                if grouped_bar_levels:
                    handles, labels_leg = handles[::-1], labels_leg[::-1]
                ax.legend(handles, labels_leg, title=legend_title,
                        bbox_to_anchor=(legend_x, 1), loc='upper left', borderaxespad=0)

        row = idx // n_cols
        col = idx % n_cols
        if always_include_zero_in_axis:
            if bar_orientation == 'horizontal':
                lo, hi = ax.get_xlim()
                ax.set_xlim(min(lo, 0), max(hi, 0))
            else:
                lo, hi = ax.get_ylim()
                ax.set_ylim(min(lo, 0), max(hi, 0))
        scale = _subplot_axis_bounds(axis_bounds, idx)
        if scale:
            if bar_orientation == 'horizontal':
                ax.set_xlim(scale[0], scale[1])
            else:
                ax.set_ylim(scale[0], scale[1])
        _fmt = _get_value_formatter(axis_tick_format, idx)
        # Prune the tick at the figure edge to prevent label overflow, but
        # never prune the 0 tick (it anchors the zero-line visually).
        if bar_orientation == 'horizontal':
            lo, hi = ax.get_xlim()
            prune = 'upper' if hi != 0 else 'lower'
            nbins = _estimate_value_nbins(lo, hi, layout.base_bar_length, _fmt, is_horizontal_axis=True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=prune))
            ax.xaxis.set_major_formatter(_fmt)
        else:
            lo, hi = ax.get_ylim()
            prune = 'upper' if hi != 0 else 'lower'
            nbins = _estimate_value_nbins(lo, hi, layout.base_bar_length, _fmt, is_horizontal_axis=False)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=prune))
            ax.yaxis.set_major_formatter(_fmt)
        # Extend axis limits AFTER tick locator to create room for value labels
        # within the plot area (13% on each side that has data).
        if value_fmt:
            if bar_orientation == 'horizontal':
                xmin, xmax = ax.get_xlim()
                if xmax > 0:
                    xmax *= 1.13
                if xmin < 0:
                    xmin *= 1.13
                ax.set_xlim(xmin, xmax)
            else:
                ymin, ymax = ax.get_ylim()
                if ymax > 0:
                    ymax *= 1.13
                if ymin < 0:
                    ymin *= 1.13
                ax.set_ylim(ymin, ymax)
        # When expand-axis is active, push ylabel further left to avoid overlapping
        # the secondary y-axis group labels. The pad is based on group_label_width in points.
        if expand_axis_levels and bar_orientation == 'horizontal':
            expand_pad = (layout.group_label_width + 0.15) * 72
        else:
            expand_pad = 0
        _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows,
                             expand_label_pad=expand_pad)

        # Add a dotted zero line on the value axis
        if bar_orientation == 'horizontal':
            ax.axvline(0, color='black', linewidth=0.5, linestyle=':')
        else:
            ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

    # ── Shared legend (one per file, anchored to top-right subplot) ──
    if legend_position == 'shared' and shared_color_map and (stack_levels or grouped_bar_levels):
        from matplotlib.patches import Patch
        legend_ax_idx = min(n_cols - 1, n_subs - 1)
        ax_legend = axes[legend_ax_idx]
        handles = [Patch(facecolor=c) for c in shared_color_map.values()]
        labels_all = list(shared_color_map.keys())

        # Generate legend title
        if grouped_bar_levels:
            if isinstance(df.columns, pd.MultiIndex):
                legend_title = ' | '.join(str(n) for n in grouped_bar_level_names)
            else:
                legend_title = str(df.columns.name) if df.columns.name else 'group'
        else:
            if isinstance(df.columns, pd.MultiIndex):
                legend_title = ' | '.join(str(n) for n in stack_level_names)
            else:
                legend_title = str(df.columns.name) if df.columns.name else 'stack'

        if bar_orientation == 'horizontal':
            axes_width = layout.base_bar_length
        else:
            axes_width = BAR_HEIGHT * max(
                (len(ep[1]) for ep in effective_plots), default=1
            )
        legend_x = 1 + LEGEND_GAP / axes_width
        ax_legend.legend(handles, labels_all, title=legend_title,
                         bbox_to_anchor=(legend_x, 1), loc='upper left', borderaxespad=0)

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Figure title at a fixed distance from the top (10px ≈ 0.14in at 72 dpi)
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(key_name, y=1 - 0.14 / fig_h, va='top')

    # Use provided filepath or generate default
    # All figures use fixed layout — no bbox_inches='tight' to keep consistent size.
    if output_filepath:
        plt.savefig(output_filepath)
    else:
        plt.savefig(f'{plot_dir}/{key_name}_d.png')
    plt.close(fig)


def plot_rowbars_stack_groupbars(df, key_name, plot_dir, stack_levels, expand_axis_levels,
        sub_levels=[], grouped_bar_levels=None,
        legend_position='right', subplots_per_row=2,
        xlabel=None, ylabel=None, bar_orientation='horizontal', base_bar_length=4,
        value_label=False, axis_bounds=None, axis_tick_format='1,.0f',
        always_include_zero_in_axis=True, max_items_per_plot=10,
        max_subplots_per_file=6, output_filepath=None):
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
    max_items_per_plot : int, optional
        Maximum bar items per subplot before splitting (default: None = no limit)
    max_subplots_per_file : int, optional
        Maximum effective subplots per file (default: None = no limit)
    """
    # Validate mutual exclusivity
    if stack_levels and grouped_bar_levels:
        raise ValueError(
            "Cannot use both 'stack_levels' and 'grouped_bar_levels' simultaneously. "
            "stack_levels creates horizontal stacks within each bar, while "
            "grouped_bar_levels creates separate bars side-by-side. "
            "Choose one approach or use neither for simple bars."
        )

    # Resolve value_label: True or 'true' → use dynamic formatting,
    # string → use as Python format spec, falsy → no labels.
    if value_label is True or value_label == 'true':
        value_fmt = 'dynamic'
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

    # Compute expand-group count so max_items_per_plot accounts for total bars
    # (each row item produces n_expand_groups visual bars)
    if expand_axis_levels and isinstance(df.columns, pd.MultiIndex):
        expand_level_name = expand_axis_level_names[0]
        n_expand_groups = len(
            df.columns.get_level_values(expand_level_name).unique()
        )
    else:
        expand_level_name = None
        n_expand_groups = 1

    # Pre-extract subplot data, drop empty rows, and split by max_items_per_plot
    effective_plots: list[tuple[str | None, pd.DataFrame]] = []
    for sub in subs:
        df_sub = _extract_subplot_data(df, sub, sub_levels)
        # Drop rows that are all-NaN (artifacts from unstack) and fill remaining NaN
        df_sub = df_sub.dropna(how='all')
        if df_sub.empty:
            continue
        df_sub = df_sub.fillna(0)
        title = (
            ' | '.join(str(v) for v in sub) if isinstance(sub, tuple)
            else str(sub) if sub is not None else None
        )
        # Compute total visual bars for this subplot
        n_rows = len(df_sub)
        total_bars = n_rows * n_expand_groups
        if max_items_per_plot and total_bars > max_items_per_plot:
            # Try splitting by row items first
            max_row_items = max(1, max_items_per_plot // n_expand_groups)
            if n_rows > max_row_items:
                # Split by rows
                for i in range(0, n_rows, max_row_items):
                    chunk = df_sub.iloc[i:i + max_row_items]
                    chunk_label = f"{title}_{i // max_row_items + 1}" if title else None
                    effective_plots.append((chunk_label, chunk))
            elif expand_level_name is not None:
                # Can't reduce rows further — split by expand-axis groups
                max_groups = max(1, max_items_per_plot // n_rows)
                all_groups = df_sub.columns.get_level_values(
                    expand_level_name
                ).unique().tolist()
                for gi, grp_start in enumerate(range(0, len(all_groups), max_groups)):
                    grp_chunk = all_groups[grp_start:grp_start + max_groups]
                    mask = df_sub.columns.get_level_values(expand_level_name).isin(grp_chunk)
                    chunk = df_sub.loc[:, mask]
                    chunk_label = f"{title}_{gi + 1}" if title else None
                    effective_plots.append((chunk_label, chunk))
            else:
                effective_plots.append((title, df_sub))
        else:
            effective_plots.append((title, df_sub))

    if not effective_plots:
        return

    # Build shared color map before splitting into file batches
    shared_color_map = None
    if legend_position == 'shared' and (stack_levels or grouped_bar_levels):
        all_labels: list[str] = []
        for _, df_sub in effective_plots:
            if grouped_bar_levels:
                if len(grouped_bar_level_names) == 1:
                    items = df_sub.columns.get_level_values(
                        grouped_bar_level_names[0]).unique().tolist()
                else:
                    item_df = df_sub.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
                    items = [tuple(row) for row in item_df.values]
            else:
                if len(stack_level_names) == 1:
                    if isinstance(df_sub.columns, pd.MultiIndex):
                        items = df_sub.columns.get_level_values(
                            stack_level_names[0]).unique().tolist()
                    else:
                        items = df_sub.columns.unique().tolist()
                else:
                    stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
                    items = [tuple(row) for row in stack_df.values]
            for item in items:
                label = _format_legend_labels([item])[0]
                if label not in all_labels:
                    all_labels.append(label)
        all_labels.sort()
        shared_color_map = build_shared_color_map(all_labels)

    # Compute layout once across ALL effective_plots for consistent margins
    layout = _compute_bar_layout(
        effective_plots, df,
        expand_axis_levels, expand_axis_level_names,
        stack_levels, stack_level_names,
        grouped_bar_levels, grouped_bar_level_names,
        legend_position, subplots_per_row,
        base_bar_length,
    )

    # Split effective_plots into file batches respecting max_subplots_per_file
    _max = max_subplots_per_file if max_subplots_per_file else len(effective_plots)
    if len(effective_plots) <= _max:
        _file_batches = [(effective_plots, output_filepath)]
    else:
        _file_batches = []
        for i in range(0, len(effective_plots), _max):
            batch = effective_plots[i:i + _max]
            file_idx = i // _max + 1  # 1-based
            if output_filepath:
                base, ext = os.path.splitext(output_filepath)
                fp = f'{base}_{file_idx:02d}{ext}'
            else:
                fp = f'{plot_dir}/{key_name}_d_{file_idx:02d}.png'
            _file_batches.append((batch, fp))

    for batch, batch_filepath in _file_batches:
        _render_bar_figure(
            batch, df, key_name, plot_dir,
            stack_levels, stack_level_names,
            expand_axis_levels, expand_axis_level_names,
            sub_levels,
            grouped_bar_levels, grouped_bar_level_names,
            legend_position, subplots_per_row,
            xlabel, ylabel, bar_orientation, base_bar_length,
            value_fmt, axis_bounds, axis_tick_format,
            always_include_zero_in_axis, batch_filepath,
            layout=layout,
            shared_color_map=shared_color_map,
        )
