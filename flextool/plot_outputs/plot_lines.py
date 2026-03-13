import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from flextool.plot_outputs.format_helpers import _get_value_formatter
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, _format_legend_labels, _should_show_legend,
)
from flextool.plot_outputs.axis_helpers import (
    _subplot_axis_bounds, _apply_subplot_label, set_smart_xticks,
    _estimate_value_nbins,
)
from flextool.plot_outputs.subplot_helpers import (
    LineLayoutParams, _calculate_grid_layout, _get_unique_levels, _extract_subplot_data,
)


# ── Layout constants (inches) ──
CHAR_WIDTH = 0.081           # Approximate width per character at font-size 9 (labels)
TICK_CHAR_WIDTH = 0.065      # Approximate width per digit character at tick font size
LEFT_PAD = 0.1              # Left edge padding
RIGHT_PAD = 0.2              # Right edge padding
SUBPLOT_VPAD = 0.25          # Space above axes for subplot title
INTER_COL_GAP = 0.2          # Horizontal gap between subplot columns
INTER_ROW_GAP = 0.25          # Vertical gap between rows (room for x-axis tick labels of row above)
YLABEL_WIDTH = 0.4           # Space reserved for y-axis label text
XLABEL_HEIGHT = 0.25         # Space reserved for x-axis label text
LEGEND_GAP = 0.15            # Gap between drawing area and legend box
TITLE_PAD = 0.3              # Top margin for figure title
BOTTOM_PAD = 0.35            # Bottom margin (room for x-axis tick labels)
MIN_VALUE_LABEL_WIDTH = 0.35 # Minimum space for y-axis value tick labels


def _get_column_items(df_sub, level_names):
    """Get unique items from column levels."""
    is_multi = isinstance(df_sub.columns, pd.MultiIndex)
    if is_multi:
        if len(level_names) == 1:
            return df_sub.columns.get_level_values(level_names[0]).unique().tolist()
        else:
            df_lvl = df_sub.columns.to_frame()[level_names].drop_duplicates()
            return [tuple(row) for row in df_lvl.values]
    else:
        return df_sub.columns.unique().tolist()


def _filter_columns_by_items(df_sub, items, level_names):
    """Filter DataFrame columns to only include specified items."""
    if len(level_names) == 1:
        mask = df_sub.columns.get_level_values(level_names[0]).isin(items)
    else:
        col_frame = df_sub.columns.to_frame()[level_names]
        mask = col_frame.apply(tuple, axis=1).isin(items).values
    return df_sub.loc[:, mask]


def _build_effective_plots(df_plot, sub_levels, item_level_names, max_items_per_plot):
    """Pre-expand subplots into effective_plots with item splitting."""
    subs = _get_unique_levels(df_plot.columns, sub_levels)
    effective_plots = []
    for sub in subs:
        df_sub = _extract_subplot_data(df_plot, sub, sub_levels)
        title = (
            ' | '.join(str(v) for v in sub) if isinstance(sub, tuple)
            else str(sub) if sub is not None else None
        )
        items = _get_column_items(df_sub, item_level_names)
        if max_items_per_plot and len(items) > max_items_per_plot:
            for i in range(0, len(items), max_items_per_plot):
                chunk_items = items[i:i + max_items_per_plot]
                df_chunk = _filter_columns_by_items(df_sub, chunk_items, item_level_names)
                chunk_idx = i // max_items_per_plot + 1
                effective_plots.append((f"{title}_{chunk_idx}", df_chunk))
        else:
            effective_plots.append((title, df_sub))
    return effective_plots


def _make_file_batches(effective_plots, max_subplots_per_file, output_filepath, plot_dir, plot_name):
    """Split effective_plots into file-sized batches with filepaths."""
    _max = max_subplots_per_file if max_subplots_per_file else len(effective_plots)
    if len(effective_plots) <= _max:
        return [(effective_plots, output_filepath)]
    batches = []
    for i in range(0, len(effective_plots), _max):
        batch = effective_plots[i:i + _max]
        file_idx = i // _max + 1
        if output_filepath:
            base, ext = os.path.splitext(output_filepath)
            fp = f'{base}_{file_idx:02d}{ext}'
        else:
            fp = f'{plot_dir}/{plot_name}_{file_idx:02d}.png'
        batches.append((batch, fp))
    return batches


# ---------------------------------------------------------------------------
#  Layout computation
# ---------------------------------------------------------------------------

def _estimate_value_label_width(
    effective_plots: list[tuple[str | None, pd.DataFrame]],
    axis_tick_format,
) -> float:
    """Estimate width needed for y-axis value tick labels across all subplots."""
    global_min = float('inf')
    global_max = float('-inf')
    for _, df_sub in effective_plots:
        vals = df_sub.values
        if len(vals) == 0:
            continue
        sub_min = np.nanmin(vals)
        sub_max = np.nanmax(vals)
        if np.isfinite(sub_min):
            global_min = min(global_min, sub_min)
        if np.isfinite(sub_max):
            global_max = max(global_max, sub_max)

    if not np.isfinite(global_min) or not np.isfinite(global_max):
        return MIN_VALUE_LABEL_WIDTH

    # Sample values at extremes and midpoint
    sample_values = [global_min, global_max, 0]
    mid = (global_min + global_max) / 2
    if mid != global_min and mid != global_max:
        sample_values.append(mid)

    # Check a representative set of formatters (first and last subplot)
    max_chars = 0
    n_subs = len(effective_plots)
    formatter_indices = set([0, n_subs - 1])
    for idx in formatter_indices:
        fmt = _get_value_formatter(axis_tick_format, idx)
        for v in sample_values:
            try:
                chars = len(fmt(v, 0))
            except (ValueError, TypeError):
                chars = len(str(v))
            max_chars = max(max_chars, chars)

    width = max_chars * TICK_CHAR_WIDTH + 0.1  # padding for tick marks
    return max(MIN_VALUE_LABEL_WIDTH, width)


def _compute_line_layout(
    effective_plots: list[tuple[str | None, pd.DataFrame]],
    item_level_names: list[str],
    legend_position: str,
    subplots_per_row: int,
    base_width: float,
    subplot_height: float,
    axis_tick_format,
) -> LineLayoutParams:
    """Compute layout parameters consistent across file batches.

    Examines ALL effective_plots so that every file uses identical margins.
    """
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # ── value-label width (y-axis tick labels) ──
    value_label_width = _estimate_value_label_width(effective_plots, axis_tick_format)

    # ── legend width (max across ALL subplots) ──
    legend_width = 0.0
    if item_level_names:
        for _, df_sub in effective_plots:
            legend_labels = _format_legend_labels(
                _get_column_items(df_sub, item_level_names)
            )
            w = estimate_legend_width(legend_labels, base_width=0.6)
            legend_width = max(legend_width, w)

    return LineLayoutParams(
        value_label_width=value_label_width,
        legend_width=legend_width,
        base_width=base_width,
        subplot_height=subplot_height,
    )


# ---------------------------------------------------------------------------
#  Lines
# ---------------------------------------------------------------------------

def _render_lines_figure(
    effective_plots, plot_name, sub_levels, line_level_names, time_index,
    subplots_per_row, legend_position,
    xlabel, ylabel,
    axis_bounds, axis_tick_format, always_include_zero_in_axis,
    output_filepath,
    layout: LineLayoutParams,
):
    """Render one file's worth of line subplots."""
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # Only reserve space for subplot titles when at least one exists
    has_titles = any(title is not None for title, _ in effective_plots)
    subplot_vpad = SUBPLOT_VPAD if has_titles else 0

    # ── Figure sizing ──
    cell_width = layout.value_label_width + layout.base_width
    if layout.legend_width > 0 and legend_position == 'all' and n_cols > 1:
        cell_width += layout.legend_width + LEGEND_GAP

    left_edge = LEFT_PAD + (YLABEL_WIDTH if ylabel else 0)
    total_width = left_edge + cell_width * n_cols + INTER_COL_GAP * max(0, n_cols - 1) + RIGHT_PAD
    # For 'right' legend (or single column), add legend space once
    if layout.legend_width > 0 and not (legend_position == 'all' and n_cols > 1):
        total_width += layout.legend_width + LEGEND_GAP

    cell_height = layout.subplot_height + subplot_vpad
    content_height = cell_height * n_rows + INTER_ROW_GAP * max(0, n_rows - 1)
    bottom_pad = BOTTOM_PAD + (XLABEL_HEIGHT if xlabel else 0)
    total_height = TITLE_PAD + content_height + bottom_pad

    # ── Axes placement ──
    fig = plt.figure(figsize=(total_width, total_height))
    axes = [None] * n_subs

    y_cursor = total_height - TITLE_PAD
    for r in range(n_rows):
        row_top = y_cursor
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= n_subs:
                break
            x_left = (left_edge + c * (cell_width + INTER_COL_GAP) + layout.value_label_width) / total_width
            ax_width = layout.base_width / total_width
            y_bottom = (row_top - cell_height) / total_height
            ax_height = layout.subplot_height / total_height
            axes[idx] = fig.add_axes([x_left, y_bottom, ax_width, ax_height])
        y_cursor -= cell_height + INTER_ROW_GAP

    # ── Per-subplot rendering ──
    for idx, (eff_title, df_sub) in enumerate(effective_plots):
        ax = axes[idx]

        # Get line combinations from line_levels
        if isinstance(df_sub, pd.Series):
            ax.plot(time_index, df_sub.values, label=str(eff_title))
        else:
            is_multiindex = isinstance(df_sub.columns, pd.MultiIndex)

            if is_multiindex:
                if len(line_level_names) == 1:
                    lines = df_sub.columns.get_level_values(line_level_names[0]).unique().tolist()
                else:
                    line_df = df_sub.columns.to_frame()[line_level_names].drop_duplicates()
                    lines = [tuple(row) for row in line_df.values]
            else:
                lines = df_sub.columns.unique().tolist()

            for line in lines:
                if is_multiindex:
                    if len(line_level_names) == 1:
                        y_data = df_sub.xs(line, level=line_level_names[0], axis=1)
                    else:
                        y_data = df_sub
                        for lvl_name, lvl_val in zip(line_level_names, line):
                            if isinstance(y_data, pd.Series):
                                break
                            if isinstance(y_data.columns, pd.MultiIndex):
                                y_data = y_data.xs(lvl_val, level=lvl_name, axis=1)
                            else:
                                y_data = y_data[lvl_val]
                else:
                    y_data = df_sub[line]

                if isinstance(y_data, pd.DataFrame):
                    y_data = y_data.sum(axis=1)

                ax.plot(time_index, y_data.values, label=str(line))

        # Subplot formatting
        if eff_title is not None:
            ax.set_title(str(eff_title), pad=2)

        if line_level_names and _should_show_legend(legend_position, sub_levels, idx, n_cols, n_subs):
            handles, labels_leg = ax.get_legend_handles_labels()
            if handles:
                legend_x = 1 + LEGEND_GAP / layout.base_width
                ax.legend(handles, labels_leg, bbox_to_anchor=(legend_x, 1), loc='upper left', fontsize=8, borderaxespad=0)

        ax.grid(True, alpha=0.3)

        # Axis scale, formatter, and labels
        row = idx // n_cols
        col = idx % n_cols
        if always_include_zero_in_axis:
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 0), max(hi, 0))
        scale = _subplot_axis_bounds(axis_bounds, idx)
        if scale:
            ax.set_ylim(scale[0], scale[1])
        _fmt = _get_value_formatter(axis_tick_format, idx)
        lo, hi = ax.get_ylim()
        ax_height_inches = layout.subplot_height
        nbins = _estimate_value_nbins(lo, hi, ax_height_inches, _fmt, is_horizontal_axis=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        ax.yaxis.set_major_formatter(_fmt)
        _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows)

        ax_width_inches = layout.base_width
        set_smart_xticks(ax, time_index, ax_width_inches)

    # ── Figure title ──
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(plot_name, y=1 - 0.14 / fig_h, va='top')

    # ── Save (fixed layout, no bbox_inches='tight') ──
    if output_filepath:
        plt.savefig(output_filepath)
    else:
        plt.savefig(f'{output_filepath}')
    plt.close(fig)


def plot_dt_sub_lines(df_plot, plot_name, plot_dir, sub_levels, line_levels,
    rows=(0,167), subplots_per_row=3, legend_position='right',
    xlabel=None, ylabel=None, base_width_per_col=6, subplot_height=4,
    axis_bounds=None, axis_tick_format=None, always_include_zero_in_axis=True,
    max_items_per_plot=None, max_subplots_per_file=None, output_filepath=None):

    # Convert level indices to level names
    if isinstance(df_plot.columns, pd.MultiIndex):
        line_level_names = [df_plot.columns.names[i] for i in line_levels]
    else:
        line_level_names = line_levels

    # Get x-axis index
    if isinstance(df_plot.index, pd.MultiIndex):
        time_index = df_plot.index.get_level_values(-1).astype(str)
    else:
        time_index = df_plot.index.astype(str)

    # Build effective_plots with item splitting
    effective_plots = _build_effective_plots(
        df_plot, sub_levels, line_level_names, max_items_per_plot
    )
    if not effective_plots:
        return

    # Compute layout once across ALL effective_plots
    layout = _compute_line_layout(
        effective_plots, line_level_names,
        legend_position, subplots_per_row,
        base_width_per_col, subplot_height,
        axis_tick_format,
    )

    # Split into file batches
    for batch, batch_filepath in _make_file_batches(
        effective_plots, max_subplots_per_file, output_filepath, plot_dir, plot_name
    ):
        _render_lines_figure(
            batch, plot_name, sub_levels, line_level_names, time_index,
            subplots_per_row, legend_position,
            xlabel, ylabel,
            axis_bounds, axis_tick_format, always_include_zero_in_axis,
            batch_filepath,
            layout=layout,
        )


# ---------------------------------------------------------------------------
#  Stacked area
# ---------------------------------------------------------------------------

def _render_stack_figure(
    effective_plots, plot_name, sub_levels, stack_level_names, time_index,
    subplots_per_row, legend_position,
    xlabel, ylabel,
    axis_bounds, axis_tick_format, always_include_zero_in_axis,
    output_filepath,
    layout: LineLayoutParams,
):
    """Render one file's worth of stacked-area subplots."""
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # Only reserve space for subplot titles when at least one exists
    has_titles = any(title is not None for title, _ in effective_plots)
    subplot_vpad = SUBPLOT_VPAD if has_titles else 0

    # ── Figure sizing ──
    cell_width = layout.value_label_width + layout.base_width
    if layout.legend_width > 0 and legend_position == 'all' and n_cols > 1:
        cell_width += layout.legend_width + LEGEND_GAP

    left_edge = LEFT_PAD + (YLABEL_WIDTH if ylabel else 0)
    total_width = left_edge + cell_width * n_cols + INTER_COL_GAP * max(0, n_cols - 1) + RIGHT_PAD
    # For 'right' legend (or single column), add legend space once
    if layout.legend_width > 0 and not (legend_position == 'all' and n_cols > 1):
        total_width += layout.legend_width + LEGEND_GAP

    cell_height = layout.subplot_height + subplot_vpad
    content_height = cell_height * n_rows + INTER_ROW_GAP * max(0, n_rows - 1)
    bottom_pad = BOTTOM_PAD + (XLABEL_HEIGHT if xlabel else 0)
    total_height = TITLE_PAD + content_height + bottom_pad

    # ── Axes placement ──
    fig = plt.figure(figsize=(total_width, total_height))
    axes = [None] * n_subs

    y_cursor = total_height - TITLE_PAD
    for r in range(n_rows):
        row_top = y_cursor
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= n_subs:
                break
            x_left = (left_edge + c * (cell_width + INTER_COL_GAP) + layout.value_label_width) / total_width
            ax_width = layout.base_width / total_width
            y_bottom = (row_top - cell_height) / total_height
            ax_height = layout.subplot_height / total_height
            axes[idx] = fig.add_axes([x_left, y_bottom, ax_width, ax_height])
        y_cursor -= cell_height + INTER_ROW_GAP

    # ── Per-subplot rendering ──
    for idx, (eff_title, df_sub) in enumerate(effective_plots):
        ax = axes[idx]

        # Get stack combinations from stack_levels
        if isinstance(df_sub, pd.Series):
            df_to_plot = df_sub.to_frame()
        else:
            is_multiindex = isinstance(df_sub.columns, pd.MultiIndex)

            if is_multiindex:
                if len(stack_level_names) == 1:
                    stacks = df_sub.columns.get_level_values(stack_level_names[0]).unique().tolist()
                else:
                    stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
                    stacks = [tuple(row) for row in stack_df.values]
            else:
                stacks = df_sub.columns.unique().tolist()

            data_dict = {}
            for stack in stacks:
                if is_multiindex:
                    if len(stack_level_names) == 1:
                        y_data = df_sub.xs(stack, level=stack_level_names[0], axis=1)
                    else:
                        y_data = df_sub.xs(stack, level=stack_level_names, axis=1)
                else:
                    y_data = df_sub[stack]

                if isinstance(y_data, pd.DataFrame):
                    y_data = y_data.sum(axis=1)

                data_dict[str(stack)] = y_data

            df_to_plot = pd.DataFrame(data_dict, index=df_sub.index)

        # Reset index to use time only (drop period)
        df_to_plot.index = time_index

        # Split columns with both positive and negative values
        for col_name in df_to_plot.columns.tolist():
            has_pos = (df_to_plot[col_name] > 0).any()
            has_neg = (df_to_plot[col_name] < 0).any()
            if has_pos and has_neg:
                df_to_plot[f'{col_name}_pos'] = df_to_plot[col_name].clip(lower=0)
                df_to_plot[f'{col_name}_neg'] = df_to_plot[col_name].clip(upper=0)
                df_to_plot = df_to_plot.drop(columns=[col_name])

        # Create stacked area plot
        n_columns = len(df_to_plot.columns)
        colors = plt.colormaps['tab10'].colors[:n_columns]
        if n_columns > 10:
            colors = plt.colormaps['tab20'].colors[:n_columns]
        df_to_plot.plot.area(stacked=True, ax=ax, alpha=1.0, legend=False, linewidth=0, color=colors, xlabel="")

        # Subplot formatting
        if eff_title is not None:
            ax.set_title(str(eff_title), pad=2)

        if _should_show_legend(legend_position, sub_levels, idx, n_cols, n_subs):
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                legend_x = 1 + LEGEND_GAP / layout.base_width
                ax.legend(handles, labels, bbox_to_anchor=(legend_x, 1), loc='upper left', fontsize=8, borderaxespad=0)

        ax.grid(True, alpha=0.3)

        # Axis scale, formatter, and labels
        row = idx // n_cols
        col = idx % n_cols
        if always_include_zero_in_axis:
            lo, hi = ax.get_ylim()
            ax.set_ylim(min(lo, 0), max(hi, 0))
        scale = _subplot_axis_bounds(axis_bounds, idx)
        if scale:
            ax.set_ylim(scale[0], scale[1])
        _fmt = _get_value_formatter(axis_tick_format, idx)
        lo, hi = ax.get_ylim()
        ax_height_inches = layout.subplot_height
        nbins = _estimate_value_nbins(lo, hi, ax_height_inches, _fmt, is_horizontal_axis=False)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune='upper'))
        ax.yaxis.set_major_formatter(_fmt)
        _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows)

        ax_width_inches = layout.base_width
        set_smart_xticks(ax, time_index, ax_width_inches)

    # ── Figure title ──
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(plot_name, y=1 - 0.14 / fig_h, va='top')

    # ── Save (fixed layout, no bbox_inches='tight') ──
    if output_filepath:
        plt.savefig(output_filepath)
    else:
        plt.savefig(f'{output_filepath}')
    plt.close(fig)


def plot_dt_stack_sub(df_plot, plot_name, plot_dir, stack_levels, sub_levels,
        rows=(0,167), subplots_per_row=3,
        legend_position='right',
        xlabel=None, ylabel=None, base_width_per_col=6, subplot_height=4,
        axis_bounds=None, axis_tick_format=None, always_include_zero_in_axis=True,
        max_items_per_plot=None, max_subplots_per_file=None, output_filepath=None):

    # Convert level indices to level names
    if isinstance(df_plot.columns, pd.MultiIndex):
        stack_level_names = [df_plot.columns.names[i] for i in stack_levels]
    else:
        stack_level_names = stack_levels

    # Get x-axis index
    if isinstance(df_plot.index, pd.MultiIndex):
        time_index = df_plot.index.get_level_values(-1).astype(str)
    else:
        time_index = df_plot.index.astype(str)

    # Build effective_plots with item splitting
    effective_plots = _build_effective_plots(
        df_plot, sub_levels, stack_level_names, max_items_per_plot
    )
    if not effective_plots:
        return

    # Compute layout once across ALL effective_plots
    layout = _compute_line_layout(
        effective_plots, stack_level_names,
        legend_position, subplots_per_row,
        base_width_per_col, subplot_height,
        axis_tick_format,
    )

    # Split into file batches
    for batch, batch_filepath in _make_file_batches(
        effective_plots, max_subplots_per_file, output_filepath, plot_dir, plot_name
    ):
        _render_stack_figure(
            batch, plot_name, sub_levels, stack_level_names, time_index,
            subplots_per_row, legend_position,
            xlabel, ylabel,
            axis_bounds, axis_tick_format, always_include_zero_in_axis,
            batch_filepath,
            layout=layout,
        )
