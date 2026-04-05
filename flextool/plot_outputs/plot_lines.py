import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from flextool.plot_outputs.format_helpers import _get_value_formatter
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, _format_legend_labels, _should_show_legend,
    build_shared_color_map,
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
TICK_CHAR_WIDTH = 0.075      # Approximate width per digit character at tick font size
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
    if df_sub.empty or len(df_sub.columns) == 0:
        return df_sub
    if len(level_names) == 1:
        mask = df_sub.columns.get_level_values(level_names[0]).isin(items)
    else:
        col_frame = df_sub.columns.to_frame()[level_names]
        if col_frame.empty:
            return df_sub
        mask = col_frame.apply(tuple, axis=1).isin(items).values
    return df_sub.loc[:, mask]


def _magnitude_label(mag: int) -> str:
    """Return a human-readable label for a magnitude bucket.

    mag is the floor(log10(max_abs)) value.  The special sentinel
    ``mag = -999`` stands for the "< 1" catch-all bucket.
    """
    if mag == -999:
        return '0\u20131'          # 0–1  (all values with max abs < 1)
    lo = 10 ** mag
    hi = 10 ** (mag + 1)
    def _fmt(v):
        if v >= 1:
            return f'{v:,.0f}'     # e.g. 1,000
        return f'{v:g}'            # e.g. 0.1
    return f'{_fmt(lo)}\u2013{_fmt(hi)}'


def _group_items_by_magnitude(
    df_sub: pd.DataFrame,
    item_level_names: list[str],
) -> dict[int, list]:
    """Group column items by order-of-magnitude of their max absolute value.

    Returns ``{magnitude_key: [items]}`` sorted by magnitude ascending.
    Items with max abs < 1 all share the key ``-999``.
    """
    items = _get_column_items(df_sub, item_level_names)
    buckets: dict[int, list] = {}
    for item in items:
        col_data = _filter_columns_by_items(df_sub, [item], item_level_names)
        max_abs = col_data.abs().max().max()
        if not np.isfinite(max_abs) or max_abs < 1e-15:
            mag = -999
        elif max_abs < 1.0:
            mag = -999              # lump everything < 1 together
        else:
            mag = math.floor(math.log10(max_abs))
        buckets.setdefault(mag, []).append(item)
    return dict(sorted(buckets.items()))


def _build_effective_plots(df_plot, sub_levels, item_level_names,
                           max_items_per_plot, subplots_by_magnitudes=False):
    """Pre-expand subplots into effective_plots with item splitting.

    When *subplots_by_magnitudes* is True the columns are first grouped
    by order-of-magnitude of their max absolute value (within each 'u'
    subplot group) and each magnitude bucket becomes its own subplot.
    """
    subs = _get_unique_levels(df_plot.columns, sub_levels)
    effective_plots = []
    for sub in subs:
        df_sub = _extract_subplot_data(df_plot, sub, sub_levels)
        base_title = (
            ' | '.join(str(v) for v in sub) if isinstance(sub, tuple)
            else str(sub) if sub is not None else None
        )

        if subplots_by_magnitudes:
            mag_groups = _group_items_by_magnitude(df_sub, item_level_names)
            for mag, mag_items in mag_groups.items():
                mag_str = _magnitude_label(mag)
                if base_title:
                    mag_title = f'{base_title} ({mag_str})'
                else:
                    mag_title = f'({mag_str})'

                if max_items_per_plot and len(mag_items) > max_items_per_plot:
                    n_chunks = math.ceil(len(mag_items) / max_items_per_plot)
                    for ci in range(n_chunks):
                        start = ci * max_items_per_plot
                        chunk_items = mag_items[start:start + max_items_per_plot]
                        df_chunk = _filter_columns_by_items(
                            df_sub, chunk_items, item_level_names
                        )
                        # mag_title ends with ')'; insert chunk index before it
                        title = f'{mag_title[:-1]}, {ci + 1}/{n_chunks})'
                        effective_plots.append((title, df_chunk))
                else:
                    df_mag = _filter_columns_by_items(
                        df_sub, mag_items, item_level_names
                    )
                    effective_plots.append((mag_title, df_mag))
        else:
            items = _get_column_items(df_sub, item_level_names)
            if max_items_per_plot and len(items) > max_items_per_plot:
                for i in range(0, len(items), max_items_per_plot):
                    chunk_items = items[i:i + max_items_per_plot]
                    df_chunk = _filter_columns_by_items(df_sub, chunk_items, item_level_names)
                    chunk_idx = i // max_items_per_plot + 1
                    effective_plots.append((f"{base_title}_{chunk_idx}", df_chunk))
            else:
                effective_plots.append((base_title, df_sub))
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
        finite_mask = np.isfinite(vals)
        if not finite_mask.any():
            continue
        sub_min = np.nanmin(vals[finite_mask])
        sub_max = np.nanmax(vals[finite_mask])
        global_min = min(global_min, sub_min)
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

    # ── legend width ──
    legend_width = 0.0
    if item_level_names:
        if legend_position == 'shared':
            # Use union of all labels across all subplots
            all_items: list = []
            for _, df_sub in effective_plots:
                for item in _get_column_items(df_sub, item_level_names):
                    if item not in all_items:
                        all_items.append(item)
            legend_labels = _format_legend_labels(all_items)
            legend_width = estimate_legend_width(legend_labels, base_width=0.6)
        else:
            # Max across individual subplots
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

def _build_lines_figure(
    effective_plots, plot_name, sub_levels, line_level_names, time_index,
    subplots_per_row, legend_position,
    xlabel, ylabel,
    axis_bounds, axis_tick_format, always_include_zero_in_axis,
    layout: LineLayoutParams,
    shared_color_map: dict[str, tuple] | None = None,
) -> plt.Figure:
    """Build a line-plot Figure and return it (without saving or closing)."""
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

    # Use integer x-positions so NaN gap rows occupy real horizontal space.
    # set_smart_xticks handles tick labelling from time_index separately.
    x_positions = np.arange(len(time_index))

    # ── Per-subplot rendering ──
    for idx, (eff_title, df_sub) in enumerate(effective_plots):
        ax = axes[idx]

        # Get line combinations from line_levels
        if isinstance(df_sub, pd.Series):
            label = str(eff_title)
            color = shared_color_map.get(label) if shared_color_map else None
            ax.plot(x_positions, df_sub.values, label=label, color=color)
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

                label = str(line)
                color = shared_color_map.get(label) if shared_color_map else None
                ax.plot(x_positions, y_data.values, label=label, color=color)

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
        if scale and scale[0] != scale[1]:
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

    # ── Shared legend (one per file, anchored to top-right subplot) ──
    if legend_position == 'shared' and shared_color_map:
        from matplotlib.lines import Line2D
        legend_ax_idx = min(n_cols - 1, n_subs - 1)
        ax_legend = axes[legend_ax_idx]
        handles = [Line2D([0], [0], color=c) for c in shared_color_map.values()]
        labels_all = list(shared_color_map.keys())
        legend_x = 1 + LEGEND_GAP / layout.base_width
        ax_legend.legend(handles, labels_all, bbox_to_anchor=(legend_x, 1),
                         loc='upper left', fontsize=8, borderaxespad=0)

    # ── Figure title ──
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(plot_name, y=1 - 0.14 / fig_h, va='top')

    return fig


def _render_lines_figure(
    effective_plots, plot_name, sub_levels, line_level_names, time_index,
    subplots_per_row, legend_position,
    xlabel, ylabel,
    axis_bounds, axis_tick_format, always_include_zero_in_axis,
    output_filepath,
    layout: LineLayoutParams,
    shared_color_map: dict[str, tuple] | None = None,
):
    """Render one file's worth of line subplots and save to disk."""
    fig = _build_lines_figure(
        effective_plots, plot_name, sub_levels, line_level_names, time_index,
        subplots_per_row, legend_position,
        xlabel, ylabel,
        axis_bounds, axis_tick_format, always_include_zero_in_axis,
        layout, shared_color_map,
    )
    if output_filepath:
        plt.savefig(output_filepath)
    else:
        plt.savefig(f'{output_filepath}')
    plt.close(fig)


def plot_dt_sub_lines(df_plot, plot_name, plot_dir, sub_levels, line_levels,
    rows=(0,167), subplots_per_row=2, legend_position='right',
    xlabel=None, ylabel=None, base_width_per_col=6, subplot_height=4,
    axis_bounds=None, axis_tick_format='1,.0f', always_include_zero_in_axis=True,
    max_items_per_plot=10, max_subplots_per_file=6, output_filepath=None,
    only_first_file=False, subplots_by_magnitudes=False):

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

    # Build effective_plots with item splitting (and optional magnitude splitting)
    effective_plots = _build_effective_plots(
        df_plot, sub_levels, line_level_names, max_items_per_plot,
        subplots_by_magnitudes=subplots_by_magnitudes,
    )
    if not effective_plots:
        return 0

    # Build shared color map before splitting into file batches
    shared_color_map = None
    if legend_position == 'shared' and line_level_names:
        all_labels: list[str] = []
        for _, df_sub in effective_plots:
            for item in _get_column_items(df_sub, line_level_names):
                label = str(item)
                if label not in all_labels:
                    all_labels.append(label)
        all_labels.sort()
        shared_color_map = build_shared_color_map(all_labels)

    # Compute layout once across ALL effective_plots
    layout = _compute_line_layout(
        effective_plots, line_level_names,
        legend_position, subplots_per_row,
        base_width_per_col, subplot_height,
        axis_tick_format,
    )

    # Split into file batches
    file_batches = _make_file_batches(
        effective_plots, max_subplots_per_file, output_filepath, plot_dir, plot_name
    )
    batches_to_render = file_batches[:1] if only_first_file else file_batches
    n_total = len(file_batches)
    for batch_idx, (batch, batch_filepath) in enumerate(batches_to_render, start=1):
        batch_title = f"{plot_name} ({batch_idx}/{n_total})" if n_total > 1 else plot_name
        _render_lines_figure(
            batch, batch_title, sub_levels, line_level_names, time_index,
            subplots_per_row, legend_position,
            xlabel, ylabel,
            axis_bounds, axis_tick_format, always_include_zero_in_axis,
            batch_filepath,
            layout=layout,
            shared_color_map=shared_color_map,
        )
    return len(file_batches) - len(batches_to_render)


# ---------------------------------------------------------------------------
#  Stacked area
# ---------------------------------------------------------------------------

def _build_stack_figure(
    effective_plots, plot_name, sub_levels, stack_level_names, time_index,
    subplots_per_row, legend_position,
    xlabel, ylabel,
    axis_bounds, axis_tick_format, always_include_zero_in_axis,
    layout: LineLayoutParams,
    shared_color_map: dict[str, tuple] | None = None,
) -> plt.Figure:
    """Build a stacked-area Figure and return it (without saving or closing)."""
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

        # Sort columns alphabetically when using shared colors so visual matches legend
        if shared_color_map:
            sorted_cols = sorted(df_to_plot.columns, key=str)
            df_to_plot = df_to_plot[sorted_cols]

        # Split columns with both positive and negative values (preserving order)
        new_cols: dict[str, pd.Series] = {}
        for col_name in df_to_plot.columns.tolist():
            has_pos = (df_to_plot[col_name] > 0).any()
            has_neg = (df_to_plot[col_name] < 0).any()
            if has_pos and has_neg:
                new_cols[f'{col_name}_pos'] = df_to_plot[col_name].clip(lower=0)
                new_cols[f'{col_name}_neg'] = df_to_plot[col_name].clip(upper=0)
            else:
                new_cols[col_name] = df_to_plot[col_name]
        df_to_plot = pd.DataFrame(new_cols, index=df_to_plot.index)

        # Create stacked area plot
        n_columns = len(df_to_plot.columns)
        if shared_color_map:
            # Handle _pos/_neg suffixes from mixed-sign column splitting
            def _lookup_color(col_name):
                if col_name in shared_color_map:
                    return shared_color_map[col_name]
                for suffix in ('_pos', '_neg'):
                    if col_name.endswith(suffix):
                        base = col_name[:-len(suffix)]
                        if base in shared_color_map:
                            return shared_color_map[base]
                return (0.5, 0.5, 0.5)
            colors = [_lookup_color(col) for col in df_to_plot.columns]
        else:
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
                # Build legend so visual top-to-bottom matches legend top-to-bottom.
                # Positive areas stack upward (col 0 = bottom → reverse for legend).
                # Negative areas stack downward (col 0 = top → keep order for legend).
                pos_pairs = []
                neg_pairs = []
                for h, l in zip(handles, labels):
                    if l in df_to_plot.columns and (df_to_plot[l] <= 0).all():
                        neg_pairs.append((h, l))
                    else:
                        pos_pairs.append((h, l))
                pos_pairs.reverse()
                ordered = pos_pairs + neg_pairs
                handles = [h for h, _ in ordered]
                labels = [l for _, l in ordered]
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
        if scale and scale[0] != scale[1]:
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

    # ── Shared legend (one per file, anchored to top-right subplot) ──
    # Reversed so top-of-stack = top-of-legend
    if legend_position == 'shared' and shared_color_map:
        from matplotlib.patches import Patch
        legend_ax_idx = min(n_cols - 1, n_subs - 1)
        ax_legend = axes[legend_ax_idx]
        keys = list(shared_color_map.keys())
        colors_list = list(shared_color_map.values())
        handles = [Patch(facecolor=c) for c in reversed(colors_list)]
        labels_all = list(reversed(keys))
        legend_x = 1 + LEGEND_GAP / layout.base_width
        ax_legend.legend(handles, labels_all, bbox_to_anchor=(legend_x, 1),
                         loc='upper left', fontsize=8, borderaxespad=0)

    # ── Figure title ──
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(plot_name, y=1 - 0.14 / fig_h, va='top')

    return fig


def _render_stack_figure(
    effective_plots, plot_name, sub_levels, stack_level_names, time_index,
    subplots_per_row, legend_position,
    xlabel, ylabel,
    axis_bounds, axis_tick_format, always_include_zero_in_axis,
    output_filepath,
    layout: LineLayoutParams,
    shared_color_map: dict[str, tuple] | None = None,
):
    """Render one file's worth of stacked-area subplots and save to disk."""
    fig = _build_stack_figure(
        effective_plots, plot_name, sub_levels, stack_level_names, time_index,
        subplots_per_row, legend_position,
        xlabel, ylabel,
        axis_bounds, axis_tick_format, always_include_zero_in_axis,
        layout, shared_color_map,
    )
    if output_filepath:
        plt.savefig(output_filepath)
    else:
        plt.savefig(f'{output_filepath}')
    plt.close(fig)


def plot_dt_stack_sub(df_plot, plot_name, plot_dir, stack_levels, sub_levels,
        rows=(0,167), subplots_per_row=2,
        legend_position='right',
        xlabel=None, ylabel=None, base_width_per_col=6, subplot_height=4,
        axis_bounds=None, axis_tick_format='1,.0f', always_include_zero_in_axis=True,
        max_items_per_plot=10, max_subplots_per_file=6, output_filepath=None,
        only_first_file=False):

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
        return 0

    # Build shared color map before splitting into file batches
    shared_color_map = None
    if legend_position == 'shared' and stack_level_names:
        all_labels: list[str] = []
        for _, df_sub in effective_plots:
            for item in _get_column_items(df_sub, stack_level_names):
                label = str(item)
                if label not in all_labels:
                    all_labels.append(label)
        all_labels.sort()
        shared_color_map = build_shared_color_map(all_labels)

    # Compute layout once across ALL effective_plots
    layout = _compute_line_layout(
        effective_plots, stack_level_names,
        legend_position, subplots_per_row,
        base_width_per_col, subplot_height,
        axis_tick_format,
    )

    # Split into file batches
    file_batches = _make_file_batches(
        effective_plots, max_subplots_per_file, output_filepath, plot_dir, plot_name
    )
    batches_to_render = file_batches[:1] if only_first_file else file_batches
    n_total = len(file_batches)
    for batch_idx, (batch, batch_filepath) in enumerate(batches_to_render, start=1):
        batch_title = f"{plot_name} ({batch_idx}/{n_total})" if n_total > 1 else plot_name
        _render_stack_figure(
            batch, batch_title, sub_levels, stack_level_names, time_index,
            subplots_per_row, legend_position,
            xlabel, ylabel,
            axis_bounds, axis_tick_format, always_include_zero_in_axis,
            batch_filepath,
            layout=layout,
            shared_color_map=shared_color_map,
        )
    return len(file_batches) - len(batches_to_render)
