import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, FixedLocator, FixedFormatter
from flextool.plot_outputs.format_helpers import _get_value_formatter, format_value_label
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, estimate_legend_height,
    _format_legend_labels, _should_show_legend,
    build_shared_color_map,
)
from flextool.plot_outputs.axis_helpers import (
    _subplot_axis_bounds, _apply_subplot_label, _estimate_value_nbins,
    _ylabel_axes_x,
)
from flextool.plot_outputs.subplot_helpers import (
    BarLayoutParams, _calculate_grid_layout, _get_unique_levels, _extract_subplot_data,
    _sort_subs,
)
from flextool.plot_outputs.plot_bars_detail import (
    _plot_grouped_bars, _plot_stacked_bars, _plot_simple_bars,
)

# ── Layout constants (inches) ──
CHAR_WIDTH = 0.081          # Approximate width per character at font-size 9
LEFT_PAD = 0.25             # Left edge padding
RIGHT_PAD = 0.35            # Right margin for value-axis tick labels
BAR_HEIGHT = 0.24           # Minimum row slot height (font-driven baseline)
# Bar thickness is fixed per row regardless of how many bars share the row.
# REFERENCE_BARS_PER_ROW + (n-1) gaps fit exactly into BAR_HEIGHT, so a row
# of REFERENCE_BARS_PER_ROW bars uses the baseline slot. Fewer bars leave
# whitespace; more bars grow the slot.
REFERENCE_BARS_PER_ROW = 4
BAR_GAP_FRACTION = 0.10            # gap between adjacent bars = 10% of bar thickness
SOLO_BAR_THICKNESS_MULT = 2.0      # one-bar-per-row case (no grouping → no legend needed for groups)
_REF_DENOM = REFERENCE_BARS_PER_ROW + (REFERENCE_BARS_PER_ROW - 1) * BAR_GAP_FRACTION
REFERENCE_BAR_THICKNESS = BAR_HEIGHT / _REF_DENOM   # ≈ 0.0558 in
SOLO_BAR_THICKNESS = REFERENCE_BAR_THICKNESS * SOLO_BAR_THICKNESS_MULT  # ≈ 0.1116 in
# When value labels are drawn on bars, the regular bar thickness leaves the
# numeric labels too close together (the center-to-center bar pitch is only
# ~0.44x the label font line height, so labels overlap). Thicken bars — and
# correspondingly grow each bar's slot height so the thicker bars still fit —
# by this factor whenever value labels are enabled. 2.5x raises the pitch to
# slightly above one font line height (~0.139" at 10pt), separating labels.
VALUE_LABEL_BAR_THICKNESS_MULT = 2.5
SUBPLOT_VPAD = 0.3          # Space above axes for subplot title
INTER_COL_GAP = 0.4         # Horizontal gap between subplot columns
INTER_ROW_GAP = 0.6         # Vertical gap between subplot rows
VALUE_LABEL_MARGIN = 0.2    # Extra right margin when value labels are shown
YLABEL_WIDTH = 0.4          # Space reserved for y-axis label text
XLABEL_HEIGHT = 0.2         # Space reserved for x-axis label text
LEGEND_GAP = 0.15           # Gap between drawing area and legend box
TITLE_PAD = 0.3             # Top margin for subplot title above axes
BOTTOM_PAD = 0.35           # Bottom margin for x-axis ticks/labels below axes

# Category (bar) tick labels are middle-ellipsised to at most this many
# characters PER index/group component. A single pathological name (e.g.
# "EGNGSC_Shoubra_El-Kheima_Steam_Turbines") would otherwise set the left
# margin for every subplot in the grid, squeezing all the bars. Middle
# truncation keeps both the head and the tail so the name stays
# recognisable; the GUI shows the full text on hover.
BAR_LABEL_MAX_CHARS = 28
_ELLIPSIS = "…"


def _truncate_middle(text: str, cap: int = BAR_LABEL_MAX_CHARS) -> str:
    """Shorten *text* to *cap* chars keeping head and tail, ellipsis in between."""
    if len(text) <= cap:
        return text
    if cap <= 1:
        return text[:cap]
    keep = cap - 1  # one char spent on the ellipsis
    head = (keep + 1) // 2
    tail = keep - head
    return text[:head] + _ELLIPSIS + (text[-tail:] if tail else "")


def _bar_label_components(idx_val, group=None) -> list[str]:
    """Flatten a bar's index value (+ optional expand group) into string parts."""
    parts = list(idx_val) if isinstance(idx_val, tuple) else [idx_val]
    if group is not None:
        parts += list(group) if isinstance(group, tuple) else [group]
    return [str(p) for p in parts]


def _format_bar_label(idx_val, group=None, *, truncate: bool = True) -> str:
    """Build a bar's display label as ``comp1 | comp2 | …`` (expand group last).

    With ``truncate`` each component is middle-ellipsised to
    ``BAR_LABEL_MAX_CHARS``. ``truncate=False`` yields the full label used for
    the GUI hover tooltip.
    """
    parts = _bar_label_components(idx_val, group)
    if truncate:
        parts = [_truncate_middle(p) for p in parts]
    return " | ".join(parts)


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
    file in a batch uses identical margins.
    """
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # ── bar-label width (max DRAWN label across ALL subplots) ──
    # The drawn tick label includes the expand-group level (folded in with
    # " | ", group last) and is middle-truncated per component, so size the
    # column to that — not to the raw index text — to keep the grid uniform.
    max_bar_label_chars = 0
    for _, df_sub in effective_plots:
        df_sub_clean = df_sub.dropna(how='all')
        if df_sub_clean.empty:
            continue
        for idx_val in df_sub_clean.index:
            max_bar_label_chars = max(
                max_bar_label_chars, len(_format_bar_label(idx_val))
            )

    # ── expand-group width (folded into the tick label, not a separate column) ──
    if expand_axis_levels:
        not_expand = list(set(range(len(df.columns.names))) - set(expand_axis_levels))
        expand_names = df.columns.droplevel(not_expand)
        if isinstance(expand_names, pd.MultiIndex):
            group_drawn = [_format_bar_label(tuple(g)) for g in expand_names]
        else:
            group_drawn = [_format_bar_label(g) for g in expand_names]
        max_group_chars = max((len(s) for s in group_drawn), default=0)
        max_bar_label_chars += 3 + max_group_chars  # " | " separator + group

    bar_label_width = max_bar_label_chars * CHAR_WIDTH
    group_label_width = 0.0  # group is now part of the tick label
    total_label_width = bar_label_width

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


def _build_bar_figure(
    effective_plots, df, key_name, plot_dir,
    stack_levels, stack_level_names,
    expand_axis_levels, expand_axis_level_names,
    sub_levels,
    grouped_bar_levels, grouped_bar_level_names,
    legend_position, subplots_per_row,
    xlabel, ylabel, bar_orientation, base_bar_length,
    value_fmt, axis_bounds, axis_tick_format,
    always_include_zero_in_axis,
    layout: BarLayoutParams | None = None,
    shared_color_map: dict[str, tuple] | None = None,
    skip_data_with_only_zeroes: bool = False,
) -> plt.Figure | None:
    """Build a bar-chart Figure and return it (without saving or closing)."""
    # Calculate subplot grid
    n_subs = len(effective_plots)
    n_rows, n_cols = _calculate_grid_layout(n_subs, subplots_per_row)

    # When value labels are on, thicken bars and grow each slot by the same
    # factor so the labels (which sit at bar centers) gain vertical
    # separation. thickness_mult is captured by the closures below and
    # passed to the draw functions, so the estimation and render passes stay
    # consistent. value_fmt is in scope here (it is also used to draw the
    # labels further down).
    thickness_mult = VALUE_LABEL_BAR_THICKNESS_MULT if value_fmt else 1.0

    # Per-position bar height: when grouped bars exceed 3, scale up the slot
    # so each individual bar stays at least 1/3 of font height (~10pt).
    # Slot height for a label row containing n bars at the FIXED bar
    # thickness (REFERENCE_BAR_THICKNESS for n>1, SOLO_BAR_THICKNESS for
    # n==1). Slot stays at the BAR_HEIGHT baseline whenever the bars +
    # gaps fit; grows once they don't.
    def _slot_height_for_n_grouped(n: int) -> float:
        """Bar slot height for n grouped bars at one label."""
        if n <= 0:
            return BAR_HEIGHT
        bar_t = (SOLO_BAR_THICKNESS if n == 1 else REFERENCE_BAR_THICKNESS) * thickness_mult
        needed = n * bar_t + max(0, n - 1) * BAR_GAP_FRACTION * bar_t
        return max(BAR_HEIGHT, needed)

    def _count_grouped(df_check: pd.DataFrame) -> int:
        """Count grouped bar members in a DataFrame."""
        if not grouped_bar_levels:
            return 1
        if isinstance(df_check.columns, pd.MultiIndex) and grouped_bar_level_names:
            if len(grouped_bar_level_names) == 1:
                return len(df_check.columns.get_level_values(
                    grouped_bar_level_names[0]).unique())
            gf = df_check.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
            return len(gf)
        elif not isinstance(df_check.columns, pd.MultiIndex) and grouped_bar_levels:
            return len(df_check.columns.unique())
        return 1

    # Add extra bottom space when xlabel is present
    bottom_pad = BOTTOM_PAD + (XLABEL_HEIGHT if xlabel else 0)
    left_edge_pad = LEFT_PAD + (YLABEL_WIDTH if ylabel else 0)

    # ── Height estimation pass ────────────────────────────────────
    # Compute accurate bars_only_h per subplot (with pruning) so the
    # figure is sized correctly. Pruning is cheap (column/row masks)
    # and will be re-done in the render pass — no data is stored.
    def _estimate_subplot_height(df_sub_raw: pd.DataFrame) -> float:
        """Return bars_only_h for one subplot, applying pruning."""
        df_s = df_sub_raw
        if skip_data_with_only_zeroes:
            df_s = df_s.loc[:, (df_s.abs() > 1e-6).any(axis=0)]
            df_s = df_s.loc[(df_s.abs() > 1e-6).any(axis=1)]
        if df_s.empty:
            return BAR_HEIGHT  # minimum

        def _sum_row_heights(df_data: pd.DataFrame, n_global: int) -> float:
            """Sum per-row slot heights. Must match the render pass exactly.

            * Simple bars (no ``grouped_bar_levels``): every kept (non-empty)
              row gets ``_slot_height_for_n_grouped(1)``; fully-empty rows
              (under pruning) contribute nothing (they are not drawn).
            * Grouped bars: every non-empty row reserves a slot for the
              GLOBAL grouped count ``n_global`` (the same count
              ``_plot_grouped_bars`` uses to position bars), so the slot is
              consistent with what is drawn. Fully-empty rows fall back to
              ``BAR_HEIGHT`` only if they are counted as rows; under pruning
              they are removed and contribute nothing.
            """
            if df_data.empty:
                return 0.0

            # Boolean mask: True where |v| > 1e-6 (row has any non-zero).
            row_has_data = (df_data.abs() > 1e-6).any(axis=1)

            # --- Case A: simple bars ---------------------------------------
            if not grouped_bar_levels:
                if skip_data_with_only_zeroes:
                    n_non_zero_rows = int(row_has_data.sum())
                else:
                    n_non_zero_rows = len(df_data.index)
                return n_non_zero_rows * _slot_height_for_n_grouped(1)

            # --- Case B: grouped bars --------------------------------------
            # Every non-empty row reserves the GLOBAL-count slot; this is the
            # render pass's behaviour (_row_height returns the global slot for
            # any row with at least one non-zero bar).
            slot_global = _slot_height_for_n_grouped(n_global)
            if skip_data_with_only_zeroes:
                n_non_empty = int(row_has_data.sum())
                return n_non_empty * slot_global
            return len(df_data.index) * slot_global

        # GLOBAL grouped count for this subplot — the same count
        # _plot_grouped_bars uses (len(grouped_bars) over the full subplot
        # columns). Every non-empty row reserves a slot for this count, so the
        # estimation total matches the render pass's per-row slot sum.
        n_global = _count_grouped(df_s) if grouped_bar_levels else 1

        if not expand_axis_levels:
            return max(_sum_row_heights(df_s, n_global), BAR_HEIGHT)

        # Per expand group: extract slice, sum per-row heights
        total_h = 0.0
        if len(expand_axis_level_names) == 1:
            groups_est = df_s.columns.get_level_values(expand_axis_level_names[0]).unique()
        elif isinstance(df_s.columns, pd.MultiIndex):
            gf = df_s.columns.to_frame()[expand_axis_level_names].drop_duplicates()
            groups_est = [tuple(r) for r in gf.values]
        else:
            groups_est = [df_s.columns[0]] if len(df_s.columns) else []

        for grp in groups_est:
            try:
                if not isinstance(df_s.columns, pd.MultiIndex):
                    df_g = df_s[[grp]] if grp in df_s.columns else None
                elif len(expand_axis_level_names) == 1:
                    df_g = df_s.xs(grp, level=expand_axis_level_names[0], axis=1)
                else:
                    df_g = df_s.xs(grp, level=expand_axis_level_names, axis=1)
            except KeyError:
                continue
            if df_g is None:
                continue
            if isinstance(df_g, pd.Series):
                df_g = df_g.to_frame()
            total_h += _sum_row_heights(df_g, n_global)
        return max(total_h, BAR_HEIGHT)

    subplot_sizes: list[float] = [
        _estimate_subplot_height(df_sub_raw) + SUBPLOT_VPAD
        for _, df_sub_raw in effective_plots
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

            fig = Figure(figsize=(total_width, total_height))

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
                layout.value_axis_width + (ss - SUBPLOT_VPAD) + RIGHT_PAD
                for ss in subplot_sizes
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

            fig = Figure(figsize=(total_width, total_height))

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
        # Prune zero data close to plotting — take a copy so we don't
        # modify the shared effective_plots data.
        if skip_data_with_only_zeroes:
            df_sub = df_sub.copy()
            # Drop all-zero columns (removes empty grouped bars)
            df_sub = df_sub.loc[:, (df_sub.abs() > 1e-6).any(axis=0)]
            # Drop all-zero rows (removes empty bar positions)
            df_sub = df_sub.loc[(df_sub.abs() > 1e-6).any(axis=1)]

        # Get unique group combinations from df_sub
        if not expand_axis_levels:
            groups = [None]
        elif len(expand_axis_level_names) == 1:
            groups = df_sub.columns.get_level_values(expand_axis_level_names[0]).unique().tolist()
        else:
            # Deduplicate multi-level expand groups
            group_frame = df_sub.columns.to_frame()[expand_axis_level_names].drop_duplicates()
            groups = [tuple(row) for row in group_frame.values]

        # Reverse ordering for horizontal bars only.
        #
        # In matplotlib horizontal bars, y=0 sits at the bottom of the axes,
        # so emitting bars in reversed data order places the first data item
        # at the top of the plot (natural top-to-bottom reading order). For
        # vertical bars, x=0 sits at the left, so we want the natural data
        # order (first period on the left → left-to-right reading order).
        reverse_order = bar_orientation == 'horizontal'

        # Reverse groups order (horizontal only — see comment above)
        if expand_axis_levels:
            if reverse_order:
                groups = groups[::-1]

        # Get bar labels from this subplot's index (not the global df).
        # Each index component is middle-truncated (full text shown on hover).
        subplot_bar_labels = [_format_bar_label(x) for x in df_sub.index]
        subplot_bar_full_labels = [
            _format_bar_label(x, truncate=False) for x in df_sub.index
        ]
        # Reverse to match the reversed groups order (horizontal only)
        if reverse_order:
            subplot_bar_labels = subplot_bar_labels[::-1]
            subplot_bar_full_labels = subplot_bar_full_labels[::-1]

        # Build list of all bars (for y-axis positioning).
        # Zero pruning already done on df_sub above (if skip_data_with_only_zeroes).
        index_order = df_sub.index[::-1] if reverse_order else df_sub.index
        all_bars = []
        if not expand_axis_levels:
            for idx_val in index_order:
                all_bars.append([None, idx_val])
        else:
            groups_with_bars: list[tuple] = []  # (group, [row_items])
            for group in groups:
                try:
                    if not isinstance(df_sub.columns, pd.MultiIndex):
                        if group in df_sub.columns:
                            df_group = df_sub[[group]]
                        else:
                            continue
                    elif len(expand_axis_level_names) == 1:
                        df_group = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
                    else:
                        df_group = df_sub.xs(group, level=expand_axis_level_names, axis=1)
                except KeyError:
                    continue
                if isinstance(df_group, pd.Series):
                    df_group = df_group.to_frame()
                # Per-group row filtering
                if skip_data_with_only_zeroes:
                    has_data = (df_group.abs() > 1e-6).any(axis=1)
                    row_items = [v for v in index_order
                                 if v in has_data.index and has_data.loc[v]]
                else:
                    row_items = [v for v in index_order
                                 if v in df_group.index]
                if row_items:
                    groups_with_bars.append((group, row_items))
                    for idx_val in row_items:
                        all_bars.append([group, idx_val])

        # Skip subplot if no bars have data (avoids zero-height axes)
        if not all_bars:
            if n_subs == 1:
                return None
            if n_subs > 1 and axes[idx] is not None:
                axes[idx].set_visible(False)
            continue

        # Per-row slot heights. _plot_grouped_bars positions every row's bar
        # group using a SINGLE total_w built from the GLOBAL grouped count
        # (len(grouped_bars) over the full subplot), applied uniformly to all
        # rows. So every row that draws any bar must reserve a slot sized for
        # that same global count — otherwise a row with fewer non-zero
        # scenarios gets a slot smaller than the group it draws and the bars
        # overflow into adjacent rows. We therefore size every non-empty row
        # to _slot_height_for_n_grouped(GLOBAL_n_grouped); only fully-empty
        # rows (no non-zero bar) fall back to BAR_HEIGHT. GLOBAL_n is constant
        # per subplot and equals the len(grouped_bars) the drawing uses.
        global_n_grouped = _count_grouped(df_sub) if grouped_bar_levels else 1

        def _row_height(row_data: pd.DataFrame) -> float:
            if grouped_bar_levels:
                # Detect a fully-empty row only to fall back to BAR_HEIGHT;
                # the non-empty slot uses the GLOBAL group count to match
                # what _plot_grouped_bars draws.
                if skip_data_with_only_zeroes:
                    pruned = row_data.loc[:, (row_data.abs() > 1e-6).any(axis=0)]
                    if pruned.empty:
                        return BAR_HEIGHT
                elif row_data.empty:
                    return BAR_HEIGHT
                return _slot_height_for_n_grouped(global_n_grouped)
            # Simple bars (no grouping) — unchanged.
            if skip_data_with_only_zeroes:
                row_data = row_data.loc[:, (row_data.abs() > 1e-6).any(axis=0)]
            if row_data.empty:
                return BAR_HEIGHT
            return _slot_height_for_n_grouped(1)

        if expand_axis_levels and groups_with_bars:
            per_bar_heights: list[float] = []
            for group, row_items in groups_with_bars:
                try:
                    if not isinstance(df_sub.columns, pd.MultiIndex):
                        df_g = df_sub[[group]] if group in df_sub.columns else df_sub
                    elif len(expand_axis_level_names) == 1:
                        df_g = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
                    else:
                        df_g = df_sub.xs(group, level=expand_axis_level_names, axis=1)
                except KeyError:
                    df_g = df_sub
                if isinstance(df_g, pd.Series):
                    df_g = df_g.to_frame()
                for ri in row_items:
                    per_bar_heights.append(_row_height(df_g.loc[[ri]]))
        else:
            per_bar_heights = []
            for _, ri in all_bars:
                per_bar_heights.append(_row_height(df_sub.loc[[ri]]))

        # Cumulative y-positions (center of each slot)
        y_positions: list[float] = []
        cumulative = 0.0
        for h in per_bar_heights:
            y_positions.append(cumulative + h / 2)
            cumulative += h
        bars_only_h = cumulative

        # Create figure for single plot (now that we know bar count)
        if n_subs == 1 and fig is None:
            subplot_h = bars_only_h + SUBPLOT_VPAD
            if bar_orientation == 'horizontal':
                fig_w = _single_width
                fig_h = subplot_h + TITLE_PAD + bottom_pad
                legend_excess = 0.0
                if layout and layout.legend_height > 0:
                    legend_excess = max(0.0, layout.legend_height - bars_only_h)
                    fig_h += legend_excess
                fig = Figure(figsize=(fig_w, fig_h))
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
                fig = Figure(figsize=(fig_w, fig_h))
                ax = fig.add_axes([
                    layout.value_axis_width / fig_w,
                    (bottom_pad + layout.total_label_width + legend_excess) / fig_h,
                    bars_only_h / fig_w,
                    layout.base_bar_length / fig_h,
                ])
            axes[0] = ax
        else:
            ax = axes[idx]

        # When axis_bounds explicitly fixes the value axis, the post-draw
        # finalisation below will override matplotlib's autoscale with
        # those bounds. Passing them in lets the bar-plot helpers compute
        # the sub-pixel threshold against the FINAL on-screen range
        # rather than the data range — important when axis_bounds is
        # tighter than the data (otherwise the helper would compute too
        # large a threshold and drop bars that the final view would show).
        # When no axis_bounds is set, the helper auto-computes from data;
        # downstream zero-inclusion and value-label margin only enlarge
        # that range, so the helper's threshold is a conservative under-
        # estimate (we keep some bars that turn out invisible — safe).
        explicit_bounds = _subplot_axis_bounds(axis_bounds, idx)
        if explicit_bounds and explicit_bounds[0] != explicit_bounds[1]:
            value_axis_lim = (float(explicit_bounds[0]), float(explicit_bounds[1]))
        else:
            value_axis_lim = None

        # Determine plotting mode and execute appropriate logic
        if grouped_bar_levels:
            # Single batched call per subplot: _plot_grouped_bars vectorises
            # across all bars per category internally (one ax.barh per
            # grouped category, with masked-out zero entries reproducing the
            # original per-row pruning behaviour).
            labeled_groups: set[str] = set()
            _plot_grouped_bars(ax, df_sub, all_bars, expand_axis_level_names,
                               grouped_bar_level_names, bar_orientation, value_fmt,
                               shared_color_map=shared_color_map,
                               y_positions=y_positions,
                               slot_heights=per_bar_heights,
                               labeled_groups=labeled_groups,
                               value_axis_lim=value_axis_lim,
                               thickness_mult=thickness_mult)
        elif stack_levels:
            _plot_stacked_bars(ax, df_sub, all_bars, expand_axis_level_names,
                               stack_level_names, bar_orientation,
                               shared_color_map=shared_color_map,
                               y_positions=y_positions, slot_heights=per_bar_heights,
                               value_axis_lim=value_axis_lim)
        else:
            _plot_simple_bars(ax, df_sub, all_bars, expand_axis_level_names,
                              bar_orientation, value_fmt,
                              y_positions=y_positions, slot_heights=per_bar_heights,
                              value_axis_lim=value_axis_lim,
                              thickness_mult=thickness_mult)

        # Set up axis with groups and bars
        # Build bar labels for display (matching all_bars structure). The
        # expand-group level is folded into each tick label as "bar | group"
        # (group last, like the non-expand "extended" pathway) instead of a
        # separate label to the left of the axis, which collided with long
        # bar labels and inflated the left margin. Full (untruncated) labels
        # are stashed on the axes for the GUI hover tooltip.
        if not expand_axis_levels:
            display_bar_labels = subplot_bar_labels
            full_bar_labels = subplot_bar_full_labels
        else:
            display_bar_labels = [
                _format_bar_label(idx_val, group) for group, idx_val in all_bars
            ]
            full_bar_labels = [
                _format_bar_label(idx_val, group, truncate=False)
                for group, idx_val in all_bars
            ]
        # Expose label data to the embedding GUI (truncated → full) so a
        # hover tooltip can reveal names cut by BAR_LABEL_MAX_CHARS.
        ax._flextool_bar_labels = list(display_bar_labels)
        ax._flextool_bar_labels_full = list(full_bar_labels)
        ax._flextool_bar_positions = list(y_positions)
        ax._flextool_bar_orientation = bar_orientation

        # Set main axis for individual bars (use cumulative y_positions).
        # Use FixedLocator/FixedFormatter rather than ax.set_yticks(...) so
        # that matplotlib defers Tick object construction until render time
        # (and then only for ticks inside the current view limits). The
        # set_yticks/set_xticks pair otherwise instantiates one Tick per
        # position up-front, which dominates build time when the bar count
        # is in the thousands.
        if bar_orientation == 'horizontal':
            ax.yaxis.set_major_locator(FixedLocator(y_positions))
            ax.yaxis.set_major_formatter(FixedFormatter(display_bar_labels))
            ax.tick_params('y', length=0)
            ax.set_ylim(0, bars_only_h)
            ax.tick_params(labelsize=10)
        else:  # vertical
            ax.xaxis.set_major_locator(FixedLocator(y_positions))
            ax.xaxis.set_major_formatter(FixedFormatter(display_bar_labels))
            ax.tick_params('x', length=0)
            ax.set_xlim(0, bars_only_h)
            ax.tick_params(labelsize=10)
            plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

        if expand_axis_levels and len(groups_with_bars) > 1:
            # The expand-group name is folded into each tick label (above), so
            # we no longer draw a separate group label to the left of the
            # axis (it collided with long bar labels). Keep only thin
            # separators between groups to keep the blocks visually distinct.
            bar_boundaries = [y_positions[0] - per_bar_heights[0] / 2]
            for i in range(len(y_positions)):
                bar_boundaries.append(y_positions[i] + per_bar_heights[i] / 2)

            group_boundaries = []
            pos_idx = 0
            for _, row_items in groups_with_bars:
                group_boundaries.append(bar_boundaries[pos_idx])
                pos_idx += len(row_items)
            group_boundaries.append(bar_boundaries[-1])

            # Interior separators only (skip the two outer edges).
            for boundary in group_boundaries[1:-1]:
                if bar_orientation == 'horizontal':
                    ax.axhline(y=boundary, color="grey", linewidth=0.8, linestyle="-")
                else:
                    ax.axvline(x=boundary, color="grey", linewidth=0.8, linestyle="-")

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
                if bar_orientation == 'horizontal':
                    axes_width = layout.base_bar_length
                else:
                    axes_width = bars_only_h
                legend_x = 1 + LEGEND_GAP / axes_width
                # _plot_stacked_bars and _plot_grouped_bars build legend entries
                # in the correct visual order — no reversal needed here.
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
        if scale and scale[0] != scale[1]:
            if bar_orientation == 'horizontal':
                ax.set_xlim(scale[0], scale[1])
            else:
                ax.set_ylim(scale[0], scale[1])
        _fmt = _get_value_formatter(axis_tick_format, idx)
        # Prune the outermost tick (away from zero) to prevent label overflow.
        if bar_orientation == 'horizontal':
            lo, hi = ax.get_xlim()
            # Prune the end farthest from zero
            if abs(hi) >= abs(lo):
                prune = 'upper'
            else:
                prune = 'lower'
            nbins = _estimate_value_nbins(lo, hi, layout.base_bar_length, _fmt, is_horizontal_axis=True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=prune))
            ax.xaxis.set_major_formatter(_fmt)
        else:
            lo, hi = ax.get_ylim()
            if abs(hi) >= abs(lo):
                prune = 'upper'
            else:
                prune = 'lower'
            nbins = _estimate_value_nbins(lo, hi, layout.base_bar_length, _fmt, is_horizontal_axis=False)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins, prune=prune))
            ax.yaxis.set_major_formatter(_fmt)
        # Extend axis limits AFTER tick locator to create room for value labels
        # within the plot area. The vertical branch keeps the simple 13% bump
        # (label overflow there is one text-line tall — negligible). The
        # horizontal branch reserves room sized to the *actual* value-label
        # text width in inches, because a fixed % of the data range under-
        # reserves for large numbers (the label width is ~constant in inches,
        # not proportional to the data range) and the labels spill into the
        # legend. It never shrinks below the old 13%.
        if value_fmt:
            if bar_orientation == 'horizontal':
                data_min, data_max = ax.get_xlim()
                # The label sits at the data extreme; reserve its text width.
                neg_anchor = min(data_min, 0)

                def _value_label_text(v):
                    if value_fmt == 'dynamic':
                        return format_value_label(v)
                    return format(v, value_fmt)

                # Per-char width slightly generous vs the font-9-calibrated
                # CHAR_WIDTH (=0.081"): bar_label font is ~10pt and CHAR_WIDTH
                # under-estimates it. Over-reserving a little is far better
                # than clipping into the legend. Add the 3pt bar_label padding.
                _label_char_w = 0.095          # inches per char at the label font
                _label_pad_in = 3 / 72         # bar_label padding=3pt
                _axw = layout.base_bar_length

                xmin, xmax = data_min, data_max
                if _axw > 0:
                    if data_max > 0:
                        label_w_in = len(_value_label_text(data_max)) * _label_char_w + _label_pad_in
                        frac = min(0.9, label_w_in / _axw)
                        new_xmax = (data_max - label_w_in * neg_anchor / _axw) / (1 - frac)
                        xmax = max(data_max * 1.13, new_xmax)
                    if data_min < 0:
                        label_w_neg = len(_value_label_text(data_min)) * _label_char_w + _label_pad_in
                        frac_neg = min(0.9, label_w_neg / _axw)
                        # Mirror of the positive solve: reserve label_w_neg to
                        # the left of data_min so the label fits in the axes.
                        pos_anchor = max(data_max, 0)
                        new_xmin = (data_min - label_w_neg * pos_anchor / _axw) / (1 - frac_neg)
                        xmin = min(data_min * 1.13, new_xmin)
                else:
                    if data_max > 0:
                        xmax = data_max * 1.13
                    if data_min < 0:
                        xmin = data_min * 1.13
                ax.set_xlim(xmin, xmax)
            else:
                ymin, ymax = ax.get_ylim()
                if ymax > 0:
                    ymax *= 1.13
                if ymin < 0:
                    ymin *= 1.13
                ax.set_ylim(ymin, ymax)
        # Y-axis label positioning. For HORIZONTAL bars the ylabel is the
        # category axis (left of the tick labels): pin it explicitly to the
        # left of the (generously estimated) tick-label region — and, when
        # expand-axis group labels occupy space to the left, to the left of
        # those too. Explicit set_label_coords avoids matplotlib's
        # environment-dependent auto-positioning. For VERTICAL bars the ylabel
        # is the value axis; keep the existing labelpad behaviour unchanged.
        if bar_orientation == 'horizontal':
            group_w = layout.group_label_width if expand_axis_levels else 0.0
            # Inches reserved between figure-left and the axes' left spine
            # (tick labels + ylabel reservation), minus LEFT_PAD which we keep
            # as figure-edge breathing room. Used to clamp the ylabel so its
            # text never crosses the figure's left edge for very wide category
            # labels.
            left_margin_in = layout.total_label_width + left_edge_pad - LEFT_PAD
            ylabel_axes_x = _ylabel_axes_x(
                layout.bar_label_width, group_w, layout.base_bar_length,
                left_margin_in=left_margin_in,
            )
            _apply_subplot_label(ax, xlabel, ylabel, idx, row, col, n_rows,
                                 ylabel_axes_x=ylabel_axes_x)
        else:
            if expand_axis_levels:
                expand_pad = (layout.group_label_width + 0.15) * 72
            else:
                expand_pad = 18
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
            axes_width = bars_only_h if bars_only_h > 0 else 1.0
        legend_x = 1 + LEGEND_GAP / axes_width
        ax_legend.legend(handles, labels_all, title=legend_title,
                         bbox_to_anchor=(legend_x, 1), loc='upper left', borderaxespad=0)

    # Hide unused subplots
    for idx in range(n_subs, len(axes)):
        axes[idx].set_visible(False)

    # Figure title at a fixed distance from the top (10px ≈ 0.14in at 72 dpi)
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(key_name, y=1 - 0.14 / fig_h, va='top')

    return fig


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
    skip_data_with_only_zeroes: bool = False,
):
    """Render one file's worth of bar subplots and save to disk."""
    fig = _build_bar_figure(
        effective_plots, df, key_name, plot_dir,
        stack_levels, stack_level_names,
        expand_axis_levels, expand_axis_level_names,
        sub_levels,
        grouped_bar_levels, grouped_bar_level_names,
        legend_position, subplots_per_row,
        xlabel, ylabel, bar_orientation, base_bar_length,
        value_fmt, axis_bounds, axis_tick_format,
        always_include_zero_in_axis,
        layout, shared_color_map, skip_data_with_only_zeroes,
    )
    if fig is None:
        return
    filepath = output_filepath or f'{plot_dir}/{key_name}_d.png'
    fig.savefig(filepath)
    plt.close(fig)


def build_bar_figures(
    df: pd.DataFrame,
    key_name: str,
    plot_dir: str,
    stack_levels: list[int],
    expand_axis_levels: list[int],
    sub_levels: list[int] | None = None,
    grouped_bar_levels: list[int] | None = None,
    legend_position: str = 'right',
    subplots_per_row: int = 2,
    xlabel: str | None = None,
    ylabel: str | None = None,
    bar_orientation: str = 'horizontal',
    base_bar_length: float = 4,
    value_label=False,
    axis_bounds=None,
    axis_tick_format='1,.0f',
    always_include_zero_in_axis: bool = True,
    max_items_per_plot: int = 10,
    max_subplots_per_file: int = 6,
    max_items_per_subplot_column: int = 40,
    only_first_file: bool = False,
    skip_data_with_only_zeroes: bool = False,
    only_file_index: int | None = None,
    *,
    color_template: dict | None = None,
    category: str | None = None,
    entity_class: str | None = None,
) -> tuple[list[tuple[str, 'plt.Figure']], int]:
    """Build bar-chart Figures and return them without saving or closing.

    Returns (figures, total_file_count) where figures is a list of
    (batch_title, Figure) pairs -- one per file batch.
    Figures where all data is empty are omitted (not returned as None).
    When only_file_index is set, only that batch is built.
    """
    if sub_levels is None:
        sub_levels = []

    # Validate mutual exclusivity
    if stack_levels and grouped_bar_levels:
        raise ValueError(
            "Cannot use both 'stack_levels' and 'grouped_bar_levels' simultaneously."
        )

    # Resolve value_label
    if value_label is True or value_label == 'true':
        value_fmt = 'dynamic'
    elif value_label:
        value_fmt = str(value_label)
    else:
        value_fmt = None

    if stack_levels is None:
        stack_levels = []
    if grouped_bar_levels is None:
        grouped_bar_levels = []

    # Convert level indices to names
    if isinstance(df.columns, pd.MultiIndex):
        stack_level_names = [df.columns.names[i] for i in stack_levels] if stack_levels else []
        expand_axis_level_names = [df.columns.names[i] for i in expand_axis_levels] if expand_axis_levels else []
        grouped_bar_level_names = [df.columns.names[i] for i in grouped_bar_levels] if grouped_bar_levels else []
    else:
        stack_level_names = stack_levels
        expand_axis_level_names = [df.columns.name] if expand_axis_levels else []
        grouped_bar_level_names = [df.columns.name] if grouped_bar_levels else []

    subs = _sort_subs(_get_unique_levels(df.columns, sub_levels))

    # Compute expand-group count. Mirrors the fix in ``_compute_bar_plan``:
    # single-level columns also fan out into one bar per unique value at
    # draw time, so we count those as expand groups too. Without this the
    # pagination undercounts and a single figure ends up gigapixel-sized.
    if expand_axis_levels:
        if isinstance(df.columns, pd.MultiIndex):
            if len(expand_axis_level_names) == 1:
                expand_level_name = expand_axis_level_names[0]
                n_expand_groups = len(df.columns.get_level_values(expand_level_name).unique())
            else:
                expand_level_name = expand_axis_level_names[0]
                expand_frame = df.columns.to_frame()[expand_axis_level_names].drop_duplicates()
                n_expand_groups = len(expand_frame)
        else:
            expand_level_name = df.columns.name
            n_expand_groups = len(df.columns.unique())
    else:
        expand_level_name = None
        n_expand_groups = 1

    # Build effective_plots — split subplots that exceed max_items_per_plot.
    # The "items" are visual bar-label rows: n_rows * n_expand_groups.
    # Splitting can happen by expand groups, by rows, or both.
    effective_plots: list[tuple[str | None, pd.DataFrame]] = []
    for sub in subs:
        df_sub = _extract_subplot_data(df, sub, sub_levels)
        df_sub = df_sub.dropna(how='all')
        if df_sub.empty:
            continue
        df_sub = df_sub.fillna(0)
        title = (
            ' | '.join(str(v) for v in sub) if isinstance(sub, tuple)
            else str(sub) if sub is not None else None
        )
        n_rows = len(df_sub)

        if not max_items_per_plot:
            effective_plots.append((title, df_sub))
            continue

        # Total visual items = n_rows * n_expand_groups
        total_items = n_rows * max(n_expand_groups, 1)
        if total_items <= max_items_per_plot:
            effective_plots.append((title, df_sub))
            continue

        # Split by expand groups first (if present), then by rows within each group chunk
        if expand_level_name is not None and n_expand_groups > 1:
            max_groups = max(1, max_items_per_plot // max(n_rows, 1))
            if isinstance(df_sub.columns, pd.MultiIndex):
                all_groups = df_sub.columns.get_level_values(expand_level_name).unique().tolist()
            else:
                all_groups = df_sub.columns.unique().tolist()
            for gi, grp_start in enumerate(range(0, len(all_groups), max_groups)):
                grp_chunk = all_groups[grp_start:grp_start + max_groups]
                if isinstance(df_sub.columns, pd.MultiIndex):
                    mask = df_sub.columns.get_level_values(expand_level_name).isin(grp_chunk)
                else:
                    mask = df_sub.columns.isin(grp_chunk)
                chunk = df_sub.loc[:, mask]
                if n_rows > max_items_per_plot:
                    n_row_chunks = (n_rows + max_items_per_plot - 1) // max_items_per_plot
                    for ri in range(n_row_chunks):
                        row_chunk = chunk.iloc[ri * max_items_per_plot:(ri + 1) * max_items_per_plot]
                        sub_label = (
                            f"{title}_{gi + 1}.{ri + 1}"
                            if title else f"{gi + 1}.{ri + 1}"
                        )
                        effective_plots.append((sub_label, row_chunk))
                else:
                    chunk_label = f"{title}_{gi + 1}" if title else None
                    effective_plots.append((chunk_label, chunk))
        elif n_rows > max_items_per_plot:
            # No expand groups — split by rows
            for i in range(0, n_rows, max_items_per_plot):
                chunk = df_sub.iloc[i:i + max_items_per_plot]
                chunk_label = f"{title}_{i // max_items_per_plot + 1}" if title else None
                effective_plots.append((chunk_label, chunk))
        else:
            effective_plots.append((title, df_sub))

    if not effective_plots:
        return [], 0

    # Build shared color map
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
        shared_color_map = build_shared_color_map(
            all_labels,
            color_template=color_template,
            category=category,
            entity_class=entity_class,
        )

    # Compute layout
    layout = _compute_bar_layout(
        effective_plots, df,
        expand_axis_levels, expand_axis_level_names,
        stack_levels, stack_level_names,
        grouped_bar_levels, grouped_bar_level_names,
        legend_position, subplots_per_row,
        base_bar_length,
    )

    # Split into file batches respecting max_subplots_per_file and, for
    # horizontal bars, max_items_per_subplot_column.
    spr = max(subplots_per_row, 1)
    _max = max_subplots_per_file if max_subplots_per_file else len(effective_plots)
    col_limit = (
        max_items_per_subplot_column
        if bar_orientation == 'horizontal' else 0
    )

    # Compute the effective visual item count per subplot.  When expand
    # groups are present, each group contributes its own set of bar labels,
    # so the count is much larger than len(df_sub).
    def _count_visual_items(df_sub: pd.DataFrame) -> int:
        if not expand_axis_level_names:
            return max(len(df_sub), 1)
        if not isinstance(df_sub.columns, pd.MultiIndex):
            return max(len(df_sub) * len(df_sub.columns.unique()), 1)
        count = 0
        if len(expand_axis_level_names) == 1:
            groups = df_sub.columns.get_level_values(expand_axis_level_names[0]).unique()
        else:
            gf = df_sub.columns.to_frame()[expand_axis_level_names].drop_duplicates()
            groups = [tuple(r) for r in gf.values]
        for grp in groups:
            try:
                if len(expand_axis_level_names) == 1:
                    df_g = df_sub.xs(grp, level=expand_axis_level_names[0], axis=1)
                else:
                    df_g = df_sub.xs(grp, level=expand_axis_level_names, axis=1)
            except KeyError:
                continue
            if isinstance(df_g, pd.Series):
                df_g = df_g.to_frame()
            count += len(df_g)
        return max(count, 1)

    visual_item_counts = [_count_visual_items(df_sub) for _, df_sub in effective_plots]

    # Group subplots into grid rows so we never break mid-row.
    grid_rows: list[list[int]] = []
    for gi in range(0, len(effective_plots), spr):
        grid_rows.append(list(range(gi, min(gi + spr, len(effective_plots)))))

    _file_batches: list[tuple[list, None]] = []
    cur: list = []
    col_counts = [0] * spr

    for row_indices in grid_rows:
        would_exceed_subplots = len(cur) + len(row_indices) > _max
        would_exceed_col = False
        if col_limit and cur:
            for j, idx in enumerate(row_indices):
                if col_counts[j] + visual_item_counts[idx] > col_limit:
                    would_exceed_col = True
                    break

        if (would_exceed_subplots or would_exceed_col) and cur:
            _file_batches.append((cur, None))
            cur = []
            col_counts = [0] * spr

        cur.extend([effective_plots[i] for i in row_indices])
        for j, idx in enumerate(row_indices):
            col_counts[j] += visual_item_counts[idx]

    if cur:
        _file_batches.append((cur, None))

    total_file_count = len(_file_batches)
    batches_to_build = _file_batches[:1] if only_first_file else _file_batches
    n_total_batches = len(_file_batches)
    result: list[tuple[str, plt.Figure]] = []
    for batch_idx, (batch, _) in enumerate(batches_to_build, start=1):
        if only_file_index is not None and (batch_idx - 1) != only_file_index:
            continue
        batch_title = f"{key_name} ({batch_idx}/{n_total_batches})" if n_total_batches > 1 else key_name
        fig = _build_bar_figure(
            batch, df, batch_title, plot_dir,
            stack_levels, stack_level_names,
            expand_axis_levels, expand_axis_level_names,
            sub_levels,
            grouped_bar_levels, grouped_bar_level_names,
            legend_position, subplots_per_row,
            xlabel, ylabel, bar_orientation, base_bar_length,
            value_fmt, axis_bounds, axis_tick_format,
            always_include_zero_in_axis,
            layout, shared_color_map, skip_data_with_only_zeroes,
        )
        if fig is not None:
            result.append((batch_title, fig))
    return result, total_file_count
