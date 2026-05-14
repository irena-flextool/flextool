"""
Bar chart rendering functions for each bar mode.

Called by plot_bars.py (and currently plot_functions.py during transition).
Each function receives an axes object and renders one subplot's bars.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flextool.plot_outputs.format_helpers import DynamicFormatter

# Bar geometry constants (kept in sync with plot_bars.py module-level
# values). Duplicated here to avoid a circular import — plot_bars.py
# imports the rendering functions from this module.
BAR_GAP_FRACTION = 0.10
REFERENCE_BAR_THICKNESS = 0.0558    # matches plot_bars.REFERENCE_BAR_THICKNESS
SOLO_BAR_THICKNESS = 0.1116         # = 2 × REFERENCE_BAR_THICKNESS


def _stack_label(stack_idx: int, stacks: list, labeled_stacks: set) -> str:
    """Return the legend label for a stack segment (empty string if already labeled)."""
    if stack_idx not in labeled_stacks:
        stack_value = stacks[stack_idx]
        label = (
            ' | '.join(str(v) for v in stack_value)
            if isinstance(stack_value, (tuple, list))
            else str(stack_value)
        )
        labeled_stacks.add(stack_idx)
    else:
        label = ''
    return label


def _plot_grouped_bars(
    ax,
    df_sub: pd.DataFrame,
    all_bars: list,
    expand_axis_level_names: list,
    grouped_bar_level_names: list,
    bar_orientation: str,
    value_fmt: str | None,
    shared_color_map: dict[str, tuple] | None = None,
    y_positions: list[float] | None = None,
    slot_heights: list[float] | None = None,
    labeled_groups: set[str] | None = None,
) -> None:
    """Render grouped side-by-side bars onto ax for one subplot.

    Vectorised across all_bars: one ax.barh / ax.bar call per grouped
    category. Each call takes a vector of sub-y-positions (one per bar) and
    a vector of widths; bars with zero value in this category are masked
    out before the draw call, mirroring the original per-bar skip behaviour
    when ``skip_data_with_only_zeroes`` had already pruned them at the
    call site.
    """
    if not all_bars:
        return

    # Get grouped bar combinations
    if len(grouped_bar_level_names) == 1:
        grouped_bars = df_sub.columns.get_level_values(grouped_bar_level_names[0]).unique().tolist()
    else:
        grouped_bar_df = df_sub.columns.to_frame()[grouped_bar_level_names].drop_duplicates()
        grouped_bars = [tuple(row) for row in grouped_bar_df.values]

    # Sort groups alphabetically when using shared colors so visual order matches legend
    if shared_color_map:
        def _group_key(gb):
            return ' | '.join(str(v) for v in gb) if isinstance(gb, (tuple, list)) else str(gb)
        grouped_bars.sort(key=_group_key)

    # Colors for grouped bars
    n_grouped = len(grouped_bars)
    if shared_color_map:
        colors = []
        for gb in grouped_bars:
            label = ' | '.join(str(v) for v in gb) if isinstance(gb, (tuple, list)) else str(gb)
            colors.append(shared_color_map.get(label, (0.5, 0.5, 0.5)))
    else:
        colors = list(plt.colormaps['tab10'].colors[:n_grouped])
        if n_grouped > 10:
            colors = list(plt.colormaps['tab20'].colors[:n_grouped])

    # Build (n_bars, n_grouped) value matrix via cached per-group slices.
    n_bars = len(all_bars)
    values_mat = np.zeros((n_bars, n_grouped), dtype=float)
    has_value = np.zeros((n_bars, n_grouped), dtype=bool)

    def _hashable_key(g):
        if isinstance(g, list):
            return tuple(g)
        return g

    group_df_cache: dict = {}
    group_cat_series: dict = {}

    def _get_cat_series(df_bar, grouped_bar):
        if isinstance(df_bar, pd.Series):
            return df_bar
        if isinstance(df_bar.columns, pd.MultiIndex):
            try:
                if len(grouped_bar_level_names) == 1:
                    df_grouped = df_bar.xs(grouped_bar, level=grouped_bar_level_names[0], axis=1)
                else:
                    df_grouped = df_bar.xs(grouped_bar, level=grouped_bar_level_names, axis=1)
            except KeyError:
                return None
        else:
            if grouped_bar in df_bar.columns:
                df_grouped = df_bar[grouped_bar]
            else:
                return None
        if isinstance(df_grouped, pd.DataFrame):
            df_grouped = df_grouped.sum(axis=1)
        return df_grouped

    for bar_idx, (group, period) in enumerate(all_bars):
        gkey = _hashable_key(group)
        if gkey not in group_df_cache:
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
            group_df_cache[gkey] = df_bar
        df_bar = group_df_cache[gkey]

        for grouped_idx, grouped_bar in enumerate(grouped_bars):
            cache_key = (gkey, grouped_idx)
            if cache_key not in group_cat_series:
                group_cat_series[cache_key] = _get_cat_series(df_bar, grouped_bar)
            sums = group_cat_series[cache_key]
            if sums is None:
                continue
            if period in sums.index:
                values_mat[bar_idx, grouped_idx] = float(sums.loc[period])
                # Preserve previous "skip zero columns at row level" semantics
                # by treating only non-zero values as drawn bars. The call
                # site already prunes whole zero columns via
                # skip_data_with_only_zeroes (the column is then absent from
                # grouped_bars for that row's batch — but with batched
                # subplot calls there is no per-row pruning, so we mask
                # value-by-value here).
                has_value[bar_idx, grouped_idx] = sums.loc[period] != 0

    # Per-category labels
    def _label_for(gb):
        return ' | '.join(str(v) for v in gb) if isinstance(gb, (tuple, list)) else str(gb)

    cat_labels = [_label_for(gb) for gb in grouped_bars]

    # Bar geometry: identical to the original formula.
    bar_w = SOLO_BAR_THICKNESS if n_grouped == 1 else REFERENCE_BAR_THICKNESS
    step = bar_w * (1 + BAR_GAP_FRACTION)
    total_w = bar_w * n_grouped + bar_w * BAR_GAP_FRACTION * max(0, n_grouped - 1)

    # y-positions (one per bar)
    if y_positions is None:
        y_pos_vec = np.arange(n_bars, dtype=float)
    else:
        y_pos_vec = np.asarray(y_positions, dtype=float)

    horizontal = bar_orientation == 'horizontal'

    # Track which categories we've emitted a legend entry for (mirrors original
    # behaviour: each category gets one entry, attached to its first-drawn
    # non-empty draw call across the whole subplot).
    if labeled_groups is None:
        labeled_groups_local: set = set()
    else:
        labeled_groups_local = labeled_groups

    # Track which categories were drawn (for the invisible-legend fallback)
    drawn_categories: set = set()

    for grouped_idx in range(n_grouped):
        if horizontal:
            offset = total_w / 2 - bar_w / 2 - grouped_idx * step  # top to bottom
        else:
            offset = -total_w / 2 + bar_w / 2 + grouped_idx * step  # left to right
        sub_y = y_pos_vec + offset

        mask = has_value[:, grouped_idx]
        if not mask.any():
            continue
        ys = sub_y[mask]
        widths = values_mat[mask, grouped_idx]
        color = colors[grouped_idx % len(colors)]
        label_str = cat_labels[grouped_idx]
        if label_str in labeled_groups_local:
            label = ''
        else:
            label = label_str
            labeled_groups_local.add(label_str)
        if horizontal:
            container = ax.barh(ys, widths, height=bar_w, label=label, color=color)
        else:
            container = ax.bar(ys, widths, width=bar_w, label=label, color=color)
        drawn_categories.add(grouped_idx)
        if value_fmt:
            if value_fmt == 'dynamic':
                _dfmt = DynamicFormatter()
                ax.bar_label(container, fmt=lambda x, _f=_dfmt: _f(x, None), padding=3)
            else:
                ax.bar_label(container, fmt=lambda x, _s=value_fmt: format(x, _s), padding=3)

    # Add invisible bars for any category that had no drawn entries, so the
    # legend still lists every category in the configured order.
    for grouped_idx in range(n_grouped):
        if grouped_idx in drawn_categories:
            continue
        label_str = cat_labels[grouped_idx]
        if label_str in labeled_groups_local:
            continue
        labeled_groups_local.add(label_str)
        color = colors[grouped_idx % len(colors)]
        if horizontal:
            ax.barh(0, 0, height=0.01, left=0, label=label_str, color=color)
        else:
            ax.bar(0, 0, width=0.01, bottom=0, label=label_str, color=color)


def _plot_stacked_bars(
    ax,
    df_sub: pd.DataFrame,
    all_bars: list,
    expand_axis_level_names: list,
    stack_level_names: list,
    bar_orientation: str,
    shared_color_map: dict[str, tuple] | None = None,
    y_positions: list[float] | None = None,
    slot_heights: list[float] | None = None,
) -> None:
    """Render stacked bars onto ax for one subplot.

    Vectorised: one ax.barh / ax.bar call per stack layer (per sign), using
    numpy cumulative sums to compute the `left`/`bottom` offsets for every
    bar simultaneously. Bars with zero value in the current layer are
    filtered out before the draw call so we don't create invisible
    Rectangles. Positive and negative signs are drawn in separate calls so
    matplotlib's draw order keeps the stacking visually correct.
    """
    # Get stack combinations (for colors and legend)
    if len(stack_level_names) == 1:
        stacks = df_sub.columns.get_level_values(stack_level_names[0]).unique().tolist()
    else:
        stack_df = df_sub.columns.to_frame()[stack_level_names].drop_duplicates()
        stacks = [tuple(row) for row in stack_df.values]

    # Sort stacks alphabetically when using shared colors so visual order matches legend
    if shared_color_map:
        def _stack_key(s):
            return ' | '.join(str(v) for v in s) if isinstance(s, (tuple, list)) else str(s)
        stacks.sort(key=_stack_key)

    # Colors for stacking
    if shared_color_map:
        colors = []
        for s in stacks:
            label = ' | '.join(str(v) for v in s) if isinstance(s, (tuple, list)) else str(s)
            colors.append(shared_color_map.get(label, (0.5, 0.5, 0.5)))
    else:
        n_stack = len(stacks)
        colors = list(plt.colormaps['tab10'].colors[:n_stack])
        if n_stack > 10:
            colors = list(plt.colormaps['tab20'].colors[:n_stack])

    if not all_bars:
        return

    n_bars = len(all_bars)
    n_stack = len(stacks)
    # Build a (n_bars, n_stack) value matrix by reusing per-group slices.
    values_mat = np.zeros((n_bars, n_stack), dtype=float)

    # Cache per-(group, stack) -> Series so each unique slice is computed once.
    group_stack_series: dict = {}

    def _get_stack_series(df_bar, stack):
        """Return per-period Series for given stack within df_bar (or None)."""
        if isinstance(df_bar, pd.Series):
            return df_bar
        if isinstance(df_bar.columns, pd.MultiIndex):
            try:
                if len(stack_level_names) == 1:
                    df_stack = df_bar.xs(stack, level=stack_level_names[0], axis=1)
                else:
                    df_stack = df_bar.xs(stack, level=stack_level_names, axis=1)
            except KeyError:
                return None
        else:
            if stack in df_bar.columns:
                df_stack = df_bar[stack]
            else:
                return None
        if isinstance(df_stack, pd.DataFrame):
            df_stack = df_stack.sum(axis=1)
        return df_stack

    # Cache per-group df slices.
    group_df_cache: dict = {}

    def _hashable_key(g):
        if isinstance(g, list):
            return tuple(g)
        return g

    for bar_idx, (group, period) in enumerate(all_bars):
        gkey = _hashable_key(group)
        if gkey not in group_df_cache:
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
            group_df_cache[gkey] = df_bar
        df_bar = group_df_cache[gkey]

        for stack_idx, stack in enumerate(stacks):
            cache_key = (gkey, stack_idx)
            if cache_key not in group_stack_series:
                group_stack_series[cache_key] = _get_stack_series(df_bar, stack)
            sums = group_stack_series[cache_key]
            if sums is None:
                continue
            if period in sums.index:
                values_mat[bar_idx, stack_idx] = float(sums.loc[period])

    # Vectorised y-positions
    if y_positions is None:
        y_pos_vec = np.arange(n_bars, dtype=float)
    else:
        y_pos_vec = np.asarray(y_positions, dtype=float)
    bar_h = SOLO_BAR_THICKNESS

    # Per-layer presence of positive / negative values (for legend ordering)
    pos_stacks = {si for si in range(n_stack) if (values_mat[:, si] > 0).any()}
    neg_stacks = {si for si in range(n_stack) if (values_mat[:, si] < 0).any()}

    # Positive widths and their running 'left' offsets.
    pos_widths = np.where(values_mat > 0, values_mat, 0.0)
    # left for layer i = sum of widths for layers 0..i-1
    pos_lefts = np.concatenate(
        [np.zeros((n_bars, 1)), np.cumsum(pos_widths, axis=1)[:, :-1]], axis=1,
    ) if n_stack > 0 else np.zeros((n_bars, 0))

    # Negative widths drawn in reversed stack order so layer N-1 sits closest
    # to zero. For stack_idx s in reversed order, left = sum of widths for
    # layers s+1..N-1 (i.e. layers drawn earlier in the reversed traversal).
    neg_widths = np.where(values_mat < 0, values_mat, 0.0)
    if n_stack > 0:
        # cumulative sum from right -> left, excluding self.
        rev_cum = np.cumsum(neg_widths[:, ::-1], axis=1)[:, ::-1]
        # left for stack s = rev_cum[s+1], with 0 for s = N-1.
        neg_lefts = np.concatenate(
            [rev_cum[:, 1:], np.zeros((n_bars, 1))], axis=1,
        )
    else:
        neg_lefts = np.zeros((n_bars, 0))

    horizontal = bar_orientation == 'horizontal'

    # Draw positives, forward order (0..N-1)
    for si in range(n_stack):
        mask = pos_widths[:, si] > 0
        if not mask.any():
            continue
        ys = y_pos_vec[mask]
        ws = pos_widths[mask, si]
        ls = pos_lefts[mask, si]
        color = colors[si % len(colors)]
        if horizontal:
            ax.barh(ys, ws, left=ls, height=bar_h, color=color)
        else:
            ax.bar(ys, ws, bottom=ls, width=bar_h, color=color)

    # Draw negatives, reversed order (N-1..0)
    for si in range(n_stack - 1, -1, -1):
        mask = neg_widths[:, si] < 0
        if not mask.any():
            continue
        ys = y_pos_vec[mask]
        ws = neg_widths[mask, si]
        ls = neg_lefts[mask, si]
        color = colors[si % len(colors)]
        if horizontal:
            ax.barh(ys, ws, left=ls, height=bar_h, color=color)
        else:
            ax.bar(ys, ws, bottom=ls, width=bar_h, color=color)

    # Build legend handles so visual order matches legend top-to-bottom:
    # - Horizontal: left-to-right = top-to-bottom.
    #   Full bar reads: [negatives far-left → 0 → positives far-right]
    #   Legend: negative-only first (sorted), then positives (sorted).
    # - Vertical: top-to-bottom = top-to-bottom.
    #   Full bar reads: [positives top → 0 → negatives bottom]
    #   Legend: positives first (reversed), then negative-only (reversed).
    from matplotlib.patches import Patch
    pos_indices = sorted(pos_stacks)
    neg_only_indices = sorted(neg_stacks - pos_stacks)  # stacks that are only negative

    if bar_orientation == 'horizontal':
        legend_order = neg_only_indices + pos_indices
    else:
        legend_order = list(reversed(pos_indices)) + list(reversed(neg_only_indices))

    for si in legend_order:
        stack_value = stacks[si]
        label = (
            ' | '.join(str(v) for v in stack_value)
            if isinstance(stack_value, (tuple, list))
            else str(stack_value)
        )
        ax.bar(0, 0, color=colors[si % len(colors)], label=label)  # invisible, just for legend


def _plot_simple_bars(
    ax,
    df_sub: pd.DataFrame,
    all_bars: list,
    expand_axis_level_names: list,
    bar_orientation: str,
    value_fmt: str | None,
    y_positions: list[float] | None = None,
    slot_heights: list[float] | None = None,
) -> None:
    """Render simple single-color bars onto ax for one subplot (no stacking, no grouping)."""
    if not all_bars:
        return

    # ── Precompute values for every (group, period) in one shot ──
    # Build a dict keyed by group → per-period sum Series (so each unique
    # group is only sliced + .sum()'d once), then read scalars per bar.
    group_sums: dict = {}
    for group, _period in all_bars:
        key = _hashable(group)
        if key in group_sums:
            continue
        if group is None:
            df_bar = df_sub
        elif len(expand_axis_level_names) == 1 and isinstance(df_sub.columns, pd.MultiIndex):
            df_bar = df_sub.xs(group, level=expand_axis_level_names[0], axis=1)
        elif len(expand_axis_level_names) == 1:
            df_bar = df_sub[group]
        else:
            df_bar = df_sub.xs(group, level=expand_axis_level_names, axis=1)
        # Pre-collapse to a per-period scalar (Series indexed by period) so
        # the per-bar lookup below is O(1) instead of a per-bar .sum().
        if isinstance(df_bar, pd.Series):
            sums = df_bar
        else:
            sums = df_bar.sum(axis=1)
        group_sums[key] = sums

    values: list[float] = []
    for group, period in all_bars:
        sums = group_sums[_hashable(group)]
        values.append(sums.loc[period] if period in sums.index else 0)

    # y-positions: vector matching all_bars length
    if y_positions is None:
        y_pos_vec = list(range(len(all_bars)))
    else:
        y_pos_vec = y_positions

    # Single vectorised draw call replacing the per-bar loop.
    bar_h = SOLO_BAR_THICKNESS
    if bar_orientation == 'horizontal':
        container = ax.barh(y_pos_vec, values, height=bar_h, color='steelblue')
    else:  # vertical
        container = ax.bar(y_pos_vec, values, width=bar_h, color='steelblue')
    if value_fmt:
        if value_fmt == 'dynamic':
            _dfmt = DynamicFormatter()
            ax.bar_label(container, fmt=lambda x, _f=_dfmt: _f(x, None), padding=3)
        else:
            ax.bar_label(container, fmt=lambda x, _s=value_fmt: format(x, _s), padding=3)


def _hashable(value):
    """Return a hashable key for grouping by expand_axis value (which may be a list)."""
    if isinstance(value, list):
        return tuple(value)
    return value
