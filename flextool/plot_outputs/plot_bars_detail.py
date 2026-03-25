"""
Bar chart rendering functions for each bar mode.

Called by plot_bars.py (and currently plot_functions.py during transition).
Each function receives an axes object and renders one subplot's bars.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flextool.plot_outputs.format_helpers import DynamicFormatter


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
) -> None:
    """Render grouped side-by-side bars onto ax for one subplot."""
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

    # Calculate bar width and offsets for side-by-side positioning.
    # Horizontal: group 0 at top (positive offset) so visual top-to-bottom = legend top-to-bottom.
    # Vertical: group 0 at left (negative offset) so visual left-to-right = legend top-to-bottom.
    total_bar_width = 0.8
    bar_width = total_bar_width / n_grouped
    if bar_orientation == 'horizontal':
        bar_offsets = np.linspace(
            total_bar_width / 2 - bar_width / 2,      # top
            -total_bar_width / 2 + bar_width / 2,      # bottom
            n_grouped,
        )
    else:
        bar_offsets = np.linspace(
            -total_bar_width / 2 + bar_width / 2,      # left
            total_bar_width / 2 - bar_width / 2,        # right
            n_grouped,
        )

    # Track which grouped bars have been labeled
    labeled_grouped_bars: set = set()

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
        if isinstance(df_bar, pd.Series):
            df_bar = df_bar.to_frame()

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
            bar_position = bar_idx + bar_offsets[grouped_idx]
            if bar_orientation == 'horizontal':
                container = ax.barh(bar_position, value, height=bar_width,
                                    label=label,
                                    color=colors[grouped_idx % len(colors)])
            else:  # vertical
                container = ax.bar(bar_position, value, width=bar_width,
                                   label=label,
                                   color=colors[grouped_idx % len(colors)])
            if value_fmt:
                if value_fmt == 'dynamic':
                    _dfmt = DynamicFormatter()
                    ax.bar_label(container, fmt=lambda x, _f=_dfmt: _f(x, None), padding=3)
                else:
                    ax.bar_label(container, fmt=lambda x, _s=value_fmt: format(x, _s), padding=3)

    # Add invisible bars for zero-value grouped bars (for legend completeness)
    for grouped_idx in range(len(grouped_bars)):
        if grouped_idx not in labeled_grouped_bars:
            grouped_bar = grouped_bars[grouped_idx]
            if isinstance(grouped_bar, (tuple, list)):
                label = ' | '.join(str(v) for v in grouped_bar)
            else:
                label = str(grouped_bar)
            if bar_orientation == 'horizontal':
                ax.barh(0, 0, height=bar_width, left=0,
                        label=label,
                        color=colors[grouped_idx % len(colors)])
            else:  # vertical
                ax.bar(0, 0, width=bar_width, bottom=0,
                       label=label,
                       color=colors[grouped_idx % len(colors)])


def _plot_stacked_bars(
    ax,
    df_sub: pd.DataFrame,
    all_bars: list,
    expand_axis_level_names: list,
    stack_level_names: list,
    bar_orientation: str,
    shared_color_map: dict[str, tuple] | None = None,
) -> None:
    """Render stacked bars onto ax for one subplot.

    Three sequential loops (positives → zeros → negatives) are intentional:
    matplotlib draws bars in call order, so mixing signs would break visual stacking.
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

    # Track which stacks have positive/negative values (for legend ordering)
    pos_stacks: set[int] = set()
    neg_stacks: set[int] = set()

    # Plot bars (no labels — legend is built explicitly afterwards)
    for bar_idx, (group, period) in enumerate(all_bars):
        # Get data for this group
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

        # Collect all values for this bar
        values = []
        for stack_idx, stack in enumerate(stacks):
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
                            df_stack = df_bar.xs(stack, level=stack_level_names, axis=1)
                        except KeyError:
                            value = 0
                            df_stack = None
                else:
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

        # Track positive/negative stacks
        for stack_idx, value in enumerate(values):
            if value > 0:
                pos_stacks.add(stack_idx)
            elif value < 0:
                neg_stacks.add(stack_idx)

        # Stack positive values (forward order: 0,1,...,N)
        left_pos = 0
        for stack_idx, value in enumerate(values):
            if value > 0:
                if bar_orientation == 'horizontal':
                    ax.barh(bar_idx, value, left=left_pos,
                            color=colors[stack_idx % len(colors)])
                else:
                    ax.bar(bar_idx, value, bottom=left_pos,
                           color=colors[stack_idx % len(colors)])
                left_pos += value

        # Stack negative values (reversed order: N,...,1,0)
        left_neg = 0
        for stack_idx in range(len(values) - 1, -1, -1):
            value = values[stack_idx]
            if value < 0:
                if bar_orientation == 'horizontal':
                    ax.barh(bar_idx, value, left=left_neg,
                            color=colors[stack_idx % len(colors)])
                else:
                    ax.bar(bar_idx, value, bottom=left_neg,
                           color=colors[stack_idx % len(colors)])
                left_neg += value

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
) -> None:
    """Render simple single-color bars onto ax for one subplot (no stacking, no grouping)."""
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
        if bar_orientation == 'horizontal':
            container = ax.barh(bar_idx, value, color='steelblue')
        else:  # vertical
            container = ax.bar(bar_idx, value, color='steelblue')
        if value_fmt:
            if value_fmt == 'dynamic':
                _dfmt = DynamicFormatter()
                ax.bar_label(container, fmt=lambda x, _f=_dfmt: _f(x, None), padding=3)
            else:
                ax.bar_label(container, fmt=lambda x, _s=value_fmt: format(x, _s), padding=3)
