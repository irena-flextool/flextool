import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from flextool.plot_outputs.format_helpers import _get_value_formatter
from flextool.plot_outputs.legend_helpers import (
    estimate_legend_width, _format_legend_labels, _should_show_legend,
)
from flextool.plot_outputs.axis_helpers import (
    _subplot_axis_scale, _apply_subplot_label, set_smart_xticks,
)
from flextool.plot_outputs.subplot_helpers import (
    _calculate_grid_layout, _get_unique_levels, _extract_subplot_data,
)


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
                        # For multiple line_levels, apply xs sequentially per level
                        # to avoid pandas 2.x multi-dimensional indexing error
                        y_data = df_sub
                        for lvl_name, lvl_val in zip(line_level_names, line):
                            if isinstance(y_data, pd.Series):
                                break
                            if isinstance(y_data.columns, pd.MultiIndex):
                                y_data = y_data.xs(lvl_val, level=lvl_name, axis=1)
                            else:
                                y_data = y_data[lvl_val]
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
        plt.savefig(f'{plot_dir}/{plot_name}.png', bbox_inches='tight')
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
        plt.savefig(f'{plot_dir}/{plot_name}.png', bbox_inches='tight')
    plt.close(fig)
