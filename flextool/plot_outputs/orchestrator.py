"""
Plot Outputs Orchestrator
=========================

Entry point for the plot_outputs module.

plot_dict_of_dataframes() — public entry point
  - Parses settings into PlotConfig
  - Applies dimension rules (sum, average, chunk-average, stack, unstack)
  - Splits data into per-file chunks via _plan_file_splits()
  - Dispatches to plot_lines or plot_bars
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import logging
from flextool.plot_outputs.format_helpers import (
    generate_split_filename, split_into_chunks, _chunk_average_df,
    insert_timeline_breaks,
)
from flextool.plot_outputs.config import PlotConfig, PLOT_FIELD_NAMES, _is_single_config
from flextool.plot_outputs.axis_helpers import _normalize_axis_bounds
from flextool.plot_outputs.plot_bars import plot_rowbars_stack_groupbars
from flextool.plot_outputs.plot_lines import plot_dt_sub_lines, plot_dt_stack_sub

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.category').disabled = True
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False


def _plan_file_splits(
    all_subs: list,
    all_items: list,
    subplot_levels: list[int],
    max_items_per_plot: int,
    max_subplots_per_file: int,
) -> list:
    """Plan how to split data into files.

    With subplot_levels: returns a list of file_chunks, each a list of
    (sub, item_chunk) pairs (one pair per effective subplot in that file).

    Without subplot_levels: returns a list of item_chunks, one per file.
    """
    needs_item_split = len(all_items) > max_items_per_plot
    item_chunks = (
        list(split_into_chunks(all_items, max_items_per_plot))
        if needs_item_split else [all_items]
    )

    if subplot_levels:
        # Combine all (sub, item_chunk) pairs, then split into files
        effective = [(sub, ic) for sub in all_subs for ic in item_chunks]
        if len(effective) > max_subplots_per_file:
            return list(split_into_chunks(effective, max_subplots_per_file))
        return [effective]
    else:
        # Each item_chunk becomes its own file
        return item_chunks


def plot_dict_of_dataframes(results_dict, plot_dir, plot_settings,
        active_settings=['default'], plot_rows=(0, 167), delete_existing_plots=True,
        plot_file_format='png', only_first_file=False,
        break_times: set[str] | None = None):
    """
    Plot dataframes from a dictionary according to key suffixes.

    Args:
        results_dict: Dictionary of pandas DataFrames
        plot_dir: Directory to save PNG files
        plot_settings: Dict mapping result keys to plot configuration
        active_settings: List of named configs to activate (default ['default'])
        plot_rows: Row slice to apply for time-series plots
        delete_existing_plots: If True, delete all existing plots in plot_dir (default True)
    """

    # Empty plot dir if requested
    if delete_existing_plots:
        for filename in os.listdir(plot_dir):
            file_path = os.path.join(plot_dir, filename)
            if filename == 'config.yaml':
                continue
            if os.path.isfile(file_path):
                os.remove(file_path)

    skipped_files = 0  # track files not plotted due to only_first_file

    for key in plot_settings:
        if key not in results_dict:
            logger.info(f"Plot key '{key}' not found in results — skipping")
            continue
        df_orig = results_dict[key]
        if df_orig.empty:
            continue

        # Collect active configs for this key
        chosen_settings = []
        entry = plot_settings[key]
        if _is_single_config(entry):
            if 'default' in active_settings:
                chosen_settings.append(entry)
        else:
            for setting_name, setting in entry.items():
                if setting_name in active_settings:
                    chosen_settings.append(setting)

        for setting in chosen_settings:
            # Parse raw dict into typed PlotConfig; warn about unknown keys
            # Backward compat: map old 'axis_scale_min_max' key to 'axis_bounds'
            unknown_keys = [k for k in setting if k not in PLOT_FIELD_NAMES]
            if unknown_keys:
                logging.warning(
                    f"Plot config '{key}': ignoring unknown setting(s): "
                    f"{', '.join(repr(k) for k in unknown_keys)}"
                )
            filtered = {k: v for k, v in setting.items() if k in PLOT_FIELD_NAMES}
            if 'axis_scale_min_max' in filtered and 'axis_bounds' not in filtered:
                filtered['axis_bounds'] = filtered.pop('axis_scale_min_max')
            elif 'axis_scale_min_max' in filtered:
                del filtered['axis_scale_min_max']
            cfg = PlotConfig(**filtered)

            map_dims = cfg.map_dimensions_for_plots
            if not map_dims or len(map_dims) < 2:
                continue
            index_types, rules = map_dims[0], map_dims[1]
            if not rules:
                continue
            rules = rules.replace('_', '')

            if index_types is None:
                continue
            parts = index_types.split('_')
            if len(parts) == 2:
                df_index_levels, df_columns_levels = parts
            else:
                raise ValueError(
                    f'plot setting {key}: map_dimensions_for_plots first element should '
                    f'contain one underscore to separate row and column index indicators'
                )

            plot_name = cfg.plot_name or key
            axis_bounds = _normalize_axis_bounds(cfg.axis_bounds)

            if 't' in rules and 'i' not in rules:
                chart_type = 'time'
                if cfg.full_timeline:
                    df = df_orig.copy()
                else:
                    df = df_orig.iloc[plot_rows[0]:plot_rows[1]].copy()
            elif 'i' in rules:
                chart_type = 'time'
                df = df_orig.copy()
            else:
                chart_type = 'bar'
                df = df_orig.copy()

            # Normalise to always-MultiIndex at entry to simplify downstream handling
            if not isinstance(df.index, pd.MultiIndex):
                df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_arrays([df.columns], names=[df.columns.name])

            nr_row_levels = df.index.nlevels
            if len(rules) != nr_row_levels + df.columns.nlevels:
                raise ValueError(
                    f"Number of plot_type rules different from the number of index + "
                    f"column levels in the dataframe. {plot_name}"
                )

            levels_to_sort = [i for i, c in enumerate(df_columns_levels) if c in ('e', 'g')]
            if levels_to_sort:
                df = df.sort_index(axis=1, level=levels_to_sort, sort_remaining=False)

            # Track which dimensions were summed/averaged for plot title
            summed_dimensions = []
            averaged_dimensions = []

            # Sum row levels marked 'm'
            sum_row_levels = [i for i, char in enumerate(rules[:nr_row_levels]) if char == 'm']
            if sum_row_levels:
                summed_dimensions.extend(df.index.names[i] for i in sum_row_levels)
                keep_levels = [i for i in range(nr_row_levels) if i not in sum_row_levels]
                if len(keep_levels) > 0:
                    df = df.groupby(level=keep_levels).sum()
                    for i in sum_row_levels:
                        rules = rules[:i] + rules[i + 1:]
                else:
                    df = df.sum(axis=0).to_frame().T
                    df.index = ['']
                    df.index.name = 'sum'

            # Sum column levels marked 'm'
            nr_column_levels = df.columns.nlevels
            sum_column_levels = [i for i, char in enumerate(rules[df.index.nlevels:]) if char == 'm']
            if sum_column_levels:
                summed_dimensions.extend(df.columns.names[i] for i in sum_column_levels)
                keep_levels = [i for i in range(nr_column_levels) if i not in sum_column_levels]
                if len(keep_levels) > 0:
                    df = df.T.groupby(level=keep_levels).sum().T
                    for i in sum_column_levels:
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]
                else:
                    df = df.sum(axis=1).to_frame()
                    df.columns = ['']
                    df.columns.name = 'sum'

            nr_row_levels = df.index.nlevels
            nr_column_levels = df.columns.nlevels

            # Average row levels marked 'a'
            mean_row_levels = [i for i, char in enumerate(rules[:nr_row_levels]) if char == 'a']
            if mean_row_levels:
                averaged_dimensions.extend(df.index.names[i] for i in mean_row_levels)
                keep_levels = [i for i in range(nr_row_levels) if i not in mean_row_levels]
                if len(keep_levels) > 1:
                    df = df.groupby(level=keep_levels).mean()
                    for i in mean_row_levels:
                        rules = rules[:i] + rules[i + 1:]
                else:
                    df = df.mean(axis=0).to_frame().T
                    df.index = ['']
                    df.index.name = 'mean'

            # Average column levels marked 'a'
            mean_column_levels = [i for i, char in enumerate(rules[nr_row_levels:]) if char == 'a']
            if mean_column_levels:
                averaged_dimensions.extend(df.columns.names[i] for i in mean_column_levels)
                keep_levels = [i for i in range(nr_column_levels) if i not in mean_column_levels]
                if len(keep_levels) > 1:
                    df = df.T.groupby(level=keep_levels).mean().T
                    for i in mean_column_levels:
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]
                else:
                    df = df.mean(axis=1).to_frame()
                    df.columns = ['']
                    df.columns.name = 'mean'

            nr_row_levels = df.index.nlevels
            nr_column_levels = df.columns.nlevels

            # Chunk-average 'i' row levels if time_average_duration is set
            i_positions = [pos for pos, char in enumerate(rules[:nr_row_levels]) if char == 'i']
            if i_positions and cfg.time_average_duration:
                i_pos = i_positions[0]
                chunk_size = int(cfg.time_average_duration)
                other_levels = [lv for lv in range(nr_row_levels) if lv != i_pos]
                if other_levels and nr_row_levels > 1:
                    parts = []
                    group_level = other_levels[0] if len(other_levels) == 1 else other_levels
                    for group_key, group_df in df.groupby(level=group_level):
                        flat = group_df.droplevel(other_levels)
                        averaged = _chunk_average_df(flat, chunk_size)
                        if not isinstance(group_key, tuple):
                            group_key = (group_key,)
                        new_tuples = []
                        for idx_val in averaged.index:
                            row = [None] * nr_row_levels
                            row[i_pos] = idx_val
                            for j, lv in enumerate(other_levels):
                                row[lv] = group_key[j]
                            new_tuples.append(tuple(row))
                        averaged.index = pd.MultiIndex.from_tuples(
                            new_tuples, names=df.index.names
                        )
                        parts.append(averaged)
                    df = pd.concat(parts)
                else:
                    df = _chunk_average_df(df, chunk_size)

                nr_row_levels = df.index.nlevels
                nr_column_levels = df.columns.nlevels

            if (not df.empty) and (len(df) > 0):
                # Track level names before rearrangement to rebuild rules after stack/unstack
                level_names_before = list(df.index.names) + list(df.columns.names)
                name_to_rule = dict(zip(level_names_before, rules))

                # Move bar/line levels from columns to index (batch to avoid Cartesian product)
                nr_row = df.index.nlevels
                col_levels_to_stack = [i - nr_row for i, c in enumerate(rules)
                                       if c in ('b', 't', 'i') and i >= nr_row]
                if col_levels_to_stack:
                    df = df.stack(col_levels_to_stack, future_stack=True)
                    if isinstance(df, pd.Series):
                        df = df.to_frame()

                # Rebuild rules to match level order after stacking
                level_names_mid = list(df.index.names) + list(df.columns.names)
                if (len(set(level_names_before)) == len(level_names_before)
                        and all(n is not None for n in level_names_before)):
                    try:
                        rules = ''.join(name_to_rule[n] for n in level_names_mid)
                    except KeyError:
                        pass

                # Move column-type levels from row index to columns (batch to avoid Cartesian product)
                row_levels_to_unstack = [i for i, c in enumerate(rules[:df.index.nlevels])
                                         if c in ('u', 'g', 's', 'l', 'e', 'f')]
                if row_levels_to_unstack:
                    df = df.unstack(row_levels_to_unstack)
                    if isinstance(df, pd.Series):
                        df = df.to_frame()

                # Rebuild rules to match actual level order after stack/unstack
                level_names_after = list(df.index.names) + list(df.columns.names)
                if (len(set(level_names_before)) == len(level_names_before)
                        and all(n is not None for n in level_names_before)):
                    try:
                        rules = ''.join(name_to_rule[n] for n in level_names_after)
                    except KeyError:
                        pass  # names changed unexpectedly, keep original rules

                # Drop any remaining 'm'/'a' row levels
                sum_mean_row_levels = [i for i, char in enumerate(rules[:df.index.nlevels])
                                       if char in ('m', 'a')]
                if df.index.nlevels - len(sum_mean_row_levels) > 0:
                    for i in reversed(sum_mean_row_levels):
                        df = df.droplevel(i)
                        rules = rules[:i] + rules[i + 1:]

                # Drop any remaining 'm'/'a' column levels
                sum_mean_column_levels = [i for i, char in enumerate(rules[df.index.nlevels:])
                                          if char in ('m', 'a')]
                if df.columns.nlevels - len(sum_mean_column_levels) > 0:
                    for i in reversed(sum_mean_column_levels):
                        df = df.droplevel(i, axis=1)
                        rules = rules[:i + df.index.nlevels] + rules[i + 1 + df.index.nlevels:]

                # Level indices for each plot role (in the column MultiIndex)
                col_rules = rules[df.index.nlevels:]
                grouped_bar_levels = [i for i, c in enumerate(col_rules) if c == 'g']
                stack_levels = [i for i, c in enumerate(col_rules) if c == 's']
                expand_axis_levels = [i for i, c in enumerate(col_rules) if c == 'e']
                subplot_levels = [i for i, c in enumerate(col_rules) if c == 'u']
                line_levels = [i for i, c in enumerate(col_rules) if c == 'l']
                file_levels = [i for i, c in enumerate(col_rules) if c == 'f']

                # Build plot title with summed/averaged info (separate from filename)
                plot_title = plot_name
                if summed_dimensions:
                    dim_str = "', '".join(str(d) for d in summed_dimensions)
                    plot_title = f"{plot_title} ('{dim_str}' summed)"
                if averaged_dimensions:
                    dim_str = "', '".join(str(d) for d in averaged_dimensions)
                    plot_title = f"{plot_title} ('{dim_str}' averaged)"

                # --- Determine file-dimension members (outer split) --------
                if file_levels:
                    if len(file_levels) == 1:
                        all_file_members: list = (
                            df.columns.get_level_values(file_levels[0]).unique().tolist()
                        )
                    else:
                        fm_df = (
                            df.columns.to_frame().iloc[:, file_levels].drop_duplicates()
                        )
                        all_file_members = [tuple(row) for row in fm_df.values]
                else:
                    all_file_members = [None]  # single pass, no file-dim split

                # Resolve 'shared' axis_bounds: compute global min/max from full data.
                # For stacked plots, sum the stacked columns within each subplot,
                # then take the max/min across subplots.
                if axis_bounds == 'shared':
                    numeric_df = df.select_dtypes(include='number')
                    if not numeric_df.empty:
                        if stack_levels and subplot_levels:
                            # Sum stacked columns per subplot, then take max across subplots
                            global_max = float('-inf')
                            global_min = float('inf')
                            for sub_val in (
                                numeric_df.columns.get_level_values(subplot_levels[0]).unique()
                                if len(subplot_levels) == 1
                                else [None]
                            ):
                                if sub_val is not None:
                                    try:
                                        sub_df = numeric_df.xs(sub_val, level=subplot_levels[0], axis=1)
                                    except KeyError:
                                        continue
                                else:
                                    sub_df = numeric_df
                                if isinstance(sub_df, pd.Series):
                                    sub_df = sub_df.to_frame()
                                pos_sum = sub_df.clip(lower=0).sum(axis=1)
                                neg_sum = sub_df.clip(upper=0).sum(axis=1)
                                global_max = max(global_max, float(pos_sum.max()))
                                global_min = min(global_min, float(neg_sum.min()))
                        elif stack_levels:
                            # No subplots — sum all stacked columns
                            pos_sum = numeric_df.clip(lower=0).sum(axis=1)
                            neg_sum = numeric_df.clip(upper=0).sum(axis=1)
                            global_max = float(pos_sum.max())
                            global_min = float(neg_sum.min())
                        else:
                            global_min = float(numeric_df.min().min())
                            global_max = float(numeric_df.max().max())
                        if cfg.always_include_zero_in_axis:
                            global_min = min(global_min, 0.0)
                            global_max = max(global_max, 0.0)
                        axis_bounds = [(global_min, global_max)]
                    else:
                        axis_bounds = None

                file_members_to_plot = all_file_members[:1] if only_first_file else all_file_members
                skipped_files += len(all_file_members) - len(file_members_to_plot)
                for file_member in file_members_to_plot:
                    # --- Filter df to this file member & drop the 'f' level(s) ---
                    if file_member is not None:
                        df_fm = df.copy()
                        if len(file_levels) == 1:
                            level_name = df.columns.names[file_levels[0]]
                            mask = df_fm.columns.get_level_values(level_name) == file_member
                            df_fm = df_fm.loc[:, mask]
                            # Drop the 'f' level — it's consumed by the filename
                            df_fm = df_fm.droplevel(file_levels[0], axis=1)
                        else:
                            col_frame = df_fm.columns.to_frame().iloc[:, file_levels]
                            mask = col_frame.apply(tuple, axis=1) == file_member
                            df_fm = df_fm.loc[:, mask.values]
                            for lvl in sorted(file_levels, reverse=True):
                                df_fm = df_fm.droplevel(lvl, axis=1)

                        # Ensure columns remain a MultiIndex after dropping
                        if not isinstance(df_fm.columns, pd.MultiIndex):
                            df_fm.columns = pd.MultiIndex.from_arrays(
                                [df_fm.columns], names=[df_fm.columns.name]
                            )

                        if df_fm.empty:
                            continue

                        # Build the member name string for filenames / titles
                        member_str = (
                            str(file_member) if not isinstance(file_member, tuple)
                            else '_'.join(str(v) for v in file_member)
                        )
                        effective_plot_name = f'{plot_title} — {member_str}'

                        # Recompute level indices after dropping 'f' levels
                        _shift = lambda lvls: [
                            l - sum(1 for fl in file_levels if fl < l)
                            for l in lvls
                        ]
                        fm_grouped_bar_levels = _shift(grouped_bar_levels)
                        fm_stack_levels = _shift(stack_levels)
                        fm_expand_axis_levels = _shift(expand_axis_levels)
                        fm_subplot_levels = _shift(subplot_levels)
                        fm_line_levels = _shift(line_levels)
                    else:
                        df_fm = df
                        effective_plot_name = plot_title
                        member_str = None
                        fm_grouped_bar_levels = grouped_bar_levels
                        fm_stack_levels = stack_levels
                        fm_expand_axis_levels = expand_axis_levels
                        fm_subplot_levels = subplot_levels
                        fm_line_levels = line_levels

                    # Determine original subplots
                    if not fm_subplot_levels:
                        all_subs = [None]
                    elif len(fm_subplot_levels) == 1:
                        all_subs = df_fm.columns.get_level_values(fm_subplot_levels[0]).unique().tolist()
                    else:
                        sub_df = df_fm.columns.to_frame().iloc[:, fm_subplot_levels].drop_duplicates()
                        all_subs = [tuple(row) for row in sub_df.values]

                    # Drop near-zero columns and rows when skip_data_with_only_zeroes is enabled
                    if cfg.skip_data_with_only_zeroes:
                        df_fm = df_fm.loc[:, (df_fm.abs() > 1e-6).any()]
                        if chart_type == 'bar':
                            df_fm = df_fm.loc[(df_fm.abs() > 1e-6).any(axis=1)]
                        if df_fm.empty:
                            continue

                    # Apply unit conversion multiplier
                    if cfg.multiply_by is not None:
                        df_fm = df_fm * cfg.multiply_by

                    # Insert NaN rows at timeline breaks for visual gaps
                    if chart_type == 'time' and break_times:
                        df_fm = insert_timeline_breaks(df_fm, break_times)

                    # Determine items and per-file limits
                    if chart_type == 'time':
                        default_max_items = 10
                        max_items = cfg.max_items_per_plot if cfg.max_items_per_plot is not None else default_max_items
                        item_levels = fm_line_levels if fm_line_levels else fm_stack_levels
                        if not item_levels:
                            all_items = [None]
                        elif len(item_levels) == 1:
                            all_items = df_fm.columns.get_level_values(item_levels[0]).unique().tolist()
                        else:
                            item_level_names = [df_fm.columns.names[i] for i in item_levels]
                            item_df = df_fm.columns.to_frame()[item_level_names].drop_duplicates()
                            all_items = [tuple(row) for row in item_df.values]
                    else:  # bar chart
                        default_max_items = 10
                        max_items = cfg.max_items_per_plot if cfg.max_items_per_plot is not None else default_max_items
                        all_items = df_fm.index.tolist()

                    if chart_type == 'bar':
                        # BAR (with or without subplots): send all data in one call.
                        # The bar function handles item splitting, subplot expansion,
                        # and file splitting (max_subplots_per_file) internally.
                        filepath = generate_split_filename(
                            plot_name, plot_dir, plot_file_format,
                            file_member=member_str,
                        )
                        skipped_files += plot_rowbars_stack_groupbars(
                            df_fm, effective_plot_name, plot_dir,
                            fm_stack_levels, fm_expand_axis_levels,
                            fm_subplot_levels, fm_grouped_bar_levels,
                            subplots_per_row=cfg.subplots_per_row, legend_position=cfg.legend,
                            xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                            bar_orientation=cfg.bar_orientation, base_bar_length=cfg.base_length,
                            value_label=cfg.value_label, axis_bounds=axis_bounds,
                            axis_tick_format=cfg.axis_tick_format,
                            always_include_zero_in_axis=cfg.always_include_zero_in_axis,
                            max_items_per_plot=max_items,
                            max_subplots_per_file=cfg.max_subplots_per_file,
                            output_filepath=filepath,
                            only_first_file=only_first_file,
                            skip_data_with_only_zeroes=cfg.skip_data_with_only_zeroes)

                    elif fm_subplot_levels or (
                        cfg.subplots_by_magnitudes and not fm_stack_levels
                    ):
                        # TIME + SUBPLOTS (or magnitude-split lines): send all data in one call.
                        # The plot functions handle item splitting and file splitting internally.
                        filepath = generate_split_filename(
                            plot_name, plot_dir, plot_file_format,
                            file_member=member_str,
                        )
                        if fm_stack_levels:
                            skipped_files += plot_dt_stack_sub(
                                df_fm, effective_plot_name, plot_dir,
                                fm_stack_levels, fm_subplot_levels,
                                rows=plot_rows, subplots_per_row=cfg.subplots_per_row,
                                legend_position=cfg.legend, xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                base_width_per_col=6, subplot_height=cfg.base_length,
                                axis_bounds=axis_bounds,
                                axis_tick_format=cfg.axis_tick_format,
                                always_include_zero_in_axis=cfg.always_include_zero_in_axis,
                                max_items_per_plot=max_items,
                                max_subplots_per_file=cfg.max_subplots_per_file,
                                output_filepath=filepath,
                                only_first_file=only_first_file)
                        else:
                            skipped_files += plot_dt_sub_lines(
                                df_fm, effective_plot_name, plot_dir,
                                fm_subplot_levels, fm_line_levels,
                                rows=plot_rows, subplots_per_row=cfg.subplots_per_row,
                                legend_position=cfg.legend, xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                base_width_per_col=6, subplot_height=cfg.base_length,
                                axis_bounds=axis_bounds,
                                axis_tick_format=cfg.axis_tick_format,
                                always_include_zero_in_axis=cfg.always_include_zero_in_axis,
                                max_items_per_plot=max_items,
                                max_subplots_per_file=cfg.max_subplots_per_file,
                                output_filepath=filepath,
                                only_first_file=only_first_file,
                                subplots_by_magnitudes=cfg.subplots_by_magnitudes)

                    else:
                        # TIME without subplots — split items into files
                        needs_item_split = len(all_items) > max_items
                        file_chunks = _plan_file_splits(
                            all_subs, all_items, fm_subplot_levels,
                            max_items, cfg.max_subplots_per_file
                        )
                        if only_first_file and len(file_chunks) > 1:
                            skipped_files += len(file_chunks) - 1
                            file_chunks = file_chunks[:1]
                        for file_idx, item_chunk in enumerate(file_chunks, start=1):
                            df_chunk = df_fm.copy()

                            # Filter by item chunk
                            if needs_item_split:
                                if item_levels and item_chunk[0] is not None:
                                    if len(item_levels) == 1:
                                        level_name = df_fm.columns.names[item_levels[0]]
                                        mask = df_chunk.columns.get_level_values(level_name).isin(item_chunk)
                                        df_chunk = df_chunk.loc[:, mask]
                                    else:
                                        item_level_names = [df_fm.columns.names[i] for i in item_levels]
                                        col_tuples = df_chunk.columns.to_frame()[item_level_names]
                                        mask = col_tuples.apply(tuple, axis=1).isin(item_chunk)
                                        df_chunk = df_chunk.loc[:, mask.values]

                            if df_chunk.empty:
                                continue

                            filepath = generate_split_filename(
                                plot_name, plot_dir, plot_file_format,
                                file_idx=file_idx, needs_split=needs_item_split,
                                file_member=member_str,
                            )

                            if fm_stack_levels:
                                plot_dt_stack_sub(
                                    df_chunk, effective_plot_name, plot_dir,
                                    fm_stack_levels, fm_subplot_levels,
                                    rows=plot_rows, legend_position=cfg.legend,
                                    xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                    base_width_per_col=6, subplot_height=cfg.base_length,
                                    axis_bounds=axis_bounds,
                                    axis_tick_format=cfg.axis_tick_format,
                                    always_include_zero_in_axis=cfg.always_include_zero_in_axis,
                                    output_filepath=filepath)
                            else:
                                plot_dt_sub_lines(
                                    df_chunk, effective_plot_name, plot_dir,
                                    fm_subplot_levels, fm_line_levels,
                                    rows=plot_rows, legend_position=cfg.legend,
                                    xlabel=cfg.xlabel, ylabel=cfg.ylabel,
                                    base_width_per_col=6, subplot_height=cfg.base_length,
                                    axis_bounds=axis_bounds,
                                    axis_tick_format=cfg.axis_tick_format,
                                    always_include_zero_in_axis=cfg.always_include_zero_in_axis,
                                    output_filepath=filepath)

            plt.close('all')  # Clean up

    if only_first_file and skipped_files > 0:
        logger.warning(
            "'Just one file per plot' active — %d file(s) not plotted.", skipped_files
        )
