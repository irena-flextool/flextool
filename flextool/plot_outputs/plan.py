"""PlotPlan -- pre-computed plot plans for instant rendering.

A PlotPlan captures everything needed to render any file page of a plot
without re-running dimension rules, layout computation, or color mapping.
Plans are saved alongside parquet files and loaded by the viewer.

File format:
  {result_key}__{sub_config}_plan.json  -- metadata, layout, colors, batch structure
  {result_key}__{sub_config}_plan.parquet -- processed DataFrame (post dimension rules)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from flextool.lean_parquet import read_lean_parquet, write_lean_parquet
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  JSON helper: make numpy / tuple types serializable
# ---------------------------------------------------------------------------

def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars, tuples, etc. into JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
#  PlotPlan dataclass
# ---------------------------------------------------------------------------

@dataclass
class PlotPlan:
    """Pre-computed plan for rendering a set of figures."""

    chart_type: str  # 'lines', 'stack', 'bar'
    plot_name: str
    total_file_count: int

    # The processed DataFrame -- post dimension rules, time-sliced,
    # file-member filtered, zero-filtered, multiplied.
    # For bar charts: index = bar labels, columns = data.
    # For time charts: index = time, columns = data.
    processed_df: pd.DataFrame

    # Effective plots structure: list of (title, column_labels_list)
    # column_labels_list identifies which columns of processed_df belong
    # to this subplot.
    # For bars: (title, {rows: [...], cols: [...]}) with row and column selectors.
    # For time-series: (title, column_labels_list).
    effective_plot_specs: list[tuple[str | None, list | dict]]

    # Batch structure: file_batches[i] = list of indices into effective_plot_specs
    file_batches: list[list[int]]

    # Visual config
    shared_color_map: dict[str, tuple] | None = None
    axis_bounds: list | None = None

    # Layout (stored as plain dict for JSON serialization)
    layout_type: str = ''  # 'line' or 'bar'
    layout_params: dict = field(default_factory=dict)

    # Figure builder parameters
    sub_levels: list[int] = field(default_factory=list)
    item_level_names: list[str] = field(default_factory=list)
    time_index_values: list[str] | None = None  # serialized time index

    # Config params for figure builder
    subplots_per_row: int = 2
    legend_position: str = 'right'
    xlabel: str | None = None
    ylabel: str | None = None
    bar_orientation: str = 'horizontal'
    axis_tick_format: str = 'dynamic'
    always_include_zero_in_axis: bool = True
    value_label: str | None = None
    base_bar_length: float = 4.0
    skip_data_with_only_zeroes: bool = False

    # Bar-specific level metadata
    stack_levels: list[int] = field(default_factory=list)
    stack_level_names: list[str] = field(default_factory=list)
    expand_axis_levels: list[int] = field(default_factory=list)
    expand_axis_level_names: list[str] = field(default_factory=list)
    grouped_bar_levels: list[int] = field(default_factory=list)
    grouped_bar_level_names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
#  save / load
# ---------------------------------------------------------------------------

def save_plot_plan(
    plan: PlotPlan, output_dir: Path, result_key: str, sub_config: str,
) -> None:
    """Save a PlotPlan to disk as JSON metadata + parquet data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{result_key}__{sub_config}"

    # write_lean_parquet handles single-level MultiIndex correctly, so no
    # flattening is needed (the old pandas-metadata approach required it).
    write_lean_parquet(plan.processed_df, output_dir / f"{prefix}_plan.parquet")

    # Serialize effective_plot_specs: titles may be None, selectors are lists
    # of scalars or tuples.  JSON-encode tuples as lists.
    specs_json: list[list] = []
    for title, selector in plan.effective_plot_specs:
        specs_json.append([title, _json_safe(selector)])

    # Build JSON-serializable metadata
    meta: dict[str, Any] = {
        'chart_type': plan.chart_type,
        'plot_name': plan.plot_name,
        'total_file_count': plan.total_file_count,
        'effective_plot_specs': specs_json,
        'file_batches': plan.file_batches,
        'shared_color_map': (
            {k: list(v) for k, v in plan.shared_color_map.items()}
            if plan.shared_color_map else None
        ),
        'axis_bounds': _json_safe(plan.axis_bounds),
        'layout_type': plan.layout_type,
        'layout_params': _json_safe(plan.layout_params),
        'sub_levels': plan.sub_levels,
        'item_level_names': plan.item_level_names,
        'time_index_values': plan.time_index_values,
        'subplots_per_row': plan.subplots_per_row,
        'legend_position': plan.legend_position,
        'xlabel': plan.xlabel,
        'ylabel': plan.ylabel,
        'bar_orientation': plan.bar_orientation,
        'axis_tick_format': plan.axis_tick_format,
        'always_include_zero_in_axis': plan.always_include_zero_in_axis,
        'value_label': plan.value_label,
        'base_bar_length': plan.base_bar_length,
        'skip_data_with_only_zeroes': plan.skip_data_with_only_zeroes,
        'stack_levels': plan.stack_levels,
        'stack_level_names': plan.stack_level_names,
        'expand_axis_levels': plan.expand_axis_levels,
        'expand_axis_level_names': plan.expand_axis_level_names,
        'grouped_bar_levels': plan.grouped_bar_levels,
        'grouped_bar_level_names': plan.grouped_bar_level_names,
        # Legacy: lean_parquet handles single-level MultiIndex now; kept
        # so old plan JSON files still load via the reconstruction below.
        'col_was_single_multi': False,
        'idx_was_single_multi': False,
    }

    with open(output_dir / f"{prefix}_plan.json", 'w') as f:
        json.dump(meta, f, indent=2, default=_json_safe)


def load_plot_plan(
    plan_dir: Path, result_key: str, sub_config: str,
) -> PlotPlan | None:
    """Load a PlotPlan from disk. Returns None if files don't exist."""
    plan_dir = Path(plan_dir)
    prefix = f"{result_key}__{sub_config}"

    json_path = plan_dir / f"{prefix}_plan.json"
    parquet_path = plan_dir / f"{prefix}_plan.parquet"

    if not json_path.exists() or not parquet_path.exists():
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        df = read_lean_parquet(parquet_path)

        # Reconstruct single-level MultiIndex if it was flattened on save
        if meta.get('col_was_single_multi', False):
            df.columns = pd.MultiIndex.from_arrays(
                [df.columns], names=[df.columns.name]
            )
        if meta.get('idx_was_single_multi', False):
            df.index = pd.MultiIndex.from_arrays(
                [df.index], names=[df.index.name]
            )

        # Reconstruct color map tuples
        color_map = None
        if meta.get('shared_color_map'):
            color_map = {k: tuple(v) for k, v in meta['shared_color_map'].items()}

        # Reconstruct effective_plot_specs as list of (title, selector)
        raw_specs = meta.get('effective_plot_specs', [])
        effective_plot_specs: list[tuple[str | None, list]] = [
            (s[0], s[1]) for s in raw_specs
        ]

        return PlotPlan(
            chart_type=meta['chart_type'],
            plot_name=meta['plot_name'],
            total_file_count=meta['total_file_count'],
            processed_df=df,
            effective_plot_specs=effective_plot_specs,
            file_batches=meta['file_batches'],
            shared_color_map=color_map,
            axis_bounds=meta.get('axis_bounds'),
            layout_type=meta.get('layout_type', ''),
            layout_params=meta.get('layout_params', {}),
            sub_levels=meta.get('sub_levels', []),
            item_level_names=meta.get('item_level_names', []),
            time_index_values=meta.get('time_index_values'),
            subplots_per_row=meta.get('subplots_per_row', 2),
            legend_position=meta.get('legend_position', 'right'),
            xlabel=meta.get('xlabel'),
            ylabel=meta.get('ylabel'),
            bar_orientation=meta.get('bar_orientation', 'horizontal'),
            axis_tick_format=meta.get('axis_tick_format', 'dynamic'),
            always_include_zero_in_axis=meta.get('always_include_zero_in_axis', True),
            value_label=meta.get('value_label'),
            base_bar_length=meta.get('base_bar_length', 4.0),
            skip_data_with_only_zeroes=meta.get('skip_data_with_only_zeroes', False),
            stack_levels=meta.get('stack_levels', []),
            stack_level_names=meta.get('stack_level_names', []),
            expand_axis_levels=meta.get('expand_axis_levels', []),
            expand_axis_level_names=meta.get('expand_axis_level_names', []),
            grouped_bar_levels=meta.get('grouped_bar_levels', []),
            grouped_bar_level_names=meta.get('grouped_bar_level_names', []),
        )
    except Exception as exc:
        logger.warning("Failed to load plot plan %s/%s: %s", result_key, sub_config, exc)
        return None


# ---------------------------------------------------------------------------
#  build_figure_from_plan  (the fast path)
# ---------------------------------------------------------------------------

def _select_bar_rows(df: pd.DataFrame, selector: list) -> pd.DataFrame:
    """Select rows from a bar DataFrame using the stored index labels."""
    if not selector:
        return df
    # Reconstruct tuples for MultiIndex row selection
    if isinstance(df.index, pd.MultiIndex):
        idx_labels = [tuple(s) if isinstance(s, list) else s for s in selector]
        mask = df.index.isin(idx_labels)
        return df.loc[mask]
    # Unwrap single-element lists that originate from a single-level MultiIndex
    # encoded via _encode_row_selector then JSON-roundtripped.
    flat_sel = [s[0] if isinstance(s, list) and len(s) == 1 else s for s in selector]
    return df.loc[df.index.isin(flat_sel)]


def _select_time_columns(df: pd.DataFrame, selector: list) -> pd.DataFrame:
    """Select columns from a time-series DataFrame using stored column labels."""
    if selector is None or len(selector) == 0:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        col_tuples = [tuple(c) if isinstance(c, list) else c for c in selector]
        mask = df.columns.isin(col_tuples)
        if mask.any():
            return df.loc[:, mask]
        # Fallback: try matching as strings
        col_strs = [str(c) for c in col_tuples]
        str_cols = [str(c) for c in df.columns]
        mask2 = pd.Index(str_cols).isin(col_strs)
        return df.loc[:, mask2]
    # Non-MultiIndex columns: unwrap single-element lists from JSON round-trip
    flat_sel = []
    for s in selector:
        if isinstance(s, list):
            flat_sel.append(s[0] if len(s) == 1 else tuple(s))
        else:
            flat_sel.append(s)
    valid = [s for s in flat_sel if s in df.columns]
    return df[valid] if valid else df


def build_figure_from_plan(
    plan: PlotPlan,
    file_index: int = 0,
    plot_rows: tuple[int, int] | None = None,
) -> 'Figure | None':
    """Build a single Figure from a pre-computed PlotPlan.

    This is the fast path -- no dimension rules, no layout computation,
    just direct figure building from pre-computed data.

    For time-series plans whose *processed_df* covers the full timeline,
    pass *plot_rows* ``(start, start + duration)`` to display a sub-range.
    """
    from matplotlib.figure import Figure
    from flextool.plot_outputs.subplot_helpers import LineLayoutParams, BarLayoutParams

    if file_index >= plan.total_file_count or file_index < 0:
        return None

    # For time-series plans, slice to the requested window.
    processed_df = plan.processed_df
    time_vals = plan.time_index_values
    if plot_rows is not None and plan.chart_type != 'bar':
        start, end = plot_rows
        processed_df = processed_df.iloc[start:end]
        if time_vals is not None:
            time_vals = time_vals[start:end]

    # Reconstruct effective_plots for the requested batch
    batch_indices = plan.file_batches[file_index]
    effective_plots: list[tuple[str | None, pd.DataFrame]] = []
    for idx in batch_indices:
        title, selector = plan.effective_plot_specs[idx]
        if plan.chart_type == 'bar':
            # selector may be a dict with 'rows'/'cols' keys (new format)
            # or a plain list of row labels (old format, backward compat)
            if isinstance(selector, dict):
                df_sub = _select_bar_rows(processed_df, selector.get('rows', []))
                col_sel = selector.get('cols')
                if col_sel:
                    df_sub = _select_time_columns(df_sub, col_sel)
            else:
                df_sub = _select_bar_rows(processed_df, selector)
        else:
            df_sub = _select_time_columns(processed_df, selector)
        effective_plots.append((title, df_sub))

    if not effective_plots:
        return None

    # Reconstruct time_index
    time_index = None
    if time_vals is not None:
        time_index = pd.Index(time_vals)

    # Reconstruct layout
    if plan.layout_type == 'line':
        layout = LineLayoutParams(**plan.layout_params)
    elif plan.layout_type == 'bar':
        layout = BarLayoutParams(**plan.layout_params)
    else:
        return None

    # Build the figure
    if plan.chart_type == 'lines':
        from flextool.plot_outputs.plot_lines import _build_lines_figure
        return _build_lines_figure(
            effective_plots, plan.plot_name, plan.sub_levels,
            plan.item_level_names, time_index,
            plan.subplots_per_row, plan.legend_position,
            plan.xlabel, plan.ylabel,
            plan.axis_bounds, plan.axis_tick_format,
            plan.always_include_zero_in_axis,
            layout, plan.shared_color_map,
        )
    elif plan.chart_type == 'stack':
        from flextool.plot_outputs.plot_lines import _build_stack_figure
        return _build_stack_figure(
            effective_plots, plan.plot_name, plan.sub_levels,
            plan.item_level_names, time_index,
            plan.subplots_per_row, plan.legend_position,
            plan.xlabel, plan.ylabel,
            plan.axis_bounds, plan.axis_tick_format,
            plan.always_include_zero_in_axis,
            layout, plan.shared_color_map,
        )
    elif plan.chart_type == 'bar':
        from flextool.plot_outputs.plot_bars import _build_bar_figure

        # Resolve value_label
        value_fmt = None
        if plan.value_label is True or plan.value_label == 'true':
            value_fmt = 'dynamic'
        elif plan.value_label:
            value_fmt = str(plan.value_label)

        batch_title = (
            f"{plan.plot_name} ({file_index + 1}/{plan.total_file_count})"
            if plan.total_file_count > 1 else plan.plot_name
        )
        return _build_bar_figure(
            effective_plots, plan.processed_df, batch_title, '',
            plan.stack_levels, plan.stack_level_names,
            plan.expand_axis_levels, plan.expand_axis_level_names,
            plan.sub_levels,
            plan.grouped_bar_levels, plan.grouped_bar_level_names,
            plan.legend_position, plan.subplots_per_row,
            plan.xlabel, plan.ylabel,
            plan.bar_orientation, plan.base_bar_length,
            value_fmt,
            plan.axis_bounds, plan.axis_tick_format,
            plan.always_include_zero_in_axis,
            layout, plan.shared_color_map,
            plan.skip_data_with_only_zeroes,
        )

    return None


# ---------------------------------------------------------------------------
#  _encode_column_selector / _encode_row_selector
# ---------------------------------------------------------------------------

def _encode_column_selector(
    df_sub: pd.DataFrame, df_full: pd.DataFrame,
) -> list:
    """Encode the columns of df_sub as a JSON-serializable selector list.

    For MultiIndex columns, each column is stored as a list of its level values.
    For simple columns, each column is stored as a scalar.
    """
    if isinstance(df_sub.columns, pd.MultiIndex):
        return [list(c) for c in df_sub.columns]
    return df_sub.columns.tolist()


def _encode_row_selector(df_sub: pd.DataFrame) -> list:
    """Encode the row index labels of df_sub for bar charts."""
    if isinstance(df_sub.index, pd.MultiIndex):
        return [list(idx) for idx in df_sub.index]
    return df_sub.index.tolist()


# ---------------------------------------------------------------------------
#  _compute_time_plan / _compute_bar_plan
# ---------------------------------------------------------------------------

def _compute_time_plan(
    df_fm: pd.DataFrame,
    effective_plot_name: str,
    cfg: 'PlotConfig',
    fm_stack_levels: list[int],
    fm_subplot_levels: list[int],
    fm_line_levels: list[int],
    axis_bounds,
    plot_rows: tuple[int, int],
) -> PlotPlan | None:
    """Compute a PlotPlan for a time-series (lines or stack) chart.

    Replicates the planning steps of build_line_figures() and
    build_stack_figures() without actually creating matplotlib Figures.
    """
    from flextool.plot_outputs.plot_lines import (
        _build_effective_plots, _compute_line_layout, _make_file_batches,
        _get_column_items,
    )
    from flextool.plot_outputs.legend_helpers import build_shared_color_map

    # Determine chart sub-type
    is_stack = bool(fm_stack_levels)
    chart_type = 'stack' if is_stack else 'lines'
    item_levels = fm_stack_levels if is_stack else fm_line_levels

    # Convert level indices to level names
    if isinstance(df_fm.columns, pd.MultiIndex):
        item_level_names = [df_fm.columns.names[i] for i in item_levels]
    else:
        item_level_names = item_levels

    # Get x-axis index
    if isinstance(df_fm.index, pd.MultiIndex):
        time_index = df_fm.index.get_level_values(-1).astype(str)
    else:
        time_index = df_fm.index.astype(str)

    # Determine max items
    default_max_items = 10
    max_items = (
        cfg.max_items_per_plot
        if cfg.max_items_per_plot is not None
        else default_max_items
    )

    # Build effective_plots with item splitting
    effective_plots = _build_effective_plots(
        df_fm, fm_subplot_levels, item_level_names, max_items,
        subplots_by_magnitudes=cfg.subplots_by_magnitudes if not is_stack else False,
    )
    if not effective_plots:
        return None

    # Build shared color map
    shared_color_map = None
    if cfg.legend == 'shared' and item_level_names:
        all_labels: list[str] = []
        for _, df_sub in effective_plots:
            for item in _get_column_items(df_sub, item_level_names):
                label = str(item)
                if label not in all_labels:
                    all_labels.append(label)
        all_labels.sort()
        shared_color_map = build_shared_color_map(all_labels)

    # Compute layout
    layout = _compute_line_layout(
        effective_plots, item_level_names,
        cfg.legend, cfg.subplots_per_row,
        6,  # base_width_per_col default
        cfg.base_length,
        cfg.axis_tick_format,
    )

    # Split into file batches
    file_batches_raw = _make_file_batches(
        effective_plots, cfg.max_subplots_per_file, None, '', effective_plot_name,
    )
    total_file_count = len(file_batches_raw)

    # Encode effective_plot_specs and build batch index lists
    effective_plot_specs: list[tuple[str | None, list]] = []
    for title, df_sub in effective_plots:
        selector = _encode_column_selector(df_sub, df_fm)
        effective_plot_specs.append((title, selector))

    file_batches: list[list[int]] = []
    # _make_file_batches returns [(batch_list, filepath), ...]
    # We need to map each batch's items back to effective_plot indices.
    offset = 0
    for batch, _filepath in file_batches_raw:
        batch_size = len(batch)
        file_batches.append(list(range(offset, offset + batch_size)))
        offset += batch_size

    # Serialize layout
    layout_params = {
        'value_label_width': layout.value_label_width,
        'legend_width': layout.legend_width,
        'base_width': layout.base_width,
        'subplot_height': layout.subplot_height,
    }

    return PlotPlan(
        chart_type=chart_type,
        plot_name=effective_plot_name,
        total_file_count=total_file_count,
        processed_df=df_fm,
        effective_plot_specs=effective_plot_specs,
        file_batches=file_batches,
        shared_color_map=shared_color_map,
        axis_bounds=axis_bounds,
        layout_type='line',
        layout_params=layout_params,
        sub_levels=fm_subplot_levels,
        item_level_names=item_level_names,
        time_index_values=time_index.tolist(),
        subplots_per_row=cfg.subplots_per_row,
        legend_position=cfg.legend,
        xlabel=cfg.xlabel,
        ylabel=cfg.ylabel,
        axis_tick_format=cfg.axis_tick_format,
        always_include_zero_in_axis=cfg.always_include_zero_in_axis,
    )


def _compute_bar_plan(
    df_fm: pd.DataFrame,
    effective_plot_name: str,
    cfg: 'PlotConfig',
    fm_stack_levels: list[int],
    fm_expand_axis_levels: list[int],
    fm_subplot_levels: list[int],
    fm_grouped_bar_levels: list[int],
    axis_bounds,
) -> PlotPlan | None:
    """Compute a PlotPlan for a bar chart.

    Replicates the planning steps of build_bar_figures() without actually
    creating matplotlib Figures.
    """
    from flextool.plot_outputs.plot_bars import _compute_bar_layout
    from flextool.plot_outputs.subplot_helpers import _get_unique_levels, _extract_subplot_data
    from flextool.plot_outputs.legend_helpers import build_shared_color_map, _format_legend_labels

    sub_levels = fm_subplot_levels or []
    stack_levels = fm_stack_levels or []
    grouped_bar_levels = fm_grouped_bar_levels or []
    expand_axis_levels = fm_expand_axis_levels or []

    # Convert level indices to names
    if isinstance(df_fm.columns, pd.MultiIndex):
        stack_level_names = [df_fm.columns.names[i] for i in stack_levels] if stack_levels else []
        expand_axis_level_names = [df_fm.columns.names[i] for i in expand_axis_levels] if expand_axis_levels else []
        grouped_bar_level_names = [df_fm.columns.names[i] for i in grouped_bar_levels] if grouped_bar_levels else []
    else:
        stack_level_names = stack_levels
        expand_axis_level_names = [df_fm.columns.name] if expand_axis_levels else []
        grouped_bar_level_names = [df_fm.columns.name] if grouped_bar_levels else []

    subs = _get_unique_levels(df_fm.columns, sub_levels)

    # Compute expand-group count
    if expand_axis_levels and isinstance(df_fm.columns, pd.MultiIndex):
        if len(expand_axis_level_names) == 1:
            expand_level_name = expand_axis_level_names[0]
            n_expand_groups = len(df_fm.columns.get_level_values(expand_level_name).unique())
        else:
            expand_frame = df_fm.columns.to_frame()[expand_axis_level_names].drop_duplicates()
            n_expand_groups = len(expand_frame)
    else:
        n_expand_groups = 1

    # Determine max items
    default_max_items = 10
    max_items = (
        cfg.max_items_per_plot
        if cfg.max_items_per_plot is not None
        else default_max_items
    )

    # Build effective_plots (mirrors build_bar_figures logic).
    # Split subplots that exceed max_items_per_plot visual bar labels
    # (n_rows * n_expand_groups).
    effective_plots: list[tuple[str | None, pd.DataFrame]] = []
    for sub in subs:
        df_sub = _extract_subplot_data(df_fm, sub, sub_levels)
        df_sub = df_sub.dropna(how='all')
        if df_sub.empty:
            continue
        df_sub = df_sub.fillna(0)
        title = (
            ' | '.join(str(v) for v in sub) if isinstance(sub, tuple)
            else str(sub) if sub is not None else None
        )
        n_rows = len(df_sub)

        if not max_items:
            effective_plots.append((title, df_sub))
            continue

        total_items = n_rows * max(n_expand_groups, 1)
        if total_items <= max_items:
            effective_plots.append((title, df_sub))
            continue

        if expand_axis_levels and expand_axis_level_names and n_expand_groups > 1:
            expand_level_name_local = expand_axis_level_names[0]
            max_groups = max(1, max_items // max(n_rows, 1))
            all_groups = df_sub.columns.get_level_values(expand_level_name_local).unique().tolist()
            for gi, grp_start in enumerate(range(0, len(all_groups), max_groups)):
                grp_chunk = all_groups[grp_start:grp_start + max_groups]
                mask = df_sub.columns.get_level_values(expand_level_name_local).isin(grp_chunk)
                chunk = df_sub.loc[:, mask]
                chunk_label = f"{title}_{gi + 1}" if title else None
                effective_plots.append((chunk_label, chunk))
        elif n_rows > max_items:
            for i in range(0, n_rows, max_items):
                chunk = df_sub.iloc[i:i + max_items]
                chunk_label = f"{title}_{i // max_items + 1}" if title else None
                effective_plots.append((chunk_label, chunk))
        else:
            effective_plots.append((title, df_sub))

    if not effective_plots:
        return None

    # Build shared color map
    shared_color_map = None
    if cfg.legend == 'shared' and (stack_levels or grouped_bar_levels):
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

    # Compute layout
    layout = _compute_bar_layout(
        effective_plots, df_fm,
        expand_axis_levels, expand_axis_level_names,
        stack_levels, stack_level_names,
        grouped_bar_levels, grouped_bar_level_names,
        cfg.legend, cfg.subplots_per_row,
        cfg.base_length,
    )

    # Split into file batches respecting max_subplots_per_file and, for
    # horizontal bars, max_items_per_subplot_column (total bar-label count
    # in any single grid column must not exceed the limit).
    spr = max(cfg.subplots_per_row, 1)
    max_per_file = cfg.max_subplots_per_file or len(effective_plots)
    col_limit = (
        cfg.max_items_per_subplot_column
        if cfg.bar_orientation == 'horizontal' else 0
    )

    # Compute visual item counts per subplot (bar labels on the y-axis).
    # With expand groups, each group contributes its own set of labels,
    # so the count can be much larger than len(df_sub).
    def _count_visual_items(df_sub: pd.DataFrame) -> int:
        if not expand_axis_level_names or not isinstance(df_sub.columns, pd.MultiIndex):
            return max(len(df_sub), 1)
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
    for i in range(0, len(effective_plots), spr):
        grid_rows.append(list(range(i, min(i + spr, len(effective_plots)))))

    raw_batches: list[list[tuple[str | None, pd.DataFrame]]] = []
    current_batch: list[tuple[str | None, pd.DataFrame]] = []
    col_counts = [0] * spr  # cumulative bar-label count per grid column

    for row_indices in grid_rows:
        # Check whether adding this row would breach either limit.
        would_exceed_subplots = len(current_batch) + len(row_indices) > max_per_file
        would_exceed_col = False
        if col_limit and current_batch:
            for j, idx in enumerate(row_indices):
                if col_counts[j] + visual_item_counts[idx] > col_limit:
                    would_exceed_col = True
                    break

        if (would_exceed_subplots or would_exceed_col) and current_batch:
            raw_batches.append(current_batch)
            current_batch = []
            col_counts = [0] * spr

        current_batch.extend([effective_plots[i] for i in row_indices])
        for j, idx in enumerate(row_indices):
            col_counts[j] += visual_item_counts[idx]

    if current_batch:
        raw_batches.append(current_batch)

    total_file_count = len(raw_batches)

    # Encode effective_plot_specs: store row index labels AND column
    # selectors so that reconstruction filters both dimensions.
    # Without column selectors, subplots with expand_axis levels
    # would show a Cartesian product of all columns.
    effective_plot_specs: list[tuple[str | None, list]] = []
    for title, df_sub in effective_plots:
        selector = {
            'rows': _encode_row_selector(df_sub),
            'cols': _encode_column_selector(df_sub, df_fm),
        }
        effective_plot_specs.append((title, selector))

    # Build batch index lists
    file_batches: list[list[int]] = []
    offset = 0
    for batch in raw_batches:
        batch_size = len(batch)
        file_batches.append(list(range(offset, offset + batch_size)))
        offset += batch_size

    # Serialize layout
    layout_params = {
        'bar_label_width': layout.bar_label_width,
        'group_label_width': layout.group_label_width,
        'total_label_width': layout.total_label_width,
        'legend_width': layout.legend_width,
        'legend_height': layout.legend_height,
        'base_bar_length': layout.base_bar_length,
        'value_axis_width': layout.value_axis_width,
    }

    return PlotPlan(
        chart_type='bar',
        plot_name=effective_plot_name,
        total_file_count=total_file_count,
        processed_df=df_fm,
        effective_plot_specs=effective_plot_specs,
        file_batches=file_batches,
        shared_color_map=shared_color_map,
        axis_bounds=axis_bounds,
        layout_type='bar',
        layout_params=layout_params,
        sub_levels=sub_levels,
        item_level_names=[],
        subplots_per_row=cfg.subplots_per_row,
        legend_position=cfg.legend,
        xlabel=cfg.xlabel,
        ylabel=cfg.ylabel,
        bar_orientation=cfg.bar_orientation,
        axis_tick_format=cfg.axis_tick_format,
        always_include_zero_in_axis=cfg.always_include_zero_in_axis,
        value_label=cfg.value_label,
        base_bar_length=cfg.base_length,
        skip_data_with_only_zeroes=cfg.skip_data_with_only_zeroes,
        stack_levels=stack_levels,
        stack_level_names=stack_level_names,
        expand_axis_levels=expand_axis_levels,
        expand_axis_level_names=expand_axis_level_names,
        grouped_bar_levels=grouped_bar_levels,
        grouped_bar_level_names=grouped_bar_level_names,
    )


# ---------------------------------------------------------------------------
#  compute_plot_plans_for_result
# ---------------------------------------------------------------------------

def compute_plot_plans_for_result(
    df: pd.DataFrame,
    result_key: str,
    plot_settings: dict,
    output_dir: Path,
    plot_rows: tuple[int, int] = (0, 167),
    break_times: set[str] | None = None,
    active_settings: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Compute and save PlotPlans for all configs of a result_key.

    Called after scenario runs to pre-compute plans.
    Returns list of (result_key, sub_config) pairs that produced valid plans.
    The *active_settings* parameter is accepted for backward compatibility
    but ignored — all configs in the YAML entry are processed so that the
    viewer can build a complete availability manifest.
    """
    from flextool.plot_outputs.config import PlotConfig, PLOT_FIELD_NAMES, _is_single_config
    from flextool.plot_outputs.orchestrator import (
        _apply_dimension_rules, _resolve_shared_axis_bounds, _process_file_member,
    )
    from flextool.plot_outputs.axis_helpers import _normalize_axis_bounds
    from flextool.plot_outputs.format_helpers import insert_timeline_breaks

    entry = plot_settings.get(result_key)
    if not isinstance(entry, dict):
        return []

    succeeded: list[tuple[str, str]] = []

    # Collect all configs (not just active ones)
    chosen: list[tuple[str, dict]] = []
    if _is_single_config(entry):
        chosen.append(('default', entry))
    else:
        for name, sub in entry.items():
            if isinstance(sub, dict):
                chosen.append((name, sub))

    for sub_config, raw_setting in chosen:
        # Parse into PlotConfig
        filtered = {k: v for k, v in raw_setting.items() if k in PLOT_FIELD_NAMES}
        if 'axis_scale_min_max' in filtered and 'axis_bounds' not in filtered:
            filtered['axis_bounds'] = filtered.pop('axis_scale_min_max')
        elif 'axis_scale_min_max' in filtered:
            del filtered['axis_scale_min_max']
        filtered.pop('variant', None)
        try:
            cfg = PlotConfig(**filtered)
        except TypeError:
            continue

        if not cfg.map_dimensions_for_plots or len(cfg.map_dimensions_for_plots) < 2:
            continue

        plot_name = cfg.plot_name or result_key

        # Check availability using the FULL data range — a variant is
        # "available" if there's any non-zero data anywhere in the time
        # series, even if the current plot_rows window is all zeros.
        # This prevents hourly variants from being marked unavailable when
        # spikes happen outside the displayed window.
        full_range = (0, len(df))
        avail_result = _apply_dimension_rules(df, cfg, full_range)
        if avail_result is None:
            continue
        df_avail = avail_result[0]
        has_data = True
        if cfg.skip_data_with_only_zeroes:
            numeric = df_avail.select_dtypes(include='number')
            has_data = not numeric.empty and (numeric.abs() >= 1e-6).any().any()
        if not has_data:
            continue
        # Record availability regardless of whether plan generation succeeds
        succeeded.append((result_key, sub_config))

        # Apply dimension rules.  For time-series plans, use the full data
        # range so the plan can be rendered at any start/duration without
        # recomputing.  For bar charts, plot_rows is irrelevant.
        dim_result = _apply_dimension_rules(df, cfg, full_range)
        if dim_result is None:
            continue
        df_processed, rules, chart_type, summed_dims, averaged_dims = dim_result

        # Identify level roles
        col_rules = rules[df_processed.index.nlevels:]
        grouped_bar_levels = [i for i, c in enumerate(col_rules) if c == 'g']
        stack_levels = [i for i, c in enumerate(col_rules) if c == 's']
        expand_axis_levels = [i for i, c in enumerate(col_rules) if c == 'e']
        subplot_levels = [i for i, c in enumerate(col_rules) if c == 'u']
        line_levels = [i for i, c in enumerate(col_rules) if c == 'l']
        file_levels = [i for i, c in enumerate(col_rules) if c == 'f']

        # Build plot title
        plot_title = plot_name
        if summed_dims:
            dim_str = "', '".join(str(d) for d in summed_dims)
            plot_title = f"{plot_title} ('{dim_str}' summed)"
        if averaged_dims:
            dim_str = "', '".join(str(d) for d in averaged_dims)
            plot_title = f"{plot_title} ('{dim_str}' averaged)"

        # File members
        if file_levels:
            if len(file_levels) == 1:
                all_file_members = (
                    df_processed.columns.get_level_values(file_levels[0]).unique().tolist()
                )
            else:
                fm_df = (
                    df_processed.columns.to_frame().iloc[:, file_levels].drop_duplicates()
                )
                all_file_members = [tuple(row) for row in fm_df.values]
        else:
            all_file_members = [None]

        # Resolve shared axis bounds
        axis_bounds = _normalize_axis_bounds(cfg.axis_bounds)
        axis_bounds = _resolve_shared_axis_bounds(
            df_processed, axis_bounds, stack_levels, subplot_levels,
            cfg.always_include_zero_in_axis,
        )

        # Process each file member
        for file_member in all_file_members:
            result = _process_file_member(
                df_processed, file_member, file_levels, plot_title,
                grouped_bar_levels, stack_levels, expand_axis_levels,
                subplot_levels, line_levels,
            )
            if result is None:
                continue
            (df_fm, effective_plot_name, member_str,
             fm_grouped_bar_levels, fm_stack_levels, fm_expand_axis_levels,
             fm_subplot_levels, fm_line_levels) = result

            # Apply skip_zeroes, multiply_by, timeline breaks
            if cfg.skip_data_with_only_zeroes:
                df_fm = df_fm.loc[:, (df_fm.abs() > 1e-6).any()]
                if chart_type == 'bar':
                    df_fm = df_fm.loc[(df_fm.abs() > 1e-6).any(axis=1)]
                if df_fm.empty:
                    continue

            if cfg.multiply_by is not None:
                df_fm = df_fm * cfg.multiply_by

            if chart_type == 'time' and break_times:
                df_fm = insert_timeline_breaks(df_fm, break_times)

            # Compute plan based on chart type
            if chart_type == 'bar':
                plan = _compute_bar_plan(
                    df_fm, effective_plot_name, cfg,
                    fm_stack_levels, fm_expand_axis_levels,
                    fm_subplot_levels, fm_grouped_bar_levels,
                    axis_bounds,
                )
            elif chart_type == 'time':
                plan = _compute_time_plan(
                    df_fm, effective_plot_name, cfg,
                    fm_stack_levels, fm_subplot_levels, fm_line_levels,
                    axis_bounds, plot_rows,
                )
            else:
                continue

            if plan is None:
                continue

            # Determine sub_config name for file member variants
            save_sub = sub_config
            if member_str:
                save_sub = f"{sub_config}__{member_str}"

            save_plot_plan(plan, output_dir, result_key, save_sub)
            # Record file-member-specific availability if different from base
            if save_sub != sub_config:
                succeeded.append((result_key, save_sub))

    return succeeded


def compute_live_plan(
    df: pd.DataFrame,
    cfg: 'PlotConfig',
    plot_name: str,
    break_times: set[str] | None = None,
) -> PlotPlan | None:
    """Compute a PlotPlan in memory without disk I/O.

    Uses the full data range so time-series plans can be rendered at any
    start/duration via :func:`build_figure_from_plan` with *plot_rows*.
    """
    from flextool.plot_outputs.config import PlotConfig  # noqa: F811
    from flextool.plot_outputs.orchestrator import (
        _apply_dimension_rules, _resolve_shared_axis_bounds, _process_file_member,
    )
    from flextool.plot_outputs.axis_helpers import _normalize_axis_bounds
    from flextool.plot_outputs.format_helpers import insert_timeline_breaks

    full_range = (0, len(df))
    dim_result = _apply_dimension_rules(df, cfg, full_range)
    if dim_result is None:
        return None
    df_processed, rules, chart_type, summed_dims, averaged_dims = dim_result

    col_rules = rules[df_processed.index.nlevels:]
    grouped_bar_levels = [i for i, c in enumerate(col_rules) if c == 'g']
    stack_levels = [i for i, c in enumerate(col_rules) if c == 's']
    expand_axis_levels = [i for i, c in enumerate(col_rules) if c == 'e']
    subplot_levels = [i for i, c in enumerate(col_rules) if c == 'u']
    line_levels = [i for i, c in enumerate(col_rules) if c == 'l']
    file_levels = [i for i, c in enumerate(col_rules) if c == 'f']

    plot_title = plot_name
    if summed_dims:
        plot_title = f"{plot_title} ('{', '.join(str(d) for d in summed_dims)}' summed)"
    if averaged_dims:
        plot_title = f"{plot_title} ('{', '.join(str(d) for d in averaged_dims)}' averaged)"

    # File members (take the first — live plans don't split across files by member)
    file_member = None
    if file_levels:
        if len(file_levels) == 1:
            members = df_processed.columns.get_level_values(file_levels[0]).unique().tolist()
        else:
            fm_df = df_processed.columns.to_frame().iloc[:, file_levels].drop_duplicates()
            members = [tuple(row) for row in fm_df.values]
        if members:
            file_member = members[0]

    axis_bounds = _normalize_axis_bounds(cfg.axis_bounds)
    axis_bounds = _resolve_shared_axis_bounds(
        df_processed, axis_bounds, stack_levels, subplot_levels,
        cfg.always_include_zero_in_axis,
    )

    result = _process_file_member(
        df_processed, file_member, file_levels, plot_title,
        grouped_bar_levels, stack_levels, expand_axis_levels,
        subplot_levels, line_levels,
    )
    if result is None:
        return None
    (df_fm, effective_plot_name, _member_str,
     fm_grouped_bar_levels, fm_stack_levels, fm_expand_axis_levels,
     fm_subplot_levels, fm_line_levels) = result

    if cfg.skip_data_with_only_zeroes:
        df_fm = df_fm.loc[:, (df_fm.abs() > 1e-6).any()]
        if chart_type == 'bar':
            df_fm = df_fm.loc[(df_fm.abs() > 1e-6).any(axis=1)]
        if df_fm.empty:
            return None

    if cfg.multiply_by is not None:
        df_fm = df_fm * cfg.multiply_by

    if chart_type == 'time' and break_times:
        df_fm = insert_timeline_breaks(df_fm, break_times)

    if chart_type == 'bar':
        return _compute_bar_plan(
            df_fm, effective_plot_name, cfg,
            fm_stack_levels, fm_expand_axis_levels,
            fm_subplot_levels, fm_grouped_bar_levels,
            axis_bounds,
        )
    elif chart_type == 'time':
        return _compute_time_plan(
            df_fm, effective_plot_name, cfg,
            fm_stack_levels, fm_subplot_levels, fm_line_levels,
            axis_bounds, full_range,
        )
    return None
