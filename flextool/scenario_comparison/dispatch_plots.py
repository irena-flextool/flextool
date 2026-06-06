"""Dispatch area plots: render and save stacked-area dispatch plots for nodeGroups and nodes."""

import pandas as pd
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from flextool.scenario_comparison.constants import DEFAULT_SPECIAL_COLORS
from flextool.scenario_comparison.data_models import TimeSeriesResults, DispatchMappings
from flextool.scenario_comparison.dispatch_data import (
    prepare_dispatch_data,
    prepare_node_dispatch_data,
)


def _auto_assign_node_colors_with_existing(
    columns, existing: dict[str, str | None],
) -> dict[str, str]:
    """Auto-assign tab20 colors for columns not already in *existing*."""
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    colors = dict(existing)
    # Also include special colors as defaults
    for col, color in DEFAULT_SPECIAL_COLORS.items():
        if col not in colors:
            colors[col] = color
    color_idx = 0
    for col in columns:
        col_str = str(col)
        if col_str in colors and colors[col_str] is not None:
            continue
        # Check base name (_in/_out suffix)
        base = col_str
        if col_str.endswith('_in'):
            base = col_str[:-3]
        elif col_str.endswith('_out'):
            base = col_str[:-4]
        if base in colors and colors[base] is not None:
            colors[col_str] = colors[base]
            continue
        colors[col_str] = matplotlib.colors.rgb2hex(cmap[color_idx % 20])
        color_idx += 1
    return colors


def get_color_for_column(col: str, colors_dict: dict[str, str | None]) -> str:
    """Get color for a column, handling _in/_out suffixes."""
    # Direct lookup
    if col in colors_dict and colors_dict[col] is not None:
        return colors_dict[col]
    # Try base name without _in/_out suffix
    if col.endswith('_in') or col.endswith('_out'):
        base_name = col[:-3] if col.endswith('_in') else col[:-4]
        if base_name in colors_dict and colors_dict[base_name] is not None:
            return colors_dict[base_name]
    # Default color
    return 'lightgray'


def _compute_ylim(
    df: pd.DataFrame,
    timeline: tuple[int, int],
    inflow: pd.Series | None = None,
) -> tuple[float, float]:
    """Compute y-axis range from a dispatch DataFrame, including line overlays.

    Accounts for the stacked area, the Curtailed dashed line, and the
    Demand (inflow) solid line so that nothing is clipped.
    """
    df_slice = df.iloc[timeline[0]:timeline[1]]
    plot_cols = [col for col in df.columns if col != 'Curtailed']
    area_slice = df_slice[plot_cols]
    pos_max = area_slice.clip(lower=0).sum(axis=1).max()
    neg_min = area_slice.clip(upper=0).sum(axis=1).min()

    # Include Curtailed line
    if 'Curtailed' in df.columns:
        curtailed_max = df_slice['Curtailed'].max()
        curtailed_min = df_slice['Curtailed'].min()
        pos_max = max(pos_max, curtailed_max)
        neg_min = min(neg_min, curtailed_min)

    # Include Demand (inflow) line
    if inflow is not None and not inflow.empty:
        inflow_slice = inflow.iloc[timeline[0]:timeline[1]]
        pos_max = max(pos_max, inflow_slice.max())
        neg_min = min(neg_min, inflow_slice.min())

    return (neg_min, pos_max)


def _build_dispatch_figure(
    df_dispatch: pd.DataFrame,
    inflow_series: pd.Series | None,
    title: str,
    ylabel: str = "MWh/h",
    colors: dict | None = None,
    timeline: tuple[int, int] = (0, 168),
    ylim: tuple[float, float] | None = None,
    break_times: set[str] | None = None,
) -> plt.Figure | None:
    """Build a dispatch stacked-area Figure and return it (without saving or closing).

    Returns None if there's nothing to plot.
    """
    from flextool.plot_outputs.format_helpers import insert_timeline_breaks

    if colors is None:
        colors = {}

    # Auto-assign colors for columns not already in the colors dict
    colors = _auto_assign_node_colors_with_existing(df_dispatch.columns, colors)

    # Get plot colors for columns (excluding 'Curtailed' which is plotted as line)
    plot_cols = [col for col in df_dispatch.columns if col != 'Curtailed']
    plot_colors = [get_color_for_column(col, colors) for col in plot_cols]

    # Slice to timeline
    df_plot = df_dispatch.iloc[timeline[0]:timeline[1]]

    # Insert NaN rows at timeline breaks for visual gaps
    if break_times:
        df_plot = insert_timeline_breaks(df_plot, break_times)
        if inflow_series is not None:
            inflow_series = insert_timeline_breaks(
                inflow_series.iloc[timeline[0]:timeline[1]].to_frame(), break_times
            ).iloc[:, 0]

    # Check if there's anything to plot (area, curtailed, or demand)
    has_area = bool(plot_cols) and not (df_plot[plot_cols].select_dtypes(include='number').abs() < 1e-6).all().all()
    has_curtailed = 'Curtailed' in df_dispatch.columns and (df_plot['Curtailed'].abs() > 1e-6).any()
    has_demand = inflow_series is not None and not inflow_series.empty
    if not has_area and not has_curtailed and not has_demand:
        return None

    # Estimate legend width to size figure appropriately
    all_labels = list(plot_cols)
    if has_curtailed:
        all_labels.append('Curtailed')
    if has_demand:
        all_labels.append('Demand')
    max_label_len = max((len(str(label)) for label in all_labels), default=0)
    legend_width_in = max(1.5, max_label_len * 0.08 + 0.6)

    plot_width = 10
    fig_width = plot_width + legend_width_in + 0.3
    fig = Figure(figsize=(fig_width, 4))

    # Position axes to leave room for legend on the right
    left_margin = 0.08
    right_margin = (legend_width_in + 0.2) / fig_width
    ax = fig.add_axes([left_margin, 0.15, 1.0 - left_margin - right_margin, 0.75])

    # Plot area chart (NaN rows create visual gaps at timeline breaks)
    if has_area:
        df_plot[plot_cols].plot.area(
            ax=ax,
            stacked=True,
            linewidth=0,
            color=plot_colors,
            legend=False
        )

    # Use integer x-positions for line overlays so NaN gap rows create visible gaps
    # (matching the area chart's internal integer positioning)
    x_positions = np.arange(len(df_plot))

    # Plot curtailed as dashed line
    if has_curtailed:
        curtailed = df_plot['Curtailed']
        ax.plot(x_positions, curtailed.values, linestyle='--', color='red', linewidth=1, label='Curtailed')

    # Plot demand line
    if has_demand:
        if break_times:
            inflow_plot = inflow_series
        else:
            inflow_plot = inflow_series.iloc[timeline[0]:timeline[1]]
        ax.plot(np.arange(len(inflow_plot)), inflow_plot.values, linestyle='solid', color='black', linewidth=1.5, label='Demand')

    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim:
        ax.set_ylim(ylim)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig


def plot_dispatch_area(
    df_dispatch: pd.DataFrame,
    inflow_series: pd.Series | None,
    output_path: str | Path,
    title: str,
    ylabel: str = "MWh/h",
    colors: dict | None = None,
    timeline: tuple[int, int] = (0, 168),
    show_plot: bool = False,
    ylim: tuple[float, float] | None = None,
    break_times: set[str] | None = None,
) -> None:
    """Create a stacked area dispatch plot with demand line."""
    fig = _build_dispatch_figure(
        df_dispatch, inflow_series, title,
        ylabel=ylabel, colors=colors, timeline=timeline,
        ylim=ylim, break_times=break_times,
    )
    if fig is None:
        return

    # Save (bbox_inches='tight' in savefig handles layout)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='png', bbox_inches='tight', dpi=150)

    if show_plot:
        plt.show()
    plt.close(fig)


def compute_dispatch_metadata_for_scenario(
    results: TimeSeriesResults,
    mappings: DispatchMappings,
    scenario: str,
    timeline: tuple[int, int] = (0, 168),
) -> dict:
    """Compute dispatch ylims + column order for ONE scenario.

    Returns a dict suitable for JSON serialization::

        {
            "nodeGroups": {
                "GroupName": {
                    "ylim": [ymin, ymax],
                    "columns": ["col1", "col2", ...]
                },
                ...
            }
        }

    No margin is applied here; downstream union code adds it. Call sites
    persist this to ``<project>/output_parquet/<scenario>/_dispatch_metadata.json``.
    """
    dispatch_groups_df = mappings.dispatch_groups
    node_groups: list[str] = []
    if dispatch_groups_df is not None and not dispatch_groups_df.empty:
        node_groups = list(dispatch_groups_df['group'].unique())

    meta: dict = {"nodeGroups": {}}
    for ng in node_groups:
        df_dispatch, inflow = prepare_dispatch_data(
            results, mappings, scenario, ng,
        )
        if df_dispatch is not None and not df_dispatch.empty:
            ymin, ymax = _compute_ylim(df_dispatch, timeline, inflow)
            meta["nodeGroups"][ng] = {
                "ylim": [float(ymin), float(ymax)],
                "columns": [str(c) for c in df_dispatch.columns],
            }
    return meta


def _fold_per_scenario_metadata(
    per_scen_meta: dict,
    ng_ylims: dict[str, tuple[float, float]],
    ng_columns: dict[str, list[str]],
) -> None:
    """Fold a per-scenario metadata dict into running ylims/columns dicts.

    Mutates *ng_ylims* and *ng_columns* in place with the union (min/max
    ylim, columns extended in first-seen order).  No margin is applied
    here — callers add the margin once after folding all scenarios.
    """
    for ng, entry in per_scen_meta.get("nodeGroups", {}).items():
        ylim = entry.get("ylim")
        columns = entry.get("columns") or []
        if not ylim or len(ylim) < 2:
            continue
        ymin, ymax = float(ylim[0]), float(ylim[1])
        if ng in ng_ylims:
            ng_ylims[ng] = (
                min(ng_ylims[ng][0], ymin),
                max(ng_ylims[ng][1], ymax),
            )
            for col in columns:
                if col not in ng_columns[ng]:
                    ng_columns[ng].append(col)
        else:
            ng_ylims[ng] = (ymin, ymax)
            ng_columns[ng] = list(columns)


def compute_dispatch_metadata(
    results: TimeSeriesResults,
    mappings: DispatchMappings,
    scenarios: list[str],
    timeline: tuple[int, int] = (0, 168),
) -> dict:
    """Compute cross-scenario ylims and column order per nodeGroup.

    Returns a dict suitable for JSON serialization::

        {
            "nodeGroups": {
                "GroupName": {
                    "ylim": [ymin, ymax],
                    "columns": ["col1", "col2", ...]
                },
                ...
            }
        }
    """
    ng_ylims: dict[str, tuple[float, float]] = {}
    ng_columns: dict[str, list[str]] = {}

    for scenario in scenarios:
        per_scen = compute_dispatch_metadata_for_scenario(
            results, mappings, scenario, timeline,
        )
        _fold_per_scenario_metadata(per_scen, ng_ylims, ng_columns)

    # Add margin
    for key, (ymin, ymax) in ng_ylims.items():
        margin = (ymax - ymin) * 0.05
        ng_ylims[key] = (ymin - margin, ymax + margin)

    meta: dict = {"nodeGroups": {}}
    for ng, ylim in ng_ylims.items():
        meta["nodeGroups"][ng] = {
            "ylim": list(ylim),
            "columns": ng_columns.get(ng, []),
        }
    return meta


def union_dispatch_metadata(
    project_path: Path,
    scenarios: list[str],
    margin_fraction: float = 0.05,
) -> dict:
    """Union per-scenario _dispatch_metadata.json files into a single
    cross-scenario manifest.

    Reads ``<project>/output_parquet/<scenario>/_dispatch_metadata.json``
    for each scenario in *scenarios*; missing files contribute nothing
    (silently skipped, like Phase C's availability union).

    Returns a dict matching the same schema as ``compute_dispatch_metadata``
    (with the 5% ylim margin applied at the union level), or
    ``{"nodeGroups": {}}`` if no per-scenario files exist.
    """
    import json as _json

    project_path = Path(project_path)
    ng_ylims: dict[str, tuple[float, float]] = {}
    ng_columns: dict[str, list[str]] = {}

    for scenario in scenarios:
        meta_path = (
            project_path / "output_parquet" / scenario / "_dispatch_metadata.json"
        )
        if not meta_path.is_file():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                per_scen = _json.load(f)
        except (OSError, _json.JSONDecodeError):
            continue
        if not isinstance(per_scen, dict):
            continue
        _fold_per_scenario_metadata(per_scen, ng_ylims, ng_columns)

    # Add margin at the union level
    for key, (ymin, ymax) in ng_ylims.items():
        margin = (ymax - ymin) * margin_fraction
        ng_ylims[key] = (ymin - margin, ymax + margin)

    meta: dict = {"nodeGroups": {}}
    for ng, ylim in ng_ylims.items():
        meta["nodeGroups"][ng] = {
            "ylim": list(ylim),
            "columns": ng_columns.get(ng, []),
        }
    return meta


def create_dispatch_plots(
    results: TimeSeriesResults,
    mappings: DispatchMappings,
    plot_dir: str | Path,
    scenarios: list[str],
    show_plot: bool = False,
    write_xlsx: bool = False,
    plot_rows: list[int] | tuple[int, int] | None = None,
    break_times: set[str] | None = None,
    debug: bool = False,
) -> None:
    """Create dispatch plots for all nodeGroups (and, in debug, all nodes).

    nodeGroup dispatch plots are always generated.  Per-node dispatch plots
    are generated for ALL nodes that have dispatch data (discovered from the
    results, never curated) ONLY when *debug* is true; otherwise no node
    plots are produced.  There is no persisted node selection.

    *scenarios* is the run's scenario set (the data-derived dispatch tags),
    passed in by the caller; there is no longer a ``config.yaml`` fallback.
    """
    plot_dir = Path(plot_dir)

    # Dispatch colors + stacking order now come from the project's
    # ``plot_settings.yaml`` (``entities`` for per-entity flows, new
    # ``categories.dispatch`` block for the special tokens), resolved by
    # ``resolve_dispatch_colors_and_order``.  This replaces the legacy
    # ``config['positive'|'negative']`` parsing.  The project path is the
    # parent of *plot_dir* (the comparison output dir lives under it).
    #
    # ``colors`` / ``config_order`` keep the same shape as before so the
    # downstream ``prepare_dispatch_data`` → ``_order_dispatch_columns`` →
    # render interface is unchanged.  ``colors`` is resolved once the union
    # of actual dispatch columns is known (after the first / ylim pass);
    # special-token colors and the entity ordering do not depend on the
    # data, only the per-column color resolution does.
    from flextool.plot_outputs.color_template import (
        load_color_template,
        resolve_dispatch_colors_and_order,
        resolve_plot_settings_path,
        template_entity_names,
    )

    settings_path = resolve_plot_settings_path(plot_dir.parent)
    template = load_color_template(settings_path)

    # ``config_order`` (entity stacking order) depends only on the template's
    # entity file order, not on the data, so resolve it once up front and use
    # it identically in the ylim pass and the plot pass.  ``colors`` needs the
    # actual column names (composite / node-level columns resolve to an entity
    # name), so it is resolved after the union of columns is known (below).
    # Resolving config_order from the template entity names keeps the two
    # passes byte-consistent; per-column colors are filled in afterwards.
    _, config_order = resolve_dispatch_colors_and_order(
        template, template_entity_names(template),
    )
    colors: dict[str, str] = {}

    # Time window comes from ``plot_rows`` (GUI/CLI, derived from
    # ``PlotSettings.start_time``/``duration``); a sane default covers the
    # rare path where the caller omits it (the old ``config.yaml``
    # ``time_to_plot`` source is gone).
    if plot_rows and len(plot_rows) >= 2:
        timeline = (int(plot_rows[0]), int(plot_rows[1]) + 1)  # plot_rows is inclusive
    else:
        timeline = (0, 168)

    # Get nodeGroups from data (dispatch_groups mapping), not config
    dispatch_groups_df = mappings.dispatch_groups
    node_groups: list[str] = []
    if dispatch_groups_df is not None and not dispatch_groups_df.empty:
        node_groups = list(dispatch_groups_df['group'].unique())

    excel_data: dict[str, pd.DataFrame] = {}

    # First pass: collect y-axis ranges across all scenarios for consistent scales
    ng_ylims: dict[str, tuple[float, float]] = {}
    node_ylims: dict[str, tuple[float, float]] = {}
    ng_columns: dict[str, list[str]] = {}
    node_columns: dict[str, list[str]] = {}
    # No node curation: ``config['nodes']`` is no longer consumed.  Per-node
    # dispatch is a debug-only convenience covering ALL nodes that have
    # dispatch data (discovered from the results), and is skipped entirely
    # when ``debug`` is false.
    from flextool.scenario_comparison.dispatch_data import (
        available_dispatch_nodes,
    )
    nodes = available_dispatch_nodes(results) if debug else []

    for scenario in scenarios:
        for ng in node_groups:
            df_dispatch, inflow = prepare_dispatch_data(
                results, mappings, scenario, ng,
                colors=colors, config_order=config_order,
            )
            if df_dispatch is not None and not df_dispatch.empty:
                ymin, ymax = _compute_ylim(df_dispatch, timeline, inflow)
                if ng in ng_ylims:
                    ng_ylims[ng] = (min(ng_ylims[ng][0], ymin), max(ng_ylims[ng][1], ymax))
                    # Add any new columns preserving existing order
                    for col in df_dispatch.columns:
                        if col not in ng_columns[ng]:
                            ng_columns[ng].append(col)
                else:
                    ng_ylims[ng] = (ymin, ymax)
                    ng_columns[ng] = list(df_dispatch.columns)

        for node in nodes:
            df_node, inflow_node = prepare_node_dispatch_data(
                results, scenario, node
            )
            if df_node is not None and not df_node.empty:
                ymin, ymax = _compute_ylim(df_node, timeline, inflow_node)
                if node in node_ylims:
                    node_ylims[node] = (min(node_ylims[node][0], ymin), max(node_ylims[node][1], ymax))
                    for col in df_node.columns:
                        if col not in node_columns[node]:
                            node_columns[node].append(col)
                else:
                    node_ylims[node] = (ymin, ymax)
                    node_columns[node] = list(df_node.columns)

    # Add small margin to y-axis limits
    for key, (ymin, ymax) in ng_ylims.items():
        margin = (ymax - ymin) * 0.05
        ng_ylims[key] = (ymin - margin, ymax + margin)
    for key, (ymin, ymax) in node_ylims.items():
        margin = (ymax - ymin) * 0.05
        node_ylims[key] = (ymin - margin, ymax + margin)

    # Resolve per-column colors now that the union of dispatch columns
    # (nodeGroup + node, base + split parts) is known.  Special tokens map to
    # ``categories.dispatch`` (== the old DEFAULT_SPECIAL_COLORS); entity /
    # composite / node-level columns map to ``entities``; anything unresolved
    # is left to ``_auto_assign_node_colors_with_existing``'s palette below.
    union_columns: list[str] = []
    seen_cols: set[str] = set()
    for col_list in list(ng_columns.values()) + list(node_columns.values()):
        for col in col_list:
            scol = str(col)
            if scol not in seen_cols:
                seen_cols.add(scol)
                union_columns.append(scol)
    resolved_colors, _ = resolve_dispatch_colors_and_order(template, union_columns)
    colors = resolved_colors

    for scenario in scenarios:
        print(f"Creating dispatch plots for scenario: {scenario}")

        # Plot nodeGroup dispatches
        for ng in node_groups:
            df_dispatch, inflow = prepare_dispatch_data(
                results, mappings, scenario, ng,
                colors=colors, config_order=config_order,
            )

            has_dispatch = df_dispatch is not None and not df_dispatch.empty
            has_demand = inflow is not None and not inflow.empty
            if has_dispatch or has_demand:
                if df_dispatch is None or df_dispatch.empty:
                    # Create a minimal DataFrame so plot_dispatch_area can
                    # draw the demand line even when there are no supply flows.
                    idx = inflow.index if has_demand else pd.RangeIndex(1)
                    df_dispatch = pd.DataFrame(index=idx)
                # Ensure consistent columns across scenarios for same nodeGroup.
                # Build the missing columns in one DataFrame and concat once;
                # the per-column ``df_dispatch[col] = 0.0`` loop triggered a
                # pandas PerformanceWarning about frame fragmentation when
                # ng_columns[ng] had many entries.
                if ng in ng_columns:
                    missing = [
                        c for c in ng_columns[ng]
                        if c not in df_dispatch.columns
                    ]
                    if missing:
                        zeros = pd.DataFrame(
                            0.0, index=df_dispatch.index, columns=missing,
                        )
                        df_dispatch = pd.concat(
                            [df_dispatch, zeros], axis=1,
                        )
                    df_dispatch = df_dispatch[ng_columns[ng]]
                output_path = plot_dir / f"dispatch_nodeGroup_{ng}_{scenario}.png"
                plot_dispatch_area(
                    df_dispatch, inflow, output_path,
                    title=f"{ng} - {scenario}",
                    colors=colors,
                    timeline=timeline,
                    show_plot=show_plot,
                    ylim=ng_ylims.get(ng),
                    break_times=break_times,
                )

                if write_xlsx:
                    excel_data[f"{ng}_{scenario}"] = df_dispatch
            else:
                print(f"  No dispatch data for nodeGroup {ng}")

        # Plot individual node dispatches
        for node in nodes:
            df_node, inflow_node = prepare_node_dispatch_data(
                results, scenario, node
            )
            has_node_dispatch = df_node is not None and not df_node.empty
            has_node_demand = inflow_node is not None and not inflow_node.empty
            if has_node_dispatch or has_node_demand:
                if df_node is None or df_node.empty:
                    idx = inflow_node.index if has_node_demand else pd.RangeIndex(1)
                    df_node = pd.DataFrame(index=idx)
                # Ensure consistent columns across scenarios for same node.
                # Same fragmentation fix as the nodeGroup branch above —
                # build missing zero-columns in one DataFrame + concat once
                # instead of per-column ``df_node[col] = 0.0`` assignment.
                if node in node_columns:
                    missing = [
                        c for c in node_columns[node]
                        if c not in df_node.columns
                    ]
                    if missing:
                        zeros = pd.DataFrame(
                            0.0, index=df_node.index, columns=missing,
                        )
                        df_node = pd.concat([df_node, zeros], axis=1)
                    df_node = df_node[node_columns[node]]
                # Seed with template-resolved colors (entity / special
                # tokens), then auto-assign palette colors for the rest.
                node_colors = _auto_assign_node_colors_with_existing(
                    df_node.columns, colors,
                )
                output_path = plot_dir / f"dispatch_node_{node}_{scenario}.png"
                plot_dispatch_area(
                    df_node, inflow_node, output_path,
                    title=f"{node} - {scenario}",
                    colors=node_colors,
                    timeline=timeline,
                    show_plot=show_plot,
                    ylim=node_ylims.get(node),
                    break_times=break_times,
                )
                if write_xlsx:
                    excel_data[f"node_{node}_{scenario}"] = df_node
            else:
                print(f"  No dispatch data for node {node}")

    # Write Excel file
    if write_xlsx and excel_data:
        excel_path = plot_dir / "dispatch_data.xlsx"
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            for name, df in excel_data.items():
                sheet_name = name[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=sheet_name)
        print(f"Wrote dispatch data to {excel_path}")
