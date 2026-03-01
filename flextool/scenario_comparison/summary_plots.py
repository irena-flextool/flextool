"""Summary bar chart plots comparing scenarios."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from flextool.scenario_comparison.config_builder import get_scenarios_from_config
from flextool.scenario_comparison.data_models import TimeSeriesResults
from flextool.scenario_comparison.dispatch_mappings import get_group_node_multiindex


def reindex_scenarios(df: pd.DataFrame | None, scen_list: list[str]) -> pd.DataFrame | None:
    """Reorder DataFrame columns to match scenario order."""
    if df is None or df.empty:
        return df
    try:
        available = [s for s in scen_list if s in df.columns.get_level_values('scenario')]
        if available:
            return df.reindex(available, axis=1, level='scenario')
    except (KeyError, ValueError):
        pass
    return df


def plot_horizontal_bar(
    df: pd.DataFrame,
    filename: str | None = None,
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
    show_plot: bool = False,
    subplot: str | None = None,
    stacked: bool | None = None,
    sum_index_level: int | None = None,
    n_subplot_cols: int = 1,
    xlabel: str | None = None,
    ylabel: str | None = None,
    max_items: int = 20,
) -> None:
    if sum_index_level is not None:
        df = df.groupby(level=sum_index_level).sum()

    # Split into multiple files if too many items (rows)
    all_items = df.index.tolist()
    n_items = len(all_items)
    needs_split = n_items > max_items

    if needs_split:
        chunks = [all_items[i:i + max_items] for i in range(0, n_items, max_items)]
    else:
        chunks = [all_items]

    for chunk_idx, item_chunk in enumerate(chunks, start=1):
        df_chunk = df.loc[item_chunk]

        # Scale figsize height proportionally to number of items in this chunk
        chunk_figsize = (figsize[0], figsize[1] * len(item_chunk) / min(n_items, max_items)) if figsize else figsize

        n_subplots = 1
        subplot_names = ['']
        if subplot is not None:
            subplot_names = df_chunk.columns.get_level_values(level=subplot).unique()
            n_subplots = len(subplot_names)
        n_subplot_rows = math.ceil(n_subplots / n_subplot_cols)
        fig, axes = plt.subplots(nrows=n_subplot_rows, ncols=n_subplot_cols, figsize=chunk_figsize, squeeze=False)
        axes = axes.flatten()

        for i, subplot_name in enumerate(subplot_names):
            if isinstance(df_chunk.columns, pd.MultiIndex):
                df_sub = df_chunk.xs(subplot_name, axis=1, level=subplot)
            else:
                df_sub = df_chunk
            ax = axes[i]
            df_sub.plot.barh(ax=ax, legend=False, title=subplot_name, xlabel=xlabel)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')

        chunk_title = f"{title} ({chunk_idx}/{len(chunks)})" if title and needs_split else title
        if chunk_title:
            fig.suptitle(chunk_title, fontweight='bold')
        plt.tight_layout()
        if filename:
            if needs_split:
                base, ext = os.path.splitext(filename)
                chunk_filename = f"{base}_{chunk_idx}{ext}"
            else:
                chunk_filename = filename
            plt.savefig(chunk_filename, bbox_inches='tight')
        if show_plot:
            plt.show()
        plt.close()


def create_basic_plots(
    results: TimeSeriesResults,
    group_node_df: pd.DataFrame | None,
    config: dict,
    plot_dir: Path | str,
    scenarios: list[str] | None = None,
    show_plot: bool = False,
) -> None:
    """Create summary bar chart plots comparing scenarios.

    Parameters
    ----------
    results : TimeSeriesResults
        Combined time-series result DataFrames.
    group_node_df : pd.DataFrame | None
        Node-to-group mapping with 'scenario' as a column.
    config : dict
        Dispatch configuration dict.
    plot_dir : Path or str
        Directory to save plots.
    scenarios : list[str] | None
        List of scenarios to include.
    show_plot : bool
        Whether to display plots.
    """
    plot_dir = Path(plot_dir)

    if scenarios is None:
        scenarios = get_scenarios_from_config(config)

    nodes = config.get('nodes', [])

    # 1. Generation by type (if nodeGroup_flows_d_gpe available)
    if results.nodeGroup_flows_d_gpe is not None:
        try:
            df = results.nodeGroup_flows_d_gpe.copy()
            # Filter to electricity/main group and sum by type
            df_type_TWh = df.stack('item', future_stack=True).groupby('item').sum().T.groupby('scenario').sum().T.div(1000000)
            df_type_TWh = reindex_scenarios(df_type_TWh, scenarios)

            if not df_type_TWh.empty:
                plot_horizontal_bar(
                    df_type_TWh,
                    filename=str(plot_dir / 'generation_by_type.png'),
                    title='Generation by type',
                    figsize=(6, 6),
                    xlabel="TWh",
                    show_plot=show_plot
                )
        except Exception as e:
            print(f"Could not create generation by type plot: {e}")

    # 2. Loss of load plots
    if results.node_slack_up_dt_e is not None and group_node_df is not None:
        try:
            df_lol = results.node_slack_up_dt_e.copy()
            df_lol = reindex_scenarios(df_lol, scenarios)

            # Get group_node mapping for first available scenario
            first_scen = scenarios[0] if scenarios else None
            if first_scen:
                group_node = get_group_node_multiindex(group_node_df, first_scen)
                if group_node is not None:
                    # Filter to nodes in nodeGroups
                    valid_nodes = group_node.get_level_values('node')
                    df_lol_filtered = df_lol.loc[:, df_lol.columns.get_level_values('node').isin(valid_nodes)]

                    if not df_lol_filtered.empty:
                        # LoL per nodeGroup — stack to long format and merge with group mapping
                        # (avoids index join which breaks when nodes belong to multiple groups)
                        df_summed = df_lol_filtered.groupby('period').sum()
                        df_long = df_summed.stack(list(range(len(df_summed.columns.names))), future_stack=True).rename('value').reset_index()
                        node_col = next(c for c in df_long.columns if c not in ('period', 'scenario', 'value'))
                        gn = group_node_df[group_node_df['scenario'] == first_scen][['group', 'node']].drop_duplicates()
                        df_merged = df_long.merge(gn, left_on=node_col, right_on='node', how='inner')
                        df_lol_sum = df_merged.groupby(['group', 'period', 'scenario'])['value'].sum().unstack('scenario')
                        df_lol_sum = reindex_scenarios(df_lol_sum, scenarios)

                        plot_horizontal_bar(
                            df_lol_sum.div(1000000),
                            filename=str(plot_dir / 'lol_TWh_nodeGroups.png'),
                            title='Loss of load by nodeGroup',
                            figsize=(5, 4),
                            sum_index_level=0,
                            xlabel="TWh",
                            show_plot=show_plot
                        )
        except Exception as e:
            print(f"Could not create loss of load plots: {e}")

    # 3. VRE share plots
    if results.nodeGroup_gd_p is not None:
        try:
            df_vre = results.nodeGroup_gd_p.copy()
            if 'vre_share' in df_vre.columns.get_level_values(1):
                df_vre_share = df_vre.xs('vre_share', axis=1, level=1).groupby('group').sum()
                df_vre_share = reindex_scenarios(df_vre_share, scenarios)

                if not df_vre_share.empty:
                    plot_horizontal_bar(
                        df_vre_share * 100,
                        filename=str(plot_dir / 'vre_share_nodeGroups.png'),
                        title='VRE share by nodeGroup',
                        figsize=(5, 4),
                        xlabel="%",
                        show_plot=show_plot
                    )
        except Exception as e:
            print(f"Could not create VRE share plots: {e}")

    # 4. Curtailment plots
    if results.unit_curtailment_outputNode_dt_ee is not None:
        try:
            df_curtail = results.unit_curtailment_outputNode_dt_ee.copy()
            df_curtail = reindex_scenarios(df_curtail, scenarios)

            # Curtailment per node (column level may be 'sink' or 'node')
            node_level = 'sink' if 'sink' in df_curtail.columns.names else 'node'
            curtail_by_node = df_curtail.sum(axis=0).groupby([node_level, 'scenario']).sum().unstack('scenario')

            # Filter to configured nodes
            if nodes:
                curtail_by_node = curtail_by_node.loc[curtail_by_node.index.isin(nodes)]

            if not curtail_by_node.empty:
                plot_horizontal_bar(
                    curtail_by_node.div(1000000),
                    filename=str(plot_dir / 'curtailment_TWh_nodes.png'),
                    title='Curtailment by node',
                    figsize=(5, 4),
                    xlabel="TWh",
                    show_plot=show_plot
                )
        except Exception as e:
            print(f"Could not create curtailment plots: {e}")

    print(f"Summary plots saved to {plot_dir}")
