import pandas as pd
import os
import math
from pathlib import Path
import matplotlib.pyplot as plt


def prepare_dispatch_data(combined_dfs, combined_mapping_dfs, scenario, output_node_group,
                          colors=None):
    """Backward-compat wrapper — delegates to dispatch_data module."""
    from flextool.scenario_comparison.dispatch_data import prepare_dispatch_data as _prepare
    from flextool.scenario_comparison.data_models import TimeSeriesResults, DispatchMappings
    results = TimeSeriesResults.from_dict(combined_dfs)
    mappings = DispatchMappings(**{k: v for k, v in combined_mapping_dfs.items()
                                   if hasattr(DispatchMappings, k)})
    return _prepare(results, mappings, scenario, output_node_group, colors=colors)


def prepare_node_dispatch_data(combined_dfs, scenario: str, node: str):
    """Backward-compat wrapper — delegates to dispatch_data module."""
    from flextool.scenario_comparison.dispatch_data import prepare_node_dispatch_data as _prepare
    from flextool.scenario_comparison.data_models import TimeSeriesResults
    results = TimeSeriesResults.from_dict(combined_dfs)
    return _prepare(results, scenario, node)


def create_basic_plots(combined_dfs, group_node_df, config, plot_dir, scenarios=None, show_plot=False):
    """
    Create summary bar chart plots comparing scenarios.

    Parameters:
    -----------
    combined_dfs : dict
        Combined result dataframes
    group_node_df : pd.DataFrame
        Node-to-group mapping
    config : dict
        Plot configuration
    plot_dir : str or Path
        Directory to save plots
    scenarios : list, optional
        List of scenarios to include
    show_plot : bool
        Whether to display plots
    """
    plot_dir = Path(plot_dir)

    if scenarios is None:
        scenarios = get_scenarios_from_config(config)

    nodes = config.get('nodes', [])

    # Reorder dataframes to match scenario order
    def reindex_scenarios(df, scen_list):
        if df is None or df.empty:
            return df
        try:
            available = [s for s in scen_list if s in df.columns.get_level_values('scenario')]
            if available:
                return df.reindex(available, axis=1, level='scenario')
        except (KeyError, ValueError):
            pass
        return df

    # 1. Generation by type (if nodeGroup_flows_d_gpe available)
    if 'nodeGroup_flows_d_gpe' in combined_dfs:
        try:
            df = combined_dfs['nodeGroup_flows_d_gpe'].copy()
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
    if 'node_slack_up_dt_e' in combined_dfs and group_node_df is not None:
        try:
            df_lol = combined_dfs['node_slack_up_dt_e'].copy()
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
    if 'nodeGroup_gd_p' in combined_dfs:
        try:
            df_vre = combined_dfs['nodeGroup_gd_p'].copy()
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
    if 'unit_curtailment_outputNode_dt_ee' in combined_dfs:
        try:
            df_curtail = combined_dfs['unit_curtailment_outputNode_dt_ee'].copy()
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


def plot_horizontal_bar(df, filename=None, title=None, figsize=(10, 6), show_plot=False, subplot=None, stacked=None, sum_index_level=None, n_subplot_cols=1, xlabel=None, ylabel=None, max_items: int = 20):
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
            _ = df_sub.plot.barh(ax=ax, legend=False, title=subplot_name, xlabel=xlabel)

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
    return ax


from flextool.scenario_comparison.db_reader import (  # noqa: E402
    read_scenario_folders,
    collect_parquet_files,
    combine_parquet_files,
    get_scenario_results,
)

from flextool.scenario_comparison.dispatch_mappings import (  # noqa: E402
    load_dispatch_mappings,
    combine_dispatch_mappings,
    get_group_node_multiindex,
)

from flextool.scenario_comparison.config_builder import (  # noqa: E402
    get_scenarios_from_config,
    compute_process_group_std_order,
    create_or_update_dispatch_config,
)

from flextool.scenario_comparison.dispatch_plots import (  # noqa: E402
    create_dispatch_plots,
    plot_dispatch_area,
)
