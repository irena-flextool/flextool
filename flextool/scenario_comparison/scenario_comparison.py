import pandas as pd


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
