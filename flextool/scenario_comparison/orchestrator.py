"""Top-level orchestration: ties together data loading, config, and plotting."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

from flextool.scenario_comparison.config_builder import (
    create_or_update_dispatch_config,
    get_scenarios_from_config,
)
from flextool.scenario_comparison.data_models import DispatchMappings
from flextool.scenario_comparison.db_reader import get_scenario_results
from flextool.scenario_comparison.dispatch_mappings import combine_dispatch_mappings
from flextool.scenario_comparison.dispatch_plots import create_dispatch_plots
from flextool.scenario_comparison.summary_plots import create_basic_plots
from flextool.plot_outputs.plot_functions import plot_dict_of_dataframes


def run(
    db_url: str,
    parquet_subdir: str,
    plot_dir: str,
    output_config_path: str,
    active_configs: list[str],
    plot_rows: list[int],
    write_to_xlsx: bool,
    write_dispatch_xlsx: bool,
    write_to_ods: bool,
    show_plots: bool,
    dispatch_plots: bool,
    basic_plots: bool,
    plot_file_format: str = 'png',
) -> None:
    """Run the full scenario-comparison pipeline.

    Takes already-resolved parameters (CLI > settings DB > defaults)
    and orchestrates: load data → build config → generate plots → write Excel.
    """
    with open(output_config_path, 'r') as f:
        settings = yaml.safe_load(f)

    scenario_folders, results = get_scenario_results(
        db_url=db_url, parquet_subdir=parquet_subdir
    )
    combined_dfs = results.to_dict()

    os.makedirs(plot_dir, exist_ok=True)

    scenarios = list(scenario_folders.keys())

    # Load and combine dispatch mappings across all scenarios
    if scenario_folders:
        mappings = combine_dispatch_mappings(scenario_folders, parquet_subdir)
        combined_mapping_dfs = {
            k: v for k, v in vars(mappings).items() if v is not None
        }
    else:
        mappings = DispatchMappings()
        combined_mapping_dfs = {}

    # Derive group_node_df for summary plots (needs 'scenario' as column)
    group_node_df = None
    group_node_combined = combined_mapping_dfs.get('group_node')
    if group_node_combined is not None and not group_node_combined.empty:
        group_node_df = group_node_combined.reset_index()

    # Create or update dispatch config
    dispatch_config = None
    if dispatch_plots or basic_plots:
        dispatch_config = create_or_update_dispatch_config(
            plot_dir, results, scenarios, mappings
        )

    # Generate original comparison plots (from default_comparison_plots.yaml)
    plot_dict_of_dataframes(
        combined_dfs, plot_dir, settings['plots'],
        active_settings=active_configs, plot_rows=plot_rows,
        delete_existing_plots=True, plot_file_format=plot_file_format,
    )
    print(f'\nPlotted comparison of {len(scenario_folders)} scenarios to folder: {plot_dir}')

    # Generate dispatch plots
    if dispatch_plots:
        if dispatch_config and combined_mapping_dfs:
            print("\nGenerating dispatch plots...")
            create_dispatch_plots(
                results, mappings, dispatch_config, plot_dir,
                scenarios=get_scenarios_from_config(dispatch_config),
                show_plot=show_plots,
                write_xlsx=write_dispatch_xlsx,
            )
        else:
            print("Warning: Cannot generate dispatch plots - missing dispatch mappings")

    # Generate summary plots
    if basic_plots:
        if dispatch_config:
            print("\nGenerating summary plots...")
            create_basic_plots(
                results, group_node_df, dispatch_config, plot_dir,
                scenarios=get_scenarios_from_config(dispatch_config),
                show_plot=show_plots,
            )

    # Write to Excel (combined results)
    if write_to_xlsx:
        excel_dir = 'output_excel_comparison'
        os.makedirs(excel_dir, exist_ok=True)
        filename = 'compare_' + str(len(scenario_folders)) + '_scens.xlsx'
        excel_path = os.path.join(excel_dir, filename)
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            used_names: set[str] = set()
            for name, df in combined_dfs.items():
                if (not df.empty) & (len(df) > 0):
                    sheet_name = name[:31]
                    if sheet_name in used_names:
                        suffix = 1
                        while f"{sheet_name[:28]}_{suffix}" in used_names:
                            suffix += 1
                        sheet_name = f"{sheet_name[:28]}_{suffix}"
                    used_names.add(sheet_name)
                    df.to_excel(writer, sheet_name=sheet_name)
        print(f'\nWrote comparison of {len(scenario_folders)} scenarios to xlsx file: {excel_path}')

    print('\nDone!')
