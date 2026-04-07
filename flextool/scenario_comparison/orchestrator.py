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
from flextool.plot_outputs.config import _is_single_config
from flextool.plot_outputs.plot_functions import plot_dict_of_dataframes


def run(
    db_url: str | None,
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
    plot_file_format: str = 'png',
    scenario_folders: dict[str, str] | None = None,
    excel_dir: str | None = None,
    shared_legend: bool = True,
    only_first_file: bool = False,
    comparison_parquet_dir: str | None = None,
) -> None:
    """Run the full scenario-comparison pipeline.

    Takes already-resolved parameters (CLI > settings DB > defaults)
    and orchestrates: load data → build config → generate plots → write Excel.

    When *scenario_folders* is provided the database is not queried and
    *db_url* may be ``None``.
    """
    with open(output_config_path, 'r') as f:
        settings = yaml.safe_load(f)

    scenario_folders, results = get_scenario_results(
        db_url=db_url, parquet_subdir=parquet_subdir,
        scenario_folders=scenario_folders,
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
    if dispatch_plots:
        dispatch_config = create_or_update_dispatch_config(
            plot_dir, results, scenarios, mappings
        )

    # If shared_legend is disabled, replace 'shared' legend with 'right' in all plot configs
    if not shared_legend:
        for result_name, config_dict in settings['plots'].items():
            if not isinstance(config_dict, dict):
                continue
            if _is_single_config(config_dict):
                # Single config: the dict itself is the plot settings
                if config_dict.get('legend') == 'shared':
                    config_dict['legend'] = 'right'
            else:
                # Named configs: iterate sub-dicts
                for config_name, plot_cfg in config_dict.items():
                    if isinstance(plot_cfg, dict) and plot_cfg.get('legend') == 'shared':
                        plot_cfg['legend'] = 'right'

    # Load timeline breaks from all scenario parquet dirs.
    # For --parquet-base-dir mode (subdir=''): parquets at base_dir/name/
    # For DB mode (subdir='output_parquet'): parquets at folder/subdir/name/
    from flextool.plot_outputs.format_helpers import load_timeline_breaks
    break_dirs = []
    for name, folder in scenario_folders.items():
        if parquet_subdir:
            break_dirs.append(os.path.join(folder, parquet_subdir, name))
        else:
            break_dirs.append(os.path.join(folder, name))
    break_times = load_timeline_breaks(*break_dirs)

    # Write combined comparison parquets if output directory specified
    if comparison_parquet_dir:
        os.makedirs(comparison_parquet_dir, exist_ok=True)
        for name, df in combined_dfs.items():
            if not df.empty:
                df.to_parquet(os.path.join(comparison_parquet_dir, f"{name}.parquet"))
        # Also write a metadata file with the scenario list
        import json
        meta = {"scenarios": scenarios}
        with open(os.path.join(comparison_parquet_dir, "_metadata.json"), "w") as f:
            json.dump(meta, f)
        if break_times:
            # Save break times so the viewer can load them
            bt_df = pd.DataFrame({"break_time": list(break_times)})
            bt_df.to_parquet(os.path.join(comparison_parquet_dir, "timeline_breaks.parquet"))
        print(f"Wrote comparison parquets to: {comparison_parquet_dir}")

    # Compute plot plans for the viewer
    plan_output_dir = comparison_parquet_dir if comparison_parquet_dir else plot_dir
    try:
        from flextool.plot_outputs.orchestrator import compute_all_plot_plans
        compute_all_plot_plans(
            combined_dfs, settings.get('plots', {}), plan_output_dir,
            active_settings=active_configs, plot_rows=plot_rows,
            break_times=break_times,
        )
        print("Computed plot plans for viewer")
    except Exception as exc:
        import logging as _logging
        _logging.warning("Plot plan computation failed (non-fatal): %s", exc)

    # Generate original comparison plots (from default_comparison_plots.yaml)
    plot_dict_of_dataframes(
        combined_dfs, plot_dir, settings['plots'],
        active_settings=active_configs, plot_rows=plot_rows,
        delete_existing_plots=True, plot_file_format=plot_file_format,
        only_first_file=only_first_file,
        break_times=break_times,
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
                break_times=break_times,
                plot_rows=plot_rows,
            )
        else:
            print("Warning: Cannot generate dispatch plots - missing dispatch mappings")

    # Write to Excel (combined results)
    if write_to_xlsx:
        filename = 'compare_' + str(len(scenario_folders)) + '_scens.xlsx'
        target_dir = excel_dir if excel_dir is not None else plot_dir
        excel_path = os.path.join(target_dir, filename)
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
