"""CLI entry point for scenario comparison: argparse + settings resolution → orchestrator."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
from spinedb_api import DatabaseMapping, from_database, Array
from spinedb_api.filters.alternative_filter import alternative_filter_config
from spinedb_api.filters.tools import append_filter_config

from flextool.scenario_comparison import orchestrator
from flextool.scenario_comparison.db_reader import build_scenario_folders_from_dir


def main() -> None:
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(
        description='Read and combine scenario results from multiple folders based on database information')
    parser.add_argument(
        'db_url', nargs='?', default=None,
        help='Database URL containing scenario information (e.g., sqlite:///scenarios.db)')
    parser.add_argument(
        '--parquet-base-dir', default=None,
        help='Base directory containing per-scenario parquet subdirectories '
             '(alternative to db_url; requires --alternatives)')
    parser.add_argument(
        '--parquet-subdir',
        default='output_parquet',
        help='Subdirectory containing parquet files (default: output_parquet)')
    parser.add_argument('--settings-db-url', default=None,
                        help='Settings database URL (fills in unset params; CLI overrides DB)')
    parser.add_argument(
        '--output-config-path', default=None,
        help='Path to output configuration YAML file (default: templates/default_comparison_plots.yaml)'
    )
    parser.add_argument('--active-configs', type=str, default=None, nargs="+",
                        help='Which plot configurations from config_path yaml to use (default: default)')
    parser.add_argument('--plot-rows', type=int, nargs=2, default=None,
                        help='First and last row to plot in time series (default: 0 167)')
    parser.add_argument('--write-to-xlsx', action='store_true', default=None,
                        help='Write combined results to Excel file')
    parser.add_argument('--write-dispatch-xlsx', action='store_true', default=None,
                        help='Write dispatch data to Excel file in plot directory')
    parser.add_argument('--write-to-ods', action='store_true', default=None)
    parser.add_argument(
        '--alternatives', metavar='S', type=str, nargs='+',
        help='Add alternative names manually')
    parser.add_argument(
        '--plot-dir', default='output_plot_comparisons',
        help='Directory to plot scenario comparison plots'
    )
    parser.add_argument(
        '--dispatch-plots', action='store_true',
        help='Generate dispatch area plots for nodes and nodeGroups'
    )
    parser.add_argument(
        '--show-plots', action='store_true', default=None,
        help='Display plots interactively (in addition to saving)'
    )
    parser.add_argument(
        '--plot-file-format', type=str, default=None,
        choices=['png', 'svg'],
        help='File format for plot output (default: png)'
    )
    parser.add_argument(
        '--excel-dir', default=None,
        help='Directory to write comparison Excel files (default: same as --plot-dir)'
    )
    parser.add_argument(
        '--shared-legend', action='store_true', default=None,
        help='Use shared legend across subplots (default: true)'
    )
    parser.add_argument(
        '--only-first-file-per-plot', action='store_true', default=False,
        help='Only produce the first file for each plot (quick overview mode)'
    )

    args = parser.parse_args()
    db_url = args.db_url
    parquet_base_dir = args.parquet_base_dir

    # Validate: need at least one data source
    if db_url is None and parquet_base_dir is None:
        parser.error('Either db_url or --parquet-base-dir must be provided')
    # Validate: --parquet-base-dir requires --alternatives
    if parquet_base_dir is not None and not args.alternatives:
        parser.error('--parquet-base-dir requires --alternatives to specify scenario names')

    # Resolve parameters: CLI args > settings DB > hardcoded defaults
    output_config_path = args.output_config_path
    active_configs = args.active_configs
    plot_rows = args.plot_rows
    write_to_xlsx = args.write_to_xlsx
    write_dispatch_xlsx = args.write_dispatch_xlsx
    write_to_ods = args.write_to_ods
    show_plots = args.show_plots
    plot_file_format = args.plot_file_format
    shared_legend = args.shared_legend

    settings_db_url = args.settings_db_url
    if settings_db_url and os.path.exists(settings_db_url.replace('sqlite:///', '')):
        with DatabaseMapping(settings_db_url) as settings_db:
            settings_entities = settings_db.get_entity_items(entity_class_name="settings")
            if len(settings_entities) == 1:
                settings_name = settings_entities[0]["name"]
                settings_params: dict = {}
                for pv in settings_db.get_parameter_value_items(entity_class_name="settings"):
                    if pv["entity_byname"] == (settings_name,):
                        settings_params[pv["parameter_definition_name"]] = from_database(pv["value"], pv["type"])

                if output_config_path is None and 'output-config-path' in settings_params:
                    output_config_path = str(settings_params['output-config-path'])

                if active_configs is None and 'active-output-configs' in settings_params:
                    val = settings_params['active-output-configs']
                    if isinstance(val, str):
                        active_configs = [val]
                    elif isinstance(val, Array):
                        active_configs = list(val.values)
                    else:
                        active_configs = list(val)

                if plot_rows is None:
                    first = settings_params.get('plot_first_timestep')
                    duration = settings_params.get('plot_duration')
                    if first is not None and duration is not None:
                        plot_rows = [int(first), int(first) + int(duration)]

                if write_to_xlsx is None and 'write-to-excel' in settings_params:
                    write_to_xlsx = bool(settings_params['write-to-excel'])
                if write_dispatch_xlsx is None and 'write-dispatch-to-excel' in settings_params:
                    write_dispatch_xlsx = bool(settings_params['write-dispatch-to-excel'])
                if write_to_ods is None and 'write-to-ods' in settings_params:
                    write_to_ods = bool(settings_params['write-to-ods'])
                if show_plots is None and 'show-plots' in settings_params:
                    show_plots = bool(settings_params['show-plots'])
                if plot_file_format is None and 'plot-file-format' in settings_params:
                    plot_file_format = str(settings_params['plot-file-format'])
                if shared_legend is None and 'shared-legend' in settings_params:
                    shared_legend = bool(settings_params['shared-legend'])

    # Apply hardcoded defaults for anything still unset
    if output_config_path is None:
        output_config_path = 'templates/default_comparison_plots.yaml'
    if active_configs is None:
        active_configs = ['default']
    if plot_rows is None:
        plot_rows = [0, 167]
    if write_to_xlsx is None:
        write_to_xlsx = False
    if write_dispatch_xlsx is None:
        write_dispatch_xlsx = False
    if write_to_ods is None:
        write_to_ods = False
    if show_plots is None:
        show_plots = False
    if plot_file_format is None:
        plot_file_format = 'png'
    if shared_legend is None:
        shared_legend = True

    # Build scenario-to-folder mapping from either --parquet-base-dir or db_url
    alternatives = args.alternatives
    pre_built_folders: dict[str, str] | None = None
    parquet_subdir = args.parquet_subdir

    if parquet_base_dir is not None:
        # Direct parquet directory mode: skip database, build mapping from filesystem
        pre_built_folders = build_scenario_folders_from_dir(
            Path(parquet_base_dir), alternatives
        )
        if not pre_built_folders:
            print("Error: no valid scenario directories found under "
                  f"{parquet_base_dir}", file=sys.stderr)
            sys.exit(1)
        # With --parquet-base-dir the parquets sit directly in base_dir/scenario/,
        # so the extra subdirectory level is not needed.
        parquet_subdir = ''
    elif alternatives and db_url is not None:
        # Database mode with alternative filters
        alternative_filter = alternative_filter_config(alternatives)
        db_url = append_filter_config(db_url, alternative_filter)

    # Resolve dispatch_plots flags
    do_dispatch = args.dispatch_plots

    orchestrator.run(
        db_url=db_url,
        parquet_subdir=parquet_subdir,
        plot_dir=args.plot_dir,
        output_config_path=output_config_path,
        active_configs=active_configs,
        plot_rows=plot_rows,
        write_to_xlsx=write_to_xlsx,
        write_dispatch_xlsx=write_dispatch_xlsx,
        write_to_ods=write_to_ods,
        show_plots=show_plots,
        dispatch_plots=do_dispatch,
        plot_file_format=plot_file_format,
        scenario_folders=pre_built_folders,
        excel_dir=args.excel_dir,
        shared_legend=shared_legend,
        only_first_file=args.only_first_file_per_plot,
    )


if __name__ == '__main__':
    main()
