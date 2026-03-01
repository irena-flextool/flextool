import matplotlib
from spinedb_api import DatabaseMapping, from_database, Array
from spinedb_api.filters.alternative_filter import alternative_filter_config
from spinedb_api.filters.tools import append_filter_config
from flextool.scenario_comparison.db_reader import get_scenario_results
from flextool.scenario_comparison.dispatch_mappings import combine_dispatch_mappings
from flextool.scenario_comparison.config_builder import (
    create_or_update_dispatch_config,
    get_scenarios_from_config,
)
from flextool.scenario_comparison.data_models import DispatchMappings
from flextool.scenario_comparison.dispatch_plots import create_dispatch_plots
from flextool.scenario_comparison.scenario_comparison import (
    create_basic_plots,
)
from flextool.plot_outputs.plot_functions import plot_dict_of_dataframes
import os
import yaml


def main():

        import argparse
        matplotlib.use('Agg')
        parser = argparse.ArgumentParser(
            description='Read and combine scenario results from multiple folders based on database information')
        parser.add_argument(
            'db_url',
            help='Database URL containing scenario information (e.g., sqlite:///scenarios.db)')
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
            '--basic-plots', action='store_true',
            help='Generate basic comparison plots'
        )
        parser.add_argument(
            '--all-plots', action='store_true',
            help='Generate all plot types (dispatch and summary)'
        )
        parser.add_argument(
            '--show-plots', action='store_true', default=None,
            help='Display plots interactively (in addition to saving)'
        )

        args = parser.parse_args()
        db_url = args.db_url
        alternatives = args.alternatives
        plot_dir = args.plot_dir

        # Resolve parameters: CLI args > settings DB > hardcoded defaults
        output_config_path = args.output_config_path
        active_configs = args.active_configs
        plot_rows = args.plot_rows
        write_to_xlsx = args.write_to_xlsx
        write_dispatch_xlsx = args.write_dispatch_xlsx
        write_to_ods = args.write_to_ods
        show_plots = args.show_plots

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

        if alternatives:
            alternative_filter = alternative_filter_config(alternatives)
            db_url = append_filter_config(db_url, alternative_filter)

        with open(output_config_path, 'r') as f:
            settings = yaml.safe_load(f)

        scenario_folders, results = get_scenario_results(db_url=db_url, parquet_subdir=args.parquet_subdir)
        combined_dfs = results.to_dict()  # dict view until downstream functions accept TimeSeriesResults

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        # Get list of scenarios
        scenarios = list(scenario_folders.keys())

        # Load and combine dispatch mappings across all scenarios
        if scenario_folders:
            mappings = combine_dispatch_mappings(scenario_folders, args.parquet_subdir)
            combined_mapping_dfs = {k: v for k, v in vars(mappings).items() if v is not None}  # dict view until downstream functions accept DispatchMappings
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
        if args.dispatch_plots or args.basic_plots or args.all_plots:
            dispatch_config = create_or_update_dispatch_config(
                plot_dir, results, scenarios, mappings
            )

        # Generate original comparison plots (from default_comparison_plots.yaml)
        plot_dict_of_dataframes(combined_dfs, plot_dir, settings['plots'], active_settings=active_configs, plot_rows=plot_rows, delete_existing_plots=True)
        print(f'\nPlotted comparison of {len(scenario_folders)} scenarios to folder: {plot_dir}')

        # Generate dispatch plots
        if args.dispatch_plots or args.all_plots:
            if dispatch_config and combined_mapping_dfs:
                print("\nGenerating dispatch plots...")
                create_dispatch_plots(
                    results, mappings, dispatch_config, plot_dir,
                    scenarios=get_scenarios_from_config(dispatch_config),
                    show_plot=show_plots,
                    write_xlsx=write_dispatch_xlsx
                )
            else:
                print("Warning: Cannot generate dispatch plots - missing dispatch mappings")

        # Generate basic plots
        if args.basic_plots or args.all_plots:
            if dispatch_config:
                print("\nGenerating summary plots...")
                create_basic_plots(
                    combined_dfs, group_node_df, dispatch_config, plot_dir,
                    scenarios=get_scenarios_from_config(dispatch_config),
                    show_plot=show_plots
                )

        # Write to excel (combined results)
        if write_to_xlsx:
            excel_dir = 'output_excel_comparison'
            os.makedirs(excel_dir, exist_ok=True)
            filename = 'compare_' + str(len(scenario_folders)) + '_scens.xlsx'
            excel_path = os.path.join(excel_dir, filename)
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                used_names = set()
                for name, df in combined_dfs.items():
                    if (not df.empty) & (len(df) > 0):
                        # Excel sheet names limited to 31 characters
                        sheet_name = name[:31]
                        # Handle duplicates from truncation
                        if sheet_name in used_names:
                            suffix = 1
                            while f"{sheet_name[:28]}_{suffix}" in used_names:
                                suffix += 1
                            sheet_name = f"{sheet_name[:28]}_{suffix}"
                        used_names.add(sheet_name)
                        df.to_excel(writer, sheet_name=sheet_name)

            print(f'\nWrote comparison of {len(scenario_folders)} scenarios to xlsx file: {excel_path}')

        print('\nDone!')

if __name__ == '__main__':
    main()
