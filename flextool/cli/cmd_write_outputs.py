from flextool.process_outputs.result_writer import write_outputs
from spinedb_api import DatabaseMapping
from spinedb_api.filters.tools import name_from_dict
import logging
import sys

def main():

        import argparse

        parser = argparse.ArgumentParser(description='Write FlexTool outputs to various formats')
        parser.add_argument('--input-db-url', type=str, help='URL of the input database with scenario filter (used when run_flextool.py calls this from Toolbox)')
        parser.add_argument('--scenario-name', type=str, help='Name of a scenario that must have raw outputs available (when re-plotting single scenario from terminal)')
        parser.add_argument('--output-locations-db-url', type=str, help='URL of the database that holds the locations of existing outputs (for re-plotting from Toolbox)')
        parser.add_argument('--settings-db-url', type=str, default=None,
                            help='URL of the settings database (fills in unset params)')
        parser.add_argument('--config-path', type=str, default=None,
                            help='Path to output configuration YAML file (default: templates/default_plots.yaml)')
        parser.add_argument('--active-configs', type=str, nargs='+', default=None,
                            help='Which plot configurations from config_path yaml to use (default: default)')
        parser.add_argument('--output-location', type=str, default=None,
                            help='Directory for the root for input and output locations (default: flextool root)')
        parser.add_argument('--subdir', type=str, default=None,
                            help='Subdirectory for outputs (default: scenario name)')
        parser.add_argument('--read-parquet-dir', type=str, default=False,
                            help='Directory to read existing parquet files from (default: False, reads from raw CSV files)')
        parser.add_argument('--write-methods', type=str, nargs='+', default=None,
                            choices=['plot', 'parquet', 'excel', 'db', 'csv'],
                            help='Output methods to use (default: plot parquet excel)')
        parser.add_argument('--plot-rows', type=int, nargs=2, default=None,
                            help='First and last row to plot in time series (default: 0 167)')
        parser.add_argument('--debug', action='store_true',
                            help='Enable debug output')
        parser.add_argument('--single-result', type=str, nargs=6,
                            metavar=('KEY', 'CSV_NAME', 'PLOT_NAME', 'PLOT_TYPE', 'SUBPLOTS_PER_ROW', 'LEGEND_POSITION'),
                            help='Process a single result (overrides --config): key csv_name plot_name plot_type subplots_per_row legend_position. Use "null" for None values.')

        args = parser.parse_args()
        input_db_url = args.input_db_url
        output_locations_db_url = args.output_locations_db_url
        if args.scenario_name:
            scenario_names = [args.scenario_name]
        else:
            scenario_names = []

        read_parquet_dir=args.read_parquet_dir
        output_location = args.output_location

        if input_db_url:
            db_map = DatabaseMapping(input_db_url)
            scenario_names = [name_from_dict(db_map.get_filter_configs()[0])]
        elif output_locations_db_url:
            db_map = DatabaseMapping(output_locations_db_url)
            filter_configs = db_map.get_filter_configs()
            if filter_configs:
                alternative_names = filter_configs[0]['alternatives']
                scenario_names = alternative_names
            read_parquet_dir = True

        if not scenario_names:
            logging.critical("No scenario provided through any of the arguments: scenario-name or output-locations-db-url (or input-db-url by run_flextool.py)")
            sys.exit(1)

        if args.subdir:
            subdir = args.subdir
        else:
            subdir = scenario_names[0]

        for i, scenario_name in enumerate(scenario_names):
            if not input_db_url and output_locations_db_url:
                param_value = db_map.get_parameter_value_item(
                    entity_class_name='scenario',
                    entity_byname=(scenario_name,),
                    parameter_definition_name='output_location',
                    alternative_name=scenario_name
                )
                if param_value:
                    output_location = param_value['parsed_value']
                else:
                    raise FileNotFoundError(f"Could not find output data location directory for scenario {scenario_name} from db {output_locations_db_url}.")
                subdir = scenario_name
        
            write_outputs(
                scenario_name=scenario_name,
                output_config_path=args.config_path,
                active_configs=args.active_configs,
                output_funcs=None,
                output_location=output_location,
                subdir=subdir,
                read_parquet_dir=read_parquet_dir,
                write_methods=args.write_methods,
                plot_rows=tuple(args.plot_rows) if args.plot_rows else None,
                debug=args.debug,
                single_result=tuple(args.single_result) if args.single_result else None,
                settings_db_url=args.settings_db_url
            )

if __name__ == '__main__':
    main()
