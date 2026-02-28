import argparse
import sys
import logging
import traceback
from typing import Callable
from functools import wraps
from pathlib import Path
from datetime import datetime
import time
import os
from flextool.process_outputs.result_writer import write_outputs
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from spinedb_api.filters.tools import name_from_dict
from spinedb_api import DatabaseMapping, to_database, DateTime
from spinedb_api.exception import NothingToCommit

class FlushingStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = FlushingStream(sys.stdout)

#return_codes
#0 : Success
#-1: Failure (Defined in the Toolbox)
#1: Infeasible or unbounded problem (not implemented in the toolbox, functionally same as -1. For a possiblity of a graphical depiction)


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Run flextool using the specified database URL. Return codes are 0: success, 1: infeasible or unbounded, -1: failure."
    parser.add_argument('input_db_url', help='Database URL to connect to (can be copied from Toolbox workflow db item')
    parser.add_argument('output_db_url', metavar='DB_URL', help='Save information about result location to database for post-processing')
    parser.add_argument('--settings-db-url', help='Settings for post-processing')
    parser.add_argument('--scenario-name', help='Name for the scenario in the database that should be executed', nargs='?', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output-spreadsheet', metavar='PATH', help='Save results to spreadsheet file')
    parser.add_argument('--write-methods', type=str, nargs='+', default=None,
                        choices=['plot', 'parquet', 'excel', 'csv'],
                        help='Output methods to use (default: plot parquet)')
    parser.add_argument('--output-config', metavar='PATH',
                        default=None,
                        help='Path to output configuration file (default: templates/default_plots.yaml)')
    parser.add_argument('--active-configs', type=str, nargs='+', default=None,
                        help='Active output configurations to use (default: default)')
    parser.add_argument('--plot-rows', type=int, nargs=2, default=None, metavar=('FIRST', 'LAST'),
                        help='First and last row to plot in time series (default: 0 167)')
    parser.add_argument('--output-location', metavar='PATH', default=None,
                        help='Override output location path')
    parser.add_argument('--flextool-location', default='template/flextool_location.txt',
                        help='When running in Spine Toolbox, this argument provides the location of FlexTool so outputs can be directed there (instead of work directories).')

    args = parser.parse_args()
    input_db_url = args.input_db_url
    settings_db_url = args.settings_db_url
    scenario_name = args.scenario_name
    DEBUG = args.debug
    output_path = Path(args.flextool_location).resolve().parent.parent

    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    timer = []
    timer.append(time.perf_counter())

    if scenario_name:
        runner = FlexToolRunner(input_db_url, output_path, scenario_name)
        timer.insert(0, time.perf_counter())
        print("--- Init time %.4s seconds ---" % (timer[0] - timer[1]))
        with open("solve_data/solve_progress.csv", "w") as solve_progress:
            solve_progress.write('scenario,' + scenario_name + '\n')
            solve_progress.write('Init time,' + str(round(timer[0] - timer[1],4)) + '\n')
        runner.write_input(input_db_url, scenario_name)
        timer.insert(0, time.perf_counter())
        print("--- Write time %.4s seconds ---" % (timer[0] - timer[1]))
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('Write input time,' + str(round(timer[0] - timer[1],4)) + '\n')

    else:
        runner = FlexToolRunner(input_db_url, output_path)
        timer.insert(0, time.perf_counter())
        print("--- Init time %.4s seconds ---" % (timer[0] - timer[1]))
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('scenario,unknown\n')
            solve_progress.write('Init time,' + str(round(timer[0] - timer[1],4)) + '\n')
        runner.write_input(input_db_url)
        timer.insert(0, time.perf_counter())
        print("--- Write time %.4s seconds ---" % (timer[0] - timer[1]))
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('Write all input time,' + str(round(timer[0] - timer[1],4)) + '\n')
        db_map = DatabaseMapping(input_db_url)
        scenario_name = name_from_dict(db_map.get_filter_configs()[0])

    print(f'Scenario: {scenario_name}')
    try:
        return_code = runner.run_model()
        timer.insert(0, time.perf_counter())
        print("--- All Flextool solves time %.4s seconds ---" % (timer[0] - timer[1]))
        with open("solve_data/solve_progress.csv", "a") as solve_progress:
            solve_progress.write('All Flextool solves,' + str(round(timer[0] - timer[1],4)) + '\n')
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)
    
    # If successful and requested, write outputs
    if return_code == 0:
        write_outputs(
            scenario_name=scenario_name,
            output_location=args.output_location,
            subdir=scenario_name,
            output_config_path=args.output_config,
            active_configs=args.active_configs,
            write_methods=args.write_methods,
            plot_rows=tuple(args.plot_rows) if args.plot_rows else None,
            settings_db_url=settings_db_url,
            fallback_output_location=str(output_path),
        )
        timer.insert(0, time.perf_counter())
    
    print("\n--- Full execution time %.4s seconds ---------------------------------------" % (timer[0] - timer[-1]))
    print("--------------------------------------------------------------------------\n")
    with open("solve_data/solve_progress.csv", "a") as solve_progress:
        solve_progress.write('Full execution time,' + str(round(timer[0] - timer[-1],4)) + '\n')

    # Write scenario information to output database if provided
    if args.output_db_url:
        # Check if database exists
        db_exists = os.path.exists(args.output_db_url.replace('sqlite:///', ''))

        with DatabaseMapping(args.output_db_url, create=not db_exists) as output_db:
            # Create/update scenario class if it doesn't exist
            output_db.add_or_update_entity_class(name="scenario")

            # Create/update parameter definition for 'folder'
            output_db.add_or_update_parameter_definition(
                entity_class_name="scenario",
                name="folder",
                description="Full path to the working directory"
            )

            # Add/update scenario entity
            output_db.add_or_update_entity(
                entity_class_name="scenario",
                name=scenario_name
            )

            output_db.add_or_update_alternative(name=scenario_name)        

            # Convert folder path to database representation
            value, type_ = to_database(str(output_path))

            # Add/update folder infio
            output_db.add_or_update_parameter_value(
                entity_class_name="scenario",
                entity_byname=(scenario_name,),
                parameter_definition_name="output_location",
                alternative_name=scenario_name,
                value=value,
                type=type_
            )

            dt_value = DateTime(datetime.now())
            value, type_ = to_database(dt_value)

            # Add/update execution time
            output_db.add_or_update_parameter_value(
                entity_class_name="scenario",
                entity_byname=(scenario_name,),
                parameter_definition_name="finish_time",
                alternative_name=scenario_name,
                value=value,
                type=type_
            )

            try:
                output_db.commit_session("Added/updated scenario information")
            except NothingToCommit:
                pass



# Debug flag
DEBUG = False  # Set via environment variable or config

def debug_only(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG:
            return func(*args, **kwargs)

    return wrapper

if __name__ == '__main__':
    main()