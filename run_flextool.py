import argparse
import sys
import logging
import traceback
import importlib.util
from typing import Callable
from functools import wraps
import time
import os
from flextool.write_outputs import write_outputs
from spinedb_api.filters.tools import name_from_dict
from spinedb_api import DatabaseMapping, to_database

class FlushingStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)


sys.stdout = FlushingStream(sys.stdout)

spec = importlib.util.spec_from_file_location("flextool.flextoolrunner", "flextool/flextoolrunner.py")
flextoolrunner = importlib.util.module_from_spec(spec)
sys.modules["flextool.flextoolrunner"] = flextoolrunner
spec.loader.exec_module(flextoolrunner)

#__file__ = os.path.abspath("run_flextool.py")
#from flextool.flextoolrunner import FlexToolRunner

#return_codes
#0 : Success
#-1: Failure (Defined in the Toolbox)
#1: Infeasible or unbounded problem (not implemented in the toolbox, functionally same as -1. For a possiblity of a graphical depiction)


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Run flextool using the specified database URL. Return codes are 0: success, 1: infeasible or unbounded, -1: failure."
    parser.add_argument('input_db_url', help='Database URL to connect to (can be copied from Toolbox workflow db item')
    parser.add_argument('output_db_url', metavar='DB_URL', help='Save information about result location to database for post-processing')
    parser.add_argument('scenario_name', help='Name for the scenario in the database that should be executed', nargs='?', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output-spreadsheet', metavar='PATH', help='Save results to spreadsheet file')
    

    args = parser.parse_args()
    input_db_url = args.input_db_url
    scenario_name = args.scenario_name
    DEBUG = args.debug

    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    timer = [] 
    timer.append(time.perf_counter())
    if scenario_name:
        runner = flextoolrunner.FlexToolRunner(input_db_url, scenario_name)
        timer.insert(0, time.perf_counter())
        print("--- Init time %.4s seconds ---" % (timer[0] - timer[1]))
        runner.write_input(input_db_url, scenario_name)
        timer.insert(0, time.perf_counter())
        print("--- write time %.4s seconds ---" % (timer[0] - timer[1]))
    else:
        runner = flextoolrunner.FlexToolRunner(input_db_url)
        timer.insert(0, time.perf_counter())
        print("--- Init time %.4s seconds ---" % (timer[0] - timer[1]))
        runner.write_input(input_db_url)
        timer.insert(0, time.perf_counter())
        print("--- write time %.4s seconds ---" % (timer[0] - timer[1]))
        db_map = DatabaseMapping(input_db_url)
        scenario_name = name_from_dict(db_map.get_filter_configs()[0])
    try:
        return_code = runner.run_model()
        timer.insert(0, time.perf_counter())
        print("--- run_model time %.4s seconds ---" % (timer[0] - timer[1]))
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)
    
    if return_code == 0:
        write_outputs(scenario_name=scenario_name)

        timer.insert(0, time.perf_counter())
        ## print("--- write outputs time %s seconds ---" % (timer[0] - timer[1]))
    print(__file__)
    print("--- full time %.4s seconds ---------------------------------------" % (timer[0] - timer[-1]))
    print("--------------------------------------------------------------------------\n\n")

    # Write scenario information to output database if provided
    if args.output_db_url:
        # Check if database exists
        db_exists = os.path.exists(args.output_db_url.replace('sqlite:///', ''))

        with DatabaseMapping(args.output_db_url, create=not db_exists) as output_db:
            # Get the full path to the current working directory
            folder_path = os.getcwd()

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

            # Convert folder path to database representation
            value, type_ = to_database(folder_path)

            # Add/update parameter value for folder
            output_db.add_or_update_parameter_value(
                entity_class_name="scenario",
                entity_byname=(scenario_name,),
                parameter_definition_name="folder",
                alternative_name="Base",
                value=value,
                type=type_
            )

            output_db.commit_session("Added/updated scenario information")



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