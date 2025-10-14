import argparse
import sys
import logging
import traceback
import importlib.util
from typing import Callable
from functools import wraps
import time

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
    parser.add_argument('scenario_name', help='Name for the scenario in the database that should be executed', nargs='?', default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--output-spreadsheet', metavar='PATH', help='Save results to spreadsheet file')
    parser.add_argument('--output-database', metavar='DB_URL', help='Save results to database')
    

    args = parser.parse_args()
    input_db_url = args.input_db_url
    scenario_name = args.scenario_name
    DEBUG = args.debug

    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    start_time = time.time()
    if scenario_name:
        runner = flextoolrunner.FlexToolRunner(input_db_url, scenario_name)
        print("--- Init time %s seconds ---" % (time.time() - start_time))
        runner.write_input(input_db_url, scenario_name)
        print("--- write time %s seconds ---" % (time.time() - start_time))
    else:
        runner = flextoolrunner.FlexToolRunner(input_db_url)
        print("--- Init time %s seconds ---" % (time.time() - start_time))
        runner.write_input(input_db_url)
        print("--- write time %s seconds ---" % (time.time() - start_time))
    try:
        return_code = runner.run_model()
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)
    
    if return_code == 0:
        # Output to spreadsheet if requested
        if args.output_spreadsheet:
            runner.process_outputs('spreadsheet', args.output_spreadsheet)
        
        # Output to database if requested
        if args.output_database:
            runner.process_outputs('database', args.output_database)
        
        # Or if neither specified, could default to spreadsheet
        if not args.output_spreadsheet and not args.output_database:
            runner.process_outputs('spreadsheet', 'results.xlsx')

    print(__file__)
    print("--- full time %.12s seconds ---------------------------------------" % (time.time() - start_time))
    print("--------------------------------------------------------------------------\n\n")
    
    

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