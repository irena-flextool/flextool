import argparse
import sys
import logging
import traceback
import importlib.util
from typing import Callable
from functools import wraps

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

    args = parser.parse_args()
    input_db_url = args.input_db_url
    scenario_name = args.scenario_name
    DEBUG = args.debug

    logging.basicConfig(
        level=logging.DEBUG if DEBUG else logging.INFO,
        format='%(levelname)s:%(filename)s:%(lineno)d:%(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if scenario_name:
        runner = flextoolrunner.FlexToolRunner(input_db_url, scenario_name)
        runner.write_input(input_db_url, scenario_name)
    else:
        runner = flextoolrunner.FlexToolRunner(input_db_url)
        runner.write_input(input_db_url)
    try:
        return_code = runner.run_model()
    except Exception as e:
        logging.error(f"Model run failed: {str(e)}\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)
    print(__file__)


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