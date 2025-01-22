import argparse
import sys
import logging
import importlib.util

sys.stdout.reconfigure(line_buffering=True)

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

    args = parser.parse_args()
    input_db_url = args.input_db_url

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    runner = flextoolrunner.FlexToolRunner(input_db_url)
    try:
        return_code = runner.run_model()
    except Exception as e:
        logging.error(f"Model run failed: {e}")
        sys.exit(1)
    print(__file__)


if __name__ == '__main__':
    main()