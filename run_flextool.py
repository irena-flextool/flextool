import argparse
import sys
import logging

sys.stdout.reconfigure(line_buffering=True)

from flextool.flextoolrunner import FlexToolRunner


#return_codes
#0 : Success
#-1: Failure (Defined in the Toolbox)
#1: Infeasible or unbounded problem (not implemented in the toolbox, functionally same as -1. For a possiblity of a graphical depiction)


def main():
    parser = argparse.ArgumentParser()
    parser.description = "Run flextool does not take arguments. It uses input folder to setup the model run. Return codes are 0: success, 1: infeasible or unbounded, -1: failure."

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    runner = FlexToolRunner()
    try:
        return_code = runner.run_model()
    except Exception as e:
        logging.error(f"Model run failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()