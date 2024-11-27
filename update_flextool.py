import argparse
import sys
import importlib.util

from flextool import update_flextool
#spec = importlib.util.spec_from_file_location("flextool.update_flextool", "flextool/update_flextool.py")
#flextool = importlib.util.module_from_spec(spec)
#sys.modules["module.name"] = flextool
#spec.loader.exec_module(flextool)

def flextool_update(skip_git):
    flextool.update_flextool(skip_git)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--skip-git",
        action="store_true",
        help="skip 'git restore' and 'git pull' steps",
    )
    args = arg_parser.parse_args()
    flextool_update(args.skip_git)