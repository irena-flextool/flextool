import argparse
from flextool.update_flextool import update_flextool


def flextool_update(skip_git):
    print(skip_git)
    update_flextool(skip_git)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--skip-git",
        action="store_true",
        help="skip 'git restore' and 'git pull' steps",
    )
    args = arg_parser.parse_args()
    flextool_update(args.skip_git)