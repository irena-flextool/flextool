import argparse
from flextool.update_flextool.self_update import update_flextool


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--skip-git",
        action="store_true",
        help="skip 'git restore' and 'git pull' steps",
    )
    args = arg_parser.parse_args()
    update_flextool(args.skip_git)


if __name__ == "__main__":
    main()
