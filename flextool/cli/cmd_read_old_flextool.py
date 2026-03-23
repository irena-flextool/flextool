"""CLI entry point for importing an old-format FlexTool .xlsm file to a Spine DB."""

import argparse
import sys

from flextool.process_inputs.read_old_flextool import read_old_flextool
from flextool.process_inputs.write_old_flextool_to_db import write_old_flextool_to_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import an old-format FlexTool Excel (.xlsm) into a Spine DB."
    )
    parser.add_argument(
        "xlsm_path",
        help="Path to the old-format FlexTool Excel file (.xlsm)",
    )
    parser.add_argument(
        "target_db_url",
        help="Target database URL (e.g. sqlite:///output.sqlite)",
    )
    parser.add_argument(
        "--alternative-name",
        default="base",
        help="Name for the alternative (default: base)",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do not purge existing data before importing",
    )
    args = parser.parse_args()

    print(f"Reading old FlexTool Excel: {args.xlsm_path}")
    data = read_old_flextool(args.xlsm_path)

    print(f"Writing to database: {args.target_db_url}")
    write_old_flextool_to_db(
        data,
        args.target_db_url,
        alternative_name=args.alternative_name,
        purge=not args.no_purge,
    )
    print("Import complete!")


if __name__ == "__main__":
    main()
