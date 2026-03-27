"""CLI entry point for importing FlexTool 2.0 sensitivities into a Spine DB."""

import argparse
import logging
import sys

from flextool.process_inputs.read_old_flextool import (
    read_old_flextool,
    read_old_flextool_sensitivities,
)
from flextool.process_inputs.write_old_flextool_to_db import write_sensitivities_to_db

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import FlexTool 2.0 sensitivities into a Spine DB."
    )
    parser.add_argument(
        "master_xlsm",
        help="Path to old FlexTool master .xlsm file (contains Sensitivity definitions sheet)",
    )
    parser.add_argument(
        "base_xlsm",
        help="Path to the base data .xlsm file (for unit_type mapping and base data)",
    )
    parser.add_argument(
        "target_db_url",
        help="Target Spine DB URL (must already contain base import)",
    )
    parser.add_argument(
        "--base-alternative",
        default="base",
        help="Name of base alternative in the DB (default: base)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    print(f"Reading base data from: {args.base_xlsm}")
    data = read_old_flextool(args.base_xlsm)

    print(f"Reading sensitivities from: {args.master_xlsm}")
    sensitivities = read_old_flextool_sensitivities(args.master_xlsm)

    if not sensitivities:
        print("No sensitivities found. Nothing to import.")
        sys.exit(0)

    print(f"Found {len(sensitivities)} sensitivity scenario(s).")
    for name, overrides in sensitivities.items():
        print(f"  - {name}: {len(overrides)} override(s)")

    print(f"Writing sensitivities to: {args.target_db_url}")
    write_sensitivities_to_db(
        sensitivities,
        data,
        args.target_db_url,
        base_alternative=args.base_alternative,
    )
    print("Sensitivity import complete!")


if __name__ == "__main__":
    main()
