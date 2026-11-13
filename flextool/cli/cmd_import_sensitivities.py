"""CLI entry point for importing FlexTool 2.0 sensitivities into a Spine DB.

The target must still be at the import schema version (56), i.e. written by
``flextool-read-old ... --no-migrate``.  The usual path is one command that
imports the base data and the sensitivities before migrating:
``flextool-read-old BASE.xlsm sqlite:///new.sqlite --master MASTER.xlsm``.
"""

import argparse
import logging
import sys

from flextool.cli._console import run_tool
from flextool.process_inputs.read_old_flextool import (
    read_old_flextool,
    read_old_flextool_sensitivities,
)
from flextool.process_inputs.write_old_flextool_to_db import (
    ImportReport,
    write_sensitivities_to_db,
)

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
        help=(
            "Target Spine DB URL: a base import still at version 56 "
            "(flextool-read-old --no-migrate)"
        ),
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
    try:
        counters = write_sensitivities_to_db(
            sensitivities,
            data,
            args.target_db_url,
            base_alternative=args.base_alternative,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    report = ImportReport()
    report.add("sensitivities", counters)
    print(report.format())
    print("Sensitivity import complete!")


def cli():
    """Console script: :func:`main` through :func:`flextool.cli._console.run_tool` (control characters escaped)."""
    run_tool(main)


if __name__ == "__main__":
    cli()
