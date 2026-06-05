"""CLI entry point for importing an old-format FlexTool .xlsm file to a Spine DB.

This runs the full pipeline: initialise the target from the frozen v56 import
template (if not already a FlexTool DB), write the workbook's data, then
``migrate_database`` the result up to the current schema version.  See
``import_old_flextool_xlsm`` for why the importer is pinned to a fixed version
and relies on migration rather than tracking the live schema.
"""

import argparse

from flextool.process_inputs.write_old_flextool_to_db import (
    import_old_flextool_xlsm,
)


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

    print(f"Importing old FlexTool Excel: {args.xlsm_path}")
    print(f"Target database: {args.target_db_url}")
    import_old_flextool_xlsm(
        args.xlsm_path,
        args.target_db_url,
        alternative_name=args.alternative_name,
        purge=not args.no_purge,
    )
    print("Import complete!")


if __name__ == "__main__":
    main()
