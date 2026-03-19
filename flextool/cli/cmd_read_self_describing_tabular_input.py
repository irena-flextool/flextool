"""CLI entry point for importing a self-describing FlexTool Excel file to a Spine DB."""

import argparse
import sys

from flextool.process_inputs.read_self_describing_excel import read_self_describing_excel
from flextool.process_inputs.write_self_describing_to_db import write_sheet_data_to_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a self-describing FlexTool Excel (.xlsx) into a Spine DB."
    )
    parser.add_argument(
        "xlsx_path",
        help="Path to the self-describing Excel file (.xlsx)",
    )
    parser.add_argument(
        "target_db_url",
        help="Target database URL (e.g. sqlite:///output.sqlite)",
    )
    parser.add_argument(
        "--no-purge",
        action="store_true",
        help="Do not purge existing data before importing",
    )
    args = parser.parse_args()

    print(f"Reading Excel: {args.xlsx_path}")
    sheets = read_self_describing_excel(args.xlsx_path, skip_sheets={"navigate", "version"})

    total_records = sum(len(s.records) for s in sheets)
    total_links = sum(len(s.link_entities) for s in sheets)
    print(f"Parsed {len(sheets)} sheets: {total_records} records, {total_links} links")

    print(f"Writing to database: {args.target_db_url}")
    write_sheet_data_to_db(sheets, args.target_db_url, purge_first=not args.no_purge)
    print("Import complete!")


if __name__ == "__main__":
    main()
