"""CLI entry point for exporting a FlexTool Spine DB to Excel (.xlsx) format."""

import argparse

from flextool.export_to_tabular import export_to_excel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a FlexTool Spine DB to Excel (.xlsx) format."
    )
    parser.add_argument(
        "db_url",
        help="URL to FlexTool input database (e.g. sqlite:///input_data.sqlite)",
    )
    parser.add_argument(
        "output_path",
        help="Output Excel file path (.xlsx)",
    )
    parser.add_argument(
        "--include-advanced",
        action="store_true",
        help="Include advanced sheets (solve sequences, stochastic data)",
    )
    parser.add_argument(
        "--old-format",
        action="store_true",
        help="Use the old v1 format instead of the self-describing v2 format",
    )
    args = parser.parse_args()
    export_to_excel(
        args.db_url,
        args.output_path,
        include_advanced=args.include_advanced,
        use_new_format=not args.old_format,
    )


if __name__ == "__main__":
    main()
