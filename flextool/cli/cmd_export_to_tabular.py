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
    parser.add_argument(
        "--groups",
        default=None,
        help=(
            "Comma-separated list of parameter_group names to keep "
            "(e.g. 'basics,model,timeline,solve_basics').  Parameters "
            "outside these groups are dropped, and sheets with no surviving "
            "columns are removed unless listed in always_include_with_groups."
        ),
    )
    parser.add_argument(
        "--no-description-row",
        action="store_true",
        help=(
            "Leave out the description row of the parameter sheets; each parameter's description stays in the "
            "note on its name (self-describing format only)"
        ),
    )
    args = parser.parse_args()
    if args.old_format and args.no_description_row:
        parser.error("--no-description-row needs the self-describing format (not --old-format)")
    include_groups = (
        [g.strip() for g in args.groups.split(",") if g.strip()]
        if args.groups
        else None
    )
    export_to_excel(
        args.db_url,
        args.output_path,
        include_advanced=args.include_advanced,
        use_new_format=not args.old_format,
        include_groups=include_groups,
        description_row=not args.no_description_row,
    )


if __name__ == "__main__":
    main()
