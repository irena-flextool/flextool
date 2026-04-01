"""CLI entry point for importing a MATPOWER .m file to a FlexTool Spine DB."""

import argparse
import sys

from flextool.process_inputs.read_matpower import (
    create_flextool_db_from_matpower,
    read_matpower,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a MATPOWER .m file into a FlexTool Spine DB "
        "with DC power flow enabled.",
    )
    parser.add_argument(
        "matpower_file",
        help="Path to the MATPOWER .m file (e.g. pglib_opf_case14_ieee.m)",
    )
    parser.add_argument(
        "target_db",
        help="Path for the output SQLite database (e.g. case14.sqlite)",
    )
    parser.add_argument(
        "--scenario-name",
        default="dc_opf_test",
        help="Name for the scenario (default: dc_opf_test)",
    )
    parser.add_argument(
        "--alternative-name",
        default="base",
        help="Name for the alternative (default: base)",
    )
    parser.add_argument(
        "--template-json",
        default=None,
        help="Path to FlexTool master template JSON "
        "(default: version/flextool_template_master.json relative to cwd)",
    )
    args = parser.parse_args()

    print(f"Reading MATPOWER file: {args.matpower_file}")
    case = read_matpower(args.matpower_file)
    print(
        f"  Parsed: {case.name} — {len(case.buses)} buses, "
        f"{len(case.generators)} generators, {len(case.branches)} branches, "
        f"baseMVA={case.base_mva}"
    )

    print(f"Writing FlexTool database: {args.target_db}")
    url = create_flextool_db_from_matpower(
        case,
        args.target_db,
        scenario_name=args.scenario_name,
        alternative_name=args.alternative_name,
        template_json=args.template_json,
    )
    print(f"Import complete! Database URL: {url}")
    print(
        f"Run with: python run_flextool.py {url} sqlite:///output.sqlite "
        f"--scenario-name {args.scenario_name}"
    )


if __name__ == "__main__":
    main()
