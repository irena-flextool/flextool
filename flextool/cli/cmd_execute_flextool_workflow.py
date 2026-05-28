#!/usr/bin/env python3
"""
FlexTool Workflow Orchestrator

Runs the FlexTool workflow in two phases:
1. Input preparation (optional): convert tabular data into a FlexTool input
   database. Runs only when --tabular-file-path or --csv-directory-path is
   given; otherwise the existing input database is used as-is.
2. Model execution + output write: run the optimisation model and write
   results in the requested formats. Parquet is always produced; --write-methods
   selects which additional formats (plot, csv, excel) to generate alongside.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from flextool.update_flextool.ensure_settings_db import ensure_settings_db


def main():
    parser = argparse.ArgumentParser(
        description='Execute complete FlexTool workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow with Excel input
  python execute_flextool_workflow.py input.sqlite results.sqlite my_scenario --tabular-file-path input.xlsx

  # Full workflow with CSV input
  python execute_flextool_workflow.py input.sqlite results.sqlite my_scenario --csv-directory-path input_data/

  # Use an existing input database (no tabular/csv source → input prep is skipped)
  python execute_flextool_workflow.py input.sqlite results.sqlite my_scenario

  # Add plots and CSV alongside the default parquet output
  python execute_flextool_workflow.py input.sqlite results.sqlite my_scenario --write-methods plot parquet csv
        """
    )

    # Required arguments
    parser.add_argument('input_db_url', help='Input database URL (e.g., sqlite:///input.sqlite or input.sqlite)')
    parser.add_argument('output_db_url', help='Output database URL for storing result metadata')
    parser.add_argument('scenario_name', help='Name of the scenario to execute')

    # Input phase arguments. Presence of either flag triggers Phase 1
    # (input preparation); when neither is given, the existing input
    # database is used as-is.
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--tabular-file-path', help='Path to Excel/ODS input file')
    input_group.add_argument('--csv-directory-path', help='Path to directory containing CSV input files')

    # Output controls — forwarded to cmd_run_flextool, which writes
    # outputs inline using the live in-memory flex_data + solution.
    parser.add_argument('--write-methods', nargs='+',
                        default=['plot', 'parquet', 'csv'],
                        choices=['plot', 'parquet', 'excel', 'csv'],
                        help='Output formats to generate (default: plot parquet csv). '
                             'Parquet is the canonical output and is always produced when this flag is omitted.')
    parser.add_argument('--output-subdir', help='Subdirectory for output files (default: scenario_name)')
    parser.add_argument('--output-config', default=None,
                        help='Path to output configuration YAML file (default: bundled schemas/default_plots.yaml)')

    # Additional options
    parser.add_argument('--debug',
                        nargs='?',
                        const='basic',
                        default='off',
                        choices=['off', 'basic', 'full'],
                        metavar='LEVEL',
                        help='Diagnostic verbosity level passed through '
                             'to cmd_run_flextool.  Bare ``--debug`` '
                             'selects ``basic``.  See cmd_run_flextool '
                             '--help for level semantics.')

    args = parser.parse_args()

    # Self-heal missing lightweight settings DBs so fresh clones don't
    # fail opaquely when the user forgot to run `flextool-update`. Only
    # seeds output_info / output_settings / comparison_settings by
    # basename; other paths are left untouched.
    for _candidate in (args.output_db_url,):
        try:
            ensure_settings_db(_candidate)
        except Exception as _exc:
            logging.warning("Failed to auto-seed %s: %s", _candidate, _exc)

    # Phase 1 runs only when a tabular/csv source is provided; otherwise
    # the existing input database is used as-is.
    run_input_prep = bool(args.tabular_file_path or args.csv_directory_path)

    # Set default output subdirectory to scenario name if not specified
    output_subdir = args.output_subdir if args.output_subdir else args.scenario_name

    # Phase 1: Input Preparation
    if run_input_prep:
        print(f"\n{'='*70}")
        print("PHASE 1: PREPARING INPUT DATA")
        print(f"{'='*70}")
        if args.tabular_file_path:
            print(f"  Input file:  {args.tabular_file_path}")
        else:
            print(f"  Input directory: {args.csv_directory_path}")
        print(f"  Target DB:   {args.input_db_url}")
        print()

        cmd = [sys.executable, '-m', 'flextool.cli.cmd_read_tabular_input', args.input_db_url]
        if args.tabular_file_path:
            cmd.extend(['--tabular-file-path', args.tabular_file_path])
        else:
            cmd.extend(['--csv-directory-path', args.csv_directory_path])

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n" + "="*70)
            print("ERROR: Input preparation failed with return code", result.returncode)
            print("="*70)
            sys.exit(result.returncode)

        print(f"\n{'='*70}")
        print("PHASE 1: COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("PHASE 1: SKIPPED (using existing input database)")
        print(f"{'='*70}\n")

    # Phase 2: Model Execution + Output Write
    # cmd_run_flextool writes outputs inline using the live in-memory
    # flex_data + solution from the cascade. There is no separate
    # standalone "write outputs" subprocess — the data is gone once the
    # solver exits.
    print(f"\n{'='*70}")
    print("PHASE 2: RUNNING FLEXTOOL MODEL + WRITING OUTPUTS")
    print(f"{'='*70}")
    print(f"  Input DB:    {args.input_db_url}")
    print(f"  Output DB:   {args.output_db_url}")
    print(f"  Scenario:    {args.scenario_name}")
    print(f"  Formats:     {', '.join(args.write_methods)}")
    print(f"  Subdirectory: {output_subdir}")
    print()

    # Resolve --output-config: bare basename or unset → bundled
    # default from the package; absolute path → used verbatim.
    from flextool._resources import package_data_path
    output_config = args.output_config
    if not output_config or Path(output_config).name in {
        'default_plots.yaml', 'default_comparison_plots.yaml'
    }:
        output_config = str(package_data_path("schemas/default_plots.yaml"))

    cmd = [sys.executable, '-m', 'flextool.cli.cmd_run_flextool',
           args.input_db_url, args.output_db_url,
           '--scenario-name', args.scenario_name,
           '--output-config', output_config,
           '--output-subdir', output_subdir,
           '--write-methods'] + args.write_methods
    if args.debug != 'off':
        cmd.append(f'--debug={args.debug}')

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n" + "="*70)
        print("ERROR: Model execution failed with return code", result.returncode)
        print("="*70)
        sys.exit(result.returncode)

    print(f"\n{'='*70}")
    print("PHASE 2: COMPLETED SUCCESSFULLY")
    print(f"{'='*70}\n")

    # Success summary
    print(f"\n{'='*70}")
    print("SUCCESS: COMPLETE WORKFLOW EXECUTED")
    print(f"{'='*70}")
    print(f"  Scenario:     {args.scenario_name}")
    print(f"  Input DB:     {args.input_db_url}")
    print(f"  Output DB:    {args.output_db_url}")
    print(f"  Output dir:   {output_subdir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
