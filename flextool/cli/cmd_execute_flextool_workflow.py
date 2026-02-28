#!/usr/bin/env python3
"""
FlexTool Workflow Orchestrator

This script executes the complete FlexTool workflow in three phases:
1. Input preparation: Convert tabular data to FlexTool input database
2. Model execution: Run the FlexTool optimization model
3. Output generation: Process and write results in various formats

Each phase can be skipped independently using --skip-* flags.
"""

import argparse
import subprocess
import sys
import os


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

  # Skip input prep if database already exists
  python execute_flextool_workflow.py input.sqlite results.sqlite my_scenario --skip-input-prep

  # Run only model (skip both input and output phases)
  python execute_flextool_workflow.py input.sqlite results.sqlite my_scenario --skip-input-prep --skip-output-write
        """
    )

    # Required arguments
    parser.add_argument('input_db_url', help='Input database URL (e.g., sqlite:///input.sqlite or input.sqlite)')
    parser.add_argument('output_db_url', help='Output database URL for storing result metadata')
    parser.add_argument('scenario_name', help='Name of the scenario to execute')

    # Input phase arguments
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--tabular-file-path', help='Path to Excel/ODS input file')
    input_group.add_argument('--csv-directory-path', help='Path to directory containing CSV input files')

    # Output phase arguments
    parser.add_argument('--output-methods', nargs='+', default=['plot', 'parquet', 'csv'],
                        choices=['plot', 'parquet', 'excel', 'csv'],
                        help='Output formats to generate (default: plot parquet csv)')
    parser.add_argument('--output-subdir', help='Subdirectory for output files (default: scenario_name)')
    parser.add_argument('--output-config', default='templates/default_plots.yaml',
                        help='Path to output configuration YAML file (default: templates/default_plots.yaml)')

    # Skip flags for individual phases
    parser.add_argument('--skip-input-prep', action='store_true',
                        help='Skip input preparation phase (assumes input database already exists)')
    parser.add_argument('--skip-model-run', action='store_true',
                        help='Skip model execution phase (assumes model has already run)')
    parser.add_argument('--skip-output-write', action='store_true',
                        help='Skip output generation phase')

    # Additional options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for model run')

    args = parser.parse_args()

    # Validate arguments
    if not args.skip_input_prep and not (args.tabular_file_path or args.csv_directory_path):
        parser.error("Must provide either --tabular-file-path or --csv-directory-path unless --skip-input-prep is used")

    # Set default output subdirectory to scenario name if not specified
    output_subdir = args.output_subdir if args.output_subdir else args.scenario_name

    # Phase 1: Input Preparation
    if not args.skip_input_prep:
        print(f"\n{'='*70}")
        print("PHASE 1: PREPARING INPUT DATA")
        print(f"{'='*70}")
        if args.tabular_file_path:
            print(f"  Input file:  {args.tabular_file_path}")
        else:
            print(f"  Input directory: {args.csv_directory_path}")
        print(f"  Target DB:   {args.input_db_url}")
        print()

        cmd = ['python', 'read_tabular_input.py', args.input_db_url]
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

    # Phase 2: Model Execution
    if not args.skip_model_run:
        print(f"\n{'='*70}")
        print("PHASE 2: RUNNING FLEXTOOL MODEL")
        print(f"{'='*70}")
        print(f"  Input DB:    {args.input_db_url}")
        print(f"  Output DB:   {args.output_db_url}")
        print(f"  Scenario:    {args.scenario_name}")
        print()

        cmd = ['python', 'run_flextool.py', args.input_db_url, args.output_db_url, args.scenario_name]
        if args.debug:
            cmd.append('--debug')

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n" + "="*70)
            print("ERROR: Model execution failed with return code", result.returncode)
            print("="*70)
            sys.exit(result.returncode)

        print(f"\n{'='*70}")
        print("PHASE 2: COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("PHASE 2: SKIPPED (using existing model results)")
        print(f"{'='*70}\n")

    # Phase 3: Output Generation
    if not args.skip_output_write:
        print(f"\n{'='*70}")
        print("PHASE 3: WRITING OUTPUTS")
        print(f"{'='*70}")
        print(f"  Scenario:    {args.scenario_name}")
        print(f"  Formats:     {', '.join(args.output_methods)}")
        print(f"  Subdirectory: {output_subdir}")
        print(f"  Config:      {args.output_config}")
        print()

        cmd = ['python', 'write_outputs.py', args.scenario_name,
               '--config_path', args.output_config,
               '--methods'] + args.output_methods + ['--subdir', output_subdir]

        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("\n" + "="*70)
            print("ERROR: Output generation failed with return code", result.returncode)
            print("="*70)
            sys.exit(result.returncode)

        print(f"\n{'='*70}")
        print("PHASE 3: COMPLETED SUCCESSFULLY")
        print(f"{'='*70}\n")
    else:
        print(f"\n{'='*70}")
        print("PHASE 3: SKIPPED")
        print(f"{'='*70}\n")

    # Success summary
    print(f"\n{'='*70}")
    print("SUCCESS: COMPLETE WORKFLOW EXECUTED")
    print(f"{'='*70}")
    print(f"  Scenario:     {args.scenario_name}")
    print(f"  Input DB:     {args.input_db_url}")
    print(f"  Output DB:    {args.output_db_url}")
    if not args.skip_output_write:
        print(f"  Output dir:   {output_subdir}/")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
