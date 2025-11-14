import pandas as pd
import argparse
import os
from pathlib import Path
from spinedb_api import DatabaseMapping


def read_scenario_folders(db_url):
    """
    Read the scenario database to get all folder paths.

    Parameters:
    -----------
    db_url : str
        Database URL containing scenario information

    Returns:
    --------
    dict : Dictionary mapping scenario names to folder paths
    """
    scenario_folders = {}

    with DatabaseMapping(db_url) as db:
        # Get all scenario entities
        scenarios = db.get_entity_items(entity_class_name="scenario")

        for scenario in scenarios:
            scenario_name = scenario["name"]

            # Get the folder parameter value for this scenario
            param_values = db.get_parameter_value_items(
                entity_class_name="scenario",
                entity_name=scenario_name,
                parameter_definition_name="folder"
            )

            if param_values:
                # Get the first (and should be only) folder parameter value
                folder_path = param_values[0]["parsed_value"]
                scenario_folders[scenario_name] = folder_path

    return scenario_folders


def collect_parquet_files(scenario_folders, output_subdir="output_parquet"):
    """
    Collect all parquet files from all scenario folders.

    Parameters:
    -----------
    scenario_folders : dict
        Dictionary mapping scenario names to folder paths
    output_subdir : str
        Subdirectory within each folder containing parquet files

    Returns:
    --------
    dict : Dictionary mapping filename to list of (scenario_name, file_path) tuples
    """
    # Dictionary to store {filename: [(scenario_name, file_path), ...]}
    files_by_name = {}

    for scenario_name, folder_path in scenario_folders.items():
        parquet_dir = Path(folder_path) / output_subdir

        if not parquet_dir.exists():
            print(f"Warning: {parquet_dir} does not exist for scenario {scenario_name}")
            continue

        # Get all parquet files in this directory
        for parquet_file in parquet_dir.glob("*.parquet"):
            filename = parquet_file.name

            if filename not in files_by_name:
                files_by_name[filename] = []

            files_by_name[filename].append((scenario_name, parquet_file))

    return files_by_name


def combine_parquet_files(files_by_name):
    """
    Combine parquet files across scenarios into dataframes.

    Parameters:
    -----------
    files_by_name : dict
        Dictionary mapping filename to list of (scenario_name, file_path) tuples

    Returns:
    --------
    dict : Dictionary mapping filename (without extension) to combined dataframe
    """
    combined_dfs = {}

    for filename, scenario_files in files_by_name.items():
        # Remove .parquet extension for the variable name
        variable_name = filename.replace('.parquet', '')

        dfs_to_concat = []

        for scenario_name, file_path in scenario_files:
            try:
                df = pd.read_parquet(file_path)

                # Ensure the dataframe has a 'scenario' column
                # If it doesn't exist, add it
                if 'scenario' not in df.columns:
                    df['scenario'] = scenario_name

                dfs_to_concat.append(df)

            except Exception as e:
                print(f"Error reading {file_path} for scenario {scenario_name}: {e}")
                continue

        if dfs_to_concat:
            # Concatenate all dataframes for this variable
            combined_df = pd.concat(dfs_to_concat, ignore_index=True)
            combined_dfs[variable_name] = combined_df
            print(f"Combined {len(dfs_to_concat)} files for variable '{variable_name}' "
                  f"({len(combined_df)} total rows)")
        else:
            print(f"Warning: No valid data found for {filename}")

    return combined_dfs


def main():
    parser = argparse.ArgumentParser(
        description='Read and combine scenario results from multiple folders based on database information'
    )
    parser.add_argument(
        'db_url',
        help='Database URL containing scenario information (e.g., sqlite:///scenarios.db)'
    )
    parser.add_argument(
        '--output-subdir',
        default='output_parquet',
        help='Subdirectory containing parquet files (default: output_parquet)'
    )
    parser.add_argument(
        '--export',
        metavar='PATH',
        help='Optional path to export combined results (as parquet files)'
    )

    args = parser.parse_args()

    # Read scenario folders from database
    print(f"Reading scenario information from {args.db_url}...")
    scenario_folders = read_scenario_folders(args.db_url)
    print(f"Found {len(scenario_folders)} scenarios: {list(scenario_folders.keys())}")

    # Collect all parquet files
    print(f"\nCollecting parquet files from {args.output_subdir} subdirectories...")
    files_by_name = collect_parquet_files(scenario_folders, args.output_subdir)
    print(f"Found {len(files_by_name)} unique result variables")

    # Combine parquet files
    print(f"\nCombining parquet files...")
    combined_dfs = combine_parquet_files(files_by_name)

    # Export if requested
    if args.export:
        export_path = Path(args.export)
        export_path.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting combined results to {export_path}...")
        for variable_name, df in combined_dfs.items():
            output_file = export_path / f"{variable_name}.parquet"
            df.to_parquet(output_file, index=False)
            print(f"  Exported {variable_name}.parquet ({len(df)} rows)")

    print(f"\nDone! Combined {len(combined_dfs)} result variables.")

    # Return the combined dataframes for use in interactive mode
    return combined_dfs


if __name__ == '__main__':
    combined_dfs = main()
