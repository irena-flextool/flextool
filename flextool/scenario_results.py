import pandas as pd
import os
import math
from pathlib import Path
from spinedb_api import DatabaseMapping
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_horizontal_bar(df, filename=None, title=None, figsize=(10, 6), show_plot=False, subplot=None, stacked=None, sum_index_level=None, n_subplot_cols=1, xlabel=None, ylabel=None):    
    if sum_index_level is not None:
        df = df.groupby(level=sum_index_level).sum()
    n_subplots = 1
    subplot_names = ['']
    if subplot is not None:
        subplot_names = df.columns.get_level_values(level=subplot).unique()
        n_subplots = len(subplot_names)
    n_subplot_rows = math.ceil(n_subplots / n_subplot_cols)
    fig, axes = plt.subplots(nrows=n_subplot_rows, ncols=n_subplot_cols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, subplot_name in enumerate(subplot_names):
        if isinstance(df.columns, pd.MultiIndex):
            df_sub = df.xs(subplot_name, axis=1, level=subplot)
        else:
            df_sub = df
        ax = axes[i]
        _ = df_sub.plot.barh(ax=ax, legend=False, title=subplot_name, xlabel=xlabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.05, 1), loc='upper left')

    if title:
        fig.suptitle(title, fontweight='bold')
    plt.tight_layout()
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    return ax

def stacked_and_grouped_bar_plot(df, output_dir, filename=None, title=None, ylabel=None, xlabel=None, show_plot=False):
    """
    Create a stacked and grouped bar plot.

    The dataframe should be pre-shaped so that:
    - Index (can be MultiIndex): dimensions to stack within each bar
    - Columns (can be MultiIndex): dimensions that create groups and bars within groups
      - If MultiIndex columns: first level creates groups, remaining levels create bars within groups

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe in the correct shape for plotting
    output_dir : str or Path
        Directory to save the SVG file (relative to current working directory)
    filename : str
        Name of the output file (without extension, .svg will be added)
    title : str, optional
        Plot title
    ylabel : str, optional
        Y-axis label
    xlabel : str, optional
        X-axis label
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # First level creates groups (x-axis positions)
        groups = df.columns.get_level_values(0).unique()
    else:
        # Single level columns - each column is a separate group
        groups = df.columns

    n_stack = len(df.index)

    # Colors for stacking dimension
    colors = plt.cm.tab20(np.linspace(0, 1, n_stack))

    # Create list of all column labels for sequential bar positioning
    all_columns = []
    if isinstance(df.columns, pd.MultiIndex):
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                all_columns.append(group)
            else:
                all_columns.extend([(group, col) for col in group_data.columns])
    else:
        all_columns = list(df.columns)

    # Plot bars sequentially
    for bar_idx, col_key in enumerate(all_columns):
        # Stack the index dimensions
        bottom = 0
        for stack_idx, stack_val in enumerate(df.index):
            value = df.loc[stack_val, col_key]

            ax.bar(bar_idx,
                   value,
                   bottom=bottom,
                   label=f'{stack_val}' if bar_idx == 0 else '',
                   color=colors[stack_idx])
            bottom += value

    # Customize the plot
    ax.set_ylabel(ylabel if ylabel else 'Value')
    ax.set_title(title if title else 'Stacked and Grouped Bar Plot')

    # Set up two-level x-axis labels
    if isinstance(df.columns, pd.MultiIndex):
        # Extract scenario labels from all_columns
        scenario_labels = [col[1] if isinstance(col, tuple) else col for col in all_columns]

        # Calculate group centers
        group_centers = []
        group_lefts = []
        bar_idx = 0
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                n_bars_in_group = 1
            else:
                n_bars_in_group = len(group_data.columns)
            # Center of group is at midpoint of its bars
            group_center = bar_idx + (n_bars_in_group - 1) / 2
            group_centers.append(group_center)
            group_lefts.append(bar_idx - 0.5)
            bar_idx += n_bars_in_group
        group_lefts.append(bar_idx - 0.5)

        # Set main x-axis for scenarios (individual bars)
        ax.set_xticks(range(len(all_columns)), labels=scenario_labels)
        # ax.xaxis.set_major_locator(mticker.MultipleLocator(base=1))
        ax.tick_params('x', length=0)

        # Set plot limits
        ax.set_xlim(-0.5, len(all_columns) - 0.5)

        # Add x-axis tick-based separators
        scen_sep_ax  = ax.secondary_xaxis(location=0)
        scen_sep_ax.set_xticks([x - 0.5 for x in range(len(all_columns) + 1)], [''] * (len(all_columns) + 1))
        scen_sep_ax.tick_params('x', length=15)

        # Add secondary x-axis for groups
        group_ax = ax.secondary_xaxis(location=0)
        group_ax.set_xticks(group_centers, labels=groups)
        group_ax.tick_params('x', length=0, pad=20)

        # Set x-axis label on group axis
        if xlabel:
            group_ax.set_xlabel(xlabel)

        group_sep_ax  = ax.secondary_xaxis(location=0)
        group_sep_ax.set_xticks(group_lefts, [''] * (len(groups) + 1))
        group_sep_ax.tick_params('x', length=30)
    else:
        ax.set_xticks(range(len(all_columns)))
        ax.set_xticklabels(all_columns)
        ax.set_xlabel(xlabel if xlabel else 'Groups')

    # Reverse legend order to match stacking order (bottom to top)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Period', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()

    # Save to SVG
    output_path = Path(os.getcwd()) / output_dir / f"{filename}.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if filename:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    if show_plot:
        plt.show()
    plt.close()


def stacked_and_grouped_barh_plot(df, output_dir, filename=None, title=None, xlabel=None, ylabel=None, show_plot=False):
    """
    Create a horizontal stacked and grouped bar plot.

    The dataframe should be pre-shaped so that:
    - Index (can be MultiIndex): dimensions to stack within each bar
    - Columns (can be MultiIndex): dimensions that create groups and bars within groups
      - If MultiIndex columns: first level creates groups, remaining levels create bars within groups

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe in the correct shape for plotting
    output_dir : str or Path
        Directory to save the SVG file (relative to current working directory)
    filename : str
        Name of the output file (without extension, .svg will be added)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    """

    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        # First level creates groups (y-axis positions)
        groups = df.columns.get_level_values(0).unique()
    else:
        # Single level columns - each column is a separate group
        groups = df.columns

    n_stack = len(df.index)

    # Colors for stacking dimension
    colors = plt.cm.tab20(np.linspace(0, 1, n_stack))

    # Create list of all column labels for sequential bar positioning
    all_columns = []
    if isinstance(df.columns, pd.MultiIndex):
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                all_columns.append(group)
            else:
                all_columns.extend([(group, col) for col in group_data.columns])
    else:
        all_columns = list(df.columns)

    fig, ax = plt.subplots(figsize=(14, (0.2 * len(all_columns) + 0.08)))

    # Plot bars sequentially (horizontal)
    for bar_idx, col_key in enumerate(all_columns):
        # Stack the index dimensions
        left = 0
        for stack_idx, stack_val in enumerate(df.index):
            value = df.loc[stack_val, col_key]

            ax.barh(bar_idx,
                    value,
                    left=left,
                    label=f'{stack_val}' if bar_idx == 0 else '',
                    color=colors[stack_idx])
            left += value

    # Customize the plot
    ax.set_xlabel(xlabel if xlabel else 'Value')
    ax.set_title(title if title else 'Stacked and Grouped Bar Plot')

    # Set up two-level y-axis labels
    if isinstance(df.columns, pd.MultiIndex):
        # Extract scenario labels from all_columns
        scenario_labels = [col[1] if isinstance(col, tuple) else col for col in all_columns]

        # Calculate group centers
        group_centers = []
        group_lefts = []
        bar_idx = 0
        for group in groups:
            group_data = df[group]
            if isinstance(group_data, pd.Series):
                n_bars_in_group = 1
            else:
                n_bars_in_group = len(group_data.columns)
            # Center of group is at midpoint of its bars
            group_center = bar_idx + (n_bars_in_group - 1) / 2
            group_centers.append(group_center)
            group_lefts.append(bar_idx - 0.5)
            bar_idx += n_bars_in_group
        group_lefts.append(bar_idx - 0.5)

        # Set main y-axis for scenarios (individual bars)
        ax.set_yticks(range(len(all_columns)), labels=scenario_labels)
        ax.tick_params('y', length=0)

        # Set plot limits
        ax.set_ylim(-0.5, len(all_columns) - 0.5)

        # Calculate padding based on maximum length of scenario labels
        # Get the longest scenario label text
        max_label_length_scens = max(len(str(label)) for label in scenario_labels)
        max_label_length_groups = max(len(str(label)) for label in groups)

        # Estimate padding: roughly 6 points per character
        pad_value_scens = max_label_length_scens * 5.8
        pad_value_groups = pad_value_scens + max_label_length_groups * 5.8

        # Add y-axis tick-based separators
        scen_sep_ax = ax.secondary_yaxis(location=0)
        scen_sep_ax.set_yticks([x - 0.5 for x in range(len(all_columns) + 1)], [''] * (len(all_columns) + 1))
        scen_sep_ax.tick_params('y', length=pad_value_scens)

        # Add secondary y-axis for groups
        group_ax = ax.secondary_yaxis(location=0)
        group_ax.set_yticks(group_centers, labels=groups)
        group_ax.tick_params('y', length=0 , pad=pad_value_scens + 3) # , labelrotation=90
        # group_ax.set_position([0.0, 0.11, 0.12, 0.88])

        # Set y-axis label on group axis
        if ylabel:
            group_ax.set_ylabel(ylabel)

        # Separators for groups
        group_sep_ax = ax.secondary_yaxis(location=0)
        group_sep_ax.set_yticks(group_lefts, [''] * (len(groups) + 1))
        group_sep_ax.tick_params('y', length=pad_value_groups)
        #group_sep_ax.set_position([0.0, 0.11, 0.12, 0.88])

    else:
        ax.set_yticks(range(len(all_columns)))
        ax.set_yticklabels(all_columns)
        ax.set_ylabel(ylabel if ylabel else 'Groups')

    # Reverse legend order to match stacking order (left to right)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='Period', bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()

    # Save to SVG
    output_path = Path(os.getcwd()) / output_dir / f"{filename}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if filename:
        plt.savefig(output_path, format='svg', bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    if show_plot:
        plt.show()
    plt.close()


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

    The parquet files are expected to have scenario information in a multi-index
    column level (highest level), so they are appended along the column axis.

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

        dfs_to_append = []

        for scenario_name, file_path in scenario_files:
            try:
                df = pd.read_parquet(file_path)
                dfs_to_append.append(df)

            except Exception as e:
                print(f"Error reading {file_path} for scenario {scenario_name}: {e}")
                continue

        if dfs_to_append:
            # Append dataframes along columns (axis=1) since scenario is in column multi-index
            combined_df = pd.concat(dfs_to_append, axis=1)
            combined_dfs[variable_name] = combined_df
            print(f"Combined {len(dfs_to_append)} files for variable '{variable_name}' "
                  f"(shape: {combined_df.shape})")
        else:
            print(f"Warning: No valid data found for {filename}")

    return combined_dfs


def load_csvs_into_scenario_dataframe(scenario_folders, csv_filename, row_level_names, col_level_names, output_subdir="output_parquet"):
    """
    Load CSV files of a particular name from scenario folders and concatenate them into a single dataframe.

    The scenario name is added as an additional dimension to the column index level as the lowest level
    (closest to the data). This means column levels are always a MultiIndex, and row index levels are
    also always a MultiIndex.

    Parameters:
    -----------
    scenario_folders : dict
        Dictionary mapping scenario names to folder paths
    csv_filename : str
        Name of the CSV file to load from each scenario folder (e.g., 'results.csv')
    n_row_index_levels : int
        Number of columns to use as row index (always the first N columns)
    n_col_index_levels : int
        Number of header rows in the CSV file to use as column index
    output_subdir : str, optional
        Subdirectory within each folder containing CSV files (default: 'output_parquet')

    Returns:
    --------
    pd.DataFrame : Combined dataframe with scenario name added as lowest column index level
    """
    dfs_to_concat = []

    for scenario_name, folder_path in scenario_folders.items():
        csv_dir = Path(folder_path)
        csv_path = csv_dir / csv_filename

        if not csv_path.exists():
            print(f"Warning: {csv_path} does not exist for scenario {scenario_name}")
            continue
        
        n_col_index_levels = len(col_level_names)
        n_row_index_levels = len(row_level_names)
        try:
            # Read CSV with multi-level headers and index
            df = pd.read_csv(
                csv_path,
                header=list(range(n_col_index_levels)) if n_col_index_levels > 0 else 0,
                index_col=list(range(n_row_index_levels))
            )

            df = df.astype(float)

            # Ensure columns are MultiIndex
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_tuples([(col,) for col in df.columns])

            # Add scenario name as the lowest (innermost) level of column index
            new_columns = pd.MultiIndex.from_tuples(
                [tuple(list(col) + [scenario_name]) for col in df.columns]
            )
            df.columns = new_columns

            dfs_to_concat.append(df)

        except Exception as e:
            print(f"Error reading {csv_path} for scenario {scenario_name}: {e}")
            continue

    if not dfs_to_concat:
        print(f"Warning: No valid data found for {csv_filename}")
        return pd.DataFrame()

    # Concatenate along columns (axis=1)
    combined_df = pd.concat(dfs_to_concat, axis=1)
    combined_df.index.names = row_level_names
    combined_df = combined_df.droplevel('solve')
    col_level_names.append('scenario')
    combined_df.columns.names = col_level_names
    

    print(f"Combined {len(dfs_to_concat)} CSV files for '{csv_filename}' (shape: {combined_df.shape})")

    return combined_df

def get_scenario_results(db_url, parquet_subdir='output_parquet'):
    # Read scenario folders from database
    print(f"Reading scenario information from {db_url}...")
    scenario_folders = read_scenario_folders(db_url)
    print(f"Found {len(scenario_folders)} scenarios: {list(scenario_folders.keys())}")

    # Collect all parquet files
    print(f"\nCollecting parquet files from {parquet_subdir} subdirectories...")
    files_by_name = collect_parquet_files(scenario_folders, parquet_subdir)
    print(f"Found {len(files_by_name)} unique result variables")

    # Combine parquet files
    print("\nCombining parquet files...")
    combined_dfs = combine_parquet_files(files_by_name)

    print(f"\nDone! Combined {len(combined_dfs)} result variables.")

    # Return the combined dataframes for use in interactive mode
    return scenario_folders, combined_dfs


if __name__ == '__main__':
    import argparse
    matplotlib.use('Agg')
    parser = argparse.ArgumentParser(
        description='Read and combine scenario results from multiple folders based on database information'
    )
    parser.add_argument(
        'db_url',
        help='Database URL containing scenario information (e.g., sqlite:///scenarios.db)'
    )
    parser.add_argument(
        '--parquet-subdir',
        default='output_parquet',
        help='Subdirectory containing parquet files (default: output_parquet)'
    )

    args = parser.parse_args()

    scenario_names, combined_dfs = get_scenario_results(db_url=args.db_url, parquet_subdir=args.parquet_subdir)

    foo = combined_dfs['nodeGroup_gd_p'].groupby('scenario', axis=1).sum()
    plot_horizontal_bar(foo, filename='foo.svg', figsize=(10, 6), sum_index_level=0, n_subplot_cols=2)
    plot_horizontal_bar(combined_dfs['nodeGroup_gd_p'], filename='foo.svg', figsize=(10, 6), subplot=1, sum_index_level=0, n_subplot_cols=2)
