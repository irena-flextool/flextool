import pandas as pd
import numpy as np
from collections import defaultdict
import re

def parse_mps_to_matrices(mps_file):
    """
    Parse MPS file and convert to matrix format with separate dataframes for each constraint group.

    Returns:
        dict containing:
        - c: objective coefficients (DataFrame)
        - constraints: dict of constraint_base_name -> DataFrame with rows for each instance,
                      columns for variables, 'sense', and 'rhs'
        - bounds_lower: lower bounds (Series)
        - bounds_upper: upper bounds (Series)
        - var_names: list of variable names
        - constraint_names: list of all constraint names
    """

    with open(mps_file, 'r') as f:
        lines = f.readlines()

    # Initialize data structures
    obj_name = None
    row_types = {}  # row_name -> type ('E', 'G', 'L', 'N')
    row_names = []
    var_names = []

    # Coefficient storage
    obj_coeffs = {}  # var_name -> coeff
    constraint_coeffs = defaultdict(dict)  # row_name -> {var_name -> coeff}
    rhs_values = {}  # row_name -> value
    bounds = {}  # var_name -> {'lower': val, 'upper': val}

    # Parse sections
    section = None

    for line in lines:
        line = line.rstrip()

        # Skip comments and empty lines
        if not line or line.startswith('*'):
            continue

        # Detect sections
        if line.startswith('NAME'):
            continue
        elif line.startswith('ROWS'):
            section = 'ROWS'
            continue
        elif line.startswith('COLUMNS'):
            section = 'COLUMNS'
            continue
        elif line.startswith('RHS'):
            section = 'RHS'
            continue
        elif line.startswith('BOUNDS'):
            section = 'BOUNDS'
            continue
        elif line.startswith('ENDATA'):
            break

        # Parse based on section
        if section == 'ROWS':
            # Format: <type> <row_name>
            parts = line.split()
            if len(parts) >= 2:
                row_type = parts[0]
                row_name = parts[1]
                row_types[row_name] = row_type
                if row_type == 'N':
                    if obj_name is None:
                        obj_name = row_name
                else:
                    row_names.append(row_name)

        elif section == 'COLUMNS':
            # Format: <var_name> <row_name> <value> [<row_name> <value>]
            parts = line.split()
            if len(parts) >= 3:
                var_name = parts[0]

                if var_name not in var_names:
                    var_names.append(var_name)

                # Process pairs of (row_name, value)
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        row_name = parts[i]
                        value = float(parts[i + 1])

                        if row_name == obj_name:
                            obj_coeffs[var_name] = value
                        else:
                            constraint_coeffs[row_name][var_name] = value

        elif section == 'RHS':
            # Format: <rhs_name> <row_name> <value> [<row_name> <value>]
            parts = line.split()
            if len(parts) >= 3:
                # Process pairs of (row_name, value)
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        row_name = parts[i]
                        value = float(parts[i + 1])
                        rhs_values[row_name] = value

        elif section == 'BOUNDS':
            # Format: <bound_type> <bound_name> <var_name> <value>
            # Common types: LO (lower), UP (upper), FX (fixed), FR (free), MI (minus infinity), PL (plus infinity)
            parts = line.split()
            if len(parts) >= 3:
                bound_type = parts[0]
                var_name = parts[2]

                if var_name not in bounds:
                    bounds[var_name] = {'lower': 0.0, 'upper': np.inf}

                if bound_type in ['LO', 'FX'] and len(parts) >= 4:
                    bounds[var_name]['lower'] = float(parts[3])
                if bound_type in ['UP', 'FX'] and len(parts) >= 4:
                    bounds[var_name]['upper'] = float(parts[3])
                elif bound_type == 'FR':
                    bounds[var_name]['lower'] = -np.inf
                    bounds[var_name]['upper'] = np.inf
                elif bound_type == 'MI':
                    bounds[var_name]['lower'] = -np.inf
                elif bound_type == 'PL':
                    bounds[var_name]['upper'] = np.inf

    # Build objective coefficient vector (use None for zeros)
    c = pd.Series(None, index=var_names, name='objective', dtype=object)
    for var_name in var_names:
        if var_name in obj_coeffs:
            c[var_name] = obj_coeffs[var_name]

    # Group constraints by base name (before the first '[')
    constraint_groups = defaultdict(list)
    for row_name in row_names:
        # Extract base name
        if '[' in row_name:
            base_name = row_name.split('[')[0]
        else:
            base_name = row_name
        constraint_groups[base_name].append(row_name)

    # Build constraint dataframes grouped by base name
    constraints = {}

    for base_name, group_row_names in constraint_groups.items():
        # Create a dataframe with rows for each constraint instance
        constraint_df = pd.DataFrame(None, index=group_row_names, columns=var_names + ['sense', 'rhs'], dtype=object)

        for row_name in group_row_names:
            # Fill in coefficients
            for var_name, coeff in constraint_coeffs[row_name].items():
                constraint_df.loc[row_name, var_name] = coeff

            # Add sense
            constraint_df.loc[row_name, 'sense'] = row_types[row_name]

            # Add RHS
            if row_name in rhs_values:
                constraint_df.loc[row_name, 'rhs'] = rhs_values[row_name]
            else:
                constraint_df.loc[row_name, 'rhs'] = 0.0

        # Remove fully empty variable columns
        var_cols = [col for col in constraint_df.columns if col not in ['sense', 'rhs']]
        non_empty_vars = [col for col in var_cols if constraint_df[col].notna().any()]
        constraint_df = constraint_df[non_empty_vars + ['sense', 'rhs']]

        constraints[base_name] = constraint_df

    # Build bounds
    bounds_lower = pd.Series(0.0, index=var_names, name='lower_bound')
    bounds_upper = pd.Series(np.inf, index=var_names, name='upper_bound')

    for var_name in var_names:
        if var_name in bounds:
            bounds_lower[var_name] = bounds[var_name]['lower']
            bounds_upper[var_name] = bounds[var_name]['upper']

    return {
        'c': c,
        'constraints': constraints,
        'bounds_lower': bounds_lower,
        'bounds_upper': bounds_upper,
        'var_names': var_names,
        'constraint_names': row_names
    }

def save_matrices_to_csv(matrices, output_dir='matrix_csv', output_prefix='lp_matrices'):
    """
    Save all matrices to CSV files - one file per constraint group.

    Args:
        matrices: dict returned by parse_mps_to_matrices
        output_dir: directory where to save files
        output_prefix: prefix for output files
    """
    import os

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save objective (empty cells for None values)
    matrices['c'].to_csv(os.path.join(output_dir, f'{output_prefix}_objective.csv'), header=True, na_rep='')

    # Save each constraint group to its own file
    for constraint_base_name, constraint_df in matrices['constraints'].items():
        # Sanitize constraint name for filename (replace problematic characters)
        safe_name = constraint_base_name.replace('[', '_').replace(']', '_').replace(',', '_').replace('/', '_')
        filename = f'{output_prefix}_constraint_{safe_name}.csv'
        constraint_df.to_csv(os.path.join(output_dir, filename), na_rep='')

    # Save bounds
    matrices['bounds_lower'].to_csv(os.path.join(output_dir, f'{output_prefix}_bounds_lower.csv'), header=True)
    matrices['bounds_upper'].to_csv(os.path.join(output_dir, f'{output_prefix}_bounds_upper.csv'), header=True)

    print(f"Matrices saved to directory '{output_dir}' with prefix '{output_prefix}'")

# Example usage
if __name__ == '__main__':
    import argparse

    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Convert MPS file to matrix format')
    parser.add_argument('mps_file', help='Path to MPS file')
    parser.add_argument('--output-dir', '-o', default='matrix_csv',
                        help='Output directory for CSV files (default: matrix_csv)')
    parser.add_argument('--prefix', '-p', default='lp_matrices',
                        help='Prefix for output files (default: lp_matrices)')

    args = parser.parse_args()

    # Parse the MPS file
    matrices = parse_mps_to_matrices(args.mps_file)

    # Display summary
    print("=" * 60)
    print("LP Model Summary")
    print("=" * 60)
    print(f"Number of variables: {len(matrices['var_names'])}")
    print(f"Number of constraint groups: {len(matrices['constraints'])}")
    print(f"Total constraint rows: {len(matrices['constraint_names'])}")
    print()

    print("Objective coefficients (first 5):")
    print(matrices['c'].head())
    print()

    # Display constraint group information
    print("Constraint groups:")
    for base_name, constraint_df in matrices['constraints'].items():
        num_rows = len(constraint_df)
        num_vars = len([col for col in constraint_df.columns if col not in ['sense', 'rhs']])
        senses = constraint_df['sense'].unique()
        print(f"  {base_name}: {num_rows} rows, {num_vars} variables, sense={list(senses)}")
    print()

    print("Sample constraint group (first group, first 5 rows):")
    first_group = list(matrices['constraints'].items())[0]
    print(f"\n  Group: {first_group[0]}")
    print(f"  {first_group[1].head().to_string()}")
    print()

    print("Variable bounds (first 5):")
    bounds_df = pd.DataFrame({
        'lower': matrices['bounds_lower'],
        'upper': matrices['bounds_upper']
    })
    print(bounds_df.head())
    print()

    # Save to CSV
    save_matrices_to_csv(matrices, output_dir=args.output_dir, output_prefix=args.prefix)
    print("\nDone!")