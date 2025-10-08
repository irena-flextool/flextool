import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Set
import csv
from io import StringIO


def check_dir(path: str) -> bool:
    if not os.path.isdir(path):
        print(f"Directory not found: {path}")
        return False
    return True


def get_text_files(directory: str) -> List[str]:
    path = Path(directory)
    return [f.name for f in path.glob("*.csv")]


def parse_csv_line(line: str) -> List[str]:
    return next(csv.reader([line]))


def is_header_line(line: str) -> bool:
    """Check if line starts with a comma."""
    return line.strip().startswith(',')


def get_header_lines(lines: List[str]) -> List[str]:
    """Get all header lines from the start of the file."""
    if not lines:
        return []
    headers = [lines[0]]  # Always include first line
    for line in lines[1:]:  # Check subsequent lines
        if is_header_line(line):
            headers.append(line)
        else:
            break
    return headers


def get_column_widths(header_rows: List[List[str]], differences: List[Tuple[str, str]], diff_cols: Set[int]) -> List[
    int]:
    widths = [0] * len(diff_cols)

    # Check headers
    for row in header_rows:
        for i, col_idx in enumerate(diff_cols):
            if col_idx < len(row):
                widths[i] = max(widths[i], len(row[col_idx]))

    # Check data rows (including Dir1/Dir2 labels)
    for line1, line2 in differences:
        if line1:
            values = parse_csv_line(line1)
            for i, col_idx in enumerate(diff_cols):
                if col_idx < len(values):
                    widths[i] = max(widths[i], len(values[col_idx]))
        if line2:
            values = parse_csv_line(line2)
            for i, col_idx in enumerate(diff_cols):
                if col_idx < len(values):
                    widths[i] = max(widths[i], len(values[col_idx]))

    return widths


def format_row(prefix: str, values: List[str], widths: List[int]) -> str:
    """Format a row with fixed column widths."""
    if prefix:
        return f"{prefix:<6}" + "  ".join(val.ljust(width) for val, width in zip(values, widths))
    return " " * 6 + "  ".join(val.ljust(width) for val, width in zip(values, widths))


def get_different_columns(file1_lines: List[str], file2_lines: List[str]) -> Tuple[List[List[str]], Set[int]]:
    """Identify different columns and return headers from file1 and column indices."""
    headers1 = get_header_lines(file1_lines)
    headers2 = get_header_lines(file2_lines)

    if not headers1:
        return [], set()

    # Parse all header rows
    header_values = [parse_csv_line(h) for h in headers1]

    # Check first 5 data rows to determine max columns
    data_start = len(headers1)
    sample_rows = file1_lines[data_start:data_start + 5]
    sample_parsed = [parse_csv_line(row) for row in sample_rows if row]

    # Use maximum number of columns from headers and sample data
    all_rows = header_values + sample_parsed
    num_cols = max(len(row) for row in all_rows) if all_rows else 0
    different_cols = set(range(num_cols))

    return header_values, different_cols

def compare_files(file1_path: str, file2_path: str) -> Tuple[
    bool, int, int, List[Tuple[str, str]], List[str], Set[int]]:
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        file1_lines = [line.rstrip('\n\r') for line in f1.readlines()]
        file2_lines = [line.rstrip('\n\r') for line in f2.readlines()]

    headers1 = get_header_lines(file1_lines)
    headers2 = get_header_lines(file2_lines)

    # Skip header lines for comparison
    data1 = file1_lines[len(headers1):]
    data2 = file2_lines[len(headers2):]

    differences = []
    for i in range(max(len(data1), len(data2))):
        line1 = data1[i] if i < len(data1) else ""
        line2 = data2[i] if i < len(data2) else ""
        if line1 != line2:
            differences.append((line1, line2))

    is_different = len(differences) > 0
    total_lines = max(len(data1), len(data2))

    # Get column structure from first file
    col_names, diff_cols = get_different_columns(file1_lines, file2_lines)

    return is_different, len(differences), total_lines, differences, col_names, diff_cols


def write_report(results: Dict, output_file: str, n_lines: int,
                 only_in_dir1: Set[str], only_in_dir2: Set[str]):
    total_files = len(results)
    different_files = sum(1 for r in results.values() if r[0])

    with open(output_file, 'w') as f:
        f.write(f"Summary Report\n=============\n")
        f.write(f"Different files: {different_files} out of {total_files}\n")
        f.write(f"Files only in directory 1: {len(only_in_dir1)}\n")
        f.write(f"Files only in directory 2: {len(only_in_dir2)}\n\n")

        if only_in_dir1:
            f.write("Files only in directory 1:\n")
            for file in sorted(only_in_dir1):
                f.write(f"- {file}\n")
            f.write("\n")

        if only_in_dir2:
            f.write("Files only in directory 2:\n")
            for file in sorted(only_in_dir2):
                f.write(f"- {file}\n")
            f.write("\n")

        f.write("File Details\n===========\n")
        for filename, (is_diff, diff_lines, total_lines, differences, header_rows, diff_cols) in sorted(results.items()):
            if is_diff:
                f.write(f"\n{filename}:\n")
                f.write(f"Different lines: {diff_lines} out of {total_lines}\n")
                if diff_lines <= n_lines:
                    f.write("All " + str(diff_lines) + " differences:\n")
                else:
                    f.write("First " + str(n_lines) + " differences out of " + str(diff_lines) + ":\n")

                widths = get_column_widths(header_rows, differences, diff_cols)

                # Show header rows
                for row in header_rows:
                    diff_values = [row[i] if i < len(row) else "" for i in diff_cols]
                    f.write(format_row("", diff_values, widths) + '\n')

                for i, (line1, line2) in enumerate(differences):
                    if i >= n_lines:
                        break

                    values1 = parse_csv_line(line1) if line1 else [""] * len(diff_cols)
                    values2 = parse_csv_line(line2) if line2 else [""] * len(diff_cols)

                    diff_values1 = [values1[i] if i < len(values1) else "" for i in diff_cols]
                    diff_values2 = [values2[i] if i < len(values2) else "" for i in diff_cols]

                    f.write(format_row("Dir1:", diff_values1, widths) + '\n')
                    f.write(format_row("Dir2:", diff_values2, widths) + '\n')
                f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Compare CSV files in two directories')
    parser.add_argument('dir1', help='First directory')
    parser.add_argument('dir2', help='Second directory')
    parser.add_argument('--output', help='Output file', default="files_comp_results.txt")
    parser.add_argument('-n', type=int, default=6, help='Number of diff lines to show (default: 6)')

    args = parser.parse_args()

    if not check_dir(args.dir1) or not check_dir(args.dir2):
        exit(1)

    files1 = set(get_text_files(args.dir1))
    files2 = set(get_text_files(args.dir2))

    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    common_files = files1 & files2
    results = {}

    for filename in sorted(common_files):
        file1_path = os.path.join(args.dir1, filename)
        file2_path = os.path.join(args.dir2, filename)
        results[filename] = compare_files(file1_path, file2_path)

    write_report(results, args.output, args.n, only_in_dir1, only_in_dir2)


if __name__ == "__main__":
    main()