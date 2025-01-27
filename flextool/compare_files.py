import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict, Set


def check_dir(path: str) -> bool:
    """Check if directory exists."""
    if not os.path.isdir(path):
        print(f"Directory not found: {path}")
        return False
    return True


def get_text_files(directory: str) -> List[str]:
    """Return list of text files in directory."""
    path = Path(directory)
    return [f.name for f in path.glob("*.csv")]


def compare_files(file1_path: str, file2_path: str) -> Tuple[bool, int, int, List[Tuple[str, str]], Tuple[str, str]]:
    """Compare two files and return differences and first lines."""
    with open(file1_path, 'r') as f1, open(file2_path, 'r') as f2:
        file1_lines = [line.rstrip('\n\r') for line in f1.readlines()]
        file2_lines = [line.rstrip('\n\r') for line in f2.readlines()]

    # Get first lines (or empty string if file is empty)
    first_line1 = file1_lines[0] if file1_lines else ""
    first_line2 = file2_lines[0] if file2_lines else ""

    # Find different lines
    differences = []
    for i in range(max(len(file1_lines), len(file2_lines))):
        line1 = file1_lines[i] if i < len(file1_lines) else ""
        line2 = file2_lines[i] if i < len(file2_lines) else ""
        if line1 != line2:
            differences.append((line1, line2))

    is_different = len(differences) > 0
    total_lines = max(len(file1_lines), len(file2_lines))

    return is_different, len(differences), total_lines, differences, (first_line1, first_line2)


def write_report(results: Dict, output_file: str, n_lines: int,
                 only_in_dir1: Set[str], only_in_dir2: Set[str]):
    """Write comparison results to output file."""
    total_files = len(results)
    different_files = sum(1 for r in results.values() if r[0])

    with open(output_file, 'w') as f:
        # Write summary
        f.write(f"Summary Report\n=============\n")
        f.write(f"Different files: {different_files} out of {total_files}\n")
        f.write(f"Files only in directory 1: {len(only_in_dir1)}\n")
        f.write(f"Files only in directory 2: {len(only_in_dir2)}\n\n")

        # List files present in only one directory
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

        # Write per-file details for common files
        f.write("File Details\n===========\n")
        for filename, (is_diff, diff_lines, total_lines, differences, first_lines) in results.items():
            if is_diff:
                f.write(f"\n{filename}:\n")
                f.write(f"Different lines: {diff_lines} out of {total_lines}\n")

                # Show first line(s)
                first_line1, first_line2 = first_lines
                if first_line1 == first_line2:
                    f.write(f"First line: {first_line1}\n")
                else:
                    f.write(f"First line Dir1: {first_line1}\n")
                    f.write(f"First line Dir2: {first_line2}\n")

                if diff_lines <= n_lines:
                    f.write("All " + str(diff_lines) + " differences:\n")
                else:
                    f.write("First " + str(n_lines) + " differences out of " + str(diff_lines) + ":\n")

                # Show differences side by side
                for i, (line1, line2) in enumerate(differences):
                    if i >= n_lines:
                        break
                    f.write(f"Dir1: {line1}\n")
                    f.write(f"Dir2: {line2}\n")
                f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Compare text files in two directories')
    parser.add_argument('dir1', help='First directory')
    parser.add_argument('dir2', help='Second directory')
    parser.add_argument('--output', help='Output file', default="files_comp_results.txt")
    parser.add_argument('-n', type=int, default=6, help='Number of diff lines to show (default: 5)')

    args = parser.parse_args()

    if not check_dir(args.dir1) or not check_dir(args.dir2):
        exit(1)

    # Get list of text files
    files1 = set(get_text_files(args.dir1))
    files2 = set(get_text_files(args.dir2))

    # Find files unique to each directory
    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1

    # Compare common files
    common_files = files1 & files2
    results = {}

    for filename in common_files:
        file1_path = os.path.join(args.dir1, filename)
        file2_path = os.path.join(args.dir2, filename)
        results[filename] = compare_files(file1_path, file2_path)

    write_report(results, args.output, args.n, only_in_dir1, only_in_dir2)


if __name__ == "__main__":
    main()