#!/usr/bin/env python3
import re
import sys
from typing import List, Tuple

def find_largest_numbers(filename: str, top_n: int = 10, threshold: float = None) -> Tuple[List[Tuple[float, int]], List[Tuple[float, int]], List[Tuple[float, int]], List[Tuple[float, int]]]:
    """
    Find numbers in four categories: largest positive, smallest negative, smallest positive, largest negative.
    
    Args:
        filename: Path to the input file
        top_n: Number of top values to return for each category (default: 10)
        threshold: Optional threshold - only return numbers with absolute value >= threshold
    
    Returns:
        Tuple of (largest_positive, smallest_negative, smallest_positive, largest_negative) 
        where each is a list of tuples (number, line_number)
    """
    positive_numbers = []
    negative_numbers = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                # Find all numbers (including decimals, negatives, and scientific notation)
                # This regex matches: optional minus, digits, optional decimal part, optional scientific notation
                numbers = re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', line)
                
                for num_str in numbers:
                    try:
                        num = float(num_str)
                        
                        # Apply threshold filter if specified
                        if threshold is not None and abs(num) < threshold:
                            continue
                        
                        # Skip zeros
                        if num > 0:
                            positive_numbers.append((num, line_num))
                        elif num < 0:
                            negative_numbers.append((num, line_num))
                            
                    except ValueError:
                        continue
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return [], [], [], []
    except Exception as e:
        print(f"Error reading file: {e}")
        return [], [], [], []
    
    # Sort positive numbers for both largest and smallest
    positive_largest = sorted(positive_numbers, key=lambda x: x[0], reverse=True)  # Descending
    positive_smallest = sorted(positive_numbers, key=lambda x: x[0], reverse=False)  # Ascending
    
    # Sort negative numbers for both smallest (most negative) and largest (closest to zero)
    negative_smallest = sorted(negative_numbers, key=lambda x: x[0], reverse=False)  # Most negative first
    negative_largest = sorted(negative_numbers, key=lambda x: x[0], reverse=True)   # Closest to zero first
    
    # Return top N results for each category
    top_positive_largest = positive_largest[:top_n] if top_n else positive_largest
    top_negative_smallest = negative_smallest[:top_n] if top_n else negative_smallest
    top_positive_smallest = positive_smallest[:top_n] if top_n else positive_smallest
    top_negative_largest = negative_largest[:top_n] if top_n else negative_largest
    
    return top_positive_largest, top_negative_smallest, top_positive_smallest, top_negative_largest

def print_results(largest_positive: List[Tuple[float, int]], smallest_negative: List[Tuple[float, int]], 
                 smallest_positive: List[Tuple[float, int]], largest_negative: List[Tuple[float, int]],
                 show_line_content: bool = False, filename: str = None):
    """Print the results in a formatted way."""
    
    print("=" * 60)
    print("LARGEST POSITIVE NUMBERS")
    print("=" * 60)
    
    if not largest_positive:
        print("No positive numbers found matching the criteria.")
    else:
        for i, (number, line_num) in enumerate(largest_positive, 1):
            print(f"{i:2d}. Line {line_num:4d}: {number}")
            
            if show_line_content and filename:
                print_line_content(filename, line_num)
            print()
    
    print("=" * 60)
    print("SMALLEST NEGATIVE NUMBERS (most negative)")
    print("=" * 60)
    
    if not smallest_negative:
        print("No negative numbers found matching the criteria.")
    else:
        for i, (number, line_num) in enumerate(smallest_negative, 1):
            print(f"{i:2d}. Line {line_num:4d}: {number}")
            
            if show_line_content and filename:
                print_line_content(filename, line_num)
            print()
    
    print("=" * 60)
    print("SMALLEST POSITIVE NUMBERS (closest to zero from positive side)")
    print("=" * 60)
    
    if not smallest_positive:
        print("No positive numbers found matching the criteria.")
    else:
        for i, (number, line_num) in enumerate(smallest_positive, 1):
            print(f"{i:2d}. Line {line_num:4d}: {number}")
            
            if show_line_content and filename:
                print_line_content(filename, line_num)
            print()
    
    print("=" * 60)
    print("LARGEST NEGATIVE NUMBERS (closest to zero from negative side)")
    print("=" * 60)
    
    if not largest_negative:
        print("No negative numbers found matching the criteria.")
    else:
        for i, (number, line_num) in enumerate(largest_negative, 1):
            print(f"{i:2d}. Line {line_num:4d}: {number}")
            
            if show_line_content and filename:
                print_line_content(filename, line_num)
            print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if largest_positive:
        print(f"Largest positive number: {largest_positive[0][0]} (Line {largest_positive[0][1]})")
    if smallest_negative:
        print(f"Smallest negative number: {smallest_negative[0][0]} (Line {smallest_negative[0][1]})")
    if smallest_positive:
        print(f"Smallest positive number: {smallest_positive[0][0]} (Line {smallest_positive[0][1]})")
    if largest_negative:
        print(f"Largest negative number: {largest_negative[0][0]} (Line {largest_negative[0][1]})")

def print_line_content(filename: str, line_num: int):
    """Print the content of a specific line."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            if line_num <= len(lines):
                line_content = lines[line_num - 1].strip()
                if len(line_content) > 80:
                    line_content = line_content[:77] + "..."
                print(f"     Content: {line_content}")
    except:
        print("     Content: (unable to read)")

def main():
    """Main function to run the script."""
    if len(sys.argv) < 2:
        print("Usage: python find_largest_numbers.py <filename> [options]")
        print("Options:")
        print("  --top N          Show top N numbers for each category (default: 10)")
        print("  --threshold T    Only show numbers with absolute value >= T")
        print("  --show-lines     Show the content of lines containing the numbers")
        print("  --all           Show all numbers (no limit)")
        print("")
        print("This script finds:")
        print("  - Top N largest positive numbers")
        print("  - Top N smallest negative numbers (most negative)")
        print("  - Top N smallest positive numbers (closest to zero from positive side)")
        print("  - Top N largest negative numbers (closest to zero from negative side)")
        return
    
    filename = sys.argv[1]
    top_n = 10
    threshold = None
    show_lines = False
    
    # Parse command line arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--top' and i + 1 < len(sys.argv):
            try:
                top_n = int(sys.argv[i + 1])
                i += 2
            except ValueError:
                print("Error: --top requires an integer value")
                return
        elif sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            try:
                threshold = float(sys.argv[i + 1])
                i += 2
            except ValueError:
                print("Error: --threshold requires a numeric value")
                return
        elif sys.argv[i] == '--show-lines':
            show_lines = True
            i += 1
        elif sys.argv[i] == '--all':
            top_n = None
            i += 1
        else:
            print(f"Unknown option: {sys.argv[i]}")
            return
    
    # Find and display results
    largest_pos, smallest_neg, smallest_pos, largest_neg = find_largest_numbers(filename, top_n, threshold)
    print_results(largest_pos, smallest_neg, smallest_pos, largest_neg, show_lines, filename)

if __name__ == "__main__":
    main()