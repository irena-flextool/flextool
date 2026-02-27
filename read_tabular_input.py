"""Convenience wrapper. Run directly: python read_tabular_input.py <args>"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.read_tabular_input import main

if __name__ == '__main__':
    main()
