"""Convenience wrapper. Run directly: python run_flextool.py <args>"""
import sys
import os

# Ensure flextool package is importable when run directly from the root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.cmd_run_flextool import main

if __name__ == '__main__':
    main()
