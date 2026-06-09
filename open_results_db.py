"""Convenience wrapper. Run directly: python open_results_db.py <args>"""
import sys
import os

# Ensure flextool package is importable when run directly from the root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.cmd_open_results_db import main

if __name__ == '__main__':
    sys.exit(main())
