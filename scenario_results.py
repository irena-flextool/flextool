"""Convenience wrapper. Run directly: python scenario_results.py <args>"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.cmd_scenario_results import cli

if __name__ == '__main__':
    cli()
