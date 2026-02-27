"""Convenience wrapper. Run directly: python execute_flextool_workflow.py <args>"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.execute_flextool_workflow import main

if __name__ == '__main__':
    main()
