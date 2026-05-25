"""Convenience wrapper. Run directly: python write_outputs.py <args>"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.cmd_write_outputs import main

if __name__ == '__main__':
    main()
