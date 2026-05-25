"""Convenience wrapper. Run directly: python update_flextool.py <args>"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.cmd_update_flextool import main

if __name__ == '__main__':
    main()
