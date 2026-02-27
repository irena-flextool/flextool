"""Convenience wrapper for Spine Toolbox. Prefer `flextool` command after pip install."""
import sys
import os

# Ensure flextool package is importable when run directly from the root directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.cli.run_flextool import main

if __name__ == '__main__':
    main()
