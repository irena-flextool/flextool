"""Convenience wrapper for Spine Toolbox. Prefer `flextool-write-outputs` command after pip install."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.process_outputs.write_outputs import write_outputs  # noqa: F401 (re-exported for backward compat)
from flextool.cli.write_outputs import main

if __name__ == '__main__':
    main()
