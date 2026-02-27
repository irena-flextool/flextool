"""Convenience wrapper. Prefer `flextool-migrate` command after pip install."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flextool.update_flextool import migrate_database  # noqa: F401
from flextool.cli.migrate_database import main

if __name__ == '__main__':
    main()
