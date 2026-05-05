"""Pytest sys.path setup for engine_polars integration tests.

Layers:
  * polar-high is installed as a real dependency (``pip install -e``).
  * ``flextool.engine_polars`` is in the flextool-engine repo and importable
    via the repo's package layout (no path injection needed).
  * ``tests/engine_polars/fixtures/`` — synthetic flextool-flavoured fixtures
    that import flextool, kept on sys.path so test files can do
    ``from flex_toy_<feature> import ...``.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))

DATA_DIR = HERE / "data"
