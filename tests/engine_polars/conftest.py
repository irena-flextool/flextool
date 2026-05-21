"""Pytest sys.path setup for engine_polars integration tests.

Layers:
  * polar-high is installed as a real dependency (``pip install -e``).
  * ``flextool.engine_polars`` is in the flextool-engine repo and importable
    via the repo's package layout (no path injection needed).
  * ``tests/engine_polars/fixtures/`` — synthetic flextool-flavoured fixtures
    that import flextool, kept on sys.path so test files can do
    ``from flex_toy_<feature> import ...``.

Also resets the global axis-enums :class:`contextvars.ContextVar` between
tests so a prior test's ``load_flextool`` doesn't pollute the next test's
joins with stale Enum vocabularies (cross-test ContextVar leak).
"""

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
if str(FIXTURES) not in sys.path:
    sys.path.insert(0, str(FIXTURES))
# Under pytest --import-mode=importlib, the directory of a test file is not
# automatically prepended to sys.path.  We host ``_golden.py`` and a couple
# of other private helpers next to the test files, and they're imported by
# bare name (``from _golden import ...``).  Make those resolvable.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DATA_DIR = HERE / "data"


@pytest.fixture(autouse=True)
def _reset_global_axis_enums():
    """Reset the cascade-wide axis_enums ContextVar before each test.

    ``load_flextool`` sets the global ContextVar on success (input.py
    finally block ~line 4089) so ``build_flextool`` and other
    post-load consumers see the live vocabulary.  But the
    ContextVar persists into the next test in the same pytest worker,
    pinning Enum dtypes that may differ from the next test's
    fixture's vocabulary — surfacing as "enum on left does not match
    enum on right" SchemaError in joins.

    Resetting before each test guarantees a clean slate.
    """
    from flextool.engine_polars._axis_enums import set_global_axis_enums
    set_global_axis_enums(None)
    yield
    set_global_axis_enums(None)
