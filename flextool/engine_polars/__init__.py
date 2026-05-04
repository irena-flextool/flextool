"""flextool integration for flexpy.

This subpackage is **flextool-specific** — it knows the on-disk layout
of flextool's ``input/`` + ``solve_data/`` CSVs (``input.py``) and the
shape of flextool's optimization model (``model.py``).

The flexpy engine (``src/flexpy``) knows nothing of flextool.  When
this package solidifies it will move to flextool's repository and
flexpy stays as the pure LP eDSL kernel.
"""

from flextool.engine_polars.input import (
    FlexData, load_flextool, load_flextool_from_db, apply_handoff,
)
from flextool.engine_polars.model import build_flextool
from flextool.engine_polars.chain import run_chain, ChainStep
from flextool.engine_polars._input_source import FlexInputSource, CsvSource, InputSource
from flextool.engine_polars._spinedb_source import SpineDbSource
from flextool.engine_polars._spinedb_reader import SpineDbReader
from flextool.engine_polars._inmemory_reader import InMemoryReader

__all__ = [
    "FlexData", "load_flextool", "load_flextool_from_db", "build_flextool",
    "apply_handoff", "run_chain", "ChainStep",
    "FlexInputSource", "CsvSource", "SpineDbSource",
    "InputSource", "SpineDbReader", "InMemoryReader",
]
