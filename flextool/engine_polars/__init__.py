"""flextool integration for flexpy.

This subpackage is **flextool-specific** — it knows the on-disk layout
of flextool's ``input/`` + ``solve_data/`` CSVs (``input.py``) and the
shape of flextool's optimization model (``model.py``).

The flexpy engine (``src/flexpy``) knows nothing of flextool.  When
this package solidifies it will move to flextool's repository and
flexpy stays as the pure LP eDSL kernel.
"""

from flextool.engine_polars.input import (
    FlexData, load_flextool, load_flextool_from_db,
)
from flextool.engine_polars.model import build_flextool
from flextool.engine_polars.chain import run_chain, ChainStep
from flextool.engine_polars._input_source import FlexInputSource, CsvSource, InputSource
from flextool.engine_polars._spinedb_source import SpineDbSource
from flextool.engine_polars._spinedb_reader import SpineDbReader
from flextool.engine_polars._inmemory_reader import InMemoryReader
from flextool.engine_polars._solve_handoff import (
    SolveHandoff, capture_post_solve, write_fix_storage_files_from_handoff,
)
from flextool.engine_polars._orchestration import (
    OrchestrationStep, run_chain_from_db, run_orchestration,
    run_single_solve_from_db,
)
from flextool.engine_polars._fast_load import (
    FastLoadError, load_flextool_source_only,
)

__all__ = [
    "FlexData", "load_flextool", "load_flextool_from_db", "build_flextool",
    "run_chain", "ChainStep",
    "FlexInputSource", "CsvSource", "SpineDbSource",
    "InputSource", "SpineDbReader", "InMemoryReader",
    # Γ.8.D — native orchestrator + handoff carrier.
    "SolveHandoff", "capture_post_solve",
    "write_fix_storage_files_from_handoff",
    "OrchestrationStep", "run_chain_from_db", "run_orchestration",
    # Δ.25 — surgical fast single-solve path.
    "run_single_solve_from_db",
    "FastLoadError", "load_flextool_source_only",
]
