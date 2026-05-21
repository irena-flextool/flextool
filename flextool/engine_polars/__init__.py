"""flextool integration for flexpy.

This subpackage is **flextool-specific** — it knows the on-disk layout
of flextool's ``input/`` + ``solve_data/`` CSVs (``input.py``) and the
shape of flextool's optimization model (``model.py``).

The flexpy engine (``src/flexpy``) knows nothing of flextool.  When
this package solidifies it will move to flextool's repository and
flexpy stays as the pure LP eDSL kernel.
"""

# Determinism wrapper for polar_high.Problem.add_var / add_cstr.
#
# polars 1.40 .unique() defaults to maintain_order=False and joins
# don't promise stable row order.  Across processes the hash-bucket
# placement of strings changes with PYTHONHASHSEED, so the index
# frames flextool passes to add_var (and the over frames passed to
# add_cstr) can land in different row orders run-to-run.  That gave
# us different LP column orderings — and HiGHS, fed the same LP with
# different column orderings, picks different vertices among alt-
# optima, which flipped scenario goldens.
#
# Site-by-site sorting at every .unique()/.join() in the cascade is
# whack-a-mole.  A single-point fix at the LP emission boundary in
# polar_high.Problem.add_var / add_cstr is robust: sort the (dim or
# axis) columns of the frame just before col_ids/row_ids are assigned
# so the LP is byte-identical across processes regardless of upstream
# polars hash-bucket randomness.
def _install_polar_high_determinism() -> None:
    import polar_high.engine as _phe
    if getattr(_phe.Problem.add_var, "_flextool_deterministic", False):
        return  # idempotent
    _orig_add_var = _phe.Problem.add_var
    _orig_add_cstr = _phe.Problem.add_cstr

    def add_var(self, name, dims, index, lower=0.0, upper=float("inf"),
                integer=False):
        if isinstance(dims, str):
            dim_tuple = (dims,)
        else:
            dim_tuple = tuple(dims)
        # Sort by dim columns when all are present so col_ids are
        # assigned in a canonical order.  Empty / scalar frames fall
        # through unchanged.
        if (index is not None and index.height > 0
                and all(d in index.columns for d in dim_tuple)):
            index = index.sort(list(dim_tuple))
        return _orig_add_var(self, name, dims, index, lower=lower,
                              upper=upper, integer=integer)
    add_var._flextool_deterministic = True

    def add_cstr(self, name, *, over=None, sense, lhs_terms,
                  rhs_terms=None):
        if over is not None and over.height > 0:
            over = over.sort(list(over.columns))
        return _orig_add_cstr(self, name, over=over, sense=sense,
                                lhs_terms=lhs_terms, rhs_terms=rhs_terms)
    add_cstr._flextool_deterministic = True

    _phe.Problem.add_var = add_var
    _phe.Problem.add_cstr = add_cstr


_install_polar_high_determinism()


from flextool.engine_polars.input import (
    FlexData, load_flextool,
)
from flextool.engine_polars.model import build_flextool
from flextool.engine_polars.chain import run_chain, ChainStep
from flextool.engine_polars._input_source import FlexInputSource, CsvSource, InputSource
from flextool.engine_polars._spinedb_reader import SpineDbReader
from flextool.engine_polars._inmemory_reader import InMemoryReader
from flextool.engine_polars._solve_handoff import (
    SolveHandoff, write_fix_storage_files_from_handoff,
)
from flextool.engine_polars._orchestration import (
    OrchestrationStep, run_chain_from_db, run_orchestration,
    run_single_solve_from_db,
)
from flextool.engine_polars._fast_load import (
    FastLoadError, load_flextool_source_only,
)

__all__ = [
    "FlexData", "load_flextool", "build_flextool",
    "run_chain", "ChainStep",
    "FlexInputSource", "CsvSource",
    "InputSource", "SpineDbReader", "InMemoryReader",
    # Γ.8.D — native orchestrator + handoff carrier.
    "SolveHandoff",
    "write_fix_storage_files_from_handoff",
    "OrchestrationStep", "run_chain_from_db", "run_orchestration",
    # Δ.25 — surgical fast single-solve path.
    "run_single_solve_from_db",
    "FastLoadError", "load_flextool_source_only",
]
