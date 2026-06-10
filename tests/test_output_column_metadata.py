"""Coverage + validity guardrails for derived output-column metadata.

The metadata is *derived* from a small transform catalog (see
``flextool/process_outputs/_output_meta.py``).  These tests keep the
derivation honest and ratchet coverage upward — a NEW processed output may
not land without either a transform declaration or an explicit allowlist
entry, and allowlist entries must be removed once the output is declared.
"""

import pandas as pd

from flextool.engine_polars._parquet_bundle import REGISTRY
from flextool.process_outputs._output_meta import (
    OUTPUT_TRANSFORM,
    FORMULA_OVERRIDE,
    DOCS_ANCHOR,
    Transform,
    Semantics,
    derive_column_meta,
)

# Processed outputs not yet given a transform declaration.  This list may only
# SHRINK: declaring one of these requires removing it here (enforced by
# ``test_allowlist_has_no_stale_entries``), and a new undeclared output trips
# ``test_no_new_undeclared_processed_outputs``.
# All processed outputs are now declared (84/84).  Any new processed output
# must be added to OUTPUT_TRANSFORM (or, exceptionally, allowlisted here).
UNDECLARED_ALLOWLIST: set[str] = set()


def _processed_keys() -> set[str]:
    return {k for k, s in REGISTRY.items() if s.category == "processed"}


def test_declarations_are_valid():
    """Every declared transform (single or per-column map) is valid."""
    for key, spec in OUTPUT_TRANSFORM.items():
        # A declaration is either a single Transform or a {column: Transform} map.
        entries = spec.values() if isinstance(spec, dict) else [spec]
        for tf in entries:
            assert isinstance(tf, Transform), f"{key}: not a Transform"
            assert isinstance(tf.semantics, Semantics), f"{key}: bad semantics"
            assert tf.tooltip, f"{key}: empty tooltip"
            # Measures must carry a unit unless dimensionless (ratio) or a pure
            # dimension/membership table.
            assert tf.unit or tf.semantics in (
                Semantics.RATIO, Semantics.DIMENSION), f"{key}: missing unit"


def test_overrides_reference_declared_outputs():
    for (key, _col) in FORMULA_OVERRIDE:
        assert key in OUTPUT_TRANSFORM, f"FORMULA_OVERRIDE references undeclared {key}"
    for key in DOCS_ANCHOR:
        assert key in OUTPUT_TRANSFORM, f"DOCS_ANCHOR references undeclared {key}"


def test_derive_marks_dimensions_and_measures():
    cols = pd.Index(
        ["period", "commodity_cost", "co2", "starts"], name="category")
    meta = derive_column_meta("annualized_costs_d_p", cols)
    assert meta is not None
    assert meta["period"].semantics is Semantics.DIMENSION
    assert meta["commodity_cost"].semantics is Semantics.ANNUALIZED
    assert meta["commodity_cost"].unit == "M CUR/a"
    # Per-column formula override is applied.
    assert "period_share" in meta["commodity_cost"].formula
    # No bogus override leaks onto other columns.
    assert meta["co2"].formula == ""


def test_derive_returns_none_for_undeclared():
    assert derive_column_meta("definitely_not_an_output", ["a", "b"]) is None


def test_no_new_undeclared_processed_outputs():
    """A new processed output must be declared or explicitly allowlisted."""
    undeclared = _processed_keys() - set(OUTPUT_TRANSFORM)
    new = undeclared - UNDECLARED_ALLOWLIST
    assert not new, (
        "New undeclared processed outputs — add a Transform to "
        f"OUTPUT_TRANSFORM (or allowlist): {sorted(new)}"
    )


def test_allowlist_has_no_stale_entries():
    """Ratchet: once an output is declared (or removed), drop it from the
    allowlist so coverage can only move forward."""
    undeclared = _processed_keys() - set(OUTPUT_TRANSFORM)
    stale = UNDECLARED_ALLOWLIST - undeclared
    assert not stale, (
        "Stale allowlist entries (now declared or no longer a processed "
        f"output) — remove them: {sorted(stale)}"
    )
