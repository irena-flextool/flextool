"""Coverage + validity guardrails for derived output-column metadata.

The metadata is *derived* from a small transform catalog (see
``flextool/process_outputs/_output_meta.py``).  These tests keep the
derivation honest and ratchet coverage upward — a NEW processed output may
not land without either a transform declaration or an explicit allowlist
entry, and allowlist entries must be removed once the output is declared.
"""

import re

import pandas as pd

from flextool.engine.output._parquet_bundle import REGISTRY
from flextool.process_outputs._output_meta import (
    OUTPUT_DESCRIPTION,
    OUTPUT_DIMS,
    OUTPUT_TRANSFORM,
    FORMULA_OVERRIDE,
    DOCS_ANCHOR,
    ENTITY_LEVELS,
    LEVEL_KINDS,
    OVERLAPPING_LEVELS,
    TIME_LIKE_LEVELS,
    Transform,
    Semantics,
    derive_column_meta,
    level_kind,
)
from flextool.process_outputs.write_spinedb import SPINEDB_OUTPUT_SPEC

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


def _result_tables() -> set[str]:
    """Every result table the docs describe: the declared outputs and the results-database tables."""
    return set(OUTPUT_TRANSFORM) | set(SPINEDB_OUTPUT_SPEC)


def test_every_result_table_has_a_description_and_dims():
    """Ratchet: OUTPUT_DESCRIPTION and OUTPUT_DIMS cover every result table, and only those."""
    tables = _result_tables()
    assert sorted(tables - set(OUTPUT_DESCRIPTION)) == [], "result tables without an OUTPUT_DESCRIPTION"
    assert sorted(set(OUTPUT_DESCRIPTION) - tables) == [], "OUTPUT_DESCRIPTION keys that are no result table"
    assert sorted(tables - set(OUTPUT_DIMS)) == [], "result tables without OUTPUT_DIMS"
    assert sorted(set(OUTPUT_DIMS) - tables) == [], "OUTPUT_DIMS keys that are no result table"


def test_descriptions_are_one_sentence():
    for key, text in OUTPUT_DESCRIPTION.items():
        assert text and text == text.strip(), key
        assert text[0].isupper() and text.endswith("."), key
        assert not re.search(r"[.;] [A-Z]", text[:-1]), f"{key}: more than one sentence"


def test_dims_are_consistent():
    """Counterparts pair up (each names the other, one per step and one per period); a table named for a
    per-step layout has a time level and one named for a per-period layout has none."""
    for key, dims in OUTPUT_DIMS.items():
        assert dims.rows, key
        assert "scenario" not in dims.rows + dims.columns, key
        if dims.counterpart is not None:
            other = OUTPUT_DIMS[dims.counterpart]
            assert other.counterpart == key, key
            assert dims.has_time != other.has_time, key
            assert dims.has_period and other.has_period, key
        if "_dt" in key:
            assert dims.has_time, key
        if re.search(r"(_d_|__d$)", key) and "_dt" not in key:
            assert not dims.has_time, key
        if isinstance(OUTPUT_TRANSFORM.get(key), Transform) and OUTPUT_TRANSFORM[key].semantics is Semantics.DIMENSION:
            assert dims.columns == (), f"{key}: a membership set has no column levels"


#: Level names that hold different quantities or parts in some table: never a default entity level.
_NOT_ENTITY_NAMES = {"parameter", "param", "param_CO2", "category", "type", "item", "scope", "path", "updown",
                     "reserve", "group", "node_group"}


def test_every_level_has_a_kind():
    """Ratchet: every level of every result table that is not a time level has a kind, declared in the table's
    ``kinds`` or given by its name (ENTITY_LEVELS / OVERLAPPING_LEVELS); only known kinds are declared, only for
    the table's own levels; and the names that mean different things in different tables are never a default."""
    assert not ENTITY_LEVELS & _NOT_ENTITY_NAMES
    assert OVERLAPPING_LEVELS == {"group", "node_group"}
    for key, dims in OUTPUT_DIMS.items():
        declared = dict(dims.kinds)
        assert len(declared) == len(dims.kinds), f"{key}: a level declared twice"
        for level, kind in dims.kinds:
            assert level in dims.rows + dims.columns, f"{key}: kinds names {level}, not a level of the table"
            assert kind in LEVEL_KINDS, f"{key}: unknown kind {kind}"
            assert level not in TIME_LIKE_LEVELS, f"{key}: a time level needs no kind"
        for level in dims.rows + dims.columns:
            if level in TIME_LIKE_LEVELS:
                continue
            assert level in declared or level in ENTITY_LEVELS or level in OVERLAPPING_LEVELS, (
                f"{key}: the level {level} has no kind (declare it in OutputDims.kinds)")


def test_level_kinds_of_some_tables():
    assert level_kind("unit_capacity_ed_p", "parameter") == "quantity"
    assert level_kind("unit_capacity_ed_p", "unit") == "entity"
    assert level_kind("unit_capacity_ed_p", "column_level_1") == "quantity"      # undeclared: never combined
    assert level_kind("annualized_costs_d_p", "category") == "part"
    assert level_kind("node_d_ep", "category") == "quantity"
    assert level_kind("nodeGroup_flows_d_gpe", "item") == "entity"
    assert level_kind("nodeGroup_flows_d_gpe", "type") == "quantity"
    assert level_kind("CO2_d_g", "group") == "overlapping"
    assert level_kind("nodeGroup_slack_reserve_d_eeg", "node_group") == "overlapping"
    assert level_kind("nodeGroup_slack_reserve_d_eeg", "reserve") == "quantity"
    assert level_kind("node_prices_dt_e", "solve") == "time"
    assert level_kind("node_prices_dt_e", "period") == "period"
    assert level_kind("no_such_table", "unit") == "entity"
