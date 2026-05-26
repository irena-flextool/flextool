"""Phase 1 ingestion guard for ``node.storage_binding_method``.

The 2026-04 list-valued design (now being reverted) silently flattened
array-typed ``storage_binding_method`` values into one row per array
element, which downstream additive logic in ``calc_storage_vre.py``
turned into double-counted state-change residuals.

The single-valued contract is enforced at ingestion in
:func:`flextool.spinedb_backend._backend.SpineDBBackend.parameter_values`
— the ``elif ptype in ("array", "time_series"):`` branch raises a
:class:`flextool.engine_polars._solve_state.FlexToolConfigError` with
an actionable message naming the offending entity, the array contents,
the six allowed single-string values, and pointing at the v52->v53
migration.  This test pins that behaviour.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import Array, DatabaseMapping, import_data

from flextool.engine_polars._solve_state import FlexToolConfigError
from flextool.spinedb_backend import SpineDBBackend


def _create_minimal_db_with_array_value(db_path: Path) -> str:
    """Build an in-memory SpineDB with one node and an array-valued
    ``storage_binding_method`` parameter.

    Mirrors the legacy v52-or-older shape that the migration must
    reject.  The value carries three methods, mirroring the
    H2_trade.sqlite fixture from audit §9.
    """
    url = f"sqlite:///{db_path}"
    with DatabaseMapping(url, create=True) as db:
        _, errors = import_data(
            db,
            entity_classes=[
                ("node", ()),
            ],
            parameter_definitions=[
                ("node", "storage_binding_method"),
            ],
        )
        assert not errors, f"DB init errors: {errors}"
        db.commit_session("init schema")

    bad_value = Array(
        ["bind_using_blended_weights", "bind_within_period"],
        value_type=str,
        index_name="i",
    )
    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            entities=[
                ("node", "ARG_H2"),
            ],
            parameter_values=[
                ("node", "ARG_H2", "storage_binding_method", bad_value),
            ],
        )
        assert not errors, f"data import errors: {errors}"
        db.commit_session("test data")
    return url


def test_array_valued_storage_binding_method_is_rejected(
    tmp_path: Path,
) -> None:
    """Array-typed ``node.storage_binding_method`` must raise
    :class:`FlexToolConfigError` with a message that names the
    entity, lists the array contents, and mentions the single-string
    contract plus ``bind_using_blended_weights`` as the recommended
    pick for representative-period blended-weights nodes.
    """
    url = _create_minimal_db_with_array_value(tmp_path / "guard.sqlite")
    backend = SpineDBBackend(url)
    try:
        with pytest.raises(FlexToolConfigError) as exc_info:
            backend.parameter_values(
                cl_pars=[("node", "storage_binding_method")],
                header="node,storage_binding_method",
            )
    finally:
        backend.close()

    msg = str(exc_info.value)
    # Entity name surfaced.
    assert "ARG_H2" in msg, (
        "guard error message must name the offending entity; got:\n"
        + msg
    )
    # Array contents surfaced.
    assert "bind_using_blended_weights" in msg, msg
    assert "bind_within_period" in msg, msg
    # Single-string contract messaging.
    assert "single" in msg.lower(), (
        "guard message must mention the single-string contract; got:\n"
        + msg
    )
    # Recommended pick called out explicitly.
    assert "bind_using_blended_weights" in msg, msg
