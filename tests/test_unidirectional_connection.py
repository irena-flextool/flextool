"""Tests for unidirectional connection transfer_method.

Verifies that a connection with ``transfer_method="unidirectional"`` is
classified into ``method_1way_1var_off``: a single non-negative flow
variable from source to sink, no reverse-direction variable.

The model side (flextool.mod) already handles ``method_1way_1var_off``
for any process — ``process_source_toSink`` admits any ``method_direct``
method, ``process_sink_toSource`` is only built for ``method_2way_2var``,
and ``p_flow_min`` is 0 for everything except ``method_2way_1var_off``.
So the feature reduces to making ``unidirectional`` a valid ct_method
that resolves correctly in ``METHODS_MAPPING`` (Python) / ``set methods``
(flextool_base.dat).
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, import_data

from flextool.input_derivation._specs import METHODS_MAPPING
from flextool.input_derivation._process_method import derive_process_method
from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.spinedb_backend import SpineDBBackend
import spinedb_api as _spinedb_api


REPO_ROOT = Path(__file__).parent.parent


def test_methods_mapping_has_unidirectional() -> None:
    """``METHODS_MAPPING`` must resolve unidirectional → method_1way_1var_off."""
    assert METHODS_MAPPING.get(
        ("unidirectional", "no_startup", "fork_no")
    ) == "method_1way_1var_off"


def test_base_dat_declares_unidirectional() -> None:
    """``flextool_base.dat`` declares unidirectional in ``set ct_method`` and
    ``set methods`` (must stay in sync with the Python ``METHODS_MAPPING``)."""
    base_dat = (REPO_ROOT / "flextool" / "flextool_base.dat").read_text()

    ct_method_block = base_dat.split("set ct_method", 1)[1].split(";", 1)[0]
    assert "unidirectional" in ct_method_block, (
        "'unidirectional' must be listed in 'set ct_method' in flextool_base.dat"
    )

    assert (
        "(unidirectional, no_startup, fork_no, method_1way_1var_off)" in base_dat
    ), (
        "'set methods' in flextool_base.dat must map (unidirectional, no_startup, "
        "fork_no) → method_1way_1var_off"
    )


def _create_minimal_db(db_path: Path) -> str:
    url = f"sqlite:///{db_path}"
    with DatabaseMapping(url, create=True) as db:
        _, errors = import_data(
            db,
            entity_classes=[
                ("node", ()),
                ("unit", ()),
                ("connection", ()),
                ("unit__inputNode", ("unit", "node")),
                ("unit__outputNode", ("unit", "node")),
                ("connection__node__node", ("connection", "node", "node")),
            ],
            parameter_definitions=[
                ("connection", "transfer_method"),
                ("connection", "startup_method"),
                ("connection", "delay"),
                ("unit", "conversion_method"),
                ("unit", "startup_method"),
                ("unit", "delay"),
                ("unit", "minimum_time_method"),
            ],
        )
        assert not errors, f"DB init errors: {errors}"
        db.commit_session("init schema")
    return url


def test_write_process_method_routes_unidirectional_to_1way_1var_off(
    tmp_path: Path,
) -> None:
    """``_write_process_method`` classifies a unidirectional connection
    as ``method_1way_1var_off`` in ``input/process_method.csv``.

    This is the user-visible plumbing check: even if a reviewer forgot
    to add the mapping, this would flag it because the process would
    either be skipped (logged warning, no CSV row) or wrongly classified.
    """
    db_url = _create_minimal_db(tmp_path / "uni.sqlite")
    with DatabaseMapping(db_url) as db:
        _, errors = import_data(
            db,
            entities=[
                ("node", "from_node"),
                ("node", "to_node"),
                ("connection", "uni_line"),
                ("connection__node__node", ("uni_line", "from_node", "to_node")),
            ],
            parameter_values=[
                ("connection", "uni_line", "transfer_method", "unidirectional"),
            ],
        )
        assert not errors, f"data import errors: {errors}"
        db.commit_session("test data")

    logger = logging.getLogger("test_unidirectional_connection")
    provider = FlexDataProvider()
    with DatabaseMapping(db_url) as db:
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        backend = SpineDBBackend.__new__(SpineDBBackend)
        backend._db = db                                  # type: ignore[attr-defined]
        backend._api = _spinedb_api                       # type: ignore[attr-defined]
        backend._precision_digits = 0                     # type: ignore[attr-defined]
        derive_process_method(backend, provider, logger,
                              ct_method_overrides=None)
    frame = provider.get("input/process_method")
    assert frame is not None and frame.height > 0, "process_method frame empty"
    methods = {row["process"]: row["method"]
               for row in frame.iter_rows(named=True)}
    assert methods.get("uni_line") == "method_1way_1var_off", (
        f"uni_line was classified as {methods.get('uni_line')!r}; "
        f"expected 'method_1way_1var_off'. Frame: {frame}"
    )
