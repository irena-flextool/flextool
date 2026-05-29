"""Tests for the warning emitted when a ``connection`` entity has no
``connection__node__node`` relationship defining its endpoints.

The warning is emitted by
``flextool.input_derivation._validators.validate_connection_node_memberships``.
Such a connection cannot transfer between nodes; if it also carries an
``invest_method`` it still becomes a degenerate (always-zero) ``v_invest``
column, which is the input shape that previously crashed the output post-
processor with ``KeyError: ['MON-PYR'] not in index``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping

from flextool.input_derivation._validators import (
    validate_connection_node_memberships,
)


def _build_db(tmp_path: Path,
              connections: list[str],
              cnn_rows: list[tuple[str, str, str]]) -> str:
    """Create a fresh SQLite DB with ``connection``, ``node`` and
    ``connection__node__node`` classes, the supplied connection entities,
    and the supplied ``connection__node__node`` byname triples.
    """
    db_path = tmp_path / "connection_warnings_test.sqlite"
    url = f"sqlite:///{db_path}"

    node_names: set[str] = set()
    for _, n1, n2 in cnn_rows:
        node_names.add(n1)
        node_names.add(n2)

    with DatabaseMapping(url, create=True) as db:
        db.add_update_item("entity_class", name="node")
        db.add_update_item("entity_class", name="connection")
        db.add_update_item("entity_class", name="connection__node__node",
                           dimension_name_list=("connection", "node", "node"))

        for c in connections:
            db.add_update_item("entity", entity_class_name="connection",
                               name=c, entity_byname=(c,))
        for n in node_names:
            db.add_update_item("entity", entity_class_name="node",
                               name=n, entity_byname=(n,))
        for row in cnn_rows:
            db.add_update_item("entity",
                               entity_class_name="connection__node__node",
                               entity_byname=row)

        db.commit_session("connection warnings test fixture")

    return url


def _run_validation(url: str, caplog) -> list[logging.LogRecord]:
    logger = logging.getLogger("test_connection_node_warnings")
    caplog.set_level(logging.WARNING, logger=logger.name)
    with DatabaseMapping(url) as db:
        db.fetch_all("entity")
        validate_connection_node_memberships(db, logger)
    return [r for r in caplog.records if r.name == logger.name]


def test_connection_without_endpoints_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A connection with no connection__node__node row must warn — this is
    the ``MON-PYR`` shape from the Cyprus model."""
    url = _build_db(
        tmp_path,
        connections=["MON-PYR", "AAT-MGS"],
        cnn_rows=[("AAT-MGS", "AAT", "MGS")],
    )
    records = _run_validation(url, caplog)
    msgs = [r.getMessage() for r in records]
    assert any("MON-PYR" in m and "connection__node__node" in m
               for m in msgs), msgs
    # The well-formed connection must NOT warn.
    assert not any("AAT-MGS" in m for m in msgs), msgs


def test_all_connections_wired_is_silent(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    url = _build_db(
        tmp_path,
        connections=["AAT-MGS", "MGS-MON"],
        cnn_rows=[("AAT-MGS", "AAT", "MGS"), ("MGS-MON", "MGS", "MON")],
    )
    records = _run_validation(url, caplog)
    assert records == [], [r.getMessage() for r in records]
