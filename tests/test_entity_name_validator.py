"""Tests for the hyphen-in-entity-name validator.

Hyphens in entity names collide with MathProg's subtraction operator and
surface at solve time as an ``out of domain`` error on a *different*
symbol (often a neighbouring node name). ``_validate_entity_names_no_hyphen``
catches them at write-time with a clear message.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, import_data

from flextool.flextoolrunner.input_writer import _validate_entity_names_no_hyphen
from flextool.flextoolrunner.runner_state import FlexToolConfigError


def _mkdb(tmp_path: Path, entities: list[tuple[str, str]]) -> str:
    """Create a minimal Spine DB with the given single-dim entities."""
    url = f"sqlite:///{tmp_path / 'v.sqlite'}"
    with DatabaseMapping(url, create=True) as db:
        _, errors = import_data(
            db,
            entity_classes=[
                ("node", ()), ("unit", ()), ("connection", ()), ("group", ()),
                ("commodity", ()), ("profile", ()), ("constraint", ()),
            ],
            entities=entities,
        )
        assert not errors, errors
        db.commit_session("init")
    return url


def test_accepts_clean_names(tmp_path: Path) -> None:
    """No hyphens → returns silently."""
    url = _mkdb(tmp_path, [
        ("node", "ARG_H2"),
        ("connection", "BRA_ARG_H2"),
        ("unit", "ARG_electrolyser"),
    ])
    with DatabaseMapping(url) as db:
        db.fetch_all("entity")
        _validate_entity_names_no_hyphen(db)  # must not raise


def test_rejects_hyphen_in_connection(tmp_path: Path) -> None:
    """A hyphenated connection name must raise with the name in the message."""
    url = _mkdb(tmp_path, [
        ("node", "ARG_H2"),
        ("node", "BRA_H2"),
        ("connection", "BRA_ARG-H2"),
    ])
    with DatabaseMapping(url) as db:
        db.fetch_all("entity")
        with pytest.raises(FlexToolConfigError) as exc:
            _validate_entity_names_no_hyphen(db)
    msg = str(exc.value)
    assert "BRA_ARG-H2" in msg
    assert "connection" in msg


def test_groups_offenders_by_class(tmp_path: Path) -> None:
    """Offenders are grouped by entity class and the total is reported."""
    url = _mkdb(tmp_path, [
        ("connection", "A-B"),
        ("connection", "C-D"),
        ("unit", "gen-x"),
        ("node", "clean_node"),
    ])
    with DatabaseMapping(url) as db:
        db.fetch_all("entity")
        with pytest.raises(FlexToolConfigError) as exc:
            _validate_entity_names_no_hyphen(db)
    msg = str(exc.value)
    assert "connection (2)" in msg
    assert "unit (1)" in msg
    assert "3 entity name(s) to fix" in msg
    assert "clean_node" not in msg  # clean names stay out of the error


def test_truncates_long_lists(tmp_path: Path) -> None:
    """More than 5 offenders in one class → show first 5 + count of remainder."""
    url = _mkdb(tmp_path, [
        ("connection", f"link-{i:02d}") for i in range(8)
    ])
    with DatabaseMapping(url) as db:
        db.fetch_all("entity")
        with pytest.raises(FlexToolConfigError) as exc:
            _validate_entity_names_no_hyphen(db)
    msg = str(exc.value)
    assert "+3 more" in msg
    assert "8 entity name(s) to fix" in msg
