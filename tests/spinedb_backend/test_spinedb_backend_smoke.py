"""Smoke test: :class:`SpineDBBackend` opens the Rivendell DB and
returns canonical-schema frames for the three spec families.

This is the Item-1 contract test for Step 2.5-A.  Subsequent items
(2-4) port the legacy spec loops into the Backend's materialiser
methods; this test gates that the package skeleton works and the
three method signatures behave on a real fixture.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.spinedb_backend import SpineDBBackend


RIVENDELL_DB = (
    Path(__file__).resolve().parents[2]
    / "projects" / "Rivendell" / "input_sources" / "rivendell.sqlite"
)


pytestmark = pytest.mark.skipif(
    not RIVENDELL_DB.exists(),
    reason=f"Rivendell DB not present at {RIVENDELL_DB}",
)


def test_backend_constructs_and_closes() -> None:
    """Constructor opens the DB, ``close()`` releases the handle."""
    backend = SpineDBBackend(str(RIVENDELL_DB))
    assert backend.db_url.startswith("sqlite:///")
    assert backend.scenario_name is None
    backend.close()
    # close() is idempotent.
    backend.close()


def test_backend_context_manager() -> None:
    """``with`` clause opens and closes the handle."""
    with SpineDBBackend(str(RIVENDELL_DB)) as backend:
        # Inside the context the DB is live.
        frame = backend.entities(
            classes=("commodity",),
            header="commodity",
        )
        assert isinstance(frame, pl.DataFrame)
    # After exit the handle is closed.


def test_entities_commodity_nonempty() -> None:
    """Rivendell has commodity entities; the canonical frame has the
    single column ``commodity`` and at least one row."""
    with SpineDBBackend(str(RIVENDELL_DB)) as backend:
        frame = backend.entities(
            classes=("commodity",),
            header="commodity",
        )
    assert frame.columns == ["commodity"]
    assert frame.height > 0
    assert frame.schema["commodity"] == pl.Utf8


def test_entities_unknown_class_returns_empty_with_schema() -> None:
    """An entity class with no rows yields an empty Utf8 frame with the
    requested columns."""
    with SpineDBBackend(str(RIVENDELL_DB)) as backend:
        frame = backend.entities(
            classes=("__nonexistent_class__",),
            header="x,y,z",
        )
    assert frame.columns == ["x", "y", "z"]
    assert frame.height == 0
    for col in ("x", "y", "z"):
        assert frame.schema[col] == pl.Utf8


def test_parameter_defaults_returns_frame() -> None:
    """The two-entry ``_DEFAULT_VALUES_SPECS`` is supported as-is."""
    with SpineDBBackend(str(RIVENDELL_DB)) as backend:
        frame = backend.parameter_defaults(
            cl_pars=[("node", "penalty_up"), ("node", "penalty_down")],
            header="class,paramName,default_value",
            filter_in_type=["float", "str", "bool"],
        )
    # The Rivendell fixture has defaults for these node params; the
    # frame schema must match the canonical header order even if rows
    # are zero.
    assert frame.columns == ["class", "paramName", "default_value"]
    for c in frame.columns:
        assert frame.schema[c] == pl.Utf8


def test_parameter_values_commodity_price_method_returns_frame() -> None:
    """A simple scalar ``str`` parameter spec materialises into a
    two-column ``(commodity, p_commodity_price_method)`` frame."""
    with SpineDBBackend(str(RIVENDELL_DB)) as backend:
        frame = backend.parameter_values(
            cl_pars=[("commodity", "price_method")],
            header="commodity,p_commodity_price_method",
        )
    assert frame.columns == ["commodity", "p_commodity_price_method"]
    assert frame.schema["commodity"] == pl.Utf8
