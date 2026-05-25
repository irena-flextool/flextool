"""Self-tests for the shared fixtures in ``loaders/conftest.py``.

Verify that ``tiny_workdir`` is a workdir ``load_flextool`` accepts
without error, and that ``write_csv`` round-trips through ``pl.read_csv``.
"""
from __future__ import annotations

import polars as pl
import pytest

from flextool.engine_polars import load_flextool

from .conftest import write_csv


def test_tiny_workdir_loads(tiny_workdir):
    """The seed workdir must load through ``load_flextool`` cleanly.

    This is the contract: any loader test that starts from
    ``tiny_workdir`` and overlays a CSV via :func:`write_csv` is then
    safe to call ``load_flextool(tiny_workdir)`` without secondary
    setup.
    """
    data = load_flextool(tiny_workdir)
    # Required-always fields populated.
    assert data.dt is not None
    assert data.dt.height > 0
    assert data.nodeBalance is not None
    assert data.nodeBalance.height > 0


def test_write_csv_roundtrip(tiny_workdir):
    """write_csv writes to solve_data/ by default; explicit prefixes honoured."""
    rows = [{"a": "x", "b": 1.5}, {"a": "y", "b": 2.5}]

    p1 = write_csv(tiny_workdir, "_probe.csv", rows)
    assert p1 == tiny_workdir / "solve_data" / "_probe.csv"
    assert p1.exists()
    df1 = pl.read_csv(p1)
    assert df1.shape == (2, 2)
    assert df1["a"].to_list() == ["x", "y"]

    p2 = write_csv(tiny_workdir, "input/_probe2.csv",
                    pl.DataFrame({"k": ["a"], "v": [1.0]}))
    assert p2 == tiny_workdir / "input" / "_probe2.csv"
    assert p2.exists()


def test_tiny_workdir_is_isolated_per_call(tiny_workdir, tmp_path):
    """Each test gets a fresh copy under its own tmp_path."""
    assert tiny_workdir.parent == tmp_path
    assert tiny_workdir.name == "work"
    # Adding a file in this test must not leak into other tests
    # (validated implicitly by other tests reading the seed).
    write_csv(tiny_workdir, "_isolation_marker.csv", [{"x": 1}])
    assert (tiny_workdir / "solve_data" / "_isolation_marker.csv").exists()
