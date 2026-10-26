"""Self-tests for the shared fixtures in ``loaders/conftest.py``.

Production-path fixtures: ``tiny_run`` gives a run and a provider clone
that ``load_scenario`` accepts; ``put_frame`` types an overlay as the
loader saw the key before (critique item 11); ``drop_frame`` removes a
key; the clone never reaches the session-cached run.

Reload-only fixtures (``tiny_workdir`` / ``write_csv``): their self-tests
are reload-only too and move to ``tests/csv_reload/`` with them (c24 S1-8).
"""
from __future__ import annotations

import polars as pl

from flextool.engine import load_flextool
from tests._support.load import load_scenario

from .conftest import drop_frame, provider_key, put_frame, write_csv


# --- production-path fixtures -----------------------------------------------

def test_tiny_run_loads(tiny_run):
    """The ``base`` provider clone loads through the production call, and equals the cascade's FlexData.

    This is the contract: a loader test that starts from ``tiny_run`` and
    overlays a key via :func:`put_frame` can call
    ``load_scenario(run, provider=P)`` without secondary setup.
    """
    run, P = tiny_run
    data = load_scenario(run, provider=P)
    # Required-always fields populated.
    assert data.dt is not None
    assert data.dt.height > 0
    assert data.nodeBalance is not None
    assert data.nodeBalance.height > 0
    assert data.dt.sort(["d", "t"]).equals(run.last_flex_data.dt.sort(["d", "t"]))


def test_provider_key_mapping():
    """Bare names land in ``solve_data/``; a folder prefix is kept; ``.csv`` is dropped."""
    assert provider_key("foo.csv") == "solve_data/foo"
    assert provider_key("foo") == "solve_data/foo"
    assert provider_key("input/foo.csv") == "input/foo"
    assert provider_key("solve_data/foo") == "solve_data/foo"


def test_put_frame_keeps_the_key_dtypes(tiny_run):
    """An overlay on an existing key is cast to that key's dtypes; extra columns keep their own.

    The cascade's ``solve_data/steps_in_use`` frame is all String, so a
    float ``1.0`` reaches the loader as ``"1.0"`` and a string ``"1.5e-3"``
    stays that text (no CSV re-inference).
    """
    run, P = tiny_run
    before = P.get("solve_data/steps_in_use")
    assert before.schema == {"period": pl.String, "step": pl.String, "step_duration": pl.String}
    df = put_frame(P, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": 1.0, "extra": 2.5},
    ])
    assert P.get("solve_data/steps_in_use") is df
    assert df.schema == {"period": pl.String, "step": pl.String, "step_duration": pl.String,
                         "extra": pl.Float64}
    assert df["step_duration"].to_list() == ["1.0"]
    df = put_frame(P, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0002", "step_duration": "1.5e-3"},
    ])
    assert df["step_duration"].to_list() == ["1.5e-3"]
    # CSV text: no inference, then the key's dtypes.
    df = put_frame(P, "steps_in_use.csv", "period,step,step_duration\np2020,t0003,2.5\n")
    assert df.schema == before.schema
    assert df["step_duration"].to_list() == ["2.5"]
    # Header-only CSV text: an empty frame with the key's dtypes.
    df = put_frame(P, "steps_in_use.csv", "period,step,step_duration\n")
    assert df.height == 0
    assert df.schema == before.schema


def test_put_frame_new_key_reads_like_a_csv_overlay(tiny_run):
    """A key the provider lacks gets the dtypes of the loader's CSV read (``_read_csv_file``)."""
    run, P = tiny_run
    assert P.get("solve_data/_probe") is None
    df = put_frame(P, "_probe.csv", [{"a": "x", "b": "1.5"}, {"a": "y", "b": "2.5"}])
    assert P.get("solve_data/_probe") is df
    assert df.schema == {"a": pl.String, "b": pl.Float64}
    assert df["b"].to_list() == [1.5, 2.5]
    df = put_frame(P, "input/_probe2.csv", "k\n")
    assert P.get("input/_probe2") is df
    assert df.height == 0
    assert df.schema == {"k": pl.String}


def test_drop_frame_removes_the_key(tiny_run):
    """``drop_frame`` removes a key from the clone only."""
    run, P = tiny_run
    drop_frame(P, "pdtNodeInflow.csv")
    assert P.get("solve_data/pdtNodeInflow") is None
    assert "solve_data/pdtNodeInflow" not in P.keys()
    assert run.last.provider.get("solve_data/pdtNodeInflow") is not None


def test_tiny_run_clone_is_isolated(tiny_run):
    """``P`` is a clone: an overlay on it never reaches the session-cached run's provider."""
    run, P = tiny_run
    assert P is not run.last.provider
    original = run.last.provider.get("solve_data/pdtNodeInflow")
    put_frame(P, "pdtNodeInflow.csv", [{"node": "west", "period": "p2020", "time": "t0001", "value": 1.0}])
    put_frame(P, "_isolation_marker.csv", [{"x": 1}])
    assert run.last.provider.get("solve_data/pdtNodeInflow") is original
    assert run.last.provider.get("solve_data/_isolation_marker") is None


# --- reload-only fixtures (S1-8 moves these tests to tests/csv_reload/) -----

def test_tiny_workdir_loads(tiny_workdir):
    """The seed workdir must load through ``load_flextool`` cleanly.

    This is the contract: any loader test that starts from
    ``tiny_workdir`` and overlays a CSV via :func:`write_csv` is then
    safe to call ``load_flextool(tiny_workdir)`` without secondary
    setup.

    Reload-only (c24 S1-7): a reader-less CSV load; S1-8 moves it to tests/csv_reload/.
    """
    data = load_flextool(tiny_workdir)
    # Required-always fields populated.
    assert data.dt is not None
    assert data.dt.height > 0
    assert data.nodeBalance is not None
    assert data.nodeBalance.height > 0


def test_write_csv_roundtrip(tiny_workdir):
    """write_csv writes to solve_data/ by default; explicit prefixes honoured.

    Reload-only (c24 S1-7): self-test of a reload-only fixture; S1-8 moves it to tests/csv_reload/.
    """
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
    """Each test gets a fresh copy under its own tmp_path.

    Reload-only (c24 S1-7): self-test of a reload-only fixture; S1-8 moves it to tests/csv_reload/.
    """
    assert tiny_workdir.parent == tmp_path
    assert tiny_workdir.name == "work"
    # Adding a file in this test must not leak into other tests
    # (validated implicitly by other tests reading the seed).
    write_csv(tiny_workdir, "_isolation_marker.csv", [{"x": 1}])
    assert (tiny_workdir / "solve_data" / "_isolation_marker.csv").exists()
