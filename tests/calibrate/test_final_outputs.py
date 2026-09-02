"""Unit tests for the post-loop parquet replay (:mod:`_final_outputs`).

These exercise the wiring only — the engine's ``write_outputs`` is stubbed so
no real parquet is read or format written; the point is that
:func:`write_final_outputs` computes the right parquet directory, forwards the
right replay kwargs, and short-circuits / raises in the right cases.
"""

from __future__ import annotations

import importlib

import pytest

from flextool.calibrate._final_outputs import (
    FINAL_WRITE_METHOD_CHOICES,
    write_final_outputs,
)

# The engine writer module. The ``flextool.process_outputs`` package re-exports
# the ``write_outputs`` FUNCTION under the same dotted name as the submodule, so
# reach the submodule object explicitly (import_module returns it from
# sys.modules regardless of that shadowing) and patch its function attribute —
# which is exactly the name ``write_final_outputs`` binds via its lazy import.
_WO_MODULE = importlib.import_module("flextool.process_outputs.write_outputs")


def _stub_write_outputs(monkeypatch, sink):
    """Patch the engine writer that ``write_final_outputs`` imports lazily."""

    def _capture(**kwargs):
        sink.update(kwargs)

    monkeypatch.setattr(_WO_MODULE, "write_outputs", _capture)


def test_empty_methods_is_noop(tmp_path, monkeypatch):
    # Empty methods must not even touch the engine writer (nor require parquet).
    def _boom(**kwargs):
        raise AssertionError("write_outputs must not be called for empty methods")

    monkeypatch.setattr(_WO_MODULE, "write_outputs", _boom)

    assert write_final_outputs(tmp_path, "scenA", []) == []
    assert write_final_outputs(tmp_path, "scenA", ("",)) == []


def test_missing_parquet_dir_raises(tmp_path, monkeypatch):
    # No output_parquet/<scenario>/ ⇒ nothing to replay ⇒ explicit error.
    called = {}
    _stub_write_outputs(monkeypatch, called)
    with pytest.raises(FileNotFoundError, match="scenA"):
        write_final_outputs(tmp_path, "scenA", ["csv"])
    assert called == {}  # never reached the writer


def test_happy_path_forwards_replay_kwargs(tmp_path, monkeypatch):
    (tmp_path / "output_parquet" / "scenA").mkdir(parents=True)
    called: dict = {}
    _stub_write_outputs(monkeypatch, called)

    written = write_final_outputs(
        tmp_path, "scenA", ["csv", "excel"], results_db_url="sqlite:///x.sqlite"
    )

    assert written == ["csv", "excel"]
    assert called["scenario_name"] == "scenA"
    assert called["output_location"] == str(tmp_path)
    assert called["subdir"] == "scenA"
    assert called["read_parquet_dir"] is True
    assert called["write_methods"] == ["csv", "excel"]
    assert called["results_db_url"] == "sqlite:///x.sqlite"
    # output_location stays non-empty even if the engine tries settings fallback.
    assert called["fallback_output_location"] == str(tmp_path)


def test_parquet_never_in_choices():
    # parquet is already on disk; it must not be an accepted final-write method.
    assert "parquet" not in FINAL_WRITE_METHOD_CHOICES
    assert set(FINAL_WRITE_METHOD_CHOICES) == {"csv", "excel", "spinedb", "plot"}
