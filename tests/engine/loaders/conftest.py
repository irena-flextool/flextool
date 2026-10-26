"""Shared fixtures for the engine loader audit (Surface A).

The loader tests drive one ``_load_*`` helper of ``load_flextool`` by
changing the input it reads, then assert the resulting ``FlexData``
field.  They load through the production call (``load_flextool`` with
the sub-solve's provider and a DB reader, see
:func:`tests._support.load.load_scenario`), so the overlay goes into a
clone of a finished run's provider, never into CSV files:

* ``tiny_run`` → ``(run, P)``: the session-cached ``base`` run (the
  smallest end-to-end scenario: 1 node, no processes, 1 period,
  dispatch-only) and a fresh :func:`clone_provider` of its last step's
  provider.  Change ``P``, then ``load_scenario(run, provider=P)``.
* :func:`put_frame` replaces one provider key with an overlay, typed as
  the loader saw that key before (critique item 11).
* :func:`drop_frame` removes a key (the provider form of "the file is
  missing").

A key the DB path re-derives on every load ignores its overlay; a test
on such a key checks a reload-only seed and belongs in
``tests/csv_reload/``.  Every overlay test therefore also asserts that
its overlay is consumed (the loaded value differs from the un-overlaid
load).

Reload-only fixtures (c24 S1-7: kept for the reload-only tests, which
S1-8 moves to ``tests/csv_reload/`` together with these two):

* ``tiny_workdir(tmp_path)`` copies the session-cached ``base`` work
  folder (provider snapshot CSVs + ``tests.sqlite``) into ``tmp_path``;
  ``load_flextool(tiny_workdir)`` is a reader-less CSV load of it.
* ``write_csv(workdir, name, rows)`` writes a CSV overlay under the
  matching sub-directory (bare names → ``solve_data/``).
"""
from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path

import polars as pl
import pytest

from flextool.engine.input._input_source import _read_csv_file
from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
from tests._support.load import ScenarioRun, clone_provider


def provider_key(name: str) -> str:
    """The provider key of a workdir CSV name.

    ``"foo.csv"`` / ``"foo"`` → ``"solve_data/foo"`` (where most loader
    overlays land); ``"input/foo.csv"`` / ``"solve_data/foo"`` keep their
    folder.  The ``.csv`` suffix is dropped.
    """
    key = name.removesuffix(".csv")
    return key if "/" in key else f"solve_data/{key}"


def _frame_as_csv_read(rows: "list[dict] | pl.DataFrame | str") -> pl.DataFrame:
    """``rows`` as the loader's CSV reader returns it (a key the provider lacks).

    ``rows`` is a list of dicts, a frame, or CSV text.  It is written to a
    CSV file and read back with ``_input_source._read_csv_file``, the
    loader's own CSV read, so the dtypes are those a CSV overlay had
    (``"2.5"`` → Float64, header-only → an empty String frame).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "overlay.csv"
        if isinstance(rows, str):
            path.write_text(rows)
        else:
            df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
            df.write_csv(path)
        return _read_csv_file(path)


def _frame_as_given(rows: "list[dict] | pl.DataFrame | str") -> pl.DataFrame:
    """``rows`` as a frame, without dtype inference from text (CSV text → all String)."""
    if isinstance(rows, pl.DataFrame):
        return rows
    if isinstance(rows, str):
        return pl.read_csv(io.StringIO(rows), infer_schema=False)
    return pl.DataFrame(rows, infer_schema_length=None)


def put_frame(P: FlexDataProvider, name: str,
              rows: "list[dict] | pl.DataFrame | str") -> pl.DataFrame:
    """Replace provider key ``name`` (see :func:`provider_key`) with ``rows``.

    The overlay gets the dtypes the loader saw for that key before
    (critique item 11):

    * the key exists: every column it already had is cast to the key's
      existing dtype (the cascade's provider frames are String, so a value
      written as ``"1.5e-3"`` reaches the loader as that text); a column
      the key did not have (a legacy layout, a column alias) keeps the
      dtype of ``rows``;
    * the key is new: ``rows`` is read as a CSV overlay was, through the
      loader's CSV reader.

    Returns the frame that was put.
    """
    key = provider_key(name)
    old = P.get(key)
    if old is None:
        df = _frame_as_csv_read(rows)
    else:
        df = _frame_as_given(rows)
        df = df.with_columns(
            pl.col(c).cast(old.schema[c]) for c in df.columns if c in old.schema
        )
    P.put(key, df)
    return df


def drop_frame(P: FlexDataProvider, name: str) -> None:
    """Remove provider key ``name`` (the provider form of a missing CSV file)."""
    key = provider_key(name)
    assert P.get(key) is not None, f"{key} is not in the provider"
    del P._frames[key]
    P._sources.pop(key, None)


@pytest.fixture(scope="function")
def tiny_run(scenario_run) -> tuple[ScenarioRun, FlexDataProvider]:
    """The session-cached ``base`` run and a fresh clone of its provider.

    ``base`` is the smallest end-to-end scenario in
    ``tests/fixtures/tests.json`` (1 node, no processes, 1 period,
    dispatch-only).  Each test gets its own clone, so its overlays never
    reach the cached run.
    """
    run = scenario_run("base")
    return run, clone_provider(run.last.provider)


# --- reload-only fixtures (S1-8 moves them to tests/csv_reload/) -----------

@pytest.fixture(scope="function")
def tiny_workdir(tmp_path: Path, scenario_workdir) -> Path:
    """Copy the session-cached ``base`` work folder into ``tmp_path / 'work'``.

    Reload-only: ``load_flextool(tiny_workdir)`` reads the snapshot CSVs
    without a DB reader (the folder is not named ``work_<scenario>``).
    Production-path loader tests use :func:`tiny_run`.
    """
    source = scenario_workdir("base")
    target = tmp_path / "work"
    shutil.copytree(source, target)
    return target


def write_csv(workdir: Path, name: str,
              rows: "list[dict] | pl.DataFrame") -> Path:
    """Write ``rows`` as a CSV under ``workdir / <subdir> / <basename>`` (reload-only overlays).

    ``name`` may be ``"foo.csv"`` (→ ``solve_data/foo.csv``) or
    ``"input/foo.csv"`` / ``"solve_data/foo.csv"``.  ``rows`` is a list of
    dicts or a ``pl.DataFrame``.  Returns the path written; overwrites.
    """
    target = workdir / name if "/" in name else workdir / "solve_data" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(rows)
    df.write_csv(target)
    return target
