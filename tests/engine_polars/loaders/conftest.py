"""Shared fixtures for the engine_polars loader audit (Surface A).

The polars loader (``flextool.engine_polars.load_flextool``) walks a
flextool workdir composed of two CSV directories — ``input/`` and
``solve_data/`` — populated by flextool's preprocessing pass.  A
hand-written workdir from scratch is impractical: ``load_flextool``
calls ~25 ``_load_*`` helpers that together read 200+ CSVs, and many
of them are tightly coupled (``period_in_use_set.csv`` must be
consistent with ``steps_in_use.csv``, ``solve_data/process.csv``, etc).

Pragmatic approach
------------------

* ``tiny_workdir(tmp_path)`` copies the smallest known full workdir
  (``tests/engine_polars/data/work_base``) into ``tmp_path`` so each
  test gets a fresh, isolated, throw-away copy.  The copy is the seed;
  loader tests overlay specific CSVs to drive the helper they care
  about.
* ``write_csv(workdir, name, rows)`` writes a CSV under the matching
  sub-directory.  Names with no slash default to ``solve_data/``
  (where most loader-test overlays land); explicit ``input/foo.csv``
  or ``solve_data/foo.csv`` paths are honoured.

This conftest is deliberately thin: every loader test is responsible
for the CSV-shape it expects ``_load_*`` to consume.  The fixture's
contract is "an isolated workdir that ``load_flextool`` accepts".
"""
from __future__ import annotations

import shutil
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="function")
def tiny_workdir(tmp_path: Path, scenario_workdir) -> Path:
    """Copy the session-cached ``base`` workdir into ``tmp_path / 'work'``.

    ``base`` is the smallest end-to-end scenario in
    ``tests/fixtures/tests.json`` (1 node, 1 process, 1 period,
    dispatch-only).  Tests layer on additional CSVs via :func:`write_csv`
    (e.g. ``rp_cost_weight.csv`` overrides) without touching the
    session-shared source — every test gets a fresh isolated copy.
    """
    source = scenario_workdir("base")
    target = tmp_path / "work"
    shutil.copytree(source, target)
    return target


def write_csv(workdir: Path, name: str,
               rows: "list[dict] | pl.DataFrame") -> Path:
    """Write ``rows`` as a CSV under ``workdir / <subdir> / <basename>``.

    ``name`` may be:
      * ``"foo.csv"`` — defaults to ``solve_data/foo.csv`` (where the
        majority of loader overlays land).
      * ``"input/foo.csv"`` or ``"solve_data/foo.csv"`` — explicit.

    ``rows`` is a list of dicts (auto-converted to ``pl.DataFrame``)
    or a ``pl.DataFrame``.

    Returns the path the file was written to.  Overwrites silently —
    the fixture's tmp_path makes this safe.
    """
    if "/" in name:
        target = workdir / name
    else:
        target = workdir / "solve_data" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows, pl.DataFrame):
        df = rows
    else:
        df = pl.DataFrame(rows)
    df.write_csv(target)
    return target
