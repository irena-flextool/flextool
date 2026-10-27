"""Γ.4 — ``FlexData.dump_csvs`` round-trip parity test.

Loads a fixture's last sub-solve as production loads it
(``load_scenario``), dumps the FlexData to a tempdir, adds the run's
metadata CSVs (``_csv_reload.copy_runner_meta``), reloads the tempdir
with ``_csv_reload.load_from_dump``, and asserts frame-level equality on
every populated Param / DataFrame field that ``dump_csvs`` knows how to
write.

This is the debug oracle's regression guard: if the dump-write path
ever diverges from the read path, this test fails.

Moved from ``tests/engine/test_dump_csvs_roundtrip.py`` (c24 S1-8).  The
writer-only checks (layout, entity-set CSVs, pre-existing stub) are in
``tests/engine/test_dump_csvs_writes.py``.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polar_high import Param
from flextool.engine._csv_reload import copy_runner_meta, load_from_dump
from flextool.engine.output._dump_csvs import DIRECT_WRITES
from tests._support.load import load_scenario


DATA = Path(__file__).resolve().parent / "data"

# Fixtures span the core CSV-write surface.  ``coal`` is the minimal
# smoke; ``lh2_three_region`` exercises the wider Param surface
# (multi-region pipes, hydrogen production paths) and is sourced from
# its own JSON fixture via ``db_fixture="lh2"``.
SCENARIO_VIA_FIXTURE: list[tuple[str, str]] = [
    ("coal", "main"),
    ("lh2_three_region", "lh2"),
    ("capacity_margin", "main"),
]


def _frame_of(value):
    if value is None:
        return None
    if isinstance(value, Param):
        return value.frame
    if isinstance(value, pl.DataFrame):
        return value
    return None


def _frames_equal_after_sort(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    """Return True iff a and b are equal after sorting by all columns."""
    if a.shape != b.shape:
        return False
    common = [c for c in a.columns if c in b.columns]
    if set(a.columns) != set(b.columns):
        return False
    aa = a.sort(common)
    bb = b.sort(common)
    if "value" in aa.columns:
        aa = aa.with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
        bb = bb.with_columns(value=pl.col("value").cast(pl.Float64, strict=False))
    return aa.equals(bb)


@pytest.mark.parametrize("scenario,db_fixture", SCENARIO_VIA_FIXTURE,
                          ids=lambda x: x)
def test_dump_csvs_roundtrip(tmp_path, scenario, db_fixture, scenario_run):
    """Load → dump → reload — every dump_csvs-mapped field round-trips.

    Per-field tolerance: frame equality after sort by all columns.
    """
    run = scenario_run(scenario, db_fixture=db_fixture)

    original = load_scenario(run)

    # Dump to tempdir, then copy the per-solve metadata from the run's work folder.
    # ``include_heavy=True`` forces the seven gigabyte-scale CSVs that
    # the cascade skips by default — the round-trip contract requires
    # every populated FlexData field, including the heavy ones, to
    # appear on disk.
    out_dir = tmp_path / "dumped"
    original.dump_csvs(out_dir, include_heavy=True)
    copy_runner_meta(run.work_folder, out_dir)

    # Reload from the dumped workdir.
    redo = load_from_dump(out_dir)

    # Every FlexData field listed in DIRECT_WRITES should round-trip.
    mismatches: list[str] = []
    for field in DIRECT_WRITES:
        a = _frame_of(getattr(original, field, None))
        b = _frame_of(getattr(redo, field, None))
        if a is None and b is None:
            continue  # neither side populated
        if a is None or b is None:
            mismatches.append(
                f"{field}: original-populated={a is not None}, "
                f"reloaded-populated={b is not None}")
            continue
        if not _frames_equal_after_sort(a, b):
            mismatches.append(
                f"{field}: shape orig={a.shape} reload={b.shape}\n"
                f"orig cols={a.columns}\nreload cols={b.columns}\n"
                f"orig:\n{a.head(5)}\nreload:\n{b.head(5)}")
    assert not mismatches, (
        f"dump_csvs round-trip failed on {len(mismatches)} field(s) "
        f"for {scenario or db_fixture}:\n" + "\n\n".join(mismatches))


def test_dump_csvs_keeps_capacity_margin(tmp_path, scenario_run):
    """A CSV-only reload keeps the capacity-margin groups (flag file) and the
    investment periods, so the reloaded model builds the same capacityMargin rows (#69)."""
    from polar_high import Problem

    from flextool.engine import build_flextool

    run = scenario_run("capacity_margin")
    original = load_scenario(run)
    assert original.groupCapacityMargin is not None and original.groupCapacityMargin.height > 0
    assert original.period_invest is not None and original.period_invest.height > 0
    out_dir = tmp_path / "dumped_cm"
    original.dump_csvs(out_dir)
    copy_runner_meta(run.work_folder, out_dir)
    redo = load_from_dump(out_dir)
    for field in ("groupCapacityMargin", "period_invest"):
        a, b = getattr(original, field), getattr(redo, field)
        assert b is not None, field
        assert _frames_equal_after_sort(a.with_columns(pl.all().cast(pl.Utf8)),
                                        b.with_columns(pl.all().cast(pl.Utf8))), field

    def _rows(fd):
        pb = Problem()
        build_flextool(pb, fd)
        recs = [r for r in pb.cstrs_named("capacityMargin") if r.name == "capacityMargin"]
        return recs[0].over.height if recs else 0
    assert _rows(redo) == _rows(original) > 0


# ---------------------------------------------------------------------------
# DB-direct variant: load via DB → dump → reload via CSV → compare.
# This is the spec's primary use case (``audit/db_direct_param_map.md §Γ.4``):
# the developer suspects a DB-direct FlexData diverges from flextool's
# preprocessing.  ``dump_csvs`` writes the DB-direct view to disk so the
# divergence shows up at file granularity under ``diff -r``.

DB_ROUNDTRIP_SCENARIOS = ["coal"]


@pytest.mark.parametrize("scenario", DB_ROUNDTRIP_SCENARIOS)
def test_dump_csvs_db_direct_roundtrip(tmp_path, scenario, scenario_run):
    """DB → dump → CSV reload — round-trip the DB-direct path.

    Loads a fixture as production does (provider + SpineDbReader,
    ``load_scenario``), dumps to a tempdir without the heavy CSVs,
    reloads from the tempdir via the CSV path (no DB reader), and asserts
    every populated DIRECT_WRITES field round-trips frame-for-frame.

    Failure modes this catches:
    * dump_csvs renames a column wrong → CSV reader produces different
      shape → frame inequality.
    * DB-direct produces extra rows the CSV reader can't represent
      → reload drops them → row count mismatch.
    """
    run = scenario_run(scenario)
    original = load_scenario(run)

    out_dir = tmp_path / "dumped_db"
    original.dump_csvs(out_dir)
    copy_runner_meta(run.work_folder, out_dir)

    redo = load_from_dump(out_dir)

    mismatches: list[str] = []
    for field in DIRECT_WRITES:
        a = _frame_of(getattr(original, field, None))
        b = _frame_of(getattr(redo, field, None))
        if a is None and b is None:
            continue
        if a is None or b is None:
            mismatches.append(
                f"{field}: original-populated={a is not None}, "
                f"reloaded-populated={b is not None}")
            continue
        if not _frames_equal_after_sort(a, b):
            mismatches.append(
                f"{field}: shape orig={a.shape} reload={b.shape}\n"
                f"orig:\n{a.head(3)}\nreload:\n{b.head(3)}")
    assert not mismatches, (
        f"DB-direct dump_csvs round-trip failed on {len(mismatches)} "
        f"field(s) for {scenario}:\n" + "\n\n".join(mismatches))
