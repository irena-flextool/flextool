"""Γ.4 — ``FlexData.dump_csvs`` round-trip parity test.

Loads a fixture via the CSV reader, dumps the FlexData back to a
tempdir, reloads from the tempdir, and asserts frame-level equality
on every populated Param / DataFrame field that ``dump_csvs`` knows
how to write.

This is the debug oracle's regression guard: if the dump-write path
ever diverges from the read path, this test fails.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from polar_high import Param
from flextool.engine_polars import load_flextool
from flextool.engine_polars._dump_csvs import DIRECT_WRITES


DATA = Path(__file__).resolve().parent / "data"

# Fixtures span the core CSV-write surface.  ``coal`` is the minimal
# smoke; ``lh2_three_region`` exercises the wider Param surface
# (multi-region pipes, hydrogen production paths) and is sourced from
# its own JSON fixture via ``db_fixture="lh2"``.
SCENARIO_VIA_FIXTURE: list[tuple[str, str]] = [
    ("coal", "main"),
    ("lh2_three_region", "lh2"),
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
def test_dump_csvs_roundtrip(tmp_path, scenario, db_fixture, scenario_workdir):
    """Load → dump → reload — every dump_csvs-mapped field round-trips.

    Per-field tolerance: frame equality after sort by all columns.
    """
    fixture = scenario_workdir(scenario, db_fixture=db_fixture)

    original = load_flextool(fixture)

    # Dump to tempdir, copying the per-solve metadata from the original.
    # ``include_heavy=True`` forces the seven gigabyte-scale CSVs that
    # the cascade skips by default — the round-trip contract requires
    # every populated FlexData field, including the heavy ones, to
    # appear on disk.
    out_dir = tmp_path / "dumped"
    original.dump_csvs(out_dir, copy_meta_from=fixture, include_heavy=True)

    # Reload from the dumped workdir.
    redo = load_flextool(out_dir)

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
        f"for {scenario or work_name}:\n" + "\n\n".join(mismatches))


def test_dump_csvs_creates_expected_layout(tmp_path, scenario_workdir):
    """``dump_csvs`` creates ``input/`` + ``solve_data/`` subdirs."""
    fixture = scenario_workdir("coal")
    original = load_flextool(fixture)
    out_dir = tmp_path / "dumped_layout"
    original.dump_csvs(out_dir, copy_meta_from=fixture)
    assert (out_dir / "input").is_dir()
    assert (out_dir / "solve_data").is_dir()
    # At least the foundational time-keeping CSV is always written.
    assert (out_dir / "solve_data" / "steps_in_use.csv").exists()


def test_dump_csvs_writes_input_entity_set_csvs(tmp_path, scenario_workdir):
    """Δ.30 — ``dump_csvs`` produces the four entity-class set CSVs and
    the wide-format unitsize CSV that handoff_writers consume.

    Every fast-path workdir must carry these for
    ``handoff_writers._load_entity_class_set`` and
    ``handoff_writers._load_unitsize`` to succeed.
    """
    fixture = scenario_workdir("base")
    original = load_flextool(fixture)
    out_dir = tmp_path / "dumped_input_sets"
    original.dump_csvs(out_dir)

    # Single-column entity-class set CSVs.
    expected_files = ("entity.csv", "node.csv", "process_unit.csv",
                       "process_connection.csv")
    for name in expected_files:
        path = out_dir / "input" / name
        assert path.exists(), f"missing input/{name}"

    # node.csv contains the nodeBalance set.
    import pandas as pd
    node_df = pd.read_csv(out_dir / "input" / "node.csv")
    assert "node" in node_df.columns
    assert "west" in set(node_df["node"].astype(str)), (
        f"node.csv missing 'west': {node_df}"
    )

    # entity.csv = nodes ∪ processes.
    entity_df = pd.read_csv(out_dir / "input" / "entity.csv")
    assert "entity" in entity_df.columns
    assert "west" in set(entity_df["entity"].astype(str))

    # p_entity_unitsize.csv: wide-transposed (entity row + value row).
    unitsize_path = out_dir / "input" / "p_entity_unitsize.csv"
    assert unitsize_path.exists()
    df = pd.read_csv(unitsize_path, index_col=0)
    assert "value" in df.index, f"value row missing: {df}"
    # work_base has no explicit p_unitsize / p_state_unitsize → default 1000.
    assert float(df.loc["value", "west"]) == 1000.0


def test_dump_csvs_writes_solve_data_pre_existing_stub(
    tmp_path, scenario_workdir,
):
    """Δ.30 — ``dump_csvs`` writes a header-only stub for
    ``solve_data/solve__p_entity_pre_existing.csv`` so
    ``handoff_writers.write_p_entity_period_existing_capacity`` reads
    an empty frame instead of crashing on the fast path.
    """
    fixture = scenario_workdir("base")
    original = load_flextool(fixture)
    out_dir = tmp_path / "dumped_pre_existing"
    original.dump_csvs(out_dir)

    stub = out_dir / "solve_data" / "solve__p_entity_pre_existing.csv"
    assert stub.exists()
    import pandas as pd
    # Empty (0, 0) when read with index_col=[0, 1].
    df = pd.read_csv(stub, index_col=[0, 1])
    assert df.empty


# ---------------------------------------------------------------------------
# DB-direct variant: load via DB → dump → reload via CSV → compare.
# This is the spec's primary use case (``audit/db_direct_param_map.md §Γ.4``):
# the developer suspects a DB-direct FlexData diverges from flextool's
# preprocessing.  ``dump_csvs`` writes the DB-direct view to disk so the
# divergence shows up at file granularity under ``diff -r``.

DB_ROUNDTRIP_SCENARIOS = ["coal"]


@pytest.mark.parametrize("scenario", DB_ROUNDTRIP_SCENARIOS)
def test_dump_csvs_db_direct_roundtrip(tmp_path, scenario, scenario_workdir):
    """DB → dump → CSV reload — round-trip the DB-direct path.

    Loads a fixture via SpineDbReader, dumps to a tempdir, reloads
    from the tempdir via the CSV path, and asserts every populated
    DIRECT_WRITES field round-trips frame-for-frame.

    Failure modes this catches:
    * dump_csvs renames a column wrong → CSV reader produces different
      shape → frame inequality.
    * DB-direct produces extra rows the CSV reader can't represent
      → reload drops them → row count mismatch.
    """
    from flextool.engine_polars import SpineDbReader

    fixture = scenario_workdir(scenario)
    sqlite_path = fixture / "tests.sqlite"

    reader = SpineDbReader(sqlite_path, scenario)
    original = load_flextool(fixture, db_reader=reader)

    out_dir = tmp_path / "dumped_db"
    original.dump_csvs(out_dir, copy_meta_from=fixture)

    redo = load_flextool(out_dir)

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
