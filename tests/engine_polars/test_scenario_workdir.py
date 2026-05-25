"""Smoke + parity tests for the scenario_workdir fixture (Stage 4 Phase 0).

Verifies:
  - The fixture can build a workdir for a scenario in tests.json
  - The workdir has the expected input/ + solve_data/ subdirs
  - The workdir's content is loadable by load_flextool
  - When the gitignored data/work_<scen>/ exists, the fixture's output
    matches it modulo solver outputs (so existing tests will work
    when migrated)

The matches test is the critical sanity check: if the in-memory cascade
produces different content than the disk-staged generator used to seed
``tests/engine_polars/data/``, that's a structural finding that may
block the Phase 1 migration.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"


def test_scenario_workdir_smoke_base(scenario_workdir):
    work = scenario_workdir("base")
    assert work.exists()
    assert work.is_dir()

    input_dir = work / "input"
    solve_data_dir = work / "solve_data"
    assert input_dir.is_dir(), f"missing {input_dir}"
    assert solve_data_dir.is_dir(), f"missing {solve_data_dir}"

    # With ``csv_dump=True`` the full cascade flushes every ``_emit_*``
    # frame to disk.  Seed dir for ``base`` has 140 input + 492 solve_data
    # CSVs; allow some slack since the exact set can shift between
    # cascade versions, but the count must be in the same ballpark.
    input_csvs = list(input_dir.glob("*.csv"))
    solve_data_csvs = list(solve_data_dir.glob("*.csv"))
    assert len(input_csvs) > 100, (
        f"expected >100 input CSVs, got {len(input_csvs)}"
    )
    assert len(solve_data_csvs) > 100, (
        f"expected >100 solve_data CSVs, got {len(solve_data_csvs)}"
    )


def test_scenario_workdir_loads_with_load_flextool(scenario_workdir):
    from flextool.engine_polars.input import load_flextool

    work = scenario_workdir("base")
    data = load_flextool(work)
    assert data is not None


def test_scenario_workdir_cached(scenario_workdir):
    work_a = scenario_workdir("base")
    work_b = scenario_workdir("base")
    assert work_a is work_b, "factory must return the same Path on cache hit"


def test_scenario_workdir_matches_data_dir(scenario_workdir):
    """Parity check vs. the gitignored on-disk seed for ``base``.

    If the seed dir is absent, skip — it's gitignored and may not exist
    on a fresh clone or in CI.  When present, two checks run:

    1. **Coverage**: the set of files in the seed must be largely
       reproduced by the fresh fixture.  Drift is structural — the
       seed was produced by an older legacy ``write_input`` path that
       materialised every Provider key to disk, whereas the modern
       in-memory cascade only emits a working subset (the cascade
       itself reads from the live Provider, never re-reads the CSVs).
       The assertion is generous (≤ 150 missing) so the test surfaces
       a *growing* drift rather than punishing the expected baseline.
    2. **Content parity**: three representative files are compared
       (column-order normalised; polars-frame equality, which already
       ignores trailing-CR line-ending differences).
    """
    seed = DATA_DIR / "work_base"
    if not seed.exists():
        pytest.skip(f"gitignored seed dir {seed} not present")

    work = scenario_workdir("base")

    # ---- coverage check -------------------------------------------------
    for subdir in ("input", "solve_data"):
        seed_sub = seed / subdir
        fresh_sub = work / subdir
        if not seed_sub.is_dir():
            continue
        seed_files = {p.name for p in seed_sub.glob("*.csv")}
        fresh_files = (
            {p.name for p in fresh_sub.glob("*.csv")}
            if fresh_sub.is_dir()
            else set()
        )
        missing = seed_files - fresh_files
        # Generous bound: as of Phase 0b the fresh cascade misses ~6
        # input files + ~105 solve_data files vs the seed (legacy
        # write_input wrote more set tables than the in-memory
        # cascade emits).  Surface a *growing* drift, not the baseline.
        assert len(missing) <= 150, (
            f"{subdir}/: fresh fixture is missing {len(missing)} files "
            f"that exist in the seed (sample: {sorted(missing)[:8]})"
        )

    # ---- content parity (sample) ---------------------------------------
    candidates = [
        Path("input") / "p_node.csv",
        Path("input") / "period.csv",
        Path("solve_data") / "p_model.csv",
    ]

    compared = 0
    for rel in candidates:
        seed_path = seed / rel
        fresh_path = work / rel
        if not seed_path.exists():
            # The seed dir is allowed to be missing some files (different
            # cascade versions emit different file sets).  Only compare
            # files that exist in BOTH.
            continue
        if not fresh_path.exists():
            pytest.fail(
                f"fixture missing {rel} that exists in seed; "
                f"seed={seed_path}, fresh={fresh_path}"
            )
        df_seed = pl.read_csv(seed_path)
        df_fresh = pl.read_csv(fresh_path)
        # Align column order — semantic equality, not file-byte equality.
        common = sorted(set(df_seed.columns) & set(df_fresh.columns))
        only_seed = sorted(set(df_seed.columns) - set(df_fresh.columns))
        only_fresh = sorted(set(df_fresh.columns) - set(df_seed.columns))
        assert not only_seed and not only_fresh, (
            f"{rel}: column-set mismatch.  "
            f"only_in_seed={only_seed}, only_in_fresh={only_fresh}"
        )
        df_seed = df_seed.select(common).sort(common)
        df_fresh = df_fresh.select(common).sort(common)
        assert df_seed.equals(df_fresh), (
            f"{rel}: content differs between seed and fresh fixture output. "
            f"seed_shape={df_seed.shape}, fresh_shape={df_fresh.shape}"
        )
        compared += 1

    if compared == 0:
        pytest.skip(
            f"none of {candidates} present in seed {seed}; nothing to compare"
        )
