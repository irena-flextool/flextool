"""Smoke + parity tests for the scenario_workdir fixture (Stage 4 Phase 0).

Verifies:
  - The fixture can build a workdir for a scenario in tests.json
  - The workdir has the expected input/ + solve_data/ subdirs
  - The factory caches the workdir per scenario

The CSV-reload checks (the snapshot loads with a provider-less
``load_flextool``; the snapshot matches the gitignored seed) are in
``tests/csv_reload/test_scenario_workdir.py`` (c24 S1-8).
"""
from __future__ import annotations

def test_scenario_workdir_smoke_base(scenario_workdir):
    work = scenario_workdir("base")
    assert work.exists()
    assert work.is_dir()

    input_dir = work / "input"
    solve_data_dir = work / "solve_data"
    assert input_dir.is_dir(), f"missing {input_dir}"
    assert solve_data_dir.is_dir(), f"missing {solve_data_dir}"

    # The cascade runs with ``csv_dump=False``; the CSVs are the
    # ``snapshot_processed_inputs`` snapshot of the last sub-solve's
    # provider.  Seed dir for ``base`` has 140 input + 492 solve_data
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


def test_scenario_workdir_cached(scenario_workdir):
    work_a = scenario_workdir("base")
    work_b = scenario_workdir("base")
    assert work_a is work_b, "factory must return the same Path on cache hit"


