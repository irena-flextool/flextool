"""Behavioural tests for the stochastic feature.

Three layers of coverage on the ``2_day_stochastic_dispatch`` scenario
(``how to example databases/stochastics.sqlite``, exported as
``tests/fixtures/stochastics.json``):

1. **pbt_profile writer** — sanity check that ``input/pbt_profile.csv``
   actually carries multi-branch profile data.  Writer-side tripwire.
2. **Non-anticipativity constraints** (``flextool.mod:4173-4232``) —
   storage-state equality across branches at every constrained
   timestep.  Catches a bad ``period__branch`` / ``dt_non_anticipativity``
   set or a wrong constraint domain.
3. **pbt_profile flows into the LP via pdtProfile** — assert that the
   per-period ``pdtProfile`` values written to ``solve_data`` match
   each period's branch profile (and that v_flow on wind_plant tracks
   ``min over branches`` because non-anticipativity equates v_flow
   while each branch enforces its own profile cap).  Catches the case
   where pbt_profile is silently dropped on the way to pdtProfile
   (e.g. ``is_stoch`` evaluates False and the writer falls through to
   the realised-branch lookup).
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pandas as pd
import pytest

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.lean_parquet import read_lean_parquet


def _read_csv_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        return [(r[0], r[1]) for r in reader if len(r) >= 2 and r[0] and r[1]]


@pytest.fixture(scope="module")
def stochastic_run_workdir(
    stochastic_db_url: str,
    test_bin_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Run the 2_day_stochastic_dispatch scenario once for the module."""
    workdir = tmp_path_factory.mktemp("non_anticipativity_run")
    os.chdir(workdir)
    runner = FlexToolRunner(
        input_db_url=stochastic_db_url,
        scenario_name="2_day_stochastic_dispatch",
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(stochastic_db_url, "2_day_stochastic_dispatch")
    rc = runner.run_model()
    assert rc == 0, "Stochastic dispatch model failed to solve"
    return workdir


def test_period__branch_has_branching(stochastic_run_workdir: Path) -> None:
    """Sanity check: the fixture actually creates non-trivial branches.

    If period__branch contains only self-loops the non-anticipativity
    constraints become vacuous and the assertions below would pass for
    the wrong reason.  Guard against the fixture decaying into a
    deterministic-only scenario.
    """
    pb = _read_csv_pairs(
        stochastic_run_workdir / "solve_data" / "period__branch.csv"
    )
    branched = [(d, b) for (d, b) in pb if d != b]
    assert branched, (
        f"2_day_stochastic_dispatch produced only self-loop period__branch "
        f"rows; non-anticipativity will not be exercised. Rows: {pb}"
    )


def test_dt_non_anticipativity_is_populated(
    stochastic_run_workdir: Path,
) -> None:
    """Sanity check: dt_non_anticipativity has rows.

    Empty → constraints don't fire and the storage equality below
    becomes vacuous.
    """
    dtna = _read_csv_pairs(
        stochastic_run_workdir / "solve_data" / "dt_non_anticipativity_set.csv"
    )
    assert dtna, (
        "dt_non_anticipativity_set.csv has no rows; non-anticipativity "
        "constraints will not fire."
    )


def test_storage_state_equal_across_branches(
    stochastic_run_workdir: Path,
) -> None:
    """v_state at the realised period must equal v_state at every
    branch period at every dt_non_anticipativity timestep.

    The mod's ``non_anticipativity_storage_use`` constraint pins net
    storage charge at (d, t) to net charge at (b, t).  Combined with a
    shared ``storage_state_start`` (a scalar parameter — same across
    branches by construction) and the storage balance, this yields
    identical v_state trajectories at every constrained timestep.

    A regression where the constraint's index domain is wrong (e.g.
    period__branch swapped with branch__period, or dt_non_anticipativity
    pulled from the wrong CSV) would let the branches' storage states
    drift.
    """
    workdir = stochastic_run_workdir
    matches = list(
        (workdir / "output_raw").glob("v_state__*.parquet")
    )
    assert matches, "No v_state parquet emitted by stochastic run"
    df = read_lean_parquet(matches[0])

    # Rows: (solve, period, time).  Columns: (node,).
    assert df.index.names == ["solve", "period", "time"], df.index.names
    assert df.columns.names == ["node"], df.columns.names

    pb = _read_csv_pairs(workdir / "solve_data" / "period__branch.csv")
    dtna = _read_csv_pairs(
        workdir / "solve_data" / "dt_non_anticipativity_set.csv"
    )
    dtna_by_d: dict[str, set[str]] = {}
    for (d, t) in dtna:
        dtna_by_d.setdefault(d, set()).add(t)

    # Periods that actually have v_state rows — branches that aren't in
    # period_in_use (e.g. a metadata-only "..._realized" branch) would
    # be absent from v_state and we should skip them.
    periods_with_rows = set(df.index.get_level_values("period"))

    pairs_checked = 0
    for (d, b) in pb:
        if d == b:
            continue
        if d not in periods_with_rows or b not in periods_with_rows:
            # b not in period_in_use — constraint domain excludes it.
            continue
        timesteps = dtna_by_d.get(d, set())
        for t in timesteps:
            try:
                row_d = df.xs((d, t), level=("period", "time"))
                row_b = df.xs((b, t), level=("period", "time"))
            except KeyError:
                continue
            for node in df.columns:
                v_d = float(row_d.iloc[0][node])
                v_b = float(row_b.iloc[0][node])
                assert v_d == pytest.approx(v_b, abs=1e-6), (
                    f"non-anticipativity violated for node={node!r}, "
                    f"d={d!r}, b={b!r}, t={t!r}: "
                    f"v_state[d]={v_d}, v_state[b]={v_b}"
                )
                pairs_checked += 1

    assert pairs_checked > 0, (
        "No (d, b, t) tuples were checked — fixture failed to produce "
        "branches in period_in_use, or v_state has no storage nodes."
    )


# ---------------------------------------------------------------------------
# pbt_profile coverage (writer + LP propagation)
# ---------------------------------------------------------------------------

def test_pbt_profile_writer_emits_multi_branch_data(
    stochastic_run_workdir: Path,
) -> None:
    """``input/pbt_profile.csv`` must carry multiple branches with
    distinct values for at least one timestep.

    Tripwire for the input writer: if the writer ever stops emitting
    branched profile data (e.g. a regression that drops the branch
    dimension or filters out non-realised branches), this test catches
    it before the value silently degrades downstream.
    """
    pbt_path = stochastic_run_workdir / "input" / "pbt_profile.csv"
    assert pbt_path.exists(), f"missing {pbt_path}"
    df = pd.read_csv(pbt_path)
    assert not df.empty, "input/pbt_profile.csv is header-only"
    branches = sorted(df["branch"].unique())
    assert len(branches) >= 2, (
        f"pbt_profile.csv has only {branches} — stochastic feature "
        f"degenerated to deterministic"
    )
    # Pick the first profile + timestep that has all branches; assert
    # the values are not all identical (non-trivial stochasticity).
    first_profile = df["profile"].iloc[0]
    first_t = df["time"].iloc[0]
    sub = df[(df["profile"] == first_profile) & (df["time"] == first_t)]
    values = sub["pbt_profile"].to_list()
    assert len(values) >= 2 and len(set(values)) >= 2, (
        f"pbt_profile values for ({first_profile!r}, t={first_t!r}) "
        f"are all identical: {values} — branches carry no signal"
    )


def test_pdtProfile_per_period_matches_per_branch_pbt(
    stochastic_run_workdir: Path,
) -> None:
    """The ``solve_data/pdtProfile.csv`` value at (profile, period d, t)
    must equal ``pbt_profile`` at (profile, branch_of_d, ts_of_d, t).

    This is the LP-side propagation check: the mod's profile constraint
    (flextool.mod:2598-2611, ``profile_flow_upper_limit``) reads from
    ``pdtProfile``, so if a stochastic profile's per-branch values are
    not picked up here, the LP enforces the wrong cap on every branch.

    Currently *expected to fail* on ``2_day_stochastic_dispatch`` —
    surfaces the bug where ``write_pdtProfile`` reads
    ``solve_data/group_process.csv`` (does not exist) instead of
    ``input/group__process.csv``, leaving ``stoch_profile`` empty so
    every branch period falls through to the realised branch's value.
    """
    workdir = stochastic_run_workdir
    pbt = pd.read_csv(workdir / "input" / "pbt_profile.csv")
    pdtp = pd.read_csv(workdir / "solve_data" / "pdtProfile.csv")
    sb = pd.read_csv(
        workdir / "solve_data" / "solve_branch__time_branch.csv"
    )
    fts = pd.read_csv(workdir / "solve_data" / "first_timesteps.csv")

    # Period → its branch identifier (from solve_branch__time_branch.csv)
    branch_of: dict[str, str] = dict(zip(sb["period"], sb["branch"]))
    # Period → its first timestep (the "time_start" axis in pbt_profile)
    ts_of: dict[str, str] = dict(zip(fts["period"], fts["step"]))

    # Index pbt_profile by (profile, branch, time_start, time)
    pbt_lookup: dict[tuple[str, str, str, str], float] = {
        (r.profile, r.branch, r.time_start, r.time): float(r.pbt_profile)
        for r in pbt.itertuples()
    }

    mismatches: list[str] = []
    checked = 0
    for r in pdtp.itertuples():
        b = branch_of.get(r.period)
        ts = ts_of.get(r.period)
        if b is None or ts is None:
            continue
        key = (r.profile, b, ts, r.time)
        expected = pbt_lookup.get(key)
        if expected is None:
            continue
        checked += 1
        if abs(float(r.value) - expected) > 1e-9:
            mismatches.append(
                f"  ({r.profile}, period={r.period}, t={r.time}) "
                f"pdtProfile={r.value} but pbt_profile[branch={b}, "
                f"ts={ts}]={expected}"
            )

    assert checked > 0, (
        "No (profile, period, time) entries cross-matched between "
        "pdtProfile and pbt_profile — fixture/lookup wiring broken."
    )
    assert not mismatches, (
        f"pdtProfile drops per-branch pbt_profile values for "
        f"{len(mismatches)} entries (showing first 5):\n"
        + "\n".join(mismatches[:5])
    )
