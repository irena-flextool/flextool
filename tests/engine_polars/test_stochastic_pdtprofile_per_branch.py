"""Regression guard mirroring flextool's ``test_non_anticipativity.py``
(commit ``042fae2``: "stochastic: restore per-branch profile values lost
in pdtProfile").

Background
----------
A flextool preprocessing bug had ``stoch_processes`` / ``stoch_nodes`` /
``stoch_profile`` ending up empty because eight call sites in
``entity_period_calc_params.py`` read group memberships from
``solve_data/group_{process,node}.csv`` (single-underscore, in
``solve_data/``) — files that don't exist.  The writer puts them at
``input/group__{process,node}.csv`` (double-underscore, in ``input/``).

Effect: ``is_stoch`` evaluated False for every profile, so the
stochastic-aware branch of the ``pdtProfile`` lookup was skipped and
every branch period inherited the *realised* period's profile values.
Stochastic 4-branch fixtures silently degenerated to deterministic
dispatch dressed up as a 4-branch model — the LP was 4× weighted but
every branch's wind cap was identical.

polar_high's posture
----------------
polar_high reads ``solve_data/pdtProfile.csv`` directly via its CSV path
(stochastic 3d_map detection in ``_derived_params.py:p_profile_value``
explicitly defers to CSV when the source returns generic ``x`` / ``i``
keys — see progress.md §3.6.1).  So polar_high is immune to the bug at the
helper level — but inherits a buggy ``pdtProfile.csv`` if a fixture
were generated from a buggy flextool.

This test pins the fixture's correctness so a future regenerate from a
stale flextool checkout (or any other path that re-introduces the
bug) can't silently revert the stochastic feature to deterministic.

Mirrors ``flextool/tests/test_non_anticipativity.py``:

* ``test_pbt_profile_writer_emits_multi_branch_data`` — pbt_profile.csv
  must carry distinct per-branch values for at least one (profile, t).
* ``test_pdtProfile_per_period_matches_per_branch_pbt`` — every
  (profile, period d, t) row in ``solve_data/pdtProfile.csv`` must
  equal ``pbt_profile`` at (profile, branch_of_d, ts_of_d, t).  Catches
  the silent degeneration where pdtProfile drops every branch's
  per-branch value to the realised branch's.
"""
from __future__ import annotations

import polars as pl
import pytest


# Phase 3d: rebuilt via ``scenario_workdir(..., db_fixture='stochastic')``.
# ``full_storage`` is the canonical scenario; ``no_storage`` adds the
# ``no_storage_override`` alternative.
STOCHASTIC_SCENARIOS = [
    "2_day_stochastic_dispatch",
    "2_day_stochastic_dispatch_no_storage",
]


@pytest.mark.parametrize("scenario", STOCHASTIC_SCENARIOS)
def test_pbt_profile_carries_distinct_branch_values(
        scenario: str, scenario_workdir) -> None:
    """``input/pbt_profile.csv`` must carry multiple branches with
    distinct values for at least one (profile, time) pair."""
    workdir = scenario_workdir(scenario, db_fixture="stochastic")
    pbt_path = workdir / "input" / "pbt_profile.csv"
    assert pbt_path.exists(), f"{scenario}: missing {pbt_path.name}"
    df = pl.read_csv(pbt_path)
    assert df.height > 0, f"{scenario}: pbt_profile.csv is header-only"
    branches = sorted(df["branch"].unique().to_list())
    assert len(branches) >= 2, (
        f"{scenario}: pbt_profile carries only {branches!r} — stochastic "
        f"feature degenerated to deterministic"
    )

    # Find at least one (profile, time) tuple where values differ
    # across branches.
    spread = (df.group_by(["profile", "time"])
                .agg([pl.col("pbt_profile").min().alias("mn"),
                      pl.col("pbt_profile").max().alias("mx"),
                      pl.col("branch").n_unique().alias("n_branch")]))
    multi = spread.filter(pl.col("n_branch") >= 2)
    assert multi.height > 0, (
        f"{scenario}: every (profile, time) tuple has only one branch "
        f"row — pbt_profile branched dimension collapsed"
    )
    nontrivial = multi.filter(pl.col("mx") - pl.col("mn") > 1e-9)
    assert nontrivial.height > 0, (
        f"{scenario}: every (profile, time) tuple has identical values "
        f"across branches — stochasticity is a no-op (max spread "
        f"{float((multi['mx'] - multi['mn']).max())})"
    )


@pytest.mark.parametrize("scenario", STOCHASTIC_SCENARIOS)
def test_pdtProfile_per_period_matches_per_branch_pbt(
        scenario: str, scenario_workdir) -> None:
    """The ``solve_data/pdtProfile.csv`` value at (profile, period d, t)
    must equal ``pbt_profile`` at (profile, branch_of_d, ts_of_d, t).

    Mirrors flextool's commit 042fae2 regression guard.  Catches the
    pre-fix degeneration where every branch period silently fell
    through to the realised branch's profile value.
    """
    workdir = scenario_workdir(scenario, db_fixture="stochastic")
    pbt = pl.read_csv(workdir / "input" / "pbt_profile.csv")
    pdtp = pl.read_csv(workdir / "solve_data" / "pdtProfile.csv")
    sb = pl.read_csv(workdir / "solve_data" / "solve_branch__time_branch.csv")
    fts = pl.read_csv(workdir / "solve_data" / "first_timesteps.csv")

    # period → branch (from solve_branch__time_branch.csv: period, branch)
    branch_of: dict[str, str] = {
        r["period"]: r["branch"] for r in sb.iter_rows(named=True)
    }
    # period → time_start (the pbt_profile's time_start axis)
    ts_of: dict[str, str] = {
        r["period"]: r["step"] for r in fts.iter_rows(named=True)
    }

    pbt_lookup: dict[tuple[str, str, str, str], float] = {
        (r["profile"], r["branch"], r["time_start"], r["time"]):
            float(r["pbt_profile"])
        for r in pbt.iter_rows(named=True)
    }

    mismatches: list[str] = []
    checked = 0
    for r in pdtp.iter_rows(named=True):
        b = branch_of.get(r["period"])
        ts = ts_of.get(r["period"])
        if b is None or ts is None:
            continue
        key = (r["profile"], b, ts, r["time"])
        expected = pbt_lookup.get(key)
        if expected is None:
            continue
        checked += 1
        if abs(float(r["value"]) - expected) > 1e-9:
            mismatches.append(
                f"  ({r['profile']!r}, period={r['period']!r}, "
                f"t={r['time']!r}) pdtProfile={r['value']} but "
                f"pbt_profile[branch={b!r}, ts={ts!r}]={expected}"
            )

    assert checked > 0, (
        f"{scenario}: no (profile, period, time) entries cross-matched "
        f"between pdtProfile and pbt_profile — fixture/lookup wiring "
        f"broken"
    )
    assert not mismatches, (
        f"{scenario}: pdtProfile drops per-branch pbt_profile values "
        f"for {len(mismatches)} entries (showing first 5):\n"
        + "\n".join(mismatches[:5])
    )
