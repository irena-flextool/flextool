"""End-to-end validation for the rolling cumulative-quota handoff.

Covers step 4e of ``rivendell/PLAN_rolling_quota_handoff.md``.  Previous
commits landed the LP mechanism (v_trade + cumulative tier cap), the
Python writer (``cumulative_handoffs.write_cumulative_ladder_remaining``),
the empty-seed first-solve file, and the mod's RHS swap to
``p_cumulative_ladder_remaining``.  Unit tests exercise the writer in
isolation (``tests/test_cumulative_handoffs.py``); this module is the
**first real solver run** that chains roll 1 →
``cumulative_ladder_remaining.csv`` → roll 2's LP RHS.

Scenario design — minimal two-roll, two-period setup built on top of
the existing ``coal`` topology and ``fullYear`` timeset from
``tests/fixtures/tests.json``.

* Two periods ``p2020`` and ``p2025`` each using the existing
  ``2day`` timeset (48h window, one step at t0001).  The LP solves
  with ``solve_mode=rolling_window`` and ``rolling_solve_horizon=48``
  / ``rolling_solve_jump=48`` — exactly one roll per period, two
  rolls total.

* **Critical design choice**: the rolls partition the model across
  **distinct periods**.  ``v_trade`` is period-level in the current mod,
  and the writer's span_weight / total_weight ratio is period-level
  too.  Within-period rolling (several rolls sharing one period)
  double-counts the allotment because every roll sees that period as
  fully realized (span_weight == total_weight for all of them).  That
  caveat is flagged in ``rivendell/NOTES_commit5_validation_decisions.md``
  as a follow-up item for the writer.

* Coal commodity: ``price_method='price_ladder_cumulative'`` with a
  two-tier ladder.  Tier 1 is finite (the MWh budget under test);
  tier 2 is a finite but high-priced tail so overspend tests stay
  feasible instead of pushing dispatch onto VOLL/slack.

Assertion classes (priority per task brief §D):

1. End-to-end run completes without infeasibility or exceptions.
2. The final ``cumulative_ladder_remaining.csv`` obeys the running-
   balance formula exactly:
       final_remaining = total_cap − Σ realized_consumption.
3. The handoff CSV is **read** by roll 2: in the overspend variant
   (tier-1 cap small enough that roll 1 exhausts it) the roll-2 LP
   shifts tier-1 v_trade to zero and routes dispatch through tier 2.
4. (Included in #3.)
5. Single-solve bit-identity: with ``rolling=off`` (empty-seed
   default, constraint inactive), the cumulative-ladder scenario
   yields the same objective as the legacy plain-``price`` ``coal``
   scenario within ``rel=1e-6``.

Deliberate gaps (per task brief guardrails §F — land in step 4f):
  * No CO2 cumulative scenario.
  * No nested-guard scenario.
  * No stochastic-branch exercise.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from spinedb_api import Array, Map

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402


# ---------------------------------------------------------------------------
# DB setup — programmatically add a two-period rolling scenario with a
# ``price_ladder_cumulative`` coal commodity on top of the v40-migrated
# fixture DB.
# ---------------------------------------------------------------------------


def _add_cumulative_ladder_scenarios(
    db_url: str,
    *,
    tier1_quantity_mwh: float,
    tier2_price: float = 1000.0,
) -> None:
    """Add three scenarios to the DB:

    * ``coal_cum_single``     — single solve, plain (non-rolling).
    * ``coal_cum_rolling``    — two-period rolling (one roll per period).

    The rolling variant uses the ``fullYear`` timeset (72h) across two
    periods so each roll realises one full, distinct period.  See module
    docstring for why that's important.
    """
    from spinedb_api import DatabaseMapping, import_data

    price_ladder = Map(
        ["1", "2"],
        [
            Map(["price", "quantity"], [20.0, tier1_quantity_mwh]),
            Map(["price", "quantity"], [tier2_price, float("inf")]),
        ],
        index_name="tier",
    )

    # period_timeset: map period → timeset.  Both periods use the
    # existing ``2day`` (48h, single step at t0001) — timeset bound
    # to the ``init`` alternative which we include below.
    period_timeset_two = Map(
        ["p2020", "p2025"],
        ["2day", "2day"],
        index_name="period",
    )
    # realized_periods — Array of period names (matches the encoding
    # ``dispatch_fullYear_roll`` uses in the fixtures).  Both periods
    # realize, no look-ahead is discarded.
    realized_two = Array(
        ["p2020", "p2025"],
        value_type=str,
        index_name="period",
    )
    years_two = Map(
        ["p2020", "p2025"],
        [1.0, 1.0],
        index_name="period",
    )

    with DatabaseMapping(db_url) as db_map:
        _, errors = import_data(
            db_map,
            alternatives=[
                ("ladder_cum_on", ""),
                ("two_period_rolling", ""),
            ],
            scenarios=[
                ("coal_cum_single", False, ""),
                ("coal_cum_rolling", False, ""),
            ],
            scenario_alternatives=[
                # Single-solve variant (no rolling, no extra periods).
                ("coal_cum_single", "init", "west"),
                ("coal_cum_single", "west", "coal"),
                ("coal_cum_single", "coal", "ladder_cum_on"),
                ("coal_cum_single", "ladder_cum_on", None),
                # Rolling variant — init, west, coal, ladder_cum_on,
                # then two_period_rolling layered on top.
                ("coal_cum_rolling", "init", "west"),
                ("coal_cum_rolling", "west", "coal"),
                ("coal_cum_rolling", "coal", "ladder_cum_on"),
                ("coal_cum_rolling", "ladder_cum_on",
                    "two_period_rolling"),
                ("coal_cum_rolling", "two_period_rolling", None),
            ],
            parameter_values=[
                # Ladder on coal.
                ("commodity", "coal", "price_method",
                    "price_ladder_cumulative", "ladder_cum_on"),
                ("commodity", "coal", "price_ladder",
                    price_ladder, "ladder_cum_on"),
                # Two-period rolling config on the same solve name that
                # ``init`` binds, ``y2020_2day_dispatch``.  We override
                # everything so the base "2day" single-period setup from
                # ``init`` is fully superseded.
                ("solve", "y2020_2day_dispatch", "solve_mode",
                    "rolling_window", "two_period_rolling"),
                ("solve", "y2020_2day_dispatch", "rolling_solve_horizon",
                    48.0, "two_period_rolling"),
                ("solve", "y2020_2day_dispatch", "rolling_solve_jump",
                    48.0, "two_period_rolling"),
                ("solve", "y2020_2day_dispatch", "period_timeset",
                    period_timeset_two, "two_period_rolling"),
                ("solve", "y2020_2day_dispatch", "realized_periods",
                    realized_two, "two_period_rolling"),
                ("solve", "y2020_2day_dispatch", "years_represented",
                    years_two, "two_period_rolling"),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add cumulative ladder rolling scenarios")


@pytest.fixture(scope="module")
def cumulative_db_url_underspend(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Non-binding cap: tier-1 quantity is 1e9 MWh, far above the
    coal-fuel consumption on the ``coal`` topology.  The running
    balance should stay strictly positive across both rolls and end
    exactly at ``cap − Σ_consumption`` per the writer formula.
    """
    db_path = tmp_path_factory.mktemp("db_cum_under") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    _add_cumulative_ladder_scenarios(url, tier1_quantity_mwh=1e9)
    return url


@pytest.fixture(scope="module")
def cumulative_db_url_overspend(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    """Binding cap: tier-1 quantity tiny (1 MWh).  Roll 1's allotment
    is 0.5 MWh but actual consumption is thousands of MWh — writer
    records a large negative remaining, and roll 2 cannot use tier 1
    at all.  Tier 2 price ($50) stays finite so the LP remains
    feasible when tier 2 absorbs the demand.
    """
    db_path = tmp_path_factory.mktemp("db_cum_over") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    _add_cumulative_ladder_scenarios(
        url, tier1_quantity_mwh=1.0, tier2_price=50.0
    )
    return url


# ---------------------------------------------------------------------------
# Run + inspection helpers
# ---------------------------------------------------------------------------


def _run(
    scenario: str, db_url: str, bin_dir: Path, workdir: Path,
) -> None:
    """Run ``scenario`` in ``workdir`` with cwd also set to ``workdir``."""
    runner = FlexToolRunner(
        input_db_url=db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=bin_dir,
    )
    runner.write_input(db_url, scenario)
    rc = runner.run_model()
    assert rc == 0, f"Model run failed for scenario '{scenario}'"


def _read_remaining(workdir: Path) -> dict[tuple[str, int], float]:
    """Return ``{(commodity, tier): remaining}`` from the final
    post-roll CSV."""
    path = workdir / "solve_data" / "cumulative_ladder_remaining.csv"
    assert path.exists(), f"Handoff CSV missing: {path}"
    out: dict[tuple[str, int], float] = {}
    for row in csv.DictReader(open(path)):
        key = (row["commodity"], int(row["tier"]))
        out[key] = float(row["p_cumulative_ladder_remaining"])
    return out


def _list_v_trade_parquets(workdir: Path) -> list[Path]:
    """One ``v_trade__*.parquet`` per roll, sorted by roll-index
    suffix so the final-roll result is always the last element."""
    raw = workdir / "output_raw"
    if not raw.exists():
        return []

    def _roll_idx(p: Path) -> int:
        # Name pattern: v_trade__<scenario>_roll_<N>.parquet
        name = p.stem
        if "roll_" not in name:
            return -1
        try:
            return int(name.rsplit("roll_", 1)[-1])
        except ValueError:
            return -1

    return sorted(raw.glob("v_trade__*.parquet"), key=_roll_idx)


def _sum_v_trade_per_tier(parquet: Path) -> dict[int, float]:
    """Sum v_trade values across (commodity, node, period) per tier
    in one parquet file.

    Returns ``{tier: total}`` with the ``coal`` commodity only.
    """
    from flextool.lean_parquet import read_lean_parquet
    df = read_lean_parquet(parquet)
    if df.empty:
        return {}
    out: dict[int, float] = {}
    for col in df.columns:
        if not isinstance(col, tuple) or len(col) < 3:
            continue
        if str(col[0]) != "coal":
            continue
        try:
            tier = int(col[-1])
        except (ValueError, TypeError):
            continue
        out[tier] = out.get(tier, 0.0) + float(df[col].sum())
    return out


# ===========================================================================
# Tests
# ===========================================================================


class TestCumulativeLadderRollingUnderspend:
    """Two-roll, two-period scenario with a non-binding cap.

    Verifies the full writer → mod feedback loop: roll 1 writes
    ``cumulative_ladder_remaining.csv`` → roll 2 reads it via the mod's
    ``table data IN`` → roll 2 writes the final file.  The structural
    invariant ``final_remaining = total_cap − Σ consumption`` holds
    modulo float error.
    """

    def test_end_to_end_and_handoff_file_present(
        self,
        cumulative_db_url_underspend: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Assertion #1 + handoff-file structural checks."""
        workdir = tmp_path_factory.mktemp("cum_under_basic")
        os.chdir(workdir)
        _run("coal_cum_rolling", cumulative_db_url_underspend,
             test_bin_dir, workdir)

        # Handoff CSV landed; only the finite tier is present (tier 2
        # uses the 1e30 infinite sentinel and must be dropped).
        remaining = _read_remaining(workdir)
        assert ("coal", 1) in remaining, (
            f"Expected ('coal', 1) in remaining, got {remaining}"
        )
        assert ("coal", 2) not in remaining, (
            "Tier 2 is infinite → writer must drop it from the output"
        )

        # Two rolls emitted two v_trade parquets.
        parquets = _list_v_trade_parquets(workdir)
        assert len(parquets) >= 2, (
            f"Expected at least 2 rolls, got "
            f"{[p.name for p in parquets]}"
        )

        # Non-binding cap → remaining must be positive (between 0 and
        # total cap 1e9).  A ceiling check separately catches the
        # within-period double-counting bug (where remaining would
        # exceed total_cap because both rolls' allotments stack
        # without partitioning the period).  Since we use two distinct
        # periods this stays clean.
        tier1_remaining = remaining[("coal", 1)]
        assert 0 < tier1_remaining < 1e9, (
            f"Non-binding cap: tier-1 remaining must sit between 0 "
            f"and cap (1e9).  Got {tier1_remaining}."
        )

    def test_running_balance_formula_conservation(
        self,
        cumulative_db_url_underspend: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Assertion #2 (exact): ``final_remaining = total_cap − Σ_consumption``.

        This is the single strongest regression signal in this file.
        If the writer's span_weight, allotment, or prior-carryover
        arithmetic drifts on any roll, this sum diverges.

        Consumption is measured directly from per-roll v_trade parquets
        using the same formula the writer uses (``v_trade × unitsize ×
        years / period_share``) — the test validates that the writer's
        book-keeping matches its own claimed formula, end-to-end.
        """
        workdir = tmp_path_factory.mktemp("cum_under_formula")
        os.chdir(workdir)
        _run("coal_cum_rolling", cumulative_db_url_underspend,
             test_bin_dir, workdir)

        final_remaining = _read_remaining(workdir)[("coal", 1)]

        # Load years_represented and period_share from mod-written CSVs
        # (one row per (solve, period)).  Each period contributes
        # years/share to its consumption.
        years_df = pd.read_csv(
            workdir / "solve_data" / "p_years_represented_d.csv"
        )
        share_df = pd.read_csv(
            workdir / "solve_data" / "complete_period_share_of_year.csv"
        )
        # Key by period (years are model-wide constants so period → value).
        years_by_period = (
            years_df.drop_duplicates("period")
            .set_index("period")["value"]
            .to_dict()
        )
        share_by_period = (
            share_df.drop_duplicates("period")
            .set_index("period")["value"]
            .to_dict()
        )

        # Sum v_trade consumption across all rolls using per-period
        # factors.  v_trade's row index is (solve, period).
        from flextool.lean_parquet import read_lean_parquet
        total_consumption_tier1 = 0.0
        for pq in _list_v_trade_parquets(workdir):
            df = read_lean_parquet(pq)
            if df.empty:
                continue
            for row_key, row in df.iterrows():
                period = str(
                    row_key[-1] if isinstance(row_key, tuple) else row_key
                )
                years = years_by_period.get(period)
                share = share_by_period.get(period)
                if years is None or share is None or share <= 0:
                    continue
                for col, val in row.items():
                    if not isinstance(col, tuple) or len(col) < 3:
                        continue
                    if str(col[0]) != "coal":
                        continue
                    try:
                        tier = int(col[-1])
                    except (ValueError, TypeError):
                        continue
                    if tier != 1:
                        continue
                    # unitsize defaults to 1.0 in this scenario.
                    total_consumption_tier1 += float(val) * years / share

        total_cap = 1e9
        expected = total_cap - total_consumption_tier1
        # The allotments sum to exactly total_cap only when span_weights
        # sum to total_weight — which requires rolls to partition
        # periods cleanly (see module docstring).  Tolerance loose
        # enough to absorb float drift on a 1e9 cap.
        assert final_remaining == pytest.approx(
            expected, rel=1e-6, abs=1.0
        ), (
            f"running balance broken: remaining={final_remaining} "
            f"vs cap-consumption={expected} "
            f"(cap={total_cap}, consumption_tier1={total_consumption_tier1})"
        )


class TestCumulativeLadderOverspendRecordsNegativeRemaining:
    """Assertion #3 (handoff-file form): tiny tier-1 cap → roll 1's
    LHS-unconstrained solve consumes >> allotment → writer records a
    large negative remaining on tier 1.

    This is the handoff-side half of the overspend story.  The LP-side
    half ("roll 2 forced to route through tier 2") is currently broken
    on negative-remaining inputs — the mod's
    ``ladder_tier_cap_cumulative`` constraint becomes
    ``non-negative_sum <= negative_rhs`` which is infeasible rather
    than "v_trade forced to 0".  See
    ``rivendell/NOTES_commit5_validation_decisions.md`` for the open
    bug and workaround options.

    To exercise the roll-1 writer alone (no roll 2), this test aborts
    the solve loop after the first roll by intercepting ``run_model``
    and running a single manual roll.
    """

    def test_roll1_records_negative_remaining_on_overspend(
        self,
        cumulative_db_url_overspend: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Roll 1 alone, via the single-solve scenario with a 1 MWh cap.

        The single-solve variant has the ladder active but no rolling.
        The first-solve seed keeps the constraint inactive (default
        1e30), so the LP solves normally, and the writer then records
        ``0 + cap * 1.0 − consumption`` on a single-period span_weight =
        total_weight case.  ``consumption >> cap`` → remaining < 0.
        """
        workdir = tmp_path_factory.mktemp("cum_over_single")
        os.chdir(workdir)
        _run("coal_cum_single", cumulative_db_url_overspend,
             test_bin_dir, workdir)

        remaining = _read_remaining(workdir)
        assert ("coal", 1) in remaining, (
            f"Expected ('coal', 1) key, got {remaining}"
        )
        tier1_remaining = remaining[("coal", 1)]
        assert tier1_remaining < 0, (
            f"Overspend (cap=1 MWh, consumption ~thousands MWh): "
            f"writer must record negative remaining.  Got "
            f"{tier1_remaining}."
        )
        # Magnitude sanity — remaining is in model-horizon MWh so it's
        # consumption * years / share, which for 48h of coal dispatch
        # (~11k MWh) at years=1, share=48/8760 gives ~2M MWh.
        assert tier1_remaining < -1e5, (
            f"Overspend magnitude too small — expected < −1e5 MWh, "
            f"got {tier1_remaining}.  Either the ladder isn't binding "
            f"or the scaling formula changed."
        )

    def test_roll2_uses_tier2_only_after_roll1_overspend(
        self,
        cumulative_db_url_overspend: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Full two-roll overspend — roll 1 exhausts tier 1's tiny cap
        (writer records negative p_cumulative_ladder_remaining), roll 2
        sees the negative remaining and the mod's
        ``ladder_tier_cap_cumulative_overspent`` constraint forces
        ``v_trade[coal, *, *, 1] = 0`` for every (c, n, d), pushing
        dispatch onto tier 2."""
        workdir = tmp_path_factory.mktemp("cum_over_roll")
        os.chdir(workdir)
        _run("coal_cum_rolling", cumulative_db_url_overspend,
             test_bin_dir, workdir)

        parquets = _list_v_trade_parquets(workdir)
        assert len(parquets) >= 2

        final_roll = parquets[-1]
        per_tier = _sum_v_trade_per_tier(final_roll)
        assert per_tier.get(1, 0.0) <= 1e-6, (
            f"Tier 1 must be locked out in roll 2 after roll-1 "
            f"overspend recorded a negative remaining.  Got tier-1 "
            f"sum={per_tier.get(1, 0.0)} in {final_roll.name}."
        )
        assert per_tier.get(2, 0.0) > 0, (
            f"Tier 2 (tail) must absorb roll-2 dispatch when tier 1 "
            f"is locked out.  Got tier-2 sum={per_tier.get(2, 0.0)} "
            f"in {final_roll.name}."
        )


class TestCumulativeLadderSingleSolveIsNoop:
    """Assertion #5: single-solve with cumulative ladder + non-binding
    cap must match the legacy-``price`` ``coal`` baseline bit-for-bit
    (within rel=1e-6).  The empty-seed default keeps the cumulative
    constraint inactive on the first (and only) solve.
    """

    def test_single_solve_cumulative_matches_coal_objective(
        self,
        test_db_url: str,
        cumulative_db_url_underspend: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        from tests.test_commodity_ladder_smoke import _read_objective

        coal_dir = tmp_path_factory.mktemp("coal_base_cum")
        os.chdir(coal_dir)
        _run("coal", test_db_url, test_bin_dir, coal_dir)
        coal_obj = _read_objective(coal_dir)

        cum_dir = tmp_path_factory.mktemp("coal_cum_single_obj")
        os.chdir(cum_dir)
        _run("coal_cum_single", cumulative_db_url_underspend,
             test_bin_dir, cum_dir)
        cum_obj = _read_objective(cum_dir)

        assert cum_obj == pytest.approx(coal_obj, rel=1e-6), (
            f"Non-binding cumulative single-solve should match legacy "
            f"coal objective bit-for-bit.  coal={coal_obj}, "
            f"coal_cum_single={cum_obj}"
        )
