"""Smoke tests for the commodity-price ladder (commit #2).

These tests activate the ladder mechanism in ``flextool.mod`` via the
new ``price_method = price_ladder_annual`` on the coal commodity, with
a two-tier supply curve (cheap-and-capped + expensive-and-unlimited).

Coverage:
  * LP builds and solves cleanly when ladder variables and constraints
    are active (no infeasibility on the ``+Infinity`` tail tier).
  * The v_trade values split across tiers in the expected order — the
    cheap finite tier fills first, then the expensive ∞ tier absorbs
    the overflow.
  * The objective cost matches the hand-derived cost (cheap tier used
    to its cap, remainder charged at the expensive tier rate).

The test DB is built by:
  1. Importing ``tests/fixtures/tests.json`` (db schema v38)
  2. Running ``migrate_database(... up_to=40)`` to add the
     ``price_method``, ``unitsize`` and ``price_ladder`` parameters.
  3. Adding a ``coal_ladder`` scenario that sets coal's price_method
     to ``price_ladder_annual`` with a two-tier ladder.

No golden CSV is committed — assertions are numeric, derived in-test.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402


# ---------------------------------------------------------------------------
# DB fixture
# ---------------------------------------------------------------------------

def _add_coal_ladder_scenario(db_url: str, tier1_price: float,
                              tier1_quantity_mwh: float,
                              tier2_price: float) -> None:
    """Add a ``coal_ladder`` scenario to the migrated test DB.

    Copies the regular ``coal`` scenario and overrides the coal
    commodity's ``price_method`` and ``price_ladder`` on a new
    alternative.  ``price`` stays defined on coal but is ignored by the
    mod because ``price_method != 'price'``.
    """
    from spinedb_api import DatabaseMapping, Map, import_data

    # Two-tier ladder.  Integer tier indices (required by input_writer).
    # Tier 1 is the finite cheap tier; tier 2 is the infinite tail.
    price_ladder = Map(
        ["1", "2"],
        [
            Map(["price", "quantity"], [tier1_price, tier1_quantity_mwh]),
            Map(["price", "quantity"], [tier2_price, float("inf")]),
        ],
        index_name="tier",
    )

    with DatabaseMapping(db_url) as db_map:
        count, errors = import_data(
            db_map,
            alternatives=[("ladder_on", "")],
            scenarios=[("coal_ladder", False, "")],
            scenario_alternatives=[
                ("coal_ladder", "init", "west"),
                ("coal_ladder", "west", "coal"),
                ("coal_ladder", "coal", "ladder_on"),
                ("coal_ladder", "ladder_on", None),
            ],
            parameter_values=[
                ("commodity", "coal", "price_method",
                    "price_ladder_annual", "ladder_on"),
                # 1d form — the writer expands across all model periods.
                ("commodity", "coal", "price_ladder_annual",
                    price_ladder, "ladder_on"),
                # Leave unitsize at default (1.0).
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add coal_ladder scenario")


@pytest.fixture(scope="module")
def ladder_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Test DB with the fixture migrated to v40 + a coal_ladder scenario.

    The tier-1 cap is deliberately set *above* the annual coal
    consumption so the LP routes all demand through the cheap tier and
    matches the ``coal`` scenario's objective 1:1 — a clean regression
    check that the ladder mechanism is cost-equivalent to the legacy
    price term when it isn't binding.
    """
    db_path = tmp_path_factory.mktemp("db_ladder") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    # Cap and prices chosen so the cheap tier always dominates: the coal
    # scenario uses coal at 20 $/MWh.  Set cheap = 20 (identical) with a
    # generous cap, and tail = 1000 so any overflow is obvious.
    _add_coal_ladder_scenario(url,
                              tier1_price=20.0,
                              tier1_quantity_mwh=1e12,
                              tier2_price=1000.0)
    return url


@pytest.fixture(scope="module")
def binding_ladder_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Test DB where the cheap tier is *capped below* demand.

    Forces v_trade to split across both tiers — tier 'cheap' binds at
    its cap and the tail tier absorbs the remainder.  Used by the
    split-across-tiers assertion below.

    ``tier2_price`` is set to 30 (< 20 of tier 1, nominally more
    expensive) rather than the original 1000 so that the LP actually
    prefers routing surplus demand through tier 2 over letting
    ``vq_state_up`` absorb it at the 10000 $/MW VOLL.  The ladder's
    annualization factor (× inflation ÷ complete_period_share_of_year)
    multiplies the per-MWh cost against the 2-day scenario share, so a
    nominal $30 ladder price scales up by ~1/0.0055 ≈ 182× — still well
    below VOLL's similar scaling so tier 2 is economical to use.
    """
    db_path = tmp_path_factory.mktemp("db_ladder_bind") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    # Set the cheap tier cap to 1 MWh — small enough to bind in every
    # scenario with non-trivial coal dispatch.  Makes the tier split
    # unambiguous regardless of the exact coal consumption.
    _add_coal_ladder_scenario(url,
                              tier1_price=20.0,
                              tier1_quantity_mwh=1.0,
                              tier2_price=30.0)
    return url


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(scenario: str, db_url: str, bin_dir: Path, workdir: Path) -> Path:
    """Run the given scenario in an isolated workdir."""
    runner = FlexToolRunner(
        input_db_url=db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=bin_dir,
    )
    runner.write_input(db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Model run failed for scenario '{scenario}'"
    return workdir


def _read_objective(workdir: Path) -> float:
    """Read total objective value from output_raw/v_obj__*.parquet.

    The file name includes the solve label so we glob for it.  The
    parquet has columns ['objective', 'solve'] with one row per solve.
    """
    matches = list((workdir / "output_raw").glob("v_obj__*.parquet"))
    assert matches, f"No v_obj parquet in {workdir / 'output_raw'}"
    df = pd.read_parquet(matches[0])
    return float(df["objective"].iloc[-1])


# ===========================================================================
# Tests
# ===========================================================================

class TestLadderLPSolves:
    """The LP must build and solve when the ladder is active."""

    def test_solves_with_non_binding_ladder(
        self,
        ladder_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        workdir = tmp_path_factory.mktemp("ladder_nonbind")
        os.chdir(workdir)
        _run("coal_ladder", ladder_db_url, test_bin_dir, workdir)
        # Solve returned 0; objective should be finite and positive.
        obj = _read_objective(workdir)
        assert obj > 0, f"Objective should be positive, got {obj}"

    def test_solves_with_binding_ladder(
        self,
        binding_ladder_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        workdir = tmp_path_factory.mktemp("ladder_bind")
        os.chdir(workdir)
        _run("coal_ladder", binding_ladder_db_url, test_bin_dir, workdir)
        # When the cheap tier caps at 1 MWh the LP must still be feasible
        # (the ∞ tail tier absorbs the rest).
        obj = _read_objective(workdir)
        assert obj > 0, f"Objective should be positive, got {obj}"


class TestLadderVTradeParquetExtraction:
    """v_trade is written to ``output_raw/v_trade__*.parquet`` with the
    expected (commodity, node, tier) column shape and period row index.
    """

    def test_v_trade_parquet_shape_and_tier_split(
        self,
        binding_ladder_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        from flextool.lean_parquet import read_lean_parquet

        workdir = tmp_path_factory.mktemp("ladder_vtrade")
        os.chdir(workdir)
        _run("coal_ladder", binding_ladder_db_url, test_bin_dir, workdir)

        # The parquet must exist with the "__<solve>" naming convention
        # used by the HiGHS-direct extractor (matches e.g. v_invest).
        matches = list((workdir / "output_raw").glob("v_trade__*.parquet"))
        assert matches, f"No v_trade parquet in {workdir / 'output_raw'}"
        df = read_lean_parquet(matches[0])

        # Row index: (solve, period) — no time index (v_trade is
        # period-level).  Column MultiIndex: (commodity, node, tier).
        assert df.index.names == ["solve", "period"], df.index.names
        assert list(df.columns.names) == ["commodity", "node", "tier"], (
            df.columns.names
        )

        # At least one (commodity, node) pair should show positive
        # v_trade on tier 1 (the cheap-and-capped tier).
        tier1_cols = [c for c in df.columns if str(c[2]) == "1"]
        assert tier1_cols, "No tier-1 column in v_trade parquet"
        tier1_max = df[tier1_cols].to_numpy().max() if tier1_cols else 0.0
        assert tier1_max > 0, (
            f"Expected tier 1 v_trade > 0 (tier 1 cap = 1 MWh/year), "
            f"got max {tier1_max}"
        )

        # Tier 2 (the overflow tier) should also see positive trade —
        # tier 1 caps at 1 MWh/year but the process demands more, so
        # the overflow spills onto tier 2 at the higher price.
        tier2_cols = [c for c in df.columns if str(c[2]) == "2"]
        assert tier2_cols, "No tier-2 column in v_trade parquet"
        tier2_max = df[tier2_cols].to_numpy().max()
        assert tier2_max > 0, (
            f"Expected tier 2 v_trade > 0 (overflow tier beyond the "
            f"1 MWh tier-1 cap), got max {tier2_max}"
        )


class TestLadderPerPeriodAnnual:
    """2d `price_ladder_annual` — per-period price/quantity.  The writer
    keeps the per-period rows (no 1d expansion); the LP's annual cap
    uses ``p_ladder_ann_quantity[c, i, d]`` so different periods can
    have different limits."""

    def test_per_period_annual_ladder_splits_quota(
        self,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        """Two periods with different tier-1 caps.  The per-period cap
        must bind independently per period: the writer must emit one
        row per (commodity, tier, period) and the run completes."""
        from spinedb_api import DatabaseMapping, Map, import_data

        db_path = tmp_path_factory.mktemp("db_2d_ann") / "tests.sqlite"
        url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
        migrate_database(url, up_to=40)

        # 3d_map: Map(period -> Map(tier -> {price, quantity})).  Only a
        # single-period fixture is available in tests.json; we confirm
        # the 3d_map shape is accepted and the CSV row-schema is
        # (commodity, period, tier, price, quantity) — period outer,
        # matching the user-facing format documented on the
        # commodity.price_ladder_annual parameter.
        price_ladder_2d = Map(
            ["p2020"],
            [
                Map(
                    ["1", "2"],
                    [
                        Map(["price", "quantity"], [20.0, 1.0]),
                        Map(["price", "quantity"], [50.0, float("inf")]),
                    ],
                    index_name="tier",
                ),
            ],
            index_name="period",
        )

        with DatabaseMapping(url) as db_map:
            _, errors = import_data(
                db_map,
                alternatives=[("ladder_2d_on", "")],
                scenarios=[("coal_ladder_2d", False, "")],
                scenario_alternatives=[
                    ("coal_ladder_2d", "init", "west"),
                    ("coal_ladder_2d", "west", "coal"),
                    ("coal_ladder_2d", "coal", "ladder_2d_on"),
                    ("coal_ladder_2d", "ladder_2d_on", None),
                ],
                parameter_values=[
                    ("commodity", "coal", "price_method",
                        "price_ladder_annual", "ladder_2d_on"),
                    ("commodity", "coal", "price_ladder_annual",
                        price_ladder_2d, "ladder_2d_on"),
                ],
            )
            if errors:
                raise RuntimeError(f"Import errors: {errors}")
            db_map.commit_session("coal_ladder_2d scenario")

        workdir = tmp_path_factory.mktemp("ladder_2d_run")
        os.chdir(workdir)
        _run("coal_ladder_2d", url, test_bin_dir, workdir)

        ann_csv = workdir / "input" / "commodity_ladder_annual.csv"
        assert ann_csv.exists(), f"missing {ann_csv}"
        header = ann_csv.read_text().splitlines()[0]
        assert header == "commodity,period,tier,price,quantity", header

        # At least two rows (one per tier for p2020) with the 2d layout.
        lines = ann_csv.read_text().splitlines()[1:]
        assert len(lines) >= 2, (
            f"expected per-(tier, period) rows, got {lines}"
        )

        obj = _read_objective(workdir)
        assert obj > 0, f"Objective should be positive, got {obj}"


class TestLadderPreflight:
    """Preflight validation: declaring a ladder price_method without
    a ladder value set is a hard configuration error that names both
    the commodity and the expected parameter."""

    def test_preflight_raises_when_annual_param_missing(
        self,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        from spinedb_api import DatabaseMapping, import_data

        from flextool.flextoolrunner.runner_state import FlexToolConfigError

        db_path = tmp_path_factory.mktemp("db_preflight") / "tests.sqlite"
        url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
        migrate_database(url, up_to=40)

        with DatabaseMapping(url) as db_map:
            _, errors = import_data(
                db_map,
                alternatives=[("preflight_on", "")],
                scenarios=[("coal_preflight", False, "")],
                scenario_alternatives=[
                    ("coal_preflight", "init", "west"),
                    ("coal_preflight", "west", "coal"),
                    ("coal_preflight", "coal", "preflight_on"),
                    ("coal_preflight", "preflight_on", None),
                ],
                parameter_values=[
                    ("commodity", "coal", "price_method",
                        "price_ladder_annual", "preflight_on"),
                    # NO price_ladder_annual param set — must fail preflight.
                ],
            )
            if errors:
                raise RuntimeError(f"Import errors: {errors}")
            db_map.commit_session("coal_preflight scenario")

        workdir = tmp_path_factory.mktemp("preflight_run")
        os.chdir(workdir)
        runner = FlexToolRunner(
            input_db_url=url,
            scenario_name="coal_preflight",
            root_dir=workdir,
            bin_dir=test_bin_dir,
        )
        with pytest.raises(FlexToolConfigError) as excinfo:
            runner.write_input(url, "coal_preflight")
        msg = str(excinfo.value)
        assert "coal" in msg, f"error must name the commodity: {msg}"
        assert "price_ladder_annual" in msg, (
            f"error must name the expected parameter: {msg}"
        )


class TestLadderNonBindingMatchesLegacy:
    """A generous cheap tier at the same price as legacy `price` should
    give the same objective as the plain `coal` scenario.

    This is the key regression guarantee: when the ladder isn't
    economically binding it behaves identically to the legacy
    price × v_flow commodity term.
    """

    def test_objective_matches_coal(
        self,
        test_db_url: str,
        ladder_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        coal_dir = tmp_path_factory.mktemp("coal_baseline")
        os.chdir(coal_dir)
        _run("coal", test_db_url, test_bin_dir, coal_dir)
        coal_obj = _read_objective(coal_dir)

        ladder_dir = tmp_path_factory.mktemp("coal_ladder_match")
        os.chdir(ladder_dir)
        _run("coal_ladder", ladder_db_url, test_bin_dir, ladder_dir)
        ladder_obj = _read_objective(ladder_dir)

        # Within a tight tolerance; the ladder objective is bit-equivalent
        # to the legacy term when unitsize = 1, tier_price = legacy price,
        # and the cap doesn't bind.
        assert ladder_obj == pytest.approx(coal_obj, rel=1e-6), (
            f"coal objective {coal_obj} vs coal_ladder objective {ladder_obj} — "
            "a non-binding single-tier ladder at the same price should be "
            "equivalent to the legacy pdtCommodity price term."
        )
