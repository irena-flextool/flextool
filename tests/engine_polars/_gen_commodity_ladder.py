"""Generate the commodity-ladder parity fixtures.

We need two fixtures:

* ``work_commodity_ladder_annual``  — coal scenario with
  ``price_method = price_ladder_annual`` and a 2-tier ladder where the
  first (cheap) tier is capped well below annual demand and the second
  tier is the +∞ tail (sentinel 1e30 in the CSV) at a higher price.
  This forces v_trade to split across two tiers.

* ``work_commodity_ladder_cumulative`` — same single-period coal scenario
  but ``price_method = price_ladder_cumulative`` with a 2-tier ladder.
  On a non-rolling single solve the cumulative cap reduces to the
  pre-refactor form (``f_d_k[d] = 1.0`` for the only realized period).

Both fixtures bind tier 1 — that's the interesting case, where two
distinct prices contribute to the objective.

Generation procedure (mirrors flextool's ``test_commodity_ladder_smoke.py``):

  1. Run ``json_to_db`` to convert ``tests.json`` → SQLite.
  2. ``migrate_database(url, up_to=40)`` to add ``price_method`` /
     ``price_ladder_*`` parameter definitions.
  3. ``import_data`` adds a new alternative + scenario layered on the
     stock ``coal`` scenario, setting ``price_method`` and the ladder
     map.
  4. ``FlexToolRunner.write_input`` + ``run_model`` produces flexpy's
     reference solution.

Usage::

    ~/venv-spi/bin/python tests/_gen_commodity_ladder.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# REPO_ROOT still points at the live flextool checkout because we use
# flextool's Python packages (FlexToolRunner, migrate_database) and its
# bin/ + .mod files live.  Only data fixtures + json_to_db are vendored.
REPO_ROOT  = Path("/home/jkiviluo/sources/flextool")
TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = TESTS_DIR / "data"
FIXTURES   = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))

from _db_utils import json_to_db                                      # noqa: E402  vendored
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner    # noqa: E402
from flextool.update_flextool.db_migration import migrate_database    # noqa: E402

# Two-tier ladder configuration shared by both scenarios.  Tier 1 is the
# finite cheap tier capped below dispatch demand so the cap binds; tier
# 2 is the +∞ tail at a higher price so the LP must spill into it.
TIER1_PRICE     = 20.0     # same as the legacy ``coal.price`` for clean book-keeping
TIER1_QUANTITY  = 1.0      # MWh — much smaller than the coal scenario's annual demand
TIER2_PRICE     = 30.0     # higher than tier 1 — overflow goes here
# Tier 2's quantity is +∞; the writer renders it as the 1e30 sentinel
# which the .mod's filters interpret as "no cap".


def _add_ladder_scenario(
    db_url: str,
    *,
    method: str,           # 'price_ladder_annual' or 'price_ladder_cumulative'
    scenario_name: str,
    alternative_name: str,
) -> None:
    """Add a coal_<method> scenario layered on the stock ``coal`` chain.

    Mirrors ``test_commodity_ladder_smoke._add_coal_ladder_scenario``
    structure: the new alternative overrides ``price_method`` and the
    relevant ladder map; ``coal.price`` stays defined but is ignored
    by the .mod because ``price_method != 'price'``.
    """
    from spinedb_api import DatabaseMapping, Map, import_data

    # Two-tier ladder.  Integer tier indices (the writer's tier_int
    # cast requires this).  Tier 1: finite cheap tier; Tier 2: ∞ tail.
    if method == "price_ladder_annual":
        # 3d_map (period -> tier -> {price, quantity}). Spine 3d_map outer
        # index = period; we use 'p2020' to match the coal scenario's
        # single dispatch period.
        price_ladder = Map(
            ["p2020"],
            [
                Map(
                    ["1", "2"],
                    [
                        Map(["price", "quantity"], [TIER1_PRICE, TIER1_QUANTITY]),
                        Map(["price", "quantity"], [TIER2_PRICE, float("inf")]),
                    ],
                    index_name="tier",
                ),
            ],
            index_name="period",
        )
    else:
        # cumulative: 2d_map (tier -> {price, quantity}).  No period
        # dimension on the parameter — applies across the whole horizon.
        price_ladder = Map(
            ["1", "2"],
            [
                Map(["price", "quantity"], [TIER1_PRICE, TIER1_QUANTITY]),
                Map(["price", "quantity"], [TIER2_PRICE, float("inf")]),
            ],
            index_name="tier",
        )

    with DatabaseMapping(db_url) as db_map:
        _, errors = import_data(
            db_map,
            alternatives=[(alternative_name, "")],
            scenarios=[(scenario_name, False, "")],
            scenario_alternatives=[
                (scenario_name, "init", "west"),
                (scenario_name, "west", "coal"),
                (scenario_name, "coal", alternative_name),
                (scenario_name, alternative_name, None),
            ],
            parameter_values=[
                ("commodity", "coal", "price_method", method, alternative_name),
                ("commodity", "coal", method, price_ladder, alternative_name),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session(f"Add {scenario_name} scenario")


def generate(
    *,
    method: str,
    workdir: Path,
    scenario_name: str,
    alternative_name: str,
) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "tests.sqlite"
    if db_path.exists():
        db_path.unlink()
    db_url = json_to_db(FIXTURES / "tests.json", db_path)
    migrate_database(db_url, up_to=40)
    _add_ladder_scenario(
        db_url,
        method=method,
        scenario_name=scenario_name,
        alternative_name=alternative_name,
    )

    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        runner = FlexToolRunner(
            input_db_url   = db_url,
            scenario_name  = scenario_name,
            flextool_dir   = REPO_ROOT / "flextool",
            bin_dir        = REPO_ROOT / "bin",
            root_dir       = workdir,
            work_folder    = workdir,
        )
        runner.write_input(db_url, scenario_name)
        rc = runner.run_model()
    finally:
        os.chdir(prev_cwd)

    if rc != 0:
        raise SystemExit(f"flextool failed for {scenario_name!r}: rc={rc}")

    # Reset rolling accumulator files to their header-only first-solve
    # state.  flextool writes the realized MWh / sim_hours into these
    # AFTER the solve completes (per-roll handoff), but flexpy replays
    # the BEFORE state (no prior roll → zero accumulator → caps reduce
    # to their pre-refactor form).  Without this reset, flexpy would
    # see the post-solve values and the cap RHS would be off by the
    # realized MWh.  Mirrors ``solve_writers.write_empty_cumulative_files``.
    sd = workdir / "solve_data"
    (sd / "ladder_cum_realized_mwh.csv").write_text(
        "commodity,tier,period,p_ladder_cum_realized_mwh\n")
    (sd / "ladder_cum_sim_hours.csv").write_text(
        "period,p_ladder_cum_sim_hours\n")

    print(f"\n{scenario_name}: generated under {workdir}")
    return workdir


if __name__ == "__main__":
    generate(
        method           = "price_ladder_annual",
        workdir          = DATA_DIR / "work_commodity_ladder_annual",
        scenario_name    = "coal_ladder_annual",
        alternative_name = "ladder_ann_on",
    )
    generate(
        method           = "price_ladder_cumulative",
        workdir          = DATA_DIR / "work_commodity_ladder_cumulative",
        scenario_name    = "coal_ladder_cumulative",
        alternative_name = "ladder_cum_on",
    )
