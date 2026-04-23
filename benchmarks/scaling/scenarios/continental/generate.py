#!/usr/bin/env python3
"""Generate the continental benchmark scenario.

Characteristics: rivendell-derived 3-region system with 14 nodes, 23 units,
3 connections. Default rivendell has 32 periods (y2019..y2050) which is
slow for benchmarking — this generator adds an alternative that caps
``period_timeset`` to the first 2 periods so the solve stays under ~60 s.

Requires ``rivendell/rivendell.sqlite`` to exist. If missing, run
``python rivendell/build_rivendell_db.py`` first (which itself needs the
Rivendell ODS/XLSX input file and ``rivendell/rivendell.json``).

Usage:
    python benchmarks/scaling/scenarios/continental/generate.py

Output:
    benchmarks/scaling/scenarios/continental/input.sqlite
"""

from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

from spinedb_api import DatabaseMapping

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DB = REPO_ROOT / "rivendell" / "rivendell.sqlite"
OUT_DIR = Path(__file__).resolve().parent
OUT_DB = OUT_DIR / "input.sqlite"

# Cap to first 2 periods for benchmark speed
CAPPED_PERIODS = ["y2019", "y2020"]
CAP_ALT_NAME = "benchmark_cap_2_periods"
NEW_SCENARIO = "continental_benchmark"
SCENARIO_NAME = NEW_SCENARIO  # harness reads this attribute


def make_period_timeset_value(periods: list[str], timeset_name: str) -> bytes:
    """Build the raw base64-decoded JSON payload for the period_timeset Map."""
    payload = {
        "index_type": "str",
        "rank": 1,
        "data": [[p, timeset_name] for p in periods],
    }
    return json.dumps(payload).encode("utf-8")


def make_years_represented(periods: list[str], years: int = 1) -> bytes:
    payload = {
        "index_type": "str",
        "rank": 1,
        "data": [[p, years] for p in periods],
    }
    return json.dumps(payload).encode("utf-8")


def make_realized_periods_array(periods: list[str]) -> bytes:
    payload = {"value_type": "str", "data": periods}
    return json.dumps(payload).encode("utf-8")


def make_invest_periods_array(periods: list[str]) -> bytes:
    return make_realized_periods_array(periods)


def main() -> int:
    if not SRC_DB.exists():
        print(
            f"ERROR: source DB not found: {SRC_DB}\n"
            f"Run `python rivendell/build_rivendell_db.py` to generate it.",
            file=sys.stderr,
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DB.unlink(missing_ok=True)
    shutil.copy2(SRC_DB, OUT_DB)

    url = f"sqlite:///{OUT_DB.resolve()}"
    with DatabaseMapping(url) as db:
        # Add the cap alternative.
        db.add_item(
            "alternative",
            name=CAP_ALT_NAME,
            description="Cap period_timeset to first 2 periods for benchmark speed",
        )
        # Override period_timeset, realized_periods, invest_periods,
        # and years_represented on the rivendell_solve entity.
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_solve",),
            parameter_definition_name="period_timeset",
            alternative_name=CAP_ALT_NAME,
            value=make_period_timeset_value(CAPPED_PERIODS, "rivendell_timeset"),
            type="map",
        )
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_solve",),
            parameter_definition_name="realized_periods",
            alternative_name=CAP_ALT_NAME,
            value=make_realized_periods_array(CAPPED_PERIODS),
            type="array",
        )
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_solve",),
            parameter_definition_name="invest_periods",
            alternative_name=CAP_ALT_NAME,
            value=make_invest_periods_array(CAPPED_PERIODS),
            type="array",
        )
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_solve",),
            parameter_definition_name="years_represented",
            alternative_name=CAP_ALT_NAME,
            value=make_years_represented(CAPPED_PERIODS, years=1),
            type="map",
        )

        # Create the benchmark scenario: B0_base_slice chain + cap
        base_chain = ["B0_base_system", "timeline_slice", CAP_ALT_NAME]
        _, err = db.add_item(
            "scenario",
            name=NEW_SCENARIO,
            active=False,
            description="Continental benchmark: rivendell B0 slice capped to 2 periods",
        )
        if err:
            print(f"WARN scenario {NEW_SCENARIO}: {err}")
        for i, alt in enumerate(base_chain):
            _, err = db.add_item(
                "scenario_alternative",
                scenario_name=NEW_SCENARIO,
                alternative_name=alt,
                rank=i + 1,
            )
            if err:
                print(f"WARN scen_alt {NEW_SCENARIO}/{alt}: {err}")
        db.commit_session("Add continental benchmark scenario with 2-period cap")

    print(f"Wrote {OUT_DB}")
    print(f"Scenario to run: {NEW_SCENARIO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
