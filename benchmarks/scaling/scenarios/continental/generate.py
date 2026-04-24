#!/usr/bin/env python3
"""Generate the continental benchmark scenario.

Characteristics: rivendell-derived 3-region system with 14 nodes, 23 units,
3 connections. Default rivendell has 32 periods (y2019..y2050) which is
slow for benchmarking — this generator adds an alternative that caps
``period_timeset`` to the first 2 periods so the solve stays under ~60 s.

Depends on the `rivendell-to-flextool` generator package (the standalone
`Rivendell_to_FlexTool` repo). Install it into the flextool venv with::

    pip install -e ../Rivendell_to_FlexTool

The source Rivendell SQLite is (re)built into a user-local cache path
the first time this benchmark is generated; no SQLite lands inside the
flextool repo.

Usage:
    python benchmarks/scaling/scenarios/continental/generate.py

Output:
    benchmarks/scaling/scenarios/continental/input.sqlite (gitignored)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from spinedb_api import DatabaseMapping


def _cache_dir() -> Path:
    """Per-user cache dir for the rebuilt rivendell SQLite."""
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "rivendell_to_flextool"


SRC_DB = _cache_dir() / "rivendell.sqlite"
OUT_DIR = Path(__file__).resolve().parent
OUT_DB = OUT_DIR / "input.sqlite"

# Cap to first 2 periods for benchmark speed
CAPPED_PERIODS = ["y2019", "y2020"]
CAP_ALT_NAME = "benchmark_cap_2_periods"
NEW_SCENARIO = "continental_benchmark"
SCENARIO_NAME = NEW_SCENARIO  # harness reads this attribute


def _ensure_source_db() -> None:
    """Build ``SRC_DB`` via the rivendell-to-flextool CLI if missing.

    The build takes ~20 s and lands in a per-user cache dir, keeping
    the flextool repo SQLite-free.
    """
    if SRC_DB.exists():
        return
    try:
        from rivendell_to_flextool.generate import main as _gen_main
        from rivendell_to_flextool.build_db import main as _build_main
    except ImportError as exc:
        print(
            "ERROR: `rivendell_to_flextool` is not importable. Install it "
            "with `pip install -e ../Rivendell_to_FlexTool` (same venv as "
            "flextool), then rerun.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    SRC_DB.parent.mkdir(parents=True, exist_ok=True)
    json_path = SRC_DB.with_suffix(".json")
    print(f"Building cached rivendell JSON → {json_path}")
    _gen_main(["--output", str(json_path)])
    print(f"Building cached rivendell SQLite → {SRC_DB} (3 RP × 48 h)")
    _build_main(["3", "48",
                 "--input", str(json_path),
                 "--output", str(SRC_DB)])


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
    _ensure_source_db()

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
        # and years_represented on the rivendell_invest entity.
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_invest",),
            parameter_definition_name="period_timeset",
            alternative_name=CAP_ALT_NAME,
            value=make_period_timeset_value(CAPPED_PERIODS, "rivendell_timeset"),
            type="map",
        )
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_invest",),
            parameter_definition_name="realized_periods",
            alternative_name=CAP_ALT_NAME,
            value=make_realized_periods_array(CAPPED_PERIODS),
            type="array",
        )
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_invest",),
            parameter_definition_name="invest_periods",
            alternative_name=CAP_ALT_NAME,
            value=make_invest_periods_array(CAPPED_PERIODS),
            type="array",
        )
        db.add_item(
            "parameter_value",
            entity_class_name="solve",
            entity_byname=("rivendell_invest",),
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
