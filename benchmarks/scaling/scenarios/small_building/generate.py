#!/usr/bin/env python3
"""Generate the small_building benchmark scenario.

Characteristics: ~1 node, 2-3 units (wind + battery + demand), short timeline,
single period, no investment. Taken from the existing ``wind_battery`` scenario
in ``templates/examples.sqlite`` (2-day 48h timeline, single 2020 period).

Usage:
    python benchmarks/scaling/scenarios/small_building/generate.py

Output:
    benchmarks/scaling/scenarios/small_building/input.sqlite
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SRC_DB = REPO_ROOT / "templates" / "examples.sqlite"
OUT_DIR = Path(__file__).resolve().parent
OUT_DB = OUT_DIR / "input.sqlite"

# Scenario inside the generated DB that the harness should execute.
SCENARIO_NAME = "wind_battery"


def main() -> int:
    if not SRC_DB.exists():
        print(f"ERROR: source DB not found: {SRC_DB}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DB.unlink(missing_ok=True)
    shutil.copy2(SRC_DB, OUT_DB)
    print(f"Wrote {OUT_DB}")
    print(f"Scenario to run: {SCENARIO_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
