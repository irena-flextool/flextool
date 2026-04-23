#!/usr/bin/env python3
"""Generate the medium_national benchmark scenario.

Characteristics: multi-node network (coal, wind, battery, CHP, water pump,
heat, EV, etc.), ~20 unit/connection/node entities, network topology, single
period. Taken from the ``network_all_tech`` scenario in
``templates/examples.sqlite``.

Timeline is 2-day (48h) — examples.sqlite does not carry a pre-built 720h
dispatch scenario for this network; the 48h case still produces a ~2500-row
MIP which is the right scale for a "medium" benchmark.

Usage:
    python benchmarks/scaling/scenarios/medium_national/generate.py

Output:
    benchmarks/scaling/scenarios/medium_national/input.sqlite
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
SCENARIO_NAME = "network_all_tech"


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
