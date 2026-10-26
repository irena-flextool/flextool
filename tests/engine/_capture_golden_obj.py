"""Γ.4 — Capture per-fixture ``golden_obj.json`` files.

For each ``CAPTURE_TARGETS`` entry (fixture dir, JSON DB fixture, scenario) this script builds the
DB from ``tests/fixtures/<json>`` (as the test session's DB fixtures do), runs the scenario's cascade
(:func:`tests._support.load.run_scenario`), rebuilds the last sub-solve's FlexData through the
production loader call (:func:`tests._support.load.load_scenario`), builds + solves the LP and
writes the resulting ``sol.obj`` into ``tests/engine/data/<fixture>/golden_obj.json``.

This is a one-shot script — invoke from the repo root::

    python tests/engine/_capture_golden_obj.py

It is NOT a pytest test (the ``_`` prefix excludes it from collection).

Re-running it overwrites existing golden files; use it only when
intentionally re-pinning the parity oracle (e.g. after a Γ.5 fix
that changes the LP).  Tests treat the JSON as immutable.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# Late import — ensure the repo root and tests/ are on sys.path when running directly.
HERE = Path(__file__).resolve().parent
TESTS = HERE.parent
ROOT = TESTS.parent
for p in (str(ROOT), str(TESTS), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from polar_high import Problem  # noqa: E402  # imports after sys.path manipulation above
from flextool.engine import build_flextool  # noqa: E402  # imports after sys.path manipulation above
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402  # imports after sys.path manipulation above

from _golden import write_golden  # noqa: E402  # imports after sys.path manipulation above
from db_utils import json_to_db  # noqa: E402  # imports after sys.path manipulation above
from tests._support.load import load_scenario, run_scenario  # noqa: E402  # imports after sys.path manipulation above

DATA_DIR = HERE / "data"
FIXTURES_DIR = TESTS / "fixtures"

# (fixture_dirname under tests/engine/data, JSON DB fixture under tests/fixtures, scenario name).
# The scenario maps to the Spine scenario filter; the JSON fixture must carry that scenario.
CAPTURE_TARGETS = [
    ("work_coal", "tests.json", "coal"),
    ("work_lh2_three_region", "lh2_three_region.json", "lh2_three_region"),
    ("work_wind_battery_invest", "tests.json", "wind_battery_invest"),
    ("work_5weeks_invest_fullYear_dispatch_coal_wind", "tests.json",
     "5weeks_invest_fullYear_dispatch_coal_wind"),
    ("work_2day_stochastic_dispatch_full_storage", "stochastics.json",
     "2_day_stochastic_dispatch"),
]


def capture(json_name: str, scenario: str, tmp: Path) -> tuple[float, float | None]:
    """Run the cascade, rebuild its last sub-solve, solve it.

    Returns ``(obj, cascade_obj)``: the rebuilt LP's objective and the objective the cascade itself
    reported for that sub-solve (``run.last_obj``).
    """
    db_path = tmp / f"{scenario}.sqlite"
    url = json_to_db(FIXTURES_DIR / json_name, db_path)
    migrate_database(url)
    solver_config_dir = tmp / "solver_config"
    solver_config_dir.mkdir(exist_ok=True)
    shutil.copy(TESTS / "highs.opt", solver_config_dir / "highs.opt")
    run = run_scenario(url, scenario, tmp / f"work_{scenario}", solver_config_dir=solver_config_dir)
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    if not sol.optimal:
        raise RuntimeError(f"non-optimal solve for {scenario}")
    return float(sol.obj), run.last_obj


def main() -> int:
    failures = []
    for work, json_name, scenario in CAPTURE_TARGETS:
        with tempfile.TemporaryDirectory(prefix=f"capture_{scenario}_") as tmp:
            try:
                obj, cascade_obj = capture(json_name, scenario, Path(tmp))
            except Exception as exc:
                print(f"[FAIL] {work}: {exc}", file=sys.stderr)
                failures.append(work)
                continue
        # Cross-check with the cascade's own objective for that sub-solve.  When divergent, flag but
        # still write — the user may explicitly want the rebuilt value as the new oracle.
        note = ""
        if cascade_obj is not None:
            rel = abs(obj - cascade_obj) / max(1.0, abs(cascade_obj))
            note = f" (vs cascade obj={cascade_obj!r}, rel={rel:.2e})"
        path = write_golden(DATA_DIR / work, obj, captured_via="production_provider")
        print(f"[OK] {work}: obj={obj}{note} → {path.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
