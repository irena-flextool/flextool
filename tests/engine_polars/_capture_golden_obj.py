"""Γ.4 — Capture per-fixture ``golden_obj.json`` files.

Reads ``CAPTURE_TARGETS`` (fixture, sqlite, scenario), loads each
via the DB-direct path, builds + solves the LP, writes the resulting
``sol.obj`` into ``<fixture>/golden_obj.json``.

This is a one-shot script — invoke from the repo root::

    python tests/_capture_golden_obj.py

It is NOT a pytest test (the ``_`` prefix excludes it from collection).

Re-running it overwrites existing golden files; use it only when
intentionally re-pinning the parity oracle (e.g. after a Γ.5 fix
that changes the LP).  Tests treat the JSON as immutable.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Late import — ensure repo root is on sys.path when running directly.
ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", ".", "tests"):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import polars as pl  # noqa: E402  # imports after sys.path manipulation above

from polar_high import Problem  # noqa: E402  # imports after sys.path manipulation above
from flextool.engine_polars import SpineDbReader, build_flextool, load_flextool  # noqa: E402  # imports after sys.path manipulation above

from _golden import write_golden, lookup_v_obj_parquet  # noqa: E402  # imports after sys.path manipulation above


# (fixture_dirname, sqlite_name, scenario_name).  Scenario maps to
# Spine scenario filter applied at SpineDbReader construction; the
# fixture's sqlite must carry that scenario.
# (fixture, sqlite, scenario, parquet_glob).  parquet_glob picks the
# specific recorded objective parquet to cross-check against (each
# fixture may have multiple per-solve parquets — by default ``capture``
# sorts the glob and picks the first match, which can be wrong when
# the scenario solves a sub-stage; pinning the filename avoids that.)
CAPTURE_TARGETS = [
    ("work_coal", "tests.sqlite", "coal",
      "v_obj__y2020_2day_dispatch.parquet"),
    ("work_lh2_three_region", "tests.sqlite", "lh2_three_region", None),
    ("work_wind_battery_invest", "tests.sqlite", "wind_battery_invest",
      "v_obj__y2020_5week.parquet"),
    ("work_5weeks_invest_fullYear_dispatch_coal_wind", "tests.sqlite",
      "5weeks_invest_fullYear_dispatch_coal_wind",
      "v_obj__y2020_fullYear_dispatch.parquet"),
    ("work_2day_stochastic_dispatch_full_storage", "tests.sqlite",
      "2_day_stochastic_dispatch",
      "v_obj__2day_dispatch.parquet"),
]


def capture(work: str, sqlite: str, scenario: str) -> tuple[float, str]:
    """Solve via DB-direct, return (obj, source_label)."""
    fixture = ROOT / "tests" / "data" / work
    sqlite_path = fixture / sqlite
    if not sqlite_path.exists():
        raise FileNotFoundError(
            f"sqlite not found for {work}: {sqlite_path}"
        )
    reader = SpineDbReader(sqlite_path, scenario)
    data = load_flextool(fixture, db_reader=reader)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    if not sol.optimal:
        raise RuntimeError(f"non-optimal solve for {work}")
    return float(sol.obj), "db_direct"


def main() -> int:
    failures = []
    for work, sqlite, scenario, parquet_pref in CAPTURE_TARGETS:
        try:
            obj, source = capture(work, sqlite, scenario)
        except Exception as exc:
            print(f"[FAIL] {work}: {exc}", file=sys.stderr)
            failures.append(work)
            continue
        # Cross-check with the recorded parquet to make sure the obj is
        # close enough to flextool's recorded value.  When divergent,
        # flag but still write — the user may explicitly want the
        # DB-direct value as the new oracle.
        fixture = ROOT / "tests" / "data" / work
        parquet = lookup_v_obj_parquet(fixture, parquet_pref)
        parquet_note = ""
        if parquet is not None:
            ftol_obj = float(pl.read_parquet(parquet)["objective"][0])
            rel = abs(obj - ftol_obj) / max(1.0, abs(ftol_obj))
            parquet_note = (
                f" (vs flextool parquet '{parquet.name}'={ftol_obj!r}, "
                f"rel={rel:.2e})"
            )
        path = write_golden(fixture, obj, captured_via=source)
        print(f"[OK] {work}: obj={obj}{parquet_note} → {path.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
