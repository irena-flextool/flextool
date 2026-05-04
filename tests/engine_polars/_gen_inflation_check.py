"""Generate the work_inflation_check fixture.

This fixture is a derivative of ``wind_battery_invest_lifetime_renew``
with one patch applied: the model-level inflation rate is changed from
``0.04`` (scenario default in ``tests.json``) to ``0.02`` (the
user-requested 2%) in the materialised SQLite BEFORE
``runner.write_input(...)``.  Both flextool's CSV writer (which reads
the DB to produce ``input/p_inflation_rate.csv``) and flexpy's
DB-direct ``p_inflation_op_full_cascade_from_source`` (which reads the
same DB) therefore see the same rate=0.02 and the cascade arithmetic
exercises non-trivial 2% compounding.

The 4-period horizon (p2020/p2025/p2030/p2035) × 5 years per period
(20 global years) makes both the per-period operations factor and the
investment-yearly factor non-degenerate sums of compounded terms; see
``tests/test_db_direct_inflation_2pct.py`` for the hand-derivation
arithmetic.

NB: this fixture deliberately keeps ``wind_plant.fixed_cost = 0``
(scenario default) so that ``ed_lifetime_fixed_cost`` is the all-zero
Param.  An earlier iteration patched in ``fixed_cost = 50.0`` for the
fixed-cost cascade hand-check, but the LP-side
battery_inverter ``existing × virtual_unitsize`` broadcast diverges
between the CSV and DB-direct paths on this scenario chain (see
``audit/handoff_full_parity_gaps.md``); the divergence is invisible
when fixed_cost=0 because the binding constraint set is unaffected,
but adding fixed_cost shifts the LP basin and the unrelated
existing-count bug surfaces as a 1% obj gap.  Until that bug is fixed
upstream, the test focuses on ``p_inflation_op`` and
``ed_entity_annual_discounted`` — both of which cleanly exercise the
2% cascade and are bit-for-bit identical between CSV and DB-direct
paths.

Usage:
    ~/venv-spi/bin/python tests/_gen_inflation_check.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# REPO_ROOT still points at the live flextool checkout — flextool's
# Python package, .mod files, and bin/ scripts are consumed live.
REPO_ROOT  = Path("/home/jkiviluo/sources/flextool")
TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = TESTS_DIR / "data"
FIXTURES   = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))

from _db_utils import json_to_db                                      # noqa: E402  vendored
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner    # noqa: E402

from _gen_input import _install_per_sub_solve_snapshot_hook           # noqa: E402


def _patch_inflation_rate_in_db(db_path: Path, new_rate: float,
                                  alt_name: str) -> None:
    """Update ``model.inflation_rate`` in the spine DB to ``new_rate`` for
    the named alternative.  Spine stores the value as a JSON-encoded
    BLOB (e.g. ``b'0.02'`` for a plain float).

    We update the existing row rather than inserting a new one — the
    base scenario already has an inflation_rate row in
    ``invest_solveSequence_5weeks``; we only need to flip its value.
    """
    new_blob = repr(float(new_rate)).encode("utf-8")
    with sqlite3.connect(db_path) as db:
        cur = db.cursor()
        cur.execute(
            "UPDATE parameter_value "
            "SET value = ? "
            "WHERE alternative_id = (SELECT id FROM alternative WHERE name = ?) "
            "  AND parameter_definition_id = "
            "      (SELECT id FROM parameter_definition WHERE name = 'inflation_rate')",
            (new_blob, alt_name),
        )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"expected to update exactly 1 inflation_rate row in "
                f"alt={alt_name!r}; updated {cur.rowcount}"
            )
        db.commit()
    print(f"patched DB {db_path}: inflation_rate({alt_name}) -> {new_rate}")




def generate(scenario: str, workdir: Path,
              inflation_rate: float = 0.02) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "tests.sqlite"
    if db_path.exists():
        db_path.unlink()
    db_url = json_to_db(FIXTURES / "tests.json", db_path)

    _install_per_sub_solve_snapshot_hook()

    runner = FlexToolRunner(
        input_db_url   = db_url,
        scenario_name  = scenario,
        flextool_dir   = REPO_ROOT / "flextool",
        bin_dir        = REPO_ROOT / "bin",
        work_folder    = workdir,
    )
    # Patch inflation_rate in the DB BEFORE write_input so flextool's
    # CSV writer emits the new rate to input/p_inflation_rate.csv and
    # any downstream solve_data factor regeneration sees rate=0.02.
    _patch_inflation_rate_in_db(
        db_path, inflation_rate, alt_name="invest_solveSequence_5weeks")
    runner.write_input(db_url, scenario)
    rc = runner.run_model()
    if rc != 0:
        raise SystemExit(f"flextool failed for {scenario!r}: rc={rc}")
    print(f"\n{scenario} (inflation={inflation_rate}): "
          f"generated under {workdir}")
    return workdir


if __name__ == "__main__":
    workdir = DATA_DIR / "work_inflation_check"
    generate("wind_battery_invest_lifetime_renew", workdir)
