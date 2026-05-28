"""Augment ``stochastics_pbt_inflow.json`` so that ``downriver`` carries a
parent-period-only stochastic 3d_map ``inflow``.

Purpose
-------
Phase C, Fixture B.  Exercises Branch 2 (parent-period fold-in) of
``write_pdtNodeInflow`` for a node that is NOT in any
``groupIncludeStochastics`` (so Branch 1 does not fire) but whose parent
period (via ``period__branch``) has pbt rows.

Structural augmentation
-----------------------
1. ``downriver`` is NOT added to ``add_stochastics`` (only ``hydro_reservoir``
   stays in the stochastic group).
2. ``downriver``'s scalar ``inflow = -100`` is REPLACED by a 3d_map
   authored on the ``realized`` branch only (the parent branch via
   ``solve_branch__time_branch.csv`` for ``period1``):

       branch ∈ {realized}             # only the parent's branch
       time_start ∈ {t0001, t0025}     # 2 time_starts for multi-ts B.2
       t ∈ {t0001..t0048}              # 48 timesteps

       value(ts_idx, t_idx) = ts_idx * 10 + t_idx + 1

   (Offset of +1 so all values are > 0, easy to inspect.)

Lookup tables (from a legacy ``2_day_stochastic_dispatch`` run)
--------------------------------------------------------------
* ``period__branch.csv`` (column 1 → column 0):
      pe_for_d[period1]         = [period1]
      pe_for_d[period1_realized] = [period1]
      pe_for_d[period1_upper]   = [period1]
      pe_for_d[period1_lower]   = [period1]
      pe_for_d[period1_mid]     = [period1]

* ``solve_branch__time_branch.csv`` (period → branch):
      tb_for_d[period1]       = [realized]
      tb_for_d[period1_upper] = [upper]
      tb_for_d[period1_lower] = [lower]
      tb_for_d[period1_mid]   = [mid]

* ``first_timesteps.csv`` (period → step), AFTER the parity test mutates
  to add a 2nd ts for ``period1_upper``:
      ts_for_d[period1]       = [t0001]
      ts_for_d[period1_upper] = [t0001, t0025]  ← mutated
      ts_for_d[period1_lower] = [t0001]
      ts_for_d[period1_mid]   = [t0001]

Expected Branch 2 fold (Branch 1 doesn't fire for ``downriver``)
----------------------------------------------------------------
For each child period d, Branch 2 sums over ``pe ∈ pe_for_d[d]``,
``tb ∈ tb_for_d[pe]``, ``ts ∈ ts_for_d[d]``.

Cell A — single-ts: ``(downriver, period1_lower, t0001)``
    pe_for_d[period1_lower] = [period1]
    tb_for_d[period1]       = [realized]
    ts_for_d[period1_lower] = [t0001]
    → fold = pbt[downriver, realized, t0001, t0001] = 0*10 + 0 + 1 = 1

Cell B — single-ts: ``(downriver, period1_mid, t0005)`` (t_idx=4)
    → fold = pbt[downriver, realized, t0001, t0005] = 0*10 + 4 + 1 = 5

Cell C — multi-ts (mutated): ``(downriver, period1_upper, t0001)`` (t_idx=0)
    pe_for_d[period1_upper] = [period1]
    tb_for_d[period1]       = [realized]
    ts_for_d[period1_upper] = [t0001, t0025]   (mutated)
    → fold = pbt[downriver, realized, t0001, t0001]
           + pbt[downriver, realized, t0025, t0001]
           = (0*10 + 0 + 1) + (1*10 + 0 + 1) = 1 + 11 = 12

Cell D — Branch 1 priority sanity: ``(hydro_reservoir, period1, t)`` is
covered by Branch 1 (already golden in Phase A test).  Branch 2 also
applies (``pe_for_d[period1] = [period1]``) but Branch 1 wins by priority.
Not asserted here — Phase A test already covers it.

Cell E — Note that for the realized period itself
``(downriver, period1, t0001)``, Branch 2 still fires because
``pe_for_d[period1] = [period1]``:
    → fold = pbt[downriver, realized, t0001, t0001] = 1

These hand-derived sums are the parity oracle for
``test_pbt_branch2_parent_period_fold``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from db_utils import db_to_json, json_to_db  # noqa: E402

from spinedb_api import DatabaseMapping, Map, to_database  # noqa: E402


SRC_JSON = REPO_ROOT / "tests/fixtures/stochastics_pbt_inflow.json"
DST_JSON = REPO_ROOT / "tests/fixtures/branch2_parent_period.json"
WORK_DIR = Path("/tmp/branch2_parent_period_build")
MIGRATE_SCRIPT = Path("/home/jkiviluo/sources/flextool/migrate_database.py")

BRANCHES = ("realized",)
TIME_STARTS = ("t0001", "t0025")
TIMES = tuple(f"t{i:04d}" for i in range(1, 49))


def _build_3d_map() -> Map:
    """branch -> time_start -> time -> float.

    Single branch (realized) × 2 time_starts × 48 times.
    value(ts_idx, t_idx) = ts_idx * 10 + t_idx + 1.
    """
    outer_indexes = list(BRANCHES)
    outer_values = []
    for _b_idx in range(len(BRANCHES)):
        ts_values = []
        for ts_idx in range(len(TIME_STARTS)):
            inner = Map(
                list(TIMES),
                [float(ts_idx * 10 + t_idx + 1)
                 for t_idx in range(len(TIMES))],
                index_name="time",
            )
            ts_values.append(inner)
        ts_map = Map(list(TIME_STARTS), ts_values, index_name="time_start")
        outer_values.append(ts_map)
    return Map(outer_indexes, outer_values, index_name="branch")


def _augment_db(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db:
        pvs = db.find_parameter_values(
            entity_class_name="node", parameter_definition_name="inflow"
        )
        target = None
        for p in pvs:
            if (
                p["entity_byname"] == ("downriver",)
                and p["alternative_name"] == "system"
            ):
                target = p
                break
        if target is None:
            raise RuntimeError(
                "downriver.inflow under alternative 'system' not found"
            )
        new_value = _build_3d_map()
        db_bytes, db_type = to_database(new_value)
        updated = target.update(value=db_bytes, type=db_type)
        if updated is None:
            raise RuntimeError("target.update returned None — value not changed")
        db.commit_session(
            "augment downriver.inflow → 3d_map (realized-only, multi-ts)"
        )
    print(f"replaced downriver.inflow with 3d_map (realized-only) in {db_path}")


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp_db = WORK_DIR / "augment_round_trip.sqlite"
    if tmp_db.exists():
        tmp_db.unlink()

    json_to_db(SRC_JSON, tmp_db)
    subprocess.run(
        [sys.executable, str(MIGRATE_SCRIPT), str(tmp_db)],
        check=True,
    )
    _augment_db(tmp_db)
    db_to_json(tmp_db, DST_JSON)
    print(f"wrote augmented JSON -> {DST_JSON}")


if __name__ == "__main__":
    main()
