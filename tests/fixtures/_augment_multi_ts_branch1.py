"""Augment ``stochastics_pbt_inflow.json`` so that ``downriver`` carries a
multi-``time_start`` stochastic 3d_map ``inflow``.

Purpose
-------
Phase C, Fixture A.  Exercises Branch 1 (stochastic fold-in) of
``write_pdtNodeInflow`` when ``ts_for_d[d]`` has more than one entry, i.e.
when the outer fold ``for ts in period_time_first[d]`` actually iterates.

In the legacy preprocessing, ``first_timesteps.csv`` is always 1 row per
period (the first timestep), so the multi-``ts`` code path is structurally
reachable but never exercised by any real scenario.  Fixture A authors
multi-``ts`` data in the source AND the parity test mutates
``solve_data/first_timesteps.csv`` to add a second ``time_start`` for
``period1`` — this is the minimal trick that lets the multi-``ts`` fold
actually fire.

Structural augmentation
-----------------------
1. ``downriver`` is added to the ``add_stochastics`` group → enters
   ``stoch_node`` set → Branch 1 fires for it.
2. ``downriver``'s scalar ``inflow = -100`` is REPLACED by a 3d_map:

       branch ∈ {realized, upper}      # 2 branches
       time_start ∈ {t0001, t0025}     # 2 time_starts
       t ∈ {t0001..t0048}              # 48 timesteps (full coverage)

       value(b_idx, ts_idx, t_idx) = b_idx * 100 + ts_idx * 10 + t_idx

   with b_idx ∈ {0, 1} = (realized, upper), ts_idx ∈ {0, 1} = (t0001, t0025),
   t_idx = int(t[1:]) - 1.

Expected Branch 1 fold (computed by the parity test, with
``first_timesteps.csv`` mutated to include ``(period1, t0025)``)
---------------------------------------------------------------
For ``(downriver, period1, t)`` where ``period1`` is the realized period
with ``tb_for_d[period1] = [realized]``:

  fold = Σ_{tb ∈ {realized}} Σ_{ts ∈ {t0001, t0025}} pbt[downriver, tb, ts, t]
       = pbt[downriver, realized, t0001, t]
         + pbt[downriver, realized, t0025, t]

* ``t = t0001`` (t_idx=0):  (0+0+0) + (0+10+0) = 10
* ``t = t0005`` (t_idx=4):  (0+0+4) + (0+10+4) = 4 + 14 = 18
* ``t = t0048`` (t_idx=47): (0+0+47) + (0+10+47) = 47 + 57 = 104

For ``(downriver, period1_upper, t)`` with single ts (we DON'T mutate that
row) — ``tb_for_d[period1_upper] = [upper]`` and
``ts_for_d[period1_upper] = [t0001]``:

  fold = pbt[downriver, upper, t0001, t]

* ``t = t0001``: 1*100 + 0 + 0 = 100
* ``t = t0010``: 1*100 + 0 + 9 = 109

These hand-derived sums are the parity oracle in
``test_pbt_branch1_multi_time_start``.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from db_utils import db_to_json, json_to_db  # noqa: E402

from spinedb_api import DatabaseMapping, Map, import_data, to_database  # noqa: E402


SRC_JSON = REPO_ROOT / "tests/fixtures/stochastics_pbt_inflow.json"
DST_JSON = REPO_ROOT / "tests/fixtures/multi_ts_branch1.json"
WORK_DIR = Path("/tmp/multi_ts_branch1_build")
MIGRATE_SCRIPT = Path("/home/jkiviluo/sources/flextool/migrate_database.py")

BRANCHES = ("realized", "upper")
TIME_STARTS = ("t0001", "t0025")
TIMES = tuple(f"t{i:04d}" for i in range(1, 49))


def _build_3d_map() -> Map:
    """branch -> time_start -> time -> float using integer formula
    value(b_idx, ts_idx, t_idx) = b_idx * 100 + ts_idx * 10 + t_idx."""
    outer_indexes = list(BRANCHES)
    outer_values = []
    for b_idx in range(len(BRANCHES)):
        ts_values = []
        for ts_idx in range(len(TIME_STARTS)):
            inner = Map(
                list(TIMES),
                [float(b_idx * 100 + ts_idx * 10 + t_idx)
                 for t_idx in range(len(TIMES))],
                index_name="time",
            )
            ts_values.append(inner)
        ts_map = Map(list(TIME_STARTS), ts_values, index_name="time_start")
        outer_values.append(ts_map)
    return Map(outer_indexes, outer_values, index_name="branch")


def _augment_db(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve()}"
    # 1. Replace scalar downriver.inflow with the 3d_map.
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
        db.commit_session("augment downriver.inflow → 3d_map")
    print(f"replaced downriver.inflow with 3d_map in {db_path}")

    # 2. Add downriver to add_stochastics via import_data
    # (group__node entity with elements [add_stochastics, downriver]).
    with DatabaseMapping(url) as db:
        count, errors = import_data(
            db,
            entities=[("group__node", ("add_stochastics", "downriver"))],
        )
        if errors:
            raise RuntimeError(f"import_data errors: {errors}")
        db.commit_session("add downriver to add_stochastics")
    print(f"added downriver to add_stochastics group in {db_path}")


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp_db = WORK_DIR / "augment_round_trip.sqlite"
    if tmp_db.exists():
        tmp_db.unlink()

    # 1. JSON -> sqlite
    json_to_db(SRC_JSON, tmp_db)

    # 2. Migrate (idempotent if already current)
    subprocess.run(
        [sys.executable, str(MIGRATE_SCRIPT), str(tmp_db)],
        check=True,
    )

    # 3. Augment
    _augment_db(tmp_db)

    # 4. Export back to JSON
    db_to_json(tmp_db, DST_JSON)
    print(f"wrote augmented JSON -> {DST_JSON}")


if __name__ == "__main__":
    main()
