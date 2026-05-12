"""Augment ``stochastics.json`` with a 3d_map ``inflow`` on ``hydro_reservoir``.

The original ``stochastics.json`` fixture carries scalar inflow on
``hydro_reservoir``. To exercise Branch 1 (stochastic fold-in) of
legacy ``write_pdtNodeInflow`` we need an inflow indexed by
``[branch, time_start, time]``.

Workflow
--------
1. Load ``tests/fixtures/stochastics.json``.
2. Materialize it as a temporary SQLite via ``tests/db_utils.json_to_db``.
3. Migrate the temp SQLite to the current flextool schema so the legacy
   migration is not a downstream surprise.
4. Open the migrated DB with ``spinedb_api``, swap the scalar
   ``hydro_reservoir.inflow`` under alternative ``system`` for a 3-level
   ``Map`` (branch -> time_start -> time -> float).
5. Export back to JSON via ``tests/db_utils.db_to_json``.

The result is ``tests/fixtures/stochastics_pbt_inflow.json``, ready for
the legacy parity-oracle run.

Branch choices
--------------
The ``2_day_stochastic_dispatch`` scenario builds branches
``realized`` / ``upper`` / ``lower`` / ``mid`` (see
``solve_branch__time_branch.csv`` emitted by a legacy run).  The model
check at ``flextool.mod:1867`` requires the inflow series to carry a
``(branch=realized, ts=t0001, t=t0001)`` row because all four
``period__time_first`` rows resolve via ``period__branch`` -> realized
parent.  We therefore include ``realized`` plus the three stochastic
branches.

Time grid
---------
``period_time_first`` for both the realized period and each stochastic
branch is ``t0001`` (see ``first_timesteps.csv`` from a legacy run), so
``time_start`` collapses to a single value.  ``time`` covers ``t0001`` ..
``t0048`` (48 hourly timesteps).

Values
------
Deterministic, reproducible, branch-dependent::

    value(branch_idx, t_idx) = 100.0 + 10.0 * branch_idx
                               + math.sin(2 * pi * t_idx / 48)

with ``branch_idx`` in {0, 1, 2, 3} ordered as
(realized, upper, lower, mid) and ``t_idx`` in 0 .. 47.  This produces
192 distinct, non-trivial values (4 branches x 1 time_start x 48 times).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from db_utils import db_to_json, json_to_db  # noqa: E402

from spinedb_api import DatabaseMapping, Map, to_database  # noqa: E402


SRC_JSON = REPO_ROOT / "tests/fixtures/stochastics.json"
DST_JSON = REPO_ROOT / "tests/fixtures/stochastics_pbt_inflow.json"
WORK_DIR = Path("/tmp/pbt_fixture_build")
MIGRATE_SCRIPT = Path("/home/jkiviluo/sources/flextool/migrate_database.py")

BRANCHES = ("realized", "upper", "lower", "mid")
TIME_START = "t0001"
TIMES = tuple(f"t{i:04d}" for i in range(1, 49))


def _build_3d_map() -> Map:
    """branch -> time_start -> time -> float."""
    outer_indexes = list(BRANCHES)
    outer_values = []
    for b_idx, _ in enumerate(BRANCHES):
        # inner-most: time -> value
        time_values = [
            100.0 + 10.0 * b_idx + math.sin(2.0 * math.pi * t_idx / 48.0)
            for t_idx in range(len(TIMES))
        ]
        innermost = Map(list(TIMES), time_values, index_name="time")
        mid = Map([TIME_START], [innermost], index_name="time_start")
        outer_values.append(mid)
    return Map(outer_indexes, outer_values, index_name="branch")


def _augment_inflow(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve()}"
    with DatabaseMapping(url) as db:
        pvs = db.find_parameter_values(
            entity_class_name="node", parameter_definition_name="inflow"
        )
        target = None
        for p in pvs:
            if (
                p["entity_byname"] == ("hydro_reservoir",)
                and p["alternative_name"] == "system"
            ):
                target = p
                break
        if target is None:
            raise RuntimeError(
                "hydro_reservoir.inflow under alternative 'system' not found"
            )
        new_value = _build_3d_map()
        db_bytes, db_type = to_database(new_value)
        updated = target.update(value=db_bytes, type=db_type)
        if updated is None:
            raise RuntimeError("target.update returned None — value not changed")
        db.commit_session("augment hydro_reservoir.inflow to 3d_map")
    print(f"augmented inflow on hydro_reservoir in {db_path}")


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    tmp_db = WORK_DIR / "augment_round_trip.sqlite"
    if tmp_db.exists():
        tmp_db.unlink()

    # 1. JSON -> sqlite
    json_to_db(SRC_JSON, tmp_db)

    # 2. Migrate to current schema (so spinedb_api can operate on it,
    # and so the JSON we export back is current-schema as well).
    subprocess.run(
        [sys.executable, str(MIGRATE_SCRIPT), str(tmp_db)],
        check=True,
    )

    # 3. Swap scalar -> 3d_map
    _augment_inflow(tmp_db)

    # 4. Export back to JSON
    db_to_json(tmp_db, DST_JSON)
    print(f"wrote augmented JSON -> {DST_JSON}")


if __name__ == "__main__":
    main()
