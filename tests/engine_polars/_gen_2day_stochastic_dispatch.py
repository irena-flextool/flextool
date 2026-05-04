"""Generate the 2_day_stochastic_dispatch parity fixtures.

We produce two fixture variants of the same flextool scenario:

* ``work_2day_stochastic_dispatch_no_storage`` — runs the
  ``2_day_stochastic_dispatch`` scenario without a storage entity (all
  hydro_reservoir / hydro_plant rows zeroed out).  Pure pdt_branch_weight
  test; storage non-anticipativity is moot.
* ``work_2day_stochastic_dispatch_full_storage`` — full topology
  (hydro_reservoir storage + hydro_plant + wind_plant + gas_plant +
  coal_plant + demand_node).  Exercises both pdt_branch_weight and
  the ``non_anticipativity_storage_use`` constraint.

Both fixtures source from ``tests/fixtures/stochastics.json`` — the
canonical flextool stochastic-feature DB dump.

Usage::

    ~/venv-spi/bin/python tests/_gen_2day_stochastic_dispatch.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# REPO_ROOT still points at the live flextool checkout because we use
# flextool's Python packages (FlexToolRunner, migrate_database) plus its
# bin/ + .mod files at run time.  Only data fixtures + json_to_db are
# vendored into flexpy_spike.
REPO_ROOT  = Path("/home/jkiviluo/sources/flextool")
TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = TESTS_DIR / "data"
FIXTURES   = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))

from _db_utils import json_to_db                                      # noqa: E402  vendored
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner    # noqa: E402
from flextool.update_flextool.db_migration import migrate_database    # noqa: E402


SCENARIO = "2_day_stochastic_dispatch"
NO_STORAGE_SCENARIO = "2_day_stochastic_dispatch_no_storage"


def _build_db(workdir: Path) -> str:
    """Build a stochastics.sqlite from the JSON dump and migrate to the
    current FLEXTOOL_DB_VERSION."""
    db_path = workdir / "stochastics.sqlite"
    if db_path.exists():
        db_path.unlink()
    src_json = FIXTURES / "stochastics.json"
    url = json_to_db(src_json, db_path)
    migrate_database(url)
    return url


def _add_no_storage_scenario(db_url: str) -> None:
    """Add a no-storage variant of 2_day_stochastic_dispatch.

    Strips the hydro_reservoir storage by setting ``node_type=commodity``.
    The resulting scenario keeps all branches and pbt_profile data but
    drops storage-related coupling, so non_anticipativity_storage_use is
    inactive and the only delta vs deterministic is the pdt_branch_weight
    factor on dispatch terms.

    Spine's ``import_data`` requires three separate calls / commits here:
    (1) the new alternative + scenario, (2) the scenario_alternative
    rows (which can only attach once the parent scenario exists), and
    (3) the parameter override (which can only attach once the
    alternative exists).
    """
    from spinedb_api import DatabaseMapping, import_data

    with DatabaseMapping(db_url) as db_map:
        existing_alts = []
        for sa in db_map.query(db_map.scenario_alternative_sq):
            scen = db_map.query(db_map.scenario_sq).filter_by(id=sa.scenario_id).one()
            if scen.name == SCENARIO:
                alt = db_map.query(db_map.alternative_sq).filter_by(id=sa.alternative_id).one()
                existing_alts.append((alt.name, sa.rank))
        existing_alts.sort(key=lambda x: x[1])

        n1, e1 = import_data(
            db_map,
            alternatives=[("no_storage_override",)],
            scenarios=[(NO_STORAGE_SCENARIO, True)],
        )
        if e1:
            print("import_data step1 errors:", e1)
        db_map.commit_session("step1: add no-storage scenario + alternative")

    with DatabaseMapping(db_url) as db_map:
        sa_data = [(NO_STORAGE_SCENARIO, alt_name) for alt_name, _ in existing_alts]
        sa_data.append((NO_STORAGE_SCENARIO, "no_storage_override"))
        n2, e2 = import_data(db_map, scenario_alternatives=sa_data)
        if e2:
            print("import_data step2 errors:", e2)
        db_map.commit_session("step2: add scenario_alternatives")

    with DatabaseMapping(db_url) as db_map:
        n3, e3 = import_data(
            db_map,
            parameter_values=[
                # Override hydro_reservoir.node_type → 'commodity' to
                # disable storage (the .mod's storage feature is gated
                # on node_type=='storage').
                ("node", "hydro_reservoir", "node_type", "commodity",
                 "no_storage_override"),
            ],
        )
        if e3:
            print("import_data step3 errors:", e3)
        db_map.commit_session("step3: hydro_reservoir.node_type = commodity")


def _run(workdir: Path, db_url: str, scenario: str) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        runner = FlexToolRunner(
            input_db_url   = db_url,
            scenario_name  = scenario,
            flextool_dir   = REPO_ROOT / "flextool",
            bin_dir        = REPO_ROOT / "bin",
            root_dir       = workdir,
            work_folder    = workdir,
        )
        runner.write_input(db_url, scenario)
        rc = runner.run_model()
    finally:
        os.chdir(prev_cwd)
    if rc != 0:
        raise SystemExit(f"flextool failed for {scenario!r}: rc={rc}")


def generate(target_full: Path | None = None,
             target_no_storage: Path | None = None) -> tuple[Path, Path]:
    target_full = target_full or (DATA_DIR / "work_2day_stochastic_dispatch_full_storage")
    target_no_storage = target_no_storage or (
        DATA_DIR / "work_2day_stochastic_dispatch_no_storage")

    # Wipe + recreate target dirs.
    for tgt in (target_full, target_no_storage):
        if tgt.exists():
            shutil.rmtree(tgt)
        tgt.mkdir(parents=True, exist_ok=True)

    # Use a single sqlite for both scenarios so we don't migrate twice.
    db_dir = TESTS_DIR / "_gen_tmp_2day_stoch"
    if db_dir.exists():
        shutil.rmtree(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    try:
        db_url = _build_db(db_dir)
        _add_no_storage_scenario(db_url)

        _run(target_full, db_url, SCENARIO)
        _run(target_no_storage, db_url, NO_STORAGE_SCENARIO)

        # Copy the migrated sqlite into the per-fixture workdirs so
        # SpineDbReader can be instantiated against it for parity tests.
        src_sqlite = db_dir / "stochastics.sqlite"
        if src_sqlite.exists():
            shutil.copy(src_sqlite, target_full / "tests.sqlite")
            shutil.copy(src_sqlite, target_no_storage / "tests.sqlite")
    finally:
        if db_dir.exists():
            shutil.rmtree(db_dir, ignore_errors=True)

    print(f"\nfull-storage:    generated under {target_full}")
    print(f"no-storage:      generated under {target_no_storage}")
    return target_full, target_no_storage


if __name__ == "__main__":
    generate()
