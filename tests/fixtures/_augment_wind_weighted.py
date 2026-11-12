"""Add the scenario ``wind_weighted`` to ``tests/fixtures/tests.json``: the ``wind`` system with the timestep weights
of ``base_weighted``.

Chain ``init → west → wind → weighted_2day`` (all existing alternatives).  ``wind_plant`` (1000 MW, profile
``wind_profile``) is the only unit, with no storage and no inter-step constraint, so each step's optimum is flow =
min(potential, demand) whatever the weights; curtailing is free.  The weights 0.5 / 1 / 1.5 / 2 per 12 steps of
t0001-t0048 make the period shares differ from the unweighted ``wind`` scenario (hand values in
``tests/expected/REGEN_LOG.md``): curtailed VRE of potential VRE 0.073269, VRE share of demand 0.946263, objective
273.7062 M CUR.

Append-only: one scenario and its chain, nothing else.  Idempotent: an existing ``wind_weighted`` keeps its chain.

After running it, regenerate the examples database JSON::

    python tests/fixtures/_augment_wind_weighted.py
    python -m flextool.update_flextool.generate_canonical templates_examples
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from db_utils import db_to_json, json_to_db  # noqa: E402

from flextool.update_flextool.db_handle import open_database_mapping  # noqa: E402
from spinedb_api import import_data  # noqa: E402

TESTS_JSON = REPO_ROOT / "tests" / "fixtures" / "tests.json"

SCENARIO = "wind_weighted"
CHAIN = ["init", "west", "wind", "weighted_2day"]

# What this script adds to tests.json, for builders that start from tests.json and must not carry it
# (see _augment_added.py).  The chain uses existing alternatives only.
ADDED_ALTERNATIVES: tuple[str, ...] = ()
ADDED_ENTITIES: tuple[tuple[str, tuple[str, ...]], ...] = ()
ADDED_SCENARIOS: tuple[str, ...] = (SCENARIO,)


def _import(db_path: Path) -> None:
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    with open_database_mapping(url) as db:
        if not any(s["name"] == SCENARIO for s in db.find_scenarios()):
            rows = [[SCENARIO, alt, CHAIN[i + 1] if i + 1 < len(CHAIN) else None] for i, alt in enumerate(CHAIN)]
            count, errors = import_data(db, scenarios=[[SCENARIO, False, ""]], scenario_alternatives=rows,
                                        on_conflict="keep")
            if errors:
                raise RuntimeError(f"import errors: {errors}")
            db.commit_session("wind_weighted scenario")
            print(f"imported {count} items")
        got = [r["alternative_name"] for r in sorted(db.find_scenario_alternatives(scenario_name=SCENARIO),
                                                     key=lambda r: r["rank"])]
        if got != CHAIN:
            raise RuntimeError(f"scenario {SCENARIO}: chain {got} != {CHAIN}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_db = Path(tmp) / "tests_augment.sqlite"
        json_to_db(TESTS_JSON, tmp_db)
        _import(tmp_db)
        db_to_json(tmp_db, TESTS_JSON)
    print(f"wrote {TESTS_JSON}")


if __name__ == "__main__":
    main()
