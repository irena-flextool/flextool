"""Regression: rolling dispatch over a ``min_load_efficiency`` unit.

Bug (SECTION-1): a rolling-window dispatch whose sub-solves get a
runtime-synthesised name (``dispatch_fullYear_roll_roll_0`` etc., not a
real Spine solve) crashed in ``build_flextool`` with::

    feature 'min_load_efficiency' is active but data fields are not
    populated (None): ['p_section']

Cause: the synthetic-solve early-return in
``flextool/engine_polars/input.py::_apply_db_overrides`` skips the
``apply_derived_*`` passes, and ``apply_derived_c`` is the *sole*
producer of ``flex_data.p_section`` after the Δ.12-drop deleted the
``pdtProcess_section.csv`` loader seed.  ``process_min_load_eff`` (a
solve-agnostic Projection Param) stays populated, so the ``MINLOAD_EFF``
consistency check raised.  The fix wires ``p_section`` from the
solve-agnostic ``p_section_from_source`` inside that early-return,
mirroring the existing RESERVE-1 patch.

The ``MINLOAD_EFF`` check fires for *every* MLE unit on the synthetic
path, so a successful optimal solve of a rolling MLE scenario is a
sufficient correctness pin: it can only complete if ``p_section`` was
populated AND the section energy-balance term (``model.py`` online
cost / flow, gated on ``p_section is not None``) was built.  Pre-fix
this test raises ``ValueError(... ['p_section'])``; post-fix it solves
to optimality.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402


def _add_min_load_rolling_scenario(db_url: str) -> None:
    """Compose ``coal_min_load_fullYear_roll``: a ``min_load_efficiency``
    coal unit (alt ``coal_min_load``) dispatched on the fullYear timeline
    under rolling-window mode (alt ``dispatch_fullYear_roll``).

    ``dispatch_fullYear_roll`` puts the model into ``rolling_window`` mode,
    whose per-roll sub-solve names (``..._roll_N``) are NOT in Spine — the
    precondition for the synthetic-solve early-return that drops
    ``p_section``.  ``coal_min_load`` sets ``coal_plant.conversion_method =
    min_load_efficiency``, making ``process_min_load_eff`` non-empty so the
    ``MINLOAD_EFF`` check requires ``p_section``.
    """
    from spinedb_api import DatabaseMapping, import_data

    scenario = "coal_min_load_fullYear_roll"
    with DatabaseMapping(db_url) as db_map:
        _, errors = import_data(
            db_map,
            scenarios=[(scenario, False, "")],
            scenario_alternatives=[
                (scenario, "init", "west"),
                (scenario, "west", "coal"),
                (scenario, "coal", "coal_min_load"),
                (scenario, "coal_min_load", "wind"),
                (scenario, "wind", "fullYear"),
                (scenario, "fullYear", "dispatch_fullYear_roll"),
                (scenario, "dispatch_fullYear_roll", None),
            ],
        )
        if errors:
            raise RuntimeError(f"Import errors: {errors}")
        db_map.commit_session("Add min_load_efficiency rolling scenario")


@pytest.fixture(scope="module")
def min_load_rolling_db_url(
    tmp_path_factory: pytest.TempPathFactory,
) -> str:
    db_path = tmp_path_factory.mktemp("db_mle_roll") / "tests.sqlite"
    url = json_to_db(TEST_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    _add_min_load_rolling_scenario(url)
    return url


def _read_objective(workdir: Path) -> float:
    matches = list((workdir / "output_raw").glob("v_obj__*.parquet"))
    assert matches, f"No v_obj parquet in {workdir / 'output_raw'}"
    df = pd.read_parquet(matches[-1])
    return float(df["objective"].iloc[-1])


def test_rolling_min_load_efficiency_solves(
    min_load_rolling_db_url: str,
    test_solver_config_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """The rolling MLE cascade must complete to optimality.

    Pre-fix this raised ``ValueError(build_flextool: feature
    'min_load_efficiency' is active but data fields are not populated
    (None): ['p_section'])`` on the first rolling sub-solve.
    """
    workdir = tmp_path_factory.mktemp("mle_roll_run")
    os.chdir(workdir)

    steps = run_chain_from_db(
        min_load_rolling_db_url, "coal_min_load_fullYear_roll",
        work_folder=workdir,
    )

    assert steps, "cascade produced no solve steps"
    # Rolling expansion → more than one sub-solve (the synthetic path that
    # dropped p_section); every step must be optimal.
    assert len(steps) >= 2, (
        f"expected a rolling cascade with multiple sub-solves, "
        f"got {len(steps)} step(s): {list(steps)}"
    )
    for name, step in steps.items():
        assert step.optimal, f"sub-solve {name!r} did not reach optimality"

    # The model solved with the section term included → a finite,
    # positive objective (the MLE coal unit carries fuel cost).
    obj = _read_objective(workdir)
    assert obj > 0 and pd.notna(obj), f"unexpected objective {obj!r}"
