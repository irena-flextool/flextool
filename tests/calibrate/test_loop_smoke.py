"""End-to-end smoke test for the calibration loop skeleton (C1a).

Drives :func:`flextool.calibrate.run_calibration` for a single iteration on
the nested-invest fixture.  It proves the PLUMBING composes — the
calibration alternative is written, the baseline solve is launched, verified
``succeeded`` by P2's detector, the per-node residual read back, and (with
C1b sizing live) a positive adder raised on the shedding node.  The exact
sizing math is pinned in ``test_sizing.py``.

Marked ``solver``/``slow``: it launches a real ``cmd_run_flextool``
subprocess.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TESTS_DIR = Path(__file__).resolve().parent.parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.calibrate._db_alt import calib_alt_name  # noqa: E402
from flextool.calibrate._loop import (  # noqa: E402
    CalibConfig,
    CalibResult,
    run_calibration,
)

SCENARIO = "multi_fullYear_battery_nested_24h_invest_one_solve"

pytestmark = [pytest.mark.solver, pytest.mark.slow]


def _alt_exists(url: str, alt: str) -> bool:
    from spinedb_api import DatabaseMapping

    with DatabaseMapping(url) as db:
        return any(a.name == alt for a in db.query(db.alternative_sq).all())


def test_run_calibration_smoke(tmp_path: Path):
    url = json_to_db(_TESTS_DIR / "fixtures" / "tests.json", tmp_path / "c.sqlite")

    config = CalibConfig(
        iterations=1,
        slack_threshold_mwh=1.0,
        damping_first=0.5,
        damping_remaining=0.5,
        over_build_tightness=0.0,
        warm_start_cache_dir=tmp_path / "cache",
        work_dir=tmp_path / "work",
        out_root=tmp_path / "out",
        debug=False,
    )

    result = run_calibration(url, SCENARIO, config)

    # The calibration alternative exists after the run.
    assert _alt_exists(url, calib_alt_name(SCENARIO))

    # A result was produced with at least the baseline iteration recorded.
    assert isinstance(result, CalibResult)
    assert result.iterations_run >= 1
    assert result.trajectory, "at least the baseline iteration was recorded"

    # The baseline residual was read as a per-node float dict (assess_solve
    # would have raised CalibError before this if the solve had not
    # succeeded — so reaching here already proves it was assessed
    # ``succeeded``).
    baseline = result.trajectory[0]
    assert baseline.iteration == 0
    assert isinstance(baseline.residual, dict)
    assert all(isinstance(v, float) for v in baseline.residual.values())

    # C1b sizing: the baseline sheds (west), so compute_step raised a
    # positive adder on it (no over-build guard yet — that is C1c).
    assert result.final_adders, "C1b sizing should have raised at least one adder"
    assert all(v > 0.0 for v in result.final_adders.values())
    assert result.guard_flagged_nodes == []
