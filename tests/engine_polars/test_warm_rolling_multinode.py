"""Warm rolling reuse on a two-node model: warm == cold, roll by roll.

A rolled sub-solve whose LP has the same shape as the previous roll's
reuses the built :class:`polar_high.WarmProblem` and only pushes the new
``p_inflow`` into ``nodeBalance_eq`` — positionally.  With two balance
nodes carrying different demands, any misalignment between that vector
and the constraint's rows moves demand from one node to the other and
changes the roll objective.  Under polars 2.0 the lazy
``nodeBalance × dt`` cross join came out right-major, so every other
``(n, t)`` demand landed on the wrong node (west demand 0 every other
hour) — no existing warm/cold equivalence fixture has two nodes with
different time-series demand in a warm-reused rolling cascade, so none
caught it.  The unit-level guard is ``test_warm_rhs_alignment.py``.

Fixture: ``tests.json`` + an override alternative (built from JSON,
never a checked-in DB): ``coal_chp`` (1000 MW) serving ``west`` and
``heat``, rolled with a 24 h horizon / 12 h jump.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.solver

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = FLEXTOOL_ROOT / "tests"
BASE_FIXTURE_JSON = TESTS_DIR / "fixtures" / "tests.json"
ALT = "warm_roll_two_nodes"
SCENARIO = f"{ALT}_test"
CHAIN = ["init", "west", "coal_chp", "heat", ALT]


def _b64(x, kind: str) -> list:
    """Spine import-format packed value."""
    return [base64.b64encode(json.dumps(x).encode()).decode(), kind]


@pytest.fixture(scope="module")
def rolling_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    data = json.loads(BASE_FIXTURE_JSON.read_text())
    data["alternatives"].append([ALT, ""])
    for cls, ent, param, val in [
        ["solve", "y2020_2day_dispatch", "solve_mode",
         _b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon",
         _b64(24.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump",
         _b64(12.0, "float")],
        # Distinct constant demands: a node mix-up changes the objective.
        ["node", "west", "inflow", _b64(-1000.0, "float")],
        ["node", "heat", "inflow", _b64(0.0, "float")],
    ]:
        data["parameter_values"].append([cls, ent, param, val, ALT])
    data["scenarios"].append([SCENARIO, False, ""])
    for a, before in zip(CHAIN, CHAIN[1:] + [None]):
        data["scenario_alternatives"].append([SCENARIO, a, before])
    work = tmp_path_factory.mktemp(f"{ALT}_db")
    json_path = work / f"{ALT}.json"
    json_path.write_text(json.dumps(data))
    url = json_to_db(json_path, work / f"{ALT}.sqlite")
    migrate_database(url)
    return url


def test_warm_rolling_two_nodes_matches_cold(
        rolling_db_url: str, tmp_path_factory) -> None:
    from flextool.engine_polars import run_chain_from_db

    runs = {}
    for warm in (False, True):
        runs[warm] = run_chain_from_db(
            rolling_db_url, SCENARIO, warm=warm, keep_solutions=True,
            work_folder=tmp_path_factory.mktemp(f"w{int(warm)}"))
    cold, hot = runs[False], runs[True]
    rolls = [k for k in cold if "_roll_" in k]
    assert len(rolls) >= 3, f"expected roll sub-solves, got {list(cold)}"
    assert list(cold) == list(hot)
    # The positional p_inflow push is only exercised on warm-reused rolls.
    assert any(hot[k].warm_used for k in rolls), (
        "no roll reused the warm problem — the test no longer exercises "
        "the positional RHS push")
    for k in rolls:
        c = cold[k].solution.obj
        w = hot[k].solution.obj
        assert abs(c - w) / max(1.0, abs(c)) < 1e-9, (
            f"{k}: warm={w}, cold={c}, warm_used={hot[k].warm_used}")
