"""Each solve's autoscaler mode is its own: ``--scaling`` / ``FLEXTOOL_SCALING`` > ``solve.scaling`` > ``full``.

A Benders solve applies the mode to its region subproblems (``_benders._apply_region_autoscale``), so the tests
record the :class:`ScalingConfig` the region build receives and stop there; nothing is solved by Benders.  The
databases are built from the ``lh2_three_region_trade_invest`` JSON fixture plus an override alternative.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from polar_high.autoscale import ScalingMode

from flextool.engine.solve import _benders
from tests.engine.test_solver_options_wiring import _lp_snapshot

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
BASE = "lh2_three_region_trade_invest"
BENDERS_SOLVE = "lh2_trade_invest"
MONOLITHIC_SOLVE = "lh2_week"


class _Stop(Exception):
    pass


def _db(tmp_path: Path, values: list) -> str:
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    data = json.loads((FIXTURES / f"{BASE}.json").read_text())
    data = _lp_snapshot()._overlay(data, name="custom", base=BASE, values=values)
    (tmp_path / "f.json").write_text(json.dumps(data))
    url = json_to_db(tmp_path / "f.json", tmp_path / "f.sqlite")
    migrate_database(url)
    return url


def _region_scaling_mode(url: str, tmp_path: Path, monkeypatch) -> ScalingMode:
    """Run the chain until the Benders solve builds its first region; return that region's autoscaler mode."""
    from flextool.engine import run_chain_from_db

    seen: list = []

    def record(pb, cfg, region_name):
        seen.append(cfg)
        raise _Stop

    monkeypatch.setattr(_benders, "_apply_region_autoscale", record)
    with pytest.raises(_Stop):
        run_chain_from_db(input_db_url=url, scenario_name="custom", work_folder=tmp_path / "work")
    assert seen, "the Benders solve built no region"
    return seen[0].mode


@pytest.fixture
def no_scaling_env(monkeypatch):
    monkeypatch.delenv("FLEXTOOL_SCALING", raising=False)
    monkeypatch.delenv("FLEXTOOL_USER_BOUND_SCALE", raising=False)


def test_benders_solve_uses_its_scaling(tmp_path, monkeypatch, no_scaling_env) -> None:
    url = _db(tmp_path, [["solve", BENDERS_SOLVE, "scaling", "off"]])
    assert _region_scaling_mode(url, tmp_path, monkeypatch) is ScalingMode.OFF


def test_benders_solve_does_not_inherit_the_previous_solves_scaling(tmp_path, monkeypatch, no_scaling_env) -> None:
    """A monolithic solve with ``scaling = off`` runs first; the Benders solve sets no scaling, so it gets ``full``."""
    url = _db(tmp_path, [
        ["model", "flexTool", "solves", {"type": "array", "value_type": "str",
                                         "data": [MONOLITHIC_SOLVE, BENDERS_SOLVE]}],
        ["solve", MONOLITHIC_SOLVE, "scaling", "off"],
    ])
    assert _region_scaling_mode(url, tmp_path, monkeypatch) is ScalingMode.FULL
    assert "FLEXTOOL_SCALING" not in os.environ


def test_environment_scaling_wins_over_the_solves(tmp_path, monkeypatch, no_scaling_env) -> None:
    url = _db(tmp_path, [["solve", BENDERS_SOLVE, "scaling", "off"]])
    monkeypatch.setenv("FLEXTOOL_SCALING", "basic")
    assert _region_scaling_mode(url, tmp_path, monkeypatch) is ScalingMode.BASIC
