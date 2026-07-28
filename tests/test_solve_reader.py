"""Tests for :func:`flextool.gui.solve_reader.read_scenario_solves`.

The Calibrate-investments GUI enumerates the solves a scenario runs and flags
which are investment solves (non-empty ``invest_periods`` Array).  These tests
assert the ordered enumeration, the invest/dispatch flag, and the loud error on
an unknown scenario.

The fixture is built from the FlexTool JSON schema (CLAUDE.md invariant #3 —
never read a checked-in ``.sqlite``): a model whose ``solves`` Array lists an
investment solve (non-empty ``invest_periods``) followed by a dispatch solve
(no ``invest_periods``), all pinned to one alternative in one scenario.
"""

from __future__ import annotations

import pytest
from spinedb_api import Array, DatabaseMapping, import_data

from flextool._resources import package_data_path
from flextool.gui.solve_reader import read_scenario_solves
from flextool.update_flextool import initialize_database

SCENARIO = "calib_scen"
ALT = "calib_base"
MODEL = "flexModel"
INVEST_SOLVE = "invest_solve"
DISPATCH_SOLVE = "dispatch_solve"


def _build_db(db_path: str) -> str:
    """Schema-complete DB: model.solves = [invest, dispatch]; one scenario."""
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"

    entities = [
        ("model", MODEL, None),
        ("solve", INVEST_SOLVE, None),
        ("solve", DISPATCH_SOLVE, None),
    ]
    parameter_values = [
        # Ordered solve list the scenario runs.
        ("model", MODEL, "solves", Array([INVEST_SOLVE, DISPATCH_SOLVE]), ALT),
        # Investment solve: non-empty invest_periods.
        ("solve", INVEST_SOLVE, "invest_periods", Array(["y2050"]), ALT),
        # Dispatch solve: invest_periods intentionally absent.
    ]

    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            alternatives=[(ALT, "calibrate fixture")],
            scenarios=[(SCENARIO, True, "calibrate scenario")],
            scenario_alternatives=[(SCENARIO, ALT)],
            entities=entities,
            parameter_values=parameter_values,
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("calibrate fixture")
    return url


def test_enumerates_solves_in_order_with_invest_flags(tmp_path):
    url = _build_db(str(tmp_path / "calib.sqlite"))
    solves = read_scenario_solves(url, SCENARIO)

    # Both solves returned, in the scenario's model.solves order.
    assert [s.name for s in solves] == [INVEST_SOLVE, DISPATCH_SOLVE]
    # Invest solve flagged True; dispatch solve (absent invest_periods) False.
    assert solves[0].has_invest_periods is True
    assert solves[1].has_invest_periods is False


def test_missing_scenario_raises(tmp_path):
    url = _build_db(str(tmp_path / "calib.sqlite"))
    with pytest.raises(ValueError, match="no_such_scenario"):
        read_scenario_solves(url, "no_such_scenario")
