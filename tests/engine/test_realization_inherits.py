"""What a realization solve keeps of its stochastic solve: the units online of the binary units and the investments.

Linear units online (``startup_method = linear``) are a relaxation, not a schedule: the realization solve decides
them again with the realized data, their minimum up and down times and the previous realization's history (no
``online_fix_eq_linear`` row).  Binary units keep the stochastic solve's commitment (``online_fix_eq_integer``).
When a realization solve keeps nothing (no binary units online, no investments), note W-0509 (its units online are
all linear) or W-0588 (it has no units online) says so, once.

Fixture: ``rolling_separate_dispatch.json`` as the reported two-stage case (one forecast + the ``realized: yes``
branch at each roll start, 24-h jump, 48-h horizon, 10-h minimum up / down times; no investment), built from JSON
under the pytest tmp dir (``tests.engine.online.test_online_free_first_step``).
"""
from __future__ import annotations

import pytest

from tests._support.coded_warnings import capture_coded
from tests.engine._lp_rows import row_terms
from tests.engine._unit_db import build
from tests.engine.online.test_online_free_first_step import ROLLS, _enc, _reporter, _run_rolling

pytestmark = pytest.mark.solver

REALIZATION = "1week_rolling__realized_dispatch"


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _no_online(data: dict) -> dict:
    """The reported case without units online: no startup_method and no minimum_time_method."""
    data = _reporter("linear")(data)
    for r in data["parameter_values"]:
        if r[0] == "unit" and r[2] == "startup_method":
            r[3] = _enc("no_startup", "str")
        elif r[0] == "unit" and r[2] == "minimum_time_method":
            r[3] = _enc("none", "str")
    return data


def _realization_rolls(steps: dict) -> list:
    return [steps[n] for n in ROLLS if REALIZATION in n]


def test_linear_units_online_are_decided_again_in_the_realization(tmp_path_factory) -> None:
    """Linear units: every realization roll has no units-online pin (no p_fix_online, no online_fix_eq_linear row)
    but its own online_change / minimum time rows, all six solves are optimal, and W-0509 names both units once
    (nothing is pinned and the stochastic solve has no investments)."""
    with capture_coded() as got:
        steps = _run_rolling(tmp_path_factory.mktemp("inherits_linear"), _reporter("linear"))
    assert list(steps) == ROLLS and all(s.solution.optimal for s in steps.values())
    for step in _realization_rolls(steps):
        assert step.flex_data.p_fix_online is None
        pb = build(step.flex_data)
        assert not row_terms(pb, "online_fix_eq_linear")
        assert len(row_terms(pb, "minimum_uptime_linear")) == 48        # 2 units × 24 steps of the jump
    assert got.names("W-0509") == ["coal_plant", "gas_plant"]
    assert len(got.lines("W-0509")) == 1
    assert "'1week_rolling__realized_dispatch'" in got.lines("W-0509")[0] and "'1week_rolling'" in got.lines(
        "W-0509")[0]
    assert "W-0588" not in got.codes()


def test_binary_units_online_stay_pinned_and_no_note(tmp_path_factory) -> None:
    """Binary units: every realization roll pins both units at each of its 24 steps to the stochastic roll's
    commitment (online_fix_eq_integer, 48 rows), and neither note is given."""
    with capture_coded() as got:
        steps = _run_rolling(tmp_path_factory.mktemp("inherits_binary"), _reporter("integer_fix_in_uc"))
    assert all(s.solution.optimal for s in steps.values())
    for k, step in enumerate(_realization_rolls(steps)):
        assert len(row_terms(build(step.flex_data), "online_fix_eq_integer")) == 48, k
    assert "W-0509" not in got.codes() and "W-0588" not in got.codes()


def test_a_realization_without_units_online_or_investment_is_noted(tmp_path_factory) -> None:
    """No units online (no startup_method, no minimum_time_method) and no investment: the realization keeps nothing
    of its stochastic solve, W-0588 says so once (entityless), W-0509 not."""
    with capture_coded() as got:
        steps = _run_rolling(tmp_path_factory.mktemp("inherits_nothing"), _no_online)
    assert all(s.solution.optimal for s in steps.values())
    assert got.entityless("W-0588") and len(got.lines("W-0588")) == 1
    assert "W-0509" not in got.codes()
