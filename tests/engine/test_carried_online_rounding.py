"""The units online, start-ups and shut-downs a solve hands to the next one are rounded where they are produced
(``flex_data.build_handoff_from_solution``): the column values of an integer variable to the nearest integer, those
of a continuous variable below ``CARRIED_ONLINE_ZERO`` (1e-6 units of unitsize) to exactly 0.  Solver noise that
depends on the platform or the thread count then does not reach the next LP: the realization pins
(``p_fix_online``), the carried units online (``p_roll_continue_online``) and the carried minimum up / down time
(``uptime_carry`` / ``downtime_carry``).

Acceptance 1 checks the rule on hand values.  Acceptance 2 adds ±3e-8 to every units-online / start-up / shut-down
value the producer reads on ``rolling_separate_dispatch.json`` (the two-stage case of
``tests.engine.test_realization_inherits``) and compares with the run without it: with ``integer_fix_in_uc`` the
pins and the carried units online are the same, exactly (the integer columns round back); with
``linear_fix_in_uc`` a pin or carried value that is 0 without the noise is exactly 0 with it, and the carried
minimum up / down time has the same cells.
"""
from __future__ import annotations

import sys

import polars as pl
import pytest

from flextool.engine.output._coarse_flows import var_is_integer
from flextool.engine.preprocess.flex_data import CARRIED_ONLINE_ZERO, snap_carried_online
from tests.engine._unit_db import utf8
from tests.engine.online.test_online_free_first_step import ROLLS, _reporter, _run_rolling


def test_integer_columns_round_to_the_nearest_integer() -> None:
    frame = pl.DataFrame({"p": list("abcde"), "value": [1.9999999, 2.0000004, 0.0000003, -0.0000002, 3.0]})
    out = snap_carried_online(frame, integer=True)
    assert out["value"].to_list() == [2.0, 2.0, 0.0, 0.0, 3.0]
    assert out["p"].to_list() == list("abcde")


def test_continuous_columns_below_the_threshold_become_zero() -> None:
    assert CARRIED_ONLINE_ZERO == 1e-6
    frame = pl.DataFrame({"p": list("abcdef"), "value": [3e-10, -4e-9, 9.9e-7, 1e-6, 0.4, 0.4000003]})
    out = snap_carried_online(frame, integer=False)
    assert out["value"].to_list() == [0.0, 0.0, 0.0, 1e-6, 0.4, 0.4000003]


def test_several_value_columns_are_snapped() -> None:
    frame = pl.DataFrame({"p": ["a", "b"], "startup": [2e-9, 1.0], "shutdown": [0.5, -1e-8]})
    out = snap_carried_online(frame, integer=False, columns=("startup", "shutdown"))
    assert out.rows() == [("a", 0.0, 0.5), ("b", 1.0, 0.0)]


_NOISY = ("v_online_linear", "v_online_integer", "v_startup_linear", "v_startup_integer", "v_shutdown_linear",
          "v_shutdown_integer")
_PRODUCERS = ("build_handoff_from_solution", "_pull_online_metric")


def _noisy(sol, frame: "pl.DataFrame | None", name: str) -> "pl.DataFrame | None":
    """*frame* with ±3e-8 (alternating by row) added to ``value`` when *name* is a units-online variable and the
    caller is the handoff producer (the results the solve writes keep the solver's values): on every row of an
    integer column, on the rows below 1e-6 of a continuous one (a continuous nonzero value is carried as it is, so
    noise there would change the next LP whatever the rounding)."""
    if (frame is None or name not in _NOISY or frame.height == 0
            or sys._getframe(2).f_code.co_name not in _PRODUCERS):
        return frame
    sign = pl.when(pl.int_range(pl.len()) % 2 == 0).then(1.0).otherwise(-1.0)
    near_zero = pl.lit(True) if var_is_integer(sol, name) else pl.col("value").abs() < 1e-6
    return frame.with_columns(pl.when(near_zero).then(pl.col("value") + 3e-8 * sign).otherwise(pl.col("value"))
                              .alias("value"))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """``{(startup_method, noise): steps}``, each run once (strict autoscale, as the other two-stage tests)."""
    from flextool.engine.output import _coarse_flows

    cache: dict = {}

    def get(startup: str, noise: bool):
        key = (startup, noise)
        if key not in cache:
            mp = pytest.MonkeyPatch()
            mp.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
            if noise:
                values, values_at = _coarse_flows.var_values, _coarse_flows.var_values_at
                mp.setattr(_coarse_flows, "var_values",
                           lambda sol, fd, name: _noisy(sol, values(sol, fd, name), name))
                mp.setattr(_coarse_flows, "var_values_at",
                           lambda sol, fd, name, d, t: _noisy(sol, values_at(sol, fd, name, d, t), name))
            try:
                cache[key] = _run_rolling(tmp_path_factory.mktemp(f"pin_{startup}_{noise}"), _reporter(startup))
            finally:
                mp.undo()
        return cache[key]
    return get


def _frame(step, field: str) -> "pl.DataFrame | None":
    value = getattr(step.flex_data, field, None)
    if value is None:
        return None
    frame = value.frame if hasattr(value, "frame") else value
    keys = [c for c in frame.columns if c != "value"]
    return utf8(frame).sort(keys)


@pytest.mark.solver
@pytest.mark.parametrize("startup", ["integer_fix_in_uc", "linear_fix_in_uc"])
def test_carried_units_online_are_the_same_with_solver_noise(runs, startup) -> None:
    """Every roll's realization pins, carried units online and carried minimum up / down time are the same with and
    without the noise, exactly.  The pins: both units at the 24 steps of each realization roll's jump (as in
    ``test_realization_inherits``); with ``integer_fix_in_uc`` whole units."""
    clean, noisy = runs(startup, False), runs(startup, True)
    assert list(clean) == ROLLS and list(noisy) == ROLLS
    pinned = carried = 0
    for roll in ROLLS:
        for field in ("p_fix_online", "p_roll_continue_online", "uptime_carry", "downtime_carry"):
            a, b = _frame(clean[roll], field), _frame(noisy[roll], field)
            assert (a is None) == (b is None), (roll, field)
            if a is None:
                continue
            assert a.equals(b), (roll, field)
            if field == "p_fix_online":
                pinned += 1
                assert a.height == 48, roll
                if startup == "integer_fix_in_uc":
                    assert all(float(v).is_integer() for v in a["value"]), roll
            carried += field == "p_roll_continue_online"
    assert pinned == len(ROLLS) // 2 and carried == len(ROLLS) - 2
