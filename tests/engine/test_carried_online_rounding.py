"""The units online, start-ups and shut-downs a solve hands to the next one are rounded where they are produced
(``flex_data.build_handoff_from_solution``).  An integer column goes to the nearest integer.  The start-ups and
shut-downs of a unit with integer units online go to whole units together, so their difference stays the change of
the units online.  Any other column goes to exactly 0 where the solve could not tell it from 0: ``|x| ≤ tol_p / cf``,
the primal feasibility tolerance in the LP's own scale (``output._coarse_flows.var_zero_tolerance``).  A unit whose
carried units online went to 0 also carries its ramp flow as 0.  An override of these carriers is set to 0 near 0
too.  Solver noise that depends on the platform or the thread count then no longer reaches the next LP's near-zero
and whole-unit values: the realization pins (``p_fix_online``), the carried units online
(``p_roll_continue_online``), the carried minimum up / down time (``uptime_carry`` / ``downtime_carry``) and the
carried ramp flow (``p_roll_continue_flow``).

Acceptance 1 checks the rules on hand values.  Acceptance 2 runs ``rolling_separate_dispatch.json`` (the two-stage
case of ``tests.engine.test_realization_inherits``, with a loose ramp limit on both units so that a ramp flow is
carried) twice, once with noise added to what the producer reads, and compares the two runs.  The noise:
±3e-8 on every value of an integer column, and also on the start-ups / shut-downs of units with integer units
online (real start-ups of 1 or 2 units included); ±0.3 × the column's tolerance on the values of a continuous
column that are exactly 0; the same on the zero ramp flows of the two units.  A nonzero continuous value gets no
noise: it is carried as it is, so noise there would change the next LP whatever the rounding.
"""
from __future__ import annotations

import sys

import numpy as np
import polars as pl
import pytest

from flextool.engine.output._coarse_flows import var_is_integer, var_zero_tolerance
from flextool.engine.preprocess.flex_data import round_whole_unit_events, snap_carried_online
from tests.engine._unit_db import utf8
from tests.engine.online.test_online_free_first_step import ROLLS, _enc, _reporter, _run_rolling

UNITS = ("coal_plant", "gas_plant")


def test_integer_columns_round_to_the_nearest_integer() -> None:
    frame = pl.DataFrame({"p": list("abcde"), "value": [1.9999999, 2.0000004, 0.0000003, -0.0000002, 3.0]})
    out = snap_carried_online(frame, True, 1e-7)
    assert out["value"].to_list() == [2.0, 2.0, 0.0, 0.0, 3.0]
    assert out["p"].to_list() == list("abcde")


def test_continuous_columns_within_the_tolerance_become_zero() -> None:
    """Only ``|v| ≤ zero`` moves; 2e-7 is above a 1e-7 tolerance and is kept as it is."""
    frame = pl.DataFrame({"p": list("abcdef"), "value": [3e-10, -4e-9, 1e-7, 2e-7, 0.4, 0.4000003]})
    out = snap_carried_online(frame, False, 1e-7)
    assert out["value"].to_list() == [0.0, 0.0, 0.0, 2e-7, 0.4, 0.4000003]


def test_several_value_columns_are_snapped() -> None:
    frame = pl.DataFrame({"p": ["a", "b"], "startup": [2e-9, 1.0], "shutdown": [0.5, -1e-8]})
    out = snap_carried_online(frame, False, 1e-7, columns=("startup", "shutdown"))
    assert out.rows() == [("a", 0.0, 0.5), ("b", 1.0, 0.0)]


def test_whole_unit_events_keep_the_change_of_units_online() -> None:
    """su − sd is the integer change of units online Δ (+ noise).  Hand values: (1.5, 0.5), Δ = 1 → (2, 1) (rounding
    each half-to-even would give (2, 0), Δ = 2); (1 + 3e-8, 0) → (1, 0); (2e-8, 1 − 1e-8), Δ = −1 → (0, 1);
    (0.5, 0.5), Δ = 0 → (1, 1); (3e-8, −2e-8) → (0, 0)."""
    frame = pl.DataFrame({"p": list("abcde"), "startup": [1.5, 1 + 3e-8, 2e-8, 0.5, 3e-8],
                          "shutdown": [0.5, 0.0, 1 - 1e-8, 0.5, -2e-8]})
    out = round_whole_unit_events(frame)
    assert list(zip(out["startup"], out["shutdown"])) == [(2.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                                                           (0.0, 0.0)]


class _Var:
    def __init__(self, col_ids: list[int], integer: bool = False) -> None:
        self.frame = pl.DataFrame({"col_id": col_ids})
        self.integer = integer


class _Sol:
    def __init__(self, factors, tol=None) -> None:
        self._vars = {"v_online_linear": _Var([2, 3]), "v_online_integer": _Var([0, 1], integer=True)}
        self.layer2_col_factors = factors
        if tol is not None:
            self.flextool_primal_tolerance = tol


def test_zero_tolerance_is_the_feasibility_tolerance_in_the_scaled_column() -> None:
    """``tol_p / cf``: HiGHS' default 1e-7 without a recorded tolerance, the solve's with one; cf = 4 (column
    2 of v_online_linear) → 2.5e-8, the change of the scaled column ``cf · x`` is then 1e-7.  No Layer 2 → tol_p."""
    factors = np.array([1.0, 1.0, 4.0, 4.0])
    assert var_zero_tolerance(_Sol(None), "v_online_linear") == 1e-7
    assert var_zero_tolerance(_Sol(factors), "v_online_linear") == 2.5e-8
    assert var_zero_tolerance(_Sol(factors, 1e-6), "v_online_linear") == 2.5e-7
    assert var_zero_tolerance(_Sol(factors), "v_online_integer") == 1e-7
    assert var_is_integer(_Sol(factors), "v_online_integer") and not var_is_integer(_Sol(factors), "v_online_linear")


def test_an_override_of_carried_units_online_is_snapped_near_zero() -> None:
    """Overrides of ``roll_end_online`` / ``online_history`` come from no LP: HiGHS' default 1e-7 is the "near"."""
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import translate_overrides_to_provider

    class _Provider:
        def __init__(self) -> None:
            self.got: dict = {}

        def put(self, key, frame, source=None) -> None:
            self.got[key] = frame

    prov = _Provider()
    translate_overrides_to_provider({
        K.HANDOFF_ROLL_END_ONLINE: pl.DataFrame({"process": ["a", "b"], "value": [5e-8, 0.3]}),
        K.HANDOFF_ONLINE_HISTORY: pl.DataFrame({"process": ["a"], "period": ["p"], "step": ["t1"],
                                                "startup": [1e-9], "shutdown": [1.0], "duration": [1.0],
                                                "hours_to_end": [1.0]}),
    }, prov)
    assert prov.got[K.OVERRIDE_ROLL_END_ONLINE]["value"].to_list() == [0.0, 0.3]
    hist = prov.got[K.OVERRIDE_ONLINE_HISTORY]
    assert hist["startup"].to_list() == [0.0] and hist["shutdown"].to_list() == [1.0]


_ONLINE = ("v_online_linear", "v_online_integer")
_EVENTS = ("v_startup_linear", "v_startup_integer", "v_shutdown_linear", "v_shutdown_integer")
_PRODUCERS = ("build_handoff_from_solution", "_pull_online_metric", "_roll_end_flow")


def _noisy(sol, frame: "pl.DataFrame | None", name: str) -> "pl.DataFrame | None":
    """*frame* with the noise of the module docstring when the caller is the handoff producer (the results the
    solve writes keep the solver's values)."""
    caller = sys._getframe(2).f_code.co_name
    if (frame is None or frame.height == 0 or name not in (*_ONLINE, *_EVENTS, "v_flow")
            or caller not in _PRODUCERS or (name == "v_flow") != (caller == "_roll_end_flow")):
        return frame
    sign = pl.when(pl.int_range(pl.len()) % 2 == 0).then(1.0).otherwise(-1.0)
    whole = var_is_integer(sol, name) or (name in _EVENTS and name.endswith("_integer")
                                          and var_is_integer(sol, "v_online_integer"))
    if whole:
        return frame.with_columns(pl.col("value") + 3e-8 * sign)
    at_zero = pl.col("value") == 0.0
    if name == "v_flow":
        at_zero = at_zero & pl.col("p").cast(pl.Utf8).is_in(list(UNITS))
    noise = 0.3 * var_zero_tolerance(sol, "v_online_linear" if name == "v_flow" else name)
    return frame.with_columns(pl.when(at_zero).then(pl.col("value") + noise * sign).otherwise(pl.col("value"))
                              .alias("value"))


def _with_ramp(startup: str):
    """The reported case (``_reporter``) with a ramp limit of 1/min on both units' output (never binding with 1-h
    steps), so that each roll carries its last flow to the next, and min up / down times of 23 h (10 h in the
    reported case), so that the carried on-line history spans nearly the whole 24-h jump and holds whole-unit
    start-ups and shut-downs."""
    base = _reporter(startup)

    def patch(data: dict) -> dict:
        data = base(data)
        for r in data["parameter_values"]:
            if r[0] == "unit" and r[2] in ("min_uptime", "min_downtime"):
                r[3] = _enc(23.0, "float")
        for unit in UNITS:
            arc = [unit, "demand_node"]
            data["parameter_values"] += [
                ["unit__outputNode", arc, "ramp_method", _enc("ramp_limit", "str"), "system"],
                ["unit__outputNode", arc, "ramp_speed_up", _enc(1.0, "float"), "system"],
                ["unit__outputNode", arc, "ramp_speed_down", _enc(1.0, "float"), "system"]]
        return data
    return patch


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
                cache[key] = _run_rolling(tmp_path_factory.mktemp(f"pin_{startup}_{noise}"), _with_ramp(startup))
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
    """Every roll's realization pins, carried units online, carried minimum up / down time and carried ramp flow are
    the same with and without the noise, exactly.  The pins: both units at the 24 steps of each realization roll's
    jump (as in ``test_realization_inherits``), whole units for ``integer_fix_in_uc``.  Each roll after the first
    pair carries units online, an on-line history (with ``integer_fix_in_uc`` whole-unit start-ups and
    shut-downs, which got noise) and a ramp flow; a
    carried ramp flow of a unit carried offline is 0."""
    clean, noisy = runs(startup, False), runs(startup, True)
    assert list(clean) == ROLLS and list(noisy) == ROLLS
    seen = {"p_fix_online": 0, "p_roll_continue_online": 0, "online_startup_history": 0, "p_roll_continue_flow": 0}
    started = stopped = 0.0
    for roll in ROLLS:
        for field in (*seen, "uptime_carry", "downtime_carry"):
            a, b = _frame(clean[roll], field), _frame(noisy[roll], field)
            assert (a is None) == (b is None), (roll, field)
            if a is None:
                continue
            assert a.equals(b), (roll, field)
            seen[field] = seen.get(field, 0) + 1
            if field == "online_startup_history":
                started = max(started, float(a["startup"].max()))
                stopped = max(stopped, float(a["shutdown"].max()))
            if field == "p_fix_online":
                assert a.height == 48, roll
                if startup == "integer_fix_in_uc":
                    assert all(float(v).is_integer() for v in a["value"]), roll
        online, flow = _frame(noisy[roll], "p_roll_continue_online"), _frame(noisy[roll], "p_roll_continue_flow")
        if online is not None and flow is not None:
            off = set(online.filter(pl.col("value") == 0.0)["p"])
            assert all(v == 0.0 for p, v in zip(flow["p"], flow["value"]) if p in off), roll
    assert seen["p_fix_online"] == len(ROLLS) // 2
    assert seen["p_roll_continue_online"] == seen["p_roll_continue_flow"] == len(ROLLS) - 2
    assert seen["online_startup_history"] == len(ROLLS) - 2
    if startup == "integer_fix_in_uc":
        assert started >= 1.0 and stopped >= 1.0
