"""A presence set comes from the values the solve uses, not from every value of the parameter.

``process_minload`` (the units of the ``minFlow_minload`` floor) is the units whose solve-resolved
``p_min_load`` is above 0 on some (d, t) of the solve.  A reserve ``(r, ud)`` has a reservation requirement
when its ``reservation``, summed over its groups and the solve's (d, t), is not 0 (the emit's rule).  A
period map whose only key is a period the solve does not have is then the same as no value: each test
compares that case with its twin that has a 0 (or no) value.

Fixtures: ``tests.json`` + an override alternative, built from JSON.  ``y2020_2day_dispatch`` solves the
period p2020 only.
"""
from __future__ import annotations

import pytest

from tests.engine._unit_db import (
    CHP, PLANT, PLANT_CHAIN, b64, build, cstr_over, make_db, run, solve, utf8,
)

OTHER_PERIOD = "p2030"


def period_map(values: dict[str, float]) -> list:
    return b64({"index_type": "str", "index_name": "period", "data": [[k, v] for k, v in values.items()]}, "map")


def _minload_units(step, kind: str = "linear") -> set[str]:
    over = cstr_over(build(step.flex_data), f"minFlow_minload_{kind}")
    return set() if over is None else set(over["p"].to_list())


def test_min_load_only_in_another_period_equals_zero_min_load(tmp_path_factory) -> None:
    """``min_load = {p2030: 0.4}`` on the p2020 solve gives no floor: the same objective and the same (empty)
    ``minFlow_minload`` rows as ``min_load = 0``.  Before, the unit was a floor unit without a ``p_min_load``
    row and the build stopped ("feature 'min_load' is active")."""
    def case(alt, value):
        return solve(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN,
                     values=[["unit", PLANT, "startup_method", b64("linear", "str")],
                             ["unit", PLANT, "min_load", value]])

    other = case("ml_other_period", period_map({OTHER_PERIOD: 0.4}))
    zero = case("ml_zero", b64(0.0, "float"))
    assert other.solution.obj == pytest.approx(zero.solution.obj, rel=1e-9)
    assert _minload_units(other) == _minload_units(zero) == set()
    assert other.flex_data.process_minload is None


def test_unit_with_min_load_only_in_another_period_has_no_floor_rows(tmp_path_factory) -> None:
    """Next to a unit with ``min_load = 0.4`` in the solve (coal_chp), the unit whose map has only p2030
    (coal_plant) gets no ``minFlow_minload`` row; the solve equals the twin where coal_plant has
    ``min_load = 0``.  Before, coal_plant got zero-floor rows."""
    chain = ["init", "west", "coal_chp", "heat", "coal"]

    def case(alt, value):
        return solve(tmp_path_factory, alt=alt, base_chain=chain,
                     values=[["constraint", "coal_chp_fix", "is_enabled", b64("no", "str")],
                             ["unit", CHP, "startup_method", b64("linear", "str")],
                             ["unit", CHP, "min_load", b64(0.4, "float")],
                             ["unit", PLANT, "startup_method", b64("linear", "str")],
                             ["unit", PLANT, "min_load", value]])

    other = case("ml2_other_period", period_map({OTHER_PERIOD: 0.4}))
    zero = case("ml2_zero", b64(0.0, "float"))
    assert _minload_units(other) == _minload_units(zero) == {CHP}
    assert sorted(utf8(other.flex_data.process_minload)["p"].to_list()) == [CHP]
    assert other.solution.obj == pytest.approx(zero.solution.obj, rel=1e-9)


def test_reservation_only_in_another_period_equals_no_reservation(tmp_path_factory) -> None:
    """``reservation = {p2030: 10}`` on the p2020 solve is no requirement: no ``v_reserve`` columns and the
    objective of the ``reservation = 0`` twin.  Before, the DB side kept the reserve's ``v_reserve`` columns
    (a value ≠ 0 somewhere) although the emit's sum over the solve was 0."""
    chain = ["init", "west", "coal", "wind", "reserve"]
    group = ["primary", "up", "electricity"]

    def case(alt, value):
        return solve(tmp_path_factory, alt=alt, base_chain=chain,
                     values=[["reserve__upDown__group", group, "reservation", value]])

    other = case("res_other_period", period_map({OTHER_PERIOD: 10.0}))
    zero = case("res_zero", b64(0.0, "float"))
    for step in (other, zero):
        prundt = step.flex_data.prundt
        assert prundt is None or prundt.height == 0
    assert other.solution.obj == pytest.approx(zero.solution.obj, rel=1e-9)


_ROLLING = [
    ["solve", "y2020_2day_dispatch", "solve_mode", b64("rolling_window", "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon", b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump", b64(12.0, "float")],
]


def test_rolling_min_load_only_in_another_period_equals_zero_min_load(tmp_path_factory) -> None:
    """The rolls (synthetic sub-solves) read the floor's units from their own ``p_min_load`` too: a rolling
    solve with ``min_load = {p2030: 0.4}`` gives every roll the objective of the ``min_load = 0`` twin."""
    def case(alt, value):
        url, scen = make_db(tmp_path_factory, alt=alt, base_chain=PLANT_CHAIN,
                            values=_ROLLING + [["unit", PLANT, "startup_method", b64("linear", "str")],
                                               ["unit", PLANT, "min_load", value]])
        return run(tmp_path_factory, url, scen)

    other = case("mlr_other_period", period_map({OTHER_PERIOD: 0.4}))
    zero = case("mlr_zero", b64(0.0, "float"))
    assert len(other) == len(zero) > 1
    for a, b in zip(other.values(), zero.values()):
        assert a.solution.obj == pytest.approx(b.solution.obj, rel=1e-9)
        assert a.flex_data.process_minload is None
