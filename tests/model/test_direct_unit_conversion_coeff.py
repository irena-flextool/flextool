"""A direct (one-input, one-output) unit's ``conversion_flow_coeff`` is folded into its
fuel-use curve (#10).

A direct unit with non-default input/output ``conversion_flow_coeff`` must behave exactly like
the same unit with coefficients left at their default of 1 and equivalent efficiencies: the
coefficient ratio is folded into the unit's slope (and, for a ``min_load_efficiency`` unit, its
section) rather than applied as a second multiplier anywhere else.  Before the fix, the ratio was
a no-op for a direct unit (built only for multi-flow/indirect units), so the coefficient had no
effect on the direct unit's fuel, cost or online behaviour.

Oracle-derived equivalence (direct unit ``coal_plant``, input ``coal_market``, output
``demand_node``; ``k_src`` / ``k_sink`` = the input / output ``conversion_flow_coeff``):

    slope  = slope0 * k_sink / k_src
    section = section0 / k_src    (min_load_efficiency only; section0 unchanged otherwise)

Case A (constant_efficiency): k_src=2.0, k_sink=0.5, efficiency=0.4 is equivalent to
coefficients left at 1 with efficiency=1.6 (slope 0.625 either way).

Case B (min_load_efficiency, linear startup, min_load=0.3): k_src=2.0, k_sink=0.5,
efficiency=0.45, efficiency_at_min_load=0.35 is equivalent to coefficients left at 1 with
efficiency=441/275, efficiency_at_min_load=441/415 (slope 215/441, section 60/441 either way).
The fuel drawn at minimum load, per unit of capacity, is then min_load*slope + section =
124.5/441 (hand-derived from the folded slope/section, independent of this fix's implementation).

Both cases are checked in a rolling solve (synthetic per-roll sub-solves, which seed their own
``p_slope``/``p_section`` from the CSV cascade rather than this module's direct-DB path) and in
a single solve, reusing the ``1week_rolling`` vehicle and ``_roll`` helper of
``test_rolling_period_maps.py``.
"""
from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from tests.engine._unit_db import b64
from tests.model.test_rolling_period_maps import _canon, _roll, _value, scalar

COAL = "coal_plant"
COAL_IN = "coal_market"
COAL_OUT = "demand_node"

_SLOPE_A = 0.625
_SLOPE_B = 215 / 441
_SECTION_B = 60 / 441
_MIN_LOAD_FUEL_B = 124.5 / 441


def _coeff(k_src: float, k_sink: float) -> list:
    return [["unit__inputNode", [COAL, COAL_IN], "conversion_flow_coeff", scalar(k_src)],
            ["unit__outputNode", [COAL, COAL_OUT], "conversion_flow_coeff", scalar(k_sink)]]


# Case A: constant_efficiency.
CASE_A_COEFF = [["unit", COAL, "efficiency", scalar(0.4)]] + _coeff(2.0, 0.5)
CASE_A_EQUIV = [["unit", COAL, "efficiency", scalar(1.6)]]


def _case_b(eff: float, eff_min: float, *, with_coeff: bool) -> list:
    values = [
        ["unit", COAL, "efficiency", scalar(eff)],
        ["unit", COAL, "conversion_method", b64("min_load_efficiency", "str")],
        ["unit", COAL, "startup_method", b64("linear", "str")],
        ["unit", COAL, "min_load", scalar(0.3)],
        ["unit", COAL, "efficiency_at_min_load", scalar(eff_min)],
    ]
    return values + _coeff(2.0, 0.5) if with_coeff else values


CASE_B_COEFF = _case_b(0.45, 0.35, with_coeff=True)
CASE_B_EQUIV = _case_b(441 / 275, 441 / 415, with_coeff=False)


def _unit_rows(frame, p: str) -> pl.DataFrame:
    return _canon(frame).filter(pl.col("p") == p)


def _assert_close(a: pl.DataFrame, b: pl.DataFrame, what, *, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> None:
    """``a`` and ``b`` agree within ``rel_tol``/``abs_tol`` on every column (order-independent).

    The "coeff" and "equiv" runs reach the same mathematical slope/section/flow through different
    float operations (a multi-step fold vs. a directly authored equivalent efficiency), so they are
    not expected to be bit-identical — only equal to the oracle's tolerance.
    """
    keys = [c for c in a.columns if c != "value"]
    try:
        assert_frame_equal(a.sort(keys), b.sort(keys), check_exact=False, rel_tol=rel_tol, abs_tol=abs_tol,
                           check_column_order=False)
    except AssertionError as exc:
        raise AssertionError(f"{what}: {exc}") from None


@pytest.mark.parametrize("single", [False, True], ids=["rolling", "single"])
def test_constant_efficiency_coeff_matches_equivalent_efficiency(tmp_path_factory, single) -> None:
    coeff = _roll(tmp_path_factory, CASE_A_COEFF, single=single)
    equiv = _roll(tmp_path_factory, CASE_A_EQUIV, single=single)
    assert list(coeff) == list(equiv)
    for name, step in coeff.items():
        other = equiv[name]
        dc, de = step.flex_data, other.flex_data
        assert _value(dc.p_slope, p=COAL) == {round(_SLOPE_A, 9)}, name
        assert _value(dc.p_slope, p=COAL) == _value(de.p_slope, p=COAL), name
        _assert_close(_canon(dc.p_slope), _canon(de.p_slope), (name, "p_slope"))
        assert step.solution.obj == pytest.approx(other.solution.obj, rel=1e-9), name
        _assert_close(_unit_rows(step.solution.value("v_flow"), COAL),
                     _unit_rows(other.solution.value("v_flow"), COAL), (name, "v_flow"))


@pytest.mark.parametrize("single", [False, True], ids=["rolling", "single"])
def test_min_load_efficiency_coeff_matches_equivalent_efficiency(tmp_path_factory, single) -> None:
    coeff = _roll(tmp_path_factory, CASE_B_COEFF, single=single)
    equiv = _roll(tmp_path_factory, CASE_B_EQUIV, single=single)
    assert list(coeff) == list(equiv)
    for name, step in coeff.items():
        other = equiv[name]
        dc, de = step.flex_data, other.flex_data
        assert _value(dc.p_slope, p=COAL) == {round(_SLOPE_B, 9)}, name
        assert _value(dc.p_section, p=COAL) == {round(_SECTION_B, 9)}, name
        assert _value(dc.p_slope, p=COAL) == _value(de.p_slope, p=COAL), name
        assert _value(dc.p_section, p=COAL) == _value(de.p_section, p=COAL), name
        # Hand-derived fuel per unit of capacity at minimum load: min_load*slope + section.
        min_load_fuel = 0.3 * next(iter(_value(dc.p_slope, p=COAL))) + next(iter(_value(dc.p_section, p=COAL)))
        assert round(min_load_fuel, 9) == round(_MIN_LOAD_FUEL_B, 9), name
        for field in ("p_slope", "p_section"):
            _assert_close(_canon(getattr(dc, field)), _canon(getattr(de, field)), (name, field))
        assert step.solution.obj == pytest.approx(other.solution.obj, rel=1e-9), name
        _assert_close(_unit_rows(step.solution.value("v_flow"), COAL),
                     _unit_rows(other.solution.value("v_flow"), COAL), (name, "v_flow"))
        _assert_close(_unit_rows(step.solution.value("v_online_linear"), COAL),
                     _unit_rows(other.solution.value("v_online_linear"), COAL), (name, "v_online_linear"))
