"""Part-load curve -> lines of a ``piecewise_linear`` unit (``flextool/engine/input/_piecewise_curve.py``).

Expected lines are the independent hand calculation of ``specs/piecewise_efficiency_oracle.md`` (exact fractions,
F = ℓ/η): O1 thermal {0.4: 0.30, 0.7: 0.40, 1.0: 0.42}, O2 electrolyser {0.2: 0.70, 0.6: 0.66, 1.0: 0.60}, O4 p2035
{0.5: 0.55, 1.0: 0.50}, O5 CHP {0.3: 0.25, 0.6: 0.35, 1.0: 0.40}.  The segment ending at load 1 must give the bits
of the min_load_efficiency section and slope (``params/_derived._mle_section`` / ``_mle_slope``).
"""
from __future__ import annotations

from fractions import Fraction as Fr

import polars as pl
import pytest

from flextool.engine.input import _piecewise_curve as pc

O1 = [(0.4, 0.30), (0.7, 0.40), (1.0, 0.42)]
O2 = [(0.2, 0.70), (0.6, 0.66), (1.0, 0.60)]
O4_P2035 = [(0.5, 0.55), (1.0, 0.50)]
O5 = [(0.3, 0.25), (0.6, 0.35), (1.0, 0.40)]


def _lines(lines: list[pc.Line]) -> list[tuple[str, float, float]]:
    return [(ln.seg, ln.slope, ln.intercept) for ln in lines]


def _approx(expected: list[tuple[str, Fr, Fr]]) -> list:
    return [(seg, pytest.approx(float(a), rel=1e-12), pytest.approx(float(b), rel=1e-12, abs=1e-15))
            for seg, a, b in expected]


def test_thermal_curve_online_form_O1() -> None:
    """Rising curve: the two segments, no origin line (oracle §2)."""
    assert _lines(pc.online_lines(O1)) == _approx([("1", Fr(25, 18), Fr(7, 9)), ("2", Fr(265, 126), Fr(5, 18))])


def test_thermal_curve_capacity_form_O1R() -> None:
    """Without online variables: the origin line through the best point only (no segment has β < 0; oracle §4)."""
    assert _lines(pc.capacity_lines(O1)) == _approx([("0", Fr(50, 21), Fr(0))])


def test_electrolyser_curve_both_forms_O2() -> None:
    """Falling curve: origin line 1/η₀ in the online form, 1/max η in the capacity form, plus both segments (β < 0;
    oracle §5 / §7)."""
    segs = [("1", Fr(120, 77), Fr(-2, 77)), ("2", Fr(125, 66), Fr(-5, 22))]
    assert _lines(pc.online_lines(O2)) == _approx([("0", Fr(10, 7), Fr(0))] + segs)
    assert _lines(pc.capacity_lines(O2)) == _approx([("0", Fr(10, 7), Fr(0))] + segs)


def test_two_point_falling_curve_is_two_lines_O4() -> None:
    assert _lines(pc.online_lines(O4_P2035)) == _approx([("0", Fr(20, 11), Fr(0)), ("1", Fr(24, 11), Fr(-2, 11))])


def test_chp_curve_O5() -> None:
    assert _lines(pc.online_lines(O5)) == _approx([("1", Fr(12, 7), Fr(24, 35)), ("2", Fr(55, 28), Fr(15, 28))])


def test_flat_curve_has_exactly_zero_intercept() -> None:
    """η₀ = η₁: β is 0.0 exactly and α is 1/η bit for bit (a constant-efficiency unit)."""
    for eta in (0.66, 0.9, 0.37):
        (line,) = pc.online_lines([(0.2, eta), (1.0, eta)])
        assert line.intercept == 0.0 and line.slope == 1.0 / eta
    assert _lines(pc.online_lines([(1.0, 0.5)])) == [("0", 2.0, 0.0)]


def test_collinear_points_merge_into_one_line() -> None:
    """F = 0.5 + 1.5ℓ through three points: one segment after the merge."""
    pts = [(0.25, 0.25 / 0.875), (0.5, 0.4), (1.0, 0.5)]
    assert pc.merge_points(pts) == [pts[0], pts[2]]
    assert _lines(pc.online_lines(pts)) == _approx([("1", Fr(3, 2), Fr(1, 2))])
    assert pc.n_lines(pts) == 1


def test_segment_ending_at_full_load_is_bit_equal_to_min_load_efficiency() -> None:
    from flextool.engine.preprocess.params._derived._process_flows import _mle_section, _mle_slope

    cases = [(0.4, 0.30, 0.42), (0.3, 0.25, 0.40), (0.35, 0.31, 0.47), (0.6, 0.40, 0.50), (0.2, 0.7, 0.6)]
    frame = pl.DataFrame(cases, schema=["ml", "eta_min", "eta"], orient="row")
    out = frame.select(section=_mle_section(pl.col("eta"), pl.col("eta_min"), pl.col("ml")),
                       slope=_mle_slope(pl.col("eta"), pl.col("eta_min"), pl.col("ml")))
    for (ml, eta_min, eta), section, slope in zip(cases, out["section"], out["slope"]):
        alpha, beta = pc.segment((ml, eta_min), (1.0, eta))
        assert beta == section and alpha == slope


def test_unit_flags() -> None:
    flags = pc.piecewise_unit_flags
    assert flags([O1]) == pc.PiecewiseFlags(True, True, False, False)
    assert flags([O2]) == pc.PiecewiseFlags(True, False, False, False)
    assert flags([[(0.4, 0.3), (1.0, 0.42)]]) == pc.PiecewiseFlags(False, False, True, False)
    assert flags([[(0.2, 0.66), (1.0, 0.66)]]) == pc.PiecewiseFlags(False, False, False, True)
    # O4: one rising line in p2030, two lines in p2035 -> multi-line in every period, online (rises in p2030).
    o4 = flags([[(0.5, 0.40), (1.0, 0.50)], O4_P2035])
    assert o4 == pc.PiecewiseFlags(True, True, False, False) and o4.auto_startup
    assert flags([[(0.4, 0.3), (1.0, 0.42)]]).auto_startup and not flags([O2]).auto_startup


def test_lower_hull_E8_example() -> None:
    """{0.4: 0.30, 0.7: 0.36, 1.0: 0.42}: α falls 2.037 -> 1.455; the hull drops 0.7, where η becomes 49/130."""
    pts = [(0.4, 0.30), (0.7, 0.36), (1.0, 0.42)]
    (a1, _), (a2, _) = pc.segments(pts)
    assert a1 == pytest.approx(55 / 27, rel=1e-12) and a2 == pytest.approx((1 / 0.42 - 0.7 / 0.36) / 0.3, rel=1e-12)
    hull = pc.lower_hull(pts)
    assert hull == [(0.4, 0.30), (1.0, 0.42)]
    assert pc.hull_efficiency(hull, 0.7) == pytest.approx(49 / 130, rel=1e-12)
    assert pc.lower_hull(O1) == O1


def test_frame_api_lines_and_flags() -> None:
    frame = pl.DataFrame(
        [("G", None, 0.4, 0.30), ("G", None, 0.7, 0.40), ("G", None, 1.0, 0.42),
         ("E", "p2030", 0.2, 0.70), ("E", "p2030", 0.6, 0.66), ("E", "p2030", 1.0, 0.60)],
        schema=pc.CURVE_SCHEMA, orient="row")
    flags = pc.unit_flags_frame(frame)
    assert flags.rows() == [("E", True, False, False, False, False), ("G", True, True, False, False, True)]
    lines = pc.curve_lines(frame, online={"G"})
    assert lines.select("p", "d", "seg").rows() == [("E", "p2030", "0"), ("E", "p2030", "1"), ("E", "p2030", "2"),
                                                    ("G", None, "1"), ("G", None, "2")]
