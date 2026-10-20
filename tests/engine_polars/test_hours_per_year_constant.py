"""Guards for the centralised ``HOURS_PER_YEAR`` unit constant.

Step 1 of the 8760 cleanup: every hours->years conversion in the engine
routes through :data:`flextool.engine_polars._constants.HOURS_PER_YEAR`
rather than a bare literal, so the serve/report/cost annualisations stay
mutually consistent and a future per-period override replaces one symbol.

The byte-parity of the ``node_full_timeline_flow`` refactor (windowing ->
``timeline_duration_in_years``) for the <=1-year case is proven by the
unchanged integration goldens (``multi_fullYear_battery_nested_multi_invest``
and ``test_a_lot_but_not_multi_year`` in ``tests/test_scenarios.py``); the
regime test below pins the intended sub-year / one-year / multi-year
behaviour of the annualisation formula itself.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

import flextool.engine_polars as ep
from flextool.engine_polars._constants import HOURS_PER_YEAR


def test_hours_per_year_value() -> None:
    assert HOURS_PER_YEAR == 8760.0


def test_no_bare_year_constant_in_engine_code() -> None:
    """No module but ``_constants`` may hardcode 8760 in executable code.

    AST-based, so comments and docstrings (which legitimately quote the
    ``/ 8760`` formulae) are ignored — only real numeric literals count.
    """
    pkg_dir = pathlib.Path(ep.__file__).parent
    offenders: list[str] = []
    for py in sorted(pkg_dir.rglob("*.py")):
        if py.name == "_constants.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, (int, float))
                and not isinstance(node.value, bool)
                and float(node.value) == 8760.0
            ):
                offenders.append(f"{py.name}:{node.lineno}")
    assert not offenders, (
        "bare 8760 literal(s) in engine code — route through "
        "flextool.engine_polars._constants.HOURS_PER_YEAR: "
        + ", ".join(offenders)
    )


def _annualise(ts_sum: float, const_per_step_hours: float,
               total_hours: float) -> float:
    """Mirror of the refactored ``node_full_timeline_flow`` annualisation.

    ``energy / timeline_duration_in_years`` with
    ``timeline_duration_in_years = total_hours / HOURS_PER_YEAR``.
    """
    duration_in_years = total_hours / HOURS_PER_YEAR
    energy = ts_sum + const_per_step_hours * total_hours
    return energy / duration_in_years


@pytest.mark.parametrize(
    "ts_sum,const,total_hours,expected",
    [
        # sub-year: half a year of inflow scales up to a full year.
        (100.0, 0.0, 4380.0, 200.0),
        # exactly one year: unchanged (duration == 1.0).
        (100.0, 0.0, 8760.0, 100.0),
        # multi-year: averages over the years (200 over 2 yr -> 100/yr).
        (200.0, 0.0, 17520.0, 100.0),
        # constant inflow of 1 per hour over a year -> 8760 energy.
        (0.0, 1.0, 8760.0, 8760.0),
    ],
)
def test_annualisation_regimes(
    ts_sum: float, const: float, total_hours: float, expected: float,
) -> None:
    assert _annualise(ts_sum, const, total_hours) == pytest.approx(expected)
