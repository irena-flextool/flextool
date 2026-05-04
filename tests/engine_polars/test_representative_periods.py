"""Closed-form Representative Period (RP) weighting test.

flextool's ``test_representative_periods.py`` exercises clustering and
weight-matrix algorithms whose Python sources live in
``flextool.representative_periods.*`` — those modules are not vendored
into ``flexpy_spike``.  The functional half of the flextool test
(`TestRPAllRepresented` / `TestRPHalfRepresented`) asserts what the
weighting math is supposed to do at the LP-objective level:

    total_cost = Σ_d (RP_cost_d × RP_weight_d).

flexpy's RP machinery surfaces this through the ``op_factor`` term in
``flextool/model.py`` (the .mod's ``Σ … · step_duration · rp_cost_weight
· inflation_op / period_share``).  Inflation_op (set per period) acts
as the period weight: scaling it by ``Y`` scales every operational
**and** investment cost contribution by ``Y``, so the LP objective is
linear in the period weight when the rest of the inputs are held
fixed.

The two on-disk RP fixtures (``work_years_represented_half`` and
``work_years_represented_2_5``) are identical except for
``p_inflation_factor_operations_yearly`` / ``…investment_yearly``:
0.5 vs. 2.5.  This test re-solves both with flexpy and asserts the
objective ratio matches the period-weight ratio (2.5 / 0.5 = 5.0)
to machine precision — i.e. the LP literally implements
``cost = Σ RP_cost · RP_weight``.
"""
from pathlib import Path

import polars as pl
import pytest

from flexpy import Problem
from flextool.engine_polars import build_flextool, load_flextool

DATA = Path(__file__).resolve().parent / "data"
WORK_HALF = DATA / "work_years_represented_half"
WORK_2_5 = DATA / "work_years_represented_2_5"


def _solve(work_dir: Path) -> float:
    data = load_flextool(work_dir)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal, f"{work_dir.name} did not solve to optimality"
    return float(sol.obj)


def _period_weight(work_dir: Path) -> float:
    """Read the per-period RP weight (``p_inflation_factor_operations_yearly``)."""
    df = pl.read_csv(
        work_dir / "solve_data" / "p_inflation_factor_operations_yearly.csv"
    )
    # Single-period fixtures: one row.
    assert df.height == 1, f"{work_dir.name}: expected one period weight row"
    return float(df["value"][0])


def _flextool_published_obj(work_dir: Path) -> float:
    parq = list(work_dir.glob("output_raw/v_obj__*.parquet"))
    if parq:
        return float(pl.read_parquet(parq[0])["objective"][0])
    return float(pl.read_csv(work_dir / "output_raw" / "v_obj.csv")["objective"][0])


def test_rp_total_cost_equals_sum_of_rp_cost_times_rp_weight():
    """Closed-form RP weighting:  total_cost = Σ_d (RP_cost_d · RP_weight_d).

    Single-period RP fixtures collapse the sum to a single term.  The two
    fixtures share every input except the per-period RP weight (0.5 vs. 2.5),
    so their objectives must be exactly proportional to the weight ratio,
    AND each must match flextool's published objective.
    """
    obj_half = _solve(WORK_HALF)
    obj_2_5 = _solve(WORK_2_5)

    w_half = _period_weight(WORK_HALF)  # 0.5
    w_2_5 = _period_weight(WORK_2_5)    # 2.5

    # 1) Linearity in RP weight: obj(Y) / obj(X) == Y / X.
    expected_ratio = w_2_5 / w_half     # 5.0
    actual_ratio = obj_2_5 / obj_half
    rel = abs(actual_ratio - expected_ratio) / expected_ratio
    assert rel < 1e-9, (
        f"RP-weight linearity broken: obj_half={obj_half}, obj_2_5={obj_2_5}, "
        f"actual ratio={actual_ratio}, expected={expected_ratio}, rel={rel}"
    )

    # 2) Same closed-form unit cost recovered from each fixture.
    unit_half = obj_half / w_half
    unit_2_5 = obj_2_5 / w_2_5
    rel_unit = abs(unit_half - unit_2_5) / max(1.0, abs(unit_half))
    assert rel_unit < 1e-9, (
        f"Per-RP unit cost disagrees across fixtures: "
        f"unit_half={unit_half}, unit_2_5={unit_2_5}, rel={rel_unit}"
    )

    # 3) flexpy matches flextool's published objective on each fixture
    #    (closed-form RP weighting is what flextool also uses).
    for work, flexpy_obj in ((WORK_HALF, obj_half), (WORK_2_5, obj_2_5)):
        ft_obj = _flextool_published_obj(work)
        rel_p = abs(flexpy_obj - ft_obj) / max(1.0, abs(ft_obj))
        assert rel_p < 1e-6, (
            f"{work.name}: flexpy={flexpy_obj} vs flextool={ft_obj}, rel={rel_p}"
        )
