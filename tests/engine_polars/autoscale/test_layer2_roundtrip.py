"""Bit-for-bit Layer-2 round-trip tests.

For each test we build the *same* tiny LP twice:

* once raw (no Layer 2);
* once with :func:`apply_layer2` applied before ``solve(...)`` and
  :func:`unscale_solution` applied after.

The two solutions must agree exactly on:

* objective value;
* primal vector ``col_value``;
* row duals;
* reduced costs (``col_dual``).

Power-of-two scaling is *bit-exact* in IEEE doubles (only the exponent
shifts), so the unscaled solution should match the raw solution
without numerical tolerance.  Empirically HiGHS' own internal
equilibration introduces sub-ULP noise; we keep a hair-tight tolerance
(``rtol=1e-12, atol=1e-10``) to absorb that without hiding real bugs.

The MIP test confirms two properties:

* integer columns are not column-scaled (so the integrality of the
  recovered solution is preserved);
* the objective and primal still match across scaled / unscaled.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from polar_high.engine import Problem, Sum

from flextool.engine_polars.autoscale import (
    QuantityType,
    apply_layer2,
    unscale_solution,
)
from flextool.engine_polars.autoscale._layer2_types import (
    CONSTRAINT_FAMILIES,
    VARIABLE_FAMILIES,
    CstrFamily,
    VarFamily,
)


# Tight but not zero — HiGHS' own equilibration introduces sub-ULP
# noise even on power-of-two user scaling.  ``rtol=1e-12, atol=1e-10``
# is well inside double precision and well outside the autoscaler's
# remit.
_RTOL = 1e-12
_ATOL = 1e-10


@pytest.fixture
def register_test_families(monkeypatch):
    """Inject test-only variable / constraint families into the
    Layer-2 registries.  Cleaned up by ``monkeypatch.undo()``.

    We use distinct names per fixture so the production registry
    isn't shadowed across tests.
    """

    added_vars: list[str] = []
    added_cstrs: list[str] = []

    def add_var(name: str, fam: VarFamily) -> None:
        VARIABLE_FAMILIES[name] = fam
        added_vars.append(name)

    def add_cstr(name: str, fam: CstrFamily) -> None:
        CONSTRAINT_FAMILIES[name] = fam
        added_cstrs.append(name)

    yield add_var, add_cstr

    for n in added_vars:
        VARIABLE_FAMILIES.pop(n, None)
    for n in added_cstrs:
        CONSTRAINT_FAMILIES.pop(n, None)


def _build_wide_lp(scale_costs: float = 1.0, scale_rhs: float = 1.0) -> Problem:
    """Tiny LP with hand-tuned coefficient magnitudes spanning >9 decades.

    Variables (3):

    * ``v_test_power``    POWER       [0, +inf)
    * ``v_test_energy``   ENERGY      [0, +inf)
    * ``v_test_money``    CURRENCY    [0, +inf)

    Constraints (3):

    * ``test_power_cap``     POWER  row:  1·v_power + 1e-3·v_energy ≤ 1e5
    * ``test_energy_cap``    ENERGY row:  1e7·v_power + 1·v_energy + 1e-2·v_money ≤ 1e9
    * ``test_money_cap``     CURRENCY row: 1·v_money ≤ 1e3 · scale_rhs

    Objective (max — equivalent to ``min -..``):  reward energy and
    money, punish power.  Cost magnitudes span 1e-4 .. 1e+4 to
    exercise the per-type bucketing.
    """
    pb = Problem()

    v_power_idx = pl.DataFrame({"i": [0]})
    v_energy_idx = pl.DataFrame({"i": [0]})
    v_money_idx = pl.DataFrame({"i": [0]})

    v_power = pb.add_var("v_test_power", "i", v_power_idx, lower=0.0, upper=float("inf"))
    v_energy = pb.add_var("v_test_energy", "i", v_energy_idx, lower=0.0, upper=float("inf"))
    v_money = pb.add_var("v_test_money", "i", v_money_idx, lower=0.0, upper=float("inf"))

    pb.add_cstr(
        "test_power_cap",
        sense="<=",
        lhs_terms={
            "p": Sum(v_power, over=("i",)),
            "e": Sum(v_energy * 1e-3, over=("i",)),
        },
        rhs_terms={"cap": 1e5 * scale_rhs},
    )
    pb.add_cstr(
        "test_energy_cap",
        sense="<=",
        lhs_terms={
            "p": Sum(v_power * 1e7, over=("i",)),
            "e": Sum(v_energy, over=("i",)),
            "m": Sum(v_money * 1e-2, over=("i",)),
        },
        rhs_terms={"cap": 1e9 * scale_rhs},
    )
    pb.add_cstr(
        "test_money_cap",
        sense="<=",
        lhs_terms={"m": Sum(v_money, over=("i",))},
        rhs_terms={"cap": 1e3 * scale_rhs},
    )

    # Maximise revenue: cost vector signs reflect ``min`` sense.
    obj = (v_power * (-1e-4 * scale_costs)
           + v_energy * (-1.0 * scale_costs)
           + v_money * (-1e4 * scale_costs))
    pb.set_objective(Sum(obj), sense="min")
    return pb


def _solve_and_collect(pb: Problem) -> dict:
    sol = pb.solve()
    return {
        "obj": float(sol.obj),
        "col_value": np.asarray(sol.col_value, dtype=np.float64).copy(),
        "row_dual": np.asarray(sol.row_dual, dtype=np.float64).copy(),
        "col_dual": np.asarray(sol.col_dual, dtype=np.float64).copy(),
        "optimal": bool(sol.optimal),
    }


def test_layer2_lp_bit_for_bit_roundtrip(register_test_families):
    """Wide-range LP: scaled solve unscaled == raw solve, to 1e-12."""
    add_var, add_cstr = register_test_families
    add_var("v_test_power", VarFamily(QuantityType.POWER))
    add_var("v_test_energy", VarFamily(QuantityType.ENERGY))
    add_var("v_test_money", VarFamily(QuantityType.CURRENCY))
    add_cstr("test_power_cap", CstrFamily(QuantityType.POWER))
    add_cstr("test_energy_cap", CstrFamily(QuantityType.ENERGY))
    add_cstr("test_money_cap", CstrFamily(QuantityType.CURRENCY))

    # Raw solve.
    raw = _solve_and_collect(_build_wide_lp())

    # Layer-2 solve.
    pb = _build_wide_lp()
    from flextool.engine_polars.autoscale import ScalingConfig
    plan = apply_layer2(pb, ScalingConfig())
    # Verify power-of-two factors are exact (no fractional bits).
    for f in plan.col_factors:
        assert math.log2(f).is_integer(), f"col_factor {f} not a power of 2"
    for f in plan.row_factors:
        assert math.log2(f).is_integer(), f"row_factor {f} not a power of 2"

    sol = pb.solve()
    unscale_solution(sol, plan)

    assert sol.optimal == raw["optimal"]
    # Objective is invariant under (c→c/cf, x→cf·x) — bit-for-bit.
    assert sol.obj == pytest.approx(raw["obj"], rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        sol.col_value, raw["col_value"], rtol=_RTOL, atol=_ATOL,
        err_msg="primal mismatch after Layer-2 unscale",
    )
    np.testing.assert_allclose(
        sol.row_dual, raw["row_dual"], rtol=_RTOL, atol=_ATOL,
        err_msg="row duals mismatch after Layer-2 unscale",
    )
    np.testing.assert_allclose(
        sol.col_dual, raw["col_dual"], rtol=_RTOL, atol=_ATOL,
        err_msg="reduced costs mismatch after Layer-2 unscale",
    )


def test_layer2_mip_preserves_integrality(register_test_families):
    """MIP: integer column survives Layer 2 with bit-exact value."""
    add_var, add_cstr = register_test_families
    add_var("v_test_count", VarFamily(QuantityType.DIMENSIONLESS))
    add_var("v_test_dollars", VarFamily(QuantityType.CURRENCY))
    add_cstr("count_cap", CstrFamily(QuantityType.DIMENSIONLESS))
    add_cstr("dollars_cap", CstrFamily(QuantityType.CURRENCY))

    def _build_mip() -> Problem:
        pb = Problem()
        v_count = pb.add_var(
            "v_test_count", "i",
            pl.DataFrame({"i": [0]}),
            lower=0.0, upper=5.0, integer=True,
        )
        v_dollars = pb.add_var(
            "v_test_dollars", "i",
            pl.DataFrame({"i": [0]}),
            lower=0.0, upper=float("inf"),
        )
        # Wide-range RHS to give Layer 2 something to scale.
        pb.add_cstr(
            "count_cap",
            sense="<=",
            lhs_terms={"c": Sum(v_count, over=("i",))},
            rhs_terms={"cap": 5.0},
        )
        pb.add_cstr(
            "dollars_cap",
            sense="<=",
            lhs_terms={
                "d": Sum(v_dollars, over=("i",)),
                "c": Sum(v_count * (-1e6), over=("i",)),
            },
            rhs_terms={"cap": 0.0},
        )
        # Maximise dollars; (-1e3) coef = "min -1e3 v_dollars".
        # Tighten with -count*1.0 so an interior point is required.
        obj = v_dollars * (-1e3) + v_count * (-1.0)
        pb.set_objective(Sum(obj), sense="min")
        return pb

    raw = _solve_and_collect(_build_mip())

    pb = _build_mip()
    from flextool.engine_polars.autoscale import ScalingConfig
    plan = apply_layer2(pb, ScalingConfig())

    # The integer column must not have been column-scaled.
    integer_cols = list(plan.skipped_integer_cols)
    # v_test_count is integer; find its col_id.
    v_count = pb._vars["v_test_count"]
    count_cids = v_count.frame["col_id"].to_numpy().tolist()
    for cid in count_cids:
        assert cid in integer_cols, f"integer col {cid} not skipped"
        assert plan.col_factors[cid] == 1.0, (
            f"integer col {cid} got col_factor {plan.col_factors[cid]}"
        )

    sol = pb.solve()
    unscale_solution(sol, plan)

    # Integer result must be exactly integer (and equal to raw).
    raw_count = raw["col_value"][count_cids[0]]
    new_count = sol.col_value[count_cids[0]]
    assert new_count == raw_count, (
        f"integer col value drifted: raw={raw_count}, scaled+unscaled={new_count}"
    )
    # Continuous column should match too.
    assert sol.obj == pytest.approx(raw["obj"], rel=_RTOL, abs=_ATOL)
    np.testing.assert_allclose(
        sol.col_value, raw["col_value"], rtol=_RTOL, atol=_ATOL,
    )
