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

from flextool.engine.autoscale import (
    QuantityType,
    apply_layer2,
    unscale_solution,
)
from flextool.engine.autoscale._layer2_types import (
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
    from flextool.engine.autoscale import ScalingConfig
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
    from flextool.engine.autoscale import ScalingConfig
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


# ---------------------------------------------------------------------------
# The live HiGHS handle after the unscale (items 194, 239)
#
# The output writers read the solve from the HiGHS handle
# (``h.getSolution()``), not from the ``Solution``.  After the unscale the
# handle must give the model's own primal, row duals and reduced costs, and
# keep a valid basis.

_DETERMINISTIC = {"presolve": "off", "threads": 1, "simplex_strategy": 1, "output_flag": False}


def _build_three_col_lp() -> Problem:
    """min x1 + 2 x2 + 3 x3  s.t.  x1 + x2 + x3 >= 1,  x1 - x2 <= 0.5,  0 <= x <= 10.

    Hand solution (independent of any solver run): x2 = 1 - x1 from the
    cover row and x1 - x2 <= 0.5 give x1 <= 0.75; cost x1 + 2 (1 - x1) falls
    with x1, so x = (0.75, 0.25, 0), objective 1.25.  Both rows are tight;
    duals from c_j = sum_i y_i a_ij on the basic x1, x2:
    1 = y1 + y2, 2 = y1 - y2  ->  y = (1.5, -0.5).  Reduced costs
    d = c - A^T y = (0, 0, 3 - 1.5) = (0, 0, 1.5).
    """
    pb = Problem()
    idx = pl.DataFrame({"i": [0]})
    x1 = pb.add_var("v_test_x1", "i", idx, lower=0.0, upper=10.0)
    x2 = pb.add_var("v_test_x2", "i", idx, lower=0.0, upper=10.0)
    x3 = pb.add_var("v_test_x3", "i", idx, lower=0.0, upper=10.0)
    pb.add_cstr(
        "test_cover",
        sense=">=",
        lhs_terms={
            "a": Sum(x1, over=("i",)),
            "b": Sum(x2, over=("i",)),
            "c": Sum(x3, over=("i",)),
        },
        rhs_terms={"one": 1.0},
    )
    pb.add_cstr(
        "test_diff",
        sense="<=",
        lhs_terms={"a": Sum(x1, over=("i",)), "b": Sum(x2 * -1.0, over=("i",))},
        rhs_terms={"half": 0.5},
    )
    pb.set_objective(Sum(x1 * 1.0 + x2 * 2.0 + x3 * 3.0), sense="min")
    return pb


_HAND_X = np.array([0.75, 0.25, 0.0])
_HAND_ROW_DUAL = np.array([1.5, -0.5])
_HAND_COL_DUAL = np.array([0.0, 0.0, 1.5])


@pytest.fixture
def three_col_families(register_test_families):
    add_var, add_cstr = register_test_families
    # One quantity type per column and row, so each gets its own factor.
    add_var("v_test_x1", VarFamily(QuantityType.POWER))
    add_var("v_test_x2", VarFamily(QuantityType.ENERGY))
    add_var("v_test_x3", VarFamily(QuantityType.EMISSION_MASS))
    add_cstr("test_cover", CstrFamily(QuantityType.DURATION))
    add_cstr("test_diff", CstrFamily(QuantityType.FRACTION))


def test_layer2_unscale_handle_gives_true_duals(three_col_families):
    """Layer 2 forced on: the handle's reduced costs and duals equal the hand values (item 239).

    HiGHS recomputes ``col_dual`` inside ``setSolution`` from the scaled LP
    on the handle, which gave (3.22, 28.25, 7.5) here for the true
    (0, 0, 1.5); the writers of ``v_dual_invest_*`` read the handle.
    """
    import highspy

    from flextool.engine.autoscale._layer2 import apply_layer2_with_exponents
    from flextool.process_outputs.read_highs_solution import handle_col_dual

    # Oracle 2: a cold solve of the same LP without Layer 2.
    raw = _build_three_col_lp().solve(options=_DETERMINISTIC, keep_solver=True)
    raw_sol = raw.highs.getSolution()
    np.testing.assert_allclose(raw_sol.col_value, _HAND_X, atol=1e-12)
    np.testing.assert_allclose(raw_sol.row_dual, _HAND_ROW_DUAL, atol=1e-12)
    np.testing.assert_allclose(raw_sol.col_dual, _HAND_COL_DUAL, atol=1e-12)

    pb = _build_three_col_lp()
    plan = apply_layer2_with_exponents(pb, {
        QuantityType.POWER: 2, QuantityType.ENERGY: -1, QuantityType.EMISSION_MASS: 1,
        QuantityType.DURATION: 3, QuantityType.FRACTION: -2,
    })
    np.testing.assert_array_equal(plan.col_factors, [4.0, 0.5, 2.0])
    np.testing.assert_array_equal(plan.row_factors, [8.0, 0.25])

    sol = pb.solve(options=_DETERMINISTIC, keep_solver=True)
    h = sol.highs
    assert isinstance(h, highspy.Highs)
    # The scaled solve's own reduced costs differ from the true ones
    # (x3: 1.5 / 2), so a pass below is not a no-op.
    assert h.getSolution().col_dual[2] == pytest.approx(0.75, abs=1e-12)

    unscale_solution(sol, plan)

    np.testing.assert_allclose(sol.col_value, _HAND_X, atol=1e-12)
    np.testing.assert_allclose(sol.row_dual, _HAND_ROW_DUAL, atol=1e-12)
    np.testing.assert_allclose(sol.col_dual, _HAND_COL_DUAL, atol=1e-12)
    hs = h.getSolution()
    np.testing.assert_allclose(hs.col_value, _HAND_X, atol=1e-12)
    np.testing.assert_allclose(hs.row_dual, _HAND_ROW_DUAL, atol=1e-12)
    # What the output writers read (``extract_variable`` / ``write_all_variables``).
    np.testing.assert_allclose(handle_col_dual(h), _HAND_COL_DUAL, atol=1e-12)
    np.testing.assert_allclose(handle_col_dual(h, hs), _HAND_COL_DUAL, atol=1e-12)

    # Solved again, the handle's own (scaled) values apply, not the kept ones.
    h.run()
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    np.testing.assert_allclose(handle_col_dual(h), [0.0, 0.0, 0.75], atol=1e-12)


def _random_lp(c: np.ndarray, A: np.ndarray, row_upper: np.ndarray, col_upper: np.ndarray):
    import highspy
    import scipy.sparse as sp

    m, n = A.shape
    h = highspy.Highs()
    for k, v in _DETERMINISTIC.items():
        h.setOptionValue(k, v)
    h.setOptionValue("solver", "simplex")
    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = m
    lp.col_cost_ = c
    lp.col_lower_ = np.zeros(n)
    lp.col_upper_ = col_upper
    lp.row_lower_ = -np.inf * np.ones(m)
    lp.row_upper_ = row_upper
    M = sp.csc_matrix(A)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
    lp.a_matrix_.start_ = M.indptr
    lp.a_matrix_.index_ = M.indices
    lp.a_matrix_.value_ = M.data
    h.passModel(lp)
    return h


def test_layer2_unscale_handle_keeps_warm_basis():
    """After the unscale the handle keeps a valid basis, so a re-solve after a change is warm (item 194).

    ``setSolution`` clears the basis-valid flag; without the restore the
    re-solve below starts cold (about 394 against 79 iterations).
    """
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    n, m = 300, 200
    A = rng.random((m, n)) * (rng.random((m, n)) < 0.05)
    c = -rng.random(n)
    cf = 2.0 ** rng.integers(-3, 4, n)
    rf = 2.0 ** rng.integers(-3, 4, m)
    # The Layer-2 transform: x' = cf x, c' = c / cf, A' = diag(rf) A diag(1/cf), rows * rf.
    c_s = c / cf
    A_s = (A * rf[:, None]) / cf[None, :]
    ru_s = np.ones(m) * rf
    cu_s = np.full(n, 10.0) * cf

    h = _random_lp(c_s, A_s, ru_s, cu_s)
    h.run()
    before = h.getBasis()
    assert before.valid
    col_status = [int(s) for s in before.col_status]
    row_status = [int(s) for s in before.row_status]
    s = h.getSolution()
    sol = SimpleNamespace(
        highs=h,
        col_value=np.asarray(s.col_value),
        col_dual=np.asarray(s.col_dual),
        row_dual=np.asarray(s.row_dual),
    )
    unscale_solution(sol, SimpleNamespace(col_factors=cf, row_factors=rf))

    after = h.getBasis()
    assert after.valid
    assert [int(x) for x in after.col_status] == col_status
    assert [int(x) for x in after.row_status] == row_status
    np.testing.assert_array_equal(np.asarray(h.getSolution().col_value), sol.col_value)

    # The same cost change on the unscaled handle and on a cold copy.
    idx = np.arange(0, n, 7)
    h.changeColsCost(len(idx), idx, c_s[idx] * 1.3)
    h.run()
    iters_fixed = h.getInfo().simplex_iteration_count
    cold = _random_lp(c_s, A_s, ru_s, cu_s)
    cold.changeColsCost(len(idx), idx, c_s[idx] * 1.3)
    cold.run()
    iters_cold = cold.getInfo().simplex_iteration_count
    assert h.getObjectiveValue() == pytest.approx(cold.getObjectiveValue(), rel=1e-9)
    assert iters_fixed * 2 < iters_cold, (iters_fixed, iters_cold)
