"""Param-tracked auto-update tests.

Targeted tests for the :meth:`polar_high.WarmProblem.declare_mutable` /
:meth:`update_param` plumbing introduced for handoff #2.  The wider
``run_chain``-level equivalence tests live in
``test_warm_chain_runner.py`` — these verify the engine-level
primitives with synthetic LPs where:

  * the LP has a known closed-form solution per Param value, and
  * the Param appears in MULTIPLE LP cells via composite expressions
    (so the test wouldn't pass under the old ``update_rhs`` /
    ``update_obj_coef``-only API).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from polar_high import Param, Problem, Sum, WarmProblem

pytestmark = pytest.mark.solver


def _toy_problem(p_cap: Param, p_price: Param, p_dur: Param) -> Problem:
    """Tiny 2-time-step LP:

        min Σ_t  v[t] * p_price[t] * p_dur[t]
        s.t.    v[t]                        ≤ p_cap[t]      ∀ t
                Σ_t v[t] * p_dur[t]         ≥ 5                       (demand)
                v[t]                        ≥ 0
    """
    p = Problem()
    dt = pl.DataFrame({"t": [0, 1]})
    v = p.add_var("v", ("t",), dt, lower=0.0)
    # bound:  v[t] ≤ p_cap[t]
    p.add_cstr("cap_eq", over=dt, sense="<=",
               lhs_terms={"v": v},
               rhs_terms={"cap": p_cap})
    # demand:  Σ v[t] * p_dur[t] ≥ 5
    p.add_cstr("demand_eq", over=None, sense=">=",
               lhs_terms={"v_dur": Sum(v * p_dur)},
               rhs_terms={"target": 5.0})
    # obj:  Σ v[t] * p_price[t] * p_dur[t]
    p.set_objective(Sum(v * p_price * p_dur), sense="min")
    return p


def _solve_cold(p_cap: Param, p_price: Param,
                p_dur: Param) -> float:
    p = _toy_problem(p_cap, p_price, p_dur)
    sol = p.solve()
    assert sol.optimal
    return float(sol.obj)


def _toy_problem_dim_lhs(p_cap: Param, p_price: Param,
                          p_eff: Param) -> Problem:
    """Per-(s, t) cstr putting tracked Param ``p_eff`` into the LHS:

        min Σ v[s, t] * p_price[s, t]
        s.t.    v[s, t] * p_eff[s, t]         ≤ p_cap[s, t]   ∀ (s, t)
                Σ v[s, t]                     ≥ 3.0
                v[s, t]                       ≥ 0

    The per-(s, t) cap cstr puts ``p_eff[s, t]`` into the LHS
    coefficient at every (row=(s,t), col=v[s,t]) cell — both
    ``p_eff``'s dims survive into the constraint axis, so the
    Sum-collapse case isn't triggered.
    """
    p = Problem()
    s_t = pl.DataFrame({
        "s": ["s1", "s1", "s2", "s2"],
        "t": [0, 1, 0, 1],
    })
    v = p.add_var("v", ("s", "t"), s_t, lower=0.0)
    # cap_eff_eq:  v[s, t] * p_eff[s, t]  ≤  p_cap[s, t]
    p.add_cstr("cap_eff_eq", over=s_t, sense="<=",
               lhs_terms={"v_eff": v * p_eff},
               rhs_terms={"cap": p_cap})
    # demand: Σ v[s, t] ≥ 3.0
    p.add_cstr("demand_eq", over=None, sense=">=",
               lhs_terms={"v_sum": Sum(v)},
               rhs_terms={"target": 3.0})
    p.set_objective(Sum(v * p_price), sense="min")
    return p


def _solve_cold_dim_lhs(p_cap: Param, p_price: Param,
                         p_eff: Param) -> float:
    p = _toy_problem_dim_lhs(p_cap, p_price, p_eff)
    sol = p.solve()
    assert sol.optimal
    return float(sol.obj)


def test_update_param_composite_lhs_coef() -> None:
    """``Σ_s v[s, t] * p_eff[s, t] ≥ ...`` per (t) puts ``p_eff`` into
    the LHS matrix at every (row=t, col=v[s, t]) cell.
    ``update_param("p_eff", new)`` must refresh every cell.
    """
    s_t = pl.DataFrame({
        "s": ["s1", "s1", "s2", "s2"],
        "t": [0, 1, 0, 1],
    })
    # cap chosen so that v ≤ cap/eff binds — and changing eff changes
    # the actual binding (so sol moves with eff).
    cap = Param(("s", "t"),
                s_t.with_columns(value=pl.Series([1.5, 1.5, 1.5, 1.5])),
                name="p_cap")
    price = Param(("s", "t"),
                  s_t.with_columns(value=pl.Series([2.0, 3.0, 1.0, 1.5])),
                  name="p_price")
    eff = Param(("s", "t"),
                s_t.with_columns(value=pl.Series([1.0, 1.0, 1.0, 1.0])),
                name="p_eff")

    p = _toy_problem_dim_lhs(cap, price, eff)
    wp = WarmProblem(p)
    wp.declare_mutable("p_eff")
    sol0 = wp.solve()
    obj0 = sol0.obj

    new_eff = Param(("s", "t"),
                    s_t.with_columns(value=pl.Series([0.5, 0.8, 1.5, 1.2])),
                    name="p_eff")
    wp.update_param("p_eff", new_eff)
    sol1 = wp.solve()

    obj_cold = _solve_cold_dim_lhs(cap, price, new_eff)
    assert abs(sol1.obj - obj_cold) < 1e-9, (
        f"warm={sol1.obj}, cold={obj_cold}")
    assert abs(sol1.obj - obj0) > 1e-9, "obj should change after update"


def test_update_param_obj_skipped_no_corruption() -> None:
    """When ``Sum(...)`` over a Param's dim collapses it, the engine
    drops tracking for that Param on the affected term — calling
    ``update_param`` for it then reaches no LP cells and silently
    no-ops on those terms.  This is correct behaviour: callers who
    need objective coef updates use ``update_obj_coef`` (which
    overwrites the cost vector).  The test verifies that misuse
    does NOT corrupt the LP — the warm solve still matches a cold
    rebuild WITH THE OLD VALUES (because the price update never
    landed)."""
    cap = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [10.0, 10.0]}),
                name="p_cap")
    price = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [3.0, 4.0]}),
                  name="p_price")
    dur = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [2.0, 2.0]}),
                name="p_dur")

    p = _toy_problem(cap, price, dur)
    wp = WarmProblem(p)
    wp.declare_mutable("p_price")  # tracked, but Sum eats the dim
    wp.solve()

    new_price = Param(("t",),
                      pl.DataFrame({"t": [0, 1], "value": [10.0, 1.0]}),
                      name="p_price")
    # Should not corrupt the LP — but won't actually update obj coefs.
    wp.update_param("p_price", new_price)
    sol = wp.solve()

    obj_cold_old_price = _solve_cold(cap, price, dur)
    assert abs(sol.obj - obj_cold_old_price) < 1e-9, (
        f"warm={sol.obj} should still match cold-with-old-price "
        f"{obj_cold_old_price} since Sum eats the dim")


def test_update_param_two_param_chain() -> None:
    """Tests that auto-update propagates correctly when a tracked
    Param is multiplied by ANOTHER Param before reaching a Var.
    Constraint:  v[s, t] * (p_eff[s, t] * p_avail[s, t]) ≤ p_cap[s, t]
    The LHS coef is the composite ``p_eff * p_avail``; updating only
    ``p_eff`` should re-derive each cell as ``p_avail × new_p_eff``.
    """
    s_t = pl.DataFrame({
        "s": ["s1", "s1", "s2", "s2"],
        "t": [0, 1, 0, 1],
    })
    cap = Param(("s", "t"),
                s_t.with_columns(value=pl.Series([1.5, 1.5, 1.5, 1.5])),
                name="p_cap")
    price = Param(("s", "t"),
                  s_t.with_columns(value=pl.Series([2.0, 3.0, 1.0, 1.5])),
                  name="p_price")
    eff = Param(("s", "t"),
                s_t.with_columns(value=pl.Series([1.0, 1.0, 1.0, 1.0])),
                name="p_eff")
    avail = Param(("s", "t"),
                  s_t.with_columns(value=pl.Series([0.9, 0.8, 1.0, 0.5])),
                  name="p_avail")

    p = Problem()
    v = p.add_var("v", ("s", "t"), s_t, lower=0.0)
    # cap_eq:  v[s, t] * (p_eff * p_avail)  ≤  p_cap[s, t]
    p.add_cstr("cap_eq", over=s_t, sense="<=",
               lhs_terms={"v_x": v * (eff * avail)},
               rhs_terms={"cap": cap})
    p.add_cstr("demand_eq", over=None, sense=">=",
               lhs_terms={"v_sum": Sum(v)},
               rhs_terms={"target": 3.0})
    p.set_objective(Sum(v * price), sense="min")

    wp = WarmProblem(p)
    wp.declare_mutable("p_eff")
    sol0 = wp.solve()

    new_eff = Param(("s", "t"),
                    s_t.with_columns(value=pl.Series([0.5, 0.7, 1.5, 1.2])),
                    name="p_eff")
    wp.update_param("p_eff", new_eff)
    sol = wp.solve()

    # Cold rebuild with new eff.
    p_cold = Problem()
    v_cold = p_cold.add_var("v", ("s", "t"), s_t, lower=0.0)
    p_cold.add_cstr("cap_eq", over=s_t, sense="<=",
                    lhs_terms={"v_x": v_cold * (new_eff * avail)},
                    rhs_terms={"cap": cap})
    p_cold.add_cstr("demand_eq", over=None, sense=">=",
                    lhs_terms={"v_sum": Sum(v_cold)},
                    rhs_terms={"target": 3.0})
    p_cold.set_objective(Sum(v_cold * price), sense="min")
    sol_cold = p_cold.solve()
    assert abs(sol.obj - sol_cold.obj) < 1e-9, (
        f"warm={sol.obj}, cold={sol_cold.obj}")
    assert abs(sol.obj - sol0.obj) > 1e-9


def test_undeclared_update_raises() -> None:
    """``update_param("p_x", ...)`` for a Param that wasn't declared
    mutable BEFORE the first solve must raise a clear error rather
    than silently no-op."""
    cap = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [10.0, 10.0]}),
                name="p_cap")
    price = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [3.0, 4.0]}),
                  name="p_price")
    dur = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [1.0, 1.0]}),
                name="p_dur")

    p = _toy_problem(cap, price, dur)
    wp = WarmProblem(p)
    # NOTE: deliberately did NOT call declare_mutable.
    wp.solve()
    with pytest.raises(ValueError, match="not declared mutable"):
        wp.update_param("p_price", price)


def test_declare_mutable_after_build_raises() -> None:
    """``declare_mutable`` must be called BEFORE the first solve;
    after the LP is built the tracking state is fixed."""
    cap = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [10.0, 10.0]}),
                name="p_cap")
    price = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [3.0, 4.0]}),
                  name="p_price")
    dur = Param(("t",), pl.DataFrame({"t": [0, 1], "value": [1.0, 1.0]}),
                name="p_dur")

    p = _toy_problem(cap, price, dur)
    wp = WarmProblem(p)
    wp.solve()
    with pytest.raises(RuntimeError, match="already been built"):
        wp.declare_mutable("p_price")


def test_param_cells_size_bound_on_real_fixture(scenario_workdir) -> None:
    """Sanity check that the side-table for the standard mutable-Param
    set on a feature-heavy fixture stays under a generous size budget.
    Acts as an early-warning for accidentally tracking a Param family
    that explodes the per-cell map."""
    from flextool.engine_polars.input import load_flextool
    from flextool.engine_polars.model import build_flextool
    from flextool.engine_polars.chain import _MUTABLE_PARAMS

    data_dir = scenario_workdir("test_a_lot")

    data = load_flextool(data_dir)
    pb = Problem()
    build_flextool(pb, data)
    wp = WarmProblem(pb)
    wp.declare_mutable(*_MUTABLE_PARAMS)
    wp.solve()

    total_cells = sum(c["rows"].size for c in wp._param_cells.values())
    # Generous bound — handoff doc estimates ~50k–500k for typical chains.
    assert total_cells < 200_000, (
        f"tracked-cell count {total_cells} exceeds 200k budget; "
        f"per-Param: { {n: c['rows'].size for n, c in wp._param_cells.items()} }"
    )
