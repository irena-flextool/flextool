"""Slice D — variant B (cross-scenario coupling) structural tests.

Design: ``specs/sliceD_recourse_invest_design.md`` §15.6.

Fixture ``stoch_two_period_invest_na.json`` (built via
``build_stoch_two_period_invest_na.py``): the proven ``stochastics.json``
storage shape (``2_day_stochastic_dispatch``) extended with a
``recourse_na`` scenario that makes the stochastic-group storage node
``hydro_reservoir`` investable under ``recourse``.  Unlike variant A the
LP is NOT separable: a stochastic-group NODE fires the dispatch
non-anticipativity net-charge pinning (``non_anticipativity_storage_use``).
"""
from __future__ import annotations

import os

import pytest

SOLVE = "2day_dispatch"


@pytest.fixture(scope="module")
def na_wf(scenario_workdir):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("recourse_na",
                                db_fixture="stoch_two_period_invest_na")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def _built(na_wf):
    from polar_high import Problem
    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(na_wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, pb, sol


@pytest.mark.solver
def test_na_coupling_active(_built):
    """§15.6 witness 1: the storage non-anticipativity family fires (the
    flag-on LP is NOT separable per leaf)."""
    data, pb, _sol = _built
    assert data.dt_non_anticipativity is not None
    assert data.dt_non_anticipativity.height > 0
    recs = pb.cstrs_named("non_anticipativity_storage_use")
    n_rows = sum(r.over.height for r in recs)
    assert n_rows > 0


@pytest.mark.solver
def test_na_recourse_active(_built):
    data, _pb, _sol = _built
    assert data.recourse_invest is True


@pytest.mark.solver
def test_na_node_invest_coefficient_branch_weighted(_built):
    """§15.6 witness 3 (the F3 node-side regression hook): the objective
    coefficient of ``v_invest_n[hydro_reservoir, d]`` equals
    ``unitsize × annu_n[d] × pd_branch_weight[d]`` — i.e. the node-side
    annuity carries the per-period branch weight."""
    data, pb, _sol = _built
    m = pb.canonicalise()

    # annu_n[d] and pd_branch_weight[d] for hydro_reservoir.
    annu = {str(r["d"]): float(r["value"])
            for r in data.ed_entity_annual_discounted.frame.iter_rows(named=True)
            if str(r["e"]) == "hydro_reservoir"}
    wts = {str(r["d"]): float(r["value"])
           for r in data.pd_branch_weight.frame.iter_rows(named=True)}
    us_frame = data.p_state_unitsize.frame
    us = {str(r[us_frame.columns[0]]): float(r["value"])
          for r in us_frame.iter_rows(named=True)}
    unit = us.get("hydro_reservoir", 1.0)

    # Objective coefficients keyed by v_invest_n column name.
    obj = {name: float(c)
           for name, c in zip(m.col_names, m.col_obj)
           if name.startswith("v_invest_n[")}
    assert obj, "no v_invest_n columns in the LP"

    # Parse the period token out of each column name and check the
    # coefficient against unitsize × annu_n × weight.
    checked = 0
    for name, coeff in obj.items():
        inner = name[len("v_invest_n["):-1]
        parts = [p.strip() for p in inner.split(",")]
        if parts[0] != "hydro_reservoir":
            continue
        period = parts[-1]
        if period not in annu or period not in wts:
            continue
        expected = unit * annu[period] * wts[period]
        assert coeff == pytest.approx(expected, rel=1e-6), (
            f"{name}: coeff {coeff} != unitsize({unit}) × "
            f"annu_n({annu[period]}) × weight({wts[period]}) = {expected}"
        )
        checked += 1
    assert checked >= 2, (
        f"expected ≥2 weighted v_invest_n[hydro_reservoir, ·] coefficients, "
        f"checked {checked}")

    # At least one weighted period carries a genuine (non-unit) factor —
    # guards against an unweighted regression that would leave every
    # coefficient == unitsize × annu_n (pd_branch_weight normalises the
    # equal-weight branch cohort to < 1 per leaf).
    weight_vals = {wts[p] for p in annu if p in wts}
    assert any(abs(w - 1.0) > 1e-9 for w in weight_vals), (
        f"branch weights all 1.0 ({weight_vals}) — the coefficient check "
        "would not discriminate an unweighted regression")


@pytest.mark.solver
def test_na_branch_state_equals_realized(_built):
    """§15.6 witness 2: at the NA-tied timesteps the branch-leaf
    ``v_state`` equals the realized value (coupling active in the
    solution)."""
    data, _pb, sol = _built
    if "v_state" not in sol._vars:
        pytest.skip("fixture has no v_state variable")
    vs = sol.value("v_state")
    # Group v_state by (node, time) across branch periods; every period
    # sharing a (node, time) NA tie must carry the same value.
    from collections import defaultdict

    tol = 1e-6
    # dt_non_anticipativity carries the (d, t) tie set; the storage NA
    # equates the net-charge, so branch v_state at those (n, t) equals the
    # realized value.  We assert the weaker, robust invariant: for each
    # (node, time), the spread across the periods present is within tol.
    by_nt: dict[tuple[str, str], list[float]] = defaultdict(list)
    cols = vs.columns
    ncol = "n" if "n" in cols else ("node" if "node" in cols else cols[0])
    tcol = "t" if "t" in cols else ("time" if "time" in cols else None)
    if tcol is None:
        pytest.skip("v_state frame has no time column")
    for r in vs.iter_rows(named=True):
        by_nt[(str(r[ncol]), str(r[tcol]))].append(float(r["value"]))
    coupled = 0
    for (_n, _t), vals in by_nt.items():
        if len(vals) > 1:
            assert max(vals) - min(vals) <= tol + 1e-6 * max(1.0, max(abs(v) for v in vals))
            coupled += 1
    assert coupled > 0, "no shared (node, time) v_state rows to witness coupling"


@pytest.mark.solver
def test_na_preconditions_clean(_built):
    """§15.6 witness 4: recourse NPV preconditions hold on this fixture.

    ``load_flextool`` invokes ``assert_recourse_npv_preconditions`` at the
    lineage-activation boundaries (D4); a violation would have raised
    during the cascade.  The ``_built`` fixture reaching an optimal solve
    is therefore the end-to-end precondition witness."""
    data, _pb, sol = _built
    assert sol.optimal
    assert data.recourse_invest is True
