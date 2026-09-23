"""Slice D — Phase D5 gate: hand-calculated per-scenario invest objective.

Design: ``specs/sliceD_recourse_invest_design.md`` §15.1 (variant A) / §18 D5.

Fixture ``stoch_two_period_invest.json`` (built via
``build_stoch_two_period_invest.py``): two 3-step periods, branches
rlz w=0.25 / low w=0.75, deterministic demand 100/130 MW, an existing
wind unit whose branch-varying ``upper_limit`` profile (rlz 0.40/0.40,
low 0.20/0.10 through the ``group__unit`` stochastic group) leaves the
investable ``peaker`` (unitsize 1, invest_cost 1, discount_rate 0.05,
lifetime 1 → annuity 1050 exactly; unbounded-forward windows 2100/1050)
residual demand rlz 60/90, low 80/120.  Under ``recourse`` the invest
axis fans, each leaf invests its own capacity, and the objective is the
probability-weighted expectation:

    0.25·(60·2100 + 30·1050) + 0.75·(80·2100 + 40·1050) = 196 875.

Verified before pinning (design §15.1 verify-before-pin):
``ed_entity_annual_discounted[peaker, ·]`` = 2100/2100/1050/1050 and the
LP optimum is (60, 80, 30, 40) — empirically confirmed on the cascade
before these asserts were written.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

OBJECTIVE = 196_875.0
SOLVE = "stoch_2p_inv"

# ed_entity_annual_discounted[peaker, d] (§15.1 step 2).
EXPECTED_ANNU = {
    "p2035": 2100.0, "p2040": 1050.0,
    "p2035_low": 2100.0, "p2040_low": 1050.0,
}
EXPECTED_WEIGHT = {
    "p2035": 0.25, "p2040": 0.25, "p2035_low": 0.75, "p2040_low": 0.75,
}
# Objective coefficient of v_invest[peaker, d] = unitsize·annu[d]·weight[d]
# (§15.2 D4): 525 / 262.5 / 1575 / 787.5.
EXPECTED_COEFF = {d: EXPECTED_ANNU[d] * EXPECTED_WEIGHT[d]
                  for d in EXPECTED_ANNU}
# Per-leaf optimal invest (§15.1 step 3).
EXPECTED_INVEST = {
    "p2035": 60.0, "p2040": 30.0, "p2035_low": 80.0, "p2040_low": 40.0,
}


@pytest.fixture(scope="module")
def recourse_wf(scenario_workdir):
    """Cascade run of the recourse fixture under STRICT autoscale (a
    missing registry row would be loud — CLAUDE.md invariant 1)."""
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("recourse",
                                db_fixture="stoch_two_period_invest")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def _resolved(recourse_wf):
    """In-process re-solve off the snapshot workdir: flex_data + solution
    (gives the BRANCH v_invest columns, which the committed parquet
    correctly omits per design §11.1)."""
    from polar_high import Problem
    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(recourse_wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, sol


# ---------------------------------------------------------------------------
# The D5 gate — hand-calc objective (§15.1)
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_recourse_objective_hand_calc(recourse_wf):
    obj = float(
        pl.read_parquet(recourse_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(OBJECTIVE, rel=1e-9)


@pytest.mark.solver
def test_recourse_per_leaf_invest(_resolved):
    """§15.2 D2: v_invest carries all four (peaker, d) columns; each leaf
    invests its own §15.1 quantities (60/30 realized, 80/40 low)."""
    _data, sol = _resolved
    vi = sol.value("v_invest_p")
    got = {r["d"]: float(r["value"]) for r in vi.iter_rows(named=True)}
    assert got == pytest.approx(EXPECTED_INVEST)


@pytest.mark.solver
def test_recourse_objective_coefficients(_resolved):
    """§15.2 D4: the objective coefficient of each v_invest[peaker, d] is
    unitsize·annu[d]·pd_branch_weight[d] = 525 / 262.5 / 1575 / 787.5.
    Asserted on the model inputs (annu, weight, unitsize=1) — with the
    §15.1 objective equality pinning the LP-level product end-to-end."""
    data, _sol = _resolved
    annu = {str(r["e"]): None for r in []}  # noqa: F841 — clarity below
    annu = {str(r["d"]): float(r["value"])
            for r in data.ed_entity_annual_discounted.frame
                        .iter_rows(named=True)}
    assert annu == pytest.approx(EXPECTED_ANNU)
    wts = {str(r["d"]): float(r["value"])
           for r in data.pd_branch_weight.frame.iter_rows(named=True)}
    for d, w in EXPECTED_WEIGHT.items():
        assert wts[d] == pytest.approx(w, abs=1e-12)
    coeff = {d: annu[d] * wts[d] for d in annu}
    assert coeff == pytest.approx(EXPECTED_COEFF)
    assert data.recourse_invest is True


@pytest.mark.solver
def test_recourse_edd_lineage_filtered(_resolved):
    """§15.2 D3 (the resolved Slice B W6 witness): edd_invest_set pairs
    only same-leaf (d_invest, d); the cross-leaf pairs are ABSENT."""
    data, _sol = _resolved
    rows = {(str(r["d_invest"]), str(r["d"]))
            for r in data.edd_invest_set.iter_rows(named=True)}
    assert rows == {
        ("p2035", "p2035"), ("p2035", "p2040"),
        ("p2035_low", "p2035_low"), ("p2035_low", "p2040_low"),
        ("p2040", "p2040"), ("p2040_low", "p2040_low"),
    }


@pytest.mark.solver
def test_recourse_committed_output_realized_only(recourse_wf):
    """§11.1: the committed v_invest output carries ONLY the realized
    periods (real names) — branch rows are horizon-output-only (D8)."""
    inv = pl.read_parquet(
        recourse_wf / "output_raw" / f"v_invest__{SOLVE}.parquet")
    got = {r["period"]: round(float(r["peaker"]), 6)
           for r in inv.iter_rows(named=True)}
    assert got == {"p2035": 60.0, "p2040": 30.0}


@pytest.fixture(scope="module")
def horizon_wf(scenario_workdir):
    """Cascade run of the recourse fixture with ``model.output_horizon``
    enabled — the non-realized branch invest rows must surface (§11.2)."""
    return scenario_workdir("recourse_horizon",
                            db_fixture="stoch_two_period_invest")


@pytest.mark.solver
def test_recourse_output_horizon_branch_rows(horizon_wf):
    """§11.2 / §15.5 (D8): under ``output_horizon`` the invest-axis output
    surfaces the non-realized branch rows (80 at p2035_low, 40 at
    p2040_low) alongside the committed realized rows (60 / 30)."""
    inv = pl.read_parquet(
        horizon_wf / "output_raw" / f"v_invest__{SOLVE}.parquet")
    got = {r["period"]: round(float(r["peaker"]), 6)
           for r in inv.iter_rows(named=True)}
    assert got.get("p2035") == 60.0
    assert got.get("p2040") == 30.0
    assert got.get("p2035_low") == 80.0
    assert got.get("p2040_low") == 40.0


@pytest.mark.solver
def test_recourse_per_path_cap_two_leaf_rows(_resolved):
    """§15.1 / §7.2 (D6): under recourse the entity total cap fans into
    ONE row per scenario leaf (peaker × {__realized, low}) under the
    ``…_path`` name — the legacy single-row ``maxInvest_entity_total``
    (which summed all four periods, 210 > 150, and would spuriously
    bind) is gone.  Each leaf row sums only its two periods; the D2
    per-leaf-invest test pins those sums to 90 (60+30) and 120 (80+40),
    both ≤ 150 → non-binding, hence objective stays 196 875.  The naive
    all-``d`` sum 210 asymmetry is the proof the caps are per-path."""
    data, _sol = _resolved
    from flextool.engine_polars import build_flextool
    from polar_high import Problem

    pb = Problem()
    build_flextool(pb, data)
    names = pb.cstr_names()
    assert "maxInvest_entity_total_path" in names
    assert "maxInvest_entity_total" not in names  # legacy replaced
    recs = pb.cstrs_named("maxInvest_entity_total_path")
    over = recs[0].over
    assert over.height == 2  # (peaker, __realized), (peaker, low)
    leaves = sorted(str(v) for v in over["leaf"].to_list())
    assert leaves == ["__realized", "low"]
    ents = {str(v) for v in over["p"].to_list()}
    assert ents == {"peaker"}


@pytest.mark.solver
def test_recourse_handoff_drops_non_realized(_resolved, recourse_wf, caplog):
    """§10.1 (D7): the handoff pre-filter drops branch-named v_invest rows
    (non-realized scenario) BEFORE the commit loop, debug-logging the
    dropped mass (low leaf 80 + 40 = 120), and commits only real periods
    (60 / 30).  Guard 3 stays inert (no raise)."""
    import logging

    data, sol = _resolved
    from flextool.engine_polars.input import build_handoff_from_solution

    with caplog.at_level(logging.DEBUG,
                         logger="flextool.engine_polars.input"):
        handoff = build_handoff_from_solution(
            sol, recourse_wf, SOLVE, flex_data=data)
    assert handoff is not None
    ri = handoff.realized_invest
    periods = {str(r["period"]) for r in ri.iter_rows(named=True)
               if float(r["value"]) > 0}
    assert periods == {"p2035", "p2040"}
    assert not (periods & {"p2035_low", "p2040_low"})
    # The pre-filter debug-logged the dropped low-branch mass = 120.
    msgs = [r.getMessage() for r in caplog.records
            if "non-realized invest" in r.getMessage()]
    assert msgs, "pre-filter debug log not emitted"
    assert "120" in msgs[0]
    assert "p2035_low" in msgs[0] and "p2040_low" in msgs[0]


@pytest.mark.solver
def test_recourse_no_invest_na_constraints(_resolved):
    """§13 (K): the fan starts at the solve's first step, so
    pd_non_anticipativity is empty — no invest-NA constraint exists."""
    data, _sol = _resolved
    pdna = data.pd_non_anticipativity
    assert pdna is None or pdna.height == 0
