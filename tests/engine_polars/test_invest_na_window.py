"""Invest-NA window (v72) — the standard two-stage stochastic-investment
mode: investment non-anticipative (a single shared here-and-now decision)
over the first period(s), per-branch (recourse) investment later,
operations free per scenario from t0.

Design: ``specs/invest_na_window_design.md`` (Revision 2).

The new solve param ``non_anticipativity_invest_periods`` ties
``v_invest`` / ``v_divest`` across stochastic branches over the resolved
window.  Composed with ``stochastic_invest_method=recourse`` (per-branch
invest columns to tie) and ``non_anticipativity_periods=[]`` (ops free) it
delivers the main mode.

Fixture ``stoch_two_period_hedge.json`` (built via
``build_stoch_two_period_hedge.py``) re-uses the Slice E hedge topology
with the branches FANNING AT t0 (the ``ws`` alternative), plus:

  * ``hedge_na_window`` — fan-at-t0 + ``non_anticipativity_invest_periods
    =[p2035]`` + ``non_anticipativity_periods=[]``.  The tie reproduces
    the Option-B hedge **RP = 173 250** BY CONSTRAINT (§9.3): p2035 invest
    shared across branches, p2040 invest per-branch (recourse).
  * ``hedge_ws`` — the same fan-at-t0 topology with the window UNSET —
    today's wait-and-see control, **WS = 157 500**.  ``RP − WS = 15 750``.
  * ``hedge_na_divest`` — ``base`` made ``invest_retire_total`` + existing
    50 so ``v_divest`` exists; the divest ties fire too (F2).
  * ``hedge_na_none`` — the window set on a ``stochastic_invest_method
    =none`` solve → graceful no-op (no invest-NA rows).
"""
from __future__ import annotations

import os

import polars as pl
import pytest

SOLVE = "stoch_2p_hedge"

RP = 173_250.0    # main mode (window = [p2035]) — shared first stage
WS = 157_500.0    # wait-and-see control (window unset)
EVPI = RP - WS    # = 15 750


# ---------------------------------------------------------------------------
# Workdir fixtures (full cascade) — mirror test_recourse_invest_hedge.py
# ---------------------------------------------------------------------------


def _strict(scenario_workdir, name):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir(name, db_fixture="stoch_two_period_hedge")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def na_window_wf(scenario_workdir):
    return _strict(scenario_workdir, "hedge_na_window")


@pytest.fixture(scope="module")
def na_ws_wf(scenario_workdir):
    return scenario_workdir("hedge_ws", db_fixture="stoch_two_period_hedge")


@pytest.fixture(scope="module")
def na_divest_wf(scenario_workdir):
    return _strict(scenario_workdir, "hedge_na_divest")


@pytest.fixture(scope="module")
def na_none_wf(scenario_workdir):
    return _strict(scenario_workdir, "hedge_na_none")


def _resolve(wf):
    """In-process re-solve off the snapshot workdir — gives the branch
    v_invest / v_divest columns the committed parquet omits."""
    from polar_high import Problem

    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, sol, pb


@pytest.fixture(scope="module")
def na_window_resolved(na_window_wf):
    return _resolve(na_window_wf)


@pytest.fixture(scope="module")
def na_divest_resolved(na_divest_wf):
    return _resolve(na_divest_wf)


# --- T-Main / T-WS: the hand-calc gate (§9.2, §9.4) ------------------------


@pytest.mark.solver
def test_main_objective_hand_calc(na_window_wf) -> None:
    """T-Main (§9.2): the main-mode objective is exactly 173 250 — the
    shared first stage costed once at full weight.  This is the FIRST test
    to fire the cross-Enum invest-NA guard join (§4.2, R3)."""
    obj = float(
        pl.read_parquet(na_window_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(RP, rel=1e-9)


@pytest.mark.solver
def test_main_invest_shared_first_stage(na_window_resolved) -> None:
    """T-Main (§9.2): p2035 v_invest is EQUAL across branches (shared
    first stage = 60), while p2040 diverges (recourse: r_rlz = 60,
    r_low = 0)."""
    _data, sol, _pb = na_window_resolved
    vi = sol.value("v_invest_p")
    got = {r["d"]: round(float(r["value"]), 4)
           for r in vi.iter_rows(named=True)}
    assert got["p2035"] == pytest.approx(60.0)
    assert got["p2035_low"] == pytest.approx(60.0)   # shared first stage
    assert got["p2035"] == got["p2035_low"]
    assert got["p2040"] == pytest.approx(60.0)        # recourse (high)
    assert got["p2040_low"] == pytest.approx(0.0)     # recourse (low)


@pytest.mark.solver
def test_ws_control_and_evpi_gap(na_ws_wf, na_window_wf) -> None:
    """T-WS (§9.2): the window-unset control is 157 500 (today's
    wait-and-see), strictly below RP.  The gap RP − WS = 15 750 proves
    the tie genuinely shares the first stage (a dropped tie collapses
    RP → WS)."""
    ws = float(
        pl.read_parquet(na_ws_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    rp = float(
        pl.read_parquet(na_window_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert ws == pytest.approx(WS, rel=1e-9)
    assert rp - ws == pytest.approx(EVPI, rel=1e-9)
    assert rp > ws


# --- T-Frame / T-Weight / T-C: structural (§9.4) ---------------------------


@pytest.mark.solver
def test_frame_window_membership(na_window_resolved) -> None:
    """T-Frame (§3.1): pd_non_anticipativity = {(p2035, p2035_low)} —
    the window ties only the p2035 anchor's synthetic copy; p2040 is
    NOT in the window so (p2040, p2040_low) is absent."""
    data, _sol, _pb = na_window_resolved
    pdna = data.pd_non_anticipativity
    pairs = {(str(r["d"]), str(r["b"]))
             for r in pdna.iter_rows(named=True)}
    assert pairs == {("p2035", "p2035_low")}


def test_frame_empty_when_window_unset(na_ws_wf) -> None:
    """T-Frame / T-P (§6.4): with the window unset the frame is EMPTY
    (byte-parity legacy path) — no CSV emitted, tie_periods None."""
    from flextool.engine_polars import load_flextool

    data = load_flextool(na_ws_wf)
    pdna = data.pd_non_anticipativity
    assert pdna is None or pdna.height == 0


@pytest.mark.solver
def test_branch_weights_partition(na_window_resolved) -> None:
    """T-Weight (F7, §9.4): the tied cohort's weights are a probability
    partition — pd_branch_weight[p2035] = 0.25, [p2035_low] = 0.75,
    summing to 1.0 across the tie (so the shared decision is costed once
    at full weight — the Option-B equivalence arithmetic)."""
    data, _sol, _pb = na_window_resolved
    df = data.pd_branch_weight.frame
    val_col = [c for c in df.columns if c != "d"][-1]
    w = {str(r["d"]): round(float(r[val_col]), 6)
         for r in df.iter_rows(named=True)}
    assert w["p2035"] == pytest.approx(0.25)
    assert w["p2035_low"] == pytest.approx(0.75)


@pytest.mark.solver
def test_invest_constraint_rows(na_window_resolved) -> None:
    """T-C (§9.4): non_anticipativity_invest_p ties (base, p2035,
    p2035_low); NO row over p2040."""
    _data, _sol, pb = na_window_resolved
    names = pb.cstr_names()
    assert "non_anticipativity_invest_p" in names
    over = pb.cstrs_named("non_anticipativity_invest_p")[0].over
    triples = {(str(r["p"]), str(r["d"]), str(r["b"]))
               for r in over.iter_rows(named=True)}
    assert triples == {("base", "p2035", "p2035_low")}


# --- T-Divest (solver, MANDATORY — F2) -------------------------------------


@pytest.mark.solver
def test_divest_family_ties_first_period(na_divest_resolved) -> None:
    """T-Divest (§4.4, F2): a divest-eligible base under recourse+window
    → non_anticipativity_divest_p ties the first-period divest over
    (base, p2035, p2035_low)."""
    _data, _sol, pb = na_divest_resolved
    names = pb.cstr_names()
    assert "non_anticipativity_divest_p" in names
    over = pb.cstrs_named("non_anticipativity_divest_p")[0].over
    triples = {(str(r["p"]), str(r["d"]), str(r["b"]))
               for r in over.iter_rows(named=True)}
    assert triples == {("base", "p2035", "p2035_low")}


@pytest.mark.solver
def test_divest_net_capacity_equal_across_branches(na_divest_resolved) -> None:
    """T-Divest (§4.4): the NET first-stage capacity
    (existing + invest − divest) is EQUAL across branches at p2035 —
    the tie shares the net decision, not just gross invest."""
    _data, sol, _pb = na_divest_resolved
    inv = {r["d"]: float(r["value"])
           for r in sol.value("v_invest_p").iter_rows(named=True)}
    div = {r["d"]: float(r["value"])
           for r in sol.value("v_divest_p").iter_rows(named=True)}
    existing = 50.0
    net_rlz = existing + inv.get("p2035", 0.0) - div.get("p2035", 0.0)
    net_low = existing + inv.get("p2035_low", 0.0) - div.get("p2035_low", 0.0)
    assert net_rlz == pytest.approx(net_low, abs=1e-6)


# --- T-None (graceful no-op) -----------------------------------------------


@pytest.mark.solver
def test_none_graceful_noop(na_none_wf) -> None:
    """T-None (§4.5): the window set on a stochastic_invest_method=none
    solve emits ZERO non_anticipativity_invest_* rows (investment is
    already shared across all periods — the domain guard / recourse gate
    makes it inert)."""
    from polar_high import Problem

    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(na_none_wf)
    pb = Problem()
    build_flextool(pb, data)
    invest_na = [n for n in pb.cstr_names()
                 if "non_anticipativity_invest" in n
                 or "non_anticipativity_divest" in n]
    assert invest_na == []


# --- T-Reg (autoscale) -----------------------------------------------------


def test_registry_has_four_invest_families() -> None:
    """T-Reg (§5): all four invest-NA family names are registered in
    CONSTRAINT_FAMILIES (a miss silently degrades the solve to an
    un-scaled LP — CLAUDE.md invariant 1)."""
    from flextool.engine_polars.autoscale._layer2_types import (
        CONSTRAINT_FAMILIES,
    )

    for name in ("non_anticipativity_invest_p", "non_anticipativity_invest_n",
                 "non_anticipativity_divest_p", "non_anticipativity_divest_n"):
        assert name in CONSTRAINT_FAMILIES
