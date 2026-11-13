"""Invest-NA window (v72) — the standard two-stage stochastic-investment
mode: investment non-anticipative (a single shared here-and-now decision)
over the first period(s), per-branch (recourse) investment later,
operations free per scenario from t0.

Design: ``specs/invest_na_window_design.md`` (Revision 2).

The new solve param ``shared_invest_periods`` ties
``v_invest`` / ``v_divest`` across stochastic branches over the resolved
window.  Composed with ``stochastic_invest_method=recourse`` (per-branch
invest columns to tie) and ``shared_operation_periods=[]`` (ops free) it
delivers the main mode.

Fixture ``stoch_two_period_hedge.json`` (built via
``build_stoch_two_period_hedge.py``) re-uses the Slice E hedge topology
with the branches FANNING AT t0 (the ``ws`` alternative), plus:

  * ``hedge_na_window`` — fan-at-t0 + ``shared_invest_periods
    =[p2035]`` + ``shared_operation_periods=[]``.  The tie reproduces
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

from tests._support.load import load_scenario

SOLVE = "stoch_2p_hedge"

RP = 173_250.0    # main mode (window = [p2035]) — shared first stage
WS = 157_500.0    # wait-and-see control (window unset)
EVPI = RP - WS    # = 15 750


# ---------------------------------------------------------------------------
# Workdir fixtures (full cascade) — mirror test_recourse_invest_hedge.py
# ---------------------------------------------------------------------------


def _strict(scenario_run, name):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_run(name, db_fixture="stoch_two_period_hedge")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def na_window_run(scenario_run):
    return _strict(scenario_run, "hedge_na_window")


@pytest.fixture(scope="module")
def na_ws_run(scenario_run):
    return scenario_run("hedge_ws", db_fixture="stoch_two_period_hedge")


@pytest.fixture(scope="module")
def na_divest_run(scenario_run):
    return _strict(scenario_run, "hedge_na_divest")


@pytest.fixture(scope="module")
def na_none_run(scenario_run):
    return _strict(scenario_run, "hedge_na_none")


@pytest.fixture(scope="module")
def na_window_wf(na_window_run):
    """The cascade's work folder (``output_raw`` parquets)."""
    return na_window_run.work_folder


@pytest.fixture(scope="module")
def na_ws_wf(na_ws_run):
    """The cascade's work folder (``output_raw`` parquets)."""
    return na_ws_run.work_folder


def _resolve(run):
    """In-process re-solve of the run's last sub-solve, loaded through the
    production provider path — gives the branch v_invest / v_divest
    columns the committed parquet omits."""
    from polar_high import Problem

    from flextool.engine import build_flextool

    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, sol, pb


@pytest.fixture(scope="module")
def na_window_resolved(na_window_run):
    return _resolve(na_window_run)


@pytest.fixture(scope="module")
def na_divest_resolved(na_divest_run):
    return _resolve(na_divest_run)


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
    """T-Main (§9.2), R3 re-grounded: the first-stage p2035 v_invest is
    EQUAL across the two forecast branches (shared first stage = 60:
    p2035_high == p2035_low), while p2040 diverges (recourse: r_high = 60
    at p2040_high, r_low = 0 at p2040_low).  Under recourse the period
    labels p2035 / p2040 have no invest column at all."""
    _data, sol, _pb = na_window_resolved
    vi = sol.value("v_invest_p")
    got = {r["d"]: round(float(r["value"]), 4)
           for r in vi.iter_rows(named=True)}
    assert got["p2035_high"] == pytest.approx(60.0)  # shared first stage
    assert got["p2035_low"] == pytest.approx(60.0)   # shared first stage
    assert got["p2035_high"] == got["p2035_low"]
    assert got["p2040_high"] == pytest.approx(60.0)   # recourse (high)
    assert got["p2040_low"] == pytest.approx(0.0)     # recourse (low)
    assert set(got) == {"p2035_high", "p2035_low", "p2040_high", "p2040_low"}


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
    """T-Frame (§3.1): pd_non_anticipativity is the forecast star of the
    p2035 fan — the first sorted forecast p2035_high is the reference and
    every other forecast is tied to it: {(p2035_high, p2035_low)}, members
    − 1 = 1 pair (the period label p2035 has no invest column under
    recourse).  p2040 is NOT in the window so its forecasts are untied."""
    data, _sol, _pb = na_window_resolved
    pdna = data.pd_non_anticipativity
    pairs = {(str(r["d"]), str(r["b"]))
             for r in pdna.iter_rows(named=True)}
    assert pairs == {("p2035_high", "p2035_low")}


def test_frame_empty_when_window_unset(na_ws_run) -> None:
    """T-Frame / T-P (§6.4): with the window unset the frame is EMPTY
    (byte-parity legacy path) — no CSV emitted, tie_periods None."""
    data = load_scenario(na_ws_run)
    pdna = data.pd_non_anticipativity
    assert pdna is None or pdna.height == 0


@pytest.mark.solver
def test_branch_weights_partition(na_window_resolved) -> None:
    """T-Weight (F7, §9.4), R3 re-grounded: under forecasts-only the tied
    cohort's weights are a probability partition over the FORECASTS —
    pd_branch_weight[p2035_high] = 0.25, [p2035_low] = 0.75, summing to
    1.0 across the fan (so the shared decision is costed once at full
    weight — the Option-B equivalence arithmetic).  The real-named trunk
    p2035 is a weight-0 phantom."""
    data, _sol, _pb = na_window_resolved
    df = data.pd_branch_weight.frame
    val_col = [c for c in df.columns if c != "d"][-1]
    w = {str(r["d"]): round(float(r[val_col]), 6)
         for r in df.iter_rows(named=True)}
    assert w["p2035"] == pytest.approx(0.0)
    assert w["p2035_high"] == pytest.approx(0.25)
    assert w["p2035_low"] == pytest.approx(0.75)


@pytest.mark.solver
def test_invest_constraint_rows(na_window_resolved) -> None:
    """T-C (§9.4): non_anticipativity_invest_p ties the p2035 forecasts
    across the fan — one row (base, p2035_high, p2035_low): the reference
    forecast p2035_high to the other forecast; NO row over p2040."""
    _data, _sol, pb = na_window_resolved
    names = pb.cstr_names()
    assert "non_anticipativity_invest_p" in names
    over = pb.cstrs_named("non_anticipativity_invest_p")[0].over
    triples = {(str(r["p"]), str(r["d"]), str(r["b"]))
               for r in over.iter_rows(named=True)}
    assert triples == {("base", "p2035_high", "p2035_low")}


# --- T-Divest (solver, MANDATORY — F2) -------------------------------------


@pytest.mark.solver
def test_divest_family_ties_first_period(na_divest_resolved) -> None:
    """T-Divest (§4.4, F2): a divest-eligible base under recourse+window →
    non_anticipativity_divest_p ties the first-period divest across the
    two forecasts — one row (base, p2035_high, p2035_low)."""
    _data, _sol, pb = na_divest_resolved
    names = pb.cstr_names()
    assert "non_anticipativity_divest_p" in names
    over = pb.cstrs_named("non_anticipativity_divest_p")[0].over
    triples = {(str(r["p"]), str(r["d"]), str(r["b"]))
               for r in over.iter_rows(named=True)}
    assert triples == {("base", "p2035_high", "p2035_low")}


@pytest.mark.solver
def test_divest_net_capacity_equal_across_branches(na_divest_resolved) -> None:
    """T-Divest (§4.4), R3 re-grounded: the NET first-stage capacity
    (existing + invest − divest) is EQUAL across the two forecast branches
    at p2035 — the tie shares the net decision, not just gross invest."""
    _data, sol, _pb = na_divest_resolved
    inv = {r["d"]: float(r["value"])
           for r in sol.value("v_invest_p").iter_rows(named=True)}
    div = {r["d"]: float(r["value"])
           for r in sol.value("v_divest_p").iter_rows(named=True)}
    existing = 50.0
    net_high = (existing + inv.get("p2035_high", 0.0)
                - div.get("p2035_high", 0.0))
    net_low = (existing + inv.get("p2035_low", 0.0)
               - div.get("p2035_low", 0.0))
    assert net_high == pytest.approx(net_low, abs=1e-6)


# --- T-None (graceful no-op) -----------------------------------------------


@pytest.mark.solver
def test_none_graceful_noop(na_none_run) -> None:
    """T-None (§4.5): the window set on a stochastic_invest_method=none
    solve emits ZERO non_anticipativity_invest_* rows (investment is
    already shared across all periods — the domain guard / recourse gate
    makes it inert)."""
    from polar_high import Problem

    from flextool.engine import build_flextool

    data = load_scenario(na_none_run)
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
    from flextool.engine.autoscale._layer2_types import (
        CONSTRAINT_FAMILIES,
    )

    for name in ("non_anticipativity_invest_p", "non_anticipativity_invest_n",
                 "non_anticipativity_divest_p", "non_anticipativity_divest_n"):
        assert name in CONSTRAINT_FAMILIES


# ---------------------------------------------------------------------------
# Investment marginals of a recourse fan (item 266 review F1)
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_recourse_fan_invest_marginals_stay_on_forecast_branches(tmp_path, monkeypatch) -> None:
    """Under ``stochastic_invest_method = recourse`` the fan's ``v_invest`` columns are on the forecast branch
    periods, so its reduced costs are read on the dispatch axis, not the investment axis with the fan's period
    labels (item 266 read the investment duals on the investment axis; on a fan that put 0 rows at the labels
    p2035 / p2040, dropped the p2040_low reduced cost and wrote a ``unit_invest_marginal__d`` of -0 rows with an
    empty solve label).  1050 is the p2040_low reduced cost the parquet held before item 266."""
    from pathlib import Path

    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs
    from flextool.update_flextool.db_migration import migrate_database
    from tests.db_utils import json_to_db

    repo = Path(__file__).resolve().parents[2]
    db_url = json_to_db(repo / "tests" / "fixtures" / "stoch_two_period_hedge.json", tmp_path / "db.sqlite")
    migrate_database(db_url)
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(tmp_path)
    wf = tmp_path / "w"
    steps = run_chain_from_db(db_url, "hedge_na_window", work_folder=wf, csv_dump=True, keep_solutions=True)
    last = next(reversed(steps.values()))
    last.flex_data_provider.snapshot_processed_inputs(wf)
    write_outputs(
        scenario_name="hedge_na_window", output_location=str(wf), subdir="out",
        output_config_path=str(repo / "flextool" / "schemas" / "default_plots.yaml"),
        write_methods=["csv"], fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider(),
    )

    duals = pl.read_parquet(wf / "output_raw" / f"v_dual_invest_unit__{SOLVE}.parquet")
    by_period = dict(zip(duals["period"], duals["base"]))
    assert by_period["p2040_low"] == pytest.approx(1050.0, rel=1e-6), duals
    # No 0-filled rows at the fan's period labels.
    assert not {"p2035", "p2040"} & set(by_period), duals
    # Nothing is invested, and no marginal table of -0 rows with an empty solve label is written.
    csv_dir = wf / "output_csv" / "out"
    capacity = pl.read_csv(csv_dir / "unit_capacity__d.csv").filter(pl.col("unit") == "base")
    assert capacity["invested"].null_count() == capacity.height, capacity
    assert not (csv_dir / "unit_invest_marginal__d.csv").exists()
