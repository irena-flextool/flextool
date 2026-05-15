"""Surface B.5 / B.8 — ramp-rate × UC startup coupling and indirect
two-input source-flow-coef tradeoff.

Three focused tests:

* ``test_ramp_sink_up_with_uc_startup_tightening`` — covers **B.5.4**.
  Extends ``toy_uc_3t`` with ``ramp_speed_up_sink`` so tight that the
  jump from off (v_flow=0) to producing (v_flow=min_load) at t02 is
  forbidden by the bare ramp RHS; the UC startup tightening
  (``+ v_startup_lin`` on the RHS, model.py:1457-1467) is exactly what
  makes the optimum feasible.  Also asserts the constraint family is
  emitted with the expected row count.

* ``test_no_ramp_load_constraint_emitted`` — covers **B.5 structural
  pin**.  ``ramp_load_ratio`` exists in the legacy .mod but is NOT
  implemented in either polars engine or v3.32.0 (audit
  ``04_old_flextool_features.md``).  This test populates ramp params on
  the smallest fixture and verifies that no constraint name contains
  ``ramp_load`` — pinning the absence so a future port doesn't silently
  introduce a different naming convention.

* ``test_indirect_two_input_source_flow_coef_tradeoff`` — covers
  **B.8.3**.  Extends ``toy_2node_chp`` to have TWO fuel inputs
  (``coal`` and ``biomass``) with different prices and per-source
  ``source_flow_coef`` weights.  Hand-calc verifies the LP picks the
  cheaper specific-fuel: at the optimum the dispatch routes 100 % of
  the input balance through the lower ``price / coef`` source, with
  the other source carrying zero flow.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

from .conftest import solver_options


def _solve(data: FlexData) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# Helper: extend a FlexData with ramp_sink_up params on an arbitrary
# (p, source, sink) row of its process_source_sink frame.

def _add_ramp_sink_up(data: FlexData, *, p: str, source: str, sink: str,
                      speed: float) -> FlexData:
    idx = pl.DataFrame({"p": [p], "source": [source], "sink": [sink]})
    speed_param = Param(("p", "sink"),
        pl.DataFrame({"p": [p], "sink": [sink], "value": [speed]}))
    return dataclasses.replace(
        data,
        process_source_sink_ramp_limit_sink_up=idx,
        p_ramp_speed_up_sink=speed_param,
    )


# ---------------------------------------------------------------------------
# B.5.4 — Ramp + UC startup/shutdown coupling.

def test_ramp_sink_up_with_uc_startup_tightening(toy_uc_3t):
    """Covers B.5.4 — UC startup tightens the ramp_sink_up RHS.

    Hand-calc setup: take ``toy_uc_3t`` (1 unit, unitsize=100, min_load=0.4,
    existing_count=1, 3 cyclic timesteps).  Set demand profile so the
    cheapest schedule has the unit OFF at t01 and ON at t02-t03 producing
    at min_load (v_flow = 0.4).  Set ramp_speed_up_sink = 0.001 so the
    bare RHS = 0.001 · 60 · 1 · 1 = 0.06 forbids the jump from 0 to 0.4
    in one step.  The startup tightening ``+ v_startup_lin`` lifts the
    RHS by 1.0 (a startup at t02), letting v_flow rise to 0.4.

    Without the startup term in rhs_terms (model.py:1458-1462) the
    optimum would have to pay slack on every step — here it does not.
    """
    d = toy_uc_3t
    # Demand: 0 at t01 (no penalty if off), 40 at t02/t03 → flow=[0, 0.4, 0.4].
    nb_dt = d.nodeBalance_dt
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("t") == "t01").then(0.0).otherwise(-40.0)
    ).select("n", "d", "t", "value"))
    # Make t01 acyclic on the prev-edge: t_previous_within_solve[t01] = t01
    # so ramp_diff[t01] = 0 (eliminates spurious cyclic ramp from t03 → t01).
    dtttdt = pl.DataFrame({
        "d": ["d1"]*3,
        "t": ["t01", "t02", "t03"],
        "t_previous": ["t01", "t01", "t02"],
        "t_previous_within_timeset": ["t01", "t01", "t02"],
        "d_previous": ["d1"]*3,
        "t_previous_within_solve": ["t01", "t01", "t02"],
    })
    data = dataclasses.replace(d, p_inflow=p_inflow, dtttdt=dtttdt)
    data = _add_ramp_sink_up(data, p="u", source="FUEL_n", sink="n", speed=0.001)
    pb, sol = _solve(data)
    assert sol.optimal
    cstr = set(pb.cstr_names())
    assert "ramp_sink_up_constraint" in cstr
    # 1 ramp arc × 1 period × 3 steps = 3 rows.
    assert pb.cstr_row_count("ramp_sink_up_constraint") == 3
    # Hand-calc: v_flow = [0, 0.4, 0.4] (no slack burnt).
    v_flow = sol.value("v_flow").sort("t")["value"].to_list()
    assert v_flow[0] == pytest.approx(0.0, abs=1e-7)
    assert v_flow[1] == pytest.approx(0.4, rel=1e-7)
    assert v_flow[2] == pytest.approx(0.4, rel=1e-7)
    # Slack on the demand node must be zero — the startup term made the
    # ramp constraint feasible exactly at the demand jump.
    vq_up = sol.value("vq_state_up")
    assert vq_up["value"].sum() == pytest.approx(0.0, abs=1e-7)


# ---------------------------------------------------------------------------
# B.5 structural pin — ``ramp_load_ratio`` is intentionally NOT emitted.

def test_no_ramp_load_constraint_emitted(toy_uc_3t):
    """Covers B.5 structural finding — ``ramp_load_ratio`` from the
    legacy .mod is NOT implemented in the polars engine (verified in
    ``.claude/test-audit/04_old_flextool_features.md``).

    Build the smallest model that activates the ramp feature gate
    (``has_ramp`` = True) and assert that no constraint name contains
    ``ramp_load`` — pinning the absence so a port does not silently
    introduce a differently-named load-ratio constraint.
    """
    data = _add_ramp_sink_up(toy_uc_3t, p="u", source="FUEL_n", sink="n",
                             speed=10.0)
    pb, sol = _solve(data)
    assert sol.optimal
    assert all("ramp_load" not in n for n in pb.cstr_names())


# ---------------------------------------------------------------------------
# B.8.3 — Two-input source_flow_coef tradeoff.

def test_indirect_two_input_source_flow_coef_tradeoff(toy_2node_chp):
    """Covers B.8.3 — non-default ``p_process_source_flow_coef`` on a
    two-input indirect process picks the cheaper specific fuel.

    Setup: extend ``toy_2node_chp`` with TWO fuel inputs:
        coal:    price=2.0,  source_flow_coef = 1.0   ⇒ specific = 2.0/1.0 = 2.0
        biomass: price=2.5,  source_flow_coef = 2.0   ⇒ specific = 2.5/2.0 = 1.25
    Single timestep (drop t02 inflow); demand: heat=5, elec=10.  Sink
    coefs (from the fixture): heat=0.5, elec=1.0, slope=1.

    Hand-calc — LHS (input balance):
        v_in_coal · 100 · 1.0  +  v_in_bio · 100 · 2.0
            == (heat · 0.5 + elec · 1.0) · slope
            ==  (5·0.5 + 10·1.0) · 1.0  =  12.5
    Cost-min: the cheaper specific fuel (biomass, 1.25 €/MWh-equiv)
    carries everything ⇒ v_in_bio = 12.5 / (100·2.0) = 0.0625, coal = 0.
    """
    d = toy_2node_chp
    # Replace the input arc with TWO inputs (coal, biomass) into chp.
    pss = pl.DataFrame({
        "p":      ["chp", "chp", "chp", "chp"],
        "source": ["coal", "biomass", "chp", "chp"],
        "sink":   ["chp", "chp", "heat", "elec"],
    })
    pss_eff = pss.clone()
    pss_dt = pss.join(d.dt, how="cross")
    flow_to_n = (pss.filter(pl.col("sink").is_in(["heat", "elec"]))
                 .with_columns(n=pl.col("sink")))
    flow_from_commodity_eff = pl.DataFrame({
        "p": ["chp", "chp"], "source": ["coal", "biomass"],
        "sink": ["chp", "chp"], "c": ["COAL", "BIO"]})
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame({"c": ["COAL", "COAL", "BIO", "BIO"],
                      "d": ["d1"]*4, "t": ["t01", "t02"]*2,
                      "value": [2.0, 2.0, 2.5, 2.5]}))
    process_input_flows = pl.DataFrame({
        "p": ["chp", "chp"], "source": ["coal", "biomass"],
        "sink": ["chp", "chp"]})
    p_process_source_flow_coef = Param(("p", "source"),
        pl.DataFrame({"p": ["chp", "chp"], "source": ["coal", "biomass"],
                      "value": [1.0, 2.0]}))
    # Single-timestep demand: zero out t02 to keep hand-calc tight.
    nb_dt = d.nodeBalance_dt
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("t") != "t01").then(0.0)
              .when(pl.col("n") == "heat").then(-5.0)
              .otherwise(-10.0)
    ).select("n", "d", "t", "value"))
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        p_flow_upper=p_flow_upper, p_commodity_price=p_commodity_price,
        process_input_flows=process_input_flows,
        p_process_source_flow_coef=p_process_source_flow_coef,
        p_inflow=p_inflow,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    v_flow = sol.value("v_flow").filter(pl.col("t") == "t01")
    coal = v_flow.filter(pl.col("source") == "coal")["value"][0]
    bio = v_flow.filter(pl.col("source") == "biomass")["value"][0]
    # Hand-calc: biomass specific cost (1.25) < coal (2.0) → bio carries all.
    assert coal == pytest.approx(0.0, abs=1e-7)
    assert bio == pytest.approx(0.0625, rel=1e-7)
