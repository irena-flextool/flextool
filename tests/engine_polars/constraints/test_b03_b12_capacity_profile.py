"""Surface B.3 / B.12 — flow-capacity (maxToSink) and profile bounds.

Focused constraint tests on the shared minimal fixtures:

* ``test_max_to_sink_availability_multiplier`` — covers **B3.2**.
  Mutates ``toy_1n1p_1d2t`` to tighten capacity to 1 and inject a
  ``p_process_availability`` of 0.5 at t01 (1.0 at t02).  Checks the
  ``flow_upper_rhs * p_process_availability`` multiplication binds
  v_flow at t01 to 0.5 and routes the 9.5 MWh shortfall to ``vq_state_up``.

* ``test_max_to_sink_neg_cap_forced_minimum`` — covers **B3.5**.
  Adds a single-row ``pd_neg_cap`` overlay so the ``maxToSink_negCap``
  (≥) companion constraint is emitted.  With existing/unitsize giving
  +1 on the bound and zero inflow, the forced-minimum semantics pin
  v_flow == 1 even with no demand.

* ``test_profile_lower_and_fixed`` — covers **B12.2** consolidated with
  **B12.3**.  Both methods share the ``_add_profile_cstr`` shape; in
  one fixture (separate processes per profile method) we pin a lower
  floor (0.6) on one process and an exact value (0.7 at t01, 0.4 at
  t02) on the other.

* ``test_profile_upper_invest_lhs_tightening`` — covers **B12.4**.
  Mutates ``toy_invest_3d`` to introduce a profile_upper=0.5 producer
  with demand 2 in period d1.  The invest LHS injection forces
  v_invest_p ≥ 2 / 0.5 = 4 to satisfy v_flow − 0.5·v_invest ≤ 0
  while still meeting demand.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars._pdt_join import compute_pss_dt, compute_nodeBalance_dt

from .conftest import solver_options


def _solve(data) -> tuple[Problem, Any]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# B3.2 — maxToSink availability multiplier.

def test_max_to_sink_availability_multiplier(toy_1n1p_1d2t):
    """Covers B3.2 — `maxToSink` availability multiplier.

    Hand-calc: capacity=1 (p_flow_upper=1), availability=[0.5, 1.0] over
    [t01, t02], demand=10/step. RHS = 1 × availability →
        t01: v_flow ≤ 0.5 ⇒ v_flow=0.5, vq_state_up = 10 − 0.5 = 9.5
        t02: v_flow ≤ 1.0 ⇒ v_flow=1.0, vq_state_up = 10 − 1.0 = 9.0
    """
    d = toy_1n1p_1d2t
    pss_dt = compute_pss_dt(d)
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    avail = Param(("p", "d", "t"),
        pl.DataFrame({"p": ["p", "p"], "d": ["d1", "d1"],
                      "t": ["t01", "t02"], "value": [0.5, 1.0]}))
    data = dataclasses.replace(d, p_flow_upper=p_flow_upper,
                               p_process_availability=avail)
    pb, sol = _solve(data)
    assert sol.optimal
    v_flow = sol.value("v_flow").sort("t")
    vq_up = sol.value("vq_state_up").sort("t")
    assert v_flow.filter(pl.col("t") == "t01")["value"][0] == pytest.approx(0.5, rel=1e-7)
    assert v_flow.filter(pl.col("t") == "t02")["value"][0] == pytest.approx(1.0, rel=1e-7)
    assert vq_up.filter(pl.col("t") == "t01")["value"][0] == pytest.approx(9.5, rel=1e-7)


# ---------------------------------------------------------------------------
# B3.5 — maxToSink_negCap (negative-unitsize forced minimum).

def test_max_to_sink_neg_cap_forced_minimum(toy_1n1p_1d2t):
    """Covers B3.5 — `maxToSink_negCap` anti-energy forced minimum.

    With unitsize=-1, existing=-1, the .mod's ``v_flow * unitsize ≤
    existing`` flips on division by unitsize to ``v_flow ≥ |existing|/|us|``
    = +1.  polar_high emits this as the ``maxToSink_negCap`` (≥) companion
    on the rows in ``pd_neg_cap`` (sharing the same +1 RHS as the ≤).
    With zero demand and small slack penalty, no force pushes v_flow
    above 1, so v_flow = 1 is the forced minimum.
    """
    d = toy_1n1p_1d2t
    pss_dt = compute_pss_dt(d)
    # Tight capacity = 1 (the .mod's |existing|/|us| RHS for both halves).
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    # Zero demand and tiny dump-slack penalty so v_flow=1 is feasible cheaply.
    nb_dt = compute_nodeBalance_dt(d)
    p_inflow_zero = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0)).select("n", "d", "t", "value"))
    pd_neg = pl.DataFrame({"p": ["p"], "d": ["d1"]})
    data = dataclasses.replace(d, p_flow_upper=p_flow_upper,
                               p_inflow=p_inflow_zero,
                               p_penalty_up=p_pen, p_penalty_down=p_pen,
                               pd_neg_cap=pd_neg)
    pb, sol = _solve(data)
    assert sol.optimal
    assert "maxToSink_negCap" in set(pb.cstr_names())
    # Hand-calc: maxToSink (≤1) ∧ maxToSink_negCap (≥1) ⇒ v_flow == 1.
    v_flow = sol.value("v_flow")
    assert v_flow["value"][0] == pytest.approx(1.0, rel=1e-7)
    assert v_flow["value"][1] == pytest.approx(1.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B12.2 + B12.3 — profile_flow_lower_limit AND profile_flow_fixed.

def test_profile_lower_and_fixed(toy_1n1p_1d2t):
    """Covers B12.2 (`profile_flow_lower_limit`) consolidated with
    B12.3 (`profile_flow_fixed`).

    Two single-source/sink processes share node ``n``: ``p_lo`` carries
    a lower-limit profile (0.6 floor, capacity=1 ⇒ floor 0.6 MW), and
    ``p_fx`` carries a fixed profile (0.7 at t01, 0.4 at t02 against
    capacity=1).  Demand and slack penalties are tuned so the LP picks
    the floor / fixed value exactly:

    Hand-calc:
        v_flow[p_lo, t01] = v_flow[p_lo, t02] = 0.6  (lower bound binds)
        v_flow[p_fx, t01] = 0.7, v_flow[p_fx, t02] = 0.4 (equality binds)
    """
    d = toy_1n1p_1d2t
    # Two processes with the same source/sink shape; both feed node ``n``.
    pss = pl.DataFrame({"p": ["p_lo", "p_fx"],
                        "source": ["src", "src"],
                        "sink":   ["n", "n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(d.dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p_lo", "p_fx"], "source": ["src", "src"],
         "sink": ["n", "n"], "c": ["FUEL", "FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["p_lo", "p_fx"], "value": [1.0, 1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").with_columns(value=pl.lit(1.0)))
    # existing_count = capacity/unitsize = 1 for both.
    p_exist_cnt = Param(("p", "d"),
        pl.DataFrame({"p": ["p_lo", "p_fx"], "d": ["d1", "d1"],
                      "value": [1.0, 1.0]}))
    # Profiles: each process maps to its own profile name.
    process_profile_lower = pl.DataFrame(
        {"p": ["p_lo"], "source": ["src"], "sink": ["n"], "f": ["fLO"]})
    process_profile_fixed = pl.DataFrame(
        {"p": ["p_fx"], "source": ["src"], "sink": ["n"], "f": ["fFX"]})
    p_profile_value = Param(("f", "d", "t"),
        pl.DataFrame({"f": ["fLO", "fLO", "fFX", "fFX"],
                      "d": ["d1"]*4, "t": ["t01", "t02", "t01", "t02"],
                      "value": [0.6, 0.6, 0.7, 0.4]}))
    # Zero inflow + small slack so the LP doesn't get pushed away from
    # the profile values; commodity price is cheap so paying for v_flow
    # is cheaper than the dump-slack we'd otherwise need.
    nb_dt = compute_nodeBalance_dt(d)
    p_inflow_zero = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1.0)).select("n", "d", "t", "value"))
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper, p_slope=p_slope,
        p_inflow=p_inflow_zero, p_penalty_up=p_pen, p_penalty_down=p_pen,
        process_profile_lower=process_profile_lower,
        process_profile_fixed=process_profile_fixed,
        p_profile_value=p_profile_value,
        p_process_existing_count=p_exist_cnt,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    cstr = set(pb.cstr_names())
    assert "profile_flow_lower_limit" in cstr
    assert "profile_flow_fixed" in cstr
    v_flow = sol.value("v_flow").sort(["p", "t"])
    lo = v_flow.filter(pl.col("p") == "p_lo").sort("t")["value"].to_list()
    fx = v_flow.filter(pl.col("p") == "p_fx").sort("t")["value"].to_list()
    # Hand-calc: lower floor binds at 0.6 each step; fixed pins 0.7, 0.4.
    assert lo[0] == pytest.approx(0.6, rel=1e-7)
    assert lo[1] == pytest.approx(0.6, rel=1e-7)
    assert fx[0] == pytest.approx(0.7, rel=1e-7)
    assert fx[1] == pytest.approx(0.4, rel=1e-7)


# ---------------------------------------------------------------------------
# B12.4 — profile_flow_upper invest LHS tightening.

def test_profile_upper_invest_lhs_tightening(toy_invest_3d):
    """Covers B12.4 — invest LHS injection on `profile_flow_upper_limit`.

    Mutates ``toy_invest_3d``: process ``u`` is invest-eligible in d1
    with existing=0, profile_upper=0.5, demand 2 over the single
    timestep in d1.

    Hand-calc: v_flow ≤ 0.5 · v_invest_p ⇒ v_invest_p ≥ 2/0.5 = 4.
    Cost-minimisation picks v_invest_p = 4, v_flow = 2; nodeBalance
    closes with no slack.
    """
    d = toy_invest_3d
    # Demand 2 at d1/t01, 0 elsewhere; small slack penalty so paying
    # invest cost (10/unit) wins over slack (the alternative would be
    # vq_state_up = 2 ⇒ slack-cost 2 vs invest 4·10 = 40, but the
    # profile-upper LHS forces invest ≥ 4 once we want v_flow > 0).
    nb_dt = compute_nodeBalance_dt(d)
    p_inflow_new = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.when(
            (pl.col("d") == "d1") & (pl.col("t") == "t01"))
            .then(-2.0).otherwise(0.0)
        ).select("n", "d", "t", "value"))
    # Slack penalty so high that buying the invest is the cost-min path.
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    # Profile upper = 0.5 in d1/t01 (loose elsewhere).
    pss = d.process_source_sink
    process_profile_upper = pl.DataFrame(
        {"p": ["u"], "source": ["FUEL_n"], "sink": ["n"], "f": ["fUP"]})
    p_profile_value = Param(("f", "d", "t"),
        d.dt.with_columns(f=pl.lit("fUP"),
                          value=pl.when((pl.col("d") == "d1")
                                        & (pl.col("t") == "t01"))
                                .then(0.5).otherwise(1.0))
            .select("f", "d", "t", "value"))
    # existing_count = 0 / 1 = 0 across all periods.
    p_exist_cnt = Param(("p", "d"),
        pl.DataFrame({"p": ["u"]*3, "d": ["d1", "d2", "d3"],
                      "value": [0.0, 0.0, 0.0]}))
    data = dataclasses.replace(d,
        p_inflow=p_inflow_new, p_penalty_up=p_pen, p_penalty_down=p_pen,
        process_profile_upper=process_profile_upper,
        p_profile_value=p_profile_value,
        p_process_existing_count=p_exist_cnt,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "profile_flow_upper_limit" in set(pb.cstr_names())
    # Hand-calc: 2 / 0.5 = 4.
    v_invest = sol.value("v_invest_p").filter(
        (pl.col("p") == "u") & (pl.col("d") == "d1"))
    assert v_invest["value"][0] == pytest.approx(4.0, rel=1e-7)
    v_flow = sol.value("v_flow").filter(
        (pl.col("d") == "d1") & (pl.col("t") == "t01"))
    assert v_flow["value"][0] == pytest.approx(2.0, rel=1e-7)
