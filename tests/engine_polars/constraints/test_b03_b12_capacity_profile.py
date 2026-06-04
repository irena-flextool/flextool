"""Surface B.3 / B.12 — flow-capacity (maxFlow) and profile bounds.

Focused constraint tests on the shared minimal fixtures:

* ``test_max_to_sink_availability_multiplier`` — covers **B3.2**.
  Mutates ``toy_1n1p_1d2t`` to tighten capacity to 1 and inject a
  ``p_process_availability`` of 0.5 at t01 (1.0 at t02).  Checks the
  ``flow_upper_rhs * p_process_availability`` multiplication binds
  v_flow at t01 to 0.5 and routes the 9.5 MWh shortfall to ``vq_state_up``.

* ``test_max_to_sink_neg_cap_forced_minimum`` — covers **B3.5**.
  Adds a single-row ``pd_neg_cap`` overlay so the ``maxFlow_negCap``
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
# B3.2 — maxFlow availability multiplier.

def test_max_to_sink_availability_multiplier(toy_1n1p_1d2t):
    """Covers B3.2 — `maxFlow` availability multiplier.

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
# B3.5 — maxFlow_negCap (negative-unitsize forced minimum).

def test_max_to_sink_neg_cap_forced_minimum(toy_1n1p_1d2t):
    """Covers B3.5 — `maxFlow_negCap` anti-energy forced minimum.

    With unitsize=-1, existing=-1, the .mod's ``v_flow * unitsize ≤
    existing`` flips on division by unitsize to ``v_flow ≥ |existing|/|us|``
    = +1.  polar_high emits this as the ``maxFlow_negCap`` (≥) companion
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
    assert "maxFlow_negCap" in set(pb.cstr_names())
    # Hand-calc: maxFlow (≤1) ∧ maxFlow_negCap (≥1) ⇒ v_flow == 1.
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


# ---------------------------------------------------------------------------
# B12.5 — profile_flow_upper with SPARSE availability (regression).

def test_profile_upper_sparse_availability_default(toy_1n1p_1d2t):
    """Regression: a profile (VRE) process with NO authored ``availability``
    must keep its default-1.0 factor, not be inner-join-dropped to RHS=0.

    ``p_process_availability`` is SPARSE — ``p_process_availability_from_source``
    emits only DB-authored rows, expecting absent processes to fall back to the
    schema default 1.0.  But ``_add_profile_cstr`` folds the factor into the
    RHS via ``Param * Param`` (an inner join), so before the densify fix a
    process absent from the sparse Param had its bound collapsed to 0 → forced
    ``v_flow = 0``.  This is what zeroed every wind/solar unit whose
    ``availability`` was left at the default in a model where OTHER units (e.g.
    hydro) authored one (making the Param non-empty but sparse).  No existing
    fixture combines authored + default availability in one solve, so the bug
    slipped the parity sweep — this test pins the mixed case.

    Two profile-upper processes share node ``n``: ``p_auth`` authors
    availability 0.5; ``p_def`` authors none (→ default 1.0).  Both carry
    profile_upper 0.8 against capacity 1; demand (10/step) far exceeds supply
    so the cost-min LP drives each to its ceiling.

    Hand-calc:
        v_flow[p_auth] ≤ 0.8 · 1 · 0.5 = 0.4
        v_flow[p_def]  ≤ 0.8 · 1 · 1.0 = 0.8   (the regression: NOT 0)
    """
    d = toy_1n1p_1d2t
    pss = pl.DataFrame({"p": ["p_auth", "p_def"],
                        "source": ["src", "src"],
                        "sink":   ["n", "n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(d.dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p_auth", "p_def"], "source": ["src", "src"],
         "sink": ["n", "n"], "c": ["FUEL", "FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["p_auth", "p_def"], "value": [1.0, 1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(1.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        pss_dt.select("p", "d", "t").with_columns(value=pl.lit(1.0)))
    # existing_count = capacity/unitsize = 1 for both.
    p_exist_cnt = Param(("p", "d"),
        pl.DataFrame({"p": ["p_auth", "p_def"], "d": ["d1", "d1"],
                      "value": [1.0, 1.0]}))
    process_profile_upper = pl.DataFrame(
        {"p": ["p_auth", "p_def"], "source": ["src", "src"],
         "sink": ["n", "n"], "f": ["fA", "fD"]})
    p_profile_value = Param(("f", "d", "t"),
        pl.DataFrame({"f": ["fA", "fA", "fD", "fD"],
                      "d": ["d1"] * 4, "t": ["t01", "t02", "t01", "t02"],
                      "value": [0.8, 0.8, 0.8, 0.8]}))
    # SPARSE: only p_auth authors availability; p_def is absent → default 1.0.
    avail = Param(("p", "d", "t"),
        pl.DataFrame({"p": ["p_auth", "p_auth"], "d": ["d1", "d1"],
                      "t": ["t01", "t02"], "value": [0.5, 0.5]}))
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper, p_slope=p_slope,
        process_profile_upper=process_profile_upper,
        p_profile_value=p_profile_value,
        p_process_existing_count=p_exist_cnt,
        p_process_availability=avail,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "profile_flow_upper_limit" in set(pb.cstr_names())
    v_flow = sol.value("v_flow").sort(["p", "t"])
    auth = v_flow.filter(pl.col("p") == "p_auth").sort("t")["value"].to_list()
    deff = v_flow.filter(pl.col("p") == "p_def").sort("t")["value"].to_list()
    # p_auth tightened by its authored 0.5; p_def keeps the default 1.0.
    assert auth[0] == pytest.approx(0.4, rel=1e-7)
    assert auth[1] == pytest.approx(0.4, rel=1e-7)
    # Regression: before the densify fix p_def was inner-join-dropped → 0.0.
    assert deff[0] == pytest.approx(0.8, rel=1e-7)
    assert deff[1] == pytest.approx(0.8, rel=1e-7)


# ---------------------------------------------------------------------------
# B3.6 — maxFlow direct-arc capacity_max_coeff (maxToSink sink-coef).

def test_max_to_sink_direct_capacity_max_coeff(toy_1n1p_1d2t):
    """Covers B3.6 — `maxFlow` per-arc ``capacity_max_coeff`` on a DIRECT
    1-in/1-out unit (the maxToSink sink-coef branch).

    A direct unit (constant efficiency, single source→sink) bounds its
    output via ``p_flow_upper_existing`` (= existing/unitsize), which the
    .mod's ``maxToSink`` then multiplies by
    ``p_process_sink_max_capacity_coefficient[p, sink]``.  The engine drops
    that coefficient unless it is folded onto the direct-arc RHS — this
    test injects ``p_arc_max_cap_coef = 0.5`` on the output arc.

    Hand-calc: existing-capacity bound = 2.0 (``p_flow_upper_existing``),
    coef = 0.5, availability = 1.0 ⇒ RHS = 2.0 × 0.5 = 1.0.  Demand = 10/
    step pushes for full output, so v_flow binds at 1.0 each step and the
    9.0 shortfall routes to ``vq_state_up``.

    Pre-fix (coef dropped) the bound stays at 2.0 → v_flow = 2.0 → the
    asserts below fail.
    """
    d = toy_1n1p_1d2t
    pss_dt = compute_pss_dt(d)
    # Existing-only direct bound = 2.0 (chosen as flow_upper_rhs because it
    # is non-None and the process is not indirect).
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss_dt.select("p", "source", "sink", "d").unique()
              .with_columns(value=pl.lit(2.0)))
    # Per-arc capacity_max_coeff = 0.5 on the output arc (sink ``n``).
    p_arc_coef = Param(("p", "source", "sink"),
        pl.DataFrame({"p": ["p"], "source": ["source_n"], "sink": ["n"],
                      "value": [0.5]}))
    data = dataclasses.replace(d,
        p_flow_upper_existing=p_flow_upper_existing,
        p_arc_max_cap_coef=p_arc_coef)
    pb, sol = _solve(data)
    assert sol.optimal
    v_flow = sol.value("v_flow").sort("t")
    vq_up = sol.value("vq_state_up").sort("t")
    # RHS = 2.0 × 0.5 = 1.0 each step.
    assert v_flow["value"][0] == pytest.approx(1.0, rel=1e-7)
    assert v_flow["value"][1] == pytest.approx(1.0, rel=1e-7)
    # demand 10 − 1.0 supplied = 9.0 unmet via slack.
    assert vq_up["value"][0] == pytest.approx(9.0, rel=1e-7)


# ---------------------------------------------------------------------------
# B3.7 — maxFlow direct-arc capacity_max_coeff (maxFromSource source-coef).

def test_max_from_source_direct_capacity_max_coeff(toy_1n1p_1d2t):
    """Covers B3.7 — `maxFlow` per-arc ``capacity_max_coeff`` on a no-output
    node-as-source DIRECT arc (the maxFromSource source-coef branch).

    A 1-way unit whose *source* is a node and that has no output node is
    emitted by the engine as a synthetic ``(p, source, p)`` arc whose
    ``sink`` is the process itself (``sink ∉ process_sink``).  The .mod's
    ``maxFromSource`` multiplies the existing-capacity RHS by
    ``p_process_source_max_capacity_coefficient[p, source]``.  This is the
    branch the engine previously omitted entirely.  At the model level the
    fold is the same per-arc multiply — the producer selects the source
    coef for these arcs; here we inject it directly.

    Hand-calc: existing-capacity bound = 2.0, source-coef = 0.5 ⇒ RHS =
    1.0.  The arc still feeds demand node ``n`` (via ``flow_to_n``) which
    pulls for full output, so v_flow binds at 1.0 each step.  The arc's
    ``sink`` label is the process itself (``sink ∉ process_sink``) — the
    real producer would pick the source coef here; at the model level the
    fold is the same per-arc multiply and the injected coef governs.
    """
    d = toy_1n1p_1d2t
    # No-output node-as-source arc: source = commodity node ``fuel_n``,
    # synthetic sink = the process ``p`` itself (so ``sink ∉ process_sink``).
    # ``flow_to_n`` still routes the produced energy to demand node ``n``.
    pss = pl.DataFrame({"p": ["p"], "source": ["fuel_n"], "sink": ["p"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(d.dt, how="cross")
    # Route the arc into demand node ``n`` (inflow −10 from the toy).
    flow_to_n = pss.with_columns(n=pl.lit("n"))
    # Fuel commodity hookup so the source side prices in.
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p"], "source": ["fuel_n"], "sink": ["p"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    # Existing-only direct bound = 2.0 on the synthetic arc.
    p_flow_upper_existing = Param(("p", "source", "sink", "d"),
        pss_dt.select("p", "source", "sink", "d").unique()
              .with_columns(value=pl.lit(2.0)))
    # Source-side capacity_max_coeff = 0.5 on arc (p, fuel_n, p).
    p_arc_coef = Param(("p", "source", "sink"),
        pl.DataFrame({"p": ["p"], "source": ["fuel_n"], "sink": ["p"],
                      "value": [0.5]}))
    # p_flow_upper must cover the rewired arc (model requires it non-None);
    # set it loose (100) so the tight existing-only bound × coef governs.
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        d.dt.with_columns(p=pl.lit("p"), value=pl.lit(1.0))
            .select("p", "d", "t", "value"))
    data = dataclasses.replace(d,
        process_source_sink=pss, process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff, pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_slope=p_slope,
        p_flow_upper=p_flow_upper,
        p_flow_upper_existing=p_flow_upper_existing,
        p_arc_max_cap_coef=p_arc_coef)
    pb, sol = _solve(data)
    assert sol.optimal
    v_flow = sol.value("v_flow").sort("t")
    # RHS = 2.0 × 0.5 = 1.0 each step — capped by the (source) coef.
    assert v_flow["value"][0] == pytest.approx(1.0, rel=1e-7)
    assert v_flow["value"][1] == pytest.approx(1.0, rel=1e-7)


# ---------------------------------------------------------------------------
# Producer-level: p_arc_max_cap_coef_from_source sink-vs-source selection.

def test_p_arc_max_cap_coef_producer_selection():
    """Unit-test the producer's maxToSink vs maxFromSource coef selection
    and its exclusion of indirect processes.

    Three direct arcs + one indirect arc:

    * ``u_sink`` — normal output arc (sink ``n`` ∈ process_sink) → picks the
      OUTPUT-node coef (0.5).
    * ``u_src``  — no-output node-as-source arc (sink ``u_src`` ∉
      process_sink) → picks the INPUT-node coef (0.25).
    * ``u_one``  — output arc whose coef is the default 1.0 → dropped from
      the carried Param.
    * ``chp``    — indirect process → excluded entirely (carried via
      ``p_flow_upper`` instead).
    """
    import polars as _pl

    from flextool.engine_polars._derived_params import (
        p_arc_max_cap_coef_from_source,
    )

    class _Src:
        """Minimal InputSource: serves entities + the two coef params."""

        def __init__(self):
            self._entities = {
                # u_sink: output arc to node n.
                "unit__outputNode": _pl.DataFrame(
                    {"unit": ["u_sink", "u_one", "chp"],
                     "node": ["n", "n2", "h"]}),
                # u_src: input arc from node n (and chp fuel input).
                "unit__inputNode": _pl.DataFrame(
                    {"unit": ["u_src", "chp"],
                     "node": ["n", "fuel"]}),
            }
            self._params = {
                ("unit__outputNode", "capacity_max_coeff"): _pl.DataFrame(
                    {"unit": ["u_sink", "u_one"],
                     "node": ["n", "n2"],
                     "value": [0.5, 1.0]}),
                ("unit__inputNode", "capacity_max_coeff"): _pl.DataFrame(
                    {"unit": ["u_src"], "node": ["n"], "value": [0.25]}),
            }

        def entities(self, ec):
            if ec in self._entities:
                return self._entities[ec]
            raise KeyError(ec)

        def parameter(self, ec, pn):
            if (ec, pn) in self._params:
                return self._params[(ec, pn)]
            raise KeyError((ec, pn))

    # pss: 3 direct arcs + 1 indirect (chp output).
    pss = _pl.DataFrame({
        "p":      ["u_sink", "u_src", "u_one", "chp"],
        "source": ["s_in",   "n",     "s2",    "chp"],
        "sink":   ["n",      "u_src", "n2",    "h"],
    })
    # Mark chp indirect via the explicit classified frame.
    classified = _pl.DataFrame({
        "p": ["chp"], "klass": ["unit"], "method": ["method_1way_nvar_off"],
    })
    res = p_arc_max_cap_coef_from_source(_Src(), pss, classified)
    assert res is not None
    got = {(r["p"], r["source"], r["sink"]): r["value"]
           for r in res.frame.iter_rows(named=True)}
    # u_sink: sink ∈ process_sink → OUTPUT coef 0.5.
    assert got[("u_sink", "s_in", "n")] == pytest.approx(0.5)
    # u_src: sink u_src ∉ process_sink → INPUT (source) coef 0.25.
    assert got[("u_src", "n", "u_src")] == pytest.approx(0.25)
    # u_one (coef 1.0) dropped; chp (indirect) excluded.
    assert ("u_one", "s2", "n2") not in got
    assert all(p != "chp" for (p, _s, _k) in got)
