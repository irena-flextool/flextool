"""γ-style child-only LP wiring tests for storage-handoff features.

Three scenarios on the shared ``toy_storage_2t`` fixture exercising the
three nested-solve handoff methods:

**Scenario A (baseline)** — cyclic binding, no handoff parameter.
Reference-price objective term does NOT fire. With no flows, no inflow,
no penalty: objective = 0 exactly.

**Scenario B (fix_price)** — cyclic binding +
``p_storage_state_reference_price[s, d1] = 100.0``. The objective term
at ``model.py:3406-3428`` fires:

    - Σ v_state[s, d1, t02] · p_state_unitsize[s] · ref_price[s, d1] · ref_price_factor
    = - v_state[s, d1, t02] · 10 · 100 · 1
    = -1000 · v_state[s, d1, t02]

LP maximises v_state[s, d1, t02], bounded by ``p_state_upper × p_state_unitsize``
= 1.0 × 10 = 10 MWh (so v_state ≤ 1.0 normalised). Cyclic binding pins
state[t01] = state[t02] = 1.0 at optimum. Objective = -1000.

Sc B is the verifier for the ``p_unitsize → p_state_unitsize`` bug
fix at model.py:3427 — previously the term cross-joined per-process
unitsize values, producing ``num_processes × 100 × v_state`` instead of
``state_unitsize × 100 × v_state``.

**Scenario C (fix_usage)** — bind_forward_only + p_state_start=0.5 +
discharge process s→n + demand at t02 + fix_usage cap.  The
``node_storage_usage_fix_le`` constraint at ``model.py:1401-1545``
fires; LHS is the cumulative net discharge over the period, RHS is
``p_fix_storage_usage[s, d1, t02]``.

With state[t01]=5 MWh (fixed by p_state_start * existing/unitsize) and
the bind_forward_only convention that the state-change term is omitted
at t01, the LP can only discharge at t02.  Demand of 2 MWh at t02
without the cap → discharge=2, obj=0.  With cap=1.0 MWh, the constraint
limits discharge to 1, forcing 1 MWh of unmet-demand slack → obj = 1e6.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool

from .conftest import solver_options


def _solve(data):
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


def _wiring_overlay(toy_storage_2t, *, with_ref_price: bool):
    """Apply the common Sc-A/Sc-B overlay to ``toy_storage_2t``.

    Zeros out the demand inflow on ``n``, adds the cyclic binding set
    and the ``period_last`` frame that the reference-price objective
    needs. When ``with_ref_price`` is True also populates
    ``p_storage_state_reference_price[s, d1] = 100.0``.
    """
    nb_dt = toy_storage_2t.nodeBalance_dt  # n+s × t01,t02
    # Zero inflow on every (n, d, t) so no penalty fires.
    p_inflow_zero = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0))
             .select("n", "d", "t", "value"))

    storage_bind = pl.DataFrame({"n": ["s"]})
    period_last = pl.DataFrame({"d": ["d1"]})

    fields = dict(
        p_inflow=p_inflow_zero,
        storage_bind_within_timeblock=storage_bind,
        period_last=period_last,
    )
    if with_ref_price:
        fields["p_storage_state_reference_price"] = Param(
            ("n", "d"),
            pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [100.0]}),
        )

    return dataclasses.replace(toy_storage_2t, **fields)


def test_storage_handoff_wiring_sc_a_baseline(toy_storage_2t):
    """Sc A — no reference price; objective = 0 exactly.

    With cyclic state binding, no inflow, no flows, and no cost terms,
    every feasible solution has cost 0. v_state[s, t01] = v_state[s, t02]
    is degenerate over [0, 1] (normalised) — assert only the objective.
    """
    data = _wiring_overlay(toy_storage_2t, with_ref_price=False)
    pb, sol = _solve(data)
    assert sol.optimal, "Sc A baseline must solve"
    assert sol.obj == pytest.approx(0.0, abs=1e-9), (
        f"Sc A baseline objective should be 0 exactly; got {sol.obj}")


def _fix_usage_overlay(toy_storage_2t, *, cap: float):
    """Build the Sc C setup on ``toy_storage_2t``.

    Replaces the cyclic binding with bind_forward_only + storage_fix_start,
    adds a single discharge process ``discharge_p`` (s → n), sets demand
    only at t02 so the fwd-fix-induced ``flow=0 at t01`` constraint
    doesn't conflict with demand, and populates the fix_usage handoff
    frames with the given cap.
    """
    nb_dt = toy_storage_2t.nodeBalance_dt  # n+s × t01,t02

    # Demand only at t02 on node n.  ``bind_forward_only`` forces
    # battery's t01 nodeBalance to flow=0 (since v_state[t01] is pinned
    # to p_state_start and no state-change term fires at t01), so demand
    # at t01 would cost slack penalty unrelated to fix_usage.
    p_inflow_demand = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.when(
            (pl.col("n") == "n") & (pl.col("t") == "t02"))
            .then(pl.lit(-2.0))
            .otherwise(pl.lit(0.0))
        ).select("n", "d", "t", "value"))

    # bind_forward_only + storage_fix_start setup.
    storage_bind_fo = pl.DataFrame({"n": ["s"]})
    storage_fix_start = pl.DataFrame({"n": ["s"]})
    p_state_start = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [0.5]}))
    # dtttdt_forward_only = dtttdt with the first (d, t) per solve
    # dropped (input.py:2493-2496).  For 2 timesteps in one period,
    # drop t01: keep only (d1, t02, ...).
    dtttdt_fo = (toy_storage_2t.dtttdt
                 .sort("d", "t").slice(1))

    # Single discharge process s → n.  method_1way_1var_off (DIRECT, no
    # online, fork_no) lands in process_source_sink_eff.  p_slope=1.0 so
    # the source_eff branch fires with identity slope.
    pss = pl.DataFrame({"p": ["discharge_p"], "source": ["s"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(toy_storage_2t.dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    # No commodity costs; provide empty noEff and a single dummy eff
    # row so build_flextool's empty-frame guards don't trip.
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["discharge_p"], "source": ["s"], "sink": ["n"], "c": ["c_s"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",),
        pl.DataFrame({"p": ["discharge_p"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        toy_storage_2t.dt.with_columns(p=pl.lit("discharge_p"),
                                        value=pl.lit(1.0))
                          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        toy_storage_2t.dt.with_columns(c=pl.lit("c_s"),
                                        value=pl.lit(0.0))
                          .select("c", "d", "t", "value"))

    # fix_usage handoff frames.  cstr anchored at nodeState_last_dt
    # = (s, d1, t02); RHS reduces to p_fix_storage_usage[s, d1, t02] = cap
    # via the period_branch × dtt_timeline_matching × period_last joins.
    n_fix = pl.DataFrame({"n": ["s"]})
    ndt_fix = pl.DataFrame({"n": ["s"], "d": ["d1"], "t": ["t02"]})
    p_fix = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "t": ["t02"],
                      "value": [cap]}))
    period_branch = pl.DataFrame({"d_upper": ["d1"], "d": ["d1"]})
    period_last = pl.DataFrame({"d": ["d1"]})
    dtt = pl.DataFrame({"d": ["d1"], "t": ["t02"], "t_upper": ["t02"]})

    return dataclasses.replace(
        toy_storage_2t,
        p_inflow=p_inflow_demand,
        storage_bind_forward_only=storage_bind_fo,
        storage_fix_start=storage_fix_start,
        p_state_start=p_state_start,
        dtttdt_forward_only=dtttdt_fo,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt,
        flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize,
        p_flow_upper=p_flow_upper,
        p_slope=p_slope,
        p_commodity_price=p_commodity_price,
        n_fix_storage_usage=n_fix,
        ndt_fix_storage_usage=ndt_fix,
        p_fix_storage_usage=p_fix,
        period_branch=period_branch,
        period_last=period_last,
        dtt_timeline_matching=dtt,
    )


def test_storage_handoff_wiring_sc_c_fix_usage_baseline(toy_storage_2t):
    """Sc C-baseline — fix_usage cap is generous (5.0), constraint slack.

    With cap large enough to admit any feasible discharge, the LP
    behaves as if fix_usage isn't there: discharge_t02 = 2 MWh meets
    demand, no slack penalty. obj = 0.

    Used as a paired baseline for the binding-cap scenario below.
    """
    data = _fix_usage_overlay(toy_storage_2t, cap=5.0)
    pb, sol = _solve(data)
    assert sol.optimal
    assert "node_storage_usage_fix_le" in set(pb.cstr_names()), (
        "Constraint must be built when handoff frames are populated")
    assert sol.obj == pytest.approx(0.0, abs=1e-6), (
        f"Sc C-baseline (cap=5) should solve at obj=0; got {sol.obj}")


def test_storage_handoff_wiring_sc_c_fix_usage_binding(toy_storage_2t):
    """Sc C-binding — fix_usage cap = 1.0 forces unmet demand.

    With cap=1 the constraint binds: discharge_t01 + discharge_t02 ≤ 1.
    bind_forward_only forces discharge_t01=0 (no state-change term at
    t01), so discharge_t02 ≤ 1.  Demand at t02 is 2, so slack of 1
    fires.  obj = 1 × 1e6 = 1,000,000.

    Hand-calc:
        Without cap:  discharge_t02=2, slack=0,    obj=0
        With cap=1:   discharge_t02=1, slack_up=1, obj=1×1e6=1e6
    """
    data = _fix_usage_overlay(toy_storage_2t, cap=1.0)
    pb, sol = _solve(data)
    assert sol.optimal
    assert "node_storage_usage_fix_le" in set(pb.cstr_names())
    assert sol.obj == pytest.approx(1_000_000.0, rel=1e-6), (
        f"Sc C-binding (cap=1) should solve at obj=1e6 (1 MWh × 1e6 "
        f"penalty); got {sol.obj}")

    # Verify discharge was capped at 1 (not the demand of 2).
    v_flow = sol.value("v_flow").filter(
        (pl.col("p") == "discharge_p") & (pl.col("t") == "t02"))
    assert v_flow.height == 1
    assert v_flow["value"][0] == pytest.approx(1.0, rel=1e-6), (
        f"Discharge at t02 should be capped at 1.0; got {v_flow['value'][0]}")


def test_fix_usage_producer_applies_slope(toy_storage_2t, tmp_path):
    """Producer test: with p_slope=0.9 on the discharge process, the
    produced p_fix_storage_usage must include the slope multiplier per
    the eff-partition branch added at input.py:5706-5837.

    Hand-calc:
      LP solves with discharge_t02 = 2 MWh (demand-driven).
      Producer's source_eff branch: v_flow * unitsize * slope * step_dur
                                  = 2 * 1.0 * 0.9 * 1.0
                                  = 1.8 MWh.
      (sink_side is 0 — battery is never a sink for any process here.)

    The pre-promotion producer would have emitted 2.0 (raw v_flow * unitsize),
    so this test fails on the legacy simplified formula and passes on
    the new full formula.  Test exists to guard against regression.
    """
    from flextool.engine_polars.input import build_handoff_from_solution

    # Build the LP with non-unit slope on discharge_p.
    data = _fix_usage_overlay(toy_storage_2t, cap=10.0)  # generous cap, non-binding
    data = dataclasses.replace(data,
        p_slope=Param(("p", "d", "t"),
            toy_storage_2t.dt.with_columns(
                p=pl.lit("discharge_p"),
                value=pl.lit(0.9),
            ).select("p", "d", "t", "value")))
    # build_handoff_from_solution looks up which nodes use fix_usage from
    # flex_data.node__storage_nested_fix_method (schema: node, method).
    nsfm = pl.DataFrame({"node": ["s"], "method": ["fix_usage"]})
    data = dataclasses.replace(data, node__storage_nested_fix_method=nsfm)

    pb, sol = _solve(data)
    assert sol.optimal, "LP must solve before producer extraction"

    # Build a minimal parent_handoff carrying fix_storage_timesteps —
    # build_handoff_from_solution reads this when no CSV is present.  Use a
    # real SolveHandoff for typed attribute access, with the timesteps
    # injected via a dynamic attribute (the field was retired in Phase 3
    # but the consumer still falls back to ``getattr(parent_handoff,
    # "fix_storage_timesteps", None)``).
    from flextool.engine_polars._solve_handoff import SolveHandoff
    fs_steps = pl.DataFrame({"period": ["d1"], "step": ["t02"]})
    parent_handoff = SolveHandoff()
    parent_handoff.fix_storage_timesteps = fs_steps  # type: ignore[attr-defined]

    # Empty solve_data dir — function falls back to in-memory inputs.
    (tmp_path / "solve_data").mkdir()

    handoff = build_handoff_from_solution(
        sol, tmp_path, "test_solve",
        flex_data=data,
        parent_handoff=parent_handoff,
    )

    assert handoff.fix_storage_usage is not None, (
        "Producer must populate fix_storage_usage for fix_usage nodes")
    fu = handoff.fix_storage_usage.filter(
        (pl.col("node") == "s") & (pl.col("step") == "t02"))
    assert fu.height == 1, f"Expected 1 row at (s, d1, t02); got {fu.height}"
    val = fu["p_fix_storage_usage"][0]
    # Slope-adjusted: 2 * 1.0 * 0.9 = 1.8; legacy raw would be 2.0.
    assert val == pytest.approx(1.8, abs=1e-6), (
        f"Producer with new slope-aware formula should emit 1.8 "
        f"(=v_flow=2 * unitsize=1.0 * slope=0.9); got {val}. "
        f"If you got 2.0, the source_eff branch isn't firing — check "
        f"that process_source_sink_eff is populated and p_slope is "
        f"not None.")


def test_storage_handoff_wiring_sc_b_fix_price(toy_storage_2t):
    """Sc B — reference price = 100.0; objective = -1000 exactly.

    The reference-price objective term at ``model.py:3406-3428`` fires
    when ``p_storage_state_reference_price`` is populated and
    ``nodeState_last_dt`` + ``period_last`` are present.

    Hand-calc with unit weights (``p_rp_cost_weight=1``,
    ``p_inflation_op=1``, ``p_period_share=1``, no ``pdt_branch_weight``):

        ref_price_factor = 1.0
        obj contribution  = - v_state[s, d1, t02]
                              · p_state_unitsize[s]   (= 10)
                              · p_storage_state_reference_price[s, d1] (= 100)
                              · ref_price_factor      (= 1)
                          = -1000 · v_state[s, d1, t02]

    LP maximises v_state[s, d1, t02] up to the bound
    ``p_state_upper × p_state_unitsize = 1.0 × 10 = 10`` MWh
    (i.e. ``v_state ≤ 1.0`` in normalised units). At the optimum
    v_state[s, d1, t02] = 1.0, so objective = -1000.

    This number is sensitive to the ``p_unitsize`` vs ``p_state_unitsize``
    distinction: the buggy pre-fix code computed
    ``-Σ_p p_unitsize[p] · 100 · v_state`` = ``-num_processes × 100``
    (independent of the state node!), which is structurally wrong.
    """
    data = _wiring_overlay(toy_storage_2t, with_ref_price=True)
    pb, sol = _solve(data)
    assert sol.optimal, "Sc B must solve"
    assert sol.obj == pytest.approx(-1000.0, abs=1e-9), (
        f"Sc B objective should be -1000 exactly (p_state_unitsize=10 "
        f"× ref_price=100 × v_state[t02]=1.0); got {sol.obj}")

    # The optimum v_state at the last timestep should be at the upper
    # bound (1.0 normalised = 10 MWh).
    v_state = sol.value("v_state").filter(
        (pl.col("n") == "s") & (pl.col("t") == "t02"))
    assert v_state.height == 1
    assert v_state["value"][0] == pytest.approx(1.0, abs=1e-9), (
        f"Sc B optimum v_state[s, d1, t02] should be at upper bound 1.0; "
        f"got {v_state['value'][0]}")
