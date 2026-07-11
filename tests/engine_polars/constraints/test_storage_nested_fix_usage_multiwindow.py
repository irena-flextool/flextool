"""Multi-coarse-window regression for ``node_storage_usage_fix_le``.

The isolation tests in ``test_storage_handoff_wiring.py`` all use a
trivial single-coarse-window timeline
(``dtt_timeline_matching = {(d1, t02, t02)}``), so they never exercise
the coarse→fine window aggregation.  The real seasonal handoff maps a
coarse ``mt_4h`` step onto MANY fine hourly steps and a fine roll spans
MANY coarse windows.

This test builds a single fine period ``d1`` with 4 fine steps split
into TWO coarse windows:

    coarse window ``tu1`` ← fine steps t01, t02
    coarse window ``tu2`` ← fine steps t03, t04

and a storage node ``s`` set to ``fix_usage`` with a per-window coarse
usage target.  ``nodeState_last_dt`` = (s, d1, t04) — so the terminal
step maps to coarse window ``tu2``.

**The bug (before the fix):** the RHS was built by joining
``dtt_timeline_matching`` on ``[d, t_upper]`` then ``nodeState_last_dt``
on ``[n, d, t]``, collapsing the cap to ONLY the terminal step's coarse
window (``tu2``), while the LHS summed flow over the whole roll.  So the
cap was ``total flow ≤ p_fix_storage_usage[tu2]`` — grossly too tight
whenever the roll spans >1 coarse window.

**The fix (reference ``storage_usage_fix`` in flextool.mod):** the cap
is a PER-PERIOD aggregate — ``total flow over period d ≤ Σ over every
coarse window overlapping d`` = ``p_fix_storage_usage[tu1] +
p_fix_storage_usage[tu2]``.

Both scenarios below FAIL against the pre-fix constraint and PASS after.
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

from .conftest import _time_axes, solver_options


def _solve(data: FlexData):
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


def _storage_4t_fix_usage(*, cap1: float, cap2: float,
                          demand_per_step: float) -> FlexData:
    """Build a child dispatch LP: 1 period × 4 fine steps × 2 coarse windows.

    Storage ``s`` (bind_forward_only, start full) discharges through
    ``discharge_p`` (s → n) to meet ``demand_per_step`` at t02, t03, t04
    (t01 is the forward-only first step: flow pinned to 0).  fix_usage
    caps come from two coarse windows tu1 (t01,t02) and tu2 (t03,t04).
    """
    periods = ["d1"]
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(periods, 4)
    steps = ["t01", "t02", "t03", "t04"]

    # nodeBalance for regular node n + storage node s.
    nb = pl.DataFrame({"n": ["n", "s"]})
    nb_dt = nb.join(dt, how="cross")

    # Demand on n at t02..t04 (negative inflow = load); 0 elsewhere.
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.when(
            (pl.col("n") == "n") & (pl.col("t").is_in(["t02", "t03", "t04"])))
            .then(pl.lit(-demand_per_step))
            .otherwise(pl.lit(0.0))
        ).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))

    # Storage node s.
    nodeState = pl.DataFrame({"n": ["s"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first().select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last().select("n", "d", "t"))
    p_state_unitsize = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [10.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [1.0]}))
    p_state_self_discharge = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [0.0]}))
    p_state_existing_capacity = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [10.0]}))
    # State starts full (1.0 normalised = 10 MWh) so plenty to discharge.
    p_state_start = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [1.0]}))

    # dtttdt: cyclic step_previous over 4 timesteps within d1.
    prev = {"t01": "t04", "t02": "t01", "t03": "t02", "t04": "t03"}
    dtttdt = pl.DataFrame({
        "d": periods * 4,
        "t": steps,
        "t_previous": [prev[t] for t in steps],
        "t_previous_within_timeset": [prev[t] for t in steps],
        "d_previous": periods * 4,
        "t_previous_within_solve": [prev[t] for t in steps],
    })
    # forward_only drops the first (d, t) so no state-change term at t01.
    dtttdt_fo = dtttdt.sort("d", "t").slice(1)

    # Discharge process s → n (DIRECT, no online; source_eff, slope 1).
    pss = pl.DataFrame({"p": ["discharge_p"], "source": ["s"], "sink": ["n"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
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
        dt.with_columns(p=pl.lit("discharge_p"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("c_s"), value=pl.lit(0.0))
          .select("c", "d", "t", "value"))

    # fix_usage handoff — TWO coarse windows.
    #   tu1 covers fine t01, t02 ; tu2 covers fine t03, t04.
    n_fix = pl.DataFrame({"n": ["s"]})
    ndt_fix = pl.DataFrame({"n": ["s", "s"], "d": ["d1", "d1"],
                            "t": ["tu1", "tu2"]})
    p_fix = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["s", "s"], "d": ["d1", "d1"],
                      "t": ["tu1", "tu2"], "value": [cap1, cap2]}))
    period_branch = pl.DataFrame({"d_upper": ["d1"], "d": ["d1"]})
    period_last = pl.DataFrame({"d": ["d1"]})
    dtt = pl.DataFrame({
        "d": ["d1", "d1", "d1", "d1"],
        "t": ["t01", "t02", "t03", "t04"],
        "t_upper": ["tu1", "tu1", "tu2", "tu2"],
    })

    return FlexData(
        dt=dt, p_step_duration=p_step, p_timestep_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=p_state_unitsize,
        p_state_upper=p_state_upper,
        p_state_self_discharge=p_state_self_discharge,
        p_state_existing_capacity=p_state_existing_capacity,
        p_state_start=p_state_start,
        dtttdt=dtttdt,
        storage_bind_forward_only=pl.DataFrame({"n": ["s"]}),
        storage_fix_start=pl.DataFrame({"n": ["s"]}),
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


def _total_discharge(sol) -> float:
    v = sol.value("v_flow").filter(pl.col("p") == "discharge_p")
    return float(v["value"].sum())


def test_fix_usage_multiwindow_aggregate_non_binding():
    """Per-period aggregate cap = cap1+cap2 = 4 ≥ demand 3 → demand met.

    Demand = 1 MWh at each of t02, t03, t04 (total 3 MWh).  Coarse caps
    tu1=3, tu2=1 → per-period aggregate = 4.  The fix lets the storage
    discharge all 3 MWh (obj 0).

    Pre-fix the RHS collapsed to only tu2 (the terminal window) = 1, so
    discharge would be capped at 1, leaving 2 MWh of unmet-demand slack
    (obj 2e6).  This assertion therefore FAILS on the old constraint.
    """
    data = _storage_4t_fix_usage(cap1=3.0, cap2=1.0, demand_per_step=1.0)
    pb, sol = _solve(data)
    assert sol.optimal
    assert "node_storage_usage_fix_le" in set(pb.cstr_names()), (
        "Constraint must be built when fix_usage frames are populated")
    assert sol.obj == pytest.approx(0.0, abs=1e-3), (
        f"Aggregate cap 4 ≥ demand 3 → no slack, obj 0; got {sol.obj}. "
        f"If obj≈2e6 the RHS is still the single-terminal-window value "
        f"(cap2=1) instead of the per-period sum (cap1+cap2=4).")
    assert _total_discharge(sol) == pytest.approx(3.0, abs=1e-3), (
        f"Storage should discharge all 3 MWh of demand; "
        f"got {_total_discharge(sol)}")


def test_fix_usage_multiwindow_aggregate_binding():
    """Per-period aggregate cap = cap1+cap2 = 2 < demand 3 → cap binds at 2.

    Coarse caps tu1=1, tu2=1 → per-period aggregate = 2.  Demand 3 →
    discharge capped at 2, 1 MWh slack (obj 1e6).

    This proves the cap is the SUM of both windows (2), not a single
    window: the pre-fix code would cap at cap2=1 (terminal window) → 2
    MWh slack → obj 2e6.  Asserting obj == 1e6 FAILS on the old code and
    PASSES on the fix.
    """
    data = _storage_4t_fix_usage(cap1=1.0, cap2=1.0, demand_per_step=1.0)
    pb, sol = _solve(data)
    assert sol.optimal
    assert "node_storage_usage_fix_le" in set(pb.cstr_names())
    assert sol.obj == pytest.approx(1_000_000.0, rel=1e-4), (
        f"Aggregate cap 2 < demand 3 → 1 MWh slack, obj 1e6; got {sol.obj}. "
        f"obj≈2e6 means the cap is still a single window (1), not the "
        f"per-period sum (2).")
    assert _total_discharge(sol) == pytest.approx(2.0, abs=1e-3), (
        f"Discharge should be capped at the aggregate 2 MWh; "
        f"got {_total_discharge(sol)}")
