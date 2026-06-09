"""Surface B — ``nodeBalancePeriod_eq`` (balance_within_period nodes).

Numerically exercises and locks in the ``nodeBalancePeriod_eq`` constraint
added in commits 36249183 / 23ab3b72 (polars port of flextool.mod:2056).

A ``balance_within_period`` node is NOT in ``nodeBalance`` — it has NO
per-(d,t) balance row.  Instead a single per-(n, period) row pins

    Σ_t(sink − source + slack_up − slack_down)  ==  Σ_t(−inflow)

so the node conserves energy across the whole period even though the
individual timesteps need not balance.  Without this constraint there is
NO equation tying the priced source to the demand at such a node, so the
cost-minimiser would set the source to 0 and feed the load "from nowhere"
(objective 0) — that is exactly the bug this feature fixes.

The toy here is the smallest model that exercises the feature:

* ONE period ``d1`` with TWO timesteps ``t01`` / ``t02`` (so "Σ over t"
  is non-trivial — the per-step demand differs between the two steps);
* ONE ``balance_within_period`` node ``N`` (in ``nodeBalancePeriod``,
  absent from ``nodeBalance`` and ``nodeState``);
* ONE priced source process ``p`` (commodity ``FUEL`` at price 1) feeding
  ``N`` through the sink-side flow, capped per step;
* the load is the node's negative inflow ``[-3, -10]`` (total −13), so the
  per-period draw is 13 MWh while neither single step balances on its own.

This file builds ``FlexData`` directly in memory in the same style as the
shared constraint fixtures (``conftest.py``); it adds NO fixture and
mutates none.
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


# Per-step inflow on the period node (negative = demand draw).  The two
# steps differ so the per-step balance is genuinely non-trivial; the
# per-period total draw is 3 + 10 = 13 MWh.
_DEMAND_T01 = -3.0
_DEMAND_T02 = -10.0
_PERIOD_DRAW = -(_DEMAND_T01 + _DEMAND_T02)  # 13.0


def _solve(data: FlexData) -> tuple[Problem, "Any"]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


def _period_node_data(flow_cap: float, penalty: float = 1e6) -> FlexData:
    """Smallest model with one balance_within_period node ``N``.

    ``flow_cap`` is the per-step upper bound on the source→N flow; set it
    above the per-step demand (10) to allow full supply, or below the
    per-period draw to force node slack (penalised at ``penalty``).
    """
    # 1 period, 2 timesteps; step_duration ≡ 1 so MW == MWh per step.
    dt = pl.DataFrame({"d": ["d1", "d1"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))

    # ``N`` is a PERIOD node: it lives in nodeBalancePeriod, NOT in
    # nodeBalance (no per-(d,t) row) and NOT in nodeState (not storage).
    nbp = pl.DataFrame({"n": ["N"]})
    # A real model also has ordinary per-(d,t) balance nodes; ``B`` is a
    # trivial one (zero inflow, no flows) so the model is well-formed (the
    # engine's slack-var widening expects nodeBalance to be non-empty when
    # period nodes exist).  ``B`` balances at 0 with zero slack and adds no
    # cost, so it never perturbs the numbers asserted on ``N``.
    nodeBalance = pl.DataFrame({"n": ["B"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")

    n_dt = pl.DataFrame({
        "n": ["N", "N", "B", "B"],
        "d": ["d1", "d1", "d1", "d1"],
        "t": ["t01", "t02", "t01", "t02"],
    })
    p_inflow = Param(("n", "d", "t"), n_dt.with_columns(
        value=pl.when(pl.col("n") == "B").then(0.0)
              .when(pl.col("t") == "t01").then(_DEMAND_T01)
              .otherwise(_DEMAND_T02)
    ).select("n", "d", "t", "value"))
    # Finite, priced node slack so an over-tight supply budget falls back
    # onto vq_state_up/down instead of going infeasible.
    p_pen_up = Param(("n", "d", "t"),
        n_dt.with_columns(value=pl.lit(penalty)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        n_dt.with_columns(value=pl.lit(penalty)).select("n", "d", "t", "value"))

    # period_in_use_set — required gate for nodeBalancePeriod_eq.
    period_in_use_set = pl.DataFrame({"d": ["d1"]})

    # Source process ``p``: commodity FUEL → N (sink side feeds N).
    pss = pl.DataFrame({"p": ["p"], "source": ["source_N"], "sink": ["N"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_commodity_eff = pl.DataFrame(
        {"p": ["p"], "source": ["source_N"], "sink": ["N"], "c": ["FUEL"]})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(flow_cap))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    p_commodity_price = Param(("c", "d", "t"),
        dt.with_columns(c=pl.lit("FUEL"), value=pl.lit(1.0))
          .select("c", "d", "t", "value"))

    return FlexData(
        dt=dt, p_step_duration=p_step, p_rp_cost_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nodeBalance, nodeBalance_dt=nodeBalance_dt,
        nodeBalancePeriod=nbp,
        period_in_use_set=period_in_use_set,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        process_source_sink=pss,
        process_source_sink_eff=pss_eff,
        process_source_sink_noEff=pss_noEff,
        pss_dt=pss_dt, flow_to_n=flow_to_n,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope, p_commodity_price=p_commodity_price,
    )


def _flow_total(sol: Any) -> float:
    """Σ_t v_flow for the single source→N process."""
    vf = sol.value("v_flow")
    return float(vf["value"].sum())


# ---------------------------------------------------------------------------
# (1)/(2) — per-period conservation + the constraint exists with the right
# shape (one row per (N, period)).

def test_nodeBalancePeriod_conserves_energy_over_period():
    """Per-period conservation: the priced source is forced to supply
    EXACTLY the period's total draw (13 MWh), even though no single
    timestep balances on its own.

    Cap = 100 per step (well above the per-step demand of 10), so the
    source is free to choose how to spread its supply over the two steps;
    the constraint only pins the SUM.  WITHOUT nodeBalancePeriod_eq the
    node has no balance row at all → cost-min sets the source to 0 and
    objective 0; the assertions below would then fail.
    """
    data = _period_node_data(flow_cap=100.0)
    pb, sol = _solve(data)
    assert sol.optimal

    # (2) The constraint exists with EXACTLY one row per (N, period).
    names = set(pb.cstr_names())
    assert "nodeBalancePeriod_eq" in names
    # There is no per-(d,t) balance node here, so nodeBalance_eq must be
    # absent (or empty) — the period node is governed solely by the
    # per-period row.
    assert pb.cstr_row_count("nodeBalancePeriod_eq") == 1  # (N, d1)
    rec = pb.cstrs_named("nodeBalancePeriod_eq")[0]
    over = rec.over
    assert set(over.columns) >= {"n", "d"}
    assert over.height == 1
    assert over["n"].to_list() == ["N"]
    assert over["d"].to_list() == ["d1"]
    # The row is per-period, NOT per-timestep: no ``t`` axis on the row set.
    assert "t" not in over.columns

    # (1) Per-period conservation.  Σ_t source flow == Σ_t demand (13).
    src_total = _flow_total(sol)
    assert src_total == pytest.approx(_PERIOD_DRAW, rel=1e-9)  # 13.0

    # Node slack must be unused — the source fully covers the draw.
    up = float(sol.value("vq_state_up").filter(pl.col("n") == "N")["value"].sum())
    dn = float(sol.value("vq_state_down").filter(pl.col("n") == "N")["value"].sum())
    assert up == pytest.approx(0.0, abs=1e-9)
    assert dn == pytest.approx(0.0, abs=1e-9)


def test_nodeBalancePeriod_row_is_per_period_not_per_timestep():
    """Structural lock-in: the per-period row collapses ``t``.

    With a 2-step period the per-(d,t) balance form would yield 2 rows for
    node N; nodeBalancePeriod_eq must yield exactly 1.  This guards against
    a regression that forgot the ``Sum(..., over=("t",))`` t-collapse.
    """
    data = _period_node_data(flow_cap=100.0)
    pb = Problem()
    build_flextool(pb, data)
    # 2 timesteps in the period — a per-step form would be 2 rows.
    assert data.dt.height == 2
    assert pb.cstr_row_count("nodeBalancePeriod_eq") == 1


# ---------------------------------------------------------------------------
# (3) — over-tight supply budget → priced node slack, NOT infeasibility.

def test_nodeBalancePeriod_overtight_budget_uses_slack_not_infeasible():
    """Tighten the per-step source cap to 4 → max period supply = 8 < the
    13 MWh draw.  The solve must stay FEASIBLE and cover the 5 MWh
    shortfall with node slack ``vq_state_up`` (priced at the node penalty),
    rather than going infeasible.  This verifies the slack-var domain was
    widened to period nodes.
    """
    penalty = 1e6
    data = _period_node_data(flow_cap=4.0, penalty=penalty)
    pb, sol = _solve(data)
    assert sol.optimal  # feasible, NOT infeasible

    # Source runs flat out: 4 + 4 = 8 MWh over the period.
    src_total = _flow_total(sol)
    assert src_total == pytest.approx(8.0, rel=1e-9)

    # The 5 MWh shortfall is carried by vq_state_up on N.
    up = float(sol.value("vq_state_up").filter(pl.col("n") == "N")["value"].sum())
    dn = float(sol.value("vq_state_down").filter(pl.col("n") == "N")["value"].sum())
    expected_slack = _PERIOD_DRAW - 8.0  # 13 - 8 = 5
    assert up == pytest.approx(expected_slack, rel=1e-6)  # 5.0
    assert dn == pytest.approx(0.0, abs=1e-6)

    # Period balance still closes: source + slack_up == draw.
    assert src_total + up == pytest.approx(_PERIOD_DRAW, rel=1e-9)


def test_nodeBalancePeriod_excludes_nodeState_nodes():
    """Guard on the ``n not in nodeState`` restriction (mod:2056): if a
    period node were ALSO a storage (nodeState) node it must be dropped
    from nodeBalancePeriod_eq's row set (storage carries its own balance).

    Layer a single-node nodeState onto the toy and assert the per-period
    row count collapses to 0 (the anti-join on nodeState removed N).
    """
    data = _period_node_data(flow_cap=100.0)
    # Promote ``N`` to a storage (nodeState) node — supply the minimal
    # storage scaffolding the ``storage`` feature gate requires.  The
    # ``n not in nodeState`` anti-join (mod:2056) must then drop ``N`` from
    # the nodeBalancePeriod_eq row set.
    nodeState = pl.DataFrame({"n": ["N"]})
    dtttdt = pl.DataFrame({
        "d": ["d1", "d1"],
        "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["d1", "d1"],
        "t_previous_within_solve": ["t02", "t01"],
    })
    p_state_unitsize = Param(("n",), pl.DataFrame({"n": ["N"], "value": [1.0]}))
    p_state_upper = Param(("n", "d"),
        pl.DataFrame({"n": ["N"], "d": ["d1"], "value": [100.0]}))
    p_state_existing = Param(("n", "d"),
        pl.DataFrame({"n": ["N"], "d": ["d1"], "value": [100.0]}))
    p_self_discharge = Param(("n",), pl.DataFrame({"n": ["N"], "value": [0.0]}))
    data = dataclasses.replace(
        data, nodeState=nodeState, dtttdt=dtttdt,
        p_state_unitsize=p_state_unitsize, p_state_upper=p_state_upper,
        p_state_existing_capacity=p_state_existing,
        p_state_self_discharge=p_self_discharge,
    )
    pb = Problem()
    build_flextool(pb, data)
    assert pb.cstr_row_count("nodeBalancePeriod_eq") == 0
