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


# ---------------------------------------------------------------------------
# (4) — SOURCE-side drain.  The tests above drain the period node via the
# SINK side (``flow_to_n`` into N) + inflow only, so they never exercise
# the ``source_eff`` / ``source_noEff`` terms of ``nodeBalancePeriod_eq``.
# That gap let a real bug ship: the cascade producer
# (``_derived_params.apply_derived_b``) seeded ``flow_from_nodeBalance_eff``
# from ``nodeBalance`` ALONE, dropping every ``balance_within_period`` node,
# so a period node that is the SOURCE on an arc (e.g. Rivendell GAS → gas
# plant) had NO source term in its period row → the constraint was VACUOUS
# and the source drew for free.  The model.py assembly was correct; the
# fixtures simply never reached it.
#
# This builder drains the period node ``G`` as a SOURCE: a conversion
# process ``p`` draws FROM ``G`` (source) and delivers INTO a regular
# per-(d,t) balance node ``B`` (the load).  ``B``'s per-step balance forces
# ``v_flow[t] == demand[t]`` each step, hence Σ_t v_flow == period demand.
# ``G``'s period balance is then
#
#     −Σ_t(v_flow · slope)  +  Σ_t(slack_up − slack_down)  ==  −Σ_t inflow_G
#
# i.e. the SOURCE draw out of G must be matched by G's own supply (inflow)
# plus period slack.  With inflow_G < demand the shortfall is forced onto
# ``vq_state_up`` (priced).  WITHOUT the ``source_eff`` term the period row
# degenerates to  slack_up − slack_down == −Σ_t inflow_G  — G's drain
# vanishes, B is fed for free, and the slack takes a different value.

# Per-step demand on the downstream load node B (negative = draw).
_B_DEMAND_T01 = -3.0
_B_DEMAND_T02 = -10.0
_B_PERIOD_DEMAND = -(_B_DEMAND_T01 + _B_DEMAND_T02)  # 13.0
# Per-step supply (positive inflow) INTO the period source node G.
_G_INFLOW_PER_STEP = 4.0
_G_PERIOD_INFLOW = _G_INFLOW_PER_STEP * 2  # 8.0
# The period balance forces the shortfall onto G's up-slack.
_G_EXPECTED_UP_SLACK = _B_PERIOD_DEMAND - _G_PERIOD_INFLOW  # 5.0


def _period_axes_base():
    """Shared 1-period / 2-timestep time axes (step_duration ≡ 1)."""
    dt = pl.DataFrame({"d": ["d1", "d1"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_rp = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    return dt, p_step, p_rp, p_infl, p_psh


def _source_drained_period_data(*, slope: float = 1.0,
                                b_penalty: float = 1e9,
                                g_penalty: float = 1.0) -> FlexData:
    """Smallest model where a ``balance_within_period`` node ``G`` is
    drained from the SOURCE side.

    Topology::

        (inflow +4/step) → [G : balance_within_period]
                              │  source-side draw  (flow_from_nodeBalance_eff)
                              ▼
                       process p  (efficiency unit, slope=``slope``)
                              │  flow_to_n into B
                              ▼
                  [B : nodeBalance] ← load (−3, −10)

    ``B`` is an ordinary per-(d,t) balance node.  Its load slack is priced
    far above ``G``'s period slack (``b_penalty`` ≫ ``g_penalty``), so the
    cost-minimiser serves B's full demand by drawing through ``p`` out of
    ``G`` rather than failing B — forcing ``Σ_t v_flow·slope == 13``.  That
    13 MWh source draw is then reconciled in ``G``'s period row against
    ``G``'s own supply (inflow 8) + period slack (the binding 5 MWh).
    """
    dt, p_step, p_rp, p_infl, p_psh = _period_axes_base()

    # G is the PERIOD (source) node; B is the regular per-(d,t) node.
    nbp = pl.DataFrame({"n": ["G"]})
    nodeBalance = pl.DataFrame({"n": ["B"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    period_in_use_set = pl.DataFrame({"d": ["d1"]})

    # Inflow lives on BOTH nodes: +supply on G, −load on B.  (n,d,t) frame.
    infl_nodes = pl.DataFrame({"n": ["G", "G", "B", "B"],
                               "d": ["d1"] * 4,
                               "t": ["t01", "t02", "t01", "t02"]})
    p_inflow = Param(("n", "d", "t"), infl_nodes.with_columns(
        value=pl.when(pl.col("n") == "G").then(pl.lit(_G_INFLOW_PER_STEP))
              .when(pl.col("t") == "t01").then(pl.lit(_B_DEMAND_T01))
              .otherwise(pl.lit(_B_DEMAND_T02))
    ).select("n", "d", "t", "value"))
    # B's load-shed slack is priced far above G's period slack so the
    # optimum fully serves B (drawing through G) rather than failing it —
    # this is what fixes Σ_t v_flow at the period demand.
    _pen_expr = (pl.when(pl.col("n") == "B").then(pl.lit(b_penalty))
                   .otherwise(pl.lit(g_penalty)))
    p_pen_up = Param(("n", "d", "t"), infl_nodes.with_columns(
        value=_pen_expr).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"), infl_nodes.with_columns(
        value=_pen_expr).select("n", "d", "t", "value"))

    # Process p: G (source) → B (sink).  ``p`` is an efficiency unit, so its
    # source-side draw lands in ``flow_from_nodeBalance_eff`` (slope-scaled).
    pss = pl.DataFrame({"p": ["p"], "source": ["G"], "sink": ["B"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    # Sink-side: p feeds B.
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    # Source-side: p draws from G — THIS is the row the bug dropped.  In a
    # real DB run the cascade builds this from process_source_sink_eff
    # filtered to ``source ∈ nodeBalance ∪ nodeBalancePeriod``; here we set
    # it explicitly so the in-engine model assembly is exercised directly.
    flow_from_nodeBalance_eff = pss.with_columns(n=pl.col("source"))

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p"), value=pl.lit(slope))
          .select("p", "d", "t", "value"))
    # No commodity on the G→B arc — G is supplied by node inflow, not a
    # priced commodity.  The ``processes`` feature gate still requires these
    # frames be non-None, so supply empty-schema frames.
    flow_from_commodity_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame(schema={"c": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                             "value": pl.Float64}))

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
        flow_from_nodeBalance_eff=flow_from_nodeBalance_eff,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_commodity_price=p_commodity_price,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope,
    )


def test_nodeBalancePeriod_source_term_present_in_row():
    """In-engine MODEL guard (diagnosis item #3): the ``source_eff`` term
    of a period node lands in the (n=G, d) period row.

    The downstream load on ``B`` forces ``Σ_t v_flow == 13`` (per-step
    balance on B), so the source draw out of ``G`` is fixed at 13.  ``G``'s
    period balance must then carry that draw on the LHS — the constraint
    cannot be satisfied with the source term absent unless the slack absorbs
    a value consistent with the *draw*, not just the inflow.  We assert the
    binding slack equals ``demand − inflow`` (5), which is ONLY true when
    the source term is present and summed over t into the (G, d) row.
    """
    data = _source_drained_period_data()
    pb = Problem()
    build_flextool(pb, data)

    # Structural: one (G, d1) period row.
    assert "nodeBalancePeriod_eq" in set(pb.cstr_names())
    assert pb.cstr_row_count("nodeBalancePeriod_eq") == 1
    rec = pb.cstrs_named("nodeBalancePeriod_eq")[0]
    assert rec.over["n"].to_list() == ["G"]
    assert rec.over["d"].to_list() == ["d1"]
    # The source-side draw row exists for the period node G.
    assert (data.flow_from_nodeBalance_eff
            .filter(pl.col("source") == "G").height) == 1

    sol = pb.solve(options=solver_options())
    assert sol.optimal

    # B's per-step balance forces the source draw out of G to equal the
    # period demand (13), regardless of the slope (=1 here).
    src_total = _flow_total(sol)
    assert src_total == pytest.approx(_B_PERIOD_DEMAND, rel=1e-9)  # 13.0

    # G's period balance: −draw + up − dn == −inflow  ⇒  up = draw − inflow.
    up = float(sol.value("vq_state_up").filter(pl.col("n") == "G")["value"].sum())
    dn = float(sol.value("vq_state_down").filter(pl.col("n") == "G")["value"].sum())
    assert up == pytest.approx(_G_EXPECTED_UP_SLACK, rel=1e-6)  # 5.0
    assert dn == pytest.approx(0.0, abs=1e-6)
    # Period balance closes: inflow + up == draw.
    assert _G_PERIOD_INFLOW + up == pytest.approx(src_total, rel=1e-9)


def test_nodeBalancePeriod_source_term_binds_without_it_vacuous():
    """Decisive numerical contrast: with the ``source_eff`` term the period
    constraint BINDS the source draw to inflow + slack; were the term absent
    (the shipped bug) the row would reduce to ``up − dn == −inflow_G`` and
    the up-slack would be 0 (inflow alone closes a source-less balance),
    leaving the 13 MWh draw unaccounted.

    We assert the slack takes the value that is only attainable WITH the
    source term (5.0), and is strictly different from the source-less value
    (0.0).  ``_G_EXPECTED_UP_SLACK`` is the witness.
    """
    data = _source_drained_period_data(slope=1.0)
    pb, sol = _solve(data)
    assert sol.optimal

    up = float(sol.value("vq_state_up").filter(pl.col("n") == "G")["value"].sum())
    # WITH source term: up = demand − inflow = 5.0.
    assert up == pytest.approx(_G_EXPECTED_UP_SLACK, rel=1e-6)
    # The source-less degenerate value would be 0.0 — assert we are NOT there.
    assert up > 1.0  # strictly bound away from the vacuous 0.0


def test_nodeBalancePeriod_source_term_slope_scaled():
    """The SOURCE term is slope-scaled (``v_flow · unitsize · slope``);
    the SINK term into B is not.  This pins the ``* d.p_slope`` factor that
    the source-side branch of ``nodeBalancePeriod_eq`` applies (model.py
    line ~881) — distinct from the sink-side ``flow_to_n`` term which omits
    it.

    B's per-step balance is in delivered-flow units, so v_flow is still 13
    (unchanged by slope).  But the energy drawn OUT of G is
    ``Σ_t v_flow · slope`` = 13·2 = 26, so the period balance forces a much
    larger up-slack: ``26 − inflow(8)`` = 18.  Were the slope factor missing
    the slack would collapse back to the slope-1 value of 5 — so the 18 is
    the witness that ``* p_slope`` is applied to the source term.
    """
    slope = 2.0
    data = _source_drained_period_data(slope=slope)
    pb, sol = _solve(data)
    assert sol.optimal

    # The sink-side term into B is NOT slope-scaled → v_flow tracks demand.
    src_flow = _flow_total(sol)
    assert src_flow == pytest.approx(_B_PERIOD_DEMAND, rel=1e-9)  # 13.0

    # Source-side energy out of G = Σ v_flow·slope = 26 → up-slack = 26−8.
    g_draw = _B_PERIOD_DEMAND * slope                       # 26.0
    expected_up = g_draw - _G_PERIOD_INFLOW                 # 18.0
    up = float(sol.value("vq_state_up").filter(pl.col("n") == "G")["value"].sum())
    assert up == pytest.approx(expected_up, rel=1e-6)       # 18.0
    # And strictly larger than the slope-1 value — the slope factor binds.
    assert up > _G_EXPECTED_UP_SLACK + 1.0


# ---------------------------------------------------------------------------
# (5) — NON-UNIFORM rp_cost_weight.  Every fixture above uses
# ``p_rp_cost_weight ≡ 1`` and so cannot catch the annualization bug: the
# per-period balance must weight each ``(d, t)`` flow term by
# ``rp_cost_weight`` BEFORE collapsing ``t`` (the per-(d,t) ``nodeBalance_eq``
# correctly omits it — each step balances locally — but a PERIOD/annual
# balance integrates each timeslice into its annual contribution).  Without
# the weight the LP balances UNWEIGHTED timeslice sums and games it by
# importing in low-weight timeslices while drawing in high-weight ones — the
# real Rivendell S16 symptom (import/draw ratio ≈ 0.838 in annual terms,
# zero slack).  This builder reuses the SOURCE-drain topology of section (4)
# but assigns DISTINCT weights ``w = [1.0, 3.0]`` to the two timesteps so the
# weighted and unweighted draws differ, and asserts the binding period slack
# takes the rp-WEIGHTED value (NOT the unweighted one), which only holds
# WITH the weight folded into both the LHS flow terms and the RHS inflow.

# rp_cost_weight per timestep — deliberately NON-uniform.
_W_T01 = 1.0
_W_T02 = 3.0
# B's per-step demand (negative = draw) forces v_flow[t] == |demand[t]|.
_BW_DEMAND_T01 = -3.0
_BW_DEMAND_T02 = -10.0
# Per-step supply into the period source node G.
_GW_INFLOW_PER_STEP = 4.0

# rp-WEIGHTED draw out of G  (Σ_t w·v_flow, v_flow == |demand|):
_GW_WEIGHTED_DRAW = (_W_T01 * -_BW_DEMAND_T01) + (_W_T02 * -_BW_DEMAND_T02)   # 1·3 + 3·10 = 33
# rp-WEIGHTED inflow into G  (Σ_t w·inflow):
_GW_WEIGHTED_INFLOW = (_W_T01 + _W_T02) * _GW_INFLOW_PER_STEP                 # (1+3)·4 = 16
# WITH the fix the period slack closes the rp-weighted balance:
_GW_EXPECTED_UP_SLACK = _GW_WEIGHTED_DRAW - _GW_WEIGHTED_INFLOW               # 33 − 16 = 17
# The PRE-FIX (uniform-weight) code would instead balance the UNWEIGHTED
# sums, giving slack = unweighted_draw − unweighted_inflow:
_GW_UNWEIGHTED_SLACK = ((-_BW_DEMAND_T01) + (-_BW_DEMAND_T02)) - (2 * _GW_INFLOW_PER_STEP)  # 13 − 8 = 5


def _nonuniform_weight_period_data(*, b_penalty: float = 1e9,
                                   g_penalty: float = 1.0) -> FlexData:
    """SOURCE-drain topology (period node ``G`` → process ``p`` → load ``B``)
    with NON-UNIFORM ``p_rp_cost_weight`` across the two timesteps.

    Identical wiring to :func:`_source_drained_period_data` (slope ≡ 1) but
    with ``p_rp_cost_weight = [_W_T01, _W_T02] = [1.0, 3.0]``.  ``B``'s load
    slack is priced far above ``G``'s period slack so the optimum serves B's
    full per-step demand by drawing through ``p`` out of ``G`` (forcing
    ``v_flow[t] == |demand[t]|`` each step); ``G``'s ANNUAL (rp-weighted)
    period balance then reconciles the weighted draw against the weighted
    inflow + period slack.
    """
    dt = pl.DataFrame({"d": ["d1", "d1"], "t": ["t01", "t02"]})
    p_step = Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))
    # NON-UNIFORM rp_cost_weight — the crux of this test.
    p_rp = Param(("d", "t"), dt.with_columns(
        value=pl.when(pl.col("t") == "t01").then(pl.lit(_W_T01))
              .otherwise(pl.lit(_W_T02))))
    p_infl = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))
    p_psh = Param(("d",), pl.DataFrame({"d": ["d1"], "value": [1.0]}))

    nbp = pl.DataFrame({"n": ["G"]})
    nodeBalance = pl.DataFrame({"n": ["B"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")
    period_in_use_set = pl.DataFrame({"d": ["d1"]})

    infl_nodes = pl.DataFrame({"n": ["G", "G", "B", "B"],
                               "d": ["d1"] * 4,
                               "t": ["t01", "t02", "t01", "t02"]})
    p_inflow = Param(("n", "d", "t"), infl_nodes.with_columns(
        value=pl.when(pl.col("n") == "G").then(pl.lit(_GW_INFLOW_PER_STEP))
              .when(pl.col("t") == "t01").then(pl.lit(_BW_DEMAND_T01))
              .otherwise(pl.lit(_BW_DEMAND_T02))
    ).select("n", "d", "t", "value"))
    _pen_expr = (pl.when(pl.col("n") == "B").then(pl.lit(b_penalty))
                   .otherwise(pl.lit(g_penalty)))
    p_pen_up = Param(("n", "d", "t"), infl_nodes.with_columns(
        value=_pen_expr).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"), infl_nodes.with_columns(
        value=_pen_expr).select("n", "d", "t", "value"))

    pss = pl.DataFrame({"p": ["p"], "source": ["G"], "sink": ["B"]})
    pss_eff = pss.clone()
    pss_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    pss_dt = pss.join(dt, how="cross")
    flow_to_n = pss.with_columns(n=pl.col("sink"))
    flow_from_nodeBalance_eff = pss.with_columns(n=pl.col("source"))

    p_unitsize = Param(("p",), pl.DataFrame({"p": ["p"], "value": [1.0]}))
    p_flow_upper = Param(("p", "source", "sink", "d", "t"),
        pss_dt.with_columns(value=pl.lit(100.0))
              .select("p", "source", "sink", "d", "t", "value"))
    p_slope = Param(("p", "d", "t"),
        dt.with_columns(p=pl.lit("p"), value=pl.lit(1.0))
          .select("p", "d", "t", "value"))
    flow_from_commodity_eff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    flow_from_commodity_noEff = pl.DataFrame(
        schema={"p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8, "c": pl.Utf8})
    p_commodity_price = Param(("c", "d", "t"),
        pl.DataFrame(schema={"c": pl.Utf8, "d": pl.Utf8, "t": pl.Utf8,
                             "value": pl.Float64}))

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
        flow_from_nodeBalance_eff=flow_from_nodeBalance_eff,
        flow_from_commodity_eff=flow_from_commodity_eff,
        flow_from_commodity_noEff=flow_from_commodity_noEff,
        p_commodity_price=p_commodity_price,
        p_unitsize=p_unitsize, p_flow_upper=p_flow_upper,
        p_slope=p_slope,
    )


def test_nodeBalancePeriod_nonuniform_rp_weight_conserves_annual_energy():
    """Decisive regression for the rp_cost_weight annualization fix.

    With NON-UNIFORM weights ``w = [1.0, 3.0]`` the rp-weighted draw out of
    ``G`` is ``1·3 + 3·10 = 33`` while the UNWEIGHTED draw is ``3 + 10 = 13``;
    the weighted inflow is ``(1+3)·4 = 16`` vs unweighted ``8``.  ``B``'s
    per-step balance fixes ``v_flow == |demand|`` (b_penalty ≫ g_penalty), so
    the only free variable in ``G``'s period row is its slack.

    WITH the fix (every LHS flow/slack term AND the RHS inflow weighted by
    ``rp_cost_weight`` before the t-sum) the period row reads, in annual
    energy units::

        −Σ_t w·v_flow + (up − dn) == −Σ_t w·inflow
        ⇒ up = Σ_t w·v_flow − Σ_t w·inflow = 33 − 16 = 17   (dn = 0)

    The witness ``up == 17`` is attainable ONLY with the weight applied.  The
    pre-fix code balances the UNWEIGHTED sums and would yield ``up = 13 − 8 =
    5`` instead (asserted to be a strictly different value below) — i.e. the
    pre-fix code FAILS this assertion.  We also reconstruct the rp-weighted
    period balance directly from the solved flows and assert it closes to 0,
    and that the import (weighted inflow + slack) equals the rp-WEIGHTED draw,
    NOT the unweighted draw.
    """
    data = _nonuniform_weight_period_data()

    # Sanity: the weights really are non-uniform (mirrors the S16 hypothesis
    # check — a uniform weight could not expose the bug).
    wvals = sorted(data.p_rp_cost_weight.frame["value"].unique().to_list())
    assert wvals == [_W_T01, _W_T02]
    assert _W_T01 != _W_T02
    # The weighted and unweighted slacks must differ, else the test is
    # vacuous (cannot distinguish fixed from buggy code).
    assert _GW_EXPECTED_UP_SLACK != _GW_UNWEIGHTED_SLACK

    pb = Problem()
    build_flextool(pb, data)
    assert "nodeBalancePeriod_eq" in set(pb.cstr_names())
    assert pb.cstr_row_count("nodeBalancePeriod_eq") == 1
    rec = pb.cstrs_named("nodeBalancePeriod_eq")[0]
    assert rec.over["n"].to_list() == ["G"]

    sol = pb.solve(options=solver_options())
    assert sol.optimal

    # B's per-step balance fixes each step's flow at the per-step demand.
    vf = sol.value("v_flow").sort("t")
    flow_by_t = {row["t"]: row["value"] for row in vf.iter_rows(named=True)}
    assert flow_by_t["t01"] == pytest.approx(-_BW_DEMAND_T01, rel=1e-9)   # 3
    assert flow_by_t["t02"] == pytest.approx(-_BW_DEMAND_T02, rel=1e-9)   # 10

    up = float(sol.value("vq_state_up").filter(pl.col("n") == "G")["value"].sum())
    dn = float(sol.value("vq_state_down").filter(pl.col("n") == "G")["value"].sum())

    # WITH the fix: the period slack closes the rp-WEIGHTED balance (17),
    # NOT the unweighted balance (5).  This is the assertion the pre-fix
    # (uniform-weight) code FAILS.
    assert up == pytest.approx(_GW_EXPECTED_UP_SLACK, rel=1e-6)           # 17
    assert dn == pytest.approx(0.0, abs=1e-6)
    # Explicitly reject the unweighted (buggy) value as the witness.
    assert abs(up - _GW_UNWEIGHTED_SLACK) > 1.0                            # 17 ≠ 5

    # Reconstruct the rp-WEIGHTED period balance from the solved flows and
    # assert it closes to zero:  Σ_t w·(−v_flow + inflow_G) + up − dn == 0.
    weighted_draw = (_W_T01 * flow_by_t["t01"]) + (_W_T02 * flow_by_t["t02"])
    residual = -weighted_draw + _GW_WEIGHTED_INFLOW + up - dn
    assert residual == pytest.approx(0.0, abs=1e-6)

    # The "import" (weighted inflow supplied + slack) equals the rp-WEIGHTED
    # draw, and is strictly LARGER than the unweighted draw — the LP cannot
    # game the balance by drawing in the high-weight timeslice.
    imported = _GW_WEIGHTED_INFLOW + up
    assert imported == pytest.approx(_GW_WEIGHTED_DRAW, rel=1e-6)          # 33
    unweighted_draw = (-_BW_DEMAND_T01) + (-_BW_DEMAND_T02)               # 13
    assert imported > unweighted_draw + 1.0
