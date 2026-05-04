"""Synthetic minimal-feature toy: capacity_margin in isolation.

A slack-only producerless scenario with one group, one node, and a
non-trivial ``pdGroup_capacity_margin`` floor.  The capacity-margin
constraint cannot be met by any producer (there are none), so the
``vq_capacity_margin`` slack absorbs the entire shortfall and is priced
at ``penalty_capacity_margin × group_capacity_for_scaling × inflation_op``
(period-only, no step_duration / rp_cost_weight / period_share — see
``audit/objective_audit.md`` §9.3 and ``flextool/_group_slack.py``
``add_objective_terms``).

Topology
--------
* 1 group ``g``; 1 node ``n`` in ``nodeBalance``; 0 processes.
* ``group_node = {(g, n)}``; ``p_group_capacity_for_scaling[g, p2020] = 1.0``.
* ``pdGroup_capacity_margin[g, p2020] = 100`` MW;
  ``pdGroup_penalty_capacity_margin[g, p2020] = 1e6`` €/MW.
* 2 timesteps with positive inflow ``+10`` MW each — feeds the
  capacity_margin RHS via ``pdtNodeInflow_per_step``.

Closed-form expected obj
------------------------
The capacity_margin constraint (per (g, d, t)) is:
    vq_capacity_margin[g,d] * 1.0  >=  100 * 1.0  -  10 * 1.0
                                    =  90  MW
(the inv_group_cap = 1.0 cancellation; producer LHS empty.)

Both timesteps yield the same RHS, so vq_capacity_margin = 90.

Slack-penalty obj (capacity_margin term, period-only):
    obj_cap = 90 * group_capacity_for_scaling * penalty_cap * inflation_op
            = 90 * 1.0 * 1e6 * 1.0
            = 9.0e7

Plus the ordinary node-balance slack (positive inflow has no consumer
so vq_state_down absorbs it):
    obj_slack = Σ_{d,t}  vq_state_down · pen_down · op_factor
              = 2 * 10 * 1.0  *  step_duration * rp_cost_weight
                                * inflation_op / period_share
              = 20 (with step_dur=rp_cw=infl=psh=1.0)

Total expected obj = 9.0e7 + 20 = 90_000_020.0.

Why this toy matters
--------------------
The merge-step-5 capacity_margin parity test would have been *trivial*
to debug here: any wrong factor on the slack term, any wrong sign on
the inflow contribution, or a missing ``inv_group_cap`` cancellation
would fail this 6-line closed-form assertion.
"""
from __future__ import annotations

import polars as pl
from flexpy import Param
from flextool.engine_polars.input import FlexData


def data() -> FlexData:
    # Time axes — single period, two timesteps, step_dur=rp_cw=infl=psh=1.
    dt = pl.DataFrame({"d": ["p2020", "p2020"], "t": ["t01", "t02"]})
    p_step_duration = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"]*2, "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_rp_cost_weight = Param(("d", "t"),
        pl.DataFrame({"d": ["p2020"]*2, "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_inflation_op = Param(("d",),
        pl.DataFrame({"d": ["p2020"], "value": [1.0]}))
    p_period_share = Param(("d",),
        pl.DataFrame({"d": ["p2020"], "value": [1.0]}))

    nodeBalance = pl.DataFrame({"n": ["n"]})
    nodeBalance_dt = nodeBalance.join(dt, how="cross")

    # Inflow +10 each timestep (positive = source on RHS via -inflow).
    p_inflow = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["n"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [10.0, 10.0]}))
    p_penalty_up = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["n"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1.0, 1.0]}))
    p_penalty_down = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["n"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [1.0, 1.0]}))

    # ── Group-slack capacity_margin data ───────────────────────────────
    groupCapacityMargin = pl.DataFrame({"g": ["g"]})
    group_node = pl.DataFrame({"g": ["g"], "n": ["n"]})

    pdGroup_capacity_margin = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["p2020"], "value": [100.0]}))
    pdGroup_penalty_capacity_margin = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["p2020"], "value": [1.0e6]}))
    p_group_capacity_for_scaling = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["p2020"], "value": [1.0]}))
    p_inv_group_cap = Param(("g", "d"),
        pl.DataFrame({"g": ["g"], "d": ["p2020"], "value": [1.0]}))

    # pdtNodeInflow_per_step = inflow / step_duration.  Same as p_inflow
    # because step_duration=1.0 here.  Required on RHS of capacityMargin.
    pdtNodeInflow_per_step = Param(("n", "d", "t"),
        pl.DataFrame({"n": ["n"]*2, "d": ["p2020"]*2,
                       "t": ["t01", "t02"], "value": [10.0, 10.0]}))

    return FlexData(
        dt=dt,
        p_step_duration=p_step_duration,
        p_rp_cost_weight=p_rp_cost_weight,
        p_inflation_op=p_inflation_op,
        p_period_share=p_period_share,
        nodeBalance=nodeBalance,
        nodeBalance_dt=nodeBalance_dt,
        p_inflow=p_inflow,
        p_penalty_up=p_penalty_up,
        p_penalty_down=p_penalty_down,
        # group-slack capacity_margin
        groupCapacityMargin=groupCapacityMargin,
        group_node=group_node,
        pdGroup_capacity_margin=pdGroup_capacity_margin,
        pdGroup_penalty_capacity_margin=pdGroup_penalty_capacity_margin,
        p_group_capacity_for_scaling=p_group_capacity_for_scaling,
        p_inv_group_cap=p_inv_group_cap,
        pdtNodeInflow_per_step=pdtNodeInflow_per_step,
    )


def expected_obj() -> float:
    """Closed-form predicted objective.

    obj_cap = vq_cap (=90) × p_group_capacity_for_scaling (=1.0)
                × penalty_cap (=1e6) × inflation_op (=1.0)
            = 9.0e7
    obj_slack = sum_{d,t} vq_state_down · pen_down · op_factor
              = 2 · 10 · 1.0 · 1.0 = 20.0
    total = 9.0e7 + 20.0 = 90_000_020.0
    """
    return 9.0e7 + 20.0
