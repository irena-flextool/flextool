"""Item 219: the slack lines of summary_solve.csv are weighted like the objective.

The Created / Removed / Inertia / NonSync / Reserve lines print ``*_d_sample`` = Σ_t w·Δt·slack over the period's
steps (``w`` the timestep weight), the sample the objective charges the penalty on.  Before they were Σ_t Δt·slack,
unweighted.  The capacity-margin slack is per period, has no weight in the objective and is unchanged.

Hand values (one period, two steps of Δt = 1 h, w = (0.5, 1.5), every slack 10 then 20, every scale 1):
weighted 0.5·10 + 1.5·20 = 35 (unweighted 30); capacity margin 7 → 7.

Item 84: the reserve penalty takes each group's own shortfall scale when the scale frame (column level ``upDown``)
lists the groups in another order than vq_reserve (level ``updown``).  Before, the two frames were aligned on the
level names and every group was multiplied by both upDown scales: the penalty came out doubled.  Hand value: groups
(rs, down, g) and (rs, up, g), slack 1 at both steps, scale 50 and 40, penalty 1000, Δt = 1, w = (0.5, 1.5):
down 1·1·0.5·1000·50 = 25000 and 1·1·1.5·1000·50 = 75000; up 20000 and 60000.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from flextool.process_outputs.calc_slacks import compute_slacks

STEPS = pd.MultiIndex.from_tuples([("p1", "t1"), ("p1", "t2")], names=["period", "time"])


def _frame(columns, values=(10.0, 20.0)) -> pd.DataFrame:
    return pd.DataFrame({c: list(values) for c in columns}, index=STEPS)


def _run(reserve_cols=None, q_reserve=None, scale=None, penalty=None):
    if reserve_cols is None:
        reserve_cols = pd.MultiIndex.from_tuples([("r", "up", "g")], names=["reserve", "updown", "group"])
    v = SimpleNamespace(
        reserve=pd.DataFrame(index=STEPS, columns=pd.MultiIndex.from_tuples(
            [], names=["process", "reserve", "updown", "node"]), dtype=float),
        q_state_up=_frame(["n"]),
        q_state_down=_frame(["n"]),
        q_inertia=_frame(["g"]),
        q_non_synchronous=_frame(["g"]),
        q_capacity_margin=pd.DataFrame({"g": [7.0]}, index=pd.Index(["p1"], name="period")),
        q_reserve=(q_reserve if q_reserve is not None
                   else pd.DataFrame({reserve_cols[0]: [10.0, 20.0]}, index=STEPS).set_axis(reserve_cols, axis=1)),
    )
    one = pd.Series(1.0, index=["n"])
    g1 = pd.Series(1.0, index=["g"])
    par = SimpleNamespace(
        step_duration=pd.Series(1.0, index=STEPS),
        timestep_weight=pd.Series([0.5, 1.5], index=STEPS),
        complete_period_share_of_year=pd.Series(2.0 / 8760.0, index=pd.Index(["p1"], name="period")),
        entity_unitsize=pd.Series(dtype=float, index=pd.Index([], name="process")),
        node_capacity_for_scaling=one, node_penalty_up=one, node_penalty_down=one,
        group_inertia_limit=_frame(["g"], (1.0, 1.0)),
        group_penalty_inertia=g1, group_capacity_for_scaling=g1, group_penalty_non_synchronous=g1,
        group_penalty_capacity_margin=g1,
        inflation_factor_operations_yearly=pd.Series(1.0, index=pd.Index(["p1"], name="period")),
        reserve_upDown_group_shortfall_scale=(scale if scale is not None
                                              else pd.DataFrame(1.0, index=STEPS, columns=reserve_cols)),
        reserve_upDown_group_penalty=penalty if penalty is not None else pd.Series(1.0, index=reserve_cols),
    )
    s = SimpleNamespace(groupNonSync=["g"], groupCapacityMargin=["g"])
    r = SimpleNamespace()
    compute_slacks(par, s, v, r)
    return r


def test_slack_lines_are_timestep_weighted() -> None:
    r = _run()
    for name, col in (("upward_node_slack_d_sample", "n"), ("downward_node_slack_d_sample", "n"),
                      ("q_inertia_d_sample", "g"), ("q_non_synchronous_d_sample", "g")):
        assert getattr(r, name).loc["p1", col] == pytest.approx(35.0, rel=1e-12), name
    assert r.q_reserves_d_sample.loc["p1"].iloc[0] == pytest.approx(35.0, rel=1e-12)


def test_capacity_margin_line_is_unchanged() -> None:
    assert _run().q_capacity_margin_d_not_annualized.loc["p1", "g"] == pytest.approx(7.0, rel=1e-12)


def test_reserve_penalty_takes_each_groups_own_scale() -> None:
    vq_cols = pd.MultiIndex.from_tuples([("rs", "down", "g"), ("rs", "up", "g")],
                                        names=["reserve", "updown", "node_group"])
    scale_cols = pd.MultiIndex.from_tuples([("rs", "up", "g"), ("rs", "down", "g")],
                                           names=["reserve", "upDown", "node_group"])
    q = pd.DataFrame(1.0, index=STEPS, columns=vq_cols)
    scale = pd.DataFrame([[40.0, 50.0], [40.0, 50.0]], index=STEPS, columns=scale_cols)
    penalty = pd.Series(1000.0, index=scale_cols)
    cost = _run(vq_cols, q, scale, penalty).costPenalty_reserve_upDown_dt
    assert list(cost.columns) == list(vq_cols)
    assert cost[("rs", "down", "g")].tolist() == pytest.approx([25000.0, 75000.0], rel=1e-12)
    assert cost[("rs", "up", "g")].tolist() == pytest.approx([20000.0, 60000.0], rel=1e-12)
