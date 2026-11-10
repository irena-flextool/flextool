"""Node-group VRE / curtailment / slack shares (item 198).

One group ``g`` with demand node ``n`` and wind unit ``w``, step duration 1 h.
Inflow 100, 50; wind potential 90, 70; wind flow 80, 30 (curtailed 10, 40).
The expected values are worked out by hand from these numbers:

- step: curtailed / potential = 10/90, 40/70;
- period, weight 1: curtailed / potential = 50/160 = 0.3125,
  curtailed / inflow = 50/150, VRE / inflow = 110/150;
- period, weights (3, 1): curtailed / potential = (30+40)/(270+70) = 70/340,
  curtailed / inflow = 70/350, VRE / inflow = (240+30)/350 = 270/350.
"""
import types

import pandas as pd
import pytest

from flextool.process_outputs.out_group import nodeGroup_indicators, nodeGroup_VRE_share


def _case(steps, inflow, potential, flow, weight, *, extra_node=None):
    """In-memory par / s / v / r for one group (no DB).

    ``extra_node`` = (node, flows) adds a member node without inflow that
    receives wind from a second unit ``w2``.
    """
    dt = pd.MultiIndex.from_tuples([("d1", t) for t in steps], names=["period", "time"])
    group_node = [("g", "n")]
    inflow_method = [("n", "use_original")]
    vre = [("w", "n")]
    flow_cols = [("w", "w", "n")]
    flow_data = {("w", "w", "n"): flow}
    if extra_node is not None:
        node, extra_flow = extra_node
        group_node.append(("g", node))
        inflow_method.append((node, "no_inflow"))
        vre.append(("w2", node))
        flow_cols.append(("w2", "w2", node))
        flow_data[("w2", "w2", node)] = extra_flow
    fcols = pd.MultiIndex.from_tuples(flow_cols, names=["process", "source", "sink"])
    s = types.SimpleNamespace(
        group_node=pd.MultiIndex.from_tuples(group_node, names=["group", "node"]),
        node__inflow_method=inflow_method,
        nodeGroupIndicators=["g"],
        dt_realize_dispatch=dt,
        process_VRE=pd.MultiIndex.from_tuples(vre, names=["process", "node"]),
        process_source_sink_alwaysProcess=fcols,
        process_sink=[(p, n) for p, n in vre],
        node_balance=["n"],
        node_balance_period=[],
    )
    par = types.SimpleNamespace(
        node_inflow=pd.DataFrame({"n": [-x for x in inflow]}, index=dt, dtype=float),
        timestep_weight=pd.Series(weight, index=dt, dtype=float),
        complete_period_share_of_year=pd.Series([1.0], index=pd.Index(["d1"], name="period")),
        step_duration=pd.Series(1.0, index=dt),
        node_capacity_for_scaling=pd.Series({"n": 1.0}),
    )
    r = types.SimpleNamespace(
        flow_dt=pd.DataFrame({c: flow_data[c] for c in flow_cols}, index=dt, dtype=float).set_axis(fcols, axis=1),
        potentialVREgen_dt=pd.DataFrame(
            [[x] for x in potential], index=dt, dtype=float,
            columns=pd.MultiIndex.from_tuples([("w", "n")], names=["process", "node"]),
        ),
    )
    v = types.SimpleNamespace(q_state_up=pd.DataFrame(), q_state_down=pd.DataFrame())
    return par, s, v, r


def _run(par, s, v, r):
    ind = {k: df for df, k in nodeGroup_indicators(par, s, v, r, False)}
    vre = {k: df for df, k in nodeGroup_VRE_share(par, s, v, r, False)}
    return ind["nodeGroup_gdt_p"], ind["nodeGroup_gd_p"].loc[("g", "d1")], vre


def test_unit_weights():
    dt, d, vre = _run(*_case(["t1", "t2"], [100, 50], [90, 70], [80, 30], [1, 1]))
    assert dt["6. Curtailed VRE of potential VRE"].tolist() == pytest.approx([10 / 90, 40 / 70])
    assert d["6. Curtailed VRE of potential VRE"] == pytest.approx(0.3125)
    assert d["4. Curtailed VRE of demand"] == pytest.approx(50 / 150)
    assert d["2. VRE share of demand"] == pytest.approx(110 / 150)
    assert vre["nodeGroup_VRE_share_d_g"].loc["d1", "g"] == pytest.approx(110 / 150)


def test_timestep_weights():
    _, d, vre = _run(*_case(["t1", "t2"], [100, 50], [90, 70], [80, 30], [3, 1]))
    assert d["6. Curtailed VRE of potential VRE"] == pytest.approx(70 / 340)
    assert d["4. Curtailed VRE of demand"] == pytest.approx(70 / 350)
    assert d["2. VRE share of demand"] == pytest.approx(270 / 350)
    assert d["5. Annualized inflow"] == pytest.approx(350)
    assert vre["nodeGroup_VRE_share_d_g"].loc["d1", "g"] == pytest.approx(270 / 350)


def test_step_with_zero_inflow_keeps_its_vre():
    # t3: inflow 0, potential 20, flow 20 → VRE 130 of inflow 150
    dt, d, vre = _run(*_case(["t1", "t2", "t3"], [100, 50, 0], [90, 70, 20], [80, 30, 20], [1, 1, 1]))
    assert d["2. VRE share of demand"] == pytest.approx(130 / 150)
    assert d["6. Curtailed VRE of potential VRE"] == pytest.approx(50 / 180)
    assert vre["nodeGroup_VRE_share_d_g"].loc["d1", "g"] == pytest.approx(130 / 150)
    # The step share is 0 at a step without inflow, never ±inf
    assert vre["nodeGroup_VRE_share_dt_g"].loc[("d1", "t3"), "g"] == 0.0
    assert dt.loc[("g", "d1", "t3"), "8. VRE share of demand"] == 0.0


def test_vre_into_member_node_without_inflow_counts_in_both_tables():
    # w2 feeds the no-inflow member node h: 5, 10 → VRE (80+5)+(30+10) = 125 of 150
    _, d, vre = _run(*_case(["t1", "t2"], [100, 50], [90, 70], [80, 30], [1, 1], extra_node=("h", [5, 10])))
    assert d["2. VRE share of demand"] == pytest.approx(125 / 150)
    assert vre["nodeGroup_VRE_share_d_g"].loc["d1", "g"] == pytest.approx(125 / 150)


def test_realized_step_without_weight_raises():
    par, s, v, r = _case(["t1", "t2"], [100, 50], [90, 70], [80, 30], [1, 1])
    par.timestep_weight = par.timestep_weight.iloc[:1]
    with pytest.raises(ValueError, match=r"\(d1, t2\)"):
        nodeGroup_indicators(par, s, v, r, False)
    with pytest.raises(ValueError, match=r"\(d1, t2\)"):
        nodeGroup_VRE_share(par, s, v, r, False)
