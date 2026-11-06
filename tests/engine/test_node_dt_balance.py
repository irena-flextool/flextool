"""``node_summary`` row-balance under timeline coarsening (step_duration > 1).

The ``node__dt`` result table (output key ``node_dt_ep``) is a MW-rate table:
flows and slacks come from ``r.flow_dt`` (MW), and the Inflow row is produced by
dividing ``par.node_inflow`` (energy per step, MWh/step) by ``par.step_duration``
in the inflow block of ``node_summary``.  Before that division the Inflow row was
MWh/step while every other row was MW, so each ``(d, t)`` row carried a residual
of ``inflow · (1 − 1/step_duration)`` and the table did not balance whenever
coarsening was active.

These tests encode the hand-calculation on a tiny synthesised case (no solver):
one balance node fed by a generator and drained by a load, with nonzero inflow
and ``step_duration = 2`` on two timesteps, chosen so the MW-rate balance closes
exactly (``gen − load + inflow/step_duration == 0``).  They assert:

(a) every ``node__dt`` row sums to ~0 — the balance closes;
(b) the ``node__dt`` inflow cell equals ``par.node_inflow / step_duration``;
(c) the ``node__d`` (``node_d_ep``) inflow equals the PRE-fix annualisation
    ``Σ_t inflow_MWh_step · weight / share`` — i.e. node__d is invariant to the
    fix (computed independently here, not read from a golden).
"""
from types import SimpleNamespace

import pandas as pd
import pytest

from flextool.process_outputs.out_node import node_summary


def _build_inputs(step_duration, *, gen=3.0, load=5.0, inflow=4.0, weight=1.0):
    """Synthesise the ``par``/``s``/``v``/``r`` namespaces ``node_summary`` reads.

    One balance node ``elec``, no storage (so the storage-hiding path short-
    circuits), fed by unit ``gen`` and drained by unit ``load`` over a single
    period with two timesteps.  ``inflow`` is energy per step (MWh/step).
    Flows/inflow are constant across the two timesteps.
    """
    dt = pd.MultiIndex.from_tuples(
        [("p1", "t1"), ("p1", "t2")], names=["period", "time"],
    )
    node_idx = pd.Index(["elec"], name="node")

    # r.flow_dt: columns (process, source, sink); gen feeds elec (sink side =>
    # "From units"), load draws from elec (source side => "To units").
    flow_cols = pd.MultiIndex.from_tuples(
        [("gen", "coal", "elec"), ("load", "elec", "dmd")],
        names=["process", "source", "sink"],
    )
    flow_dt = pd.DataFrame(
        [[gen, load], [gen, load]], index=dt, columns=flow_cols, dtype=float,
    )

    inflow_df = pd.DataFrame({"elec": [inflow, inflow]}, index=dt, dtype=float)
    inflow_df.columns.name = "node"

    qzero = pd.DataFrame({"elec": [0.0, 0.0]}, index=dt, dtype=float)
    qzero.columns.name = "node"

    sd = pd.Series([step_duration, step_duration], index=dt, dtype=float)
    w = pd.Series([weight, weight], index=dt, dtype=float)
    share = pd.Series(
        [sd.sum() / 8760.0], index=pd.Index(["p1"], name="period"),
    )

    par = SimpleNamespace(
        node_inflow=inflow_df,
        node_capacity_for_scaling=pd.Series([1.0], index=node_idx),
        step_duration=sd,
        timestep_weight=w,
        complete_period_share_of_year=share,
    )
    s = SimpleNamespace(
        node=node_idx,
        node_balance=node_idx,
        node_balance_period=pd.Index([], name="node"),
        node_state=pd.Index([], name="node"),
        node_self_discharge=pd.Index([], name="node"),
        process_unit=pd.Index(["gen", "load"], name="process"),
        process_connection=pd.Index([], name="process"),
        process_source=pd.MultiIndex.from_tuples([], names=["process", "source"]),
        process_sink=pd.MultiIndex.from_tuples([], names=["process", "sink"]),
        dt_realize_dispatch=dt,
    )
    v = SimpleNamespace(q_state_up=qzero.copy(), q_state_down=qzero.copy())
    r = SimpleNamespace(
        flow_dt=flow_dt,
        self_discharge_loss_dt=pd.DataFrame(
            index=dt, columns=pd.Index([], name="node"),
        ),
    )
    return par, s, v, r


def _run(par, s, v, r):
    return {tag: df for df, tag in node_summary(par, s, v, r, debug=False)}


@pytest.mark.parametrize("step_duration", [2.0, 3.0])
def test_node_dt_rows_balance_under_coarsening(step_duration):
    # Pick flows so the MW-rate balance closes: gen - load + inflow/sd == 0.
    inflow = 4.0
    gen = 3.0
    load = gen + inflow / step_duration
    par, s, v, r = _build_inputs(step_duration, gen=gen, load=load, inflow=inflow)

    res = _run(par, s, v, r)
    node_dt = res["node_dt_ep"]

    # (a) every (d, t) row sums to ~0 — the balance closes.
    row_sums = node_dt.sum(axis=1)
    assert (row_sums.abs() < 1e-9).all(), row_sums

    # (b) the inflow cell is a MW rate == par.node_inflow / step_duration.
    expected_rate = inflow / step_duration
    inflow_cell = node_dt[("elec", "Inflow")]
    assert (inflow_cell - expected_rate).abs().max() < 1e-12

    # Sanity: the flow-derived rows are unchanged MW values.
    assert (node_dt[("elec", "From units")] - gen).abs().max() < 1e-12
    assert (node_dt[("elec", "To units")] + load).abs().max() < 1e-12


@pytest.mark.parametrize("step_duration", [2.0, 3.0])
def test_node_d_inflow_invariant_to_fix(step_duration):
    inflow = 4.0
    gen = 3.0
    load = gen + inflow / step_duration
    par, s, v, r = _build_inputs(step_duration, gen=gen, load=load, inflow=inflow)

    res = _run(par, s, v, r)
    node_d = res["node_d_ep"]

    # (c) node__d inflow == PRE-fix formula: Σ_t inflow_MWh_step·weight / share,
    # computed independently (NOT read from a golden).  inflow_MWh_step is the
    # raw par.node_inflow (energy per step); weight == 1; share == Σ_t sd / 8760.
    weighted = (par.node_inflow["elec"] * par.timestep_weight).sum()
    share = par.complete_period_share_of_year.loc["p1"]
    expected_inflow_d = weighted / share

    got = node_d[("elec", "Inflow")].loc["p1"]
    assert abs(got - expected_inflow_d) < 1e-6


def test_step_duration_one_is_identity():
    """At step_duration == 1 the Inflow row equals the raw energy-per-step
    value (the fix is an identity), so node__dt is byte-unchanged vs the old
    MWh/step convention and still balances."""
    inflow = 4.0
    gen = 3.0
    load = gen + inflow  # sd == 1 => rate == energy
    par, s, v, r = _build_inputs(1.0, gen=gen, load=load, inflow=inflow)

    res = _run(par, s, v, r)
    node_dt = res["node_dt_ep"]

    assert (node_dt[("elec", "Inflow")] - inflow).abs().max() < 1e-12
    assert (node_dt.sum(axis=1).abs() < 1e-9).all()
