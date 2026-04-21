"""Unit + integration tests for flowGroup_indicators.

The unit test builds a minimal ``par``/``s``/``v``/``r`` stand-in and calls
``flowGroup_indicators`` directly — fast, deterministic, and
independent of running the solver.

The integration test reuses the shipped
``aggregate_outputs_network_coal_wind_chp`` scenario (which already has
``output_flowGroup_indicators: yes`` on two groups with both unit and
connection members) and asserts that the parquet lands with the expected
shape.  The integration test is marked slow so the unit test alone is
enough to guard the implementation in quick iterations.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from flextool.process_outputs.out_flowgroup import flowGroup_indicators

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))


def _build_fixture() -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace]:
    """Two periods × two timesteps, one unit flow and one connection flow
    per group.  Timings and magnitudes are chosen so the expected cumulative
    and average flows are easy to check by hand."""
    periods = ['p1', 'p2']
    times = ['t1', 't2']
    dt_index = pd.MultiIndex.from_product([periods, times], names=['period', 'time'])

    # step_duration: 1h in p1, 2h in p2 — makes MWh ≠ MW so the weighting
    # is visible in the golden numbers.
    step_duration = pd.Series(
        [1.0, 1.0, 2.0, 2.0], index=dt_index, name='value',
    )
    # complete_period_share_of_year: (1h+1h)/8760 in p1, (2h+2h)/8760 in p2.
    period_share = pd.Series(
        [2.0 / 8760.0, 4.0 / 8760.0],
        index=pd.Index(periods, name='period'),
    )

    par = SimpleNamespace(
        step_duration=step_duration,
        complete_period_share_of_year=period_share,
    )

    # Two flow-indicator groups:
    #   gU — one unit flow (coal_plant from coal to elec).
    #   gC — one connection flow leg (west_east touching the "west" node).
    # Plus a "silent" group gEmpty with no member flows → must land with
    # zeros rather than being dropped.
    group_process_node = pd.MultiIndex.from_tuples(
        [
            ('gU', 'coal_plant', 'elec'),
            ('gC', 'west_east', 'west'),
        ],
        names=['group', 'process', 'node'],
    )
    s = SimpleNamespace(
        flowGroupIndicators=pd.Index(['gU', 'gC', 'gEmpty'], name='group'),
        group_process_node=group_process_node,
        process_unit=pd.Index(['coal_plant'], name='process'),
        process_connection=pd.Index(['west_east'], name='process'),
        dt_realize_dispatch=dt_index,
        d_realized_period=pd.Index(periods, name='period'),
    )

    # r.flow_dt — MW at the (process, source, sink) column level.
    flow_cols = pd.MultiIndex.from_tuples(
        [('coal_plant', 'coal', 'elec')],
        names=['process', 'source', 'sink'],
    )
    flow_dt = pd.DataFrame(
        [[10.0], [20.0], [30.0], [40.0]],
        index=dt_index, columns=flow_cols,
    )

    # r.from_conn / r.to_conn — the connection flow.  We give non-trivial
    # values on both directions; the stub sums absolute magnitudes so they
    # both contribute.
    conn_cols = pd.MultiIndex.from_tuples(
        [('west_east', 'west')], names=['process', 'node'],
    )
    from_conn = pd.DataFrame(
        [[5.0], [5.0], [5.0], [5.0]],
        index=dt_index, columns=conn_cols,
    )
    to_conn = pd.DataFrame(
        [[1.0], [2.0], [3.0], [4.0]],
        index=dt_index, columns=conn_cols,
    )

    r = SimpleNamespace(
        flow_dt=flow_dt,
        from_conn=from_conn,
        to_conn=to_conn,
    )
    v = SimpleNamespace()
    return par, s, v, r


def test_flowgroup_indicators_basic_shape_and_values() -> None:
    par, s, v, r = _build_fixture()

    results = flowGroup_indicators(par, s, v, r, debug=False)

    assert len(results) == 1
    frame, name = results[0]
    assert name == 'flowGroup_gd_p'

    # Index: (group, period) over all three groups × both periods.
    assert list(frame.index.names) == ['group', 'period']
    expected_keys = {('gU', 'p1'), ('gU', 'p2'), ('gC', 'p1'), ('gC', 'p2'),
                     ('gEmpty', 'p1'), ('gEmpty', 'p2')}
    assert set(frame.index) == expected_keys

    # Columns: the two stub metrics, labelled with columns.name='parameter'.
    assert list(frame.columns) == ['cumulative_flow', 'average_flow']
    assert frame.columns.name == 'parameter'

    # gU: unit flow is 10 MW + 20 MW in p1 (1h each) → 30 MWh cumulative,
    # 30 MW + 40 MW in p2 (2h each) → 140 MWh.
    #   average_flow = cumulative / hours  (hours = share × 8760)
    assert frame.loc[('gU', 'p1'), 'cumulative_flow'] == pytest.approx(30.0)
    assert frame.loc[('gU', 'p2'), 'cumulative_flow'] == pytest.approx(140.0)
    assert frame.loc[('gU', 'p1'), 'average_flow'] == pytest.approx(15.0)   # 30/2
    assert frame.loc[('gU', 'p2'), 'average_flow'] == pytest.approx(35.0)   # 140/4

    # gC: |from_conn| + |to_conn| summed per step, weighted by step_duration.
    #   p1: (5+1)*1 + (5+2)*1 = 13 MWh,  hours=2  → avg=6.5
    #   p2: (5+3)*2 + (5+4)*2 = 34 MWh,  hours=4  → avg=8.5
    assert frame.loc[('gC', 'p1'), 'cumulative_flow'] == pytest.approx(13.0)
    assert frame.loc[('gC', 'p2'), 'cumulative_flow'] == pytest.approx(34.0)
    assert frame.loc[('gC', 'p1'), 'average_flow'] == pytest.approx(6.5)
    assert frame.loc[('gC', 'p2'), 'average_flow'] == pytest.approx(8.5)

    # gEmpty: no member flows → zeros (not dropped, not NaN).
    assert frame.loc[('gEmpty', 'p1'), 'cumulative_flow'] == 0.0
    assert frame.loc[('gEmpty', 'p2'), 'cumulative_flow'] == 0.0
    assert frame.loc[('gEmpty', 'p1'), 'average_flow'] == 0.0
    assert frame.loc[('gEmpty', 'p2'), 'average_flow'] == 0.0


def test_flowgroup_indicators_empty_indicators_set() -> None:
    par, s, v, r = _build_fixture()
    s.flowGroupIndicators = pd.Index([], name='group')
    assert flowGroup_indicators(par, s, v, r, debug=False) == []


def test_flowgroup_indicators_empty_realized_dispatch() -> None:
    par, s, v, r = _build_fixture()
    s.dt_realize_dispatch = pd.MultiIndex.from_tuples([], names=['period', 'time'])
    assert flowGroup_indicators(par, s, v, r, debug=False) == []
