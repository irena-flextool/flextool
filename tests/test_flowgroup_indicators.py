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

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import polars as pl
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

    # Now returns two frames: the per-period summary plus the signed
    # per-(period,time) net-flow series.
    assert len(results) == 2
    frame, name = results[0]
    assert name == 'flowGroup_gd_p'
    assert results[1][1] == 'flowGroup_gd_t'

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


def test_flowgroup_period_frame_byte_identical_regression() -> None:
    """The unsigned per-period frame must be unchanged by the addition of the
    signed series — a byte-identical regression guard.  The expected values
    below replicate exactly what the period-only implementation produced."""
    par, s, v, r = _build_fixture()

    results = flowGroup_indicators(par, s, v, r, debug=False)
    frame = results[0][0]

    expected = pd.DataFrame(
        [
            {'group': 'gU', 'period': 'p1', 'cumulative_flow': 30.0,
             'average_flow': 15.0},
            {'group': 'gU', 'period': 'p2', 'cumulative_flow': 140.0,
             'average_flow': 35.0},
            {'group': 'gC', 'period': 'p1', 'cumulative_flow': 13.0,
             'average_flow': 6.5},
            {'group': 'gC', 'period': 'p2', 'cumulative_flow': 34.0,
             'average_flow': 8.5},
            {'group': 'gEmpty', 'period': 'p1', 'cumulative_flow': 0.0,
             'average_flow': 0.0},
            {'group': 'gEmpty', 'period': 'p2', 'cumulative_flow': 0.0,
             'average_flow': 0.0},
        ]
    ).set_index(['group', 'period'])[['cumulative_flow', 'average_flow']]
    expected.columns.name = 'parameter'

    pd.testing.assert_frame_equal(frame, expected, check_exact=True)


def test_flowgroup_gd_t_signed_net_flow() -> None:
    """The signed per-(period,time) series uses into-node +, out-of-node −.

    Fixture sign breakdown:
      * gU — unit flow ``(coal_plant, coal, elec)`` with node ``elec`` as the
        *sink* → all into-node (+): 10, 20, 30, 40.
      * gC — connection ``west_east`` touching node ``west``:
        ``from_conn`` = 5,5,5,5 (into node, +); ``to_conn`` = 1,2,3,4 (out of
        node, −).  Signed net = 5−1, 5−2, 5−3, 5−4 = 4, 3, 2, 1.
        (The UNSIGNED magnitude summed |5|+|1| etc → 6, 7, 8, 9.)
      * gEmpty — no member flows → zeros.
    """
    par, s, v, r = _build_fixture()

    results = flowGroup_indicators(par, s, v, r, debug=False)

    by_name = {name: frame for frame, name in results}
    assert 'flowGroup_gd_t' in by_name
    net = by_name['flowGroup_gd_t']

    assert list(net.index.names) == ['group', 'period', 'time']
    assert list(net.columns) == ['net_flow']
    assert net.columns.name == 'parameter'

    # All three groups present at every realized (period, time).
    expected_keys = {
        (g, p, t)
        for g in ('gU', 'gC', 'gEmpty')
        for p in ('p1', 'p2')
        for t in ('t1', 't2')
    }
    assert set(net.index) == expected_keys

    # gU — unit sink, all positive.
    assert net.loc[('gU', 'p1', 't1'), 'net_flow'] == pytest.approx(10.0)
    assert net.loc[('gU', 'p1', 't2'), 'net_flow'] == pytest.approx(20.0)
    assert net.loc[('gU', 'p2', 't1'), 'net_flow'] == pytest.approx(30.0)
    assert net.loc[('gU', 'p2', 't2'), 'net_flow'] == pytest.approx(40.0)

    # gC — signed net = from_conn − to_conn.
    assert net.loc[('gC', 'p1', 't1'), 'net_flow'] == pytest.approx(4.0)
    assert net.loc[('gC', 'p1', 't2'), 'net_flow'] == pytest.approx(3.0)
    assert net.loc[('gC', 'p2', 't1'), 'net_flow'] == pytest.approx(2.0)
    assert net.loc[('gC', 'p2', 't2'), 'net_flow'] == pytest.approx(1.0)

    # gEmpty — zeros.
    for t in ('t1', 't2'):
        for p in ('p1', 'p2'):
            assert net.loc[('gEmpty', p, t), 'net_flow'] == 0.0

    # signed ≠ unsigned guard: for gC the signed magnitudes (4,3,2,1) sum
    # strictly below the unsigned magnitudes (6,7,8,9).
    signed_gc = net.xs('gC', level='group')['net_flow'].abs().sum()
    unsigned_gc = 6.0 + 7.0 + 8.0 + 9.0
    assert signed_gc < unsigned_gc


class _FakeProvider:
    """Minimal stand-in for the cascade FlexDataProvider — only the ``has``
    / ``get`` keyed-frame interface used by ``_provider_lookup_df``."""

    def __init__(self, frames: dict[str, pl.DataFrame]):
        self._frames = frames

    def has(self, name: str) -> bool:
        return name in self._frames

    def get(self, name: str) -> pl.DataFrame:
        return self._frames[name]


def test_backfill_group_process_node_from_provider() -> None:
    """red→green guard for the latent backfill bug: ``read_sets`` leaves
    ``s.group_process_node`` empty; the backfill must populate it from the
    Provider key ``input/flowGroup__process__node``."""
    # write_outputs is re-exported as a *function* on the package, so a plain
    # ``import flextool.process_outputs.write_outputs`` resolves to the
    # function, not the module.  Reach the module via importlib.
    wo = importlib.import_module('flextool.process_outputs.write_outputs')

    # read_sets-empty starting state.
    s = SimpleNamespace(
        nodeGroupDispatch=pd.Index([], name='group'),
        nodeGroupIndicators=pd.Index([], name='group'),
        flowGroupIndicators=pd.Index([], name='group'),
        group_process_node=pd.MultiIndex.from_tuples(
            [], names=['group', 'process', 'node']),
    )
    assert len(s.group_process_node) == 0  # red before

    gpn_frame = pl.DataFrame(
        {
            'flowGroup': ['gA', 'gA', 'gB'],
            'process': ['coal_plant', 'west_east', 'wind_plant'],
            'node': ['elec', 'west', 'elec'],
        }
    )
    provider = _FakeProvider({'input/flowGroup__process__node': gpn_frame})

    wo._backfill_group_indicator_sets(s, output_dir=None, provider=provider)

    # green after: the membership index is populated from the Provider.
    assert isinstance(s.group_process_node, pd.MultiIndex)
    assert list(s.group_process_node.names) == ['group', 'process', 'node']
    assert set(s.group_process_node) == {
        ('gA', 'coal_plant', 'elec'),
        ('gA', 'west_east', 'west'),
        ('gB', 'wind_plant', 'elec'),
    }


def test_backfill_group_process_node_absent_stays_empty() -> None:
    """A group-less model (Provider carries no flowGroup membership key) must
    leave ``s.group_process_node`` empty — tolerant, no error."""
    wo = importlib.import_module('flextool.process_outputs.write_outputs')

    s = SimpleNamespace(
        nodeGroupDispatch=pd.Index([], name='group'),
        nodeGroupIndicators=pd.Index([], name='group'),
        flowGroupIndicators=pd.Index([], name='group'),
        group_process_node=pd.MultiIndex.from_tuples(
            [], names=['group', 'process', 'node']),
    )
    provider = _FakeProvider({})  # empty Provider — no keys

    wo._backfill_group_indicator_sets(s, output_dir=None, provider=provider)

    assert len(s.group_process_node) == 0
