"""Unit tests for ``_synthesize_invest_dual`` (out_ancillary).

The synthesized ``dual_invest_effective_*`` output is the COMPLETE, SIGNED
marginal value (objective per MW) of one more MW of investment capacity per
(entity, period).  Sign convention (user-confirmed):

    POSITIVE = more investment in this entity would IMPROVE (lower) the
    objective.

Each binding regime is asserted with the documented sign:

  - (a) not built — v_invest column reduced cost ≥ 0 → negate → NEGATIVE.
  - (b) upper-cap binds — ``<=`` row dual < 0 → negate → POSITIVE.
  - (c) lower-floor binds — ``>=`` row dual > 0 → negate → NEGATIVE.  This
        case also exercises the min-side (Increment 3) duals.
  - (d) interior — 0.

The frames fed here mirror the post-``drop_levels`` runtime shapes: a
``period`` row index and an ``entity`` (or 'unit'/'connection'/'node' for the
column-dual frames) column axis.  Crucially, ``dual_maxInvest_total`` is fed
period-LESS (single-row ``RangeIndex``, the ``_V_SOLVE_ONLY`` shape) while
``dual_minInvest_total`` is fed per-(entity, period) (the ``_V_DROP`` shape),
to prove the broadcast asymmetry is handled correctly.
"""
from types import SimpleNamespace

import pandas as pd

from flextool.process_outputs.out_ancillary import _synthesize_invest_dual


def _empty_entity(periods):
    df = pd.DataFrame(index=pd.Index(periods, name='period'))
    df.columns.name = 'entity'
    return df


def _empty_group(periods):
    df = pd.DataFrame(index=pd.Index(periods, name='period'))
    df.columns.name = 'group'
    return df


def _empty_col(periods, axis_name):
    df = pd.DataFrame(index=pd.Index(periods, name='period'))
    df.columns.name = axis_name
    return df


def _base_v(periods):
    """A ``v`` namespace with every invest-dual frame present but empty,
    plus an empty group-entity map.  Tests selectively populate frames."""
    v = SimpleNamespace()
    v.dual_maxInvest_period = _empty_entity(periods)
    # solve-only collapsed shape (RangeIndex) — populated per test.
    v.dual_maxInvest_total = pd.DataFrame(index=pd.RangeIndex(0))
    v.dual_maxInvest_total.columns.name = 'entity'
    v.dual_maxCumulative = _empty_entity(periods)
    v.dual_maxInvestGroup_period = _empty_group(periods)
    v.dual_maxInvestGroup_total = _empty_group(periods)
    v.dual_maxInvestGroup_cumulative = _empty_group(periods)
    v.dual_minInvest_period = _empty_entity(periods)
    # per-(entity, period) shape (retains period axis).
    v.dual_minInvest_total = _empty_entity(periods)
    v.dual_minCumulative = _empty_entity(periods)
    v.dual_minInvestGroup_period = _empty_group(periods)
    v.dual_minInvestGroup_total = _empty_group(periods)
    v.dual_minInvestGroup_cumulative = _empty_group(periods)
    v.dual_invest_unit = _empty_col(periods, 'unit')
    v.dual_invest_connection = _empty_col(periods, 'connection')
    v.dual_invest_node = _empty_col(periods, 'node')
    v.group_entity_invest = pd.DataFrame(columns=['group', 'entity'])
    return v


def _par(unitsizes):
    s = pd.Series(unitsizes, dtype=float)
    s.index.name = 'entity'
    s.name = 'entity'
    return SimpleNamespace(entity_unitsize=s)


def test_upper_cap_binding_is_positive():
    """(b) An upper-cap ``<=`` row dual arrives NEGATIVE; negating it yields
    a POSITIVE effective value (the cap holds the entity back)."""
    periods = ['p1', 'p2']
    v = _base_v(periods)
    # Raw <= dual is negative (HiGHS convention for a binding cap in a min).
    v.dual_maxInvest_period = pd.DataFrame(
        {'wind': [-30.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_maxInvest_period.columns.name = 'entity'
    par = _par({'wind': 100.0})

    out = _synthesize_invest_dual(v, par)
    assert out.loc['p1', 'wind'] == 30.0  # negate(-30) = +30, POSITIVE
    assert out.loc['p2', 'wind'] == 0.0


def test_max_total_broadcast_vs_min_total_per_period():
    """Asymmetry: ``dual_maxInvest_total`` is period-less (broadcast across
    all periods) while ``dual_minInvest_total`` is already per-period (added
    directly)."""
    periods = ['p1', 'p2']
    v = _base_v(periods)
    # Need a period_ref so the period-less max-total can be broadcast.
    v.dual_maxInvest_period = pd.DataFrame(
        {'wind': [0.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_maxInvest_period.columns.name = 'entity'
    # Solve-only single-row max-total (negative raw): broadcast to BOTH periods.
    v.dual_maxInvest_total = pd.DataFrame({'wind': [-5.0]}, index=pd.RangeIndex(1))
    v.dual_maxInvest_total.columns.name = 'entity'
    # Per-(entity, period) min-total (positive raw): added as-is, no broadcast.
    v.dual_minInvest_total = pd.DataFrame(
        {'wind': [7.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_minInvest_total.columns.name = 'entity'
    par = _par({'wind': 100.0})

    out = _synthesize_invest_dual(v, par)
    # max-total broadcast: +5 in BOTH periods.  min-total per-period: -7 in p1.
    assert out.loc['p1', 'wind'] == 5.0 - 7.0  # = -2.0
    assert out.loc['p2', 'wind'] == 5.0


def test_max_total_broadcasts_via_regime_a_period_ref():
    """Regression: ``dual_maxInvest_total`` is the SOLE binding family.

    Every per-period constraint-dual family is empty (each is independently
    emission-gated), but the regime-(a) ``dual_invest_unit`` frame is always
    present with a realized-``period`` index for any investable entity.  It
    must serve as the ``period_ref`` so the period-less max-total broadcasts
    to a POSITIVE value across BOTH periods instead of being silently dropped
    to 0 (the bug)."""
    periods = ['p1', 'p2']
    v = _base_v(periods)
    # Solve-only single-row max-total (negative raw): the only binding family.
    v.dual_maxInvest_total = pd.DataFrame({'wind': [-9.0]}, index=pd.RangeIndex(1))
    v.dual_maxInvest_total.columns.name = 'entity'
    # Regime-(a) column reduced cost present but ZERO-valued — the only frame
    # carrying the period axis to broadcast across.  Column axis is 'unit'.
    v.dual_invest_unit = pd.DataFrame(
        {'wind': [0.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_invest_unit.columns.name = 'unit'
    par = _par({'wind': 100.0})

    out = _synthesize_invest_dual(v, par)
    # Without the regime-(a) fallback, period_ref is None and max-total is
    # dropped -> out is 0.  With the fix it broadcasts negate(-9) = +9.
    assert out.loc['p1', 'wind'] == 9.0
    assert out.loc['p2', 'wind'] == 9.0


def test_floor_binding_is_negative():
    """(c) A lower-floor ``>=`` row dual arrives POSITIVE; negating yields a
    NEGATIVE effective value (the floor over-forces capacity).  Also proves
    the Increment-3 min-side duals actually populate the output."""
    periods = ['p1', 'p2']
    v = _base_v(periods)
    v.dual_minInvest_period = pd.DataFrame(
        {'battery': [12.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_minInvest_period.columns.name = 'entity'
    v.dual_minCumulative = pd.DataFrame(
        {'battery': [0.0, 4.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_minCumulative.columns.name = 'entity'
    par = _par({'battery': 50.0})

    out = _synthesize_invest_dual(v, par)
    assert out.loc['p1', 'battery'] == -12.0  # negate(+12) = -12, NEGATIVE
    assert out.loc['p2', 'battery'] == -4.0


def test_not_built_is_negative():
    """(a) The v_invest column reduced cost is ≥ 0 (lower bound, unprofitable
    to build); divide by unitsize then negate → NEGATIVE per MW."""
    periods = ['p1']
    v = _base_v(periods)
    # Reduced cost in obj/v_invest-unit; unitsize converts to obj/MW.
    v.dual_invest_unit = pd.DataFrame(
        {'solar': [2000.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_invest_unit.columns.name = 'unit'
    par = _par({'solar': 200.0})

    out = _synthesize_invest_dual(v, par)
    # 2000 / 200 = 10 obj/MW reduced cost; negate → -10.
    assert out.loc['p1', 'solar'] == -10.0


def test_interior_is_zero():
    """(d) An entity that binds nothing and has zero reduced cost contributes
    exactly 0."""
    periods = ['p1', 'p2']
    v = _base_v(periods)
    v.dual_maxInvest_period = pd.DataFrame(
        {'wind': [0.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_maxInvest_period.columns.name = 'entity'
    v.dual_minInvest_period = pd.DataFrame(
        {'wind': [0.0, 0.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_minInvest_period.columns.name = 'entity'
    par = _par({'wind': 100.0})

    out = _synthesize_invest_dual(v, par)
    assert (out['wind'] == 0.0).all()


def test_group_caps_and_floors_expand_and_sign():
    """Group ``maxInvestGroup_*`` (<= , raw<0 → +) and ``minInvestGroup_*``
    (>=, raw>0 → -) duals expand to member entities with the right sign."""
    periods = ['p1']
    v = _base_v(periods)
    v.group_entity_invest = pd.DataFrame(
        {'group': ['renewables', 'renewables'], 'entity': ['wind', 'solar']}
    )
    v.dual_maxInvestGroup_period = pd.DataFrame(
        {'renewables': [-8.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_maxInvestGroup_period.columns.name = 'group'
    v.dual_minInvestGroup_cumulative = pd.DataFrame(
        {'renewables': [3.0]}, index=pd.Index(periods, name='period')
    )
    v.dual_minInvestGroup_cumulative.columns.name = 'group'
    par = _par({'wind': 100.0, 'solar': 200.0})

    out = _synthesize_invest_dual(v, par)
    # Each member gets negate(-8) + negate(+3) = +8 - 3 = +5.
    assert out.loc['p1', 'wind'] == 5.0
    assert out.loc['p1', 'solar'] == 5.0


def test_all_regimes_combined_one_entity_per_regime():
    """End-to-end across regimes: each entity sits in a distinct regime and
    reads the documented sign; the combined frame carries them side by side."""
    periods = ['p1']
    idx = pd.Index(periods, name='period')
    v = _base_v(periods)
    # (b) capped -> positive
    v.dual_maxInvest_period = pd.DataFrame({'capped': [-20.0]}, index=idx)
    v.dual_maxInvest_period.columns.name = 'entity'
    # (c) floored -> negative
    v.dual_minInvest_period = pd.DataFrame({'floored': [15.0]}, index=idx)
    v.dual_minInvest_period.columns.name = 'entity'
    # (a) not built -> negative
    v.dual_invest_unit = pd.DataFrame({'notbuilt': [1000.0]}, index=idx)
    v.dual_invest_unit.columns.name = 'unit'
    par = _par({'capped': 100.0, 'floored': 100.0, 'notbuilt': 100.0})

    out = _synthesize_invest_dual(v, par)
    assert out.loc['p1', 'capped'] == 20.0    # POSITIVE
    assert out.loc['p1', 'floored'] == -15.0  # NEGATIVE
    assert out.loc['p1', 'notbuilt'] == -10.0  # 1000/100 = 10, negate -> -10


def test_no_duals_returns_empty():
    """With every frame empty the synthesis returns an empty frame (caller
    skips emitting)."""
    v = _base_v(['p1'])
    par = _par({})
    out = _synthesize_invest_dual(v, par)
    assert out.empty
