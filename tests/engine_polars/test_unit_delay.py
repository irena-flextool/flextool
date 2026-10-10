"""Unit ``delay``: a constant or a map of delays (index: hours, value:
weight).

The delayed input term of ``conversion_indirect`` joins
``process_delayed__duration`` (p, td) with the time-shift table
``dtt__delay_duration`` and the weights ``p_process_delay_weight``.  The
duration set used to come from a map-only reader: a constant ``delay``
gave no rows, the unit's inputs were taken out of the undelayed sum and
not put back, and the unit was forced to zero output without a message.
In rolling sub-solves the shift table and weights came from text CSV seeds
while the durations were floats, and the build crashed on the ``td`` join.
All three tables now come from one read of ``delay`` (Float64 ``td``) on
every path, and :func:`_delay.check_delay_tables` raises when they do not
line up.

Fixtures: ``tests.json`` + an override alternative, built from JSON.
``coal_chp``: indirect, 1000 MW, efficiency 0.9, coal_market -> west +
heat (ratio-fixing user constraint switched off).
"""
from __future__ import annotations

import polars as pl
import pytest

from tests.engine_polars._unit_db import (
    CHP, CHP_CHAIN, COAL, b64, demand, flow, make_db, no_ratio_constraint,
    run, time_map, utf8,
)

pytestmark = pytest.mark.solver

EFF = 0.9
N_STEPS = 48  # the y2020_2day_dispatch solve: two days of hourly steps
# West demand varies so that a time shift is visible in the coal flow.
WEST = {f"t{i:04d}": 300.0 + 10.0 * (i % 7) for i in range(1, N_STEPS + 1)}


def _delay_map(pairs: list[tuple[float, float]]) -> list:
    return b64({"index_type": "float", "rank": 1,
                "data": [[td, w] for td, w in pairs],
                "index_name": "constraint", "type": "map"}, "map")


def _values(delay: list) -> list:
    return no_ratio_constraint() + [
        ["node", "west", "inflow",
         time_map({t: -v for t, v in WEST.items()}, -300.0)],
        demand("heat", 0.0),
        ["unit", CHP, "delay", delay],
    ]


_ROLLING = [
    ["solve", "y2020_2day_dispatch", "solve_mode",
     b64("rolling_window", "str")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_horizon",
     b64(24.0, "float")],
    ["solve", "y2020_2day_dispatch", "rolling_solve_jump",
     b64(12.0, "float")],
]


def _out_and_coal(step) -> tuple[list[float], list[float]]:
    out = flow(step, CHP, CHP, "west") + flow(step, CHP, CHP, "heat")
    return out.to_list(), flow(step, CHP, COAL, CHP).to_list()


def _shifted(xs: list[float], k: int) -> list[float]:
    """``xs[t − k]`` with the period's cyclic wrap."""
    return xs[-k:] + xs[:-k]


def _solve(tpf, alt: str, delay: list, extra: list | None = None):
    url, scen = make_db(tpf, alt=alt, values=_values(delay) + (extra or []),
                        base_chain=CHP_CHAIN)
    return run(tpf, url, scen)


def test_constant_delay_unit_runs_and_shifts(tmp_path_factory) -> None:
    """``delay = 1.0``: the unit serves west (it used to be forced to 0 with
    all demand unserved) and the coal at ``t − 1`` feeds the output at
    ``t``: ``out[t] = 0.9 · coal[t − 1]``."""
    step = list(_solve(tmp_path_factory, "dly_const",
                       b64(1.0, "float")).values())[-1]
    pdd = utf8(step.flex_data.process_delayed__duration)
    assert pdd.to_dicts() == [{"p": CHP, "td": "1.0"}]
    out, coal = _out_and_coal(step)
    assert out == pytest.approx(list(WEST.values()), rel=1e-6)
    assert out == pytest.approx(
        [EFF * c for c in _shifted(coal, 1)], rel=1e-6)


def test_constant_delay_equals_single_entry_map(tmp_path_factory) -> None:
    """A constant delay of 1 h and the map ``{1: 1.0}`` give the same
    flows and objective."""
    const = list(_solve(tmp_path_factory, "dly_c1",
                        b64(1.0, "float")).values())[-1]
    as_map = list(_solve(tmp_path_factory, "dly_m1",
                         _delay_map([(1.0, 1.0)])).values())[-1]
    assert const.solution.obj == pytest.approx(as_map.solution.obj, rel=1e-9)
    for a, b in zip(_out_and_coal(const), _out_and_coal(as_map)):
        assert a == pytest.approx(b, rel=1e-9, abs=1e-9)


def test_weighted_map_delay(tmp_path_factory) -> None:
    """Map ``{1: 0.25, 2: 0.75}``: ``out[t] = 0.9 · (0.25 · coal[t − 1] +
    0.75 · coal[t − 2])``."""
    step = list(_solve(tmp_path_factory, "dly_w",
                       _delay_map([(1.0, 0.25), (2.0, 0.75)])).values())[-1]
    out, coal = _out_and_coal(step)
    assert out == pytest.approx(list(WEST.values()), rel=1e-6)
    expected = [EFF * (0.25 * a + 0.75 * b)
                for a, b in zip(_shifted(coal, 1), _shifted(coal, 2))]
    assert out == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("kind", ["constant", "map"])
def test_rolling_delayed_unit(tmp_path_factory, kind: str) -> None:
    """Rolling sub-solves (synthetic-solve loader path) build and serve
    west with a delayed unit; every roll's three delay tables share the
    Float64 ``td`` (the roll used to crash on the ``td`` join)."""
    delay = (b64(1.0, "float") if kind == "constant"
             else _delay_map([(1.0, 1.0)]))
    steps = _solve(tmp_path_factory, f"dly_roll_{kind}", delay, _ROLLING)
    rolls = [k for k in steps if "_roll_" in k]
    assert len(rolls) >= 2, f"expected roll sub-solves, got {list(steps)}"
    for name in rolls:
        fd = steps[name].flex_data
        assert fd.process_delayed__duration.schema["td"] == pl.Float64
        assert fd.dtt__delay_duration.schema["td"] == pl.Float64
        assert fd.p_process_delay_weight.frame.schema["td"] == pl.Float64
        out, coal = _out_and_coal(steps[name])
        assert sum(out) > 0.0, name
        assert out[1:] == pytest.approx(
            [EFF * c for c in coal[:-1]], rel=1e-6), name


# ── The check: tables that do not line up are an error ────────────────
def _toy():
    from tests.engine_polars.fixtures import flex_toy_delay
    return flex_toy_delay.data()


def test_check_accepts_consistent_tables() -> None:
    from flextool.engine_polars import _delay

    _delay.check_delay_tables(_toy())


def test_check_raises_without_durations() -> None:
    from flextool.engine_polars import _delay

    d = _toy()
    d.process_delayed__duration = None
    with pytest.raises(_delay.DelayDataError,
                       match="p_d: no delay durations"):
        _delay.check_delay_tables(d)


def test_check_raises_on_td_dtype_mismatch() -> None:
    """A text ``td`` against the Float64 shift table would silently empty
    the join (or crash on it): an error naming the tables."""
    from flextool.engine_polars import _delay

    d = _toy()
    d.process_delayed__duration = pl.DataFrame({"p": ["p_d"], "td": ["1.0"]})
    with pytest.raises(_delay.DelayDataError, match="dtypes differ"):
        _delay.check_delay_tables(d)


def test_check_raises_on_duration_without_shift_rows() -> None:
    from flextool.engine_polars import _delay

    d = _toy()
    d.process_delayed__duration = pl.DataFrame({"p": ["p_d"], "td": [2.0]})
    with pytest.raises(_delay.DelayDataError,
                       match="without time-shift rows"):
        _delay.check_delay_tables(d)


def test_check_raises_on_duration_without_weight() -> None:
    from polar_high import Param

    from flextool.engine_polars import _delay

    d = _toy()
    d.p_process_delay_weight = Param(("p", "td"), pl.DataFrame(
        {"p": ["other"], "td": [1.0], "value": [1.0]}))
    with pytest.raises(_delay.DelayDataError, match="without a weight"):
        _delay.check_delay_tables(d)


# ── Delayed connections ───────────────────────────────────────────────
@pytest.mark.parametrize("delay", ["constant", "map"])
def test_delayed_connection_builds(tmp_path_factory, delay: str) -> None:
    """A connection with ``delay`` is not an indirect unit, so the check
    leaves it alone and the model builds and solves.  (The delay itself is
    not applied to connections: the delayed term exists only in
    ``conversion_indirect``; see docs/dev/unit_floors_and_reserves.md.)"""
    value = (b64(2.0, "float") if delay == "constant"
             else _delay_map([(2.0, 1.0)]))
    url, scen = make_db(
        tmp_path_factory, alt=f"dly_conn_{delay}",
        values=[["connection", "west_north", "transfer_method",
                 b64("no_losses_no_variable_cost", "str")],
                ["connection", "west_north", "delay", value]],
        base_chain=["init", "west", "coal", "wind", "network"])
    step = list(run(tmp_path_factory, url, scen).values())[-1]
    pdd = utf8(step.flex_data.process_delayed__duration)
    assert pdd.to_dicts() == [{"p": "west_north", "td": "2.0"}]
