"""Surface A.1 (Temporal Foundation & Cost Weighting) and A.2
(Node Capacity & Balance Topology) loader tests.

Entry point: ``load_flextool`` reached top-down (``_load_time`` /
``_load_node`` are not called directly).  Each test overlays one input on
the smallest dispatch-only seed (``base``) and asserts the resulting
``FlexData`` field literally.

* The node-inflow tests load through the production call: the overlay
  goes into a clone of the ``base`` run's provider (``tiny_run`` +
  ``put_frame``) and ``load_scenario`` loads it with the DB reader.
* The step-duration, period-share, timestep-weight and penalty tests
  check reload-only seeds: with a DB reader, ``apply_derived_a``
  (``params/_derived.py``) re-derives ``dt`` / ``p_step_duration`` (from
  the DB timeline, unless ``solve.new_stepduration`` is set),
  ``p_period_share`` / ``p_window_fraction`` (from that dt),
  ``p_timestep_weight`` (non-rp solves) and ``p_penalty_up`` /
  ``p_penalty_down`` on every load, so their overlays are never read on
  the production path.  They stay on the reader-less CSV load of
  ``tiny_workdir`` and move to ``tests/csv_reload/`` (c24 S1-8).
"""
from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flextool.engine import load_flextool
from tests._support.load import load_scenario

from .conftest import put_frame, write_csv


# --- A.1 ------------------------------------------------------------

_SCATTERED_STEPS = [
    {"period": "p2020", "step": "t0001", "step_duration": "2.5"},
    {"period": "p2020", "step": "t0003", "step_duration": "1.0"},
    {"period": "p2030", "step": "t0002", "step_duration": "0.5"},
    {"period": "p2030", "step": "t0004", "step_duration": "1.0"},
]


def _write_scattered_periods(workdir, share_hours: dict[str, float]) -> None:
    """Two scattered periods in ``steps_in_use.csv`` plus the per-period files that go with a period set.

    The seed workdir has one period (p2020); adding p2030 to ``steps_in_use.csv`` alone leaves the
    period-keyed companions without a p2030 row.  ``complete_period_share_of_year_calc.csv`` (the
    complete solve's hours of each period / 8760) and ``p_inflation_factor_operations_yearly.csv``
    weight every operational cost of their period, so they are written for both periods.
    """
    write_csv(workdir, "steps_in_use.csv", _SCATTERED_STEPS)
    write_csv(workdir, "complete_period_share_of_year_calc.csv",
              [{"period": d, "value": h / 8760.0} for d, h in share_hours.items()])
    write_csv(workdir, "p_inflation_factor_operations_yearly.csv",
              [{"period": d, "value": 1.0} for d in share_hours])


def test_step_duration_string_cast_and_scattered_periods(tiny_workdir):
    """Covers A1-step_duration_casts_to_float (direct) +
    A1-multi_period_steps_scattered (direct).  Sparse 2-period scatter
    with string ``step_duration`` confirms (a) no spurious cross-join
    injects (p2020,t0002) etc, and (b) ``cast(Float64)`` parses "2.5".

    Reload-only seed (c24 S1-7): see the module docstring; S1-8 moves it to tests/csv_reload/.
    """
    _write_scattered_periods(tiny_workdir, {"p2020": 3.5, "p2030": 1.5})
    data = load_flextool(tiny_workdir)
    # Hand-calc: dt = exactly the 4 input rows, no cross-product.
    # Phase 4: data.dt's d/t columns are Enum (post-vocab activation);
    # cast the expected frame's columns to match the live cascade dtypes.
    from flextool.engine.input._axis_enums import schema_dtype
    expected_dt = pl.DataFrame({
        "d": ["p2020", "p2020", "p2030", "p2030"],
        "t": ["t0001", "t0003", "t0002", "t0004"],
    }, schema={
        "d": schema_dtype(None, "d"),
        "t": schema_dtype(None, "t"),
    })
    assert_frame_equal(data.dt.sort(["d", "t"]),
                        expected_dt.sort(["d", "t"]))
    # Hand-calc: string "2.5" -> 2.5 via cast(Float64).
    sd = data.p_step_duration.frame.sort(["d", "t"])
    assert sd["value"].to_list() == pytest.approx([2.5, 1.0, 0.5, 1.0],
                                                    rel=1e-7)


def test_scattered_periods_window_fraction(tiny_workdir):
    """``p_window_fraction`` of a reader-less load: the dt hours of each period over the complete
    solve's hours (``complete_period_share_of_year_calc.csv``), summed over the steps of dt only.

    Hand-calc: p2020 dt hours 2.5 + 1.0 = 3.5 of a complete 3.5 h -> exactly 1.0 (snapped);
    p2030 dt hours 0.5 + 1.0 = 1.5 of a complete 6.0 h -> 0.25.

    Reload-only seed (c24 S1-7): see the module docstring; S1-8 moves it to tests/csv_reload/.
    """
    _write_scattered_periods(tiny_workdir, {"p2020": 3.5, "p2030": 6.0})
    data = load_flextool(tiny_workdir)
    share = data.p_period_share.frame.sort("d")
    assert [str(x) for x in share["d"]] == ["p2020", "p2030"]
    assert share["value"].to_list() == pytest.approx([3.5 / 8760, 6.0 / 8760], rel=1e-12)
    wf = data.p_window_fraction.frame.sort("d")
    assert data.p_window_fraction.dims == ("d",)
    assert [str(x) for x in wf["d"]] == ["p2020", "p2030"]
    assert wf["value"][0] == 1.0
    assert wf["value"][1] == pytest.approx(0.25, rel=1e-12)


def test_period_share_without_a_dt_period_raises(tiny_workdir):
    """A workdir whose ``complete_period_share_of_year_calc.csv`` lacks a period of dt does not load.

    Every operational cost of a period is divided by its share (``op_factor``) and the
    ``co2_max_period`` cap is scaled by its window fraction, so a dt period without a share would drop
    out of the objective and the caps.  The load boundary refuses it, naming the period.

    Reload-only seed (c24 S1-7): see the module docstring; S1-8 moves it to tests/csv_reload/.
    """
    write_csv(tiny_workdir, "steps_in_use.csv", _SCATTERED_STEPS)  # the seed's share has p2020 only
    with pytest.raises(ValueError, match=r"p_period_share has no row for dt periods \['p2030'\]"):
        load_flextool(tiny_workdir)


def test_timestep_weight_strict_false_scientific_notation(tiny_workdir):
    """Covers A1-timestep_weight_strict_false (direct).  String values
    in scientific notation must round-trip via cast(Float64,
    strict=False) and the override must merge over the per-(d,t)
    default of 1.0.

    Reload-only seed (c24 S1-7): see the module docstring; S1-8 moves it to tests/csv_reload/.
    """
    write_csv(tiny_workdir, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": 1.0},
        {"period": "p2020", "step": "t0002", "step_duration": 1.0},
    ])
    write_csv(tiny_workdir, "timestep_weight.csv", [
        {"period": "p2020", "time": "t0001", "value": "1.5e-3"},
    ])
    data = load_flextool(tiny_workdir)
    rp = data.p_timestep_weight.frame.sort(["d", "t"])
    # Hand-calc: t0001 override 1.5e-3, t0002 default 1.0.
    assert rp["value"].to_list() == pytest.approx([1.5e-3, 1.0], rel=1e-7)


# --- A.2 ------------------------------------------------------------

def test_inflow_long_new_format_with_null_fill(tiny_run):
    """Covers A2-inflow_long_format_new (direct) +
    A2-inflow_null_fill_zero (direct).  Long ``pdtNodeInflow.csv`` with
    one explicit null exercises ``_read_wide_per_entity``'s long branch
    plus ``fill_null(0.0)``.

    Production path: ``p_inflow`` is the ``solve_data/pdtNodeInflow`` seed
    on every load (``apply_derived_a`` step 6 keeps it), so the overlay is
    what the model sees.  dt comes from the DB timeline (48 steps of
    ``base``); the overlay covers two of them.
    """
    run, P = tiny_run
    put_frame(P, "pdtNodeInflow.csv", [
        {"node": "west", "period": "p2020", "time": "t0001", "value": 42.0},
        {"node": "west", "period": "p2020", "time": "t0002", "value": None},
    ])
    data = load_scenario(run, provider=P)
    inf = data.p_inflow.frame.sort(["n", "d", "t"])
    # Hand-calc: 42.0 stays, null -> 0.0.
    assert inf["value"].to_list() == pytest.approx([42.0, 0.0], rel=1e-7)
    assert inf["n"].to_list() == ["west", "west"]
    assert inf["t"].to_list() == ["t0001", "t0002"]
    # The overlay is consumed: the un-overlaid base inflow is 48 rows starting -589.0, -537.0.
    base = run.last_flex_data.p_inflow.frame.sort(["n", "d", "t"])
    assert base.height == 48
    assert base["value"].head(2).to_list() != inf["value"].to_list()


def test_inflow_legacy_wide_format_unpivot(tiny_run):
    """Covers A2-inflow_long_format_legacy_wide (direct).  Legacy
    ``solve``-prefixed wide layout (one column per entity) goes through
    the ``unpivot`` branch in ``_read_wide_per_entity``.  Two timesteps
    times one node -> 2 long rows.

    Production path, as the long-format test above.
    """
    run, P = tiny_run
    put_frame(P, "pdtNodeInflow.csv", [
        {"solve": "s1", "period": "p2020", "time": "t0001", "west": 100.0},
        {"solve": "s1", "period": "p2020", "time": "t0002", "west": 200.0},
    ])
    data = load_scenario(run, provider=P)
    inf = data.p_inflow.frame.sort(["n", "d", "t"])
    # Hand-calc: unpivot of the two-row wide CSV preserves order.
    assert inf.height == 2
    assert inf["value"].to_list() == pytest.approx([100.0, 200.0], rel=1e-7)
    assert inf["n"].to_list() == ["west", "west"]
    # The overlay is consumed: the un-overlaid base inflow has 48 rows.
    assert run.last_flex_data.p_inflow.frame.height == 48


def test_pdtNode_penalty_extraction_empty_up_cast_down(tiny_workdir):
    """Covers A2-penalty_up_from_pdtNode (indirect via empty assertion)
    + A2-penalty_down_from_pdtNode (direct) + A2-penalty_empty_when_missing
    (direct) + A2-penalty_cast_with_null_fill (direct).  Single
    ``pdtNode.csv`` carries only ``penalty_down`` rows (one string,
    one null) -> ``p_penalty_up`` is height-0 (empty-Param convention),
    ``p_penalty_down`` has the cast ``200.0`` and the null filled to
    ``0.0``.

    Reload-only seed (c24 S1-7): see the module docstring; S1-8 moves it to tests/csv_reload/.
    """
    write_csv(tiny_workdir, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": 1.0},
        {"period": "p2020", "step": "t0002", "step_duration": 1.0},
    ])
    write_csv(tiny_workdir, "pdtNode.csv", [
        {"node": "west", "param": "penalty_down",
         "period": "p2020", "time": "t0001", "value": "200"},
        {"node": "west", "param": "penalty_down",
         "period": "p2020", "time": "t0002", "value": None},
    ])
    data = load_flextool(tiny_workdir)
    # Hand-calc: no penalty_up rows in pdtNode -> empty Param frame.
    assert data.p_penalty_up.frame.height == 0
    dn = data.p_penalty_down.frame.sort(["n", "d", "t"])
    # Hand-calc: "200" -> 200.0 via cast(strict=False); null -> 0.0
    # via fill_null.
    assert dn["value"].to_list() == pytest.approx([200.0, 0.0], rel=1e-7)
    assert dn["n"].to_list() == ["west", "west"]
