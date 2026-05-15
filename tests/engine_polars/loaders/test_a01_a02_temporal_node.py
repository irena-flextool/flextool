"""Surface A.1 (Temporal Foundation & Cost Weighting) and A.2
(Node Capacity & Balance Topology) loader tests.

Entry point: ``flextool.engine_polars.load_flextool(workdir)`` (mirrors
``tests/engine_polars/test_db_direct_parity.py::_load_pair`` — the
canonical CSV path through ``_load_time`` / ``_load_node`` is reached
top-down through ``load_flextool``, not by calling those private helpers
directly).  Each test mutates exactly one ``solve_data/*.csv`` overlay
on top of ``tiny_workdir`` (the smallest dispatch-only seed) and asserts
the resulting ``FlexData`` field literally.
"""
from __future__ import annotations

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flextool.engine_polars import load_flextool

from .conftest import write_csv


# --- A.1 ------------------------------------------------------------

def test_step_duration_string_cast_and_scattered_periods(tiny_workdir):
    """Covers A1-step_duration_casts_to_float (direct) +
    A1-multi_period_steps_scattered (direct).  Sparse 2-period scatter
    with string ``step_duration`` confirms (a) no spurious cross-join
    injects (p2020,t0002) etc, and (b) ``cast(Float64)`` parses "2.5".
    """
    write_csv(tiny_workdir, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": "2.5"},
        {"period": "p2020", "step": "t0003", "step_duration": "1.0"},
        {"period": "p2030", "step": "t0002", "step_duration": "0.5"},
        {"period": "p2030", "step": "t0004", "step_duration": "1.0"},
    ])
    data = load_flextool(tiny_workdir)
    # Hand-calc: dt = exactly the 4 input rows, no cross-product.
    expected_dt = pl.DataFrame({
        "d": ["p2020", "p2020", "p2030", "p2030"],
        "t": ["t0001", "t0003", "t0002", "t0004"],
    })
    assert_frame_equal(data.dt.sort(["d", "t"]),
                        expected_dt.sort(["d", "t"]))
    # Hand-calc: string "2.5" -> 2.5 via cast(Float64).
    sd = data.p_step_duration.frame.sort(["d", "t"])
    assert sd["value"].to_list() == pytest.approx([2.5, 1.0, 0.5, 1.0],
                                                    rel=1e-7)


def test_rp_cost_weight_strict_false_scientific_notation(tiny_workdir):
    """Covers A1-rp_cost_weight_strict_false (direct).  String values
    in scientific notation must round-trip via cast(Float64,
    strict=False) and the override must merge over the per-(d,t)
    default of 1.0.
    """
    write_csv(tiny_workdir, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": 1.0},
        {"period": "p2020", "step": "t0002", "step_duration": 1.0},
    ])
    write_csv(tiny_workdir, "rp_cost_weight.csv", [
        {"period": "p2020", "time": "t0001", "value": "1.5e-3"},
    ])
    data = load_flextool(tiny_workdir)
    rp = data.p_rp_cost_weight.frame.sort(["d", "t"])
    # Hand-calc: t0001 override 1.5e-3, t0002 default 1.0.
    assert rp["value"].to_list() == pytest.approx([1.5e-3, 1.0], rel=1e-7)


# --- A.2 ------------------------------------------------------------

def test_inflow_long_new_format_with_null_fill(tiny_workdir):
    """Covers A2-inflow_long_format_new (direct) +
    A2-inflow_null_fill_zero (direct).  Long ``pdtNodeInflow.csv`` with
    one explicit null exercises ``_read_wide_per_entity``'s long branch
    plus ``fill_null(0.0)``.
    """
    write_csv(tiny_workdir, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": 1.0},
        {"period": "p2020", "step": "t0002", "step_duration": 1.0},
    ])
    write_csv(tiny_workdir, "pdtNodeInflow.csv", [
        {"node": "west", "period": "p2020", "time": "t0001", "value": 42.0},
        {"node": "west", "period": "p2020", "time": "t0002", "value": None},
    ])
    data = load_flextool(tiny_workdir)
    inf = data.p_inflow.frame.sort(["n", "d", "t"])
    # Hand-calc: 42.0 stays, null -> 0.0.
    assert inf["value"].to_list() == pytest.approx([42.0, 0.0], rel=1e-7)
    assert inf["n"].to_list() == ["west", "west"]


def test_inflow_legacy_wide_format_unpivot(tiny_workdir):
    """Covers A2-inflow_long_format_legacy_wide (direct).  Legacy
    ``solve``-prefixed wide layout (one column per entity) goes through
    the ``unpivot`` branch in ``_read_wide_per_entity``.  Two timesteps
    times one node -> 2 long rows.
    """
    write_csv(tiny_workdir, "steps_in_use.csv", [
        {"period": "p2020", "step": "t0001", "step_duration": 1.0},
        {"period": "p2020", "step": "t0002", "step_duration": 1.0},
    ])
    write_csv(tiny_workdir, "pdtNodeInflow.csv", [
        {"solve": "s1", "period": "p2020", "time": "t0001", "west": 100.0},
        {"solve": "s1", "period": "p2020", "time": "t0002", "west": 200.0},
    ])
    data = load_flextool(tiny_workdir)
    inf = data.p_inflow.frame.sort(["n", "d", "t"])
    # Hand-calc: unpivot of the two-row wide CSV preserves order.
    assert inf.height == 2
    assert inf["value"].to_list() == pytest.approx([100.0, 200.0], rel=1e-7)
    assert inf["n"].to_list() == ["west", "west"]


def test_pdtNode_penalty_extraction_empty_up_cast_down(tiny_workdir):
    """Covers A2-penalty_up_from_pdtNode (indirect via empty assertion)
    + A2-penalty_down_from_pdtNode (direct) + A2-penalty_empty_when_missing
    (direct) + A2-penalty_cast_with_null_fill (direct).  Single
    ``pdtNode.csv`` carries only ``penalty_down`` rows (one string,
    one null) -> ``p_penalty_up`` is height-0 (empty-Param convention),
    ``p_penalty_down`` has the cast ``200.0`` and the null filled to
    ``0.0``.
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
