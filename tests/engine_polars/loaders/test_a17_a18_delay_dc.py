"""Surface A.17 (delayed-dispatch processes) and A.18 (DC power-flow inputs)
loader tests.

A.17: ``_delay.load_data`` returns a blank dict when the feature is
inactive (no ``solve_data/process_delayed.csv``); ``has_feature`` is
False on the resulting FlexData-shaped object.

A.18: pin the truncated π literal ``_PI_LITERAL = 3.14159265`` used as
the ±bound on ``v_angle``.  Regression-pin: a refactor that swaps to
``math.pi`` would silently change LP coefficients vs flextool's MPS
golden output.  See ``flextool.mod:1680-1681``.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from polar_high import Problem

from flextool.engine_polars import _delay, _dc_power_flow


# ====================================================================
# A.17 — _delay.load_data feature gate
# ====================================================================

def test_feature_gate_missing_csv_returns_blank(tmp_path: Path):
    """Covers A17-feature_gate_missing_csv_returns_blank.

    Empty ``solve_data/`` (no ``process_delayed.csv``) ⇒ load_data
    returns the all-None blank dict; ``has_feature`` False on the
    resulting FlexData.
    """
    inp = tmp_path / "input"; sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()

    out = _delay.load_data(inp, sd)
    # Hand-calc: missing CSV → blank dict, every value None, all 8 keys present.
    expected_keys = {
        "process_delayed", "process_delayed__duration",
        "process_source_delayed", "process_source_undelayed",
        "process_source_sink_delayed", "process_source_sink_undelayed",
        "dtt__delay_duration", "p_process_delay_weight",
    }
    assert set(out.keys()) == expected_keys
    assert all(v is None for v in out.values())
    # has_feature on a FlexData-shaped object built from the blank dict.
    d = SimpleNamespace(**out)
    assert _delay.has_feature(d) is False


# ====================================================================
# A.18 — _dc_power_flow.add_variables ±π bound literal
# ====================================================================

def test_angle_var_bounds_pi_literal(tmp_path: Path):
    """Covers A18-angle_var_bounds_pi_literal.

    Regression-pin: the angle Var bound is the truncated literal
    ``3.14159265`` (NOT ``math.pi`` ≈ 3.141592653589793).  A swap would
    silently shift LP coefficients vs flextool's MPS bit pattern.
    """
    # Hand-calc: literal is the documented 8-digit truncation of π.
    assert _dc_power_flow._PI_LITERAL == 3.14159265
    # Sanity: NOT equal to math.pi (would be 3.141592653589793).
    assert _dc_power_flow._PI_LITERAL != math.pi

    # Minimal DC PF setup: 1 node, 1 connection, 1 timestep.
    d = SimpleNamespace(
        node_dc_power_flow=pl.DataFrame({"n": ["n1"]}),
        connection_dc_power_flow=pl.DataFrame({"p": ["line1"]}),
        node_reference_angle=None,
        p_connection_susceptance=None,
        dt=pl.DataFrame({"d": ["d1"], "t": ["t01"]}),
        process_source_sink=None,  # no v_flow_back path needed
    )
    assert _dc_power_flow.has_feature(d) is True

    m = Problem()
    out = _dc_power_flow.add_variables(m, d)
    v_angle = out["v_angle"]
    # Hand-calc: bounds are ±_PI_LITERAL == ±3.14159265 exactly.
    assert v_angle.lower == -3.14159265
    assert v_angle.upper == 3.14159265
