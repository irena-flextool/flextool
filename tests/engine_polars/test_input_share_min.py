"""``unit__inputNode.input_share_min`` — the input mixing minimum.

While the unit is running, input ``s`` supplies at least ``share_s`` of
the unit's CURRENT input energy (flow × ``conversion_flow_coeff``)::

    conv_s · in_s  ≥  share_s · Σ_{s'} conv_{s'} · in_{s'}       (minInputShare)

The row is homogeneous (``0 ≥ 0`` when the unit has no input), so it holds
for units with or without online variables.  Validation (once per
scenario) rejects shares outside [0, 1], shares summing above 1, a share
above the input's ``input_share_max`` and shares on a unit with a negative
input ``conversion_flow_coeff``.

Fixture: ``tests.json`` ``coal_chp`` (1000 MW, efficiency 0.9) plus a gas
input, built from JSON.
"""
from __future__ import annotations

import logging

import polars as pl
import pytest

from tests.engine_polars._unit_db import (
    CHP, CHP_CHAIN, COAL, GAS, GAS_ENTITIES, b64, build, cstr_over, flow,
    gas, make_db, no_ratio_constraint, online, run, solve, time_map, uc,
    utf8,
)

ZERO_T = {"t0005": 0.0, "t0006": 0.0}   # no demand: the unit is idle


def _share(node: str, share: float, name: str = "input_share_min") -> list:
    return ["unit__inputNode", [CHP, node], name, b64(share, "float")]


def _demands() -> list:
    return [["node", "west", "inflow", time_map(ZERO_T, -500.0)],
            ["node", "heat", "inflow", time_map(ZERO_T, -300.0)]]


def _solve(tpf, alt: str, values: list):
    return solve(tpf, alt=alt, base_chain=CHP_CHAIN, entities=GAS_ENTITIES,
                 values=no_ratio_constraint() + gas(100.0) + _demands()
                 + values)


def _cflow(step, source, sink):
    return flow(step, CHP, source, sink)


def _check_share(step, share: float, gas_conv: float = 1.0) -> None:
    coal = _cflow(step, COAL, CHP)
    g = _cflow(step, GAS, CHP) * gas_conv
    total = coal + g
    running = total > 1e-6
    assert running.sum() > 0
    ratio = (g.filter(running) / total.filter(running)).to_list()
    assert ratio == pytest.approx([share] * len(ratio), rel=1e-6)
    # idle timesteps: both inputs 0 (the homogeneous row is 0 >= 0)
    assert total.filter(~running).abs().max() in (None, pytest.approx(0.0))


@pytest.mark.parametrize("gas_conv", [1.0, 2.0])
def test_co_mixing_minimum(tmp_path_factory, gas_conv: float) -> None:
    """Gas at 100 €/MWh would never be used, but ``input_share_min 0.3``
    makes it supply exactly 30 % of the conversion-weighted input
    whenever the unit runs; at the zero-demand steps both inputs are 0."""
    step = _solve(tmp_path_factory, f"ism_mix_{int(gas_conv)}",
                  [_share(GAS, 0.3),
                   ["unit__inputNode", [CHP, GAS], "conversion_flow_coeff",
                    b64(gas_conv, "float")]])
    _check_share(step, 0.3, gas_conv)
    for t in (4, 5):
        assert _cflow(step, COAL, CHP)[t] == pytest.approx(0.0, abs=1e-6)
    over = cstr_over(build(step.flex_data), "minInputShare")
    assert over is not None
    assert set(over.select("p", "source", "sink").unique().iter_rows()) == {
        (CHP, GAS, CHP)}


def test_with_unit_commitment(tmp_path_factory) -> None:
    """The mixing row and the per-unit min-load floor coexist; the unit
    still shuts down at the zero-demand steps."""
    step = _solve(tmp_path_factory, "ism_uc", [_share(GAS, 0.3)]
                  + uc(CHP, 0.4))
    _check_share(step, 0.3)
    on = online(step, CHP)
    assert on[4] == pytest.approx(0.0, abs=1e-6)
    assert on[5] == pytest.approx(0.0, abs=1e-6)


def test_share_one_runs_on_that_input_only(tmp_path_factory) -> None:
    """``input_share_min = 1`` on one of two inputs is legal ("this input
    only"): the other input stays at 0 (its row coefficient is exactly 0)."""
    step = _solve(tmp_path_factory, "ism_one", [_share(GAS, 1.0)])
    assert _cflow(step, COAL, CHP).abs().max() == pytest.approx(0.0, abs=1e-6)
    assert _cflow(step, GAS, CHP).max() > 1.0


def test_delayed_unit_shares_hold_at_input_time(tmp_path_factory) -> None:
    """A delayed unit (map delay 1 h) mixes its inputs at the input-side
    timestep: the share holds for every t where it burns."""
    delay = b64({"index_type": "float", "rank": 1, "data": [[1.0, 1.0]],
                 "index_name": "constraint", "type": "map"}, "map")
    step = _solve(tmp_path_factory, "ism_delay",
                  [_share(GAS, 0.3), ["unit", CHP, "delay", delay]])
    _check_share(step, 0.3)


# ── validation ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("alt,values,match", [
    ("ism_sum", [_share(GAS, 0.6), _share(COAL, 0.6)], "sums to 1.2"),
    ("ism_range", [_share(GAS, 1.5)], r"outside \[0, 1\]"),
    ("ism_gt_max", [_share(GAS, 0.7),
                    _share(GAS, 0.6, "input_share_max")],
     "could not run at full load"),
    ("ism_negconv", [_share(GAS, 0.3),
                     ["unit__inputNode", [CHP, COAL],
                      "conversion_flow_coeff", b64(-1.0, "float")]],
     "negative input conversion_flow_coeff"),
])
def test_invalid_shares_are_errors(tmp_path_factory, alt, values,
                                   match) -> None:
    from flextool.engine_polars._solve_state import FlexToolConfigError

    url, scen = make_db(tmp_path_factory, alt=alt, base_chain=CHP_CHAIN,
                        entities=GAS_ENTITIES,
                        values=no_ratio_constraint() + gas(100.0) + values)
    with pytest.raises(FlexToolConfigError, match=match):
        run(tmp_path_factory, url, scen)


def test_share_on_zero_coefficient_input_warns(tmp_path_factory,
                                               caplog) -> None:
    """A share on an input with ``conversion_flow_coeff = 0`` has no
    effect: a warning, and no minInputShare row."""
    caplog.set_level(logging.WARNING)
    step = _solve(tmp_path_factory, "ism_conv0",
                  [_share(GAS, 0.3),
                   ["unit__inputNode", [CHP, GAS], "conversion_flow_coeff",
                    b64(0.0, "float")]])
    assert any("no effect" in r.getMessage() and GAS in r.getMessage()
               for r in caplog.records)
    assert cstr_over(build(step.flex_data), "minInputShare") is None


def test_share_on_single_input_unit_warns(tmp_path_factory, caplog) -> None:
    caplog.set_level(logging.WARNING)
    solve(tmp_path_factory, alt="ism_single", base_chain=CHP_CHAIN,
          values=no_ratio_constraint() + [_share(COAL, 0.5)])
    assert any("single input" in r.getMessage() for r in caplog.records)


def test_build_guard_rejects_bad_shares(tmp_path_factory) -> None:
    """Hand-built FlexData bypass the validator; the emitter guards the
    range itself."""
    import dataclasses

    from polar_high import Param

    from flextool.engine_polars._solve_state import FlexToolConfigError

    step = _solve(tmp_path_factory, "ism_guard", [_share(GAS, 0.3)])
    fd = step.flex_data
    frame = utf8(fd.p_process_source_input_share_min.frame).with_columns(
        value=pl.lit(1.4))
    bad = dataclasses.replace(
        fd, p_process_source_input_share_min=Param(("p", "source"), frame))
    with pytest.raises(FlexToolConfigError, match="input_share_min"):
        build(bad)


# ── rolling: warm path, validation once ─────────────────────────────────
def test_rolling_warm_matches_cold_and_validates_once(tmp_path_factory,
                                                      caplog) -> None:
    from flextool.engine_polars import run_chain_from_db

    caplog.set_level(logging.WARNING)
    rolling = [
        ["solve", "y2020_2day_dispatch", "solve_mode",
         b64("rolling_window", "str")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_horizon",
         b64(24.0, "float")],
        ["solve", "y2020_2day_dispatch", "rolling_solve_jump",
         b64(12.0, "float")],
    ]
    url, scen = make_db(tmp_path_factory, alt="ism_roll", base_chain=CHP_CHAIN,
                        entities=GAS_ENTITIES,
                        values=no_ratio_constraint() + gas(100.0)
                        + _demands() + [_share(GAS, 0.3),
                                        _share(COAL, 1.5, "input_share_max")]
                        + uc(CHP, 0.4) + rolling)
    warm = run_chain_from_db(url, scen, keep_solutions=True, warm=True,
                             work_folder=tmp_path_factory.mktemp("ism_rw"))
    n_warn = sum("input_share_max 1.5 is above 1" in r.getMessage()
                 for r in caplog.records)
    assert n_warn == 1
    cold = run_chain_from_db(url, scen, keep_solutions=True, warm=False,
                             work_folder=tmp_path_factory.mktemp("ism_rc"))
    assert len(warm) > 1
    for w, c in zip(warm.values(), cold.values()):
        for src in (COAL, GAS):
            assert _cflow(w, src, CHP).to_list() == pytest.approx(
                _cflow(c, src, CHP).to_list(), rel=1e-6, abs=1e-6)
        _check_share(w, 0.3)
