"""Model-build tests for ``node.penalty_method == 'off'``.

A node whose ``penalty_method`` is ``'off'`` must be built with NO balance
slack variables (``vq_state_up`` / ``vq_state_down``), turning its node
balance into a hard equality.  ``'regular'`` (the default, threaded as
``nodeBalance_penalty_off = None``) keeps the slack, byte-identical to
prior behaviour.

The fixtures here are the minimal slack-only LP: one or two balance nodes
carrying pure exogenous demand (negative inflow) and no supply, so the
balance can ONLY be met by the up-slack.  This isolates the slack-var
gating from every other subsystem:

* ``'regular'`` node → up-slack absorbs the demand at ``penalty_up`` cost;
  the LP is feasible with a hand-computable objective.
* ``'off'`` node → the up-slack column is gone, ``0 == demand`` cannot
  hold, and the LP is infeasible.
"""
from __future__ import annotations

import polars as pl

from polar_high import Param, Problem

from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import FlexData

# Two timesteps in one period; unit step-duration / weights so the
# objective is a clean penalty × demand × steps product.
_DT = pl.DataFrame({"d": ["p", "p"], "t": ["t01", "t02"]})
_DEMAND = 3.0      # MWh of exogenous demand per step (negative inflow)
_PENALTY = 100.0   # CUR/MWh up-slack penalty


def _dt_param(value: float) -> Param:
    return Param(("d", "t"), _DT.with_columns(value=pl.lit(value)))


def _build(nodes: list[str], off: list[str] | None) -> FlexData:
    """Minimal slack-only FlexData: every node in *nodes* is a balance
    node with ``-_DEMAND`` inflow each step; nodes listed in *off* carry
    ``penalty_method == 'off'`` (no slack vars)."""
    nb = pl.DataFrame({"n": nodes})
    nb_dt = nb.join(_DT, how="cross")
    inflow = Param(("n", "d", "t"),
                   nb_dt.with_columns(value=pl.lit(-_DEMAND))
                        .select("n", "d", "t", "value"))
    penalty = Param(("n", "d", "t"),
                    nb_dt.with_columns(value=pl.lit(_PENALTY))
                         .select("n", "d", "t", "value"))
    kw = dict(
        dt=_DT,
        p_step_duration=_dt_param(1.0),
        p_timestep_weight=_dt_param(1.0),
        p_inflation_op=Param(("d",), pl.DataFrame({"d": ["p"], "value": [1.0]})),
        p_period_share=Param(("d",), pl.DataFrame({"d": ["p"], "value": [1.0]})),
        nodeBalance=nb,
        nodeBalance_dt=nb_dt,
        p_inflow=inflow,
        p_penalty_up=penalty,
        p_penalty_down=penalty,
    )
    if off is not None:
        # ``off == []`` deliberately builds a height-0 frame so the empty-set
        # guard in model.py (``penalty_off.height > 0``) is exercised.
        kw["nodeBalance_penalty_off"] = pl.DataFrame(
            {"n": off}, schema={"n": pl.Utf8})
    return FlexData(**kw)


def _slack_nodes(pb: Problem, var: str) -> list[str]:
    frame = pb._vars[var].frame
    return sorted(set(frame["n"].to_list())) if frame.height else []


def _solve(pb: Problem):
    return pb.solve(options={"random_seed": 42, "parallel": "off"})


def test_regular_node_keeps_slack_and_is_feasible() -> None:
    d = _build(["load"], off=None)
    pb = Problem()
    build_flextool(pb, d)
    assert _slack_nodes(pb, "vq_state_up") == ["load"]
    assert _slack_nodes(pb, "vq_state_down") == ["load"]
    sol = _solve(pb)
    assert sol.optimal
    # up-slack must absorb _DEMAND each of the two steps at _PENALTY each.
    assert abs(sol.obj - _PENALTY * _DEMAND * 2) < 1e-9


def test_off_node_has_no_slack_variables() -> None:
    d = _build(["load"], off=["load"])
    pb = Problem()
    build_flextool(pb, d)
    assert _slack_nodes(pb, "vq_state_up") == []
    assert _slack_nodes(pb, "vq_state_down") == []


def test_off_node_makes_balance_hard_infeasible() -> None:
    d = _build(["load"], off=["load"])
    pb = Problem()
    build_flextool(pb, d)
    sol = _solve(pb)
    # No slack + unbalanceable demand ⇒ hard equality 0 == demand ⇒ infeasible.
    assert not sol.optimal


def test_off_is_selective_regular_node_untouched() -> None:
    # Two balance nodes; only 'drop' is penalty_method='off'.  The build
    # must keep 'keep's slack vars and remove only 'drop's.
    d = _build(["keep", "drop"], off=["drop"])
    pb = Problem()
    build_flextool(pb, d)
    assert _slack_nodes(pb, "vq_state_up") == ["keep"]
    assert _slack_nodes(pb, "vq_state_down") == ["keep"]


def test_empty_off_set_is_byte_identical_to_none() -> None:
    # An empty (height-0) off frame must be a no-op — same slack index as
    # the None default (guards the byte-parity invariant on vq_idx).
    d_none = _build(["a", "b"], off=None)
    d_empty = _build(["a", "b"], off=[])
    pb_none, pb_empty = Problem(), Problem()
    build_flextool(pb_none, d_none)
    build_flextool(pb_empty, d_empty)
    assert (pb_none._vars["vq_state_up"].frame
            .equals(pb_empty._vars["vq_state_up"].frame))
