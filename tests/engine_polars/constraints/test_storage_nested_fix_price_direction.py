"""End-to-end sign/scale check for the ``fix_price`` storage handoff.

The nested ``fix_price`` handoff funnels the PARENT (coarse) solve's
``nodeBalance_eq`` row dual — the storage node's marginal energy value
(water value) — into the CHILD (fine) solve as a reference price on the
terminal ``v_state``.  The child §10.1 objective term is a REVENUE term
(``obj -= p_ref * v_state * unitsize * factor``), so a HIGHER coarse
water value must produce a POSITIVE reference price that drives the
child's terminal storage state UP (incentivise keeping storage full).

The reference AMPL model writes ``-dual`` because GLPK returns the
equality-balance dual with the opposite sign to HiGHS.  polar_high's
``constraint_dual`` returns the HiGHS shadow price ∂obj/∂rhs directly
(a positive water value → dual = +C), so the native producer must take
``+dual``.  These tests pin that end to end:

* ``test_fix_price_producer_sign_and_scale`` — a parent whose storage
  node has a known positive water value +C yields
  ``p_fix_storage_price ≈ +C`` (positive, scaling linearly with C).
  Pre-fix (``-dual``) this was ``-C`` and the assertion FAILS.
* ``test_fix_price_round_trip_raises_terminal_state`` — feeding the
  produced price into a child LP drives the terminal ``v_state`` to its
  upper bound (full) for a high water value, and leaves it un-driven for
  a ~zero water value.  Pre-fix the negative price drove it to the lower
  bound (empty) — the economically inverted outcome.
"""
from __future__ import annotations

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool
from flextool.engine_polars.input import (
    FlexData,
    build_handoff_from_solution,
)
from flextool.engine_polars._solve_handoff import SolveHandoff

from .conftest import _time_axes, solver_options


def _one(dt):
    return Param(("d", "t"), dt.with_columns(value=pl.lit(1.0)))


def _parent_with_water_value(penalty: float) -> FlexData:
    """Parent LP: storage node ``s`` whose marginal energy value = penalty.

    ``s`` carries a unit demand at t02 met only by upward slack priced at
    ``penalty``, so the LP pays ``penalty`` per unit and
    ``nodeBalance_eq[s, d1, t02]`` has dual = +penalty (the water value).
    Node ``s`` is flagged ``fix_price`` so the producer extracts its dual.
    """
    periods = ["d1"]
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(periods, 2)
    nb = pl.DataFrame({"n": ["s"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"), nb_dt.with_columns(
        value=pl.when(pl.col("t") == "t02").then(pl.lit(-1.0))
                .otherwise(pl.lit(0.0))
    ).select("n", "d", "t", "value"))
    p_pen_up = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(penalty)).select("n", "d", "t", "value"))
    p_pen_dn = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(penalty)).select("n", "d", "t", "value"))
    nsfm = pl.DataFrame({"node": ["s"], "method": ["fix_price"]})
    return FlexData(
        dt=dt, p_step_duration=p_step, p_timestep_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen_up, p_penalty_down=p_pen_dn,
        node__storage_nested_fix_method=nsfm,
    )


def _produce_fix_price(penalty: float, tmp_path):
    data = _parent_with_water_value(penalty)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    assert sol.optimal

    parent_handoff = SolveHandoff()
    parent_handoff.fix_storage_timesteps = pl.DataFrame(
        {"period": ["d1"], "step": ["t02"]})  # type: ignore[attr-defined]
    (tmp_path / "solve_data").mkdir(parents=True, exist_ok=True)
    handoff = build_handoff_from_solution(
        sol, tmp_path, "parent_solve",
        flex_data=data, parent_handoff=parent_handoff)
    return handoff


def test_fix_price_producer_sign_and_scale(tmp_path):
    """Positive water value +C → p_fix_storage_price ≈ +C (positive)."""
    for C in (10.0, 50.0):
        handoff = _produce_fix_price(C, tmp_path / f"c{int(C)}")
        assert handoff.fix_storage_price is not None, (
            "Producer must populate fix_storage_price for fix_price nodes")
        fp = handoff.fix_storage_price.filter(
            (pl.col("node") == "s") & (pl.col("step") == "t02"))
        assert fp.height == 1
        price = float(fp["p_fix_storage_price"][0])
        # infl = share = scale = 1 → price == +dual == +C.
        assert price == pytest.approx(C, rel=1e-4), (
            f"Water value +{C} must yield reference price +{C} (positive), "
            f"got {price}. A negative value means the HiGHS dual sign was "
            f"not flipped (the ported AMPL -dual is wrong for HiGHS).")


def _child_ref_price_lp(ref_price: float) -> FlexData:
    """Child LP valuing terminal v_state[s, d1, t02] at ``ref_price``.

    Mirrors the storage-handoff wiring Sc B setup: a storage node ``s``
    with cyclic binding, no flows/inflow, and a §10.1 reference-price
    term.  The optimum terminal state is driven to the upper bound when
    ref_price > 0 (revenue) and to the lower bound when ref_price < 0.
    """
    periods = ["d1"]
    dt, p_step, p_rp, p_infl, p_psh = _time_axes(periods, 2)
    nb = pl.DataFrame({"n": ["s"]})
    nb_dt = nb.join(dt, how="cross")
    p_inflow = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(0.0)).select("n", "d", "t", "value"))
    p_pen = Param(("n", "d", "t"),
        nb_dt.with_columns(value=pl.lit(1e6)).select("n", "d", "t", "value"))
    nodeState = pl.DataFrame({"n": ["s"]})
    nodeState_dt = nodeState.join(dt, how="cross")
    nodeState_first_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).first().select("n", "d", "t"))
    nodeState_last_dt = (nodeState_dt.sort(["n", "d", "t"])
        .group_by(["n", "d"], maintain_order=True).last().select("n", "d", "t"))
    dtttdt = pl.DataFrame({
        "d": ["d1", "d1"], "t": ["t01", "t02"],
        "t_previous": ["t02", "t01"],
        "t_previous_within_timeset": ["t02", "t01"],
        "d_previous": ["d1", "d1"],
        "t_previous_within_solve": ["t02", "t01"],
    })
    data = FlexData(
        dt=dt, p_step_duration=p_step, p_timestep_weight=p_rp,
        p_inflation_op=p_infl, p_period_share=p_psh,
        nodeBalance=nb, nodeBalance_dt=nb_dt,
        p_inflow=p_inflow, p_penalty_up=p_pen, p_penalty_down=p_pen,
        nodeState=nodeState, nodeState_dt=nodeState_dt,
        nodeState_first_dt=nodeState_first_dt,
        nodeState_last_dt=nodeState_last_dt,
        p_state_unitsize=Param(("n",),
            pl.DataFrame({"n": ["s"], "value": [10.0]})),
        p_state_upper=Param(("n", "d"),
            pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [1.0]})),
        p_state_self_discharge=Param(("n",),
            pl.DataFrame({"n": ["s"], "value": [0.0]})),
        p_state_existing_capacity=Param(("n", "d"),
            pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [10.0]})),
        dtttdt=dtttdt,
        storage_bind_within_timeblock=pl.DataFrame({"n": ["s"]}),
        period_last=pl.DataFrame({"d": ["d1"]}),
        p_storage_state_reference_price=Param(("n", "d"),
            pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [ref_price]})),
    )
    return data


def _solve_child(ref_price: float):
    data = _child_ref_price_lp(ref_price)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    assert sol.optimal
    v = sol.value("v_state").filter(
        (pl.col("n") == "s") & (pl.col("t") == "t02"))
    return float(v["value"][0])


def test_fix_price_round_trip_raises_terminal_state(tmp_path):
    """Parent water value ↑ ⇒ child terminal v_state ↑ (end to end).

    Produce the reference price from a high-water-value parent and from a
    ~zero-water-value parent, feed each into an identical child, and
    assert the high-water-value price drives the terminal state to the
    upper bound while the ~zero price does not.
    """
    hi = _produce_fix_price(100.0, tmp_path / "hi")
    price_hi = float(hi.fix_storage_price.filter(
        pl.col("step") == "t02")["p_fix_storage_price"][0])
    assert price_hi > 0, (
        f"High water value must produce a positive reference price; "
        f"got {price_hi}")

    state_hi = _solve_child(price_hi)
    state_zero = _solve_child(0.0)

    assert state_hi == pytest.approx(1.0, abs=1e-6), (
        f"A positive reference price (from high coarse water value) must "
        f"drive the child terminal state to its upper bound 1.0; got "
        f"{state_hi}. A value near 0 means the produced price was "
        f"non-positive (inverted sign).")
    assert state_hi > state_zero + 0.5, (
        f"Terminal state must respond upward to the coarse water value: "
        f"state(high)={state_hi} should exceed state(zero)={state_zero}")
