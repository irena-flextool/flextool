"""Surface B.1 / B.2 — energy-balance + storage state binding.

Two focused constraint tests on the shared minimal fixtures:

* ``test_node_balance_fix_quantity_eq_lower`` — covers **B1.5**.
  Pins v_state[s, d_last, t_last] · unitsize == Σ p_fix_storage_quantity
  via the upper→lower handoff path (mod:2760).  Layered onto
  ``toy_storage_2t`` with hand-built ``n_fix_storage_quantity`` /
  ``ndt_fix_storage_quantity`` / ``p_fix_storage_quantity`` /
  ``period_branch`` / ``period_last`` / ``dtt_timeline_matching``
  overlay frames — no separate fixture needed.

* ``test_profile_state_upper_invest_tightening`` — covers **B2.4**.
  Mutates ``toy_storage_2t`` to make ``s`` invest-eligible with zero
  existing capacity, adds a ``profile_state_upper`` constraint
  (profile=0.4) and a cyclic in-period balance with positive inflow that
  forces v_state to absorb energy at one timestep.  At the optimum
  v_invest_n must reach the level required by the invest-tightened RHS:
  ``v_state ≤ profile · v_invest_n``.
"""
from __future__ import annotations

import dataclasses
from typing import Any  # noqa: F401  # used in string annotation on _solve

import polars as pl
import pytest

from polar_high import Param, Problem
from flextool.engine_polars import build_flextool

from .conftest import solver_options


def _solve(data) -> tuple[Problem, "Any"]:
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve(options=solver_options())
    return pb, sol


# ---------------------------------------------------------------------------
# B1.5 — node_balance_fix_quantity_eq_lower (lower→upper handoff).

def test_node_balance_fix_quantity_eq_lower(toy_storage_2t):
    """Covers B1.5 — `node_balance_fix_quantity_eq_lower` lower→upper handoff.

    Hand-calc: with one upper anchor row p_fix_storage_quantity[s, du, t_up]=4
    mapped via dtt_timeline_matching (du, t_up) → (d1, t02), the
    constraint pins v_state[s, d1, t02] · unitsize(=10) == 4
    ⇒ v_state[s, d1, t02] = 0.4.
    """
    # B1.5 setup is not part of toy_storage_2t — the fix-quantity overlay
    # frames are absent there.  Add them inline (small, single-row each).
    n = "s"
    n_fix = pl.DataFrame({"n": [n]})
    ndt_fix = pl.DataFrame({"n": [n], "d": ["d1"], "t": ["t_up"]})
    p_fix = Param(("n", "d", "t"),
                  pl.DataFrame({"n": [n], "d": ["d1"], "t": ["t_up"],
                                "value": [4.0]}))
    # period_branch: (d_upper="d1", d="d1") — single-period self-map.
    period_branch = pl.DataFrame({"d_upper": ["d1"], "d": ["d1"]})
    period_last = pl.DataFrame({"d": ["d1"]})
    # dtt_timeline_matching: lower (d=d1, t=t02) maps to upper (t_upper=t_up).
    dtt = pl.DataFrame({"d": ["d1"], "t": ["t02"], "t_upper": ["t_up"]})

    data = dataclasses.replace(
        toy_storage_2t,
        n_fix_storage_quantity=n_fix,
        ndt_fix_storage_quantity=ndt_fix,
        p_fix_storage_quantity=p_fix,
        period_branch=period_branch,
        period_last=period_last,
        dtt_timeline_matching=dtt,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "node_balance_fix_quantity_eq_lower" in set(pb.cstr_names())

    v_state = sol.value("v_state")
    pinned = v_state.filter((pl.col("n") == "s") & (pl.col("t") == "t02"))
    assert pinned.height == 1
    # Hand-calc: 4.0 / unitsize(10.0) = 0.4
    assert pinned["value"][0] == pytest.approx(0.4, rel=1e-7)


# ---------------------------------------------------------------------------
# B2.4 — profile_state_* invest tightening.

def test_profile_state_upper_invest_tightening(toy_storage_2t):
    """Covers B2.4 — invest tightening on profile_state_upper.

    Mutates toy_storage_2t: storage node ``s`` invest-eligible with
    existing_capacity=0, profile_upper=0.4 over both timesteps, plus a
    cyclic +2 / -2 inflow that forces v_state[s, t02] = 2 via balance.
    The invest-tightened upper bound v_state ≤ 0.4 · (existing + invest)
    then forces v_invest_n[s, d1] = 2 / 0.4 = 5.
    """
    f = "profA"
    nb_dt_s = pl.DataFrame({
        "n": ["n", "n", "s", "s"],
        "d": ["d1"] * 4,
        "t": ["t01", "t02", "t01", "t02"],
    })
    # Inflow: zero out node "n" demand so no slack mass; on s set
    # [-2, +2] so the cyclic balance forces state[t02] = state[t01] + 2.
    p_inflow_new = Param(("n", "d", "t"),
        nb_dt_s.with_columns(
            value=pl.when(pl.col("n") == "n").then(0.0)
                  .when(pl.col("t") == "t01").then(-2.0)
                  .otherwise(2.0)
        ).select("n", "d", "t", "value"))

    # Storage cycling: bind state across timesteps via storage_bind_within_timeblock.
    storage_bind = pl.DataFrame({"n": ["s"]})

    # Existing capacity = 0; unitsize = 1 so state_change is in MWh
    # directly (state[t02] − state[t01] = 2 with the chosen inflow).
    # Loose maxState upper so it doesn't dominate the profile UB.
    p_state_existing_zero = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [0.0]}))
    p_state_upper_loose = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [10.0]}))
    p_state_unitsize_one = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [1.0]}))

    # Invest sets/params for storage node ``s``.
    nd_invest = pl.DataFrame({"n": ["s"], "d": ["d1"]})
    ed_invest = pl.DataFrame({"e": ["s"], "d": ["d1"]})
    edd_invest = pl.DataFrame({"e": ["s"], "d_invest": ["d1"], "d": ["d1"]})
    p_entity_max_units = Param(("e", "d"),
        pl.DataFrame({"e": ["s"], "d": ["d1"], "value": [100.0]}))
    ed_invest_period_set = ed_invest.clone()
    ed_invest_max_period = Param(("e", "d"),
        pl.DataFrame({"e": ["s"], "d": ["d1"], "value": [1000.0]}))
    ed_annu = Param(("e", "d"),
        pl.DataFrame({"e": ["s"], "d": ["d1"], "value": [10.0]}))
    ed_lf = Param(("e", "d"),
        pl.DataFrame({"e": ["s"], "d": ["d1"], "value": [0.0]}))

    # Profile data for the upper bound.
    node_profile_upper = pl.DataFrame({"n": ["s"], "f": [f]})
    p_profile_value = Param(("f", "d", "t"),
        pl.DataFrame({"f": [f, f], "d": ["d1", "d1"],
                      "t": ["t01", "t02"], "value": [0.4, 0.4]}))

    data = dataclasses.replace(
        toy_storage_2t,
        p_inflow=p_inflow_new,
        storage_bind_within_timeblock=storage_bind,
        p_state_existing_capacity=p_state_existing_zero,
        p_state_upper=p_state_upper_loose,
        p_state_unitsize=p_state_unitsize_one,
        nd_invest_set=nd_invest,
        ed_invest_set=ed_invest,
        edd_invest_set=edd_invest,
        p_entity_max_units=p_entity_max_units,
        ed_invest_period_set=ed_invest_period_set,
        ed_invest_max_period=ed_invest_max_period,
        ed_entity_annual_discounted=ed_annu,
        ed_lifetime_fixed_cost=ed_lf,
        node_profile_upper=node_profile_upper,
        p_profile_value=p_profile_value,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "profile_state_upper_limit" in set(pb.cstr_names())

    # Hand-calc: cyclic balance with inflow [+2, -2] and v_state ≥ 0
    # forces v_state[s, t01]=0, v_state[s, t02]=2.  Profile upper says
    # v_state ≤ 0.4·(0 + v_invest_n) ⇒ v_invest_n ≥ 5; cost-min picks 5.
    v_state = sol.value("v_state").filter(pl.col("n") == "s")
    s_t02 = v_state.filter(pl.col("t") == "t02")["value"][0]
    assert s_t02 == pytest.approx(2.0, rel=1e-7)
    v_inv = sol.value("v_invest_n").filter(pl.col("n") == "s")
    assert v_inv["value"][0] == pytest.approx(5.0, rel=1e-7)


def test_profile_state_sparse_availability_default(toy_storage_2t):
    """Regression: a profile storage node with NO authored ``availability``
    must keep its default-1.0 factor, not be inner-join-dropped to RHS=0.

    Node-state twin of ``test_profile_upper_sparse_availability_default``
    (B12.5).  ``p_node_availability`` is SPARSE, and ``_add_node_profile_cstr``
    folds it into the ``profile_state_*`` RHS via ``Param * Param`` (an inner
    join), so before the densify fix a node absent from the sparse Param had
    its existing-capacity bound collapsed to 0 → forced ``v_state = 0``.  No
    existing fixture combined an authored-availability node with a default-
    availability profile node, so the parity sweep never tripped it.

    Storage node ``s`` has existing capacity 10, unitsize 1 (so v_state is in
    MWh and existing_count = 10), and a ``profile_upper = 0.4`` ⇒ the RHS bound
    is ``0.4 · 10 · availability``.  A cyclic +2 / −2 inflow forces
    v_state[t02] = 2.  We additionally author ``availability`` on the UNRELATED
    demand node ``n`` (0.5) so ``p_node_availability`` is non-empty but absent
    for ``s`` — the exact trigger.

    Hand-calc: with the fix ``s`` keeps availability 1.0 ⇒ bound = 4 ≥ 2 and
    v_state[t02] = 2.  Before the fix ``s``'s RHS dropped to 0 ⇒ v_state ≤ 0,
    and the cyclic demand is forced onto loss-of-load slack ⇒ v_state[t02] = 0.
    """
    f = "profA"
    nb_dt_s = pl.DataFrame({
        "n": ["n", "n", "s", "s"],
        "d": ["d1"] * 4,
        "t": ["t01", "t02", "t01", "t02"],
    })
    # Zero node "n" demand; cyclic [-2, +2] on s forces v_state[t02] = 2.
    p_inflow_new = Param(("n", "d", "t"),
        nb_dt_s.with_columns(
            value=pl.when(pl.col("n") == "n").then(0.0)
                  .when(pl.col("t") == "t01").then(-2.0)
                  .otherwise(2.0)
        ).select("n", "d", "t", "value"))
    storage_bind = pl.DataFrame({"n": ["s"]})
    # unitsize 1 ⇒ v_state in MWh; existing 10 ⇒ existing_count = 10.
    p_state_unitsize_one = Param(("n",),
        pl.DataFrame({"n": ["s"], "value": [1.0]}))
    p_state_existing_ten = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [10.0]}))
    # Loose maxState so the profile_upper (0.4·10·avail) is the binding bound.
    p_state_upper_loose = Param(("n", "d"),
        pl.DataFrame({"n": ["s"], "d": ["d1"], "value": [100.0]}))
    node_profile_upper = pl.DataFrame({"n": ["s"], "f": [f]})
    p_profile_value = Param(("f", "d", "t"),
        pl.DataFrame({"f": [f, f], "d": ["d1", "d1"],
                      "t": ["t01", "t02"], "value": [0.4, 0.4]}))
    # SPARSE: only the unrelated demand node ``n`` authors availability;
    # the profile node ``s`` is absent → must fall back to default 1.0.
    p_node_availability = Param(("n",),
        pl.DataFrame({"n": ["n"], "value": [0.5]}))

    data = dataclasses.replace(
        toy_storage_2t,
        p_inflow=p_inflow_new,
        storage_bind_within_timeblock=storage_bind,
        p_state_existing_capacity=p_state_existing_ten,
        p_state_upper=p_state_upper_loose,
        p_state_unitsize=p_state_unitsize_one,
        node_profile_upper=node_profile_upper,
        p_profile_value=p_profile_value,
        p_node_availability=p_node_availability,
    )
    pb, sol = _solve(data)
    assert sol.optimal
    assert "profile_state_upper_limit" in set(pb.cstr_names())
    v_state = sol.value("v_state").filter(pl.col("n") == "s")
    s_t02 = v_state.filter(pl.col("t") == "t02")["value"][0]
    # Regression: ``s`` keeps default availability 1.0 ⇒ bound 4 ⇒ state = 2.
    # Before the densify fix ``s`` was dropped from the RHS → bound 0 →
    # v_state forced to 0 (demand shunted to slack).
    assert s_t02 == pytest.approx(2.0, rel=1e-7)
