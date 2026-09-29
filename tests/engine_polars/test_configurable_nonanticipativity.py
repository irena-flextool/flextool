"""Slice G — configurable operational non-anticipativity window.

Design: ``specs/sliceG_configurable_nonanticipativity_design.md`` (§4
byte-parity, §7 hand-checkable test).

The new ``solve.non_anticipativity_periods`` parameter (v71) decouples the
operational-NA window (the (d,t) set over which the four
``non_anticipativity_*`` families pin storage / online / reserve dispatch
across stochastic branches) from ``realized_periods``.  Tri-state:

* unset       -> legacy ``realized_dispatch ∪ fix_storage`` window
  (byte-identical to prior behaviour — the ``pinned_ops`` guard);
* ``[]``      -> empty window -> operations branch freely from t0 (the
  classic two-stage capacity expansion — the ``free_ops`` scenario);
* ``[p, ...]``-> tie only over those periods' timesteps.

Fixture ``stoch_two_period_free_ops.json`` (built via
``build_stoch_two_period_free_ops.py``): two 3-step periods p2035 / p2040,
three branches mid / up / low declared at t0001 (t0-branching), a single
shared investable ``base`` unit (``stochastic_invest_method=none`` -> ONE
``v_invest``), and a stochastic-group storage node ``resv`` fed by
branch-asymmetric wind timing so each branch has a UNIQUE cost-optimal
storage trajectory (design F6).  Two scenarios differ ONLY in the
``non_anticipativity_periods`` value.

Hand-verified reference numbers (deterministic LP; see the builder):

* ``pinned_ops`` (unset): storage NA fires (12 rows), the set CSV holds
  the whole 6-step horizon, and ``v_state[resv]`` is IDENTICAL across
  mid / up / low (all 0.5) — the branches cannot follow their opposite
  surplus/deficit timing.  Objective 1_920_800; shared v_invest 12.
* ``free_ops`` (``[]``): zero NA rows, header-only set CSV, and
  ``v_state[resv]`` DIVERGES (up rises above 0.5, low falls below).
  Objective 757_369.4 < pinned; shared v_invest 4.5.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

SOLVE = "stoch_2p_free"

# Deterministic LP reference objectives (see the fixture builder).
PINNED_OBJ = 1_920_800.0
FREE_OBJ = 757_369.4


# ---------------------------------------------------------------------------
# Work-folder fixtures (full cascade + snapshot, AUTOSCALE_STRICT on)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def free_ops_wf(scenario_workdir):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("free_ops",
                                db_fixture="stoch_two_period_free_ops")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def pinned_ops_wf(scenario_workdir):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("pinned_ops",
                                db_fixture="stoch_two_period_free_ops")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


def _resolve(wf):
    """In-process build+solve off the snapshot workdir; returns
    ``(data, sol, pb)``."""
    from polar_high import Problem
    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, sol, pb


@pytest.fixture(scope="module")
def free_ops_resolved(free_ops_wf):
    return _resolve(free_ops_wf)


@pytest.fixture(scope="module")
def pinned_ops_resolved(pinned_ops_wf):
    return _resolve(pinned_ops_wf)


def _v_state_by_branch(sol, period_prefix: str) -> dict[str, list[float]]:
    """Map branch-period name -> ordered ``v_state[resv]`` trajectory for
    every (d, t) whose period ``d`` starts with *period_prefix*."""
    vs = sol.value("v_state")
    out: dict[str, list[tuple[str, float]]] = {}
    for r in vs.iter_rows(named=True):
        if str(r["n"]) != "resv":
            continue
        d = str(r["d"])
        if not d.startswith(period_prefix):
            continue
        out.setdefault(d, []).append((str(r["t"]), round(float(r["value"]), 4)))
    return {d: [v for _t, v in sorted(ts)] for d, ts in out.items()}


# ---------------------------------------------------------------------------
# free_ops (non_anticipativity_periods = []) — operations free from t0
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_free_ops_set_csv_is_header_only(free_ops_wf) -> None:
    """§4.4.2/§7: an empty window emits a present-but-empty (header-only)
    dt_non_anticipativity_set.csv."""
    p = free_ops_wf / "solve_data" / "dt_non_anticipativity_set.csv"
    assert p.exists(), "set CSV must still be emitted (present-but-empty)"
    assert pl.read_csv(p).height == 0


@pytest.mark.solver
def test_free_ops_no_na_constraint_rows(free_ops_resolved) -> None:
    """§7: an empty window -> the model's dtna.height==0 early-out skips
    ALL four non_anticipativity_* families (zero rows in the built LP)."""
    _data, _sol, pb = free_ops_resolved
    na = [n for n in pb.cstr_names() if "non_anticipativity" in n]
    assert na == [], f"expected no NA families, got {na}"


@pytest.mark.solver
def test_free_ops_shared_single_v_invest(free_ops_resolved) -> None:
    """§7: with stochastic_invest_method=none investment stays realized-only
    (R-O6) -> exactly ONE shared v_invest for the ``base`` unit (no
    per-branch fan)."""
    _data, sol, _pb = free_ops_resolved
    vi = sol.value("v_invest_p")
    keys = {str(r["d"]) for r in vi.iter_rows(named=True)}
    # Only the trunk invest period p2035 — never a branch copy
    # (p2035_up / p2035_low), which would signal a fanned invest axis.
    assert keys == {"p2035"}
    val = {str(r["d"]): round(float(r["value"]), 3)
           for r in vi.iter_rows(named=True)}
    assert val == pytest.approx({"p2035": 4.5})


@pytest.mark.solver
def test_free_ops_v_state_diverges_across_branches(free_ops_resolved) -> None:
    """§7 (F6): with operations freed each branch reaches its OWN storage
    optimum, so v_state[resv] is strictly distinct across mid / up / low —
    deterministic given the branch-asymmetric wind timing.  ``up`` (surplus
    early) charges above 0.5; ``low`` (deficit early) discharges below;
    ``mid`` (flat) stays idle."""
    _data, sol, _pb = free_ops_resolved
    traj = _v_state_by_branch(sol, "p2035")
    assert set(traj) == {"p2035", "p2035_up", "p2035_low"}
    up = traj["p2035_up"]
    low = traj["p2035_low"]
    mid = traj["p2035"]
    # mid trunk (flat wind) keeps storage idle at the 0.5 reference.
    assert mid == [0.5, 0.5, 0.5]
    # up strictly above, low strictly below — at every step, and the two
    # branches are pairwise distinct (a UNIQUE per-branch optimum).
    assert all(u > 0.5 for u in up), up
    assert all(lo < 0.5 for lo in low), low
    assert up != low != mid


@pytest.mark.solver
def test_free_ops_objective(free_ops_wf) -> None:
    """§7: the free-ops objective is the deterministic LP optimum
    (per-branch dispatch under the shared capacity)."""
    obj = float(
        pl.read_parquet(free_ops_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(FREE_OBJ, rel=1e-6)


# ---------------------------------------------------------------------------
# pinned_ops (unset) — legacy window, byte-parity guard
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_pinned_ops_set_csv_is_legacy_horizon(pinned_ops_wf) -> None:
    """§4.4.1: unset window -> the set CSV holds the legacy
    realized_dispatch ∪ fix_storage union = the whole 6-step horizon."""
    p = pinned_ops_wf / "solve_data" / "dt_non_anticipativity_set.csv"
    assert p.exists()
    df = pl.read_csv(p)
    assert df.height == 6
    # Exactly the two periods' three steps each.
    pairs = {(str(r["period"]), str(r["time"])) for r in df.iter_rows(named=True)}
    assert pairs == {
        ("p2035", "t0001"), ("p2035", "t0002"), ("p2035", "t0003"),
        ("p2040", "t0004"), ("p2040", "t0005"), ("p2040", "t0006"),
    }


@pytest.mark.solver
def test_pinned_ops_storage_na_fires(pinned_ops_resolved) -> None:
    """§7: unset window -> the storage NA family is present and ranges over
    the whole horizon (12 rows: 6 (d,t) × the up/low sibling pairs)."""
    _data, _sol, pb = pinned_ops_resolved
    names = pb.cstr_names()
    assert "non_anticipativity_storage_use" in names
    over = pb.cstrs_named("non_anticipativity_storage_use")[0].over
    assert over.height == 12


@pytest.mark.solver
def test_pinned_ops_v_state_equal_across_branches(pinned_ops_resolved) -> None:
    """§7 (byte-parity guard): with the legacy window the storage NA family
    ties the branches' net-charge equal, so v_state[resv] is IDENTICAL
    across mid / up / low on every constrained (d,t)."""
    _data, sol, _pb = pinned_ops_resolved
    for prefix in ("p2035", "p2040"):
        traj = _v_state_by_branch(sol, prefix)
        assert len(traj) == 3, traj
        vals = list(traj.values())
        assert vals[0] == vals[1] == vals[2], (prefix, traj)


@pytest.mark.solver
def test_pinned_ops_objective(pinned_ops_wf) -> None:
    obj = float(
        pl.read_parquet(pinned_ops_wf / "output_raw"
                        / f"v_obj__{SOLVE}.parquet")["objective"][0]
    )
    assert obj == pytest.approx(PINNED_OBJ, rel=1e-6)


# ---------------------------------------------------------------------------
# Monotonicity — freeing operations only relaxes the feasible set
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_free_ops_objective_le_pinned(free_ops_wf, pinned_ops_wf) -> None:
    """§7: free_ops objective <= pinned_ops objective (non-strict —
    here strict, since the pinned window forces a costly common
    trajectory)."""
    free = float(
        pl.read_parquet(free_ops_wf / "output_raw"
                        / f"v_obj__{SOLVE}.parquet")["objective"][0]
    )
    pinned = float(
        pl.read_parquet(pinned_ops_wf / "output_raw"
                        / f"v_obj__{SOLVE}.parquet")["objective"][0]
    )
    assert free <= pinned
    assert free < pinned  # the fixture provably forces strict improvement
