"""Slice G — configurable operational non-anticipativity window.

Design: ``specs/sliceG_configurable_nonanticipativity_design.md`` (§4
byte-parity, §7 hand-checkable test).

The new ``solve.shared_operation_periods`` parameter (v71) decouples the
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
``shared_operation_periods`` value.

Hand-verified reference numbers (deterministic LP; see the builder):

* ``pinned_ops`` (unset): storage NA fires, the set CSV holds the whole
  6-step horizon, and ``v_state[resv]`` is IDENTICAL across the two
  forecast copies up / low — the branches cannot follow their opposite
  surplus/deficit timing.  Objective 2_037_600.
* ``free_ops`` (``[]``): zero NA rows, header-only set CSV, and
  ``v_state[resv]`` DIVERGES (up rises above 0.5, low falls below).
  Shared v_invest 2 and objective 428_076 < pinned (hand derivation at
  ``FREE_INVEST``).

R3 forecasts-only note: the fixture declares a realized designation
``mid`` (realized:yes) alongside the two forecasts ``up`` / ``low``.
Under the corrected model ``mid`` is PULLED from the weighted fan and the
two forecasts renormalise to {up:0.5, low:0.5}; the objectives above are
the driven optima under that weighting.  The period labels p2035 / p2040
have no dispatch, and the designation adds a realization solve after the
stochastic solve; the tests here read the stochastic solve.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

from tests._support.load import load_scenario
from tests.engine._unit_db import is_realization

SOLVE = "stoch_2p_free"

# Deterministic LP reference objectives (see the fixture builder).
PINNED_OBJ = 2_037_600.0
# free_ops hand derivation.  Per forecast the wind gives 60 MWh against 72 MWh demand, so ``base`` must
# supply 12 MWh over the 6 steps; the storage ends where it starts (reference 0.5) and is not limiting
# (100 MW, 1000 MWh), so ``base`` runs flat at 12 / 6 = 2 MW and the shared capacity is 2 MW:
#   up : steps 1-3 wind 20 + base 2 - demand 12 = charge 10; steps 4-6 wind 0 + base 2 -> discharge 10.
#   low: the mirror (discharge 10 early, charge 10 late).  Each forecast discharges 30 MWh.
# Operating cost per forecast = (5 · 12 + 0.01 · 30) CUR × 8760 / 3 (3 h periods, 1 year each) = 60.3 · 2920,
# weighted 0.5 + 0.5 -> 176 076.  Investment: 2 MW × the LP's invest coefficient (read from FlexData).
FREE_INVEST = 2.0
FREE_OPS_COST = 60.3 * 8760.0 / 3.0


# ---------------------------------------------------------------------------
# Cascade-run fixtures (full cascade + snapshot, AUTOSCALE_STRICT on)
# ---------------------------------------------------------------------------


def _strict_run(scenario_run, name):
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_run(name, db_fixture="stoch_two_period_free_ops")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def free_ops_run(scenario_run):
    return _strict_run(scenario_run, "free_ops")


@pytest.fixture(scope="module")
def pinned_ops_run(scenario_run):
    return _strict_run(scenario_run, "pinned_ops")


@pytest.fixture(scope="module")
def free_ops_wf(free_ops_run):
    """The cascade's work folder (emitted CSVs + ``output_raw``)."""
    return free_ops_run.work_folder


@pytest.fixture(scope="module")
def pinned_ops_wf(pinned_ops_run):
    """The cascade's work folder (emitted CSVs + ``output_raw``)."""
    return pinned_ops_run.work_folder


def _resolve(run):
    """In-process build+solve of the run's stochastic solve (the last
    sub-solve before the realization solve the ``mid`` designation adds),
    loaded through the production provider path; returns ``(data, sol, pb)``."""
    from polar_high import Problem
    from flextool.engine import build_flextool

    stochastic = [s.solve_name for s in run.steps if not is_realization(s.solve_name)]
    data = load_scenario(run, step=stochastic[-1])
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()
    assert sol.optimal
    return data, sol, pb


@pytest.fixture(scope="module")
def free_ops_resolved(free_ops_run):
    return _resolve(free_ops_run)


@pytest.fixture(scope="module")
def pinned_ops_resolved(pinned_ops_run):
    return _resolve(pinned_ops_run)


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
# free_ops (shared_operation_periods = []) — operations free from t0
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
    # Only the period label p2035 (under ``none`` it keeps the shared
    # ``v_invest``) — never a branch copy (p2035_up / p2035_low), which
    # would signal a fanned invest axis.
    assert keys == {"p2035"}
    val = {str(r["d"]): round(float(r["value"]), 3)
           for r in vi.iter_rows(named=True)}
    assert val == pytest.approx({"p2035": FREE_INVEST})


@pytest.mark.solver
def test_free_ops_v_state_diverges_across_branches(free_ops_resolved) -> None:
    """§7 (F6), R3 re-grounded: with operations freed each FORECAST reaches
    its OWN storage optimum, so v_state[resv] is strictly distinct across
    the two forecasts ``up`` / ``low`` — deterministic given the
    branch-asymmetric wind timing.  ``up`` (surplus early) charges above
    0.5; ``low`` (deficit early) discharges below.  The realized
    designation ``mid`` is a weight-0 phantom under forecasts-only (its
    real-named trunk carries no probability mass), so its storage
    trajectory is an uncosted free variable (driven to 0) and is NOT the
    idle-at-0.5 member it was when ``mid`` still carried fan weight; the
    probability-relevant divergence is between the two forecasts.  The
    period label p2035 has no dispatch, so no storage trajectory."""
    _data, sol, _pb = free_ops_resolved
    traj = _v_state_by_branch(sol, "p2035")
    assert set(traj) == {"p2035_up", "p2035_low"}
    up = traj["p2035_up"]
    low = traj["p2035_low"]
    # up strictly above, low strictly below — at every step, and the two
    # FORECASTS are distinct (a UNIQUE per-forecast optimum).
    assert all(u > 0.5 for u in up), up
    assert all(lo < 0.5 for lo in low), low
    assert up != low


@pytest.mark.solver
def test_free_ops_objective(free_ops_wf, free_ops_resolved) -> None:
    """§7: the free-ops objective is the deterministic LP optimum
    (per-branch dispatch under the shared capacity, hand derivation at
    ``FREE_INVEST``)."""
    data, _sol, _pb = free_ops_resolved
    annual = data.ed_entity_annual_discounted.frame
    inv_cost = float(annual.filter((pl.col("e").cast(pl.Utf8) == "base")
                                   & (pl.col("d").cast(pl.Utf8) == "p2035"))["value"].item())
    obj = float(
        pl.read_parquet(free_ops_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(FREE_INVEST * inv_cost + FREE_OPS_COST, rel=1e-6)


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
    the whole horizon.

    Under the forecast-to-forecast re-ground the cohort is a per-group star
    keyed on the reference forecast (the lexicographically smallest of the
    two forecasts ``up`` / ``low`` -> ``p2035_low`` / ``p2040_low``), so each
    group contributes ONE sibling pair (``low`` -> ``up``) over its 3 tied
    steps: 6 rows total (3 (d,t) × 2 period groups × 1 sibling), down from
    the pre-re-ground 12 (which anchored both forecasts to the trunk)."""
    import polars as pl

    _data, _sol, pb = pinned_ops_resolved
    names = pb.cstr_names()
    assert "non_anticipativity_storage_use" in names
    over = pb.cstrs_named("non_anticipativity_storage_use")[0].over
    assert over.height == 6
    # The hubs are the reference forecasts; the trunk base periods are tied
    # to nothing.
    ov = over.with_columns(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8))
    assert set(ov["d"].unique().to_list()) == {"p2035_low", "p2040_low"}
    assert set(ov["b"].unique().to_list()) == {"p2035_up", "p2040_up"}
    assert not ({"p2035", "p2040"} & set(ov["d"].unique().to_list()))
    assert not ({"p2035", "p2040"} & set(ov["b"].unique().to_list()))


@pytest.mark.solver
def test_pinned_ops_v_state_equal_across_branches(pinned_ops_resolved) -> None:
    """§7 (re-grounded): with the legacy window the storage NA family ties
    the FORECASTS' net-charge equal, so v_state[resv] is IDENTICAL across the
    forecast copies ``up`` / ``low`` on every constrained (d,t).

    Forecast-to-forecast: the period label (the bare-period key) has no
    dispatch; only the two forecast copies carry a trajectory."""
    _data, sol, _pb = pinned_ops_resolved
    for prefix in ("p2035", "p2040"):
        traj = _v_state_by_branch(sol, prefix)
        assert set(traj) == {f"{prefix}_up", f"{prefix}_low"}, traj
        assert traj[f"{prefix}_up"] == traj[f"{prefix}_low"], (prefix, traj)


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
