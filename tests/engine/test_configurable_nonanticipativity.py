"""Slice G — the operations of a stochastic investment solve branch at the reveal.

Design: ``specs/sliceG_configurable_nonanticipativity_design.md`` (§7
hand-checkable test); data version 84 (``invest_solve_branching_method``):
an investment solve ties no operations of its forecasts (the operational
non-anticipativity window is empty), whatever its settings.  Before v84 the
unset ``shared_operation_periods`` tied them over the committed window; that
setting is gone (a dispatch solve still ties its forecasts over its committed
window: ``test_nonanticipativity_nreal0.py``).

Fixture ``stoch_two_period_free_ops.json`` (built via
``build_stoch_two_period_free_ops.py``): two 3-step periods p2035 / p2040,
three branches mid / up / low declared at t0001 (t0-branching), a single
shared investable ``base`` unit (``operations_only`` -> ONE ``v_invest``),
and a stochastic-group storage node ``resv`` fed by branch-asymmetric wind
timing so each branch has a UNIQUE cost-optimal storage trajectory (design
F6).

Hand-verified reference numbers (deterministic LP; derivation at
``FREE_INVEST``): zero NA rows, header-only set CSV, ``v_state[resv]``
DIVERGES (up rises above 0.5, low falls below), shared v_invest 2 and
objective 256_076.

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

# Investment cost coefficient of ``base`` built in p2035, CUR/MW (ed_entity_annual_discounted).
# Annuity = invest_cost · 1000 · r / (1 − (1 + r)^−life) / (1 + r) = 20 · 1000 · 0.05 / (1 − 1.05^−1) / 1.05
# = 20 000 (one payment at the start of the year, the default discount_offset_investment 0: the cost itself).
# lifetime_method reinvest_automatic: the capacity serves every later calendar period of the horizon, p2035 and
# p2040 (1 year each, every investment inflation factor 1.0), so the sum is 2 × 20 000 = 40 000.  Each calendar
# period counts once: the period labels p2035 / p2040 and their forecast copies p20xx_up / p20xx_low are the
# same calendar years, not four more years of service.
INVEST_COEFF = 2 * 20_000.0
# Operating cost scale: 3-h periods of 1 year each -> 8760 / 3 per CUR of per-step cost; forecasts 0.5 / 0.5.
OPS_SCALE = 8760.0 / 3.0
# free_ops hand derivation.  Per forecast the wind gives 60 MWh against 72 MWh demand, so ``base`` must
# supply 12 MWh over the 6 steps; the storage ends where it starts (reference 0.5) and is not limiting
# (100 MW, 1000 MWh), so ``base`` runs flat at 12 / 6 = 2 MW and the shared capacity is 2 MW:
#   up : steps 1-3 wind 20 + base 2 - demand 12 = charge 10; steps 4-6 wind 0 + base 2 -> discharge 10.
#   low: the mirror (discharge 10 early, charge 10 late).  Each forecast discharges 30 MWh.
# Operating cost per forecast = (5 · 12 + 0.01 · 30) CUR × 8760 / 3 (3 h periods, 1 year each) = 60.3 · 2920,
# weighted 0.5 + 0.5 -> 176 076.  Investment: 2 MW × INVEST_COEFF = 80 000.  Objective 256 076.
FREE_INVEST = 2.0
FREE_OPS_COST = 60.3 * OPS_SCALE


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
def free_ops_wf(free_ops_run):
    """The cascade's work folder (emitted CSVs + ``output_raw``)."""
    return free_ops_run.work_folder


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
# free_ops — operations free from the reveal (t0)
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
    """§7: with operations_only investment stays realized-only
    (R-O6) -> exactly ONE shared v_invest for the ``base`` unit (no
    per-branch fan)."""
    _data, sol, _pb = free_ops_resolved
    vi = sol.value("v_invest_p")
    keys = {str(r["d"]) for r in vi.iter_rows(named=True)}
    # Only the period label p2035 (under ``operations_only`` it keeps the shared
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
def test_free_ops_objective(free_ops_wf) -> None:
    """§7: the free-ops objective is the deterministic LP optimum
    (per-branch dispatch under the shared capacity, hand derivation at
    ``FREE_INVEST``): 2 × 40 000 + 176 076 = 256 076."""
    obj = float(
        pl.read_parquet(free_ops_wf / "output_raw" / f"v_obj__{SOLVE}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(FREE_INVEST * INVEST_COEFF + FREE_OPS_COST, rel=1e-6)


@pytest.mark.solver
def test_here_and_now_invest_annuity_counts_each_calendar_period_once(free_ops_resolved) -> None:
    """Under ``operations_only`` the one here-and-now ``v_invest`` (on the period label p2035)
    pays its annuity once per calendar period it serves (p2035, p2040), not once per period label and forecast
    copy in ``period_in_use``: the coefficient is 40 000 (hand derivation at ``INVEST_COEFF``), not 6 × 20 000."""
    data, _sol, _pb = free_ops_resolved
    annual = data.ed_entity_annual_discounted.frame
    coeff = {str(r["d"]): float(r["value"])
             for r in annual.filter(pl.col("e").cast(pl.Utf8) == "base").iter_rows(named=True)}
    assert coeff == pytest.approx({"p2035": INVEST_COEFF}, rel=1e-12)


# ---------------------------------------------------------------------------
# Units online are not tied in an investment solve
# ---------------------------------------------------------------------------


def _online_tie_rows(tmp_path_factory, alt: str, base_chain: list[str]) -> int:
    """Rows of non_anticipativity_online_linear in the stochastic solve of the fixture with ``base`` given unit
    commitment (startup_method linear, min_load 0.3) in alternative ``alt`` on top of ``base_chain``."""
    from pathlib import Path

    from tests.engine._unit_db import build, cstr_over, last_stochastic, make_db, run, uc

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "stoch_two_period_free_ops.json"
    url, scenario = make_db(tmp_path_factory, alt=alt, values=uc("base", 0.3),
                            base_chain=base_chain, fixture_json=fixture)
    step = last_stochastic(run(tmp_path_factory, url, scenario))
    over = cstr_over(build(step.flex_data), "non_anticipativity_online_linear")
    return 0 if over is None else over.height


@pytest.mark.solver
def test_free_ops_leaves_units_online_untied(tmp_path_factory) -> None:
    """An investment solve frees the operations of its forecasts from the reveal (here the first timestep), units
    online included: with nothing tied there is no non_anticipativity_online row, however many online steps the
    forecasts have."""
    assert _online_tie_rows(tmp_path_factory, "online_free", ["init", "base", "stoch"]) == 0
