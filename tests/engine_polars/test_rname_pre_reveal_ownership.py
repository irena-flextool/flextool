"""R-NAME — the reference forecast owns the real period names when n_real=0.

Corrected stochastic model: the stochastic fan is FORECASTS ONLY and the
``realized`` tag is an OPTIONAL 0-or-1 designation.  When a mid-horizon fan
carries NO realized branch (``n_real = 0``), no branch owns the un-suffixed
(real) period names, so a PRE-REVEAL real-named period — the shared trunk,
which has no suffixed forecast copy of its own — would get no
``solve_branch__time_branch`` (SBTB) owner and its branch-keyed data
(profiles, pdt params) would silently fall through to the zero fallback.

R-NAME binds such a pre-reveal trunk to the REFERENCE forecast (the
lexically-smallest forecast raw token — the same ``d_ref`` R-NA's
``db_pairs`` star uses) so the pre-reveal data binds instead of zeroing.
The guard is scoped to PRE-REVEAL periods only: the post-reveal weight-0
phantom self-row is left untouched (binding it would collide with the
recourse lineage chain grouping — see the R-NAME design).

Two witnesses, both of which fail if the guard is removed:

* ``test_rname_sbtb_binds_reference_forecast_prereveal`` — solver-free: drive
  ``create_stochastic_periods`` directly with a hand-built n_real=0
  mid-horizon fan and assert the pre-reveal SBTB row ``(p2035, high)`` appears
  (``high`` = lexical-min forecast) while the reveal period's phantom stays
  unbound.
* the solve-based tests on the ``stoch_two_period_midhorizon_nreal0`` fixture
  (DEFAULT ``stochastic_invest_method = none``) — the pre-reveal wind binds
  NONZERO (so ``base`` need only cover the real residual 60, not the inflated
  120) and the here-and-now invest is the sensible 60.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from types import SimpleNamespace

import pytest

from flextool.engine_polars._solve_state import ActiveTimeEntry
from flextool.engine_polars._stochastic import StochasticSolver


# ---------------------------------------------------------------------------
# Solver-free unit witness: the guard writes the pre-reveal SBTB owner.
# ---------------------------------------------------------------------------
def _ats(names_and_indices):
    return [ActiveTimeEntry(timestep=t, index=i, duration=1.0)
            for t, i in names_and_indices]


P1 = _ats([("t0001", 0), ("t0002", 1), ("t0003", 2)])
P2 = _ats([("t0004", 3), ("t0005", 4), ("t0006", 5)])

# Genuine forecasts-only fan revealed at p2040 (mid-horizon): both ``high``
# and ``low`` are ``realized: no`` with nonzero weight ⟹ n_real = 0.
# (period, branch, start_step, realized, weight)
BRANCH_INFO_NREAL0_MID = [
    ("p2040", "high", "t0004", "no", "1"),
    ("p2040", "low", "t0004", "no", "3"),
]


def _make_solver() -> StochasticSolver:
    state = SimpleNamespace(
        logger=logging.getLogger("test_rname_pre_reveal_ownership"),
        timeline=SimpleNamespace(stochastic_timesteps=defaultdict(list)),
    )
    return StochasticSolver(state)


def test_rname_sbtb_binds_reference_forecast_prereveal() -> None:
    """n_real=0 mid-horizon fan: the pre-reveal trunk's real name binds the
    reference forecast (lexical-min token), and the reveal period's weight-0
    phantom is left unbound."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}
    fix_storage: dict = {}

    (_pb, sbtb, _atl, _jumps, _fstl, _rtl, _bstl) = (
        solver.create_stochastic_periods(
            {"s": BRANCH_INFO_NREAL0_MID},
            ["s"],
            {"s": "s"},
            {"s": active},
            {"s": fix_storage},
            {"s": realized},
        )
    )
    rows = sbtb["s"]

    # d_ref is the lexically-smallest forecast raw token.
    d_ref = sorted({"high", "low"})[0]
    assert d_ref == "high"

    # The PRE-REVEAL trunk p2035 binds the reference forecast (the R-NAME
    # guard row).  Without the guard this row is absent and p2035's
    # branch-keyed data falls through to the zero fallback.
    assert ("p2035", d_ref) in rows, rows

    # Scope check: the fan copies carry their own data; the reveal period's
    # weight-0 phantom self-row is NEVER bound by the guard (that would trip
    # the recourse lineage chain precondition).
    assert ("p2040_high", "high") in rows, rows
    assert ("p2040_low", "low") in rows, rows
    assert ("p2040", "high") not in rows, rows
    assert ("p2040", "low") not in rows, rows


# ---------------------------------------------------------------------------
# Solve-based witness: the pre-reveal profile binds nonzero (not the zero
# fallback) and the here-and-now invest is the sensible residual-driven value.
# ---------------------------------------------------------------------------
SOLVE = "stoch_2p_mid"
PRE_REVEAL_PERIOD = "p2035"
# Reference forecast = lexically-smallest forecast raw token of the fan.
D_REF = "high"
# With the guard: p2035 wind binds 0.6·100 = 60 → the investable ``base``
# covers the real pre-reveal residual 120 − 60 = 60.  Without the guard the
# profile zeroes → residual 120 → invest forced to 120 (the broken artifact).
EXPECTED_PRE_REVEAL_INVEST = 60.0
EXPECTED_PRE_REVEAL_BASE_DISPATCH = 60.0


@pytest.fixture(scope="module")
def midhorizon_wf(scenario_workdir):
    import os

    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir(
            "midhorizon", db_fixture="stoch_two_period_midhorizon_nreal0")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def midhorizon_outputs(midhorizon_wf):
    """Re-solve from the workdir and return (optimal, v_flow, v_invest, sbtb)."""
    import polars as pl

    from polar_high import Problem

    from flextool.engine_polars import build_flextool, load_flextool

    data = load_flextool(midhorizon_wf)
    pb = Problem()
    build_flextool(pb, data)
    sol = pb.solve()

    v_flow = pl.read_parquet(
        midhorizon_wf / "output_raw" / f"v_flow__{SOLVE}.parquet")
    v_invest = pl.read_parquet(
        midhorizon_wf / "output_raw" / f"v_invest__{SOLVE}.parquet")
    sbtb = pl.read_csv(
        midhorizon_wf / "solve_data" / "solve_branch__time_branch.csv")
    return sol.optimal, v_flow, v_invest, sbtb


def _wind_col(df):
    return next(c for c in df.columns if "wind" in c)


def _base_col(df):
    return next(c for c in df.columns if c.startswith("('base'"))


@pytest.mark.solver
def test_midhorizon_solve_optimal(midhorizon_outputs):
    """The n_real=0 mid-horizon ``none`` solve reaches a strict optimum."""
    optimal, *_ = midhorizon_outputs
    assert optimal


@pytest.mark.solver
def test_midhorizon_prereveal_profile_binds_nonzero(midhorizon_outputs):
    """The pre-reveal trunk's branch-keyed wind profile binds NONZERO.

    Contrast: without R-NAME the profile falls through to the zero fallback
    (wind 0 at p2035) and ``base`` is forced to supply the full 120.  With the
    guard, free wind binds to its cap so ``base`` supplies only the real
    residual 60.
    """
    _opt, v_flow, _inv, _sbtb = midhorizon_outputs
    pre = v_flow.filter(v_flow["period"] == PRE_REVEAL_PERIOD)
    assert pre.height > 0

    wind = pre[_wind_col(pre)].to_list()
    base = pre[_base_col(pre)].to_list()
    # Wind bound nonzero (not the zero fallback) at every pre-reveal step.
    assert all(w > 0.0 for w in wind), wind
    # ``base`` covers only the real residual 60 (would be 120 under the
    # zero fallback) — a non-degenerate witness: base carries a positive
    # dispatch cost so free wind is strictly preferred up to its cap.
    assert base == pytest.approx(
        [EXPECTED_PRE_REVEAL_BASE_DISPATCH] * len(base)), base


@pytest.mark.solver
def test_midhorizon_prereveal_sbtb_owner_is_reference_forecast(
    midhorizon_outputs,
):
    """SBTB binds the pre-reveal trunk to the reference forecast (lexical-min
    token), and the reveal-period phantom stays unbound."""
    _opt, _flow, _inv, sbtb = midhorizon_outputs
    pairs = {(r["period"], r["branch"])
             for r in sbtb.iter_rows(named=True)}

    # The forecast fan tokens; the reference is the lexically-smallest.
    forecasts = {b for (p, b) in pairs if p.startswith("p2040_")}
    assert forecasts, pairs
    assert D_REF == sorted(forecasts)[0] == "high"

    # The pre-reveal trunk real name is owned by the reference forecast.
    assert (PRE_REVEAL_PERIOD, D_REF) in pairs, pairs
    # Pre-reveal scope: the reveal period's weight-0 phantom is NOT bound.
    assert ("p2040", "high") not in pairs, pairs
    assert ("p2040", "low") not in pairs, pairs


@pytest.mark.solver
def test_midhorizon_here_and_now_invest_is_sensible(midhorizon_outputs):
    """The here-and-now (pre-reveal) invest covers the REAL residual 60.

    Driven by this fixture's own data: pre-reveal residual 60 (demand 120 −
    wind 60) is the binding period for the single ``none`` invest var
    (cheaper ``invest_cost`` at p2035; both post-reveal residuals 20/40 are
    smaller), so the shared first-stage capacity is 60.  Without the guard the
    pre-reveal residual inflates to 120 and the invest is forced to 120.
    """
    _opt, _flow, v_invest, _sbtb = midhorizon_outputs
    pre = v_invest.filter(v_invest["period"] == PRE_REVEAL_PERIOD)
    assert pre.height == 1, pre
    assert pre["base"].to_list()[0] == pytest.approx(
        EXPECTED_PRE_REVEAL_INVEST), pre
