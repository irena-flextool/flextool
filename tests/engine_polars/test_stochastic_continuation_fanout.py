"""Mid-horizon stochastic continuation fan-out — e2e tests (Option B,
"realized-not-past-reveal").

Fixture ``tests/fixtures/stoch_mid_horizon.json`` (built from JSON via
``json_to_db`` — CLAUDE.md invariant 3): one 9-step timeline split into
THREE 3-step periods — ``p2030`` (t0001-t0003, COMMITTED), ``p2035``
(t0004-t0006, REVEAL) and ``p2040`` (t0007-t0009, CONTINUATION).  Demand
100 / 100 / 110 MW, a free wind unit capped by a branch-varying profile
(stochastic group membership via ``group__unit`` only), and a
50 EUR/MWh thermal backstop.  Branches ``rlz`` (realized, weight 1.0),
``low`` (weight 3.0) and ``high`` (weight 1.0) are declared at
``p2035``/t0004 — a MID-HORIZON reveal with ``p2030`` as the committed
pre-reveal region.

Option B invariants under test:

* The realized trunk STOPS at the reveal.  Committed ``p2030`` + reveal
  ``p2035`` are real-named; ``p2040`` carries ONLY the synthetic
  forecast copies ``p2040_low`` / ``p2040_high`` (no real-named trunk,
  no ``(p2040, p2040)`` self-row, ``p2040`` not in ``realized_dispatch``).
* Weights: committed ``p2030`` = 1.0 (cohort-of-one); the reveal cohort
  {p2035, p2035_low, p2035_high} sums to 1 (0.2 / 0.6 / 0.2); the
  post-reveal cohort {p2040_low, p2040_high} renormalizes to sum 1
  (0.75 / 0.25) with NO realized term.
* Realized (output_horizon off) outputs cover the committed region only
  (p2030 + p2035); post-reveal realized (d, t) cells are absent
  (empty/NaN, not 0).  Forecast operational results are still emitted
  (output_horizon on) for debugging.
"""
from __future__ import annotations

import os

import polars as pl
import pytest

MID_OBJECTIVE = 88_914_000.0
DET_OBJECTIVE = 61_320_000.0
WIND_EXISTING = 100.0
THERMAL_EXISTING = 200.0

WIND_COL = "('wind', 'wind', 'city')"
THERMAL_COL = "('thermal', 'thermal', 'city')"

P0_STEPS = ["t0001", "t0002", "t0003"]   # committed p2030
P1_STEPS = ["t0004", "t0005", "t0006"]   # reveal    p2035
P2_STEPS = ["t0007", "t0008", "t0009"]   # continue  p2040

# period__branch (design): committed self-row; reveal self + fan;
# continuation FAN ONLY (no (p2040, p2040) self-row — trunk stops at
# the reveal).  Fan rows follow DB branch order rlz / low / high.
EXPECTED_PERIOD_BRANCH = [
    ("p2030", "p2030"),
    ("p2035", "p2035"),
    ("p2035", "p2035_rlz"),
    ("p2035", "p2035_low"),
    ("p2035", "p2035_high"),
    ("p2040", "p2040_rlz"),
    ("p2040", "p2040_low"),
    ("p2040", "p2040_high"),
]

# Per-cohort normalized weights (Stage C).
EXPECTED_PD_BRANCH_WEIGHT = {
    "p2030": 1.0,        # committed — cohort-of-one
    "p2035": 0.2,        # reveal cohort {rlz, low, high} = 1/5
    "p2035_low": 0.6,
    "p2035_high": 0.2,
    "p2040_low": 0.75,   # post-reveal cohort {low, high} = 3/4
    "p2040_high": 0.25,  # 1/4
}

# Realized (committed + reveal) wind / thermal MW, hand-checked.
EXPECTED_REALIZED_MW = {
    "p2030": (60.0, 40.0),
    "p2035": (60.0, 40.0),
}
# Forecast (output_horizon on) wind / thermal MW.
EXPECTED_FORECAST_MW = {
    "p2030": (60.0, 40.0),
    "p2035": (60.0, 40.0),
    "p2035_low": (20.0, 80.0),
    "p2035_high": (40.0, 60.0),
    "p2040_low": (10.0, 100.0),
    "p2040_high": (30.0, 80.0),
}


@pytest.fixture(scope="module")
def mid_wf(scenario_workdir):
    """``mid`` scenario (output_horizon off — realized-only outputs),
    run under ``FLEXTOOL_AUTOSCALE_STRICT=1`` so an autoscale-registry
    gap would be loud."""
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        return scenario_workdir("mid", db_fixture="stoch_mid_horizon")
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev


@pytest.fixture(scope="module")
def mid_horizon_wf(scenario_workdir):
    return scenario_workdir("mid_horizon", db_fixture="stoch_mid_horizon")


@pytest.fixture(scope="module")
def det_wf(scenario_workdir):
    return scenario_workdir("det", db_fixture="stoch_mid_horizon")


def _objective(wf) -> float:
    return float(
        pl.read_parquet(wf / "output_raw" / "v_obj__stoch_mid.parquet")
        ["objective"][0]
    )


def _v_flow(wf) -> pl.DataFrame:
    return pl.read_parquet(wf / "output_raw" / "v_flow__stoch_mid.parquet")


def _per_period_mw(wf) -> dict[str, tuple[float, float]]:
    flows = _v_flow(wf)
    out: dict[str, tuple[float, float]] = {}
    for p in flows["period"].unique().to_list():
        row = flows.filter(pl.col("period") == p).row(0, named=True)
        out[p] = (row[WIND_COL] * WIND_EXISTING,
                  row[THERMAL_COL] * THERMAL_EXISTING)
    return out


def _csv_rows(wf, name: str) -> list[tuple]:
    df = pl.read_csv(wf / "solve_data" / name)
    return [(r[0], r[1]) for r in df.rows()]


# ---------------------------------------------------------------------------
# ``mid`` scenario (output_horizon off — realized-only outputs)
# ---------------------------------------------------------------------------


def test_mid_solves_optimal(mid_wf) -> None:
    """The mid-horizon stochastic LP solves to the expected optimum."""
    assert _objective(mid_wf) == pytest.approx(MID_OBJECTIVE, rel=1e-6)


def test_mid_realized_stops_at_reveal(mid_wf) -> None:
    """Realized output covers the COMMITTED region only (p2030 + p2035);
    post-reveal realized (d, t) cells are ABSENT (empty/NaN, not 0)."""
    flows = _v_flow(mid_wf)
    assert sorted(set(flows["period"].to_list())) == ["p2030", "p2035"], (
        "post-reveal / branch periods leaked into realized outputs"
    )
    assert flows.height == 6
    # realized_dispatch is the committed + reveal steps only; p2040 absent.
    rd = _csv_rows(mid_wf, "realized_dispatch.csv")
    assert rd == (
        [("p2030", t) for t in P0_STEPS] + [("p2035", t) for t in P1_STEPS]
    )
    assert all(d != "p2040" for d, _t in rd)


def test_mid_committed_and_reveal_realized_flows(mid_wf) -> None:
    """Committed + reveal realized flows match the hand calc."""
    mw = _per_period_mw(mid_wf)
    assert set(mw) == set(EXPECTED_REALIZED_MW)
    for p, (wind, thermal) in EXPECTED_REALIZED_MW.items():
        assert mw[p][0] == pytest.approx(wind, abs=1e-6), p
        assert mw[p][1] == pytest.approx(thermal, abs=1e-6), p


def test_mid_branch_weights(mid_wf) -> None:
    """Stage C: committed realized = 1.0; the post-reveal cohort
    {p2040_low, p2040_high} renormalizes to sum 1 with NO realized
    term; the reveal cohort sums to 1."""
    pdw = pl.read_csv(mid_wf / "solve_data" / "pd_branch_weight.csv")
    got = {r[0]: float(r[1]) for r in pdw.rows()}
    assert set(got) == set(EXPECTED_PD_BRANCH_WEIGHT)
    for p, w in EXPECTED_PD_BRANCH_WEIGHT.items():
        assert got[p] == pytest.approx(w, abs=1e-12), p
    # committed cohort-of-one.
    assert got["p2030"] == pytest.approx(1.0, abs=1e-12)
    # post-reveal forecasts renormalize to 1 (no realized term — there is
    # no real-named p2040 weight at all).
    assert "p2040" not in got
    assert got["p2040_low"] + got["p2040_high"] == pytest.approx(
        1.0, abs=1e-12
    )
    # reveal cohort (realized + forecasts) sums to 1.
    assert (got["p2035"] + got["p2035_low"] + got["p2035_high"]
            == pytest.approx(1.0, abs=1e-12))
    # Raw per-branch input weights (pre-normalization).
    sbw = pl.read_csv(mid_wf / "solve_data" / "solve_branch_weight.csv")
    got_input = {r[0]: float(r[1]) for r in sbw.rows()}
    assert got_input == {
        "p2030": 1.0, "p2035": 1.0,
        "p2035_low": 3.0, "p2035_high": 1.0,
        "p2040_low": 3.0, "p2040_high": 1.0,
    }


def test_mid_na_and_period_branch(mid_wf) -> None:
    """dt_non_anticipativity covers the committed region only;
    period__branch carries the committed self-row, the reveal self + fan,
    and the continuation FAN ONLY (no (p2040, p2040) self-row)."""
    dtna = _csv_rows(mid_wf, "dt_non_anticipativity_set.csv")
    assert dtna == (
        [("p2030", t) for t in P0_STEPS] + [("p2035", t) for t in P1_STEPS]
    )
    pb = _csv_rows(mid_wf, "period__branch.csv")
    assert pb == EXPECTED_PERIOD_BRANCH
    assert ("p2040", "p2040") not in pb


# ---------------------------------------------------------------------------
# ``mid_horizon`` scenario (output_horizon on — forecast rows visible)
# ---------------------------------------------------------------------------


def test_mid_horizon_objective_identical(mid_horizon_wf) -> None:
    """output_horizon only widens the output gate, not the LP."""
    assert _objective(mid_horizon_wf) == pytest.approx(
        MID_OBJECTIVE, rel=1e-6
    )


def test_mid_horizon_forecast_results_emitted(mid_horizon_wf) -> None:
    """Stochastic/forecast operational results are STILL emitted
    (debugging): the reveal and continuation forecast copies appear with
    their branch-specific flows."""
    flows = _v_flow(mid_horizon_wf)
    assert flows.height == 18
    mw = _per_period_mw(mid_horizon_wf)
    assert set(mw) == set(EXPECTED_FORECAST_MW)
    assert {"p2035_low", "p2035_high", "p2040_low", "p2040_high"} <= set(mw)
    for p, (wind, thermal) in EXPECTED_FORECAST_MW.items():
        assert mw[p][0] == pytest.approx(wind, abs=1e-6), p
        assert mw[p][1] == pytest.approx(thermal, abs=1e-6), p


# ---------------------------------------------------------------------------
# ``det`` scenario (deterministic control)
# ---------------------------------------------------------------------------


def test_det_control(det_wf) -> None:
    """Deterministic control: full horizon, no branch periods anywhere."""
    assert _objective(det_wf) == pytest.approx(DET_OBJECTIVE, rel=1e-6)
    flows = _v_flow(det_wf)
    assert sorted(set(flows["period"].to_list())) == [
        "p2030", "p2035", "p2040",
    ]
    assert flows.height == 9
