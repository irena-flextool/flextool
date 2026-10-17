"""W4c — the auto-spawned realized-dispatch sibling reads each stochastic
parameter PROJECTED onto the REALIZED branch.

W4b proved the engine auto-spawns a deterministic realized-dispatch
sibling (``<parent>__realized_dispatch``) that consumes the parent's
committed capacity and owns the realized window — but only when the
realized operational data was DETERMINISTIC (a 2d profile).  A
branch-distinct stochastic 3d_map ``(branch, time_start, time) -> value``
was DROPPED by the sibling (no branch scaffolding -> zero-fallback ->
wind 0 -> VOLL), so forecasts could not diverge from the realized data.

W4c closes that gap: a post-pass in ``_native_run_model`` projects the
parent's REALIZED-branch ``solve_branch__time_branch`` rows onto the
sibling (keyed off the explicit ``SolveConfig.realized_dispatch_parent``
map, recorded by ``spawn_realized_dispatch_sibling``).  The per-solve
folds then select the realized branch unchanged, and the sibling stays
NON-genuinely-stochastic (zero branch LP variables).

This module pins:

  * the divergent-forecast e2e: with wind profile rlz=0.50 / high=0.80 /
    low=0.10, the sibling dispatches the REALIZED rlz=0.50 (wind 50 MW,
    peaker 50 MW/step, objective 4 380 000) — NOT a forecast (80/20 or
    10/90) and NOT the collapsed-to-0 VOLL fallback;
  * last-wins: the parent's realized p2035 window is emptied and the
    sibling owns p2035/t0001-t0003;
  * the spawn primitive records ``realized_dispatch_parent[sibling]``;
  * the projection helper carries ONE realized row per realized period
    for a MULTI-period realized window (continuation-loop coverage).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
DIVERGENT_FIXTURE = (
    FLEXTOOL_ROOT / "tests" / "fixtures" / "w4c_divergent_dispatch.json"
)

SCENARIO = "w8_realized_dispatch"
PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"
REALIZED_STEPS = {"t0001", "t0002", "t0003"}

# Hand-calc constants (realized rlz branch: wind 0.50 -> 50 MW, residual
# 50 MW peaker, annualized over 8760 h at 10 CUR/MWh).
EXISTING_WIND_MW = 100.0
RLZ_PROFILE = 0.50
WIND_MW = RLZ_PROFILE * EXISTING_WIND_MW          # 50 MW
PEAKER_MW = 100.0 - WIND_MW                        # 50 MW residual
HOURS_PER_YEAR = 8760.0
PEAKER_VAR_COST = 10.0
DISPATCH_OBJECTIVE = PEAKER_MW * PEAKER_VAR_COST * HOURS_PER_YEAR  # 4 380 000

# Negative-control peaker levels the sibling must NOT land on.
PEAKER_HIGH_MW = 100.0 - 0.80 * EXISTING_WIND_MW   # 20 MW  (high forecast)
PEAKER_LOW_MW = 100.0 - 0.10 * EXISTING_WIND_MW    # 90 MW  (low forecast)
PEAKER_COLLAPSED_MW = 100.0                         # wind 0 -> all peaker


def _build_db(fixture_json: Path, tmp_path: Path) -> str:
    """Build-from-JSON + migrate (CLAUDE.md invariant 3; never read a
    checked-in .sqlite)."""
    if not fixture_json.exists():
        pytest.skip(f"fixture JSON not present: {fixture_json}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{fixture_json.stem}.sqlite"
    url = json_to_db(fixture_json, db_path)
    migrate_database(url)
    return url


def _run_chain(fixture_json: Path, tmp_path: Path):
    """Run the single-solve chain once under ``FLEXTOOL_AUTOSCALE_STRICT=1``
    with ``warm=False`` + ``keep_solutions=True``; return
    ``(steps, work_folder)``."""
    from flextool.engine_polars import run_chain_from_db

    url = _build_db(fixture_json, tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            url, SCENARIO, work_folder=work_folder,
            warm=False, keep_solutions=True,
        )
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    assert steps, "run_chain_from_db produced no orchestration steps"
    return steps, work_folder


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("w4c_divergent_dispatch")
    return _run_chain(DIVERGENT_FIXTURE, tmp_path)


def _step(steps, needle):
    key = next((k for k in steps if k == needle), None) or next(
        (k for k in steps if needle in k), None
    )
    assert key is not None, f"no {needle!r} step in {list(steps)}"
    return steps[key]


# ===========================================================================
# (1) Divergent-forecast e2e — the sibling reads the REALIZED branch
# ===========================================================================


@pytest.mark.solver
def test_sibling_reads_realized_branch_not_forecast(chain) -> None:
    """With branch-distinct wind (rlz 0.50 / high 0.80 / low 0.10) the
    auto sibling dispatches the REALIZED rlz=0.50: wind 50 MW, peaker
    50 MW/step.  NOT a forecast (80/20 or 10/90) and NOT the collapsed
    wind-0 VOLL fallback (wind > 0)."""
    steps, _wf = chain
    dispatch = _step(steps, SIBLING)
    assert dispatch.solution is not None and dispatch.optimal

    # Deterministic sibling — zero branch LP variables, no v_invest.
    disp_vars = set(getattr(dispatch.solution, "_vars", {}).keys())
    assert not any("v_invest" in v for v in disp_vars), (
        f"sibling built v_invest variables {sorted(disp_vars)}"
    )

    flows = dispatch.solution.value("v_flow")
    peaker = flows.filter(pl.col("source") == "peaker")
    wind = flows.filter(pl.col("source") == "wind")
    assert peaker.height == 3
    assert wind.height == 3

    peaker_vals = peaker["value"].to_list()
    wind_vals = wind["value"].to_list()

    # Realized rlz: peaker 50 MW/step; wind v_flow is the profile fraction
    # (physical wind = 0.50 * 100 existing = 50 MW).
    assert all(v == pytest.approx(PEAKER_MW) for v in peaker_vals), (
        f"sibling peaker != {PEAKER_MW} MW/step (realized rlz=0.50): "
        f"{peaker_vals}"
    )
    assert all(v == pytest.approx(RLZ_PROFILE) for v in wind_vals), (
        f"sibling wind fraction != {RLZ_PROFILE} (realized rlz): {wind_vals}"
    )

    # NEGATIVE CONTROLS — not a forecast, not the collapsed-to-0 fallback.
    for v in peaker_vals:
        assert v != pytest.approx(PEAKER_HIGH_MW), "sibling read the HIGH forecast"
        assert v != pytest.approx(PEAKER_LOW_MW), "sibling read the LOW forecast"
        assert v != pytest.approx(PEAKER_COLLAPSED_MW), (
            "sibling collapsed to wind=0 (3d_map dropped) -> all-peaker VOLL"
        )
    for v in wind_vals:
        assert v > 0.0, "wind collapsed to 0 — the realized 3d_map was dropped"


@pytest.mark.solver
def test_sibling_objective_hand_calc(chain) -> None:
    """Sibling objective = 50 MW * 8760 h * 10 CUR/MWh = 4 380 000 at
    rel=1e-9.  This single value excludes every negative control (any
    forecast level or a VOLL collapse would move it well outside 1e-9)."""
    steps, wf = chain
    _step(steps, SIBLING)  # ensure the sibling step exists
    obj = float(
        pl.read_parquet(wf / "output_raw" / f"v_obj__{SIBLING}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(DISPATCH_OBJECTIVE, rel=1e-9)
    # Sanity: nowhere near a VOLL collapse (which would add ~50 MW unserved
    # * 8760 h * 10000 CUR/MWh on top of fuel).
    assert obj < DISPATCH_OBJECTIVE * 100.0


@pytest.mark.solver
def test_parent_forecasts_genuinely_diverge(chain) -> None:
    """The parent's FORECAST branches genuinely diverge: peaker hits 20
    (high) and 90 (low) MW — proving the forecast data are DISTINCT.

    W4d-A note: under ``separate_dispatch`` the parent hedge's REALIZED
    operational branch now carries OPERATIONAL weight 0 (its operations are
    owned by the W4c realized-dispatch sibling).  A weight-0 branch is
    DEGENERATE in the LP objective — the realized peaker is free in [0, 90]
    and the solver lands it anywhere (empirically 0 here, not the old 50) —
    so we no longer assert the realized 50 MW level on the parent.  The
    realized dispatch is pinned on the SIBLING instead
    (``test_sibling_reads_realized_branch_not_forecast``).  We assert ONLY
    the forecast divergence {20, 90}."""
    steps, _wf = chain
    parent = _step(steps, PARENT)
    assert parent.solution is not None and parent.optimal
    flows = parent.solution.value("v_flow")
    peaker_vals = set(
        round(v, 6)
        for v in flows.filter(pl.col("source") == "peaker")["value"].to_list()
    )
    for expected in (PEAKER_HIGH_MW, PEAKER_LOW_MW):
        assert any(v == pytest.approx(expected) for v in peaker_vals), (
            f"parent peaker levels {sorted(peaker_vals)} missing {expected}"
        )


# W4d-A hand-calc: the parent hedge objective is the FORECAST-only
# expectation (realized operational weight 0) + the invest annuity.
#   high forecast: 20 MW * 8760 h * 10 CUR/MWh = 1 752 000
#   low  forecast: 90 MW * 8760 h * 10 CUR/MWh = 7 884 000
#   expectation  : 0.5 * 1 752 000 + 0.5 * 7 884 000 = 4 818 000
#   + invest annuity                                 =   283 500
#   = 5 101 500
PARENT_FORECAST_ONLY_OBJECTIVE = 5_101_500.0
# Negative control: the OLD value when the realized branch was kept in the
# normalisation denominator (realized self-weight 1.0 in the cohort).
PARENT_REALIZED_IN_DENOM_OBJECTIVE = 4_851_214.2857


@pytest.mark.solver
def test_parent_objective_is_forecast_only_expectation(chain) -> None:
    """W4d-A: under ``separate_dispatch`` the parent hedge's OPERATIONAL
    branch weights are de-scenario'd — forecasts normalise among themselves
    (realized excluded from the denominator) and the realized branch's own
    operational weight → 0.  The parent objective is therefore the
    FORECAST-ONLY expectation (0.5/0.5 of the 20 MW / 90 MW forecasts =
    4 818 000) plus the invest annuity (283 500) = 5 101 500.

    Negative control: 4 851 214.2857 — the OLD objective with the realized
    branch kept in the normalisation denominator.  INVEST weight
    (``pd_branch_weight``) is UNTOUCHED, so the annuity term is unchanged."""
    steps, wf = chain
    _step(steps, PARENT)  # ensure the parent step exists
    obj = float(
        pl.read_parquet(wf / "output_raw" / f"v_obj__{PARENT}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(PARENT_FORECAST_ONLY_OBJECTIVE, rel=1e-9)
    assert obj != pytest.approx(PARENT_REALIZED_IN_DENOM_OBJECTIVE, rel=1e-6)


@pytest.mark.solver
def test_realized_window_owned_by_sibling(chain) -> None:
    """Last-wins: the parent's realized p2035 window is emptied; the
    sibling owns p2035/t0001-t0003."""
    steps, _wf = chain
    parent = _step(steps, PARENT)
    dispatch = _step(steps, SIBLING)

    parent_rd = parent.flex_data.realized_dispatch
    if parent_rd is not None:
        remaining = parent_rd.filter(pl.col("period") == "p2035")
        assert remaining.height == 0, (
            "parent still owns realized p2035 cells — last-wins did not hand "
            f"them to the auto sibling:\n{parent_rd}"
        )

    disp_rd = dispatch.flex_data.realized_dispatch
    assert disp_rd is not None and disp_rd.height > 0
    owned = disp_rd.filter(pl.col("period") == "p2035")
    assert set(owned["step"].to_list()) == REALIZED_STEPS


# ===========================================================================
# (2) Spawn primitive records the sibling -> parent map
# ===========================================================================


def test_spawn_records_realized_dispatch_parent(tmp_path):
    """``spawn_realized_dispatch_sibling`` records the EXPLICIT sibling ->
    parent mapping consumed by the W4c post-pass (no ``__realized_dispatch``
    name-parse)."""
    from flextool.engine_polars._solve_config import SolveConfig

    url = _build_db(DIVERGENT_FIXTURE, tmp_path)
    sc = SolveConfig.load_from_db_url(url, SCENARIO)

    assert sc.realized_dispatch_parent == {}
    sc.spawn_realized_dispatch_sibling(PARENT, SIBLING)
    assert sc.realized_dispatch_parent[SIBLING] == PARENT


# ===========================================================================
# (3) Multi-period realized window — one realized row per realized period
# ===========================================================================


def test_projection_helper_multi_period_realized_window():
    """Lock the continuation-loop coverage: for a 2-period realized window
    the projection carries ONE realized-branch row per realized period and
    excludes every synthetic forecast row — in parent order."""
    from flextool.engine_polars._native_run_model import (
        project_realized_branch_time_branch_rows,
    )

    # Parent ``solve_branch__time_branch`` for a 2-period fan-at-first:
    # forecast rows (<period>_<branch>, branch) + realized rows
    # (period, realized_branch) for BOTH p2035 and p2040 (the continuation
    # loop appends one realized row per continued period).
    parent_time_branch = [
        ("p2035_high", "high"),
        ("p2035_low", "low"),
        ("p2035", "rlz"),
        ("p2040_high", "high"),
        ("p2040_low", "low"),
        ("p2040", "rlz"),
    ]
    # Parent ``period__branch``: realized self-row (period, period) per
    # period + fan rows (period, <period>_<branch>).
    parent_period_branch = [
        ("p2035", "p2035"),
        ("p2035", "p2035_high"),
        ("p2035", "p2035_low"),
        ("p2040", "p2040"),
        ("p2040", "p2040_high"),
        ("p2040", "p2040_low"),
    ]

    projected = project_realized_branch_time_branch_rows(
        parent_time_branch, parent_period_branch
    )
    assert projected == [("p2035", "rlz"), ("p2040", "rlz")]


def test_projection_helper_single_period_and_empty():
    """Single-period realized window -> one realized row; a parent with no
    realized self-row (defensive / parent-empty) -> empty projection (no
    crash)."""
    from flextool.engine_polars._native_run_model import (
        project_realized_branch_time_branch_rows,
    )

    single = project_realized_branch_time_branch_rows(
        [("p2035_high", "high"), ("p2035", "rlz")],
        [("p2035", "p2035"), ("p2035", "p2035_high")],
    )
    assert single == [("p2035", "rlz")]

    # No (p, p) self-row at all (e.g. an empty / pre-branch parent) -> [].
    assert project_realized_branch_time_branch_rows([], []) == []
    assert project_realized_branch_time_branch_rows(
        [("p2035_high", "high")], [("p2035", "p2035_high")]
    ) == []
