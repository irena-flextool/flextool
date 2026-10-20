"""Single here-and-now invest + stochastic operational fan (the DEFAULT
stochastic model's defining structure).

Owner decision 6 / the Q1 coverage gap: under the DEFAULT
``stochastic_invest_method = none``, a solve carrying BOTH an invest
decision AND a genuinely divergent operational forecast fan must produce

  * a SINGLE here-and-now ``v_invest`` per REAL period — one unambiguous
    capacity decision, NOT branch-suffixed copies (that would be the
    opt-in ``recourse`` wait-and-see structure, which this test must
    genuinely reject); while
  * ``v_flow`` fans across the forecast branches (the branch-suffixed
    forecast periods own their own operational LP variables and the
    forecasts genuinely diverge); and
  * the solve is Optimal.

This is the "single here-and-now invest + stochastic ops" structure — the
engine default the owner wanted — as opposed to recourse (per-branch
post-reveal invest), which is NOT exercised here.

Witness fixture: ``w4c_divergent_dispatch.json`` — the W4c parent
``stoch_rd`` is a genuinely stochastic INVEST solve (peaker ``invest_total``)
on the DEFAULT ``none`` invest method, with branch-DISTINCT wind
(rlz 0.50 / high 0.80 / low 0.10) so the forecasts genuinely diverge
(peaker 20 MW on the high-wind forecast, 90 MW on the low-wind forecast).
We read the parent solve's raw LP solution directly (the forecast-period
flow variables are LP variables regardless of output reporting), so the
assertions pin the LP structure, not an output-formatting choice.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_JSON = (
    FLEXTOOL_ROOT / "tests" / "fixtures" / "w4c_divergent_dispatch.json"
)

SCENARIO = "w8_realized_dispatch"
PARENT = "stoch_rd"

# Real (un-suffixed) period and its two forecast-branch copies.
REAL_PERIOD = "p2035"
FORECAST_PERIODS = {"p2035_high", "p2035_low"}
# Divergent peaker levels on the two forecasts (100 MW demand minus
# 0.80 / 0.10 wind of 100 MW existing).
PEAKER_HIGH_MW = 20.0
PEAKER_LOW_MW = 90.0


def _build_db(tmp_path: Path) -> str:
    if not FIXTURE_JSON.exists():
        pytest.skip(f"fixture JSON not present: {FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{FIXTURE_JSON.stem}.sqlite"
    url = json_to_db(FIXTURE_JSON, db_path)
    migrate_database(url)
    return url


def _run_chain(tmp_path: Path):
    from flextool.engine_polars import run_chain_from_db

    url = _build_db(tmp_path)
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
    tmp_path = tmp_path_factory.mktemp("single_invest_stochastic_ops")
    return _run_chain(tmp_path)


def _parent(steps):
    assert PARENT in steps, f"no {PARENT!r} step in {list(steps)}"
    return steps[PARENT]


@pytest.mark.solver
def test_parent_optimal(chain) -> None:
    """The combined invest + stochastic-ops solve is Optimal."""
    steps, _wf = chain
    parent = _parent(steps)
    assert parent.solution is not None and parent.optimal


@pytest.mark.solver
def test_invest_is_single_here_and_now(chain) -> None:
    """``v_invest`` is SINGLE-valued per real period: it lives ONLY on the
    real period ``p2035`` — there are NO branch-suffixed invest copies
    (``p2035_high`` / ``p2035_low``).

    This genuinely fails if invest were to fan across forecast branches
    (the opt-in recourse structure): a fanned invest would add
    ``p2035_high`` / ``p2035_low`` rows to ``v_invest`` and the period set
    would no longer be exactly ``{p2035}``."""
    steps, wf = chain
    _parent(steps)  # ensure the parent step ran
    inv_path = wf / "output_raw" / f"v_invest__{PARENT}.parquet"
    assert inv_path.exists(), f"missing {inv_path}"
    inv = pl.read_parquet(inv_path)
    assert "period" in inv.columns, (
        f"v_invest has no 'period' column: {inv.columns}"
    )
    invest_periods = set(inv["period"].drop_nulls().to_list())
    # Exactly the real here-and-now period — no branch-suffixed copies.
    assert invest_periods == {REAL_PERIOD}, (
        f"v_invest fanned across periods {sorted(invest_periods)}; "
        f"expected a single here-and-now invest on {{{REAL_PERIOD}}}"
    )
    assert not (invest_periods & FORECAST_PERIODS), (
        f"v_invest carries forecast-branch periods "
        f"{sorted(invest_periods & FORECAST_PERIODS)} — invest must not fan "
        f"under stochastic_invest_method=none"
    )
    # Exactly one invested (entity, period) row — a single capacity decision.
    assert inv.height == 1, (
        f"expected a single v_invest row (one here-and-now decision); "
        f"got {inv.height}:\n{inv}"
    )


@pytest.mark.solver
def test_flow_fans_across_forecast_branches(chain) -> None:
    """``v_flow`` fans across the forecast branches: both branch-suffixed
    forecast periods carry operational variables and the peaker dispatch
    genuinely diverges (20 MW on the high-wind forecast, 90 MW on the
    low-wind forecast)."""
    steps, _wf = chain
    parent = _parent(steps)
    flows = parent.solution.value("v_flow")
    assert "d" in flows.columns, f"v_flow period column missing: {flows.columns}"
    periods = set(flows["d"].unique().to_list())
    # The branch-suffixed forecast periods own operational LP variables.
    assert FORECAST_PERIODS <= periods, (
        f"v_flow missing forecast-branch periods "
        f"{sorted(FORECAST_PERIODS - periods)}; got {sorted(periods)}"
    )

    peaker = flows.filter(pl.col("source") == "peaker")

    def _level(period: str) -> set[float]:
        return {
            round(v, 6)
            for v in peaker.filter(pl.col("d") == period)["value"].to_list()
        }

    high = _level("p2035_high")
    low = _level("p2035_low")
    assert high == {PEAKER_HIGH_MW}, (
        f"high-forecast peaker {sorted(high)} != {{{PEAKER_HIGH_MW}}}"
    )
    assert low == {PEAKER_LOW_MW}, (
        f"low-forecast peaker {sorted(low)} != {{{PEAKER_LOW_MW}}}"
    )
    # The forecasts GENUINELY diverge (not a collapsed single value).
    assert high != low, (
        "forecast branches did not diverge — v_flow is not a stochastic fan"
    )
