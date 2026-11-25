"""``invest_solve_branching_method`` on ``stoch_two_period_invest`` (data version 84): the three methods, by hand.

Fixture (``tests/fixtures/build_stoch_two_period_invest.py``): periods p2035 / p2040 of one year each, forecasts
``high`` (weight 1) and ``low`` (weight 3) from p2035 (the reveal, the first period), probabilities 0.25 / 0.75.
Deterministic demand 100 / 130 MW; the free existing wind gives high 40 / 40, low 20 / 10 MW, so the investable
``peaker`` has to cover the residual high 60 / 90, low 80 / 120 MW (the shortfall penalty 10 000 CUR/MWh dominates;
operations cost nothing).  Peaker annuity 1 · 1000 · 0.05 / (1 − 1.05^−1) / 1.05 = 1000 CUR/MW a year (paid at the
start of the year); a p2035 build serves p2035 and p2040 (lifetime 1, replaced automatically), so its window costs
2000 / MW, a p2040 build 1000 / MW.  x = the p2035 build, y = the p2040 build.

* T4 ``operations_only`` (scenario ``operations_only``): one plan for both forecasts.  x ≥ max(60, 80) = 80 and
  x + y ≥ max(90, 120) = 120; a p2040 MW is cheaper, so x = 80, y = 40:
  objective = 80·2000 + 40·1000 = **200 000**.
* T5 ``investments_after_reveal_period`` (scenario ``after_reveal``, realized p2035): x is shared and must cover
  low's 80 (building more at p2035 costs 2000 against 1000 at p2040), so x = 80; p2040 per forecast: high
  90 − 80 = 10, low 120 − 80 = 40: objective = 80·2000 + 0.25·10·1000 + 0.75·40·1000 = 160 000 + 2 500 + 30 000
  = **192 500**.  The realized p2035 investment 80 is reported under p2035.
* T6 ``investments_at_reveal_period`` (scenario ``recourse``, nothing realized: the reveal is the first period):
  each forecast plans on its own: high 60 + 30, low 80 + 40:
  objective = 0.25·(60·2000 + 30·1000) + 0.75·(80·2000 + 40·1000) = 37 500 + 150 000 = **187 500**.

The value of sharing less: 187 500 < 192 500 < 200 000.  LP parity with the settings these replace (old ``none``
with ``shared_operation_periods = []``, ``recourse`` with ``shared_invest_periods = [reveal]`` and unset) is proven
by ``tools/lp_snapshot.py`` (spec ``stochastic_controls``), not here.
"""
from __future__ import annotations

import polars as pl
import pytest

SOLVE = "stoch_2p_inv"
EXPECTED = {"operations_only": 200_000.0, "after_reveal": 192_500.0, "recourse": 187_500.0}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _objective(run) -> float:
    return float(pl.read_parquet(run.work_folder / "output_raw" / f"v_obj__{SOLVE}.parquet")["objective"][0])


@pytest.mark.solver
@pytest.mark.parametrize("scenario", list(EXPECTED))
def test_objective_by_hand(scenario_run, scenario) -> None:
    run = scenario_run(scenario, db_fixture="stoch_two_period_invest")
    assert _objective(run) == pytest.approx(EXPECTED[scenario], rel=1e-9)


@pytest.mark.solver
def test_after_reveal_reports_the_shared_p2035_investment(scenario_run) -> None:
    """T5: the reported investment of a realized period is the label's own ``v_invest`` (0: the decision is on the
    branch copies) plus the branch-weighted ``v_invest_fan`` (``calc_costs._with_fan``).  The tied copies hold 80 each,
    weighted 0.25 + 0.75: p2035 reports 80 MW.  p2040 is per forecast and not realized: no row."""
    run = scenario_run("after_reveal", db_fixture="stoch_two_period_invest")
    raw = run.work_folder / "output_raw"
    got: dict[str, float] = {}
    for name in ("v_invest", "v_invest_fan"):
        for r in pl.read_parquet(raw / f"{name}__{SOLVE}.parquet").iter_rows(named=True):
            got[r["period"]] = got.get(r["period"], 0.0) + float(r["peaker"])
    assert got == pytest.approx({"p2035": 80.0}, abs=1e-6)


@pytest.mark.solver
def test_operations_only_reports_both_shared_investments(scenario_run) -> None:
    """T4: every investment is one plan and both periods are realized: p2035 80, p2040 40 MW."""
    run = scenario_run("operations_only", db_fixture="stoch_two_period_invest")
    inv = pl.read_parquet(run.work_folder / "output_raw" / f"v_invest__{SOLVE}.parquet")
    got = {r["period"]: round(float(r["peaker"]), 6) for r in inv.iter_rows(named=True)}
    assert got == {"p2035": 80.0, "p2040": 40.0}
