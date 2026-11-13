"""Stochastic data before a mid-horizon reveal (item 250, owner decision Q2).

The stochastic fan is FORECASTS ONLY and ``realized`` is an OPTIONAL 0-or-1
designation.  A period before the reveal has no forecast copy of its own: it
reads the series of the realized branch, the forecast made at the solve's
start.  A forecasts-only fan (no realized branch) has no series for such a
period, and FlexTool never picks one of the forecasts (it used to bind the
lexically-smallest one): a stochastic value read there is a configuration
error, while plain (non-stochastic) data runs.

Witnesses, on the ``stoch_two_period_midhorizon_nreal0`` fixture
(p2035 = t0001-t0003 before the reveal, p2040 fans out at t0004 into high /
low, weights 1 : 3, DEFAULT ``stochastic_invest_method = none``):

* ``test_forecasts_only_fan_binds_no_forecast_before_the_reveal`` --
  solver-free: ``create_stochastic_periods`` gives the pre-reveal period no
  ``solve_branch__time_branch`` (SBTB) row.
* ``midhorizon`` (plain profile) runs: p2035 wind 60, base 60, invest 60.
* ``midhorizon_stoch_profile`` (stochastic profile, no realized branch) is the
  configuration error, with its message.
* ``midhorizon_realized`` (realized branch ``real``) runs: p2035 reads
  real@t0001 = 0.6 (wind 60), base covers the residual 60 and the
  here-and-now invest is 60 -- also in the stochastic solve's ``v_invest``
  output and the processed capacity / investment-cost outputs (item 258).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine.input._solve_state import ActiveTimeEntry, FlexToolConfigError
from flextool.engine.solve._stochastic import StochasticSolver


# ---------------------------------------------------------------------------
# Solver-free: no forecast owns the pre-reveal period.
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
        logger=logging.getLogger("test_pre_reveal_stochastic_data"),
        timeline=SimpleNamespace(stochastic_timesteps=defaultdict(list)),
    )
    return StochasticSolver(state)


def test_forecasts_only_fan_binds_no_forecast_before_the_reveal() -> None:
    """n_real=0 mid-horizon fan: SBTB holds exactly the forecast copies; p2035 (before the reveal) and the
    reveal period's label p2040 get no row (before: (p2035, high), the first forecast in sorted order)."""
    solver = _make_solver()
    active = {"p2035": P1, "p2040": P2}
    realized = {"p2035": P1[0:], "p2040": P2[0:]}

    (_pb, sbtb, _atl, _jumps, _fstl, _rtl, _bstl, _fan_label_times) = (
        solver.create_stochastic_periods(
            {"s": BRANCH_INFO_NREAL0_MID},
            ["s"],
            {"s": "s"},
            {"s": active},
            {"s": {}},
            {"s": realized},
        )
    )
    assert sorted(sbtb["s"]) == [("p2040_high", "high"), ("p2040_low", "low")]


# ---------------------------------------------------------------------------
# Solve-based witnesses.
# ---------------------------------------------------------------------------
SOLVE = "stoch_2p_mid"
PRE_REVEAL_PERIOD = "p2035"
DB_FIXTURE = "stoch_two_period_midhorizon_nreal0"
# p2035 wind 0.6·100 = 60 → the investable ``base`` covers 120 − 60 = 60 at every step, and the here-and-now
# invest (cheaper at p2035, every later residual below 60) is 60.
EXPECTED_PRE_REVEAL_BASE_DISPATCH = 60.0
# The raw ``v_flow`` is per unit size; wind's unit size is its existing 100 MW, so 60 MW reads 0.6.
EXPECTED_PRE_REVEAL_WIND = 0.6
EXPECTED_PRE_REVEAL_INVEST = 60.0


def _wind_col(df):
    return next(c for c in df.columns if "wind" in c)


def _base_col(df):
    return next(c for c in df.columns if c.startswith("('base'"))


def _outputs(run, scenario: str):
    """The pre-reveal ``v_flow`` rows (of the realization dispatch when there is a realized branch: it holds the
    realized operational results) and the stochastic solve's ``v_invest``."""
    wf = run.work_folder
    flow_solve = f"{SOLVE}__realized_dispatch" if scenario == "midhorizon_realized" else SOLVE
    v_flow = pl.read_parquet(wf / "output_raw" / f"v_flow__{flow_solve}.parquet")
    v_invest = pl.read_parquet(wf / "output_raw" / f"v_invest__{SOLVE}.parquet")
    return v_flow.filter(pl.col("period") == PRE_REVEAL_PERIOD), v_invest


@pytest.fixture(scope="module", params=["midhorizon", "midhorizon_realized"])
def midhorizon_run(request, scenario_run, monkeypatch_module):
    monkeypatch_module.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    return request.param, scenario_run(request.param, db_fixture=DB_FIXTURE)


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.mark.solver
def test_pre_reveal_period_reads_plain_or_realized_data(midhorizon_run):
    """Plain profile (forecasts-only fan) and realized branch: p2035 wind binds 60 and base covers 60 at every
    step.  Forecasts-only: the here-and-now invest at p2035 is 60."""
    scenario, run = midhorizon_run
    pre, v_invest = _outputs(run, scenario)
    assert pre.height == 3
    assert pre[_wind_col(pre)].to_list() == pytest.approx([EXPECTED_PRE_REVEAL_WIND] * 3)
    assert pre[_base_col(pre)].to_list() == pytest.approx([EXPECTED_PRE_REVEAL_BASE_DISPATCH] * 3)
    # Item 258: with a realized branch the realization dispatch solve holds the dispatch of every period, so the
    # stochastic solve has no dispatch axis; its here-and-now investment at p2035 still has its row.
    assert v_invest["period"].to_list() == [PRE_REVEAL_PERIOD, "p2040"], v_invest
    assert v_invest["base"].to_list() == pytest.approx([EXPECTED_PRE_REVEAL_INVEST, 0.0])


# The here-and-now investment of 60 MW at p2035, at invest_cost 1 CUR/kW (1000 CUR/MW), lifetime 1 year and
# discount_rate 0.05: annuity 0.05 / (1 - 1.05**-1) = 1.05, so 60 * 1000 * 1.05 = 63 000 CUR = 0.063 M CUR a year.
EXPECTED_PRE_REVEAL_ANNUALIZED_INVEST_MCUR = 0.063


@pytest.mark.solver
def test_realized_branch_reports_the_pre_reveal_investment(
        stoch_two_period_midhorizon_nreal0_db_url, tmp_path, monkeypatch):
    """Item 258: with a realized branch the processed outputs show the p2035 investment (before: no invested
    capacity and no investment cost, as the stochastic solve's ``v_invest`` had no p2035 row)."""
    from flextool.engine import run_chain_from_db
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.process_outputs.write_outputs import write_outputs

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    monkeypatch.chdir(tmp_path)
    wf = tmp_path / "w"
    steps = run_chain_from_db(stoch_two_period_midhorizon_nreal0_db_url, "midhorizon_realized",
                              work_folder=wf, csv_dump=True, keep_solutions=True)
    last = next(reversed(steps.values()))
    last.flex_data_provider.snapshot_processed_inputs(wf)
    write_outputs(
        scenario_name="midhorizon_realized", output_location=str(wf), subdir="out",
        output_config_path=str(Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "default_plots.yaml"),
        write_methods=["csv"], fallback_output_location=str(wf), raw_output_dir=str(wf / "output_raw"),
        solution=last.solution, solve_name=last.solve_name,
        solve_steps=[(s.solve_name, s.flex_data, s.effective_solution) for s in steps.values()],
        flex_data_provider=FlexDataProvider(),
    )
    csv_dir = wf / "output_csv" / "out"
    capacity = pl.read_csv(csv_dir / "unit_capacity__d.csv").filter(
        (pl.col("unit") == "base") & (pl.col("period") == PRE_REVEAL_PERIOD))
    assert capacity["invested"].to_list() == pytest.approx([EXPECTED_PRE_REVEAL_INVEST])
    costs = pl.read_csv(csv_dir / "annualized_costs__d.csv").filter(pl.col("period") == PRE_REVEAL_PERIOD)
    assert costs["unit investment & retirement"].to_list() == pytest.approx(
        [EXPECTED_PRE_REVEAL_ANNUALIZED_INVEST_MCUR])


@pytest.mark.solver
def test_realized_branch_owns_the_pre_reveal_period(midhorizon_run):
    """SBTB binds p2035 (and the realization's real name p2040) to the realized branch ``real`` with it, and
    neither to any forecast without it."""
    scenario, run = midhorizon_run
    sbtb = pl.read_csv(run.work_folder / "solve_data" / "solve_branch__time_branch.csv")
    pairs = {(r["period"], r["branch"]) for r in sbtb.iter_rows(named=True)}
    pre = {b for p, b in pairs if p == PRE_REVEAL_PERIOD}
    owner = {"real"} if scenario == "midhorizon_realized" else set()
    assert pre == owner, pairs
    assert {b for p, b in pairs if p == "p2040"} == owner, pairs


def test_forecasts_only_fan_with_stochastic_data_before_it_is_an_error(
        stoch_two_period_midhorizon_nreal0_db_url, tmp_path, monkeypatch):
    """A stochastic wind profile and no realized branch: p2035 has no series to read before the reveal (before:
    the forecast 'high', the first in sorted order), a configuration error naming the profile, the solve, the
    period, its steps and the fix."""
    from flextool.engine import run_chain_from_db

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    with pytest.raises(FlexToolConfigError) as err:
        run_chain_from_db(stoch_two_period_midhorizon_nreal0_db_url, "midhorizon_stoch_profile",
                          work_folder=tmp_path / "w")
    assert str(err.value) == (
        "profile 'wprof', parameter 'profile': the value has forecasts, and solve 'stoch_2p_mid' reads it in "
        "period 'p2035' (time steps 't0001', 't0002', 't0003'), before its forecasts branch out.  No forecast of "
        "the solve is marked realized: yes, so there is no series to read before the forecasts branch out; "
        "FlexTool does not pick one of the forecasts, nor the default value.  Designate a realized branch "
        "(realized: yes in the solve's stochastic_branches) -- it provides the shared data before the forecasts "
        "branch out, at analysis_time 't0001' -- or give 'profile' a value without forecasts.  Check which "
        "alternative of the scenario gives this value.")
