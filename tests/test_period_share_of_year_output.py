"""``period_share_of_year__d``: the share of a year each realised period covers, the divisor of the ``…__d`` tables.

The value is ``complete_period_share_of_year`` of the solve that realises the period (``drop_levels.
_reduce_share_per_period``: the largest share over the solves), for the realised periods only.  Expected values
are the periods' hours over 8760:

* ``coal`` — one 48-h dispatch period: 48 / 8760 = 0.005479…;
* ``multi_fullYear_battery_nested_24h_invest_one_solve`` — a 24-h-step investment solve with a nested 6-h storage
  solve and a rolling dispatch that realises the four periods; the fixture's ``fullYear`` timeline is 72 h, so every
  realised period has 72 / 8760 (also the investment solve's share);
* a synthetic dispatch that covers the whole year after an investment solve on sampled days: the realised period has
  1.0 (the dispatch's share, not the sample's) and a period no solve realises is left out.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flextool.engine import run_chain_from_db
from flextool.process_outputs.write_outputs import write_outputs

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parquets(db_url: str, scenario: str, solver_cfg: Path, work: Path) -> Path:
    """Run ``scenario`` the way the CLI does and write its parquets; returns the scenario's parquet folder."""
    steps = run_chain_from_db(input_db_url=db_url, scenario_name=scenario, work_folder=work,
                              solver_config_dir=solver_cfg, warm=False)
    assert steps, scenario
    last = next(reversed(list(steps.values())))
    write_outputs(
        scenario, output_location=str(work / "out"), subdir=scenario,
        output_config_path=str(REPO_ROOT / "flextool" / "schemas" / "default_plots.yaml"),
        write_methods=["parquet"], raw_output_dir=str(work / "output_raw"), flex_data=last.flex_data,
        solution=last.solution, solve_name=last.solve_name, flex_data_provider=last.flex_data_provider)
    return work / "out" / "output_parquet" / scenario


def _share(folder: Path) -> dict[str, float]:
    frame = pd.read_parquet(folder / "period_share_of_year__d.parquet")
    frame = frame.reset_index() if "period" not in frame.columns else frame
    # The value column is ``(scenario, "period_share_of_year")`` (flattened to its string in the parquet).
    (col,) = [c for c in frame.columns if c != "period" and "period_share_of_year" in str(c)]
    return dict(zip(frame["period"].astype(str), frame[col].astype(float)))


def _years_periods(folder: Path) -> set[str]:
    frame = pd.read_parquet(folder / "years_represented__d.parquet")
    frame = frame.reset_index() if "period" not in frame.columns else frame
    return set(frame["period"].astype(str))


@pytest.mark.parametrize("scenario, expected", [
    ("coal", {"p2020": 48 / 8760}),
    ("multi_fullYear_battery_nested_24h_invest_one_solve", {d: 72 / 8760 for d in ("p2020", "p2025", "p2030", "p2035")}),
])
def test_period_share_of_year(scenario, expected, test_db_url, test_solver_config_dir, tmp_path) -> None:
    folder = _parquets(test_db_url, scenario, test_solver_config_dir, tmp_path)
    share = _share(folder)
    assert share, "no realised period"
    assert set(share) <= _years_periods(folder)
    assert share == pytest.approx(expected, rel=1e-12)


def test_full_year_dispatch_after_sampled_investment() -> None:
    from types import SimpleNamespace

    from flextool.process_outputs.drop_levels import _reduce_share_per_period
    from flextool.process_outputs.out_ancillary import period_share_of_year_d

    # Union order: the dispatch row of p2020 comes first, the investment solve's sampled share last.
    csoy = pd.Series([1.0, 72 / 8760, 72 / 8760],
                     index=pd.MultiIndex.from_tuples([("dispatch", "p2020"), ("invest", "p2020"),
                                                      ("invest", "p2025")], names=["solve", "period"]))
    par = SimpleNamespace(complete_period_share_of_year=_reduce_share_per_period(csoy))
    s = SimpleNamespace(d_realized_period=pd.Index(["p2020"], name="period"))
    frame = period_share_of_year_d(par, s)
    assert frame.index.name == "period" and frame.columns.name == "param"
    assert frame["period_share_of_year"].to_dict() == {"p2020": 1.0}
