"""Invest on representative periods, dispatch the full timeline: the invested capacity is handed over (b5 chain).

The model (``tests/engine/_tiny_model_db.py``): one node with demand 50, 60, 80, 70 MWh in t1..t4 (1 h steps),
upward penalty 10 000 CUR/MWh; one unit to invest in at 100 CUR/kW, lifetime 1 year, discount rate 0.05, other
operational cost 1 CUR/MWh.  ``invest_rp`` sees only the representative block t1..t2 (weight 2 × 1 / 2 = 1, period share 2/8760);
``dispatch`` runs t1..t4 on the capacity ``invest_rp`` built.

Hand values (independent of the code):

* Investment.  Below 60 MW every MW short costs 1 MWh × 4380 × 1e4 = 4.38e7 CUR/a, far above the annuity
  (r = 0.05, n = 1: 1e5 × 0.05 / (1 − 1.05⁻¹) = 1.05e5 CUR/MW/a); a MW above
  60 saves nothing.  So **60 MW**, and the investment cost is 60 × 1.05e5 = **6.3 M CUR**, counted once.
* Dispatch on 60 MW.  Generation 50, 60, 60, 60, unserved 0, 0, **20, 10** MWh (30 MWh); annualised with the
  dispatch share 4/8760: 30 × 8760 / 4 = **65 700 MWh/a**.
* Without the handover the dispatch solve has 0 MW: 260 MWh unserved (569 400 MWh/a).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.engine import _tiny_model_db as T

pytestmark = pytest.mark.solver

INVESTED_MW = 60.0
UNSERVED_MWH = {"t1": 0.0, "t2": 0.0, "t3": 20.0, "t4": 10.0}
UNSERVED_MWH_PER_YEAR = 30.0 * 8760.0 / 4.0          # 65 700
UNSERVED_WITHOUT_HANDOVER_MWH = 50.0 + 60.0 + 80.0 + 70.0  # 260
INVEST_COST_MCUR = INVESTED_MW * 100.0 * 1000.0 * 1.05 / 1e6  # 6.3


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    from flextool.engine import run_chain_from_db
    from flextool.process_outputs.write_outputs import write_outputs

    work = tmp_path_factory.mktemp("rp_invest_dispatch_chain")
    url = T.build_db(work, {"invest_rp": T.invest_rp_solve(), "dispatch": T.dispatch_solve()})
    steps = run_chain_from_db(url, T.SCENARIO, work_folder=work / "w", csv_dump=True, keep_solutions=True)
    assert list(steps) == ["invest_rp", "dispatch"]
    assert all(s.solution is not None and s.solution.optimal for s in steps.values())
    last = steps["dispatch"]
    write_outputs(
        scenario_name="tiny", output_location=str(work / "out"), subdir="tiny", write_methods=["csv"],
        flex_data=last.flex_data, solution=last.solution, solve_name=last.solve_name,
        flex_data_provider=getattr(last, "flex_data_provider", None),
        raw_output_dir=str(work / "w" / "output_raw"),
    )
    return steps, Path(work / "out" / "output_csv" / "tiny")


def _detail_csv(path: Path) -> pd.DataFrame:
    """A CSV with a second header row of column details (node__dt / node__d)."""
    return pd.read_csv(path, header=[0, 1])


def test_the_investment_solve_builds_60_mw(chain) -> None:
    _, csv = chain
    cap = pd.read_csv(csv / "unit_capacity__d.csv")
    row = cap[(cap["unit"] == "gen") & (cap["period"] == "p1")]
    assert len(row) == 1
    assert float(row["invested"].iloc[0]) == pytest.approx(INVESTED_MW, rel=1e-9)
    assert float(row["total"].iloc[0]) == pytest.approx(INVESTED_MW, rel=1e-9)
    # An investment table names the solve that realized the investment (#210).
    assert row["solve"].iloc[0] == "invest_rp"


def test_the_dispatch_solve_runs_on_the_handed_over_capacity(chain) -> None:
    _, csv = chain
    node_dt = _detail_csv(csv / "node__dt.csv")
    solve_col = node_dt.columns[0]
    time_col = node_dt.columns[2]
    lol = [c for c in node_dt.columns if c[0] == "n" and c[1] == "Loss of load"]
    assert len(lol) == 1
    unserved = dict(zip(node_dt[time_col], node_dt[lol[0]].astype(float)))
    assert unserved == pytest.approx(UNSERVED_MWH, abs=1e-6)
    assert sum(unserved.values()) != pytest.approx(UNSERVED_WITHOUT_HANDOVER_MWH)
    assert set(node_dt[solve_col]) == {"dispatch"}

    node_d = _detail_csv(csv / "node__d.csv")
    lol_d = [c for c in node_d.columns if c[0] == "n" and c[1] == "Loss of load"]
    assert float(node_d[lol_d[0]].iloc[0]) == pytest.approx(UNSERVED_MWH_PER_YEAR, rel=1e-9)


def _invest_rows(step) -> int:
    try:
        return step.solution.value("v_invest_p").height
    except KeyError:
        return 0


def test_the_dispatch_solve_has_no_investment_variable(chain) -> None:
    steps, _ = chain
    assert _invest_rows(steps["invest_rp"]) == 1
    assert _invest_rows(steps["dispatch"]) == 0


def test_the_investment_cost_is_counted_once(chain) -> None:
    _, csv = chain
    costs = pd.read_csv(csv / "costs_discounted.csv").set_index("category").iloc[:, 0]
    assert float(costs["unit investment & retirement"]) == pytest.approx(INVEST_COST_MCUR, rel=1e-9)
