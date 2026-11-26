"""``investments_after_reveal_period`` on representative periods with differing forecasts, by hand.

Model (built in the test from the schema JSON): one period ``p1`` (one year, ``years_represented`` 1) on the
timeline ``tl`` of four 1-h steps.  Its timeset ``rp`` has two representative blocks of one step each, A = t0001 and
B = t0004, and four base periods t0001 .. t0004 with ``representative_period_weights`` t0001, t0002, t0003 ← A and
t0004 ← B (weight 1 each): A stands for 3 base hours, B for 1.  The forecasts ``high`` and ``low`` (weight 0.5
each, ``realized: no``) start at p1 / t0001, the solve's first period, which is the reveal period.

Node ``elec`` (balance, ``penalty_up`` 10 000 CUR/MWh, in the ``include_stochastics`` group) has the stochastic
demand high 100 (A) / 60 (B), low 80 (A) / 40 (B) MW.  Unit ``base`` gives up to 50 MW for free (existing 50).
Unit ``inv`` is invested (``invest_total``, ``virtual_unitsize`` 1, ``invest_cost`` 1 CUR/kW, ``discount_rate``
0.05, ``lifetime_technical`` 1) and costs 10 CUR/MWh to run.  Its annuity is 1000 · 0.05 / (1 − 1.05^−1) / 1.05
= 1000 CUR/MW a year (paid at the start of the year), and one period of one year gives the window cost
a = 1000 CUR/MW (as the p2040 build of ``test_invest_solve_branching``).  ``social_discount_rate`` is 0 (default).

Weights (``_solve_writers._compute_rp_period_frames``): the timestep weight of a block is the sum of the base
weights it stands for, times n_rep / n_base = 2 / 4, so w_A = 3 · 0.5 = 1.5 and w_B = 1 · 0.5 = 0.5 (they sum to
the 2 modelled steps).  The period's share of the year is its modelled hours over 8760, 2 / 8760.  An operating
cost term is weighted by step_duration · w_t / period_share · branch_weight (``core/`` objective header), so a
1-h step of block A costs c · 1.5 · 8760 / 2 = c · 6570 · branch weight, one of block B c · 0.5 · 8760 / 2 =
c · 2190 · branch weight: per hour of the year, 0.75 · c_A + 0.25 · c_B.

Under ``investments_after_reveal_period`` the investment of the reveal period is one decision of both forecasts:
x covers the largest residual over the forecasts and blocks, max(100, 60, 80, 40) − 50 = 50 MW (each MW less
leaves high's block A short, 0.5 · 6570 · 10 000 CUR per MW, far above a = 1000).  ``inv`` then runs the residual:
high A 50, B 10; low A 30, B 0 MW.  Expected operating cost

    10 · 8760 · [0.5 · (0.75 · 50 + 0.25 · 10) + 0.5 · (0.75 · 30 + 0.25 · 0)]
      = 10 · 8760 · [0.5 · 40 + 0.5 · 22.5] = 10 · 8760 · 31.25 = 2 737 500

(per base period of 4 h: 10 · [0.5 · (3 · 50 + 1 · 10) + 0.5 · (3 · 30 + 0)] = 1 250, times 8760 / 4).
Objective = 50 · 1000 + 2 737 500 = **2 787 500**.  The reported investment is 50 MW under p1: the label's own
``v_invest`` is 0, and the tied branch copies hold 50 each, weighted 0.5 + 0.5 in ``v_invest_fan``.

The contrast, ``investments_at_reveal_period`` (scenario ``at_reveal_s``, nothing realized: the reveal is the first
period, W-0639): each forecast invests for its own residual, high 50, low 30 MW, with the same operation:
0.5 · 50 · 1000 + 0.5 · 30 · 1000 + 2 737 500 = **2 777 500**, lower by the value of deciding after the reveal.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

REPO = Path(__file__).resolve().parents[2]
SCHEMA_JSON = REPO / "flextool" / "schemas" / "spinedb_schema.json"
SOLVE = "sol"
STEPS = [f"t{i:04d}" for i in range(1, 5)]
DEMAND = {"high": {"A": 100.0, "B": 60.0}, "low": {"A": 80.0, "B": 40.0}}
BLOCK_OF = {"t0001": "A", "t0002": "A", "t0003": "A", "t0004": "B"}


@pytest.fixture(autouse=True)
def _strict_autoscale(monkeypatch):
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")


def _m(data: dict, name: str) -> api.Map:
    return api.Map(list(data), list(data.values()), index_name=name)


@pytest.fixture(scope="module")
def rp_db(tmp_path_factory) -> tuple[str, Path]:
    """Build the database; return its url and folder."""
    from flextool.update_flextool.db_handle import open_database_mapping
    from flextool.update_flextool.initialize_database import initialize_database

    work = tmp_path_factory.mktemp("rp_after_reveal")
    db_path = work / "rp.sqlite"
    initialize_database(str(SCHEMA_JSON), str(db_path))
    url = f"sqlite:///{db_path.resolve().as_posix()}"
    entities = [("model", "flextool"), ("timeline", "tl"), ("timeset", "rp"), ("node", "elec"),
                ("unit", "base"), ("unit", "inv"), ("unit__outputNode", ("base", "elec")),
                ("unit__outputNode", ("inv", "elec")), ("group", "g"), ("group__node", ("g", "elec")),
                ("solve", SOLVE)]
    branches = _m({"p1": _m({b: _m({"t0001": _m({"no": 0.5}, "realized")}, "analysis_time")
                             for b in DEMAND}, "branch")}, "period")
    inflow = _m({b: _m({"t0001": _m({t: -DEMAND[b][BLOCK_OF[t]] for t in STEPS}, "time")}, "analysis_time")
                 for b in DEMAND}, "branch")
    rp_weights = _m({base: _m({"t0004" if BLOCK_OF[base] == "B" else "t0001": 1.0}, "representative_period")
                     for base in STEPS}, "base_period")
    values = [
        ("model", "flextool", "solves", api.Array([SOLVE])),
        ("timeline", "tl", "timestep_duration", _m({t: 1.0 for t in STEPS}, "time")),
        ("timeset", "rp", "timeline", "tl"),
        ("timeset", "rp", "timeset_duration", _m({"t0001": 1.0, "t0004": 1.0}, "time")),
        ("timeset", "rp", "representative_period_weights", rp_weights),
        ("node", "elec", "node_type", "balance"),
        ("node", "elec", "penalty_up", 10000.0),
        ("node", "elec", "penalty_down", 10000.0),
        ("node", "elec", "inflow", inflow),
        ("group", "g", "include_stochastics", "yes"),
        ("unit", "base", "existing", 50.0),
        ("unit", "base", "efficiency", 1.0),
        ("unit", "inv", "existing", 0.0),
        ("unit", "inv", "efficiency", 1.0),
        ("unit", "inv", "virtual_unitsize", 1.0),
        ("unit", "inv", "invest_method", "invest_total"),
        ("unit", "inv", "invest_cost", 1.0),
        ("unit", "inv", "discount_rate", 0.05),
        ("unit", "inv", "lifetime_technical", 1.0),
        ("unit", "inv", "invest_max_total", 1000.0),
        ("unit__outputNode", ("inv", "elec"), "other_operational_cost", 10.0),
        ("solve", SOLVE, "solve_mode", "single_solve"),
        ("solve", SOLVE, "period_timeset", _m({"p1": "rp"}, "period")),
        ("solve", SOLVE, "years_represented", _m({"p1": 1.0}, "period")),
        ("solve", SOLVE, "invest_periods", api.Array(["p1"])),
        ("solve", SOLVE, "realized_invest_periods", api.Array(["p1"])),
        ("solve", SOLVE, "realized_periods", api.Array(["p1"])),
        ("solve", SOLVE, "stochastic_branches", branches),
        ("solve", SOLVE, "invest_solve_branching_method", "investments_after_reveal_period"),
    ]
    at_reveal = [
        ("solve", SOLVE, "invest_solve_branching_method", "investments_at_reveal_period"),
        ("solve", SOLVE, "realized_invest_periods", api.Array([], value_type=str)),
        ("solve", SOLVE, "realized_periods", api.Array([], value_type=str)),
    ]
    with open_database_mapping(url) as db:
        _count, errors = api.import_data(
            db, alternatives=[("rp", ""), ("at_reveal", "")],
            scenarios=[("rp_s", False, ""), ("at_reveal_s", False, "")],
            scenario_alternatives=[("rp_s", "rp", None), ("at_reveal_s", "rp", "at_reveal"),
                                   ("at_reveal_s", "at_reveal", None)],
            entities=entities,
            entity_alternatives=[("node", ("elec",), "rp", True), ("unit", ("base",), "rp", True),
                                 ("unit", ("inv",), "rp", True)],
            parameter_values=([(c, e, p, v, "rp") for c, e, p, v in values]
                              + [(c, e, p, v, "at_reveal") for c, e, p, v in at_reveal]))
        assert not errors, errors[:5]
        db.commit_session("representative periods, investments that branch")
    return url, work


def _run(rp_db, scenario: str) -> Path:
    from flextool.engine import run_chain_from_db

    url, work = rp_db
    steps = run_chain_from_db(url, scenario, work_folder=work / scenario, keep_solutions=True)
    for step in steps.values():
        assert step.solution is not None and step.solution.optimal
    return work / scenario


@pytest.fixture(scope="module")
def rp_after_reveal(rp_db) -> Path:
    return _run(rp_db, "rp_s")


def _objective(work: Path) -> float:
    return float(pl.read_parquet(work / "output_raw" / f"v_obj__{SOLVE}.parquet")["objective"][0])


@pytest.mark.solver
def test_objective_is_the_shared_investment_plus_the_weighted_operating_cost(rp_after_reveal) -> None:
    assert _objective(rp_after_reveal) == pytest.approx(50 * 1000.0 + 2_737_500.0, rel=1e-9)


@pytest.mark.solver
def test_the_reveal_period_investment_is_one_decision_reported_under_p1(rp_after_reveal) -> None:
    raw = rp_after_reveal / "output_raw"
    got = {name: {r["period"]: float(r["inv"])
                  for r in pl.read_parquet(raw / f"{name}__{SOLVE}.parquet").iter_rows(named=True)}
           for name in ("v_invest", "v_invest_fan")}
    assert got["v_invest"] == pytest.approx({"p1": 0.0}, abs=1e-6)
    assert got["v_invest_fan"] == pytest.approx({"p1": 50.0}, abs=1e-6)


@pytest.mark.solver
def test_investments_at_the_reveal_period_invest_per_forecast(rp_db) -> None:
    assert _objective(_run(rp_db, "at_reveal_s")) == pytest.approx(
        0.5 * 50 * 1000.0 + 0.5 * 30 * 1000.0 + 2_737_500.0, rel=1e-9)
