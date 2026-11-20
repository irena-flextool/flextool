"""Item 287 P2: a Benders solve ends with its final solve, and the run writes the result tables of a monolithic run.

After the Benders loop, the whole model of the solve is solved once with every investment decision fixed to the
Benders incumbent (``_orchestration._run_one``), on the normal monolithic path.  Oracles are relational: each
Benders run is compared with its **monolithic twin**, the same database with ``solve.decomposition = none`` on
the Benders solve (the values come from the twin, never from the new code).

* **T3** ``lh2_three_region_trade_invest`` (one Benders solve, the last): every csv the twin writes is written;
  ``costs_discounted``, ``summary_solve`` and the capacity tables (``invested`` included) equal the twin's within
  ``max(gap, 1e-6) * |objective|``; the investment marginal values are written and finite; the final objective is
  between the Benders bounds; exit 0, parquet ``written``, the run record has ``benders.final_solve.accepted``.
* **T4** the same with ``--benders-final-solve off``: no table, W-0570, parquet ``skipped``, exit 0.
* **T7** ``lh2_three_region_invest`` (Benders invest solve, then a dispatch solve) against the monolithic chain:
  cost, capacity and summary tables within the gap, the investment marginal tables written, the same warning
  codes; flows only as totals per period.
* **T8** ``benders_max_iter = 1``: the Benders solve does not converge; the final solve still runs and the tables
  are written, the non-convergence banner fires, the run record keeps ``not_converged``, exit 0.
* **T9** (the last-solve half) the final solve not accepted (acceptance patched): W-0572, the Benders step and its
  handoff stay, ``benders.final_solve.accepted`` false, no table, parquet ``skipped``, exit 0.

The runs are CLI subprocesses (``cmd_run_flextool``), started together; T9 runs the CLI in this process.
"""
from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from flextool import run_record
from tests.engine._unit_db import b64, make_db

pytestmark = pytest.mark.solver

REPO = Path(__file__).resolve().parents[2]
LH2_JSON = REPO / "tests" / "fixtures" / "lh2_three_region.json"
TRADE_JSON = REPO / "tests" / "fixtures" / "lh2_three_region_trade_invest.json"
TRADE_CHAIN = ["lh2_three_region", "lh2_three_region_trade_invest"]
INVEST_CHAIN = ["lh2_three_region", "lh2_three_region_invest"]
TRADE_SOLVE = "lh2_trade_invest"
INVEST_SOLVE = "lh2_invest"

#: Tables compared value by value (numeric columns, rows keyed by their text columns).
COST_TABLES = ("costs_discounted.csv", "summary_solve.csv", "annualized_costs__d.csv", "costs_discounted__d.csv")
CAPACITY_TABLES = ("unit_capacity__d.csv", "connection_capacity__d.csv", "node_capacity__d.csv")
INVEST_MARGINAL = ("connection_invest_marginal__d.csv", "unit_invest_marginal__d.csv",
                   "node_invest_marginal__d.csv")


def _none(solve: str) -> list:
    return [["solve", solve, "decomposition", b64("none", "str")]]


#: name -> (fixture, base chain, override values, extra CLI flags)
RUNS = {
    "ti_benders": (TRADE_JSON, TRADE_CHAIN, [], []),
    "ti_mono": (TRADE_JSON, TRADE_CHAIN, _none(TRADE_SOLVE), []),
    "ti_off": (TRADE_JSON, TRADE_CHAIN, [], ["--benders-final-solve", "off"]),
    "ti_one_iter": (TRADE_JSON, TRADE_CHAIN, [["solve", TRADE_SOLVE, "benders_max_iter", b64(1.0, "float")]], []),
    "inv_benders": (LH2_JSON, INVEST_CHAIN, [], []),
    "inv_mono": (LH2_JSON, INVEST_CHAIN, _none(INVEST_SOLVE), []),
}


class Run:
    def __init__(self, name: str, scenario: str, root: Path, result: subprocess.CompletedProcess):
        self.name, self.scenario, self.root, self.result = name, scenario, root, result
        self.csv = root / "out" / "output_csv" / scenario
        self.parquet = root / "out" / "output_parquet" / scenario

    @property
    def out(self) -> str:
        return self.result.stdout + self.result.stderr

    def status(self) -> dict:
        status = run_record.read_status_file(self.parquet)
        assert status is not None, self.out
        return status

    def table(self, name: str) -> list[dict]:
        with open(self.csv / name, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def codes(self) -> set[str]:
        return set(re.findall(r"\[(W-\d{4})\]", self.out))

    def bounds(self) -> tuple[float, float]:
        lb = re.search(r"lower bound \(valid\) = (\S+)", self.out)
        ub = re.search(r"upper bound \(best\)  = (\S+)", self.out)
        assert lb and ub, self.out
        return float(lb.group(1)), float(ub.group(1))

    def gap(self) -> float:
        return float(re.findall(r"optimality gap\s+= (\S+)", self.out)[-1])


def _cli(name: str, url: str, scenario: str, root: Path, flags: list[str]) -> Run:
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FLEXTOOL_THREADS="1", PYTHONIOENCODING="utf-8")
    env.pop("FLEXTOOL_BENDERS_FINAL_SOLVE", None)
    result = subprocess.run(
        [sys.executable, "-m", "flextool.cli.cmd_run_flextool", url, "--scenario-name", scenario,
         "--work-folder", str(root / "work"), "--output-location", str(root / "out"),
         "--write-methods", "csv", "parquet", *flags],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO), env=env, timeout=900,
    )
    return Run(name, scenario, root, result)


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> dict[str, Run]:
    jobs = []
    for name, (fixture, chain, values, flags) in RUNS.items():
        url, scenario = make_db(tmp_path_factory, alt=f"p287p2_{name}", values=values, base_chain=chain,
                                fixture_json=fixture)
        jobs.append((name, url, scenario, tmp_path_factory.mktemp(f"p287p2_{name}_run"), flags))
    with ThreadPoolExecutor(max_workers=min(len(jobs), 6)) as pool:
        done = list(pool.map(lambda job: _cli(*job), jobs))
    return {run.name: run for run in done}


def _ok(run: Run) -> None:
    assert run.result.returncode == 0, run.out


def _numeric(value: str) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _cells(run: Run, name: str) -> list[list[str]]:
    with open(run.csv / name, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _assert_table_close(a: Run, b: Run, name: str, atol: float) -> None:
    """Row by row (the twins write the same rows in the same order): equal text cells, numbers within *atol*
    (an empty cell is 0)."""
    ra, rb = _cells(a, name), _cells(b, name)
    assert len(ra) == len(rb), f"{name}: {len(ra)} rows, the twin {len(rb)}"
    for row_a, row_b in zip(ra, rb):
        if row_a and "Output at" in row_a[0]:
            continue  # the time the table was written
        assert len(row_a) == len(row_b), (name, row_a, row_b)
        for x, y in zip(row_a, row_b):
            nx, ny = _numeric(x), _numeric(y)
            if (nx is None and x != "") or (ny is None and y != ""):
                assert x == y, (name, row_a, row_b)
            else:
                assert (nx or 0.0) == pytest.approx(ny or 0.0, abs=atol), (name, row_a, row_b)


def _objective(run: Run) -> float:
    """The objective (CUR) of the run's first solve, from its run record."""
    status = run.status()
    objs = [s["objective"] for s in status["solves"] if s["objective"] is not None]
    assert objs, status
    return objs[0]


def _tolerance(benders: Run, mono_obj: float) -> float:
    # Costs are in M CUR, the objective of the run record in CUR.
    return max(benders.gap(), 1e-6) * abs(mono_obj) / 1e6


# ---------------------------------------------------------------------------------------------------------------
# T3: Benders last, final solve on
# ---------------------------------------------------------------------------------------------------------------


def test_benders_last_writes_the_tables_of_the_monolith(runs):
    bend, mono = runs["ti_benders"], runs["ti_mono"]
    _ok(bend)
    _ok(mono)
    mono_files = {p.name for p in mono.csv.glob("*.csv")} - {"timings.csv"}
    bend_files = {p.name for p in bend.csv.glob("*.csv")}
    assert mono_files <= bend_files, sorted(mono_files - bend_files)
    atol = _tolerance(bend, _objective(mono))
    for name in COST_TABLES:
        _assert_table_close(bend, mono, name, atol)
    for name in CAPACITY_TABLES:
        _assert_table_close(bend, mono, name, max(atol, 1e-6))
    invested = [r for r in bend.table("connection_capacity__d.csv") if _numeric(r.get("invested", "")) or 0.0]
    assert invested, "the trade pipes' investment is not in connection_capacity__d"
    marginal = bend.table("connection_invest_marginal__d.csv")
    assert marginal and all(math.isfinite(v) for r in marginal for v in map(_numeric, r.values()) if v is not None)
    assert (bend.parquet / "dual_invest_effective_connection_d_e.parquet").is_file()


def test_benders_last_final_objective_is_inside_the_bounds_and_recorded(runs):
    bend = runs["ti_benders"]
    _ok(bend)
    lb, ub = bend.bounds()
    status = bend.status()
    assert status["state"] == "succeeded" and status["exit_code"] == 0
    assert status["outputs"]["parquet_state"] == "written"
    [solve] = status["solves"]
    assert (solve["kind"], solve["state"], solve["acceptance"]) == ("benders", "accepted", "converged")
    final = solve["benders"]["final_solve"]
    assert final["accepted"] is True and final["model_status"] == "kOptimal"
    slack = 1e-5 * abs(ub)  # the log prints the bounds to 6 digits
    assert lb - slack <= final["objective"] <= ub + slack
    assert "W-0572" not in bend.codes()
    assert "--benders-final-solve off" in bend.out


# ---------------------------------------------------------------------------------------------------------------
# T4: Benders last, final solve off
# ---------------------------------------------------------------------------------------------------------------


def test_benders_last_with_the_final_solve_off_writes_no_table(runs):
    off = runs["ti_off"]
    _ok(off)
    assert "W-0570" in off.codes(), off.out
    assert not [p for p in off.csv.glob("*.csv") if p.name != "timings.csv"] if off.csv.exists() else True
    status = off.status()
    assert status["outputs"]["parquet_state"] == "skipped"
    [solve] = status["solves"]
    assert solve["acceptance"] == "converged" and "final_solve" not in (solve["benders"] or {})
    assert "final solve: the whole model" not in off.out


# ---------------------------------------------------------------------------------------------------------------
# T7: Benders invest solve, then a dispatch solve
# ---------------------------------------------------------------------------------------------------------------


def test_benders_chain_writes_the_tables_of_the_monolithic_chain(runs):
    bend, mono = runs["inv_benders"], runs["inv_mono"]
    _ok(bend)
    _ok(mono)
    atol = _tolerance(bend, sum(s["objective"] for s in mono.status()["solves"]))
    for name in COST_TABLES:
        _assert_table_close(bend, mono, name, atol)
    for name in CAPACITY_TABLES:
        _assert_table_close(bend, mono, name, max(atol, 1e-6))
    present = [n for n in INVEST_MARGINAL if (mono.csv / n).is_file()]
    assert present and all((bend.csv / n).is_file() for n in present), present
    assert bend.codes() == mono.codes(), (bend.codes(), mono.codes())
    # Flows: totals per period only (two optimal dispatches may differ).
    for name in ("unit__outputNode__d.csv", "connection__d.csv"):
        def totals(run: Run) -> dict[str, float]:
            out: dict[str, float] = {}
            for row in run.table(name):
                period = row.get("period", "")
                out[period] = out.get(period, 0.0) + sum(abs(_numeric(v) or 0.0) for k, v in row.items()
                                                         if k not in ("solve", "period"))
            return out
        tb, tm = totals(bend), totals(mono)
        assert tb.keys() == tm.keys()
        for period in tm:
            assert tb[period] == pytest.approx(tm[period], rel=1e-3), (name, period)


def test_benders_chain_records_the_final_solve(runs):
    bend = runs["inv_benders"]
    _ok(bend)
    status = bend.status()
    assert status["outputs"]["parquet_state"] == "written"
    entry = next(s for s in status["solves"] if s["name"] == INVEST_SOLVE)
    assert entry["acceptance"] == "converged" and entry["benders"]["final_solve"]["accepted"] is True


# ---------------------------------------------------------------------------------------------------------------
# T8: the Benders solve does not converge
# ---------------------------------------------------------------------------------------------------------------


def test_not_converged_benders_runs_the_final_solve_and_stays_not_converged(runs):
    one = runs["ti_one_iter"]
    _ok(one)
    assert "DID NOT CONVERGE" in one.out
    assert "did NOT meet its convergence tolerance" in one.out, "the non-convergence banner did not fire"
    status = one.status()
    [solve] = status["solves"]
    assert (solve["state"], solve["acceptance"]) == ("accepted", "not_converged")
    assert status["verdict"] == "not_converged"
    assert solve["benders"]["final_solve"]["accepted"] is True
    assert status["outputs"]["parquet_state"] == "written"
    assert (one.csv / "summary_solve.csv").is_file()
    lb, ub = one.bounds()
    assert solve["benders"]["final_solve"]["objective"] <= ub + 1e-5 * abs(ub)


# ---------------------------------------------------------------------------------------------------------------
# T9 (last-solve half): the final solve is not accepted
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture
def restore_env():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def test_final_solve_not_accepted_keeps_the_benders_result(restore_env, monkeypatch, tmp_path_factory, capsys,
                                                          caplog):
    import dataclasses

    from flextool.cli import cmd_run_flextool
    from flextool.engine.solve import _orchestration

    real = _orchestration.classify_acceptance
    seen = []

    def reject(sol, **kwargs):
        acc = real(sol, **kwargs)
        seen.append(kwargs.get("solve_name"))
        return dataclasses.replace(acc, accepted=False, near_optimal=False, model_status="kInfeasible",
                                   message="patched: not accepted")

    monkeypatch.setattr(_orchestration, "classify_acceptance", reject)
    url, scenario = make_db(tmp_path_factory, alt="p287p2_reject", values=[], base_chain=TRADE_CHAIN,
                            fixture_json=TRADE_JSON)
    root = tmp_path_factory.mktemp("p287p2_reject_run")
    os.environ.pop("FLEXTOOL_BENDERS_FINAL_SOLVE", None)
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", url, "--scenario-name", scenario, "--work-folder", str(root / "work"),
        "--output-location", str(root / "out"), "--write-methods", "csv", "parquet",
    ])
    code = 0
    try:
        cmd_run_flextool.main()
    except SystemExit as exc:
        code = exc.code or 0
    out = capsys.readouterr()
    folder = root / "out" / "output_parquet" / scenario
    text = out.out + out.err + "\n".join(r.getMessage() for r in caplog.records)
    assert code == 0, text
    assert seen == [TRADE_SOLVE], seen
    assert "failed (kInfeasible) [W-0572]" in text, text
    csv_dir = root / "out" / "output_csv" / scenario
    assert not (csv_dir.exists() and [p for p in csv_dir.glob("*.csv") if p.name != "timings.csv"])
    status = run_record.read_status_file(folder)
    assert status["state"] == "succeeded" and status["outputs"]["parquet_state"] == "skipped"
    [solve] = status["solves"]
    assert (solve["state"], solve["acceptance"]) == ("accepted", "converged")
    assert solve["benders"]["final_solve"] == {"accepted": False, "model_status": "kInfeasible",
                                               "objective": solve["benders"]["final_solve"]["objective"]}


def test_final_solve_not_accepted_keeps_the_benders_step_and_handoff(monkeypatch, tmp_path_factory):
    import dataclasses

    from flextool.engine import run_chain_from_db
    from flextool.engine.solve import _orchestration

    real = _orchestration.classify_acceptance
    monkeypatch.setattr(_orchestration, "classify_acceptance", lambda sol, **kw: dataclasses.replace(
        real(sol, **kw), accepted=False, near_optimal=False, model_status="kInfeasible", message="patched"))
    monkeypatch.delenv("FLEXTOOL_BENDERS_FINAL_SOLVE", raising=False)
    url, scenario = make_db(tmp_path_factory, alt="p287p2_reject_steps", values=[], base_chain=TRADE_CHAIN,
                            fixture_json=TRADE_JSON)
    steps = run_chain_from_db(url, scenario, work_folder=tmp_path_factory.mktemp("p287p2_reject_steps_w"),
                              keep_solutions=True)
    [step] = steps.values()
    assert step.is_benders and step.benders_results == "none" and step.optimal is True
    assert isinstance(step.solution, _orchestration.SnapshotSolution)
    assert step.handoff is not None and not step.handoff.is_empty()
