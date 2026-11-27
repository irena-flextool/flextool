"""Item 287 P2/P3: a Benders solve ends with its final solve, and the run writes the result tables of a monolithic
run; without the final solve result a chained Benders solve writes its investment-only results (P3).

After the Benders loop, the whole model of the solve is solved once with every investment decision fixed to the
Benders incumbent (``_run_one`` in ``_orchestration._cascade``), on the normal monolithic path.  Oracles are relational: each
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
* **T1** (P3) the chained Benders solve with ``--benders-final-solve off``: its investments, their costs and its
  ``summary_solve`` row equal the monolithic chain's within the gap; ``existing + invested - divested = total``;
  no investment marginal table, W-0571 once.
* **T2** (P3) the same with built capacity retired (build 100 MW, retire 40): the retirement and salvage lines.
* **T9** (P3, the chained half) the final solve of the chained Benders solve not accepted: W-0572 and the same
  investment-only results as T1.

* **M1** (review) the final solve's solver path raises: a solve outcome (``SolveOutcomeError`` /
  ``FlexToolSolveOutcomeError``) or a ``MemoryError`` (writing the MPS, building the model) is W-0572 and exit 0;
  a licence, configuration or internal error fails the run (non-zero exit, no W-0572).

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
from tests.engine.test_benders_final_pin import RETIRE_BUILT

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
    # P3: the chained Benders solve with its final solve off writes its investment-only results.
    "inv_off": (LH2_JSON, INVEST_CHAIN, [], ["--benders-final-solve", "off"]),
    "ret_off": (LH2_JSON, INVEST_CHAIN, RETIRE_BUILT, ["--benders-final-solve", "off"]),
    "ret_mono": (LH2_JSON, INVEST_CHAIN, RETIRE_BUILT + _none(INVEST_SOLVE), []),
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
    with ThreadPoolExecutor(max_workers=min(len(jobs), 9)) as pool:
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
    assert "W-0571" not in off.codes(), "a Benders last solve with the final solve off writes no investment slice"
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

    real = _orchestration._cascade.classify_acceptance
    seen = []

    def reject(sol, **kwargs):
        acc = real(sol, **kwargs)
        seen.append(kwargs.get("solve_name"))
        return dataclasses.replace(acc, accepted=False, near_optimal=False, model_status="kInfeasible",
                                   message="patched: not accepted")

    monkeypatch.setattr(_orchestration._cascade, "classify_acceptance", reject)
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

    real = _orchestration._cascade.classify_acceptance
    monkeypatch.setattr(_orchestration._cascade, "classify_acceptance", lambda sol, **kw: dataclasses.replace(
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


# ---------------------------------------------------------------------------------------------------------------
# P3, T1 / T2: a chained Benders solve with its final solve off writes its investment-only results
# ---------------------------------------------------------------------------------------------------------------


def _assert_investment_only_results(run: Run, mono: Run, *, capacities: bool = True) -> None:
    """*run* (a chained Benders solve without a final solve result) against the monolithic chain *mono*: the cost
    and summary tables within the gap, the capacity tables (*capacities*: the fixture's optimum is unique, invariant
    4 of the design), the capacity rows add up, no investment marginal table."""
    _ok(run)
    _ok(mono)
    atol = _tolerance(run, sum(s["objective"] for s in mono.status()["solves"]))
    for name in COST_TABLES:
        _assert_table_close(run, mono, name, atol)
    if capacities:
        for name in CAPACITY_TABLES:
            _assert_table_close(run, mono, name, max(atol, 1e-6))
    assert any(row and row[0] == INVEST_SOLVE for row in _cells(run, "summary_solve.csv")), "no summary_solve row"
    rows = run.table("unit_capacity__d.csv")
    assert any((_numeric(r["invested"]) or 0.0) > 0 for r in rows), "no investment in unit_capacity__d"
    assert {r["solve"] for r in rows} == {INVEST_SOLVE}
    for name in CAPACITY_TABLES:
        for r in run.table(name):
            if _numeric(r.get("in_construction", "")):
                continue
            got = (_numeric(r["existing"]) or 0.0) + (_numeric(r["invested"]) or 0.0) - (
                _numeric(r["divested"]) or 0.0)
            assert got == pytest.approx(_numeric(r["total"]) or 0.0, abs=1e-6), (name, r)
    present = [n for n in INVEST_MARGINAL if (mono.csv / n).is_file()]
    assert present, "the monolithic chain writes investment marginal values"
    assert not [n for n in INVEST_MARGINAL if (run.csv / n).is_file()]
    assert not (run.parquet / "dual_invest_effective_unit_d_e.parquet").exists()
    assert run.status()["outputs"]["parquet_state"] == "written"


def test_chained_benders_with_the_final_solve_off_writes_its_investments_and_costs(runs):
    off, mono = runs["inv_off"], runs["inv_mono"]
    _assert_investment_only_results(off, mono)
    assert off.out.count("[W-0571]") == 1, off.out
    assert "W-0570" not in off.codes() and "W-0572" not in off.codes()
    assert off.codes() - {"W-0571"} == mono.codes(), (off.codes(), mono.codes())
    assert "final solve: the whole model" not in off.out


def test_chained_benders_with_the_final_solve_off_writes_its_retirements(runs):
    off, mono = runs["ret_off"], runs["ret_mono"]
    # The salvage makes the objective large and the Benders gap loose for wind_B (an investment of a few M CUR
    # within the gap): the capacities are compared on the retiring unit only.
    _assert_investment_only_results(off, mono, capacities=False)
    [wind_off] = [r for r in off.table("unit_capacity__d.csv") if r["unit"] == "wind_A"]
    [wind_mono] = [r for r in mono.table("unit_capacity__d.csv") if r["unit"] == "wind_A"]
    assert (_numeric(wind_mono["invested"]), _numeric(wind_mono["divested"])) == (100.0, 40.0), wind_mono
    for col in ("existing", "invested", "divested", "total"):
        assert _numeric(wind_off[col]) == pytest.approx(_numeric(wind_mono[col]), rel=1e-6), (col, wind_off)
    lines_off = {r[0]: _numeric(r[1]) for r in _cells(off, "costs_discounted.csv")[1:] if len(r) > 1}
    lines_mono = {r[0]: _numeric(r[1]) for r in _cells(mono, "costs_discounted.csv")[1:] if len(r) > 1}
    for line in ("retirement", "salvage"):
        assert lines_mono[line], (line, lines_mono)
        assert lines_off[line] == pytest.approx(lines_mono[line], rel=1e-6), line


# ---------------------------------------------------------------------------------------------------------------
# P3, T9 (the chained half): the final solve of a chained Benders solve is not accepted
# ---------------------------------------------------------------------------------------------------------------


def test_chained_final_solve_not_accepted_writes_the_investment_only_results(runs, restore_env, monkeypatch,
                                                                             tmp_path_factory, capsys, caplog):
    import dataclasses

    from flextool.cli import cmd_run_flextool
    from flextool.engine.solve import _orchestration

    real = _orchestration._cascade.classify_acceptance

    def reject(sol, **kwargs):
        acc = real(sol, **kwargs)
        if kwargs.get("solve_name") != INVEST_SOLVE:
            return acc
        return dataclasses.replace(acc, accepted=False, near_optimal=False, model_status="kInfeasible",
                                   message="patched: not accepted")

    monkeypatch.setattr(_orchestration._cascade, "classify_acceptance", reject)
    url, scenario = make_db(tmp_path_factory, alt="p287p3_reject", values=[], base_chain=INVEST_CHAIN,
                            fixture_json=LH2_JSON)
    root = tmp_path_factory.mktemp("p287p3_reject_run")
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
    text = out.out + out.err + "\n".join(r.getMessage() for r in caplog.records)
    assert code == 0, text
    assert "failed (kInfeasible) [W-0572]" in text, text
    assert "[W-0571]" not in text
    rejected = Run("reject", scenario, root, subprocess.CompletedProcess([], 0, stdout=text, stderr=""))
    _assert_investment_only_results(rejected, runs["inv_mono"])
    entry = next(s for s in rejected.status()["solves"] if s["name"] == INVEST_SOLVE)
    assert entry["acceptance"] == "converged" and entry["benders"]["final_solve"]["accepted"] is False



# ---------------------------------------------------------------------------------------------------------------
# Review M1: only a genuine solve outcome (or running out of memory) is a failed final solve; a configuration,
# licence or internal error fails the run
# ---------------------------------------------------------------------------------------------------------------


def _cli_in_process(alt: str, monkeypatch, tmp_path_factory, capsys, caplog) -> tuple[int, str]:
    """``cmd_run_flextool`` on the trade-invest chain (one Benders solve, the last), in this process: the exit
    code and the console + log text."""
    from flextool.cli import cmd_run_flextool

    url, scenario = make_db(tmp_path_factory, alt=alt, values=[], base_chain=TRADE_CHAIN, fixture_json=TRADE_JSON)
    root = tmp_path_factory.mktemp(f"{alt}_run")
    monkeypatch.delenv("FLEXTOOL_BENDERS_FINAL_SOLVE", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "cmd_run_flextool", url, "--scenario-name", scenario, "--work-folder", str(root / "work"),
        "--output-location", str(root / "out"), "--write-methods", "csv",
    ])
    code = 0
    try:
        cmd_run_flextool.main()
    except SystemExit as exc:
        code = exc.code or 0
    out = capsys.readouterr()
    return code, out.out + out.err + "\n".join(r.getMessage() for r in caplog.records)


def _final_solve_raises(monkeypatch, exc: BaseException) -> list[str]:
    """The final solve's ``run_one_solve`` raises *exc* (the Benders loop does not go through it)."""
    from flextool.engine.solve import _solver_dispatch

    calls: list[str] = []

    def failing(pb, cfg, **kwargs):
        calls.append(kwargs.get("solve_name"))
        raise exc

    monkeypatch.setattr(_solver_dispatch, "run_one_solve", failing)
    return calls


@pytest.mark.parametrize("make_exc", [
    lambda: __import__("flextool.engine.solve._subprocess_solve", fromlist=["x"]).SolveOutcomeError(
        "'gurobi' reports that solve 'lh2_trade_invest' is infeasible; there is no solution to use."),
    lambda: __import__("flextool.engine.solve._solver_dispatch", fromlist=["x"]).FlexToolSolveOutcomeError(
        "Solver 'gurobi' failed for solve 'lh2_trade_invest': did not stop at its own time limit"),
], ids=["outcome", "outcome_via_dispatch"])
def test_final_solve_without_a_solution_is_w0572_and_exit_0(make_exc, restore_env, monkeypatch, tmp_path_factory,
                                                             capsys, caplog):
    calls = _final_solve_raises(monkeypatch, make_exc())
    code, text = _cli_in_process("m1_outcome", monkeypatch, tmp_path_factory, capsys, caplog)
    assert calls == [TRADE_SOLVE], calls
    assert code == 0, text
    assert "failed (the solver gave no solution) [W-0572]" in text, text


@pytest.mark.parametrize("make_exc", [
    lambda: __import__("flextool.engine.solve._solver_dispatch", fromlist=["x"]).FlexToolUserError(
        "Solver 'gurobi' subprocess failed a license check.  Details: no licence."),
    lambda: RuntimeError("'gurobi' CLI binary ('gurobi_cl') was not found on $PATH"),
    lambda: ValueError("an internal error"),
], ids=["licence", "configuration", "internal"])
def test_final_solve_configuration_or_internal_error_fails_the_run(make_exc, restore_env, monkeypatch,
                                                                    tmp_path_factory, capsys, caplog):
    calls = _final_solve_raises(monkeypatch, make_exc())
    code, text = _cli_in_process("m1_config", monkeypatch, tmp_path_factory, capsys, caplog)
    assert calls == [TRADE_SOLVE], calls
    assert code != 0, text
    assert "[W-0572]" not in text, text


def test_final_solve_out_of_memory_writing_the_mps_is_w0572(restore_env, monkeypatch, tmp_path_factory, capsys,
                                                              caplog):
    _final_solve_raises(monkeypatch, MemoryError())
    code, text = _cli_in_process("m1_mem_write", monkeypatch, tmp_path_factory, capsys, caplog)
    assert code == 0, text
    assert "failed (out of memory; see --benders-final-solve off) [W-0572]" in text, text


def test_final_solve_out_of_memory_building_the_model_is_w0572(restore_env, monkeypatch, tmp_path_factory, capsys,
                                                                 caplog):
    from flextool.engine.model import core

    real = core.build_flextool
    calls: list[str] = []

    def failing(pb, data, **kwargs):
        if kwargs.get("solve_name") == TRADE_SOLVE:  # the final solve (the Benders loop names no solve)
            calls.append(TRADE_SOLVE)
            raise MemoryError("Unable to allocate 40.0 GiB")
        return real(pb, data, **kwargs)

    monkeypatch.setattr(core, "build_flextool", failing)
    code, text = _cli_in_process("m1_mem_build", monkeypatch, tmp_path_factory, capsys, caplog)
    assert calls == [TRADE_SOLVE], calls
    assert code == 0, text
    assert "failed (out of memory; see --benders-final-solve off) [W-0572]" in text, text
