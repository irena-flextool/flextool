#!/usr/bin/env python3
"""Benchmark harness for the LP-scaling project.

Runs one or all of the four benchmark scenarios, captures a fixed set of
numerical measurements (objective, matrix/coefficient ranges, nnz, solve
wall-time, slack totals), and writes them as JSON baselines in
``benchmarks/scaling/baseline/<scenario>.json``.

Typical usage::

    # Generate scenario DBs (once, or whenever generator scripts change)
    python benchmarks/scaling/run_benchmarks.py --generate

    # Run all scenarios and (re)write baselines
    python benchmarks/scaling/run_benchmarks.py --write-baseline

    # Compare a current run against an existing baseline
    python benchmarks/scaling/run_benchmarks.py --compare benchmarks/scaling/baseline/composite.json

    # Run just one scenario
    python benchmarks/scaling/run_benchmarks.py --scenario composite --write-baseline

Scenarios live under ``benchmarks/scaling/scenarios/<name>/``; each has a
``generate.py`` that writes ``input.sqlite`` next to itself and declares a
module-level ``SCENARIO_NAME`` naming the scenario inside that DB that the
harness should execute.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH_ROOT = Path(__file__).resolve().parent
SCENARIOS_DIR = BENCH_ROOT / "scenarios"
BASELINE_DIR = BENCH_ROOT / "baseline"
WORK_DIR = BENCH_ROOT / "work"

SCENARIOS: list[str] = [
    "small_building",
    "medium_national",
    "continental",
    "composite",
]


# ---------------------------------------------------------------------------
# Scenario generation
# ---------------------------------------------------------------------------


def load_scenario_module(name: str):
    script = SCENARIOS_DIR / name / "generate.py"
    if not script.exists():
        raise FileNotFoundError(f"No generator for scenario {name!r}: {script}")
    spec = importlib.util.spec_from_file_location(f"gen_{name}", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def generate_scenario(name: str) -> tuple[Path, str]:
    mod = load_scenario_module(name)
    rc = mod.main()
    if rc != 0:
        raise RuntimeError(f"Generator for {name!r} exited with {rc}")
    db = SCENARIOS_DIR / name / "input.sqlite"
    if not db.exists():
        raise RuntimeError(f"Generator for {name!r} did not produce {db}")
    scen_name = getattr(mod, "SCENARIO_NAME")
    return db, scen_name


def ensure_scenario(name: str) -> tuple[Path, str]:
    db = SCENARIOS_DIR / name / "input.sqlite"
    mod = load_scenario_module(name)
    scen_name = getattr(mod, "SCENARIO_NAME")
    if not db.exists():
        print(f"[{name}] input.sqlite missing — regenerating")
        return generate_scenario(name)
    return db, scen_name


# ---------------------------------------------------------------------------
# HiGHS log parsing
# ---------------------------------------------------------------------------


_RANGE_BLOCK = re.compile(
    r"Coefficient ranges:\s*\n"
    r"\s*Matrix\s*\[([^,]+),\s*([^\]]+)\]\s*\n"
    r"\s*Cost\s*\[([^,]+),\s*([^\]]+)\]\s*\n"
    r"\s*Bound\s*\[([^,]+),\s*([^\]]+)\]\s*\n"
    r"\s*RHS\s*\[([^,]+),\s*([^\]]+)\]",
)

_NNZ_INITIAL = re.compile(
    r"(?:LP|MIP|QP)\s+flextool\s+has\s+(\d+)\s+rows;\s+(\d+)\s+cols;\s+(\d+)\s+nonzeros"
)

# Post-presolve line — HiGHS 1.14 writes either
#   "<rows> rows, <cols> cols, <nnz> nonzeros <X>s"        (LP)
# or "Presolve reductions: rows <r>(-d); columns <c>(-d); nonzeros <n>(-d)" (MIP)
_NNZ_POST_PRESOLVE_MIP = re.compile(
    r"Presolve reductions:\s+rows\s+(\d+).*?;\s+columns\s+(\d+).*?;\s+nonzeros\s+(\d+)"
)
_NNZ_POST_PRESOLVE_LP = re.compile(
    r"Presolve reductions:\s+rows\s+(\d+).*?columns\s+(\d+).*?nonzeros\s+(\d+)"
)

_OBJECTIVE = re.compile(
    r"Objective value\s*:\s*([+\-0-9.eE]+)|"
    r"Primal bound\s+([+\-0-9.eE]+)"
)


def _parse_float(s: str) -> float:
    # HiGHS uses forms like "4e-01", "2e+03", "0e+00" — robustly float().
    return float(s.strip())


def parse_highs_log(log_text: str) -> dict:
    """Extract numerical fields from a HiGHS log."""
    result: dict = {
        "matrix_range": None,
        "cost_range": None,
        "bound_range": None,
        "rhs_range": None,
        "rows_initial": None,
        "cols_initial": None,
        "nnz_initial": None,
        "rows_postpresolve": None,
        "cols_postpresolve": None,
        "nnz_postpresolve": None,
    }

    m = _RANGE_BLOCK.search(log_text)
    if m:
        mn, mx, cn, cx, bn, bx, rn, rx = m.groups()
        result["matrix_range"] = [_parse_float(mn), _parse_float(mx)]
        result["cost_range"] = [_parse_float(cn), _parse_float(cx)]
        result["bound_range"] = [_parse_float(bn), _parse_float(bx)]
        result["rhs_range"] = [_parse_float(rn), _parse_float(rx)]

    m = _NNZ_INITIAL.search(log_text)
    if m:
        result["rows_initial"] = int(m.group(1))
        result["cols_initial"] = int(m.group(2))
        result["nnz_initial"] = int(m.group(3))

    m = _NNZ_POST_PRESOLVE_MIP.search(log_text)
    if m:
        result["rows_postpresolve"] = int(m.group(1))
        result["cols_postpresolve"] = int(m.group(2))
        result["nnz_postpresolve"] = int(m.group(3))

    return result


_TIME_LINE = re.compile(r"--- Solver \(HiGHS\):\s+([\d.]+)\s+seconds")


def parse_solve_time_from_stdout(stdout: str) -> float | None:
    m = _TIME_LINE.search(stdout)
    if not m:
        return None
    return float(m.group(1))


# ---------------------------------------------------------------------------
# MPS matrix range (fallback + verification)
# ---------------------------------------------------------------------------


def mps_matrix_range(mps_path: Path) -> tuple[float, float] | None:
    """Scan MPS file COLUMNS section for absolute non-zero coefficient range."""
    if not mps_path.exists():
        return None
    mn, mx = float("inf"), 0.0
    in_columns = False
    in_marker_int = False
    with mps_path.open("r") as f:
        for line in f:
            if not in_columns:
                if line.startswith("COLUMNS"):
                    in_columns = True
                continue
            if line.startswith("RHS") or line.startswith("RANGES") or line.startswith("BOUNDS") or line.startswith("ENDATA"):
                break
            if "'MARKER'" in line:
                in_marker_int = "'INTORG'" in line
                continue
            toks = line.split()
            # COLUMNS layout: colname rowname value [rowname value]
            # values are at positions 2 and (optionally) 4
            for idx in (2, 4):
                if len(toks) > idx:
                    try:
                        v = abs(float(toks[idx]))
                    except ValueError:
                        continue
                    if v > 0.0:
                        if v < mn:
                            mn = v
                        if v > mx:
                            mx = v
    if mn == float("inf"):
        return None
    return (mn, mx)


# ---------------------------------------------------------------------------
# Slack totals (read from parquet outputs)
# ---------------------------------------------------------------------------


SLACK_VARS: list[str] = [
    "vq_state_up",
    "vq_state_down",
    "vq_reserve",
    "vq_inertia",
    "vq_non_synchronous",
    "vq_capacity_margin",
    "vq_state_up_group",
]


def compute_slack_totals(output_raw_dir: Path) -> dict[str, float]:
    """Sum absolute values across all parquet shards of each slack variable.

    Reads ``output_raw/<slack>__*.parquet`` files produced by the HiGHS →
    parquet extractor. Missing files -> 0.0.
    """
    import pandas as pd
    from flextool.lean_parquet import read_lean_parquet

    totals: dict[str, float] = {v: 0.0 for v in SLACK_VARS}
    if not output_raw_dir.exists():
        return totals
    for slack in SLACK_VARS:
        shards = list(output_raw_dir.glob(f"{slack}__*.parquet"))
        total = 0.0
        for shard in shards:
            try:
                df = read_lean_parquet(shard)
            except Exception:
                continue
            # Empty frames have shape (N, 0)
            if df.empty or df.shape[1] == 0:
                continue
            # Values are numeric; sum absolute values.
            try:
                total += float(df.abs().sum().sum())
            except Exception:
                pass
        totals[slack] = total
    return totals


# ---------------------------------------------------------------------------
# Objective value
# ---------------------------------------------------------------------------


def read_v_obj(output_raw_dir: Path) -> float | None:
    """Read the v_obj parquet; there should be exactly one per solve."""
    import pandas as pd
    from flextool.lean_parquet import read_lean_parquet

    if not output_raw_dir.exists():
        return None
    shards = list(output_raw_dir.glob("v_obj__*.parquet"))
    if not shards:
        return None
    # Sum across solves (for rolling / nested; here should be just one).
    total = 0.0
    seen = False
    for shard in shards:
        try:
            df = read_lean_parquet(shard)
            total += float(df.to_numpy().sum())
            seen = True
        except Exception:
            continue
    return total if seen else None


# ---------------------------------------------------------------------------
# Git commit lookup
# ---------------------------------------------------------------------------


def git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        )
        return out.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    scenario: str
    scenario_name_in_db: str
    timestamp: str
    git_commit: str
    objective: float | None
    matrix_range: list[float] | None
    cost_range: list[float] | None
    bound_range: list[float] | None
    rhs_range: list[float] | None
    rows_initial: int | None
    cols_initial: int | None
    nnz_initial: int | None
    rows_postpresolve: int | None
    cols_postpresolve: int | None
    nnz_postpresolve: int | None
    matrix_range_from_mps: list[float] | None
    solve_wall_time_s: float | None
    total_wall_time_s: float
    slack_totals: dict[str, float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def run_scenario(name: str, keep_work: bool = False) -> BenchmarkResult:
    db, scen_name = ensure_scenario(name)
    scenario_work = WORK_DIR / name
    if scenario_work.exists():
        shutil.rmtree(scenario_work)
    scenario_work.mkdir(parents=True, exist_ok=True)

    # Ephemeral output-info DB (harness does not inspect it)
    out_info_src = REPO_ROOT / "templates" / "output_info.sqlite"
    out_info = scenario_work / "output_info.sqlite"
    shutil.copy2(out_info_src, out_info)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "run_flextool.py"),
        f"sqlite:///{db.resolve()}",
        f"sqlite:///{out_info.resolve()}",
        "--scenario-name",
        scen_name,
        "--output-location",
        str(scenario_work),
        "--work-folder",
        str(scenario_work),
        "--write-methods",
        "parquet",
    ]
    print(f"[{name}] running scenario={scen_name} ...")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    total_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(
            f"[{name}] flextool run failed with exit code {proc.returncode}"
        )

    log_path = scenario_work / "HiGHS.log"
    log_text = log_path.read_text() if log_path.exists() else ""
    parsed = parse_highs_log(log_text)
    mps_path = scenario_work / "flextool.mps"
    mps_range = mps_matrix_range(mps_path)

    slack_totals = compute_slack_totals(scenario_work / "output_raw")
    objective = read_v_obj(scenario_work / "output_raw")
    solve_time = parse_solve_time_from_stdout(proc.stdout)

    result = BenchmarkResult(
        scenario=name,
        scenario_name_in_db=scen_name,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=git_commit(),
        objective=objective,
        matrix_range=parsed["matrix_range"],
        cost_range=parsed["cost_range"],
        bound_range=parsed["bound_range"],
        rhs_range=parsed["rhs_range"],
        rows_initial=parsed["rows_initial"],
        cols_initial=parsed["cols_initial"],
        nnz_initial=parsed["nnz_initial"],
        rows_postpresolve=parsed["rows_postpresolve"],
        cols_postpresolve=parsed["cols_postpresolve"],
        nnz_postpresolve=parsed["nnz_postpresolve"],
        matrix_range_from_mps=list(mps_range) if mps_range else None,
        solve_wall_time_s=solve_time,
        total_wall_time_s=round(total_wall, 3),
        slack_totals=slack_totals,
    )

    if not keep_work:
        # Keep HiGHS.log and flextool.mps for debuggability, drop heavy parquet/CSV.
        for d in ("output_raw", "solve_data", "input", "output_plots"):
            p = scenario_work / d
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
    return result


# ---------------------------------------------------------------------------
# Baseline I/O and compare
# ---------------------------------------------------------------------------


def baseline_path(name: str) -> Path:
    return BASELINE_DIR / f"{name}.json"


def write_baseline(result: BenchmarkResult) -> Path:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    p = baseline_path(result.scenario)
    p.write_text(result.to_json() + "\n")
    return p


def _pct(current: float | None, baseline: float | None) -> str:
    if current is None or baseline is None:
        return "n/a"
    if baseline == 0:
        return "∞" if current != 0 else "0%"
    return f"{(current - baseline) / baseline * 100:+.4f}%"


def compare(result: BenchmarkResult, baseline_file: Path) -> bool:
    """Print deltas between a current run and a saved baseline.

    Returns True if baselines differ materially, False otherwise.
    """
    baseline = json.loads(baseline_file.read_text())
    current = asdict(result)
    differ = False

    def cmp(key: str, flag_diff: bool = True):
        nonlocal differ
        b = baseline.get(key)
        c = current.get(key)
        if flag_diff and b != c:
            differ = True
        print(f"  {key}: baseline={b!r}  current={c!r}  delta={_pct(c, b) if isinstance(c, (int, float)) and isinstance(b, (int, float)) else ''}")

    print(f"[{result.scenario}] compare vs {baseline_file.name}")
    print(f"  baseline commit: {baseline.get('git_commit')}")
    print(f"  current  commit: {current.get('git_commit')}")
    cmp("objective")
    cmp("matrix_range")
    cmp("matrix_range_from_mps")
    cmp("cost_range")
    cmp("bound_range")
    cmp("rhs_range")
    cmp("rows_initial")
    cmp("cols_initial")
    cmp("nnz_initial")
    cmp("rows_postpresolve")
    cmp("cols_postpresolve")
    cmp("nnz_postpresolve")
    # Wall time is jittery; report but do not flag as a regression by itself.
    cmp("solve_wall_time_s", flag_diff=False)
    for slack in SLACK_VARS:
        b = baseline.get("slack_totals", {}).get(slack)
        c = current.get("slack_totals", {}).get(slack)
        if b != c:
            differ = True
        print(f"  slack_totals.{slack}: baseline={b!r}  current={c!r}  delta={_pct(c, b)}")
    return differ


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Run just one scenario (default: all four)",
    )
    ap.add_argument(
        "--generate",
        action="store_true",
        help="Regenerate scenario input.sqlite files before running",
    )
    ap.add_argument(
        "--write-baseline",
        action="store_true",
        help="After running, write baseline JSON(s) to benchmarks/scaling/baseline/",
    )
    ap.add_argument(
        "--compare",
        type=Path,
        help="Compare current run against this baseline JSON (only valid with --scenario)",
    )
    ap.add_argument(
        "--keep-work",
        action="store_true",
        help="Retain the per-scenario work folder (parquet, CSVs) for inspection",
    )
    args = ap.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS

    if args.generate:
        for s in scenarios:
            print(f"[{s}] generating input.sqlite")
            generate_scenario(s)

    if args.compare is not None and not args.scenario:
        ap.error("--compare requires --scenario")

    differ_any = False
    results: list[BenchmarkResult] = []
    for s in scenarios:
        res = run_scenario(s, keep_work=args.keep_work)
        results.append(res)
        print(f"[{s}] DONE")
        print(f"  objective        : {res.objective}")
        print(f"  matrix_range     : {res.matrix_range}")
        print(f"  matrix_range_mps : {res.matrix_range_from_mps}")
        print(f"  rows/cols/nnz    : {res.rows_initial}/{res.cols_initial}/{res.nnz_initial}")
        print(f"  solve_wall_time  : {res.solve_wall_time_s}")
        print(f"  slack_totals     : {res.slack_totals}")

        if args.write_baseline:
            p = write_baseline(res)
            print(f"[{s}] wrote baseline {p}")

        if args.compare is not None:
            differ = compare(res, args.compare)
            if differ:
                differ_any = True

    return 2 if differ_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
