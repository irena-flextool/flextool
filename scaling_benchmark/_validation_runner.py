#!/usr/bin/env python3
"""Agent 11 validation driver.

Runs each of the four benchmark scenarios under three configurations:

1. ``default`` — no ``--auto-scale``, ``FLEXTOOL_FORCE_ROW_SCALING`` unset.
2. ``auto_scale`` — ``--auto-scale`` on.
3. ``force_row_scaling`` — ``FLEXTOOL_FORCE_ROW_SCALING=1``.

Captures objective, matrix/cost/bound ranges, slack totals, solve time,
and the one-line summary from the scaling_report.txt for each
(scenario, mode) cell.  Emits a single JSON document so that
``VALIDATION_REPORT.md`` can be rendered from data rather than hand-pasted.

Run from the repo root with the venv activated::

    python scaling_benchmark/_validation_runner.py > scaling_benchmark/_validation_data.json

This helper is for the Agent 11 validation step only and is safe to delete
afterwards — it lives under ``scaling_benchmark/`` purely so relative paths
are identical to the other benchmark scripts.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent

SCENARIOS = ["small_building", "medium_national", "continental", "composite"]
MODES = ["default", "auto_scale", "force_row_scaling"]


def _parse_scaling_report_summary(report_path: Path) -> str:
    """Return the summary line (section 9) or an empty string."""
    if not report_path.exists():
        return ""
    text = report_path.read_text(errors="replace")
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith("-- 9. Summary"):
            # Next non-blank line holds the one-sentence verdict.
            for ln2 in lines[i + 1 :]:
                if ln2.strip():
                    return ln2.strip()
    return ""


def run_one(scenario: str, mode: str) -> dict:
    import importlib.util

    spec_file = BENCH / "scenarios" / scenario / "generate.py"
    spec = importlib.util.spec_from_file_location(f"gen_{scenario}", spec_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scen_name = getattr(mod, "SCENARIO_NAME")
    db = BENCH / "scenarios" / scenario / "input.sqlite"

    work = BENCH / "work_validation" / f"{scenario}__{mode}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    out_info = work / "output_info.sqlite"
    shutil.copy2(REPO / "templates" / "output_info.sqlite", out_info)

    cmd = [
        sys.executable,
        str(REPO / "run_flextool.py"),
        f"sqlite:///{db.resolve()}",
        f"sqlite:///{out_info.resolve()}",
        "--scenario-name",
        scen_name,
        "--output-location",
        str(work),
        "--work-folder",
        str(work),
        "--write-methods",
        "parquet",
    ]
    env = os.environ.copy()
    if mode == "auto_scale":
        cmd.append("--auto-scale")
        env.pop("FLEXTOOL_FORCE_ROW_SCALING", None)
    elif mode == "force_row_scaling":
        env["FLEXTOOL_FORCE_ROW_SCALING"] = "1"
    else:
        env.pop("FLEXTOOL_FORCE_ROW_SCALING", None)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env)
    total_wall = time.perf_counter() - t0
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"{scenario}/{mode} failed rc={proc.returncode}")

    # Import harness helpers via sys.path trick.
    sys.path.insert(0, str(BENCH))
    try:
        import run_benchmarks as rb
    finally:
        sys.path.pop(0)

    log_text = (work / "HiGHS.log").read_text() if (work / "HiGHS.log").exists() else ""
    parsed = rb.parse_highs_log(log_text)
    mps_range = rb.mps_matrix_range(work / "flextool.mps")
    slack_totals = rb.compute_slack_totals(work / "output_raw")
    objective = rb.read_v_obj(work / "output_raw")
    solve_time = rb.parse_solve_time_from_stdout(proc.stdout)

    report_path = work / "solve_data" / "scaling_report.txt"
    summary_line = _parse_scaling_report_summary(report_path)

    # Capture scaling_analysis.json for use_row_scaling recommendation visibility.
    scaling_json = work / "solve_data" / "scaling_analysis.json"
    rec = None
    if scaling_json.exists():
        try:
            rec = json.loads(scaling_json.read_text())
            rec = {
                "use_row_scaling": rec.get("use_row_scaling"),
                "scale_the_objective": rec.get("scale_the_objective"),
                "unitsize_spread_log10": rec.get("unitsize_spread_log10"),
                "rough_obj_estimate": rec.get("rough_obj_estimate"),
            }
        except Exception:
            pass

    return {
        "scenario": scenario,
        "mode": mode,
        "returncode": proc.returncode,
        "objective": objective,
        "matrix_range": parsed["matrix_range"],
        "cost_range": parsed["cost_range"],
        "bound_range": parsed["bound_range"],
        "rhs_range": parsed["rhs_range"],
        "rows_initial": parsed["rows_initial"],
        "cols_initial": parsed["cols_initial"],
        "nnz_initial": parsed["nnz_initial"],
        "matrix_range_from_mps": list(mps_range) if mps_range else None,
        "solve_wall_time_s": solve_time,
        "total_wall_time_s": round(total_wall, 3),
        "slack_totals": slack_totals,
        "scaling_report_summary": summary_line,
        "recommendation": rec,
    }


def main() -> int:
    data: dict = {"_meta": {"generated": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    for scenario in SCENARIOS:
        data[scenario] = {}
        for mode in MODES:
            sys.stderr.write(f"[validate] {scenario} / {mode} ...\n")
            sys.stderr.flush()
            data[scenario][mode] = run_one(scenario, mode)
            sys.stderr.write(
                f"           obj={data[scenario][mode]['objective']}  "
                f"time={data[scenario][mode]['solve_wall_time_s']}s\n"
            )
    print(json.dumps(data, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
