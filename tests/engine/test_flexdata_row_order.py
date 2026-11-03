"""The rows of the FlexData set frames and of the output files come in the same order on every run.

The LP does not depend on row order (``add_var`` / ``add_cstr`` sort their domains), but the outputs, dumps
and any first-row pick do.  Polars ``unique()`` / ``group_by`` / joins return their rows in an order that
changes from run to run, also within one process, and iterating a Python ``set`` of strings follows the
hash seed.  Each producer therefore either keeps its input order (``unique(maintain_order=True)``) or
sorts by its keys.

Each case runs the cascade in two child processes (``PYTHONHASHSEED`` 1 and 2), twice in each, and writes
every ``pl.DataFrame`` field of every step's FlexData and every ``output_raw`` parquet.  The four runs must
be ``.equals`` without sorting: run 1 == run 2 in one process, and process 1 == process 2.  The sweep
covers every FlexData frame field, so it also catches producers not named here.  Fields the fixtures
exercise: ``nodeState_first_dt``, ``storage_bind_{within_timeblock,forward_only,within_solve}``,
``period_block``, ``period_block_time``, ``edd_invest_set`` (and the ``edd_invest__*`` outputs),
``ndt_fix_storage_quantity``, ``period_branch``, ``label_steps``, ``period_last``, the rp sets and the
entity-wide output columns.  No fixture uses ``bind_within_period``; ``storage_bind_within_period`` is
built by the same code as the other three.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

# (case id, db fixture, scenarios, FlexData fields that must be present in some step of the case)
_CASES = [
    ("lh2", "lh2_db_url", ["lh2_three_region_invest"],
     ["nodeState_first_dt", "storage_bind_within_timeblock", "storage_bind_forward_only",
      "storage_bind_within_solve", "period_block", "period_block_time", "edd_invest_set"]),
    ("main", "test_db_url",
     ["wind_battery_invest_lifetime_renew_4solve", "network_coal_wind_battery_invest_cumulative", "test_a_lot"],
     ["edd_invest_set", "ed_invest_cumulative", "period_in_use_set", "dtt__delay_duration",
      "flow_constraint_idx"]),
    pytest.param(("nested", "test_db_url", ["multi_fullYear_battery_nested_multi_invest"],
                  ["ndt_fix_storage_quantity", "dtt_timeline_matching"]),
                 marks=pytest.mark.slow),  # ~4 min: four runs of a nested rolling cascade
    ("stochastic", "stochastic_db_url", ["2_day_stochastic_dispatch", "1_week_rolling_wind"],
     ["label_steps", "period_last", "period_branch"]),
    ("rp", "lh2_rp_invest_db_url", ["lh2_three_region_rp_invest"],
     ["nodeState_rp", "p_rp_last_step", "rp_base_first", "rp_base_last", "rp_base_period_set",
      "rp_block_first"]),
]


def _dump_run(db_url: str, scenario: str, work: Path, dump: Path, solver_config_dir: Path) -> None:
    """Run the cascade; write each step's FlexData frame fields to ``dump/fd/<step>/<field>.parquet``."""
    from flextool.engine.solve._orchestration import run_chain_from_db

    steps = run_chain_from_db(input_db_url=db_url, scenario_name=scenario, work_folder=work,
                              solver_config_dir=solver_config_dir, csv_dump=False, keep_solutions=True)
    for name, step in steps.items():
        fd = getattr(step, "flex_data", None)
        if fd is None:
            continue
        folder = dump / "fd" / name
        folder.mkdir(parents=True, exist_ok=True)
        for field in dataclasses.fields(fd):
            value = getattr(fd, field.name, None)
            if isinstance(value, pl.DataFrame):
                value.write_parquet(folder / f"{field.name}.parquet")


def _child(spec_path: str) -> None:
    import flextool

    print("flextool:", flextool.__file__, flush=True)
    spec = json.loads(Path(spec_path).read_text())
    out = Path(spec["out"])
    for rep in range(2):
        for scenario in spec["scenarios"]:
            base = out / f"run{rep}" / scenario
            _dump_run(spec["db_url"], scenario, base / "work", base / "dump", Path(spec["solver_config_dir"]))


def _frames(root: Path) -> dict[str, pl.DataFrame]:
    files = [f for f in sorted(root.rglob("*.parquet"))
             if "dump" in f.relative_to(root).parts or "output_raw" in f.relative_to(root).parts]
    return {str(f.relative_to(root)): pl.read_parquet(f) for f in files}


def _differing(a: dict[str, pl.DataFrame], b: dict[str, pl.DataFrame]) -> list[str]:
    assert a.keys() == b.keys(), sorted(set(a) ^ set(b))[:20]
    return [k for k, ref in a.items() if not ref.equals(b[k])]


@pytest.mark.solver
@pytest.mark.parametrize("case", _CASES, ids=["lh2", "main", "nested", "stochastic", "rp"])
def test_row_order_same_across_runs_and_processes(case, request, tmp_path, test_solver_config_dir) -> None:
    case_id, db_fixture, scenarios, fields = case
    db_url = request.getfixturevalue(db_fixture)
    runs: dict[int, dict[int, dict[str, pl.DataFrame]]] = {}
    for seed in (1, 2):
        out = tmp_path / f"seed{seed}"
        spec = tmp_path / f"spec{seed}.json"
        spec.write_text(json.dumps({"out": str(out), "db_url": db_url, "scenarios": scenarios,
                                    "solver_config_dir": str(test_solver_config_dir)}))
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "POLARS_MAX_THREADS": "4"}
        proc = subprocess.run([sys.executable, __file__, str(spec)], env=env, capture_output=True, text=True,
                              timeout=1200)
        assert proc.returncode == 0, proc.stderr[-4000:]
        runs[seed] = {rep: _frames(out / f"run{rep}") for rep in range(2)}

    present = {Path(k).stem for k in runs[1][0] if "fd" in Path(k).parts}
    assert set(fields) <= present, sorted(set(fields) - present)
    same_process = _differing(runs[1][0], runs[1][1]) + _differing(runs[2][0], runs[2][1])
    assert not same_process, f"{case_id}: row order differs between two runs in one process: {same_process}"
    cross = _differing(runs[1][0], runs[2][0])
    assert not cross, f"{case_id}: row order differs between PYTHONHASHSEED 1 and 2: {cross}"


if __name__ == "__main__":
    _child(sys.argv[1])
