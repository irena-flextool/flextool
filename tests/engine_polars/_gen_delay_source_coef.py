"""Generate the work_delay_source_coef fixture.

This fixture is a derivative of ``water_pump_delayed`` with one
modification: ``p_process_source_flow_coefficient`` for
``(water_pump, water_source)`` is changed from 1.0 to 2.0.  This
exposes the delay-path source-coefficient bug — flextool's
.mod (line 2573) applies the coefficient to the delayed source flows;
flexpy's ``_delay.delayed_input_expr`` did not, so the LP solutions
differ when the coefficient is non-default.

Generation procedure:

  1. Run the standard ``_gen_input.py`` pipeline for water_pump_delayed
     into ``work_delay_source_coef/`` (same scenario, different output
     directory).
  2. After ``write_input`` completes, patch
     ``input/p_process_source_flow_coefficient.csv`` to set
     ``water_pump, water_source`` from 1.0 to 2.0.
  3. Run the model so the solver sees the patched CSV and emits a
     reference solution that *does* multiply delayed source flows by 2.0
     (per the .mod's handling).

Usage:
    ~/venv-spi/bin/python tests/_gen_delay_source_coef.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# REPO_ROOT still points at the live flextool checkout because we use
# flextool's Python package (FlexToolRunner) and its bin/ + .mod files
# live.  Only data fixtures + json_to_db are vendored.
REPO_ROOT  = Path("/home/jkiviluo/sources/flextool")
TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = TESTS_DIR / "data"
FIXTURES   = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))

from _db_utils import json_to_db                                      # noqa: E402  vendored
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner    # noqa: E402

from _gen_input import _install_per_sub_solve_snapshot_hook           # noqa: E402


def _patch_source_flow_coef(workdir: Path) -> None:
    """Set p_process_source_flow_coefficient(water_pump, water_source) = 2.0.

    Replaces the existing 1.0 row.  Leaves the (water_pump, west) = 0.0 row
    intact (zero-coef rows are dropped from the conversion equation by
    both flextool and flexpy, so keeping it harmless).
    """
    csv = workdir / "input" / "p_process_source_flow_coefficient.csv"
    text = csv.read_text()
    new_text = text.replace(
        "water_pump,water_source,1.0",
        "water_pump,water_source,2.0",
    )
    if new_text == text:
        raise RuntimeError(
            f"source coef patch did not match anything in {csv}; CSV content was:\n{text}"
        )
    csv.write_text(new_text)
    print(f"patched {csv}: water_pump,water_source 1.0 -> 2.0")


def _patch_water_sink_demand(workdir: Path) -> None:
    """Add an outflow (negative inflow) to water_sink so the sink-side flow
    becomes binding in the LP — without this, the conversion equation's
    coefficient is degenerate and the bug is invisible to the objective.

    With water_sink draining at -1.0 MWh/step, the LP must satisfy
    ``coef * source_flow = sink_flow >= drain_rate``.  When coef=2, half
    as much source flow is needed; when coef=1 (flexpy bug), full source
    flow is needed → different coal cost → different objective.
    """
    csv = workdir / "input" / "p_node.csv"
    text = csv.read_text()
    if "water_sink,inflow," in text:
        raise RuntimeError("water_sink already has an inflow row; refuse to clobber")
    if not text.endswith("\n"):
        text += "\n"
    text += "water_sink,inflow,-1.0\n"
    text += "water_sink,penalty_up,500.0\n"
    text += "water_sink,penalty_down,500.0\n"
    csv.write_text(text)
    print(f"patched {csv}: added water_sink inflow=-1.0 (constant drain) "
          "and penalty_up/down=500")


def generate(scenario: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    db_path = workdir / "tests.sqlite"
    if db_path.exists():
        db_path.unlink()
    db_url = json_to_db(FIXTURES / "tests.json", db_path)

    _install_per_sub_solve_snapshot_hook()

    runner = FlexToolRunner(
        input_db_url   = db_url,
        scenario_name  = scenario,
        flextool_dir   = REPO_ROOT / "flextool",
        bin_dir        = REPO_ROOT / "bin",
        work_folder    = workdir,
    )
    runner.write_input(db_url, scenario)
    _patch_source_flow_coef(workdir)
    _patch_water_sink_demand(workdir)
    rc = runner.run_model()
    if rc != 0:
        raise SystemExit(f"flextool failed for {scenario!r}: rc={rc}")
    print(f"\n{scenario} (with patched source coef): generated under {workdir}")
    return workdir


if __name__ == "__main__":
    workdir = DATA_DIR / "work_delay_source_coef"
    generate("water_pump_delayed", workdir)
