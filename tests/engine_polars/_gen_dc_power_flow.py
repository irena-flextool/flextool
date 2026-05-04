"""Generate the work_dc_power_flow fixture (PGLib case14 IEEE).

Mirrors the pattern of flextool's ``test_dc_power_flow.py::TestPGLibCase14Integration``
fixture: parse the MATPOWER ``pglib_opf_case14_ieee.m`` file, build a
FlexTool Spine DB with DC power flow physics enabled, run flextool's full
pipeline so we capture the reference v_obj parquet under
``output_raw/v_obj__dispatch.parquet``.

Usage::

    ~/venv-spi/bin/python tests/_gen_dc_power_flow.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# REPO_ROOT still points at the live flextool checkout because we use
# flextool's Python packages (FlexToolRunner, read_matpower) and its
# bin/ + .mod files live.  Only the MATPOWER .m fixture is vendored.
REPO_ROOT  = Path("/home/jkiviluo/sources/flextool")
TESTS_DIR  = Path(__file__).resolve().parent
DATA_DIR   = TESTS_DIR / "data"
FIXTURES   = TESTS_DIR / "fixtures"

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner    # noqa: E402
from flextool.process_inputs.read_matpower import (                   # noqa: E402
    create_flextool_db_from_matpower,
    read_matpower,
)


SCENARIO   = "dc_opf_test"
CASE14_M   = FIXTURES / "pglib_opf_case14_ieee.m"


def generate(workdir: Path | None = None) -> Path:
    workdir = workdir or (DATA_DIR / "work_dc_power_flow")
    workdir.mkdir(parents=True, exist_ok=True)

    # Parse MATPOWER and build FlexTool DB.
    case = read_matpower(str(CASE14_M))
    db_path = workdir / "case14.sqlite"
    if db_path.exists():
        db_path.unlink()
    db_url = create_flextool_db_from_matpower(
        case, str(db_path), scenario_name=SCENARIO,
    )

    prev_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        runner = FlexToolRunner(
            input_db_url   = db_url,
            scenario_name  = SCENARIO,
            flextool_dir   = REPO_ROOT / "flextool",
            bin_dir        = REPO_ROOT / "bin",
            root_dir       = workdir,
            work_folder    = workdir,
        )
        runner.write_input(db_url, SCENARIO)
        rc = runner.run_model()
    finally:
        os.chdir(prev_cwd)

    if rc != 0:
        raise SystemExit(f"flextool failed for {SCENARIO!r}: rc={rc}")
    print(f"\n{SCENARIO}: generated under {workdir}")
    return workdir


if __name__ == "__main__":
    generate()
