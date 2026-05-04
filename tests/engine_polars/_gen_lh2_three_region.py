"""Re-generate flextool ``input/`` + ``solve_data/`` + ``output_raw/``
for the ``lh2_three_region`` scenario into
``tests/data/work_lh2_three_region/``.

This scenario is NOT in flextool's ``tests.json`` baseline — it lives
in its own committed JSON fixture
(``flextool/tests/fixtures/lh2_three_region.json``) which already has
the v51 schema additions (decomposition_method, new_stepduration)
layered onto a pruned baseline.  We just need to materialise the JSON
into a SQLite and run flextool against it.

Usage::

    python tests/_gen_lh2_three_region.py
"""

from __future__ import annotations

import os
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


SCENARIO   = "lh2_three_region"
JSON_PATH  = FIXTURES / "lh2_three_region.json"


def generate(workdir: Path | None = None) -> Path:
    workdir = workdir or (DATA_DIR / f"work_{SCENARIO}")
    workdir.mkdir(parents=True, exist_ok=True)

    db_path = workdir / "tests.sqlite"
    if db_path.exists():
        db_path.unlink()
    db_url = json_to_db(JSON_PATH, db_path)

    # FlexToolRunner reads cwd at construction for some downstream
    # writers; pin it like the upstream test does.
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
