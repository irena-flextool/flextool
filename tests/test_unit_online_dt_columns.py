"""Regression test for Rivendell bug 4 — ``process_online_dt`` missing
column for a UC unit (specs/model_bugs.md ``# Rivendell bug 4``).

Bug recap
---------
For ``S12_uc_flextool_linear_roll`` (Rivendell), post-processing crashed in
``flextool/process_outputs/out_flows.py:271``::

    online_units_dt = r.process_online_dt[
        s.process_unit.intersection(s.process_online)
    ]
    # KeyError: "None of [Index(['RVN_PP_NGS_UC_C'], dtype='str',
    #           name='process')] are in the [columns]"

The unit was tagged as ``process_online`` (UC machinery enabled), so it
appeared in the ``s.process_unit ∩ s.process_online`` set the
post-processor selected on.  But ``v_online_linear`` and
``v_online_integer`` were emitted with zero process columns, because the
per-solve ``p_online_dt_set`` was empty — the writer derived it strictly
from ``process_block ⋈ block_step_duration`` and the input data had no
``process_block`` row for the UC unit.

Root cause + fix
----------------
``flextool/engine_polars/_emit_per_solve.py::write_per_solve_sets`` —
when a process in ``process_online`` had no ``process_block`` entry, the
emitted ``p_online_dt`` collapsed to zero rows, so the LP build skipped
``v_online_linear[(p, d, t)]`` for that process entirely, the post-solve
parquet for that solve carried no column for the unit, and the
post-processing column-select crashed.

Fix (commit ``91edd891``): when a UC process has no ``process_block``
row, fall back to the per-step timeline (``steps_in_use.csv``), matching
``_native_input_writer``'s documented intent that empty entity_block /
process_block stubs "fall through to identity (every entity mapped to
'default')".  See ``# BUG p_online_dt_empty_no_blocks`` in
``specs/model_bugs.md``.

What this test guards
---------------------
The unit-level fallback already has its own assertion
(``tests/engine_polars/constraints/test_p_online_dt_fallback.py``).  The
test here closes the loop at the post-processing layer: it runs the full
``run_chain_from_db`` → ``write_outputs`` pipeline for a UC fixture and
asserts the invariant the bug 4 traceback violated::

    ∀ p ∈ s.process_unit ∩ s.process_online :
        p ∈ r.process_online_dt.columns

i.e. for every UC unit, ``r.process_online_dt`` (the
``v_online_linear + v_online_integer`` matrix downstream code selects
on) must have that unit as a column.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.process_outputs.write_outputs import write_outputs  # noqa: E402


def test_process_online_dt_has_uc_unit_columns(
    test_db_url: str,
    test_solver_config_dir: Path,
) -> None:
    """For a UC fixture, ``r.process_online_dt`` must carry every
    ``s.process_unit ∩ s.process_online`` member as a column.

    Uses the ``coal_wind_min_uptime`` fixture (a UC scenario without an
    explicit ``process_block`` row — the same fixture exercised by
    ``test_p_online_dt_fallback.py``).  The end-to-end run goes through
    ``write_outputs`` so the column-select in
    ``out_flows.unit_online_and_startup`` is exercised; if the bug
    regresses, the call inside ``write_outputs`` raises ``KeyError``.
    """
    scenario = "coal_wind_min_uptime"

    with tempfile.TemporaryDirectory() as wd:
        workdir = Path(wd)
        # write_outputs reads ``output_raw/`` relative to CWD by
        # default; ``raw_output_dir`` makes the lookup explicit.
        cwd = os.getcwd()
        try:
            os.chdir(workdir)

            steps = run_chain_from_db(
                test_db_url,
                scenario,
                work_folder=workdir,
                solver_config_dir=test_solver_config_dir,
                keep_solutions=True,
            )
            assert steps, (
                f"run_chain_from_db returned no steps for '{scenario}'"
            )
            last = next(reversed(steps.values()))
            assert last.solution is not None and last.solution.optimal, (
                f"'{scenario}' did not solve optimally"
            )

            # The crash in bug 4 is inside ``write_outputs`` →
            # ``unit_online_and_startup`` → ``r.process_online_dt[...]``.
            # If the regression returns, this call raises ``KeyError``.
            write_outputs(
                scenario_name=scenario,
                output_location=str(workdir),
                subdir=scenario,
                output_config_path=OUTPUT_CONFIG,
                write_methods=["csv"],
                fallback_output_location=str(workdir),
                raw_output_dir=str(workdir / "output_raw"),
                solution=last.solution,
                solve_name=last.solve_name,
                solve_steps=[
                    (s.solve_name, s.flex_data, s.effective_solution)
                    for s in steps.values()
                ],
            )

            # Belt-and-braces — verify the actual output file carries
            # the UC unit.  ``unit_online_dt_e`` is the table produced
            # by ``unit_online_and_startup``; ``unit_online__dt.csv`` is
            # its user-facing CSV name (templates/default_plots.yaml).
            csv_path = workdir / "output_csv" / scenario / "unit_online__dt.csv"
            assert csv_path.exists(), (
                f"unit_online__dt.csv was not written for '{scenario}' — "
                f"unit_online_and_startup either returned empty or "
                f"templates/default_plots.yaml has drifted from the "
                f"out_flows output_name."
            )
            df = pd.read_csv(csv_path)
            # CSV layout is wide: index columns ('solve', 'period',
            # 'time') + one column per UC unit.  The UC unit must
            # appear as a column.
            assert "coal_plant" in set(df.columns), (
                f"UC unit 'coal_plant' missing from {csv_path.name} "
                f"columns {sorted(df.columns)} — r.process_online_dt "
                f"regressed to dropping the column.  See "
                f"specs/model_bugs.md '# Rivendell bug 4' and "
                f"'# BUG p_online_dt_empty_no_blocks'."
            )
            # The unit must carry non-null values across the dt grid —
            # an all-NaN column would mean the underlying parquet
            # joined an empty solve-side and the post-processor merely
            # reindexed.  Bug 4's failure mode had the column missing
            # entirely; this assertion guards a softer regression where
            # the column reappears but is empty.
            assert df["coal_plant"].notna().any(), (
                f"'coal_plant' column in {csv_path.name} is entirely "
                f"NaN — r.process_online_dt has the column shape but "
                f"the per-solve parquet carried no values."
            )
        finally:
            os.chdir(cwd)
