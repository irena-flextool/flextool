"""Tier 7 emission test — maxToSink row count.

The ``maxToSink`` constraint family in ``flextool.mod`` is indexed by
``(p, source, sink) in process__source__sinkIsNode`` × ``(d, t) in dt``.
The emitted row count must match this product exactly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner

from ._mps_parser import parse_mps_row_families


@pytest.mark.skip(
    reason=(
        "Δ.22: this Tier-7 emission test parses ``flextool.mps`` written "
        "by glpsol --wfreemps in the legacy MathProg pipeline.  The "
        "native cascade builds the LP directly via polar_high and never "
        "emits an MPS file, so the parser cannot run.  Row-count "
        "invariants are now structural to the cascade's LP build and "
        "are exercised by the per-family unit tests in tests/engine_polars/."
    )
)
@pytest.mark.emission
def test_maxToSink_emits_one_row_per_pss_dt(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    scenario = "coal"
    runner = FlexToolRunner(
        input_db_url=test_db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(test_db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Model run failed for scenario '{scenario}'"

    families = parse_mps_row_families(workdir / "flextool.mps")

    # Constraint domain: process__source__sinkIsNode (single set, not
    # split by eff/noEff in the .mod) × dt.
    pss = pd.read_csv(workdir / "solve_data" / "process__source__sinkIsNode.csv")
    dt = pd.read_csv(workdir / "solve_data" / "dt.csv")
    expected = len(pss) * len(dt)

    actual = families.get("maxToSink", 0)
    assert actual == expected, (
        f"maxToSink row count mismatch: actual={actual} "
        f"expected={expected} (|process__source__sinkIsNode|={len(pss)} "
        f"* |dt|={len(dt)})"
    )
