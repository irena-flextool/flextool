"""Tier 7 emission test — minimum_uptime row count.

In ``flextool.mod`` the constraint is::

    s.t. minimum_uptime {(p, d, t) in pdt_uptime : p in process_online ...}

For the ``coal_wind_min_uptime`` scenario, the ``pdt_uptime_set.csv``
domain already represents the post-filter set (the linear-online
variant is what the LP carries), so the emitted row count must equal
``|pdt_uptime_set ∩ process_online|``.
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
def test_minimum_uptime_emits_correct_rows(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    scenario = "coal_wind_min_uptime"
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

    pdt_uptime = pd.read_csv(workdir / "solve_data" / "pdt_uptime_set.csv")
    process_online = pd.read_csv(workdir / "solve_data" / "process_online.csv")
    online_processes = set(process_online["process"])
    expected = int(pdt_uptime["process"].isin(online_processes).sum())

    actual = families.get("minimum_uptime", 0)
    assert actual == expected, (
        f"minimum_uptime row count mismatch: actual={actual} "
        f"expected={expected} (|pdt_uptime_set ∩ process_online|)"
    )
