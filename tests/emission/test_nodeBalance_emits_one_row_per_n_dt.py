"""Tier 7 emission test — nodeBalance_eq row count.

Confirms the MPS contains exactly one ``nodeBalance_eq`` row per
``(node-with-balance, dt)`` pair. The count is derived on the fly from
``solve_data/nodeBalance.csv`` (the actual constraint domain in
``flextool.mod`` is ``n in nodeBalance``; ``nodeBalancePeriod`` drives a
different aggregation and produces no rows in this family) and
``solve_data/dt.csv``.
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
def test_nodeBalance_emits_one_row_per_n_dt(
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

    # The constraint set in flextool.mod is `n in nodeBalance`. Take the
    # union with nodeBalancePeriod for futureproofing in case the family
    # is ever extended (today nodeBalancePeriod has no rows in this
    # family, so the union degenerates to nodeBalance).
    node_balance = pd.read_csv(workdir / "solve_data" / "nodeBalance.csv")
    node_balance_period_path = workdir / "solve_data" / "nodeBalancePeriod.csv"
    if node_balance_period_path.exists():
        node_balance_period = pd.read_csv(node_balance_period_path)
        nodes = set(node_balance["node"]) | set(node_balance_period["node"])
    else:
        nodes = set(node_balance["node"])

    dt = pd.read_csv(workdir / "solve_data" / "dt.csv")
    expected = len(nodes) * len(dt)

    actual = families.get("nodeBalance_eq", 0)
    assert actual == expected, (
        f"nodeBalance_eq row count mismatch: actual={actual} "
        f"expected={expected} (|nodes|={len(nodes)} * |dt|={len(dt)})"
    )
