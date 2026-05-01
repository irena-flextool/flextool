"""Tier 7 emission test — storage_state_start_binding row count.

In ``flextool.mod`` the constraint is::

    s.t. storage_state_start_binding {n in nodeState, (n, b_n) in node__block,
         (b_n, d, t) in block__period__time_first
         : p_nested_model['solveFirst']
         && (n, 'bind_forward_only') not in node__storage_binding_method
         && (n, 'bind_within_period')   not in node__storage_binding_method
         && (n, 'bind_within_solve')    not in node__storage_binding_method
         && d in period_first
         && ((n, 'fix_start')      in node__storage_start_end_method
             || (n, 'fix_start_end') in node__storage_start_end_method)}

For ``wind_battery``: 1 nodeState (battery, ``bind_within_timeset`` —
not in any of the three excluded binding methods, and ``fix_start``)
× 1 ``period_first_of_solve`` = 1 row.

Expected count: ``|nodeState_eligible| × |period_first_of_solve|`` where
``nodeState_eligible`` is ``nodeState`` minus nodes whose
``storage_binding_method`` is one of the three excluded variants and
which have ``fix_start`` or ``fix_start_end``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner

from ._mps_parser import parse_mps_row_families


_EXCLUDED_BINDING = {
    "bind_forward_only",
    "bind_within_period",
    "bind_within_solve",
}
_FIX_START_METHODS = {"fix_start", "fix_start_end"}


@pytest.mark.emission
def test_storage_state_start_binding_emits_only_in_period_first(
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> None:
    scenario = "wind_battery"
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

    node_state = pd.read_csv(workdir / "solve_data" / "nodeState.csv")
    period_first = pd.read_csv(workdir / "solve_data" / "period_first_of_solve.csv")

    binding = pd.read_csv(workdir / "input" / "node__storage_binding_method.csv")
    excluded_nodes = set(
        binding.loc[binding["storage_binding_method"].isin(_EXCLUDED_BINDING), "node"]
    )

    start_end = pd.read_csv(workdir / "input" / "node__storage_start_end_method.csv")
    pinned_nodes = set(
        start_end.loc[start_end["storage_start_end_method"].isin(_FIX_START_METHODS), "node"]
    )

    eligible = (set(node_state["node"]) & pinned_nodes) - excluded_nodes
    expected = len(eligible) * len(period_first)

    actual = families.get("storage_state_start_binding", 0)
    assert actual == expected, (
        f"storage_state_start_binding row count mismatch: actual={actual} "
        f"expected={expected} (|eligible_nodeState|={len(eligible)} * "
        f"|period_first_of_solve|={len(period_first)})"
    )
