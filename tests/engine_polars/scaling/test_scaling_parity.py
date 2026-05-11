"""Parity tests for engine_polars/scaling.py — analyze_solve.

Runs the in-memory scaling analyzer on work_all and work_network_all_tech
and checks:
  1. Structural invariants on the returned ScaleTable.
  2. The JSON file is written to tmp_path/solve_data/scaling_analysis.json.
  3. Parsed JSON matches ScaleTable fields.
  4. Key fields vs the pre-committed baseline JSONs (with xfail markers on
     fields that are known to diverge from the CSV-based baseline).

Known divergences (bugs in scaling.py — do not fix here):
------------------------------------------------------------
B) node_inflow.n_nonzero:
   The original CSV reader only extracted values for the active 5-week /
   2-day horizon.  ``flex_data.p_inflow`` carries all periods and
   timesteps, so n_nonzero is larger (200 vs 72, or 288 vs 72).
   This does not affect the final decision in tested models.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars.scaling import (
    ScaleTable,
    analyze_solve,
    clear_cache,
)

HERE = Path(__file__).resolve().parent.parent  # tests/engine_polars/
DATA = HERE / "data"

# Baseline JSON files committed alongside the test fixtures.
_BASELINE_ALL = DATA / "work_all" / "solve_data" / "scaling_analysis.json"
_BASELINE_NET = DATA / "work_network_all_tech" / "solve_data" / "scaling_analysis.json"

# Solve names match what's recorded in baseline JSONs (and in model__solve.csv).
_SOLVE_ALL = "y2020_5week"
_SOLVE_NET = "y2020_2day_dispatch"

EXPECTED_FAMILY_KEYS = frozenset(
    [
        "entity_unitsize",
        "node_inflow",
        "node_annual_flow",
        "vom_and_op_costs",
        "capex_invest",
        "node_penalty",
    ]
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flex_all():
    """Load FlexData for work_all once per module."""
    return load_flextool(DATA / "work_all")


@pytest.fixture(scope="module")
def flex_net():
    """Load FlexData for work_network_all_tech once per module."""
    return load_flextool(DATA / "work_network_all_tech")


@pytest.fixture(autouse=True)
def _clear_scale_cache():
    """Ensure the module-level cache does not bleed between tests."""
    clear_cache()
    yield
    clear_cache()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _load_baseline(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Structural invariant tests
# ---------------------------------------------------------------------------


class TestScaleTableInvariantsWorkAll:
    """ScaleTable shape / type invariants for work_all."""

    def test_returns_scale_table(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert isinstance(table, ScaleTable)

    def test_use_row_scaling_is_yes_or_no(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert table.use_row_scaling in ("yes", "no"), (
            f"use_row_scaling={table.use_row_scaling!r} is not 'yes' or 'no'"
        )

    def test_scale_the_objective_is_positive_finite(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert math.isfinite(table.scale_the_objective)
        assert table.scale_the_objective > 0.0

    def test_unitsize_spread_log10_nonnegative(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert table.unitsize_spread_log10 >= 0.0

    def test_family_ranges_has_expected_keys(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert set(table.family_ranges.keys()) == EXPECTED_FAMILY_KEYS

    def test_row_scaling_trigger_valid(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert table.row_scaling_trigger in ("unitsize", "rhs", "cost", "none")

    def test_solve_name_is_stored(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert table.solve_name == _SOLVE_ALL


class TestScaleTableInvariantsWorkNet:
    """ScaleTable shape / type invariants for work_network_all_tech."""

    def test_returns_scale_table(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert isinstance(table, ScaleTable)

    def test_use_row_scaling_is_yes_or_no(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert table.use_row_scaling in ("yes", "no")

    def test_scale_the_objective_is_positive_finite(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert math.isfinite(table.scale_the_objective)
        assert table.scale_the_objective > 0.0

    def test_unitsize_spread_log10_nonnegative(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert table.unitsize_spread_log10 >= 0.0

    def test_family_ranges_has_expected_keys(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert set(table.family_ranges.keys()) == EXPECTED_FAMILY_KEYS

    def test_solve_name_is_stored(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert table.solve_name == _SOLVE_NET


# ---------------------------------------------------------------------------
# JSON file output tests
# ---------------------------------------------------------------------------


class TestJsonFileOutput:
    """JSON file is written to solve_data/scaling_analysis.json."""

    def test_json_written_for_work_all(self, flex_all, tmp_path):
        analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        out = tmp_path / "solve_data" / "scaling_analysis.json"
        assert out.exists(), f"JSON not written: {out}"
        assert out.stat().st_size > 0

    def test_json_written_for_work_net(self, flex_net, tmp_path):
        analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        out = tmp_path / "solve_data" / "scaling_analysis.json"
        assert out.exists(), f"JSON not written: {out}"
        assert out.stat().st_size > 0

    def test_json_parseable_work_all(self, flex_all, tmp_path):
        analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        out = tmp_path / "solve_data" / "scaling_analysis.json"
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_json_parseable_work_net(self, flex_net, tmp_path):
        analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        out = tmp_path / "solve_data" / "scaling_analysis.json"
        data = json.loads(out.read_text())
        assert isinstance(data, dict)

    def test_json_fields_match_scale_table_work_all(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        out = tmp_path / "solve_data" / "scaling_analysis.json"
        data = json.loads(out.read_text())
        assert data["use_row_scaling"] == table.use_row_scaling
        assert data["scale_the_objective"] == pytest.approx(
            table.scale_the_objective, rel=1e-9
        )
        assert data["unitsize_spread_log10"] == pytest.approx(
            table.unitsize_spread_log10, rel=1e-9
        )
        assert data["row_scaling_trigger"] == table.row_scaling_trigger
        assert data["solve_name"] == table.solve_name
        assert set(data["family_ranges"].keys()) == EXPECTED_FAMILY_KEYS


# ---------------------------------------------------------------------------
# Baseline parity tests — fields that MATCH the CSV-based baseline
# ---------------------------------------------------------------------------
# Fields that are known to DIVERGE from the CSV-based baseline are marked
# with pytest.mark.xfail and a clear comment explaining the root cause.
# See the module docstring for a summary of the two divergence classes.


@pytest.mark.skipif(
    not _BASELINE_ALL.exists(),
    reason=f"baseline JSON missing: {_BASELINE_ALL}",
)
class TestBaselineParityWorkAll:
    """Key ScaleTable fields vs the pre-committed baseline for work_all.

    The final use_row_scaling / scale_the_objective / row_scaling_trigger
    happen to match the baseline for work_all despite the unitsize-source
    bug, because the node_penalty family spread is wide enough to trigger
    row scaling independently.
    """

    def test_use_row_scaling_matches_baseline(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_ALL)
        assert table.use_row_scaling == ref["use_row_scaling"], (
            f"use_row_scaling: got {table.use_row_scaling!r}, "
            f"expected {ref['use_row_scaling']!r}"
        )

    @pytest.mark.xfail(
        reason=(
            "Two-sided cost-band guard (cost_abs_min + cost_abs_max) "
            "intentionally diverges from the single-sided baseline.  "
            "The new guard moves the LP cost range UPWARD to keep "
            "cost_min × scale above HiGHS' 1e-7 'excessively small' "
            "threshold, replacing the legacy 'minimise objective "
            "magnitude' heuristic.  For work_all this raises the "
            "recommendation from 1e-9 (legacy single-sided) to "
            "~5e-9 (two-sided clamp).  Baseline kept for historical "
            "record; intentional behaviour drift documented in the "
            "commit that added the two-sided guard."
        ),
        strict=True,
    )
    def test_scale_the_objective_matches_baseline(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_ALL)
        assert table.scale_the_objective == pytest.approx(
            ref["scale_the_objective"], rel=1e-6
        ), (
            f"scale_the_objective: got {table.scale_the_objective}, "
            f"expected {ref['scale_the_objective']}"
        )

    def test_row_scaling_trigger_matches_baseline(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_ALL)
        assert table.row_scaling_trigger == ref["row_scaling_trigger"], (
            f"row_scaling_trigger: got {table.row_scaling_trigger!r}, "
            f"expected {ref['row_scaling_trigger']!r}"
        )

    def test_unitsize_spread_matches_baseline(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_ALL)
        assert table.unitsize_spread_log10 == pytest.approx(
            ref["unitsize_spread_log10"], rel=1e-9
        )

    def test_entity_unitsize_abs_max_matches_baseline(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_ALL)
        got = table.family_ranges["entity_unitsize"].abs_max
        expected = ref["family_ranges"]["entity_unitsize"]["abs_max"]
        assert got == pytest.approx(expected, rel=1e-6)

    @pytest.mark.xfail(
        reason=(
            "Bug B: node_inflow.n_nonzero — p_inflow carries all periods "
            "and timesteps; the CSV-based baseline only counted the active "
            "5-week horizon (72 rows). In-memory count is larger (200)."
        ),
        strict=True,
    )
    def test_node_inflow_n_nonzero_matches_baseline(self, flex_all, tmp_path):
        table = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_ALL)
        got = table.family_ranges["node_inflow"].n_nonzero
        expected = ref["family_ranges"]["node_inflow"]["n_nonzero"]
        assert got == expected, (
            f"node_inflow.n_nonzero: got {got}, expected {expected}"
        )


@pytest.mark.skipif(
    not _BASELINE_NET.exists(),
    reason=f"baseline JSON missing: {_BASELINE_NET}",
)
class TestBaselineParityWorkNet:
    """Key ScaleTable fields vs the pre-committed baseline for work_network_all_tech.

    This model exposes Bug A more clearly: the missing node unitsizes cause
    the unitsize spread to fall below the 3-decade threshold, so
    use_row_scaling flips from 'yes' to 'no'.
    """

    @pytest.mark.xfail(
        reason=(
            "Bug A: p_unitsize excludes node unitsizes. For work_network_all_tech "
            "the node 'water_sink' (1e6) and 'battery' (1e5) entries are missing, "
            "dropping unitsize spread from 5.0 to 2.0 decades (below 3-decade "
            "threshold). This causes use_row_scaling to flip 'yes' -> 'no'."
        ),
        strict=True,
    )
    def test_use_row_scaling_matches_baseline(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_NET)
        assert table.use_row_scaling == ref["use_row_scaling"]

    @pytest.mark.xfail(
        reason=(
            "Bug A: scale_the_objective is derived from rough_obj which depends "
            "on unitsize spread. With missing node unitsizes the estimate changes "
            "(1e-8 vs baseline 1e-9)."
        ),
        strict=True,
    )
    def test_scale_the_objective_matches_baseline(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_NET)
        assert table.scale_the_objective == pytest.approx(
            ref["scale_the_objective"], rel=1e-6
        )

    @pytest.mark.xfail(
        reason=(
            "Bug A: row_scaling_trigger is 'none' (no trigger fires) instead of "
            "'unitsize' because the unitsize spread is 2.0 decades (below "
            "UNITSIZE_SPREAD_THRESHOLD=3.0) due to missing node entries."
        ),
        strict=True,
    )
    def test_row_scaling_trigger_matches_baseline(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_NET)
        assert table.row_scaling_trigger == ref["row_scaling_trigger"]

    @pytest.mark.xfail(
        reason=(
            "Bug A: unitsize_spread_log10 is 2.0 in-memory vs 5.0 in baseline "
            "because node unitsizes (water_sink=1e6, battery=1e5) are absent "
            "from p_unitsize."
        ),
        strict=True,
    )
    def test_unitsize_spread_matches_baseline(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_NET)
        assert table.unitsize_spread_log10 == pytest.approx(
            ref["unitsize_spread_log10"], rel=1e-9
        )

    @pytest.mark.xfail(
        reason=(
            "Bug B: node_inflow.n_nonzero — p_inflow carries all periods; "
            "in-memory count is 288 vs baseline 72."
        ),
        strict=True,
    )
    def test_node_inflow_n_nonzero_matches_baseline(self, flex_net, tmp_path):
        table = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        ref = _load_baseline(_BASELINE_NET)
        got = table.family_ranges["node_inflow"].n_nonzero
        expected = ref["family_ranges"]["node_inflow"]["n_nonzero"]
        assert got == expected


# ---------------------------------------------------------------------------
# Cache idempotency
# ---------------------------------------------------------------------------


class TestCacheIdempotency:
    """Second call returns the cached table — same object identity."""

    def test_second_call_returns_cached_for_work_all(self, flex_all, tmp_path):
        t1 = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        t2 = analyze_solve(_SOLVE_ALL, flex_all, work_folder=tmp_path)
        assert t1 is t2

    def test_second_call_returns_cached_for_work_net(self, flex_net, tmp_path):
        t1 = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        t2 = analyze_solve(_SOLVE_NET, flex_net, work_folder=tmp_path)
        assert t1 is t2
