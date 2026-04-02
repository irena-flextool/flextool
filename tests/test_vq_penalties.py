"""Tests that verify vq penalty variables produce correct cost values.

Each test runs an existing scenario from the test database, reads the
output CSV files, and asserts that penalty costs match hand-calculated
expected values.

Penalty cost formulas (from flextool.mod objective function):
  vq_state_up cost = vq_state_up * node_capacity_for_scaling * penalty_up
                     * step_duration * discount_factor / period_share_of_year
  vq_capacity_margin cost = vq_capacity_margin * group_capacity_for_scaling
                            * penalty_capacity_margin * discount_factor

In the test scenarios:
  - node_capacity_for_scaling = 1 (always, by definition in flextool.mod)
  - step_duration = 1 hour (y2020 timeline uses 1h steps)
  - discount_factor and period_share_of_year affect annualized costs but
    the costs__dt.csv reports NON-annualized per-timestep costs

Therefore the simple formula for costs__dt.csv "upward slack penalty" is:
  slack_MW * penalty_up * step_duration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

TEST_DIR = Path(__file__).parent
REPO_ROOT = TEST_DIR.parent
OUTPUT_CONFIG = str(REPO_ROOT / "templates" / "default_plots.yaml")

if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
from flextool.process_outputs.write_outputs import write_outputs


# ---------------------------------------------------------------------------
# Known input data for hand-calculations
# ---------------------------------------------------------------------------

# West node inflow for first 48 timesteps (2day timeset), from 'west' alternative.
# Negative values = demand (consumption).
WEST_INFLOW_48 = [
    -589, -537, -506, -482, -472, -454, -423, -438,
    -474, -518, -538, -558, -572, -564, -553, -541,
    -534, -567, -689, -696, -670, -627, -550, -458,
    -436, -446, -436, -452, -469, -507, -549, -594,
    -682, -730, -774, -802, -807, -793, -780, -768,
    -755, -767, -872, -865, -808, -735, -669, -597,
]

PENALTY_UP_WEST = 900.0   # CUR/MWh
PENALTY_DOWN_WEST = 800.0  # CUR/MWh
STEP_DURATION = 1.0        # hours (all steps in y2020 timeline)
COAL_EXISTING_MW = 500.0   # MW capacity of coal_plant
COAL_EFFICIENCY = 0.4
COAL_PRICE = 20.0          # CUR/MWh (fuel)


def _run_scenario(
    scenario: str,
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> Path:
    """Run a scenario and write CSV outputs. Returns the output CSV directory."""
    runner = FlexToolRunner(
        input_db_url=test_db_url,
        scenario_name=scenario,
        root_dir=workdir,
        bin_dir=test_bin_dir,
    )
    runner.write_input(test_db_url, scenario)
    return_code = runner.run_model()
    assert return_code == 0, f"Model run failed for scenario '{scenario}'"

    write_outputs(
        scenario_name=scenario,
        output_location=str(workdir),
        subdir=scenario,
        output_config_path=OUTPUT_CONFIG,
        write_methods=["csv"],
        fallback_output_location=str(workdir),
    )
    return workdir / "output_csv" / scenario


def _read_costs_dt(csv_dir: Path) -> pd.DataFrame:
    """Read costs__dt.csv and return as a DataFrame."""
    return pd.read_csv(csv_dir / "costs__dt.csv", index_col=[0, 1, 2])


def _read_slack_up_dt(csv_dir: Path) -> pd.DataFrame:
    """Read the upward node state slack CSV."""
    return pd.read_csv(
        csv_dir / "slack__upward__node_state__dt.csv", index_col=[0, 1, 2]
    )


def _read_slack_capacity_margin_d(csv_dir: Path) -> pd.DataFrame:
    """Read the capacity margin slack CSV."""
    return pd.read_csv(
        csv_dir / "slack__capacity_margin__d.csv", index_col=[0, 1]
    )


# ===================================================================
# Test: vq_state_up — base scenario (no supply, 100% slack)
# ===================================================================

class TestVqStateUpBase:
    """Verify vq_state_up penalties in the 'base' scenario.

    The base scenario has:
      - One node ('west') with demand (negative inflow) and no supply units
      - penalty_up = 900 CUR/MWh, step_duration = 1h
      - All demand must be met by vq_state_up (upward slack)

    Expected: slack_MW[t] = |inflow[t]| for every timestep
              penalty_cost[t] = slack_MW[t] * 900 * 1
    """

    @pytest.fixture(scope="class")
    def base_csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        """Run the base scenario once for all tests in this class."""
        workdir = tmp_path_factory.mktemp("vq_base")
        import os
        os.chdir(workdir)
        return _run_scenario("base", test_db_url, test_bin_dir, workdir)

    def test_slack_equals_demand(self, base_csv_dir: Path) -> None:
        """Every timestep's upward slack equals the absolute demand (no supply)."""
        slack = _read_slack_up_dt(base_csv_dir)
        expected_slack = [abs(x) for x in WEST_INFLOW_48]
        actual_slack = slack["west"].values.tolist()
        assert len(actual_slack) == 48
        for t, (actual, expected) in enumerate(zip(actual_slack, expected_slack)):
            assert actual == pytest.approx(expected, abs=0.1), (
                f"Timestep t{t+1:04d}: slack={actual}, expected={expected}"
            )

    def test_penalty_cost_formula(self, base_csv_dir: Path) -> None:
        """Upward slack penalty = slack_MW * penalty_up * step_duration."""
        costs = _read_costs_dt(base_csv_dir)
        for t_idx in range(48):
            demand_mw = abs(WEST_INFLOW_48[t_idx])
            expected_cost = demand_mw * PENALTY_UP_WEST * STEP_DURATION
            actual_cost = costs["upward slack penalty"].iloc[t_idx]
            assert actual_cost == pytest.approx(expected_cost, rel=1e-4), (
                f"Timestep {t_idx}: cost={actual_cost}, "
                f"expected={demand_mw}*{PENALTY_UP_WEST}*{STEP_DURATION}={expected_cost}"
            )

    def test_no_downward_slack(self, base_csv_dir: Path) -> None:
        """No downward slack penalty (no oversupply possible without units)."""
        costs = _read_costs_dt(base_csv_dir)
        assert (costs["downward slack penalty"] == 0).all()

    def test_no_commodity_cost(self, base_csv_dir: Path) -> None:
        """No commodity cost (no units, no fuel consumption)."""
        costs = _read_costs_dt(base_csv_dir)
        assert (costs["commodity_cost"] == 0).all()

    def test_total_penalty_cost(self, base_csv_dir: Path) -> None:
        """Sum of all penalty costs matches expected total."""
        costs = _read_costs_dt(base_csv_dir)
        total_demand_mwh = sum(abs(x) for x in WEST_INFLOW_48) * STEP_DURATION
        expected_total = total_demand_mwh * PENALTY_UP_WEST
        actual_total = costs["upward slack penalty"].sum()
        assert actual_total == pytest.approx(expected_total, rel=1e-4), (
            f"Total penalty cost: {actual_total}, expected: {expected_total}"
        )


# ===================================================================
# Test: vq_state_up — coal scenario (partial slack)
# ===================================================================

class TestVqStateUpCoal:
    """Verify vq_state_up penalties in the 'coal' scenario.

    The coal scenario has:
      - One node ('west') with demand and one coal plant (500 MW)
      - When demand > 500 MW, the shortfall hits vq_state_up
      - When demand <= 500 MW, no slack needed
      - penalty_up = 900 CUR/MWh
      - commodity_cost = generation_MW / efficiency * fuel_price * step_duration
    """

    @pytest.fixture(scope="class")
    def coal_csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        """Run the coal scenario once for all tests in this class."""
        workdir = tmp_path_factory.mktemp("vq_coal")
        import os
        os.chdir(workdir)
        return _run_scenario("coal", test_db_url, test_bin_dir, workdir)

    def test_no_slack_when_supply_sufficient(self, coal_csv_dir: Path) -> None:
        """No upward slack when demand <= coal capacity (500 MW)."""
        costs = _read_costs_dt(coal_csv_dir)
        for t_idx in range(48):
            demand = abs(WEST_INFLOW_48[t_idx])
            if demand <= COAL_EXISTING_MW:
                actual = costs["upward slack penalty"].iloc[t_idx]
                assert actual == pytest.approx(0, abs=0.01), (
                    f"Timestep {t_idx}: demand={demand} <= {COAL_EXISTING_MW}, "
                    f"but slack penalty={actual}"
                )

    def test_slack_equals_shortfall(self, coal_csv_dir: Path) -> None:
        """Upward slack = demand - coal_capacity when demand > 500 MW."""
        slack = _read_slack_up_dt(coal_csv_dir)
        for t_idx in range(48):
            demand = abs(WEST_INFLOW_48[t_idx])
            if demand > COAL_EXISTING_MW:
                expected_slack = demand - COAL_EXISTING_MW
                actual_slack = slack["west"].iloc[t_idx]
                assert actual_slack == pytest.approx(expected_slack, abs=0.1), (
                    f"Timestep {t_idx}: demand={demand}, slack={actual_slack}, "
                    f"expected={expected_slack}"
                )

    def test_penalty_cost_matches_shortfall(self, coal_csv_dir: Path) -> None:
        """Upward slack penalty cost = shortfall * penalty_up * step_duration."""
        costs = _read_costs_dt(coal_csv_dir)
        for t_idx in range(48):
            demand = abs(WEST_INFLOW_48[t_idx])
            shortfall = max(0, demand - COAL_EXISTING_MW)
            expected_cost = shortfall * PENALTY_UP_WEST * STEP_DURATION
            actual_cost = costs["upward slack penalty"].iloc[t_idx]
            assert actual_cost == pytest.approx(expected_cost, rel=1e-4), (
                f"Timestep {t_idx}: demand={demand}, shortfall={shortfall}, "
                f"penalty_cost={actual_cost}, expected={expected_cost}"
            )

    def test_commodity_cost_at_full_load(self, coal_csv_dir: Path) -> None:
        """When demand >= 500 MW, coal runs at full capacity.

        commodity_cost = 500 MW / 0.4 efficiency * 20 CUR/MWh * 1h = 25000 CUR
        """
        costs = _read_costs_dt(coal_csv_dir)
        expected_full_load_cost = (
            COAL_EXISTING_MW / COAL_EFFICIENCY * COAL_PRICE * STEP_DURATION
        )
        for t_idx in range(48):
            demand = abs(WEST_INFLOW_48[t_idx])
            if demand >= COAL_EXISTING_MW:
                actual = costs["commodity_cost"].iloc[t_idx]
                assert actual == pytest.approx(expected_full_load_cost, rel=1e-4), (
                    f"Timestep {t_idx}: demand={demand}, "
                    f"commodity_cost={actual}, expected={expected_full_load_cost}"
                )

    def test_commodity_cost_at_partial_load(self, coal_csv_dir: Path) -> None:
        """When demand < 500 MW, coal matches demand exactly.

        commodity_cost = demand_MW / efficiency * fuel_price * step_duration
        """
        costs = _read_costs_dt(coal_csv_dir)
        for t_idx in range(48):
            demand = abs(WEST_INFLOW_48[t_idx])
            if demand < COAL_EXISTING_MW:
                expected_cost = demand / COAL_EFFICIENCY * COAL_PRICE * STEP_DURATION
                actual = costs["commodity_cost"].iloc[t_idx]
                assert actual == pytest.approx(expected_cost, rel=1e-4), (
                    f"Timestep {t_idx}: demand={demand}, "
                    f"commodity_cost={actual}, expected={expected_cost}"
                )


# ===================================================================
# Test: vq_capacity_margin
# ===================================================================

class TestVqCapacityMargin:
    """Verify vq_capacity_margin penalties in the 'capacity_margin' scenario.

    The capacity_margin scenario has:
      - The 'base' model (west node, no units) PLUS
      - A group 'capacity_margin' with:
        - has_capacity_margin = 'yes'
        - capacity_margin = 100 MW
        - penalty_capacity_margin = 1,000,000 CUR/MW
      - invest_periods includes p2020 (needed for capacity margin to activate)
      - Nodes: west, east, north (all in the capacity_margin group)

    The capacity margin constraint requires that installed capacity
    exceeds peak demand by 'capacity_margin' MW. With no units,
    ALL demand + margin is unmet.
    """

    @pytest.fixture(scope="class")
    def cm_csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        """Run the capacity_margin scenario once."""
        workdir = tmp_path_factory.mktemp("vq_cm")
        import os
        os.chdir(workdir)
        return _run_scenario("capacity_margin", test_db_url, test_bin_dir, workdir)

    def test_capacity_margin_slack_positive(self, cm_csv_dir: Path) -> None:
        """Capacity margin slack should be positive (insufficient capacity)."""
        slack = _read_slack_capacity_margin_d(cm_csv_dir)
        assert (slack["capacity_margin"] > 0).all(), (
            f"Expected positive capacity margin slack, got: {slack['capacity_margin'].values}"
        )

    def test_capacity_margin_slack_has_expected_columns(self, cm_csv_dir: Path) -> None:
        """The capacity margin slack CSV should contain the group name as column."""
        slack = _read_slack_capacity_margin_d(cm_csv_dir)
        assert "capacity_margin" in slack.columns


# ===================================================================
# Test: vq_state_up cost is strictly proportional to penalty_up
# ===================================================================

class TestVqStateUpProportionality:
    """Verify that the penalty cost is exactly proportional to penalty_up.

    Using the base scenario: cost[t] / slack[t] should equal
    penalty_up * step_duration for every timestep.
    """

    @pytest.fixture(scope="class")
    def base_csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("vq_prop")
        import os
        os.chdir(workdir)
        return _run_scenario("base", test_db_url, test_bin_dir, workdir)

    def test_cost_per_mwh_equals_penalty(self, base_csv_dir: Path) -> None:
        """cost / (slack * step_duration) = penalty_up for every timestep."""
        costs = _read_costs_dt(base_csv_dir)
        slack = _read_slack_up_dt(base_csv_dir)

        for t_idx in range(48):
            slack_mw = slack["west"].iloc[t_idx]
            cost = costs["upward slack penalty"].iloc[t_idx]
            if slack_mw > 0:
                implied_penalty = cost / (slack_mw * STEP_DURATION)
                assert implied_penalty == pytest.approx(PENALTY_UP_WEST, rel=1e-4), (
                    f"Timestep {t_idx}: implied penalty={implied_penalty}, "
                    f"expected={PENALTY_UP_WEST}"
                )


# ===================================================================
# Test: Consistency between slack output and cost output
# ===================================================================

class TestSlackCostConsistency:
    """Cross-check that slack * penalty = cost for the coal scenario."""

    @pytest.fixture(scope="class")
    def coal_csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("vq_consistency")
        import os
        os.chdir(workdir)
        return _run_scenario("coal", test_db_url, test_bin_dir, workdir)

    def test_slack_times_penalty_equals_cost(self, coal_csv_dir: Path) -> None:
        """slack_MW * step_duration * penalty_up = upward_slack_penalty for every timestep.

        The slack CSV reports MW (vq_state_up * node_capacity_for_scaling).
        The cost CSV reports CUR (slack_MW * step_duration * penalty_up).
        With step_duration=1h these are numerically equal: slack_MW * 900.
        """
        costs = _read_costs_dt(coal_csv_dir)
        slack = _read_slack_up_dt(coal_csv_dir)

        for t_idx in range(48):
            slack_mw = slack["west"].iloc[t_idx]
            cost = costs["upward slack penalty"].iloc[t_idx]
            expected_cost = slack_mw * STEP_DURATION * PENALTY_UP_WEST
            assert cost == pytest.approx(expected_cost, abs=0.1), (
                f"Timestep {t_idx}: slack_MW={slack_mw}, cost={cost}, "
                f"expected={expected_cost}"
            )
