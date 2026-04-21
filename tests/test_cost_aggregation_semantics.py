"""Cost-aggregation semantic tests (P3b audit).

Purpose
-------
These tests pin down the exact factors the Python post-processing in
``flextool/process_outputs/calc_costs.py`` and
``flextool/process_outputs/calc_slacks.py`` must apply so that the
period-level and horizon-level cost totals it publishes match the LP
objective value reported by the solver.

Each test runs an existing small scenario, reads the CSV outputs that the
normal pipeline produces (``summary_solve.csv``, ``costs__dt.csv``,
``slack__*__d.csv``, etc.), and compares them to hand-derived expected
values computed from the inputs and the variable values.

Running
-------
.. code-block:: bash

   pytest tests/test_cost_aggregation_semantics.py -v

Do NOT run the full ``tests/`` suite from here -- another agent is working
on scenario golden files concurrently.

Variables covered
-----------------
* ``vq_state_up / vq_state_down``         — per-timestep node slack (MW)
* ``vq_capacity_margin``                  — per-period group slack (MW)
* ``vq_inertia``                          — per-timestep dimensionless slack
* ``vq_non_synchronous``                  — per-timestep group slack (MW)
* ``p_rp_cost_weight``                    — per-(period, timestep) weight
* Pre-existing fixed cost / divest fixed cost (constants)

Semantic classes
----------------
The mod objective for each cost term uses a characteristic scaling set.
After aggregation, the Python side must apply the same set exactly:

================================  =================================================
term                              Python factors that must appear
================================  =================================================
commodity/co2/varCost/reserve     flow  x step_duration x rp_weight x inflation
startup                           startup x unitsize x rp_weight x inflation
node-state slack                  q x capacity x step_duration x rp_weight
                                  x penalty x inflation
inertia slack                     q x inertia_limit x step_duration x rp_weight
                                  x penalty_inertia x inflation
non-sync slack                    q x group_capacity x step_duration x rp_weight
                                  x penalty_non_sync x inflation
reserve slack                     q x reservation x step_duration x rp_weight
                                  x penalty_reserve x inflation
capacity-margin slack             q x group_capacity x penalty x 1000 x inflation
                                  (NO step_duration, NO rp_weight — period event)
================================  =================================================
"""
from __future__ import annotations

import csv
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
# Helpers
# ---------------------------------------------------------------------------

def _run_scenario(
    scenario: str,
    test_db_url: str,
    test_bin_dir: Path,
    workdir: Path,
) -> Path:
    """Run the FlexTool pipeline end-to-end for a scenario. Returns the
    ``output_csv/<scenario>`` directory produced by ``write_outputs``.
    """
    import os
    os.chdir(workdir)

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


def _read_summary_solve(csv_dir: Path) -> dict:
    """Parse ``summary_solve.csv`` into a flat dict of the top-level totals.

    Returns keys like ``objective`` (M CUR, from solver), ``total_calc`` (M
    CUR, Python full-horizon calculated total), ``penalty_total`` (M CUR),
    etc.  Values past the first comma-separated tokens are ignored.
    """
    rows = []
    with open(csv_dir / "summary_solve.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                rows.append(row)

    out: dict[str, float] = {}
    for row in rows:
        if not row:
            continue
        label = row[0]
        if label.startswith('"Solve"'):
            continue
        # The solver objective is on the line after '"Solve",...' — scan all
        # rows and pick any numeric second column that isn't another label.
        if "Total cost (calculated) full horizon" in label:
            out["total_calc"] = float(row[1])
        elif "Operational costs for realized periods" in label:
            out["operational"] = float(row[1])
        elif "Investment costs for realized periods" in label:
            out["investment"] = float(row[1])
        elif "Retirement costs" in label:
            out["retirement"] = float(row[1])
        elif "Fixed costs for pre-existing entities" in label:
            out["fixed_pre_existing"] = float(row[1])
        elif "Fixed costs for invested entities" in label:
            out["fixed_invested"] = float(row[1])
        elif "Fixed cost removal" in label:
            out["fixed_divested"] = float(row[1])
        elif "Penalty (slack) costs" in label:
            out["penalty"] = float(row[1])

    # The solver objective line is the 5th line (row index 4 after header blanks)
    # matching the pattern '<solve_name>,<value>'.  Pick the first such line.
    for row in rows:
        if (
            len(row) >= 2
            and not row[0].startswith('"')
            and not row[0].startswith("Created")
            and not row[0].startswith("CapMargin")
            and not row[0].startswith("Period")
            and "objective" not in row[0].lower()
            and row[0]  # non-empty
        ):
            try:
                out["objective"] = float(row[1])
                break
            except (ValueError, IndexError):
                continue
    return out


def _read_costs_dt(csv_dir: Path) -> pd.DataFrame:
    return pd.read_csv(csv_dir / "costs__dt.csv", index_col=[0, 1, 2])


def _read_slack_up_dt(csv_dir: Path) -> pd.DataFrame:
    return pd.read_csv(csv_dir / "slack__upward__node_state__dt.csv", index_col=[0, 1, 2])


def _read_slack_capacity_margin_d(csv_dir: Path) -> pd.DataFrame:
    return pd.read_csv(csv_dir / "slack__capacity_margin__d.csv", index_col=[0, 1])


# ---------------------------------------------------------------------------
# Known inputs (copied from tests/test_vq_penalties.py — same fixture)
# ---------------------------------------------------------------------------

# ``west`` node inflow for the 48 timesteps of the 2day timeset (alternative
# ``west`` in the JSON fixture).  Negative = demand.
WEST_INFLOW_48 = [
    -589, -537, -506, -482, -472, -454, -423, -438,
    -474, -518, -538, -558, -572, -564, -553, -541,
    -534, -567, -689, -696, -670, -627, -550, -458,
    -436, -446, -436, -452, -469, -507, -549, -594,
    -682, -730, -774, -802, -807, -793, -780, -768,
    -755, -767, -872, -865, -808, -735, -669, -597,
]
PENALTY_UP_WEST = 900.0          # CUR/MWh
PENALTY_DOWN_WEST = 800.0        # CUR/MWh
STEP_DURATION_H = 1.0            # 1-hour timesteps in the y2020 timeline
PERIOD_SHARE_2DAY = 48.0 / 8760  # 48h / year

# Weighted-timeset (used by ``base_weighted``): raw weights 0.5/1.0/1.5/2.0
# for the four consecutive 12-h blocks.  Total = 60, scaled by 48/60 so
# every block ends up at 0.4/0.8/1.2/1.6 (mean 1.0, preserving period
# share).
WEIGHTED_2DAY = [0.4] * 12 + [0.8] * 12 + [1.2] * 12 + [1.6] * 12

# ``capacity_margin`` scenario: group penalty and margin.
CAP_MARGIN_PENALTY = 1_000_000.0   # CUR/kW (→ × 1000 in objective to reach CUR/MW)
CAP_MARGIN_VALUE = 100.0           # MW (group capacity margin target)


# ===========================================================================
# Control: ``base`` — no rp_weight, no capacity margin, no invest.
#
# This control test is expected to PASS both before and after the fixes
# because no weighting mismatch is triggered.  If it ever fails, we know
# the aggregation got broken in an unrelated way.
# ===========================================================================

class TestBaseControl:
    @pytest.fixture(scope="class")
    def csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("ctrl_base")
        return _run_scenario("base", test_db_url, test_bin_dir, workdir)

    def test_solver_obj_matches_python_total(self, csv_dir: Path) -> None:
        s = _read_summary_solve(csv_dir)
        # Solver objective is reported in M CUR (already divided).  The
        # Python total should match within the solver's tolerance (this
        # is a pure-penalty problem, no presolve-dropped constant).
        assert s["objective"] == pytest.approx(s["total_calc"], rel=1e-4), (
            f"objective={s['objective']} vs total_calc={s['total_calc']}"
        )

    def test_hand_derived_penalty_matches(self, csv_dir: Path) -> None:
        """Penalty = Σ |demand| * penalty_up / period_share (annualized).

        With uniform weights and 1-h steps, rp_cost_weight == 1 everywhere
        and ``step_duration`` is 1, so the hand-calc is just
        Σ demand * penalty / period_share.
        """
        s = _read_summary_solve(csv_dir)
        total_mwh = sum(abs(x) for x in WEST_INFLOW_48) * STEP_DURATION_H
        expected_M = total_mwh * PENALTY_UP_WEST / PERIOD_SHARE_2DAY / 1e6
        assert s["penalty"] == pytest.approx(expected_M, rel=1e-4), (
            f"penalty={s['penalty']} vs expected={expected_M}"
        )
        assert s["objective"] == pytest.approx(expected_M, rel=1e-4)


# ===========================================================================
# Target: ``base_weighted`` — non-uniform rp_cost_weight.
#
# EXPECTED TO FAIL on current code (missing rp_cost_weight in Python
# aggregation).  PASSES after calc_slacks and calc_costs are fixed.
# ===========================================================================

class TestRpCostWeightSlackPenalty:
    @pytest.fixture(scope="class")
    def csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("rpw_base_weighted")
        return _run_scenario("base_weighted", test_db_url, test_bin_dir, workdir)

    def test_solver_matches_python(self, csv_dir: Path) -> None:
        """The decisive test.  LP objective (from HiGHS) must equal the
        full-horizon calculated total.  Delta on current code is the
        missing ``rp_cost_weight`` factor on the slack penalty.
        """
        s = _read_summary_solve(csv_dir)
        assert s["objective"] == pytest.approx(s["total_calc"], rel=1e-4), (
            f"Objective {s['objective']} != Python total {s['total_calc']} "
            f"(delta = {s['objective'] - s['total_calc']:+.4f} M CUR)"
        )

    def test_weighted_hand_calc(self, csv_dir: Path) -> None:
        """Penalty = Σ |demand[t]| * penalty_up * step_duration *
        rp_cost_weight[t] / period_share.
        """
        s = _read_summary_solve(csv_dir)
        weighted_mwh = sum(
            abs(x) * w * STEP_DURATION_H
            for x, w in zip(WEST_INFLOW_48, WEIGHTED_2DAY)
        )
        expected_M = weighted_mwh * PENALTY_UP_WEST / PERIOD_SHARE_2DAY / 1e6
        assert s["penalty"] == pytest.approx(expected_M, rel=1e-4), (
            f"penalty={s['penalty']} vs expected (weighted)={expected_M}"
        )


# ===========================================================================
# Target: ``capacity_margin`` — vq_capacity_margin slack.
#
# EXPECTED TO FAIL on current code (missing `× penalty_capacity_margin ×
# 1000` on the capacity margin penalty).  PASSES after calc_slacks is fixed.
# ===========================================================================

class TestCapacityMarginPenalty:
    @pytest.fixture(scope="class")
    def csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("cm_cap_margin")
        return _run_scenario("capacity_margin", test_db_url, test_bin_dir, workdir)

    def test_solver_matches_python(self, csv_dir: Path) -> None:
        s = _read_summary_solve(csv_dir)
        # Huge delta expected here on current code: capacity_margin_penalty
        # is (1577.6 × 1 × 1_000_000 × 1000) / 1e6 ≈ 1.58e9 M CUR, but
        # Python only reports (1577.6 × 1 × 1) / 1e6 ≈ 0.0016 M CUR.
        assert s["objective"] == pytest.approx(s["total_calc"], rel=1e-4), (
            f"Objective {s['objective']} != Python total {s['total_calc']} "
            f"(delta = {s['objective'] - s['total_calc']:+.2f} M CUR)"
        )

    def test_capacity_margin_hand_calc(self, csv_dir: Path) -> None:
        """cap-margin penalty = vq * group_cap * penalty * 1000 * inflation.

        ``group_capacity_for_scaling`` = 1.0 (default when no cap present).
        ``inflation_factor_operations_yearly`` = 1.0 (single 2020 period,
        inflation = 0).

        In this scenario the capacity_margin group has three nodes
        (west, east, north).  Instead of enumerating their individual
        demands, we read the total upward-slack MWh from the
        ``slack__upward__node_state__dt.csv`` output and annualize it
        exactly the way the objective does.
        """
        s = _read_summary_solve(csv_dir)
        slack_cm = _read_slack_capacity_margin_d(csv_dir)
        vq = slack_cm["capacity_margin"].iloc[0]  # MW
        expected_cap_margin_penalty_M = vq * 1.0 * CAP_MARGIN_PENALTY * 1000.0 * 1.0 / 1e6

        slack_up = _read_slack_up_dt(csv_dir)
        total_mwh = slack_up.sum().sum()  # MW × 1-h steps sums to MWh
        upward_penalty_M = total_mwh * PENALTY_UP_WEST / PERIOD_SHARE_2DAY / 1e6

        expected_total = upward_penalty_M + expected_cap_margin_penalty_M
        assert s["penalty"] == pytest.approx(expected_total, rel=1e-4), (
            f"penalty={s['penalty']} vs expected cap_margin "
            f"+ upward = {expected_total} "
            f"(cap_margin_alone={expected_cap_margin_penalty_M}, "
            f"upward_alone={upward_penalty_M}, total_slack_mwh={total_mwh})"
        )


# ===========================================================================
# Target: ``coal`` — control for commodity + variable operational cost.
#
# Expected to PASS both before and after; no rp weighting and no
# capacity-margin slack are involved.
# ===========================================================================

class TestCoalControl:
    @pytest.fixture(scope="class")
    def csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("ctrl_coal")
        return _run_scenario("coal", test_db_url, test_bin_dir, workdir)

    def test_solver_matches_python(self, csv_dir: Path) -> None:
        s = _read_summary_solve(csv_dir)
        assert s["objective"] == pytest.approx(s["total_calc"], rel=1e-4), (
            f"Objective {s['objective']} != Python total {s['total_calc']}"
        )


# ===========================================================================
# Target: ``coal_wind_inertia`` — inertia group present.
#
# Not forcing a violation, but still verifying that the python total
# matches the solver.  If the group has no violation, vq_inertia = 0 and
# the missing step_duration on the inertia penalty does not manifest.
# This test is a regression guard for the inertia column rather than a
# new-bug detector.
# ===========================================================================

class TestCoalWindInertiaControl:
    @pytest.fixture(scope="class")
    def csv_dir(
        self, test_db_url: str, test_bin_dir: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> Path:
        workdir = tmp_path_factory.mktemp("ctrl_inertia")
        return _run_scenario("coal_wind_inertia", test_db_url, test_bin_dir, workdir)

    def test_solver_matches_python(self, csv_dir: Path) -> None:
        s = _read_summary_solve(csv_dir)
        assert s["objective"] == pytest.approx(s["total_calc"], rel=1e-4), (
            f"Objective {s['objective']} != Python total {s['total_calc']}"
        )


# ===========================================================================
# TODO items documented as xfail markers (user-decisions).
# ===========================================================================

@pytest.mark.xfail(
    reason=(
        "TODO(user): calc_costs.cost_entity_divest_d uses "
        "entity_annual_divest_discounted (salvage annuity) which has no "
        "counterpart in the mod objective. The objective uses "
        "ed_lifetime_fixed_cost_divest only. Decide: is the salvage "
        "annuity an intentional Python-only accounting item (then this "
        "test should be removed), or is it a bug (then cost_entity_divest_d "
        "should be changed)? See calc_costs.py:103."
    ),
    strict=False,
)
def test_divest_salvage_matches_mod() -> None:
    # Placeholder — no scenario in the fixture exercises divestment with
    # salvage value large enough to distinguish the two formulas.
    assert False, "Needs user decision"


# ===========================================================================
# Target: ``coal_min_load_wind`` with a varCost added on the source
# (coal_plant, coal_market).  The mod objective for pssdt_varCost_eff_unit_source
# adds both a flow-slope term AND a section term when the process uses
# ``min_load_efficiency`` (flextool.mod ~line 2352-2367):
#
#     + sum {(p,source,sink,d,t) in pssdt_varCost_eff_unit_source}
#         ( - pdtProcess_source[p, source, 'other_operational_cost', d, t]
#             * ( v_flow * unitsize * slope * (sink_coef / source_coef)
#                 + online * section * unitsize   (if min_load_efficiency)
#               )
#             * step_duration * rp_cost_weight * inflation / period_share
#             * branch_weight
#         )
#
# The Python aggregation in calc_costs.py line 113-115 intersects
# ``r.flow_dt`` with ``par.process_source_sink_varCost``.  The source-side
# entry of ``r.flow_dt`` already contains ``slope*v_flow*unitsize +
# section*online*unitsize`` (set up in calc_capacity_flows.py line 49-59),
# so multiplying by varCost recovers BOTH terms.  Multiply by
# ``step_duration * rp_cost_weight`` to match the objective.
#
# This test is the regression guard for that algebra: we add a non-zero
# ``other_operational_cost`` on (coal_plant, coal_market) inside the
# existing ``coal_min_load`` alternative of the test fixture DB, run the
# ``coal_min_load_wind`` scenario, and verify the ``other operational``
# bucket in ``costs__dt.csv`` matches the exact hand-derived formula.
# ===========================================================================

class TestMinLoadEfficiencySectionTerm:
    """When a min_load_efficiency process has an ``other_operational_cost``
    on its source (fuel) edge, the Python ``other operational`` bucket must
    include BOTH the slope (variable, proportional to output) AND the
    section (no-load, proportional to online hours) terms to match the LP
    objective.

    The hand-derivation (per (d, t), for coal_plant):

        bucket_dt = (flow_source_dt * varCost) * step * rp_weight
                  = ( slope * v_flow * unitsize
                    + section * online * unitsize ) * varCost * step * rp_weight

    where ``slope`` and ``section`` are piecewise-linear conversion curve
    parameters (``pdtProcess_slope``, ``pdtProcess_section``), derived in
    flextool.mod lines 1548-1560 from ``efficiency``, ``min_load``, and
    ``efficiency_at_min_load``.
    """

    VAR_COST = 5.0   # CUR/MWh on the coal_plant <- coal_market input edge

    @pytest.fixture(scope="class")
    def patched_db_url(
        self, test_db_url: str, tmp_path_factory: pytest.TempPathFactory
    ) -> str:
        """Clone the session DB and add ``other_operational_cost`` on
        ``unit__inputNode (coal_plant, coal_market)`` in the existing
        ``coal_min_load`` alternative.  All downstream scenarios that
        inherit that alternative (including ``coal_min_load_wind``) pick
        up the new value without having to modify the JSON fixture.
        """
        import shutil
        from spinedb_api import DatabaseMapping, import_data

        src_path = Path(test_db_url.replace("sqlite:///", ""))
        dst_dir = tmp_path_factory.mktemp("min_load_section_db")
        dst_path = dst_dir / "tests_patched.sqlite"
        shutil.copy(src_path, dst_path)
        url = f"sqlite:///{dst_path.resolve()}"

        with DatabaseMapping(url) as db:
            count, errors = import_data(
                db,
                parameter_values=[
                    (
                        "unit__inputNode",
                        ("coal_plant", "coal_market"),
                        "other_operational_cost",
                        self.VAR_COST,
                        "coal_min_load",
                    ),
                ],
            )
            assert not errors, f"Import errors: {errors}"
            db.commit_session("Add other_operational_cost on coal_plant source")
        return url

    @pytest.fixture(scope="class")
    def csv_dir(
        self,
        patched_db_url: str,
        test_bin_dir: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        workdir = tmp_path_factory.mktemp("min_load_section_run")
        return _run_scenario(
            "coal_min_load_wind", patched_db_url, test_bin_dir, workdir,
        )

    def test_other_operational_matches_hand_derived(self, csv_dir: Path) -> None:
        """Hand-derive the expected ``other operational`` bucket from the
        output flow and online CSVs, and compare with the Python pipeline's
        published value in ``costs__dt.csv``.
        """
        # Actual Python-published value of the "other operational" bucket.
        costs_dt = pd.read_csv(csv_dir / "costs__dt.csv", index_col=[0, 1, 2])
        actual = costs_dt["other operational"]

        # Inputs needed for the hand calc:
        #   - v_flow (MW output at sink, via unit__outputNode__dt.csv)
        #   - v_online (unit_online__dt.csv)
        #   - slope, section (pdtProcess_slope, pdtProcess_section)
        #   - unitsize, step_duration, rp_cost_weight
        # All live under the solver/work dir -- work folder is csv_dir's
        # parent-parent (``output_csv/<scenario>`` inside workdir).
        workdir = csv_dir.parent.parent
        solve_data = workdir / "solve_data"
        input_dir = workdir / "input"

        # v_flow output at west (1-unit-scaled -- we multiply by unitsize below).
        # unit__outputNode__dt.csv has a 2-row header: [unit, node].
        flow_out = pd.read_csv(
            csv_dir / "unit__outputNode__dt.csv",
            header=[0, 1],
            index_col=[0, 1, 2],
        )
        flow_out.index.names = ["solve", "period", "time"]
        # Series indexed by (solve, period, time) giving MW output of coal_plant.
        flow_coal = flow_out[("coal_plant", "west")].astype(float)

        # v_online for coal_plant (0..1 linear variable in coal_min_load_wind).
        online = pd.read_csv(
            csv_dir / "unit_online__dt.csv", index_col=[0, 1, 2],
        )
        online.index.names = ["solve", "period", "time"]
        online_coal = online["coal_plant"].astype(float)

        # pdtProcess_slope and pdtProcess_section (per (d, t)).
        slope = pd.read_csv(
            solve_data / "pdtProcess_slope.csv", index_col=[0, 1, 2],
        )["coal_plant"].astype(float)
        slope.index.names = ["solve", "period", "time"]
        section = pd.read_csv(
            solve_data / "pdtProcess_section.csv", index_col=[0, 1, 2],
        )["coal_plant"].astype(float)
        section.index.names = ["solve", "period", "time"]

        # Entity unitsize (virtual_unitsize). coal_plant has virtual_unitsize
        # = 250 in the coal_unit_size alternative -- but coal_min_load_wind
        # does NOT include that alternative, so unitsize defaults to 1.0.
        # Read it from the written input CSV to avoid hard-coding.
        unitsize_df = pd.read_csv(
            input_dir / "p_entity_unitsize.csv", index_col=0,
        )
        unitsize_coal = float(unitsize_df.loc["value", "coal_plant"])

        # Per-step scaling factor used by compute_costs.
        step_duration = pd.read_csv(
            solve_data / "p_step_duration.csv", index_col=[0, 1, 2],
        )["value"].astype(float)
        step_duration.index.names = ["solve", "period", "time"]
        rp_cost_weight = pd.read_csv(
            solve_data / "p_rp_cost_weight.csv", index_col=[0, 1, 2],
        )["value"].astype(float)
        rp_cost_weight.index.names = ["solve", "period", "time"]

        # Align everything on the actual cost index.
        idx = actual.index

        # Source-side fuel flow at each timestep:
        #   slope * v_flow_output + section * online * unitsize
        # v_flow_output at west is already in physical MW (= v_flow*unitsize),
        # so dividing by unitsize recovers per-unit v_flow for the mod's
        # ``v_flow * unitsize * slope`` term.  Since sink_coef = source_coef
        # = 1 here, the mod's ``sink_coef/source_coef`` multiplier is 1.
        flow_source = (
            slope.reindex(idx).mul(flow_coal.reindex(idx), fill_value=0.0)
            + section.reindex(idx).mul(online_coal.reindex(idx), fill_value=0.0)
              * unitsize_coal
        )

        # Hand-derived bucket:
        expected = (
            flow_source
            * self.VAR_COST
            * step_duration.reindex(idx)
            * rp_cost_weight.reindex(idx)
        )

        # The bucket should be non-trivially non-zero somewhere (sanity).
        assert expected.abs().sum() > 0.0, (
            "Hand-derived expected bucket is identically zero -- "
            "the scenario is not exercising the min_load_efficiency path."
        )

        # Assert elementwise match.  rtol 1e-4 = same precision as goldens.
        pd.testing.assert_series_equal(
            actual.rename("expected"),
            expected.rename("expected"),
            check_names=False,
            rtol=1e-4,
            atol=1e-6,
        )

    def test_section_term_actually_contributes(self, csv_dir: Path) -> None:
        """Positive control: without the section term, the hand-derived
        bucket would be ``slope*v_flow * varCost * step * rpw`` only.
        Verify that the section contribution (section*online*unitsize *
        varCost * step * rpw) is numerically non-negligible -- so the
        main assertion above actually exercises the section path and is
        not vacuously satisfied by section = 0.
        """
        workdir = csv_dir.parent.parent
        solve_data = workdir / "solve_data"
        input_dir = workdir / "input"

        online = pd.read_csv(
            csv_dir / "unit_online__dt.csv", index_col=[0, 1, 2],
        )["coal_plant"].astype(float)
        online.index.names = ["solve", "period", "time"]
        section = pd.read_csv(
            solve_data / "pdtProcess_section.csv", index_col=[0, 1, 2],
        )["coal_plant"].astype(float)
        section.index.names = ["solve", "period", "time"]
        step_duration = pd.read_csv(
            solve_data / "p_step_duration.csv", index_col=[0, 1, 2],
        )["value"].astype(float)
        step_duration.index.names = ["solve", "period", "time"]
        rp_cost_weight = pd.read_csv(
            solve_data / "p_rp_cost_weight.csv", index_col=[0, 1, 2],
        )["value"].astype(float)
        rp_cost_weight.index.names = ["solve", "period", "time"]

        unitsize_coal = float(
            pd.read_csv(input_dir / "p_entity_unitsize.csv", index_col=0)
            .loc["value", "coal_plant"]
        )

        idx = online.index
        section_contrib = (
            section.reindex(idx)
            * online.reindex(idx)
            * unitsize_coal
            * self.VAR_COST
            * step_duration.reindex(idx)
            * rp_cost_weight.reindex(idx)
        )
        # Sanity: section term must be a meaningful share of the bucket --
        # otherwise the main assertion doesn't actually exercise the code
        # path.  Empirically ~9200 CUR for coal_min_load_wind + VAR_COST=5.
        assert section_contrib.abs().sum() > 100.0, (
            f"Section contribution is negligible ({section_contrib.abs().sum()}); "
            "scenario does not reliably exercise the section path."
        )


@pytest.mark.xfail(
    reason=(
        "TODO(user): calc_costs.cost_entity_fixed_pre_existing (line 104) "
        "uses par.entity_pre_existing.  The mod objective (line 2396) uses "
        "p_entity_all_existing.  For single-solve dispatch-only runs they "
        "are equal.  In multi-solve investment runs "
        "(p_model['solveFirst'] = False on later solves), they differ "
        "because p_entity_all_existing includes earlier-period v_invest "
        "carried forward.  Decide: fix or accept?"
    ),
    strict=False,
)
def test_pre_existing_vs_all_existing() -> None:
    assert False, "Needs user decision"


@pytest.mark.xfail(
    reason=(
        "TODO(user): Storage-state reference-price credit (mod line "
        "2390-2394) is not implemented in Python at all.  Subtracts "
        "p_storage_state_reference_price * v_state * unitsize from the "
        "objective at the last timestep of the last period.  No existing "
        "fixture uses use_reference_price, so no automated test possible "
        "yet.  Decide: add Python implementation?"
    ),
    strict=False,
)
def test_storage_state_reference_price_credit() -> None:
    assert False, "Needs user decision"


@pytest.mark.xfail(
    reason=(
        "TODO(user): pdt_branch_weight is computed in the mod but never "
        "written to CSV.  Python does not load branch weights.  Manifests "
        "in stochastic branching scenarios.  To fix: add a writer in "
        "flextool.mod for solve_data/pdt_branch_weight.csv, a loader in "
        "read_parameters.py, and a _PAR_DROP entry in drop_levels.py, "
        "then apply branch_weight to all per-(d,t) cost terms.  No "
        "fixture in tests.json currently exercises stochastic branches."
    ),
    strict=False,
)
def test_stochastic_branch_weight() -> None:
    assert False, "Needs branching fixture"
