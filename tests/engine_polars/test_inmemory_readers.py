"""Δ.31 — tests for the in-memory ``read_parameters`` / ``read_sets``.

Validates that the polars-driven replacements of
:mod:`flextool.process_outputs.read_parameters` and
:mod:`flextool.process_outputs.read_sets` produce a SimpleNamespace
shape that downstream :mod:`flextool.process_outputs.write_outputs`
can consume end-to-end on the fast single-solve path.

Scope is the **shape** of the resulting namespace — values are spot-
checked for the post-solve derived attributes
(``entity_all_capacity`` and the three sister capacity tables) which
are the dispatch's load-bearing case.  Numeric parity vs the slow /
CSV path lives in the broader test suite (out of scope for Δ.31).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from flextool.engine_polars import (
    build_flextool,
    load_flextool,
    run_single_solve_from_db,
)
from flextool.process_outputs.read_parameters import read_parameters
from flextool.process_outputs.read_sets import read_sets

from polar_high import Problem


pytestmark = pytest.mark.solver


# ---------------------------------------------------------------------------
# Shape sanity — verify the namespace carries every legacy attribute.
# ---------------------------------------------------------------------------


_REQUIRED_PARAM_ATTRS: tuple[str, ...] = (
    "node",
    "entity_unitsize",
    "commodity_co2_content",
    "process_sink_conversion_flow_coeff",
    "process_source_conversion_flow_coeff",
    "reserve_upDown_group_penalty",
    "step_duration",
    "rp_cost_weight",
    "flow_min",
    "flow_max",
    "process_source",
    "process_sink",
    "process_slope",
    "process_section",
    "process_availability",
    "process_source_sink_varCost",
    "node_self_discharge_loss",
    "node_penalty_up",
    "node_penalty_down",
    "node_inflow",
    "commodity_price",
    "group_co2_price",
    "reserve_upDown_group_reservation",
    "profile",
    "years_from_start_d",
    "years_represented_d",
    "entity_max_units",
    "entity_all_existing",
    "entity_pre_existing",
    "entity_all_capacity",
    "process_startup_cost",
    "entity_fixed_cost",
    "entity_lifetime_fixed_cost",
    "entity_lifetime_fixed_cost_divest",
    "node_annual_flow",
    "group_penalty_inertia",
    "group_penalty_non_synchronous",
    "group_penalty_capacity_margin",
    "group_inertia_limit",
    "group_capacity_margin",
    "entity_annuity",
    "entity_annual_discounted",
    "entity_annual_divest_discounted",
    "inflation_factor_operations_yearly",
    "inflation_factor_investment_yearly",
    "node_capacity_for_scaling",
    "group_capacity_for_scaling",
    "complete_period_share_of_year",
    "nested_model",
)


_REQUIRED_SET_ATTRS: tuple[str, ...] = (
    "entity",
    "node",
    "process",
    "entityInvest",
    "entityDivest",
    "process_unit",
    "process_connection",
    "process_profile",
    "process_online",
    "process_online_integer",
    "process_online_linear",
    "process_VRE",
    "process_source_sink",
    "process_method_sources_sinks",
    "process_method",
    "process__ct_method",
    "method_1var_per_way",
    "method_nvar",
    "period",
    "dt",
    "d_realized_period",
    "d_realize_invest",
    "d_realize_dispatch_or_invest",
    "dt_realize_dispatch",
    "dt_fix_storage_timesteps",
    "dtttdt",
    "dtt",
    "ed_invest",
    "ed_divest",
    "edd_invest",
    "process__node__profile__profile_method",
    "node_state",
    "node_balance",
    "node_balance_period",
    "node_commodity",
    "node_self_discharge",
    "node__storage_binding_method",
    "node__storage_start_end_method",
    "node__inflow_method",
    "node__storage_nested_fix_method",
    "process_source",
    "process_sink",
    "process__source__sink__profile__profile_method",
    "commodity_node",
    "commodity_node_co2",
    "process__commodity__node",
    "process__commodity__node_co2",
    "group_co2_price",
    "group_co2_limit",
    "groupInertia",
    "groupNonSync",
    "groupCapacityMargin",
    "group_node",
    "group_process",
    "group_process_node",
    "upDown",
    "enable_optional_outputs",
    "node_dc_power_flow",
    "connection_dc_power_flow",
    "connection_susceptance",
)


@pytest.fixture(scope="module")
def base_solution(scenario_workdir):
    """Fast solve of the ``work_base`` fixture, returning the
    OrchestrationStep."""
    fixture = scenario_workdir("base")
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        step = run_single_solve_from_db(
            f"sqlite:///{db}",
            scenario_name="base",
            work_folder=work,
        )
        yield step


def test_read_parameters_namespace_shape(base_solution):
    """Every legacy attribute is present on the namespace."""
    par = read_parameters(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    missing = [a for a in _REQUIRED_PARAM_ATTRS if not hasattr(par, a)]
    assert not missing, f"missing par attributes: {missing}"


def test_read_sets_namespace_shape(base_solution):
    """Every legacy set attribute is present on the namespace."""
    sets = read_sets(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    missing = [a for a in _REQUIRED_SET_ATTRS if not hasattr(sets, a)]
    assert not missing, f"missing s attributes: {missing}"


def test_read_parameters_step_duration_shape(base_solution):
    """``step_duration`` is a Series with (solve, period, time)
    MultiIndex."""
    par = read_parameters(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    assert isinstance(par.step_duration, pd.Series)
    assert par.step_duration.index.names == ["solve", "period", "time"]
    assert (par.step_duration > 0).all()
    # work_base has 48 timesteps in one period.
    assert len(par.step_duration) == 48


def test_read_parameters_node_inflow_shape(base_solution):
    """``node_inflow`` is wide-format (solve, period, time) × node."""
    par = read_parameters(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    assert isinstance(par.node_inflow, pd.DataFrame)
    assert par.node_inflow.index.names == ["solve", "period", "time"]
    assert par.node_inflow.columns.name == "node"
    assert "west" in par.node_inflow.columns
    assert len(par.node_inflow) == 48


def test_read_parameters_inflation_factors(base_solution):
    """Inflation factor Series with (solve, period) index, values=1.0
    on a fixture without inflation."""
    par = read_parameters(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    s = par.inflation_factor_operations_yearly
    assert isinstance(s, pd.Series)
    assert s.index.names == ["solve", "period"]
    # work_base has p_inflation_op = 1.0.
    assert s.iloc[0] == pytest.approx(1.0)


def test_read_sets_period_shape(base_solution):
    """``period`` set is a (solve, period) MultiIndex."""
    sets = read_sets(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    assert isinstance(sets.period, pd.MultiIndex)
    assert sets.period.names == ["solve", "period"]
    assert len(sets.period) == 1
    assert sets.period[0] == ("base", "p2020")


def test_read_sets_node_balance(base_solution):
    """``node_balance`` set carries the node names with name='node'."""
    sets = read_sets(
        base_solution.flex_data,
        base_solution.solution,
        solve_name=base_solution.solve_name,
    )
    assert isinstance(sets.node_balance, pd.Index)
    assert sets.node_balance.name == "node"
    assert "west" in sets.node_balance


# ---------------------------------------------------------------------------
# entity_all_capacity — the post-solve derived attribute the dispatch flagged.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coal_wind_solution(scenario_workdir):
    """Run the ``coal_wind_inertia`` fixture for a non-trivial
    capacity table.

    work_base has no processes so its entity_all_capacity is single-
    entity (just the node) and zero-valued.  For the dispatch's
    load-bearing test case (the four header-only capacity tables),
    we want a fixture with at least one unit-typed process and an
    existing capacity.
    """
    fixture = scenario_workdir("coal_wind_inertia")
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "work"
        work.mkdir()
        step = run_single_solve_from_db(
            f"sqlite:///{db}",
            scenario_name="coal_wind_inertia",
            work_folder=work,
        )
        yield step


def test_entity_all_capacity_non_empty(coal_wind_solution):
    """``entity_all_capacity`` is populated for a fixture with
    pre-existing process capacity.  Validates the post-solve
    derivation from FlexData + Solution.
    """
    par = read_parameters(
        coal_wind_solution.flex_data,
        coal_wind_solution.solution,
        solve_name=coal_wind_solution.solve_name,
    )
    eac = par.entity_all_capacity
    assert isinstance(eac, pd.DataFrame)
    assert eac.columns.name == "entity"
    assert eac.index.names == ["solve", "period"]
    # Non-empty: at least the coal_plant + wind_plant are in the columns.
    assert "coal_plant" in eac.columns
    assert "wind_plant" in eac.columns
    # Pre-existing capacity for both.
    row = eac.iloc[0]
    assert row["coal_plant"] == pytest.approx(500.0, rel=1e-6)
    assert row["wind_plant"] == pytest.approx(1000.0, rel=1e-6)


# ---------------------------------------------------------------------------
# End-to-end write_outputs check — fast path emits all four output dirs.
# ---------------------------------------------------------------------------


def test_write_outputs_fast_path_emits_all_dirs(tmp_path: Path, scenario_workdir)-> None:
    """Running ``write_outputs`` after the fast single-solve path
    populates output_csv/, output_parquet/, output_excel/ (skipped
    intentionally on csv-only run) — all from the in-memory readers,
    no CSV reads from solve_data/.

    Δ.31 acceptance bar item 3.
    """
    fixture = scenario_workdir("base")
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    from flextool.process_outputs.write_outputs import write_outputs

    work = tmp_path / "fast"
    work.mkdir()
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="base",
        work_folder=work,
    )
    assert step.solution is not None and step.solution.optimal
    assert step.flex_data is not None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_outputs(
            scenario_name="base",
            write_methods=["parquet", "csv"],
            output_location=str(work),
            subdir="base",
            raw_output_dir=str(work / "output_raw"),
            flex_data=step.flex_data,
            solution=step.solution,
            solve_name=step.solve_name,
        )

    csv_dir = work / "output_csv" / "base"
    pq_dir = work / "output_parquet" / "base"
    assert csv_dir.exists() and any(csv_dir.iterdir())
    assert pq_dir.exists() and any(pq_dir.iterdir())


def test_unit_capacity_period_table_includes_invested(tmp_path: Path, scenario_workdir)-> None:
    """Δ.31 acceptance bar item 4 — ``unit_capacity__d.csv`` for a
    multi-period investment fixture emits per-period invest values
    AND a correctly-cumulative ``total`` column.

    This exercises the post-solve derivation
    ``total = existing + cumulative_invested - cumulative_divested``
    via the FlexData ``edd_invest_set`` indirection — the legacy
    GMPL-phase 3 path produced this row.
    """
    fixture = scenario_workdir("network_coal_wind_battery_invest_cumulative")
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    from flextool.process_outputs.write_outputs import write_outputs

    work = tmp_path / "fast"
    work.mkdir()
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="network_coal_wind_battery_invest_cumulative",
        work_folder=work,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_outputs(
            scenario_name="ncwbic",
            write_methods=["csv"],
            output_location=str(work),
            subdir="ncwbic",
            raw_output_dir=str(work / "output_raw"),
            flex_data=step.flex_data,
            solution=step.solution,
            solve_name=step.solve_name,
        )

    csv = work / "output_csv" / "ncwbic" / "unit_capacity__d.csv"
    assert csv.exists()
    text = csv.read_text()
    # Reads as a header + 8 rows (coal_plant × 4 periods + wind_plant × 4).
    df = pd.read_csv(csv)
    coal = df[df["unit"] == "coal_plant"].sort_values("period")
    assert len(coal) == 4, f"expected 4 coal rows, got {len(coal)}"
    # invested should be > 0 on every period (it's an active invest entity).
    assert (coal["invested"] > 0).all(), (
        f"invested column has zero rows: {coal[['period', 'invested']]}"
    )
    # total should be monotone non-decreasing (cumulative invest).
    assert coal["total"].is_monotonic_increasing or all(
        a <= b for a, b in zip(coal["total"], coal["total"].iloc[1:])
    ), f"unit_capacity total not cumulative: {coal[['period', 'total']]}"
    # And total = existing + invested_cumulative on the first period.
    assert coal.iloc[0]["total"] == pytest.approx(
        coal.iloc[0]["existing"] + coal.iloc[0]["invested"], rel=1e-6,
    )


def test_unit_capacity_period_table_non_empty(tmp_path: Path, scenario_workdir)-> None:
    """Δ.31 acceptance bar item 4 — ``unit_capacity__d.csv`` (the
    user-facing output_csv table for unit capacity by period) emits
    non-zero rows for a fixture with pre-existing unit capacity.
    """
    fixture = scenario_workdir("coal_wind_inertia")
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    from flextool.process_outputs.write_outputs import write_outputs

    work = tmp_path / "fast"
    work.mkdir()
    step = run_single_solve_from_db(
        f"sqlite:///{db}",
        scenario_name="coal_wind_inertia",
        work_folder=work,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        write_outputs(
            scenario_name="coal_wind_inertia",
            write_methods=["csv"],
            output_location=str(work),
            subdir="coal_wind_inertia",
            raw_output_dir=str(work / "output_raw"),
            flex_data=step.flex_data,
            solution=step.solution,
            solve_name=step.solve_name,
        )

    csv = work / "output_csv" / "coal_wind_inertia" / "unit_capacity__d.csv"
    assert csv.exists(), f"unit_capacity__d.csv missing under {csv.parent}"
    text = csv.read_text()
    # Header line + at least one data row (coal_plant or wind_plant).
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) > 1, f"unit_capacity__d.csv has only header: {text!r}"
    assert "coal_plant" in text
    assert "wind_plant" in text
