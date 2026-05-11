"""Writer-port Phase 1 (L0-L2) — parity tests for native leaf-set writers.

Asserts byte-identical output between the legacy preprocessing helpers
in ``flextool.flextoolrunner.preprocessing`` and their native polars
ports in :mod:`flextool.engine_polars._writer_leaf_sets`.

Per family we run both writers on the same fixture ``input/`` (and an
``solve_data/`` seeded from the fixture's checked-in copy for the
solve_data → solve_data projections), then ``filecmp`` the outputs.

Fixtures exercised:
    * ``work_base``       — minimal smoke test
    * ``work_coal``       — has commodities, processes, periods
    * ``work_test_a_lot`` — exercises invest methods, co2 methods,
      multi-period projections, optional outputs.

Anchored on the four families (legacy line ranges in
``flextool/flextoolrunner/input_writer.py``):
    period_param_sets   (line 1886)
    invest_method_sets  (line 1887)
    co2_method_sets     (line 1888)
    simple_projections  (lines 1889-1939)
"""
from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

import pytest

from flextool.engine_polars import _writer_arc_unions as native_arc
from flextool.engine_polars import _writer_calc_params as native_calc
from flextool.engine_polars import _writer_chain_params as native_chain
from flextool.engine_polars import _writer_dispatchers as native_disp
from flextool.engine_polars import _writer_leaf_sets as native
from flextool.engine_polars import _writer_mid_sets as native_mid
from flextool.engine_polars import _writer_pdt_params as native_pdt
from flextool.engine_polars import _writer_entity_annual as native_entity_annual
from flextool.engine_polars import _writer_lp_scaling as native_lp_scaling
from flextool.engine_polars import _writer_per_solve as native_per_solve
from flextool.engine_polars import _writer_period_params as native_period
from flextool.flextoolrunner.preprocessing import (
    co2_method_sets as legacy_co2,
    dc_angle_bounds as legacy_dc,
    entity_annual_calc_params as legacy_entity_annual,
    entity_period_calc_params as legacy_entity_period,
    entity_total_caps as legacy_entity_total,
    invest_divest_sets as legacy_invest_divest,
    invest_method_sets as legacy_invest,
    invest_total_sets as legacy_invest_total,
    lp_scaling_params as legacy_lp_scaling,
    method_with_fallback_sets as legacy_method_fb,
    node_type_sets as legacy_node_type,
    nonsync_sets as legacy_nonsync,
    period_param_sets as legacy_period,
    per_solve_sets as legacy_per_solve,
    process_arc_unions as legacy_arc_unions,
    process_method_sets as legacy_process_method,
    reserve_method_partitions as legacy_reserve_part,
    simple_projections as legacy_simple,
    structural_filters as legacy_struct,
    union_sets as legacy_union,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
FIXTURES = ["work_base", "work_coal", "work_test_a_lot"]
# Extra fixture exercising DC power-flow bounds.
FIXTURES_WITH_DC = FIXTURES + ["work_dc_power_flow"]
# Extra fixture exercising explicit float param values (invest_max_total),
# stresses repr(float) precision parity in entity_total_caps.
FIXTURES_WITH_INVEST = FIXTURES + ["work_network_coal_wind_battery_invest_cumulative"]


def _seed_workdir(tmp_path: Path, fixture: str) -> tuple[Path, Path, Path, Path]:
    """Copy a fixture's ``input/`` + ``solve_data/`` into two parallel
    workdirs (legacy + native).  Returns (legacy_input, legacy_sd,
    native_input, native_sd).

    We copy ``solve_data/`` because a couple of the projections read
    from solve_data files written earlier in the legacy chain (e.g.
    ``write_period_solve`` reads ``solve_period.csv`` produced by
    ``write_simple_setof_projections``).  Seeding from the fixture's
    checked-in ``solve_data/`` provides those upstream files for
    isolated parity tests.
    """
    src_input = DATA_DIR / fixture / "input"
    src_sd = DATA_DIR / fixture / "solve_data"

    legacy_root = tmp_path / "legacy"
    native_root = tmp_path / "native"
    shutil.copytree(src_input, legacy_root / "input")
    shutil.copytree(src_sd, legacy_root / "solve_data")
    shutil.copytree(src_input, native_root / "input")
    shutil.copytree(src_sd, native_root / "solve_data")

    return (
        legacy_root / "input", legacy_root / "solve_data",
        native_root / "input", native_root / "solve_data",
    )


def _assert_files_equal(legacy_path: Path, native_path: Path) -> None:
    """Byte-identical CSV comparison.

    Both writers should emit the same row order; if a divergence is
    found we dump both files to make the failure inspectable.
    """
    if not legacy_path.exists() and not native_path.exists():
        return  # both absent — both wrote nothing, parity holds
    assert legacy_path.exists(), f"Legacy missing: {legacy_path}"
    assert native_path.exists(), f"Native missing: {native_path}"
    if not filecmp.cmp(legacy_path, native_path, shallow=False):
        legacy_text = legacy_path.read_text()
        native_text = native_path.read_text()
        raise AssertionError(
            f"CSV byte mismatch for {legacy_path.name}\n"
            f"--- legacy ---\n{legacy_text}\n"
            f"--- native ---\n{native_text}"
        )


# ---------------------------------------------------------------------------
# Family 1 — period_param_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_period_param_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_period.write_period_param_sets(lin, lsd)
    native.write_period_param_sets(nin, nsd)
    for fname in (
        "period_group.csv", "period_node.csv",
        "period_commodity.csv", "period_process.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 2 — invest_method_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_invest_method_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_invest.write_invest_method_sets(lin, lsd)
    native.write_invest_method_sets(nin, nsd)
    for fname in (
        "entityInvest.csv", "entityDivest.csv",
        "group_invest.csv", "group_divest.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 3 — co2_method_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_co2_method_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_co2.write_co2_method_sets(lin, lsd)
    native.write_co2_method_sets(nin, nsd)
    for fname in (
        "group_co2_price.csv",
        "group_co2_max_period.csv",
        "group_co2_max_total.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 4 — simple_projections
#
# These are ordering-sensitive: ``write_simple_setof_projections`` must
# run BEFORE ``write_period_solve`` (which reads ``solve_period.csv``)
# and BEFORE ``write_commodity_tier_sets`` (which reads
# ``commodity__tier_ann.csv``).  ``write_enable_optional_outputs``
# requires ``optional_yes`` and ``def_optional_yes`` to exist.
# ``write_node_state_subsets`` reads ``nodeState`` +
# ``node__storage_binding_method``.
# ``write_process_delayed`` reads ``solve_data/process_delayed__duration.csv``.
# We run the legacy chain in the exact order of ``write_input``
# (lines 1889-1939) and compare each output independently.
# ---------------------------------------------------------------------------

_SIMPLE_PROJECTION_OUTPUTS = (
    # written by the early L0 batch 1 calls
    "optional_yes.csv",
    "reserve__upDown__group.csv",
    "group_loss_share.csv",
    # L0 batch 4 / 6
    "def_optional_yes.csv",
    "process_delayed.csv",
    "process_side.csv",
    "solve_period.csv",
    "timeline.csv",
    "timeline_steps.csv",
    "commodity__tier_ann.csv",
    "period_solve.csv",
    "time.csv",
    "enable_optional_outputs.csv",
    "nodeState_rp.csv",
    "nodeStateBlock.csv",
    "commodity__tier.csv",
    "tier.csv",
)


def _run_legacy_simple_chain(input_dir: Path, sd: Path) -> None:
    """Replays the legacy ordering from ``write_input``."""
    legacy_simple.write_optional_yes(input_dir, sd)
    legacy_simple.write_reserve_upDown_group(input_dir, sd)
    legacy_simple.write_group_loss_share(input_dir, sd)
    legacy_simple.write_def_optional_yes(input_dir, sd)
    legacy_simple.write_process_delayed(input_dir, sd)
    legacy_simple.write_process_side(sd)
    legacy_simple.write_simple_setof_projections(input_dir, sd)
    legacy_simple.write_period_solve(sd)
    legacy_simple.write_time_set(input_dir, sd)
    legacy_simple.write_enable_optional_outputs(sd)
    legacy_simple.write_node_state_subsets(sd)
    legacy_simple.write_commodity_tier_sets(input_dir, sd)


def _run_native_simple_chain(input_dir: Path, sd: Path) -> None:
    """Same ordering but using native polars implementations."""
    native.write_optional_yes(input_dir, sd)
    native.write_reserve_upDown_group(input_dir, sd)
    native.write_group_loss_share(input_dir, sd)
    native.write_def_optional_yes(input_dir, sd)
    native.write_process_delayed(input_dir, sd)
    native.write_process_side(sd)
    native.write_simple_setof_projections(input_dir, sd)
    native.write_period_solve(sd)
    native.write_time_set(input_dir, sd)
    native.write_enable_optional_outputs(sd)
    native.write_node_state_subsets(sd)
    native.write_commodity_tier_sets(input_dir, sd)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_simple_projections_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    _run_legacy_simple_chain(lin, lsd)
    _run_native_simple_chain(nin, nsd)
    for fname in _SIMPLE_PROJECTION_OUTPUTS:
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Native-derive return type smoke — make sure ``derive_*`` returns a
# polars DataFrame (the in-memory contract).
# ---------------------------------------------------------------------------

def test_derive_returns_dataframe(tmp_path: Path) -> None:
    import polars as pl
    lin, lsd, _, _ = _seed_workdir(tmp_path, "work_base")
    # A handful of representative derive_* signatures.
    assert isinstance(native.derive_period_param_set(lin, "pd_node.csv"), pl.DataFrame)
    assert isinstance(native.derive_entity_invest(lin), pl.DataFrame)
    assert isinstance(native.derive_group_co2(lin, "price"), pl.DataFrame)
    assert isinstance(native.derive_process_side(), pl.DataFrame)
    assert isinstance(native.derive_optional_yes(lin), pl.DataFrame)
    # L3-L6 native-derive return type smoke.
    assert isinstance(native_mid.derive_node_effective_type(lin), pl.DataFrame)
    assert isinstance(native_mid.derive_group_entity(lin), pl.DataFrame)
    lower, upper = native_mid.derive_dc_angle_bounds(lin)
    assert isinstance(lower, pl.DataFrame)
    assert isinstance(upper, pl.DataFrame)
    assert isinstance(native_mid.derive_reserve_universe(lin), pl.DataFrame)
    assert isinstance(native_mid.derive_entity_lifetime_method(lin), pl.DataFrame)
    assert isinstance(native_mid.derive_connection_param(lin), pl.DataFrame)
    # L7-L9 native-derive return type smoke.
    assert isinstance(native_calc.derive_process_online_linear(lin), pl.DataFrame)
    assert isinstance(native_calc.derive_process_online_integer(lin), pl.DataFrame)
    assert isinstance(native_calc.derive_process_method_indirect(lin), pl.DataFrame)
    assert isinstance(native_calc.derive_process_VRE(lin), pl.DataFrame)
    assert isinstance(
        native_calc.derive_entity_total_cap(
            [], frozenset(), frozenset(), {}, {}, "invest_max_total",
        ),
        pl.DataFrame,
    )


# ===========================================================================
# Phase 1 (L3-L6) — mid-level set / param families
# ===========================================================================

# ---------------------------------------------------------------------------
# Family 5 — node_type_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_node_type_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_node_type.write_node_type_sets(lin, lsd)
    native_mid.write_node_type_sets(nin, nsd)
    for fname in (
        "nodeCommodity.csv", "nodeBalance.csv",
        "nodeState.csv", "nodeBalancePeriod.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 6 — union_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_union_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_union.write_group_entity(lin, lsd)
    legacy_union.write_process_delayed__duration(lin, lsd)
    native_mid.write_group_entity(nin, nsd)
    native_mid.write_process_delayed__duration(nin, nsd)
    for fname in ("group_entity.csv", "process_delayed__duration.csv"):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 7 — dc_angle_bounds (work_dc_power_flow exercises the non-empty path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES_WITH_DC)
def test_dc_angle_bounds_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_dc.write_dc_angle_bounds(lin, lsd)
    native_mid.write_dc_angle_bounds(nin, nsd)
    for fname in ("p_angle_lower.csv", "p_angle_upper.csv"):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 8 — reserve_method_partitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_reserve_method_partitions_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_reserve_part.write_reserve_partitions(lin, lsd)
    native_mid.write_reserve_partitions(nin, nsd)
    for fname in (
        "reserve.csv",
        "reserve__upDown__group__method_timeseries.csv",
        "reserve__upDown__group__method_dynamic.csv",
        "reserve__upDown__group__method_n_1.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 9 — nonsync_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_nonsync_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_nonsync.write_process_group_inside_group_nonsync(lin, lsd)
    legacy_nonsync.write_process__sink_nonSync(lin, lsd)
    native_mid.write_process_group_inside_group_nonsync(nin, nsd)
    native_mid.write_process__sink_nonSync(nin, nsd)
    for fname in (
        "process__sink_nonSync.csv",
        "process__group_inside_group_nonSync.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 10 — method_with_fallback_sets
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_method_with_fallback_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_method_fb.write_entity_lifetime_method(lin, lsd)
    legacy_method_fb.write_process_ct_method(lin, lsd)
    legacy_method_fb.write_process_startup_method(lin, lsd)
    legacy_method_fb.write_node_inflow_method(lin, lsd)
    legacy_method_fb.write_node_storage_binding_method(lin, lsd)
    native_mid.write_entity_lifetime_method(nin, nsd)
    native_mid.write_process_ct_method(nin, nsd)
    native_mid.write_process_startup_method(nin, nsd)
    native_mid.write_node_inflow_method(nin, nsd)
    native_mid.write_node_storage_binding_method(nin, nsd)
    for fname in (
        "entity__lifetime_method.csv",
        "process__ct_method.csv",
        "process__startup_method.csv",
        "node__inflow_method.csv",
        "node__storage_binding_method.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 11 — invest_total_sets
#
# Depends on entityInvest/entityDivest/group_invest/group_divest being
# written first (Phase 1 L0-L2 — invest_method_sets) and on
# ``commodity_with_ladder_cumulative.csv`` existing in solve_data (it's
# checked into the fixture directly).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_invest_total_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    # Re-seed entityInvest / entityDivest / group_invest / group_divest
    # using the legacy invest_method_sets writer so both chains have a
    # consistent universe.
    from flextool.flextoolrunner.preprocessing import invest_method_sets as _legacy_im
    _legacy_im.write_invest_method_sets(lin, lsd)
    _legacy_im.write_invest_method_sets(nin, nsd)
    legacy_invest_total.write_invest_total_sets(lin, lsd)
    legacy_invest_total.write_ci_ladder_cumulative(lin, lsd)
    native_mid.write_invest_total_sets(nin, nsd)
    native_mid.write_ci_ladder_cumulative(nin, nsd)
    for fname in (
        "e_invest_total.csv", "e_divest_total.csv",
        "g_invest_total.csv", "g_divest_total.csv",
        "g_invest_cumulative.csv",
        "ci_ladder_cumulative.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 12 — structural_filters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES)
def test_structural_filters_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_struct.write_connection_param(lin, lsd)
    legacy_struct.write_nodegroup_dispatch_node(lin, lsd)
    legacy_struct.write_commodity_node_co2(lin, lsd)
    legacy_struct.write_process__commodity__node(lin, lsd)
    legacy_struct.write_process_coeff_zero_sets(lin, lsd)
    native_mid.write_connection_param(nin, nsd)
    native_mid.write_nodegroup_dispatch_node(nin, nsd)
    native_mid.write_commodity_node_co2(nin, nsd)
    native_mid.write_process__commodity__node(nin, nsd)
    native_mid.write_process_coeff_zero_sets(nin, nsd)
    for fname in (
        "connection__param.csv",
        "nodeGroupDispatch_node.csv",
        "commodity_node_co2.csv",
        "process__commodity__node.csv",
        "process_source_coeff_zero.csv",
        "process_sink_coeff_zero.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ===========================================================================
# Phase 1 (L7-L9) — calculated-param + process-method families
# ===========================================================================

# ---------------------------------------------------------------------------
# Family 13 — entity_total_caps  (calculated-param, repr(float) precision)
#
# Depends on entityInvest / entityDivest being present in solve_data — the
# fixture's checked-in solve_data has them.  We also stress an extra
# fixture with explicit non-trivial float values (invest_max_total =
# 800.0, etc.) to gate the repr(float)-parity strategy.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES_WITH_INVEST)
def test_entity_total_caps_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_total.write_entity_total_caps(lin, lsd)
    native_calc.write_entity_total_caps(nin, nsd)
    for fname in (
        "e_invest_max_total.csv",
        "e_divest_max_total.csv",
        "e_invest_min_total.csv",
        "e_divest_min_total.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Family 14 — process_method_sets
#
# Four legacy writers; each emits multiple solve_data CSVs.  We run all of
# them on each fixture and compare every output independently.
# ---------------------------------------------------------------------------

_PROCESS_METHOD_OUTPUTS = (
    # write_process_method_projections
    "process_online_linear.csv",
    "process_online_integer.csv",
    "process__method_indirect.csv",
    # write_process_VRE
    "process_VRE.csv",
    # write_process_arc_method_joins
    "process_sink_toProcess.csv",
    "process_process_toSource.csv",
    "process_source_toSink.csv",
    "process_source_toProcess_direct.csv",
    "process_process_toSink_direct.csv",
    "process_sink_toProcess_direct.csv",
    "process_sink_toSource.csv",
    "process_process_toSink_noConversion.csv",
    "process_source_toProcess_noConversion.csv",
    "process_process_toSource_direct.csv",
    # write_process_profile_method_joins
    "process__profileProcess__toSink__profile__profile_method.csv",
    "process__source__toProfileProcess__profile__profile_method.csv",
)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_INVEST)
def test_process_method_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_process_method.write_process_method_projections(lin, lsd)
    legacy_process_method.write_process_VRE(lin, lsd)
    legacy_process_method.write_process_arc_method_joins(lin, lsd)
    legacy_process_method.write_process_profile_method_joins(lin, lsd)
    native_calc.write_process_method_projections(nin, nsd)
    native_calc.write_process_VRE(nin, nsd)
    native_calc.write_process_arc_method_joins(nin, nsd)
    native_calc.write_process_profile_method_joins(nin, nsd)
    for fname in _PROCESS_METHOD_OUTPUTS:
        _assert_files_equal(lsd / fname, nsd / fname)


# ===========================================================================
# Phase 1 follow-up — process_arc_unions leaf-like writers
#                   + entity_period_calc_params subset
#
# Each test mirrors the legacy↔native pattern.  Where the legacy writer
# emits multiple CSVs (e.g. the four ``sinkIsNode*`` variants, or the
# two ``delayed/undelayed`` partition halves) we compare every emitted
# file independently.  Extra fixture ``work_coal_ramp_limit`` exercises
# ``write_process_source_sink_ramp_method`` against a non-empty
# ``process__node__ramp_method.csv`` input.
# ===========================================================================

FIXTURES_WITH_RAMP = FIXTURES + ["work_coal_ramp_limit"]


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_param_t_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_param_t(lin, lsd)
    native_arc.write_process_source_sink_param_t(nin, nsd)
    _assert_files_equal(
        lsd / "process_source_sink_param_t.csv",
        nsd / "process_source_sink_param_t.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_node_time_param_in_use_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_node_time_param_in_use(lin, lsd)
    native_arc.write_node_time_param_in_use(nin, nsd)
    _assert_files_equal(
        lsd / "node__TimeParam_in_use.csv",
        nsd / "node__TimeParam_in_use.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_delayed_partition_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_delayed_partition(lin, lsd)
    native_arc.write_process_source_delayed_partition(nin, nsd)
    for fname in ("process_source_delayed.csv", "process_source_undelayed.csv"):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_param_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_param(lin, lsd)
    native_arc.write_process_source_sink_param(nin, nsd)
    _assert_files_equal(
        lsd / "process__source__sink__param.csv",
        nsd / "process__source__sink__param.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_profile_method_connection_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_profile_method_connection(lin, lsd)
    native_arc.write_process_source_sink_profile_method_connection(nin, nsd)
    _assert_files_equal(
        lsd / "process__source__sink__profile__profile_method_connection.csv",
        nsd / "process__source__sink__profile__profile_method_connection.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_method_sources_sinks_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_method_sources_sinks(lin, lsd)
    native_arc.write_process_method_sources_sinks(nin, nsd)
    _assert_files_equal(
        lsd / "process_method_sources_sinks.csv",
        nsd / "process_method_sources_sinks.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_ed_history_realized_first_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_ed_history_realized_first(lin, lsd)
    native_arc.write_ed_history_realized_first(nin, nsd)
    _assert_files_equal(
        lsd / "ed_history_realized_first.csv",
        nsd / "ed_history_realized_first.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_is_node_sink_1way_no_sink_or_more_than_1_source_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(lin, lsd)
    native_arc.write_process_source_is_node_sink_1way_no_sink_or_more_than_1_source(nin, nsd)
    _assert_files_equal(
        lsd / "process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.csv",
        nsd / "process__sourceIsNode__sink_1way_noSinkOrMoreThan1Source.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES_WITH_RAMP)
def test_process_source_sink_ramp_method_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_ramp_method(lin, lsd)
    native_arc.write_process_source_sink_ramp_method(nin, nsd)
    _assert_files_equal(
        lsd / "process__source__sink__ramp_method.csv",
        nsd / "process__source__sink__ramp_method.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_coeff_zero_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_coeff_zero(lin, lsd)
    native_arc.write_process_source_sink_coeff_zero(nin, nsd)
    _assert_files_equal(
        lsd / "process_source_sink_coeff_zero.csv",
        nsd / "process_source_sink_coeff_zero.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_is_node_family_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_is_node_family(lin, lsd)
    native_arc.write_process_source_sink_is_node_family(nin, nsd)
    for fname in (
        "process__source__sinkIsNode.csv",
        "process__source__sinkIsNode_2way1var.csv",
        "process__source__sinkIsNode_not2way1var.csv",
        "process__source__sinkIsNode_2way2var.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_delayed_partition_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_delayed_partition(lin, lsd)
    native_arc.write_process_source_sink_delayed_partition(nin, nsd)
    for fname in (
        "process_source_sink_delayed.csv",
        "process_source_sink_undelayed.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_INVEST)
def test_p_process_source_sink_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pProcess_source_sink(lin, lsd)
    native_arc.write_pProcess_source_sink(nin, nsd)
    _assert_files_equal(
        lsd / "pProcess_source_sink.csv",
        nsd / "pProcess_source_sink.csv",
    )


# ---------------------------------------------------------------------------
# pdtProcess / pdtNode / pdtProcess_source / pdtProcess_sink  (PdtLookup port)
# ---------------------------------------------------------------------------

# Extra fixture exercising the stochastic-branch and parent-period-branch
# fold-in branches of PdtLookup (Branches 1-2).  The non-stochastic
# fixtures only exercise branches 3+.
FIXTURES_WITH_STOCH = FIXTURES + ["work_2day_stochastic_dispatch_full_storage"]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_STOCH)
def test_pdtProcess_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProcess(lin, lsd)
    native_pdt.write_pdtProcess(nin, nsd)
    _assert_files_equal(lsd / "pdtProcess.csv", nsd / "pdtProcess.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_STOCH)
def test_pdtNode_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtNode(lin, lsd)
    native_pdt.write_pdtNode(nin, nsd)
    _assert_files_equal(lsd / "pdtNode.csv", nsd / "pdtNode.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_STOCH)
def test_pdtProcess_source_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProcess_source(lin, lsd)
    native_pdt.write_pdtProcess_source(nin, nsd)
    _assert_files_equal(
        lsd / "pdtProcess_source.csv", nsd / "pdtProcess_source.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES_WITH_STOCH)
def test_pdtProcess_sink_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProcess_sink(lin, lsd)
    native_pdt.write_pdtProcess_sink(nin, nsd)
    _assert_files_equal(
        lsd / "pdtProcess_sink.csv", nsd / "pdtProcess_sink.csv",
    )


# ---------------------------------------------------------------------------
# process_source_sink_ramp_family (5 CSVs) + ramp_unions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", FIXTURES_WITH_RAMP)
def test_process_source_sink_ramp_family_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_ramp_family(lin, lsd)
    native_arc.write_process_source_sink_ramp_family(nin, nsd)
    for fname in (
        "process_source_sink_ramp_limit_source_up.csv",
        "process_source_sink_ramp_limit_sink_up.csv",
        "process_source_sink_ramp_limit_source_down.csv",
        "process_source_sink_ramp_limit_sink_down.csv",
        "process_source_sink_ramp_cost.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_RAMP)
def test_process_source_sink_ramp_unions_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    # The unions writer reads the 5 ramp CSVs produced by the family
    # writer; seed both legacy and native trees with those files first.
    legacy_arc_unions.write_process_source_sink_ramp_family(lin, lsd)
    native_arc.write_process_source_sink_ramp_family(nin, nsd)
    legacy_arc_unions.write_process_source_sink_ramp_unions(lin, lsd)
    native_arc.write_process_source_sink_ramp_unions(nin, nsd)
    _assert_files_equal(
        lsd / "process_source_sink_ramp.csv",
        nsd / "process_source_sink_ramp.csv",
    )


# ---------------------------------------------------------------------------
# group_commodity_node_period_co2_total
# ---------------------------------------------------------------------------

FIXTURES_WITH_CO2 = FIXTURES + ["work_coal_co2_limit"]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CO2)
def test_group_commodity_node_period_co2_total_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_group_commodity_node_period_co2_total(lin, lsd)
    native_arc.write_group_commodity_node_period_co2_total(nin, nsd)
    _assert_files_equal(
        lsd / "group_commodity_node_period_co2_total.csv",
        nsd / "group_commodity_node_period_co2_total.csv",
    )


# ---------------------------------------------------------------------------
# Phase 1 follow-up 3 — heavy per-(d, t) emitters
# ---------------------------------------------------------------------------

# pdtNodeInflow: stress branch 3 (deterministic additive sum) via the
# scale_to_peak_flow fixture which exercises peak/annual flow scaling.
# work_2day_stochastic_dispatch_full_storage carries empty pbt_node_inflow
# (Branches 1+2 are Gap E — no fixture exercises them).
FIXTURES_WITH_INFLOW = FIXTURES + ["work_scale_to_peak_flow"]

# pdtProfile: stress branch 1 (stochastic profile fold-in) via the 2-day
# stochastic fixture which carries non-empty pbt_profile + stochastic
# group__node membership tying through node__profile__profile_method.
FIXTURES_WITH_PROFILE = FIXTURES + [
    "work_2day_stochastic_dispatch_full_storage",
    "work_scale_to_peak_flow",
]

# pdtConversion_rate / section / slope: stress the process_minload branch
# via work_coal_min_load and work_coal_chp (cogeneration efficiency).
FIXTURES_WITH_CONVERSION = FIXTURES + [
    "work_coal_min_load",
    "work_coal_chp",
]

# pdtProcess_source_sink: stress branches 5-11 via fixtures that carry
# pt_process_sink / pt_process_source / p_process_source rows.  The
# stochastic fixture exercises branches 1-4 indirectly when present.
FIXTURES_WITH_PSS = FIXTURES + [
    "work_coal_chp",
    "work_2day_stochastic_dispatch_full_storage",
]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_INFLOW)
def test_pdtNodeInflow_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtNodeInflow(lin, lsd)
    native_period.write_pdtNodeInflow(nin, nsd)
    _assert_files_equal(lsd / "pdtNodeInflow.csv", nsd / "pdtNodeInflow.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PROFILE)
def test_pdtProfile_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProfile(lin, lsd)
    native_period.write_pdtProfile(nin, nsd)
    _assert_files_equal(lsd / "pdtProfile.csv", nsd / "pdtProfile.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CONVERSION)
def test_pdtConversion_rate_section_slope_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtConversion_rate_section_slope(lin, lsd)
    native_period.write_pdtConversion_rate_section_slope(nin, nsd)
    for fname in (
        "pdtConversion_rate.csv",
        "pdtProcess_section.csv",
        "pdtProcess_slope.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PSS)
def test_pdtProcess_source_sink_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProcess_source_sink(lin, lsd)
    native_period.write_pdtProcess_source_sink(nin, nsd)
    _assert_files_equal(
        lsd / "pdtProcess_source_sink.csv",
        nsd / "pdtProcess_source_sink.csv",
    )


# ---------------------------------------------------------------------------
# Phase 1 follow-up 4 — pdGroup / pdtGroup / pdCommodity / pdtCommodity,
# positive/negative inflow, param-in-use family, and the dispatch-inside set.
# ---------------------------------------------------------------------------

# Group / commodity fallback writers: stress co2_price / co2_max_total /
# inertia / capacity_margin penalty defaults via the dedicated fixtures.
FIXTURES_WITH_GROUP = FIXTURES + [
    "work_coal_co2_limit",
    "work_coal_co2_price",
    "work_capacity_margin",
    "work_coal_wind_inertia",
]
FIXTURES_WITH_COMMODITY = FIXTURES + [
    "work_coal_co2_price",
    "work_commodity_ladder_annual",
    "work_commodity_ladder_cumulative",
]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_GROUP)
def test_pdGroup_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdGroup(lin, lsd)
    native_period.write_pdGroup(nin, nsd)
    _assert_files_equal(lsd / "pdGroup.csv", nsd / "pdGroup.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_GROUP)
def test_pdtGroup_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtGroup(lin, lsd)
    native_period.write_pdtGroup(nin, nsd)
    _assert_files_equal(lsd / "pdtGroup.csv", nsd / "pdtGroup.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_COMMODITY)
def test_pdCommodity_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdCommodity(lin, lsd)
    native_period.write_pdCommodity(nin, nsd)
    _assert_files_equal(lsd / "pdCommodity.csv", nsd / "pdCommodity.csv")


@pytest.mark.parametrize("fixture", FIXTURES_WITH_COMMODITY)
def test_pdtCommodity_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtCommodity(lin, lsd)
    native_period.write_pdtCommodity(nin, nsd)
    _assert_files_equal(lsd / "pdtCommodity.csv", nsd / "pdtCommodity.csv")


# p_positive/negative_inflow consumes pdtNodeInflow.csv (already native);
# we seed both legacy and native trees with the legacy-emitted pdtNodeInflow
# to keep this test focused on the positive/negative split.
@pytest.mark.parametrize("fixture", FIXTURES_WITH_INFLOW)
def test_p_positive_negative_inflow_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtNodeInflow(lin, lsd)
    legacy_entity_period.write_pdtNodeInflow(nin, nsd)
    legacy_entity_period.write_p_positive_negative_inflow(lin, lsd)
    native_period.write_p_positive_negative_inflow(nin, nsd)
    for fname in ("p_positive_inflow.csv", "p_negative_inflow.csv"):
        _assert_files_equal(lsd / fname, nsd / fname)


# param_in_use_sets emits 7 CSVs; FIXTURES_WITH_INVEST stresses the
# invest-gated branches (NODE_PERIOD_PARAM_INVEST / PROCESS_PERIOD_PARAM_INVEST).
@pytest.mark.parametrize("fixture", FIXTURES_WITH_INVEST)
def test_param_in_use_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_param_in_use_sets(lin, lsd)
    native_arc.write_param_in_use_sets(nin, nsd)
    for fname in (
        "node__PeriodParam_in_use.csv",
        "process__PeriodParam_in_use.csv",
        "process_TimeParam_in_use.csv",
        "process_source_sourceSinkTimeParam_in_use.csv",
        "process_sink_sourceSinkTimeParam_in_use.csv",
        "process_source_sourceSinkPeriodParam_in_use.csv",
        "process_sink_sourceSinkPeriodParam_in_use.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# nodeGroupDispatch__process_fully_inside: stress fixtures with multi-node
# dispatch groups that contain non-trivial processes.
@pytest.mark.parametrize("fixture", FIXTURES)
def test_node_group_dispatch_process_fully_inside_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_node_group_dispatch_process_fully_inside(lin, lsd)
    native_arc.write_node_group_dispatch_process_fully_inside(nin, nsd)
    _assert_files_equal(
        lsd / "nodeGroupDispatch__process_fully_inside.csv",
        nsd / "nodeGroupDispatch__process_fully_inside.csv",
    )


# ---------------------------------------------------------------------------
# Phase 1 follow-up 5 — small_set_derivations + small arc-union writers
# + entity_period_calc_params varCost / cap_reduction / ed_period_params.
# ---------------------------------------------------------------------------

# Fixtures specialised to exercise each follow-up 5 writer's branches.
FIXTURES_WITH_DELAY = FIXTURES + ["work_delay_source_coef", "work_water_pump_delayed"]
FIXTURES_WITH_ONLINE = FIXTURES + [
    "work_coal_min_load",
    "work_coal_min_load_MIP_wind",
    "work_coal_wind_min_uptime",
]


# write_small_set_derivations emits 6 CSVs in one call.  It depends on
# pdtNode + pdProcess + a small army of upstream solve_data files — all
# pre-seeded in the fixture's checked-in solve_data tree.
@pytest.mark.parametrize("fixture", FIXTURES + ["work_coal_min_load"])
def test_small_set_derivations_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_small_set_derivations(lin, lsd)
    native_arc.write_small_set_derivations(nin, nsd)
    for fname in (
        "ed_history_realized.csv",
        "process__source__sink__profile__profile_method.csv",
        "process_sinkIsNode_2way1var.csv",
        "nodeSelfDischarge.csv",
        "pdt_online_linear.csv",
        "pdt_online_integer.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# write_process_source_sink_param_with_time — extension of the param_t
# writer; stress with the standard FIXTURES set (work_test_a_lot exercises
# multiple params + process_connection).
@pytest.mark.parametrize("fixture", FIXTURES)
def test_process_source_sink_param_with_time_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_source_sink_param_with_time(lin, lsd)
    native_arc.write_process_source_sink_param_with_time(nin, nsd)
    _assert_files_equal(
        lsd / "process__source__sink__param_t.csv",
        nsd / "process__source__sink__param_t.csv",
    )


# write_gdt_instant_flow_sets — needs non-empty pdtGroup with
# max/min_instant_flow rows; the standard fixtures cover the 0-row case
# byte-identically and that's also a parity assertion.
@pytest.mark.parametrize("fixture", FIXTURES)
def test_gdt_instant_flow_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_gdt_instant_flow_sets(lin, lsd)
    native_arc.write_gdt_instant_flow_sets(nin, nsd)
    for fname in ("gdt_maxInstantFlow.csv", "gdt_minInstantFlow.csv"):
        _assert_files_equal(lsd / fname, nsd / fname)


# write_p_process_delay_weight — exercise via the delay fixtures.
@pytest.mark.parametrize("fixture", FIXTURES_WITH_DELAY)
def test_p_process_delay_weight_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_p_process_delay_weight(lin, lsd)
    native_arc.write_p_process_delay_weight(nin, nsd)
    _assert_files_equal(
        lsd / "p_process_delay_weight.csv",
        nsd / "p_process_delay_weight.csv",
    )


# write_gcndt_co2_price + write_group_commodity_node_period_co2_period —
# exercise via the CO2-price fixture; the standard fixtures cover the
# 0-row path.
FIXTURES_WITH_CO2_PRICE = FIXTURES + ["work_coal_co2_price", "work_coal_co2_limit"]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CO2_PRICE)
def test_gcndt_co2_price_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_gcndt_co2_price(lin, lsd)
    native_arc.write_gcndt_co2_price(nin, nsd)
    _assert_files_equal(
        lsd / "gcndt_co2_price.csv", nsd / "gcndt_co2_price.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CO2_PRICE)
def test_group_commodity_node_period_co2_period_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_group_commodity_node_period_co2_period(lin, lsd)
    native_arc.write_group_commodity_node_period_co2_period(nin, nsd)
    _assert_files_equal(
        lsd / "group_commodity_node_period_co2_period.csv",
        nsd / "group_commodity_node_period_co2_period.csv",
    )


# write_peedt — cross-product of process_source_sink × dt; standard
# fixtures cover small + medium row counts.
@pytest.mark.parametrize("fixture", FIXTURES)
def test_peedt_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_peedt(lin, lsd)
    native_arc.write_peedt(nin, nsd)
    _assert_files_equal(lsd / "peedt.csv", nsd / "peedt.csv")


# write_pdtProcess__source__sink__dt_varCost_pair — emits 2 CSVs.  Stress
# via the standard FIXTURES (work_test_a_lot exercises varCost path).
@pytest.mark.parametrize("fixture", FIXTURES)
def test_pdtProcess__source__sink__dt_varCost_pair_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProcess__source__sink__dt_varCost_pair(lin, lsd)
    native_period.write_pdtProcess__source__sink__dt_varCost_pair(nin, nsd)
    for fname in (
        "pdtProcess__source__sink__dt_varCost.csv",
        "pdtProcess__source__sink__dt_varCost_alwaysProcess.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# write_pssdt_varCost_filters — emits 4 CSVs; depends on the
# varCost_pair writer having run first.  We invoke both legacy and
# native writers in sequence so the input file is present in each
# workdir.
@pytest.mark.parametrize("fixture", FIXTURES)
def test_pssdt_varCost_filters_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_pdtProcess__source__sink__dt_varCost_pair(lin, lsd)
    native_period.write_pdtProcess__source__sink__dt_varCost_pair(nin, nsd)
    legacy_entity_period.write_pssdt_varCost_filters(lin, lsd)
    native_period.write_pssdt_varCost_filters(nin, nsd)
    for fname in (
        "pssdt_varCost_noEff.csv",
        "pssdt_varCost_eff_unit_source.csv",
        "pssdt_varCost_eff_unit_sink.csv",
        "pssdt_varCost_eff_connection.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# write_cap_reduction_params — emits 4 CSVs; non-zero rows require
# online + ramp_speed > 0 (exercised by min_uptime / min_load fixtures).
@pytest.mark.parametrize("fixture", FIXTURES_WITH_ONLINE)
def test_cap_reduction_params_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_cap_reduction_params(lin, lsd)
    native_period.write_cap_reduction_params(nin, nsd)
    for fname in (
        "p_startup_cap_reduction_sink.csv",
        "p_shutdown_cap_reduction_sink.csv",
        "p_startup_cap_reduction_source.csv",
        "p_shutdown_cap_reduction_source.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# write_ed_period_params — emits 6 CSVs.  Exercise via the invest
# fixtures (FIXTURES_WITH_INVEST already covers the invest path).
@pytest.mark.parametrize("fixture", FIXTURES_WITH_INVEST)
def test_ed_period_params_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_ed_period_params(lin, lsd)
    native_period.write_ed_period_params(nin, nsd)
    for fname in (
        "ed_invest_max_period.csv",
        "ed_invest_min_period.csv",
        "ed_divest_max_period.csv",
        "ed_divest_min_period.csv",
        "ed_cumulative_max_capacity.csv",
        "ed_cumulative_min_capacity.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Phase 1 follow-up 6 — flow-bound + state-slack + storage reference price
# + 12-CSV nodeGroupDispatch dispatch set family.
# ---------------------------------------------------------------------------

# Fixtures exercising flow_min/flow_max paths beyond the trivial 0-row case.
# work_network_coal_wind_battery_invest_cumulative + work_5weeks_battery_intraperiod_blocks
# exercise indirect+min_load+sink_coef branches in flow_max via peedt rows.
FIXTURES_WITH_FLOW_BOUNDS = FIXTURES + [
    "work_network_coal_wind_battery_invest_cumulative",
    "work_5weeks_battery_intraperiod_blocks",
]

# Fixtures with nodeGroupDispatch + flowAggregator (work_test_a_lot stresses
# the multi-group case; the invest fixture has fully_inside rows).
FIXTURES_WITH_NGD = FIXTURES + [
    "work_network_coal_wind_battery_invest_cumulative",
]

# Fixtures exercising storage_state_reference_price (needs nodeState).
FIXTURES_WITH_STORAGE = FIXTURES + [
    "work_5weeks_battery_intraperiod_blocks",
    "work_network_coal_wind_battery_invest_cumulative",
    "work_2day_stochastic_dispatch_full_storage",
]


# write_p_flow_min — sparse for all fixtures with empty sinkIsNode_2way1var,
# but the writer's shape (header-only emit) is also a parity assertion.
@pytest.mark.parametrize("fixture", FIXTURES_WITH_FLOW_BOUNDS)
def test_p_flow_min_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_p_flow_min(lin, lsd)
    native_arc.write_p_flow_min(nin, nsd)
    _assert_files_equal(lsd / "p_flow_min.csv", nsd / "p_flow_min.csv")


# write_p_flow_max — emits one value per peedt row.  Stresses repr(float)
# precision parity across two-branch value formula (coeff_zero vs indirect +
# slope/section).
@pytest.mark.parametrize("fixture", FIXTURES_WITH_FLOW_BOUNDS)
def test_p_flow_max_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_p_flow_max(lin, lsd)
    native_arc.write_p_flow_max(nin, nsd)
    _assert_files_equal(lsd / "p_flow_max.csv", nsd / "p_flow_max.csv")


# write_p_state_slack_share — empty for all fixtures (group_loss_share is
# empty), but the writer's contract is still asserted byte-for-byte.
@pytest.mark.parametrize("fixture", FIXTURES)
def test_p_state_slack_share_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_p_state_slack_share(lin, lsd)
    native_arc.write_p_state_slack_share(nin, nsd)
    _assert_files_equal(
        lsd / "p_state_slack_share.csv", nsd / "p_state_slack_share.csv"
    )


# write_p_storage_state_reference_price — nodes_state × period_in_use rows.
# The storage fixtures stress non-empty nodeState; baselines exercise the
# 0-row path.
@pytest.mark.parametrize("fixture", FIXTURES_WITH_STORAGE)
def test_p_storage_state_reference_price_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_p_storage_state_reference_price(lin, lsd)
    native_arc.write_p_storage_state_reference_price(nin, nsd)
    _assert_files_equal(
        lsd / "p_storage_state_reference_price.csv",
        nsd / "p_storage_state_reference_price.csv",
    )


# write_node_group_dispatch_sets — emits 12 CSVs.  work_test_a_lot
# is the strongest stress (multiple groups + 5 fully_inside rows).
@pytest.mark.parametrize("fixture", FIXTURES_WITH_NGD)
def test_node_group_dispatch_sets_parity(tmp_path: Path, fixture: str) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_node_group_dispatch_sets(lin, lsd)
    native_arc.write_node_group_dispatch_sets(nin, nsd)
    for fname in (
        "nodeGroupDispatch__process__unit__to_node_Not_in_aggregate.csv",
        "nodeGroupDispatch__process__node__to_unit_Not_in_aggregate.csv",
        "nodeGroupDispatch__group_aggregate__process__unit__to_node.csv",
        "nodeGroupDispatch__group_aggregate__process__node__to_unit.csv",
        "nodeGroupDispatch__process__node__to_connection_Not_in_aggregate.csv",
        "nodeGroupDispatch__process__connection__to_node_Not_in_aggregate.csv",
        "nodeGroupDispatch__connection_Not_in_aggregate.csv",
        "nodeGroupDispatch__group_aggregate__process__connection__to_node.csv",
        "nodeGroupDispatch__group_aggregate__process__node__to_connection.csv",
        "nodeGroupDispatch__group_aggregate_Connection.csv",
        "nodeGroupDispatch__group_aggregate_Unit_to_group.csv",
        "nodeGroupDispatch__group_aggregate_Group_to_unit.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Phase 1 follow-up 7 — param_t projections + time-param joins (8 CSVs).
# ---------------------------------------------------------------------------

# work_test_a_lot is the strongest stress (145-row pt_process.csv + 40-row
# p_process.csv + 5-row process_connection.csv).  The 2-day stochastic
# fixture exercises a separate code path (pt_process_source / sink can
# differ from work_test_a_lot's enum).
FIXTURES_WITH_PARAM_T = FIXTURES + [
    "work_2day_stochastic_dispatch_full_storage",
]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PARAM_T)
def test_param_t_projections_and_time_params_parity(
    tmp_path: Path, fixture: str,
) -> None:
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_param_t_projections_and_time_params(lin, lsd)
    native_arc.write_param_t_projections_and_time_params(nin, nsd)
    for fname in (
        # Projections (drop time)
        "process__param_t.csv",
        "connection__param__time.csv",
        "connection__param_t.csv",
        "process__source__param_t.csv",
        "process__sink__param_t.csv",
        # Static-∪-temporal joins
        "process__source__timeParam.csv",
        "process__sink__timeParam.csv",
        "process__timeParam.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Phase 1 follow-up 8 — chain-cluster entity-period params (4 writers,
# 11 output CSVs).  Cumulative-invest fixture is included to exercise
# multi-period investment semantics; the lifetime-renew fixture exercises
# ed_divest_period rows that aren't trivially zero.
# ---------------------------------------------------------------------------

# Fixtures with non-trivial invest / divest / lifetime data.  The
# cumulative-invest fixture is the explicit must-include from the brief.
FIXTURES_WITH_CHAIN = FIXTURES + [
    "work_network_coal_wind_battery_invest_cumulative",
    "work_wind_battery_invest_lifetime_renew_4solve",
]


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CHAIN)
def test_p_entity_pre_existing_parity(tmp_path: Path, fixture: str) -> None:
    """12-branch lifetime-method × kind × virtual_unitsize gate."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_p_entity_pre_existing(lin, lsd)
    native_chain.write_p_entity_pre_existing(nin, nsd)
    _assert_files_equal(
        lsd / "p_entity_pre_existing.csv",
        nsd / "p_entity_pre_existing.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CHAIN)
def test_p_entity_divest_cumulative_max_parity(
    tmp_path: Path, fixture: str,
) -> None:
    """3-branch cumulative divest ceiling per (entity, period)."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_p_entity_divest_cumulative_max(lin, lsd)
    native_chain.write_p_entity_divest_cumulative_max(nin, nsd)
    _assert_files_equal(
        lsd / "p_entity_divest_cumulative_max.csv",
        nsd / "p_entity_divest_cumulative_max.csv",
    )


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CHAIN)
def test_p_entity_existing_chain_parity(tmp_path: Path, fixture: str) -> None:
    """5 cascading existing-capacity params (later_solves, all_existing,
    existing_count, existing_integer_count, previously_invested_capacity).

    Uses the file-based handoff path (prior_handoff=None) — that's what
    fixtures carry on disk via ``p_entity_period_existing_capacity.csv``
    and ``p_entity_divested.csv`` checked-in copies.  The cumulative-invest
    fixture exercises non-trivial edd_history.
    """
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_p_entity_existing_chain(lin, lsd)
    native_chain.write_p_entity_existing_chain(nin, nsd)
    for fname in (
        "p_entity_existing_capacity_later_solves.csv",
        "p_entity_all_existing.csv",
        "p_entity_existing_count.csv",
        "p_entity_existing_integer_count.csv",
        "p_entity_previously_invested_capacity.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_CHAIN)
def test_p_entity_capacity_max_chain_parity(
    tmp_path: Path, fixture: str,
) -> None:
    """4 cascading capacity-ceiling params (max_capacity, max_units,
    invest_cumulative_max, dispatch_capacity_max).  Depends on
    p_entity_all_existing from the existing-chain writer above — we
    invoke that first to provide the missing fixture row, then run
    both legacy / native variants of the cap-max chain in isolation.
    """
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    # p_entity_all_existing.csv is written by write_p_entity_existing_chain
    # in the live cascade.  We replay it on both sides so the cap-max
    # writer's read sees a deterministic upstream.
    legacy_entity_period.write_p_entity_existing_chain(lin, lsd)
    native_chain.write_p_entity_existing_chain(nin, nsd)

    legacy_entity_period.write_p_entity_capacity_max_chain(lin, lsd)
    native_chain.write_p_entity_capacity_max_chain(nin, nsd)
    for fname in (
        "p_entity_max_capacity.csv",
        "p_entity_max_units.csv",
        "p_entity_invest_cumulative_max.csv",
        "p_entity_dispatch_capacity_max.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ---------------------------------------------------------------------------
# Phase 1 closeout — top-level dispatcher own-compute.
#
# ``write_process_arc_unions`` emits 14 CSVs and is called from both
# ``input_writer.write_input`` (Phase 1) and ``preprocessing.solve_time``
# (Phase 2).  ``write_entity_period_calc_params`` emits 5 CSVs and is
# called from ``preprocessing.solve_time``.  Both functions are pure
# own-compute (no sub-writer calls), so the override hook can swap them
# atomically without touching the per-solve chain wiring.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture", FIXTURES)
def test_write_process_arc_unions_parity(tmp_path: Path, fixture: str) -> None:
    """Top-level dispatcher emits 14 CSVs from arc-union derivations."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_arc_unions.write_process_arc_unions(lin, lsd)
    native_disp.write_process_arc_unions(nin, nsd)
    for fname in (
        "process__profileProcess__toSink.csv",
        "process__source__toProfileProcess.csv",
        "process_profile.csv",
        "process_source_toProcess.csv",
        "process_process_toSink.csv",
        "process_source_sink_eff.csv",
        "process_source_sink_noEff.csv",
        "process_online.csv",
        "process_minload.csv",
        "process__commodity__node_co2.csv",
        "process_co2.csv",
        "process_source_sink.csv",
        "process_source_sink_alwaysProcess.csv",
        "process__source__sink__profile__profile_method_direct.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_write_entity_period_calc_params_parity(
    tmp_path: Path, fixture: str,
) -> None:
    """Top-level dispatcher emits pdProcess / pdNode + 3 ed_* CSVs."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_period.write_entity_period_calc_params(lin, lsd)
    native_disp.write_entity_period_calc_params(nin, nsd)
    for fname in (
        "pdProcess.csv",
        "pdNode.csv",
        "edEntity_lifetime.csv",
        "ed_fixed_cost.csv",
        "p_entity_unitsize.csv",
    ):
        _assert_files_equal(lsd / fname, nsd / fname)


# ===========================================================================
# Writer-port Phase 2 (sub-dispatch 1) — per-solve set + invest-divest writers.
#
# Both families are called from
# ``flextool.flextoolrunner.preprocessing.solve_time.run`` rather than from
# ``input_writer.write_input``.  The override hook in
# ``_native_input_writer.py`` does not currently wrap solve_time, so wiring
# is deferred to a follow-up dispatch; these tests assert native parity in
# isolation against the legacy emitter so the wiring step can land safely.
# ===========================================================================


# Fixtures with a populated per-solve solve_data state (period__branch,
# steps_in_use, ed_invest seeds etc.).  ``work_base`` / ``work_coal`` /
# ``work_test_a_lot`` all check those in.  Invest-cumulative covers richer
# ed_invest / edd_history scenarios.
FIXTURES_WITH_PER_SOLVE = FIXTURES + [
    "work_network_coal_wind_battery_invest_cumulative",
]


_PER_SOLVE_OUTPUTS = (
    "branch_set.csv",
    "year_set.csv",
    "period_from_period_time_set.csv",
    "period_in_use_set.csv",
    "time_in_use_set.csv",
    "complete_time_in_use_set.csv",
    "rp_base_period_set.csv",
    "rp_rep_period_set.csv",
    "period_block_set.csv",
    "dtt_set.csv",
    "d_fix_storage_period_set.csv",
    "period_set.csv",
    "periodAll_set.csv",
    "block_set.csv",
    "period__timeline_set.csv",
    "dt_realize_dispatch_set.csv",
    "d_realized_period_set.csv",
    "d_realize_dispatch_or_invest_set.csv",
    "dt_non_anticipativity_set.csv",
    "pdt_uptime_set.csv",
    "pdt_downtime_set.csv",
    "cnd_ladder_set.csv",
    "cndi_ladder_cum_set.csv",
    "cndi_ladder_ann_set.csv",
    "cndi_ladder_set.csv",
    "dtdt_next_set.csv",
    "n_fix_storage_quantity_set.csv",
    "n_fix_storage_price_set.csv",
    "n_fix_storage_usage_set.csv",
    "p_online_dt_set.csv",
)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PER_SOLVE)
def test_per_solve_sets_parity(tmp_path: Path, fixture: str) -> None:
    """Native ``write_per_solve_sets`` emits the same 30 CSVs as the legacy
    helper, byte-for-byte."""
    _lin, lsd, _nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_per_solve.write_per_solve_sets(lsd)
    native_per_solve.write_per_solve_sets(nsd)
    for fname in _PER_SOLVE_OUTPUTS:
        _assert_files_equal(lsd / fname, nsd / fname)


_INVEST_DIVEST_OUTPUTS = (
    "ed_invest.csv",
    "ed_divest.csv",
    "ed_invest_period.csv",
    "ed_divest_period.csv",
    "ed_invest_cumulative.csv",
    "pd_invest.csv",
    "nd_invest.csv",
    "pd_divest.csv",
    "nd_divest.csv",
    "edd_history_choice.csv",
    "edd_history_automatic.csv",
    "edd_history_no_investment.csv",
    "edd_history.csv",
    "edd_history_invest.csv",
    "edd_invest.csv",
    "gd_invest.csv",
    "gd_divest.csv",
    "gd_invest_period.csv",
    "gd_divest_period.csv",
)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PER_SOLVE)
def test_invest_divest_sets_parity(tmp_path: Path, fixture: str) -> None:
    """Native ``write_invest_divest_sets`` mirrors the legacy 19 CSVs."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_invest_divest.write_invest_divest_sets(lin, lsd)
    native_per_solve.write_invest_divest_sets(nin, nsd)
    for fname in _INVEST_DIVEST_OUTPUTS:
        _assert_files_equal(lsd / fname, nsd / fname)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PER_SOLVE)
def test_ed_invest_forbidden_no_investment_parity(
    tmp_path: Path, fixture: str,
) -> None:
    """Native ``write_ed_invest_forbidden_no_investment`` mirrors legacy."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    # Seed ed_invest.csv (consumed by both legacy + native helpers) by
    # running the predecessor write first; the seeded workdir's
    # solve_data/ed_invest.csv may be from a different model state.
    legacy_invest_divest.write_invest_divest_sets(lin, lsd)
    native_per_solve.write_invest_divest_sets(nin, nsd)
    legacy_invest_divest.write_ed_invest_forbidden_no_investment(lin, lsd)
    native_per_solve.write_ed_invest_forbidden_no_investment(nin, nsd)
    _assert_files_equal(
        lsd / "ed_invest_forbidden_no_investment.csv",
        nsd / "ed_invest_forbidden_no_investment.csv",
    )


# ---------------------------------------------------------------------------
# Phase 2 (sub-dispatch 2) — entity_annual + lp_scaling parity tests.
# ---------------------------------------------------------------------------

_ENTITY_ANNUAL_OUTPUTS = (
    "ed_entity_annual.csv",
    "ed_entity_annual_discounted.csv",
    "ed_entity_annual_divest.csv",
    "ed_entity_annual_divest_discounted.csv",
    "ed_lifetime_fixed_cost.csv",
    "ed_lifetime_fixed_cost_divest.csv",
)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PER_SOLVE)
def test_entity_annual_calc_params_parity(
    tmp_path: Path, fixture: str,
) -> None:
    """Native ``write_entity_annual_calc_params`` emits the six
    annuity / discounted / lifetime-fixed-cost CSVs byte-identically
    to the legacy helper.  Float values stress ``repr(float)``
    precision parity."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_entity_annual.write_entity_annual_calc_params(lin, lsd)
    native_entity_annual.write_entity_annual_calc_params(nin, nsd)
    for fname in _ENTITY_ANNUAL_OUTPUTS:
        _assert_files_equal(lsd / fname, nsd / fname)


_LP_SCALING_OUTPUTS = (
    "_node_cap_unitsize_sum.csv",
    "_node_cap_raw.csv",
    "_node_cap_pow10.csv",
    "node_capacity_for_scaling.csv",
    "inv_node_cap.csv",
    "_group_cap_raw.csv",
    "_group_cap_pow10.csv",
    "group_capacity_for_scaling.csv",
    "inv_group_cap.csv",
)


@pytest.mark.parametrize("fixture", FIXTURES_WITH_PER_SOLVE)
def test_lp_scaling_params_parity(
    tmp_path: Path, fixture: str,
) -> None:
    """Native ``write_lp_scaling_params`` mirrors the legacy 9 CSVs
    (node-level + group-level capacity proxies and their reciprocals)."""
    lin, lsd, nin, nsd = _seed_workdir(tmp_path, fixture)
    legacy_lp_scaling.write_lp_scaling_params(lin, lsd)
    native_lp_scaling.write_lp_scaling_params(nin, nsd)
    for fname in _LP_SCALING_OUTPUTS:
        _assert_files_equal(lsd / fname, nsd / fname)
