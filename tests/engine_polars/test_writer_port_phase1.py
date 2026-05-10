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

from flextool.engine_polars import _writer_leaf_sets as native
from flextool.flextoolrunner.preprocessing import (
    co2_method_sets as legacy_co2,
    invest_method_sets as legacy_invest,
    period_param_sets as legacy_period,
    simple_projections as legacy_simple,
)


DATA_DIR = Path(__file__).resolve().parent / "data"
FIXTURES = ["work_base", "work_coal", "work_test_a_lot"]


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
