"""Phase C — per-sub-solve FlexData accumulator parity test.

The accumulator captures the writer-port thin-wrapper writers'
derived frames during one sub-solve's preprocessing pass.  This test
verifies that every captured frame equals what the equivalent CSV
contains on disk after the cascade runs — proving the accumulator's
in-memory frames are a drop-in replacement for the disk-read path
(which Phase D will exploit by adding a ``seed`` kwarg to
``load_flextool``).

The test runs ``run_chain_from_db`` on the smallest fixture
(``work_base``), reads the last step's accumulator off the
:class:`OrchestrationStep`, and asserts each captured frame
``frames[csv_name]`` equals ``pl.read_csv(work / "solve_data" /
csv_name)`` after a sort-and-compare.

Phase C scope is the 37 ``OK_thin_wrapper`` writers identified in
``specs/phase_b_writer_audit.md``.  Special-handling writers (multi-
CSV streamed monoliths and row-by-row emitters; 103 in the audit)
are NOT wired into the accumulator yet — those require deeper
refactoring that is deferred to follow-up dispatches.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import run_chain_from_db
from flextool.engine_polars._flex_data_accumulator import (
    FlexDataAccumulator,
)


pytestmark = pytest.mark.solver


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
WORK_NAME = "work_base"
SCENARIO_NAME = "base"


# ---------------------------------------------------------------------------
# CSV files known to be populated by the 37 OK_thin_wrapper writers.
#
# Sourced from ``specs/phase_b_writer_audit.md`` ("none (OK)" entries).
# Frames written by writers further down the chain (Phase C special-
# handling list) are NOT in scope here — Phase D will populate the
# corresponding FlexData fields via load_flextool's disk read.
# ---------------------------------------------------------------------------

ACCUMULATOR_COVERAGE: tuple[str, ...] = (
    # _writer_leaf_sets (L0-L2)
    "period_group.csv",
    "period_node.csv",
    "period_commodity.csv",
    "period_process.csv",
    "entityInvest.csv",
    "entityDivest.csv",
    "group_invest.csv",
    "group_divest.csv",
    "group_co2_price.csv",
    "group_co2_max_period.csv",
    "group_co2_max_total.csv",
    "optional_yes.csv",
    "reserve__upDown__group.csv",
    "group_loss_share.csv",
    "def_optional_yes.csv",
    "process_delayed.csv",
    "process_side.csv",
    "period_solve.csv",
    "time.csv",
    "enable_optional_outputs.csv",
    "nodeState_rp.csv",
    "nodeStateBlock.csv",
    "commodity__tier.csv",
    "tier.csv",
    "timeline.csv",
    "timeline_steps.csv",
    "commodity__tier_ann.csv",
    # _writer_mid_sets (L3-L6)
    "group_entity.csv",
    "process_delayed__duration.csv",
    "process__sink_nonSync.csv",
    "entity_lifetime_method.csv",
    "process_ct_method.csv",
    "process_startup_method.csv",
    "node_inflow_method.csv",
    "node_storage_binding_method.csv",
    "connection_param.csv",
    "nodegroup_dispatch_node.csv",
    "commodity_node_co2.csv",
    "process__commodity__node.csv",
    # _writer_calc_params (L7-L9)
    "process_VRE.csv",
)


def _normalise(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by all columns + reset index to make order-insensitive
    comparison robust to writer / reader ordering differences."""
    if df.is_empty():
        return df
    return df.sort(df.columns)


def _read_solve_data_csv(
    work_folder: Path, csv_name: str
) -> pl.DataFrame | None:
    """Read ``solve_data/<csv_name>`` if present, else None.

    The writer-port CSVs all have a header row.  Empty-body CSVs
    (header only) read back as a zero-row DataFrame with the
    correct columns.
    """
    p = work_folder / "solve_data" / csv_name
    if not p.exists():
        return None
    # Some derived CSVs are empty (header only); pl.read_csv handles that
    # by returning an empty frame with the header's column names.  Use
    # ``infer_schema_length=None`` so type inference doesn't drop columns
    # that happen to be empty in the small ``work_base`` fixture.
    return pl.read_csv(p, infer_schema_length=10000)


def test_phase_c_accumulator_matches_disk_csvs(tmp_path: Path) -> None:
    """Run a single-solve cascade on ``work_base`` and assert the
    captured accumulator frames equal the on-disk CSVs.

    The accumulator is parallel-write only at this phase: CSVs still
    land on disk under ``<work>/solve_data/`` AND the accumulator
    captures the derived frames.  This test asserts the captured
    frames match the disk frames — proving Phase D's seed path can
    skip the disk-read for these 37+ CSVs without semantic drift.
    """
    fixture = DATA / WORK_NAME
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / WORK_NAME
    sols = run_chain_from_db(
        db, scenario_name=SCENARIO_NAME, work_folder=work,
    )
    assert sols, "cascade produced no sub-solves"

    # Phase C wires the accumulator onto the latest sub-solve's
    # OrchestrationStep.  ``work_base`` is single-solve so there is
    # one step; multi-roll fixtures retain only the latest sub-solve's
    # accumulator (per-sub-solve memory discipline).
    last_step = next(reversed(sols.values()))
    accum = last_step.flex_data_accumulator
    assert accum is not None, (
        "OrchestrationStep.flex_data_accumulator should be populated by "
        "_native_run_model — got None.  Wiring regression."
    )
    assert isinstance(accum, FlexDataAccumulator), (
        f"unexpected accumulator type: {type(accum).__name__}"
    )

    # The accumulator should have collected at least *some* of the 37
    # tracked CSVs (depends on which writer paths the fixture exercises;
    # work_base is the minimal fixture so it covers the leaf-set core).
    assert accum.frames, (
        "accumulator captured zero frames — capture_frames context not "
        "active around preprocessing call?"
    )

    # Frame-by-frame: every CSV the accumulator captured AND that is in
    # our tracked coverage list must equal the on-disk CSV exactly
    # (modulo column order).  CSVs not exercised by this fixture are
    # silently skipped — they belong to Phase C's special-handling
    # follow-ups OR to entity domains absent from work_base.
    asserted_keys: list[str] = []
    skipped_not_captured: list[str] = []
    skipped_no_csv: list[str] = []
    for csv_name in ACCUMULATOR_COVERAGE:
        in_accum = csv_name in accum.frames
        disk = _read_solve_data_csv(work, csv_name)
        if not in_accum:
            skipped_not_captured.append(csv_name)
            continue
        if disk is None:
            skipped_no_csv.append(csv_name)
            continue
        captured = accum.get(csv_name)
        assert captured is not None
        # Compare on column set first — writers may pick a canonical
        # order; CSV reader picks header order.  Both should produce
        # the same set of columns.
        assert set(captured.columns) == set(disk.columns), (
            f"{csv_name}: captured cols {captured.columns} ≠ "
            f"disk cols {disk.columns}"
        )
        # Then row-level equality after sort.  Cast disk to captured's
        # dtypes — pl.read_csv may infer a different (e.g. Int64 vs
        # String) dtype for empty / sparse columns.
        disk_aligned = disk.select(captured.columns)
        # If shapes differ but contents are equivalent under cast, the
        # equals() check below catches that.  Otherwise widen.
        try:
            disk_aligned = disk_aligned.cast(
                {c: captured.schema[c] for c in captured.columns}
            )
        except Exception:  # noqa: BLE001
            # Dtype mismatch unrecoverable via cast — keep disk dtypes
            # and let the equals() check below decide.
            pass
        a = _normalise(captured)
        b = _normalise(disk_aligned)
        assert a.equals(b), (
            f"{csv_name}: accumulator frame differs from disk CSV.\n"
            f"  captured ({a.shape}):\n{a}\n"
            f"  on-disk  ({b.shape}):\n{b}\n"
        )
        asserted_keys.append(csv_name)

    # Sanity floor: assert SOMETHING got compared — protects against a
    # silent miss where the accumulator wires up but captures nothing
    # work_base touches.  work_base is the simplest fixture so we expect
    # at least the canonical leaf sets.
    assert len(asserted_keys) >= 10, (
        f"only {len(asserted_keys)} csv(s) parity-checked: "
        f"{asserted_keys}.  Expected work_base to exercise the leaf-set "
        f"core (period_*.csv, entity*.csv, group_*.csv …).  "
        f"not_captured={skipped_not_captured} no_csv={skipped_no_csv}"
    )


def test_phase_c_accumulator_keys_are_csv_basenames(tmp_path: Path) -> None:
    """The accumulator keys are basenames (no directory component).

    Phase D will map basenames into FlexData field names; the keying
    convention is part of the public contract.
    """
    fixture = DATA / WORK_NAME
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    work = tmp_path / WORK_NAME
    sols = run_chain_from_db(
        db, scenario_name=SCENARIO_NAME, work_folder=work,
    )
    last_step = next(reversed(sols.values()))
    accum = last_step.flex_data_accumulator
    assert accum is not None

    bad = [k for k in accum.frames if "/" in k or "\\" in k]
    assert not bad, f"non-basename keys leaked into accumulator: {bad}"

    # All captured keys end with .csv — they are CSV target filenames.
    not_csv = [k for k in accum.frames if not k.endswith(".csv")]
    assert not not_csv, f"non-CSV keys captured: {not_csv}"
