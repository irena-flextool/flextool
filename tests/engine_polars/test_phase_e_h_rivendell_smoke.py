"""Phase E-h regression — seed-aware existence guard on solve_data readers.

The Rivendell ``B0_base_hourly_rp`` scenario regressed from optimal
(objective 24.96) to HiGHS-presolve-infeasible after Phase E-c gated CSV
emission behind ``--csv-dump``.  Root cause: a handful of
``_derived_params.py`` reader helpers used bare ``path.exists()`` checks
on ``solve_data/`` CSVs the writer-port accumulator covers.  When CSV
emission is disabled (Phase E-c default), the disk file is absent even
though the active in-memory accumulator holds the frame; the bare
``exists()`` short-circuit returned ``None`` and the caller fell back to
a raw-existing source path that doesn't apply the lifetime-cumulative
chain.  For Rivendell B0, that wrong-fallback inflated
``p_flow_upper_existing[RVN_PP_NGS_C, y2039]`` from ``0.002`` to ``1.0``,
which in turn skewed the LP's row-bound range and pushed HiGHS'
``user_bound_scale`` heuristic from ``-10`` to ``-19`` — small bounds
got scaled below feasibility tolerance and HiGHS presolved the model
infeasible.

The fix (Phase E-h) replaces every bare ``path.exists()`` guard on an
accumulator-tracked ``solve_data`` CSV in ``_derived_params.py`` with
:func:`_seed_or_exists`, the seed-aware check Phase E-f introduced for
the same pattern elsewhere.

This test directly exercises the offending pattern with a tiny
synthetic seed — no full LP solve required.  It fails on the pre-fix
HEAD (988efb2e) and passes after the fix.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import _input_source as _ipsrc
from flextool.engine_polars._derived_params import (
    _ctx_read,
    _read_p_entity_all_existing_csv,
)
from flextool.engine_polars._flex_data_accumulator import FlexDataAccumulator


# ---------------------------------------------------------------------------
# A. Direct guard tests — the bug pattern in isolation.
# ---------------------------------------------------------------------------


def test_read_p_entity_all_existing_csv_uses_seed_when_disk_absent(
    tmp_path: Path,
) -> None:
    """The core regression: when the cascade runs with CSV emission
    disabled, ``p_entity_all_existing.csv`` lives only in the in-memory
    accumulator.  ``_read_p_entity_all_existing_csv`` MUST consult the
    seed before bailing out, otherwise downstream
    ``p_flow_upper_existing`` falls back to the raw-existing source path
    and produces wrong values (the Rivendell B0_base_hourly_rp regression).
    """
    work = tmp_path / "work"
    (work / "solve_data").mkdir(parents=True)
    assert not (work / "solve_data" / "p_entity_all_existing.csv").exists(), (
        "fixture invariant: disk CSV must be absent so the seed is the "
        "only source"
    )

    accum = FlexDataAccumulator()
    seeded = pl.DataFrame(
        {
            "entity": ["RVN_PP_NGS_C", "RVN_PP_NGS_C"],
            "period": ["y2039", "y2050"],
            "value": [2.0, 0.0],
        }
    )
    accum.capture(work / "solve_data" / "p_entity_all_existing.csv", seeded)

    _ipsrc._install_seed(accum)
    try:
        out = _read_p_entity_all_existing_csv(work)
    finally:
        _ipsrc._install_seed(None)

    assert out is not None, (
        "Phase E-h regression: with the seed active and disk absent, "
        "_read_p_entity_all_existing_csv returned None — the bare "
        "path.exists() guard short-circuited before consulting the seed. "
        "Downstream p_flow_upper_existing falls back to the wrong path."
    )
    # Schema invariant: rename to (e, d, value), cast value to Float64.
    assert out.columns == ["e", "d", "value"]
    row = out.filter(
        (pl.col("e") == "RVN_PP_NGS_C") & (pl.col("d") == "y2039")
    )
    assert row.height == 1
    assert row["value"][0] == pytest.approx(2.0)


def test_ctx_read_uses_seed_when_disk_absent(tmp_path: Path) -> None:
    """Companion to the above — ``_ctx_read`` is the generic gateway
    used by many derived helpers (period_branch, fix_storage_quantity,
    p_roll_continue_state, …).  Same pattern, same fix: bare
    ``path.exists()`` must be replaced by the seed-aware variant or
    those helpers also lose the in-memory frame when CSV emission is
    disabled.
    """
    work = tmp_path / "work"
    (work / "solve_data").mkdir(parents=True)
    assert not (work / "solve_data" / "fix_storage_quantity.csv").exists()

    accum = FlexDataAccumulator()
    seeded = pl.DataFrame(
        {
            "period": ["y2030"],
            "step": ["t0001"],
            "node": ["RVN_ELC_B"],
            "p_fix_storage_quantity": [12.5],
        }
    )
    accum.capture(
        work / "solve_data" / "fix_storage_quantity.csv", seeded,
    )

    _ipsrc._install_seed(accum)
    try:
        out = _ctx_read(None, work, "fix_storage_quantity.csv")
    finally:
        _ipsrc._install_seed(None)

    assert out is not None, (
        "Phase E-h regression: _ctx_read short-circuited on bare "
        "path.exists() and never consulted the active seed. Generic "
        "rolling-handoff helpers that route through _ctx_read lose "
        "their in-memory frame when CSV emission is disabled."
    )
    assert out.height == 1
    assert out["p_fix_storage_quantity"][0] == pytest.approx(12.5)


def test_seed_inactive_still_falls_through_when_disk_absent(
    tmp_path: Path,
) -> None:
    """Negative control — with no seed active and disk absent, the
    helpers must still return ``None`` (pre-Phase-E-f behaviour
    preserved on the csv-emission-on path)."""
    work = tmp_path / "work"
    (work / "solve_data").mkdir(parents=True)
    # Ensure the global seed slot is clear (defensive — other tests in
    # the same process might have left it set).
    _ipsrc._install_seed(None)

    assert _read_p_entity_all_existing_csv(work) is None
    assert _ctx_read(None, work, "fix_storage_quantity.csv") is None


# ---------------------------------------------------------------------------
# B. Bug-shape integration — confirm the end-to-end path the regression
#    travelled.  Synthesises the lookup chain ``_flow_upper_existing_from
#    _chained_csv`` depends on, sans a real solve.
# ---------------------------------------------------------------------------


def test_chained_csv_path_active_when_seed_holds_frame(
    tmp_path: Path,
) -> None:
    """Higher-level proof: with a seeded ``p_entity_all_existing`` the
    ``_flow_upper_existing_from_chained_csv`` helper must return a
    non-None Param (i.e. it walks the chained CSV path), so the caller
    in ``p_flow_upper_existing_from_source`` does NOT fall through to
    the raw-existing source path.  On the pre-fix HEAD this helper
    returned None and downstream p_flow_upper_existing was wrong.
    """
    from flextool.engine_polars._derived_params import (
        _flow_upper_existing_from_chained_csv,
    )

    work = tmp_path / "work"
    (work / "solve_data").mkdir(parents=True)
    accum = FlexDataAccumulator()
    accum.capture(
        work / "solve_data" / "p_entity_all_existing.csv",
        pl.DataFrame(
            {
                "entity": ["P1", "P1"],
                "period": ["d1", "d2"],
                "value": [10.0, 0.0],
            }
        ),
    )

    # Minimal pss frame the helper expects (one arc).
    pss = pl.DataFrame(
        {
            "p": ["P1"],
            "source": ["S1"],
            "sink": ["K1"],
        }
    )

    # We need a stand-in ``source`` with the entity-class / unitsize
    # accessors the helper calls.  The chained-CSV path only reads
    # ``_entity_classes_lookup`` (for the process filter) and
    # ``_entity_unitsize_lf`` (for the cascade); we shim both via a tiny
    # fake source object that satisfies the duck-typed contract.
    import types

    fake_source = types.SimpleNamespace()

    # Patch the two module-level helpers _flow_upper_existing_from_chained_csv
    # actually calls.  Mocking is appropriate here because the test's
    # intent is to verify the seed-aware existence check, not the join
    # arithmetic (covered elsewhere).
    from flextool.engine_polars import _derived_params as _dp

    original_ecl = _dp._entity_classes_lookup
    original_eus = _dp._entity_unitsize_lf
    _dp._entity_classes_lookup = lambda src: ({"P1"}, set(), set())
    _dp._entity_unitsize_lf = lambda src: pl.DataFrame(
        {"e": ["P1"], "us": [5.0]}
    ).lazy()

    _ipsrc._install_seed(accum)
    try:
        out = _flow_upper_existing_from_chained_csv(
            fake_source, pss, workdir=work,
        )
    finally:
        _ipsrc._install_seed(None)
        _dp._entity_classes_lookup = original_ecl
        _dp._entity_unitsize_lf = original_eus

    assert out is not None, (
        "Phase E-h regression: the chained-CSV path returned None even "
        "though the seed holds p_entity_all_existing.  Downstream "
        "p_flow_upper_existing then takes the raw-existing fallback — "
        "the exact failure mode that infeasibled Rivendell B0_base_hourly_rp."
    )
    # value = 10.0 / unitsize 5.0 = 2.0; arc-broadcasted to (P1, S1, K1, d1)
    row = out.frame.filter(pl.col("d") == "d1")
    assert row.height == 1
    assert row["value"][0] == pytest.approx(2.0)
