"""Tests for the in-memory ``SolveHandoff`` carrier + its CSV bridges.

Two surfaces:

* :func:`flextool.flextoolrunner.solve_handoff.capture_post_solve` —
  reads each carrier file from ``solve_data/`` after a solve completes
  and parks a :class:`SolveHandoff` into ``state.handoffs[solve_name]``.
* :func:`flextool.flextoolrunner.solve_handoff.write_fix_storage_files_from_handoff`
  — fans the wide ``fix_storage`` carrier back out to the three on-disk
  files, with NULL-aware per-metric independence.

These tests build a skeleton ``RunnerState`` (PathConfig + MagicMock for
unused fields) the same way ``tests/test_solver_options.py`` does, so
no solver runs.  Critically, ``state.handoffs = {}`` opts in — without
it ``capture_post_solve`` short-circuits at ``solve_handoff.py:137``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import polars as pl
import pytest

from flextool.flextoolrunner.runner_state import PathConfig, RunnerState
from flextool.flextoolrunner.solve_handoff import (
    capture_post_solve,
    write_fix_storage_files_from_handoff,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(tmp_path: Path) -> RunnerState:
    """Skeleton ``RunnerState`` with handoffs opted-in (``{}``).

    Mirrors the pattern from ``tests/test_solver_options.py:218-245`` —
    only ``state.paths.work_folder`` and ``state.handoffs`` are read by
    the SolveHandoff hooks; everything else is a placeholder.
    """
    wf = tmp_path / "work"
    wf.mkdir(parents=True)
    (wf / "solve_data").mkdir()
    paths = PathConfig(
        flextool_dir=tmp_path,
        bin_dir=tmp_path,
        root_dir=tmp_path,
        output_path=tmp_path,
        work_folder=wf,
    )
    state = RunnerState(
        paths=paths,
        solve=MagicMock(),
        timeline=MagicMock(),
        logger=logging.getLogger("test_solve_handoff"),
    )
    # Opt in to in-memory handoffs (the default ``None`` short-circuits
    # capture_post_solve at line 137).
    state.handoffs = {}
    return state


# ---------------------------------------------------------------------------
# capture_post_solve — cum_sim_hours (lines 257-265)
# ---------------------------------------------------------------------------


def test_capture_post_solve_cum_sim_hours(tmp_path: Path) -> None:
    """``ladder_cum_sim_hours.csv`` → ``handoff.cum_sim_hours`` with schema
    ``(period, value)``.  Missing file → field stays ``None``."""
    state = _make_state(tmp_path)
    sd = state.paths.work_folder / "solve_data"
    (sd / "ladder_cum_sim_hours.csv").write_text(
        "period,p_ladder_cum_sim_hours\n"
        "p2020,8760\n"
        "p2025,17520\n"
    )

    capture_post_solve(state, "s1")
    h = state.handoffs["s1"]
    assert h.cum_sim_hours is not None
    assert h.cum_sim_hours.columns == ["period", "value"]
    assert h.cum_sim_hours.to_dicts() == [
        {"period": "p2020", "value": 8760.0},
        {"period": "p2025", "value": 17520.0},
    ]

    # Missing file path: a fresh state with no CSV → field is None.
    state2 = _make_state(tmp_path / "second")
    capture_post_solve(state2, "s1")
    assert state2.handoffs["s1"].cum_sim_hours is None


# ---------------------------------------------------------------------------
# capture_post_solve — cumulative_commodity (lines 239-255)
# ---------------------------------------------------------------------------


def test_capture_post_solve_cumulative_commodity(tmp_path: Path) -> None:
    """``commodity_ladder_cumulative.csv`` → ``handoff.cumulative_commodity``;
    accepts either ``mwh`` or ``p_ladder_cum_realized_mwh`` as the value
    column (the dual-column tolerance is the bug surface)."""
    # Variant A: value column named ``mwh``.
    state_a = _make_state(tmp_path / "a")
    (state_a.paths.work_folder / "solve_data" / "commodity_ladder_cumulative.csv").write_text(
        "commodity,tier,period,mwh\n"
        "gas,t1,p2020,100\n"
        "gas,t2,p2020,50\n"
    )
    capture_post_solve(state_a, "s1")
    cc_a = state_a.handoffs["s1"].cumulative_commodity
    assert cc_a is not None
    assert cc_a.columns == ["commodity", "tier", "period", "mwh"]
    assert cc_a.to_dicts() == [
        {"commodity": "gas", "tier": "t1", "period": "p2020", "mwh": 100.0},
        {"commodity": "gas", "tier": "t2", "period": "p2020", "mwh": 50.0},
    ]

    # Variant B: same data, value column named ``p_ladder_cum_realized_mwh``.
    state_b = _make_state(tmp_path / "b")
    (state_b.paths.work_folder / "solve_data" / "commodity_ladder_cumulative.csv").write_text(
        "commodity,tier,period,p_ladder_cum_realized_mwh\n"
        "gas,t1,p2020,100\n"
        "gas,t2,p2020,50\n"
    )
    capture_post_solve(state_b, "s1")
    cc_b = state_b.handoffs["s1"].cumulative_commodity
    assert cc_b is not None
    # Despite the different on-disk column name, the in-memory schema is
    # identical — ``mwh`` is the canonical name in the carrier.
    assert cc_b.columns == ["commodity", "tier", "period", "mwh"]
    assert cc_b.to_dicts() == cc_a.to_dicts()


# ---------------------------------------------------------------------------
# capture_post_solve — periods_already_emitted (lines 267-270)
# ---------------------------------------------------------------------------


def test_capture_post_solve_periods_already_emitted(tmp_path: Path) -> None:
    """``period_capacity.csv`` → ``handoff.periods_already_emitted`` with
    duplicates collapsed; missing file or missing ``period`` column → None."""
    # Case A: file with duplicated period rows → deduped via ``.unique()``.
    state_a = _make_state(tmp_path / "a")
    (state_a.paths.work_folder / "solve_data" / "period_capacity.csv").write_text(
        "period\np2020\np2025\np2020\n"
    )
    capture_post_solve(state_a, "s1")
    pae_a = state_a.handoffs["s1"].periods_already_emitted
    assert pae_a is not None
    assert pae_a.columns == ["period"]
    assert sorted(pae_a["period"].to_list()) == ["p2020", "p2025"]

    # Case B: missing file → None (the ``_read`` helper returns None when
    # the file doesn't exist; the guard at line 269 then short-circuits).
    state_b = _make_state(tmp_path / "b")
    capture_post_solve(state_b, "s1")
    assert state_b.handoffs["s1"].periods_already_emitted is None

    # Case C: file present but no ``period`` column → None (column-presence
    # guard at line 269; without it the ``.select("period")`` would raise).
    state_c = _make_state(tmp_path / "c")
    (state_c.paths.work_folder / "solve_data" / "period_capacity.csv").write_text(
        "wrong_col\np2020\n"
    )
    capture_post_solve(state_c, "s1")
    assert state_c.handoffs["s1"].periods_already_emitted is None


# ---------------------------------------------------------------------------
# write_fix_storage_files_from_handoff — round-trip with NULL-aware fan-out
# ---------------------------------------------------------------------------


def test_solve_handoff_round_trip_fix_storage(tmp_path: Path) -> None:
    """Wide ``[node, period, time, quantity, price, usage]`` → three per-metric
    files, each containing ONLY rows where its metric is non-NULL.  Verifies
    per-metric independence (mixed NULLs across rows) plus the on-disk
    ``time → step`` rename in ``solve_handoff.py:301``."""
    sd = tmp_path / "solve_data"
    sd.mkdir()
    # Row 1: only quantity.  Row 2: only price.  Row 3: all three set.
    fix_storage = pl.DataFrame(
        {
            "node":     ["battery", "battery", "tank"],
            "period":   ["p2020",   "p2020",   "p2025"],
            "time":     ["t0001",   "t0024",   "t0001"],
            "quantity": [50.0,      None,      11.0],
            "price":    [None,      -1600.0,   22.0],
            "usage":    [None,      None,      33.0],
        },
        schema={
            "node":     pl.Utf8,
            "period":   pl.Utf8,
            "time":     pl.Utf8,
            "quantity": pl.Float64,
            "price":    pl.Float64,
            "usage":    pl.Float64,
        },
    )

    write_fix_storage_files_from_handoff(fix_storage, sd)

    # quantity: rows where ``quantity`` is non-null (rows 1 and 3).
    qdf = pl.read_csv(sd / "fix_storage_quantity.csv")
    assert qdf.columns == ["node", "period", "step", "p_fix_storage_quantity"]
    assert qdf.to_dicts() == [
        {"node": "battery", "period": "p2020", "step": "t0001",
         "p_fix_storage_quantity": 50.0},
        {"node": "tank",    "period": "p2025", "step": "t0001",
         "p_fix_storage_quantity": 11.0},
    ]

    # price: rows where ``price`` is non-null (rows 2 and 3).
    pdf = pl.read_csv(sd / "fix_storage_price.csv")
    assert pdf.columns == ["node", "period", "step", "p_fix_storage_price"]
    assert pdf.to_dicts() == [
        {"node": "battery", "period": "p2020", "step": "t0024",
         "p_fix_storage_price": -1600.0},
        {"node": "tank",    "period": "p2025", "step": "t0001",
         "p_fix_storage_price": 22.0},
    ]

    # usage: rows where ``usage`` is non-null (only row 3).
    udf = pl.read_csv(sd / "fix_storage_usage.csv")
    assert udf.columns == ["node", "period", "step", "p_fix_storage_usage"]
    assert udf.to_dicts() == [
        {"node": "tank", "period": "p2025", "step": "t0001",
         "p_fix_storage_usage": 33.0},
    ]
