"""Unit-level tests for the cascade's in-memory ``SolveHandoff`` consume helpers.

Phase 3 of ``specs/provider_consolidation.md`` deleted the disk-reading
``capture_post_solve`` constructor; the cascade builds ``SolveHandoff``
directly from the flexpy ``Solution`` via ``build_handoff_from_flexpy``.
End-to-end coverage lives in ``test_chain_handoff_writers.py``.

The 2 unit-level tests preserved here verify the consume helpers in
isolation:

* :func:`test_cumulative_loaders_consume_from_handoff` — the three
  cumulative-handoff prior loaders read from a populated
  ``SolveHandoff`` instead of disk when the optional ``prior_handoff``
  kwarg is supplied.
* :func:`test_write_fix_storage_files_from_handoff` — the wide
  ``fix_storage`` handoff frame fans back out to three long-format
  ``fix_storage_*.csv`` files on disk, with NULL metric rows excluded
  per file and the time-axis column renamed from ``time`` to ``step``.
"""
from __future__ import annotations

import polars as pl

from flextool.flextoolrunner.solve_handoff import (
    SolveHandoff,
    write_fix_storage_files_from_handoff,
)


def test_cumulative_loaders_consume_from_handoff(tmp_path):
    """Unit-level: the three cumulative-handoff prior loaders read from
    a populated ``SolveHandoff`` instead of disk when the optional
    ``prior_handoff`` kwarg is supplied.  Verifies disk is bypassed by
    pointing the file path at a non-existent location — only the
    handoff path can produce non-empty output."""
    from flextool.process_outputs.cumulative_handoffs import (
        _load_prior_co2_cum_realized_tonnes,
        _load_prior_cum_realized_mwh,
        _load_prior_cum_sim_hours,
    )

    # Doesn't exist — file-only path would return {}.
    nope = tmp_path / "does_not_exist.csv"

    h = SolveHandoff(
        cumulative_co2=pl.DataFrame({
            "group":  ["g1", "g2"],
            "period": ["p2025", "p2025"],
            "value":  [12.5, 99.0],
        }),
        # Phase 4.1a — handoff carrier schemas match the canonical
        # ``solve_data/`` column names so the iteration-start translator
        # can route them through unchanged.
        cumulative_commodity=pl.DataFrame({
            "commodity":                 ["coal"],
            "tier":                      [1],
            "period":                    ["p2025"],
            "p_ladder_cum_realized_mwh": [42.0],
        }),
        cum_sim_hours=pl.DataFrame({
            "period":                 ["p2025", "p2030"],
            "p_ladder_cum_sim_hours": [8760.0, 4380.0],
        }),
    )

    co2 = _load_prior_co2_cum_realized_tonnes(nope, prior_handoff=h)
    assert co2 == {("g1", "p2025"): 12.5, ("g2", "p2025"): 99.0}

    mwh = _load_prior_cum_realized_mwh(nope, prior_handoff=h)
    assert mwh == {("coal", 1, "p2025"): 42.0}

    hrs = _load_prior_cum_sim_hours(nope, prior_handoff=h)
    assert hrs == {"p2025": 8760.0, "p2030": 4380.0}

    # Sanity: without the handoff, the same call returns empty (proving
    # the disk path is what the handoff is bypassing).
    assert _load_prior_co2_cum_realized_tonnes(nope) == {}
    assert _load_prior_cum_realized_mwh(nope) == {}
    assert _load_prior_cum_sim_hours(nope) == {}


def test_write_fix_storage_files_from_handoff(tmp_path):
    """Unit-level: the wide ``fix_storage`` handoff frame fans back out
    to three long-format ``fix_storage_*.csv`` files on disk, with NULL
    metric rows excluded per file and the time-axis column renamed
    from ``time`` to ``step`` (the on-disk convention).

    This is the only test for the fix_storage consume helper; no
    current fixture exercises the orchestration-level shutil.copy
    branch it replaces (all 18 scenarios have empty fix_storage_*.csv
    on disk)."""
    sd = tmp_path / "solve_data"
    sd.mkdir()

    # Wide row with mixed NULLs — the producer's natural output shape
    # from ``build_handoff_from_flexpy``.
    fix_storage = pl.DataFrame({
        "node":     ["battery", "battery", "tank"],
        "period":   ["p2025",   "p2025",   "p2030"],
        "time":     ["t0001",   "t0002",   "t0001"],
        "quantity": [10.0, 20.0, None],
        "price":    [None, 5.0,  None],
        "usage":    [None, None, 0.7],
    })

    write_fix_storage_files_from_handoff(fix_storage, sd)

    # Each per-metric file should contain only its non-NULL rows, with
    # the on-disk column name and ``step`` (not ``time``) for the axis.
    qty = pl.read_csv(sd / "fix_storage_quantity.csv")
    assert qty.columns == ["node", "period", "step", "p_fix_storage_quantity"]
    assert qty.height == 2
    assert sorted(qty["p_fix_storage_quantity"].to_list()) == [10.0, 20.0]

    price = pl.read_csv(sd / "fix_storage_price.csv")
    assert price.columns == ["node", "period", "step", "p_fix_storage_price"]
    assert price.height == 1
    assert price["p_fix_storage_price"][0] == 5.0
    assert price["step"][0] == "t0002"

    usage = pl.read_csv(sd / "fix_storage_usage.csv")
    assert usage.columns == ["node", "period", "step", "p_fix_storage_usage"]
    assert usage.height == 1
    assert usage["node"][0] == "tank"
    assert usage["p_fix_storage_usage"][0] == 0.7
