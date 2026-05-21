"""Unit-level tests for the cascade's in-memory ``SolveHandoff`` consume helpers.

Phase 3 of ``specs/provider_consolidation.md`` deleted the disk-reading
``capture_post_solve`` constructor; the cascade builds ``SolveHandoff``
directly from the flexpy ``Solution`` via ``build_handoff_from_flexpy``.
End-to-end coverage lives in ``test_chain_handoff_writers.py``.

The unit-level test preserved here verifies a consume helper in
isolation:

* :func:`test_cumulative_loaders_consume_from_handoff` — the three
  cumulative-handoff prior loaders read from a populated
  ``SolveHandoff`` instead of disk when the optional ``prior_handoff``
  kwarg is supplied.

Phase 4.1i retired ``write_fix_storage_files_from_handoff``; the
corresponding disk-fan-out unit test was deleted alongside the helper.
"""
from __future__ import annotations

import polars as pl

from flextool.flextoolrunner.solve_handoff import (
    SolveHandoff,
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


# Phase 4.1i — ``test_write_fix_storage_files_from_handoff`` was deleted
# alongside the helper.  The wide → narrow CSV fan-out path has no
# consumers (per-metric ``handoff/*`` Provider keys cover all readers);
# end-to-end coverage of the new path lives in the rolling/chain
# handoff suites.
