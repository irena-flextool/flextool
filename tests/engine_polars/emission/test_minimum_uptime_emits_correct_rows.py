"""Tier-7 #25: ``minimum_uptime`` is emitted on the right (p, d, t) set.

Fixture: ``work_coal_wind_min_uptime`` (linear-online unit-commitment
on coal_plant; no integer UC).  We expect the linear-variant constraint
to show one LP row per ``(p, d, t)`` in ``data.pdt_uptime_set`` and the
integer-variant family to be absent entirely (count = 0).
"""

from __future__ import annotations

import pytest

from tests.engine_polars.conftest import DATA_DIR
from tests.engine_polars.emission._helpers import (
    assert_cstr_absent,
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_minimum_uptime_emits_correct_rows() -> None:
    pb, data = build(DATA_DIR / "work_coal_wind_min_uptime")

    assert data.pdt_uptime_set is not None and data.pdt_uptime_set.height > 0, (
        "fixture invariant: pdt_uptime_set must be non-empty for this test"
    )

    # Linear-online UC variant: one LP row per (p, d, t) in pdt_uptime_set.
    assert_cstr_row_count(
        pb, "minimum_uptime_linear", data.pdt_uptime_set.height
    )

    # Integer-online variant must NOT be emitted on a linear-only fixture.
    # process_online_integer is empty here, so the block is skipped.
    assert_cstr_absent(pb, "minimum_uptime_integer")

    # Cross-check: the prefix-match accessor returns just the linear record.
    recs = pb.cstrs_named("minimum_uptime")
    assert [r.name for r in recs] == ["minimum_uptime_linear"]
    assert recs[0].over.height == data.pdt_uptime_set.height

    # Sanity: minimum_uptime exists in cstr_names listing.
    assert_cstr_present(pb, "minimum_uptime_linear")
