"""Tier-7 #26: ``minimum_downtime`` is emitted on the right (p, d, t) set.

Fixture: ``work_coal_wind_min_uptime`` (same as #25 — the fixture
exercises both uptime and downtime constraints on the linear-online
coal_plant).  Expects one LP row per ``(p, d, t)`` in
``data.pdt_downtime_set`` and the integer-variant family to be absent.
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
def test_minimum_downtime_emits_correct_rows() -> None:
    pb, data = build(DATA_DIR / "work_coal_wind_min_uptime")

    assert data.pdt_downtime_set is not None and data.pdt_downtime_set.height > 0, (
        "fixture invariant: pdt_downtime_set must be non-empty for this test"
    )

    # Linear-online UC variant: one LP row per (p, d, t) in pdt_downtime_set.
    assert_cstr_row_count(
        pb, "minimum_downtime_linear", data.pdt_downtime_set.height
    )

    # Integer-online variant must NOT be emitted on a linear-only fixture.
    assert_cstr_absent(pb, "minimum_downtime_integer")

    # Prefix accessor returns just the linear record.
    recs = pb.cstrs_named("minimum_downtime")
    assert [r.name for r in recs] == ["minimum_downtime_linear"]
    assert recs[0].over.height == data.pdt_downtime_set.height

    assert_cstr_present(pb, "minimum_downtime_linear")
