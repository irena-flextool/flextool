"""Tier-7 #28: ``maxToSink`` is emitted on the full ``pss_dt`` cross.

Fixture: ``work_coal``.  The capacity bound on every flow:
``v_flow[p, source, sink, d, t]  ≤  cap[p, d]`` should produce exactly
one LP row per ``(p, source, sink, d, t)`` tuple in ``data.pss_dt``.

A regression that broadened or narrowed the constraint domain (e.g.
forgetting to cross with ``dt``, or accidentally filtering via an
unrelated set) would change this count and trip the assertion.
"""

from __future__ import annotations

import pytest

from tests.engine_polars.conftest import DATA_DIR
from tests.engine_polars.emission._helpers import (
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_maxToSink_emits_one_row_per_pss_dt() -> None:
    pb, data = build(DATA_DIR / "work_coal")

    assert data.pss_dt is not None and data.pss_dt.height > 0, (
        "fixture invariant: pss_dt must be non-empty"
    )

    assert_cstr_row_count(pb, "maxToSink", data.pss_dt.height)
    assert_cstr_present(pb, "maxToSink")

    # The constraint should be a single record (no _linear/_integer split).
    recs = pb.cstrs_named("maxToSink")
    assert len(recs) == 1 and recs[0].name == "maxToSink"
    # `over` should carry the full (p, source, sink, d, t) axis.
    assert set(recs[0].over.columns) >= {"p", "source", "sink", "d", "t"}
