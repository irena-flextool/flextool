"""Tier-7 #28: ``maxFlow`` is emitted on the full ``pss_dt`` cross.

Fixture: ``work_coal``.  The capacity bound on every flow:
``v_flow[p, source, sink, d, t]  ≤  cap[p, d]`` should produce exactly
one LP row per ``(p, source, sink, d, t)`` tuple in ``data.pss_dt``.

A regression that broadened or narrowed the constraint domain (e.g.
forgetting to cross with ``dt``, or accidentally filtering via an
unrelated set) would change this count and trip the assertion.
"""

from __future__ import annotations

import pytest

from flextool.engine_polars._pdt_join import compute_pss_dt
from tests.engine_polars.emission._helpers import (
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_maxFlow_emits_one_row_per_pss_dt(scenario_workdir)-> None:
    pb, data = build(scenario_workdir("coal"))

    pss_dt = compute_pss_dt(data)
    assert pss_dt is not None and pss_dt.height > 0, (
        "fixture invariant: pss_dt must be non-empty"
    )

    assert_cstr_row_count(pb, "maxFlow", pss_dt.height)
    assert_cstr_present(pb, "maxFlow")

    # The constraint should be a single record (no _linear/_integer split).
    recs = pb.cstrs_named("maxFlow")
    assert len(recs) == 1 and recs[0].name == "maxFlow"
    # `over` should carry the full (p, source, sink, d, t) axis.
    assert set(recs[0].over.columns) >= {"p", "source", "sink", "d", "t"}
