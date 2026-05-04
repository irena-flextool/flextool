"""Smoke (Tier-7): ``ramp_sink_up_constraint`` is emitted on the right
``(p, source, sink) × dt`` set.

Fixture: ``work_coal_ramp_limit`` — exercises sink-up ramp limits on
``coal_plant``.  The .mod runs the ramp constraint once per
``process_source_sink_ramp_limit_sink_up × dt`` tuple.  We assert the
LP row count = ``|process_source_sink_ramp_limit_sink_up| × |dt|`` for
each emitted ramp family that is wired by this fixture (only
``sink_up`` here; the other three ramp families have empty index sets).

A regression that broadened the domain (e.g. ramp on every (p, s, s)
not just the flagged ones) would inflate the row count.  A regression
that dropped the constraint emission entirely would leave coal_plant
free to ramp instantaneously and the obj would no longer match
flextool.

Goal: <100 ms; build-only.
"""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from tests.emission._helpers import (
    assert_cstr_absent,
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_ramp_sink_up_emits_correct_rows() -> None:
    pb, data = build(DATA_DIR / "work_coal_ramp_limit")

    idx = data.process_source_sink_ramp_limit_sink_up
    assert idx is not None and idx.height > 0, (
        "fixture invariant: process_source_sink_ramp_limit_sink_up "
        "must be non-empty"
    )
    assert data.dt is not None and data.dt.height > 0

    expected = idx.height * data.dt.height
    assert_cstr_row_count(pb, "ramp_sink_up_constraint", expected)
    assert_cstr_present(pb, "ramp_sink_up_constraint")

    # The other three ramp families have empty index sets in this fixture
    # (only sink_up is used) — assert they did NOT emit any rows so a
    # regression that fires them on the empty-but-present idx surfaces
    # here.
    assert_cstr_absent(pb, "ramp_sink_down_constraint")
    assert_cstr_absent(pb, "ramp_source_up_constraint")
    assert_cstr_absent(pb, "ramp_source_down_constraint")

    recs = pb.cstrs_named("ramp_sink_up_constraint")
    assert len(recs) == 1
    assert set(recs[0].over.columns) >= {"p", "source", "sink", "d", "t"}, (
        f"ramp_sink_up_constraint `over` should carry (p, source, sink, d, t); "
        f"got {recs[0].over.columns}"
    )
