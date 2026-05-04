"""Smoke (Tier-7): ``profile_flow_upper_limit`` is emitted on the right
``(p, source, sink, f) × dt`` set.

Fixture: ``work_wind`` — wind_plant has a ``wind_profile`` with method
``upper_limit``.  The .mod's profile_flow_upper constraint runs once
per ``(p, source, sink, profile) × dt`` tuple, capping the realised
flow by the profile.

We assert:
  * row count = ``|process_profile_upper| × |dt|``;
  * the constraint family is present;
  * the lower / fixed family variants are absent on this fixture (only
    upper is wired).

A regression that flips the sense to ``==`` or that drops the
constraint family entirely would silently let the wind_plant overshoot
its own profile, breaking parity.

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
def test_profile_flow_upper_emits_correct_rows() -> None:
    pb, data = build(DATA_DIR / "work_wind")

    idx = data.process_profile_upper
    assert idx is not None and idx.height > 0, (
        "fixture invariant: process_profile_upper must be non-empty"
    )
    assert data.dt is not None and data.dt.height > 0

    # process_profile_upper is keyed on (p, source, sink, f).  The
    # constraint is one row per (p, source, sink, f, d, t).
    expected = idx.height * data.dt.height
    assert_cstr_row_count(pb, "profile_flow_upper_limit", expected)
    assert_cstr_present(pb, "profile_flow_upper_limit")

    # Lower / fixed variants must NOT be emitted on this fixture.
    assert_cstr_absent(pb, "profile_flow_lower_limit")
    assert_cstr_absent(pb, "profile_flow_fixed")

    recs = pb.cstrs_named("profile_flow_upper_limit")
    assert len(recs) == 1
    assert "<=" == recs[0].proto.sense, (
        f"profile_flow_upper_limit must use sense '<='; "
        f"got {recs[0].proto.sense!r}"
    )
    assert set(recs[0].over.columns) >= {"p", "source", "sink", "d", "t"}, (
        f"profile_flow_upper_limit `over` should carry "
        f"(p, source, sink, d, t); got {recs[0].over.columns}"
    )
