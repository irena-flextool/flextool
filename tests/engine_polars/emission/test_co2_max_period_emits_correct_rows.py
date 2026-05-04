"""Smoke (Tier-7): ``co2_max_period`` is emitted once per priced-CO2
``(g, d)`` row.

Fixture: ``work_coal_co2_limit`` — applies a CO2 cap on the
``co2_limit`` group for every period.  The .mod's co2_max_period
constraint binds the per-period sum of CO2-capped flow against
``p_co2_max_period[g, d]``.

We assert:
  * the constraint emits exactly ``p_co2_max_period.frame.height``
    rows (one per (g, d) tuple);
  * the constraint family is present;
  * the ``flow_from_co2_capped`` set is non-empty (otherwise the LHS
    is identically zero and the cap is a no-op — would not catch
    regressions).

A regression that inverts the sense (``>=``) or that drops the
constraint family entirely would let the LP emit unlimited CO2 and
break parity.

Goal: <100 ms; build-only.
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
def test_co2_max_period_emits_correct_rows() -> None:
    pb, data = build(DATA_DIR / "work_coal_co2_limit")

    assert data.flow_from_co2_capped is not None and data.flow_from_co2_capped.height > 0, (
        "fixture invariant: flow_from_co2_capped must be non-empty (would "
        "be a no-op cap otherwise)"
    )
    assert data.p_co2_max_period is not None, (
        "fixture invariant: p_co2_max_period must be present"
    )
    assert data.p_co2_max_period.frame.height > 0, (
        "fixture invariant: p_co2_max_period must be non-empty"
    )

    expected = data.p_co2_max_period.frame.height
    assert_cstr_row_count(pb, "co2_max_period", expected)
    assert_cstr_present(pb, "co2_max_period")

    recs = pb.cstrs_named("co2_max_period")
    assert len(recs) == 1
    assert "<=" == recs[0].proto.sense, (
        f"co2_max_period must use sense '<='; got {recs[0].proto.sense!r}"
    )
    assert set(recs[0].over.columns) >= {"g", "d"}, (
        f"co2_max_period `over` should carry (g, d); "
        f"got {recs[0].over.columns}"
    )
