"""Smoke (Tier-7): ``maxCumulative_capacity_p`` and ``..._n`` are
emitted on the process- and node-side cumulative-invest entity / period
sets.

Fixture: ``work_network_coal_wind_battery_invest_cumulative`` — wires
the ``ed_invest_cumulative`` entity-period set used by
``_cumulative_invest._emit_cumulative_capacity``.  The .mod splits the
LP rows by entity kind (process / node), one row per
``(p, d) ∈ ed_invest_cumulative ∩ process_side`` and one per
``(n, d) ∈ ed_invest_cumulative ∩ node_side``.

We assert:
  * the per-side row counts add up to a non-zero total <=
    ``ed_invest_cumulative.height`` (the split discards entities that
    are neither processes nor storage nodes);
  * both the process- and node-side variants are present;
  * the sense is ``<=`` (max-cap variant).

A regression that drops the cumulative-cap path would let invests run
unbounded over time and break parity hard.

Goal: <100 ms; build-only.
"""

from __future__ import annotations

import pytest

from tests.engine_polars.emission._helpers import (
    assert_cstr_present,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_cumulative_capacity_emits_correct_rows(scenario_workdir)-> None:
    pb, data = build(
        scenario_workdir("network_coal_wind_battery_invest_cumulative")
    )

    assert data.ed_invest_cumulative is not None, (
        "fixture invariant: ed_invest_cumulative must be present"
    )
    assert data.ed_invest_cumulative.height > 0, (
        "fixture invariant: ed_invest_cumulative must be non-empty"
    )
    assert data.ed_cumulative_max_capacity is not None, (
        "fixture invariant: ed_cumulative_max_capacity must be present"
    )

    rows_p = pb.cstr_row_count("maxCumulative_capacity_p")
    rows_n = pb.cstr_row_count("maxCumulative_capacity_n")
    assert rows_p > 0, "maxCumulative_capacity_p must emit at least one row"
    assert rows_n > 0, "maxCumulative_capacity_n must emit at least one row"

    # Each emitted row must correspond to a real (e, d) entry — total
    # cannot exceed the index set.  Strict less-than is allowed because
    # the split discards entities that are not in any v_invest_p / _n.
    assert rows_p + rows_n <= data.ed_invest_cumulative.height, (
        f"row split inconsistent: rows_p={rows_p}, rows_n={rows_n}, "
        f"ed_invest_cumulative.height={data.ed_invest_cumulative.height}"
    )

    assert_cstr_present(pb, "maxCumulative_capacity_p")
    assert_cstr_present(pb, "maxCumulative_capacity_n")

    for name in ("maxCumulative_capacity_p", "maxCumulative_capacity_n"):
        recs = pb.cstrs_named(name)
        assert len(recs) == 1, (
            f"{name} should be a single record; got {len(recs)}"
        )
        assert "<=" == recs[0].proto.sense, (
            f"{name} must use sense '<='; got {recs[0].proto.sense!r}"
        )
