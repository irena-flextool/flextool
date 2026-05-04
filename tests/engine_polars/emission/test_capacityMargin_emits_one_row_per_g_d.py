"""Tier-7 #30: ``capacityMargin`` is emitted on the right group / time
domain.

Fixture: ``work_capacity_margin``.  The .mod's capacity-margin
constraint runs once per ``(g, d, t)`` — i.e. ``groupCapacityMargin ×
dt``, not just per ``(g, d)``.  The proposal text originally said
"one row per ``(g, d)``" / "= ``data.pdGroup_capacity_margin.height``"
but that is the row count of the **parameter** (one per period), not of
the constraint.  This test checks the emitted shape and the parameter
shape separately so a future refactor does not silently break either.

See ``audit/phase3_notes.md`` for the discrepancy note.
"""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from tests.emission._helpers import (
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_capacityMargin_emits_one_row_per_g_d() -> None:
    pb, data = build(DATA_DIR / "work_capacity_margin")

    assert data.groupCapacityMargin is not None, (
        "fixture invariant: groupCapacityMargin set must be present"
    )
    assert data.pdGroup_capacity_margin is not None, (
        "fixture invariant: pdGroup_capacity_margin must be present"
    )
    assert data.dt is not None and data.dt.height > 0

    # Constraint domain = groupCapacityMargin × dt.  The parameter
    # `pdGroup_capacity_margin` lives on (g, d) but the constraint
    # itself is bound per (g, d, t) so the per-step margin floor can
    # vary with inflow / step duration.
    expected = data.groupCapacityMargin.height * data.dt.height
    assert_cstr_row_count(pb, "capacityMargin", expected)
    assert_cstr_present(pb, "capacityMargin")

    recs = pb.cstrs_named("capacityMargin")
    assert len(recs) == 1
    # Sanity: `over` carries the (g, d, t) axis.
    assert set(recs[0].over.columns) >= {"g", "d", "t"}, (
        f"capacityMargin `over` should be keyed on (g, d, t); got "
        f"{recs[0].over.columns}"
    )

    # Cross-check the parameter shape is (g, d) so the audit-style
    # "per (g, d) entry" interpretation is also recorded — a regression
    # that doubled the parameter rows would surface here even though it
    # would not affect the constraint row count.
    assert data.pdGroup_capacity_margin.frame.height == (
        data.groupCapacityMargin.height
        * data.pdGroup_capacity_margin.frame.select("d").unique().height
    )
