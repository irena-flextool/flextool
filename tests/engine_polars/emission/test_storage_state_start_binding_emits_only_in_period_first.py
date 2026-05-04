"""Tier-7 #27: ``storage_state_start_binding`` only fires at first-step
of each period for storage nodes flagged ``fix_start``.

Fixture: ``work_wind_battery``.  The .mod constrains
``v_state[n, d, t_first] · unitsize  ==  state_start[n] · existing[n, d]``
exactly once per ``(n, d)`` where ``d ∈ period_first`` and the node is
in ``storage_fix_start``.

We assert the LP row count equals ``data.nodeState_first_dt`` filtered
to ``storage_fix_start`` — i.e. the constraint never leaks onto an
interior timestep.  A regression that emitted the binding for every
(d, t) (instead of only the period-first step) would fail loudly here.
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
def test_storage_state_start_binding_emits_only_in_period_first() -> None:
    pb, data = build(DATA_DIR / "work_wind_battery")

    assert data.storage_fix_start is not None and data.storage_fix_start.height > 0, (
        "fixture invariant: storage_fix_start must be non-empty"
    )
    assert data.nodeState_first_dt is not None, (
        "fixture invariant: nodeState_first_dt must be present"
    )

    # The binding domain is nodeState_first_dt ∩ storage_fix_start (on n).
    # In this fixture every storage node is fix_start and every state
    # node has one period_first row, so the join is non-narrowing.
    expected = (
        data.nodeState_first_dt
        .join(data.storage_fix_start, on="n", how="inner")
        .height
    )
    assert expected > 0, "fixture invariant: at least one fix_start row"

    assert_cstr_row_count(pb, "storage_state_start_binding", expected)
    assert_cstr_present(pb, "storage_state_start_binding")

    # Sanity: the constraint's `over` columns include both n and the
    # period-time index — so the constraint is not collapsed onto a
    # plain (n,) or (n, d) shape that would silently skip rows.
    recs = pb.cstrs_named("storage_state_start_binding")
    assert len(recs) == 1
    over_cols = set(recs[0].over.columns)
    assert {"n", "d", "t"}.issubset(over_cols), (
        f"storage_state_start_binding `over` should carry (n, d, t); "
        f"got {sorted(over_cols)}"
    )
