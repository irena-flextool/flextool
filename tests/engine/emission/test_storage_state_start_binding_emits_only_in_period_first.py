"""Tier-7 #27: the fix_start pin of a ``bind_within_timeblock`` node fires once per first step, at the step that
first step starts from.

Fixture: ``work_wind_battery`` (battery bind_within_timeblock + fix_start).  The first step's state change is
``v_state[t_first] − v_state[t_wrap]`` with ``t_wrap`` its ``t_previous_within_timeset`` (the last step of the first
timeblock), so the start state is pinned there:
``v_state[n, d, t_wrap] · unitsize  ==  state_start[n] · existing[n, d]``, in
``storage_state_start_binding_cyclic_period``, exactly once per ``(n, d, t_first)`` of ``nodeState_first_dt`` ∩
``storage_fix_start``.  The plain ``storage_state_start_binding`` (representative-period nodes only) has no row.
A regression that emitted the binding for every (d, t), or on the first step itself, fails here.
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.engine.emission._helpers import (
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_storage_state_start_binding_emits_only_in_period_first(scenario_run)-> None:
    pb, data = build(scenario_run("wind_battery"))

    assert data.storage_fix_start is not None and data.storage_fix_start.height > 0, (
        "fixture invariant: storage_fix_start must be non-empty"
    )
    assert data.nodeState_first_dt is not None, (
        "fixture invariant: nodeState_first_dt must be present"
    )

    # One pin per (n, d, t_first) of nodeState_first_dt ∩ storage_fix_start, at the wrap target of t_first.
    first = data.nodeState_first_dt.join(data.storage_fix_start, on="n", how="inner")
    expected = first.height
    assert expected > 0, "fixture invariant: at least one fix_start row"
    wrap = (first.join(data.dtttdt, on=["d", "t"], how="inner")
            .select("n", "d", pl.col("t_previous_within_timeset").alias("t")))
    assert wrap.height == expected

    assert_cstr_row_count(pb, "storage_state_start_binding_cyclic_period", expected)
    assert_cstr_present(pb, "storage_state_start_binding_cyclic_period")
    assert not [r for r in pb.cstrs_named("storage_state_start_binding") if r.name == "storage_state_start_binding"]

    # The rows sit on the wrap targets, with (n, d, t) columns (not collapsed to (n,) or (n, d)).
    recs = [r for r in pb.cstrs_named("storage_state_start_binding_cyclic_period")
            if r.name == "storage_state_start_binding_cyclic_period"]
    assert len(recs) == 1
    over = recs[0].over
    assert {"n", "d", "t"}.issubset(set(over.columns)), sorted(over.columns)
    as_str = [pl.col(c).cast(pl.Utf8) for c in ("n", "d", "t")]
    assert sorted(over.select(as_str).rows()) == sorted(wrap.select(as_str).rows())
