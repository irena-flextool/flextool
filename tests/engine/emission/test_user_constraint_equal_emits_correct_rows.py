"""Smoke (Tier-7): ``process_constraint_equal`` is emitted on the right
``(c, d, t)`` set, and the LE / GE variants are absent on a fixture
that only declares ``equal`` constraints.

Fixture: ``work_coal_chp`` — defines one user constraint
``coal_chp_fix`` with sense ``equal`` (forces the CHP unit's heat /
power flow ratio).  Other fixtures exercise the LE / GE variants.

We assert:
  * row count = the ``data.cdt_eq`` rows of the constraints with a flow
    term (``flow_constraint_idx``); the fixture's other ``equal``
    constraints have no term in this solve (their entities are not in
    it), so they are capacity-only constraints and get no per-step rows;
  * the GE / LE variants are *absent* (their cdt_*  sets are empty);
  * the constraint family is present;
  * the constraint sense is ``==``.

A regression that broadens the domain or that drops the user constraint
machinery entirely would silently free the CHP heat / power coupling
and break parity.

Goal: <100 ms; build-only.
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.engine.emission._helpers import (
    assert_cstr_absent,
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_user_constraint_equal_emits_correct_rows(scenario_run)-> None:
    pb, data = build(scenario_run("coal_chp"))

    assert data.cdt_eq is not None and data.cdt_eq.height > 0, (
        "fixture invariant: cdt_eq must be non-empty"
    )

    flow_cns = data.flow_constraint_idx.get_column("cn").cast(pl.Utf8).unique().to_list()
    assert "coal_chp_fix" in flow_cns
    per_step = data.cdt_eq.filter(pl.col("cn").cast(pl.Utf8).is_in(flow_cns)).height
    assert 0 < per_step < data.cdt_eq.height
    assert_cstr_row_count(pb, "process_constraint_equal", per_step)
    assert_cstr_present(pb, "process_constraint_equal")

    # LE / GE variants are not wired in this fixture (cdt_le / cdt_ge None).
    assert_cstr_absent(pb, "process_constraint_less_than")
    assert_cstr_absent(pb, "process_constraint_greater_than")

    recs = pb.cstrs_named("process_constraint_equal")
    assert len(recs) == 1
    assert "==" == recs[0].proto.sense, (
        f"process_constraint_equal must use sense '=='; "
        f"got {recs[0].proto.sense!r}"
    )
    assert set(recs[0].over.columns) >= {"cn", "d", "t"}, (
        f"process_constraint_equal `over` should carry (cn, d, t); "
        f"got {recs[0].over.columns}"
    )
