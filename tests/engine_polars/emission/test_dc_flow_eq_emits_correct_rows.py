"""Tier-7: ``dc_flow_eq`` is emitted with the right shape on a DC PF fixture.

Fixture: ``work_dc_power_flow`` (PGLib case14 IEEE).  The constraint
family ``dc_flow_eq`` (.mod:3236) is::

    v_flow[p, source, sink, d, t] * unitsize[p]
      ==
    susceptance[p] * (v_angle[source, d, t] - v_angle[sink, d, t])

so the row count equals
``connection_dc_power_flow × process_source_toSink × dt`` filtered to
``source, sink ∈ node_dc_power_flow``.  For case14 this is 20 lines × 1
timestep = 20 rows.  We also verify that ``dc_reference_angle_eq`` pins
exactly the ``node_reference_angle`` set (1 row for case14, since the
fixture has a single connected component).
"""

from __future__ import annotations

import pytest

from tests.conftest import DATA_DIR
from tests.emission._helpers import (
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.emission
def test_dc_flow_eq_emits_one_row_per_dc_arc_dt() -> None:
    pb, data = build(DATA_DIR / "work_dc_power_flow")

    # Fixture invariants — the loader populated DC PF data.
    assert data.node_dc_power_flow is not None
    assert data.connection_dc_power_flow is not None
    assert data.node_reference_angle is not None
    assert data.p_connection_susceptance is not None
    n_dc_arcs = data.connection_dc_power_flow.height
    n_dt = data.dt.height
    assert n_dc_arcs > 0 and n_dt > 0

    # dc_flow_eq: one row per (DC arc) × (d, t) tuple.
    assert_cstr_row_count(pb, "dc_flow_eq", n_dc_arcs * n_dt)
    recs = pb.cstrs_named("dc_flow_eq")
    assert len(recs) == 1
    assert set(recs[0].over.columns) >= {"p", "source", "sink", "d", "t"}

    # Reference angle pin: one row per (ref_node, d, t).
    n_ref = data.node_reference_angle.height
    assert_cstr_row_count(pb, "dc_reference_angle_eq", n_ref * n_dt)
    assert_cstr_present(pb, "dc_reference_angle_eq")

    # v_angle variable was declared with the right column count
    # (n_dc_nodes × dt).
    v_angle = pb._vars.get("v_angle")
    assert v_angle is not None, "v_angle variable was not declared"
    assert v_angle.frame.height == data.node_dc_power_flow.height * n_dt

    # The symmetric back-flow capacity bound rounds out the picture for
    # method_2way_1var_off DC PF connections.
    assert_cstr_row_count(pb, "maxToSink_back", n_dc_arcs * n_dt)
