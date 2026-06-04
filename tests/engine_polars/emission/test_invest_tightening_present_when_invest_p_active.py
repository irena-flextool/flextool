"""Tier-7 #29: ``maxFlow`` carries an ``invest`` LHS branch when
``v_invest_p`` is active.

Fixture: ``work_wind_battery_invest`` (process-side investments active
— ``data.pd_invest_set`` non-empty).  The proposal originally pointed
at ``work_coal_unit_size_MIP_wind`` as a multi-solve invest fixture, but
that fixture has ``pd_invest_set is None`` at first-solve build time
(invest carryover happens between solves, not within the initial build).
We use ``work_wind_battery_invest`` instead — invest is active in the
single-solve build — and document the swap in ``audit/phase3_notes.md``.

What we verify:
  * ``maxFlow`` is emitted once per ``(p, source, sink, d, t)``;
  * the LHS expression carries **two terms** (``v_flow`` and
    ``invest_neg``) instead of just one — exposing the invest-tightening
    branch the .mod attaches when invest_p is active;
  * ``pd_invest_set`` is non-empty (invest is in fact active in the LP).
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
def test_invest_tightening_present_when_invest_p_active(scenario_workdir)-> None:
    pb, data = build(scenario_workdir("wind_battery_invest"))

    # Invest is active in this fixture.
    assert data.pd_invest_set is not None and data.pd_invest_set.height > 0, (
        "fixture invariant: pd_invest_set must be non-empty for this test"
    )

    # Row count matches pss_dt — same LHS shape as the no-invest case;
    # the *additional* LHS term is what exposes invest tightening.
    assert_cstr_row_count(pb, "maxFlow", compute_pss_dt(data).height)
    assert_cstr_present(pb, "maxFlow")

    # Inspect the LHS proto: we should see *more than one* additive
    # variable term — the second is the invest tightening (``invest_neg``).
    recs = pb.cstrs_named("maxFlow")
    assert len(recs) == 1
    proto = recs[0].proto
    n_terms = len(proto.expr.terms)
    assert n_terms >= 2, (
        f"maxFlow LHS should carry v_flow + invest tightening "
        f"(>=2 terms); got {n_terms}.  This regression would silently "
        f"loosen the capacity bound when v_invest_p > 0."
    )

    # Cross-check: at least one of those terms is keyed on (p, d) without
    # source/sink — that's the invest sum collapsed over (d_invest).
    invest_term_dims = [tuple(t.dims) for t in proto.expr.terms]
    has_invest_shape = any(
        ("p" in dims and "source" not in dims and "sink" not in dims)
        for dims in invest_term_dims
    )
    assert has_invest_shape, (
        f"no LHS term has invest-style dims (p, d); got {invest_term_dims}"
    )
