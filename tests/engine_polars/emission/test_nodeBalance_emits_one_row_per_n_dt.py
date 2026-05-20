"""Smoke (Tier-7): ``nodeBalance_eq`` is emitted on the full
``nodeBalance × dt`` cross.

Fixture: ``work_coal`` (1 nodeBalance node × 48 (d, t) steps = 48 rows).

The basic flow-conservation constraint is the foundation under every
flextool scenario.  A regression that drops the constraint family
entirely, or that emits it on the wrong shape (e.g. ``(n, d)`` instead
of ``(n, d, t)`` after a state-change refactor), would silently let the
LP find any feasible flow assignment and break parity in subtle ways.
This is the canonical sanity check for that family.

Goal: <100 ms; build-only (no LP solve).
"""

from __future__ import annotations

import pytest

from flextool.engine_polars._pdt_join import compute_nodeBalance_dt
from tests.engine_polars.conftest import DATA_DIR
from tests.engine_polars.emission._helpers import (
    assert_cstr_present,
    assert_cstr_row_count,
    build,
)


@pytest.mark.smoke
@pytest.mark.emission
def test_nodeBalance_emits_one_row_per_n_dt() -> None:
    pb, data = build(DATA_DIR / "work_coal")

    assert data.nodeBalance is not None and data.nodeBalance.height > 0, (
        "fixture invariant: nodeBalance must be non-empty"
    )
    assert data.dt is not None

    # nodeBalance_dt = nodeBalance × dt; the constraint is one row per (n, d, t).
    # Phase E.3: built on demand via the helper instead of pre-materialised.
    nb_dt = compute_nodeBalance_dt(data)
    expected = data.nodeBalance.height * data.dt.height
    assert nb_dt is not None and nb_dt.height == expected, (
        f"helper invariant: compute_nodeBalance_dt height should equal "
        f"|nodeBalance| × |dt|; got "
        f"{nb_dt.height if nb_dt is not None else 'None'} != {expected}"
    )

    assert_cstr_row_count(pb, "nodeBalance_eq", expected)
    assert_cstr_present(pb, "nodeBalance_eq")

    recs = pb.cstrs_named("nodeBalance_eq")
    assert len(recs) == 1 and recs[0].name == "nodeBalance_eq"
    assert set(recs[0].over.columns) >= {"n", "d", "t"}, (
        f"nodeBalance_eq `over` should carry (n, d, t); got "
        f"{recs[0].over.columns}"
    )
