"""Surface A.2 (Node Capacity & Balance Topology) loader tests.

Entry point: ``load_flextool`` reached top-down (``_load_node`` is not
called directly).  Each test overlays one input on the smallest
dispatch-only seed (``base``) and asserts the resulting ``FlexData``
field literally.  The node-inflow tests load through the production call:
the overlay goes into a clone of the ``base`` run's provider (``tiny_run``
+ ``put_frame``) and ``load_scenario`` loads it with the DB reader.

The A.1 step-duration, period-share and timestep-weight tests and the A.2
penalty test check reload-only seeds (with a DB reader ``apply_derived_a``
re-derives those fields on every load); they are in
``tests/csv_reload/test_a01_a02_temporal_node.py`` (c24 S1-8).
"""
from __future__ import annotations

import pytest

from tests._support.load import load_scenario

from .conftest import put_frame


# --- A.2 ------------------------------------------------------------

def test_inflow_long_new_format_with_null_fill(tiny_run):
    """Covers A2-inflow_long_format_new (direct) +
    A2-inflow_null_fill_zero (direct).  Long ``pdtNodeInflow.csv`` with
    one explicit null exercises ``_read_wide_per_entity``'s long branch
    plus ``fill_null(0.0)``.

    Production path: ``p_inflow`` is the ``solve_data/pdtNodeInflow`` seed
    on every load (``apply_derived_a`` step 6 keeps it), so the overlay is
    what the model sees.  dt comes from the DB timeline (48 steps of
    ``base``); the overlay covers two of them.
    """
    run, P = tiny_run
    put_frame(P, "pdtNodeInflow.csv", [
        {"node": "west", "period": "p2020", "time": "t0001", "value": 42.0},
        {"node": "west", "period": "p2020", "time": "t0002", "value": None},
    ])
    data = load_scenario(run, provider=P)
    inf = data.p_inflow.frame.sort(["n", "d", "t"])
    # Hand-calc: 42.0 stays, null -> 0.0.
    assert inf["value"].to_list() == pytest.approx([42.0, 0.0], rel=1e-7)
    assert inf["n"].to_list() == ["west", "west"]
    assert inf["t"].to_list() == ["t0001", "t0002"]
    # The overlay is consumed: the un-overlaid base inflow is 48 rows starting -589.0, -537.0.
    base = run.last_flex_data.p_inflow.frame.sort(["n", "d", "t"])
    assert base.height == 48
    assert base["value"].head(2).to_list() != inf["value"].to_list()


def test_inflow_legacy_wide_format_unpivot(tiny_run):
    """Covers A2-inflow_long_format_legacy_wide (direct).  Legacy
    ``solve``-prefixed wide layout (one column per entity) goes through
    the ``unpivot`` branch in ``_read_wide_per_entity``.  Two timesteps
    times one node -> 2 long rows.

    Production path, as the long-format test above.
    """
    run, P = tiny_run
    put_frame(P, "pdtNodeInflow.csv", [
        {"solve": "s1", "period": "p2020", "time": "t0001", "west": 100.0},
        {"solve": "s1", "period": "p2020", "time": "t0002", "west": 200.0},
    ])
    data = load_scenario(run, provider=P)
    inf = data.p_inflow.frame.sort(["n", "d", "t"])
    # Hand-calc: unpivot of the two-row wide CSV preserves order.
    assert inf.height == 2
    assert inf["value"].to_list() == pytest.approx([100.0, 200.0], rel=1e-7)
    assert inf["n"].to_list() == ["west", "west"]
    # The overlay is consumed: the un-overlaid base inflow has 48 rows.
    assert run.last_flex_data.p_inflow.frame.height == 48


