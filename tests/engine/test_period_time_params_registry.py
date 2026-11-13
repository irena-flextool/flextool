"""Pin test for ``PERIOD_TIME_PARAMS`` (spec fix_map_reading.md §3.1 / STEP 1).

``PERIOD_TIME_PARAMS`` is derived from ``PARAM_ALLOWED_SHAPES`` so the two
cannot drift.  This test pins the derivation RULE, the exact count, and
the membership — in the style of
``tests/engine/autoscale/test_registry_coverage.py`` — so a future
registry edit that silently changes which parameters get per-row
period/time placement fails loudly.
"""
from __future__ import annotations

from flextool.engine.input._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    PERIOD_TIME_PARAMS,
    Shape,
)

# The three EXACT Shape members that define membership (NOT substring
# matching — e.g. MAP_PERIOD_TIER_FACET must be EXCLUDED).
_PERIOD_TIME_SHAPES = {Shape.MAP_PERIOD, Shape.MAP_TIME, Shape.MAP_PERIOD_TIME}


def test_derivation_rule_matches_registry():
    expected = {
        key for key, shapes in PARAM_ALLOWED_SHAPES.items()
        if _PERIOD_TIME_SHAPES & set(shapes)
    }
    assert set(PERIOD_TIME_PARAMS) == expected


def test_count_is_66_split_22_dual_1_time_only_42_period_only_1_period_map_only():
    assert len(PERIOD_TIME_PARAMS) == 66
    # 22 dual-axis: 21 with the full 2-D shape (incl. node.energy_margin_adder, which has no time-only map) +
    # commodity.price (two 1-D).
    dual_2d = {k for k, v in PARAM_ALLOWED_SHAPES.items()
               if Shape.MAP_PERIOD_TIME in v}
    assert len(dual_2d) == 21
    assert PARAM_ALLOWED_SHAPES[("node", "energy_margin_adder")] == {
        Shape.SCALAR, Shape.MAP_PERIOD, Shape.MAP_PERIOD_TIME}
    commodity_price = {k for k, v in PARAM_ALLOWED_SHAPES.items()
                       if Shape.MAP_TIME in v and Shape.MAP_PERIOD_TIME not in v
                       and Shape.MAP_PERIOD in v}
    assert commodity_price == {("commodity", "price")}
    # 1 time-only: the per-timestep weight map of a timeset.
    time_only = {k for k, v in PARAM_ALLOWED_SHAPES.items() if set(v) == {Shape.MAP_TIME}}
    assert time_only == {("timeset", "timeset_weights")}
    # 42 period-only (incl. group cumulative_{max,min}_capacity, capacity_margin_share, the user-constraint
    # constant and unit__inputNode.startup_energy): exactly {SCALAR, MAP_PERIOD}.
    period_only = {k for k, v in PARAM_ALLOWED_SHAPES.items()
                   if set(v) == {Shape.SCALAR, Shape.MAP_PERIOD}}
    assert len(period_only) == 42
    assert ("group", "capacity_margin_share") in period_only
    assert ("unit__inputNode", "startup_energy") in period_only
    # 1 period map only (no constant): the years each period of a solve represents.
    period_map_only = {k for k, v in PARAM_ALLOWED_SHAPES.items() if set(v) == {Shape.MAP_PERIOD}}
    assert period_map_only == {("solve", "years_represented")}
    assert set(PERIOD_TIME_PARAMS) == dual_2d | commodity_price | time_only | period_only | period_map_only


def test_facet_and_scalar_str_shapes_excluded():
    # Facet (price-ladder) + scalar-string (invest_method) keys must NOT
    # be routed through per-row period/time placement.
    excluded = set(PARAM_ALLOWED_SHAPES) - set(PERIOD_TIME_PARAMS)
    assert excluded == {
        ("commodity", "price_ladder_cumulative"),
        ("commodity", "price_ladder_annual"),
        ("node", "invest_method"),
        ("unit", "invest_method"),
        ("connection", "invest_method"),
    }
    for key in excluded:
        assert key not in PERIOD_TIME_PARAMS


def test_known_dual_axis_members_present():
    for key in [
        ("group", "co2_price"),
        ("flowGroup", "min_instant_flow"),
        ("flowGroup", "max_instant_flow"),
        ("node", "availability"),
        ("node", "storage_state_reference_value"),
        ("node", "self_discharge_loss"),
        ("node", "penalty_up"),
        ("node", "penalty_down"),
        ("unit", "availability"),
        ("connection", "availability"),
        ("unit", "efficiency"),
        ("connection", "efficiency"),
        ("unit", "efficiency_at_min_load"),
        ("unit", "min_load"),
        ("commodity", "price"),
        ("unit__inputNode", "other_operational_cost"),
        ("unit__outputNode", "other_operational_cost"),
        ("connection", "other_operational_cost"),
        ("reserve__upDown__group", "reservation"),
    ]:
        assert key in PERIOD_TIME_PARAMS

    # Period-only members are included on purpose (so a stray Map(time)
    # raises in the §3.3 check).
    for key in [
        ("group", "co2_max_period"),
        ("unit", "startup_cost"),
        ("connection", "startup_cost"),
        ("unit", "invest_max_period"),
        ("connection", "cumulative_min_capacity"),
        ("unit", "invest_cost"),
        ("node", "invest_cost"),
        ("connection", "invest_cost"),
        ("unit", "salvage_value"),
        ("node", "salvage_value"),
        ("connection", "salvage_value"),
    ]:
        assert key in PERIOD_TIME_PARAMS
