"""Per-class param taxonomies — enum subsets from flextool_base.dat.

Mirror of the ``set X := a, b, c;`` definitions in
flextool/flextool_base.dat lines 143-180. Used by the ``*_in_use`` set
preprocessing in `process_arc_unions.py` (and similar) to decide
which (entity, paramName) pairs are required, optional-invest-related,
or otherwise active.

Update both sites in lockstep if base.dat changes.
"""
from __future__ import annotations


# flextool_base.dat:148
PROCESS_PERIOD_PARAM_REQUIRED: frozenset[str] = frozenset((
    "fixed_cost", "other_operational_cost", "lifetime", "existing",
))

# flextool_base.dat:149-152
PROCESS_PERIOD_PARAM_INVEST: frozenset[str] = frozenset((
    "discount_rate", "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period",
))

# flextool_base.dat:144-147 (processPeriodParam — full)
PROCESS_PERIOD_PARAM: frozenset[str] = frozenset((
    "fixed_cost", "other_operational_cost", "lifetime", "existing",
    "discount_rate", "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period", "startup_cost",
))

# flextool_base.dat:153
PROCESS_TIME_PARAM: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load",
    "other_operational_cost", "availability",
))

# flextool_base.dat:154
PROCESS_TIME_PARAM_REQUIRED: frozenset[str] = frozenset((
    "efficiency", "other_operational_cost", "availability",
))

# flextool_base.dat:158-159 (sourceSinkTimeParam[Required])
SOURCE_SINK_TIME_PARAM: frozenset[str] = frozenset((
    "efficiency", "efficiency_at_min_load", "min_load", "other_operational_cost",
))

SOURCE_SINK_TIME_PARAM_REQUIRED: frozenset[str] = frozenset((
    "efficiency", "other_operational_cost",
))

# flextool_base.dat:160-161 (sourceSinkPeriodParam[Required])
SOURCE_SINK_PERIOD_PARAM: frozenset[str] = SOURCE_SINK_TIME_PARAM
SOURCE_SINK_PERIOD_PARAM_REQUIRED: frozenset[str] = SOURCE_SINK_TIME_PARAM_REQUIRED

# flextool_base.dat:168-172 (nodePeriodParam — full)
NODE_PERIOD_PARAM: frozenset[str] = frozenset((
    "annual_flow", "peak_inflow", "fixed_cost", "discount_rate",
    "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period", "lifetime",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period",
    "virtual_unitsize",
    "storage_state_reference_price", "existing", "penalty_up", "penalty_down",
))

# flextool_base.dat:173
NODE_PERIOD_PARAM_REQUIRED: frozenset[str] = frozenset((
    "annual_flow", "peak_inflow", "fixed_cost", "lifetime",
    "storage_state_reference_price", "existing",
    "penalty_up", "penalty_down",
))

# flextool_base.dat:174-177
NODE_PERIOD_PARAM_INVEST: frozenset[str] = frozenset((
    "discount_rate", "invest_cost", "salvage_value",
    "invest_max_period", "invest_min_period",
    "cumulative_max_capacity", "cumulative_min_capacity",
    "retire_forced", "retire_max_period", "retire_min_period",
    "virtual_unitsize",
))

# flextool_base.dat:178
NODE_TIME_PARAM: frozenset[str] = frozenset((
    "inflow", "penalty_down", "penalty_up", "self_discharge_loss",
    "availability", "storage_state_reference_value",
))

# flextool_base.dat:179
NODE_TIME_PARAM_REQUIRED: frozenset[str] = frozenset((
    "inflow", "penalty_down", "penalty_up",
))
