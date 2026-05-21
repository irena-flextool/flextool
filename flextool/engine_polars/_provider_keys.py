"""Canonical Provider key constants.

Centralises the parent-qualified Provider keys
(``"solve_data/<basename>"``, ``"input/<basename>"``, etc.) used by
emit_* writers, cascade readers, and orchestration code.

Why this module exists
----------------------

Phase 0a of ``specs/provider_consolidation.md`` retired the dual-key
registration in :func:`_emit_provider_io._emit`.  Every frame lives
under exactly one Provider key — the parent-qualified form.  Stringly-
typed lookups (``provider.get("solve_data/active_timelines")``) are
typo-prone: a rename in one site without the other becomes a silent
``None`` return that the cascade tolerates by emitting an empty frame,
which then surfaces as wrong numbers downstream.

This module defines named Python constants for every key the cascade
emits or consumes.  Producers do ``provider.put(K.SOLVE_DATA_ACTIVE_TIMELINES, df)``;
consumers do ``provider.get(K.SOLVE_DATA_ACTIVE_TIMELINES)``.  IDE
rename / Find-Usages stays sound; a typo is a ``NameError`` at import
time, not a silent runtime miss.

Naming convention
-----------------

* The canonical form is **``"<parent>/<basename>"``** — no ``.csv``
  suffix.  Phase 3b retired the ``.csv`` suffix from key names: it
  reinforced a misleading "this is a file" mental model, even though
  the Provider stores frames in memory and strips ``.csv`` at lookup
  time anyway.  ``snapshot_processed_inputs`` adds the ``.csv``
  suffix on disk dump.
* Constant names are uppercased, underscore-separated, prefixed with
  the parent: ``SOLVE_DATA_ACTIVE_TIMELINES``, ``INPUT_TIMELINE``,
  ``DERIVED_CT_METHOD_OVERRIDES``, ``HANDOFF_REALIZED_INVEST``.
* Group constants by parent dir, then alphabetically within the group.
* New keys: add the constant here first, then use it from the call sites.

Coverage scope
--------------

This module currently covers the **cross-solve carrier set**, the
**handoff carriers** (Phase 2 translator), and other high-traffic keys
touched by orchestration and the ``_solve_context`` loaders.  The
single-use ``solve_data/*`` emit-and-forget keys (most of
:func:`_flex_data_accumulator.expected_basenames`) still appear as
inline string literals at their producer call sites — migrating them
to constants is a mechanical follow-up that can land in batches
without changing semantics.
"""
from __future__ import annotations


SOLVE_DATA_FIX_STORAGE_QUANTITY = "solve_data/fix_storage_quantity"
SOLVE_DATA_FIX_STORAGE_PRICE = "solve_data/fix_storage_price"
SOLVE_DATA_FIX_STORAGE_USAGE = "solve_data/fix_storage_usage"
SOLVE_DATA_P_ENTITY_PRE_EXISTING = "solve_data/p_entity_pre_existing"
SOLVE_DATA_P_ENTITY_DIVEST_CUMULATIVE_MAX = (
    "solve_data/p_entity_divest_cumulative_max"
)
SOLVE_DATA_P_ENTITY_INVESTED = "solve_data/p_entity_invested"
SOLVE_DATA_P_ENTITY_DIVESTED = "solve_data/p_entity_divested"
SOLVE_DATA_P_ENTITY_PERIOD_EXISTING_CAPACITY = (
    "solve_data/p_entity_period_existing_capacity"
)
SOLVE_DATA_P_ROLL_CONTINUE_STATE = "solve_data/p_roll_continue_state"
SOLVE_DATA_CO2_CUM_REALIZED_TONNES = "solve_data/co2_cum_realized_tonnes"
SOLVE_DATA_LADDER_CUM_SIM_HOURS = "solve_data/ladder_cum_sim_hours"
SOLVE_DATA_LADDER_CUM_REALIZED_MWH = "solve_data/ladder_cum_realized_mwh"
SOLVE_DATA_ED_HISTORY_REALIZED = "solve_data/ed_history_realized"
SOLVE_DATA_ED_HISTORY_REALIZED_FIRST = "solve_data/ed_history_realized_first"
SOLVE_DATA_EDD_HISTORY = "solve_data/edd_history"


# ---------------------------------------------------------------------------
# Frequently-referenced solve_data/ keys — touched by multiple cascade
# modules.  Add to this section as call sites accumulate.
# ---------------------------------------------------------------------------

SOLVE_DATA_ACTIVE_TIMELINES = "solve_data/active_timelines"
SOLVE_DATA_INVEST_PERIODS_OF_CURRENT_SOLVE = (
    "solve_data/invest_periods_of_current_solve"
)
SOLVE_DATA_PROCESS_SOURCE_TOSINK = "solve_data/process_source_toSink"
SOLVE_DATA_REGION_COUPLING = "solve_data/region_coupling"


# ---------------------------------------------------------------------------
# input/ keys — populated by ``seed_provider_from_dir(kind="input")`` and
# the input_derivation modules.  Subset listed; expand as call sites grow.
# ---------------------------------------------------------------------------

INPUT_COMMODITY_LADDER_ANNUAL = "input/commodity_ladder_annual"
INPUT_COMMODITY_LADDER_CUMULATIVE = "input/commodity_ladder_cumulative"
INPUT_CONNECTION_DC_POWER_FLOW = "input/connection_dc_power_flow"
INPUT_GROUP__CO2_METHOD = "input/group__co2_method"
INPUT_NODE_DC_POWER_FLOW = "input/node_dc_power_flow"
INPUT_NODE_REFERENCE_ANGLE = "input/node_reference_angle"
INPUT_P_CONNECTION_SUSCEPTANCE = "input/p_connection_susceptance"
INPUT_PERIODS_AVAILABLE = "input/periods_available"
INPUT_PROCESS_METHOD = "input/process_method"


# ---------------------------------------------------------------------------
# derived/ keys — populated by input_derivation between Spine read and
# cascade build.  Distinct from input/ so derivation intermediates don't
# collide with raw user input.
# ---------------------------------------------------------------------------

DERIVED_CT_METHOD_OVERRIDES = "derived/ct_method_overrides"


# ---------------------------------------------------------------------------
# handoff/ keys — populated by ``_provider_translators.translate_handoff_to_provider``
# at iteration start, from the previous sub-solve's ``SolveHandoff``.
# One key per consumed handoff field; empty header-only frame written
# when the corresponding handoff field is ``None`` (first sub-solve, or
# carrier not active for the previous solve).  Consumers should treat
# ``provider.get(K.HANDOFF_X).height == 0`` as the "no prior carrier"
# signal instead of checking the SolveHandoff object directly.
# ---------------------------------------------------------------------------

HANDOFF_REALIZED_INVEST = "handoff/realized_invest"
HANDOFF_REALIZED_EXISTING = "handoff/realized_existing"
HANDOFF_DIVEST_CUMULATIVE = "handoff/divest_cumulative"
HANDOFF_ROLL_END_STATE = "handoff/roll_end_state"
HANDOFF_CUMULATIVE_CO2 = "handoff/cumulative_co2"
HANDOFF_CUMULATIVE_COMMODITY = "handoff/cumulative_commodity"
HANDOFF_CUM_SIM_HOURS = "handoff/cum_sim_hours"
HANDOFF_FIX_STORAGE_QUANTITY = "handoff/fix_storage_quantity"
HANDOFF_FIX_STORAGE_PRICE = "handoff/fix_storage_price"
HANDOFF_FIX_STORAGE_USAGE = "handoff/fix_storage_usage"
