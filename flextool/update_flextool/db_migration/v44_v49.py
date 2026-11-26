"""Data versions 44-49: parameter groups and their colours, row scaling, 47-49."""
from spinedb_api import SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import (
    _commit_step,
    _upsert_definition,
    add_parameters_manual,
    add_value_list_manual,
)


def _migrate_v47(db) -> None:
    # Register "unidirectional" as a first-class connection
    # transfer_method.  It maps to method_1way_1var_off — a
    # single non-negative flow variable, no reverse-direction
    # variable.  This covers the common "one-way transmission"
    # use case that previously forced users to work around
    # with a unit instead of a connection.
    #
    # Schema touch points:
    #
    # 1. transfer_methods value list (attached to
    #    connection.transfer_method) — the per-connection
    #    dropdown must offer "unidirectional".
    # 2. transfer_methods_group value list (attached to
    #    group.transfer_method) — the group-level override
    #    must also accept "unidirectional".
    # 3. Description refreshes on both parameter definitions
    #    so the dropdown tooltip advertises the new option.
    add_value_list_manual(db, [
        ["transfer_methods", "unidirectional"],
        ["transfer_methods_group", "unidirectional"],
    ])
    db.add_update_item(
        "parameter_definition",
        entity_class_name="connection",
        name="transfer_method",
        description=(
            "Choice of transfer method. Options: regular (default), "
            "no_losses_no_variable_cost, variable_cost_only, exact, "
            "unidirectional. 'unidirectional' restricts flow to "
            "source → sink only (single non-negative variable); "
            "'regular'/'exact'/'variable_cost_only' are bidirectional "
            "two-variable variants; 'no_losses_no_variable_cost' is a "
            "single signed variable with no losses and no VOM."
        ),
    )
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="transfer_method",
        description=(
            "Override transfer_method for all connections within "
            "this nodeGroup. Options: use_connection_transfer_methods "
            "(default, no override), no_losses_no_variable_cost, "
            "regular, exact, variable_cost_only, unidirectional, "
            "dc_power_flow_with_angles. Setting 'unidirectional' "
            "gates every member connection to one-way flow (source "
            "→ sink only). Setting 'dc_power_flow_with_angles' "
            "uses B-theta DC power flow (requires reactance on "
            "connections)."
        ),
    )
    try:
        _commit_step(db,
            "Added 'unidirectional' to transfer_methods and "
            "transfer_methods_group value lists"
        )
    except SpineDBAPIError:
        pass


def _migrate_v48(db) -> None:
    # Restore the 0.05 default on entity-level discount_rate.
    # Migration v28 renamed interest_rate → discount_rate but
    # did not preserve the default, so DBs without an explicit
    # discount_rate value fell through to 0 and produced a
    # 0 / 0 divide inside the annuity factor
    # r / (1 − 1/(1+r)^n) in flextool.mod.
    default_val, default_type = to_database(0.05)
    for cls in ("connection", "node", "unit"):
        db.add_update_item(
            "parameter_definition",
            entity_class_name=cls,
            name="discount_rate",
            default_value=default_val,
            default_type=default_type,
        )
    try:
        _commit_step(db,
            "v48: set default_value=0.05 on "
            "connection/node/unit discount_rate "
            "(fixes 0/0 in annuity factor when unset)"
        )
    except SpineDBAPIError:
        pass


def _migrate_v49(db) -> None:
    # Enum defaults that were never set on their parameter
    # definitions.  Each is a method/sense whose absence puts
    # the entity into a silently-wrong state: a constraint
    # with no sense is never enforced, a node with no
    # inflow_method defaults back to use_original inside the
    # model anyway, and a reserve without reserve_method has
    # all its method-based branches turned off.  All values
    # are already in the corresponding parameter_value_list.
    for (cls, param, value) in (
        ("node", "inflow_method", "use_original"),
        ("constraint", "sense", "equal"),
        ("reserve__upDown__group", "reserve_method", "no_reserve"),
    ):
        dv, dt = to_database(value)
        db.add_update_item(
            "parameter_definition",
            entity_class_name=cls,
            name=param,
            default_value=dv,
            default_type=dt,
        )
    try:
        _commit_step(db,
            "v49: restore advisory enum defaults — "
            "node.inflow_method=use_original; "
            "constraint.sense=equal; "
            "reserve__upDown__group.reserve_method=no_reserve"
        )
    except SpineDBAPIError:
        pass


# ---------------------------------------------------------------------------
# v44: full parameter_group metadata across every parameter_definition.
# ---------------------------------------------------------------------------

# Group definitions: (name, color_6hex, priority).  See
# the parameter-groups proposal for the rationale behind the
# tiered priority scheme (asset physics → decision overlays → model plumbing).
_V44_PARAMETER_GROUPS: tuple[tuple[str, str, int], ...] = (
    ("basics",         "b3cde3", 10),
    ("investment",     "fdbf6f", 20),
    ("retirement",     "ffb870", 25),
    ("storage",        "cab2d6", 30),
    ("tech_advanced",  "b2df8a", 35),
    ("reserve",        "fb9a99", 40),
    ("emission",       "ccebc5", 45),
    ("network",        "80b1d3", 50),
    ("flow_limit",     "fccde5", 55),
    ("constraint",     "bc80bd", 70),
    ("model",          "d9d9d9", 80),
    ("solve_basics",   "bebada", 85),
    ("solve_advanced", "9f94c6", 87),
    ("timeline",       "ffed6f", 90),
    # "output" is kept with its existing colour (was the v43 "Outputs"
    # foothold); only its casing + priority change.
    ("output",         "a6cee3", 95),
)


def _v44_build_parameter_group_map() -> dict[tuple[str, str], str]:
    """Return a {(entity_class, parameter_name): group_name} dict.

    This is the data-driven membership table — one entry per
    parameter_definition row in the v43 master template.  See the
    proposal document for the rationale behind each assignment.
    """
    m: dict[tuple[str, str], str] = {}

    # --- basics ---------------------------------------------------------
    basics_map: dict[str, tuple[str, ...]] = {
        "commodity": (
            "price", "unitsize", "price_method",
            "price_ladder_annual", "price_ladder_cumulative",
        ),
        "connection": (
            "availability", "efficiency", "existing", "virtual_unitsize",
            "other_operational_cost", "transfer_method",
        ),
        "connection__profile": ("profile_method",),
        "node": (
            "availability", "existing", "virtual_unitsize", "annual_flow",
            "peak_inflow", "inflow", "inflow_method", "node_type",
            "penalty_up", "penalty_down",
        ),
        "node__profile": ("profile_method",),
        "profile": ("profile",),
        "unit": (
            "availability", "efficiency", "efficiency_at_min_load",
            "existing", "virtual_unitsize", "conversion_method", "min_load",
        ),
        "unit__inputNode": (
            "flow_coefficient", "max_capacity_coefficient",
            "min_capacity_coefficient", "other_operational_cost",
        ),
        "unit__outputNode": (
            "flow_coefficient", "max_capacity_coefficient",
            "min_capacity_coefficient", "other_operational_cost",
        ),
        "unit__node__profile": ("profile_method",),
    }
    for ec, params in basics_map.items():
        for p in params:
            m[(ec, p)] = "basics"

    # --- investment ----------------------------------------------------
    invest_shared = (
        "invest_cost", "invest_method", "invest_max_period",
        "invest_max_total", "invest_min_period", "invest_min_total",
        "cumulative_max_capacity", "cumulative_min_capacity",
        "lifetime", "lifetime_method", "discount_rate", "fixed_cost",
    )
    for ec in ("connection", "node", "unit"):
        for p in invest_shared:
            m[(ec, p)] = "investment"
    m[("node", "invest_forced")] = "investment"
    for p in (
        "invest_method", "invest_max_period", "invest_max_total",
        "invest_min_period", "invest_min_total",
        "cumulative_max_capacity", "cumulative_min_capacity",
        "capacity_margin", "has_capacity_margin",
        "penalty_capacity_margin",
    ):
        m[("group", p)] = "investment"

    # --- retirement ----------------------------------------------------
    for ec in ("connection", "node", "unit"):
        for p in (
            "retire_max_period", "retire_max_total",
            "retire_min_period", "retire_min_total", "salvage_value",
        ):
            m[(ec, p)] = "retirement"

    # --- storage -------------------------------------------------------
    for p in (
        "self_discharge_loss", "storage_binding_method",
        "storage_nested_fix_method", "storage_solve_horizon_method",
        "storage_start_end_method", "storage_state_start",
        "storage_state_end", "storage_state_reference_price",
        "storage_state_reference_value",
    ):
        m[("node", p)] = "storage"

    # --- tech_advanced -------------------------------------------------
    for ec in ("connection", "unit"):
        for p in ("startup_cost", "startup_method"):
            m[(ec, p)] = "tech_advanced"
    for p in ("min_uptime", "min_downtime", "minimum_time_method"):
        m[("unit", p)] = "tech_advanced"
    for ec in ("connection", "unit"):
        m[(ec, "delay")] = "tech_advanced"
    for ec in ("unit__inputNode", "unit__outputNode"):
        for p in (
            "ramp_cost", "ramp_method",
            "ramp_speed_up", "ramp_speed_down",
        ):
            m[(ec, p)] = "tech_advanced"

    # --- reserve -------------------------------------------------------
    for p in (
        "reservation", "reserve_method",
        "penalty_reserve", "increase_reserve_ratio",
    ):
        m[("reserve__upDown__group", p)] = "reserve"
    for ec in (
        "reserve__upDown__connection__node",
        "reserve__upDown__unit__node",
    ):
        for p in (
            "increase_reserve_ratio", "large_failure_ratio",
            "max_share", "reliability",
        ):
            m[(ec, p)] = "reserve"

    # --- emission ------------------------------------------------------
    m[("commodity", "co2_content")] = "emission"
    for p in ("co2_method", "co2_max_period", "co2_max_total", "co2_price"):
        m[("group", p)] = "emission"

    # --- network -------------------------------------------------------
    for p in ("is_DC", "reactance"):
        m[("connection", p)] = "network"
    for p in (
        "base_MVA", "reference_node",
        "candidate_precapacity_to_avoid_big_m", "transfer_method",
        "has_inertia", "inertia_limit", "penalty_inertia",
        "has_non_synchronous", "non_synchronous_limit",
        "penalty_non_synchronous",
    ):
        m[("group", p)] = "network"
    for ec in ("unit__inputNode", "unit__outputNode"):
        for p in ("is_non_synchronous", "inertia_constant"):
            m[(ec, p)] = "network"

    # --- flow_limit ----------------------------------------------------
    for p in (
        "max_cumulative_flow", "min_cumulative_flow",
        "max_instant_flow", "min_instant_flow", "share_loss_of_load",
    ):
        m[("group", p)] = "flow_limit"

    # --- constraint ----------------------------------------------------
    for p in ("constant", "sense"):
        m[("constraint", p)] = "constraint"
    for ec in ("connection__node", "unit__inputNode", "unit__outputNode"):
        m[(ec, "constraint_flow_coefficient")] = "constraint"
    for ec in ("connection", "node", "unit"):
        for p in (
            "constraint_invested_capacity_coefficient",
            "constraint_cumulative_pre_built_capacity_coefficient",
        ):
            m[(ec, p)] = "constraint"
    m[("node", "constraint_state_coefficient")] = "constraint"

    # --- model ---------------------------------------------------------
    for p in (
        "version", "solves", "periods_available", "inflation_rate",
        "inflation_offset_operations", "inflation_offset_investment",
        "max_flow_for_unconstrained_variables",
    ):
        m[("model", p)] = "model"
    m[("group", "include_stochastics")] = "solve_advanced"

    # --- solve_basics --------------------------------------------------
    for p in (
        "solver", "solve_mode", "period_timeset", "realized_periods",
        "invest_periods", "years_represented",
    ):
        m[("solve", p)] = "solve_basics"

    # --- solve_advanced -----------------------------------------------
    # timeline_hole_multiplier is created + tagged in v44 Step 4 (it
    # didn't exist in the schema before that step).  Listing it here too
    # would be a no-op since Step 3's loop pre-dates its existence.
    for p in (
        "solver_arguments", "solver_precommand", "highs_presolve",
        "highs_method", "highs_parallel", "rolling_duration",
        "rolling_solve_horizon", "rolling_solve_jump",
        "realized_invest_periods", "fix_storage_periods",
        "stochastic_branches", "contains_solves",
    ):
        m[("solve", p)] = "solve_advanced"

    # --- timeline ------------------------------------------------------
    m[("timeline", "timestep_duration")] = "timeline"
    for p in (
        "timeline", "timeset_duration", "new_stepduration",
    ):
        m[("timeset", p)] = "timeline"
    m[("timeset", "timeset_weights")] = "solve_advanced"

    # --- output --------------------------------------------------------
    for p in (
        "output_nodeGroup_dispatch", "output_nodeGroup_indicators",
        "output_flowGroup_indicators", "flow_aggregator",
    ):
        m[("group", p)] = "output"
    for p in (
        "debug", "exclude_entity_outputs", "output_horizon",
        "output_node_balance_t", "output_ramp_envelope",
        "output_unit__node_flow_t", "output_unit__node_ramp_t",
        "output_connection_flow_separate",
        "output_connection__node__node_flow_t",
    ):
        m[("model", p)] = "output"

    return m


def _migrate_v44_parameter_groups(db) -> None:
    """Apply full parameter_group metadata.

    Behaviour:
      1. Rename the existing ``"Outputs"`` parameter_group to ``"output"``
         (casing fix), keeping its colour ``a6cee3`` and bumping its
         priority to 95.  Rename cascades to every parameter_definition
         that references it (Spine tracks the link by id, not by name).
      2. Add the other 14 parameter_groups from
         :data:`_V44_PARAMETER_GROUPS`.
      3. Assign every parameter_definition to its group from the
         data-driven membership map built by
         :func:`_v44_build_parameter_group_map`.
      4. Define ``solve.timeline_hole_multiplier`` (float, default 1.0)
         and assign it to ``solve_advanced``.  The parameter has been
         consumed end-to-end by the writer chain
         (``input_writer._WRITE_ENTITY_PARAMETER_SPECS`` →
         ``solve_writers.write_hole_multiplier`` → ``solve_data/
         solve_hole_multiplier.csv`` → mod's ``p_hole_multiplier``)
         since well before v44, but was never declared in the schema —
         flextool relied on a Python-convention default of 1.0.  This
         step closes the gap so the schema is the single source of
         truth (matters for downstream consumers like the native engine
         that read defaults from ``parameter_definition.default_value``).
    """
    # Step 1: rename the existing Outputs group (or create it if it is
    # somehow missing) and update its priority.  Doing an id-keyed update
    # preserves the link to the four output-parameter_definitions that
    # v43 already tagged.  Use find_parameter_groups (plural) so a
    # re-iterated migration on a DB that already has "output" (post-v44
    # state, version rolled back) doesn't blow up on the lookup.
    outputs_groups = list(db.find_parameter_groups(name="Outputs"))
    outputs_group = outputs_groups[0] if outputs_groups else None
    if outputs_group is not None:
        db.update_item(
            "parameter_group",
            id=outputs_group["id"],
            name="output",
            color="a6cee3",
            priority=95,
        )
    else:
        db.add_update_item(
            "parameter_group",
            name="output",
            color="a6cee3",
            priority=95,
        )

    # Step 2: add the remaining 14 groups.  add_update_item is idempotent
    # on name, so re-running is safe.
    for name, color, priority in _V44_PARAMETER_GROUPS:
        if name == "output":
            continue  # handled in step 1
        db.add_update_item(
            "parameter_group",
            name=name,
            color=color,
            priority=priority,
        )

    # Step 3: assign every parameter_definition to its group.
    group_map = _v44_build_parameter_group_map()
    for (entity_class_name, param_name), group_name in group_map.items():
        _upsert_definition(db, 44, entity_class_name, param_name, parameter_group_name=group_name)

    # Step 4: declare solve.timeline_hole_multiplier (float, default 1.0)
    # and tag it to solve_advanced.  add_update_item is idempotent on
    # (entity_class_name, name) so re-running on a DB that already has
    # the parameter (e.g. one carried in by hand) is a no-op except for
    # back-filling missing fields (default, type, group).
    thm_default_val, thm_default_type = to_database(1.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="timeline_hole_multiplier",
        parameter_type_list=("float",),
        default_value=thm_default_val,
        default_type=thm_default_type,
        parameter_group_name="solve_advanced",
        description=(
            "[unitless] Multiplier applied to the inverse-step-duration "
            "term in nodeBalance_eq and storage-binding constraints "
            "across timeline gaps (holes).  Tunes how strongly state "
            "differences are penalised across discontinuities in the "
            "timeline.  Default 1.0 mirrors the .mod default and matches "
            "pre-v44 Python-convention behaviour."
        ),
    )

    try:
        _commit_step(db,
            "v44: renamed 'Outputs' parameter_group to 'output'; added 14 "
            "new parameter_groups (basics, investment, retirement, storage, "
            "tech_advanced, reserve, emission, network, flow_limit, "
            "constraint, model, solve_basics, solve_advanced, timeline); "
            "assigned every parameter_definition to its group; "
            "declared solve.timeline_hole_multiplier (float, default 1.0) "
            "in solve_advanced — closes the schema gap for a parameter "
            "the writer chain has consumed for some time via a "
            "Python-convention default."
        )
    except SpineDBAPIError:
        pass


# ---------------------------------------------------------------------------
# v45: recolour parameter_groups for light/dark-theme compatibility.
# ---------------------------------------------------------------------------
#
# The v44 palette used ColorBrewer pastels (luminance ~0.7-0.85) which wash
# out on a light IDE background and read as low-contrast on a dark one.
# v45 replaces them with mid-tone colours (relative luminance ~0.25-0.55)
# in three emotional registers:
#   * Calm cool tones for the 10 groups users normally reach for.
#   * Mild warm (soft salmon / gold) for specialised overlays that engage
#     optional physics or silently conflict with dispatch — network,
#     flow_limit.
#   * Stronger warm (amber / brick / rose) for groups where misuse can
#     break a solve — tech_advanced, solve_advanced, constraint.
#
# Names + priorities are unchanged; only the colour field is updated.
_V45_GROUP_COLORS: tuple[tuple[str, str], ...] = (
    # calm tier
    ("basics",         "6fa8c7"),  # sky blue — foundational
    ("investment",     "7fb095"),  # sage green
    ("retirement",     "a3b08c"),  # muted olive
    ("storage",        "a598c7"),  # lavender
    ("reserve",        "7aaeb0"),  # teal
    ("emission",       "8cbb8a"),  # leaf green
    ("model",          "a3a3a3"),  # neutral gray
    ("solve_basics",   "8c94b0"),  # slate
    ("timeline",       "c2b870"),  # muted gold
    ("output",         "9ac2d1"),  # cyan
    # mild warn tier
    ("network",        "d9a8a0"),  # soft salmon
    ("flow_limit",     "d9bf96"),  # soft amber
    # strong warn tier
    ("tech_advanced",  "d9925c"),  # amber-orange
    ("solve_advanced", "b56f6f"),  # brick red
    ("constraint",     "a36784"),  # dusky rose
)


def _migrate_v45_parameter_group_colors(db) -> None:
    """Update the colour on every parameter_group to the v45 palette.

    Does not touch names or priorities — both are stable from v44.  Does
    not touch parameter_definition group assignments — those already point
    by id and follow the renames/updates transparently.
    """
    for name, color in _V45_GROUP_COLORS:
        item = db.item(db.mapped_table("parameter_group"), name=name)
        if item is None:
            # Shouldn't happen if v44 ran, but don't crash if a group is
            # missing — just skip.
            continue
        db.update_item(
            "parameter_group",
            id=item["id"],
            color=color,
        )
    try:
        _commit_step(db,
            "v45: updated parameter_group colours to a mid-tone palette "
            "(readable on both light and dark IDE themes); calm cool tones "
            "for the common groups, warm tones for advanced / risk-prone "
            "groups"
        )
    except SpineDBAPIError:
        pass


def _migrate_v46_use_row_scaling(db) -> None:
    """Add the ``use_row_scaling`` parameter on the ``solve`` entity (Agent 5).

    Default "no" preserves pre-Agent-5 behaviour exactly; setting it to
    "yes" on a solve makes the AMPL model derive
    ``node_capacity_for_scaling`` / ``group_capacity_for_scaling`` from
    connected-unit ``unitsize`` (rounded to the nearest power of 10) so
    matrix coefficients stay in a narrower band.
    """
    add_parameters_manual(db, [[
        "solve",
        "use_row_scaling",
        "no",
        "yes_no",
        "Enable automatic row scaling (experimental): derive "
        "node_capacity_for_scaling / group_capacity_for_scaling from "
        "connected-unit unitsizes (rounded to nearest power of 10) so "
        "HiGHS sees matrix coefficients on a narrower range.  Default "
        "'no' preserves pre-scaling behaviour exactly.  See "
        "flextool/SLACK_CONVENTION.md.",
    ]])
    try:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="use_row_scaling",
            parameter_group_name="solve_advanced",
        )
        _commit_step(db,
            "v46: added solve.use_row_scaling parameter (Agent 5 LP-scaling): "
            "per-solve opt-in for automatic row scaling; default 'no' "
            "preserves pre-scaling behaviour."
        )
    except SpineDBAPIError:
        pass
