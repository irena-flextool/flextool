"""Data versions 50-55: step durations on the solve, group block resolution, solver selection, storage binding."""
import logging

from spinedb_api import from_database, SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _attach_list_defaults,
    _commit_step,
    _detach_list_defaults,
    _unless_renamed,
    add_value_list_manual,
    FlexToolMigrationError,
)


def _migrate_v50_new_stepduration_to_solve(db) -> None:
    """Move ``new_stepduration`` from ``timeset`` to ``solve``.

    Rationale
    ---------
    Pre-v50 the parameter lived on ``timeset``, but FlexTool already
    rejected any configuration in which two timesets used by the same
    solve carried different ``new_stepduration`` values (see
    ``timeline_config.create_averaged_timeseries`` — "More than one
    timeline in the solve or the same timeline with different step
    durations in different timesets").  The parameter was therefore
    solve-scoped in practice; this migration makes that explicit.

    Steps
    -----
    1. Capture the description + default from the existing
       ``timeset.new_stepduration`` parameter_definition so the new one
       carries identical metadata.
    2. Add ``solve.new_stepduration`` with the captured metadata, under
       the ``timeline`` parameter_group if it exists.
    3. For every existing ``timeset.new_stepduration`` parameter_value,
       find the solves that use that timeset via their
       ``solve.period_timeset`` map and copy the value onto each such
       solve under the same alternative.  If two timesets used by the
       same solve carry different values under the same alternative,
       raise :class:`FlexToolMigrationError` — the pre-v50 runtime
       would have errored on that config too.
    4. Remove ``timeset.new_stepduration`` (cascades to its values).
    """

    parameter_definitions = db.mapped_table("parameter_definition")

    # --- Step 1: capture existing definition metadata ---------------
    # ``db.item`` raises ``SpineDBAPIError`` when the row is absent
    # (spinedb_api ≥ 0.34); the fallback branch below was written for
    # the older silently-None return, so guard with try/except.
    try:
        timeset_def = db.item(
            parameter_definitions,
            entity_class_name="timeset",
            name="new_stepduration",
        )
    except SpineDBAPIError:
        timeset_def = None
    if timeset_def is None:
        # Nothing to migrate — schema was already missing the old
        # definition (hand-edited DB).  Create the solve-level
        # definition with sensible defaults and return.
        default_val, default_type = to_database(None)
        description = (
            "Hours. Creates a new `timeline` from the old for this "
            "`solve` with this timestep duration. The new timeline "
            "will sum or average the other timeseries data like "
            "`profile` and `inflow` for the new timesteps."
        )
    else:
        default_val = timeset_def["default_value"]
        default_type = timeset_def["default_type"]
        description = (
            timeset_def.get("description")
            or "Hours. Creates a new `timeline` from the old for this "
            "`solve` with this timestep duration. The new timeline "
            "will sum or average the other timeseries data like "
            "`profile` and `inflow` for the new timesteps."
        )
        # Refresh the wording to point at "solve" now that we're moving.
        description = description.replace("for this `timeset`", "for this `solve`")

    # --- Step 2: create solve.new_stepduration definition -----------
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="new_stepduration",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=description,
    )
    # Attach the timeline parameter_group when it exists (v44+).
    if db.item(db.mapped_table("parameter_group"), name="timeline") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="new_stepduration",
            parameter_group_name="timeline",
        )

    # --- Step 3: propagate values from timesets to their solves -----

    # Build timeset -> list[(solve, alternative)] from every
    # solve.period_timeset map.  A period_timeset map has index =
    # period and value = timeset_name; we only care about the set of
    # timesets each solve refers to per alternative.
    solves_by_timeset_alt: dict[tuple[str, str], set[str]] = {}
    for pv in db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="period_timeset",
    ):
        solve_name = pv["entity_byname"][0]
        alt = pv["alternative_name"]
        parsed = from_database(pv["value"], pv["type"])
        # parsed is a spinedb_api.Map: indexes=periods, values=timesets.
        try:
            timesets_in_map = list(parsed.values)
        except AttributeError:
            # Unexpected scalar — skip (no timeset reference).
            continue
        for ts in timesets_in_map:
            ts_name = str(ts)
            solves_by_timeset_alt.setdefault((ts_name, alt), set()).add(solve_name)

    # Also index period_timeset values by solve so we can propagate
    # across alternatives when a timeset value's alternative doesn't
    # match any period_timeset alternative on the same solve.
    timesets_of_solve_any_alt: dict[str, set[str]] = {}
    for (ts_name, alt), solves in solves_by_timeset_alt.items():
        for solve_name in solves:
            timesets_of_solve_any_alt.setdefault(solve_name, set()).add(ts_name)

    # Walk every existing timeset.new_stepduration value and derive
    # the corresponding solve.new_stepduration values.
    # Track assignments so we can detect conflicts.
    written: dict[tuple[str, str], tuple[bytes, str]] = {}

    for pv in list(db.find_parameter_values(
        entity_class_name="timeset",
        parameter_definition_name="new_stepduration",
    )):
        timeset_name = pv["entity_byname"][0]
        ts_alt = pv["alternative_name"]
        value = pv["value"]
        vtype = pv["type"]

        # Which solves reference this timeset?  Match by alternative
        # first; fall back to "any alternative" when the timeset value
        # is defined in an alt where the solve didn't also define its
        # period_timeset.  (Common when both live under the scenario's
        # timeline alternative only.)
        candidate_solves = solves_by_timeset_alt.get((timeset_name, ts_alt), set())
        if not candidate_solves:
            candidate_solves = {
                s for s, ts_set in timesets_of_solve_any_alt.items()
                if timeset_name in ts_set
            }

        for solve_name in candidate_solves:
            key = (solve_name, ts_alt)
            if key in written:
                prev_value, prev_type = written[key]
                if (prev_value, prev_type) != (value, vtype):
                    raise FlexToolMigrationError(
                        f"solve '{solve_name}' (alternative '{ts_alt}') "
                        f"has timesets with conflicting "
                        f"new_stepduration values.  The pre-v50 runtime "
                        f"rejected this configuration at solve time; "
                        f"the v50 migration cannot reconcile it "
                        f"automatically.  Set a single consistent value "
                        f"on all timesets used by this solve before "
                        f"re-running the migration."
                    )
                continue
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=(solve_name,),
                parameter_definition_name="new_stepduration",
                alternative_name=ts_alt,
                value=value,
                type=vtype,
            )
            written[key] = (value, vtype)

    # --- Step 4: drop timeset.new_stepduration ---------------------
    if timeset_def is not None:
        db.remove_parameter_definition(id=timeset_def["id"])

    try:
        _commit_step(db,
            "v50: move new_stepduration from timeset to solve; "
            "propagate values via solve.period_timeset; drop "
            "timeset.new_stepduration (parameter was already "
            "effectively solve-scoped, see timeline_config.py)."
        )
    except SpineDBAPIError:
        pass


def _migrate_v51_group_block_resolution(db) -> None:
    """Add group-level ``new_stepduration`` and ``decomposition_method``.

    Rationale
    ---------
    FlexTool's ``group`` abstraction already carries a variety of
    cross-entity concerns (CO2 caps, reserves, inertia, transfer method
    overrides).  Agent 1.1 introduces two new group-level parameters
    that are the entry point for the flex-temporal / decomposition
    refactor:

    * ``new_stepduration`` — hours; when set, members of the group
      (nodes / units / connections) dispatch at this stepduration,
      overriding the solve-level ``new_stepduration`` (v50) for those
      entities.  This makes mixed-resolution models possible (e.g.
      hourly power + daily hydrogen in the same solve).
    * ``decomposition_method`` — enum string on the
      ``decomposition_methods`` value list (``none``,
      ``lagrangian_region``).  Default ``none`` preserves existing
      behaviour; ``lagrangian_region`` is reserved for Agent 3.2 and
      has no LP behaviour attached yet.

    Only schema + defaults land in this migration — block derivation
    and overlap-set generation are Python-side work.
    """

    # --- Value list for decomposition_method ----------------------------
    add_value_list_manual(db, _unless_renamed(db, 51, "decomposition_methods", "lagrangian_region", "benders_regional"))

    # --- group.new_stepduration -----------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="new_stepduration",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Hours. Members of this group operate at this step "
            "duration. Overrides the solve-level new_stepduration "
            "for these entities. Used for multi-resolution models "
            "where some nodes (e.g. fuel markets) need coarser "
            "dispatch resolution than others (e.g. power systems)."
        ),
    )
    # Attach the solve_advanced parameter_group when it exists (v44+).
    if db.item(db.mapped_table("parameter_group"), name="solve_advanced") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="new_stepduration",
            parameter_group_name="solve_advanced",
        )

    # --- group.decomposition_method -------------------------------------
    default_val, default_type = to_database("none")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="decomposition_method",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="decomposition_methods",
        parameter_type_list=("str",),
        description=(
            "Decomposition strategy to apply to this group. "
            "Currently supported: 'none' (no decomposition — "
            "default), 'lagrangian_region' (group is solved as an "
            "independent region with shared-commodity coupling)."
        ),
    )
    # Attach decomposition_method to the solve_advanced parameter_group
    # — the flag is experimental and nests under the same heading as
    # ``solve.use_row_scaling`` and similar opt-in features.
    if db.item(db.mapped_table("parameter_group"), name="solve_advanced") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="decomposition_method",
            parameter_group_name="solve_advanced",
        )

    try:
        _commit_step(db,
            "v51: added group.new_stepduration and "
            "group.decomposition_method (Agent 1.1 flex-temporal + "
            "decomposition foundation); no LP behaviour yet."
        )
    except SpineDBAPIError:
        pass


def _migrate_v52_solver_selection(db) -> None:
    """Add per-solve solver-selection parameters (Phase 1 multi-solver).

    Rationale
    ---------
    FlexTool historically had a single ``solve.solver`` parameter bound
    to a small value list (``glpsol`` / ``highs`` / ``cplex``) and ran
    HiGHS via highspy in-process.  Phase 1 of the polar-high multi-
    solver handoff (see ``specs/flextool-multi-solver-handoff.md``)
    moves solver dispatch behind polar-high and exposes per-solve user
    controls.  This migration lands the schema:

    * New value lists ``solvers``, ``solver_io_apis``, ``solver_log_levels``.
    * ``solver`` re-bound to the new ``solvers`` value list (HiGHS stays
      default; commercial solvers Gurobi / CPLEX / Xpress / COPT are
      opt-in).
    * Six new parameters: ``solver_io_api``, ``solver_options`` (free-
      form map forwarded to the solver), ``solver_time_limit``,
      ``solver_mip_gap``, ``solver_threads``, ``solver_log_level``.

    All seven parameters attach to the ``solve_advanced`` parameter
    group (consistent with ``solver_precommand`` / ``solver_arguments``
    in v44).  No LP / writer behaviour changes here — engine wiring
    arrives in later phases.
    """

    # --- Migrate legacy 'glpsol' values to 'highs' --------------------
    # GLPK/glpsol was retired in Δ.22 (binary deleted, model file
    # deleted).  Any pre-existing ``solve.solver == "glpsol"`` value
    # would point at a non-functional solver and would also fail the
    # new ``solvers`` value-list check below (glpsol is no longer a
    # member).  Rewrite in place to 'highs' while the parameter is
    # still bound to the legacy ``solver`` value list (which contains
    # 'highs' as well as 'glpsol').
    highs_value, highs_type = to_database("highs")
    for pv in db.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="solver",
    ):
        if pv["parsed_value"] == "glpsol":
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=pv["entity_byname"],
                parameter_definition_name="solver",
                alternative_name=pv["alternative_name"],
                value=highs_value,
                type=highs_type,
            )

    # --- Transform legacy ``solver`` value list into ``solvers`` ------
    # Pre-v52 ``solve.solver`` was bound to a list named ``solver``
    # with members [glpsol, highs, cplex].  v52 expands this to
    # [highs, gurobi, cplex, xpress, copt] and renames to ``solvers``.
    #
    # Update the existing list in place (rather than drop + recreate)
    # so the parameter_definition's foreign-key reference stays valid
    # and dependent parameter_value rows keep their list_value_ref
    # entries through the migration.  Spine cascades the drop to
    # dependent rows, which made the drop-and-readd path unreliable.
    legacy_solver_list = db.get_item("parameter_value_list", name="solver")
    if not legacy_solver_list and db.get_item("parameter_value_list", name="solvers"):
        _already_applied(52, "value list solver is renamed to solvers")
    elif legacy_solver_list:
        legacy_id = legacy_solver_list["id"]
        # Add missing members: gurobi, xpress, copt.
        for member in ("gurobi", "xpress", "copt"):
            v_bytes, v_type = to_database(member)
            db.add_update_item(
                "list_value",
                parameter_value_list_name="solver",  # current (pre-rename) name
                value=v_bytes,
                type=v_type,
            )
        # Remove the glpsol member.  No parameter_value rows reference
        # it after the glpsol→highs rewrite above, so the cascade-delete
        # is a no-op here.  ``db.item`` raises when no match exists,
        # so iterate via ``find_list_values`` and match by value.
        glpsol_value_bytes, _ = to_database("glpsol")
        for lv in db.find_list_values(
            parameter_value_list_name="solver",
        ):
            if lv["value"] == glpsol_value_bytes:
                db.remove_item("list_value", lv["id"])
                break
        # Rename the list to ``solvers`` (plural).  The
        # parameter_definition's FK is by id; parameter_value rows'
        # list_value_ref entries are unaffected.
        db.update_item(
            "parameter_value_list", id=legacy_id, name="solvers",
        )

    # --- Value lists ---------------------------------------------------
    # ``solvers`` either already exists from the transform above (when
    # the legacy ``solver`` list was renamed) OR needs to be created
    # from scratch (fresh DB / no pre-v52 history).  ``add_value_list_manual``
    # is idempotent on member names so we can call it either way.
    add_value_list_manual(db, [
        ["solvers", "highs"],
        ["solvers", "gurobi"],
        ["solvers", "cplex"],
        ["solvers", "xpress"],
        ["solvers", "copt"],
        ["solver_io_apis", "direct"],
        ["solver_io_apis", "mps"],
        ["solver_io_apis", "lp"],
        ["solver_log_levels", "silent"],
        ["solver_log_levels", "normal"],
        ["solver_log_levels", "verbose"],
    ])

    has_solve_advanced = (
        db.item(db.mapped_table("parameter_group"), name="solve_advanced")
        is not None
    )

    # --- solve.solver (rebind to new ``solvers`` value list) -----------
    # Existing parameter from v1; only the value list and description
    # change.  ``add_update_item`` keyed on (entity_class_name, name)
    # is idempotent and preserves any user-set defaults via
    # parameter_value rows (we only touch the definition's default).
    default_val, default_type = to_database("highs")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="solvers",
        parameter_type_list=("str",),
        description=(
            "Solver to use for this solve. One of polar-high's "
            "available_solvers ('highs', 'gurobi', 'cplex', 'xpress', "
            "'copt'). HiGHS is the bundled default; commercial solvers "
            "require a separate install and license."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_io_api ------------------------------------------
    default_val, default_type = to_database("direct")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_io_api",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="solver_io_apis",
        parameter_type_list=("str",),
        description=(
            "How the model is handed to the solver: 'direct' (in-process "
            "API, fastest), 'mps' or 'lp' (file fallback for solvers or "
            "environments without a direct binding)."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_io_api",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_options -----------------------------------------
    # Free-form map of key->value options passed through to the solver.
    # Default is left unset (None); user must populate the map
    # explicitly.  This mirrors the pre-existing convention for
    # solver_arguments (array, ``parameter_type_list=("array",)``):
    # awkward defaults are avoided by leaving the slot empty.
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_options",
        default_value=default_val,
        default_type=default_type,
        # Spine encodes Map rank inline: "1d_map" means Map rank 1
        # (single index level → scalar value).  ``solver_options`` is
        # a flat key→value mapping (option_name → string/float), so
        # rank 1 is correct.  Bare "map" is invalid (Map requires
        # rank ≥ 1).
        parameter_type_list=("1d_map",),
        description=(
            "Map of solver-specific option name -> value, forwarded "
            "raw to the chosen solver. Use the convenience parameters "
            "(solver_time_limit, solver_mip_gap, solver_threads) for "
            "the common cases; this map is for anything else."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_options",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_time_limit --------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_time_limit",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Wall-clock time limit for the solver in seconds. "
            "Normalised across solvers by polar-high. None = no limit."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_time_limit",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_mip_gap -----------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_mip_gap",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Relative MIP optimality gap at which the solver may stop "
            "early. Normalised across solvers by polar-high. None = "
            "solver default."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_mip_gap",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_threads -----------------------------------------
    default_val, default_type = to_database(None)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_threads",
        default_value=default_val,
        default_type=default_type,
        # Spine has no integer value type; everything numeric is
        # stored as float.  Users entering ``4`` get a float on
        # disk; the runtime coerces back to int via ``_opt_int``
        # in _solve_config.py.
        parameter_type_list=("float",),
        description=(
            "Maximum number of solver worker threads. Normalised "
            "across solvers by polar-high. None = solver default "
            "(usually all available cores)."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_threads",
            parameter_group_name="solve_advanced",
        )

    # --- solve.solver_log_level ---------------------------------------
    default_val, default_type = to_database("normal")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="solver_log_level",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="solver_log_levels",
        parameter_type_list=("str",),
        description=(
            "Verbosity of the solver log: 'silent' (suppress), "
            "'normal' (default summary output), 'verbose' (detailed "
            "per-iteration output)."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="solver_log_level",
            parameter_group_name="solve_advanced",
        )


    try:
        _commit_step(db,
            "v52: added solver-selection parameters on solve "
            "(solver rebound to 'solvers' value list; new "
            "solver_io_api, solver_options, solver_time_limit, "
            "solver_mip_gap, solver_threads, solver_log_level); "
            "migrated legacy 'glpsol' values to 'highs'; "
            "removed legacy 'solver' value list."
        )
    except SpineDBAPIError:
        pass


def _migrate_v53_storage_binding_value_list(db) -> None:
    """Wire ``storage_binding_methods`` value list to
    ``node.storage_binding_method`` parameter_definition.

    Rationale
    ---------
    The ``storage_binding_methods`` parameter_value_list was first
    created in v30 (``bind_using_blended_weights``) and extended in
    v31 (``bind_intraperiod_blocks``); ``update_timestructure`` (called
    from the v22 step) further mutates one of its members from
    ``bind_within_timeblock`` to ``bind_within_timeset``.  However, the
    ``node.storage_binding_method`` parameter_definition was never
    bound to this list — its ``parameter_value_list_id`` stayed NULL.
    The practical effect was that Spine UI cheerfully accepted
    arbitrary strings (and even ``array``-typed values, see the
    H2_trade.sqlite case in
    ``_audit_reports/storage_binding_method_callsites.md`` §9) and the
    backend silently flattened the array, fueling the 2026-04
    additive-flag bug now being reverted.

    Phase 1 of the single-valued migration closes the schema gap.
    Phase 2 ports existing array-valued data onto the new contract.
    The ingestion guard in ``flextool/spinedb_backend/_backend.py``
    (see ``parameter_values``) is the runtime safety net that fires
    if a v52-or-older DB containing array values is opened directly,
    without running this migration first.
    """
    pvl_table = db.mapped_table("parameter_value_list")
    sbm_list = db.item(pvl_table, name="storage_binding_methods")
    if sbm_list is None:
        # Pre-v30 DB that somehow skipped the value-list creation —
        # surface the issue rather than silently no-op.  v30 / v31 are
        # idempotent ``add_value_list_manual`` calls so any reasonable
        # upgrade path will have populated this list before reaching
        # v53.  If it is genuinely missing we cannot wire anything; the
        # assertion below will trip and force the operator to look.
        raise SpineDBAPIError(
            "v53 migration: parameter_value_list "
            "'storage_binding_methods' not found.  Re-run the "
            "migration starting from a version <=31 so the list is "
            "created, then retry."
        )

    parameter_definitions = db.mapped_table("parameter_definition")
    param = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    if param is None:
        raise SpineDBAPIError(
            "v53 migration: parameter_definition "
            "'node.storage_binding_method' not found.  The v1 schema "
            "is expected to define it; cannot wire value_list."
        )

    db.update_parameter_definition(
        id=param["id"],
        parameter_value_list_name="storage_binding_methods",
    )

    # In-migration assertion: the wiring actually took effect.  We
    # re-read the parameter_definition row and compare its
    # parameter_value_list_id (or _name) against the list we wired.
    parameter_definitions = db.mapped_table("parameter_definition")
    param_after = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    wired_id = param_after.get("parameter_value_list_id")
    wired_name = param_after.get("parameter_value_list_name")
    if wired_id != sbm_list["id"] and wired_name != "storage_binding_methods":
        raise SpineDBAPIError(
            "v53 migration: post-write verification failed — "
            "node.storage_binding_method.parameter_value_list_id is "
            f"{wired_id!r} / name {wired_name!r}, expected list id "
            f"{sbm_list['id']!r} (name 'storage_binding_methods')."
        )

    _commit_step(
        db,
        "v53: wired storage_binding_methods value_list to "
        "node.storage_binding_method parameter_definition (Phase 1 of "
        "the single-valued storage_binding_method migration).",
    )


#: Priority order (highest first) used by the v54 migration to collapse
#: array-valued ``node.storage_binding_method`` rows down to a single
#: string.  Mirrors the audit's H2_trade.sqlite recommendation
#: (see ``_audit_reports/storage_binding_method_callsites.md`` §9):
#: when multiple methods coexist in a single array, the RP-aware
#: ``bind_using_blended_weights`` wins because it carries the most
#: state-tracking machinery; ``bind_forward_only`` loses because it is
#: the silent default and should only surface when nothing else asked
#: for richer semantics.
_STORAGE_BINDING_PRIORITY: tuple[str, ...] = (
    "bind_using_blended_weights",
    "bind_intraperiod_blocks",
    "bind_within_solve",
    "bind_within_period",
    "bind_within_timeset",
    "bind_forward_only",
)


def _migrate_v54_storage_binding_arrays_to_scalar(db) -> None:
    """Rewrite every array-valued ``node.storage_binding_method`` row
    as a single string per :data:`_STORAGE_BINDING_PRIORITY`.

    Background
    ----------
    The 2026-04 list-valued design (now being reverted) silently
    flattened array-typed ``storage_binding_method`` values into one
    row per array element, which downstream additive logic in
    ``calc_storage_vre.py`` turned into double-counted state-change
    residuals.  v52 ingestion guard (Phase 1) now rejects arrays at
    load time; this v54 step is the data-side counterpart that ports
    pre-existing databases (e.g. H2_trade.sqlite, with 15 array-valued
    entries — see ``_audit_reports/storage_binding_method_callsites.md``
    §9) onto the new single-string contract.

    Behaviour
    ---------
    For each ``parameter_value`` row of
    ``node.storage_binding_method``:

    - If ``type != "array"`` (scalar string or other): leave untouched.
    - If ``type == "array"``: pick the highest-priority element from
      :data:`_STORAGE_BINDING_PRIORITY` that appears in the array's
      values, then overwrite the row in-place with that single string.
      Preserves ``entity_id`` / ``alternative_id`` / ``entity_byname``;
      only ``value`` and ``type`` change.
    - If the array contains *only* strings that are not in the priority
      list (i.e. nothing matched), raise ``SpineDBAPIError`` naming
      the entity and the unknown contents.  We refuse to guess.

    Post-migration assertion: every remaining row for
    ``node.storage_binding_method`` is verified to have ``type == "str"``.
    """
    set(_STORAGE_BINDING_PRIORITY)

    updated: list[tuple[tuple[str, ...], str, list, str]] = []
    for pv in list(db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    )):
        if pv["type"] != "array":
            continue

        entity_byname = pv["entity_byname"]
        alt_name = pv["alternative_name"]
        parsed = pv["parsed_value"]
        try:
            members = list(parsed.values)
        except AttributeError as exc:
            raise SpineDBAPIError(
                "v54 migration: node.storage_binding_method row for "
                f"entity {entity_byname!r} alternative {alt_name!r} "
                f"has type='array' but parsed_value {parsed!r} does "
                "not expose a .values list — cannot port to scalar."
            ) from exc

        picked: str | None = None
        for candidate in _STORAGE_BINDING_PRIORITY:
            if candidate in members:
                picked = candidate
                break
        if picked is None:
            raise SpineDBAPIError(
                "v54 migration: node.storage_binding_method array for "
                f"entity {entity_byname!r} (alternative {alt_name!r}) "
                f"contains only unknown methods {members!r}.  Expected "
                "at least one of "
                f"{list(_STORAGE_BINDING_PRIORITY)!r}.  Refusing to "
                "guess; fix the source data and retry."
            )

        new_value, new_type = to_database(picked)
        db.update_parameter_value(
            id=pv["id"],
            value=new_value,
            type=new_type,
        )
        updated.append((entity_byname, alt_name, members, picked))

    # Post-write verification: every remaining row must be scalar str.
    for pv in db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    ):
        if pv["type"] != "str":
            raise SpineDBAPIError(
                "v54 migration: post-write verification failed — "
                "node.storage_binding_method row for entity "
                f"{pv['entity_byname']!r} alternative "
                f"{pv['alternative_name']!r} still has type "
                f"{pv['type']!r} (expected 'str').  Migration is "
                "incomplete; aborting before commit."
            )

    if updated:
        summary_lines = [
            f"  {ent} ({alt}): {arr} -> {pick}"
            for ent, alt, arr, pick in updated
        ]
        logging.info(
            "v54 migration ported %d array-valued "
            "node.storage_binding_method rows to scalar strings:\n%s",
            len(updated),
            "\n".join(summary_lines),
        )

    _commit_step(
        db,
        "v54: ported array-valued node.storage_binding_method "
        "parameter_value rows to scalar strings (Phase 2 of the "
        "single-valued storage_binding_method migration).  "
        f"{len(updated)} row(s) rewritten.",
    )


#: Map of legacy ``node.storage_binding_method`` scalar values to the
#: clean-seven-method names introduced by the storage-binding
#: restructure (Phase A).  Used by v55 to rewrite parameter_value rows
#: in-place and to determine which value_list members to drop.
#:
#: - ``bind_within_timeset`` becomes ``bind_within_timeblock``: a timeset
#:   can contain several timeblocks; the binding always operated per
#:   block, not per set — the old name was misleading.
#: - ``bind_using_blended_weights`` becomes
#:   ``bind_within_solve_blended_weights``: makes the cycle-closure
#:   scope explicit (a solve), in line with the two new variants the
#:   restructure adds (per-period and forward-only blended weights).
#: - ``bind_within_model`` becomes ``bind_within_solve``: ``model`` was
#:   already removed from the v53/v54 value_list, but any stray
#:   parameter_value rows still using it migrate to the largest in-scope
#:   binding window we offer.
_STORAGE_BINDING_RENAMES_V55: dict[str, str] = {
    "bind_within_timeset": "bind_within_timeblock",
    "bind_using_blended_weights": "bind_within_solve_blended_weights",
    "bind_within_model": "bind_within_solve",
}

#: Members of the ``storage_binding_methods`` value_list to drop in v55
#: (the three "rename-from" names).  Removed via ``find_list_values`` +
#: ``remove_item`` because Spine's value_list operations don't
#: re-validate dependent parameter_value rows; the data-rewrite in step
#: (a) of the v55 helper has already cleared those references.
_STORAGE_BINDING_DROPPED_V55: tuple[str, ...] = (
    "bind_within_timeset",
    "bind_using_blended_weights",
    "bind_within_model",
)

#: Members to add (idempotently) to the ``storage_binding_methods``
#: value_list in v55.  Two of these (``bind_within_period_blended_weights``
#: and ``bind_forward_only_blended_weights``) gain their constraint
#: implementations only in later phases of the restructure — Phase A
#: only declares them in the value_list so the wired Spine UI can offer
#: them and so subsequent phases have a stable enum to key off.
_STORAGE_BINDING_ADDED_V55: tuple[str, ...] = (
    "bind_within_timeblock",
    "bind_within_solve_blended_weights",
    "bind_within_period_blended_weights",
    "bind_forward_only_blended_weights",
)

#: Expected exact membership of ``storage_binding_methods`` after the
#: v55 step completes.  Used by the post-migration verification block
#: to fail loudly if drops or adds didn't take effect.
_STORAGE_BINDING_EXPECTED_V55: frozenset[str] = frozenset({
    "bind_within_period",
    "bind_within_solve",
    "bind_within_timeblock",
    "bind_forward_only",
    "bind_within_solve_blended_weights",
    "bind_within_period_blended_weights",
    "bind_forward_only_blended_weights",
    "bind_intraperiod_blocks",
})

# Canonical post-v55 description for ``node.storage_binding_method``.
# Mirrors ``flextool/schemas/spinedb_schema.json``'s parameter_definition
# entry verbatim so that DBs migrated through v55 carry the same text in
# their own ``parameter_definition`` row as freshly-seeded DBs.  When the
# schema description text changes (Phase F-style scrub or later), update
# this constant in lockstep so existing-DB migration stays aligned with
# new-DB seeding.
_STORAGE_BINDING_METHOD_DESCRIPTION_V55: str = (
    "Choice how the storage state will be maintained over discontinuous "
    "timelines. Seven cycle-scope methods (the state-continuity family): "
    "'bind_within_timeblock' cycles state within each timeblock (cycle "
    "closes at block boundaries); 'bind_within_period' cycles within each "
    "FlexTool period and chains blocks inside the period; "
    "'bind_within_solve' cycles across the whole solve horizon; "
    "'bind_forward_only' (default) chains state forward across the solve "
    "with no end-to-start closure; 'bind_within_solve_blended_weights' is "
    "the representative-period variant of bind_within_solve (RP weighting "
    "+ solve-level cycle closure); 'bind_within_period_blended_weights' "
    "is the RP variant of bind_within_period (per-period RP weighting, "
    "each period closes independently); "
    "'bind_forward_only_blended_weights' is the RP variant of "
    "bind_forward_only (RP weighting, no cycle closure). One additional "
    "value, 'bind_intraperiod_blocks', is structurally an aggregation "
    "method rather than a cycle-scope method: state is held constant "
    "within each block and the block-total flow is balanced at the "
    "boundary. Silent-degrade behaviour: any '*_blended_weights' method "
    "on a node in a solve whose active timeset has no "
    "representative_period_weights is automatically downgraded to the "
    "corresponding non-RP variant for that solve, so the same storage "
    "entity can drive an RP investment solve and a chronological "
    "dispatch solve back-to-back. Separate parameters (e.g. "
    "'storage_state_start') can force additional bindings. By default, "
    "storage start state is bound to 0."
)


def _migrate_v55_storage_binding_rename_and_extend(db) -> None:
    """Rename legacy ``storage_binding_method`` scalar values to their
    clean-seven-method names and refresh the
    ``storage_binding_methods`` value_list.

    Background
    ----------
    Phase A of the storage-binding-methods restructure.  v53/v54 left
    the value_list carrying three names that the new design replaces:

    - ``bind_within_timeset``         → ``bind_within_timeblock``
    - ``bind_using_blended_weights``  → ``bind_within_solve_blended_weights``
    - ``bind_within_model``           → ``bind_within_solve`` (legacy
      ``bind_within_model`` was already dropped from the value_list in
      ``update_timestructure``-era history; this step catches any
      stray parameter_value rows still carrying the string)

    Phase A also seeds the value_list with the two upcoming
    blended-weights variants (``bind_within_period_blended_weights``
    and ``bind_forward_only_blended_weights``) whose constraint
    implementations land in Phases D and E.  Adding them now (with an
    empty implementation) lets the wired Spine UI surface the
    enumeration without further schema churn later.

    Behaviour
    ---------
    Step (a) — value_list extension.  Add the four new members
    (idempotent via ``add_value_list_manual``) BEFORE rewriting any
    parameter_value rows.  Order matters: v53 wired the value_list to
    ``node.storage_binding_method``, so Spine validates every
    ``update_parameter_value`` against current list membership.
    Writing ``bind_within_solve_blended_weights`` to a row before that
    name exists in the list would raise.

    Step (b) — data rewrite.  For each ``parameter_value`` row of
    ``node.storage_binding_method`` whose scalar string value is one of
    :data:`_STORAGE_BINDING_RENAMES_V55`, rewrite the row in-place with
    the renamed string (same entity, alternative, type).  Other values
    pass through untouched.  Array-typed rows are not expected at this
    point (v54 collapsed them to scalars and v52's ingestion guard
    rejects new arrays); the helper does not iterate them.

    Step (c) — value_list cleanup.  Drop the three rename-from members
    from the list.  Safe now that step (b) has cleared every
    parameter_value row referencing them.

    Step (d) — refresh the parameter_definition description.  The
    schema-template text in ``flextool/schemas/spinedb_schema.json``
    was rewritten in Phase F to enumerate the seven cycle-scope
    methods + ``bind_intraperiod_blocks`` (aggregation) + the
    silent-degrade behaviour.  Fresh DBs seeded from the schema get
    the new text automatically; existing DBs being migrated up to
    v55 must have their own ``parameter_definition`` row rewritten
    so the in-DB help text matches.  Mirrors
    :data:`_STORAGE_BINDING_METHOD_DESCRIPTION_V55` verbatim.

    Step (e) — in-migration verification.  Re-query
    ``node.storage_binding_method`` rows and assert no legacy string
    remains; re-query the value_list and assert exact expected
    membership.  ``SpineDBAPIError`` is raised with the offending
    entries if either check fails — surface, don't guess.
    """
    # ---- Step (a): extend the storage_binding_methods value_list ----
    # Add the four new members BEFORE the data rewrite so the v53
    # wiring on node.storage_binding_method accepts the renamed names.
    pvl_table = db.mapped_table("parameter_value_list")
    sbm_list = db.item(pvl_table, name="storage_binding_methods")
    if sbm_list is None:
        raise SpineDBAPIError(
            "v55 migration: parameter_value_list "
            "'storage_binding_methods' not found.  v30/v31/v53 are "
            "expected to have populated and wired it before this step; "
            "cannot extend a list that does not exist."
        )

    add_value_list_manual(
        db,
        [["storage_binding_methods", name]
         for name in _STORAGE_BINDING_ADDED_V55],
    )

    # ---- Step (b): rename parameter_value rows ----------------------
    renamed: list[tuple[tuple[str, ...], str, str, str]] = []
    for pv in list(db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    )):
        if pv["type"] != "str":
            # v54 left every row scalar-str; anything else is a
            # surprise we refuse to mutate silently.
            raise SpineDBAPIError(
                "v55 migration: node.storage_binding_method row for "
                f"entity {pv['entity_byname']!r} alternative "
                f"{pv['alternative_name']!r} has unexpected type "
                f"{pv['type']!r} (expected 'str' post-v54).  Re-run v54 "
                "first, or fix the source row, before retrying."
            )
        old_value = pv["parsed_value"]
        new_value = _STORAGE_BINDING_RENAMES_V55.get(old_value)
        if new_value is None:
            # Pass-through value (already in the clean set).
            continue
        new_value_bytes, new_value_type = to_database(new_value)
        db.update_parameter_value(
            id=pv["id"],
            value=new_value_bytes,
            type=new_value_type,
        )
        renamed.append((
            pv["entity_byname"], pv["alternative_name"], old_value, new_value,
        ))

    # ---- Step (b) verification: no legacy names remain --------------
    legacy = set(_STORAGE_BINDING_RENAMES_V55)
    offenders_a: list[tuple[tuple[str, ...], str, str]] = []
    for pv in db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_binding_method",
    ):
        if pv["type"] == "str" and pv["parsed_value"] in legacy:
            offenders_a.append((
                pv["entity_byname"], pv["alternative_name"], pv["parsed_value"],
            ))
    if offenders_a:
        raise SpineDBAPIError(
            "v55 migration: post-rewrite verification failed — "
            "node.storage_binding_method rows still carry legacy "
            f"names: {offenders_a!r}.  Expected zero rows in "
            f"{sorted(legacy)!r}."
        )

    # ---- Step (c): drop the three rename-from value_list members ----
    # ``find_list_values`` walks the list; we match by encoded value
    # bytes (same pattern as the v52 ``glpsol`` drop a few hundred
    # lines above).
    dropped_lvs: list[str] = []
    pending = _detach_list_defaults(db, "storage_binding_methods", _STORAGE_BINDING_RENAMES_V55)
    for legacy_name in _STORAGE_BINDING_DROPPED_V55:
        legacy_bytes, _ = to_database(legacy_name)
        for lv in list(db.find_list_values(
            parameter_value_list_name="storage_binding_methods",
        )):
            if lv["value"] == legacy_bytes:
                db.remove_item("list_value", lv["id"])
                dropped_lvs.append(legacy_name)
                break
    _attach_list_defaults(db, pending)

    # ---- Step (d): refresh parameter_definition description ---------
    # Mirror the Phase F rewrite of the schema-template description so
    # existing-DB migration emits the same in-DB help text as fresh-DB
    # seeding from spinedb_schema.json.
    parameter_definitions = db.mapped_table("parameter_definition")
    sbm_def = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    if sbm_def is None:
        raise SpineDBAPIError(
            "v55 migration: parameter_definition "
            "'node.storage_binding_method' not found.  v1 schema is "
            "expected to define it; cannot refresh description."
        )
    db.update_parameter_definition(
        id=sbm_def["id"],
        description=_STORAGE_BINDING_METHOD_DESCRIPTION_V55,
    )
    sbm_def_after = db.item(
        parameter_definitions,
        entity_class_name="node",
        name="storage_binding_method",
    )
    if sbm_def_after["description"] != _STORAGE_BINDING_METHOD_DESCRIPTION_V55:
        raise SpineDBAPIError(
            "v55 migration: parameter_definition description refresh "
            "did not take effect for node.storage_binding_method."
        )

    # ---- Step (e) verification: exact value_list membership ---------
    members_after = {
        from_database(lv["value"], lv["type"])
        for lv in db.find_list_values(
            parameter_value_list_name="storage_binding_methods",
        )
    }
    if members_after != _STORAGE_BINDING_EXPECTED_V55:
        missing = sorted(_STORAGE_BINDING_EXPECTED_V55 - members_after)
        extra = sorted(members_after - _STORAGE_BINDING_EXPECTED_V55)
        raise SpineDBAPIError(
            "v55 migration: post-refresh value_list membership "
            f"mismatch — missing {missing!r}, extra {extra!r}.  "
            f"Expected exactly {sorted(_STORAGE_BINDING_EXPECTED_V55)!r}; "
            f"got {sorted(members_after)!r}."
        )

    if renamed:
        summary_lines = [
            f"  {ent} ({alt}): {old} -> {new}"
            for ent, alt, old, new in renamed
        ]
        logging.info(
            "v55 migration renamed %d node.storage_binding_method "
            "rows:\n%s",
            len(renamed),
            "\n".join(summary_lines),
        )
    if dropped_lvs:
        logging.info(
            "v55 migration dropped storage_binding_methods members: %s",
            sorted(dropped_lvs),
        )

    _commit_step(
        db,
        "v55: renamed legacy node.storage_binding_method scalar values "
        "to the clean-seven-method set (bind_within_timeset -> "
        "bind_within_timeblock; bind_using_blended_weights -> "
        "bind_within_solve_blended_weights; bind_within_model -> "
        "bind_within_solve) and refreshed the storage_binding_methods "
        "value_list (dropped the three rename-from members, added "
        "bind_within_timeblock, bind_within_solve_blended_weights, "
        "bind_within_period_blended_weights, "
        "bind_forward_only_blended_weights).  Also refreshed the "
        "node.storage_binding_method parameter_definition description "
        "to match the post-Phase-F schema-template text.  Phase A of "
        "the storage-binding restructure.  "
        f"{len(renamed)} row(s) renamed.",
    )
