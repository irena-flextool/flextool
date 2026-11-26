"""Data version 56, second part: the ``none`` members and the yes/no value lists, the co2 methods, ``is_enabled``; :func:`_migrate_v56`, the whole step."""
from spinedb_api import from_database, SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _attach_list_defaults,
    _commit_step,
    _detach_list_defaults,
    _later_definition,
    _pdef_or_none,
    _remove_items_bounded,
    add_value_list_manual,
)
from flextool.update_flextool.db_migration.v56_renames import (
    _migrate_v56_add_group_cumulative_capacity_descriptions,
    _migrate_v56_fix_wrong_defaults,
    _migrate_v56_fold_highs_method_into_solver_arguments,
    _migrate_v56_fold_highs_parallel_into_solver_arguments,
    _migrate_v56_fold_highs_presolve_into_solver_arguments,
    _migrate_v56_fold_solver_options_into_solver_arguments,
    _migrate_v56_remove_exclude_entity_outputs,
    _migrate_v56_remove_highs_method,
    _migrate_v56_remove_highs_parallel,
    _migrate_v56_remove_highs_presolve,
    _migrate_v56_remove_model_debug,
    _migrate_v56_remove_output_connection__node__node_flow_t,
    _migrate_v56_remove_output_connection_flow_separate,
    _migrate_v56_remove_output_node_balance_t,
    _migrate_v56_remove_output_ramp_envelope,
    _migrate_v56_remove_output_unit__node_flow_t,
    _migrate_v56_remove_output_unit__node_ramp_t,
    _migrate_v56_remove_solver_io_api,
    _migrate_v56_remove_solver_log_level,
    _migrate_v56_remove_solver_options,
    _migrate_v56_remove_solver_threads,
    _migrate_v56_remove_solver_time_limit,
    _migrate_v56_remove_use_row_scaling,
    _migrate_v56_rename_constraint_coefficient_to_coeff,
    _migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff,
    _migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff,
    _migrate_v56_rename_min_capacity_coefficient_to_capacity_min_coeff,
    _migrate_v56_retype_solver_arguments_to_1d_map,
)


def _migrate_v56(db) -> None:
    # Drop ``model.debug`` from the schema.  The parameter
    # was a leftover from the legacy flextoolrunner / GAMS
    # path that emitted ``input/debug.csv``; no
    # engine module reads it.  Debug control now
    # lives purely on the CLI (``--debug={off,basic,full}``)
    # and in the GUI's ProjectSettings.debug_level — see
    # ``flextool/cli/cmd_run_flextool.py`` and
    # ``flextool/gui/data_models.py``.  Removes the
    # parameter_definition AND any parameter_value rows
    # carrying it.
    _migrate_v56_remove_model_debug(db)
    # Backfill the missing ``description`` field on
    # ``group.cumulative_max_capacity`` and
    # ``group.cumulative_min_capacity``.  Both
    # parameter_definitions exist in the schema since v22
    # but only ``node``/``connection``/``unit`` got their
    # descriptions populated in the v22 migration block —
    # the ``group`` rows were left with NULL/empty text.
    # The schema-template JSON already carries the canonical
    # phrasing; this brings legacy databases in line.
    _migrate_v56_add_group_cumulative_capacity_descriptions(db)
    # Clear ``default_value`` on five parameter_definition
    # rows whose schema-declared default disagrees with how
    # the engine actually consumes the parameter.  The
    # rationale per row lives in
    # ``_audit_reports/v56_default_audit.md`` and in the
    # helper's docstring.  Only ``high``-confidence rows are
    # patched here; ``medium`` / ``surface`` rows are left
    # for user review.
    _migrate_v56_fix_wrong_defaults(db)
    # Shorten the ``_coefficient`` suffix on the four user-
    # constraint coefficient parameters to ``_coeff`` across
    # every entity class that declares them
    # (connection / connection__node / node / unit /
    # unit__inputNode / unit__outputNode).  Pure name change
    # — descriptions, default values, value-list bindings
    # and engine semantics are untouched.  The schema
    # template JSON is updated in the same commit so a
    # fresh v55 init lands on the shortened names; the
    # engine, input_derivation, autoscale and export
    # modules are renamed in lock-step.
    _migrate_v56_rename_constraint_coefficient_to_coeff(db)
    # Rename ``flow_coefficient`` → ``conversion_flow_coeff``
    # on ``unit__inputNode`` / ``unit__outputNode``.  Same
    # suffix shape as the four constraint-coefficient
    # renames immediately above (``_coeff``), with the
    # ``conversion_`` prefix making the parameter's role
    # explicit: it scales the conversion of input → output
    # energy in unit dispatch (the node-balance and
    # ``conversion_indirect`` equations).  Pure name
    # change — description, default value (1.0), value-
    # list bindings and engine semantics are untouched.
    # The schema template JSON, engine frame attrs,
    # CSV filename suffixes, input_derivation cl_pars,
    # autoscale quantity types, pandas accessors and the
    # docs are renamed in lock-step.
    _migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff(db)
    # Rename ``max_capacity_coefficient`` →
    # ``capacity_max_coeff`` on ``unit__inputNode`` /
    # ``unit__outputNode``.  Same ``_coeff`` suffix as the
    # earlier batches; the noun ``capacity`` now leads and
    # the qualifier ``max`` follows so the parameter sorts
    # alphabetically with the other capacity-related rows
    # (``capacity``, ``capacity_existing``, ``capacity_max``
    # invest cap, …).  Pure name change — description,
    # default value (1.0), parameter_value_list and engine
    # semantics are untouched.  Engine_polars derived
    # params, autoscale quantity types, input_derivation
    # cl_pars, CSV filename suffixes
    # (``p_process_source_max_capacity_coefficient.csv`` →
    # ``p_process_source_capacity_max_coeff.csv`` and sink),
    # and docs are renamed in lock-step.
    _migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff(db)
    # Rename ``min_capacity_coefficient`` →
    # ``capacity_min_coeff`` on ``unit__inputNode`` /
    # ``unit__outputNode``.  Mirror of the immediately
    # preceding ``capacity_max_coeff`` rename: same
    # ``_coeff`` suffix, same noun-leads-qualifier-follows
    # ordering so the parameter sorts alphabetically with
    # the other capacity-related rows
    # (``capacity_max_coeff``, ``capacity_min_coeff``, …).
    # Pure name change — description, default value (1.0),
    # parameter_value_list and engine semantics are
    # untouched.  Autoscale quantity types,
    # input_derivation cl_pars, CSV filename suffixes
    # (``p_process_source_min_capacity_coefficient.csv`` →
    # ``p_process_source_capacity_min_coeff.csv`` and sink),
    # and docs are renamed in lock-step.
    _migrate_v56_rename_min_capacity_coefficient_to_capacity_min_coeff(db)
    # Drop ``model.exclude_entity_outputs``.  The parameter
    # was a gate on the three per-period capacity dumps
    # (``unit_capacity.csv``, ``connection_capacity.csv``,
    # ``node_capacity.csv``) emitted by
    # :mod:`flextool.process_outputs.handoff_writers`.  The
    # schema's ``"yes"`` default made "exclude" silently the
    # default for every database — which inverts the
    # parameter name's apparent intent ("exclude" reads as
    # opt-in, behaved as opt-out).  Per-entity capacity rows
    # now always emit; aggregated/group outputs continue to
    # be controlled by the three ``group.output_*`` set
    # selectors (unaffected).  The gate site in
    # ``handoff_writers``, the cl_pars emitter in
    # ``input_derivation/_specs.py``, the SET_LIKE_NAMES
    # entry in ``spinedb_backend/_backend.py``, the autoscale
    # quantity-type row, the ``export_settings.yaml`` params
    # listing and the v44 parameter_group membership map are
    # all stripped in the same commit.
    _migrate_v56_remove_exclude_entity_outputs(db)
    # Drop ``model.output_node_balance_t``.  Dead toggle:
    # nothing in ``engine`` reads its row from the
    # ``optional_outputs.csv`` set, and the ``enable_set``
    # check in :mod:`flextool.engine.preprocess.emit._per_solve`
    # only looks for ``output_horizon``.  Schema row, the
    # SET_LIKE_NAMES bookkeeping entry, the autoscale
    # quantity-type table row and the export_settings.yaml
    # params list entry are stripped in the same commit.
    _migrate_v56_remove_output_node_balance_t(db)
    # Drop ``model.output_ramp_envelope``.  Dead toggle:
    # the flag is plumbed into the ``optional_outputs.csv``
    # multi-emitter but nothing on the engine side reads
    # its row from ``enable_optional_outputs`` (only
    # ``output_horizon`` is consulted).  Schema row,
    # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
    # autoscale quantity-type row and export_settings.yaml
    # params list entry are stripped in the same commit.
    _migrate_v56_remove_output_ramp_envelope(db)
    # Drop ``model.output_unit__node_flow_t``.  Dead toggle:
    # the flag was plumbed into the ``optional_outputs.csv``
    # multi-emitter but nothing on the engine side reads
    # its row from ``enable_optional_outputs`` (only
    # ``output_horizon`` is consulted).  Schema row,
    # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
    # autoscale quantity-type row, export_settings.yaml
    # params list and the legacy regen_lh2_three_region.py
    # ``yes`` override are stripped in the same commit.
    _migrate_v56_remove_output_unit__node_flow_t(db)
    # Drop ``model.output_unit__node_ramp_t``.  Dead toggle:
    # plumbed into ``optional_outputs.csv`` but nothing on
    # the engine side reads its row.  Schema row,
    # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
    # autoscale quantity-type row and export_settings.yaml
    # params list entry are stripped in the same commit.
    _migrate_v56_remove_output_unit__node_ramp_t(db)
    # Drop ``model.output_connection__node__node_flow_t``.
    # Dead toggle: plumbed into ``optional_outputs.csv`` but
    # nothing on the engine side reads its row.  Schema row,
    # input_derivation cl_pars entry, SET_LIKE_NAMES entry,
    # autoscale quantity-type row, export_settings.yaml
    # params list entry and the legacy
    # regen_lh2_three_region.py ``yes`` override are
    # stripped in the same commit.
    _migrate_v56_remove_output_connection__node__node_flow_t(db)
    # Drop ``model.output_connection_flow_separate``.  Last
    # of the Batch-B dead-toggle removals; same shape as the
    # other six.  After this commit the
    # ``optional_outputs.csv`` cl_pars emitter holds only
    # ``output_horizon`` — the one flag actually consumed
    # by ``_per_solve``.
    _migrate_v56_remove_output_connection_flow_separate(db)
    # Batch C.1 — first commit of the solver-knob
    # consolidation.  Retype ``solve.solver_arguments`` from
    # ``array`` to ``1d_map`` so it can hold the HiGHS
    # solver options the engine layers on top of
    # ``solver_config/highs.opt``.  Subsequent commits
    # (C.2-C.5) fold the sibling ``solver_options`` Map and
    # the three ``highs_method`` / ``highs_parallel`` /
    # ``highs_presolve`` shortcut parameters into this
    # canonical home before removing them.  Existing
    # parameter_value rows carrying Array values are
    # parsed entry-by-entry as ``key=value`` HiGHS option
    # lines and converted to Map entries; unparseable
    # entries raise so the user can inspect their data.
    _migrate_v56_retype_solver_arguments_to_1d_map(db)
    # Batch C.2 — fold the legacy ``solver_options`` Map
    # into the freshly retyped ``solver_arguments`` 1d-map
    # (existing solver_arguments entries win on key
    # collision; collisions logged to stdout) and drop the
    # now-duplicate ``solver_options`` parameter
    # definition.  All in-repo fixtures carry null values
    # for ``solver_options`` at v55 so the fold path is
    # exercised only by user databases that authored
    # entries.
    _migrate_v56_fold_solver_options_into_solver_arguments(db)
    _migrate_v56_remove_solver_options(db)
    # Batch C.3 — fold ``highs_method`` into
    # ``solver_arguments['solver']`` (HiGHS' own name for
    # the LP-method selection option, verified against
    # ``DETERMINISM_OPTIONS`` and the
    # the former ``_HIGHS_PARAM_MAP`` table of the flex_data loader) and drop the
    # shortcut parameter.  Existing ``solver_arguments``
    # entries for ``solver`` win on collision (logged).
    _migrate_v56_fold_highs_method_into_solver_arguments(db)
    _migrate_v56_remove_highs_method(db)
    # Batch C.4 — fold ``highs_parallel`` into
    # ``solver_arguments['parallel']`` (HiGHS' own option
    # name for parallelism control) and drop the
    # shortcut + its value list.  Existing
    # ``solver_arguments`` entries for ``parallel`` win
    # on collision (logged).
    _migrate_v56_fold_highs_parallel_into_solver_arguments(db)
    _migrate_v56_remove_highs_parallel(db)
    # Batch C.5 — fold ``highs_presolve`` into
    # ``solver_arguments['presolve']`` (HiGHS' own option
    # name for presolve control) and drop the shortcut +
    # its value list.  Existing ``solver_arguments``
    # entries for ``presolve`` win on collision (logged).
    # After this commit all three legacy ``highs_*``
    # shortcuts are gone and the new resolver is the
    # sole consumer of these overrides.
    _migrate_v56_fold_highs_presolve_into_solver_arguments(db)
    _migrate_v56_remove_highs_presolve(db)
    # Batch C.6 — drop ``solver_threads`` parameter.
    # User-stored value DROPPED (per the Q-C-2 design:
    # GUI-bound knobs were rarely scenario-relevant and
    # the break is acceptable).  Use --highs-threads CLI
    # flag (or the GUI's max_cores_per_job, which plumbs
    # it automatically); the GUI control for the DB-
    # stored equivalent is deferred to the v56 follow-up
    # PR (task #26).
    _migrate_v56_remove_solver_threads(db)
    # Batch C.7 — drop ``solver_log_level`` parameter.
    # User-stored value DROPPED.  Audit confirmed the
    # field was loaded into SolverConfig.log_level but
    # never consumed by any engine module.  Use the new
    # --solver-log-level CLI flag (silent / normal /
    # verbose → HiGHS output_flag + log_dev_level); GUI
    # control deferred to v56 follow-up PR (task #26).
    # Also removes the dedicated ``solver_log_levels``
    # parameter_value_list.
    _migrate_v56_remove_solver_log_level(db)
    # Batch C.8 — drop ``solver_time_limit`` parameter.
    # User-stored value DROPPED.  Use the new
    # --solver-time-limit CLI flag (plumbed via the
    # existing FLEXTOOL_HIGHS_TIME_LIMIT env var that the
    # orchestrator's CLI-overrides builder already
    # consults — maps to HiGHS ``time_limit`` key); GUI
    # control deferred to v56 follow-up PR (task #26).
    _migrate_v56_remove_solver_time_limit(db)
    # Batch C.9 — drop ``solver_io_api`` parameter.
    # User-stored value DROPPED.  Use the new
    # --matrix-file-format {mps,lp} CLI flag (plumbed via
    # FLEXTOOL_MATRIX_FILE_FORMAT env var into
    # SolveConfig.load_from_db, overriding the per-solve
    # SolverConfig.io_api default).  The in-process vs.
    # file decision is implicit: HiGHS + no --save-memory
    # stays on the direct binding; commercial solvers
    # always write a file using the chosen format.
    # Also removes the dedicated ``solver_io_apis``
    # parameter_value_list.
    _migrate_v56_remove_solver_io_api(db)
    # Batch C.10 — drop ``use_row_scaling`` parameter.
    # User-stored value DROPPED.  Use --scaling CLI flag
    # (already exists from the autoscale work); the
    # FLEXTOOL_FORCE_ROW_SCALING env-var test hook also
    # remains.  Engine side hard-wires the per-solve dict
    # to {} so every solve emits p_use_row_scaling=0 (the
    # default branch), preserving the Mode A
    # pre-scaling behaviour.  After this commit only the
    # five intended solver params remain on the solve
    # entity class: solver, solve_mode, solver_arguments,
    # solver_mip_gap, solver_precommand.
    _migrate_v56_remove_use_row_scaling(db)
    # Batch D.1 — add ``none`` off-member to
    # ``profile_methods`` and ``ramp_methods`` value-lists,
    # retype the four ``parameter_definition`` defaults
    # (connection__profile.profile_method,
    # node__profile.profile_method,
    # unit__inputNode.ramp_method,
    # unit__outputNode.ramp_method) to ``"none"``, and
    # backfill an explicit ``method='none'`` row on every
    # legacy entity that did not author a value.  Preserves
    # the pre-D.1 silent "no method" semantics on legacy
    # data and makes the off-state explicit going forward.
    # ``unit__node__profile.profile_method`` is handled
    # separately in D.2 because it gets a different new
    # default (``upper_limit``); see
    # ``_migrate_v56_set_unit_node_profile_default_upper_limit``.
    none_before = _v56_none_members(db)
    _migrate_v56_add_profile_and_ramp_method_none(db, none_before)
    # Batch D.2 — flip
    # ``unit__node__profile.profile_method`` default from
    # ``null`` to ``upper_limit`` (the dominant authoring
    # case for this relationship class) and backfill
    # legacy entities lacking an explicit method to
    # ``none`` so the new default does not silently
    # activate the constraint on pre-D.2 data.
    _migrate_v56_set_unit_node_profile_default_upper_limit(db, none_before)
    # Batch D.3 — retype six boolean-as-enum parameters
    # (connection.is_DC, group.has_capacity_margin,
    # group.has_inertia, group.has_non_synchronous,
    # unit__inputNode.is_non_synchronous,
    # unit__outputNode.is_non_synchronous) from their
    # dedicated single-``yes`` value-lists to the shared
    # ``yes_no`` list; set default to ``"no"`` on each;
    # drop the five obsolete single-member value-lists.
    # Existing ``yes`` parameter_value rows are preserved
    # unchanged because ``yes`` is also a member of
    # ``yes_no``.
    _migrate_v56_retype_yes_only_to_yes_no(db)
    # Batch D.4 — drop the redundant ``no`` member from
    # the ``storage_nested_fix_method`` value-list (the
    # canonical off-name is ``fix_nothing``, matching the
    # active-member naming convention on the same list).
    # Existing ``no`` parameter_value rows are rewritten
    # to ``fix_nothing`` so the user's intent is preserved
    # under the canonical spelling.  The parameter_definition
    # default is already ``fix_nothing`` — unchanged.
    _migrate_v56_drop_storage_nested_fix_method_no(db)
    # Batch D.7 — rename ``co2_methods.no_method`` to
    # ``co2_methods.none`` so the off-member on the
    # ``co2_methods`` value-list aligns with every other
    # off-style member used by the schema
    # (``conversion_methods.none``,
    # ``decomposition_methods.none``,
    # ``minimum_time_methods.none``, ``profile_methods.none``,
    # ``ramp_methods.none``, …).  Adds the new member,
    # rewrites every ``group.co2_method = no_method``
    # parameter_value row to ``"none"``, drops the legacy
    # ``no_method`` list_value, and sets the
    # parameter_definition default to ``"none"``.  Engine
    # semantics are unchanged: every consumer in
    # :mod:`flextool.engine.preprocess.emit._co2_accumulators`
    # filters by exact-match on the active method names
    # (``total`` / ``price_total`` / ``period_total`` /
    # ``price_period_total``); ``no_method`` and ``none``
    # both fall through identically.
    _migrate_v56_rename_co2_methods_no_method_to_none(db)
    # Batch F Option A — re-add ``is_enabled`` parameter on
    # the three classes whose pre-v56 gating used
    # entity_alternative.active.  Materialises explicit
    # parameter_values from existing entity_alternative
    # rows, backfills Base ``is_enabled="no"`` for the two
    # reserve classes whose ``active_by_default`` flips
    # False → True, drops the entity_alternative rows for
    # the three classes, and removes the orphan
    # ``is_active`` value list.  Named ``is_enabled`` (not
    # ``is_active``) to bypass spinedb_api's
    # compatibility.py shim which auto-collapses
    # ``is_active`` parameter_values back to
    # entity_alternative on every commit.  Engine wire-up
    # lives in SpineDBBackend.find_entities / entities /
    # parameter_values and SpineDbReader.
    _migrate_v56_reactivate_is_enabled_parameter(db)


def _v56_none_members(db) -> frozenset[str]:
    """Which of ``profile_methods`` / ``ramp_methods`` have the ``none`` member v56 D.1 adds (read before D.1)."""
    return frozenset(name for name in ("profile_methods", "ramp_methods")
                     if "none" in {lv["parsed_value"] for lv in db.find_list_values(parameter_value_list_name=name)})


def _v56_backfilled(db, entity_class_name: str, name: str, target: str, none_before: frozenset[str]) -> bool:
    """The default of *entity_class_name.name* is *target* and its list had ``none`` before D.1: the default and the
    ``none`` backfill (one commit) are done, and a rerun must not backfill the entities made since."""
    pdef = db.get_item("parameter_definition", entity_class_name=entity_class_name, name=name)
    if (not pdef or pdef["parameter_value_list_name"] not in none_before
            or from_database(pdef["default_value"], pdef["default_type"]) != target):
        return False
    return _already_applied(56, f"{entity_class_name}.{name} default is {target}")


def _v56_checked(db, item_type: str, **fields) -> None:
    """``add_update_item`` that raises on the error it returns (a v56 ``Base`` write must not fail unseen)."""
    error = db.add_update_item(item_type, **fields)[2]
    if error:
        raise SpineDBAPIError(f"v56: could not write the {item_type} {fields}: {error}")


def _v56_ensure_base(db) -> None:
    """Make alternative ``Base`` when the database has none (item 292: a database from the pre-v26 templates), or
    put an empty ``Base`` that no scenario uses into the scenarios (the one spinedb_api makes next to a template's
    ``base``): either way the v56 values written into it would apply in no scenario.

    ``Base`` goes to the lowest rank of every scenario: one below the scenario's lowest (after the v38 floor), rank 1
    in a scenario without alternatives.  So the v56 values in it apply in every scenario and a value in any other
    alternative wins.  A ``Base`` with values, entity_alternative rows or a scenario is left as it is (adding it to
    a scenario would change the results of its own values).  The rows ride on the calling step's commit.
    """
    if db.get_item("alternative", name="Base"):
        if (db.find_scenario_alternatives(alternative_name="Base") or db.find_parameter_values(alternative_name="Base")
                or db.find_entity_alternatives(alternative_name="Base")):
            return
    else:
        _v56_checked(db, "alternative", name="Base", description=(
            "Made by the upgrade to data version 56: holds the values that keep the behaviour of the entities made "
            "before it. At the lowest rank of every scenario."))
    lowest = {scenario["name"]: 2 for scenario in db.find_scenarios()}
    for sa in db.find_scenario_alternatives():
        lowest[sa["scenario_name"]] = min(lowest[sa["scenario_name"]], sa["rank"])
    for name, rank in lowest.items():
        _v56_checked(db, "scenario_alternative", scenario_name=name, alternative_name="Base", rank=rank - 1)


def _v56_write_base(db, **value) -> None:
    """Write the ``parameter_value`` *value* into ``Base`` (:func:`_v56_ensure_base` first): v56 D.1, D.2 and
    Batch F keep there the older behaviour of the entities without a value."""
    _v56_ensure_base(db)
    _v56_checked(db, "parameter_value", alternative_name="Base", **value)


def _migrate_v56_add_profile_and_ramp_method_none(db, none_before: frozenset[str] = frozenset()) -> None:
    """Batch D.1 — introduce a ``none`` off-member for the
    ``profile_methods`` and ``ramp_methods`` value-lists, set the four
    user-facing defaults to ``none``, and backfill an explicit
    ``method = none`` row on every legacy entity that did not author a
    value.

    The pre-D.1 schema declared ``connection__profile.profile_method``,
    ``node__profile.profile_method``, ``unit__inputNode.ramp_method``
    and ``unit__outputNode.ramp_method`` with default ``null`` and no
    "off" member on the corresponding value-list.  Engine consumers
    treat an unset row as "no method" (the parameter is silently
    skipped); D.1 makes that semantics explicit by introducing
    ``profile_methods.none`` / ``ramp_methods.none`` and routing the
    schema default through it.  Legacy entities that lacked a value
    must be backfilled to ``none`` so the new schema default (``none``)
    does not silently change their behaviour now and a future flip of
    the default to a non-``none`` member does not silently activate the
    method on those legacy entities.

    Step (a) — extend the two value-lists with the new ``none`` member.
    Idempotent via :func:`add_value_list_manual` (no-op if the row
    already exists).

    Step (b) — set the four ``parameter_definition`` defaults to
    ``none``.

    Step (c) — backfill an explicit ``method = none`` row on every
    entity that owns one of the four classes and has zero
    ``parameter_value`` rows for the relevant method parameter (across
    every alternative).  Writes to the ``Base`` alternative (made by
    :func:`_v56_ensure_base` when the database has none).

    Sister migration :func:`_migrate_v56_set_unit_node_profile_default_upper_limit`
    handles ``unit__node__profile.profile_method`` separately because
    its new default is ``upper_limit`` (not ``none``); the backfill
    contract there still writes ``none`` to preserve legacy "no method"
    semantics.

    Verification: re-read the four defaults and confirm each is
    ``"none"``; re-read each backfilled entity and confirm a
    ``parameter_value`` row exists.  Mismatch raises
    :class:`SpineDBAPIError`.

    *none_before*: :func:`_v56_none_members` before this step; a target found done (:func:`_v56_backfilled`, a rerun
    after a kill) is not backfilled again.
    """
    # ---- Step (a): extend profile_methods / ramp_methods ------------
    add_value_list_manual(
        db,
        [["profile_methods", "none"], ["ramp_methods", "none"]],
    )

    none_value, none_type = to_database("none")

    targets: tuple[tuple[str, str], ...] = (
        ("connection__profile", "profile_method"),
        ("node__profile",       "profile_method"),
        ("unit__inputNode",     "ramp_method"),
        ("unit__outputNode",    "ramp_method"),
    )

    done = {target for target in targets if _v56_backfilled(db, *target, "none", none_before)}

    # ---- Step (b): retype defaults to 'none' ------------------------
    for entity_class_name, name in targets:
        db.add_update_item(
            "parameter_definition",
            entity_class_name=entity_class_name,
            name=name,
            default_value=none_value,
            default_type=none_type,
        )

    # ---- Step (c): backfill legacy entities -------------------------
    backfilled_count = 0
    for entity_class_name, name in targets:
        if (entity_class_name, name) in done:
            continue
        entities_with_value = {
            pv["entity_byname"]
            for pv in db.find_parameter_values(
                entity_class_name=entity_class_name,
                parameter_definition_name=name,
            )
        }
        for ent in db.find_entities(entity_class_name=entity_class_name):
            byname = ent["entity_byname"]
            if byname in entities_with_value:
                continue
            _v56_write_base(db, entity_class_name=entity_class_name, entity_byname=byname,
                            parameter_definition_name=name, value=none_value, type=none_type)
            backfilled_count += 1

    # ---- Verification ------------------------------------------------
    parameter_definitions = db.mapped_table("parameter_definition")
    for entity_class_name, name in targets:
        defn = db.item(
            parameter_definitions,
            entity_class_name=entity_class_name,
            name=name,
        )
        if defn is None:
            raise SpineDBAPIError(
                f"v56 D.1: parameter_definition "
                f"{entity_class_name}.{name} not found after retype."
            )
        if from_database(defn["default_value"], defn["default_type"]) != "none":
            raise SpineDBAPIError(
                f"v56 D.1: default_value retype failed for "
                f"{entity_class_name}.{name}: got "
                f"{from_database(defn['default_value'], defn['default_type'])!r}."
            )

    _commit_step(
        db,
        "v56 D.1: added 'none' off-member to profile_methods and "
        "ramp_methods value-lists; set defaults to 'none' on "
        "connection__profile.profile_method, node__profile.profile_method, "
        "unit__inputNode.ramp_method, unit__outputNode.ramp_method; "
        f"backfilled {backfilled_count} legacy entity row(s) with an "
        "explicit method='none' value to preserve the pre-D.1 'no method "
        "set' semantics.",
    )


def _migrate_v56_set_unit_node_profile_default_upper_limit(db, none_before: frozenset[str] = frozenset()) -> None:
    """Batch D.2 — flip ``unit__node__profile.profile_method`` default
    from ``null`` to ``upper_limit`` and backfill legacy entities to
    ``none``.

    The ``unit__node__profile`` class is unusual: when a user creates
    one of these relationships they almost always want a per-time-step
    upper-bound profile (the dominant use-case — capping the unit's
    output to a node by a time series).  The pre-D.2 default of
    ``null`` forced every new entity to set ``profile_method``
    explicitly even for the dominant case.  D.2 makes ``upper_limit``
    the new default so that authoring a ``unit__node__profile`` row
    with no ``profile_method`` set lands on the dominant semantics.

    To prevent silent activation on legacy data: every existing
    ``unit__node__profile`` entity without an explicit
    ``profile_method`` row is backfilled with ``profile_method='none'``
    (the off-member D.1 introduced).  That preserves the pre-D.2
    "no method set" behaviour for those entities while the new default
    only kicks in for newly authored rows.

    Note: this is the third ``profile_method`` parameter in the schema
    (alongside ``connection__profile`` and ``node__profile`` covered
    in D.1).  We keep this in a separate helper because its post-D.2
    default differs (``upper_limit`` instead of ``none``); sharing a
    helper would obscure that.

    Verification: re-read the default and confirm it is
    ``"upper_limit"``.  Mismatch raises :class:`SpineDBAPIError`.

    *none_before* as in :func:`_migrate_v56_add_profile_and_ramp_method_none` (a user-set ``upper_limit`` default
    before v56 has no ``none`` member, so it still gets the backfill).
    """
    entity_class_name = "unit__node__profile"
    name = "profile_method"

    upper_limit_value, upper_limit_type = to_database("upper_limit")
    none_value, none_type = to_database("none")

    done = _v56_backfilled(db, entity_class_name, name, "upper_limit", none_before)

    # ---- Step (a): flip default to upper_limit ----------------------
    db.add_update_item(
        "parameter_definition",
        entity_class_name=entity_class_name,
        name=name,
        default_value=upper_limit_value,
        default_type=upper_limit_type,
    )

    # ---- Step (b): backfill legacy entities to 'none' ---------------
    entities_with_value = {
        pv["entity_byname"]
        for pv in db.find_parameter_values(
            entity_class_name=entity_class_name,
            parameter_definition_name=name,
        )
    }
    backfilled_count = 0
    for ent in db.find_entities(entity_class_name=entity_class_name) if not done else ():
        byname = ent["entity_byname"]
        if byname in entities_with_value:
            continue
        _v56_write_base(db, entity_class_name=entity_class_name, entity_byname=byname,
                        parameter_definition_name=name, value=none_value, type=none_type)
        backfilled_count += 1

    # ---- Verification ------------------------------------------------
    parameter_definitions = db.mapped_table("parameter_definition")
    defn = db.item(
        parameter_definitions,
        entity_class_name=entity_class_name,
        name=name,
    )
    if defn is None:
        raise SpineDBAPIError(
            "v56 D.2: parameter_definition "
            "unit__node__profile.profile_method not found after retype."
        )
    if from_database(defn["default_value"], defn["default_type"]) != "upper_limit":
        raise SpineDBAPIError(
            "v56 D.2: default_value retype failed for "
            "unit__node__profile.profile_method: got "
            f"{from_database(defn['default_value'], defn['default_type'])!r}."
        )

    _commit_step(
        db,
        "v56 D.2: set unit__node__profile.profile_method default to "
        f"'upper_limit'; backfilled {backfilled_count} legacy "
        "unit__node__profile entity row(s) with an explicit "
        "method='none' value to preserve the pre-D.2 'no method set' "
        "semantics on legacy data.",
    )


#: v56 D.3: the suffix of the definition that holds a parameter's values while its value list is rebound.
_V56_YES_NO_STAGE_SUFFIX = "__v56_yes_no_stage"


def _migrate_v56_retype_yes_only_to_yes_no(db) -> None:
    """Batch D.3 — retype the six boolean-as-enum parameters from their
    single-``yes`` dedicated value-lists to the shared ``yes_no``
    value-list, and drop the five obsolete single-member lists.

    Pre-D.3 the schema declared five dedicated value-lists each carrying
    a single member ``yes``:

    - ``is_DC``           (referenced by ``connection.is_DC``)
    - ``has_capacity_margin``   (``group.has_capacity_margin``)
    - ``has_inertia``           (``group.has_inertia``)
    - ``has_non_synchronous``   (``group.has_non_synchronous``)
    - ``is_non_synchronous``    (``unit__inputNode.is_non_synchronous``
                                 + ``unit__outputNode.is_non_synchronous``)

    That carried two costs: (1) the parameters could not be flipped to
    ``"no"`` through their value-list bindings even though the engine
    treats ``no`` as the off state; (2) the six similarly-shaped
    parameters did not share the same value-list, so schema reviewers
    had to read each list separately to confirm the enum domain.  D.3
    rebinds all six parameters to the shared ``yes_no`` list (which
    carries both ``yes`` and ``no``), sets each default to ``"no"``,
    and drops the five obsolete single-yes lists.

    Existing ``parameter_value`` rows are PRESERVED unchanged: every
    pre-D.3 value is the string ``"yes"`` (the only member of the old
    lists) which is also a valid member of ``yes_no``.  No rewrite is
    required.

    Step (a) — for each of the six parameters, ``add_update_item`` to
    rebind ``parameter_value_list_name`` to ``"yes_no"`` and set
    ``default_value`` to ``"no"``.

    Step (b) — drop the five obsolete value-lists via
    :func:`db.remove_items` on the matching
    ``parameter_value_list`` rows.  Safe after step (a) because no
    parameter_definition still references them.

    Step (c) — verification.  Re-read each of the six
    parameter_definitions and confirm both the value-list rebind and
    the ``"no"`` default.  Re-query
    ``parameter_value_list``/``list_value`` and confirm each of the
    five legacy lists is gone.  Re-query the surviving
    ``parameter_value`` rows for the six parameters and confirm each
    string value is still either ``"yes"`` or ``"no"`` (i.e. a member
    of the shared list).  Mismatch raises
    :class:`SpineDBAPIError`.
    """
    targets: tuple[tuple[str, str], ...] = (
        ("connection",       "is_DC"),
        ("group",            "has_capacity_margin"),
        ("group",            "has_inertia"),
        ("group",            "has_non_synchronous"),
        ("unit__inputNode",  "is_non_synchronous"),
        ("unit__outputNode", "is_non_synchronous"),
    )
    # A target with a later name (v65 has_capacity_margin -> capacity_margin_method) is retyped already.
    renamed = [(cls, name, _later_definition(db, 56, cls, name)) for cls, name in targets]
    for cls, name, later in renamed:
        if later is not None:
            _already_applied(56, f"{cls}.{name} is {later[0]}.{later[1]}, retyped already")
    targets = tuple((cls, name) for cls, name, later in renamed if later is None)
    legacy_lists: tuple[str, ...] = (
        "is_DC",
        "has_capacity_margin",
        "has_inertia",
        "has_non_synchronous",
        "is_non_synchronous",
    )

    no_value, no_type = to_database("no")

    # ---- Step (a): rebind each parameter_definition -----------------
    # spinedb_api refuses to mutate ``parameter_value_list_name`` on a
    # parameter that already has stored values ("can't modify the
    # parameter value list of a parameter that already has values").
    # The pre-D.3 lists each carry only ``"yes"`` so any saved value is
    # the string ``"yes"`` — also a valid member of ``yes_no``.  The
    # values are copied to a stage definition, removed, the
    # parameter_definition is rebound, the same rows are written back
    # from the stage and the stage is removed; the round-trip is
    # byte-stable.  The stage is in the database before the first
    # removal is committed (a large removal commits in batches), so a
    # rerun after an interruption restores every value from it.  In one
    # commit the stage is added and removed again and never reaches the
    # database.
    for entity_class_name, name in targets:
        stage_name = f"{name}{_V56_YES_NO_STAGE_SUFFIX}"
        if _pdef_or_none(db, entity_class_name, stage_name) is None:
            db.add_parameter_definition(entity_class_name=entity_class_name, name=stage_name)
            for pv in db.find_parameter_values(entity_class_name=entity_class_name,
                                               parameter_definition_name=name):
                db.add_parameter_value(
                    entity_class_name=entity_class_name, entity_byname=pv["entity_byname"],
                    parameter_definition_name=stage_name, alternative_name=pv["alternative_name"],
                    value=pv["value"], type=pv["type"])
        # A rerun finds, beside the stage, the values not yet removed or those already written back (all
        # copies of the stage's); the stage has every value until it is removed itself.
        saved = {}
        for source in (name, stage_name):
            for pv in db.find_parameter_values(entity_class_name=entity_class_name,
                                               parameter_definition_name=source):
                saved[(tuple(pv["entity_byname"]), pv["alternative_name"])] = (pv["value"], pv["type"])
        saved_values = [(byname, alt, raw_val, raw_type) for (byname, alt), (raw_val, raw_type) in saved.items()]
        _remove_items_bounded(
            db, "parameter_value",
            [pv["id"] for pv in db.find_parameter_values(
                entity_class_name=entity_class_name,
                parameter_definition_name=name,
            )],
            f"v56: removed the {entity_class_name}.{name} values to rebind its value list",
        )
        db.add_update_item(
            "parameter_definition",
            entity_class_name=entity_class_name,
            name=name,
            parameter_value_list_name="yes_no",
            default_value=no_value,
            default_type=no_type,
        )
        for entity_byname, alt_name, raw_val, raw_type in saved_values:
            db.add_update_item(
                "parameter_value",
                entity_class_name=entity_class_name,
                entity_byname=entity_byname,
                parameter_definition_name=name,
                alternative_name=alt_name,
                value=raw_val,
                type=raw_type,
            )
        _remove_items_bounded(
            db, "parameter_definition",
            [_pdef_or_none(db, entity_class_name, stage_name)["id"]],
            f"v56: removed the stage of {entity_class_name}.{name}",
        )

    # ---- Step (b): drop the five obsolete value-lists ---------------
    pvl_table = db.mapped_table("parameter_value_list")
    dropped: list[str] = []
    for legacy_name in legacy_lists:
        try:
            vl = db.item(pvl_table, name=legacy_name)
        except SpineDBAPIError:
            vl = None
        if vl is None:
            continue
        db.remove_items("parameter_value_list", vl["id"])
        dropped.append(legacy_name)

    # ---- Step (c): verification -------------------------------------
    parameter_definitions = db.mapped_table("parameter_definition")
    for entity_class_name, name in targets:
        defn = db.item(
            parameter_definitions,
            entity_class_name=entity_class_name,
            name=name,
        )
        if defn is None:
            raise SpineDBAPIError(
                f"v56 D.3: parameter_definition {entity_class_name}.{name} "
                "not found after retype."
            )
        if defn["parameter_value_list_name"] != "yes_no":
            raise SpineDBAPIError(
                f"v56 D.3: {entity_class_name}.{name} value-list rebind "
                "failed: got "
                f"{defn['parameter_value_list_name']!r}, expected 'yes_no'."
            )
        if from_database(defn["default_value"], defn["default_type"]) != "no":
            raise SpineDBAPIError(
                f"v56 D.3: {entity_class_name}.{name} default retype "
                "failed: got "
                f"{from_database(defn['default_value'], defn['default_type'])!r}, "
                "expected 'no'."
            )

    for legacy_name in legacy_lists:
        # ``db.item`` raises ``SpineDBAPIError`` when the item was
        # explicitly removed earlier in this session (vs returning
        # ``None`` for a never-present name); both outcomes mean
        # "list is gone", which is what we want.
        try:
            vl = db.item(pvl_table, name=legacy_name)
        except SpineDBAPIError:
            vl = None
        if vl is not None:
            raise SpineDBAPIError(
                f"v56 D.3: legacy value-list {legacy_name!r} still "
                "present after removal."
            )

    for entity_class_name, name in targets:
        for pv in db.find_parameter_values(
            entity_class_name=entity_class_name,
            parameter_definition_name=name,
        ):
            if pv["type"] != "str":
                raise SpineDBAPIError(
                    f"v56 D.3: {entity_class_name}.{name} row "
                    f"{pv['entity_byname']!r}/{pv['alternative_name']!r} "
                    f"has non-str type {pv['type']!r} after retype."
                )
            if pv["parsed_value"] not in {"yes", "no"}:
                raise SpineDBAPIError(
                    f"v56 D.3: {entity_class_name}.{name} row "
                    f"{pv['entity_byname']!r}/{pv['alternative_name']!r} "
                    f"carries value {pv['parsed_value']!r}, expected "
                    "'yes' or 'no'."
                )

    _commit_step(
        db,
        "v56 D.3: retyped six boolean-as-enum parameters to the shared "
        "yes_no value-list (connection.is_DC, group.has_capacity_margin, "
        "group.has_inertia, group.has_non_synchronous, "
        "unit__inputNode.is_non_synchronous, "
        "unit__outputNode.is_non_synchronous); set each default to 'no' "
        "and dropped the five obsolete single-yes value-lists: "
        f"{sorted(dropped)!r}.  Existing 'yes' parameter_value rows "
        "are preserved (still valid under yes_no).",
    )


def _migrate_v56_drop_storage_nested_fix_method_no(db) -> None:
    """Batch D.4 — drop the redundant ``no`` member from the
    ``storage_nested_fix_method`` value-list; rewrite any existing
    ``no`` parameter_value rows to ``fix_nothing``.

    Pre-D.4 the value-list carried two members that meant the same
    thing — ``fix_nothing`` and ``no``.  ``fix_nothing`` is the
    canonical off-name (matches the surrounding off-vs-active member
    naming on the same list: ``fix_quantity`` / ``fix_price`` /
    ``fix_usage`` for the active modes).  ``no`` was a legacy artefact;
    keeping both lets two authors author the same intent two different
    ways.  D.4 collapses them: rewrite any ``no`` rows to
    ``fix_nothing`` (preserves the user's intent under the canonical
    spelling), then drop ``no`` from the list.

    The parameter_definition default is already ``fix_nothing`` (the
    canonical off-name) and stays unchanged.

    Step (a) — rewrite parameter_value rows.  Mirror the v55
    storage_binding_rename rewrite: for each ``no`` row, write a
    fresh ``fix_nothing`` value via ``db.update_parameter_value``.
    Step (b) — drop the ``no`` list_value.  Same pattern as
    :func:`_migrate_v55_storage_binding_rename_and_extend` (locate
    the list_value row by encoded value bytes, ``remove_item``).
    """
    rewritten = 0
    for pv in list(db.find_parameter_values(
        entity_class_name="node",
        parameter_definition_name="storage_nested_fix_method",
    )):
        if pv["type"] != "str":
            continue
        if pv["parsed_value"] != "no":
            continue
        new_value_bytes, new_value_type = to_database("fix_nothing")
        db.update_parameter_value(
            id=pv["id"],
            value=new_value_bytes,
            type=new_value_type,
        )
        rewritten += 1

    no_bytes, _ = to_database("no")
    dropped = False
    pending = _detach_list_defaults(db, "storage_nested_fix_method", {"no": "fix_nothing"})
    for lv in list(db.find_list_values(
        parameter_value_list_name="storage_nested_fix_method",
    )):
        if lv["value"] == no_bytes:
            db.remove_item("list_value", lv["id"])
            dropped = True
            break
    _attach_list_defaults(db, pending)

    _commit_step(
        db,
        "v56 D.4: dropped redundant 'no' member from the "
        "storage_nested_fix_method value-list (collapses to canonical "
        f"'fix_nothing' off-name); rewrote {rewritten} legacy 'no' "
        "parameter_value row(s) on node.storage_nested_fix_method to "
        f"'fix_nothing'.  list_value dropped: {dropped}.",
    )


def _migrate_v56_rename_co2_methods_no_method_to_none(db) -> None:
    """Batch D.7 — rename ``co2_methods.no_method`` → ``co2_methods.none``.

    The ``co2_methods`` value-list shipped with ``no_method`` as its
    off-member while every other "off"-style member on the schema uses
    ``none``:

    * ``conversion_methods.none``
    * ``decomposition_methods.none``
    * ``minimum_time_methods.none``
    * ``profile_methods.none`` (added in D.1)
    * ``ramp_methods.none`` (added in D.1)
    * ``reserve_methods.no_reserve`` (a sister off-name not touched here)

    D.7 collapses the inconsistency by renaming the
    ``co2_methods.no_method`` member to ``co2_methods.none`` so the
    schema's off-vocabulary is uniform.  Audit reference:
    ``_audit_reports/v56_method_none_audit.md``.

    Steps (mirrors the D.4 ``no -> fix_nothing`` pattern, plus a default
    flip and an explicit add-before-drop ordering to keep the existing
    ``group.co2_method = no_method`` parameter_value rows valid against
    the value-list at every intermediate state):

    1. Add the new ``co2_methods.none`` member.
    2. Rewrite every ``group.co2_method = no_method`` parameter_value
       row to value ``"none"``.
    3. Drop the legacy ``co2_methods.no_method`` list_value (safe now
       because step 2 cleared every parameter_value reference).
    4. Update the ``group.co2_method`` parameter_definition default to
       ``"none"`` so newly created entities land on the canonical
       off-name.

    Engine consumer behaviour is unchanged: the only filter site is
    :func:`flextool.engine.preprocess.emit._co2_accumulators._emitting_groups`,
    which keeps groups whose method is in
    ``{"total", "price_total", "period_total", "price_period_total"}``;
    ``no_method`` and ``none`` both fall through identically.
    """
    # ---- Step 1: add the new ``none`` member -----------------------------
    add_value_list_manual(db, [["co2_methods", "none"]])

    # ---- Step 2: rewrite ``no_method`` parameter_value rows to ``none`` --
    rewritten = 0
    none_bytes, none_type = to_database("none")
    for pv in list(db.find_parameter_values(
        entity_class_name="group",
        parameter_definition_name="co2_method",
    )):
        if pv["type"] != "str":
            continue
        if pv["parsed_value"] != "no_method":
            continue
        db.update_parameter_value(
            id=pv["id"],
            value=none_bytes,
            type=none_type,
        )
        rewritten += 1

    # ---- Step 3: drop the legacy ``no_method`` list_value ----------------
    no_method_bytes, _ = to_database("no_method")
    dropped = False
    pending = _detach_list_defaults(db, "co2_methods", {"no_method": "none"})
    for lv in list(db.find_list_values(
        parameter_value_list_name="co2_methods",
    )):
        if lv["value"] == no_method_bytes:
            db.remove_item("list_value", lv["id"])
            dropped = True
            break
    _attach_list_defaults(db, pending)

    # ---- Step 4: flip the parameter_definition default to ``none`` -------
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="co2_method",
        default_value=none_bytes,
        default_type=none_type,
    )

    # ---- Verification ---------------------------------------------------
    parameter_definitions = db.mapped_table("parameter_definition")
    defn = db.item(
        parameter_definitions,
        entity_class_name="group",
        name="co2_method",
    )
    if defn is None:
        raise SpineDBAPIError(
            "v56 D.7: parameter_definition group.co2_method not found "
            "after rename."
        )
    if from_database(defn["default_value"], defn["default_type"]) != "none":
        raise SpineDBAPIError(
            "v56 D.7: default_value retype failed for "
            "group.co2_method: got "
            f"{from_database(defn['default_value'], defn['default_type'])!r}."
        )

    _commit_step(
        db,
        "v56 D.7: renamed co2_methods.no_method -> co2_methods.none for "
        "consistency with the other off-style value-list members "
        "(conversion_methods/decomposition_methods/minimum_time_methods/"
        "profile_methods/ramp_methods all use 'none'); rewrote "
        f"{rewritten} legacy group.co2_method='no_method' "
        f"parameter_value row(s) to 'none'.  list_value dropped: "
        f"{dropped}.  parameter_definition default set to 'none'.",
    )


def _migrate_v56_reactivate_is_enabled_parameter(db) -> None:
    """Batch F Option A — re-add ``is_enabled`` on three entity classes
    that previously used the ``entity_alternative.active`` pattern.

    Affected classes (parameter group in parentheses):

    - ``constraint``                           (constraint)
    - ``reserve__upDown__unit__node``          (reserve)
    - ``reserve__upDown__connection__node``    (reserve)

    Why a NEW parameter (not ``is_active``)?
    ----------------------------------------

    spinedb_api's ``compatibility.py`` runs
    ``convert_tool_feature_method_to_entity_alternative`` on every
    ``commit_session``: it scans parameter_values named exactly
    ``is_active`` and auto-collapses them back into
    ``entity_alternative.active`` rows.  Naming our replacement
    ``is_enabled`` bypasses that shim.  Verified in attempt #1 (named
    ``is_active``: values vanished on commit) vs attempt #2 (named
    ``is_enabled``: values survived).

    Why migrate at all?
    -------------------

    Pre-v56 the constraint and reserve activation lived on the
    Entity Alternative tab — invisible from the parameter table, easy
    to miss when reviewing inputs, and forced an ``active_by_default
    = False`` schema on the two reserve classes (so any new reserve
    relationship was disabled until the user added an
    ``entity_alternative`` row).  Replacing the gate with an explicit
    ``is_enabled = yes/no`` parameter:

    * surfaces the on/off state in the parameter table where every
      other entity-scoped flag lives;
    * lets the two reserve classes flip to ``active_by_default =
      True`` (their natural default — declared reserves are usually
      meant to be active);
    * ends ``constraint.active_by_default`` True as well (it is True
      already except in a database from before the SpineDB entity
      schema, v3.0.0 .. v3.5.1, where spinedb_api's ``is_active``
      conversion left it False: item 1023).

    Class-specific behaviour
    ------------------------

    For **all three** classes:

    (a) Every ``entity_alternative`` row of the class is materialised
        as an explicit ``is_enabled`` parameter_value in the same
        alternative: ``yes`` if ``active=True``, ``no`` if
        ``active=False``.
    (b) After (a), every ``entity_alternative`` row of the class is
        dropped — the old gate is retired.

    For each of the three classes whose ``active_by_default`` is False
    when the step starts (the two reserve classes; ``constraint`` too in
    a database from before the SpineDB entity schema) **only**:

    (c) For each entity in the class that has no Base
        ``entity_alternative`` row pre-migration, write Base
        ``is_enabled="no"``.  Preserves the legacy semantic ``no
        row → inactive`` across the default flip (``Base`` made by
        :func:`_v56_ensure_base` when the database has none).  A class
        that is True already does NOT get this backfill: the
        post-migration ``is_enabled`` default ``"yes"`` matches.
    (d) Flip ``active_by_default`` to True on the entity_class.

    The schema-template JSON adds the three ``parameter_definitions``
    rows (with default ``"yes"`` against the shared ``yes_no``
    value-list) and the three ``parameter_types`` str-scalar rows in
    the same commit; the orphan ``is_active`` value list entry is
    dropped at the same time.

    Engine wire-up
    --------------

    :meth:`flextool.spinedb_backend._backend.SpineDBBackend.find_entities`
    / :meth:`SpineDBBackend.entities` / :meth:`parameter_values`
    post-filter the three classes by ``is_enabled != "no"`` resolved
    against the active scenario.  Likewise
    :class:`flextool.engine.input._spinedb_reader.SpineDbReader`
    drops disabled entities from its per-class caches at construction
    time.  Together these replace the entity_alternative-based
    gating spinedb_api's scenario_filter applied pre-v56.
    """
    affected_classes = (
        "constraint",
        "reserve__upDown__unit__node",
        "reserve__upDown__connection__node",
    )
    entity_class_table = db.mapped_table("entity_class")
    # The classes off by default when the step starts: on a rerun after a
    # kill, a class whose flip was committed is True and its backfill is
    # done (both ride on the step's last commit).
    classes_with_flip = tuple(
        cls_name for cls_name in affected_classes
        if (ec := db.item(entity_class_table, name=cls_name)) is not None
        and ec.get("active_by_default") is False
    )
    group_by_class = {
        "constraint": "constraint",
        "reserve__upDown__unit__node": "reserve",
        "reserve__upDown__connection__node": "reserve",
    }

    description = (
        "Whether the entity is enabled. Set to 'no' to disable "
        "without deleting the entity. Constant."
    )

    # ---- Step 1: ensure the parameter_definition exists on each class -
    yes_value, yes_type = to_database("yes")
    for cls_name in affected_classes:
        db.add_update_item(
            "parameter_definition",
            entity_class_name=cls_name,
            name="is_enabled",
            default_value=yes_value,
            default_type=yes_type,
            parameter_value_list_name="yes_no",
            description=description,
            parameter_type_list=("str",),
            parameter_group_name=group_by_class[cls_name],
        )

    # ---- Step 2: walk entity_alternative rows ------------------------
    # Per affected class: collect every (entity_byname, alt, active),
    # then materialise an is_enabled parameter_value mirroring the
    # active flag.  Also track per-class Base coverage so the backfill
    # (step 3) can fill the gaps.
    yes_no_values = {
        True: to_database("yes"),
        False: to_database("no"),
    }
    base_alt_covered: dict[str, set[tuple]] = {
        cls: set() for cls in affected_classes
    }
    materialised_count = 0
    for cls_name in affected_classes:
        # ``list(...)`` snapshot so the subsequent ``remove_items``
        # iteration is safe.
        ea_rows = list(
            db.find_entity_alternatives(entity_class_name=cls_name),
        )
        for ea in ea_rows:
            byname = ea["entity_byname"]
            alt = ea["alternative_name"]
            active = bool(ea.get("active", True))
            value_bytes, value_type = yes_no_values[active]
            db.add_update_item(
                "parameter_value",
                entity_class_name=cls_name,
                entity_byname=byname,
                parameter_definition_name="is_enabled",
                alternative_name=alt,
                value=value_bytes,
                type=value_type,
            )
            materialised_count += 1
            if alt == "Base":
                base_alt_covered[cls_name].add(tuple(byname))

    # ---- Step 3: Base is_enabled="no" backfill for the classes that ---
    # flip.  Mirrors the pre-migration "no entity_alternative row =
    # inactive" semantic across the active_by_default flip.
    backfilled_count = 0
    no_value, no_type = yes_no_values[False]
    # An entity that already has a Base is_enabled value is covered too:
    # before v56 only this step writes one, so it is the value an earlier
    # (interrupted or later-failed) run of the step materialised from a
    # Base entity_alternative row that is gone now.
    for cls_name in classes_with_flip:
        covered = base_alt_covered[cls_name] | {
            tuple(pv["entity_byname"]) for pv in db.find_parameter_values(
                entity_class_name=cls_name, parameter_definition_name="is_enabled", alternative_name="Base")}
        for ent in db.find_entities(entity_class_name=cls_name):
            byname = ent["entity_byname"]
            if tuple(byname) in covered:
                continue
            _v56_write_base(db, entity_class_name=cls_name, entity_byname=byname,
                            parameter_definition_name="is_enabled", value=no_value, type=no_type)
            backfilled_count += 1

    # ---- Step 4: drop entity_alternative rows for the three classes --
    dropped_ea_count = 0
    for cls_name in affected_classes:
        ea_rows = list(
            db.find_entity_alternatives(entity_class_name=cls_name),
        )
        _remove_items_bounded(db, "entity_alternative", [ea["id"] for ea in ea_rows],
                              f"v56: removed the {cls_name} entity_alternative rows")
        dropped_ea_count += len(ea_rows)

    # ---- Step 5: flip active_by_default on the classes off by default ---
    entity_class_table = db.mapped_table("entity_class")
    flipped_classes: list[str] = []
    for cls_name in classes_with_flip:
        ec = db.item(entity_class_table, name=cls_name)
        if ec is None:
            continue
        if ec.get("active_by_default") is False:
            db.add_update_item(
                "entity_class",
                name=cls_name,
                active_by_default=True,
            )
            flipped_classes.append(cls_name)

    # ---- Step 6: drop the ``is_active`` value list ------------------
    # In a database made from a FlexTool 3 shipped file both reserve classes still have their ``is_active``
    # definition on it (v19's ``remove_parameters_manual`` searches object classes only).  Removing the list removes
    # them, before this step's commit runs spinedb_api's ``is_active`` transformations.  Safe when absent.
    pvl_table = db.mapped_table("parameter_value_list")
    dropped_is_active_vl = False
    try:
        vl = db.item(pvl_table, name="is_active")
    except SpineDBAPIError:
        vl = None
    if vl is not None:
        db.remove_items("parameter_value_list", vl["id"])
        dropped_is_active_vl = True

    # ---- Verification -----------------------------------------------
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls_name in affected_classes:
        defn = db.item(
            parameter_definitions,
            entity_class_name=cls_name,
            name="is_enabled",
        )
        if defn is None:
            raise SpineDBAPIError(
                f"v56 Batch F: parameter_definition {cls_name}.is_enabled "
                "missing after migration."
            )
        if defn["parameter_value_list_name"] != "yes_no":
            raise SpineDBAPIError(
                f"v56 Batch F: {cls_name}.is_enabled bound to value-list "
                f"{defn['parameter_value_list_name']!r}, expected 'yes_no'."
            )
        if from_database(defn["default_value"], defn["default_type"]) != "yes":
            raise SpineDBAPIError(
                f"v56 Batch F: {cls_name}.is_enabled default is "
                f"{from_database(defn['default_value'], defn['default_type'])!r}, "
                "expected 'yes'."
            )

    for cls_name in affected_classes:
        leftover = list(
            db.find_entity_alternatives(entity_class_name=cls_name),
        )
        if leftover:
            raise SpineDBAPIError(
                f"v56 Batch F: {len(leftover)} entity_alternative row(s) "
                f"remain for {cls_name} after migration."
            )

    for cls_name in classes_with_flip:
        ec = db.item(entity_class_table, name=cls_name)
        if ec is None or ec.get("active_by_default") is not True:
            raise SpineDBAPIError(
                f"v56 Batch F: {cls_name}.active_by_default not flipped "
                "to True after migration."
            )

    _commit_step(
        db,
        "v56 Batch F (Option A): re-added is_enabled parameter on "
        "constraint + reserve__upDown__unit__node + "
        "reserve__upDown__connection__node (replaces the pre-v56 "
        "entity_alternative.active gating); materialised "
        f"{materialised_count} parameter_value(s) from "
        f"{dropped_ea_count} entity_alternative row(s); backfilled "
        f"Base is_enabled='no' on {backfilled_count} entity "
        "row(s) without a pre-migration Base row; flipped "
        f"active_by_default True on {flipped_classes!r}.  Orphan "
        f"is_active value list dropped: {dropped_is_active_vl}.",
    )
