"""Data version 56, first part: default fixes, ``*_coefficient`` renames, removed output flags and the folds into ``solver_arguments``."""
from spinedb_api import from_database, Map, SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import _commit_step, remove_parameters_manual


def _migrate_v56_remove_model_debug(db) -> None:
    """Drop the ``model.debug`` parameter from the schema.

    The parameter dates back to the legacy flextoolrunner / GAMS path
    that emitted ``input/debug.csv`` for the .mod file to consume.  In
    the engine rewrite (FlexTool v4) nothing reads it: the
    cl_pars emitter in :mod:`flextool.input_derivation._specs` still
    produced the CSV, but no downstream module touched the file.

    Debug-level control is now purely a runtime concern, exposed
    through:

    * ``flextool/cli/cmd_run_flextool.py`` — the tri-valued
      ``--debug={off,basic,full}`` flag, default ``off``, bare
      ``--debug`` → ``basic``.
    * ``flextool/gui/data_models.py`` — ``ProjectSettings.debug_level``
      persisted in ``settings.yaml`` and surfaced as a dropdown in the
      main window.

    Removing the DB parameter eliminates a silently-broken contract
    (any value users set was discarded) and reduces schema noise.

    Side effects: every ``parameter_value`` row referencing
    ``model.debug`` is dropped alongside the ``parameter_definition``
    when ``remove_parameters_manual`` invokes ``db.remove_items``
    (cascading delete is handled by spinedb_api).
    """
    remove_parameters_manual(db, [["model", "debug"]])


def _migrate_v56_add_group_cumulative_capacity_descriptions(db) -> None:
    """Populate the missing description text on the two
    ``group.cumulative_*_capacity`` parameter_definitions.

    Both parameters have existed in the schema since the v22 migration
    that introduced cumulative-capacity bounds across
    ``node``/``connection``/``unit``/``group``.  That migration block
    seeded ``description`` for the first three entity classes but not
    for ``group``, so any database initialised before this helper ran
    carries NULL/empty description text on those two rows.  The
    canonical phrasing already lives in
    ``flextool/schemas/spinedb_schema.json`` (and is mirrored in the
    quantity-type comments under
    ``flextool/engine/autoscale/_quantity_types.py``); this
    helper brings legacy databases in line.
    """
    db.update_item(
        "parameter_definition",
        entity_class_name="group",
        name="cumulative_max_capacity",
        description=(
            "[MW or MWh] Maximum cumulative capacity for a group of "
            "entities (considers existing, invested and retired "
            "capacity). Constant or period."
        ),
    )
    db.update_item(
        "parameter_definition",
        entity_class_name="group",
        name="cumulative_min_capacity",
        description=(
            "[MW or MWh] Minimum cumulative capacity for a group of "
            "entities (considers existing, invested and retired "
            "capacity). Constant or period."
        ),
    )
    _commit_step(
        db,
        "Populated description text on group.cumulative_max_capacity "
        "and group.cumulative_min_capacity (left blank by the v22 "
        "migration that introduced them).",
    )


def _migrate_v56_fix_wrong_defaults(db) -> None:
    """Fix ``default_value`` on seven ``parameter_definition`` rows
    whose schema-declared default disagrees with how the engine actually
    consumes the parameter.

    Audit: ``_audit_reports/v56_default_audit.md``.  Five of the rows
    (the ``rows_to_clear`` tuple below) were classified ``high`` and are
    cleared to ``(None, None)`` because the engine reads them via
    ``parameter_explicit`` (schema default silently dropped) or because
    the current default is a corrupt artefact.  The remaining two rows
    were originally classified ``medium`` and have since been resolved
    by user approval (see audit "high-confidence (resolved)" section):

    * ``model.inflation_offset_investment`` — current default ``1.0``,
      patched to ``0.0`` to match the engine fallback in
      :func:`flextool.engine.preprocess.params._discount._solve_inflation_scalars`
      and the symmetric fallback in ``_period_calc.py``
      (line 295).

    * ``commodity.unitsize`` — current default ``1.0``, patched to
      ``None``.  The engine reads via
      :func:`flextool.engine.preprocess.params._direct.p_commodity_unitsize_from_source`
      (``_entity_scalar_explicit``) so the schema default is dropped;
      the price-ladder consumer in
      :func:`flextool.engine.model._commodity_ladder._commodity_unitsize_param`
      substitutes ``1.0`` internally when the explicit Param is absent.
      The description is rewritten to name the gating feature
      (``commodity.price_method = price_ladder_*``) and explain the
      absent → identity behaviour.

    * ``reserve__upDown__connection__node.large_failure_ratio`` and
      ``reserve__upDown__unit__node.large_failure_ratio`` — currently
      carry an empty-string default (``""``).  The N-1 reserve consumer
      in :func:`flextool.engine.preprocess.emit._reserve_sets._compute_reserve_filters`
      gates each ``(p, r, ud, n)`` on
      ``p_prn.get((..., "large_failure_ratio"), 0.0) > 0`` — any
      non-zero enables the constraint, so the contract is "absent /
      0 / null = disabled, explicit positive value = enabled".  The
      empty string is a corrupt artefact for a ``float``-typed
      parameter; the sister rows ``increase_reserve_ratio`` on the
      same two classes already carry ``null, null``.

    * ``reserve__upDown__group.penalty_reserve`` — currently ``5000.0``.
      :func:`flextool.engine.preprocess.params._direct.p_reserve_upDown_group_penalty_reserve_from_source`
      reads via ``parameter_explicit`` and drops the broadcast default
      on the floor (the function's own docstring claims "None default
      — explicit rows only").  Soft-reserve violations enter the
      objective as ``vq_reserve * reservation * penalty * op_factor``
      (``_reserve.py``); the schema default of 5000 misleads users
      into thinking a soft-slack term is enabled by default when in
      fact no explicit row → no penalty term emitted.

    * ``reserve__upDown__connection__node.max_share`` — currently
      ``0.0``.  The consumer
      :func:`flextool.engine.preprocess.params._direct._process_reserve_node_param`
      also uses ``parameter_explicit``; the sister row on
      ``reserve__upDown__unit__node`` already has ``null, null``.

    * ``node.storage_state_start`` — currently ``0.0``.
      :func:`flextool.engine.preprocess.params._direct.p_state_start_from_source`
      reads explicit rows only; the docstring already claims
      "Default ``None`` (schema).".  If the schema default were
      actually honoured every storage node would be force-pinned to
      state 0 at the start of each rolling solve under the (also
      default) ``fix_start`` binding — a silent LP perturbation.

    The matching schema-template rows in
    ``flextool/schemas/spinedb_schema.json`` are updated in the same
    commit so a fresh v55 init lands on the corrected contract.
    """
    rows_to_clear: tuple[tuple[str, str], ...] = (
        ("reserve__upDown__connection__node", "large_failure_ratio"),
        ("reserve__upDown__unit__node",       "large_failure_ratio"),
        ("reserve__upDown__group",            "penalty_reserve"),
        ("reserve__upDown__connection__node", "max_share"),
        ("node",                              "storage_state_start"),
    )
    for entity_class_name, name in rows_to_clear:
        db.update_item(
            "parameter_definition",
            entity_class_name=entity_class_name,
            name=name,
            default_value=None,
            default_type=None,
        )

    # model.inflation_offset_investment — engine fallback is 0.0, not
    # the schema's 1.0.  Use to_database() per CONTRIBUTING.md to
    # encode the float default safely.
    inflation_default_value, inflation_default_type = to_database(0.0)
    db.update_item(
        "parameter_definition",
        entity_class_name="model",
        name="inflation_offset_investment",
        default_value=inflation_default_value,
        default_type=inflation_default_type,
    )

    # commodity.unitsize — clear the silently-dropped 1.0 default and
    # rewrite the description to name the price-ladder gate
    # (commodity.price_method = price_ladder_annual /
    # price_ladder_cumulative) and the absent → identity semantics.
    commodity_unitsize_description = (
        "Per-commodity scaling coefficient applied to the v_trade tier "
        "variable when the commodity uses the price-ladder feature "
        "(gated by commodity.price_method = price_ladder_annual or "
        "price_ladder_cumulative).  When set, v_trade is expressed in "
        "user-MWh divided by this value; pick the unitsize so the "
        "largest tier quantity sits at O(10) in the scaled LP.  The "
        "coefficient multiplies v_trade in the commodity_ladder_balance "
        "LHS, the per-tier cap LHS, and the per-tier objective term.  "
        "When absent (the default), the price-ladder consumer "
        "substitutes 1.0 internally so v_trade is in user-MWh "
        "(identity scaling).  Ignored entirely when "
        "commodity.price_method is not a price_ladder_* value."
    )
    db.update_item(
        "parameter_definition",
        entity_class_name="commodity",
        name="unitsize",
        default_value=None,
        default_type=None,
        description=commodity_unitsize_description,
    )

    _commit_step(
        db,
        "v56 wrong-default cleanup: cleared default_value/default_type on "
        "reserve__upDown__{connection,unit}__node.large_failure_ratio, "
        "reserve__upDown__group.penalty_reserve, "
        "reserve__upDown__connection__node.max_share, "
        "node.storage_state_start, and commodity.unitsize (the latter "
        "also gets a rewritten description naming the price-ladder "
        "gate); set model.inflation_offset_investment default to 0.0 "
        "to match the engine fallback.  See "
        "_audit_reports/v56_default_audit.md.",
    )


def _migrate_v56_rename_constraint_coefficient_to_coeff(db) -> None:
    """Rename the four user-constraint ``*_coefficient`` parameters to
    ``*_coeff`` on every entity class that declares them.

    The four parameters and their per-class footprint match the
    schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json``:

    * ``constraint_flow_coefficient`` →
      ``constraint_flow_coeff`` on
      ``connection__node`` / ``unit__inputNode`` / ``unit__outputNode``.
    * ``constraint_cumulative_pre_built_capacity_coefficient`` →
      ``constraint_cumulative_pre_built_capacity_coeff`` on
      ``connection`` / ``node`` / ``unit``.
    * ``constraint_invested_capacity_coefficient`` →
      ``constraint_invested_capacity_coeff`` on
      ``connection`` / ``node`` / ``unit``.
    * ``constraint_state_coefficient`` →
      ``constraint_state_coeff`` on ``node``.

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value,
    parameter_value_list, parameter_group, valid types) is preserved
    by passing ``description`` through unchanged.  Existing
    ``parameter_value`` rows that reference the old name follow the
    rename automatically because spinedb_api tracks the link by id,
    not by name.

    The engine frame attributes, autoscale quantity-type
    table, input_derivation cl_pars specs, export_to_tabular
    settings, and the docs are renamed in the same commit so the
    pipeline stays internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("connection",       "constraint_cumulative_pre_built_capacity_coefficient",
                             "constraint_cumulative_pre_built_capacity_coeff"),
        ("connection",       "constraint_invested_capacity_coefficient",
                             "constraint_invested_capacity_coeff"),
        ("connection__node", "constraint_flow_coefficient",
                             "constraint_flow_coeff"),
        ("node",             "constraint_cumulative_pre_built_capacity_coefficient",
                             "constraint_cumulative_pre_built_capacity_coeff"),
        ("node",             "constraint_invested_capacity_coefficient",
                             "constraint_invested_capacity_coeff"),
        ("node",             "constraint_state_coefficient",
                             "constraint_state_coeff"),
        ("unit",             "constraint_cumulative_pre_built_capacity_coefficient",
                             "constraint_cumulative_pre_built_capacity_coeff"),
        ("unit",             "constraint_invested_capacity_coefficient",
                             "constraint_invested_capacity_coeff"),
        ("unit__inputNode",  "constraint_flow_coefficient",
                             "constraint_flow_coeff"),
        ("unit__outputNode", "constraint_flow_coefficient",
                             "constraint_flow_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename constraint_*_coefficient -> constraint_*_coeff: "
        "connection.constraint_cumulative_pre_built_capacity_coefficient -> "
        "constraint_cumulative_pre_built_capacity_coeff; "
        "connection.constraint_invested_capacity_coefficient -> "
        "constraint_invested_capacity_coeff; "
        "connection__node.constraint_flow_coefficient -> "
        "constraint_flow_coeff; "
        "node.constraint_cumulative_pre_built_capacity_coefficient -> "
        "constraint_cumulative_pre_built_capacity_coeff; "
        "node.constraint_invested_capacity_coefficient -> "
        "constraint_invested_capacity_coeff; "
        "node.constraint_state_coefficient -> "
        "constraint_state_coeff; "
        "unit.constraint_cumulative_pre_built_capacity_coefficient -> "
        "constraint_cumulative_pre_built_capacity_coeff; "
        "unit.constraint_invested_capacity_coefficient -> "
        "constraint_invested_capacity_coeff; "
        "unit__inputNode.constraint_flow_coefficient -> "
        "constraint_flow_coeff; "
        "unit__outputNode.constraint_flow_coefficient -> "
        "constraint_flow_coeff.",
    )


def _migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff(db) -> None:
    """Rename ``flow_coefficient`` to ``conversion_flow_coeff`` on
    every entity class that declares it.

    Footprint matches the schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json`` — ``unit__inputNode`` and
    ``unit__outputNode``.  The new name shares the ``_coeff`` suffix
    with the four user-constraint coefficients renamed by
    :func:`_migrate_v56_rename_constraint_coefficient_to_coeff` and
    keeps the ``conversion_`` prefix that signals the parameter's
    role: it scales the conversion of input → output energy in the
    unit dispatch / node-balance / ``conversion_indirect`` equations
    (see flextool.mod:2557-2580 and the engine dispatch path
    in ``model/core/`` §F.4).

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value of 1.0,
    parameter_value_list, parameter_group ``basics``, valid types) is
    preserved.  Existing ``parameter_value`` rows that reference the
    old name follow the rename automatically because spinedb_api
    tracks the link by id, not by name.

    The engine frame attributes (``p_process_source_flow_coef``
    → ``p_process_source_conversion_flow_coeff`` and the sink
    counterpart), autoscale quantity-type table, input_derivation
    cl_pars specs, the CSV filename suffixes
    (``p_process_source_flow_coefficient.csv`` →
    ``p_process_source_conversion_flow_coeff.csv`` and sink), the
    pandas accessor names in ``process_outputs/read_parameters.py``,
    and the docs are renamed in the same commit so the pipeline stays
    internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("unit__inputNode",  "flow_coefficient", "conversion_flow_coeff"),
        ("unit__outputNode", "flow_coefficient", "conversion_flow_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename flow_coefficient -> conversion_flow_coeff: "
        "unit__inputNode.flow_coefficient -> conversion_flow_coeff; "
        "unit__outputNode.flow_coefficient -> conversion_flow_coeff.",
    )


def _migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff(db) -> None:
    """Rename ``max_capacity_coefficient`` to ``capacity_max_coeff`` on
    every entity class that declares it.

    Footprint matches the schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json`` — ``unit__inputNode`` and
    ``unit__outputNode``.  The shortened ``_coeff`` suffix aligns with
    the v56 convention introduced by
    :func:`_migrate_v56_rename_constraint_coefficient_to_coeff` and
    :func:`_migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff`.
    Reordering puts the noun ``capacity`` first and the qualifier
    ``max`` second, which groups the parameter alphabetically with the
    other capacity-related parameters on these classes.

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value of 1.0,
    parameter_value_list, parameter_group ``basics``, valid types) is
    preserved.  Existing ``parameter_value`` rows that reference the
    old name follow the rename automatically because spinedb_api
    tracks the link by id, not by name.

    The engine derived-param helpers
    (``_arc_max_capacity_coef_lf`` / ``_process_source_sink_coeff_zero_lf``),
    autoscale quantity-type table, input_derivation cl_pars specs,
    the CSV filename suffixes
    (``p_process_source_max_capacity_coefficient.csv`` →
    ``p_process_source_capacity_max_coeff.csv`` and sink), and the
    docs are renamed in the same commit so the pipeline stays
    internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("unit__inputNode",  "max_capacity_coefficient", "capacity_max_coeff"),
        ("unit__outputNode", "max_capacity_coefficient", "capacity_max_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename max_capacity_coefficient -> capacity_max_coeff: "
        "unit__inputNode.max_capacity_coefficient -> capacity_max_coeff; "
        "unit__outputNode.max_capacity_coefficient -> capacity_max_coeff.",
    )


def _migrate_v56_rename_min_capacity_coefficient_to_capacity_min_coeff(db) -> None:
    """Rename ``min_capacity_coefficient`` to ``capacity_min_coeff`` on
    every entity class that declares it.

    Footprint matches the schema-template snapshot under
    ``flextool/schemas/spinedb_schema.json`` — ``unit__inputNode`` and
    ``unit__outputNode``.  The shortened ``_coeff`` suffix aligns with
    the v56 convention introduced by
    :func:`_migrate_v56_rename_constraint_coefficient_to_coeff` and
    :func:`_migrate_v56_rename_flow_coefficient_to_conversion_flow_coeff`.
    Reordering puts the noun ``capacity`` first and the qualifier
    ``min`` second, which groups the parameter alphabetically with the
    other capacity-related parameters on these classes — directly
    after ``capacity_max_coeff`` (renamed by the sibling helper
    :func:`_migrate_v56_rename_max_capacity_coefficient_to_capacity_max_coeff`).

    Pure name change: every other column on the
    ``parameter_definition`` row (description, default value of 1.0,
    parameter_value_list, parameter_group ``basics``, valid types) is
    preserved.  Existing ``parameter_value`` rows that reference the
    old name follow the rename automatically because spinedb_api
    tracks the link by id, not by name.

    The autoscale quantity-type table, input_derivation cl_pars
    specs, the CSV filename suffixes
    (``p_process_source_min_capacity_coefficient.csv`` →
    ``p_process_source_capacity_min_coeff.csv`` and sink), and the
    docs are renamed in the same commit so the pipeline stays
    internally consistent.
    """
    renames: tuple[tuple[str, str, str], ...] = (
        ("unit__inputNode",  "min_capacity_coefficient", "capacity_min_coeff"),
        ("unit__outputNode", "min_capacity_coefficient", "capacity_min_coeff"),
    )
    parameter_definitions = db.mapped_table("parameter_definition")
    for cls, old_name, new_name in renames:
        # ``db.item()`` raises ``SpineDBAPIError`` (not None) when the
        # row doesn't exist; that's the steady-state once the schema
        # template JSON has been re-synced to the renamed names and a
        # fresh DB is bootstrapped from it.  Treat "row already renamed"
        # as idempotent — the helper must be safe to re-run.
        try:
            param = db.item(parameter_definitions,
                            entity_class_name=cls, name=old_name)
        except SpineDBAPIError:
            param = None
        if param:
            db.update_parameter_definition(
                id=param["id"],
                name=new_name,
                description=param.get("description"),
            )
    _commit_step(
        db,
        "v56 rename min_capacity_coefficient -> capacity_min_coeff: "
        "unit__inputNode.min_capacity_coefficient -> capacity_min_coeff; "
        "unit__outputNode.min_capacity_coefficient -> capacity_min_coeff.",
    )


def _migrate_v56_remove_exclude_entity_outputs(db) -> None:
    """Drop the ``model.exclude_entity_outputs`` parameter from the schema.

    The parameter was the single gate behind
    :func:`flextool.process_outputs.handoff_writers._exclude_entity_outputs_active`
    which short-circuited the three per-period capacity dumps
    (``unit_capacity.csv``, ``connection_capacity.csv``,
    ``node_capacity.csv``) whenever its value resolved to ``"yes"``.  The
    schema's default of ``"yes"`` made "exclude" the silent default for
    every database that did not override it explicitly, which inverted
    the intent of the parameter name ("exclude" reads as an opt-in but
    behaved as an opt-out).

    The user-facing semantic is now simply "always emit per-entity
    capacity rows".  Aggregated/group-level outputs continue to be
    controlled by the three ``group.output_*`` set selectors
    (``output_nodeGroup_dispatch``, ``output_nodeGroup_indicators``,
    ``output_flowGroup_indicators``) — those are unaffected.

    The gate site in :mod:`flextool.process_outputs.handoff_writers` is
    deleted in the same commit, along with the cl_pars emitter in
    :mod:`flextool.input_derivation._specs` (which produced
    ``input/exclude_entity_outputs.csv``, the only file the gate read),
    and the bookkeeping rows in
    :data:`flextool.spinedb_backend._backend.SET_LIKE_NAMES`,
    :mod:`flextool.engine.autoscale._quantity_types`,
    :mod:`flextool.export_to_tabular.export_settings` and the v44
    parameter_group membership map above.

    Side effects: every ``parameter_value`` row referencing
    ``model.exclude_entity_outputs`` is dropped alongside the
    ``parameter_definition`` when ``remove_parameters_manual`` invokes
    ``db.remove_items`` (cascading delete is handled by spinedb_api).
    """
    remove_parameters_manual(db, [["model", "exclude_entity_outputs"]])


def _migrate_v56_remove_output_node_balance_t(db) -> None:
    """Drop the ``model.output_node_balance_t`` parameter from the schema.

    Dead toggle: no module in :mod:`flextool.engine` reads the
    ``optional_outputs.csv`` row for this flag and no per-flag branch
    exists in the per-solve emitter
    (:mod:`flextool.engine.preprocess.emit._per_solve`).  Only
    ``output_horizon`` is checked from the ``enable_optional_outputs``
    set.  The remaining cl_pars entry in
    :mod:`flextool.input_derivation._specs` (the ``optional_outputs.csv``
    multi-param emitter) already does NOT include
    ``output_node_balance_t`` — this helper just clears the schema row,
    the SET_LIKE_NAMES bookkeeping entry, the autoscale quantity-type
    table row and the ``export_settings.yaml`` params list entry.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_node_balance_t`` is dropped alongside the
    ``parameter_definition`` when ``remove_parameters_manual`` invokes
    ``db.remove_items`` (cascading delete is handled by spinedb_api).
    """
    remove_parameters_manual(db, [["model", "output_node_balance_t"]])


def _migrate_v56_remove_output_ramp_envelope(db) -> None:
    """Drop the ``model.output_ramp_envelope`` parameter from the schema.

    Dead toggle: the flag IS plumbed into the multi-param
    ``optional_outputs.csv`` emitter in
    :mod:`flextool.input_derivation._specs`, but nothing on the engine
    side reads its row from the resulting ``enable_optional_outputs``
    set — only ``output_horizon`` is checked in
    :mod:`flextool.engine.preprocess.emit._per_solve`.  Any value users set
    has been silently dropped.

    This helper removes the schema row; sibling edits in the same
    commit strip it from the input_derivation cl_pars, the
    SET_LIKE_NAMES table, the autoscale quantity-type table and the
    export_settings.yaml params list.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_ramp_envelope`` is dropped alongside the
    ``parameter_definition`` when ``remove_parameters_manual`` invokes
    ``db.remove_items``.
    """
    remove_parameters_manual(db, [["model", "output_ramp_envelope"]])


def _migrate_v56_remove_output_unit__node_flow_t(db) -> None:
    """Drop the ``model.output_unit__node_flow_t`` parameter from the schema.

    Dead toggle: the flag IS plumbed into the multi-param
    ``optional_outputs.csv`` emitter, but nothing on the engine side
    reads its row from the resulting ``enable_optional_outputs`` set —
    only ``output_horizon`` is checked.  The
    ``unit__inputNode__dt`` / ``unit__outputNode__dt`` golden CSVs
    that tests rely on are produced by the always-on
    write-handoff path, not by this gate.

    This helper removes the schema row; sibling edits in the same
    commit strip it from the input_derivation cl_pars, the
    SET_LIKE_NAMES table, the autoscale quantity-type table and the
    export_settings.yaml params list.  The legacy
    ``tests/fixtures/regen_lh2_three_region.py`` generator no longer
    appends a ``yes`` override for the parameter.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_unit__node_flow_t`` is dropped alongside the
    ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_unit__node_flow_t"]])


def _migrate_v56_remove_output_unit__node_ramp_t(db) -> None:
    """Drop the ``model.output_unit__node_ramp_t`` parameter from the schema.

    Dead toggle: the flag was plumbed into the multi-param
    ``optional_outputs.csv`` emitter but nothing on the engine side
    reads its row from ``enable_optional_outputs`` — only
    ``output_horizon`` is consulted.  No per-flag emission branch
    exists; any value users set was silently dropped.

    Sibling edits in the same commit strip it from the input_derivation
    cl_pars, the SET_LIKE_NAMES table, the autoscale quantity-type
    table and the export_settings.yaml params list.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_unit__node_ramp_t`` is dropped alongside the
    ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_unit__node_ramp_t"]])


def _migrate_v56_remove_output_connection__node__node_flow_t(db) -> None:
    """Drop the ``model.output_connection__node__node_flow_t`` parameter
    from the schema.

    Dead toggle: the flag was plumbed into the multi-param
    ``optional_outputs.csv`` emitter but nothing on the engine side
    reads its row from ``enable_optional_outputs`` — only
    ``output_horizon`` is consulted.  Any value users set was silently
    dropped.

    Sibling edits in the same commit strip it from the input_derivation
    cl_pars, the SET_LIKE_NAMES table, the autoscale quantity-type
    table, the export_settings.yaml params list and the legacy
    ``tests/fixtures/regen_lh2_three_region.py`` ``yes`` override.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_connection__node__node_flow_t`` is dropped alongside
    the ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_connection__node__node_flow_t"]])


def _migrate_v56_remove_output_connection_flow_separate(db) -> None:
    """Drop the ``model.output_connection_flow_separate`` parameter
    from the schema.

    Dead toggle and the last of the Batch-B output flag removals: the
    parameter was plumbed into the multi-param ``optional_outputs.csv``
    emitter but nothing on the engine side reads its row from
    ``enable_optional_outputs`` — only ``output_horizon`` is consulted.
    Any value users set was silently dropped.

    Sibling edits in the same commit strip it from the input_derivation
    cl_pars (which leaves only ``output_horizon`` in the
    ``optional_outputs.csv`` emitter — the parameter that IS actually
    consumed), the SET_LIKE_NAMES table, the autoscale quantity-type
    table and the export_settings.yaml params list.

    Side effects: every ``parameter_value`` row referencing
    ``model.output_connection_flow_separate`` is dropped alongside the
    ``parameter_definition``.
    """
    remove_parameters_manual(db, [["model", "output_connection_flow_separate"]])


def _migrate_v56_retype_solver_arguments_to_1d_map(db) -> None:
    """Retype ``solve.solver_arguments`` from ``array`` to ``1d_map``.

    Batch C.1 — first commit of the solver-knob consolidation.  The
    parameter changes role from "list of raw command-line arguments to
    pass to the legacy flextoolrunner / GAMS solver invocation" to
    "key→value map of HiGHS solver options layered on top of
    ``solver_config/highs.opt``".  In subsequent C.2-C.5 commits the
    sibling Map ``solver_options`` and the three ``highs_method`` /
    ``highs_parallel`` / ``highs_presolve`` shortcut parameters fold
    their content into this canonical home before being removed.

    Schema retype: the parameter_definition's ``parameter_type_list`` is
    flipped from ``("array",)`` to ``("1d_map",)`` and the description
    is rewritten to reflect the new role.

    Value translation: every existing ``solver_arguments`` parameter
    value with the legacy Array shape is converted to an equivalent
    1d-map.  Each Array entry is parsed as a HiGHS ``key=value`` option
    line (the format ``solver_config/highs.opt`` uses) and becomes one
    Map entry.  Entries that do not parse cleanly cause a
    ``SpineDBAPIError`` to be raised — silent lossy conversion is
    intentionally not allowed because the legacy semantics differed
    enough (raw CLI args for flextoolrunner vs HiGHS option keys) that
    the user must inspect any non-empty array on a per-DB basis.  None
    of the in-repo fixtures or canonical databases carry a non-null
    ``solver_arguments`` value at v55 (audited 2026-05-27), so the
    no-real-data fast path is the only one exercised by the gate.

    Engine consumption: the new 1d-map values are read by the resolver
    in :mod:`flextool.engine.solve._solver_dispatch` and merged with
    ``solver_config/highs.opt`` (floor) and the CLI flags (top) into
    the final HiGHS options dict.  See
    :func:`flextool.engine.solve._solver_dispatch._resolve_effective_highs_options`.
    """
    parameter_definitions = db.mapped_table("parameter_definition")
    try:
        param = db.item(parameter_definitions,
                        entity_class_name="solve", name="solver_arguments")
    except SpineDBAPIError:
        param = None
    if param is None:
        # Steady-state once the schema template is in sync; helper must
        # be safe to re-run.
        return

    # Translate existing Array values to 1d-map.  In-repo fixtures all
    # carry null values for this parameter at v55 but a user's DB may
    # carry author-supplied arrays — convert them on a best-effort
    # basis and STOP on the first unparseable entry.
    new_description = (
        "Map of HiGHS solver options (option name -> value), layered on "
        "top of the floor values in ``solver_config/highs.opt`` and "
        "below any CLI overrides (e.g. ``--solver-time-limit``).  Use "
        "this for ad-hoc HiGHS knobs that do not have a dedicated "
        "FlexTool parameter (``solver_mip_gap``, ``solver_precommand``)."
    )
    existing = list(db.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="solver_arguments",
    ))
    for pv in existing:
        try:
            raw_value = from_database(pv["value"], pv["type"])
        except Exception:  # pragma: no cover — best-effort
            raw_value = None
        if raw_value is None:
            continue
        # Already a Map?  Idempotent path — leave alone.
        if isinstance(raw_value, Map):
            continue
        # Legacy Array path.  Parse each "key=value" entry.
        from spinedb_api import Array as _Array
        if not isinstance(raw_value, _Array):
            raise SpineDBAPIError(
                f"solver_arguments on {pv['entity_byname']!r} has "
                f"unexpected type {type(raw_value).__name__!r}; cannot "
                "convert to 1d-map.  Inspect the value manually before "
                "re-running the migration."
            )
        entries: list[tuple[str, str]] = []
        for entry in raw_value.values:
            text = str(entry).strip()
            if not text:
                continue
            # Accept both "key=value" (highs.opt format) and "key value"
            # (legacy CLI-arg style); reject anything else.
            if "=" in text:
                key, _, val = text.partition("=")
            elif " " in text:
                key, _, val = text.partition(" ")
            else:
                raise SpineDBAPIError(
                    f"solver_arguments on {pv['entity_byname']!r} "
                    f"carries entry {text!r} that does not match the "
                    "expected 'key=value' or 'key value' form; cannot "
                    "convert to 1d-map.  Edit the value to the new map "
                    "shape manually before re-running the migration."
                )
            entries.append((key.strip(), val.strip()))
        new_map = Map([k for k, _ in entries], [v for _, v in entries])
        new_value, new_type = to_database(new_map)
        db.update_item(
            "parameter_value",
            id=pv["id"],
            value=new_value, type=new_type,
        )

    # Flip the parameter_definition's type list.
    db.update_parameter_definition(
        id=param["id"],
        name="solver_arguments",
        parameter_type_list=("1d_map",),
        description=new_description,
    )
    _commit_step(
        db,
        "v56 retype solve.solver_arguments array -> 1d-map: canonical "
        "home for HiGHS solver-option overrides; values layered on top "
        "of solver_config/highs.opt by the engine's effective-options "
        "resolver (Batch C.1).",
    )


def _migrate_v56_fold_solver_options_into_solver_arguments(db) -> None:
    """Fold the legacy ``solve.solver_options`` Map into the
    ``solve.solver_arguments`` 1d-map and remove ``solver_options``.

    Batch C.2 — second commit of the solver-knob consolidation.  The
    ``solver_options`` Map covered the same surface as the just-retyped
    ``solver_arguments`` 1d-map (free-form HiGHS option key → value
    overrides); collapsing both into one canonical home eliminates the
    "which one wins?" ambiguity and the redundant DB axis.

    Collision policy: where a key appears in both Maps on the same
    (solve, alternative), the existing ``solver_arguments`` value
    wins — it was the more explicit / more recently introduced home
    (v52+) and we treat the migration as a no-op for that key.  The
    helper logs collisions to stdout so a user reviewing the
    migration log can spot drifted values; no user data is silently
    overwritten.

    All in-repo fixtures carry null values for ``solver_options`` at
    v55 (audited 2026-05-27), so the fold path is exercised only by
    user databases that authored entries.

    Engine consumption: ``_resolve_effective_highs_options`` in
    :mod:`flextool.engine.solve._solver_dispatch` already reads the
    1d-map ``solver_arguments`` via the resolver; after this commit
    every override the user authored is routed through that single
    path.  ``solver_options`` is no longer read from the DB.

    Side effects: every ``parameter_value`` row referencing
    ``solver_options`` is dropped alongside the
    ``parameter_definition`` by the companion
    :func:`_migrate_v56_remove_solver_options` call in the
    elif block.
    """
    parameter_values = db.mapped_table("parameter_value")  # noqa: F841 — touch the table cache
    options_rows = list(db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="solver_options",
    ))
    if not options_rows:
        # Nothing to fold; the companion removal helper handles the
        # idempotent definition strip.
        return

    # Build an (entity_name, alternative_name) -> existing solver_arguments
    # row index so we can merge in place when the user authored both.
    args_rows = list(db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="solver_arguments",
    ))
    args_index: dict[tuple[str, str], dict] = {}
    for pv in args_rows:
        args_index[(pv["entity_name"], pv["alternative_name"])] = pv

    for opt_pv in options_rows:
        try:
            opt_value = from_database(opt_pv["value"], opt_pv["type"])
        except Exception:  # pragma: no cover — best-effort
            opt_value = None
        if opt_value is None:
            continue
        if not isinstance(opt_value, Map):
            print(
                f"v56 fold solver_options -> solver_arguments: solve."
                f"{opt_pv['entity_byname']!r}.solver_options is not a "
                f"Map ({type(opt_value).__name__!r}); skipping fold."
            )
            continue
        opt_dict: dict[str, str] = {
            str(k): str(v)
            for k, v in zip(list(opt_value.indexes), list(opt_value.values))
        }
        existing = args_index.get(
            (opt_pv["entity_name"], opt_pv["alternative_name"])
        )
        existing_map: dict[str, str] = {}
        if existing is not None:
            try:
                cur = from_database(existing["value"], existing["type"])
            except Exception:  # pragma: no cover — best-effort
                cur = None
            if isinstance(cur, Map):
                existing_map = {
                    str(k): str(v)
                    for k, v in zip(list(cur.indexes), list(cur.values))
                }
        merged: dict[str, str] = dict(existing_map)
        collisions: list[str] = []
        for key, val in opt_dict.items():
            if key in merged and merged[key] != val:
                collisions.append(
                    f"{key}: solver_arguments={merged[key]!r} wins over "
                    f"solver_options={val!r}"
                )
                continue
            merged[key] = val
        if collisions:
            print(
                "v56 fold solver_options -> solver_arguments collisions on "
                f"solve.{opt_pv['entity_byname']!r} "
                f"(alt={opt_pv['alternative_name']!r}): "
                + "; ".join(collisions)
            )
        new_map = Map(list(merged.keys()), list(merged.values()))
        new_value, new_type = to_database(new_map)
        if existing is not None:
            db.update_item(
                "parameter_value",
                id=existing["id"],
                value=new_value, type=new_type,
            )
        else:
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=opt_pv["entity_byname"],
                parameter_definition_name="solver_arguments",
                alternative_name=opt_pv["alternative_name"],
                value=new_value, type=new_type,
            )

    _commit_step(
        db,
        "v56 fold solve.solver_options -> solver_arguments: existing "
        "solver_arguments entries win on key collision (Batch C.2).",
    )


def _migrate_v56_remove_solver_options(db) -> None:
    """Remove the ``solve.solver_options`` parameter definition.

    Companion to :func:`_migrate_v56_fold_solver_options_into_solver_arguments`
    (Batch C.2).  After the fold helper has copied every Map entry
    into ``solver_arguments``, this helper drops the now-duplicate
    parameter_definition.  ``parameter_value`` rows referencing the
    definition cascade-delete alongside it via spinedb_api.
    """
    remove_parameters_manual(db, [["solve", "solver_options"]])


def _fold_highs_shortcut_into_solver_arguments(
    db, *, shortcut_param: str, highs_key: str, label: str,
) -> None:
    """Shared helper for the three ``highs_*`` shortcut folds (C.3-C.5).

    Walks every ``solve.<shortcut_param>`` parameter_value, injects
    ``solver_arguments[<highs_key>] = <value>`` on the same
    (solve, alternative).  When the user already authored the same
    key in ``solver_arguments``, the explicit ``solver_arguments``
    value wins (logged); we treat the migration as a no-op for that
    key.  Mirrors :func:`_migrate_v56_fold_solver_options_into_solver_arguments`
    semantics for consistency across the C-batch.

    *label* appears in the migration commit message and the
    collision-log lines so the source shortcut is identifiable.
    """
    shortcut_rows = list(db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name=shortcut_param,
    ))
    if not shortcut_rows:
        return
    args_rows = list(db.find_parameter_values(
        entity_class_name="solve", parameter_definition_name="solver_arguments",
    ))
    args_index: dict[tuple[str, str], dict] = {}
    for pv in args_rows:
        args_index[(pv["entity_name"], pv["alternative_name"])] = pv

    for sh_pv in shortcut_rows:
        try:
            sh_value = from_database(sh_pv["value"], sh_pv["type"])
        except Exception:  # pragma: no cover — best-effort
            sh_value = None
        if sh_value is None:
            continue
        if not isinstance(sh_value, str):
            print(
                f"v56 fold {label} -> solver_arguments[{highs_key!r}]: "
                f"solve.{sh_pv['entity_byname']!r}.{shortcut_param} is "
                f"not a str ({type(sh_value).__name__!r}); skipping fold."
            )
            continue
        existing = args_index.get(
            (sh_pv["entity_name"], sh_pv["alternative_name"])
        )
        existing_map: dict[str, str] = {}
        if existing is not None:
            try:
                cur = from_database(existing["value"], existing["type"])
            except Exception:  # pragma: no cover — best-effort
                cur = None
            if isinstance(cur, Map):
                existing_map = {
                    str(k): str(v)
                    for k, v in zip(list(cur.indexes), list(cur.values))
                }
        if highs_key in existing_map and existing_map[highs_key] != sh_value:
            print(
                f"v56 fold {label} -> solver_arguments[{highs_key!r}] "
                f"collision on solve.{sh_pv['entity_byname']!r} "
                f"(alt={sh_pv['alternative_name']!r}): "
                f"solver_arguments[{highs_key!r}]={existing_map[highs_key]!r} "
                f"wins over {shortcut_param}={sh_value!r}"
            )
            continue
        merged = dict(existing_map)
        merged[highs_key] = sh_value
        new_map = Map(list(merged.keys()), list(merged.values()))
        new_value, new_type = to_database(new_map)
        if existing is not None:
            db.update_item(
                "parameter_value",
                id=existing["id"],
                value=new_value, type=new_type,
            )
        else:
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=sh_pv["entity_byname"],
                parameter_definition_name="solver_arguments",
                alternative_name=sh_pv["alternative_name"],
                value=new_value, type=new_type,
            )

    _commit_step(
        db,
        f"v56 fold solve.{shortcut_param} -> "
        f"solver_arguments[{highs_key!r}]: shortcut value injected "
        "into the canonical HiGHS option-overrides Map (existing "
        f"solver_arguments[{highs_key!r}] wins on collision).",
    )


def _migrate_v56_fold_highs_method_into_solver_arguments(db) -> None:
    """Fold ``solve.highs_method`` values into
    ``solver_arguments['solver']`` (HiGHS' name for the method-
    selection option, per ``DETERMINISM_OPTIONS`` and the
    ``flex_data.py:_HIGHS_PARAM_MAP`` table).

    Batch C.3 — first of the three ``highs_*`` shortcut folds.  The
    parameter values in flextool fixtures (``"simplex"``,
    ``"choose"``) are the exact HiGHS-side spelling so no value
    translation is needed; only the key name flips
    (``highs_method`` → ``solver``).

    Sibling :func:`_migrate_v56_remove_highs_method` strips the
    parameter_definition + every value row.  Collision policy
    matches :func:`_migrate_v56_fold_solver_options_into_solver_arguments`:
    explicit ``solver_arguments`` entry wins, collision logged.
    """
    _fold_highs_shortcut_into_solver_arguments(
        db, shortcut_param="highs_method",
        highs_key="solver", label="highs_method",
    )


def _migrate_v56_remove_highs_method(db) -> None:
    """Remove the ``solve.highs_method`` parameter definition + value list.

    Companion to
    :func:`_migrate_v56_fold_highs_method_into_solver_arguments`
    (Batch C.3).  After the fold helper has injected every value as
    ``solver_arguments['solver']``, this helper drops the now-
    duplicate parameter_definition AND its dedicated
    ``highs_method`` parameter_value_list (which the parameter is the
    sole referent of).  ``parameter_value`` rows cascade-delete
    alongside the definition via spinedb_api.
    """
    remove_parameters_manual(db, [["solve", "highs_method"]])
    try:
        vl = db.item(db.mapped_table("parameter_value_list"), name="highs_method")
    except SpineDBAPIError:
        vl = None
    if vl:
        db.remove_items("parameter_value_list", vl["id"])
        try:
            _commit_step(db, "v56 removed solve.highs_method parameter_value_list")
        except SpineDBAPIError:
            pass


def _migrate_v56_fold_highs_parallel_into_solver_arguments(db) -> None:
    """Fold ``solve.highs_parallel`` values into
    ``solver_arguments['parallel']`` (HiGHS' own option name for
    parallelism control, per ``DETERMINISM_OPTIONS`` and the
    ``flex_data.py:_HIGHS_PARAM_MAP`` table).

    Batch C.4 — second of the three ``highs_*`` shortcut folds.
    Fixture values (``"on"``, ``"off"``) are HiGHS-canonical so no
    value translation is needed; only the key name flips.

    Sibling :func:`_migrate_v56_remove_highs_parallel` strips the
    parameter_definition + the dedicated value list.  Collision
    policy matches C.3: explicit ``solver_arguments['parallel']``
    wins on collision (logged).
    """
    _fold_highs_shortcut_into_solver_arguments(
        db, shortcut_param="highs_parallel",
        highs_key="parallel", label="highs_parallel",
    )


def _migrate_v56_remove_highs_parallel(db) -> None:
    """Remove ``solve.highs_parallel`` parameter definition + value list.

    Companion to
    :func:`_migrate_v56_fold_highs_parallel_into_solver_arguments`
    (Batch C.4).  Same shape as C.3's removal helper.
    """
    remove_parameters_manual(db, [["solve", "highs_parallel"]])
    try:
        vl = db.item(
            db.mapped_table("parameter_value_list"), name="highs_parallel",
        )
    except SpineDBAPIError:
        vl = None
    if vl:
        db.remove_items("parameter_value_list", vl["id"])
        try:
            _commit_step(db, "v56 removed solve.highs_parallel parameter_value_list")
        except SpineDBAPIError:
            pass


def _migrate_v56_fold_highs_presolve_into_solver_arguments(db) -> None:
    """Fold ``solve.highs_presolve`` values into
    ``solver_arguments['presolve']`` (HiGHS' own option name for
    presolve control, per ``DETERMINISM_OPTIONS`` and the
    ``flex_data.py:_HIGHS_PARAM_MAP`` table).

    Batch C.5 — third (and last) of the ``highs_*`` shortcut folds.
    Fixture values (``"on"``, ``"off"``, ``"choose"``) are
    HiGHS-canonical so no value translation is needed; only the key
    name flips.

    Sibling :func:`_migrate_v56_remove_highs_presolve` strips the
    parameter_definition + the dedicated value list.  Collision
    policy matches C.3-C.4: explicit ``solver_arguments['presolve']``
    wins on collision (logged).
    """
    _fold_highs_shortcut_into_solver_arguments(
        db, shortcut_param="highs_presolve",
        highs_key="presolve", label="highs_presolve",
    )


def _migrate_v56_remove_highs_presolve(db) -> None:
    """Remove ``solve.highs_presolve`` parameter definition + value list.

    Companion to
    :func:`_migrate_v56_fold_highs_presolve_into_solver_arguments`
    (Batch C.5).  Same shape as C.3-C.4's removal helpers.
    """
    remove_parameters_manual(db, [["solve", "highs_presolve"]])
    try:
        vl = db.item(
            db.mapped_table("parameter_value_list"), name="highs_presolve",
        )
    except SpineDBAPIError:
        vl = None
    if vl:
        db.remove_items("parameter_value_list", vl["id"])
        try:
            _commit_step(db, "v56 removed solve.highs_presolve parameter_value_list")
        except SpineDBAPIError:
            pass


def _migrate_v56_remove_solver_threads(db) -> None:
    """Remove ``solve.solver_threads`` parameter definition + values.

    Batch C.6 — drop the GUI/CLI-only knob's DB axis.  User-stored
    values are intentionally NOT migrated: per the Q-C-2 design
    decision, GUI-bound knobs were rarely scenario-relevant and the
    break is acceptable.  Equivalent control remains available via
    the existing ``--highs-threads N`` CLI flag (CLI > solver_arguments
    > highs.opt precedence inside
    :func:`flextool.engine.solve._solver_dispatch._resolve_effective_highs_options`)
    and via the GUI's per-job ``max_cores_per_job`` setting which
    plumbs the flag automatically.  GUI controls for this knob's
    DB-stored equivalent are deferred to the v56 follow-up PR
    (task #26).

    The engine read site in :mod:`flextool.engine.input._solve_config`
    (where it fed :class:`SolverConfig.threads` → ``build_solver_options``)
    is dropped in the same commit.  After this commit ``SolverConfig.threads``
    stays as a None-default field — the commercial-solver path still
    accepts a Python int there but FlexTool no longer authors it.
    """
    remove_parameters_manual(db, [["solve", "solver_threads"]])


def _migrate_v56_remove_use_row_scaling(db) -> None:
    """Remove ``solve.use_row_scaling`` parameter definition + values.

    Batch C.10 (last commit of the solver-knob consolidation) — drop
    the DB-level row-scaling opt-in.  User-stored values are
    intentionally NOT migrated (per Q-C-2).  Equivalent control is
    already exposed via the ``--scaling {off,solver_only,basic,full}``
    CLI flag (from the autoscale work); the env-var test hook
    ``FLEXTOOL_FORCE_ROW_SCALING=1`` also remains in place for the
    Mode B un-scaling benchmark harness.

    Engine cleanup: the per-solve dict read in
    :meth:`SolveConfig.load_from_db` is hard-wired to ``{}`` so
    every solve emits ``p_use_row_scaling=0`` (the
    ``use_row_scaling.get(solve, "no")`` default branch) and the
    Mode A pre-scaling behaviour is preserved.  The autoscaler's
    Layer 2 + Layer 3 (driven by ``--scaling``) are unaffected and
    continue to compress LP coefficient ranges in the cold and warm
    paths the same way.
    """
    remove_parameters_manual(db, [["solve", "use_row_scaling"]])


def _migrate_v56_remove_solver_io_api(db) -> None:
    """Remove ``solve.solver_io_api`` parameter definition + value list.

    Batch C.9 — drop the GUI/CLI-only knob's DB axis.  User-stored
    values are intentionally NOT migrated (per Q-C-2).  Equivalent
    control is exposed via the new ``--matrix-file-format {mps,lp}``
    CLI flag, plumbed via the ``FLEXTOOL_MATRIX_FILE_FORMAT`` env var
    into :meth:`SolveConfig.load_from_db` where it overrides the
    per-solve :class:`SolverConfig.io_api` default for every solve.
    The in-process vs. file dispatch is implicit: HiGHS without
    ``--save-memory`` stays on the direct binding; commercial
    solvers (gurobi/cplex/xpress/copt) always write a file using
    the chosen format.  GUI controls for the DB-stored equivalent
    are deferred to the v56 follow-up PR (task #26).

    The ``solver_io_apis`` parameter_value_list (``direct``, ``mps``,
    ``lp``) is also removed since the parameter was its sole referent.
    """
    remove_parameters_manual(db, [["solve", "solver_io_api"]])
    try:
        vl = db.item(
            db.mapped_table("parameter_value_list"), name="solver_io_apis",
        )
    except SpineDBAPIError:
        vl = None
    if vl:
        db.remove_items("parameter_value_list", vl["id"])
        try:
            _commit_step(db, "v56 removed solve.solver_io_apis parameter_value_list")
        except SpineDBAPIError:
            pass


def _migrate_v56_remove_solver_time_limit(db) -> None:
    """Remove ``solve.solver_time_limit`` parameter definition + values.

    Batch C.8 — drop the GUI/CLI-only knob's DB axis.  User-stored
    values are intentionally NOT migrated (per Q-C-2).  Equivalent
    control is exposed via the new ``--solver-time-limit SECONDS``
    CLI flag, plumbed via the existing ``FLEXTOOL_HIGHS_TIME_LIMIT``
    env var into the engine-side
    :func:`flextool.engine.solve._solver_dispatch._resolve_effective_highs_options`
    resolver as a CLI override on the HiGHS ``time_limit`` key
    (CLI > solver_arguments > highs.opt precedence).  GUI controls
    for the DB-stored equivalent are deferred to the v56 follow-up
    PR (task #26).

    The engine read site in :mod:`flextool.engine.input._solve_config`
    (where it fed :class:`SolverConfig.time_limit` →
    ``build_solver_options``) is dropped in the same commit.
    ``SolverConfig.time_limit`` stays as a None-default field — the
    commercial-solver path still accepts a Python float there but
    FlexTool no longer authors it.
    """
    remove_parameters_manual(db, [["solve", "solver_time_limit"]])


def _migrate_v56_remove_solver_log_level(db) -> None:
    """Remove ``solve.solver_log_level`` parameter definition + value list.

    Batch C.7 — drop the GUI/CLI-only knob's DB axis.  User-stored
    values are intentionally NOT migrated.  Audit (2026-05-27) found
    no engine module consuming ``SolverConfig.log_level`` — the field
    was loaded into the dataclass but never read by any solver-
    dispatch / autoscale / writer site, so the v55 schema knob was
    fully dead.  Equivalent control is now exposed via the new
    ``--solver-log-level {silent,normal,verbose}`` CLI flag, plumbed
    via the ``FLEXTOOL_SOLVER_LOG_LEVEL`` env var into the engine-side
    :func:`flextool.engine.solve._solver_dispatch._resolve_effective_highs_options`
    resolver (CLI > solver_arguments > highs.opt precedence).
    ``silent`` maps to HiGHS ``output_flag=false``; ``verbose`` maps
    to ``output_flag=true`` plus ``log_dev_level=2``; ``normal`` maps
    to ``output_flag=true`` (default).  GUI controls for the DB-
    stored equivalent are deferred to the v56 follow-up PR (task #26).

    The ``solver_log_levels`` parameter_value_list (``silent``,
    ``normal``, ``verbose``) is also removed since the parameter was
    its sole referent.
    """
    remove_parameters_manual(db, [["solve", "solver_log_level"]])
    try:
        vl = db.item(
            db.mapped_table("parameter_value_list"), name="solver_log_levels",
        )
    except SpineDBAPIError:
        vl = None
    if vl:
        db.remove_items("parameter_value_list", vl["id"])
        try:
            _commit_step(db, "v56 removed solve.solver_log_levels parameter_value_list")
        except SpineDBAPIError:
            pass
