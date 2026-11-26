import argparse
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from typing import Callable, NamedTuple

from spinedb_api import from_database, import_data, Map, to_database

from flextool.sqlite_url import sqlite_url_string
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _append_list_members,
    _attach_list_defaults,
    _commit_step,
    _describe_pv,
    _detach_list_defaults,
    _ensure_parameter_group,
    _list_members,
    _pair_is,
    _pdef_or_none,
    _pre_v26_template,
    _remove_items_bounded,
    _rename_or_drop_parameter_definition,
    _replace_superseded_pairs,
    _replace_text_pairs,
    add_new_parameters,
    add_parameters_manual,
    add_value_list_manual,
    add_version,
    ALREADY_APPLIED_PREFIX,
    BEHAVIOUR_CHANGE_MARKER_PREFIX,
    FlexToolMigrationError,
    MigrationCancelled,
    remove_parameters_manual,
)
from flextool.update_flextool.db_migration.v01_v25 import (
    _migrate_v12,
    _migrate_v13,
    _migrate_v14,
    _migrate_v15,
    _migrate_v17,
    _migrate_v18,
    _migrate_v21,
    _migrate_v22,
    _migrate_v23,
    _migrate_v24,
    _migrate_v25,
    change_optional_output_type,
)
from flextool.update_flextool.db_migration.v26_v43 import (
    _migrate_v26,
    _migrate_v27,
    _migrate_v28,
    _migrate_v29,
    _migrate_v32,
    _migrate_v33,
    _migrate_v35,
    _migrate_v36,
    _migrate_v37,
    _migrate_v38,
    _migrate_v39,
    _migrate_v40,
    _migrate_v41,
    _migrate_v42,
    _migrate_v43,
)
from flextool.update_flextool.db_migration.v44_v49 import (
    _migrate_v44_parameter_groups,
    _migrate_v45_parameter_group_colors,
    _migrate_v46_use_row_scaling,
    _migrate_v47,
    _migrate_v48,
    _migrate_v49,
)
from flextool.update_flextool.db_migration.v50_v55 import (
    _migrate_v50_new_stepduration_to_solve,
    _migrate_v51_group_block_resolution,
    _migrate_v52_solver_selection,
    _migrate_v53_storage_binding_value_list,
    _migrate_v54_storage_binding_arrays_to_scalar,
    _migrate_v55_storage_binding_rename_and_extend,
)
from flextool.update_flextool.db_migration.v56_methods import _migrate_v56
from flextool.update_flextool.db_migration.v57_v69 import (
    _migrate_v57,
    _migrate_v58_carve_flowgroup_out_of_group,
    _migrate_v59,
    _migrate_v60_solve_decomposition,
    _migrate_v61,
    _migrate_v62_benders_rename,
    _migrate_v63_add_in_out_weight,
    _migrate_v64_add_solve_scaling,
    _migrate_v65_energy_and_capacity_margin_methods,
    _migrate_v66_energy_margin_adder,
    _migrate_v67_node_penalty_method,
    _migrate_v68_rp_group_flag,
    _migrate_v69_backfill_parameter_groups,
)
from flextool.update_flextool.db_migration.v70_v76 import (
    _migrate_v70_capacity_coefficient_descriptions,
    _migrate_v71_stochastic_invest_method,
    _migrate_v72_non_anticipativity_periods,
    _migrate_v73_non_anticipativity_invest_periods,
    _migrate_v74_construction_lead_time,
    _migrate_v75,
    _migrate_v76_dispatch_realization_method,
)
from flextool.update_flextool.db_migration.v77 import (
    _inactive_in_scenarios,
    _merge_marker_lists,
    _migrate_v77_schema_batch,
    _refresh_unit_count_texts,
    _scenario_alternatives,
    _SUPERSEDED_UNIT_COUNT_TEXTS,
    _unit_count_vu_text,
    _V77_NO_SCENARIO,
    _v77_set_marker,
    _v77_winner,
)
from flextool.update_flextool.db_migration.v78 import _migrate_v78_unit_prefixes
from flextool.update_flextool.db_migration.v79 import (
    _migrate_v79_schema_batch,
    _v79_numbers,
    V79_NODE_UNIT_OF_MEASURE_DESCRIPTION,
)

__all__ = ["migrate_database", "migrate_repository_database", "MigrationCancelled", "FlexToolMigrationError", "BEHAVIOUR_CHANGE_MARKER_PREFIX", "ALREADY_APPLIED_PREFIX", "REFRESH_COMMIT", "RERUNNABLE_HEAD_STEP", "remove_behaviour_change_markers"]


def migrate_database(
    database_path,
    up_to: int | None = None,
    *,
    progress_callback: Callable[[int, int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    allow_frozen: bool = False,
    backup: bool = True,
):
    """Migrate a FlexTool database to a target schema version.

    A local file is migrated all or nothing: it is copied aside before the first step and copied back when any step
    fails or the migration is cancelled (:func:`~flextool.update_flextool.migration_backup.restored_on_failure`), so
    it is never left half migrated.

    Args:
        database_path: Path or URL to the SQLite database.
        up_to: Target version to migrate to.  When ``None`` (the default),
            migrates all the way to :data:`FLEXTOOL_DB_VERSION`.
        progress_callback: Optional callable invoked BEFORE each step's
            logic runs with ``(current_version, target_version,
            next_version_being_applied)``.  Lets a GUI report which step
            is in progress.
        cancel_check: Optional callable returning ``True`` if the user
            has requested cancellation.  Checked at the top of each loop
            iteration, BEFORE step logic.  When True, raises
            :class:`MigrationCancelled` carrying the last successfully
            completed version.  The check does not fire inside a step,
            so an in-flight commit always finishes before cancellation
            takes effect.
        allow_frozen: Migrate a database that carries the frozen marker
            (:mod:`flextool.update_flextool.frozen`); the marker is kept with
            its version part updated.  Without it, a frozen database older
            than this FlexTool raises
            :class:`~flextool.update_flextool.frozen.FrozenDatabaseError`
            before the file is opened, and a frozen database that is
            already current is left unchanged.
        backup: Copy a local file aside and restore it on failure (the default).  ``False`` only for a caller that
            holds its own backup of the file for the whole migration (the window's upgrade).
    """
    from flextool.update_flextool.frozen import FrozenCheckError, sqlite_path_from
    from flextool.update_flextool.migrate_before_run import migration_lock
    from flextool.update_flextool.migration_backup import backup_path, refuse_kept_copy

    # A pre-migration copy left by an earlier migration may be the only intact original (the file itself may be half
    # migrated, or truncated by an interrupted restore, so that even the frozen check below cannot read it): refuse
    # first, naming the copy.  A current file has nothing to migrate and is left alone.
    kept_path = sqlite_path_from(database_path)
    if (backup and kept_path is not None and kept_path.is_file() and backup_path(kept_path).exists()
            and _current_version_ro(kept_path, up_to) is None):
        refuse_kept_copy(kept_path)

    # The frozen guard runs next, read-only, before anything can open the file with ``upgrade=True``.  A marker that
    # cannot be read now (another migration is committing) is checked again under the migration lock.
    try:
        proceed, frozen = _frozen_check(database_path, allow_frozen)
    except FrozenCheckError:
        if sqlite_path_from(database_path) is None:
            raise
        proceed, frozen = None, None
    if proceed is False:
        return

    if database_path.startswith('sqlite://') or database_path.startswith('http://'):
        mapping_name = database_path
    elif os.path.exists(database_path) and database_path.endswith(".sqlite"):
        mapping_name = sqlite_url_string(database_path)
    else:
        logging.critical("No sqlite file at " + database_path)
        exit(-1)

    # Every in-place migration of a local file holds ``<db>.migrate.lock``; the version is read again under it.  A
    # file that reads as current (read-only) needs no migration: it neither waits for the lock nor creates the file.
    path = sqlite_path_from(database_path)
    if proceed is True and path is not None and path.is_file():
        current = _current_version_ro(path, up_to)
        if current is not None:
            print(database_path + " already up-to-date at version " + str(current))
            return
    with migration_lock(path) if path is not None and path.is_file() else contextlib.nullcontext():
        if proceed is None:
            proceed, frozen = _frozen_check(database_path, allow_frozen)
            if proceed is False:
                return
        from flextool.update_flextool.migration_backup import restored_on_failure

        local = path is not None and path.is_file()
        with restored_on_failure(path) if backup and local else contextlib.nullcontext():
            if local:
                # Before the first spinedb_api open: it reads a list default stored as text as None (item 279).
                from flextool.update_flextool.list_default_repair import repair_list_defaults

                repair_list_defaults(path)
                current = _current_version_ro(path, up_to)
                if current is not None:
                    print(database_path + " already up-to-date at version " + str(current))
                    return
            else:
                logging.debug("%s is not a local file; list defaults not checked", database_path)
            _migrate_open(database_path, mapping_name, up_to, frozen, progress_callback, cancel_check)


def _current_version_ro(path, up_to: int | None) -> int | None:
    """The data version of *path* when it has nothing to migrate or refresh (read read-only), else ``None``.

    Nothing to do means: the current data version (and not below *up_to*), no list default stored as text that
    :func:`~flextool.update_flextool.list_default_repair.repair_list_defaults` repairs (checked first: the later
    probes parse defaults), a current SpineDB schema revision, no superseded unit-count description
    (:func:`_refresh_unit_count_texts` would rewrite one), every definition and list member the current version's
    schema step adds and no superseded shipped default and description pair or description
    (:func:`_head_defaults_stale_ro`; :func:`_refresh_head_schema` would add or replace one).  Anything that cannot
    be read gives ``None``: it is decided under the lock.
    """
    from flextool.update_flextool.frozen import read_db_version_ro, schema_outdated_ro
    from flextool.update_flextool.list_default_repair import list_defaults_repairable_ro

    version = read_db_version_ro(path)
    if version != FLEXTOOL_DB_VERSION or (up_to is not None and up_to > version):
        return None
    if (list_defaults_repairable_ro(path) or schema_outdated_ro(path) or _unit_count_texts_stale_ro(path)
            or _head_schema_missing_ro(path) or _head_defaults_stale_ro(path)):
        return None
    return version


def _unit_count_texts_stale_ro(path) -> bool:
    """Whether *path* carries a superseded ``existing`` / ``virtual_unitsize`` description (read-only; ``True`` when
    it cannot be read)."""
    import sqlite3

    from flextool.update_flextool.frozen import _connect_ro

    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return True
    try:
        rows = con.execute(
            "SELECT ec.name, pd.name, pd.description FROM parameter_definition pd "
            "JOIN entity_class ec ON pd.entity_class_id = ec.id WHERE pd.name IN ('existing', 'virtual_unitsize')"
        ).fetchall()
    except sqlite3.Error:
        return True
    finally:
        con.close()
    return any(desc in _SUPERSEDED_UNIT_COUNT_TEXTS.get((cls, name), ()) for cls, name, desc in rows)


def _frozen_check(database_path, allow_frozen: bool):
    """``(proceed, frozen marker)``: the frozen guard of :func:`migrate_database`; prints why it stops."""
    from flextool.update_flextool.frozen import FrozenDatabaseError, frozen_guard, is_current_ro, sqlite_path_from

    try:
        frozen = frozen_guard(database_path, allow_frozen=allow_frozen)
    except FrozenDatabaseError as exc:
        if exc.up_to_date:
            print(f"{database_path} is frozen ({exc.info.describe()}) and up to date; left unchanged")
            return False, None
        raise
    if frozen is not None and is_current_ro(sqlite_path_from(database_path)):
        # Even with allow_frozen, a frozen database that is current is not opened for writing.
        print(f"{database_path} is frozen ({frozen.describe()}) and up to date; left unchanged")
        return False, None
    return True, frozen


def _migrate_open(database_path, mapping_name, up_to, frozen, progress_callback, cancel_check) -> None:
    """The steps of :func:`migrate_database` on the opened database (the version is read here, under the lock)."""
    from flextool.update_flextool.frozen import set_frozen_version

    with open_database_mapping(mapping_name, create=False, upgrade=True) as db:
        sq= db.object_parameter_definition_sq
        settings_parameter = db.query(sq).filter(sq.c.object_class_name == "model").filter(sq.c.parameter_name == "version").one_or_none()
        if settings_parameter is None:
            #if no version assume version 0
            print("No version found. Assuming version 0, if older, migration might not work")
            version = 0
        else:
            version = from_database(settings_parameter.default_value, settings_parameter.default_type)

        next_version = int(version) + 1
        new_version = up_to if up_to is not None else FLEXTOOL_DB_VERSION

        last_completed_version = int(version)
        while next_version <= new_version:
            if cancel_check is not None and cancel_check():
                raise MigrationCancelled(last_completed_version=last_completed_version)
            if progress_callback is not None:
                progress_callback(last_completed_version, new_version, next_version)
            if 1 <= next_version <= 11 and _template_step_superseded(db, next_version):
                pass
            elif next_version == 0:
                add_version(db)
            elif next_version == 1:
                add_new_parameters(db, _pre_v26_template('flextool_template_v2.json'))
            elif next_version == 2:
                add_new_parameters(db, _pre_v26_template('flextool_template_rolling_window.json'))
            elif next_version == 3:
                add_new_parameters(db, _pre_v26_template('flextool_template_lifetime_method.json'))
            elif next_version == 4:
                add_new_parameters(db, _pre_v26_template('flextool_template_drop_down.json'))
            elif next_version == 5:
                add_new_parameters(db, _pre_v26_template('flextool_template_optional_outputs.json'))
            elif next_version == 6:
                add_new_parameters(db, _pre_v26_template('flextool_template_default_value.json'))
            elif next_version == 7:
                add_new_parameters(db, _pre_v26_template('flextool_template_rolling_start_remove.json'))
            elif next_version == 8:
                add_new_parameters(db, _pre_v26_template('flextool_template_output_node_flows.json'))
            elif next_version == 9:
                add_new_parameters(db, _pre_v26_template('flextool_template_constant_default.json'))
            elif next_version == 10:
                add_new_parameters(db, _pre_v26_template('flextool_template_storage_binding_defaults.json'))
            elif next_version == 11:
                change_optional_output_type(db, _pre_v26_template('flextool_template_default_optional_output.json'))
            elif next_version == 12:
                _migrate_v12(db)
            elif next_version == 13:
                _migrate_v13(db)
            elif next_version == 14:
                _migrate_v14(db)
            elif next_version == 15:
                _migrate_v15(db)
            elif next_version == 16:
                add_parameters_manual(db, [["group", "co2_max_period", None, None, "[tCO2] Annualized maximum limit for emitted CO2 in each period."]])
            elif next_version == 17:
                _migrate_v17(db)
            elif next_version == 18:
                _migrate_v18(db)
            elif next_version == 19:
                remove_parameters_manual(db, [["constraint", "is_active"], ["reserve__upDown__unit__node", "is_active"]])
            elif next_version == 20:
                remove_parameters_manual(db, [["connection", "is_active"], ["node", "is_active"], ["unit", "is_active"], ["reserve__upDown__connection", "is_active"]])
            elif next_version == 21:
                _migrate_v21(db)
            elif next_version == 22:
                _migrate_v22(db)
            elif next_version == 23:
                _migrate_v23(db)
            elif next_version == 24:
                _migrate_v24(db)
            elif next_version == 25:
                _migrate_v25(db)
            elif next_version == 26:
                _migrate_v26(db)
            elif next_version == 27:
                _migrate_v27(db)
            elif next_version == 28:
                _migrate_v28(db)
            elif next_version == 29:
                _migrate_v29(db)
            elif next_version == 30:
                add_value_list_manual(db, [
                    ["storage_binding_methods", "bind_using_blended_weights"],
                ])
            elif next_version == 31:
                add_value_list_manual(db, [
                    ["storage_binding_methods", "bind_intraperiod_blocks"],
                ])
            elif next_version == 32:
                _migrate_v32(db)
            elif next_version == 33:
                _migrate_v33(db)
            elif next_version == 34:
                # New lifetime_method 'no_investment': asset retires after
                # lifetime (like reinvest_choice) but no further v_invest is
                # allowed once the first-period lifetime window has elapsed —
                # a one-shot investment, no rebuild. Motivating case: life-
                # extension refurbishments that cannot be physically repeated.
                add_value_list_manual(db, [
                    ["lifetime_methods", "no_investment"],
                ])
            elif next_version == 35:
                _migrate_v35(db)
            elif next_version == 36:
                _migrate_v36(db)
            elif next_version == 37:
                _migrate_v37(db)
            elif next_version == 38:
                _migrate_v38(db)
            elif next_version == 39:
                _migrate_v39(db)
            elif next_version == 40:
                _migrate_v40(db)
            elif next_version == 41:
                _migrate_v41(db)
            elif next_version == 42:
                _migrate_v42(db)
            elif next_version == 43:
                _migrate_v43(db)
            elif next_version == 44:
                _migrate_v44_parameter_groups(db)
            elif next_version == 45:
                _migrate_v45_parameter_group_colors(db)
            elif next_version == 46:
                _migrate_v46_use_row_scaling(db)
            elif next_version == 47:
                _migrate_v47(db)
            elif next_version == 48:
                _migrate_v48(db)
            elif next_version == 49:
                _migrate_v49(db)
            elif next_version == 50:
                # Move new_stepduration from entity_class `timeset` to
                # entity_class `solve`.
                #
                # Pre-v50 semantics: new_stepduration was declared per
                # timeset, but FlexTool rejected any config where two
                # timesets used by the same solve carried different
                # new_stepduration values (see the "More than one
                # timeline in the solve or the same timeline with
                # different step durations in different timesets"
                # check in timeline_config.py).  So the parameter was
                # effectively solve-scoped already — this migration
                # makes the scope explicit.
                _migrate_v50_new_stepduration_to_solve(db)
            elif next_version == 51:
                # Group-level temporal resolution + decomposition schema
                # (Agent 1.1).  Adds two new parameters on entity_class
                # `group`:
                #
                # * ``new_stepduration`` — float, default None.  When set,
                #   members of this group (nodes / units / connections)
                #   operate at this stepduration, overriding the
                #   solve-level new_stepduration for those entities.
                #   Enables e.g. hourly electricity nodes alongside daily
                #   H2 nodes in the same solve.
                # * ``decomposition_method`` — enum string, default
                #   "none".  Reserved for Agent 3.2 (Lagrangian region
                #   decomposition); only the schema lands here.
                #
                # Also adds the ``decomposition_methods`` value list.
                _migrate_v51_group_block_resolution(db)
            elif next_version == 52:
                # Multi-solver dispatch (Phase 1 of the polar-high
                # multi-solver handoff).  Adds seven new solver-selection
                # parameters on the ``solve`` entity so a user can choose
                # a solver per-solve (HiGHS stays default; Gurobi /
                # CPLEX / Xpress / COPT become opt-in via polar-high).
                # No LP / engine behaviour changes here — schema only.
                _migrate_v52_solver_selection(db)
            elif next_version == 53:
                # Storage-binding single-valued migration, Phase 1.
                # Wires the existing ``storage_binding_methods``
                # parameter_value_list to the ``node.storage_binding_method``
                # parameter_definition so Spine UI enforces the
                # enumeration and rejects free-form strings at edit
                # time.  The value-list members themselves were added in
                # v30 / v31 / ``update_timestructure``; this step only
                # closes the schema-level wiring.
                _migrate_v53_storage_binding_value_list(db)
            elif next_version == 54:
                # Storage-binding single-valued migration, Phase 2.
                # Ports existing array-valued
                # ``node.storage_binding_method`` rows onto the new
                # single-string contract.  For each array, the highest
                # priority element present is picked (see
                # ``_STORAGE_BINDING_PRIORITY`` inside the step).  Rows
                # that are already scalar strings are left untouched.
                # Arrays whose contents are entirely outside the known
                # method set raise ``SpineDBAPIError`` naming the entity
                # — surface, don't guess.
                _migrate_v54_storage_binding_arrays_to_scalar(db)
            elif next_version == 55:
                # Storage-binding restructure Phase A — rename the
                # three legacy method names that v53/v54 left in the
                # value_list to their clean-set counterparts, drop the
                # legacy members, and add the four new members the
                # restructure introduces (two of which gain their
                # constraint implementations only in later phases).
                # Touches both parameter_value rows and the
                # ``storage_binding_methods`` parameter_value_list.
                _migrate_v55_storage_binding_rename_and_extend(db)
            elif next_version == 56:
                _migrate_v56(db)
            elif next_version == 57:
                _migrate_v57(db)
            elif next_version == 58:
                _migrate_v58_carve_flowgroup_out_of_group(db)
            elif next_version == 59:
                _migrate_v59(db)
            elif next_version == 60:
                _migrate_v60_solve_decomposition(db)
            elif next_version == 61:
                _migrate_v61(db)
            elif next_version == 62:
                _migrate_v62_benders_rename(db)
            elif next_version == 63:
                _migrate_v63_add_in_out_weight(db)
            elif next_version == 64:
                _migrate_v64_add_solve_scaling(db)
            elif next_version == 65:
                _migrate_v65_energy_and_capacity_margin_methods(db)
            elif next_version == 66:
                _migrate_v66_energy_margin_adder(db)
            elif next_version == 67:
                _migrate_v67_node_penalty_method(db)
            elif next_version == 68:
                _migrate_v68_rp_group_flag(db)
            elif next_version == 69:
                _migrate_v69_backfill_parameter_groups(db)
            elif next_version == 70:
                _migrate_v70_capacity_coefficient_descriptions(db)
            elif next_version == 71:
                _migrate_v71_stochastic_invest_method(db)
            elif next_version == 72:
                _migrate_v72_non_anticipativity_periods(db)
            elif next_version == 73:
                _migrate_v73_non_anticipativity_invest_periods(db)
            elif next_version == 74:
                _migrate_v74_construction_lead_time(db)
            elif next_version == 75:
                _migrate_v75(db)
            elif next_version == 76:
                _migrate_v76_dispatch_realization_method(db)
            elif next_version == 77:
                _migrate_v77_schema_batch(db)
            elif next_version == 78:
                _migrate_v78_unit_prefixes(db)
            elif next_version == 79:
                _migrate_v79_schema_batch(db)
            elif next_version == 80:
                _migrate_v80_schema_batch(db)
            elif next_version == 81:
                _migrate_v81_schema_batch(db)
            elif next_version == 82:
                _migrate_v82_schema_batch(db)
            elif next_version == 83:
                _migrate_v83_schema_batch(db)
            elif next_version == 84:
                _migrate_v84_schema_batch(db)
            else:
                print("Version invalid")
            _stage_version(db, next_version)
            last_completed_version = next_version
            next_version += 1

        if version < new_version:
            version_up = [["model", "version", new_version, None,
                           "[version number] Contains database version information."]]
            (num,log) = import_data(db, object_parameters = version_up)
            if frozen is not None:
                set_frozen_version(db, new_version)
            print(database_path+ " updated to version "+ str(new_version))
            _commit_step(db,"Updated Flextool data structure to version " + str(new_version))
        else:
            if int(version) == FLEXTOOL_DB_VERSION:
                _refresh_unit_count_texts(db)
                _refresh_head_schema(db)
                _commit_step(db, REFRESH_COMMIT)
            print(database_path+ " already up-to-date at version "+ str(version))


def _template_step_superseded(db, version: int) -> bool:
    """Steps 1-11 import whole old templates.  On content from v12 on (``group.output_aggregate_flows`` from v12 to
    v41, ``group.share_loss_of_load`` from v13 on; in no older template) they would bring back what later steps
    removed or renamed: skip them (owner Q1).  A stamp 0-10 there comes from a hand edit or a lost ``model.version``."""
    for name in ("share_loss_of_load", "output_aggregate_flows"):
        if db.get_item("parameter_definition", entity_class_name="group", name=name):
            return _already_applied(version, f"group.{name} (v12 / v13) is there")
    return False


def _stage_version(db, version: int) -> None:
    """Write ``model.version = version`` into the session after step *version*, without a commit of its own.

    The next commit carries it: the next step's first commit, or the final version commit.  A migration killed
    outright (no restore of the copy) thus leaves the stamp at most one step behind what was committed, and the rerun
    repeats only that step, which therefore must converge when its effect is already there (:func:`_already_applied`).
    """
    pdef = db.get_item("parameter_definition", entity_class_name="model", name="version")
    if pdef:
        val, typ = to_database(version)
        db.update_item("parameter_definition", id=pdef["id"], default_value=val, default_type=typ)
    else:
        import_data(db, object_parameters=[["model", "version", version, None,
                                            "[version number] Contains database version information."]])


#: The commit of the current-version refresh (``migrate_database`` on a current database); the repair tool does not
#: count it as a hand edit (``list_default_restore.changed_by_hand``).
REFRESH_COMMIT = ("Refreshed existing / virtual_unitsize descriptions, added the current version's new definitions "
                  "and replaced its superseded defaults")


def remove_behaviour_change_markers(db) -> int:
    """Remove every ``flextool_behaviour_change:*`` metadata item from an
    open mapping (not committed).  Used by the tools that regenerate the
    repository's template, example and fixture JSONs: those are new
    databases, not upgraded ones, and must not print notices.  Returns the
    number removed."""
    ids = [md["id"] for md in db.get_items("metadata")
           if str(md["name"]).startswith(BEHAVIOUR_CHANGE_MARKER_PREFIX)]
    if ids:
        db.remove_items("metadata", *ids)
    return len(ids)


def migrate_repository_database(database_path: str) -> None:
    """Migrate a repository database (master template, canonical example,
    test fixture) and drop the behaviour-change markers the migration wrote.

    The one shared path of the tools that regenerate the repository's JSONs:
    a database built from those JSONs is new, so it must not carry markers
    (only upgraded user databases print behaviour-change notices).
    """
    migrate_database(database_path)
    url = database_path if "://" in database_path else sqlite_url_string(database_path)
    with open_database_mapping(url) as db:
        if remove_behaviour_change_markers(db):
            db.commit_session("Removed behaviour-change markers (repository database)")


#: v80: the sentence that says a coarse ``new_stepduration`` step mixing no-limit and finite time steps has no limit.
_V80_COARSE_STEP_MAX = (" With a solve's new_stepduration, a solve step that combines inf time steps with finite ones "
                        "has no limit for the whole solve step (a warning says so).")
_V80_COARSE_STEP_MIN = (" With a solve's new_stepduration, a solve step that combines -inf time steps with finite ones "
                        "has no limit for the whole solve step (a warning says so).")


def _v80_suffixes() -> dict[tuple[str, str], tuple[str, ...]]:
    """The sentences v80 appends to the description of each period-map limit: ``inf`` (``-inf`` for a lower limit)
    means no limit in that period (the row of that period is left out), and, for the four time-series limits, that a
    solve step of ``new_stepduration`` that combines a no-limit time step with finite ones has no limit for the
    whole step.  The capacity-margin limits get their whole v80 text instead (:func:`_v80_texts`)."""
    return {
        ('connection', 'cumulative_max_capacity'): (' inf means no limit in that period.',),
        ('connection', 'cumulative_min_capacity'): (' 0 or -inf means no minimum in that period.',),
        ('connection', 'invest_max_period'): (' inf means no limit in that period.',),
        ('connection', 'invest_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('connection', 'retire_max_period'): (' inf means no limit in that period.',),
        ('connection', 'retire_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('constraint', 'constant'): (
            ' inf (sense less_than) or -inf (greater_than) means no constraint in that period.',),
        ('flowGroup', 'max_cumulative_flow'): (' inf means no limit in that period.',),
        ('flowGroup', 'max_instant_flow'): (' inf means no limit in that period or time step.', _V80_COARSE_STEP_MAX),
        ('flowGroup', 'min_cumulative_flow'): (' -inf means no limit in that period.',),
        ('flowGroup', 'min_instant_flow'): (' -inf means no limit in that period or time step.', _V80_COARSE_STEP_MIN),
        ('group', 'co2_max_period'): (' inf means no limit in that period.',),
        ('group', 'cumulative_max_capacity'): (' inf means no limit in that period.',),
        ('group', 'cumulative_min_capacity'): (' 0 or -inf means no minimum in that period.',),
        ('group', 'inertia_limit'): (' -inf also means no requirement in that period or time step.',
                                     " With a solve's new_stepduration, a solve step that combines -inf time steps "
                                     "with finite ones has no requirement for the whole solve step (a warning says "
                                     "so)."),
        ('group', 'invest_max_period'): (' inf means no limit in that period.',),
        ('group', 'invest_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('group', 'non_synchronous_limit'): (' inf means no limit in that period or time step.', _V80_COARSE_STEP_MAX),
        ('group', 'retire_max_period'): (' inf means no limit in that period.',),
        ('group', 'retire_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('node', 'cumulative_max_capacity'): (' inf means no limit in that period.',),
        ('node', 'cumulative_min_capacity'): (' 0 or -inf means no minimum in that period.',),
        ('node', 'invest_max_period'): (' inf means no limit in that period.',),
        ('node', 'invest_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('node', 'retire_max_period'): (' inf means no limit in that period.',),
        ('node', 'retire_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('unit', 'cumulative_max_capacity'): (' inf means no limit in that period.',),
        ('unit', 'cumulative_min_capacity'): (' 0 or -inf means no minimum in that period.',),
        ('unit', 'invest_max_period'): (' inf means no limit in that period.',),
        ('unit', 'invest_min_period'): (' 0 or -inf means no minimum in that period.',),
        ('unit', 'retire_max_period'): (' inf means no limit in that period.',),
        ('unit', 'retire_min_period'): (' 0 or -inf means no minimum in that period.',),
    }


def _v80_append_sentences(description: str, sentences: tuple[str, ...]) -> str:
    """``description`` ending with all of ``sentences``: a description that already ends with the first ``k`` of them
    gets only the rest (so a rerun appends nothing)."""
    description = description.rstrip()
    for k in range(len(sentences), 0, -1):
        if description.endswith("".join(sentences[:k])):
            return description + "".join(sentences[k:])
    return description + "".join(sentences)


def _v80_limit_inf_descriptions(db) -> None:
    """``inf`` / ``-inf`` in a period-map limit is no limit in that period (part of v80): each description gets the
    sentences of :func:`_v80_suffixes` it does not end with yet.  No value is rewritten.  Idempotent."""
    for (cls, name), sentences in _v80_suffixes().items():
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        description = _v80_append_sentences(pdef["description"] or "", sentences)
        if description != (pdef["description"] or ""):
            db.update_parameter_definition(id=pdef["id"], description=description)


#: v80: ``group`` capacity-margin parameters renamed (old name -> new name).
V80_CAPACITY_MARGIN_RENAMES: dict[str, str] = {
    "capacity_margin": "capacity_margin_absolute",
    "capacity_margin_share": "capacity_margin_relative",
}
#: v80: ``capacity_margin_methods`` members renamed (old -> new), in the list order.
V80_CAPACITY_MARGIN_METHODS: dict[str, str] = {
    "none": "none",
    "manual": "absolute",
    "share_of_peak": "relative_of_peak_demand",
}


def _v80_texts() -> dict[tuple[str, str], str]:
    """The whole descriptions v80 writes (keyed by the v80 names)."""
    return {
        ("group", "capacity_margin_method"): (
            "[choice] Whether the group needs a capacity margin in investment solves, and how the margin is given. "
            "none (default): no margin. absolute: capacity_margin_absolute MW. relative_of_peak_demand: "
            "capacity_margin_relative times the group's peak demand in the period. At every time step of an "
            "investment period, the capacity of the group's firm units (capacity x availability x "
            "capacity_max_coeff) and VRE units (capacity x profile x availability) must cover the group's demand "
            "plus the margin. Units fed from a storage node, connections and the units consuming from the group "
            "count by their actual flow, so exports add to the requirement; connections and lossy units inside the "
            "group are left out (their losses are not required). Units fed from fuel or commodity nodes count at "
            "capacity: the fuel is assumed to last."),
        ("group", "capacity_margin_absolute"): (
            "[MW] Capacity the group needs above its demand at every time step of an investment period. Used only "
            "with capacity_margin_method absolute. No value means 0 (the demand must still be covered). Constant or "
            "period. -inf means no capacity-margin requirement in that period."),
        ("group", "capacity_margin_relative"): (
            "[-] e.g. 0.15 means 15%. Capacity margin as a share of the group's peak demand: the largest sum of the "
            "demand of the group's nodes over the period's full time series (before representative periods and "
            "new_stepduration). Used only with capacity_margin_method relative_of_peak_demand. No value means 0. "
            "Constant or period. -inf means no capacity-margin requirement in that period."),
        ("solve", "solver_precommand"): (
            "[text] Not used: FlexTool never runs this command. Set a solver wrapper on your computer instead (see "
            "the solver documentation). Kept only so that older databases still read."),
        ("solve", "solver_arguments"): (
            "[text] Map of solver options (option name -> value), with the solver's own option names (e.g. presolve "
            "for HiGHS, Presolve for Gurobi). They apply to whichever solver the solve uses (solver) and are checked "
            "against FlexTool's list of that solver's options before the first solve: a name that is not one of "
            "its options is refused. They override the same option in the solver's option file "
            "(solver_config/<solver>.opt) and solver_mip_gap; the run's flags (e.g. --solver-time-limit) override "
            "them."),
        ("unit", "conversion_method"): (
            "[choice] How the unit converts inputs to outputs. constant_efficiency (default): output = efficiency "
            "× input. min_load_efficiency: the fuel use is a straight line from min_load (efficiency_at_min_load) "
            "to full load (efficiency); needs online variables. piecewise_linear: the fuel use follows the "
            "part-load curve piecewise_linear_efficiency; efficiency is not used. none: for a unit without inputs "
            "or without outputs (a unit with both converts as with constant_efficiency)."),
        ("solve", "reduce_unit_detail_method"): (
            "[choice] Detail of units and connections that this solve leaves out, to make it faster (typically an "
            "investment solve). none (default): all detail. integers: binary startup_method is solved as "
            "continuous and exact connections behave as regular. integers_ramps: integers, and no ramp limits or "
            "ramp costs. integers_min_limits: integers, and no min_load, capacity_min_coeff floors or minimum up / "
            "down times; a min_load_efficiency unit that is more efficient at min load than at full load "
            "(efficiency_at_min_load above efficiency) then runs at its full-load efficiency. "
            "integers_ramps_min_limits: integers, no ramps and no min limits. Under the integers levels, a unit or "
            "connection whose online variables nothing in the solve still uses (no start-up cost, min_load, "
            "capacity_min_coeff floor, minimum up / down time, min_load_efficiency curve, piecewise_linear curve "
            "whose efficiency rises from its first point, constraint_online_coeff in a user constraint, reserve, "
            "inertia or ramp limit left) has none, as with startup_method no_startup; the run log names them. "
            "online: everything of integers_min_limits, and no online variables: no start-up costs, user "
            "constraints that count units online are left out whole, min_load_efficiency and piecewise_linear "
            "units run at their full-load efficiency (the same fuel at full output; at part load less or more fuel "
            "than with their curve), and ramp limits apply to the capacity. Reserves are then limited by built "
            "capacity, not online capacity, and inertia is counted from flows. online_ramps: online, and no ramp "
            "limits or ramp costs. Never adds detail."),
        ("group", "print_dispatch"): (
            "[yes/no] Combines unit/connection flows, flowGroup aggregated flows, and other balance elements "
            "(slack, demand, internal_losses, etc.) into a set of timeseries and a preconfigured plot per group. "
            "Also gives the group's ramp envelope (ramp_envelope_t: how much the flows at its member nodes could "
            "still change toward the next step, next to the net-load ramp) and its Ramp envelope and Ramp "
            "frequency plots."),
    }


#: v80: ``unit.piecewise_linear_efficiency`` -- the part-load curve of a ``piecewise_linear`` unit.
V80_PIECEWISE_LINEAR_EFFICIENCY_DESCRIPTION = (
    "[-] Part-load efficiency curve of a piecewise_linear unit: a map from load share (0–1 of the unit's online "
    "capacity; of its built capacity when it has no online variables) to the efficiency at that load (output / "
    "input, each flow weighted by its conversion_flow_coeff, like efficiency; may be above 1). The last point must "
    "be at load 1 (full load). Fuel use is linear between the points. The extra input per extra output (the "
    "incremental heat rate) must not fall from one segment to the next: such a curve is an error, and the message "
    "gives the nearest curve that is accepted. When the efficiency rises from the first point to the second (the "
    "usual thermal curve), the unit burns fuel at zero output and needs online variables: a unit without "
    "startup_method gets linear start-ups (an info line says so). Below the first point the first segment is "
    "extended down to min_load; when the efficiency falls from the first point to the second (e.g. an "
    "electrolyser), the first point's efficiency applies below it instead. min_load is set separately and does not "
    "move the curve. A piecewise_linear unit does not use efficiency or efficiency_at_min_load, and its efficiency "
    "must not vary in time. One curve (map: load share → efficiency) or a curve per period (map: period → load "
    "share → efficiency).")


def _v80_piecewise_sentences() -> dict[tuple[str, str], tuple[str, ...]]:
    """The sentences v80 appends to the descriptions of the unit parameters a ``piecewise_linear`` unit reads
    differently (part 4)."""
    return {
        ("unit", "efficiency"): (" Not used by a piecewise_linear unit: the point at load 1 of its "
                                 "piecewise_linear_efficiency is its full-load efficiency.",),
        ("unit", "efficiency_at_min_load"): (" Used by min_load_efficiency units only.",),
        ("unit", "min_load"): (" For a piecewise_linear unit, min_load is separate from the first point of its "
                               "curve.",),
    }


def _v80_piecewise_linear(db) -> None:
    """``conversion_method`` ``piecewise_linear`` and ``unit.piecewise_linear_efficiency`` (part 4 of v80).

    ``piecewise_linear`` is appended to the ``conversion_methods`` value list (index = max + 1);
    ``unit.piecewise_linear_efficiency`` (types ``1d_map`` / ``2d_map``: a curve, or a curve per period; no default,
    no value list, group ``basics``) is added or updated to these fields; ``efficiency``, ``efficiency_at_min_load``
    and ``min_load`` get the sentences of :func:`_v80_piecewise_sentences` they do not end with yet (the
    ``conversion_method`` text is in :func:`_v80_texts`).  No value is rewritten.  Idempotent.
    """
    if not db.get_item("entity_class", name="unit"):
        return
    if db.get_item("parameter_value_list", name="conversion_methods"):
        items = db.get_items("list_value", parameter_value_list_name="conversion_methods")
        if "piecewise_linear" not in {x["parsed_value"] for x in items}:
            val, typ = to_database("piecewise_linear")
            db.add_list_value(parameter_value_list_name="conversion_methods", value=val, type=typ,
                              index=max((x["index"] for x in items), default=-1) + 1)
    fields = {"parameter_type_list": ("1d_map", "2d_map"), "parameter_group_name": "basics",
              "description": V80_PIECEWISE_LINEAR_EFFICIENCY_DESCRIPTION}
    pdef = _pdef_or_none(db, "unit", "piecewise_linear_efficiency")
    if pdef is None:
        db.add_parameter_definition(entity_class_name="unit", name="piecewise_linear_efficiency", **fields)
    elif any((tuple(pdef[k] or ()) if k == "parameter_type_list" else pdef[k]) != v for k, v in fields.items()):
        db.update_parameter_definition(id=pdef["id"], **fields)
    for (cls, name), sentences in _v80_piecewise_sentences().items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        description = _v80_append_sentences(pdef["description"] or "", sentences)
        if description != (pdef["description"] or ""):
            db.update_parameter_definition(id=pdef["id"], description=description)


def _v80_capacity_margin_rename(db) -> None:
    """``group.capacity_margin`` -> ``capacity_margin_absolute``, ``capacity_margin_share`` ->
    ``capacity_margin_relative`` and the ``capacity_margin_methods`` members ``manual`` -> ``absolute``,
    ``share_of_peak`` -> ``relative_of_peak_demand`` (part of v80).

    A definition rename keeps its values (they hang on the definition).  Each list member is renamed in place (its
    index, so the list order, is kept); when the new name is already a member, the old member is removed instead of
    renamed (never a duplicate).  Every stored ``capacity_margin_method`` value (all alternatives), and a default,
    then reads the new name, so each is a member of the list again; each renamed stored value is printed.  The values
    change only by name.  Idempotent: a rerun finds the new names.

    A stored value and a default refer to their list member.  The open mapping keeps reading the old name for them
    after the in-place rename, so each value is written again with the new name.  A default that still refers to the
    old member makes every later update of its definition fail ("not in the list"), so it is cleared before the
    members change and set to the new name after (:func:`_detach_list_defaults`).
    """
    if not db.get_item("entity_class", name="group"):
        return
    for old, new in V80_CAPACITY_MARGIN_RENAMES.items():
        _rename_or_drop_parameter_definition(db, "group", old, new)
    renames = {old: new for old, new in V80_CAPACITY_MARGIN_METHODS.items() if old != new}
    values = db.find_parameter_values(entity_class_name="group", parameter_definition_name="capacity_margin_method")
    renamed = []
    for pv in values:
        parsed = from_database(pv["value"], pv["type"])
        if isinstance(parsed, str) and parsed in renames:
            renamed.append(f"group {'/'.join(pv['entity_byname'])} [{pv['alternative_name']}]: {parsed} -> "
                           f"{renames[parsed]}")
    pending = []
    if db.get_item("parameter_value_list", name="capacity_margin_methods"):
        pending = _detach_list_defaults(db, "capacity_margin_methods", renames)
        items = db.get_items("list_value", parameter_value_list_name="capacity_margin_methods")
        members = {x["parsed_value"] for x in items}
        for item in sorted(items, key=lambda x: x["index"]):
            new = renames.get(item["parsed_value"], item["parsed_value"])
            if new == item["parsed_value"]:
                continue
            if new in members:
                # The new name is already a member (a database built from the current template but stamped with
                # an older version: an earlier step re-added the old name beside it).  Drop the old member; its
                # stored values are rewritten to the new name below.
                db.remove_item("list_value", item["id"])
            else:
                val, typ = to_database(new)
                db.update_list_value(id=item["id"], value=val, type=typ)
                members.add(new)
    # A value still reads the old name (the mapping keeps it after the in-place rename, as does a value of a removed
    # old member): write the new name.
    for pv in db.find_parameter_values(entity_class_name="group", parameter_definition_name="capacity_margin_method"):
        parsed = from_database(pv["value"], pv["type"])
        if isinstance(parsed, str) and parsed in renames:
            val, typ = to_database(renames[parsed])
            db.update_parameter_value(id=pv["id"], value=val, type=typ)
    _attach_list_defaults(db, pending)
    if renamed:
        print("v80: capacity_margin_method values renamed (capacity_margin is now capacity_margin_absolute, "
              "capacity_margin_share capacity_margin_relative): " + ", ".join(renamed))


def _v80_descriptions(db) -> None:
    """The whole descriptions of :func:`_v80_texts` (part of v80).  No value is rewritten.  Idempotent."""
    for (cls, name), text in _v80_texts().items():
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, name)
        if pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)


def _migrate_v80_schema_batch(db) -> None:
    """All schema changes of database version 80 (v79 -> v80), in one step.

    1. ``inf`` / ``-inf`` in a period-map limit is no limit in that period, and a ``new_stepduration`` solve step
       that combines a no-limit time step with finite ones has no limit for the whole step: the limit descriptions
       (:func:`_v80_limit_inf_descriptions`).
    2. ``group.capacity_margin`` is ``capacity_margin_absolute`` and ``capacity_margin_share`` is
       ``capacity_margin_relative``; the ``capacity_margin_method`` values ``manual`` / ``share_of_peak`` are
       ``absolute`` / ``relative_of_peak_demand``, in the value list and in every stored value
       (:func:`_v80_capacity_margin_rename`).
    3. The descriptions of the capacity-margin parameters (new names), of ``solve.solver_precommand`` (never run;
       use a solver wrapper) and of ``solve.solver_arguments`` (the solver's own option names, for whichever solver
       the solve uses, checked, over the option file), of ``unit.conversion_method`` (with piecewise_linear) and of
       ``solve.reduce_unit_detail_method`` (min_load_efficiency and piecewise_linear units run at their full-load
       efficiency without online variables, as do min_load_efficiency units more efficient at min load under the
       min_limits levels) and of ``group.print_dispatch`` (also the group's ramp envelope and its Ramp envelope and
       Ramp frequency plots) (:func:`_v80_descriptions`).
    4. ``conversion_method`` ``piecewise_linear``: the ``conversion_methods`` member, ``unit.piecewise_linear_efficiency``
       (a part-load curve, or a curve per period) and the sentences on ``efficiency``, ``efficiency_at_min_load`` and
       ``min_load`` (:func:`_v80_piecewise_linear`).

    Values change only by the renames of part 2.  Each part is idempotent, so a rerun of the step changes nothing
    (a database already at v80 gets part 4 by rerunning the step).
    """
    _v80_limit_inf_descriptions(db)
    _v80_capacity_margin_rename(db)
    _v80_descriptions(db)
    _v80_piecewise_linear(db)
    _commit_step(db, "v80: inf / -inf in a limit is no limit in that period, capacity_margin_absolute / "
                     "capacity_margin_relative and the capacity_margin_method values absolute / "
                     "relative_of_peak_demand, solver_precommand, solver_arguments, "
                     "reduce_unit_detail_method and print_dispatch descriptions, "
                     "conversion_method piecewise_linear and unit.piecewise_linear_efficiency")


# ─── v81: technical lifetime and payback time, net salvage value, no_reinvestment ───
#
# The step is table-driven: the tables below say what it adds, moves, renames and rewrites.  The tables are frozen
# since v82.  A database migrated to 81 by a development build of this step is brought up to them by the v81 refresh
# (:func:`_v81_refresh`, part 1 of v82): it adds the missing definitions and list members, and a shipped field of a
# definition that was superseded (:func:`_v81_superseded`: a default and its description, together; a definition
# without a default has the pair (None, description)) is replaced where the database still has exactly a pair
# FlexTool shipped, never a value and never an edited pair.  The refresh writes neither the behaviour-change markers
# nor a description that is not part of a superseded pair.

#: v81: parameters whose values move to new names (old name -> the new names that each get a copy of every value).
#: ``lifetime`` moves to ``lifetime_technical`` only: an entity without ``payback_time`` pays back over its
#: ``lifetime_technical`` (``engine/input/_lifetime_roles.py``), so a copy into ``payback_time`` would only repeat it.
#: Databases migrated to 81 by an earlier development build of this step also hold a ``payback_time`` copy of each
#: ``lifetime``; they are not migrated again and keep it, which builds the same model as the fallback (equal values,
#: both read at the period the capacity comes online, also after a construction lead time).
V81_VALUE_MOVES: dict[str, tuple[str, ...]] = {
    "lifetime": ("lifetime_technical",),
    "salvage_value": ("net_salvage_value",),
}
#: v81: value-list members renamed (list -> {old member: new member}); every stored value is rewritten.
V81_LIST_MEMBER_RENAMES: dict[str, dict[str, str]] = {
    "lifetime_methods": {"no_investment": "no_reinvestment"},
}
#: v81: value lists to create (when absent) with these members, or to complete with the members they lack (appended
#: in this order).
V81_NEW_VALUE_LISTS: dict[str, tuple[str, ...]] = {
    "end_of_life_methods": ("closest_seam", "previous_seam", "next_seam"),
}
#: v81: the default of ``end_of_life_method`` (the rounding of an end of life to a period start).
V81_END_OF_LIFE_METHOD_DEFAULT = "closest_seam"
#: v81: the default of ``end_of_life_method`` that FlexTool shipped before it became closest_seam (its description:
#: :func:`_v81_superseded_end_of_life_text`).
V81_SUPERSEDED_END_OF_LIFE_METHOD_DEFAULT = "next_seam"
#: The shipped default of ``discount_rate`` (unit, connection, node), which v81 keeps; with the superseded description
#: it forms a superseded pair (:func:`_v81_superseded`).
V81_DISCOUNT_RATE_DEFAULT = 0.05

_V81_CAPACITY = {"unit": ("capacity", "unit", "kW"), "connection": ("capacity", "connection", "kW"),
                 "node": ("storage capacity", "node", "kWh")}


def _v81_texts() -> dict[tuple[str, str], str]:
    """The whole descriptions v81 writes: those of its new definitions and of the existing definitions whose text
    names a renamed parameter or value."""
    t: dict[tuple[str, str], str] = {}
    for cls, (cap, what, per) in _V81_CAPACITY.items():
        t[(cls, "lifetime_technical")] = (
            f"[years] Technical lifetime: how long the {cap} operates. It starts when the {cap} starts to operate: "
            "with a construction lead time, after the construction. Its end is rounded to a period start by "
            "end_of_life_method; for existing capacity it counts from the model start minus age_at_model_start "
            "(or model.start_year - build_year). What happens at its end depends on "
            f"lifetime_method: reinvest_automatic keeps the {cap} to the end of the model horizon, reinvest_choice "
            "retires it, no_reinvestment retires it and allows investment only during the first technical lifetime "
            "from the model start. Without payback_time, invest_cost is annualized over it. Required when the "
            f"{what} invests or retires with a cost (invest_cost, net_salvage_value) and has no payback_time, and "
            "when lifetime_method is reinvest_choice or no_reinvestment (also for existing capacity without "
            f"investment). Constant or period: the value of the period when the {cap} comes online is used, for "
            f"existing {cap} that of the first period.")
        t[(cls, "payback_time")] = _v81_payback_time_text(cap, "changes nothing (a whole number of years)")
        t[(cls, "net_salvage_value")] = _v81_net_salvage_text(cap, per)
        t[(cls, "lifetime_method")] = (
            "[choice] What happens when capacity reaches the end of its lifetime_technical. reinvest_automatic "
            "(default): the capacity is kept to the end of the model horizon and the annualized investment cost is "
            "paid on, without a model choice. reinvest_choice: the capacity retires at the end of its "
            "lifetime_technical and the model decides how much new capacity to build. no_reinvestment: the "
            "capacity retires at the end of its lifetime_technical; investment is possible only during the first "
            "technical lifetime from the model start, and the capacity is not rebuilt. With reinvest_choice and "
            "no_reinvestment the investment cost is paid as a level yearly cost of the lifetime_technical (the "
            "annuity over payback_time spread over it) for each year the capacity is in service.")
        t[(cls, "discount_rate")] = (
            "[-] e.g. 0.05 means 5% per year. Discount rate for investments (WACC). Reflects the financing cost and "
            "risk premium for this technology. When the model inflation_rate > 0, this should be a nominal rate. "
            "When inflation_rate = 0, this should be a real rate. Used to annualize investment costs over the "
            "payback time, sized so that the yearly payments are worth invest_cost at this rate when the capacity "
            "starts to operate: annuity = invest_cost x r / (1 - (1 + r)^-payback_time) x (1 + r)^"
            "(inflation_offset_investment - 1). 0 gives the straight-line annuity invest_cost / payback_time. No "
            "value means 0.05. Must be 0 or above. Constant or period.")
        t[(cls, "invest_cost")] = (
            f"[CUR/{per}] Overnight investment cost of new {cap}. It is annualized over payback_time with "
            "discount_rate; with lifetime_method reinvest_choice or no_reinvestment that annuity is spread evenly "
            f"over lifetime_technical and paid for each year the {cap} is in service (every year to the end of the "
            "model horizon with reinvest_automatic). Construction time costs with "
            f"their interest must be included here. Constant or period: the value of the period when the {cap} is "
            "built (comes online) is used.")
        t[(cls, "age_at_model_start")] = (
            f"[years] Age of the existing {cap} at the start of the model. Its end of life is lifetime_technical "
            "minus the age from the model start, rounded to a period start by end_of_life_method; with "
            f"lifetime_method reinvest_automatic the {cap} is kept and the age has no effect. Give "
            "age_at_model_start or build_year, not both. No value: new at the model start. Constant, 0 or more.")
        t[(cls, "build_year")] = (
            f"[year] Calendar year the existing {cap} was built: its age at the model start is model.start_year "
            "minus build_year. Needs model.start_year and must not be after it. Not with age_at_model_start. "
            "Constant.")
        t[(cls, "end_of_life_method")] = (
            f"[choice] How the end of life of {cap} (start of operation plus lifetime_technical; existing {cap}: "
            "lifetime_technical minus its age, from the model start) is rounded to a period start. closest_seam "
            "(default): the nearer start, a tie goes to the later one. next_seam: the start at or after it "
            "(operates to the end of the period its life ends in; the rounding of FlexTool 4.5 and earlier). "
            "previous_seam: the start at or before it (gone for that whole period; for capacity-adequacy studies, "
            "which must not count capacity past its life). The yearly investment cost is paid for each year in "
            f"service. An end of life after the model horizon keeps the {cap} to the end. Used with "
            "lifetime_method reinvest_choice / no_reinvestment.")
    t[("model", "start_year")] = (
        "[year] The calendar year in which the first period of the model's first solve starts. Used by "
        "build_year (age = start_year - build_year).")
    t[("node", "unit_of_measure")] = V79_NODE_UNIT_OF_MEASURE_DESCRIPTION.replace(
        "(invest_cost, fixed_cost, salvage_value)", "(invest_cost, fixed_cost, net_salvage_value)")
    t[("group", "penalty_capacity_margin")] = (
        "[CUR/kW/a] Penalty for violating the capacity margin constraint. Uses operational discounting (not "
        "annualized over a payback time like investment costs), so the value is not directly comparable to "
        "annualized investment costs. Constant or period.")
    return t


def _v81_payback_time_text(cap: str, equal_rates: str) -> str:
    """The description of ``payback_time`` on a class whose capacity is *cap*; *equal_rates* says what payback_time
    does when discount_rate equals inflation_rate (the shipped and the superseded wording differ only there)."""
    return ("[years] Years over which invest_cost is annualized with discount_rate (the annuity factor): with "
            "discount_rate it sets the size of the yearly investment cost. With lifetime_method reinvest_choice or "
            "no_reinvestment the present value of the annuity (discounted at the model inflation_rate) is spread "
            "evenly over lifetime_technical and paid for each year the capacity is in service: with discount_rate "
            f"equal to inflation_rate payback_time {equal_rates}, below it a shorter payback_time costs more. "
            "With reinvest_automatic the annuity is paid in every year to the end of the model horizon. No value: "
            "lifetime_technical. Must be above 0. Constant or period: the value of the period when the "
            f"{cap} comes online is used.")


def _v81_net_salvage_text(cap: str, per: str) -> str:
    """The description of ``net_salvage_value`` on a class whose capacity is *cap*, priced per *per*."""
    return (f"[CUR/{per}] Net salvage value: scrap value minus decommissioning cost, negative when decommissioning "
            "costs more than the scrap is worth. A retirement of existing capacity receives it in full at the start "
            "of the retirement period (the investment in existing capacity is sunk). With lifetime_method "
            "reinvest_choice or no_reinvestment the value at the end of life is a level yearly credit over the "
            f"lifetime at discount_rate: it lowers the yearly cost of new {cap}, existing {cap} holds it over its "
            "service years, and a retirement gives the credits back. Constant or period: the value of the period "
            "the salvage happens in, after the last period the last value.")


def _v81_superseded_net_salvage_text(cap: str, per: str) -> str:
    """The description of ``net_salvage_value`` that FlexTool shipped before the salvage terms of retirement and of
    new capacity (verbatim)."""
    return (f"[CUR/{per}] Net salvage value: scrap value minus decommissioning cost, received once when {cap} is "
            "retired (at end of life or earlier). Negative when decommissioning costs more than the scrap is worth. "
            "Does not depend on age. Constant or period.")


def _v81_superseded_end_of_life_text(cls: str) -> str:
    """The description of ``end_of_life_method`` on ``cls`` that FlexTool shipped with the default ``next_seam``
    (verbatim)."""
    cap = _V81_CAPACITY[cls][0]
    return (f"[choice] How the end of life of {cap} (start of operation plus lifetime_technical; existing {cap}: "
            "lifetime_technical minus its age, from the model start) is rounded to a period start. next_seam "
            "(default): the start at or after it (operates to the end of the period its life ends in). "
            "closest_seam: the nearer start, a tie goes to the later one. previous_seam: the start at or before "
            f"it (gone for that whole period). An end of life after the model horizon keeps the {cap} to the "
            "end. Used with lifetime_method reinvest_choice / no_reinvestment.")


def _v81_superseded_discount_rate_text() -> str:
    """The description of ``discount_rate`` (unit, connection, node) that FlexTool shipped with the annuity formula
    before the yearly payment was sized for when it is paid (verbatim)."""
    return ("[-] e.g. 0.05 means 5% per year. Discount rate for investments (WACC). Reflects the financing cost and "
            "risk premium for this technology. When the model inflation_rate > 0, this should be a nominal rate. "
            "When inflation_rate = 0, this should be a real rate. Used to annualize investment costs over the "
            "payback time: annuity = invest_cost x r / (1 - (1 + r)^-payback_time). 0 gives the straight-line "
            "annuity invest_cost / payback_time. No value means 0.05. Must be 0 or above. Constant or period.")


def _v81_superseded() -> dict[tuple[str, str], tuple[str | None, str]]:
    """``{(class, name): (default, description)}``: the default and description pairs of v81 definitions that
    FlexTool shipped and has since superseded (the current pair is in :func:`_v81_definitions`; for a definition
    that v81 does not add, only the description of :func:`_v81_texts` is current and the default stays).  A
    database at 81 that has exactly such a pair gets the current one (:func:`_refresh_head_schema`)."""
    out: dict[tuple[str, str], tuple[str | None, str]] = {}
    for cls, (cap, _what, per) in _V81_CAPACITY.items():
        out[(cls, "end_of_life_method")] = (V81_SUPERSEDED_END_OF_LIFE_METHOD_DEFAULT,
                                            _v81_superseded_end_of_life_text(cls))
        out[(cls, "payback_time")] = (None, _v81_payback_time_text(cap, "changes nothing"))
        out[(cls, "net_salvage_value")] = (None, _v81_superseded_net_salvage_text(cap, per))
        out[(cls, "discount_rate")] = (V81_DISCOUNT_RATE_DEFAULT, _v81_superseded_discount_rate_text())
    return out


def _v81_definitions(texts: dict[tuple[str, str], str] | None = None) -> dict[tuple[str, str], dict]:
    """The definitions v81 adds, with the fields it gives them (types, group, default, value list, description).
    *texts* replaces a description of :func:`_v81_texts` (a later step's text of the same definition)."""
    texts = {**_v81_texts(), **(texts or {})}
    out: dict[tuple[str, str], dict] = {}
    for cls in _V81_CAPACITY:
        for name, group in (("lifetime_technical", "investment"), ("payback_time", "investment"),
                            ("net_salvage_value", "retirement")):
            out[(cls, name)] = {"parameter_type_list": ("float", "1d_map"), "parameter_group_name": group,
                                "default_value": None, "default_type": None, "parameter_value_list_name": None,
                                "description": texts[(cls, name)]}
        for name, types in (("age_at_model_start", ("float",)), ("build_year", ("float",))):
            out[(cls, name)] = {"parameter_type_list": types, "parameter_group_name": "retirement",
                                "default_value": None, "default_type": None, "parameter_value_list_name": None,
                                "description": texts[(cls, name)]}
        default_value, default_type = to_database(V81_END_OF_LIFE_METHOD_DEFAULT)
        out[(cls, "end_of_life_method")] = {"parameter_type_list": ("str",), "parameter_group_name": "retirement",
                                          "default_value": default_value, "default_type": default_type,
                                          "parameter_value_list_name": "end_of_life_methods",
                                          "description": texts[(cls, "end_of_life_method")]}
    out[("model", "start_year")] = {"parameter_type_list": ("float",), "parameter_group_name": "model",
                                    "default_value": None, "default_type": None, "parameter_value_list_name": None,
                                    "description": texts[("model", "start_year")]}
    return out


#: Colour and priority of the parameter groups v81 definitions use (as v44 / v45 create them), for a database built
#: without the standard groups.
_V81_GROUPS = {"investment": ("7fb095", 20), "retirement": ("a3b08c", 25), "model": ("a3a3a3", 80)}


def _v81_value_lists(db) -> None:
    """Part 1 of v81: each list of :data:`V81_NEW_VALUE_LISTS` exists and has its members (the missing ones are
    appended in the table's order).  Idempotent."""
    for name, members in V81_NEW_VALUE_LISTS.items():
        if not db.get_item("parameter_value_list", name=name):
            db.add_parameter_value_list(name=name)
        _append_list_members(db, name, members)


def _v81_definitions_part(db, *, update: bool = True, texts: dict[tuple[str, str], str] | None = None) -> None:
    """Part 2 of v81: each definition of :func:`_v81_definitions` (with *texts*) on an existing class is added, or
    (``update``) set to the given fields.  ``update=False`` only adds the missing ones (an existing one keeps its own
    fields, e.g. a description the user edited).  Idempotent."""
    for (cls, name), fields in _v81_definitions(texts).items():
        if not db.get_item("entity_class", name=cls):
            continue
        group = fields["parameter_group_name"]
        if group is not None:
            _ensure_parameter_group(db, group, *_V81_GROUPS[group])
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            db.add_parameter_definition(entity_class_name=cls, name=name,
                                        **{k: v for k, v in fields.items() if v is not None})
        elif update and any((tuple(pdef[k] or ()) if k == "parameter_type_list" else pdef[k]) != v
                            for k, v in fields.items()):
            db.update_parameter_definition(id=pdef["id"], **fields)


def _v81_move_values(db) -> None:
    """Part 3 of v81: the values of each old parameter of :data:`V81_VALUE_MOVES` go to each of its new names, then
    the old definition is removed.

    Every value goes as it is stored (its bytes and type: a constant, a map with any index, every alternative), on
    each class that defines the old name, to each new name where that (entity, alternative) has no value yet; a
    value already there is kept, and the old one it would have replaced is printed.  Without the old definition the
    part does nothing, so a rerun never brings back a value.

    A value goes to the last new name by moving it there (it keeps its row) and to any other new name by a copy, so
    the only values removed with the old definition are those a new name already had, in bounded commits
    (:func:`_remove_items_bounded`): a database of any size moves.
    """
    for old, news in V81_VALUE_MOVES.items():
        for pdef in [d for d in db.get_items("parameter_definition") if d["name"] == old]:
            cls = pdef["entity_class_name"]
            values = list(db.find_parameter_values(entity_class_name=cls, parameter_definition_name=old))
            targets = [new for new in news if _pdef_or_none(db, cls, new) is not None]
            for new in targets:
                move = new == targets[-1]
                taken = {(tuple(pv["entity_byname"]), pv["alternative_name"])
                         for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name=new)}
                kept = []
                for pv in values:
                    key = (tuple(pv["entity_byname"]), pv["alternative_name"])
                    if key in taken:
                        kept.append(pv)
                    elif move:
                        db.update_parameter_value(id=pv["id"], parameter_definition_name=new)
                    else:
                        db.add_parameter_value(entity_class_name=cls, entity_byname=key[0],
                                               parameter_definition_name=new, alternative_name=key[1],
                                               value=pv["value"], type=pv["type"])
                if kept:
                    print(f"v81: {cls} {new} already had a value where {old} had one; kept the {new} value, the "
                          f"{old} value was not copied to {new}:")
                    for pv in kept:
                        print("  " + _describe_pv(pv))
            _remove_items_bounded(db, "parameter_definition", [pdef["id"]], f"v81: removed {cls}.{old}")


def _v81_list_member_renames(db) -> None:
    """Part 4 of v81: the value-list members of :data:`V81_LIST_MEMBER_RENAMES` are renamed.

    Each member is renamed in place (its index, so the list order, is kept); when the new name is already a member
    (a database built from the current template but stamped with an older version), the old member is removed
    instead (never a duplicate).  Every stored value, and a default, of every definition that uses the list then
    reads the new name; each renamed stored value is printed.  Idempotent: a rerun finds the new names.

    A stored value and a default refer to their list member, so a value follows the in-place rename on its own and
    is found by its old name only before it.  A default that still refers to the old member makes every later update
    of its definition fail ("not in the list"), so it is cleared before the members change and set to the new name
    after (:func:`_detach_list_defaults`, :func:`_attach_list_defaults`).
    """
    for list_name, renames in V81_LIST_MEMBER_RENAMES.items():
        if not db.get_item("parameter_value_list", name=list_name):
            continue
        pdefs = [d for d in db.get_items("parameter_definition") if d["parameter_value_list_name"] == list_name]
        pending = _detach_list_defaults(db, list_name, renames)
        renamed = []
        for pdef in pdefs:
            for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                               parameter_definition_name=pdef["name"]):
                parsed = from_database(pv["value"], pv["type"])
                if isinstance(parsed, str) and parsed in renames:
                    renamed.append(f"{pv['entity_class_name']} {'/'.join(pv['entity_byname'])} "
                                   f"{pdef['name']} [{pv['alternative_name']}]: {parsed} -> {renames[parsed]}")
        items = db.get_items("list_value", parameter_value_list_name=list_name)
        members = {x["parsed_value"] for x in items}
        for item in sorted(items, key=lambda x: x["index"]):
            new = renames.get(item["parsed_value"])
            if new is None:
                continue
            if new in members:
                db.remove_item("list_value", item["id"])
            else:
                val, typ = to_database(new)
                db.update_list_value(id=item["id"], value=val, type=typ)
                members.add(new)
        for pdef in pdefs:
            # A value of a removed old member (the new name was already a member) still reads the old name.
            for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                               parameter_definition_name=pdef["name"]):
                parsed = from_database(pv["value"], pv["type"])
                if isinstance(parsed, str) and parsed in renames:
                    val, typ = to_database(renames[parsed])
                    db.update_parameter_value(id=pv["id"], value=val, type=typ)
        _attach_list_defaults(db, pending)
        if renamed:
            print("v81: value names changed: " + ", ".join(renamed))


def _v81_descriptions(db) -> None:
    """Part 5 of v81: the whole descriptions of :func:`_v81_texts`.  No value is rewritten.  Idempotent."""
    for (cls, name), text in _v81_texts().items():
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, name)
        if pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)


def _v81_annuity_marker(db) -> None:
    """Part 6 of v81: the behaviour-change marker ``annuity_over_payback_time`` (``{"since": 81, "entities": [...]}``)
    on a database where an entity with the ``lifetime_method`` ``reinvest_choice`` or ``no_reinvestment`` (in some
    alternative) has a non-zero ``invest_cost``: its investment cost is now the level yearly cost of the plant over
    its ``lifetime_technical`` (the annuity over ``payback_time``, its present value spread evenly over the
    lifetime), paid for each year the capacity is in service (before: the annuity in every period that started
    before the end of the lifetime).  A run names those whose investment cost changes.  No value is rewritten.
    Idempotent: a rerun finds the marker written.
    """
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        bounded = {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name=cls, parameter_definition_name="lifetime_method")
            if from_database(pv["value"], pv["type"]) in ("reinvest_choice", "no_reinvestment")}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="invest_cost"):
            name = pv["entity_byname"][0]
            if name in bounded and any(v != 0.0 for v in _v79_numbers(from_database(pv["value"], pv["type"]))):
                entities.add(name)
    if entities:
        _v77_set_marker(db, "annuity_over_payback_time", json.dumps({"since": 81, "entities": sorted(entities)}))


def _v81_end_of_life_marker(db) -> None:
    """Part 7 of v81: the behaviour-change marker ``end_of_life_closest_seam`` (``{"since": 81, "entities": [...]}``)
    on a database with an entity whose ``lifetime_method`` is ``reinvest_choice`` or ``no_reinvestment`` (in some
    alternative): ``end_of_life_method`` now defaults to ``closest_seam``, so an end of life inside a period may be
    rounded to the start of that period (before: always to the next period start), which moves its capacity in
    service and fixed cost.  A run names those whose end of life moves and that set no ``end_of_life_method``.  No
    value is rewritten.  Idempotent: a rerun finds the marker written.
    """
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        entities |= {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name=cls, parameter_definition_name="lifetime_method")
            if from_database(pv["value"], pv["type"]) in ("reinvest_choice", "no_reinvestment")}
    if entities:
        _v77_set_marker(db, "end_of_life_closest_seam", json.dumps({"since": 81, "entities": sorted(entities)}))


def _v81_fixed_cost_marker(db) -> None:
    """Part 8 of v81: the behaviour-change marker ``fixed_cost_per_operating_period`` (``{"since": 81, "entities":
    [...]}``) on a database where an entity with an ``invest_method`` that allows investment (in some alternative)
    has a ``fixed_cost`` whose values differ (a period map, in some alternative): new capacity now pays the
    fixed_cost of each period it operates in (before: that of the period it came online, in every period).  A run
    names those whose fixed cost changes.  No value is rewritten.  Idempotent: a rerun finds the marker written.
    """
    no_invest = {"not_allowed", "retire_period", "retire_total", "retire_no_limit", "retire_period_total"}
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        investing = {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name=cls, parameter_definition_name="invest_method")
            if from_database(pv["value"], pv["type"]) not in no_invest}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="fixed_cost"):
            name = pv["entity_byname"][0]
            if name in investing and len(set(_v79_numbers(from_database(pv["value"], pv["type"])))) > 1:
                entities.add(name)
    if entities:
        _v77_set_marker(db, "fixed_cost_per_operating_period", json.dumps({"since": 81, "entities": sorted(entities)}))


def _v81_annuity_timing_marker(db) -> None:
    """Part 9 of v81: the behaviour-change marker ``annuity_timing`` (``{"since": 81, "entities": [...]}``) on a
    database where an entity has a non-zero ``invest_cost`` with an ``invest_method`` that allows investment, or a
    non-zero ``net_salvage_value`` with one that allows retirement (in some alternative), and a ``discount_rate``
    that is not 0 in every alternative: the yearly payment of the annuity is now sized for the moment
    ``inflation_offset_investment`` places it in the year (the annuity times (1 + discount_rate)^(offset − 1)), so
    the loan is worth the investment cost at the discount_rate; before, at the default offset 0 it was worth the
    cost times 1 + discount_rate.  A run names those whose annuity changes (offset not 1).  No value is rewritten.
    Idempotent: a rerun finds the marker written.
    """
    no_invest = {"not_allowed", "retire_period", "retire_total", "retire_no_limit", "retire_period_total"}
    no_retire = {"not_allowed", "invest_period", "invest_total", "invest_no_limit", "invest_period_total"}
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        methods: dict[str, set[str]] = {}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="invest_method"):
            methods.setdefault(pv["entity_byname"][0], set()).add(from_database(pv["value"], pv["type"]))
        rates: dict[str, list[float]] = {}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="discount_rate"):
            rates.setdefault(pv["entity_byname"][0], []).extend(
                _v79_numbers(from_database(pv["value"], pv["type"])))
        for param, excluded in (("invest_cost", no_invest), ("net_salvage_value", no_retire)):
            for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name=param):
                name = pv["entity_byname"][0]
                if (methods.get(name, set()) - excluded
                        and any(v != 0.0 for v in _v79_numbers(from_database(pv["value"], pv["type"])))
                        and not (rates.get(name) and all(v == 0.0 for v in rates[name]))):
                    entities.add(name)
    if entities:
        _v77_set_marker(db, "annuity_timing", json.dumps({"since": 81, "entities": sorted(entities)}))


def _v81_net_salvage_marker(db) -> None:
    """Part 10 of v81: the behaviour-change marker ``net_salvage_once`` (``{"since": 81, "entities": [...]}``) on a
    database where an entity has a non-zero ``net_salvage_value`` (in some alternative) and an ``invest_method`` that
    allows retirement, or one that allows investment together with the ``lifetime_method`` ``reinvest_choice`` or
    ``no_reinvestment``: a retirement of existing capacity now receives the whole net salvage value at the start of
    the retirement period (before: its annuity over lifetime_technical, inside the horizon), and the net salvage
    value at the end of life of new capacity lowers its yearly cost (before: no effect).  A run names those whose
    costs change.  No value is rewritten.  Idempotent: a rerun finds the marker written.
    """
    no_invest = {"not_allowed", "retire_period", "retire_total", "retire_no_limit", "retire_period_total"}
    no_retire = {"not_allowed", "invest_period", "invest_total", "invest_no_limit", "invest_period_total"}
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        methods: dict[str, set[str]] = {}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="invest_method"):
            methods.setdefault(pv["entity_byname"][0], set()).add(from_database(pv["value"], pv["type"]))
        bounded = {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name=cls, parameter_definition_name="lifetime_method")
            if from_database(pv["value"], pv["type"]) in ("reinvest_choice", "no_reinvestment")}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="net_salvage_value"):
            name = pv["entity_byname"][0]
            own = methods.get(name, set())
            if (any(v != 0.0 for v in _v79_numbers(from_database(pv["value"], pv["type"])))
                    and (own - no_retire or (name in bounded and own - no_invest))):
                entities.add(name)
    if entities:
        _v77_set_marker(db, "net_salvage_once", json.dumps({"since": 81, "entities": sorted(entities)}))


def _v81_built_retirable_marker(db) -> None:
    """Part 11 of v81: the behaviour-change marker ``built_capacity_retirable`` (``{"since": 81, "entities":
    [...]}``) on a database where an entity has an ``invest_method`` (in some alternative) that allows both
    investment and retirement and the ``lifetime_method`` ``reinvest_choice`` or ``no_reinvestment``: the capacity
    the model builds can now be retired too (before: only the existing capacity).  A run names those with a
    retirement option of model-built capacity.  No value is rewritten.  Idempotent: a rerun finds the marker
    written.
    """
    no_invest = {"not_allowed", "retire_period", "retire_total", "retire_no_limit", "retire_period_total"}
    no_retire = {"not_allowed", "invest_period", "invest_total", "invest_no_limit", "invest_period_total"}
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue
        bounded = {pv["entity_byname"][0] for pv in db.find_parameter_values(
            entity_class_name=cls, parameter_definition_name="lifetime_method")
            if from_database(pv["value"], pv["type"]) in ("reinvest_choice", "no_reinvestment")}
        for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name="invest_method"):
            method = from_database(pv["value"], pv["type"])
            if pv["entity_byname"][0] in bounded and method not in no_invest and method not in no_retire:
                entities.add(pv["entity_byname"][0])
    if entities:
        _v77_set_marker(db, "built_capacity_retirable", json.dumps({"since": 81, "entities": sorted(entities)}))


def _migrate_v81_schema_batch(db) -> None:
    """All schema changes of database version 81 (v80 -> v81), in one step.

    1. The value lists of :data:`V81_NEW_VALUE_LISTS` and their members (:func:`_v81_value_lists`):
       ``end_of_life_methods`` (closest_seam, previous_seam, next_seam).
    2. ``lifetime_technical`` and ``payback_time`` (group ``investment``) and ``net_salvage_value`` (group
       ``retirement``) on unit, connection and node: a constant or a period map, no default; on the same
       classes (group ``retirement``) ``age_at_model_start`` and ``build_year`` (a constant, no default) and
       ``end_of_life_method`` (a choice of ``end_of_life_methods``, default ``closest_seam``); ``model.start_year``
       (group ``model``, a constant, no default) (:func:`_v81_definitions_part`).
    3. ``lifetime`` values go to ``lifetime_technical`` (how long the capacity operates); ``payback_time`` (the
       years of the annuity) gets no value: without one an entity pays back over its ``lifetime_technical``.
       ``salvage_value`` values go to ``net_salvage_value``; the old definitions are removed
       (:func:`_v81_move_values`).  A database migrated to 81 by an earlier development build of this step, which
       also copied each ``lifetime`` value to ``payback_time``, keeps that copy (a value move never reaches a
       database already at 81); the copy equals the fallback, and both are read at the period the capacity comes
       online (also after a construction lead time), so the model is the same.
    4. The ``lifetime_method`` value ``no_investment`` is ``no_reinvestment``, in the value list and in every stored
       value (:func:`_v81_list_member_renames`).
    5. The descriptions of the new definitions and of ``lifetime_method``, ``discount_rate``, ``invest_cost``,
       ``node.unit_of_measure`` and ``group.penalty_capacity_margin`` (:func:`_v81_descriptions`).
    6. The behaviour-change marker ``annuity_over_payback_time`` on the entities whose investment cost is now the
       level yearly cost of the lifetime, paid for each year in service (:func:`_v81_annuity_marker`).
    7. The behaviour-change marker ``end_of_life_closest_seam`` on the entities with ``lifetime_method``
       ``reinvest_choice`` / ``no_reinvestment``, whose end of life is now rounded to the closest period start
       (:func:`_v81_end_of_life_marker`).
    8. The behaviour-change marker ``fixed_cost_per_operating_period`` on the investing entities whose
       ``fixed_cost`` differs between periods: new capacity now pays the fixed cost of each period it operates in
       (:func:`_v81_fixed_cost_marker`).
    9. The behaviour-change marker ``annuity_timing`` on the entities with an investment or salvage annuity: its
       yearly payment is now sized for where inflation_offset_investment places it in the year
       (:func:`_v81_annuity_timing_marker`).
    10. The behaviour-change marker ``net_salvage_once`` on the entities with a net_salvage_value that a retirement
        or a new plant's yearly cost now counts differently (:func:`_v81_net_salvage_marker`).
    11. The behaviour-change marker ``built_capacity_retirable`` on the reinvest_choice / no_reinvestment entities
        that may invest and retire: the capacity the model builds can now be retired (:func:`_v81_built_retirable_marker`).

    Values change only by the moves of part 3 and the renames of part 4.  Each part is idempotent, so a rerun of the
    step changes nothing.
    """
    _v81_value_lists(db)
    _v81_definitions_part(db)
    _v81_move_values(db)
    _v81_list_member_renames(db)
    _v81_descriptions(db)
    _v81_annuity_marker(db)
    _v81_end_of_life_marker(db)
    _v81_fixed_cost_marker(db)
    _v81_annuity_timing_marker(db)
    _v81_net_salvage_marker(db)
    _v81_built_retirable_marker(db)
    _commit_step(db, "v81: lifetime is lifetime_technical (payback_time added, no value), salvage_value is "
                     "net_salvage_value, the lifetime_method value no_investment is no_reinvestment")


# ─── v82: social_discount_rate and the discount offsets ───
#
# The step renames the model's discount parameters and rewrites the descriptions that name them.  While 82 is the
# current data version, new schema entries go to the v82 tables; the v81 tables are frozen.  A database already at 81
# from a development build of v81 gets, from the v81 refresh, the v81 definitions and list members it lacks and the
# current pair in place of a superseded shipped one, then the whole v82 step; the v81 behaviour-change markers and a
# v81 description that is not a superseded pair are not written to it.

#: v82: ``model`` parameters renamed (old name -> new name).  Each keeps its id, so its values, default, types and
#: group (published for the assistant's old-name translation).
V82_PARAMETER_RENAMES: dict[str, str] = {
    "inflation_rate": "social_discount_rate",
    "inflation_offset_investment": "discount_offset_investment",
    "inflation_offset_operations": "discount_offset_operations",
}
#: v82: the parameters of a life extension, added on unit, connection and node (types ``float`` and ``1d_map``, group
#: ``investment``, no default).
V82_LIFE_EXTENSION: tuple[str, ...] = ("life_extension_cost", "life_extension_years")
#: v82: the default a definition of :func:`_v82_definitions` gets when the step adds it (a renamed one keeps its own).
#: ``social_discount_rate`` has none: without a value the engine uses 0; a life extension has none: it is opt-in.
V82_DEFAULTS: dict[str, float | None] = {
    "social_discount_rate": None,
    "discount_offset_investment": 0.0,
    "discount_offset_operations": 0.5,
    **{name: None for name in V82_LIFE_EXTENSION},
}


def _v82_discount_rate_text() -> str:
    """The description of ``discount_rate`` on unit, connection and node from v82 on."""
    return ("[-] e.g. 0.05 means 5% per year. Discount rate for investments (WACC). Reflects the financing cost and "
            "risk premium for this technology. In the same terms as the cost inputs: a real rate for inputs in real "
            "prices, a nominal rate for nominal inputs. Not the model's social_discount_rate, which weighs costs over "
            "time. Used to annualize investment costs over the payback time, sized so that the yearly payments are "
            "worth invest_cost at this rate when the capacity starts to operate: annuity = invest_cost x r / (1 - "
            "(1 + r)^-payback_time) x (1 + r)^(discount_offset_investment - 1). 0 gives the straight-line annuity "
            "invest_cost / payback_time. No value means 0.05. Must be 0 or above. Constant or period.")


def _v82_texts() -> dict[tuple[str, str], str]:
    """The whole descriptions v82 writes: those of the renamed definitions and of the definitions whose text names
    one of them."""
    t: dict[tuple[str, str], str] = {
        ("model", "social_discount_rate"): (
            "[-] e.g. 0.03 means 3% per year. The model's time discount rate: a cost or income is weighed by 1 / "
            "(1 + social_discount_rate)^t, with t the years from the start of the run to the moment it falls (its "
            "year plus discount_offset_investment or discount_offset_operations). Give it in the same terms as the "
            "cost inputs. 0 (default): no time preference; a cost in a later period weighs the same as now (with "
            "real-price inputs, inflation is already in the prices). Above 0: later costs and incomes weigh less "
            "(the planner's preference for now over later); with nominal inputs, use the combined rate "
            "(1 + inflation) x (1 + social rate) - 1. Not the discount_rate of a unit, connection or node: that is "
            "the investor's rate (WACC), which turns invest_cost into yearly payments. Constant."),
        ("model", "discount_offset_investment"): (
            "[years] Where in each year the yearly investment payments and the yearly salvage credits fall, "
            "counted from the start of the year, between 0 and 1: 0 (default) the start, 1 the end. The loan over "
            "payback_time is sized so that, at the entity's discount_rate, its payments are worth invest_cost when "
            "the capacity starts to operate, at any offset; the salvage credits are sized the same way. The offset "
            "moves the payments and credits that social_discount_rate weighs. No effect when both rates are 0. "
            "Constant."),
        ("model", "discount_offset_operations"): (
            "[years] Where in each year operating and fixed costs are weighed by social_discount_rate, counted from "
            "the start of the year, between 0 and 1: 0.5 (default) the middle of the year, standing for costs "
            "spread over the year. No effect when social_discount_rate is 0. Constant."),
    }
    t.update(_v82_storage_texts())
    for cls, (cap, what, per) in _V81_CAPACITY.items():
        t[(cls, "discount_rate")] = _v82_discount_rate_text()
        t[(cls, "payback_time")] = _v82_shipped_payback_time_text(cap).replace(
            "With reinvest_automatic the annuity is paid in every year to the end of the model horizon.",
            "With reinvest_automatic each plant (the first one and each replacement) is priced the same way, at the "
            "values of the period it comes online.")
        t.update(_v82_replacement_texts(cls, cap, what, per))
        t.update(_v82_life_extension_texts(cls, cap, per))
    return t


def _v82_storage_texts() -> dict[tuple[str, str], str]:
    """The descriptions of ``storage_binding_method`` and ``storage_nested_fix_method``: v82 closes cyclic storage
    over the rolls of a rolling solve, pins the representative-period chain ends, matches a forward chain's end to
    a free start, and makes the nested fixes cumulative (fix_usage) and interpolated (fix_quantity)."""
    return {
        ("node", "storage_binding_method"): (
            "[choice] How the storage state is kept over discontinuous timelines and from roll to roll. "
            "'bind_within_timeblock' cycles the state within each timeblock; 'bind_within_period' cycles within each "
            "FlexTool period (blocks chain inside the period); 'bind_within_solve' cycles over the whole solve; "
            "'bind_forward_only' (default) chains the state forward with no closure (where the solve chooses the "
            "start, the model ends at least at it). 'bind_within_solve_blended_weights', "
            "'bind_within_period_blended_weights' and 'bind_forward_only_blended_weights' are the representative-"
            "period variants: the cycle or chain runs on the base-period chain of the chronological level, where "
            "storage_state_start, storage_state_end and use_reference_value act; in a solve whose timesets have no "
            "representative_period_weights a blended method is used as its plain variant. 'bind_intraperiod_blocks' "
            "is an aggregation method: the state is held constant within each block, the block-total flow is "
            "balanced and the blocks of a period loop. In a rolling_window solve without representative periods a "
            "cycle is closed over the rolls: a roll that starts inside the solve or period continues it from the "
            "level realised by the previous roll and ends its part at least at that level, and the roll(s) holding "
            "its end end it at the level it started from; such a solve uses bind_within_timeblock as "
            "bind_within_period and needs a continuous timeline (a period with several timeblocks is a "
            "configuration error). A rolling solve on representative periods is a configuration error, and so is "
            "bind_forward_only (also the default), bind_within_solve, bind_within_period or bind_intraperiod_blocks "
            "in a solve on representative periods (use a blended method or bind_within_timeblock). "
            "storage_start_end_method fixes the start and the end of the whole model."),
        ("node", "storage_nested_fix_method"): (
            "[choice] Used in nested solve sequences: makes this storage a long-term storage whose planned level the "
            "upper-level solve hands to the lower-level solves it contains (at the steps of its "
            "fix_storage_periods). fix_quantity: the lower solve's state at the end of its horizon is the upper "
            "level at that moment (interpolated inside an upper step). fix_usage: the lower solve's net use of the "
            "storage is at most the upper solve's, counted from the start of the period (less what its earlier "
            "rolls used; an upper step that the horizon cuts counts pro rata). fix_price: the lower solve values "
            "its end state at the upper solve's price; requires storage_state_reference_price. A lower solve whose "
            "end is fixed this way starts from the upper solve's planned level where neither a realised level nor "
            "fix_start applies."),
    }


def _v82_life_extension_texts(cls: str, cap: str, per: str) -> dict[tuple[str, str], str]:
    """The descriptions of :data:`V82_LIFE_EXTENSION` on the class *cls* whose capacity is *cap* (priced per
    *per*)."""
    both = "Give both life_extension_cost and life_extension_years, or neither."
    return {
        (cls, "life_extension_cost"): (
            f"[CUR/{per}] One-off cost of keeping {cap} in service for life_extension_years beyond the end of its "
            "lifetime_technical, paid when the extension starts: at the end of life (rounded by end_of_life_method). "
            f"The model decides how much of the {cap} still in service at its end of life to extend, at most all of "
            "it; no invest_method is needed, but the period the extension starts in must be an invest_period of the "
            "solve. The extension is priced as a plant of its own: life_extension_cost plus the net_salvage_value "
            "the capacity would have received at its end of life (given up: it stays in the plant), financed at "
            "discount_rate over life_extension_years, less the net_salvage_value received at the extended end of "
            "life, paid as a level yearly cost for each year in service, with the fixed_cost of each year. With "
            "lifetime_method reinvest_choice it competes with new capacity; with no_reinvestment it is the only "
            "way to keep the capacity after its end of life. Not with reinvest_automatic, whose capacity is "
            "replaced at its end of life (an input error). An extension is never extended again. Extended capacity "
            "can be retired when invest_method allows retirement, and counts as capacity, not as investment, in "
            f"the investment limits. {both} Constant or period: the value of the period the extension starts in is "
            "used."),
        (cls, "life_extension_years"): (
            f"[years] How long {cap} can be kept in service beyond the end of its lifetime_technical for "
            "life_extension_cost (see there): the extended end of life is the end of life plus "
            "life_extension_years, rounded by end_of_life_method; the extension is financed over these years at "
            f"discount_rate. {both} Must be above 0. Constant or period: the value of the period the extension "
            "starts in is used."),
    }


def _v82_shipped_payback_time_text(cap: str) -> str:
    """The description of ``payback_time`` a development build of v82 shipped (superseded, :func:`_v82_superseded`)."""
    return _v81_payback_time_text(cap, "changes nothing (a whole number of years)").replace(
        "inflation_rate", "social_discount_rate")


def _v82_replacement_texts(cls: str, cap: str, what: str, per: str) -> dict[tuple[str, str], str]:
    """The descriptions v82 writes for the replacement of reinvest_automatic capacity at its end of life, on the
    class *cls* whose capacity is *cap* (of a *what*, priced per *per*)."""
    v81 = _v81_texts()
    return {
        (cls, "lifetime_method"): (
            "[choice] What happens when capacity reaches the end of its lifetime_technical. reinvest_automatic "
            "(default): the plant is replaced, without a model choice, by a plant at the invest_cost, discount_rate, "
            "payback_time, lifetime_technical and net_salvage_value of the period the replacement comes online, and "
            "so on to the end of the model horizon (existing capacity too, free without an invest_cost); each plant "
            "pays its own level yearly cost for its years in service, and retiring the capacity avoids the "
            "replacements still to come. reinvest_choice: the capacity retires at the end of its lifetime_technical "
            "and the model decides how much new capacity to build. no_reinvestment: the capacity retires at the end "
            "of its lifetime_technical; investment is possible only during the first technical lifetime from the "
            "model start, and the capacity is not rebuilt. The investment cost of a plant is paid as a level yearly "
            "cost of its lifetime_technical (the annuity over payback_time spread over it, its net_salvage_value "
            "netted in) for each year it is in service."),
        (cls, "lifetime_technical"): v81[(cls, "lifetime_technical")].replace(
            f"lifetime_method: reinvest_automatic keeps the {cap} to the end of the model horizon, reinvest_choice",
            f"lifetime_method: reinvest_automatic replaces the {cap} (to the end of the model horizon), "
            "reinvest_choice"),
        (cls, "invest_cost"): v81[(cls, "invest_cost")].replace(
            "(every year to the end of the model horizon with reinvest_automatic)",
            "(with reinvest_automatic: each replacement at the invest_cost of the period it comes online, for its "
            "years in service)"),
        (cls, "age_at_model_start"): v81[(cls, "age_at_model_start")].replace(
            f"with lifetime_method reinvest_automatic the {cap} is kept and the age has no effect.",
            f"with lifetime_method reinvest_automatic the existing {cap} reaches its end of life (its age less the "
            "replacements before the model start) and is then replaced."),
        (cls, "net_salvage_value"): v81[(cls, "net_salvage_value")].replace(
            "With lifetime_method reinvest_choice or no_reinvestment the value at the end of life is a level yearly "
            "credit over the lifetime at discount_rate:",
            "The value at the end of life is a level yearly credit over the lifetime at discount_rate (with "
            "reinvest_automatic for each plant of the chain; a replacement without an invest_cost keeps it):"),
        (cls, "end_of_life_method"): v81[(cls, "end_of_life_method")].replace(
            "Used with lifetime_method reinvest_choice / no_reinvestment.",
            "Used with every lifetime_method (with reinvest_automatic it sets when each plant is replaced)."),
    }


def _v82_shipped_life_extension_cost_text(cap: str, per: str) -> str:
    """The description of ``life_extension_cost`` a development build of v82 shipped, when a life extension was
    also offered under reinvest_automatic (superseded, :func:`_v82_superseded`)."""
    return (f"[CUR/{per}] One-off cost of keeping {cap} in service for life_extension_years beyond the end of its "
            "lifetime_technical, paid when the extension starts: at the end of life (rounded by end_of_life_method). "
            f"The model decides how much of the {cap} still in service at its end of life to extend, at most all of "
            "it; no invest_method is needed, but the period the extension starts in must be an invest_period of the "
            "solve. The extension is priced as a plant of its own: life_extension_cost plus the net_salvage_value "
            "the capacity would have received at its end of life (given up: it stays in the plant), financed at "
            "discount_rate over life_extension_years, less the net_salvage_value received at the extended end of "
            "life, paid as a level yearly cost for each year in service, with the fixed_cost of each year. With "
            "lifetime_method reinvest_choice it competes with new capacity, with no_reinvestment it is the only "
            "way to keep the capacity after its end of life, and with reinvest_automatic it competes with the "
            "replacement: the extended capacity is replaced at the extended end of life instead (each end of life "
            "of the chain can be extended). An extension, and the replacements of an extension, are never extended "
            "again. Extended capacity can be retired when invest_method allows retirement, and counts as capacity, "
            "not as investment, in the investment limits. Give both life_extension_cost and life_extension_years, "
            "or neither. Constant or period: the value of the period the extension starts in is used.")


def _v82_superseded() -> dict[tuple[str, str], tuple[str | None, str]]:
    """``{(class, name): (default, description)}``: the default and description pairs of v82 definitions that a
    development build of v82 shipped and that have since been superseded (the current description is in
    :func:`_v82_texts`; the default stays).  A database at 82 that has exactly such a pair gets the current
    description (:func:`_refresh_head_schema`)."""
    out: dict[tuple[str, str], tuple[str | None, str]] = {("model", "discount_offset_operations"): (
        0.5, "[years] Where in each year operating and fixed costs are weighed by social_discount_rate, counted from "
             "the start of the year: 0.5 (default) the middle of the year, standing for costs spread over the year. "
             "No effect when social_discount_rate is 0. Constant.")}
    # The descriptions of the replacement of reinvest_automatic capacity: a database at 82 has the v81 text (the
    # shipped v82 text of payback_time).
    v81 = _v81_texts()
    defaults = {"lifetime_method": "reinvest_automatic", "end_of_life_method": "closest_seam"}
    for cls, (cap, what, per) in _V81_CAPACITY.items():
        for _cls, name in _v82_replacement_texts(cls, cap, what, per):
            out[(cls, name)] = (defaults.get(name), v81[(cls, name)])
        out[(cls, "payback_time")] = (None, _v82_shipped_payback_time_text(cap))
        out[(cls, "life_extension_cost")] = (None, _v82_shipped_life_extension_cost_text(cap, per))
    return out


def _v82_definitions() -> dict[tuple[str, str], dict]:
    """The definitions v82 adds, with the fields it gives them: the ``model`` definitions of
    :data:`V82_PARAMETER_RENAMES` (new names; a constant, ``float``, group ``model``) and those of
    :data:`V82_LIFE_EXTENSION` on unit, connection and node (``float`` and ``1d_map``, group ``investment``), each
    with its description.  The default is not among them: a renamed definition keeps its own
    (:func:`_v82_definitions_part`)."""
    texts = _v82_texts()
    out = {("model", new): {"parameter_type_list": ("float",), "parameter_group_name": "model",
                            "description": texts[("model", new)]}
           for new in V82_PARAMETER_RENAMES.values()}
    for cls in _V81_CAPACITY:
        for name in V82_LIFE_EXTENSION:
            out[(cls, name)] = {"parameter_type_list": ("float", "1d_map"), "parameter_group_name": "investment",
                                "description": texts[(cls, name)]}
    return out


def _v82_rename_definitions(db) -> None:
    """Part 2 of v82: each old name of :data:`V82_PARAMETER_RENAMES` becomes its new name.

    Without the new definition the old one is renamed in place: it keeps its id, so its values (every alternative,
    their bytes and types), its default, its types and its group; no value row is touched.  When both exist (the
    user added the new name, a template was re-applied, or old-named data was imported into a database at 81), each
    old value moves to the new name where that alternative has no value under it yet, a value already there is kept
    and the old one is printed, and the old definition is removed (:func:`_remove_items_bounded`).  Without the old
    definition the part does nothing, so a rerun changes nothing.
    """
    if not db.get_item("entity_class", name="model"):
        return
    for old, new in V82_PARAMETER_RENAMES.items():
        old_def = _pdef_or_none(db, "model", old)
        if old_def is None:
            continue
        if _pdef_or_none(db, "model", new) is None:
            db.update_parameter_definition(id=old_def["id"], name=new)
            continue
        taken = {(tuple(pv["entity_byname"]), pv["alternative_name"])
                 for pv in db.find_parameter_values(entity_class_name="model", parameter_definition_name=new)}
        kept = []
        for pv in list(db.find_parameter_values(entity_class_name="model", parameter_definition_name=old)):
            if (tuple(pv["entity_byname"]), pv["alternative_name"]) in taken:
                kept.append(pv)
            else:
                db.update_parameter_value(id=pv["id"], parameter_definition_name=new)
        if kept:
            print(f"v82: model {new} already had a value where {old} had one; kept the {new} value, the {old} "
                  f"value was not moved to {new}:")
            for pv in kept:
                print("  " + _describe_pv(pv))
        _remove_items_bounded(db, "parameter_definition", [old_def["id"]], f"v82: removed model.{old}")


def _v82_definitions_part(db, *, update: bool = True) -> None:
    """Part 3 of v82: each definition of :func:`_v82_definitions` on an existing class is added (with the default of
    :data:`V82_DEFAULTS`, and its parameter group), or (``update``) given its types, group and description.  A
    default is never compared or written on an existing definition (a renamed one keeps its own, also a stored empty
    one).  ``update=False`` only adds the missing ones.  Idempotent."""
    for (cls, name), fields in _v82_definitions().items():
        if not db.get_item("entity_class", name=cls):
            continue
        _ensure_parameter_group(db, fields["parameter_group_name"], *_V81_GROUPS[fields["parameter_group_name"]])
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            default = V82_DEFAULTS[name]
            extra = {} if default is None else dict(zip(("default_value", "default_type"), to_database(default)))
            db.add_parameter_definition(entity_class_name=cls, name=name, **fields, **extra)
        elif update and any((tuple(pdef[k] or ()) if k == "parameter_type_list" else (pdef[k] or "")) != v
                            for k, v in fields.items()):
            db.update_parameter_definition(id=pdef["id"], **fields)


def _v82_descriptions(db) -> None:
    """Part 4 of v82: the whole descriptions of :func:`_v82_texts`.  No value is rewritten.  Idempotent."""
    for (cls, name), text in _v82_texts().items():
        if not db.get_item("entity_class", name=cls):
            continue
        pdef = _pdef_or_none(db, cls, name)
        if pdef is not None and (pdef["description"] or "") != text:
            db.update_parameter_definition(id=pdef["id"], description=text)


def _v82_automatic_replacement_marker(db) -> None:
    """Part 5 of v82: the behaviour-change marker ``automatic_replacement_cost`` (``{"since": 82, "entities":
    [...]}``) on a database where a unit, connection or node with ``lifetime_method`` ``reinvest_automatic`` (in some
    alternative, or without one: the default) and a ``lifetime_technical`` either may invest and has a period map
    of ``invest_cost``, ``discount_rate``, ``lifetime_technical`` or ``payback_time``, a ``payback_time`` or a
    non-zero ``net_salvage_value``, or may retire: its capacity is now replaced at the end of its life at the
    values of the period the replacement comes online, and a retirement avoids the replacements still to come.  A
    run names those whose costs change.  No value is rewritten.  Idempotent: a rerun finds the marker written.
    """
    no_invest = {"not_allowed", "retire_period", "retire_total", "retire_no_limit", "retire_period_total"}
    no_retire = {"not_allowed", "invest_period", "invest_total", "invest_no_limit", "invest_period_total"}
    entities: set[str] = set()
    for cls in ("unit", "connection", "node"):
        if not db.get_item("entity_class", name=cls):
            continue

        def values(name: str) -> list[tuple[str, object]]:
            return [(pv["entity_byname"][0], from_database(pv["value"], pv["type"]))
                    for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name=name)]

        methods = values("lifetime_method")
        bounded_only = {e for e, _v in methods} - {e for e, v in methods if v == "reinvest_automatic"}
        with_life = {e for e, _v in values("lifetime_technical")} - bounded_only
        invest: dict[str, set[str]] = {}
        for e, v in values("invest_method"):
            invest.setdefault(e, set()).add(v)
        priced: set[str] = {e for e, _v in values("payback_time")}
        for name in ("invest_cost", "discount_rate", "lifetime_technical"):
            priced |= {e for e, v in values(name) if isinstance(v, Map)}
        priced |= {e for e, v in values("net_salvage_value") if any(x != 0.0 for x in _v79_numbers(v))}
        for e in with_life:
            own = invest.get(e, set())
            if (own - no_invest and e in priced) or own - no_retire:
                entities.add(e)
    if entities:
        _v77_set_marker(db, "automatic_replacement_cost", json.dumps({"since": 82, "entities": sorted(entities)}))


def _v82_rolling_storage_markers(db) -> None:
    """Part 6 of v82: the behaviour-change markers of the storage carry in rolling and nested solves.  No value is
    rewritten.  Idempotent: a rerun finds the markers written.

    * ``rolling_cyclic_storage_anchored`` (``{"since": 82, "nodes": [...]}``) when a solve has ``solve_mode``
      ``rolling_window`` and a storage node has a cyclic ``storage_binding_method`` (``bind_within_solve``,
      ``bind_within_period``, their ``_blended_weights`` variants, ``bind_within_timeblock`` or
      ``bind_intraperiod_blocks``, in some alternative): such a node now continues each roll from the previous roll
      (per period or timeblock where a roll starts inside one), ends a roll's part at least at its start level and
      closes the solve, period or timeblock at the level it started from.
    * ``nested_storage_start_borrowed`` (``{"since": 82, "nodes": [...]}``) when a solve has ``contains_solves``
      and a node has a ``storage_nested_fix_method``: its first lower-level solve can now start from the upper
      solve's planned level.
    * ``nested_storage_fix_cumulative`` (``{"since": 82, "nodes": [...]}``) when a solve has ``contains_solves`` and
      a node has ``storage_nested_fix_method`` ``fix_usage`` or ``fix_quantity``: the usage cap is cumulative and pro
      rata, and the quantity target is interpolated inside an upper step.

    The markers are broader than the changes (the migration does not read which alternatives a scenario
    combines); a run names the nodes whose rows change.
    """
    if not (db.get_item("entity_class", name="solve") and db.get_item("entity_class", name="node")):
        return

    def values(cls: str, name: str) -> list[tuple[str, object]]:
        return [(pv["entity_byname"][0], from_database(pv["value"], pv["type"]))
                for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name=name)]

    storage = {n for n, v in values("node", "node_type") if v == "storage"}
    if any(v == "rolling_window" for _s, v in values("solve", "solve_mode")):
        cyclic = ("bind_within_solve", "bind_within_solve_blended_weights", "bind_within_period",
                  "bind_within_period_blended_weights", "bind_within_timeblock", "bind_intraperiod_blocks")
        nodes = sorted({n for n, v in values("node", "storage_binding_method") if v in cyclic} & storage)
        if nodes:
            _v77_set_marker(db, "rolling_cyclic_storage_anchored", json.dumps({"since": 82, "nodes": nodes}))
    if any(v is not None for _s, v in values("solve", "contains_solves")):
        fix = values("node", "storage_nested_fix_method")
        nodes = sorted({n for n, v in fix if v} & storage)
        if nodes:
            _v77_set_marker(db, "nested_storage_start_borrowed", json.dumps({"since": 82, "nodes": nodes}))
        nodes = sorted({n for n, v in fix if v in ("fix_usage", "fix_quantity")} & storage)
        if nodes:
            _v77_set_marker(db, "nested_storage_fix_cumulative", json.dumps({"since": 82, "nodes": nodes}))


def _v82_storage_accounting_markers(db) -> None:
    """Part 7 of v82: the behaviour-change markers of the storage start and end accounting.  No value is rewritten.
    Idempotent: a rerun finds the markers written.

    * ``rp_storage_chain_pinned`` (``{"since": 82, "nodes": [...]}``) when a timeset has
      ``representative_period_weights`` and a storage node has a ``*_blended_weights`` ``storage_binding_method``
      (in some alternative): storage_state_start / storage_state_end / use_reference_value now set the level at the
      ends of its base-period chain, and a forward chain's start is within the capacity.
    * ``storage_end_matches_start`` (``{"since": 82, "nodes": [...]}``) when a storage node chains forward
      (``storage_binding_method`` bind_forward_only or bind_forward_only_blended_weights in some alternative, or no
      value: the default bind_forward_only): where the solve decides its start, the model now ends at least at it.
    * ``storage_reference_start`` (``{"since": 82, "nodes": [...]}``) when a storage node has
      ``storage_solve_horizon_method`` use_reference_value and storage_start_end_method fix_nothing (in some
      alternative; fix_end turns the reference off): it now starts from the reference value instead of a free
      level.

    The markers are broader than the changes (the migration does not read which alternatives a scenario combines,
    nor which solves invest); a run names the nodes whose rows change.
    """
    if not db.get_item("entity_class", name="node"):
        return

    def values(cls: str, name: str) -> list[tuple[str, object]]:
        return [(pv["entity_byname"][0], from_database(pv["value"], pv["type"]))
                for pv in db.find_parameter_values(entity_class_name=cls, parameter_definition_name=name)]

    storage = {n for n, v in values("node", "node_type") if v == "storage"}
    if not storage:
        return
    methods = values("node", "storage_binding_method")
    if db.get_item("entity_class", name="timeset") and any(
            v is not None for _t, v in values("timeset", "representative_period_weights")):
        nodes = sorted({n for n, v in methods if isinstance(v, str) and v.endswith("_blended_weights")} & storage)
        if nodes:
            _v77_set_marker(db, "rp_storage_chain_pinned", json.dumps({"since": 82, "nodes": nodes}))
    with_method = {n for n, _v in methods}
    forward = {n for n, v in methods if v in ("bind_forward_only", "bind_forward_only_blended_weights")}
    nodes = sorted((forward | (storage - with_method)) & storage)
    if nodes:
        _v77_set_marker(db, "storage_end_matches_start", json.dumps({"since": 82, "nodes": nodes}))
    # Every value of each node (one per alternative): a node that is fix_nothing in some alternative, or has no
    # value in some (the default fix_nothing), qualifies whichever alternative comes last.
    start_end: dict[str, set] = {}
    for n, v in values("node", "storage_start_end_method"):
        start_end.setdefault(n, set()).add(v)
    reference = {n for n, v in values("node", "storage_solve_horizon_method") if v == "use_reference_value"}
    nodes = sorted(n for n in reference & storage
                   if n not in start_end or "fix_nothing" in start_end[n] or None in start_end[n])
    if nodes:
        _v77_set_marker(db, "storage_reference_start", json.dumps({"since": 82, "nodes": nodes}))


def _migrate_v82_schema_batch(db) -> None:
    """All schema changes of database version 82 (v81 -> v82), in one step.

    1. The v81 refresh (:func:`_v81_refresh`): a database migrated to 81 by a development build of v81 gets the v81
       definitions and list members it lacks and the current pair in place of a superseded shipped one (with the
       v82 descriptions); after a full v81 it changes nothing.
    2. ``model.inflation_rate`` is ``social_discount_rate``, ``inflation_offset_investment`` is
       ``discount_offset_investment`` and ``inflation_offset_operations`` is ``discount_offset_operations``, with
       every value and the default (:func:`_v82_rename_definitions`).
    3. The three definitions: a constant, group ``model``, the description; a missing one is added with its shipped
       default (:func:`_v82_definitions_part`).  With them, ``life_extension_cost`` and ``life_extension_years`` on
       unit, connection and node (a constant or a period map, group ``investment``, no default): capacity at its end
       of life can be kept in service longer for a one-off cost.
    4. The descriptions of the three and of ``discount_rate`` and ``payback_time`` (unit, connection, node), which
       name them, and of ``lifetime_method``, ``lifetime_technical``, ``invest_cost``, ``age_at_model_start``,
       ``net_salvage_value`` and ``end_of_life_method``, which say what reinvest_automatic does: the capacity is
       replaced at the end of its life, and of ``storage_binding_method`` and ``storage_nested_fix_method``, which
       say how storage closes over rolls and nested solves (:func:`_v82_descriptions`).
    5. The behaviour-change marker ``automatic_replacement_cost`` on the reinvest_automatic entities with a
       lifetime_technical whose investment cost or retirement now follows the replacements of their capacity
       (:func:`_v82_automatic_replacement_marker`).
    6. The behaviour-change markers ``rolling_cyclic_storage_anchored``, ``nested_storage_start_borrowed`` and
       ``nested_storage_fix_cumulative`` of the storage carry in rolling and nested solves
       (:func:`_v82_rolling_storage_markers`).
    7. The behaviour-change markers ``rp_storage_chain_pinned``, ``storage_end_matches_start`` and
       ``storage_reference_start`` of the storage start and end accounting (:func:`_v82_storage_accounting_markers`).

    Values change only by the renames of part 2 (they keep their rows; a collision moves them).  Each part is
    idempotent, so a rerun of the step changes nothing.
    """
    _v81_refresh(db, _v82_texts())
    _v82_rename_definitions(db)
    _v82_definitions_part(db)
    _v82_descriptions(db)
    _v82_automatic_replacement_marker(db)
    _v82_rolling_storage_markers(db)
    _v82_storage_accounting_markers(db)
    _commit_step(db, "v82: inflation_rate is social_discount_rate, inflation_offset_investment is "
                     "discount_offset_investment, inflation_offset_operations is discount_offset_operations")


# --- Version 83 part 1: minimum_time_method (C rev 3 §6) --------------------------------------------------------

#: ``startup_method`` values that give a unit online variables, in both vocabularies (part 2 renames ``binary``; a
#: rerun of part 1 after a kill sees the new names).
_V83_ONLINE = frozenset(("linear", "binary", "linear_fix_in_uc", "integer", "integer_fix_in_uc"))
#: The methods that enable each time.
_V83_ENABLES = {"min_uptime": frozenset(("min_uptime", "both")), "min_downtime": frozenset(("min_downtime", "both"))}
#: The members of ``minimum_time_methods``, in the order part 1 appends the missing ones.
_V83_MIN_TIME_METHODS = ("min_uptime", "min_downtime", "both", "none")
#: The unit parameters whose parsed values decide a unit's online variables and min up/down rows.
_V83_READ = ("startup_method", "conversion_method", "piecewise_linear_efficiency", "minimum_time_method",
             "min_uptime", "min_downtime")
#: The behaviour-change marker of part 1 (W-0634).
V83_MIN_TIME_MARKER = "minimum_time_method_written"
_V83_MEMBER_MESSAGE = ("A version 83 upgrade step changed an existing member of the value list {list} ({members}); "
                       "the upgrade stops so that no default or value changes with it. The file was restored from "
                       "the backup. Please report this.")


@dataclass(frozen=True)
class MinTimePlan:
    """What part 1 of v83 writes.  ``writes``: ``(unit, alternative, method)``; ``undecided``: ``(unit, reason,
    candidate scenarios, ((candidate scenario, the kinds it needs), ...))`` (nothing is written for such a unit)."""
    writes: tuple = ()
    undecided: tuple = ()


def _v83_write_values(db, entity_class: str, name: str, writes) -> None:
    """Write each ``(entity, alternative, value)`` of *writes* into *name* of *entity_class*, by value: a row
    already there (a null one: :func:`_v83_read` reads it as unset) is updated, else one is added."""
    rows = {(pv["entity_byname"][0], pv["alternative_name"]): pv["id"]
            for pv in db.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=name)}
    for entity, alternative, value in writes:
        val, typ = to_database(value)
        if (entity, alternative) in rows:
            db.update_parameter_value(id=rows[(entity, alternative)], value=val, type=typ)
        else:
            db.add_parameter_value(entity_class_name=entity_class, entity_byname=(entity,),
                                   parameter_definition_name=name, alternative_name=alternative, value=val, type=typ)


def _v83_scenarios_text(scenarios) -> str:
    """``scenario S``, ``scenarios S1, S2`` or, without scenarios (:data:`_V77_NO_SCENARIO`), ``the database``
    (the v83 prints)."""
    if tuple(scenarios) == (_V77_NO_SCENARIO,):
        return "the database"
    return f"scenario{'s' if len(scenarios) > 1 else ''} {', '.join(scenarios)}"


def _v83_check_members(db, name: str, before: dict, removed=frozenset()) -> None:
    """The v83 member guard: every member of the value list *name* in *before* (:func:`_list_members`, taken before
    the part) that the part does not remove on purpose (*removed*: database ids) is still there with the same
    ``(list, value, type)``, and every id of *removed* is gone.  Raises :class:`FlexToolMigrationError` (before the
    step's commit) otherwise."""
    after = _list_members(db, [name])
    changed = sorted(f"{from_database(v, t)!r}{' (its planned removal did not happen)' if i in removed else ''}"
                     for i, (_lst, v, t) in before.items()
                     if (i in after if i in removed else after.get(i) != (_lst, v, t)))
    if changed:
        raise FlexToolMigrationError(_V83_MEMBER_MESSAGE.format(list=name, members=", ".join(changed)))


def _v83_is_time(value) -> bool:
    """The engine's test for a min up/down time: ``float(v) > 0``, of any leaf of a non-scalar value."""
    leaves = getattr(value, "values", None)
    if leaves is not None:
        return any(_v83_is_time(v) for v in leaves)
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _v83_read(alternatives, table, key, default=None) -> tuple:
    """``(value, alternative)`` of *key* in a scenario of *alternatives* (lowest rank first), as the engine reads
    it: the value of the winning row of *table* ``{(key, alternative): parsed value}`` (the highest-ranked
    alternative holding a row, a null one included), else *default*; ``alternative`` is where that row is (None
    without one).  A winning null reads as unset, so *default* applies, not a lower alternative's value: the engine's
    scenario filter keeps the top row, ``SpineDbReader._unroll_rows`` drops a whole-null value and
    ``_parameter_frame`` fills in the definition default (``engine/input/_spinedb_reader.py``); the solve reader
    ``params_to_dict`` (``engine/input/_solve_config.py``) skips it, and
    ``SolveConfig.dispatch_realization_method_for`` reads the absent value as within_solve."""
    where = _v77_winner(alternatives, table, key)
    value = table[(key, where)] if where is not None else None
    return (default if value is None else value), where


def _v83_method(unit: str, alternatives, table, default) -> tuple:
    """``(resolves, method, alternative)`` of *unit*'s ``minimum_time_method`` in a scenario of *alternatives*
    (:func:`_v83_read`: a winning null reads the definition *default*).  It resolves when a non-null value wins, or
    when the default that reads is a member other than ``none``; ``alternative``: where the winning row is (a null
    row there still hides a write below it)."""
    method, where = _v83_read(alternatives, table, unit, default)
    explicit = where is not None and table[(unit, where)] is not None
    return explicit or method not in (None, "none"), method, where


class _V83State(NamedTuple):
    """A unit in one scenario as the engine reads it before data version 83 (C rev 3 §6.2)."""
    online: bool  #: it has online variables
    need: frozenset  #: the kinds with a time when online (their rows are built)
    resolves: bool  #: a minimum_time_method value wins, or the default is a member other than none
    method: object  #: the method that reads
    where: str | None  #: where the winning method row is (None for none)
    off_by: tuple  #: when not online: the alternatives whose winning values keep it off (startup_method, then
    #: conversion_method, then the curve)


def _v83_unit_state(unit: str, alternatives, values, defaults) -> _V83State:
    """*unit* in a scenario of *alternatives* (lowest rank first) as the engine reads it before data version 83.
    ``online``: ``startup_method`` in :data:`_V83_ONLINE`, else ``no_startup`` with ``minimum_time_method``
    promoting it, else with a ``piecewise_linear`` curve with ``auto_startup``.  Every value through
    :func:`_v83_read` (a winning null reads the default)."""
    def read(name):
        return _v83_read(alternatives, values.get(name, {}), unit, defaults.get(name))

    startup, st_alt = read("startup_method")
    startup = startup or "no_startup"
    resolves, method, where = _v83_method(unit, alternatives, values.get("minimum_time_method", {}),
                                          defaults.get("minimum_time_method"))
    conversion, ct_alt = read("conversion_method")
    curve, curve_alt = read("piecewise_linear_efficiency")
    piecewise = (conversion or "constant_efficiency") == "piecewise_linear"
    online = startup in _V83_ONLINE or startup == "no_startup" and (
        method in ("min_uptime", "min_downtime", "both") or piecewise and bool(curve))
    need = frozenset(k for k in _V83_ENABLES if online and _v83_is_time(read(k)[0]))
    off_by = () if online else tuple(a for a, keeps_off in (
        (st_alt, True), (ct_alt, startup == "no_startup" and not piecewise),
        (curve_alt, startup == "no_startup" and piecewise and not curve)) if a is not None and keeps_off)
    return _V83State(online, need, resolves, method, where, off_by)


def _v83_unit_writes(unit, method, candidates, before, values, held, scenarios, defaults) -> tuple[dict, str | None]:
    """C rev 3 §6.3 steps 3-5 for one unit: ``({alternative: method}, None)``, or ``({}, reason)`` when it stays
    undecided.  *held*: the ``(unit, alternative)`` holding any value of the unit."""
    planned: dict[str, str] = {}
    for s in candidates:
        target = _v77_winner(scenarios[s], values.get("startup_method", {}), unit)
        if target is None:
            target = _v77_winner(scenarios[s], held, unit)
        null_row = before[s].where  # a winning null method row (a candidate resolves no method), None without one
        if null_row is not None and (target is None or scenarios[s].index(null_row) > scenarios[s].index(target)):
            target = null_row  # it reads as not set and would hide a write below it: the write goes into it
        if target is None:
            return {}, f"no alternative of scenario {s} holds a value of the unit"
        planned[target] = method
    own = values.get("minimum_time_method", {})
    for s, state in before.items():
        alts = scenarios[s]
        winner = _v77_winner(alts, {**own, **{(unit, a): m for a, m in planned.items()}}, unit)
        if not state.online and winner in planned and planned[winner] != "none":
            # The switch-off: none into the alternative that keeps u off in s, else (it ranks too low to win, or
            # there is none) into the highest alternative of s.
            off = next((a for a in state.off_by if alts.index(a) > alts.index(winner)), alts[-1])
            if off in planned:
                return {}, f"a value in alternative {off} would give the unit online variables in scenario {s}"
            planned[off] = "none"
    table = {**own, **{(unit, a): m for a, m in planned.items()}}
    for s, state in before.items():
        new_resolves, new, winner = _v83_method(unit, scenarios[s], table, defaults.get("minimum_time_method"))
        new = new if new_resolves else None
        if state.resolves and new != state.method:
            return {}, (f"a value in alternative {winner} would replace your minimum_time_method in alternative "
                        f"{state.where} in scenario {s}")
        if s in candidates and not all(new in _V83_ENABLES[k] for k in state.need):
            return {}, f"the values written would not enable its {' and '.join(sorted(state.need))} in scenario {s}"
        if not state.online and new in ("min_uptime", "min_downtime", "both"):
            return {}, f"a value in alternative {winner} would give the unit online variables in scenario {s}"
    return planned, None


def plan_minimum_time_method(units, values, scenarios, defaults, inactive=frozenset()) -> MinTimePlan:
    """Where part 1 of v83 writes ``minimum_time_method`` so that every unit keeps its min up/down rows once the
    engine reads the method (C rev 3 §6.3).  Pure: no database.

    *units*: the unit names; *values* ``{parameter: {(unit, alternative): parsed value}}`` with every stored unit
    parameter (those outside :data:`_V83_READ` only mark the alternatives that hold a value of the unit; a null
    value is kept: it reads the default), the value of ``piecewise_linear_efficiency`` being ``auto_startup`` of its
    curves; *scenarios* ``{scenario: [alternative, ...]}`` lowest rank first; *defaults* ``{parameter: parsed
    definition default}``; *inactive* ``{(scenario, unit)}`` (left out of that scenario).

    A unit's candidate scenarios are those where it has rows today and no method resolves.  It gets ``both`` when
    they need both times, else the one time, written into each candidate scenario's ``startup_method`` winner (else
    its highest alternative holding any value of the unit), or into the winning null ``minimum_time_method`` row
    when that one ranks higher (it reads as not set; its row is updated), plus ``none`` where the written method would win in a
    scenario where the unit has no online variables today: into the alternative whose value keeps it off there
    (``startup_method``, else ``conversion_method``, else the curve), else, when that one ranks too low to win, into
    the scenario's highest alternative.  Then every scenario is checked: a method that resolved is unchanged, every
    needed time is enabled, a unit without online variables gets no enabling method.  A unit that fails any of these
    gets no write (undecided, the reason of the first failing scenario by name)."""
    writes, undecided = [], []
    held: dict[str, set] = {}
    for table in values.values():
        for key in table:
            held.setdefault(key[0], set()).add(key)
    for unit in sorted(units):
        before = {s: _v83_unit_state(unit, scenarios[s], values, defaults)
                  for s in sorted(scenarios) if (s, unit) not in inactive}
        candidates = [s for s, state in before.items() if state.need and not state.resolves]
        if not candidates:
            continue
        kinds = frozenset().union(*(before[s].need for s in candidates))
        method = "both" if len(kinds) == 2 else next(iter(kinds))
        planned, reason = _v83_unit_writes(unit, method, candidates, before, values, held.get(unit, set()),
                                           scenarios, defaults)
        if reason is None:
            writes.extend((unit, alternative, m) for alternative, m in sorted(planned.items()))
        else:
            undecided.append((unit, reason, tuple(candidates),
                              tuple((s, tuple(k for k in _V83_ENABLES if k in before[s].need)) for s in candidates)))
    return MinTimePlan(writes=tuple(writes), undecided=tuple(undecided))


def _v83_unit_inputs(db) -> tuple:
    """:func:`plan_minimum_time_method` arguments read from *db* (parsed values only, C rev 3 §6.1)."""
    from flextool.engine.input._piecewise_curve import piecewise_unit_flags
    from flextool.input_derivation._piecewise import curves_of_value

    units = sorted(e["name"] for e in db.get_items("entity", entity_class_name="unit"))
    values: dict[str, dict] = {}
    for pv in db.find_parameter_values(entity_class_name="unit"):
        name = pv["parameter_definition_name"]
        value = pv["parsed_value"] if name in _V83_READ else True
        if name == "piecewise_linear_efficiency" and value is not None:
            curves, problems = curves_of_value(value)
            value = not problems and piecewise_unit_flags(points for _p, points in curves).auto_startup
        values.setdefault(name, {})[(pv["entity_byname"][0], pv["alternative_name"])] = value
    defaults = {}
    for name in _V83_READ:
        pdef = _pdef_or_none(db, "unit", name)
        if name != "piecewise_linear_efficiency" and pdef is not None and pdef["default_value"] is not None:
            defaults[name] = from_database(pdef["default_value"], pdef["default_type"])
    scenarios = _scenario_alternatives(db)
    return units, values, scenarios, defaults, _inactive_in_scenarios(db, "unit", units, scenarios)


def _v83_minimum_time_method(db) -> None:
    """Part 1 of v83: the engine now builds a unit's min up/down rows only when ``minimum_time_method`` enables the
    time.  Appends the missing members of ``minimum_time_methods``, then writes the method where a unit relied on a
    time alone (:func:`plan_minimum_time_method`, results unchanged), and records the writes and the undecided units
    in the marker ``minimum_time_method_written`` (``{"since": 83, "writes": [[unit, alternative, method], ...],
    "undecided": [[unit, reason, [scenario, ...]], ...]}``; writes merged, undecided replaced).  Skips a database
    without the class, the definition or the list.  Idempotent; no commit."""
    if (not db.get_item("entity_class", name="unit") or _pdef_or_none(db, "unit", "minimum_time_method") is None
            or not db.get_item("parameter_value_list", name="minimum_time_methods")):
        return
    before = _list_members(db, ["minimum_time_methods"])
    _append_list_members(db, "minimum_time_methods", _V83_MIN_TIME_METHODS)
    _v83_check_members(db, "minimum_time_methods", before)  # the writes below touch no member
    marked = any(md["name"] == BEHAVIOUR_CHANGE_MARKER_PREFIX + V83_MIN_TIME_MARKER for md in db.get_items("metadata"))
    plan = plan_minimum_time_method(*_v83_unit_inputs(db))
    _v83_write_values(db, "unit", "minimum_time_method", plan.writes)
    if marked and not plan.writes:
        _already_applied(83, "minimum_time_method: already written")
    if plan.writes:
        print("v83: minimum_time_method written so that min_uptime / min_downtime keep applying (results unchanged):")
        for unit, alternative, method in plan.writes:
            print(f"  unit {unit} [{alternative}]: {method}")
    for unit, reason, _candidates, needs in plan.undecided:
        groups: dict[tuple, list] = {}
        for s, kinds in needs:
            groups.setdefault(kinds, []).append(s)
        lost = "; ".join(f"in {_v83_scenarios_text(ss)} its {' and '.join(kinds)} no longer "
                         f"{'applies' if len(kinds) == 1 else 'apply'}" for kinds, ss in groups.items())
        print(f"v83: minimum_time_method not written for unit {unit}: {reason}. {lost[0].upper()}{lost[1:]} (results "
              "change). Set minimum_time_method yourself (both, min_uptime or min_downtime) where it should apply.")
    _merge_marker_lists(db, V83_MIN_TIME_MARKER, 83, "writes", plan.writes,
                        [[unit, reason, list(candidates)] for unit, reason, candidates, _n in plan.undecided])


#: The members part 2 of v83 appends to ``startup_methods`` (when missing), in this order, and the one it replaces.
_V83_STARTUP_METHODS = ("linear_fix_in_uc", "integer", "integer_fix_in_uc")
_V83_STARTUP_RENAMES = {"binary": "integer_fix_in_uc"}
_V83_ONLINE_NEED = ("Needs online variables (startup_method linear or binary",
                    "Needs online variables (a startup_method other than no_startup")
_V83_CONTINUOUS = ("With linear startup the number is continuous.",
                   "With startup_method linear or linear_fix_in_uc the number is continuous.")


def _v83_startup_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions that name a ``startup_method`` member, as every database at 82
    has them unless the user edited them, and their v83 texts (part 2 of v83, :func:`_replace_superseded_texts`)."""
    online_coeff = ("[unit of the constraint per unit online] A map of coefficients (Index: constraint name, value: "
                    "coefficient). Puts the number of units {what}online in each time step on the left side of the "
                    "user constraint. One unit is virtual_unitsize; without it, the largest existing capacity, else "
                    "1000 MW (set virtual_unitsize when the capacity is invested). Needs online variables "
                    "(startup_method linear or binary{mt}); a coefficient without them is an error. With linear "
                    "startup the number is continuous. For MW online, use the unit size (MW per unit) as the "
                    "coefficient.")
    startup_input = (
        "[choice] What the input arc supplies. just_input (default): the normal input flow. startup_and_input: the "
        "normal input flow, and each start-up of the unit also takes startup_energy x MW started from this node. "
        "startup_only: only the start-up energy; the arc has no flow and takes no part in the conversion, the "
        "capacity, input shares or ramps. The start-up input can be fuel (oil to start a coal unit), heat (from the "
        "tank of a CSP plant) or electricity. Needs online variables (startup_method linear or binary, or a "
        "minimum_time_method). The start-up input is priced as the node's commodity, counts in CO2 prices and caps "
        "and is dropped with the start-ups by reduce_unit_detail_method online. It is not part of "
        "constraint_flow_coeff, flow groups or the capacity margin.")
    reduce_detail = (
        "[choice] Detail of units and connections that this solve leaves out, to make it faster (typically an "
        "investment solve). none (default): all detail. integers: binary startup_method is solved as continuous and "
        "exact connections behave as regular. integers_ramps: integers, and no ramp limits or ramp costs. "
        "integers_min_limits: integers, and no min_load, capacity_min_coeff floors or minimum up / down times; a "
        "min_load_efficiency unit that is more efficient at min load than at full load (efficiency_at_min_load "
        "above efficiency) then runs at its full-load efficiency. integers_ramps_min_limits: integers, no ramps and "
        "no min limits. Under the integers levels, a unit or connection whose online variables nothing in the solve "
        "still uses (no start-up cost, min_load, capacity_min_coeff floor, minimum up / down time, "
        "min_load_efficiency curve, piecewise_linear curve whose efficiency rises from its first point, "
        "constraint_online_coeff in a user constraint, reserve, inertia or ramp limit left) has none, as with "
        "startup_method no_startup; the run log names them. online: everything of integers_min_limits, and no "
        "online variables: no start-up costs, user constraints that count units online are left out whole, "
        "min_load_efficiency and piecewise_linear units run at their full-load efficiency (the same fuel at full "
        "output; at part load less or more fuel than with their curve), and ramp limits apply to the capacity. "
        "Reserves are then limited by built capacity, not online capacity, and inertia is counted from flows. "
        "online_ramps: online, and no ramp limits or ramp costs. Never adds detail.")
    out = {
        ("unit", "startup_method"): (
            "[choice] Whether the unit has online variables. linear: any share of the capacity can be online. "
            "binary: whole units only (slower to solve). Online status links to the previous time step; that link "
            "stops at a gap in the timeline.",
            "[choice] Whether the unit has online variables. linear: any share of the capacity can be online. "
            "integer: whole units only (a MIP, slower to solve); several units of one entity are counted as whole "
            "units. linear_fix_in_uc and integer_fix_in_uc: the same, and in a separate realization solve (a "
            "rolling separate_dispatch solve, or a single solve with a realized: yes branch) the units online over "
            "the jump (the realized periods) are kept at what the stochastic (unit commitment) solve chose; without "
            "_fix_in_uc the realization decides them again with the realized data. Elsewhere, including a rolling "
            "within_solve solve, a member and its _fix_in_uc variant are the same. Online status links to the "
            "previous time step; that link stops at a gap in the timeline."),
        ("connection", "startup_method"): (
            "[choice] Gives the connection online capacity in both directions. Flows and reserves are limited by the "
            "online capacity, and startup_cost is paid per MW brought online. No minimum load or minimum up/down "
            "time. Not allowed for dc_power_flow_with_angles lines.",
            "[choice] Gives the connection online capacity in both directions (linear: any share; integer: whole "
            "units, a MIP; linear_fix_in_uc and integer_fix_in_uc: the same, and in a separate realization solve "
            "the online capacity over the jump (the realized periods) is kept at what the stochastic solve chose). "
            "Flows and reserves are limited by the online capacity, and startup_cost is paid per MW brought online. "
            "No minimum load or minimum up/down time. Not allowed for dc_power_flow_with_angles lines."),
        ("unit__inputNode", "startup_input_method"): (startup_input, startup_input.replace(*_V83_ONLINE_NEED)),
        ("solve", "reduce_unit_detail_method"): (reduce_detail, reduce_detail.replace(
            "integers: binary startup_method is solved as continuous",
            "integers: startup_method integer and integer_fix_in_uc are solved as continuous")),
    }
    for cls, what, mt in (("unit", "", ", or a minimum_time_method"), ("connection", "of the connection ", "")):
        old = online_coeff.format(what=what, mt=mt)
        out[(cls, "constraint_online_coeff")] = (old, old.replace(*_V83_ONLINE_NEED).replace(*_V83_CONTINUOUS))
    return out


def _v83_check_no_references(db, list_name: str, removing: dict) -> None:
    """Before members of the value list *list_name* are removed (*removing*: ``{mapped id: parsed value}``), check that
    nothing still refers to them: no default of any definition and no value of a definition on the list holds a
    reference to one, or (a plain string) its value.  At the commit a removed member's id is reused by a member added
    in the same session, so a reference left to it would silently read as that member.  Raises
    :class:`FlexToolMigrationError` (before the step's commit) otherwise."""
    names = set(removing.values())
    left = []
    for pdef in db.get_items("parameter_definition"):
        on_list = pdef["parameter_value_list_name"] == list_name
        if pdef["list_value_id"] in removing or (on_list and pdef["parsed_value"] in names):
            left.append(f"{pdef['entity_class_name']}.{pdef['name']} default")
        if not on_list:
            continue
        for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                           parameter_definition_name=pdef["name"]):
            if pv["list_value_id"] in removing or (isinstance(pv["parsed_value"], str) and pv["parsed_value"] in names):
                left.append(f"{_describe_pv(pv)} ({pdef['name']})")
    if left:
        raise FlexToolMigrationError(
            f"A version 83 upgrade step would remove the member(s) {', '.join(sorted(map(repr, names)))} of the value "
            f"list {list_name} while still in use ({'; '.join(left)}); the upgrade stops so that no default or value "
            "changes with it. The file was restored from the backup. Please report this.")


def _v83_startup_method(db) -> None:
    """Part 2 of v83: ``startup_methods`` becomes no_startup, linear, linear_fix_in_uc, integer, integer_fix_in_uc.

    Appends the missing new members, rewrites every value ``binary`` of every definition on the list (every class
    and alternative; a list reference or a plain string) and a ``binary`` default to ``integer_fix_in_uc`` (the
    same model), then removes the ``binary`` member (after the rewrites, checked by
    :func:`_v83_check_no_references`: at the commit a removed member's id is reused by a member added in the same
    session, so a reference left to it would read as that member), and replaces the six descriptions that name the
    members (:func:`_v83_startup_texts`, by text alone).  ``linear`` is untouched.  Without the list, only the
    descriptions are replaced.  Idempotent; no commit."""
    if db.get_item("parameter_value_list", name="startup_methods"):
        before = _list_members(db, ["startup_methods"])
        planned = {i for i, (_lst, v, t) in before.items() if from_database(v, t) in _V83_STARTUP_RENAMES}
        _append_list_members(db, "startup_methods", _V83_STARTUP_METHODS)
        _v83_check_members(db, "startup_methods", before)  # a changed member would fail the by-value writes first
        rewritten = []
        for pdef in db.get_items("parameter_definition"):
            if pdef["parameter_value_list_name"] != "startup_methods":
                continue
            for pv in db.find_parameter_values(entity_class_name=pdef["entity_class_name"],
                                               parameter_definition_name=pdef["name"]):
                new = _V83_STARTUP_RENAMES.get(pv["parsed_value"]) if isinstance(pv["parsed_value"], str) else None
                if new is not None:
                    rewritten.append(f"{_describe_pv(pv)} ({pdef['name']})")
                    val, typ = to_database(new)
                    db.update_parameter_value(id=pv["id"], value=val, type=typ)
        if rewritten:
            print("v83: the startup_methods member binary is now integer_fix_in_uc (the same model) for:")
            for line in rewritten:
                print("  " + line)
        pending = _detach_list_defaults(db, "startup_methods", _V83_STARTUP_RENAMES)
        removing = {lv["id"]: lv["parsed_value"]
                    for lv in db.get_items("list_value", parameter_value_list_name="startup_methods")
                    if getattr(lv["id"], "db_id", lv["id"]) in planned}
        _v83_check_no_references(db, "startup_methods", removing)
        for lv_id in removing:
            db.remove_item("list_value", lv_id)
        _attach_list_defaults(db, pending)
        _v83_check_members(db, "startup_methods", before, removed=planned)
    _replace_text_pairs(db, _v83_startup_texts())


#: Part 5 of v83: the member appended to ``dispatch_realization_methods`` (the new shipped default) and the method
#: written on the stochastic solves that relied on the default.
_V83_DETERMINISTIC = "deterministic"
_V83_WITHIN_SOLVE = "within_solve"


def _v83_dispatch_realization_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the ``dispatch_realization_method`` description as every database at 82 has
    it unless the user edited it, and its v83 text (part 5 of v83, :func:`_replace_superseded_texts`)."""
    return {("solve", "dispatch_realization_method"): (
        "[choice] How the committed part of a stochastic solve is realized. within_solve (default): a realized: yes "
        "branch gets a realization solve automatically. separate_dispatch: always runs a separate realized dispatch "
        "solve; needs a realized: yes branch (error otherwise).",
        "[choice] How a solve with stochastic_branches is solved. deterministic (default): the forecasts are ignored "
        "and the solve uses the data of the realized: yes branch, with a warning (an error without that branch). "
        "within_solve: the forecasts are solved together; a rolling solve realizes each jump with the realized: yes "
        "branch and looks ahead with the forecasts made at the end of the jump; a single solve with a realized: yes "
        "branch gets a separate realization solve. separate_dispatch: a first, stochastic solve decides the units "
        "online of _fix_in_uc units; a realization solve keeps them and dispatches the realized: yes branch over "
        "the jump and, in a rolling solve, looks ahead with the forecasts made at the end of the jump (a single "
        "solve realizes its realized periods only); needs a realized: yes branch. Without stochastic_branches, deterministic and within_solve give the same model.")}


def plan_dispatch_realization_method(branches, methods, scenarios) -> tuple[tuple, tuple]:
    """Where part 5 of v83 writes ``dispatch_realization_method`` ``within_solve`` so that every stochastic solve
    keeps running within_solve once an absent method means ``deterministic`` (plan §2a step 2).  Pure: no database.

    *branches* ``{(solve, alternative): the stochastic_branches value is non-empty}``; *methods* ``{(solve,
    alternative): method, None for a null row}``; *scenarios* ``{scenario: [alternative, ...]}`` lowest rank first.
    A scenario is a candidate for a solve when its winning ``stochastic_branches`` is non-empty and no method
    resolves (a winning null row reads as no method, :func:`_v83_read`); the target is the alternative of that
    winning value, or the winning null method row when that one ranks higher (it is overwritten).  With the writes
    in place every scenario where a method resolved must read the same method and every candidate must read
    within_solve, else the solve gets no write at all (the reason names every scenario that fails).  Returns
    ``(writes, undecided)``: ``((solve, alternative), ...)`` and ``((solve, reason, candidate scenarios), ...)``."""
    writes, undecided = [], []
    for solve in sorted({key[0] for key in branches}):
        before = {s: _v83_read(alts, methods, solve) for s, alts in sorted(scenarios.items())}
        candidates, targets = [], set()
        for s, alts in sorted(scenarios.items()):
            where = _v77_winner(alts, branches, solve)
            if where is not None and branches[(solve, where)] and before[s][0] is None:
                candidates.append(s)
                null_row = before[s][1]  # a winning null method row (None without one)
                targets.add(where if null_row is None or alts.index(null_row) < alts.index(where) else null_row)
        if not candidates:
            continue
        table = {**methods, **{(solve, a): _V83_WITHIN_SOLVE for a in targets}}
        reasons = []
        for s, (own, where) in before.items():
            new, winner = _v83_read(scenarios[s], table, solve)
            if own is not None and new != own:
                reasons.append(f"a value in alternative {winner} would replace your dispatch_realization_method in "
                               f"alternative {where} in scenario {s}")
            elif s in candidates and new != _V83_WITHIN_SOLVE:
                reasons.append(f"the within_solve written for scenario {s} would not win there")
        if not reasons:
            writes.extend((solve, a) for a in sorted(targets))
        else:
            undecided.append((solve, "; ".join(reasons), tuple(candidates)))
    return tuple(writes), tuple(undecided)


def _v83_solve_inputs(db) -> tuple:
    """:func:`plan_dispatch_realization_method` arguments read from *db* (parsed values only; a null method row
    is kept: it reads as no method where it wins, :func:`_v83_read`; an empty map has no branch)."""
    branches, methods = {}, {}
    for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="stochastic_branches"):
        value = pv["parsed_value"]
        branches[(pv["entity_byname"][0], pv["alternative_name"])] = value is not None and not (
            isinstance(value, Map) and len(value) == 0)
    for pv in db.find_parameter_values(entity_class_name="solve",
                                       parameter_definition_name="dispatch_realization_method"):
        methods[(pv["entity_byname"][0], pv["alternative_name"])] = pv["parsed_value"]
    return branches, methods, _scenario_alternatives(db)


def _v83_dispatch_realization_method(db) -> None:
    """Part 5 of v83: ``dispatch_realization_methods`` gets ``deterministic``, the new shipped default.

    Appends the member, writes ``within_solve`` where a stochastic solve relied on the default (the engine reads no
    solve definition default: an absent method ran within_solve; :func:`plan_dispatch_realization_method`, results
    unchanged), sets a ``within_solve`` definition default to ``deterministic`` (a ``NULL`` or another member stays)
    and replaces the description by text alone.  Skips the member, writes and default without the class, the
    definition or the list.  Idempotent; no commit, no marker."""
    name = "dispatch_realization_methods"
    if (db.get_item("entity_class", name="solve") and db.get_item("parameter_value_list", name=name)
            and _pdef_or_none(db, "solve", "dispatch_realization_method") is not None):
        before = _list_members(db, [name])
        # Not from *before*: _list_members leaves out the members not yet committed (a rerun in the session).
        rerun = any(x["parsed_value"] == _V83_DETERMINISTIC
                    for x in db.get_items("list_value", parameter_value_list_name=name))
        _append_list_members(db, name, (_V83_DETERMINISTIC,))
        _v83_check_members(db, name, before)  # a changed member would fail the by-value writes first
        writes, undecided = plan_dispatch_realization_method(*_v83_solve_inputs(db))
        _v83_write_values(db, "solve", "dispatch_realization_method",
                          [(solve, alternative, _V83_WITHIN_SOLVE) for solve, alternative in writes])
        if rerun:
            _already_applied(83, "dispatch_realization_method: already written")
        if writes:
            print("v83: dispatch_realization_method within_solve written on the stochastic solves that relied on "
                  "the default (results unchanged):")
            for solve, alternative in writes:
                print(f"  solve {solve} [{alternative}]")
        for solve, reason, candidates in undecided:
            print(f"v83: dispatch_realization_method not written for solve {solve}: {reason}. In "
                  f"{_v83_scenarios_text(candidates)} it now runs deterministic: the forecasts are ignored (results "
                  "change), or the run stops when there is no realized: yes branch. Set dispatch_realization_method "
                  "within_solve yourself where it should apply.")
        pdef = _pdef_or_none(db, "solve", "dispatch_realization_method")
        if pdef["parsed_value"] == _V83_WITHIN_SOLVE:
            val, typ = to_database(_V83_DETERMINISTIC)
            db.update_parameter_definition(id=pdef["id"], default_value=val, default_type=typ)
    _replace_text_pairs(db, _v83_dispatch_realization_texts())


#: Part 3 of v83: the behaviour-change markers of the realization changes (``{"since": 83, "solves": [...]}``).
V83_WITHIN_SOLVE_MARKER = "rolling_within_solve_one_solve"
V83_SEPARATE_DISPATCH_MARKER = "separate_dispatch_realization_looks_ahead"


def plan_realization_notices(branches, modes, methods, scenarios) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The solves part 3 of v83 lists in the markers :data:`V83_WITHIN_SOLVE_MARKER` and
    :data:`V83_SEPARATE_DISPATCH_MARKER` (plan §2a "Notices").  Pure: no database.

    *branches* ``{(solve, alternative): the stochastic_branches value is non-empty}``, *modes* and *methods*
    ``{(solve, alternative): parsed solve_mode / dispatch_realization_method, None for a null row}``, *scenarios*
    ``{scenario: [alternative, ...]}`` lowest rank first.  In each scenario a solve is *stochastic* when its winning
    ``stochastic_branches`` is non-empty (one forecast counts), *rolling* when its winning ``solve_mode`` is
    ``rolling_window``, and its method is the winning explicit ``dispatch_realization_method`` (part 5's writes
    included; no definition default is read: the engine reads none).  Listed for within_solve: rolling, stochastic
    and ``within_solve`` in some scenario; for separate_dispatch: stochastic and ``separate_dispatch`` in some
    scenario, rolling or single.  A solve that is not stochastic in the scenario where its method resolves is not
    listed for it.  Returns ``(within_solve solves, separate_dispatch solves)``, sorted."""
    within, separate = set(), set()
    for solve in sorted({key[0] for key in branches}):
        for alts in scenarios.values():
            where = _v77_winner(alts, branches, solve)
            if where is None or not branches[(solve, where)]:
                continue
            method = _v83_read(alts, methods, solve)[0]
            if method == "separate_dispatch":
                separate.add(solve)
            elif method == _V83_WITHIN_SOLVE and _v83_read(alts, modes, solve)[0] == "rolling_window":
                within.add(solve)
    return tuple(sorted(within)), tuple(sorted(separate))


def _v83_realization_notices(db) -> None:
    """Part 3 of v83: rolling ``within_solve`` solves now realize each jump in one solve, and ``separate_dispatch``
    realizations look ahead after the jump (their first solve shares only the units online of ``_fix_in_uc``
    units).  Lists the solves whose results change (:func:`plan_realization_notices`) in the markers
    ``rolling_within_solve_one_solve`` and ``separate_dispatch_realization_looks_ahead`` (``{"since": 83,
    "solves": [...]}``, merged with a stored list).  Runs after part 5, whose writes count.  Metadata only;
    idempotent; no commit."""
    if not db.get_item("entity_class", name="solve"):
        return
    branches, methods, scenarios = _v83_solve_inputs(db)
    modes = {(pv["entity_byname"][0], pv["alternative_name"]): pv["parsed_value"]
             for pv in db.find_parameter_values(entity_class_name="solve", parameter_definition_name="solve_mode")}
    for change_id, solves in zip((V83_WITHIN_SOLVE_MARKER, V83_SEPARATE_DISPATCH_MARKER),
                                 plan_realization_notices(branches, modes, methods, scenarios)):
        name = BEHAVIOUR_CHANGE_MARKER_PREFIX + change_id
        stored: list = []
        for md in db.get_items("metadata"):
            if md["name"] == name:
                try:
                    stored = list(json.loads(md["value"]).get("solves") or [])
                except (TypeError, ValueError, AttributeError):
                    stored = []
        merged = stored + [s for s in solves if s not in stored]
        if merged:
            _v77_set_marker(db, change_id, json.dumps({"since": 83, "solves": merged}))


# --- Version 83: the batch ---------------------------------------------------------------------------------------

#: The members v83 adds to each value list (parts 1, 2 and 5): what a database at 83 has on the lists it has.
_V83_NEW_MEMBERS: dict[str, tuple[str, ...]] = {"minimum_time_methods": _V83_MIN_TIME_METHODS,
                                                "startup_methods": _V83_STARTUP_METHODS,
                                                "dispatch_realization_methods": (_V83_DETERMINISTIC,)}


#: Part 4 of v83 (item 278): the descriptions whose shipped text names an old member or an old set of value shapes,
#: ``{(class, name): (old, new)}``.  The old texts are what the v25 template has at 82; the new ones name the current
#: members and the shapes each parameter takes.
_V83_SHAPE_TEXTS: dict[tuple[str, str], tuple[str, str]] = {
    ("connection", "availability"): (
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for connection flows. Constant or time.",
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for connection flows. Constant, period, time or "
        "period+time."),
    ("connection", "other_operational_cost"): (
        "[CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, Period or time.",
        "[CUR/MWh] Other operational variable cost for trasferring over the connection. Constant, period, time or "
        "period+time."),
    ("group", "co2_method"): (
        "[choice] Choice of the CO2 method: no_method, price, period, total, price_period, price_total, period_total, "
        "price_period_total",
        "[choice] Choice of the CO2 method: none, price, period, total, price_period, price_total, period_total, "
        "price_period_total"),
    ("group", "co2_price"): (
        "[CUR/tCO2] CO2 price for a group of nodes. Constant, period or time.",
        "[CUR/tCO2] CO2 price for a group of nodes. Constant, period, time or period+time."),
    ("node", "availability"): (
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for storage. Constant or time.",
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for storage. Constant, period, time, period+time or "
        "stochastic time."),
    ("node", "penalty_up"): (
        "[CUR/MWh] Penalty cost for decreasing consumption in the node (energy not served). Constant, Period or Time.",
        "[CUR/MWh] Penalty cost for decreasing consumption in the node (energy not served). Constant, period, time, "
        "period+time, or stochastic time."),
    ("node", "penalty_down"): (
        "[CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, Period or Time.",
        "[CUR/MWh] Penalty cost for increasing consumption in the node (excess energy). Constant, period, time, "
        "period+time, or stochastic time."),
    ("node", "storage_state_reference_value"): (
        "[-] 0-1, share of the storage capacity. Relative state of storage at then end of each solve (can be "
        "overwritten in the next solve).  Requires 'use_reference_value' in 'storage_solve_horizon_method'. Constant.",
        "[-] 0-1, share of the storage capacity. Relative state of storage at then end of each solve (can be "
        "overwritten in the next solve).  Requires 'use_reference_value' in 'storage_solve_horizon_method'. Constant, "
        "period, time or period+time."),
    ("reserve__upDown__group", "reservation"): (
        "[MW] Amount of reserve to be reserved. Constant or time.",
        "[MW] Amount of reserve to be reserved. Constant, period, time, period+time or stochastic time."),
    ("unit", "availability"): (
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for flows from/to the unit. For online units, the "
        "online variable is multiplied by the availability. Constant or time.",
        "[-] e.g. 0.9 means 90%. Fraction of capacity available for flows from/to the unit. For online units, the "
        "online variable is multiplied by the availability. Constant, period, time, period+time or stochastic time."),
    **{(cls, "other_operational_cost"): (
        "[CUR/MWh] Other operational variable cost for energy flows. Constant, Period or Time.",
        "[CUR/MWh] Other operational variable cost for energy flows. Constant, period, time, period+time or "
        "stochastic time.") for cls in ("unit__inputNode", "unit__outputNode")},
}


#: Part 4 of v83 (item 278, D26): the descriptions that a database made from a release's shipped database or template
#: (FlexTool 3.22.0-3.43.0) still holds at 82, because the step that rewrote the text ran before the release's data
#: version: ``{(class, name): ((old texts...), new)}``.  The old texts are read from the shipped databases of those tags
#: migrated to 82 (``""`` is a description that was never written); the new ones are the shipped schema's.  Kept apart
#: from :data:`_V83_SHAPE_TEXTS`, whose single old text per definition is what the v25 template holds at 82.
_V83_RELEASE_TEXTS: dict[tuple[str, str], tuple[tuple[str, ...], str]] = {
    ("unit", "minimum_time_method"): (
        ("[choice] Not functional yet. Choice between minimum up- and downtimes (<empty>, min_downtime, min_uptime, "
         "both).",),
        "[choice] Choice between minimum up- and downtimes (none, min_downtime, min_uptime, both). Setting this to "
        "anything other than 'none' will activate online variables (at least linear) for the unit. Default: none (no "
        "minimum time constraints)."),
    ("node", "penalty_up"): (
        ("[CUR/MW] Penalty cost for decreasing consumption in the node. Constant, Period or Time.",),
        _V83_SHAPE_TEXTS[("node", "penalty_up")][1]),
    ("node", "penalty_down"): (
        ("[CUR/MW] Penalty cost for increasing consumption in the node. Constant, Period or Time.",),
        _V83_SHAPE_TEXTS[("node", "penalty_down")][1]),
    ("group", "penalty_inertia"): (
        ("[CUR/MWs per hour] Penalty for violating the inertia constraint. Constant or period.",),
        "[CUR/MWs per hour] Penalty for violating the inertia constraint. Cost scales with the duration of the "
        "violation. Constant or period."),
    ("connection", "reactance"): (
        ("",),
        "[p.u.] Per-unit reactance of the transmission line. Used for DC power flow when the connection is within a "
        "nodeGroup that has transfer_method set to dc_power_flow_with_angles. The susceptance used in the flow "
        "equation is base_MVA / reactance."),
    ("connection", "transfer_method"): (
        ("[choice] Choice of transfer method",),
        "[choice] Choice of transfer method. Options: regular (default), no_losses_no_variable_cost, "
        "variable_cost_only, exact, unidirectional. 'unidirectional' restricts flow to source → sink only (single "
        "non-negative variable); 'regular'/'exact'/'variable_cost_only' are bidirectional two-variable variants; "
        "'no_losses_no_variable_cost' is a single signed variable with no losses and no VOM."),
    ("group", "transfer_method"): (
        ("[choice] Override transfer_method for all connections within this nodeGroup. Options: "
         "use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, "
         "variable_cost_only, dc_power_flow_with_angles. When set to dc_power_flow_with_angles, connections between "
         "member nodes use B-theta DC power flow (requires reactance parameter on connections).",),
        "[choice] Override transfer_method for all connections within this nodeGroup. Options: "
        "use_connection_transfer_methods (default, no override), no_losses_no_variable_cost, regular, exact, "
        "variable_cost_only, unidirectional, dc_power_flow_with_angles. Setting 'unidirectional' gates every member "
        "connection to one-way flow (source → sink only). Setting 'dc_power_flow_with_angles' uses B-theta DC power "
        "flow (requires reactance on connections)."),
    ("commodity", "price_ladder_cumulative"): (
        ("[CUR/MWh; MWh]",
         "[CUR/MWh; MWh] Stepped supply curve for price_method='price_ladder_cumulative'.  Structure: Map(tier -> "
         "{price, quantity}).  1-based integer tier index.  quantity=inf marks an unbounded tail tier.  "
         "Period-agnostic — the cap is a single total across the full model horizon."),
        "[CUR/MWh; MWh] Stepped supply curve for price_method='price_ladder_cumulative'.  2d map with rows "
        "'tier,price,quantity' — one row per tier, giving the tier's unit price and its cumulative quantity cap.  "
        "1-based integer tier index.  quantity=inf marks an unbounded tail tier.  Period-agnostic — the cap is a "
        "single total across the full model horizon."),
}


def _replace_release_texts(db) -> None:
    """The current description in place of each old shipped one of :data:`_V83_RELEASE_TEXTS` (exactly equal; a
    description the user edited is kept).  Idempotent."""
    for key, (olds, new) in _V83_RELEASE_TEXTS.items():
        pdef = _pdef_or_none(db, *key)
        if pdef is not None and (pdef["description"] or "") in olds:
            db.update_parameter_definition(id=pdef["id"], description=new)


def _v83_description_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions part 4 of v83 replaces by text alone
    (:func:`_replace_superseded_texts`): the twelve of :data:`_V83_SHAPE_TEXTS` and ``virtual_unitsize`` of units
    and connections, whose v77 text (:func:`_unit_count_vu_text`) names the removed ``binary`` startups."""
    out = dict(_V83_SHAPE_TEXTS)
    for cls in ("unit", "connection"):
        old = _unit_count_vu_text(cls)
        out[(cls, "virtual_unitsize")] = (old, old.replace(
            "Binary startups commit", "Integer startups (startup_method integer or integer_fix_in_uc) commit"))
    return out


def _v83_descriptions(db) -> None:
    """Part 4 of v83 (item 278): the current description in place of each old shipped one of
    :func:`_v83_description_texts` and of :data:`_V83_RELEASE_TEXTS` (by text alone, whatever the default; a
    description the user edited is kept), and the value list ``solver`` deleted when no definition is on it (``solve.solver`` is on ``solvers`` since v52; an
    older template can still hold the unused list).  Descriptions and an unused list only.  Idempotent."""
    _replace_text_pairs(db, _v83_description_texts())
    _replace_release_texts(db)
    solver = db.get_item("parameter_value_list", name="solver")
    if solver and not any(pd["parameter_value_list_name"] == "solver" for pd in db.get_items("parameter_definition")):
        db.remove_item("parameter_value_list", solver["id"])


def _v82_superseded_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions of :func:`_v82_superseded` (a development build of v82
    shipped them; the default stays) and their current text (:func:`_v82_texts`), replaced by text alone."""
    texts = _v82_texts()
    return {key: (old, texts[key]) for key, (_default, old) in _v82_superseded().items()}


def _v83_superseded_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions the parts of v83 replace by text alone."""
    return {**_v83_startup_texts(), **_v83_dispatch_realization_texts(), **_v83_description_texts()}


def _v83_carry_over(db) -> None:
    """Part 0 of v83: a database migrated to 82 by a development build of v82 gets the v82 definitions it lacks
    (an existing definition keeps its fields) and the current description in place of a superseded shipped one
    (:func:`_v82_superseded_texts`, by text alone).  After a full v82 it changes nothing.  Idempotent."""
    _v82_definitions_part(db, update=False)
    _replace_text_pairs(db, _v82_superseded_texts())


def _migrate_v83_schema_batch(db) -> None:
    """All schema changes of database version 83 (v82 -> v83), in one step.  The parts are numbered by name and
    run in the order below; each is idempotent, and the step commits once at the end.

    0. The v82 carry-over (:func:`_v83_carry_over`).
    1. ``minimum_time_method`` now decides whether a unit's min up/down times apply: the method is written where a
       unit relied on a time alone (results unchanged), with the marker ``minimum_time_method_written``
       (:func:`_v83_minimum_time_method`).
    2. ``startup_methods`` gets ``linear_fix_in_uc``, ``integer`` and ``integer_fix_in_uc``; every ``binary`` becomes
       ``integer_fix_in_uc`` (the same model) and the ``binary`` member is removed (:func:`_v83_startup_method`).
    5. ``dispatch_realization_methods`` gets ``deterministic``, the new shipped default; ``within_solve`` is written
       on the stochastic solves that relied on the default (results unchanged;
       :func:`_v83_dispatch_realization_method`).
    3. Rolling ``within_solve`` solves realize each jump in one solve and ``separate_dispatch`` realizations look
       ahead: the solves whose results change are listed in two behaviour-change markers (after part 5, whose
       writes count; :func:`_v83_realization_notices`).
    4. Item 278: the current description in place of fourteen old shipped ones (value shapes, ``no_method``,
       ``binary`` startups) and of the older texts that databases made from a release still hold
       (:data:`_V83_RELEASE_TEXTS`), and the unused value list ``solver`` deleted (:func:`_v83_descriptions`).

    On a non-local URL the list defaults are not repaired before the step (item 279): a default still stored as its
    member's text reads as nothing, and a description written on its definition stores it as ``NULL``; it read as
    nothing before too.
    """
    _v83_carry_over(db)
    _v83_minimum_time_method(db)
    _v83_startup_method(db)
    _v83_dispatch_realization_method(db)
    _v83_realization_notices(db)
    _v83_descriptions(db)
    _commit_step(db, "v83: minimum_time_method decides the min up/down times, startup_method binary is "
                     "integer_fix_in_uc, dispatch_realization_method deterministic is the default, the "
                     "realization of stochastic solves looks ahead, descriptions updated")


# --- Version 84: one branching setting of investment solves (item 1024): ``v84_branching`` ------------------------

def _migrate_v84_schema_batch(db) -> None:
    """All schema changes of database version 84 (v83 -> v84), in one step:
    :func:`v84_branching._migrate_v84_schema_batch` (imported here: it uses the helpers of this module)."""
    from flextool.update_flextool import v84_branching

    v84_branching._migrate_v84_schema_batch(db)


#: The schema step of the current data version.  The template sync reruns it on the master template, so that schema
#: entries appended to its tables after the version was first stamped reach the shipped schema (a database at the
#: current version is otherwise not migrated again).
RERUNNABLE_HEAD_STEP: dict[int, Callable] = {84: _migrate_v84_schema_batch}


def _head_schema_additions() -> tuple[set[tuple[str, str]], dict[str, tuple[str, ...]]]:
    """``(definitions, list members)`` the schema steps of the current release add: what a database at that
    version must have (on the classes it has).  4.7.0 has two steps, 83 and 84 (v84 inside the release)."""
    if FLEXTOOL_DB_VERSION != 84:
        return set(), {}
    from flextool.update_flextool.v84_branching import _V84_LIST, _V84_METHODS, _v84_definitions

    return set(_v82_definitions()) | set(_v84_definitions()), {**_V83_NEW_MEMBERS, _V84_LIST: _V84_METHODS}


def _head_schema_missing_ro(path) -> bool:
    """Whether the database *path* (read read-only) lacks a definition (on a class it has) or a list member (of a
    list it has) that the schema step of the current data version adds (``True`` when it cannot be read)."""
    import sqlite3

    from flextool.update_flextool.frozen import _connect_ro

    definitions, lists = _head_schema_additions()
    if not definitions and not lists:
        return False
    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return True
    try:
        classes = {r[0] for r in con.execute("SELECT name FROM entity_class").fetchall()}
        have = set(con.execute(
            "SELECT ec.name, pd.name FROM parameter_definition pd "
            "JOIN entity_class ec ON pd.entity_class_id = ec.id").fetchall())
        members: dict[str, set] = {r[0]: set() for r in con.execute("SELECT name FROM parameter_value_list")}
        for name, value, typ in con.execute(
                "SELECT pvl.name, lv.value, lv.type FROM list_value lv "
                "JOIN parameter_value_list pvl ON lv.parameter_value_list_id = pvl.id").fetchall():
            members.setdefault(name, set()).add(from_database(value, typ))
    except sqlite3.Error:
        return True
    finally:
        con.close()
    if any(cls in classes and (cls, name) not in have for cls, name in definitions):
        return True
    return any(name in members and not set(wanted) <= members[name] for name, wanted in lists.items())


def _head_superseded() -> dict[tuple[str, str], list[tuple[str | None, str]]]:
    """The superseded shipped ``(default, description)`` pairs that the refresh of the current data version replaces,
    per definition.  None at 83: every database at 82 passed the v81 refresh inside step 82, and the v82 entries are
    description-only (:func:`_head_superseded_texts`)."""
    return {}


def _head_superseded_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the superseded shipped descriptions that the refresh of the current data
    version replaces by text alone, whatever the default (:func:`_v82_superseded_texts`,
    :func:`_v83_superseded_texts` and :func:`v84_branching._v84_dispatch_realization_texts`, applied in that order)."""
    if FLEXTOOL_DB_VERSION != 84:
        return {}
    return {**_v82_superseded_texts(), **_v83_superseded_texts()}


def _is_superseded_pair(cls: str, name: str, default, description: str | None) -> bool:
    """Whether the definition ``(cls, name)`` has exactly a superseded shipped pair of :func:`_head_superseded`:
    its parsed default and its description both equal to what FlexTool shipped.  The test of
    :func:`_head_defaults_stale_ro`; :func:`_replace_superseded_pairs` applies the same one (:func:`_pair_is`)."""
    return any(_pair_is(pair, default, description) for pair in _head_superseded().get((cls, name), ()))


def _head_defaults_stale_ro(path) -> bool:
    """Whether the database *path* (read read-only) has a definition with a superseded shipped default and
    description (:func:`_is_superseded_pair`) or a superseded shipped description (:func:`_head_superseded_texts`
    and :data:`_V83_RELEASE_TEXTS`, by text alone; ``True`` when it cannot be read)."""
    import sqlite3

    from flextool.update_flextool.frozen import _connect_ro

    superseded = _head_superseded()
    texts = {key: old for key, (old, _new) in _head_superseded_texts().items()}
    release = {key: olds for key, (olds, _new) in _V83_RELEASE_TEXTS.items()} if FLEXTOOL_DB_VERSION == 84 else {}
    if FLEXTOOL_DB_VERSION == 84:
        from flextool.update_flextool.v84_branching import _v84_dispatch_realization_texts

        texts.update({key: old for key, (old, _new) in _v84_dispatch_realization_texts().items()})
    if not superseded and not texts and not release:
        return False
    names = sorted({name for _cls, name in (*superseded, *texts, *release)})
    try:
        con = _connect_ro(path)
    except sqlite3.Error:
        return True
    try:
        rows = con.execute(
            "SELECT ec.name, pd.name, pd.default_value, pd.default_type, pd.description FROM parameter_definition pd "
            "JOIN entity_class ec ON pd.entity_class_id = ec.id WHERE pd.name IN (%s)" % ",".join("?" * len(names)),
            names).fetchall()
        # A default from a value list is stored as a reference to its list member.
        members = {int(i): (v, t) for i, v, t in con.execute("SELECT id, value, type FROM list_value").fetchall()}
    except sqlite3.Error:
        return True
    finally:
        con.close()

    def parsed(value, typ):
        if value is None:
            return None
        if typ == "list_value_ref":
            # A payload that is not an id (a member's text, item 279) reads as None, as in spinedb_api.
            raw = value.encode() if isinstance(value, str) else bytes(value)
            member = members.get(int(raw)) if raw.isdigit() else None
            return from_database(*member) if member is not None else None
        return from_database(value, typ)

    return any(texts.get((cls, name)) == (desc or "") or (desc or "") in release.get((cls, name), ())
               or _is_superseded_pair(cls, name, parsed(value, typ), desc) for cls, name, value, typ, desc in rows)


def _v81_refresh(db, texts: dict[tuple[str, str], str]) -> None:
    """Bring a database migrated to 81 by a development build of v81 up to the frozen v81 tables: add the value lists,
    list members and definitions it lacks (an existing definition keeps its fields) and replace a superseded shipped
    default and description pair (:func:`_v81_superseded`) with the current one, both together.  *texts* are the
    current descriptions where a later step rewrote a v81 one (an added definition and a replaced pair get them).  A
    description the user edited, a default the user set and every value are kept; the behaviour-change markers of v81
    and its other descriptions are not written.  After a full v81 it changes nothing.  Idempotent."""
    texts = {**_v81_texts(), **texts}
    _v81_value_lists(db)
    _v81_definitions_part(db, update=False, texts=texts)
    _replace_superseded_pairs(db, _v81_superseded(), _v81_definitions(texts), texts)


def _refresh_head_schema(db) -> None:
    """Bring a database at the current data version to the schema of its release's steps: the v82 definitions it
    lacks (:func:`_v82_definitions_part`; an existing definition keeps its fields), the members v83 adds to the lists
    it has (:data:`_V83_NEW_MEMBERS`; no value is written, no member removed, no default changed: a database at 84
    has had steps 83 and 84), the v84 value list and definition it lacks (:func:`v84_branching._v84_definitions_part`) and the current description in place of a superseded shipped one (:func:`_head_superseded_texts`
    and :data:`_V83_RELEASE_TEXTS`, by text alone).  A description the user edited, a default the user set and every value are kept; otherwise a
    description is never rewritten.  Idempotent."""
    if FLEXTOOL_DB_VERSION == 84:
        from flextool.update_flextool.v84_branching import _v84_definitions_part, _v84_dispatch_realization_texts

        _v82_definitions_part(db, update=False)
        for name, members in _V83_NEW_MEMBERS.items():
            if db.get_item("parameter_value_list", name=name):
                _append_list_members(db, name, members)
        _v84_definitions_part(db, update=False)
        _replace_text_pairs(db, _head_superseded_texts())
        _replace_release_texts(db)
        _replace_text_pairs(db, _v84_dispatch_realization_texts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename',help= "The filepath of the database to be migrated")
    args = parser.parse_args()
    migrate_database(args.filename)
