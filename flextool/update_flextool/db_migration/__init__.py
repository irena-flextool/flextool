import argparse
import contextlib
import logging
import os
from typing import Callable

from spinedb_api import from_database, import_data, to_database

from flextool.sqlite_url import sqlite_url_string
from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_handle import open_database_mapping
from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _append_list_members,
    _commit_step,
    _pair_is,
    _pre_v26_template,
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
    _migrate_v77_schema_batch,
    _refresh_unit_count_texts,
    _SUPERSEDED_UNIT_COUNT_TEXTS,
)
from flextool.update_flextool.db_migration.v78 import _migrate_v78_unit_prefixes
from flextool.update_flextool.db_migration.v79 import _migrate_v79_schema_batch
from flextool.update_flextool.db_migration.v80 import _migrate_v80_schema_batch
from flextool.update_flextool.db_migration.v81 import _migrate_v81_schema_batch
from flextool.update_flextool.db_migration.v82 import (
    _migrate_v82_schema_batch,
    _v82_definitions,
    _v82_definitions_part,
    _v82_superseded_texts,
)
from flextool.update_flextool.db_migration.v83 import (
    _migrate_v83_schema_batch,
    _replace_release_texts,
    _V83_NEW_MEMBERS,
    _V83_RELEASE_TEXTS,
    _v83_superseded_texts,
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


# --- Version 84: one branching setting of investment solves (item 1024): ``v84_branching`` ------------------------

def _migrate_v84_schema_batch(db) -> None:
    """All schema changes of database version 84 (v83 -> v84), in one step:
    :func:`v84_branching._migrate_v84_schema_batch` (imported here: it uses the helpers of this module)."""
    from flextool.update_flextool.db_migration import v84_branching

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
    from flextool.update_flextool.db_migration.v84_branching import _V84_LIST, _V84_METHODS, _v84_definitions

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
        from flextool.update_flextool.db_migration.v84_branching import _v84_dispatch_realization_texts

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


def _refresh_head_schema(db) -> None:
    """Bring a database at the current data version to the schema of its release's steps: the v82 definitions it
    lacks (:func:`_v82_definitions_part`; an existing definition keeps its fields), the members v83 adds to the lists
    it has (:data:`_V83_NEW_MEMBERS`; no value is written, no member removed, no default changed: a database at 84
    has had steps 83 and 84), the v84 value list and definition it lacks (:func:`v84_branching._v84_definitions_part`) and the current description in place of a superseded shipped one (:func:`_head_superseded_texts`
    and :data:`_V83_RELEASE_TEXTS`, by text alone).  A description the user edited, a default the user set and every value are kept; otherwise a
    description is never rewritten.  Idempotent."""
    if FLEXTOOL_DB_VERSION == 84:
        from flextool.update_flextool.db_migration.v84_branching import _v84_definitions_part, _v84_dispatch_realization_texts

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
