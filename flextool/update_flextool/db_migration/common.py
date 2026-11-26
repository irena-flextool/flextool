"""The helpers the migration steps share: commits, bounded removals, definitions, value lists, texts."""
import json
import logging

from spinedb_api import from_database, import_data, SpineDBAPIError, to_database
from spinedb_api.exception import NothingToCommit

from flextool._resources import package_data_path


def _pre_v26_template(name: str) -> str:
    """Return the absolute filesystem path of a bundled pre-v26 template JSON."""
    return str(package_data_path(f"schemas/pre_v26/{name}"))


def _commit_step(db, message):
    """Commit a migration step, tolerating the no-op case.

    spinedb_api raises ``NothingToCommit`` when ``add_update_item`` /
    ``update_item`` calls produced zero net changes — typically because
    the requested rows already exist with identical fields (a step that
    has been hand-applied or carried in by an earlier partial migration
    that never bumped ``model.version``).  Treating that signal as
    fatal would abort the rest of the migration mid-chain.  Log and
    continue so subsequent steps still run.  The version stamp is not
    committed here on its own: :func:`_stage_version` puts it in the
    session and the next commit carries it.
    """
    try:
        transformations, _info = db.commit_session(message)
    except NothingToCommit:
        logging.info("Migration step idempotent (no changes): %s", message)
        return
    _reread_transformed(db, transformations)


def _reread_transformed(db, transformations) -> None:
    """Make the open session read what spinedb_api's commit wrote behind it.

    Every commit runs spinedb_api's compatibility transformations on the file: while a class has an ``is_active``
    definition bound to a list with ``yes``, they set its ``active_by_default`` from that definition's default (0
    unless it is ``yes``) and turn its ``is_active`` values into ``entity_alternative`` rows.  The open mapping keeps
    what it read before.  On a file from before the SpineDB entity schema (v3.0.0 .. v3.5.1) that is 1 for
    ``constraint`` and the two reserve classes while the file holds 0, and v56 would decide on the 1 (item 1023).
    The item types the transformations changed are reset, so the next step reads the file; the commit has just
    written every change of the session, so nothing is lost.
    """
    changed = set()
    for item_type, (added, updated, removed) in transformations:
        if item_type == "entity_class":
            if any(db.get_item("entity_class", id=x["id"]).get("active_by_default") != x["active_by_default"]
                   for x in updated):
                changed.add(item_type)
        elif added or updated or removed:
            changed.add(item_type)
    if changed:
        db.reset(*changed)


#: The most runs of consecutive ids one commit deletes from a table.  spinedb_api deletes the removed rows of a table
#: in one ``DELETE ... WHERE id >= ? AND id <= ? OR ...`` statement, one term and two bound variables per run.  SQLite
#: refuses an expression deeper than 1000 (``Expression tree is too large``) and, before 3.32, more than 999 bound
#: variables: the scattered values of a large database exceed both.
_MAX_REMOVED_RANGES = 400


def _id_runs(ids) -> list[list[int]]:
    """The runs of consecutive integers in *ids*, in order."""
    runs: list[list[int]] = []
    for i in sorted(set(ids)):
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _pending_removed_db_ids(db, item_type: str) -> set[int]:
    """The database ids of the *item_type* rows removed in the session and not yet committed (what the next commit
    deletes)."""
    from spinedb_api.mapped_item_status import Status
    from spinedb_api.temp_id import resolve

    return {resolve(item["id"]) for item in db.mapped_table(item_type).values()
            if item.status == Status.to_remove and item.has_valid_id}


#: What spinedb_api removes with an item (item type -> (dependent item type, the fields that refer to the item)), the
#: dependents that can be many.  A dependent not listed here (a dimension row, a type-list row) shares the item's id
#: in its DELETE or is one of a few per item.
_REMOVAL_CASCADES: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "entity_class": (("entity", ("class_id",)), ("parameter_definition", ("entity_class_id",))),
    "entity": (("entity", ("element_id_list",)), ("parameter_value", ("entity_id",)),
               ("entity_alternative", ("entity_id",)), ("entity_metadata", ("entity_id",)),
               ("entity_group", ("entity_id", "member_id")), ("entity_location", ("entity_id",))),
    "parameter_definition": (("parameter_value", ("parameter_definition_id",)),),
    "parameter_value": (("parameter_value_metadata", ("parameter_value_id",)),),
}


def _remove_items_bounded(db, item_type: str, ids, message: str) -> None:
    """Remove the *item_type* items *ids* so that no commit deletes more than :data:`_MAX_REMOVED_RANGES` runs of
    ids from one table.

    When they fit beside the removals the session already holds, they are only removed (the caller's commit deletes
    them, as a plain ``remove_items``).  Otherwise the session is committed first and the items are removed and
    committed in batches, each commit named *message* with its batch number.  What cascades from an item
    (:data:`_REMOVAL_CASCADES`) is removed first, the same way.  The result is that of ``remove_items``.
    """
    from spinedb_api.temp_id import resolve

    def key(id_):
        # The database id; an item added in this session (no database id yet) is its own key.
        db_id = resolve(id_)
        return id_ if db_id is None else db_id

    by_key = {key(i): i for i in ids}
    if not by_key:
        return
    for child_type, refs in _REMOVAL_CASCADES.get(item_type, ()):
        children = [x["id"] for x in db.get_items(child_type)
                    if any((key(x[ref]) in by_key) if not ref.endswith("_list") else
                           any(key(e) in by_key for e in x[ref]) for ref in refs)]
        _remove_items_bounded(db, child_type, children, message)
    in_db = {k for k in by_key if isinstance(k, int)}
    if len(_id_runs(_pending_removed_db_ids(db, item_type) | in_db)) <= _MAX_REMOVED_RANGES:
        db.remove_items(item_type, *by_key.values())
        return
    # An item added in this session deletes no row.
    db.remove_items(item_type, *(i for k, i in by_key.items() if not isinstance(k, int)))
    _commit_step(db, f"{message} (before removing {len(in_db)} {item_type} items in batches)")
    runs = _id_runs(in_db)
    batches = [runs[i:i + _MAX_REMOVED_RANGES] for i in range(0, len(runs), _MAX_REMOVED_RANGES)]
    for n, batch in enumerate(batches, start=1):
        db.remove_items(item_type, *(by_key[k] for run in batch for k in run))
        _commit_step(db, f"{message} (removed {item_type} items, batch {n}/{len(batches)})")


def _ensure_parameter_group(db, name, color, priority):
    """Get-or-create a ``parameter_group`` so a new parameter can be
    attached to it.

    Every parameter must belong to a parameter_group or it is dropped from
    group-filtered tabular exports (v69 invariant, guarded by
    tests/test_parameter_group_coverage.py).  Real DBs always carry the
    standard groups (created in v44, recoloured in v45), but a DB built
    from scratch at a version above v44 — e.g. the minimal fixtures the
    migration tests seed — never ran v44 and so has no parameter_groups.
    Use the non-raising ``get_item`` (``db.item`` raises SpineDBAPIError
    when absent); the create only fires on such group-less DBs and must
    mirror v44/v45's canonical colour/priority for the group.
    """
    if not db.get_item("parameter_group", name=name):
        db.add_update_item(
            "parameter_group",
            name=name,
            color=color,
            priority=priority,
        )


#: The fixed start of the INFO record :func:`_already_applied` writes.
ALREADY_APPLIED_PREFIX = "Migration step already applied:"


def _already_applied(version: int, what: str) -> bool:
    """Log that step *version* finds its effect *what* already in the database and return ``True``.

    That happens on a rerun of the step after a migration was killed during or right after it (the stamp lags by one
    step, :func:`_stage_version`); a clean migration never logs it.
    """
    logging.info("%s v%d %s", ALREADY_APPLIED_PREFIX, version, what)
    return True


def _rename_or_drop_parameter_definition(db, entity_class_name, old_name, new_name, description=None):
    """Collision-safe parameter_definition rename within an entity class.

    Renames ``old_name`` -> ``new_name`` on ``entity_class_name`` via
    ``update_parameter_definition``.  If ``new_name`` is *already present*
    (an interrupted/re-run migration can leave the rename target in place
    while the source name is re-introduced by a template re-apply), the
    bare rename would hit spinedb_api's unique-name constraint and abort
    the whole migration.  In that case the stale ``old_name`` definition
    is removed instead, so the step converges to a single, correctly-named
    definition.  A no-op when ``old_name`` is absent.
    """
    def _find(name):
        # db.item raises SpineDBAPIError when there is no match; treat that
        # as "absent" rather than propagating.
        try:
            return db.item(parameter_definitions, entity_class_name=entity_class_name, name=name)
        except SpineDBAPIError:
            return None

    parameter_definitions = db.mapped_table("parameter_definition")
    old = _find(old_name)
    if not old:
        return
    new = _find(new_name)
    if new:
        # Target already exists (carried in by a prior partial migration);
        # the rename would collide.  Drop the stale source definition.
        _remove_items_bounded(db, "parameter_definition", [old["id"]],
                              f"removed {entity_class_name}.{old_name} ({new_name} is there)")
        return
    kwargs = {"id": old["id"], "name": new_name}
    if description is not None:
        kwargs["description"] = description
    db.update_parameter_definition(**kwargs)


def _detach_list_defaults(db, list_name: str, renames: dict[str, str]) -> list[tuple[object, str]]:
    """Clear each default of a definition on value list *list_name* that names an old member of *renames*.

    A default refers to its list member.  After an in-place member rename the open mapping still holds the old name
    for the definition, so every later update of it in the same migration run fails ("default value ... is not in
    the list"); after a member removal the reference dangles and later reads as whichever member takes the freed id.
    So a step that renames or removes members calls this before, and :func:`_attach_list_defaults` with the result
    after.  Returns ``(definition id, new name)`` pairs.
    """
    pending = []
    for pdef in db.get_items("parameter_definition"):
        if pdef["parameter_value_list_name"] != list_name:
            continue
        default = from_database(pdef["default_value"], pdef["default_type"])
        if isinstance(default, str) and default in renames and renames[default] != default:
            db.update_parameter_definition(id=pdef["id"], default_value=None, default_type=None)
            pending.append((pdef["id"], renames[default]))
    return pending


def _attach_list_defaults(db, pending: list[tuple[object, str]]) -> None:
    """Set the defaults cleared by :func:`_detach_list_defaults` to their new member names."""
    for pdef_id, new in pending:
        val, typ = to_database(new)
        db.update_parameter_definition(id=pdef_id, default_value=val, default_type=typ)


def add_version(db):
    # this function adds the version information to the database if there is none

    version_up = [["model", "version", 1, None, "Contains database version information."]]
    (num,log) = import_data(db, object_parameters = version_up)
    _commit_step(db,"Added version parameter")

    return 0

def remove_parameters_manual(db,obj_param_names):
    sq_def = db.object_parameter_definition_sq
    id_list = []
    for name_list in obj_param_names:
        object_name = name_list[0]
        parameter_name = name_list[1]
        param = db.query(sq_def).filter(sq_def.c.object_class_name == object_name).filter(sq_def.c.parameter_name == parameter_name).one_or_none()
        if param is not None:
            id_list.append(param.id)

    try:
        _remove_items_bounded(db, 'parameter_definition', id_list, "Removed parameters")
        _commit_step(db,"Removed parameters")
    except SpineDBAPIError:
        print("This removal has been done before, continuing")
    return 0

def add_parameters_manual(db,new_parameters):
    (num,log) = import_data(db, object_parameters = new_parameters)
    try:
        _commit_step(db,"Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def add_relationships_manual(db, new_relationships):
    (num,log) = import_data(db, relationship_parameters = new_relationships)
    try:
        _commit_step(db,"Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0

def _list_members(db, names) -> dict:
    """``{database id: (list, value, type)}`` of the stored members of the value lists *names*."""
    members = {}
    for name in names:
        for lv in db.get_items("list_value", parameter_value_list_name=name):
            db_id = getattr(lv["id"], "db_id", lv["id"])
            if db_id is not None:
                members[db_id] = (lv["parameter_value_list_name"], lv["value"], lv["type"])
    return members


def _import_value_lists_checked(db, lists):
    """``import_data`` of the value lists *lists* with a post-condition guard: an import adds members, it never
    rewrites one.  spinedb_api 0.31-0.34 could overwrite a list's first member here (and every default and value
    that used it), see value_list_history."""
    before = _list_members(db, dict.fromkeys(name for name, *_ in lists))
    num_log = import_data(db, parameter_value_lists=lists)
    after = _list_members(db, dict.fromkeys(name for name, *_ in lists))
    changed = sorted(f"{lst}: {from_database(v, t)!r}" for i, (lst, v, t) in before.items() if after.get(i) != (lst, v, t))
    if changed:
        raise FlexToolMigrationError("Adding value list members changed existing members (" + ", ".join(changed)
                                     + "); the migration stops so that no default or value changes with them. "
                                     "The file was restored from the backup; check that spinedb_api >= 0.35 is "
                                     "installed (pip show spinedb_api).")
    return num_log


def add_value_list_manual(db, new_value_lists):
    _import_value_lists_checked(db, new_value_lists)
    try:
        _commit_step(db,"Added new parameter value lists")
    except SpineDBAPIError:
        print("These value lists have been added before, continuing")
    return 0

def add_new_parameters(db, filepath):

    #get template JSON. This can be the master or old template if conflicting migrations in between
    with open(filepath, encoding="utf-8") as json_file:
        template = json.load(json_file)

    #Parameter values need to be added first! Otherwise the new value_list_name cannot be connected!
    #Add parameter_value_lists. Note that object_parameter import and value_list import work differently. Former replaces all, latter adds what is missing.
    (num,log) = _import_value_lists_checked(db, template["parameter_value_lists"])

    #With objective parameters, no duplicates are created. These will replace the old ones or create new. There will always be imports.
    (num,log) = import_data(db, object_parameters = template["object_parameters"])

    try:
        _commit_step(db,"Added new parameters")
    except SpineDBAPIError:
        print("These parameters have been added before, continuing") 
    return 0


def _per_class(version, classes, old, new):
    return tuple((version, cls, old, cls, new) for cls in classes)


_IO = ("unit__inputNode", "unit__outputNode")
_CAPACITY_CLASSES = ("unit", "connection", "node")

#: Every rename of a parameter definition: (version of the rename step, class, old name, class, new name), in
#: version order.  ``timeline.timeline_duration_in_years``, removed at v25, has ``timeset.timeline`` (added by the same
#: step) for its successor.  A step that creates or updates a definition by the name it had then does not bring that
#: name back beside a later one (:func:`_later_definition`): a database whose migration failed or was killed under
#: FlexTool 4.6.0 or older can hold a later step's result under an older stamp.  Names that come back later are in
#: :data:`_REUSED_DEFINITIONS`.
_DEFINITION_RENAMES: tuple[tuple[int, str, str, str, str], ...] = (
    (25, "solve", "period_timeblockSet", "solve", "period_timeset"),
    (25, "timeblockSet", "block_duration", "timeset", "timeset_duration"),
    (25, "timeblockSet", "new_stepduration", "timeset", "new_stepduration"),
    (25, "timeline", "timeline_duration_in_years", "timeset", "timeline"),
    *_per_class(28, _CAPACITY_CLASSES, "interest_rate", "discount_rate"),
    (28, "model", "discount_rate", "model", "inflation_rate"),
    (28, "model", "discount_offset_investment", "model", "inflation_offset_investment"),
    (28, "model", "discount_offset_operations", "model", "inflation_offset_operations"),
    *_per_class(32, _CAPACITY_CLASSES, "constraint_capacity_coefficient", "constraint_invested_capacity_coefficient"),
    *_per_class(35, _IO, "coefficient", "flow_coefficient"),
    (42, "group", "output_node_flows", "group", "output_nodeGroup_dispatch"),
    (42, "group", "output_aggregate_flows", "group", "flow_aggregator"),
    (50, "timeset", "new_stepduration", "solve", "new_stepduration"),
    *_per_class(56, _CAPACITY_CLASSES, "constraint_invested_capacity_coefficient", "constraint_invested_capacity_coeff"),
    *_per_class(56, _CAPACITY_CLASSES, "constraint_cumulative_pre_built_capacity_coefficient",
                "constraint_cumulative_pre_built_capacity_coeff"),
    *_per_class(56, ("connection__node", *_IO), "constraint_flow_coefficient", "constraint_flow_coeff"),
    (56, "node", "constraint_state_coefficient", "node", "constraint_state_coeff"),
    *_per_class(56, _IO, "flow_coefficient", "conversion_flow_coeff"),
    *_per_class(56, _IO, "max_capacity_coefficient", "capacity_max_coeff"),
    *_per_class(56, _IO, "min_capacity_coefficient", "capacity_min_coeff"),
    (58, "group", "output_nodeGroup_dispatch", "group", "print_dispatch"),
    (58, "group", "output_nodeGroup_indicators", "group", "print_indicators"),
    *((58, "group", name, "flowGroup", name) for name in (
        "flow_aggregator", "max_instant_flow", "min_instant_flow", "max_cumulative_flow", "min_cumulative_flow")),
    (65, "group", "has_capacity_margin", "group", "capacity_margin_method"),
    (66, "node", "energy_margin", "node", "energy_margin_multiplier"),
    (70, "unit__inputNode", "capacity_max_coeff", "unit__inputNode", "input_share_max"),
    (70, "unit__inputNode", "capacity_min_coeff", "unit__inputNode", "input_share_min"),
    (75, "solve", "non_anticipativity_periods", "solve", "shared_operation_periods"),
    (75, "solve", "non_anticipativity_invest_periods", "solve", "shared_invest_periods"),
    (80, "group", "capacity_margin", "group", "capacity_margin_absolute"),
    (80, "group", "capacity_margin_share", "group", "capacity_margin_relative"),
    *_per_class(81, _CAPACITY_CLASSES, "lifetime", "lifetime_technical"),
    *_per_class(81, _CAPACITY_CLASSES, "salvage_value", "net_salvage_value"),
    (82, "model", "inflation_rate", "model", "social_discount_rate"),
    (82, "model", "inflation_offset_investment", "model", "discount_offset_investment"),
    (82, "model", "inflation_offset_operations", "model", "discount_offset_operations"),
)

#: The definition names a later step brings back after an earlier step removed or renamed them: (class, name) ->
#: (version that brings it back, the definitions that show the later one is there).  v23 types ``node.invest_forced``
#: and ``node.storate_state_end`` again after v15 / v13 removed them (no definition had value types before v23; v15
#: also removed ``unit.invest_forced``, v13 added ``group.share_loss_of_load``); v82 gives ``model.discount_offset_*``
#: back their pre-v28 names (and renames ``inflation_rate``, the v28 name of ``model.discount_rate``, to
#: ``social_discount_rate``).  An earlier step's rerun leaves the later definition alone (:func:`_reused_later`).
_REUSED_DEFINITIONS: dict[tuple[str, str], tuple[int, tuple[tuple[str, str, bool], ...]]] = {
    ("node", "invest_forced"): (23, (("unit", "invest_forced", False),)),
    ("node", "storate_state_end"): (23, (("group", "share_loss_of_load", True),)),
    ("model", "discount_offset_investment"): (82, (("model", "social_discount_rate", True),
                                                   ("model", "inflation_offset_investment", False))),
    ("model", "discount_offset_operations"): (82, (("model", "social_discount_rate", True),
                                                   ("model", "inflation_offset_operations", False))),
}


def _later_definition(db, version: int, cls: str, name: str) -> tuple[str, str] | None:
    """The later name of definition *cls*.*name* (as step *version* knows it) in *db* when the definition itself is
    absent: the database then has the renames of :data:`_DEFINITION_RENAMES` made after *version*.  ``None`` when the
    definition is there or none of its later names is."""
    if db.get_item("parameter_definition", entity_class_name=cls, name=name):
        return None
    key = (cls, name)
    for rename_version, old_cls, old, new_cls, new in _DEFINITION_RENAMES:
        if rename_version > version and (old_cls, old) == key:
            key = (new_cls, new)
            if key != (cls, name) and db.get_item("parameter_definition", entity_class_name=new_cls, name=new):
                return key
    return None


def _upsert_definition(db, version: int, cls: str, name: str, **fields) -> None:
    """``add_update_item`` of definition *cls*.*name* with *fields*, unless it is absent and a later name of it
    exists (:func:`_later_definition`): then step *version* is already applied to it."""
    later = _later_definition(db, version, cls, name)
    if later is not None:
        _already_applied(version, f"{cls}.{name} is {later[0]}.{later[1]}")
        return
    db.add_update_item("parameter_definition", entity_class_name=cls, name=name, **fields)


def _reused_later(db, version: int, cls: str, name: str) -> bool:
    """Whether definition *cls*.*name* in *db* is the one a step after *version* brought back
    (:data:`_REUSED_DEFINITIONS`), which step *version* must not remove or rename."""
    entry = _REUSED_DEFINITIONS.get((cls, name))
    if entry is None or entry[0] <= version:
        return False
    pdef = db.get_item("parameter_definition", entity_class_name=cls, name=name)
    if not pdef or (entry[0] == 23 and not pdef["parameter_type_list"]):
        return False
    if all(bool(db.get_item("parameter_definition", entity_class_name=c, name=n)) == present
           for c, n, present in entry[1]):
        return _already_applied(version, f"{cls}.{name} is the definition v{entry[0]} brought back")
    return False


def _unless_renamed(db, version: int, list_name: str, member: str, later: str) -> list[list[str]]:
    """The members ``none`` and *member* of value list *list_name* for ``add_value_list_manual``, without *member*
    when the list has *later* (its v62 name) instead: step *version* is then already applied."""
    members = {lv["parsed_value"] for lv in db.find_list_values(parameter_value_list_name=list_name)}
    if later in members and member not in members:
        _already_applied(version, f"{list_name} member {member} is {later}")
        return [[list_name, "none"]]
    return [[list_name, "none"], [list_name, member]]


class FlexToolMigrationError(RuntimeError):
    """Raised when a database migration hits an unresolvable data
    inconsistency and cannot continue safely."""


class MigrationCancelled(RuntimeError):
    """Raised when :func:`migrate_database` is interrupted via its
    ``cancel_check`` callback.

    Carries the last successfully completed version number so the caller
    can report partial progress.  Cancellation is checked at the top of
    each step, never inside a commit, so an in-flight step always
    finishes before the exception is raised.
    """

    def __init__(self, last_completed_version: int):
        super().__init__(
            f"Migration cancelled after version {last_completed_version}"
        )
        self.last_completed_version = last_completed_version


# Behaviour changes recorded on every database migrated from below v77; a
# notice for one fires only while its marker is present.  The storage
# default marker is written by the storage step (conditional on the old
# default).
BEHAVIOUR_CHANGE_MARKER_PREFIX = "flextool_behaviour_change:"


def _describe_pv(pv) -> str:
    return (f"{pv['entity_class_name']} {'/'.join(pv['entity_byname'])} "
            f"[{pv['alternative_name']}] = {pv['parsed_value']}")


def _pdef_or_none(db, entity_class_name: str, name: str):
    """``parameter_definition`` row or ``None`` (``db.item`` raises
    ``SpineDBAPIError`` for a missing row)."""
    try:
        return db.item(db.mapped_table("parameter_definition"),
                       entity_class_name=entity_class_name, name=name)
    except SpineDBAPIError:
        return None


def _append_list_members(db, name: str, members) -> None:
    """Append the *members* missing (by parsed value) from the existing value list *name*, in the given order, at
    1 + the largest stored index.  A stored member is never reordered or rewritten."""
    items = db.get_items("list_value", parameter_value_list_name=name)
    have = {x["parsed_value"] for x in items}
    index = max((x["index"] for x in items), default=-1)
    for member in members:
        if member in have:
            continue
        index += 1
        val, typ = to_database(member)
        db.add_list_value(parameter_value_list_name=name, value=val, type=typ, index=index)
        have.add(member)


def _replace_superseded_texts(db, superseded: dict[tuple[str, str], str], texts: dict[tuple[str, str], str]) -> None:
    """Replace the description of each definition ``(class, name)`` of *superseded* that is exactly the old shipped
    text with the text of *texts*, whatever its default (a description-only update; :func:`_replace_superseded_pairs`
    is for a default and description pair).  A description the user edited is kept.  Idempotent."""
    for key, old in superseded.items():
        pdef = _pdef_or_none(db, *key)
        if pdef is not None and (pdef["description"] or "") == old:
            db.update_parameter_definition(id=pdef["id"], description=texts[key])


def _replace_text_pairs(db, texts: dict[tuple[str, str], tuple[str, str]]) -> None:
    """:func:`_replace_superseded_texts` for *texts* ``{(class, name): (old shipped text, new text)}``."""
    _replace_superseded_texts(db, {k: old for k, (old, _new) in texts.items()},
                              {k: new for k, (_old, new) in texts.items()})


def _pair_is(old: tuple[object, str] | None, default, description: str | None) -> bool:
    """Whether a parsed default and a description are exactly the shipped pair *old* (``None``: no pair)."""
    return old is not None and (default, description or "") == old


def _replace_superseded_pairs(db, superseded: dict[tuple[str, str], tuple[str | None, str]],
                              current: dict[tuple[str, str], dict], texts: dict[tuple[str, str], str]) -> None:
    """Replace each definition that has exactly a *superseded* shipped ``(default, description)`` pair: with the
    default and description of *current* where the step defines the whole definition, else with the description
    of *texts* (the default stays).  An edited description or default is kept.  Idempotent."""
    for (cls, name), old in superseded.items():
        pdef = _pdef_or_none(db, cls, name)
        if pdef is None:
            continue
        default = (from_database(pdef["default_value"], pdef["default_type"])
                   if pdef["default_value"] is not None else None)
        if not _pair_is(old, default, pdef["description"]):
            continue
        fields = current.get((cls, name))
        if fields is None:  # a definition the step does not add: only its description was superseded
            db.update_parameter_definition(id=pdef["id"], description=texts[(cls, name)])
        else:
            db.update_parameter_definition(id=pdef["id"], default_value=fields["default_value"],
                                           default_type=fields["default_type"], description=fields["description"])
