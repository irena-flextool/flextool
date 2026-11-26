"""Data version 81: technical lifetime and payback time, net salvage value, no_reinvestment."""
import json

from spinedb_api import from_database, to_database

from flextool.update_flextool.db_migration.common import (
    _append_list_members,
    _attach_list_defaults,
    _commit_step,
    _describe_pv,
    _detach_list_defaults,
    _ensure_parameter_group,
    _pdef_or_none,
    _remove_items_bounded,
    _replace_superseded_pairs,
)
from flextool.update_flextool.db_migration.v77 import _v77_set_marker
from flextool.update_flextool.db_migration.v79 import _v79_numbers, V79_NODE_UNIT_OF_MEASURE_DESCRIPTION


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
