"""Data version 82: social_discount_rate and the discount offsets."""
import json

from spinedb_api import from_database, Map, to_database

from flextool.update_flextool.db_migration.common import (
    _commit_step,
    _describe_pv,
    _ensure_parameter_group,
    _pdef_or_none,
    _remove_items_bounded,
)
from flextool.update_flextool.db_migration.v77 import _v77_set_marker
from flextool.update_flextool.db_migration.v79 import _v79_numbers
from flextool.update_flextool.db_migration.v81 import (
    _V81_CAPACITY,
    _V81_GROUPS,
    _v81_payback_time_text,
    _v81_refresh,
    _v81_texts,
)


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


def _v82_superseded_texts() -> dict[tuple[str, str], tuple[str, str]]:
    """``{(class, name): (old, new)}``: the descriptions of :func:`_v82_superseded` (a development build of v82
    shipped them; the default stays) and their current text (:func:`_v82_texts`), replaced by text alone."""
    texts = _v82_texts()
    return {key: (old, texts[key]) for key, (_default, old) in _v82_superseded().items()}
