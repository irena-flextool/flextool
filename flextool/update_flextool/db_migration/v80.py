"""Data version 80: the schema batch."""
from spinedb_api import from_database, to_database

from flextool.update_flextool.db_migration.common import (
    _attach_list_defaults,
    _commit_step,
    _detach_list_defaults,
    _pdef_or_none,
    _rename_or_drop_parameter_definition,
)


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
