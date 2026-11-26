"""Data versions 70-76."""
from spinedb_api import to_database

from flextool.update_flextool.db_migration.common import (
    _commit_step,
    _describe_pv,
    _ensure_parameter_group,
    _pdef_or_none,
    _remove_items_bounded,
    _rename_or_drop_parameter_definition,
    add_value_list_manual,
)


def _migrate_v71_stochastic_invest_method(db) -> None:
    """Add the solve.stochastic_invest_method opt-in (v70 -> v71).

    Two-stage stochastic investment with recourse (recourse plan
    §6b Slice D) is enabled per solve by this enum.  'none' (default,
    byte-identical to prior behaviour) keeps the R-O6 invariant —
    stochastic branches never enter the invest axis; v_invest stays
    realized-only.  'recourse' selects the Slice D machinery.

    Slice C ships only the schema + the validation guards: flag-on is
    HARD-REJECTED until Slice D lands (branch periods on the invest
    axis).  Bound to a new 'stochastic_invest_methods' value list;
    grouped under 'solve_advanced' (created if the DB lacks it).
    """
    add_value_list_manual(db, [
        ["stochastic_invest_methods", "none"],
        ["stochastic_invest_methods", "recourse"],
    ])

    default_val, default_type = to_database("none")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="stochastic_invest_method",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="stochastic_invest_methods",
        description=(
            "How stochastic branches participate in investment. 'none' "
            "(default) keeps investment realized-only: stochastic "
            "branches never enter the invest axis (v_invest stays on "
            "realized periods), byte-identical to prior behaviour. "
            "'recourse' enables two-stage stochastic investment with "
            "recourse — shared (non-anticipative) investment before the "
            "branch point plus per-branch recourse investment at and "
            "after the branching period, with probability-weighted "
            "investment costs. Reserved for future expected-value and "
            "nested-tree modes."
        ),
    )
    # solve_advanced is this param's canonical home in the schema.
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="stochastic_invest_method",
        parameter_group_name="solve_advanced",
    )

    _commit_step(db,
        "v71: added solve.stochastic_invest_method "
        "(stochastic_invest_methods: none/recourse, default none); "
        "opt-in for recourse stochastic investment (Slice D)."
    )


def _migrate_v72_non_anticipativity_periods(db) -> None:
    """Add the solve.non_anticipativity_periods window knob (v71 -> v72).

    Slice G decouples the operational non-anticipativity window from
    ``realized_periods``.  This Array-of-periods parameter selects the
    (d,t) set over which the four ``non_anticipativity_*`` families
    (storage/online/reserve) are pinned across stochastic branches:

    * unset (default) -> legacy ``realized_dispatch u fix_storage``
      window, byte-identical to prior behaviour;
    * empty Array ``[]`` -> empty window, operations branch freely from
      t0 (classic two-stage capacity expansion);
    * ``[p, ...]`` -> tie only over those periods' timesteps
      (intersected with steps_in_use at emit).

    Array-shaped (read by ``periods_to_tuples``, same as
    ``realized_periods`` / ``fix_storage_periods``): ``default_value`` and
    ``default_type`` are ``None`` and it has NO parameter_value_list.
    Migrating a DB adds only the definition (no per-solve value), so every
    migrated solve resolves to the unset -> legacy-window case.  Grouped
    under 'solve_advanced' (created if the DB lacks it).
    """
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="non_anticipativity_periods",
        default_value=None,
        default_type=None,
        description=(
            "Array of periods over which operational non-anticipativity "
            "(storage / online / reserve dispatch tied across stochastic "
            "branches) is enforced. Unset (default) uses the legacy window "
            "= all realized-dispatch timesteps union the fix_storage "
            "timesteps (byte-identical to prior behaviour). An explicitly "
            "empty Array frees operations from t0 (classic two-stage "
            "capacity expansion: one shared here-and-now investment plus "
            "operations that branch freely per scenario). A period list "
            "ties operations only over those periods' timesteps."
        ),
    )
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="non_anticipativity_periods",
        parameter_group_name="solve_advanced",
    )

    _commit_step(db,
        "v72: added solve.non_anticipativity_periods (Array of periods, "
        "default unset); decouples the operational non-anticipativity "
        "window from realized_periods (Slice G)."
    )


def _migrate_v73_non_anticipativity_invest_periods(db) -> None:
    """Add the solve.non_anticipativity_invest_periods knob (v72 -> v73).

    The invest-NA window is the standard two-stage stochastic-investment
    mode: an Array-of-periods parameter selecting the periods over which
    ``v_invest`` / ``v_divest`` are TIED across stochastic branches (a
    single shared here-and-now decision).  It pins the four
    ``non_anticipativity_invest_p/n`` and ``non_anticipativity_divest_p/n``
    constraint families across the branch fan of those periods:

    * unset (default) OR empty Array ``[]`` -> NO tie, byte-identical to
      prior behaviour (``recourse`` stays per-branch everywhere; ``none``
      stays shared);
    * ``[p, ...]`` -> tie invest/divest at each branch's copy of those
      periods to the realized anchor -> shared first-stage investment with
      per-branch (recourse) investment in later periods.

    Unlike the operational ``non_anticipativity_periods`` (v72), the
    unset/``[]`` distinction carries no behaviour here (both = no tie),
    so it is read by the plain ``periods_to_tuples`` (not a tri-state).

    Array-shaped (read by ``periods_to_tuples``, same as
    ``realized_periods`` / ``non_anticipativity_periods``):
    ``default_value`` and ``default_type`` are ``None`` and it has NO
    parameter_value_list.  Migrating a DB adds only the definition (no
    per-solve value), so every migrated solve resolves to unset -> no
    tie -> today's behaviour.  Grouped under 'solve_advanced' (created if
    the DB lacks it).
    """
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="non_anticipativity_invest_periods",
        default_value=None,
        default_type=None,
        description=(
            "Array of periods over which investment is non-anticipative "
            "(v_invest and v_divest tied across stochastic branches — a "
            "single shared here-and-now decision). Unset (default) or an "
            "empty Array means NO tie (byte-identical to prior behaviour: "
            "recourse stays per-branch, none stays shared). A period list "
            "-- e.g. the first period -- ties investment over those "
            "periods while later periods invest per-branch (recourse), "
            "the standard two-stage stochastic capacity-expansion mode. "
            "Requires stochastic_invest_method=recourse; a no-op under "
            "none (investment is already shared across all periods)."
        ),
    )
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="non_anticipativity_invest_periods",
        parameter_group_name="solve_advanced",
    )

    _commit_step(db,
        "v73: added solve.non_anticipativity_invest_periods (Array of "
        "periods, default unset); the standard two-stage stochastic "
        "investment mode — ties v_invest/v_divest across branches over "
        "the window (shared here-and-now decision), recourse later."
    )


def _migrate_v74_construction_lead_time(db) -> None:
    """Add construction_lead_time + _method per invest entity (v73 -> v74).

    Slice H — construction lead time.  Two per-entity parameters on each
    invest entity class (``unit`` / ``connection`` / ``node``):

    * ``construction_lead_time`` — ``[years]`` physical build time.
      Capacity ordered in a period becomes available this many years
      later, snapped to a period boundary per
      ``construction_lead_time_method``.  Constant or period (float / map,
      mirroring ``lifetime``).  Default 0 (instantaneous, as before).
    * ``construction_lead_time_method`` — how ``yr(order) +
      construction_lead_time`` snaps to a period boundary: ``immediate``
      (no lag; capacity from the decision period; byte-identical to prior
      behaviour), ``closest_seam`` (nearest boundary, exact tie rounds up;
      the effective default when a lead time is set), ``previous_seam``
      (round down), ``next_seam`` (round up).

    Both defaults are ``null`` (see design §0.2 / §6.2): the *unset* vs
    explicit-``immediate`` distinction is what lets the reader implement
    the two-level default (unset resolves to ``immediate`` when the lead
    time is 0, else ``closest_seam``).  A literal ``immediate`` default
    would materialise on every entity and destroy that distinction.

    Migrating a DB adds only the definitions (never a per-entity value),
    so every migrated entity resolves to unset -> ``L=0`` -> ``immediate``
    -> byte-identical to prior behaviour.  Grouped under ``investment``
    (mirrors ``lifetime``; created if the DB lacks it).
    """
    add_value_list_manual(db, [
        ["construction_lead_time_methods", "immediate"],
        ["construction_lead_time_methods", "closest_seam"],
        ["construction_lead_time_methods", "previous_seam"],
        ["construction_lead_time_methods", "next_seam"],
    ])

    lead_desc = (
        "[years] Construction lead time: capacity ordered in a period "
        "becomes available this many years later, snapped to a period "
        "boundary per construction_lead_time_method (defaults to "
        "closest_seam when set; use immediate to keep capacity available "
        "from the decision period). Constant or period. Default 0 "
        "(instantaneous, as before)."
    )
    method_desc = (
        "How yr(order)+construction_lead_time snaps to a period boundary: "
        "immediate (no lag; capacity from the decision period; "
        "byte-identical to prior behaviour), closest_seam (nearest "
        "boundary, exact tie rounds up; the effective default when a lead "
        "time is set), previous_seam (round down), next_seam (round up). "
        "Unset resolves to immediate when no lead time, else closest_seam."
    )
    # ``investment`` is construction_lead_time's canonical home.
    _ensure_parameter_group(db, "investment", "7fb095", 20)
    for class_name in ("unit", "connection", "node"):
        db.add_update_item(
            "parameter_definition",
            entity_class_name=class_name,
            name="construction_lead_time",
            default_value=None,
            default_type=None,
            parameter_group_name="investment",
            description=lead_desc,
        )
        # ``parameter_type_list=("float","map")`` assigns BOTH types rank 0
        # in spinedb_api (a quirk of the convenience arg — the map row then
        # collides with float on the unique (definition, rank) key and is
        # dropped on re-import).  Add the two allowed types as explicit
        # ``parameter_type`` items with distinct ranks (float 0, map 1),
        # matching how ``lifetime`` is stored.
        for rank, type_name in enumerate(("float", "map")):
            db.add_update_item(
                "parameter_type",
                entity_class_name=class_name,
                parameter_definition_name="construction_lead_time",
                type=type_name,
                rank=rank,
            )
        db.add_update_item(
            "parameter_definition",
            entity_class_name=class_name,
            name="construction_lead_time_method",
            default_value=None,
            default_type=None,
            parameter_type_list=("str",),
            parameter_value_list_name="construction_lead_time_methods",
            parameter_group_name="investment",
            description=method_desc,
        )

    _commit_step(db,
        "v74: added construction_lead_time (float/map, [years], default 0) "
        "and construction_lead_time_method (immediate/closest_seam/"
        "previous_seam/next_seam, default unset) on unit/connection/node; "
        "Slice H construction lead time — commissioning lag lives in the "
        "availability walk, cost stays at the order period."
    )


# Positive, non-negated descriptions for the renamed stochastic
# shared-decision window parameters (v75).  The old ``non_anticipativity_*``
# names were negations of the same concept; the new names state it
# positively: these periods are SHARED (identical) across branches.
_SHARED_OPERATION_PERIODS_DESC = (
    "Array of periods over which operations are SHARED (identical) across "
    "stochastic branches — the non-anticipative first-stage decision; "
    "later periods are per-branch recourse. Unset (default) uses the "
    "legacy window = all realized-dispatch timesteps union the fix_storage "
    "timesteps (byte-identical to prior behaviour). An explicitly empty "
    "Array frees operations from t0 (classic two-stage capacity "
    "expansion: one shared here-and-now investment plus operations that "
    "branch freely per scenario). A period list shares operations only "
    "over those periods' timesteps."
)
_SHARED_INVEST_PERIODS_DESC = (
    "Array of periods over which investment is SHARED (identical) across "
    "stochastic branches — the non-anticipative first-stage decision; "
    "later periods are per-branch recourse. Unset (default) or an empty "
    "Array means NO shared tie (byte-identical to prior behaviour: "
    "recourse stays per-branch, none stays shared). A period list -- e.g. "
    "the first period -- shares investment over those periods while later "
    "periods invest per-branch (recourse), the standard two-stage "
    "stochastic capacity-expansion mode. Requires "
    "stochastic_invest_method=recourse; a no-op under none (investment is "
    "already shared across all periods)."
)


def _migrate_v75(db) -> None:
    """Rename the stochastic shared-decision windows + add reserve_duration
    (v74 -> v75).

    Two folded changes:

    1. **Rename** the two (unreleased) non-anticipativity window params on
       the ``solve`` class to positive, non-negated names:

       * ``non_anticipativity_periods``        -> ``shared_operation_periods``
       * ``non_anticipativity_invest_periods`` -> ``shared_invest_periods``

       The renames are collision-safe (``_rename_or_drop_parameter_definition``)
       and carry their existing per-solve Array values by definition id, so
       a migrated DB keeps identical behaviour under the new names.  The
       descriptions are rewritten to state the meaning positively (these
       periods are SHARED/identical across branches — the non-anticipative
       first-stage decision).  No-ops when the source names are absent (a DB
       built fresh above v74 already carries the new names).

    2. **Add** ``reserve__upDown__group.reserve_duration`` for DBs that
       predate the parameter (mirrors ``_migrate_v71_reserve_duration`` from
       the original branch).  A single scalar (hours) coupling a
       storage-backed reserve provider's committed reserve power to the
       stored energy (up/discharge) or free headroom (down/charge) needed
       to sustain it for ``reserve_duration`` hours (issue #322).  Absent or
       0 means no storage-energy coupling, keeping the LP byte-identical to
       prior behaviour, so the parameter defaults to null and carries no
       value list.  Idempotent via ``add_update_item`` — a no-op when the
       definition is already present (e.g. the master template, which
       already carries it).  Attached to the existing ``reserve`` parameter
       group when that group is present.
    """
    # 1. Rename the two shared-decision window parameters.
    _rename_or_drop_parameter_definition(
        db, "solve", "non_anticipativity_periods",
        "shared_operation_periods", _SHARED_OPERATION_PERIODS_DESC,
    )
    _rename_or_drop_parameter_definition(
        db, "solve", "non_anticipativity_invest_periods",
        "shared_invest_periods", _SHARED_INVEST_PERIODS_DESC,
    )

    # 2. Add reserve__upDown__group.reserve_duration (storage energy
    #    adequacy for reserves, #322) for DBs that predate it.
    db.add_update_item(
        "parameter_definition",
        entity_class_name="reserve__upDown__group",
        name="reserve_duration",
        default_value=None,
        default_type=None,
        parameter_value_list_name=None,
        description=(
            "[h] Time span over which committed reserve power must be "
            "sustainable from storage. Energy obligation = reservation_MW × "
            "reserve_duration, corrected by the reserve provider's "
            "efficiency. Only affects reserve provided by units/connections "
            "that draw on (up) or charge (down) a storage node. Absent or 0 "
            "means no storage-energy coupling. Constant."
        ),
    )
    # Attach to the 'reserve' parameter group when it exists (consistent
    # with reservation / reserve_method).
    if db.get_item("parameter_group", name="reserve"):
        db.add_update_item(
            "parameter_definition",
            entity_class_name="reserve__upDown__group",
            name="reserve_duration",
            parameter_group_name="reserve",
        )

    _commit_step(db,
        "v75: renamed non_anticipativity_periods -> shared_operation_periods "
        "and non_anticipativity_invest_periods -> shared_invest_periods "
        "(positive names for the stochastic shared-decision windows); added "
        "reserve__upDown__group.reserve_duration ([h] scalar, default none; "
        "storage energy-adequacy coupling for reserves, #322)."
    )


_V76_DISPATCH_REALIZATION_METHOD_DESCRIPTION = (
    "How the stochastic branches of a solve are realized into a committed "
    "dispatch. 'within_solve' (default) is the normal path: realization "
    "stays inside the single optimisation — branches and their realized "
    "periods are resolved within one solve, byte-identical to prior "
    "behaviour. 'separate_dispatch' is an available advanced option that "
    "splits realization into a dedicated downstream dispatch stage, "
    "automatically spawning a separate realized-dispatch solve (a second "
    "market-clearing pass over the revealed branch) — the 2-stage seed of "
    "a future multi-stage sequential-market structure. Its "
    "automatic-selection defaults are still being refined."
)
_V76_STOCHASTIC_BRANCHES_DESCRIPTION = (
    "[4d-Map], Sets branches included in the solve. [Period, branch, "
    "start_time (time_step), realized (yes/no), weight (number)]. The "
    "branches form a probability-weighted forecast fan (forecasts-only). "
    "'realized' is an optional designation, at most one (0 or 1) per "
    "start_time, marking which branch's data are treated as the actuals "
    "for the committed / realized-dispatch results."
)


def _migrate_v76_dispatch_realization_method(db) -> None:
    """Add the solve.dispatch_realization_method opt-in (v75 -> v76).

    Selects how the stochastic branches of a solve are realized into a
    committed dispatch.  This is the 2-stage seed of a future multi-stage
    market structure: ``'within_solve'`` (default, byte-identical to prior
    behaviour) keeps realization inside the single optimisation — the
    branches and their realized periods are resolved within one solve, as
    today.  ``'separate_dispatch'`` will split realization into a dedicated
    downstream dispatch stage (a second market-clearing pass over the
    revealed branch), the first step toward a multi-stage sequential-market
    structure.

    W4 Slice 1 (W4a) shipped the schema; W4b now IMPLEMENTS the engine
    side: ``'separate_dispatch'`` is ACCEPTED for a single top-level
    ``single_solve``, for which the engine automatically registers a
    deterministic realized-dispatch sibling ``<solve>__realized_dispatch``
    (full invest→dispatch handoff + last-wins).  Other shapes (rolling,
    contains_solves children, nested) and solves that author
    fix_storage_periods remain rejected (later slices).  Bound to a new
    'dispatch_realization_methods' value list; grouped under
    'solve_advanced' (created if the DB lacks it).  This is a structural
    clone of the v71 ``stochastic_invest_method`` line.

    This step also refreshes two stale user-facing descriptions to the
    corrected stochastics model (description-only, no structural change):
    ``solve.dispatch_realization_method`` (``separate_dispatch`` is now an
    available advanced option, not "not yet implemented") and the older
    ``solve.stochastic_branches`` ('realized' is an optional at-most-one
    designation over a forecasts-only fan, not a one-of-N requirement).
    """
    add_value_list_manual(db, [
        ["dispatch_realization_methods", "within_solve"],
        ["dispatch_realization_methods", "separate_dispatch"],
    ])

    default_val, default_type = to_database("within_solve")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="dispatch_realization_method",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="dispatch_realization_methods",
        description=_V76_DISPATCH_REALIZATION_METHOD_DESCRIPTION,
    )
    # solve_advanced is this param's canonical home in the schema.
    _ensure_parameter_group(db, "solve_advanced", "b56f6f", 87)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="dispatch_realization_method",
        parameter_group_name="solve_advanced",
    )

    # Correct the stale stochastic_branches description (an older param,
    # created pre-v76 — line ~160) to the corrected stochastics model: the
    # fan is a probability-weighted forecasts-only set and 'realized' is an
    # optional designation (at most one per start_time), NOT a one-of-N
    # requirement.  Any DB upgrading to v76 passes through this step, so it
    # is the natural place to refresh the user-facing text.  Idempotent.
    if _pdef_or_none(db, "solve", "stochastic_branches") is not None:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="stochastic_branches",
            description=_V76_STOCHASTIC_BRANCHES_DESCRIPTION,
        )

    _commit_step(db,
        "v76: added solve.dispatch_realization_method "
        "(dispatch_realization_methods: within_solve/separate_dispatch, "
        "default within_solve); 2-stage seed of a future multi-stage market "
        "structure. 'separate_dispatch' accepted (W4b) for a single "
        "top-level single_solve — engine auto-spawns the "
        "<solve>__realized_dispatch sibling."
    )


_V70_MAX_FLOW_UNCONSTRAINED_DESCRIPTION = (
    "[MW] Upper bound assigned to LP variables that have no other cap "
    "(invest_no_limit / invest_retire_no_limit capacity; flows through "
    "edges whose conversion_flow_coeff is zero; infinite-capacity "
    "commodity tiers).  Keep large enough not to constrain the physical "
    "solution but small enough to avoid numerical issues (default "
    "1,000,000)."
)
_V70_CAPACITY_MAX_COEFF_DESCRIPTION = (
    "[factor, default 1.0] Fraction of the unit's capacity (existing + "
    "invested - retired) available to this edge's upper cap (maxToSink / "
    "maxFromSource / ramp). 0 is a zero cap; to remove the edge from the "
    "capacity constraints altogether use conversion_flow_coeff = 0. For "
    "extraction CHP set to 1.0 on each output so each can reach full "
    "capacity when the other drops."
)
_V70_INPUT_SHARE_MAX_DESCRIPTION = (
    "[factor, default 1.0] Largest share of the unit's full-load input "
    "energy (flow x conversion_flow_coeff) that this input may supply, "
    "relative to the unit's capacity (existing + invested - retired) "
    "times availability. 1 = this input alone can run the unit at full "
    "output (no effective limit); e.g. two inputs with 0.6 each can each "
    "supply at most 60% of the full-load input, so full output needs both. "
    "A lower-energy input is handled by its conversion_flow_coeff, not "
    "here. For a unit without outputs: fraction of the unit's capacity "
    "available to this input. Not applied to inputs with "
    "conversion_flow_coeff = 0."
)
_V70_INPUT_SHARE_MIN_DESCRIPTION = (
    "[factor, default 0] While the unit is running, the smallest share of "
    "the unit's current input energy (flow x conversion_flow_coeff) that "
    "this input must supply. 0 = no minimum. The shares of a unit's inputs "
    "may not sum above 1, and a share may not exceed the input's "
    "input_share_max. Setting it on every input with shares summing to 1 "
    "fixes the blend. Not applied to inputs with conversion_flow_coeff = 0."
)
_V70_CAPACITY_MIN_COEFF_DESCRIPTION = (
    "[factor, default 0] Optional extra floor on this output of a unit "
    "with several outputs: while the unit is online, this output must be at "
    "least capacity_min_coeff x min_load x the online capacity. The unit's "
    "minimum load (min_load) applies once, to the sum of its outputs, and "
    "is not scaled by this coefficient. 0 = no separate floor on this "
    "output. Range 0..1."
)

_V70_PENALTY_RESERVE_DEFAULT = 5000.0
_V70_PENALTY_RESERVE_DESCRIPTION = (
    "[CUR/MW] Penalty for each MW of reserve shortfall per hour (the "
    "reserve requirement of the group is soft). Constant. Default 5000 - "
    "applied to every reserve group that has no value of its own."
)
_V70_CONNECTION_DELAY_DESCRIPTION = (
    "[hours] Not yet applied to connections (planned): a connection delay "
    "is ignored (the connection is modelled as if no delay were set) and "
    "the model warns about it. Model a delayed flow with a unit. Intended "
    "meaning: a time delay "
    "between the input node and the output node, either a constant time "
    "difference in hours or a map of time differences (index: time "
    "difference in hours, value: weight; the weights should sum to 1)."
)


def _numeric_elements(parsed) -> list[float]:
    """Numeric elements of a parsed value (float, or Map / TimeSeries
    values, recursively); ``[]`` for non-numeric values."""
    if parsed is None or isinstance(parsed, (str, bool)):
        return []
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    values = getattr(parsed, "values", None)
    if values is None:
        return []
    out: list[float] = []
    for v in values:
        out.extend(_numeric_elements(v))
    return out


def _migrate_v70_capacity_coefficient_descriptions(db) -> None:
    """Capacity coefficients, input shares and the per-unit minimum load
    (v69 -> v70; the "input-share step").

    * ``model.max_flow_for_unconstrained_variables`` said it bounds "flows
      through edges whose capacity_max_coeff is zero".  That was an
      artefact of the v35 rename (the old ``coefficient`` was the flow
      coefficient): the uncapped edges are the ``conversion_flow_coeff = 0``
      ones, while ``capacity_max_coeff = 0`` is a zero cap.
    * ``unit__outputNode.capacity_max_coeff``: the description now states
      that it scales the whole capacity (existing + invested - retired) and
      that 0 is a zero cap.
    * ``unit__inputNode.capacity_max_coeff`` -> ``input_share_max`` (pure
      rename, values move unchanged): the largest share of the unit's
      full-load input energy the input may supply.
    * ``unit__inputNode.capacity_min_coeff`` -> ``input_share_min``, default
      0, and EVERY stored value is deleted (listed on stdout).  The 4.x
      engine never applied them, their old meaning (a multiplier on the
      GLPK-era input-side min-load floor, last honoured by v3.32) is
      unrelated to the new mixing share (the smallest share of the unit's
      current input the input must supply), and a carried 1.0 would pin a
      100 % share.
    * ``unit__outputNode.capacity_min_coeff``: the minimum load is now once
      per unit on the sum of its outputs, without a coefficient; this
      parameter becomes an optional extra per-output floor, default 1 -> 0.
      Stored values outside [0, 1] are deleted (the v36 backfill copied old
      flow coefficients such as 2 or -0.025 there; they were never floor
      intents), and so are the v36 leftovers inside [0, 1] (stored
      ``capacity_min_coeff == capacity_max_coeff != 1`` in one
      alternative — the same backfill).  Both deletions are listed, and the
      units whose remaining values now act differently are listed as a
      notice.

    * ``reserve__upDown__group.penalty_reserve`` gets back its default
      5000 [CUR/MW per hour of shortfall] (v18 set it, as did the GMPL
      ``reserveParam_defaults``; v56 cleared it because the 4.x engine read
      explicit values only, leaving the shortfall unpenalised).  The engine
      now applies the default to every reserve group without a value.
    * ``connection.delay``: the description says it is not applied (the
      engine delays unit flows only; a connection delay is ignored with a
      warning).

    Idempotent: a definition is renamed only when the old one exists and
    the new one does not; the input-side deletion, the v36-leftover
    deletion and the notices run only in that (pre-step) case; the
    out-of-range deletion and the definition defaults / descriptions are
    no-ops once applied.
    """
    db.add_update_item(
        "parameter_definition",
        entity_class_name="model",
        name="max_flow_for_unconstrained_variables",
        description=_V70_MAX_FLOW_UNCONSTRAINED_DESCRIPTION,
    )
    db.add_update_item(
        "parameter_definition",
        entity_class_name="unit__outputNode",
        name="capacity_max_coeff",
        description=_V70_CAPACITY_MAX_COEFF_DESCRIPTION,
    )
    zero_val, zero_type = to_database(0.0)
    pre_step = (_pdef_or_none(db, "unit__inputNode", "capacity_min_coeff")
                is not None
                and _pdef_or_none(db, "unit__inputNode", "input_share_min")
                is None)

    # ── unit__outputNode.capacity_min_coeff: per-output floor, default 0 ─
    # Before the input-side rename: the v36 leftovers are recognised only
    # while ``pre_step`` holds, and a large removal commits in batches,
    # so a rerun after an interruption must still see the old input names.
    out_min = _pdef_or_none(db, "unit__outputNode", "capacity_min_coeff")
    if out_min:
        db.update_parameter_definition(
            id=out_min["id"], default_value=zero_val, default_type=zero_type,
            description=_V70_CAPACITY_MIN_COEFF_DESCRIPTION)
        out_vals = list(db.find_parameter_values(
            entity_class_name="unit__outputNode",
            parameter_definition_name="capacity_min_coeff"))
        out_of_range = [pv for pv in out_vals
                        if any(v < 0.0 or v > 1.0 for v in
                               _numeric_elements(pv["parsed_value"]))]
        leftovers = []
        if pre_step:
            cmax = {(tuple(pv["entity_byname"]), pv["alternative_name"]):
                    pv["parsed_value"]
                    for pv in db.find_parameter_values(
                        entity_class_name="unit__outputNode",
                        parameter_definition_name="capacity_max_coeff")}
            bad_ids = {pv["id"] for pv in out_of_range}
            for pv in out_vals:
                if pv["id"] in bad_ids:
                    continue
                v = pv["parsed_value"]
                vmax = cmax.get((tuple(pv["entity_byname"]),
                                 pv["alternative_name"]))
                if (isinstance(v, float) and isinstance(vmax, float)
                        and v == vmax and v != 1.0):
                    leftovers.append(pv)
        if out_of_range:
            print("v70: deleted unit__outputNode capacity_min_coeff values "
                  "outside [0, 1] (now an optional per-output floor; these "
                  "were copied from old flow coefficients):")
            for pv in out_of_range:
                print("  " + _describe_pv(pv))
        if leftovers:
            print("v70: deleted unit__outputNode capacity_min_coeff values "
                  "equal to capacity_max_coeff (left by the v36 coefficient "
                  "backfill, not a floor intent):")
            for pv in leftovers:
                print("  " + _describe_pv(pv))
        removed = out_of_range + leftovers
        if removed:
            _remove_items_bounded(db, "parameter_value", [pv["id"] for pv in removed],
                                  "v70: removed unit__outputNode capacity_min_coeff values")
        if pre_step:
            _print_v70_min_load_notices(
                db, [pv for pv in out_vals
                     if pv["id"] not in {r["id"] for r in removed}])
    # ── unit__inputNode: rename; input_share_min loses its old values ──
    if pre_step:
        old_vals = list(db.find_parameter_values(
            entity_class_name="unit__inputNode",
            parameter_definition_name="capacity_min_coeff"))
        if old_vals:
            print("v70: deleted the old input-side capacity_min_coeff "
                  "values (now input_share_min, a different meaning; "
                  "re-author if needed):")
            for pv in old_vals:
                print("  " + _describe_pv(pv))
            _remove_items_bounded(db, "parameter_value", [pv["id"] for pv in old_vals],
                                  "v70: removed the unit__inputNode capacity_min_coeff values")
    for old_name, new_name, description, default in (
        ("capacity_max_coeff", "input_share_max",
         _V70_INPUT_SHARE_MAX_DESCRIPTION, None),
        ("capacity_min_coeff", "input_share_min",
         _V70_INPUT_SHARE_MIN_DESCRIPTION, (zero_val, zero_type)),
    ):
        old = _pdef_or_none(db, "unit__inputNode", old_name)
        new = _pdef_or_none(db, "unit__inputNode", new_name)
        fields = {"description": description}
        if default is not None:
            fields["default_value"], fields["default_type"] = default
        if old and not new:
            db.update_parameter_definition(id=old["id"], name=new_name,
                                           **fields)
        elif new:
            db.update_parameter_definition(id=new["id"], **fields)

    # ── reserve__upDown__group.penalty_reserve: default 5000 restored ──
    # (v18 set it, v56 cleared it because the engine ignored it; the
    # engine now applies it to every group without an authored value).
    pen = _pdef_or_none(db, "reserve__upDown__group", "penalty_reserve")
    if pen:
        pen_val, pen_type = to_database(_V70_PENALTY_RESERVE_DEFAULT)
        db.update_parameter_definition(
            id=pen["id"], default_value=pen_val, default_type=pen_type,
            description=_V70_PENALTY_RESERVE_DESCRIPTION)

    # ── connection.delay: not applied by the engine; say so ──
    conn_delay = _pdef_or_none(db, "connection", "delay")
    if conn_delay:
        db.update_parameter_definition(
            id=conn_delay["id"],
            description=_V70_CONNECTION_DELAY_DESCRIPTION)

    _commit_step(db,
        "v70: capacity_max_coeff = 0 is a zero cap; uncapped edges are "
        "conversion_flow_coeff = 0; unit__inputNode.capacity_max_coeff -> "
        "input_share_max, unit__inputNode.capacity_min_coeff -> "
        "input_share_min (old values deleted, default 0); "
        "unit__outputNode.capacity_min_coeff is a per-output floor "
        "(default 0, out-of-range and v36-leftover values deleted); "
        "reserve__upDown__group.penalty_reserve default 5000; "
        "connection.delay described as not applied."
    )


def _print_v70_min_load_notices(db, remaining) -> None:
    """List units whose kept output ``capacity_min_coeff`` values behave
    differently under the per-unit minimum load."""
    outputs: dict[str, set[str]] = {}
    for ent in db.find_entities(entity_class_name="unit__outputNode"):
        outputs.setdefault(ent["entity_byname"][0], set()).add(
            ent["entity_byname"][1])
    by_unit: dict[str, list[float]] = {}
    for pv in remaining:
        vals = _numeric_elements(pv["parsed_value"])
        if vals:
            by_unit.setdefault(pv["entity_byname"][0], []).extend(vals)
    lowered = sorted(u for u, v in by_unit.items() if max(v) < 1.0)
    floors = sorted(u for u, v in by_unit.items()
                    if len(outputs.get(u, ())) >= 2 and max(v) > 0.0)
    if lowered:
        print("v70: capacity_min_coeff no longer lowers or switches off a "
              "unit's minimum load (it now applies once per unit, to the "
              "sum of its outputs); lower min_load instead (0 switches it "
              "off) for: " + ", ".join(lowered))
    if floors:
        print("v70: capacity_min_coeff > 0 on an output of a multi-output "
              "unit is now a separate floor on that output "
              "(output >= coefficient x min_load x online capacity) for: "
              + ", ".join(floors))
