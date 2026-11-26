"""Data versions 57-69."""
from spinedb_api import from_database, SpineDBAPIError, to_database

from flextool.update_flextool.db_migration.common import (
    _already_applied,
    _attach_list_defaults,
    _commit_step,
    _detach_list_defaults,
    _remove_items_bounded,
    _rename_or_drop_parameter_definition,
    _unless_renamed,
    add_value_list_manual,
    remove_parameters_manual,
)


def _migrate_v57(db) -> None:
    # Clarify the descriptions of the three parameters that
    # together drive timeslice weighting / annualisation:
    # ``node.inflow_method`` (how the annual-energy scaling
    # uses each timestep's share of the year),
    # ``timeset.timeset_duration`` (defines the *selection*
    # of timesteps) and ``timeset.timeset_weights`` (defines
    # the *representativeness* — each timestep's share of the
    # year, used identically for inflow scaling, energy
    # reporting and the operating-cost objective).  Wording
    # per spec ``specs/timeslice_weight_alignment.md`` section
    # B2.  Pure description text — engine semantics, default
    # values and value-list bindings are untouched.
    db.update_item(
        "parameter_definition",
        entity_class_name="node",
        name="inflow_method",
        description="Choice how to treat inflow time series. Empty defaults to 'use_original', which does not scale the time series. 'no_inflow' ignores the inflow time series. 'scale_to_annual_flow' scales the inflow time series by a single factor (shape preserved) so its annual energy equals 'annual_flow'. The annualisation uses each timestep's share of the year: for a full or evenly sampled timeline this is 8760/'hours_in_solve' (every step weighted equally); for a representative timeset with 'timeset_weights', each timestep is weighted by its declared share of the year, so a biased sample of timesteps still annualises correctly. The same weighting is used for energy reporting and operating-cost accounting, so the energy the model is scaled to serve, reports, and costs all coincide. If 'timeset_weights' do not reflect the true representativeness of your selected timesteps, the scaling factor - and therefore the absolute demand level - shifts accordingly; see the per-node 'f' diagnostic (inflow_scaling_diagnostics.csv). 'scale_in_proprotion' calculates a scaling factor by dividing 'annual_flow' with the sum of time series inflow (after it has been annualized using 'timeline_duration_in_years'). (With a representative timeset, the annual-energy match uses 'timeset_weights' as above.) 'scale_to_annual_and_peak_flow' scales the time series to match the 'annual_flow' target while transforming the time series to match the highest load with the 'peak_inflow'. (With a representative timeset, the annual-energy match uses 'timeset_weights' as above; the peak match stays unweighted.)",
    )
    db.update_item(
        "parameter_definition",
        entity_class_name="timeset",
        name="timeset_duration",
        description="Index: name of the the timestep that starts the timeset, value: duration of the block in timesteps. Defines the *selection*: which timesteps are in the solve and how many (this sets 'hours_in_solve'). Representativeness - each timestep's share of the year - is set separately by 'timeset_weights'.",
    )
    db.update_item(
        "parameter_definition",
        entity_class_name="timeset",
        name="timeset_weights",
        description="Per-timestep weight map (index: timestep name, value: float) defining *representativeness*: each timestep's share of the year. Drives the annualisation of demand, energy, and cost alike (the inflow scaling, the annual energy outputs, and the operating-cost objective all use this same weight), so the energy the model is scaled to serve, reports, and costs all coincide. Use for non-RP models where timesteps represent unequal fractions of the year (e.g. seasonal yearsplit on a coarse timeslice structure). Weights are normalized per period to sum to 1 and then scaled by the number of active timesteps so that uniform input reproduces the default (weight = 1 per step). Must not be combined with representative_period_weights on the same timeset.",
    )
    _commit_step(db, "Clarify inflow_method/timeset_duration/timeset_weights descriptions for timeslice weighting")


def _migrate_v59(db) -> None:
    default_val, default_type = to_database(0.0001)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="model", name="small_number_threshold",
        default_value=default_val, default_type=default_type,
        parameter_type_list=("float",),
        description="Coefficients and right-hand-side terms whose "
        "absolute value is below this are treated as 0 when "
        "building the optimisation problem. Reduces the numerical "
        "range (conditioning) of the LP. Default 0.0001.",
    )
    _commit_step(db, "Added model.small_number_threshold parameter")


def _migrate_v61(db) -> None:
    db.add_update_item(
        "parameter_definition",
        entity_class_name="timeset",
        name="representative_period_weights",
        parameter_type_list=("2d_map",),
        parameter_group_name="solve_advanced",
        description=(
            "Representative-period weighting map (rank-2 nested "
            "Map: base period start timestep -> representative "
            "period start timestep -> weight).  Created by the "
            "representative-periods preprocess tool; declares how "
            "much each representative period stands in for each "
            "base period.  Its presence enables RP-aware "
            "storage-binding ('*_blended_weights'); its absence "
            "silently downgrades those methods to the non-RP "
            "behaviour.  Mutually exclusive with 'timeset_weights' "
            "on the same timeset."
        ),
    )
    _commit_step(db, "Added timeset.representative_period_weights parameter")


def _migrate_v60_solve_decomposition(db) -> None:
    """Add per-solve decomposition scheme selection + Lagrangian knobs.

    Rationale
    ---------
    Decomposition was previously an all-or-nothing CLI choice
    (``--decomposition lagrangian``) that bypassed the normal
    orchestrator.  This migration moves the choice into the database so
    it can be made *per solve*: a single solve chain can run a
    Lagrangian investment solve followed by a monolithic dispatch solve.

    Parameters added to entity_class ``solve``:

    * ``decomposition`` — enum string on the new ``decomposition_schemes``
      value list (``none``, ``lagrangian``).  Default ``none`` preserves
      existing behaviour.  When ``lagrangian`` the orchestrator runs that
      solve through the Lagrangian driver over the group members whose
      ``decomposition_method`` is ``lagrangian_region`` (the
      ``decomp_<REG>`` groups).  Note this is *distinct* from the
      group-level ``decomposition_methods`` value list (``none`` /
      ``lagrangian_region``) added in v51: the group parameter declares
      *which groups are regions*, the solve parameter declares *whether
      this solve decomposes*.
    * ``lagrangian_alpha`` — float, default 0.1.  Subgradient base step
      size.
    * ``lagrangian_max_iter`` — float, default 80.  Max outer iterations.
    * ``lagrangian_tolerance`` — float, default 1.0.  Tail-averaged
      coupling-imbalance convergence threshold.

    These three replace the removed ``--lagrangian-*`` CLI flags, giving
    the tuning knobs a per-solve home.

    Idempotency
    -----------
    The PLEXOS-to-FlexTool writer self-defines and populates
    ``solve.<name>.decomposition = lagrangian`` (under a ``lagrangian``
    alternative) using the same value-list / value names.  ``import_data``
    for value lists only adds what is missing and ``add_update_item``
    updates an existing definition in place, so re-running against a DB
    that already carries the writer's definition converges to the same
    schema without duplicate value-list rows or definitions.
    """

    # --- Value list for solve.decomposition -----------------------------
    add_value_list_manual(db, _unless_renamed(db, 60, "decomposition_schemes", "lagrangian", "benders"))

    # --- solve.decomposition --------------------------------------------
    default_val, default_type = to_database("none")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="decomposition",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="decomposition_schemes",
        parameter_type_list=("str",),
        description=(
            "Decomposition scheme for this solve. 'none' (default) "
            "solves monolithically. 'lagrangian' runs the solve through "
            "the Lagrangian region-decomposition driver over the groups "
            "whose group.decomposition_method is 'lagrangian_region' "
            "(requires at least two such region groups). The choice is "
            "per solve, so an investment solve can decompose while a "
            "later dispatch solve in the same chain stays monolithic."
        ),
    )

    # --- solve.lagrangian_alpha -----------------------------------------
    default_val, default_type = to_database(0.1)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="lagrangian_alpha",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Lagrangian decomposition: subgradient base step size for "
            "the multiplier update. Only used when decomposition = "
            "'lagrangian'. Default 0.1."
        ),
    )

    # --- solve.lagrangian_max_iter --------------------------------------
    default_val, default_type = to_database(80.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="lagrangian_max_iter",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Lagrangian decomposition: maximum number of outer "
            "(subgradient) iterations. Only used when decomposition = "
            "'lagrangian'. Default 80."
        ),
    )

    # --- solve.lagrangian_tolerance -------------------------------------
    default_val, default_type = to_database(1.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="lagrangian_tolerance",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Lagrangian decomposition: tail-averaged coupling-imbalance "
            "threshold for convergence of the outer loop. Only used when "
            "decomposition = 'lagrangian'. Default 1.0."
        ),
    )

    # Attach all four to the solve_advanced parameter_group when it
    # exists (v44+) — same heading as other experimental opt-in solve
    # controls.
    if db.item(db.mapped_table("parameter_group"), name="solve_advanced") is not None:
        for _pname in (
            "decomposition",
            "lagrangian_alpha",
            "lagrangian_max_iter",
            "lagrangian_tolerance",
        ):
            db.add_update_item(
                "parameter_definition",
                entity_class_name="solve",
                name=_pname,
                parameter_group_name="solve_advanced",
            )

    try:
        _commit_step(db,
            "v60: added solve.decomposition (decomposition_schemes value "
            "list) + solve.lagrangian_alpha/max_iter/tolerance for "
            "per-solve Lagrangian decomposition."
        )
    except SpineDBAPIError:
        pass


def _migrate_v62_benders_rename(db) -> None:
    """Rename the decomposition method from Lagrangian to Benders.

    Rationale
    ---------
    The dual-subgradient Lagrangian region decomposition (v51/v60) is
    replaced wholesale by a Benders decomposition (master trade layer +
    per-region recourse subproblems with cut append / warm restart).
    Benders is a different algorithm — a master/subproblem scheme that
    produces a VALID lower bound and converges on the optimality gap —
    so the subgradient-specific knobs (``lagrangian_alpha`` step size,
    the subgradient ``max_iter`` / tail-averaged-imbalance ``tolerance``)
    have no meaning under Benders and are dropped; Benders gets its own
    iteration cap + relative-gap tolerance.

    Schema changes (entity_class ``solve`` / ``group``):

    * Rename the ``decomposition_schemes`` value-list member
      ``lagrangian`` → ``benders`` (and every authored
      ``solve.decomposition = lagrangian`` value).
    * Rename the ``decomposition_methods`` value-list member
      ``lagrangian_region`` → ``benders_regional`` (and every authored
      ``group.decomposition_method = lagrangian_region`` value).
    * DROP ``solve.lagrangian_alpha`` / ``lagrangian_max_iter`` /
      ``lagrangian_tolerance`` (subgradient knobs).
    * ADD ``solve.benders_max_iter`` (default 50) +
      ``solve.benders_tolerance`` (default 1e-3) — Benders has no
      subgradient step, so there is no ``alpha`` analogue.
    * Update the ``decomposition`` / ``decomposition_method``
      descriptions to describe Benders.

    Idempotency
    -----------
    list_value members are renamed in place by id (the
    parameter_definition FK and dependent parameter_value rows are
    unaffected); the parameter_value rewrite is keyed by parsed value so
    a re-run over an already-migrated DB finds no ``lagrangian`` /
    ``lagrangian_region`` rows to rewrite.  ``add_update_item`` /
    ``remove_parameters_manual`` converge to the same definitions
    whether or not the legacy ones are present.
    """

    # --- 1. Rename solve.decomposition value 'lagrangian' -> 'benders' ----
    old_val_bytes, _ = to_database("lagrangian")
    new_val_bytes, new_val_type = to_database("benders")
    # Rewrite authored parameter_values first (while the old list member
    # still exists), then swap the list_value member.
    for pv in list(db.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="decomposition",
    )):
        if pv["parsed_value"] == "lagrangian":
            db.add_update_item(
                "parameter_value",
                entity_class_name="solve",
                entity_byname=pv["entity_byname"],
                parameter_definition_name="decomposition",
                alternative_name=pv["alternative_name"],
                value=new_val_bytes,
                type=new_val_type,
            )
    pending = _detach_list_defaults(db, "decomposition_schemes", {"lagrangian": "benders"})
    for lv in db.find_list_values(
        parameter_value_list_name="decomposition_schemes",
    ):
        if lv["value"] == old_val_bytes:
            db.update_item(
                "list_value", id=lv["id"],
                value=new_val_bytes, type=new_val_type,
            )
            break
    _attach_list_defaults(db, pending)

    # --- 2. Rename group method 'lagrangian_region' -> 'benders_regional' -
    old_meth_bytes, _ = to_database("lagrangian_region")
    new_meth_bytes, new_meth_type = to_database("benders_regional")
    for pv in list(db.find_parameter_values(
        entity_class_name="group",
        parameter_definition_name="decomposition_method",
    )):
        if pv["parsed_value"] == "lagrangian_region":
            db.add_update_item(
                "parameter_value",
                entity_class_name="group",
                entity_byname=pv["entity_byname"],
                parameter_definition_name="decomposition_method",
                alternative_name=pv["alternative_name"],
                value=new_meth_bytes,
                type=new_meth_type,
            )
    pending = _detach_list_defaults(db, "decomposition_methods", {"lagrangian_region": "benders_regional"})
    for lv in db.find_list_values(
        parameter_value_list_name="decomposition_methods",
    ):
        if lv["value"] == old_meth_bytes:
            db.update_item(
                "list_value", id=lv["id"],
                value=new_meth_bytes, type=new_meth_type,
            )
            break
    _attach_list_defaults(db, pending)

    try:
        _commit_step(db,
            "v62: renamed decomposition value lagrangian -> benders and "
            "group method lagrangian_region -> benders_regional."
        )
    except SpineDBAPIError:
        pass

    # --- 3. Drop the three subgradient knobs ------------------------------
    # remove_parameters_manual commits internally (cascades the values).
    remove_parameters_manual(db, [
        ["solve", "lagrangian_alpha"],
        ["solve", "lagrangian_max_iter"],
        ["solve", "lagrangian_tolerance"],
    ])

    has_solve_advanced = (
        db.item(db.mapped_table("parameter_group"), name="solve_advanced")
        is not None
    )

    # --- 4. Add the Benders knobs -----------------------------------------
    default_val, default_type = to_database(50.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="benders_max_iter",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Benders decomposition: maximum number of outer (master/"
            "subproblem) iterations. Only used when decomposition = "
            "'benders'. Default 50."
        ),
    )
    default_val, default_type = to_database(1e-3)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="benders_tolerance",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Benders decomposition: relative optimality-gap tolerance "
            "(best_UB - LB)/|best_UB| for convergence of the outer loop. "
            "Only used when decomposition = 'benders'. Default 0.001."
        ),
    )
    if has_solve_advanced:
        for _pname in ("benders_max_iter", "benders_tolerance"):
            db.add_update_item(
                "parameter_definition",
                entity_class_name="solve",
                name=_pname,
                parameter_group_name="solve_advanced",
            )

    # --- 5. Refresh the routing-parameter descriptions for Benders --------
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="decomposition",
        description=(
            "Decomposition scheme for this solve. 'none' (default) "
            "solves monolithically. 'benders' runs the solve through the "
            "Benders region-decomposition driver (a master trade layer "
            "coordinating per-region recourse subproblems via appended "
            "optimality cuts, producing a valid lower bound) over the "
            "groups whose group.decomposition_method is 'benders_regional' "
            "(requires at least two such region groups). The choice is "
            "per solve, so an investment solve can decompose while a later "
            "dispatch solve in the same chain stays monolithic."
        ),
    )
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="decomposition_method",
        description=(
            "Decomposition strategy to apply to this group. "
            "Currently supported: 'none' (no decomposition — default), "
            "'benders_regional' (group is solved as an independent region "
            "in the Benders master/subproblem decomposition, with "
            "shared-commodity trade coupling handled by the master)."
        ),
    )

    try:
        _commit_step(db,
            "v62: dropped solve.lagrangian_alpha/max_iter/tolerance; added "
            "solve.benders_max_iter (50) + benders_tolerance (1e-3); "
            "refreshed decomposition/decomposition_method descriptions for "
            "Benders."
        )
    except SpineDBAPIError:
        pass


def _migrate_v63_add_in_out_weight(db) -> None:
    """Add the ``solve.benders_in_out_weight`` knob (v62 -> v63).

    Rationale
    ---------
    Promotes the previously machine-local env knob
    ``FLEXTOOL_BENDERS_IN_OUT_WEIGHT`` to a proper per-solve DB parameter.
    It is the in-out separation stabilization weight ``lambda`` in
    ``f_sep = lambda*centre + (1-lambda)*f_out`` (Ben-Ameur & Neto 2007),
    fed one InOutStabilizer per region.  ``0.0`` (default) = OFF = exact
    Benders (byte-identical); values in ``(0, 1)`` turn it on, larger =
    more stabilisation.  Only used when ``decomposition = 'benders'``.

    Mirrors the "Add the Benders knobs" section of
    :func:`_migrate_v62_benders_rename`: a single ``add_update_item``
    (an upsert, so re-run-safe) plus the ``solve_advanced`` group
    assignment when that group exists.
    """
    has_solve_advanced = (
        db.item(db.mapped_table("parameter_group"), name="solve_advanced")
        is not None
    )

    default_val, default_type = to_database(0.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="benders_in_out_weight",
        default_value=default_val,
        default_type=default_type,
        parameter_type_list=("float",),
        description=(
            "Benders decomposition: in-out separation stabilization weight "
            "(lambda) in f_sep = lambda*centre + (1-lambda)*f_out. 0.0 "
            "(default) = off (exact Benders); values in (0,1) turn it on, "
            "larger = more stabilization. Only used when decomposition = "
            "'benders'. Default 0.0."
        ),
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="benders_in_out_weight",
            parameter_group_name="solve_advanced",
        )

    _commit_step(db,
        "v63: added solve.benders_in_out_weight (in-out separation weight, "
        "default 0.0)."
    )


def _migrate_v64_add_solve_scaling(db) -> None:
    """Add the ``solve.scaling`` autoscaler-mode knob (v63 -> v64).

    Rationale
    ---------
    The numerical autoscaler mode was previously a run-time-only control
    (``--scaling`` CLI flag / ``FLEXTOOL_SCALING`` env, resolved
    globally, default ``full``).  This promotes it to a per-solve DB
    parameter so a model can pin the mode that solves it correctly — in
    particular ``basic`` (Layer 1 + Layer 3, no Layer 2) for models
    whose Layer-2 physical-unit rescaling mis-solves (e.g. the H2
    commodity-ladder trade model).  Precedence at run time:
    ``--scaling`` / ``FLEXTOOL_SCALING`` (explicit operator override) >
    ``solve.scaling`` (DB) > schema default ``full``.

    Bound to a new ``scaling_modes`` value list mirroring the CLI's four
    modes.  ``full`` is the default (unchanged behaviour for every
    existing solve).  Grouped under ``solve_advanced`` when that group
    exists, alongside ``solver`` / ``decomposition``.
    """
    add_value_list_manual(db, [
        ["scaling_modes", "off"],
        ["scaling_modes", "solver_only"],
        ["scaling_modes", "basic"],
        ["scaling_modes", "full"],
    ])

    default_val, default_type = to_database("full")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="solve",
        name="scaling",
        default_value=default_val,
        default_type=default_type,
        parameter_value_list_name="scaling_modes",
        description=(
            "Numerical LP-autoscaler mode for this solve. 'full' (default) "
            "runs Layer 1 (detection) + Layer 2 (per-quantity-type "
            "column/row/cost rescaling) + Layer 3 (HiGHS user_*_scale). "
            "'basic' drops Layer 2 (Layer 1 + Layer 3 only); 'solver_only' "
            "leaves scaling to the solver; 'off' disables all scaling "
            "including HiGHS-internal. Use 'basic' (or lower) for models "
            "whose Layer-2 rescaling mis-solves. The run-time --scaling flag "
            "/ FLEXTOOL_SCALING env override this per-solve value."
        ),
    )
    has_solve_advanced = (
        db.item(db.mapped_table("parameter_group"), name="solve_advanced")
        is not None
    )
    if has_solve_advanced:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="solve",
            name="scaling",
            parameter_group_name="solve_advanced",
        )

    _commit_step(db,
        "v64: added solve.scaling (autoscaler mode: off/solver_only/basic/"
        "full, default full)."
    )


def _migrate_v65_energy_and_capacity_margin_methods(db) -> None:
    """Add ``node.energy_margin`` + method knobs and convert
    ``group.has_capacity_margin`` to ``group.capacity_margin_method`` (v64 -> v65).

    Two related additions, kept in one migration to minimise version churn:

    1. **``node.energy_margin`` (+ ``node.energy_margin_method``).** A new
       investment-stage lever: multiply a node's inflow (demand) in the
       *investment* solve only, to offset representative-period VRE
       optimism (the rep-period mean VRE availability exceeds the true
       annual mean, so the invest stage underbuilds and the full-year
       dispatch eats an unserved-energy penalty).  ``energy_margin_method``
       selects behaviour: ``none`` (default, off, byte-identical) or
       ``inflow_multiplier`` (apply the ``energy_margin`` factor).  The
       enum leaves room for a future ``inflow_adder`` (additive margin)
       without a further schema change.

    2. **``group.capacity_margin_method`` replaces ``has_capacity_margin``.**
       The former yes/no flag becomes a method enum for consistency with
       (1) and to leave room for a future ``automatic`` peak-margin sizing.
       ``has_capacity_margin = yes`` -> ``capacity_margin_method = manual``;
       ``no``/absent -> ``none`` (the default, so no row needed).  The old
       flag definition is then removed.
    """
    # --- 1. energy_margin_method value list + node params -------------
    add_value_list_manual(db, [
        ["energy_margin_methods", "none"],
        ["energy_margin_methods", "inflow_multiplier"],
    ])

    em_default_val, em_default_type = to_database("none")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node",
        name="energy_margin_method",
        default_value=em_default_val,
        default_type=em_default_type,
        parameter_value_list_name="energy_margin_methods",
        description=(
            "How the node's investment-stage energy margin is applied. "
            "'none' (default) = off, no scaling. 'inflow_multiplier' "
            "multiplies this node's inflow by 'energy_margin' in the solve "
            "that carries investment periods only (dispatch solves are "
            "unaffected), to offset representative-period VRE optimism that "
            "would otherwise cause underinvestment. Reserved for a future "
            "'inflow_adder' mode."
        ),
    )

    em_val, em_type = to_database(1.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node",
        name="energy_margin",
        default_value=em_val,
        default_type=em_type,
        description=(
            "[factor] Multiplier applied to this node's inflow in the "
            "investment solve only, when energy_margin_method = "
            "'inflow_multiplier'. "
            "Offsets representative-period VRE optimism (rep-period average "
            "VRE availability exceeds the annual average). 1 = off. e.g. "
            "1.1 builds ~10% more capacity to serve the true annual demand "
            "under the true (lower) annual VRE. Constant or period."
        ),
    )

    # --- 2. capacity_margin_method value list + group param ----------
    add_value_list_manual(db, [
        ["capacity_margin_methods", "none"],
        ["capacity_margin_methods", "manual"],
    ])

    cm_default_val, cm_default_type = to_database("none")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="capacity_margin_method",
        default_value=cm_default_val,
        default_type=cm_default_type,
        parameter_value_list_name="capacity_margin_methods",
        description=(
            "Whether the node group enforces a capacity margin in the "
            "investment mode. 'none' (default) = no constraint. 'manual' "
            "requires the group to have 'capacity_margin' MW of capacity "
            "above the peak net load. Replaces the former has_capacity_margin "
            "yes/no flag; reserved for a future 'automatic' mode."
        ),
    )

    # Assign both new params to the 'investment' parameter group when it
    # exists (mirrors capacity_margin's grouping).
    has_investment_group = bool(
        db.get_item("parameter_group", name="investment")
    )
    if has_investment_group:
        for cls, pname in (
            ("node", "energy_margin_method"),
            ("node", "energy_margin"),
            ("group", "capacity_margin_method"),
        ):
            db.add_update_item(
                "parameter_definition",
                entity_class_name=cls,
                name=pname,
                parameter_group_name="investment",
            )

    # --- 3. convert has_capacity_margin values -> capacity_margin_method
    for pv in list(db.find_parameter_values(
        entity_class_name="group",
        parameter_definition_name="has_capacity_margin",
    )):
        parsed = pv.get("parsed_value")
        if parsed is None or str(parsed).strip().lower() != "yes":
            # 'no'/absent maps to the default 'none' — no row needed.
            continue
        new_val, new_type = to_database("manual")
        db.add_update_item(
            "parameter_value",
            entity_class_name="group",
            entity_byname=pv["entity_byname"],
            parameter_definition_name="capacity_margin_method",
            alternative_name=pv["alternative_name"],
            value=new_val,
            type=new_type,
        )

    # --- 4. drop the old has_capacity_margin definition --------------
    has_cm_def = db.get_item("parameter_definition", entity_class_name="group", name="has_capacity_margin")
    if has_cm_def:
        db.remove_parameter_definition(id=has_cm_def["id"])
    else:
        _already_applied(65, "group.has_capacity_margin is removed")

    _commit_step(db,
        "v65: added node.energy_margin (+ energy_margin_method) and "
        "converted group.has_capacity_margin -> capacity_margin_method "
        "(none/manual)."
    )


def _migrate_v66_energy_margin_adder(db) -> None:
    """Rename ``node.energy_margin`` -> ``energy_margin_multiplier`` and add
    the additive counterpart ``node.energy_margin_adder`` (v65 -> v66).

    The v65 energy-margin lever was a single multiplicative factor
    (``energy_margin`` applied via ``energy_margin_method =
    'inflow_multiplier'``).  v66 makes the two modes symmetric:

    1. The multiplicative parameter is renamed to the self-documenting
       ``energy_margin_multiplier`` (a plain definition rename preserves
       any FK'd ``parameter_value`` rows, so no value migration is needed).
    2. A new additive lever ``energy_margin_adder`` [MWh] is added, applied
       via the freshly-live ``energy_margin_methods`` value ``inflow_adder``
       — extra demand (negative inflow) added to the node in the investment
       solve only, raising adequacy headroom without changing dispatch
       demand.  The ``inflow_adder`` enum value, reserved-but-unused in v65,
       is now populated and the ``energy_margin_method`` description is
       updated to reflect that it is live.
    """
    # --- 1. rename energy_margin -> energy_margin_multiplier ----------
    _rename_or_drop_parameter_definition(
        db, "node", "energy_margin", "energy_margin_multiplier",
        description=(
            "[factor] Multiplier applied to this node's inflow in the "
            "investment solve only, when energy_margin_method = "
            "'inflow_multiplier'. "
            "Offsets representative-period VRE optimism (rep-period average "
            "VRE availability exceeds the annual average). 1 = off. e.g. "
            "1.1 builds ~10% more capacity to serve the true annual demand "
            "under the true (lower) annual VRE. Constant or period."
        ),
    )

    # --- 2. populate the inflow_adder enum value ----------------------
    add_value_list_manual(db, [["energy_margin_methods", "inflow_adder"]])

    # --- 3. energy_margin_method description now that inflow_adder is live
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node",
        name="energy_margin_method",
        description=(
            "How the node's investment-stage energy margin is applied. "
            "'none' (default) = off, no scaling. 'inflow_multiplier' "
            "multiplies this node's inflow by 'energy_margin_multiplier' in "
            "the solve that carries investment periods only (dispatch solves "
            "are unaffected). 'inflow_adder' instead adds "
            "'energy_margin_adder' MWh of extra demand (negative inflow) in "
            "that same investment solve. Both offset representative-period "
            "VRE optimism that would otherwise cause underinvestment."
        ),
    )

    # --- 4. add node.energy_margin_adder ------------------------------
    ema_val, ema_type = to_database(0.0)
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node",
        name="energy_margin_adder",
        default_value=ema_val,
        default_type=ema_type,
        description=(
            "[MWh] Extra demand (negative inflow) added to this node in the "
            "investment solve only, when energy_margin_method = "
            "'inflow_adder'. Raises adequacy headroom without changing "
            "dispatch demand. 0 = off. Constant or period."
        ),
    )

    # Assign the new param to the 'investment' parameter group when it
    # exists (mirrors the v65 energy_margin grouping).
    has_investment_group = bool(
        db.get_item("parameter_group", name="investment")
    )
    if has_investment_group:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="node",
            name="energy_margin_adder",
            parameter_group_name="investment",
        )

    _commit_step(db,
        "v66: renamed node.energy_margin -> energy_margin_multiplier; "
        "added node.energy_margin_adder + energy_margin_methods.inflow_adder."
    )


def _migrate_v67_node_penalty_method(db) -> None:
    """Add ``node.penalty_method`` to toggle balance-penalty slack (v66 -> v67).

    FlexTool nodes carry ``penalty_up`` / ``penalty_down`` slack variables
    on their balance equation: unserved-energy and over-supply slacks that
    let the node balance be violated at a (usually large) cost.  This keeps
    solves feasible, but on adequacy studies that same slack can silently
    absorb a genuine shortfall — the solve reports an expensive but
    "optimal" result instead of the infeasibility that would reveal
    inadequate capacity.

    ``penalty_method`` is a per-node enum (value list ``penalty_methods``):

    * ``regular`` (default, byte-identical to prior behaviour) keeps the
      ``penalty_up`` / ``penalty_down`` slack variables.
    * ``off`` removes those slack variables entirely, turning the node
      balance into a hard equality — the solve becomes infeasible if the
      node cannot balance, forcing true adequacy on the selected nodes.

    The parameter is deliberately left out of any parameter group: a
    penalty toggle is not investment-specific.
    """
    add_value_list_manual(db, [
        ["penalty_methods", "regular"],
        ["penalty_methods", "off"],
    ])

    pm_default_val, pm_default_type = to_database("regular")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node",
        name="penalty_method",
        default_value=pm_default_val,
        default_type=pm_default_type,
        parameter_value_list_name="penalty_methods",
        description=(
            "Whether this node has balance penalty (unserved-energy / "
            "over-supply slack) variables. 'regular' (default) keeps the "
            "penalty_up / penalty_down slack variables that let the node "
            "balance be violated at a cost. 'off' removes those slack "
            "variables entirely, making the node balance a hard equality "
            "(the solve becomes infeasible if the node cannot balance). "
            "Use 'off' only if certain that it will not make the model "
            "infeasible (a real solution exists) - infeasible models can be "
            "very difficult to debug. At the same time, 'off' will reduce "
            "variables and can direct infeasibilities to other, more "
            "desirable, nodes."
        ),
    )

    _commit_step(db,
        "v67: added node.penalty_method (penalty_methods: regular/off); "
        "'off' removes node balance slack (penalty) variables."
    )


def _migrate_v68_rp_group_flag(db) -> None:
    """Add ``group.use_for_representative_periods`` (v67 -> v68).

    A new per-group ``yes_no`` flag that marks a region-group as an
    aggregation unit for net-load representative-period selection.  When a
    group carries ``use_for_representative_periods == 'yes'`` the
    representative-periods preprocessor aggregates that group's member
    nodes' net load (demand minus available VRE) into a single regional
    signal used for clustering; when no group carries the flag the
    selection falls back to per-node net-load signals.

    The flag is consumed only by the representative-periods preprocessor,
    never by the LP model, so there is no engine wiring here — this
    migration lands the schema definition alone.  It reuses the existing
    ``yes_no`` value list (mirroring ``has_inertia`` /
    ``include_stochastics``) and attaches the parameter to the
    ``solve_advanced`` parameter group (consistent with the other
    representative-period parameters such as
    ``representative_period_weights``).
    """
    rp_default_val, rp_default_type = to_database("no")
    db.add_update_item(
        "parameter_definition",
        entity_class_name="group",
        name="use_for_representative_periods",
        default_value=rp_default_val,
        default_type=rp_default_type,
        parameter_value_list_name="yes_no",
        description=(
            "A flag whether this group of nodes is an aggregation unit for "
            "net-load representative-period selection. When 'yes', the "
            "representative-periods preprocessor aggregates the member "
            "nodes' net load (demand minus available VRE) into a single "
            "regional signal used for clustering; when 'no' (default) the "
            "group is ignored by that selection. If no group carries the "
            "flag, the selection falls back to per-node net-load signals."
        ),
    )

    # Attach to the 'solve_advanced' parameter group when it exists
    # (consistent with representative_period_weights).
    has_solve_advanced_group = bool(
        db.get_item("parameter_group", name="solve_advanced")
    )
    if has_solve_advanced_group:
        db.add_update_item(
            "parameter_definition",
            entity_class_name="group",
            name="use_for_representative_periods",
            parameter_group_name="solve_advanced",
        )

    _commit_step(db,
        "v68: added group.use_for_representative_periods (yes_no flag) for "
        "net-load representative-period aggregation-unit selection."
    )


def _migrate_v69_backfill_parameter_groups(db) -> None:
    """Assign parameter groups to two previously ungrouped params (v68 -> v69).

    ``model.small_number_threshold`` (added in v59) and
    ``node.penalty_method`` (added in v67) were introduced without a
    ``parameter_group_name``.  Every parameter must belong to a
    ``parameter_group`` so it is not dropped from group-filtered tabular
    exports (this invariant is guarded by
    ``tests/test_parameter_group_coverage.py``).

    Group assignments follow sibling convention within each class:

    * ``model.small_number_threshold`` -> ``model``.  It is a numerical-
      conditioning knob for building the LP; its closest class sibling,
      ``model.max_flow_for_unconstrained_variables`` (also an LP-
      conditioning knob), sits in ``model``, and every other ``model.*``
      parameter is in ``model`` except ``output_horizon``.
    * ``node.penalty_method`` -> ``basics``.  Its balance-behaviour
      siblings ``node.penalty_up`` / ``node.penalty_down`` / ``node.inflow``
      / ``node.node_type`` are all in ``basics``.

    Idempotent: ``add_update_item`` backfills the group on existing DBs and
    is a no-op on fresh builds that already carry it.  Both target groups
    exist in every FlexTool DB, so no existence guard is needed.
    """
    db.add_update_item(
        "parameter_definition",
        entity_class_name="model",
        name="small_number_threshold",
        parameter_group_name="model",
    )
    db.add_update_item(
        "parameter_definition",
        entity_class_name="node",
        name="penalty_method",
        parameter_group_name="basics",
    )

    _commit_step(db,
        "v69: backfilled parameter groups for model.small_number_threshold "
        "(-> model) and node.penalty_method (-> basics)."
    )


_V58_FLOW_VALUE_PARAMS = ("max_instant_flow", "min_instant_flow", "max_cumulative_flow", "min_cumulative_flow")
_V58_BOOL_FLOW_PARAMS = ("flow_aggregator", "output_flowGroup_indicators")
_V58_ALL_FLOW_PARAMS = _V58_FLOW_VALUE_PARAMS + _V58_BOOL_FLOW_PARAMS
_V58_THREE_DIM = ("group__unit__node", "group__connection__node")
_V58_FLOWGROUP_THREE_DIM = {
    "group__unit__node": "flowGroup__unit__node",
    "group__connection__node": "flowGroup__connection__node",
}
_V58_GROUP_MEMBERSHIP = ("group__node", "group__unit", "group__connection")
_V58_FLOW_LIMIT_DEFS = {
    "max_instant_flow": (
        ("float", "1d_map", "2d_map"),
        "[MW] Maximum instantenous flow for the aggregated flow of all group members. Constant, period, time or period+time.",
    ),
    "min_instant_flow": (
        ("float", "1d_map", "2d_map"),
        "[MW] Minimum instantenous flow for the aggregated flow of all group members. Constant, period, time or period+time.",
    ),
    "max_cumulative_flow": (
        ("float", "1d_map"),
        "[MW] Maximum average flow, which limits the cumulative flow for a group of connection_nodes and/or unit_nodes. The average value is multiplied by the model duration to get the cumulative limit (e.g. by 8760 if a single year is modelled). Applied for each solve. Constant or period.",
    ),
    "min_cumulative_flow": (
        ("float", "1d_map"),
        "[MW] Minimum average flow, which limits the cumulative flow for a group of connection_nodes and/or unit_nodes. The average value is multiplied by the model duration to get the cumulative limit (e.g. by 8760 if a single year is modelled). Applied for each solve. Constant or period.",
    ),
}
_V58_ENUM_FROM_BOOLS = {
    ("yes", "no"): "dispatch_plots_only",
    ("no", "yes"): "standalone_aggregator_only",
    ("yes", "yes"): "both",
    ("no", "no"): "none",
}


def _v58_find_one(db, table_name, **keys):
    try:
        return db.item(db.mapped_table(table_name), **keys)
    except SpineDBAPIError:
        return None


def _v58_check(item_error, what):
    """Raise on a truthy error from add/update_item ``(item, error)`` tuples."""
    if isinstance(item_error, tuple) and len(item_error) == 2:
        _, error = item_error
        if error:
            raise SpineDBAPIError(f"v58: {what} failed: {error}")
    return item_error



#: v58: the descriptions of the renamed group output flags (``print_dispatch`` kept this text up to v79).
_V58_GROUP_OUTPUT_FLAG_DESCRIPTIONS: dict[str, str] = {
    "print_dispatch": (
        "Combines unit/connection flows, flowGroup aggregated flows, and other balance elements "
        "(slack, demand, internal_losses, etc.) into a set of timeseries and a preconfigured plot per group."
    ),
    "print_indicators": (
        "Output indicators for the set of nodes in the group (Loss of Load, VRE generation, etc.)."
    ),
}

def _migrate_v58_carve_flowgroup_out_of_group(db) -> None:
    # Step 1: create classes (guarded / idempotent)
    if not db.find_entity_classes(name="flowGroup"):
        # display_icon = icon_code (low 24 bits, group's icon) + (color #828094 << 24)
        db.add_entity_class(name="flowGroup", display_icon=143488754774337)
    if not db.find_entity_classes(name="flowGroup__unit__node"):
        db.add_entity_class(name="flowGroup__unit__node", dimension_name_list=("flowGroup", "unit", "node"))
    if not db.find_entity_classes(name="flowGroup__connection__node"):
        db.add_entity_class(name="flowGroup__connection__node", dimension_name_list=("flowGroup", "connection", "node"))
    _commit_step(db, "v58: create flowGroup + 3-dim membership classes")

    # Step 2: value-list + flowGroup param defs
    add_value_list_manual(db, [
        ["flow_aggregator_methods", "none"],
        ["flow_aggregator_methods", "dispatch_plots_only"],
        ["flow_aggregator_methods", "standalone_aggregator_only"],
        ["flow_aggregator_methods", "both"],
    ])
    for name, (type_list, desc) in _V58_FLOW_LIMIT_DEFS.items():
        if not db.find_parameter_definitions(entity_class_name="flowGroup", name=name):
            db.add_parameter_definition(
                entity_class_name="flowGroup", name=name,
                parameter_type_list=type_list, parameter_group_name="flow_limit", description=desc,
            )
    if not db.find_parameter_definitions(entity_class_name="flowGroup", name="flow_aggregator"):
        none_val, none_type = to_database("none")
        db.add_parameter_definition(
            entity_class_name="flowGroup", name="flow_aggregator",
            default_value=none_val, default_type=none_type,
            parameter_value_list_name="flow_aggregator_methods",
            parameter_type_list=("str",), parameter_group_name="output",
            description=(
                "Choice of flow-aggregation method for this flowGroup: 'none' (used for flow limits only, "
                "no aggregated output), 'dispatch_plots_only' (used as an aggregated flowGroup within "
                "'print_dispatch' group of nodes - do not include one flow in multiple flow aggregators to "
                "avoid double counting), 'standalone_aggregator_only' (just make time series available for "
                "manual output processing / simple plots), 'both'."
            ),
        )
    _commit_step(db, "v58: flowGroup value-list + parameter definitions")

    # Step 3: classify + migrate (snapshots taken before any mutation)
    flow_mem = {}
    for cls in _V58_THREE_DIM:
        for ent in db.find_entities(entity_class_name=cls):
            flow_mem.setdefault(ent["entity_byname"][0], {}).setdefault(cls, []).append(ent["entity_byname"])
    group_mem = {}
    for cls in _V58_GROUP_MEMBERSHIP:
        for ent in db.find_entities(entity_class_name=cls):
            group_mem.setdefault(ent["entity_byname"][0], set()).add(cls)
    flow_pvals = {}
    nonflow_param_names = {}
    flag_only = set()
    for pv in db.find_parameter_values(entity_class_name="group"):
        g = pv["entity_byname"][0]
        p = pv["parameter_definition_name"]
        if p in _V58_FLOW_VALUE_PARAMS:
            flow_pvals.setdefault(g, []).append(pv)
        elif p in _V58_BOOL_FLOW_PARAMS:
            # bool flow flags: re-mapped to the flow_aggregator enum below if the
            # entity has flow membership; recorded here so a flag-only entity
            # (no membership, no value, no non-flow param) is surfaced as vacuous.
            flag_only.add(g)
        else:
            nonflow_param_names.setdefault(g, set()).add(p)
    group_names = [e["name"] for e in db.find_entities(entity_class_name="group")]
    migrated = []
    split = []
    dropped_vacuous = []
    removed_entities = []
    for g in group_names:
        # A flowGroup of the same name was made by an earlier run of this step (before v58 there is no flowGroup)
        # that stopped before it removed every flow membership of the group: the group is still being re-homed.
        has_flow_mem = g in flow_mem or bool(db.find_entities(entity_class_name="flowGroup", name=g))
        has_group_role = (g in group_mem) or (g in nonflow_param_names)
        if not has_flow_mem:
            # No flow membership: any flow content on this group is vacuous and
            # will be dropped when its def leaves `group` (Step 4).  Surface a
            # log entry both for stray flow VALUES and for flag-only orphans
            # (only a flow_aggregator/output_flowGroup_indicators flag, with no
            # 3-dim membership, no node/2-dim membership, and no non-flow param).
            # The bare group entity is left in place (an empty group is harmless).
            if (g in flow_pvals) or (g in flag_only and not has_group_role):
                dropped_vacuous.append(g)
            continue
        if not db.find_entities(entity_class_name="flowGroup", name=g):
            db.add_entity(entity_class_name="flowGroup", name=g)
        for src_cls, target_cls in _V58_FLOWGROUP_THREE_DIM.items():
            for byname in flow_mem.get(g, {}).get(src_cls, []):
                new_byname = (g,) + tuple(byname[1:])
                if not db.find_entities(entity_class_name=target_cls, entity_byname=new_byname):
                    db.add_entity(entity_class_name=target_cls, entity_byname=new_byname)
        for pv in flow_pvals.get(g, []):
            if not db.find_parameter_values(
                entity_class_name="flowGroup", entity_byname=(g,),
                parameter_definition_name=pv["parameter_definition_name"], alternative_name=pv["alternative_name"],
            ):
                db.add_parameter_value(
                    entity_class_name="flowGroup", entity_byname=(g,),
                    parameter_definition_name=pv["parameter_definition_name"], alternative_name=pv["alternative_name"],
                    value=pv["value"], type=pv["type"],
                )
        # bool flags -> method enum, per alternative (data-faithful; no collapse)
        agg_by_alt = {}
        ind_by_alt = {}
        for pv in db.find_parameter_values(entity_class_name="group", entity_byname=(g,), parameter_definition_name="flow_aggregator"):
            agg_by_alt[pv["alternative_name"]] = from_database(pv["value"], pv["type"])
        for pv in db.find_parameter_values(entity_class_name="group", entity_byname=(g,), parameter_definition_name="output_flowGroup_indicators"):
            ind_by_alt[pv["alternative_name"]] = from_database(pv["value"], pv["type"])
        for alt in set(agg_by_alt) | set(ind_by_alt):
            enum_val = _V58_ENUM_FROM_BOOLS.get((agg_by_alt.get(alt, "no"), ind_by_alt.get(alt, "no")), "none")
            if enum_val == "none":
                continue
            ev, et = to_database(enum_val)
            if not db.find_parameter_values(
                entity_class_name="flowGroup", entity_byname=(g,),
                parameter_definition_name="flow_aggregator", alternative_name=alt,
            ):
                db.add_parameter_value(
                    entity_class_name="flowGroup", entity_byname=(g,),
                    parameter_definition_name="flow_aggregator", alternative_name=alt, value=ev, type=et,
                )
        for ea in list(db.find_entity_alternatives(entity_class_name="group", entity_byname=(g,))):
            if not db.find_entity_alternatives(entity_class_name="flowGroup", entity_byname=(g,), alternative_name=ea["alternative_name"]):
                db.add_entity_alternative(
                    entity_class_name="flowGroup", entity_byname=(g,),
                    alternative_name=ea["alternative_name"], active=bool(ea.get("active", True)),
                )
        if has_group_role:
            # SPLIT: keep the group entity (node/2-dim membership + non-flow
            # params), drop only its 3-dim flow membership rows.
            for src_cls in _V58_THREE_DIM:
                removed_entities += [ent["id"] for ent in db.find_entities(entity_class_name=src_cls)
                                     if ent["entity_byname"][0] == g]
            split.append(g)
        else:
            ge = _v58_find_one(db, "entity", entity_class_name="group", name=g)
            if ge is not None:
                removed_entities.append(ge["id"])
            migrated.append(g)
    # A group removes its flow memberships, their values and entity_alternatives with it: many rows in a large
    # database, so the removal is bounded.
    _remove_items_bounded(db, "entity", removed_entities, "v58: removed the flow memberships of group")
    _commit_step(db, f"v58: re-homed {migrated!r}; split {split!r}; vacuous dropped {dropped_vacuous!r}")

    # Step 4: drop old flow defs + 3-dim classes from group (cascades values/members)
    for name in _V58_ALL_FLOW_PARAMS:
        pdef = _v58_find_one(db, "parameter_definition", entity_class_name="group", name=name)
        if pdef is not None:
            _remove_items_bounded(db, "parameter_definition", [pdef["id"]], f"v58: removed group.{name}")
    for cls in _V58_THREE_DIM:
        ec = _v58_find_one(db, "entity_class", name=cls)
        if ec is not None:
            _remove_items_bounded(db, "entity_class", [ec["id"]], f"v58: removed {cls}")
    _commit_step(db, "v58: dropped old group flow defs + 3-dim classes")

    # Step 5: rename output flags on group + set their new descriptions
    _v58_rename_descs = _V58_GROUP_OUTPUT_FLAG_DESCRIPTIONS
    for old, new in (("output_nodeGroup_dispatch", "print_dispatch"), ("output_nodeGroup_indicators", "print_indicators")):
        pdef = _v58_find_one(db, "parameter_definition", entity_class_name="group", name=old)
        if pdef is not None:
            _v58_check(
                db.update_item(
                    "parameter_definition", id=pdef["id"], name=new, description=_v58_rename_descs[new],
                ),
                f"rename group.{old}->{new}",
            )
    _commit_step(db, "v58: renamed output_nodeGroup_* -> print_*")

    # Step 6: §8 metadata heal on group stragglers (no-op where already present)
    _v58_metadata_heal = {
        "base_MVA": (("float",), "[MVA] Base power for the per-unit system used in DC power flow. Default 100. susceptance = base_MVA / reactance_pu."),
        "candidate_precapacity_to_avoid_big_m": (("float",), "[MW] Small pre-existing capacity assigned to investment candidate connections in DC power flow groups that have zero existing capacity. This avoids the need for big-M / MIP constraints by ensuring the angle constraint is always active. Default 0.1 MW. The value should be small enough to not affect results but large enough for numerical stability."),
        "reference_node": (("str",), "Name of the reference bus node (angle fixed to zero) for DC power flow. Optional — if not specified, automatically selected as the node with the largest existing capacity in each connected component of the DC power flow network."),
        "transfer_method": (("str",), None),
        "cumulative_max_capacity": (("float", "1d_map"), None),
        "cumulative_min_capacity": (("float", "1d_map"), None),
    }
    for name, (type_list, desc) in _v58_metadata_heal.items():
        pdef = _v58_find_one(db, "parameter_definition", entity_class_name="group", name=name)
        if pdef is None:
            continue
        kwargs = {"parameter_type_list": type_list}
        if desc is not None:
            kwargs["description"] = desc
        _v58_check(db.update_item("parameter_definition", entity_class_name="group", name=name, **kwargs), f"heal group.{name}")
    # §9: node.invest_forced gains its missing description (keep its type_list).
    if _v58_find_one(db, "parameter_definition", entity_class_name="node", name="invest_forced") is not None:
        _v58_check(
            db.update_item(
                "parameter_definition", entity_class_name="node", name="invest_forced",
                description="[MWh] Forces the model to invest exactly this amount of new storage capacity, overriding the investment optimisation (equivalent to setting invest_min and invest_max equal). Constant or period.",
            ),
            "heal node.invest_forced description",
        )
    _commit_step(db, "v58: heal missing group/node parameter metadata (§8/§9)")

    # Verification (raise on miss)
    for cls in ("flowGroup", "flowGroup__unit__node", "flowGroup__connection__node"):
        if not db.find_entity_classes(name=cls):
            raise SpineDBAPIError(f"v58: {cls} missing")
    for cls in _V58_THREE_DIM:
        if db.find_entity_classes(name=cls):
            raise SpineDBAPIError(f"v58: old {cls} still present")
    for name in _V58_ALL_FLOW_PARAMS:
        if db.find_parameter_definitions(entity_class_name="group", name=name):
            raise SpineDBAPIError(f"v58: group.{name} not dropped")
    for old in ("output_nodeGroup_dispatch", "output_nodeGroup_indicators"):
        if db.find_parameter_definitions(entity_class_name="group", name=old):
            raise SpineDBAPIError(f"v58: group.{old} not renamed")
