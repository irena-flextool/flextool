"""The solve-side warning codes (W-0500..W-0999) of the registry :mod:`flextool.run_warnings.registry`: the checks of
the engine (``flextool/engine``: solves, the preprocessing of a solve, the model build, the outputs of a solve).
Same rules as the registry's own codes (code order, a short lowercase ``message``, a ``body`` with **How to fix:**);
the registry merges them.

* W-050x solves: realization, carried states, look-backs
* W-051x inputs of a solve: inflow scaling, limits, maps, time steps
* W-052x lifetimes, investments and units
* W-053x outputs of a solve and storage notes
* W-054x what the solver's outcome means (notes after a solve that ended optimal, but not cleanly); from W-0543
  the scaling guard (the solver's accuracy on the LP it gets)
* W-055x warm start and how the solves are run (in this process or a separate one)
* W-057x the results of a Benders solve: its final solve (the whole model with the Benders investments fixed)
* W-058x W-0580..W-0583 storage notes (continued from W-053x); from W-0584 (and W-059x) the overflow of the full
  sub-ranges (W-050x solves, W-051x inputs, W-052x lifetimes: the topic of each code says which)
* W-06xx behaviour-change notices (their texts: :mod:`flextool.run_warnings._behaviour_specs`)
"""
from __future__ import annotations

from flextool.run_warnings._behaviour_specs import BEHAVIOUR_SPECS
from flextool.run_warnings.registry import NOTE, WarningSpec

_SOLVES = "Solves and realization"
_INPUTS = "Inputs of a solve"
_LIFETIME = "Lifetimes, investments and units"
_OUTPUTS = "Results of a solve"
_STORAGE = "Storage"
_SOLVER = "Solver outcome"
_WARM = "Warm start and memory"
_SCALING = "Scaling guard"

_CROSSOVER_FIX = (
    "add `run_crossover` = `on` to the solve's `solver_arguments` to run crossover (or set `solver` = "
    "`simplex` there). Crossover turns the interior point into a vertex solution that HiGHS certifies "
    "against the absolute tolerances; it takes extra time, which on a large model can be much more than the "
    "interior point itself. Scaling problems (see the `Coefficient ranges` HiGHS prints and the autoscale line) "
    "make the outcome more likely: fixing the units of very large or very small numbers helps too."
)

ENGINE_SPECS: tuple[WarningSpec, ...] = (
    WarningSpec(
        code="W-0500",
        message="solves realize the same {steps} time steps ({periods}); the later is kept",
        title="Two solves realize the same dispatch",
        topic=_SOLVES,
        body=(
            "Both named solves realize dispatch in the same time steps of the named periods. The solve that runs "
            "later keeps those time steps; the earlier solve's results for them are not reported. The line counts "
            "the time steps and periods over all levels of the two solves (the rolls of a rolling solve count as "
            "that solve). This is expected when, for example, a dispatch solve is nested inside an investment "
            "solve.\n\n"
            "**How to fix:** if the overlap is not meant, check the `realized_periods` of the two solves (and, "
            "for rolling solves, `rolling_jump` / `rolling_horizon`) so that each time step is realized by one "
            "solve."
        ),
    ),
    WarningSpec(
        code="W-0501",
        message="two solves both realize the investments of {periods}",
        title="Two solves realize the same investment periods",
        topic=_SOLVES,
        body=(
            "Both named solves list the same period(s) in `realized_invest_periods`. Investments of a period should "
            "be realized by only one solve. If the later solve can also invest in that period, it treats the earlier "
            "solve's investment as existing capacity and may invest more on top of it; the reported invested "
            "capacity and investment cost of the period then come from the later solve only, although the total "
            "capacity includes both.\n\n"
            "**How to fix:** remove the period from the `realized_invest_periods` of one of the solves."
        ),
    ),
    WarningSpec(
        code="W-0502",
        message="a realised state ends after solve '{solve}' starts; not carried",
        title="Realised state not before the solve's start",
        topic=_SOLVES,
        body=(
            "A solve continues the states (storage levels, on-line units, ramps) that the earlier solves of its "
            "nesting level realised before its first time step. The named earlier solve realised a state that ends "
            "at or after the start of this solve, so it is not carried into this solve. The debug log gives the "
            "time steps.\n\n"
            "**How to fix:** check the `realized_periods` and the time steps of the two solves: a solve that "
            "continues another should start after the last time step the other realizes."
        ),
    ),
    WarningSpec(
        code="W-0503",
        message="solve '{solve}' starts a level: storage not from the higher level",
        title="First solve of a nesting level does not continue the higher level",
        topic=_SOLVES,
        body=(
            "The solve is the first solve of its nesting level, so its storage nodes start from their "
            "`storage_start_end_method` (`fix_start`) or free, not from the state a solve of a higher level "
            "realised before it. This happens when a nesting level first appears later in the model.\n\n"
            "**How to fix:** usually nothing. If the storage should continue the higher level's state, start the "
            "nested level together with the higher level, or give the storage nodes a `storage_state_start`."
        ),
    ),
    WarningSpec(
        code="W-0504",
        message="minimum {parameter} look-back cut short at a start of solve '{solve}'",
        title="Minimum up / down time look-back cut short",
        topic=_SOLVES,
        body=(
            "The process has `min_uptime` or `min_downtime`, but at a first time step of a period of the solve the "
            "look-back covers 0 hours: the time steps before it belong to an earlier solve, and this step does not "
            "continue the last step that solve realized. No realized on-line history reaches it, so the earlier "
            "start-ups and shut-downs do not constrain it. The debug log names the time step.\n\n"
            "**How to fix:** make the solves continue each other in time (a hole or an overlap between the "
            "realized time steps of the solves causes this), or accept the looser constraint at that step."
        ),
    ),
    WarningSpec(
        code="W-0505",
        message="group new_stepduration shorter than the time step of solve '{solve}'",
        title="Group new_stepduration shorter than the solve's time step",
        topic=_SOLVES,
        body=(
            "The group's `new_stepduration` would make its nodes coarser than the solve's time step, but it is "
            "shorter than that step, so it has no effect: the group's nodes are balanced at the solve's own time "
            "step. The debug log gives both durations.\n\n"
            "**How to fix:** remove `new_stepduration` from the group, or give it a multiple of the solve's time "
            "step (the solve's own `new_stepduration`, or its timeline's step)."
        ),
    ),
    WarningSpec(
        code="W-0506",
        message="rolls of solve '{solve}' realize the same {steps} time steps ({periods}); the later is kept",
        title="Rolls of a rolling solve realize the same dispatch",
        topic=_SOLVES,
        entityless=True,
        body=(
            "Two rolls of the same rolling solve realize dispatch in the same time steps of the named periods. "
            "Each roll should realize only the time steps up to the start of the next roll (`rolling_jump`); here "
            "a roll realizes time steps that a later roll realizes again, and the later roll's results are kept. "
            "The line counts the time steps and periods over all pairs of rolls; the debug log names each pair "
            "and period.\n\n"
            "**How to fix:** check the solve's `rolling_jump`, `rolling_horizon` and `rolling_duration`, and its "
            "`realized_periods`: each roll should realize only the `rolling_jump` time steps before the next "
            "roll starts."
        ),
    ),
    WarningSpec(
        code="W-0507",
        message="unit or connection in a new_stepduration group links nodes outside it",
        title="Unit or connection listed in a coarse group",
        topic=_SOLVES,
        body=(
            "The unit or connection (named with the group) is listed in a group that has a `new_stepduration` "
            "longer than the solve's time step, and it connects to node(s) outside that group. Listing a unit or "
            "connection in such a group has no effect on time resolution: its flows and online status stay at "
            "the solve's time step, and each coarse node sums the flows over its coarse step. The debug log names "
            "the nodes outside the group.\n\n"
            "**How to fix:** usually nothing; remove the unit or connection from the group "
            "(`group__unit` / `group__connection`) to silence the warning. Only the nodes of a group get its "
            "time step."
        ),
    ),
    WarningSpec(
        code="W-0508",
        message="stochastic_branches forecasts where no solve or roll fans out; ignored",
        title="Forecasts where no solve or roll fans out",
        topic=_SOLVES,
        body=(
            "`stochastic_branches` declares forecasts (a solve, a period and an `analysis_time`) where no solve "
            "or roll fans out, and FlexTool ignores them. A solve or roll fans out once, at its first reveal, and "
            "its branches read the forecasts made then through the rest of its horizon. Each case is a complete "
            "solve, a period and an analysis time; the debug log gives the forecast branches and the solve or "
            "roll whose horizon reaches them.\n\n"
            "**How to fix:** to use these forecasts, make a roll start there (`rolling_solve_jump`) or put the "
            "later fan in a separate solve; otherwise remove these `stochastic_branches` rows."
        ),
    ),
    WarningSpec(
        code="W-0510",
        message="annual_flow is above what peak_inflow can reach",
        title="annual_flow above the reach of peak_inflow",
        topic=_INPUTS,
        body=(
            "The node uses `inflow_method` `scale_to_annual_and_peak_flow`, but its `annual_flow` is above "
            "|`peak_inflow`| times the time steps of a year: no profile with that peak reaches the annual flow, "
            "so the scaled inflow exceeds `peak_inflow` at its off-peak time steps. The debug log gives the "
            "values.\n\n"
            "**How to fix:** raise `peak_inflow` or lower `annual_flow`."
        ),
    ),
    WarningSpec(
        code="W-0511",
        message="the scaled inflow changes direction in some time steps",
        title="Scaled inflow changes direction",
        topic=_INPUTS,
        body=(
            "The node uses `inflow_method` `scale_to_annual_and_peak_flow`, and its `annual_flow` is too small "
            "for this `peak_inflow` and profile shape: the low-demand (or low-supply) time steps turn into supply "
            "(or demand). The debug log gives the number of time steps and the values.\n\n"
            "**How to fix:** raise `annual_flow` or lower |`peak_inflow`|."
        ),
    ),
    WarningSpec(
        code="W-0512",
        message="timeset_weights change the annualised inflow by over 1%",
        title="timeset_weights change the annualised inflow",
        topic=_INPUTS,
        body=(
            "For a scaled inflow (`inflow_method` `scale_to_annual_flow` or `scale_to_annual_and_peak_flow`), "
            "the non-uniform `timeset_weights` imply an annualisation that differs from an even sample of the "
            "selected time steps by more than 1 %. The debug log gives the factor per node and period.\n\n"
            "**How to fix:** check that the `timeset_weights` match how representative the selected time steps "
            "are."
        ),
    ),
    WarningSpec(
        code="W-0513",
        message="limit is {use} in only part of a new_stepduration step; dropped",
        title="No-limit value in part of a solve step",
        topic=_INPUTS,
        body=(
            "The parameter is a limit, and it is `inf` / `-inf` (no limit) in some but not all time steps of a "
            "solve step of `new_stepduration`. A solve step takes the mean of its time steps, which is then "
            "`inf` / `-inf`, so the limit is dropped for the whole solve step. The debug log names the solve "
            "steps.\n\n"
            "**How to fix:** give `inf` / `-inf` in all or none of the time steps of a solve step, or use a "
            "shorter `new_stepduration`."
        ),
    ),
    WarningSpec(
        code="W-0514",
        message="limit has a very large value (numerically unstable)",
        title="Very large limit value",
        topic=_INPUTS,
        body=(
            "The parameter is used as a limit, and a value this large (1e10 or more) makes the solve numerically "
            "unstable. The debug log gives the values and where they are.\n\n"
            "**How to fix:** if no limit is meant, give `inf` (an upper limit) or `-inf` (a lower limit) there; "
            "otherwise check the unit of the value."
        ),
    ),
    WarningSpec(
        code="W-0515",
        message="maps without index names; read from their depth and keys",
        title="Maps without index names",
        topic=_INPUTS,
        severity=NOTE,
        body=(
            "Some values of the parameter are maps whose index levels have no names, so FlexTool reads them "
            "(over periods, time or both) from their depth and keys. The debug log says how each entity's maps "
            "were read.\n\n"
            "**How to fix:** name the index levels of the maps (for example `period` and `time`) to make the "
            "reading explicit."
        ),
    ),
    WarningSpec(
        code="W-0516",
        message="timeset block length is not a whole number of time steps",
        title="Timeset block of a fractional length",
        topic=_INPUTS,
        body=(
            "A block of the timeset's `timeset_duration` has a length that is not a whole number of time steps. "
            "The whole time steps are used. The debug log gives the block and its length.\n\n"
            "**How to fix:** give the block length as a whole number of time steps."
        ),
    ),
    WarningSpec(
        code="W-0517",
        message="timeset block runs past the end of its timeline; cut there",
        title="Timeset block past the end of the timeline",
        topic=_INPUTS,
        body=(
            "A block of the timeset's `timeset_duration` has more time steps than its timeline has from the "
            "block's start, so the block is cut at the end of the timeline. The debug log gives the block, its "
            "length and the time steps left.\n\n"
            "**How to fix:** shorten the block or start it earlier, or extend the timeline."
        ),
    ),
    WarningSpec(
        code="W-0518",
        message="a new_stepduration step has values for only some of its time steps",
        title="Time series with gaps inside a solve step",
        topic=_INPUTS,
        body=(
            "The named time series (`class.parameter`) has values for only some of the time steps that a solve "
            "step of `new_stepduration` combines. The solve step is made of the time steps that have a value; the "
            "default of the other time steps is not part of it. The debug log gives an example entity and step.\n\n"
            "**How to fix:** give the time series a value in every time step (or none of a solve step's), or check "
            "that its time steps are those of the timeline."
        ),
    ),
    WarningSpec(
        code="W-0519",
        message="value of {parameter} varies within a solve step; its curve is approximate",
        title="Value that varies within a solve step makes a curve approximate",
        topic=_INPUTS,
        body=(
            "`unit.min_load` varies within a solve step of `new_stepduration` for the named units with "
            "`min_load_efficiency`. The solve step uses the mean `min_load` and the harmonic means of `efficiency` "
            "and `efficiency_at_min_load`, so the min_load_efficiency curve (slope and section) of those steps is "
            "an approximation of the step mean of the hourly curves. It is exact for a `min_load` that is constant "
            "within each solve step.\n\n"
            "**How to fix:** usually nothing. For an exact curve, keep `min_load` constant within each solve step "
            "(or use a shorter `new_stepduration`)."
        ),
    ),
    WarningSpec(
        code="W-0520",
        message="existing capacity replaced at its end of life in the horizon",
        title="Existing capacity replaced automatically",
        topic=_LIFETIME,
        severity=NOTE,
        body=(
            "The existing capacity of the entity (`lifetime_method` `reinvest_automatic`) reaches its end of life "
            "inside the model horizon and is replaced at the `invest_cost` of the period the replacement comes "
            "online. The replacement is reported as investment pre-existing, outside the optimised objective (as "
            "the fixed cost of existing capacity). The debug log gives the year of the replacement.\n\n"
            "**How to fix:** nothing, if this is meant. Use `lifetime_method` `reinvest_choice` to let the model "
            "decide the replacement, or `no_reinvestment` to let the capacity retire."
        ),
    ),
    WarningSpec(
        code="W-0521",
        message="existing capacity past its end of life at the model start",
        title="Existing capacity past its end of life",
        topic=_LIFETIME,
        body=(
            "The end of life of the entity's existing capacity (`lifetime_technical`, with its age or "
            "`build_year`) is before or at the start of the model, or the `end_of_life_method` rounds it to the "
            "model start, so its existing capacity is 0 in every period. The debug log gives the years.\n\n"
            "**How to fix:** check the entity's age (`age_at_model_start` or `build_year`) and "
            "`lifetime_technical`."
        ),
    ),
    WarningSpec(
        code="W-0522",
        message="investments would never operate (end of life at start)",
        title="Investments that would never operate",
        topic=_LIFETIME,
        body=(
            "The investment decisions of the entity would be in service in no period: `lifetime_technical` "
            "with the `end_of_life_method` rounds their end of life to the start of the period they start to "
            "operate in. They are not offered (their investment minimums do not apply). The debug log gives the "
            "periods.\n\n"
            "**How to fix:** use a longer `lifetime_technical` or an `end_of_life_method` that keeps the "
            "capacity for its period (for example `next_seam`)."
        ),
    ),
    WarningSpec(
        code="W-0523",
        message="life extensions would never operate (end of life at start)",
        title="Life extensions that would never operate",
        topic=_LIFETIME,
        body=(
            "The life extensions of the entity would be in service in no period: `life_extension_years` with "
            "the `end_of_life_method` rounds their end of life to the start of the period they start in. They "
            "are not offered. The debug log gives the periods.\n\n"
            "**How to fix:** use a longer `life_extension_years` or an `end_of_life_method` that keeps the "
            "capacity for its period."
        ),
    ),
    WarningSpec(
        code="W-0524",
        message="investments would start after the model horizon; not offered",
        title="Investments that would start after the model horizon",
        topic=_LIFETIME,
        body=(
            "With its `construction_lead_time`, the capacity of the entity's investment decisions in some periods "
            "would start to operate after the end of the model horizon, so these investments are not offered. "
            "The debug log gives the lead time and the periods.\n\n"
            "**How to fix:** nothing, if this is meant. Otherwise extend the model horizon, shorten the "
            "`construction_lead_time` or use `construction_lead_time_method` `immediate`."
        ),
    ),
    WarningSpec(
        code="W-0525",
        message="investments would start after solve '{solve}'; not offered",
        title="Investments that would start after the solve",
        topic=_LIFETIME,
        body=(
            "With its `construction_lead_time`, the capacity of the entity's investment decisions in some periods "
            "would start to operate after the end of the solve, so these investments are not offered in this "
            "solve. The debug log gives the lead time and the periods.\n\n"
            "**How to fix:** use a longer solve horizon (more investment periods in the solve) or "
            "`construction_lead_time_method` `immediate`."
        ),
    ),
    WarningSpec(
        code="W-0526",
        message="integer online units: existing is not whole units",
        title="Fractional number of existing units",
        topic=_LIFETIME,
        body=(
            "The unit or connection has integer online variables (whole units: `startup_method` *integer* or "
            "*integer_fix_in_uc*), but its existing capacity divided "
            "by its unit size is not a whole number of units in some period. Integer online units cannot use the "
            "fraction (below 1 the unit can never be online, which can make a minimum such as `min_instant_flow` "
            "infeasible). Without `virtual_unitsize`, one unit is the largest existing capacity over the periods. "
            "The debug log gives the counts per period.\n\n"
            "**How to fix:** set `virtual_unitsize` to the size of one unit and `existing` to the number of "
            "units, or use `startup_method` `linear` (`linear_fix_in_uc` in place of `integer_fix_in_uc`)."
        ),
    ),
    WarningSpec(
        code="W-0527",
        message="net_salvage_value worth more than invest_cost: yearly cost below 0",
        title="Salvage value above the investment cost",
        topic=_LIFETIME,
        body=(
            "A plant the model can build (an investment, or a generation of its replacement chain) has a "
            "`net_salvage_value` worth more than its `invest_cost` (the salvage at the end of its life, "
            "discounted at `discount_rate`): its yearly cost is below 0, so the model builds as much as the "
            "investment limits allow. Each entity is named once per run.\n\n"
            "**How to fix:** check `net_salvage_value` against `invest_cost` (also in the periods a replacement "
            "comes online)."
        ),
    ),
    WarningSpec(
        code="W-0528",
        message="free automatic replacement keeps its net_salvage_value",
        title="Free replacement with a salvage value",
        topic=_LIFETIME,
        body=(
            "The existing capacity (`lifetime_method` `reinvest_automatic`) is replaced at its end of life by a "
            "plant without an `invest_cost` that still has a `net_salvage_value`: the replacement is free and "
            "keeps its salvage value, so keeping the capacity earns that salvage value and a retirement gives it "
            "up. Each entity is named once per run.\n\n"
            "**How to fix:** set `invest_cost` if the replacement, and so the retirement, should be priced."
        ),
    ),
    WarningSpec(
        code="W-0529",
        message="building and retiring the capacity earns money",
        title="Building to retire earns money",
        topic=_LIFETIME,
        body=(
            "Building the capacity and retiring it earns money: the `net_salvage_value` received at retirement "
            "is worth more than what the capacity costs until then (`invest_cost`, `fixed_cost`, "
            "`discount_rate`). The model builds as much as the investment limits allow just to retire it; "
            "without a limit the problem is unbounded. Each entity is named once per run.\n\n"
            "**How to fix:** check `net_salvage_value` against `invest_cost`."
        ),
    ),
    WarningSpec(
        code="W-0530",
        message="capacity margin not met in solve '{solve}'; penalty paid",
        title="Capacity margin shortfall",
        topic=_OUTPUTS,
        body=(
            "The group's capacity margin constraint is not met in the investment period (the name is "
            "`group, period`): the solve pays `penalty_capacity_margin` for the shortfall. The debug log gives "
            "the shortfall (MW), the time step that binds and the penalty; the results have "
            "`capacity_margin_report` tables.\n\n"
            "**How to fix:** allow more investment (or existing capacity) in the group, lower "
            "`capacity_margin`, or accept the penalty if it is meant as a price."
        ),
    ),
    WarningSpec(
        code="W-0531",
        message="piecewise unit used more input than its curve needs",
        title="Piecewise unit burns input for nothing",
        topic=_OUTPUTS,
        body=(
            "A unit with a piecewise efficiency curve used more weighted input (input × `conversion_flow_coeff`) "
            "than its curve needs in some realized time steps. Its input had no cost or a negative cost there (a "
            "surplus, a free or curtailed input, a CO2 credit), or a reserve or a minimum on its input made it pay, "
            "so the solution is not unique or burns input for nothing. The debug log gives the time steps and "
            "the amounts.\n\n"
            "**How to fix:** give the input a cost, or check the reserves and minimums on the unit's input."
        ),
    ),
    WarningSpec(
        code="W-0532",
        message="rolling '{solve}': bind_within_timeblock used as bind_within_period",
        title="Rolling solve: bind_within_timeblock as bind_within_period",
        topic=_STORAGE,
        severity=NOTE,
        body=(
            "In a rolling solve each period is one continuous timeblock, so the storage nodes with "
            "`storage_binding_method` `bind_within_timeblock` use `bind_within_period` in this solve.\n\n"
            "**How to fix:** nothing; set `bind_within_period` on the nodes to say it explicitly."
        ),
    ),
    WarningSpec(
        code="W-0533",
        message="rolls of '{solve}' start inside a period; storage carries on",
        title="Rolls start inside a period",
        topic=_STORAGE,
        severity=NOTE,
        body=(
            "The rolls of the rolling solve start inside a period. Storage nodes with `bind_within_period` (also "
            "`bind_within_timeblock`) or `bind_intraperiod_blocks` continue that period from the level the "
            "previous roll realised; a roll ends its part at least at that level, and the roll(s) holding the "
            "period's end end it at the level it started from.\n\n"
            "**How to fix:** nothing; this is how a rolling solve binds storage within a period."
        ),
    ),
    WarningSpec(
        code="W-0534",
        message="storage ends free in '{solve}'; its storage_state_start energy is usable",
        title="Storage start energy can be used up (an endowment)",
        topic=_STORAGE,
        body=(
            "The storage nodes start the solve at `storage_state_start` on capacity the solve cannot change "
            "(existing capacity, no investment or life extension in the start period) and chain forward to a "
            "free end: `bind_forward_only`, no `fix_end`, no `use_reference_value` and no end fixed by an "
            "upper-level solve. The run may then use this start energy without producing it. This is a valid "
            "model of an initial stock (an endowment). The warning is given in the first solve of each level "
            "(a start carried from an earlier solve is not an endowment). The debug log gives the energy per "
            "node and in total (MWh, on the existing capacity before any retirement in the solve).\n\n"
            "**How to fix:** nothing, if the initial stock is meant. If the storage should return it, use "
            "`storage_start_end_method` `fix_start_end`, a cyclic `storage_binding_method` or "
            "`storage_solve_horizon_method` `use_reference_value`."
        ),
    ),
    WarningSpec(
        code="W-0535",
        message="storage start on investable capacity not matched at the end in '{solve}'",
        title="Start of investable storage not matched at the end",
        topic=_STORAGE,
        body=(
            "The storage nodes start the solve at a level the solve decides by investing: `storage_state_start` "
            "(or the reference value of `use_reference_value`) on capacity the solve can build or extend in the "
            "start period, and chain forward. The matching end (the end held at least at the start) needs both "
            "ends in one solve, and this solve does not hold the last step of the model, so it is not applied: "
            "built storage can deliver its start energy without producing it. A free start (`fix_nothing`) is "
            "not warned about: the later solve that reaches the end of the model matches it.\n\n"
            "**How to fix:** use `storage_binding_method` `bind_within_solve` (or "
            "`bind_within_solve_blended_weights`), or `storage_start_end_method` `fix_start_end` with "
            "`storage_state_end` at least `storage_state_start`."
        ),
    ),
    WarningSpec(
        code="W-0536",
        message="storage_state_start differs from the reference in '{solve}'; start used",
        title="Start differs from the reference value",
        topic=_STORAGE,
        body=(
            "The storage nodes have `storage_solve_horizon_method` `use_reference_value` and a "
            "`storage_start_end_method` that fixes the start (`fix_start` or `fix_start_end`), and their "
            "`storage_state_start` differs from `storage_state_reference_value` at their first step. The start "
            "uses `storage_state_start`. (A node that does not fix its start starts from the reference value.) "
            "The debug log gives both values per node.\n\n"
            "**How to fix:** give `storage_state_start` the reference value of the first step, or set "
            "`storage_start_end_method` to `fix_nothing` (or `fix_end`) so that the start follows the reference "
            "value."
        ),
    ),
    WarningSpec(
        code="W-0537",
        message="storage slack in solve '{solve}' can bypass a linked hard balance",
        title="Storage balance slack bypasses a hard balance",
        topic=_STORAGE,
        body=(
            "The storage nodes have balance slack (`penalty_method` not `off`) and a unit or a connection links "
            "them to node(s) with `penalty_method` `off`. The storage slack can supply or absorb energy for "
            "those nodes and so bypass their hard balance at the storage node's penalty, which can be far below "
            "what an unmet demand there should cost. Only pure storage nodes (no own inflow or demand, e.g. a "
            "battery) are named: the slack of a node with its own demand is the intended outlet of that demand. "
            "The debug log gives the linked nodes of each storage node.\n\n"
            "**How to fix:** if the balance should hold, set `penalty_method` `off` on the storage node too (or "
            "give it a high penalty), so that a shortfall shows where it arises."
        ),
    ),
    WarningSpec(
        code="W-0538",
        message="reference end below the start in '{solve}'; caps storage investment",
        title="Reference value below the start of investable storage",
        topic=_STORAGE,
        body=(
            "The storage nodes have `storage_solve_horizon_method` `use_reference_value`, their reference value "
            "at the end lies below their start, and the solve can invest in their storage in the period they "
            "start in. As for `fix_start_end`, the matching end stays (the end at least at the start), so "
            "storage built must still return its start energy, which caps that investment.\n\n"
            "**How to fix:** give `storage_state_reference_value` at least the start if the storage should be "
            "built freely."
        ),
    ),
    WarningSpec(
        code="W-0539",
        message="storage in '{solve}' cycles over representative periods unweighted",
        title="Plain storage cycle on representative periods",
        topic=_STORAGE,
        body=(
            "The solve is on representative periods (its timesets have `representative_period_weights`), and "
            "the storage nodes use a `storage_binding_method` (`bind_within_solve`, `bind_within_period` or "
            "`bind_intraperiod_blocks`) that moves the storage level from one representative period to the next "
            "without their weights, while the flows of each step count with the weight of its period. This is "
            "reasonable when the representative periods have equal weights or when the timing of the storage's "
            "inflows does not matter (e.g. pumped hydro, large heat storage). Nodes with `storage_start_end_method` "
            "`fix_start_end` are not named (their binding method is not used). The debug log gives the "
            "timesets and the method of each node.\n\n"
            "**How to fix:** if the weights matter, use the `*_blended_weights` method, which weights the "
            "transfer: `bind_within_solve` -> `bind_within_solve_blended_weights`, `bind_within_period` and "
            "`bind_intraperiod_blocks` -> `bind_within_period_blended_weights`."
        ),
    ),
    WarningSpec(
        code="W-0540",
        message="solve '{solve}' optimal by HiGHS at up to {factor}x its relative tolerances{rolls}",
        title="Optimal within a multiple of the relative tolerances",
        topic=_SOLVER,
        severity=NOTE,
        entityless=True,
        body=(
            "HiGHS called the solve optimal, although its solution misses some of HiGHS's tolerances (by default "
            "1e-7) also in relative terms, by up to the factor given (HiGHS's own log says `Model status is "
            "\"Optimal\" since relative violation of tolerances is no more than X`; the note gives that X). After "
            "an interior-point solve without crossover (`solver` = `ipm`, `run_crossover` = `off`) X measures "
            "the relative violations, relative to the size of the numbers in each row and column; after simplex "
            "or crossover it measures the absolute infeasibilities and the gap between the primal and dual "
            "objectives. The factor tells how serious it is: a factor up to about 10 means the flows, storage "
            "levels and costs are accurate to about a millionth (1e-6) of their size, which is far below the "
            "accuracy of any input data: the results are usable. The duals (prices) are less exact than the primal values: small "
            "differences in prices between equivalent nodes or time steps can be numerical noise. The debug log "
            "names the checks and their largest violations. Each solve is noted once: for a rolling solve the "
            "note comes after its rolls, says in how many rolls it happened and gives the largest factor; the "
            "debug log names each roll, the worst first.\n\n"
            "**How to fix:** usually nothing. If prices or small values matter, or the factor is large: "
            + _CROSSOVER_FIX
        ),
    ),
    WarningSpec(
        code="W-0541",
        message=("these HiGHS IPX warnings ('imprecise', 'Unwelcome IPX status', 'Model status changed from Unknown "
                 "to Optimal') are expected in this solver mode and the solve was accepted by FlexTool's checks"),
        title="Interior point without crossover: HiGHS's tolerance warnings",
        topic=_SOLVER,
        severity=NOTE,
        entityless=True,
        body=(
            "The solve ran the interior-point method without crossover (`solver` = `ipm`, `run_crossover` = "
            "`off`) and HiGHS called the result optimal. Such a solution is an interior point, not a vertex, and "
            "HiGHS checks it against its tolerances (by default 1e-7) both in absolute terms and relative to the "
            "size of the numbers in each row and column. Its log then often has `WARNING` lines that read as "
            "alarms but are expected here:\n\n"
            "- `Status interior point solve: imprecise`, `WARNING: Unwelcome IPX status ... run_crossover is "
            "\"off\"`, `WARNING: Running postsolve on non-optimal solution of reduced LP` and `WARNING: Model "
            "status changed from \"Unknown\" to \"Optimal\"`: the interior point stopped a little short of "
            "HiGHS's own target, and the solution of the full model meets the tolerances after postsolve.\n"
            "- `WARNING: Solution optimality conditions` (also `: Before postsolve`) with a table of `num/max` "
            "violations: absolute violations in rows or columns with large numbers. The `relative` numbers in "
            "the same table are what counts.\n"
            "- `WARNING: LP solver claims optimality, but with ... relative ...` and `Model status is "
            "\"Optimal\" since relative violation of tolerances is no more than X`: the relative violations are "
            "at most X times the tolerance. A \"Before postsolve\" one concerns the presolved model only. An X "
            "up to about 10 means the flows, storage levels and costs are accurate to about a millionth of their "
            "size, far below the accuracy of any input data.\n\n"
            "The results are usable. The duals (prices) are less exact than the primal values: small price "
            "differences between equivalent nodes or time steps can be numerical noise.\n\n"
            "The note is printed **once per run**, at the first interior-point solve without crossover, whether "
            "or not HiGHS printed these warnings for it; it covers the warnings of the later solves of the run "
            "too. FlexTool cannot read HiGHS's log. A solve that HiGHS accepted only at a multiple of its "
            "tolerances also gets W-0540, with that multiple.\n\n"
            "**How to fix:** usually nothing. If a certified optimum matters: " + _CROSSOVER_FIX
        ),
    ),
    WarningSpec(
        code="W-0542",
        message="solve '{solve}' not certified optimal by HiGHS; its feasible solution is used",
        title="Near-optimal solve accepted",
        topic=_SOLVER,
        severity=NOTE,
        entityless=True,
        body=(
            "HiGHS ended the solve with model status `Unknown`: after an interior-point solve without crossover, "
            "postsolve left the dual objective slightly inconsistent, so HiGHS cannot certify the optimum. "
            "FlexTool checked the solution itself: the primal solution (the flows, storage levels and "
            "investments the results are made of) is feasible within the tolerances, and the gap between the "
            "primal and dual objectives is within the accepted optimality gap (`FLEXTOOL_ACCEPT_PD_GAP`, by "
            "default 1 %), so the solution is used as near-optimal. The objective can be off the true optimum "
            "by at most that gap. The debug log and the run record (`_run_status.json`) give the numbers. A solve "
            "that fails these checks stops the run with an error instead. Each solve is noted once.\n\n"
            "**How to fix:** usually nothing. If the exact optimum or the prices matter: " + _CROSSOVER_FIX
        ),
    ),
    WarningSpec(
        code="W-0543",
        message="limits and bounds span over 13 decades; small limits may be violated",
        title="Scaling guard: wide span of limits and bounds",
        topic=_SCALING,
        body=(
            "The scaling guard checks the LP the solver gets, after FlexTool's scaling, before the solve starts. "
            "Here the limits and bounds (variable bounds and constraint right-hand sides) span more than 13 "
            "decades: the solver may violate the smallest limits by up to its feasibility tolerance, scaled. The "
            "case is the solve (with the roll or the Benders region). The run log (`_run.log` in the run's "
            "`output_parquet` folder; the debug log with `--debug`) gives the span, the largest and smallest "
            "values and where they are, and how much the smallest limit could be violated; the YAML audit "
            "`solve_data/autoscale_<solve>.yaml` in the work folder has the numbers too. The solve continues. A warning is shown once per run for each solve, Benders region and kind of finding; a "
            "later roll that gets a scale of its own shows it again.\n\n"
            "**How to fix:** a placeholder for \"no limit\" (`1e18`, `1e30`, ...): give `inf` instead. Tiny "
            "values that are really zero: give `0`, or leave the parameter out. Genuinely different scales: "
            "change units, or split the model. If the run log says that Layer 2 widened the span, "
            "`--scaling basic` avoids that. `--scaling-guard off` (`FLEXTOOL_SCALING_GUARD=off`; in the GUI: "
            "Solver options, Scaling guard: Off) hides this check."
        ),
    ),
    WarningSpec(
        code="W-0544",
        message="limits and bounds do not fit the solver's accurate range; solved anyway",
        title="Scaling guard: limits and bounds cannot be scaled",
        topic=_SCALING,
        body=(
            "The limits and bounds of the solve span more than the solver handles accurately at its feasibility "
            "tolerance after FlexTool's scaling (more than 17 decades for HiGHS), or the scale given with "
            "`--user-bound-scale` / `FLEXTOOL_USER_BOUND_SCALE` or `user_bound_scale` in the solver options puts "
            "them outside that range. Under `--scaling-guard error` (the default) this stops the solve; it shows "
            "as this warning under `--scaling-guard warn`, and always for a commercial solver (which gets no "
            "FlexTool scaling). The solve goes on and may violate the smallest limits. The case is the solve "
            "(with the roll or the Benders region). The run log (`_run.log` in the run's `output_parquet` folder; "
            "the debug log with `--debug`) gives the span, the largest and smallest values and where they are, "
            "and how much the smallest limit could be violated; the YAML audit `solve_data/autoscale_<solve>.yaml` "
            "in the work folder has the numbers too.\n\n"
            "**How to fix:** a placeholder for \"no limit\" (`1e18`, `1e30`, ...): give `inf` instead. Tiny "
            "values that are really zero: give `0`, or leave the parameter out. Genuinely different scales: "
            "change units, or split the model. If the run log says that Layer 2 widened the span, "
            "`--scaling basic` avoids that."
        ),
    ),
    WarningSpec(
        code="W-0545",
        message="objective costs span over 13 decades or near zero after scaling",
        title="Scaling guard: objective costs",
        topic=_SCALING,
        body=(
            "After FlexTool's scaling, the costs of the objective span more than 13 decades (the cheapest choices "
            "may be decided by rounding), or the smallest scaled cost is below 10 times the dual feasibility "
            "tolerance (the solver may treat it as zero). The case is the solve (with the roll or the Benders "
            "region). The run log (`_run.log` in the run's `output_parquet` folder; the debug log with `--debug`) "
            "gives the span and the objective terms at each end; the YAML audit `solve_data/autoscale_<solve>.yaml` "
            "in the work folder has the numbers too. The solve continues.\n\n"
            "**How to fix:** check the objective terms the run log names (their units, or values that are "
            "really zero). `--scaling-guard off` (`FLEXTOOL_SCALING_GUARD=off`; in the GUI: Solver options, "
            "Scaling guard: Off) hides this check."
        ),
    ),
    WarningSpec(
        code="W-0546",
        message="coefficient terms HiGHS drops or refuses",
        title="Scaling guard: coefficients HiGHS drops or refuses",
        topic=_SCALING,
        body=(
            "After FlexTool's scaling, some constraint coefficients are 1e-9 or less (HiGHS drops them from the "
            "model) or 1e15 or more (HiGHS refuses such a model). The case is the solve (with the roll or the "
            "Benders region). The run log (`_run.log` in the run's `output_parquet` folder; the debug log with "
            "`--debug`) gives the values and where they are; the YAML audit `solve_data/autoscale_<solve>.yaml` "
            "in the work folder has the numbers too. HiGHS equilibrates "
            "the matrix itself; the solve continues.\n\n"
            "**How to fix:** check the inputs behind the coefficients the run log names (their units, or "
            "values that are really zero). `--scaling-guard off` (`FLEXTOOL_SCALING_GUARD=off`; in the GUI: "
            "Solver options, Scaling guard: Off) hides this check."
        ),
    ),
    WarningSpec(
        code="W-0547",
        message="scaling guard could not run{why}; solves go on unchecked",
        title="Scaling guard could not run",
        topic=_SCALING,
        entityless=True,
        body=(
            "The scaling guard could not judge a solve: the range readout of the LP failed, or the check itself "
            "failed. The solves go on without the check. It is shown once per run; the debug log has the "
            "details.\n\n"
            "**How to fix:** nothing is needed for the solve. If it repeats, report it with the debug log "
            "(`--debug`)."
        ),
    ),
    WarningSpec(
        code="W-0550",
        message="warm-start basis not {step} for solve '{solve}'; {effect}",
        title="Warm-start basis not used",
        topic=_WARM,
        entityless=True,
        body=(
            "The run has warm start on (`--warm-start` / `FLEXTOOL_WARM_START=1`), and FlexTool could not use the "
            "cached HiGHS basis of the solve: either loading the basis into the solve failed (`not loaded`: the "
            "solve ran from scratch) or saving the basis of the finished solve failed (`not saved`: a later "
            "solve of the same model, in this run or a later one sharing the basis cache, starts without it). "
            "The results are not affected; only the solve time can be longer. The rolls of a rolling solve share "
            "one line; the debug log has the error of each.\n\n"
            "**How to fix:** usually nothing. If it happens in every run, check that the basis cache folder "
            "(`--basis-cache-dir`, by default `<work folder>/basis_cache`) is writable and its files are not "
            "damaged (deleting the folder is safe), and report it with the debug log (`--debug`)."
        ),
    ),
    WarningSpec(
        code="W-0551",
        message="save-memory mode: every solve is rebuilt and solved by HiGHS in a separate process",
        title="Save-memory mode",
        topic=_WARM,
        severity=NOTE,
        entityless=True,
        body=(
            "The run has `--save-memory` / `FLEXTOOL_SAVE_MEMORY=1` on. Each solve then builds its LP from scratch, "
            "writes it to an MPS file and lets HiGHS solve it in a separate process, so the memory HiGHS needs "
            "while solving is not added to this process's memory. An LP is not reused between solves (warm reuse "
            "is off), and each solve takes some tens of seconds more for the file I/O. The note is printed once "
            "per run.\n\n"
            "**How to fix:** nothing, if the mode was meant (for models that run out of memory otherwise). For "
            "faster solves when memory is enough, run without `--save-memory` and unset `FLEXTOOL_SAVE_MEMORY`."
        ),
    ),
    WarningSpec(
        code="W-0552",
        message="warm start is not available for solver '{solver}'; every solve is built from scratch",
        title="No warm start for this solver",
        topic=_WARM,
        severity=NOTE,
        entityless=True,
        body=(
            "Reusing the LP of an earlier solve (and the warm-start basis cache) works with HiGHS only. With "
            "another solver every solve builds its LP from scratch, so a run with many solves (for example the "
            "rolls of a rolling solve) takes longer than with HiGHS. The results are not affected. The note is "
            "printed once per run.\n\n"
            "**How to fix:** nothing, if the solver was meant. For warm reuse, use HiGHS (the solve's `solver`)."
        ),
    ),
    WarningSpec(
        code="W-0553",
        message="warm reuse is off (warm=False); HiGHS solves run in a separate process",
        title="Warm reuse off",
        topic=_WARM,
        severity=NOTE,
        entityless=True,
        body=(
            "The run was started with `warm=False` (`run_chain_from_db` / `run_orchestration` used from Python), "
            "so no LP is reused between solves. Each HiGHS solve then builds its LP from scratch, writes it to an "
            "MPS file and is solved by HiGHS in a separate process (as in save-memory mode), which keeps the "
            "memory of this process low. Each solve takes some tens of seconds more for the file I/O. The results "
            "are not affected. The note is printed once per run.\n\n"
            "**How to fix:** nothing, if it was meant. For faster solves, leave `warm` at its default (`True`)."
        ),
    ),
    WarningSpec(
        code="W-0570",
        message="benders solve '{solve}' is last and its final solve is off; no result tables written",
        title="Benders solve without results",
        topic=_OUTPUTS,
        entityless=True,
        body=(
            "The named solve is decomposed with Benders (`decomposition` = `benders`), it is the last solve of the "
            "run, and its final solve is off (`--benders-final-solve off` / `FLEXTOOL_BENDERS_FINAL_SOLVE=off`). "
            "The final solve is what solves the whole model once with the investments the Benders solve found, "
            "and it is what writes the solve's results. Without it no result table (csv, parquet, results "
            "database, plots) is written. The Benders bounds and gap are in the log, and the run ends normally.\n\n"
            "**How to fix:** drop `--benders-final-solve off` (and unset `FLEXTOOL_BENDERS_FINAL_SOLVE`) if the "
            "whole model fits in memory. If it does not, add a dispatch solve after the Benders solve (for "
            "example a rolling solve): it receives the investments, and the Benders solve then writes its "
            "investments and their costs."
        ),
    ),
    WarningSpec(
        code="W-0571",
        message="investment marginal values of benders solve '{solve}' are not in the results",
        title="Benders solve without investment marginal values",
        topic=_OUTPUTS,
        severity=NOTE,
        entityless=True,
        body=(
            "The named solve is decomposed with Benders and its final solve is off (`--benders-final-solve off` / "
            "`FLEXTOOL_BENDERS_FINAL_SOLVE=off`). The solve writes its investment and retirement decisions, their "
            "costs and its objective; its dispatch comes from the solves after it. The marginal values of the "
            "investments (the `*_invest_marginal__d` and `dual_invest_*` tables) come only from a solve of the whole "
            "model, which the final solve is, so they are not written for this solve. FlexTool does not "
            "estimate them from the decomposition.\n\n"
            "**How to fix:** drop `--benders-final-solve off` (and unset `FLEXTOOL_BENDERS_FINAL_SOLVE`) if the "
            "whole model fits in memory."
        ),
    ),
    WarningSpec(
        code="W-0572",
        message="final solve of benders solve '{solve}' {problem}",
        title="Final solve of a Benders solve failed or disagrees",
        topic=_OUTPUTS,
        entityless=True,
        body=(
            "After a Benders solve, FlexTool solves the whole model once with the Benders investments fixed (the "
            "final solve) and writes the solve's results from it. The line says what went wrong:\n\n"
            "* **failed**: the final solve did not end with an accepted solution (infeasible, unbounded, a time "
            "limit, the solver process stopped or was killed, or the whole model did not fit in memory: then "
            "the line says *out of memory*). The run still ends normally, since the Benders result itself is "
            "valid. A configuration, licence or internal error in the final solve (an option the solver "
            "refuses, a missing solver or licence) is not this warning: it stops the run with an error, as "
            "without Benders. "
            "When the Benders solve is followed by other solves, it writes its investments and their costs (as "
            "with `--benders-final-solve off`) and the later solves give the dispatch; when it is the last solve, "
            "no result table is written.\n"
            "* **objective outside the Benders bounds**: the final solve's objective is not between the lower and "
            "upper bound of the Benders solve (the line gives the three values). The results of the final solve "
            "are written. The decomposed and the whole model then do not agree on the cost of the same "
            "investments.\n\n"
            "The Benders incumbent is a feasible plan of the whole model, so the final solve should always find a "
            "solution at most the upper bound. A failure points to numerical trouble (scaling) or to a problem in "
            "the decomposition.\n\n"
            "**How to fix:** out of memory: `--benders-final-solve off` (`FLEXTOOL_BENDERS_FINAL_SOLVE=off`) "
            "skips the final solve; add a dispatch solve after the Benders solve to get results. Otherwise check "
            "the solver log of the final solve and the scaling lines (`--debug` shows "
            "more). Try the solve without decomposition (`decomposition` = `none`) on a smaller case; if the "
            "two disagree, report it with the debug log."
        ),
    ),
    WarningSpec(
        code="W-0580",
        message="rolling '{solve}': instance start unknown; storage closes at the carry",
        title="Rolling solve: start level of an instance not known",
        topic=_STORAGE,
        body=(
            "In a rolling solve, the storage nodes with `bind_within_solve` continue each roll from the level the "
            "previous roll realised, and the closing roll ends them at the level the instance (one run of the "
            "rolling solve within its upper-level solve) started from. That start level was not recorded for this instance, so the "
            "closing roll ends at the carried level instead: the storage does not return to its start over the "
            "instance. The debug log names the closing roll.\n\n"
            "**How to fix:** usually nothing; report it with the debug log (`--debug`) if the storage balance "
            "over the instance matters."
        ),
    ),
    WarningSpec(
        code="W-0581",
        message="rolling '{solve}': no start for upper-fixed storage; first roll cycles",
        title="Nested rolling solve: no start level for the first roll",
        topic=_STORAGE,
        body=(
            "The storage nodes are fixed by the upper-level solve (a nested solve), but neither that solve "
            "gives them a start level for the first roll nor do they have `storage_state_start` with a "
            "`storage_start_end_method` that fixes the start. The first roll then cycles on itself: it ends at "
            "the level it starts from. The debug log names the roll.\n\n"
            "**How to fix:** set `storage_state_start` and `storage_start_end_method` `fix_start` on the nodes, "
            "so the first roll starts from a known level."
        ),
    ),
    WarningSpec(
        code="W-0582",
        message="rolling '{solve}': period start unknown; storage closes at the carry",
        title="Rolling solve: start level of a period not known",
        topic=_STORAGE,
        body=(
            "The rolls of the rolling solve continue a period that an earlier roll started (storage nodes with "
            "`bind_within_period`, `bind_within_timeblock` or `bind_intraperiod_blocks`), but the level the "
            "node started the period at was not recorded. The roll continues the period from the carried level "
            "and closes it there, so the storage does not return to the period's start level. The debug log "
            "names the roll and the period.\n\n"
            "**How to fix:** usually nothing; report it with the debug log (`--debug`) if the storage balance "
            "over the period matters."
        ),
    ),
    WarningSpec(
        code="W-0583",
        message="storage fix_start in '{solve}' differs from the upper solve's level",
        title="Nested solve: fix_start differs from the upper plan",
        topic=_STORAGE,
        body=(
            "The storage nodes are fixed by an upper-level solve (`storage_nested_fix_method`), and the lower "
            "solve starts them from their explicit start (`storage_start_end_method` `fix_start` or "
            "`fix_start_end`, at the start of the model) instead of the level the upper solve planned. After the "
            "solve, the two start levels differ: the storage capacity or the start share is not the same in the "
            "two solves. The difference is energy the lower solve gets or loses against the upper plan. The "
            "debug log gives both levels per node (MWh).\n\n"
            "**How to fix:** check that the storage capacity at the start (existing and invested) is the same in "
            "the lower solve as in the upper-level solve."
        ),
    ),
    WarningSpec(
        code="W-0584",
        message="retirement option without existing capacity or an investment option",
        title="Retirement option with nothing to retire",
        topic=_LIFETIME,
        body=(
            "The entity has a retirement option (`invest_method`) but no `existing` capacity in any period (none "
            "given, or 0 everywhere) and no investment option. Only existing capacity and capacity the model "
            "builds can be retired, so there is no retirement variable; retired-capacity terms of user "
            "constraints and group retirement limits are 0. Each entity is named once per run.\n\n"
            "**How to fix:** give the entity `existing` capacity or an investment option, or choose an "
            "`invest_method` without retirement."
        ),
    ),
    WarningSpec(
        code="W-0585",
        message="capacity margin relative_of_peak_demand without demand; margin 0",
        title="Relative capacity margin without demand",
        topic=_INPUTS,
        body=(
            "The group has `capacity_margin_method` `relative_of_peak_demand` and a positive "
            "`capacity_margin_relative`, but its nodes have no demand in some investment period: the peak demand "
            "is 0 there, so the margin there is 0. The debug log names the periods.\n\n"
            "**How to fix:** check the demand (`inflow`) of the group's nodes in those periods, or give the "
            "margin with `capacity_margin_method` `absolute`."
        ),
    ),
    WarningSpec(
        code="W-0586",
        message="reserve group has no nodes in the solve; its reserve is left out",
        title="Reserve group without nodes",
        topic=_INPUTS,
        body=(
            "The group has a `reserve_method` other than `no_reserve`, but no node in the solve (no `group__node` "
            "row). No node of the group can hold its reserve, so its reserve requirements are left out of the "
            "solve. Each group is named once per run.\n\n"
            "**How to fix:** add the group's nodes (`group__node`), or remove the reserve from the group."
        ),
    ),
    WarningSpec(
        code="W-0587",
        message="stochastic-time {parameter} has no folded value in the solve; not applied",
        title="Stochastic-time value outside the solve's use",
        topic=_INPUTS,
        body=(
            "The entity has a stochastic-time value (a map over branches and time) of the parameter, but the "
            "solve does not use the parameter for it (the entity is outside the solve's use of the parameter), "
            "so the value has no folded row in the solve and is not applied. Each entity is named once per run "
            "and parameter.\n\n"
            "**How to fix:** nothing, if the value is not meant for this solve; otherwise check that the entity "
            "takes part in the solve with the parameter."
        ),
    ),
    WarningSpec(
        code="W-0588",
        message="realization '{solve}' keeps nothing of '{parent}': its results do not depend on it",
        title="Realization solve that inherits nothing",
        topic=_SOLVES,
        severity=NOTE,
        entityless=True,
        body=(
            "The realization solve takes from its stochastic solve the units online of the units and connections "
            "with `startup_method` *linear_fix_in_uc* or *integer_fix_in_uc*, and the investments. Here there are "
            "neither: no unit or connection with a _fix_in_uc method has units online in the realization solve "
            "(units with *linear* or *integer* are decided again with the realized data; units without online "
            "variables, or with `reduce_unit_detail_method` leaving them out, have none), and the stochastic solve "
            "decides no investments (no `invest_periods` or `realized_invest_periods`; a rolling solve builds no "
            "investment in its rolls). Storage and the other states continue from the previous realization, "
            "not from the stochastic solve, and reserve is met again with the realized data. So the realized "
            "results do not depend on the stochastic solve, and its forecasts only change its own results. The "
            "realization of a single solve dispatches its committed window with the realized data and no "
            "look-ahead; the realization added to a rolling `separate_dispatch` solve dispatches each jump with "
            "the realized data and then looks ahead with the forecasts made at the end of the jump. Two kinds of "
            "rolling solve keep a realization solve that does not look ahead with the forecasts: in a model with "
            "delayed units (note W-0594) a rolling `separate_dispatch` or `within_solve` solve, and a rolling "
            "stochastic solve followed by a realization solve you configure yourself (that solve dispatches as you configured it). "
            "The note is printed once per realization solve.\n\n"
            "**How to fix:** nothing, if that is meant. For a commitment made in the stochastic solve that holds "
            "in the realization, give those units `startup_method` *integer_fix_in_uc* or *linear_fix_in_uc*. "
            "For a rolling solve that should keep nothing of a separate unit-commitment solve, use "
            "`dispatch_realization_method` *within_solve*: each roll is then one solve (not in a model with "
            "delayed units, note W-0594)."
        ),
    ),
    WarningSpec(
        code="W-0589",
        message="one-variable connection: efficiency converts units, the same both ways",
        title="Efficiency of a one-variable connection as a unit conversion",
        topic=_INPUTS,
        body=(
            "A one-variable connection (`transfer_method` *no_losses_no_variable_cost*) carries both directions "
            "in one signed flow. Its two nodes have different `unit_of_measure` values, so its `efficiency` "
            "(other than 1) is used as a unit conversion (e.g. MWh to t): the flow f arriving at the second node "
            "takes f / efficiency from the first node, and in the reverse direction the first node gets f / "
            "efficiency for the f taken from the second. The same conversion holds both ways, so it cannot model "
            "losses. The debug log gives the efficiency of each connection. The warning is printed once per "
            "solve.\n\n"
            "**How to fix:** nothing, if the efficiency is a unit conversion. For losses, use `transfer_method` "
            "*regular* or *exact*."
        ),
    ),
    WarningSpec(
        code="W-0590",
        message="one-variable connection has other_operational_cost; ignored",
        title="Variable cost of a one-variable connection",
        topic=_INPUTS,
        body=(
            "A one-variable connection (`transfer_method` *no_losses_no_variable_cost*, and the lines of a "
            "`dc_power_flow_with_angles` group) carries both directions in one signed flow, so a cost on that "
            "flow would be a cost in one direction and a revenue in the other. Its `other_operational_cost` is "
            "ignored (treated as zero) in the solve and in the cost results. The warning is printed once per "
            "solve.\n\n"
            "**How to fix:** use `transfer_method` *variable_cost_only*, *regular* or *exact* for a connection "
            "with a variable cost, or remove the cost."
        ),
    ),
    WarningSpec(
        code="W-0591",
        message="cannot model losses with one flow variable; efficiency ignored",
        title="Efficiency of a one-variable connection ignored",
        topic=_INPUTS,
        body=(
            "A one-variable connection (`transfer_method` *no_losses_no_variable_cost*, and the lines of a "
            "`dc_power_flow_with_angles` group, which use it internally) carries both directions in one signed "
            "flow, so it cannot model losses: a loss in one direction would be a gain in the other. Its "
            "`efficiency` is ignored (treated as 1) on a line of a `dc_power_flow_with_angles` group, and on a "
            "*no_losses_no_variable_cost* connection unless both its nodes have a `unit_of_measure` and the two "
            "differ (then the efficiency is a unit conversion: W-0589). The debug log gives the efficiency that "
            "was ignored. The warning is printed once per solve.\n\n"
            "**How to fix:** for losses, use `transfer_method` *regular* or *exact* (for a DC power-flow line, "
            "accept a lossless line), or remove the efficiency."
        ),
    ),
    WarningSpec(
        code="W-0592",
        message="an outage ended an obligation carried over from an earlier solve",
        title="Outage ends a carried obligation",
        topic=_SOLVES,
        severity=NOTE,
        body=(
            "An earlier solve started these units, or left them ramping down. Their minimum up-time, or "
            "ramp-down limit, would have kept them running into time steps where they cannot run: their "
            "availability is below their `min_load` there (for the ramp-down limit, the flow their capacity "
            "allows there is below what the limit asks for). That solve could not see the outage, because it lay "
            "beyond its horizon or only the realized data have it. FlexTool ends the carried obligation at the "
            "first such step, and the rest of its minimum up-time is dropped. A realization solve keeps the units "
            "online its stochastic solve committed up to such an outage and decides the unit again from the outage "
            "on. A solve that sees an outage coming still keeps the minimum up-time: it does not start a unit it "
            "cannot keep. The debug log gives the solve and the step of each unit. The note is printed once per "
            "run.\n\n"
            "**How to fix:** nothing, if the outage could not be foreseen. If it was planned, make sure the "
            "solve that starts the unit sees it: give it in the stochastic solve's forecasts, or lengthen "
            "`rolling_solve_horizon`."
        ),
    ),
    WarningSpec(
        code="W-0593",
        message="solve '{solve}' has dispatch_realization_method deterministic: forecasts not solved",
        title="Deterministic solve with stochastic branches",
        topic=_SOLVES,
        entityless=True,
        body=(
            "The solve is a dispatch solve with `stochastic_branches`, and its `dispatch_realization_method` is "
            "`deterministic` (the default of a dispatch solve since FlexTool 4.7.0; an investment solve, a solve "
            "that is not rolling and has `invest_periods` or `realized_invest_periods`, does not read the method "
            "and always solves its forecasts). It is solved with the data of the `realized: yes` "
            "branch only, as one "
            "timeline: each time step reads the realized series that covers it (where several do, the one made "
            "nearest to the start of the solve, or of the roll in a rolling solve). The forecasts "
            "and the realization solve are ignored. Without a "
            "`realized: yes` branch, or at a time step no realized series covers, the run stops with an error.\n\n"
            "**How to fix:** set `dispatch_realization_method` to `within_solve` or `separate_dispatch` to solve "
            "the forecasts."
        ),
    ),
    WarningSpec(
        code="W-0639",
        message="investment solve '{solve}': investments branch at its first period (expert option)",
        title="Investments branch at the first period",
        topic=_SOLVES,
        entityless=True,
        body=(
            "The solve is an investment solve with `invest_solve_branching_method` "
            "`investments_at_reveal_period`, and its forecasts start at its first period. Every forecast then "
            "invests and operates on its own from the start, as one deterministic solve per forecast weighted "
            "together (the wait-and-see value of the forecasts). No decision is shared, so nothing can be "
            "realized: the solve gets no realization dispatch, and the normal results hold no investment or "
            "operation of the forecasts. They are written only by the `model.output_horizon` dump.\n\n"
            "**How to fix:** nothing, if the wait-and-see value is intended. To decide one investment for every "
            "forecast at the first period, set `investments_after_reveal_period` (or leave the setting out: "
            "`operations_only`); to learn after a first period, start the forecasts at a later period."
        ),
    ),
    WarningSpec(
        code="W-0640",
        message="investment solve '{solve}': dispatch_realization_method separate_dispatch ignored",
        title="separate_dispatch on an investment solve",
        topic=_SOLVES,
        entityless=True,
        body=(
            "The solve is an investment solve (not rolling, with `invest_periods` or `realized_invest_periods`), "
            "and it sets `dispatch_realization_method` `separate_dispatch`, which applies to dispatch solves only. "
            "An investment solve does not read `dispatch_realization_method`: it solves its forecasts under its "
            "`invest_solve_branching_method`, and with a `realized: yes` branch it gets a realization dispatch of "
            "its own. The value is ignored.\n\n"
            "**How to fix:** remove `dispatch_realization_method` from the solve. To dispatch in two steps, put a "
            "dispatch solve after it in `model.solves`: no `invest_periods`, `realized_periods` = the periods to "
            "dispatch, `stochastic_branches` with a `realized: yes` branch, `dispatch_realization_method` "
            "`separate_dispatch`; leave those periods out of the `realized_periods` of the investment solve. The "
            "dispatch solve starts from the capacity the investment solve realized."
        ),
    ),
    WarningSpec(
        code="W-0594",
        message="solve '{solve}' has delayed units: each jump is realized without look-ahead",
        title="Realization without look-ahead: delayed units",
        topic=_SOLVES,
        severity=NOTE,
        body=(
            "The solve is a rolling solve with `stochastic_branches` and `dispatch_realization_method` "
            "`within_solve` or `separate_dispatch`, and the model has units with a `delay`. Such a solve realizes "
            "each jump in a realization solve that dispatches it on the realized data, continuing from the "
            "previous realization and keeping what it inherits from the roll's stochastic solve (note W-0588): the "
            "realization itself does not look past the jump with the forecasts made at its end. The look-ahead "
            "is not available with delays: within the roll a delay wraps the last time steps' flows back to its "
            "first ones, and the forecasts that follow the jump would all wrap into the same first time steps. The "
            "note names the delayed units and is printed once per run.\n\n"
            "**How to fix:** nothing, if that is meant. For the look-ahead, model the units without `delay`."
        ),
    ),
) + BEHAVIOUR_SPECS
