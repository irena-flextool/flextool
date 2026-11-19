"""The solve-side warning codes (W-0500..W-0999) of the registry :mod:`flextool.run_warnings.registry`: the checks of
the engine (``flextool/engine``: solves, the preprocessing of a solve, the model build, the outputs of a solve).
Same rules as the registry's own codes (code order, a short lowercase ``message``, a ``body`` with **How to fix:**);
the registry merges them.

* W-050x solves: realization, carried states, look-backs
* W-051x inputs of a solve: inflow scaling, limits, maps, time steps
* W-052x lifetimes, investments and units
* W-053x outputs of a solve and storage notes
* W-054x what the solver's outcome means (notes after a solve that ended optimal, but not cleanly)
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
            "The unit or connection has integer online variables (whole units), but its existing capacity divided "
            "by its unit size is not a whole number of units in some period. Integer online units cannot use the "
            "fraction (below 1 the unit can never be online, which can make a minimum such as `min_instant_flow` "
            "infeasible). Without `virtual_unitsize`, one unit is the largest existing capacity over the periods. "
            "The debug log gives the counts per period.\n\n"
            "**How to fix:** set `virtual_unitsize` to the size of one unit and `existing` to the number of "
            "units, or use `startup_method` `linear`."
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
        code="W-0540",
        message="solve '{solve}' optimal by HiGHS at up to {factor}x its relative tolerances",
        title="Optimal within a multiple of the relative tolerances",
        topic=_SOLVER,
        severity=NOTE,
        entityless=True,
        body=(
            "HiGHS called the solve optimal, although its solution misses some of HiGHS's tolerances (by default "
            "1e-7) also in relative terms, by up to the factor given (HiGHS's own log says `Model status is "
            "\"Optimal\" since relative violation of tolerances is no more than X`). This happens after an "
            "interior-point solve without crossover (`solver` = `ipm`, `run_crossover` = `off`), and the factor "
            "tells how serious it is: a factor up to about 10 means the flows, storage levels and costs are "
            "accurate to about a millionth (1e-6) of their size, which is far below the accuracy of any input "
            "data: the results are usable. The duals (prices) are less exact than the primal values: small "
            "differences in prices between equivalent nodes or time steps can be numerical noise. The debug log "
            "names the checks and their largest violations. Each solve is noted once; a later roll of a rolling "
            "solve with the same outcome goes to the debug log.\n\n"
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
            "The note is printed **once per run**, after the first interior-point solve without crossover whose "
            "solution HiGHS reports as not certainly clean; it covers the same HiGHS warnings of the later solves "
            "of the run, which go to the debug log only. FlexTool cannot read HiGHS's log, and when polar-high's "
            "autoscale has scaled the problem (`user_objective_scale` / `user_bound_scale` in the HiGHS log), "
            "HiGHS keeps only part of these checks in what it reports to FlexTool, so the note can appear also "
            "when HiGHS printed none of these warnings. When the reported checks are complete and clean, there is "
            "no note. A solve whose relative tolerances are missed gets its own note, W-0540.\n\n"
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
) + BEHAVIOUR_SPECS
