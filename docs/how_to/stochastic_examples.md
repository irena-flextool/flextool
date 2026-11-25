# Stochastic examples

Two small models in `stochastics.sqlite` that put the stochastic settings side by side: the realization methods of a stochastic rolling dispatch, and the investment choices of a stochastic investment solve. [Stochastics](stochastics.md) explains every setting used here.

## How to compare the realization methods of a stochastic rolling dispatch
**(stochastics.sqlite scenarios: rolling_deterministic, rolling_within_solve, rolling_separate_dispatch, rolling_perfect_within_solve, rolling_perfect_separate_dispatch)**

The model (alternative *rolling_demo*): three periods *period1* - *period3* of three days each (216 hours, one time step per hour), each representing one year, and one rolling solve *rolling_dispatch* with `rolling_solve_jump` 24 and `rolling_solve_horizon` 48: each roll dispatches a day and looks a day ahead. One node *elec* with a daily demand curve (peak at 19:00); *wind_farm* 650 MW on the stochastic profile *wind_cf*; *coal_units* 3 x 100 MW (`startup_method` *integer_fix_in_uc*, minimum up time 8 h, start-up 5000 EUR a unit); *gas_peaker* 350 MW (`startup_method` *linear*); *battery* 400 MWh with the 100 MW *battery_inverter*. The wind farm and the battery are in the `include_stochastics` group *rolling_stochastic*.

The wind profile is a (branch, analysis_time, time) map. At the start of each day (t0001, t0025, ... t0193) there are two forecasts, *high* and *low*, of the next 48 hours, and the actual wind of that day in branch *actual*. The solve's `stochastic_branches` gives, at each of these times, *high* and *low* with weight 0.5 each (`realized: no`) and *actual* (`realized: yes`). The scenarios differ only in `dispatch_realization_method`, and two of them in the forecasts:

- **rolling_deterministic** (rolling_demo): `deterministic`. The forecasts are not solved: each roll reads the actual wind as one timeline, also in its look-ahead, so it knows the next day's wind exactly. This is the perfect-foresight benchmark of the rolling solve; warning W-0593 says that the forecasts are ignored.
- **rolling_within_solve** (rolling_demo, within_solve): one solve per roll. The day is dispatched once, on the actual wind, under the period's own name; from the end of the day the roll looks ahead with the forecasts made then, each in its own branch period (*period1_high*, *period1_low*, ...). The day's units online and battery use hedge against both forecasts of the next day.
- **rolling_separate_dispatch** (rolling_demo, separate_dispatch): two solves per roll. *rolling_dispatch_roll_k* is stochastic from the roll's first hour on the forecasts made then, and ties across the forecasts only the units online of *coal_units* (*integer_fix_in_uc*) over the day. *rolling_dispatch__realized_dispatch_roll_k* keeps that commitment, dispatches the day on the actual wind and looks ahead with the forecasts made at the end of the day. The peaker (*linear*) and the battery are decided again there.
- **rolling_perfect_within_solve** and **rolling_perfect_separate_dispatch** (rolling_demo, perfect_forecast, the method): alternative *perfect_forecast* replaces the forecasts with one forecast *exact* (weight 1) equal to the actual wind. Both methods then give the deterministic result, roll by roll.

Results over the 216 hours. They are the realized operation only: the look-ahead of the forecasts is not in the results. The cost is the fuel, start-ups, battery inverter and unserved energy of the realized hours (the sum of `costs__dt`). Starts and energies are over the simulated hours.

| scenario | solves | cost (k EUR) | coal starts | peaker starts (350 MW each) | battery discharged (MWh) | battery charged (MWh) | coal (GWh) | peaker (GWh) | wind (GWh) | unserved (MWh) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rolling_deterministic | 9 | 1,853.6 | 10 | 2.0 | 2,682 | 3,190 | 41.05 | 6.95 | 65.95 | 0.0 |
| rolling_within_solve | 9 | 1,854.1 | 10 | 2.1 | 2,652 | 3,155 | 41.05 | 6.95 | 65.95 | 0.0 |
| rolling_separate_dispatch | 18 | 1,876.9 | 11 | 2.1 | 2,280 | 2,714 | 41.32 | 6.93 | 65.63 | 0.0 |
| rolling_perfect_within_solve | 9 | 1,853.6 | 10 | 2.0 | 2,682 | 3,190 | 41.05 | 6.95 | 65.95 | 0.0 |
| rolling_perfect_separate_dispatch | 18 | 1,853.6 | 10 | 2.0 | 2,682 | 3,190 | 41.05 | 6.95 | 65.95 | 0.0 |

*rolling_within_solve* dispatches each day on the actual wind and is uncertain only about the next day, so it comes close to the benchmark. *rolling_separate_dispatch* commits the coal units for the day before it sees the day's actual wind, so it starts a coal unit more and uses the battery less. With a perfect forecast both methods give the benchmark exactly.

## How to compare stochastic investment choices
**(stochastics.sqlite scenarios: invest_operations_only, invest_after_reveal_period, invest_then_dispatch)**

The model (alternative *invest_demo*): two periods *y2030* and *y2035* of four hours each, each representing 5 years (`years_represented`), no discounting (`social_discount_rate` 0). One node *town* whose demand is 100 MW in y2030; in y2035 it is 160 MW in forecast *high* and 120 MW in forecast *low* (weight 0.5 each). The demand is a stochastic `inflow` (branch, analysis_time, time) with every forecast made at t0001, and *town* is in the `include_stochastics` group *invest_stochastic*. The only supply is *new_plant*, built by the model (`invest_method` *invest_no_limit*): `invest_cost` 500 EUR/kW in y2030 and 1500 EUR/kW in y2035, `discount_rate` 0, `lifetime_technical` 10 years, no operating cost. Unserved energy costs 3000 EUR/MWh, so demand is always met.

The solve *invest_plan* (`solve_mode` *single_solve*) invests in both periods (`invest_periods`) and realizes y2030 only (`realized_invest_periods` and `realized_periods`): y2035 is look-ahead. Its forecasts start at y2030 (`stochastic_branches` at t0001), with branch *actual* (`realized: yes`, the high demand) for the realization dispatch. The uncertainty is about the 2035 demand: the forecasts do not differ in 2030.

With a `discount_rate` of 0 a plant pays `invest_cost` x 1000 / `lifetime_technical` a year for each modelled year it serves (see [Investment parameters](investment.md)). A MW built in 2030 serves both periods, 10 years: 500 x 100 x 10 = 500 000 EUR. A MW built in 2035 serves 5 modelled years: 1500 x 100 x 5 = 750 000 EUR. Building early is cheaper per MW, but a plant built early for a future that does not come is wasted.

- **invest_operations_only** (invest_demo): `invest_solve_branching_method` is not set, so it is `operations_only`: every period's investment is one decision for both forecasts. The y2035 fleet must serve the high demand, and building it in 2030 is cheaper, so the solve builds 160 MW in 2030: 160 x 500 000 = 80.0 M EUR.
- **invest_after_reveal_period** (invest_demo, after_reveal_period): `investments_after_reveal_period`. The y2030 investment is still one decision for both forecasts, but the y2035 investment follows each forecast. The solve builds 120 MW in 2030 (the low demand) and, in the forecast *high*, 40 MW more in 2035: 120 x 500 000 + 0.5 x 40 x 750 000 = 75.0 M EUR. Building a 121st MW in 2030 would cost 500 000 EUR and save 750 000 EUR only in the forecast *high*, worth 375 000 EUR. Being able to adapt later is worth 80.0 − 75.0 = 5.0 M EUR here.
- **invest_then_dispatch** (invest_demo, after_reveal_period, invest_then_dispatch): the structure that warning W-0640 points to, an investment solve followed by a dispatch solve. `model.solves` is *invest_first*, *dispatch_2030*. *invest_first* is *invest_plan* with `investments_after_reveal_period`, no `realized_periods` and no realized branch: it only decides and realizes the y2030 investment. *dispatch_2030* is a dispatch solve of y2030 (no `invest_periods`; `realized_periods` y2030) with its own `stochastic_branches` (the forecasts and *actual*) and `dispatch_realization_method` *separate_dispatch*. It starts from the 120 MW that *invest_first* realized. Its forecasts do not differ in 2030, and *new_plant* has no units online to commit, so the result equals the realization dispatch of *invest_after_reveal_period*; note W-0588 says that the realization keeps nothing of the stochastic solve. The structure pays off when the dispatch step differs from the investment solve, for example with a finer time resolution, a full year after representative periods, or forecasts made closer to the operation.

| scenario | solves | objective of the investment solve (M EUR) | y2030 investment (MW) | its yearly cost (M EUR) | y2030 dispatch (MWh) |
|---|---|---:|---:|---:|---:|
| invest_operations_only | invest_plan, invest_plan__realized_dispatch | 80.0 | 160 | 16.0 | 400 |
| invest_after_reveal_period | invest_plan, invest_plan__realized_dispatch | 75.0 | 120 | 12.0 | 400 |
| invest_then_dispatch | invest_first, dispatch_2030, dispatch_2030__realized_dispatch | 75.0 | 120 | 12.0 | 400 |

The results hold the realized y2030 investment and the y2030 dispatch. The y2035 investments are look-ahead, and under `investments_after_reveal_period` they differ per forecast, so they are only in the `model.output_horizon` dump. To decide them, add a later solve for y2035 (see "Investing year by year under uncertainty" in [Stochastics](stochastics.md#investing-year-by-year-under-uncertainty-fan-realize-next-year)). For a perfect-foresight benchmark, run a scenario without `stochastic_branches` and with the actual demand as a plain `inflow`.
