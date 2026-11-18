# Advanced Concepts

## Penalties

FlexTool uses penalty (slack) variables to relax constraints that might otherwise cause infeasibility. Each penalty has a very high cost in the objective function that makes the optimizer try to avoid these violations unless no other option is viable.

### How penalties work

When a constraint cannot be satisfied (e.g., demand exceeds available supply), a slack variable absorbs the deficit. The cost of this violation is added to the objective function. The penalty cost should therefore be set high enough to prevent violations unless there is genuinely no feasible alternative, but not so high that it causes numerical issues. The penalty could also be based on perceived cost of the violation: e.g. value of lost load in case of node balance violation.

### Penalty scaling with time

Most penalties scale with the duration of the violation -- a 3-hour violation costs 3 times as much as a 1-hour violation. The table below summarizes how each penalty is treated:

| Penalty parameter | Unit | Constraint type | Scales with step_duration | Notes |
|---|---|---|---|---|
| `penalty_up` (node) | CUR/MWh | Energy balance | Yes | Cost per MWh of energy not served |
| `penalty_down` (node) | CUR/MWh | Energy balance | Yes | Cost per MWh of excess energy |
| `penalty_reserve` (reserve group) | CUR/MW | Reserve requirement | Yes | Cost per MW-hour of reserve shortfall |
| `penalty_inertia` (group) | CUR/MWs | Inertia requirement | Yes | Cost per MWs-hour of inertia shortfall |
| `penalty_non_synchronous` (group) | CUR/MWh | Non-synchronous limit | Yes | Cost per MWh of non-sync limit violation |
| `penalty_capacity_margin` (group) | CUR/kW | Capacity margin | No | Cost per kW of capacity shortfall per period. Analogous to investment cost -- but not annualized. It is still discounted like an operating cost. |

### Capacity margin penalty

The capacity margin penalty is different from the others: it represents the cost of not having sufficient installed capacity. It is analogous to an investment cost (CUR/kW) but is NOT annualized over a lifetime like actual investments. Instead, it applies as a lump cost per period. This means a `penalty_capacity_margin` of 1000 CUR/kW is much more expensive than an `invest_cost` of 1000 CUR/kW (which would be annualized to roughly 50-100 CUR/kW/year depending on lifetime and discount rate). Set the penalty to reflect the annual cost of capacity shortfall, not the total investment cost.

## Economic Modelling

### Overview

FlexTool minimizes total system cost over a planning horizon. The economic framework uses two kinds of rate:

- **`social_discount_rate`** (model-level): the model's time discount rate. It weighs a cost or income in a later year less than the same amount now.
- **`discount_rate`** (per technology): the investor's rate (WACC). It turns an `invest_cost` into yearly payments and reflects the financing cost and risk of each investment.

### Real vs nominal values

Economic values can be expressed in two ways:

- **Real (constant prices)**: All values are expressed in today's money. A power plant that costs 1000 EUR/kW today still costs 1000 EUR/kW in 2040 in real terms -- the purchasing power is the same.
- **Nominal (current prices)**: Values include inflation. The same plant might cost 1300 EUR/kW in 2040 at 2% annual inflation, even though its real cost hasn't changed.

**Example**: A coal plant costs 1500 EUR/kW today. At 2% inflation:

- Real cost in 2035: 1500 EUR/kW (unchanged)
- Nominal cost in 2035: 1500 x (1.02)^10 = 1829 EUR/kW approximately

**Rule**: Never mix real and nominal values in the same model. Both rates are given in the same terms as the cost inputs: real rates for inputs in real prices, nominal rates for nominal inputs.

### The social discount rate

The model-level `social_discount_rate` (no value: 0) weighs every cost and income by `1 / (1 + social_discount_rate)^t`, where `t` is the years from the start of the run to the moment the cost falls. It applies uniformly to all cost types: investment payments, fuel costs, fixed O&M, penalties, salvage values, etc.

- **0** (no value): no time preference. A cost in a later period weighs the same as the same cost now. With real-price inputs, inflation is already in the prices.
- **Above 0**: later costs and incomes weigh less, the planner's (or society's) preference for now over later; the discounted costs are present values at that rate. With real inputs, give the real social rate (for example 0.03). With nominal inputs, use the combined rate (1 + inflation) × (1 + social rate) − 1: for 2 % inflation and a 3 % social rate, 1.02 × 1.03 − 1 = 0.0506.

The yearly investment payments fall at the start of each year by default (the annual payment is sized for that, see below) and operating costs at the middle of the year: `discount_offset_investment` (default 0) and `discount_offset_operations` (default 0.5) move them within the year.

### Discounting between milestone years

**Only `social_discount_rate` discounts.** FlexTool discounts the costs of later periods (milestone years) with the `model` parameter `social_discount_rate` and nothing else. Each period's costs are multiplied by a discount factor: the sum, over the years the period represents, of `1 / (1 + social_discount_rate)^years_from_start`. The per-technology `discount_rate` does not discount between periods: it only turns an `invest_cost` into a yearly payment (see [The discount rate](#the-discount-rate-per-technology)).

**How the years are counted.** The years from the start of the run come from `years_represented`: a period starts after the years that the periods before it represent. The period names are only labels, so *p2030* is not read as the year 2030. Give each milestone the number of years until the next milestone (for example 5 for *p2025* when the next period is *p2030*). Investment payments are discounted from the start of a year (`discount_offset_investment`, default 0), operating costs from the middle of it (`discount_offset_operations`, default 0.5).

**With `social_discount_rate` 0 there is no discounting.** A cost in 2040 then weighs the same as the same cost in 2025: each period's yearly costs are only multiplied by its `years_represented`. This is right for a model that should not prefer earlier or later costs, but it is not what most planning studies want. Keep the per-technology `discount_rate` at the cost of capital of each technology in either case.

**Where to see it.** `summary_solve.csv` lists the *Operational discount factor* and the *Investment discount factor* of each period. The cost results come both annualized (one year of the period, not discounted) and discounted (present value over the years the periods represent); see [Costs](results.md#costs).

### The discount rate (per technology)

Each unit, connection, and storage node can have its own `discount_rate` parameter (default: 0.05, i.e., 5%). This represents the **weighted average cost of capital (WACC)** -- the return that investors require to finance the technology.

The discount rate converts a lump-sum investment cost into annual payments over the technology's payback time (`payback_time`, else its `lifetime_technical`) using the standard annuity formula:

```
annual_payment = invest_cost * discount_rate / (1 - (1 / (1 + discount_rate))^payback_time)
                 * (1 + discount_rate)^(discount_offset_investment - 1)
```

The loan is taken when the capacity comes online (`invest_cost` is the overnight cost) and each yearly payment falls
`discount_offset_investment` years into its service year: at the default 0 at the start of each year, the first one on
the day the capacity comes online; at 1 at the end of each year. The last factor sizes the payments for that moment,
so the payments are worth exactly `invest_cost` at the `discount_rate` whatever the offset. For example, invest cost
1000, discount rate 5 % and payback time 10 years give 129.50 a year paid at year ends (offset 1) and 129.50 / 1.05 =
123.34 a year paid at year starts (offset 0); both are worth 1000 when discounted at 5 % from the moment the capacity
comes online. The same factor sizes the level salvage credit (below).

With `lifetime_method` *reinvest_choice* or *no_reinvestment* the plant pays a level yearly cost over its real lifetime: the present value of the `payback_time` annual payments, discounted at the model `social_discount_rate`, spread evenly over the `lifetime_technical` years, `annual_payment · G(payback_time) / G(lifetime_technical)` with `G(x)` the discounted sum of x years (a fractional last year counts its fraction). It is paid for each year the capacity is in service, from the year it starts to operate to its end of life as `end_of_life_method` rounds it: a 12-year plant whose life is rounded to 10 years pays the yearly cost of the 12-year plant for 10 years, and one rounded to 15 years for 15. The price does not depend on when the plant is built or on the period lengths. `payback_time` and `discount_rate` set only its size: with `discount_rate` equal to `social_discount_rate` the yearly cost is the annuity over `lifetime_technical` whatever the payback time, and with a lower `discount_rate` (an investor who demands faster payback) a shorter payback time costs more. With *reinvest_automatic* the annual payment is made in every year to the end of the model horizon. Years after the model horizon are not counted.

The `net_salvage_value` at the end of the real lifetime lowers that yearly cost: its present value at `discount_rate` levelled over the `lifetime_technical` years, `net_salvage_value · (1 + discount_rate)^−lifetime_technical · annual factor`, the same annual factor as the investment (with `payback_time` equal to `lifetime_technical`, the textbook capital recovery with salvage). Existing capacity, whose investment is sunk, holds its salvage value as the same level credit over its service years from the model start. A retirement receives the whole `net_salvage_value` at the start of the retirement period and gives the credits back. Since years after the horizon are not counted, a retirement late in the horizon collects a salvage value whose credits a kept plant would earn mostly after the horizon, so it can pay; this is a consequence of the finite horizon, not a model error.

A `discount_rate` of 0 gives the straight-line annuity `invest_cost / payback_time` (no interest). Without a value the rate is 0.05; a negative rate is an input error. Neither lifetime has a default: an entity that invests or retires with a cost (`invest_cost`, `net_salvage_value`) needs a `payback_time` or a `lifetime_technical`, and one whose `lifetime_method` is `reinvest_choice` or `no_reinvestment` needs a `lifetime_technical` (also for existing capacity without investment), since the capacity then ends after it.

An `invest_method` other than `not_allowed` offers investment (retirement) in every investment period, also without an `invest_cost` (`net_salvage_value`) or where it is 0: that capacity has no annuity, and its `fixed_cost` still counts (a retirement saves it). A 0 in an `invest_cost` map therefore does not block investment in that period; to forbid investment in a period, use a period method (`invest_period`, `invest_period_total` or their `invest_retire_` forms) and set `invest_max_period` 0 for it; the other methods do not use `invest_max_period`. With neither `invest_cost` nor `fixed_cost` any invested amount is optimal, so the result is arbitrary. A zero-cost investment needs a `lifetime_technical` only with `lifetime_method` `reinvest_choice` or `no_reinvestment`.

**Example**: A solar plant costs 800 EUR/kW with a 25-year lifetime and 5% discount rate, paid at the start of each year (the default `discount_offset_investment` 0):
```
annual_payment = 800 * 0.05 / (1 - 1/1.05^25) * 1.05^(0 - 1) = 800 * 0.0710 / 1.05 = 54.1 EUR/kW/year approximately
```

Different technologies can have different discount rates reflecting their risk profiles:

- Low risk (e.g., established solar PV): 3-5%
- Medium risk (e.g., natural gas turbine): 5-7%
- High risk (e.g., novel technology): 8-12%

### Consistency between rates

The `discount_rate` and the `social_discount_rate` must use the same price basis as the cost inputs of the model instance (either real or nominal):

| Inputs | social_discount_rate | discount_rate (entity) |
|--------|---------------|----------------------|
| Real (constant prices) | 0, or the real social rate (e.g., 3%) | Real WACC (e.g., 5%) |
| Nominal (current and future prices) | (1 + inflation) × (1 + real social rate) − 1 (e.g., 5.06% at 2% and 3%) | Nominal WACC (e.g., 7%) |

The relationship between real and nominal rates follows the Fisher equation:
```
(1 + nominal_rate) = (1 + real_rate) * (1 + inflation)
```
Approximation: nominal = real + inflation approximately (e.g., 5% + 2% = 7%).

### Fixed costs

Fixed costs (`fixed_cost` parameter on units, connections, and nodes) represent annual operation and maintenance costs that are incurred regardless of how much the asset is used. They are expressed in CUR/kW/year (or CUR/kWh/year for storage).

For **existing** assets, fixed costs are applied each year, discounted like the operating costs.

For **invested** assets, the fixed cost is paid in every period the new capacity operates within the model horizon: until its `lifetime_technical` ends, or in every later period with `lifetime_method` *reinvest_automatic* (the same years as the yearly investment cost, see above). It is the `fixed_cost` of each of those periods, as for existing capacity (a period map can let it rise with the age of the fleet), discounted like the other costs of that period. Years after the model horizon are not counted, the same as for the investment annuity, so the investment decision weighs the costs of the years the model covers. In the results, these costs are reported as a separate line item (*fixed cost invested*) from the investment annuity.

### Summary of economic parameters

| Parameter | Level | Default | Description |
|-----------|-------|---------|-------------|
| `social_discount_rate` | Model | 0 | The model's time discount rate; 0 is no time preference |
| `discount_offset_investment` | Model | 0 | Where in the year the yearly investment payments and salvage credits fall (0 = start, 1 = end); the annual payment is sized for it |
| `discount_offset_operations` | Model | 0.5 | Where in the year operating and fixed costs fall (0.5 = middle) |
| `discount_rate` | Entity | 0.05 | Technology-specific WACC for annualizing investments; 0 gives invest_cost / payback_time |
| `invest_cost` | Entity | - | Overnight investment cost [CUR/kW] |
| `lifetime_technical` | Entity | - | How long the capacity operates [years]; required when the entity invests or retires with a cost and has no `payback_time` |
| `payback_time` | Entity | `lifetime_technical` | Years over which `invest_cost` is annualized |
| `fixed_cost` | Entity | - | Annual fixed O&M cost [CUR/kW/year] |
| `net_salvage_value` | Entity | - | Scrap value minus decommissioning cost when the capacity stops operating [CUR/kW] |

### How multi-year periods are handled internally

A period can represent multiple years via the `years_represented` parameter on the solve entity. For example, a period `p2025` with `years_represented = 5` means that the modelled timeline (e.g., one representative week) stands in for 5 full years (2025-2029).

Internally, FlexTool expands this into per-year entries. A period with `years_represented = 5` becomes 5 rows in the solve data, each representing 1 year at a different distance from the start of the run (the first period of the model's first solve; a later solve counts the years of the earlier solves' periods first). The discount factor then sums the per-year discount factors:

```
discount_factor = Σ_y  p_years_represented[d,y] × 1/(1 + social_discount_rate)^(years_from_run_start[d,y] + p_years_represented[d,y] × offset)
```

where `years_from_run_start[d,y]` is the start of year `y` of period `d` (a fractional year counts as its fraction: after a period of 2.5 years the next one starts at 2.5) and `offset` is `discount_offset_investment` or `discount_offset_operations`.

With `social_discount_rate = 0`, each term contributes `p_years_represented[d,y] × 1 = 1`, and the sum equals `years_represented` (i.e., 5). With `social_discount_rate = 2%`, each later year is discounted slightly more, giving a sum of approximately 4.81 (at offset 0).

This mechanism also supports fractional values. With `years_represented = 0.5` (half-year resolution for investment decisions), the discount factor is `0.5 × 1/(1 + social_discount_rate)^y`, correctly counting only half a year of costs.

In the objective function, operational costs follow this pattern:

```
cost_per_timestep × step_duration × discount_factor / period_share_of_year
```

where `period_share_of_year = hours_in_period / 8760` converts the modelled period to an annual equivalent, and the discount factor then scales across all represented years. Investment costs follow a similar pattern using a separate discount factor that accounts for investment timing (`discount_offset_investment`).

### Using periods that are not one year long

A period does not have to represent exactly one year. It can be shorter
(e.g. a half-year period for six-monthly investment decisions) or stand in
for several years (see *How multi-year periods are handled internally*).
FlexTool keeps the accounting correct for any period length — you mostly
just need to know what is scaled and what is not.

#### What a period's length means

Each period has a length derived from its modelled timeline,
`period_share_of_year = hours_in_period / 8760`. Investment granularity is set separately
with `years_represented` on the solve — including fractional values such
as `years_represented = 0.5` for half-year steps.

#### What stays correct automatically

- **Operating costs and energy balances are absolute.** They are summed
  over the timesteps actually in the period, so a shorter period simply
  incurs less; a longer one, more. No year constant enters these totals.
- **Annual (per-year) parameters are prorated by period length.** Annual
  `fixed_cost`, for instance, is applied as `fixed_cost × period_share`, so
  a half-year period pays half a year of fixed cost. Investment annuities
  and discounting likewise use the period's (possibly fractional) year
  count.

#### What "annualised" in the results means

The `..._annualized` result columns divide a period's totals by
`period_share_of_year` to express them as a per-year rate. This is only a
reporting convenience so that periods of different lengths can be compared
on the same footing — it does **not** change the optimisation or the
period totals. If you want the actual amount incurred in the period, read
the plain (non-annualised) total.

#### When NOT to rely on inflow scaling

The `inflow_method` options `scale_to_annual_flow`, `scale_in_proportion`
and `scale_to_annual_and_peak_flow` assume the modelled timeline is a
representative year: they normalise the inflow by its sum over (one year
of) the timeline. If your timeline is **not** a representative year — a
non-representative sub-year slice, or several years with genuinely
different shapes — do not use these methods. Instead provide an accurate
absolute inflow time series and leave `inflow_method` at its default
(`use_original`); then no annual assumption is applied to your inflow.

(There is no equivalent "scale to annual" for `profile`-based limits such
as capacity factors. A capacity factor is bounded by installed capacity,
so there is no easy way to scale it accurately.)

#### One approximation to be aware of

When a single period represents multiple years (`years_represented > 1`),
costs within that period are discounted at the period's single point in
time rather than year by year. The totals remain correct; only the
within-period discounting is approximated. Use shorter periods (more of
them) if you need finer discounting resolution.
