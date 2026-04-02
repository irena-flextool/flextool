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
| `penalty_capacity_margin` (group) | CUR/kW | Capacity margin | No | Cost per kW of capacity shortfall per period. Analogous to investment cost -- but not annualized. It still has operational inflation adjustment. |

### Capacity margin penalty

The capacity margin penalty is different from the others: it represents the cost of not having sufficient installed capacity. It is analogous to an investment cost (CUR/kW) but is NOT annualized over a lifetime like actual investments. Instead, it applies as a lump cost per period. This means a `penalty_capacity_margin` of 1000 CUR/kW is much more expensive than an `invest_cost` of 1000 CUR/kW (which would be annualized to roughly 50-100 CUR/kW/year depending on lifetime and discount rate). Set the penalty to reflect the annual cost of capacity shortfall, not the total investment cost.

## Economic Modelling

### Overview

FlexTool minimizes total system cost over a planning horizon. The economic framework uses two key rates:

- **`inflation_rate`** (model-level): Adjusts for general price level changes over time
- **`discount_rate`** (per technology): Reflects the financing cost and risk of each investment

### Real vs nominal values

Economic values can be expressed in two ways:

- **Real (constant prices)**: All values are expressed in today's money. A power plant that costs 1000 EUR/kW today still costs 1000 EUR/kW in 2040 in real terms -- the purchasing power is the same.
- **Nominal (current prices)**: Values include inflation. The same plant might cost 1300 EUR/kW in 2040 at 2% annual inflation, even though its real cost hasn't changed.

**Example**: A coal plant costs 1500 EUR/kW today. At 2% inflation:
- Real cost in 2035: 1500 EUR/kW (unchanged)
- Nominal cost in 2035: 1500 x (1.02)^10 = 1829 EUR/kW approximately

**Rule**: Never mix real and nominal values in the same model. If `inflation_rate` = 0, all inputs must be in real terms. If `inflation_rate` > 0, all inputs must be in nominal terms.

### The inflation rate

The model-level `inflation_rate` parameter (default: 0) adjusts all future costs to a common price base. It applies uniformly to all cost types: investment costs, fuel costs, fixed O&M, penalties, etc.

- **Real inputs** (most common): Set `inflation_rate` = 0. All costs in all years use today's prices. A fuel cost of 50 EUR/MWh means the same purchasing power whether it occurs in 2025 or 2040.
- **Nominal inputs**: Set `inflation_rate` to expected inflation (e.g., 0.02 for 2%). The model deflates future costs: a nominal cost of 60 EUR/MWh in 2035 is treated as 60 / (1.02)^10 = 49.2 EUR/MWh approximately in today's money.

The inflation rate applies via the factor `1 / (1 + inflation_rate)^years`. Investment costs are assumed to occur at the beginning of each year; operational costs at the middle of the year (adjustable with `inflation_offset_investment` and `inflation_offset_operations`).

### The discount rate (per technology)

Each unit, connection, and storage node can have its own `discount_rate` parameter (default: 0.05, i.e., 5%). This represents the **weighted average cost of capital (WACC)** -- the return that investors require to finance the technology.

The discount rate converts a lump-sum investment cost into annual payments over the technology's lifetime using the standard annuity formula:

```
annual_payment = invest_cost * discount_rate / (1 - (1 / (1 + discount_rate))^lifetime)
```

**Example**: A solar plant costs 800 EUR/kW with a 25-year lifetime and 5% discount rate:
```
annual_payment = 800 * 0.05 / (1 - 1/1.05^25) = 800 * 0.0710 = 56.8 EUR/kW/year approximately
```

Different technologies can have different discount rates reflecting their risk profiles:
- Low risk (e.g., established solar PV): 3-5%
- Medium risk (e.g., natural gas turbine): 5-7%
- High risk (e.g., novel technology): 8-12%

### Consistency between rates

The `discount_rate` and `inflation_rate` must use the same price basis across one model instance (either real or nominal):

| Inputs | inflation_rate | discount_rate (entity) |
|--------|---------------|----------------------|
| Real (constant prices) | 0 | Real WACC (e.g., 5%) |
| Nominal (current and future prices) | Expected inflation (e.g., 2%) | Nominal WACC (e.g., 7%) |

The relationship between real and nominal rates follows the Fisher equation:
```
(1 + nominal_rate) = (1 + real_rate) * (1 + inflation_rate)
```
Approximation: nominal = real + inflation approximately (e.g., 5% + 2% = 7%).

### Fixed costs

Fixed costs (`fixed_cost` parameter on units, connections, and nodes) represent annual operation and maintenance costs that are incurred regardless of how much the asset is used. They are expressed in CUR/kW/year (or CUR/kWh/year for storage).

For **existing** assets, fixed costs are applied each year with inflation adjustment.

For **invested** assets, fixed costs are calculated over the full economic lifetime at the time of the investment decision. This allows the optimizer to correctly weigh the total cost of ownership when deciding whether to invest. In the results, these costs are reported as a separate line item from the investment annuity.

### Summary of economic parameters

| Parameter | Level | Default | Description |
|-----------|-------|---------|-------------|
| `inflation_rate` | Model | 0 | General inflation rate. Set to 0 for real inputs. |
| `inflation_offset_investment` | Model | 0 | When in the year investment costs occur (0 = start) |
| `inflation_offset_operations` | Model | 0.5 | When in the year operational costs occur (0.5 = middle) |
| `discount_rate` | Entity | 0.05 | Technology-specific WACC for annualizing investments |
| `invest_cost` | Entity | - | Overnight investment cost [CUR/kW] |
| `lifetime` | Entity | - | Economic lifetime [years] |
| `fixed_cost` | Entity | - | Annual fixed O&M cost [CUR/kW/year] |
| `salvage_value` | Entity | - | Residual value at end of life [CUR/kW] |
