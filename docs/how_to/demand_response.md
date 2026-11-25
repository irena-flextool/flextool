# Demand response

Demand response is demand that changes with the price or the needs of the power system: it is cut when power is
scarce, or moved to cheaper hours. FlexTool has no separate demand response entity. Instead, it is built from units,
storage nodes and connections. The sections below go from a simple price-sensitive reduction to EV smart charging.
They share one small model in `demand_response.sqlite`, with one scenario per section.

**The model** (alternative *base*, scenario *no_dr*). It has 48 one-hour time steps: a Friday (t0001–t0024) and a
Saturday (t0025–t0048). The demand node *town* is served by *base_plant* (700 MW, 40 €/MWh) and *peak_plant* (300
MW, 150 €/MWh). Both are units without inputs, and their cost is `other_operational_cost` on the
`unit__outputNode`. Demand that the plants cannot serve goes unserved at the `penalty_up` of *town*, 3000 €/MWh.
The demand (MW) is:

| Friday | 00–05 | 06–07 | 08–16 | 17–18 | 19–20 | 21–23 |
|---|---|---|---|---|---|---|
| demand | 680 | 800 | 950 | 1080 | 1130 | 780 |

| Saturday | 00–07 | 08–16 | 17–20 | 21–23 |
|---|---|---|---|---|
| demand | 550 | 820 | 980 | 650 |

So the price of *town* (`price_t` in the `node` results) is 40 €/MWh below 700 MW and 150 €/MWh between 700 and
1000 MW. Above 1000 MW it is 3000 €/MWh. In *no_dr*, 80 MW is unserved at Friday 17–18 and 130 MW at 19–20
(*Loss of load* in the node balance, and the `slack_up_t` of *town* per time step).

The scenarios *dr_decrease_demand*, *dr_increase_demand* and *dr_shift_demand* of `examples.sqlite` are
smaller versions of the same ideas on the node *west*.

## How to model price-sensitive demand reduction

**(demand_response.sqlite, scenario: dr_price)**
***base - dr_price***

Consumers who are willing to cut their consumption when the price exceeds a given level are modelled as a unit that
feeds the demand node. Each MWh the unit "produces" is a MWh of demand that is not consumed. The unit *dr_reduce*
has:

- a `unit__outputNode` *dr_reduce – town*, and no input;
- `existing`: 150 [MW], the largest reduction;
- `other_operational_cost` on *dr_reduce – town*: 200 [€/MWh], the price at which the consumers start to cut.

The model uses the reduction only when it is cheaper than the other ways to serve the demand, that is, when the
price would otherwise be above 200 €/MWh. In *dr_price*, *dr_reduce* cuts 80 MW at Friday 17–18 and 130 MW at
19–20, exactly the demand that was unserved in *no_dr*. In all other hours the price is at most 150 €/MWh and
*dr_reduce* stays at 0. The price at Friday 17–20 is now 200 €/MWh instead of 3000 €/MWh. The reduction is the
`unit__node` result `flow_t` of *dr_reduce – town*.

## How to model a price-elastic demand (stepwise price blocks)

**(demand_response.sqlite, scenario: dr_blocks)**
***base - dr_price - dr_blocks***

A demand curve, where the demand falls step by step as the price rises, is a set of reduction units with rising
costs. Each unit is one block of the curve. The alternative *dr_blocks* builds on *dr_price*:

- *dr_reduce* gets `existing` 50 [MW] and becomes the middle block (200 €/MWh);
- *dr_reduce_low*: 50 MW at 100 €/MWh (a block below the cost of the peak plant);
- *dr_reduce_high*: 50 MW at 500 €/MWh.

Results:

- *dr_reduce_low* cuts 50 MW in every hour with demand above 700 MW, where the price would otherwise be 150 €/MWh.
  The price there is still 150 €/MWh, set by *peak_plant*.
- *dr_reduce* cuts 30 MW at Friday 17–18 and sets the price there to 200 €/MWh. It cuts its full 50 MW at 19–20.
- *dr_reduce_high* cuts 30 MW at Friday 19–20 only (700 + 300 + 50 + 50 + 30 = 1130 MW). The price there is
  500 €/MWh.

No demand goes unserved. More blocks give a finer curve.

## How to make demand response available only part of the time

**(demand_response.sqlite, scenario: dr_window)**
***base - dr_price - dr_window***

Many demand response programmes can be called only at given times, e.g. on weekday daytime when an office or a
factory is operating. The unit `availability` time series sets this. The alternative *dr_window* builds on
*dr_price* and gives *dr_reduce* an `availability` of 1 on Friday 08–18 (t0009–t0019) and 0 at every other hour.

In *dr_window*, *dr_reduce* cuts 80 MW at Friday 17–18, as in *dr_price*. At 19–20 it is outside its window, so it
cuts nothing, and 130 MW of demand goes unserved although the price is 3000 €/MWh. On Saturday it is not available
at all.

An `availability` between 0 and 1 gives the share of the capacity that can be called, e.g. 0.3 for a programme that
can call 30 % of its 150 MW at night.

## How to model load shifting as a storage

**(demand_response.sqlite, scenario: dr_shift)**
***base - dr_shift***

Some consumption can be moved to another time, e.g. heating, cooling or industrial batches. Moving it means
consuming less at one time and more at another, and that is a storage: a storage node connected to the demand node.
Energy that flows from the storage node to *town* is consumption cut now. Energy that flows from *town* to the
storage node is that consumption made up at another time. The alternative *dr_shift* adds:

- node *town_shift*: `node_type` *storage*, `existing` 500 [MWh] (how much consumption can be moved),
  `storage_binding_method` *bind_within_solve* (all the consumption that is moved is made up within the solve),
  and `penalty_up` / `penalty_down` 3000;
- connection *shift_link* between *town* and *town_shift*: `existing` 150 [MW] (how much consumption can be moved
  in an hour), `efficiency` 1, `transfer_method` *variable_cost_only* and `other_operational_cost` 60 [€/MWh]. The
  cost applies in both directions and stands for the lost comfort or wear of moving consumption: 120 €/MWh for each
  MWh moved and made up.

In *dr_shift* the storage delivers 80 MW at Friday 17–18 and 130 MW at 19–20 (420 MWh), and no demand goes
unserved. The 420 MWh are consumed in hours with a price of 40 €/MWh: Friday 00–05 (20 MW, the room left on
*base_plant*), Saturday 07:00 and Saturday 21–23. Some of it is made up on Saturday night. The storage cycles over
the solve, so this energy also covers the start of Friday. Consumption is not moved out of the hours with a price
of 150 €/MWh, since a 110 €/MWh price difference does not pay the 120 €/MWh of a round trip. With a lower
`other_operational_cost`, consumption would also move from those hours to the 40 €/MWh hours. When several hours
are equally cheap, the hours the solver picks are one choice of several equally good ones.

Energy that is moved can also be lost (e.g. a building that cools down): give *town_shift* a
`self_discharge_loss`. The storage does not require the moved consumption to be made up within the same day or
within a given number of hours. That would need a storage balance every N hours, which is a possible future
option.

## How to model EV smart charging

**(demand_response.sqlite, scenario: ev_smart_charging)**
***base - ev_charging***

A fleet of electric vehicles is a storage that is connected to the grid only while the cars are plugged in. Its
energy is used for driving whether the cars are plugged in or not. The alternative *ev_charging* adds:

- node *ev_battery*: `node_type` *storage*, `existing` 300 [MWh] (the fleet's battery energy),
  `storage_binding_method` *bind_within_solve*, and `inflow` −40 [MW] at 07–08 and 16–17 each day (the driving,
  160 MWh a day);
- unit *ev_charger* from *town* (`unit__inputNode`) to *ev_battery* (`unit__outputNode`): `existing` 60 [MW],
  `efficiency` 0.9, and `availability` 1 at 00–06 and 18–23, 0 at 07–17 while the cars are away.

The charger cannot charge while `availability` is 0, so the battery must hold the energy for the drives during the
day. In *ev_smart_charging* the charger has no flow at 07–17. It charges 320 MWh into *ev_battery* over the two
days, 355.6 MWh from *town*, and all of it in hours with a price of 40 €/MWh. It does not charge in the Friday
evening, where demand is unserved. For vehicle-to-grid, add a second unit from *ev_battery* to *town* with the same
`availability`.
