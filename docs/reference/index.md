# Model parameters

These pages list the entity classes of a FlexTool model and the parameters you can give them, one page per topic.

- [Entities and time structure](entities_and_time.md) — the main entities, the timeline, periods, timesets, solves and the units of measure.
- [Nodes](nodes.md) — balance and storage nodes: inflow, penalties, storage and investment parameters.
- [Units](units.md) — conversion units, their inputs and outputs, investment, retirement and profile limits.
- [Connections](connections.md) — transfers between nodes, their transfer methods and investment parameters.
- [Commodities](commodities.md) — fuels and other commodities bought or sold at a price.
- [Groups](groups.md) — capacity, investment and emission limits, stochastics, decomposition and outputs for a set of entities.
- [Flow groups](flow_groups.md) — cumulative and instant limits on summed flows.
- [Reserves](reserves.md) — reserve requirements and the units and connections that provide them.
- [User constraints](user_constraints.md) — linear constraints you write yourself over flows, states and capacities.
- [Parameter groups and conventions](parameter_groups.md) — the parameter group labels and the behaviour-change notices of upgraded databases.
- [Parameter catalogue](../parameter_catalogue.md) — every parameter in one table per entity class: unit, allowed shapes, default, choices and group, generated from the schema of this FlexTool version. `python -m flextool.docs_build --xlsx catalogue.xlsx` writes it as an Excel data dictionary (see [How to make a data dictionary for your model](../how_to/outputs.md#how-to-make-a-data-dictionary-for-your-model)).
