# How-to recipes

How-to section contains examples on how to include common energy system components in your model. The examples assume that you have sufficient understanding of FlexTool basics (e.g. by doing the tutorial).
Each example will either include an example database file that is located in the 'how to examples databases' folder or the example is included in the examples.sqlite as a scenario. If the example is in its own database, you can switch to that database by selecting the 'input' data store in the workflow and then changing the database by clicking the folder icon next to the current database file path in the 'Data store properties' widget. Navigate to the 'how to example databases' folder and choose the appropriate database.

## [Basics: time, VRE, network and demand](basics.md)

- [How to create basic temporal structures for your model](basics.md#how-to-create-basic-temporal-structures-for-your-model)
- [How to create a PV, wind or run-of-river hydro power plant](basics.md#how-to-create-a-pv-wind-or-run-of-river-hydro-power-plant)
- [How to connect nodes in the same energy network](basics.md#how-to-connect-nodes-in-the-same-energy-network)
- [How to set the demand in a node](basics.md#how-to-set-the-demand-in-a-node)
- [How to create a delay between two nodes](basics.md#how-to-create-a-delay-between-two-nodes)
- [How to use timeset weights (non-RP weighted timesteps)](basics.md#how-to-use-timeset-weights-non-rp-weighted-timesteps)

## [Storage and hydro](storage_hydro.md)

- [How to add a storage unit (battery)](storage_hydro.md#how-to-add-a-storage-unit-battery)
- [How to create a hydro reservoir](storage_hydro.md#how-to-create-a-hydro-reservoir)
- [How to create a hydro pump storage](storage_hydro.md#how-to-create-a-hydro-pump-storage)

## [Thermal, CHP and reserves](thermal_chp.md)

- [How to create combined heat and power (CHP)](thermal_chp.md#how-to-create-combined-heat-and-power-chp)
- [How to add a reserve](thermal_chp.md#how-to-add-a-reserve)
- [How to add a minimum load, start-up and ramp](thermal_chp.md#how-to-add-a-minimum-load-start-up-and-ramp)
- [How to add minimum uptime and downtime](thermal_chp.md#how-to-add-minimum-uptime-and-downtime)
- [How to make a unit run all the time (must-run)](thermal_chp.md#how-to-make-a-unit-run-all-the-time-must-run)

## [Investment and multi-year models](investment.md)

- [How to make investments (storage/unit)](investment.md#how-to-make-investments-storageunit)
- [How to create a multi-year model](investment.md#how-to-create-a-multi-year-model)

## [Solve sequences, rolling and nested solves](solve_sequences.md)

- [How to run solves in a sequence (investment + dispatch)](solve_sequences.md#how-to-run-solves-in-a-sequence-investment-dispatch)
- [How to use a rolling window for a dispatch model](solve_sequences.md#how-to-use-a-rolling-window-for-a-dispatch-model)
- [How to use Nested Rolling window solves (investments and long-term storage)](solve_sequences.md#how-to-use-nested-rolling-window-solves-investments-and-long-term-storage)
    - [How to force-include adequacy-critical weeks in representative-period selection](solve_sequences.md#how-to-force-include-adequacy-critical-weeks-in-representative-period-selection)
    - [How to cluster representative periods on net load](solve_sequences.md#how-to-cluster-representative-periods-on-net-load)
    - [How to add an energy margin for investment adequacy](solve_sequences.md#how-to-add-an-energy-margin-for-investment-adequacy)
- [How to use representative-period clustering with scalar profiles](solve_sequences.md#how-to-use-representative-period-clustering-with-scalar-profiles)

## [Stochastics](stochastics.md)

- [How to use stochastics (representing uncertainty)](stochastics.md#how-to-use-stochastics-representing-uncertainty)

## [Emissions and system limits](emissions.md)

- [How to add CO2 emissions, costs and limits](emissions.md#how-to-add-co2-emissions-costs-and-limits)
- [How to compare emission trajectories](emissions.md#how-to-compare-emission-trajectories)
- [How to create a non-synchronous limit](emissions.md#how-to-create-a-non-synchronous-limit)

## [Outputs, diagnostics and Excel templates](outputs.md)

- [How to see the VRE curtailment and VRE share results for a node](outputs.md#how-to-see-the-vre-curtailment-and-vre-share-results-for-a-node)
- [How to create aggregate outputs](outputs.md#how-to-create-aggregate-outputs)
- [How to enable/disable outputs](outputs.md#how-to-enabledisable-outputs)
- [How to read the LP scaling diagnostic report](outputs.md#how-to-read-the-lp-scaling-diagnostic-report)
- [How to export a partial Excel template (parameter-group picker)](outputs.md#how-to-export-a-partial-excel-template-parameter-group-picker)
- [How to make a data dictionary for your model](outputs.md#how-to-make-a-data-dictionary-for-your-model)

## [Performance and solvers](performance.md)

- [How to use CPLEX as the solver](performance.md#how-to-use-cplex-as-the-solver)
- [How to make the Flextool run faster](performance.md#how-to-make-the-flextool-run-faster)
- [How to use Benders decomposition (spatial)](performance.md#how-to-use-benders-decomposition-spatial)
