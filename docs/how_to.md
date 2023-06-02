# How to

Here are some examples on how to build parts of the system.
Each example will include an example database file that are located in the 'how to examples databases' folder. You can change the filepath to database used as the input data by clicking the input_data tool.

## How to create a hydro reservoir
**hydro_reservoir.sq**

Note: This applies to single reservoir rivers. If multiple plants in a series on the same river and their storages are dependent on the usages of others, things get more complicated.

The objective is to create a hydro power plant with a reservoir and connect it to a demand node.

Hydro reservoir power plant requires three components:

- Reservoir (node)
- Unit 
- Output node

Good common practice is to create a new alternative for this plant to be able to include and exclude it from the scenarios.

The reservoir is made with a node as it can have storage capacity. The incoming water can be represented by the inflow parameter. It can be a constant or a time mapping. The unit of the inflow should be the power that can be created from the quantity of the incoming water at maximum efficiency [MW]. In the same way, the existing capacity should be the energy that can be created from the storage [MWh].
The speciality of this storage is the option of decreasing the storage by spilling ie. without running it through the plant. The simplest way of allowing spilling is setting the downward penalty of the node to 0. This way the energy can disappear from the storage without a cost. The quantity of spilled energy can be seen from the results as the 'downward slack' of the node.

The required parameters of the reservoir are (node_c and node_t in excel):

- Is_active: yes
- has_balance: yes
- has_storage: yes
- inflow: Mapping of the incoming water as the potential power [MW]
- existing: The maximum size of the reservoir as the potential energy [MWh]
- penalty_up: a large number
- penalty_down: 0 or a large number (spilling or not)
- a storage_method to set the behaviour on how the storage levels should be managed

The unit is connected to the reservoir and the output nodeA (unit_c and unit_node_c in excel):

- The efficiency of the unit can be set to 1 as that information is in the reservoir.
- Set existing capacity [MW]
- is_active: yes 
- Create relations unit_inputNode: hydro_plant|reservoir and unit_outputNode: hydro_plant|nodeA.

![Cplex parameters](./hydro_reservoir.png)

## How to use CPLEX as the solver

Using CPLEX requires that you have installed the software, have a licence for it and have added it to PATH or to the environment where you are using the FlexTool, so that the tool can find the solver.

CPLEX is used when the **solve** parameter *solver* is set to 'cplex'. The tool passes the built optimization model to the CPLEX solver and converts the solution file to the filetype the tool requires. The solver will produce two additional files to the work directory: 'cplex.log' and 'flexModel3_cplex.sol'. The former is the logfile of the solver and the latter contains the solution in the CPLEX format.

The tool uses [Interactive Optimizer](https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-interactive-optimizer) to pass the problem to the solver. The default command used:
  
```shell
cplex -c 'read flexModel3.mps' 'opt' 'write flexModel3_cplex.sol' 'quit'
```

Additional parameters:

- *solver_wrapper* creates a text in front of the cplex call. This is useful when dealing with floating licences and if the licence system you are using allows to reserve the licence for the duration of the cplex program with a command line argument.
- *solver_command* is an array containing additional CPLEX solver commands

With these parameters, the command line call is:

```shell
'solver_wrapper' cplex -c 'read flexModel3.mps' 'solver_command1' 'solver_command2' ... 'solver_command_last' 'opt' 'write flexModel3_cplex.sol' 'quit'
```

![Cplex parameters](./CPLEX.png)