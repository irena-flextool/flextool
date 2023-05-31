# Examples on how to

## How to use CPLEX as the solver

Using CPLEX requires that you have installed the software, have a licence for it and have added it to PATH or to the environment where you are using the FlexTool, so that the tool can find the solver.

CPLEX is used when the **solve** parameter *solver* is set to 'cplex'. The tool passes the built optimization model to the CPLEX solver and converts the solutionfile to the filetype it requires. The solver will produce two additional files to the work directory: 'cplex.log' and 'flexModel3_cplex.sol'. The former is the logfile of the solver and the latter contains the solution in the CPLEX format.

The tool uses [Interactive Optimizer](https://www.ibm.com/docs/en/icos/12.8.0.0?topic=cplex-interactive-optimizer) to pass the problem to the solver. The default command used:
  
```shell
cplex -c 'read inputfile' 'opt' 'write outputfile' 'quit'
```

Additional parameters:

- *solver_wrapper* creates a text in front of the cplex call. This is useful when dealing with floating licences and if the licence system you are using allows to reserve the licence for the duration of the cplex program with a command line argument.
- *solver_command* is an array containing additional CPLEX solver commands

With these parameters, the command line call is:

```shell
solver_wrapper cplex -c 'read inputfile' 'solver_command1' 'solver_command2' ... 'solver_command_last' 'opt' 'write outputfile' 'quit'
```

![Cplex parameters](./CPLEX.png)