"""
FlexTool Runner
===============
Entry point  : FlexToolRunner (flextoolrunner.py)
  .write_input()   — reads DB, writes input/ CSV files for the GLPK model
  .run_model()     — runs the full solve loop (delegates to orchestration.py)

State objects: runner_state.py (RunnerState, PathConfig)
               solve_config.py (SolveConfig — all solve-level parameters)
               timeline_config.py (TimelineConfig — all timeline definitions)

Solve logic  : recursive_solves.py (RecursiveSolveBuilder — rolling/nested/recursive solve structure)
               stochastic.py (StochasticSolver — stochastic branch handling)
               orchestration.py (run_model — the main solve loop)

I/O          : solve_writers.py  (pure functions — write solve_data/ CSV files)
               input_writer.py   (write_input + helpers — write input/ CSV files)
               solver_runner.py  (SolverRunner — invoke glpsol/highs/cplex binaries)
               db_reader.py      (pure functions — read from spinedb_api)
"""
from flextool.flextoolrunner.flextoolrunner import FlexToolRunner
__all__ = ['FlexToolRunner']
