# flextool.engine

The engine turns a FlexTool input database into solved LPs and output files.
It reads the data, splits the run into sub-solves (rolling, nested,
stochastic), builds each LP with polar_high, solves it and passes results
(investments, storage states, …) on to the next sub-solve.

## Public API (`flextool/engine/__init__.py`)

- `run_chain_from_db(input_db_url, scenario_name, work_folder)`: the usual entry point.
- `run_orchestration`, `OrchestrationStep`: the loop behind it.
- `run_chain`, `ChainStep`: older wrapper around `run_chain_from_db`.
- `load_flextool`, `FlexData`: per-solve model data. `build_flextool`: the LP.
- `SolveHandoff`: what one sub-solve passes to the next.
- `SpineDbReader`, `InMemoryReader`, `CsvSource`, `FlexInputSource`, `InputSource`: input adapters.

## Pipeline

1. `solve/_orchestration.run_chain_from_db` is the entry point.
2. `preprocess/_input_writer` reads the database (via `flextool.input_derivation`) and fills the in-memory Provider.
3. `input/_solve_config` and `input/_timeline` load solve settings and timeline into a `RunnerState`.
4. `solve/_run_model.native_run_model` expands solves into sub-solves (`solve/_recursive_solve`, `solve/_stochastic`).
5. Per sub-solve:
   - `preprocess/emit/` writes the sub-solve's sets and parameters into the Provider.
   - `preprocess/flex_data.load_flextool` assembles `FlexData` and runs the parameter cascade (`preprocess/params/`).
   - `model/core.build_flextool` builds the LP (or, when `solve/_warm` finds the sub-solve identical to the previous one, the previous LP is re-solved).
   - The LP is solved (`solve/_solver_dispatch`), with `autoscale/` scaling around it.
   - `output/_writer` writes outputs; `flex_data.build_handoff_from_solution` builds the handoff for the next sub-solve.
   - A solve with `decomposition = benders` goes to `solve/_benders` instead of a single LP.

## Where things live

| Folder | What |
|---|---|
| `input/` | Database and in-memory readers, solve settings, timeline, `RunnerState`, parameter shapes, constants |
| `preprocess/` | `FlexData` (`flex_data.py`), the Provider and its keys, input writer, per-solve context |
| `preprocess/params/` | Parameter cascade: direct, projected and derived parameters |
| `preprocess/emit/` | Per-solve sets and parameters written into the Provider |
| `model/` | The LP: `core.py` plus one module per optional feature |
| `solve/` | Orchestration, sub-solve expansion, solvers, warm start, Benders, handoff |
| `output/` | Solution to output parquet files; CSV dumps |
| `autoscale/` | LP scaling around the solve |

## Import direction

Imports point down the pipeline: solve → model → preprocess → input
(`output/` and `autoscale/` are used by `solve/`). Known exceptions:
`preprocess/flex_data.py` imports model feature modules at top level and,
inside functions, `solve/_orchestration`, `solve/_handoff` and
`output/_dump_csvs`; `input/_timeline.py` imports `preprocess/_provider_io`;
several modules import `SolveHandoff` from `solve/_handoff` inside functions.

## Finding things

- An equation: grep `EQ. ─── <name>` (map in the `model/core.py` docstring).
- A Provider key: `preprocess/_provider_keys.py`.
- Did my change alter the LP? `tools/lp_snapshot.py` (snapshot, then compare).
- More depth: `docs/dev/engine.md`, `docs/dev/architecture.md`.
- Tests: `tests/engine/`.
