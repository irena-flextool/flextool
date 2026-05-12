# Spatial-Lagrangian decomposition (`--decomposition lagrangian`)

FlexTool supports a **spatial Lagrangian decomposition** where the
scenario is partitioned into geographic regions, each solving an
independent LP coupled only by shared pipeline / transmission flows.
The coordinator drives a damped sub-gradient loop on the shared flow
constraints until the averaged primal imbalance converges.

This mode is **optional**; the default remains the monolithic
orchestration path.  Use it when:

* Each region is moderate-scale (so that one region's LP solves in
  seconds) but the monolithic union is too large to solve as a single
  HiGHS instance;
* You're interested in exploring the dual price structure of
  inter-region energy trade (each pipeline's λ converges to a shadow
  price);
* You want to validate a decomposition scheme against a known
  monolithic optimum (see the LH2 three-region fixture for the
  integration test).

## Setting up a decomposition scenario

Every region is a **group** in the Spine input DB whose
`decomposition_method` parameter equals the string
`lagrangian_region`.  The group's `group__node` / `group__unit` /
`group__connection` membership lists identify which entities belong to
that region.

```python
# In a fixture builder (or via the Toolbox GUI):
entities += [
    ("group", "region_A"),
    ("group", "region_B"),
    ("group", "region_C"),
]
parameter_values += [
    ("group", "region_A", "decomposition_method", "lagrangian_region", ALT),
    ("group", "region_B", "decomposition_method", "lagrangian_region", ALT),
    ("group", "region_C", "decomposition_method", "lagrangian_region", ALT),
]
for r in ("A", "B", "C"):
    for node in [f"elec_{r}", f"h2_{r}", f"lh2_{r}", f"battery_{r}"]:
        entities.append(("group__node", (f"region_{r}", node)))
```

Cross-region connections (pipelines / transmission lines) are
identified automatically — a `connection` whose source endpoint lies
in one region group and sink endpoint lies in another is rewritten
into a pair of **import / export half-flows** through a virtual
commodity node.  The half-flow for the exporting region is named
`hf_<pipe>__export__<region>` with a new virtual node
`<pipe>__export__<region>`; the importing region gets
`hf_<pipe>__import__<region>` and the matching virtual node.  The
coupling constraint is simply

```
sum(v_flow[hf_<pipe>__export__<region_A>, …]) = sum(v_flow[hf_<pipe>__import__<region_B>, …])
```

dualized with a shared multiplier λ_pipe; in equilibrium
`export = import` for every cross-region pipeline.

## Running

```
python run_flextool.py \
    sqlite:///my_input.sqlite \
    sqlite:///my_output.sqlite \
    --scenario-name my_scenario \
    --decomposition lagrangian \
    --lagrangian-alpha 0.1 \
    --lagrangian-max-iter 80 \
    --lagrangian-tolerance 1.0 \
    --work-folder /tmp/lagrangian_run
```

Flags:

| Flag | Default | Meaning |
| --- | ---: | --- |
| `--decomposition lagrangian` | `none` | Switch on spatial Lagrangian |
| `--lagrangian-alpha` | `0.1` | Base sub-gradient step size (step at iter k is `α / √k`) |
| `--lagrangian-max-iter` | `80` | Outer-loop iteration cap |
| `--lagrangian-tolerance` | `1.0` | Tail-averaged imbalance threshold for convergence |

## What the converged solution represents

After the outer loop finishes (or runs out of iterations), the
coordinator runs a **primal recovery** pass: each region's coupling
flow columns are fixed to their tail-averaged values, costs are reset
to zero, and every region is re-solved.  The summed objective is
reported as `total_objective`.

For LP subproblems (no integer variables) the averaged-primal
objective converges to the monolithic optimum; a small residual gap
(~0.5–2%) is normal and comes from:

* The sub-gradient step size not shrinking fast enough to eliminate
  the last trace of bang-bang oscillation in the coupling flows;
* Efficiency mismatches when the half-flow formulation drops the
  pipeline's loss coefficient (the default injection sets
  `efficiency = 1.0` on the virtual half-flow; for loss-aware
  decomposition the exporter-side half-flow inherits the original
  pipeline efficiency).

For MIP subproblems (unit commitment etc.) there is an intrinsic
duality gap that the Lagrangian cannot close; a follow-up bundle /
branch-and-bound outer loop is required (not implemented).

## Diagnostics

Every outer iteration logs:

```
Lagrangian iter 42: α_k=0.0154  max|imb|=8.4e+03  Σ_obj=-16543  λ={'pipe_AB': -24.3, 'pipe_BC': 0.0}
```

The `iteration_log` field of the :class:`LagrangianResult` carries
this data as a list of dicts — useful for plotting λ trajectories
and debugging oscillation.

Per-region artefacts live under
`work_folder/region_<group>/`:

* `input/` — the filtered per-region input CSV directory
* `input_monolithic_backup/` — pristine copy of the unfiltered input
  (for reference)
* `solve_data/` — the per-region pre-solve data
* `flextool.mps` — the GMPL-generated regional MPS (loaded into the
  live HiGHS instance for all outer iterations)
* `HiGHS.log` — HiGHS solver log for the initial pass

The union coupling manifest is at
`work_folder/solve_data/region_coupling.csv` with columns
`region, process, side, virtual_node`.

## Python API

```python
from flextool.flextoolrunner.lagrangian import run_lagrangian, LagrangianResult

result: LagrangianResult = run_lagrangian(
    db_url="sqlite:///my_input.sqlite",
    scenario="my_scenario",
    alpha=0.1,
    max_iterations=80,
    tolerance=1.0,
    work_folder=Path("/tmp/lag"),
)
assert result.converged
print(f"Total objective: {result.total_objective:.3f}")
for pipe, lam in result.final_lambdas.items():
    print(f"λ[{pipe}] = {lam:.3f}")
```

## Limitations (as of Agent 3.2)

* Pure LP subproblems only — MIP regions converge to the LP relaxation
  of the monolithic optimum, not the MIP optimum.
* Each region's LP is loaded into HiGHS once from the GMPL-generated
  MPS; the coordinator modifies only column costs on the coupling
  variables between iterations (no row additions).
* The filter assumes every cross-region connection is bilateral — star
  topologies collapse to a single λ per pipe across all sharing
  regions.
* Pipeline efficiency is currently set to `1.0` on the virtual half-
  flow; the true pipe loss coefficient is not yet propagated.
