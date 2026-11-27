# FlexTool scaling user guide

Short, practical reference for the numerical scaling that runs on every
FlexTool solve. Paired with the deeper reference in
[slack_convention.md](slack_convention.md) and the architecture overview
in [architecture.md](architecture.md).

## What is scaling doing for me?

FlexTool builds an LP / MIP whose matrix, cost row, and bound-vector
coefficients can span many orders of magnitude when a user mixes
entities of very different physical scale (tiny heat pumps next to
multi-GW coal plants, multi-year invest horizons next to hourly
dispatch, etc.). Wide coefficient spreads make the solver scale the
problem badly, miss symmetry, slow down, or — in extreme cases — declare
false infeasibility.

The autoscaler does three things on every solve, then writes a YAML
audit so you can see what it did:

1. **Layer 1 — detect.** Walks the assembled LP and computes log10
   ranges for the matrix, cost row, bound vector, and RHS, plus a
   cross-group max-ratio. Always runs; results land in the YAML.
   When the spread is comfortable HiGHS handles the LP on its own and
   Layers 2/3 are no-ops.
2. **Layer 2 — semantic per-type scaling.** Buckets matrix columns by
   physical quantity-type (energy, capacity, monetary, etc.) and
   multiplies each bucket by a power-of-2 scaler that compresses the
   median value onto O(1). The output writer applies the inverse so
   downstream consumers see absolute units.

Both readouts (the Layer 1 ranges and the Layer 2 magnitude buckets)
are **memory-bounded**. Rather than materialise the whole coefficient
product just to read a statistic, they walk the constraint / column
spine in fixed row-batches — only one batch's coefficients are ever
held in memory at once. This requires `polar-high >= 2.4.0`, which
FlexTool now depends on; on the large RETO-Africa DES scenario it cuts
the autoscale peak working set roughly in half. There is no longer a
size cap that skips very large families: every family's coefficients are
read and folded into the scaling decision, so no family is silently
left out.
3. **Layer 3 — HiGHS native.** Picks power-of-two `user_bound_scale` and
   `user_objective_scale` exponents.  The bound exponent keeps every
   meaningful limit inside `[10 × tol, 1e18 × tol]` (`tol` = the solve's
   primal feasibility tolerance): when the limits are so far apart that
   their small end would leave HiGHS' comfort zone, Layer 3 scales from the
   smallest side — the smallest limit lands at 100 × tol (10 × tol when the
   large end needs the room) and the large end takes the rest.  The exponent
   chosen there therefore depends on the tolerance.  Values at or above
   HiGHS' `infinite_bound` (1e20) are "no limit" to HiGHS and are not
   limits here either; float residue below 10 × tol is left out of the small
   end.
4. **Scaling guard.** Before the solver runs, FlexTool checks that the
   result fits (see [The scaling guard](#the-scaling-guard)).

The autoscaler never changes the LP optimum — every transform is
reversible and the reverse is wired into the output writer.

## How do I enable / disable it?

The autoscaler is **on by default**. Two ways to turn it off:

```bash
python run_flextool.py <input.sqlite> <output_info.sqlite> \
    --scenario-name <name> --scaling=off
```

Or set the environment variable:

```bash
export FLEXTOOL_SCALING=off
```

`--scaling` takes one of `off`, `solver_only`, `basic`, `full` (default
`full`). With `--scaling=off`, Layer 1 still runs (the YAML audit is
always written) but Layers 2 and 3 are skipped — the LP arrives at
HiGHS in raw user units, and HiGHS' own equilibration is also
disabled. The intermediate modes are `solver_only` (HiGHS-internal
equilibration only), `basic` (Layer 1 + Layer 3 bound-scale, no Layer
2 column rewriting), and `full` (the default — all three layers).

### Manual `user_bound_scale` override

When the auto-pick for Layer 3 misses a value you know works better
(e.g. HiGHS itself nudged with "*Consider setting the
`user_bound_scale` option to N*"), pin it explicitly:

```bash
python run_flextool.py ... --user-bound-scale 6
```

Or per-solve in the DB via `solve.user_bound_scale`. A non-zero
integer disables the auto-pick and the value is clamped to the
HiGHS-safe range.

## The scaling guard

After FlexTool's scaling, some models still have limits too far apart for
the solver: a period CO2 cap of `1e18` used as "no limit" next to a
capacity of `0.5` spans 18 decades, and no single scale lets HiGHS enforce
the small one while it still reads the large one accurately.  The solve
would then return a result that silently breaks the small limits (or fail
with a numerical status).  The guard checks the LP the solver gets, after
FlexTool's scaling, before the solve starts.

In full mode (the default) such a lone `1e18` usually only **warns**:
Layer 2 shrinks the quantity type it belongs to, the span drops to about
13 decades, and Layer 3 then picks a scale that keeps the small limits
enforceable.  The warning says so ("Layer 2 narrowed the span from 18.3
decades; under --scaling basic this would stop") whenever the span before
Layer 2 was more than 17 decades.  The solve **stops** under `--scaling
basic` (no Layer 2), or in full mode when Layer 2 cannot shrink the value
(for example when it shares a quantity type with the small limits):

```
Solve y2030 cannot be scaled into a range the solver handles accurately.
  Limits and bounds span 18.3 decades after FlexTool's scaling (0.5 … 1e+18); at most 17 fit (primal feasibility tolerance 1e-07, HiGHS default).
  Largest:  1e+18 (model units) in co2_max_period (constraint right-hand side) at g=co2_cap, d=y2030 — a stand-in for "no limit"? Use inf.
  Smallest: 0.5 (model units) in maxFlow (constraint right-hand side), 412 values within 10x of it, e.g. at:
      p=..., source=..., sink=..., d=y2030, t=t0001
      ...
  Fitting the largest value, the smallest limit could be violated by up to 1.68 (model units), 3.36x its size.
  To solve anyway: --scaling-guard warn (FLEXTOOL_SCALING_GUARD=warn; in the GUI: Solver options, Scaling guard: Warn).
```

At most five entries are listed at each end, and only those within 10x of
it.  A warning's message ends with what the solver may accept: "With
user_bound_scale N, the smallest limit could be violated by up to X (model
units); every limit by at most r %."

| What | Warning | Stop (error) |
|---|---|---|
| Limits and bounds (variable bounds, constraint right-hand sides) | span more than 13 decades | no scale fits them into the 17 decades the solver handles at its tolerance, or a `--user-bound-scale` / `highs.opt` scale puts them outside it |
| Objective costs | span more than 13 decades, or the smallest is below 10 × the dual tolerance | never |
| Constraint coefficients | a term the solver drops (≤ 1e-9) or refuses (≥ 1e15) | never |

How to fix the model, in order of preference:

1. **A placeholder for "no limit"** (`1e18`, `1e30`, …): give `inf` instead —
   an `inf` limit has no row at all.
2. **Tiny values that are really zero** (`1e-12` capacities, availabilities):
   give `0`, or leave the parameter out.
3. **Genuinely different scales**: change units (MWh → GWh for one
   commodity), or split the model (invest → dispatch handoff).
4. **A Layer 2 widening** (the message says "Layer 2 widened the span"):
   `--scaling basic` runs without Layer 2.

To run anyway: `--scaling-guard warn` (or `FLEXTOOL_SCALING_GUARD=warn`,
or in the GUI Solver options → Scaling guard → Warn) shows it as a coded
warning and solves; `--scaling-guard off` does not check.  A warning is one
console line naming the solve (W-0543 limits and bounds, W-0544 limits that
cannot be scaled, W-0545 objective costs, W-0546 coefficients; see the
warnings page); the message above, with the values and where they are,
goes to the run's `_run.log` (not the console; `run_warnings.warn_grouped(...,
run_log=True)` through `run_record.write_log_only`, a `LOG_ONLY_PREFIX` line
for a pipe owner) and, with `--debug`, to the debug log.  It shows once per run for each solve,
Benders region and kind of finding; a later roll whose finding becomes an
error, or that gets a scale of its own, shows it again.  A commercial solver (Gurobi, CPLEX, Xpress, COPT) gets no
FlexTool scaling, so the guard only warns there, judged at that solver's
own feasibility tolerance.  A rolling solve is checked on its first roll
and again on every later roll; a roll whose smallest limit no longer fits
the first roll's scale gets a scale of its own (logged), and a roll that
cannot be scaled stops the run with the roll named.  `--scaling off` and
`--scaling solver_only` run no autoscaler and so no guard.

## The YAML audit

Every solve writes `solve_data/autoscale_<solve>.yaml`, before the solve
(so it is there when the guard stops it) and again after it.  Top-level
sections:

| Section | Contents |
|---|---|
| `layer1` | matrix / cost / bound / RHS log10 ranges; `cross_group_max_ratio`; `trigger` flag (whether Layers 2/3 fired). |
| `layer2` | per-quantity-type power-of-2 exponents; post-Layer-2 ranges per type; list of quantity-types actually present in the LP. |
| `layer3` | `user_bound_scale` / `user_objective_scale` exponents applied to HiGHS; `bound_tag` (the tier: `in-zone`, `clamp-large`, `clamp-small`, `smallest-side`, `refuse`, with `+raised` / `+lowered` when the tolerance window moved it, `unscalable+smallest-side` when nothing fits); `bound_source` (`auto` / `manual` / `external`); the tolerance window. |
| `guard` | the scaling guard: `verdict` (`ok` / `warn` / `error` / `off`), the tolerances and where they came from, the window, every finding (`bound_rhs`, `cost`, `matrix`) with its numbers (spread `decades`, before Layer 2 `decades_pre`, scaled ends `s_lo` / `s_hi`, `r_lo` = the fraction of the smallest limit the solver may violate, `violation_model`), the family at each end and the entries named there; `info` counts residue and values at or above the solver infinity; `roll_user_bound_scale` when a later roll got its own exponent. |

A single-line console summary echoes after each solve; it ends with
`guard=ok|warn|error`. When a solve is
*rejected* and its **post-autoscale** LP is still wide-ranged, a hint also
prints — framed as a possibility, since a wide range does not by itself
explain a non-optimal status; the YAML shows which layer (or which
quantity-type) could not compress the spread. (Note a solve HiGHS could
not certify as optimal is not automatically rejected — see the solve
acceptance policy in `_solve_acceptance.py`.)

## Limitations to be aware of

The autoscaler is a numerical tool, not a model fixer. Some things it
**cannot** do:

- **Compress within-type spread.** Layer 2 scales whole quantity-types
  uniformly. If two `capacity` entities differ by 6 decades inside the
  same type, no single power-of-2 scaler shrinks both onto O(1).
- **Cure composite physical models.** A 10 kW residential heat pump
  directly connected to a 10 GW continental grid forces both sides
  into the same node-balance row at their physical magnitudes. The
  fix is structural — aggregate the small-side units (e.g. 1000
  buildings as one), or run the two subsystems as sequential models
  (invest → dispatch handoff). The Layer 1 ranges will flag this; the
  YAML's quantity-type sections will show the wide pre-/post-L2 range.
- **Fit limits more than 17 decades apart.** The tolerance window HiGHS
  handles accurately is 17 decades wide at any tolerance; a wider spread
  stops the solve (the scaling guard) unless `--scaling-guard warn` lets it
  run.  The guard reads the LP as built: a later roll of a rolling solve
  is rechecked, but a solve that reuses an already built LP (warm reuse)
  is not.
- **Help when the Rivendell guard fires.** Layer 2 has a guard that
  refuses to over-shrink columns carrying mixed-magnitude coefficients
  on a single row (the *Rivendell case*). When the guard activates,
  Layer 2 leaves those columns alone — the YAML records that the
  scaler was clamped.

## Small coefficients rounded to zero

Two cutoffs remove near-zero coefficients before any scaling (both are
judged on the coefficient's own, unscaled value):

- **Matrix and right-hand side** — `model.small_number_threshold`
  (default `1e-4`; polar-high `Problem.coef_zero_threshold`). A
  constraint coefficient or RHS term with `|value| < threshold`
  becomes 0. It does not apply to the objective.
- **Tiny cost coefficients (objective)** — polar-high
  `Problem.obj_coef_zero_rel_threshold`, which FlexTool sets to
  `OBJ_COEF_ZERO_REL_THRESHOLD = 1e-14` (`flextool/engine/model/core/_context.py`,
  not a database parameter). A variable's total cost coefficient below
  `2**floor(log2(1e-14 · max|cost|))` becomes 0. Layer 3 centres the
  objective on the cost range `[min|c|, max|c|]`; one float leftover
  such as a `1e-17` cost (from an algebraic cancellation) stretched
  that range to 17 decades, Layer 3 then scaled the objective by
  `2**25` and HiGHS' dual simplex failed with "excessive dual values"
  (#140). The cutoff is applied to the cost vector the solver gets,
  the pre-solve range readout that Layer 3 uses, and the warm-start
  cost updates.

  The cutoff is **relative** to the largest cost on purpose. Costs
  legitimately span many decades. Across the LP-snapshot fixtures the
  smallest real cost is `4.1e-5` (slack penalties of
  `rolling_separate_dispatch`), so reusing `small_number_threshold`
  (`1e-4`) would delete real costs. The widest real spread is
  min/max = `1.4e-12` (11.8 decades). `1e-14` is about 100 machine
  epsilons below the largest cost: below every real spread, above
  cancellation residue. Being relative, it also scales with
  `scale_the_objective`. It changes none of the snapshot LPs.

## Legacy LP row-scaling

Separate from the autoscale package, FlexTool still ships an LP
row-scaling family that multiplies node-balance / group-balance rows by
`node_capacity_for_scaling` / `group_capacity_for_scaling` derived from
connected-entity unitsizes (rounded to powers of 10 to preserve HiGHS
symmetry detection). The former per-solve `solve.use_row_scaling` DB
parameter was **removed**; row scaling is now controlled by the
`--scaling` CLI flag (env var `FLEXTOOL_SCALING`) alongside the rest of
the autoscaler. Every solve emits `p_use_row_scaling=0` by default, so
the legacy family is off unless forced via the
`FLEXTOOL_FORCE_ROW_SCALING` test hook. Output un-scaling is wired into
`process_outputs/read_highs_solution.py` either way.

## Input values are not rounded

FlexTool passes every input value to the model at full double
precision; nothing rounds parameter values before the LP is built.
(An earlier `--precision-digits` option claimed to round inputs to 10
significant figures, but it never reached a normal run; it was removed.)

## Troubleshooting checklist

- **"My objective doesn't match a pre-scaling reference."** Compare
  with `--scaling=off`: the scaled and unscaled solves agree within the
  solver tolerance, not bit for bit.
- **"Solve … cannot be scaled into a range the solver handles
  accurately."** The scaling guard stopped the solve before it ran; see
  [The scaling guard](#the-scaling-guard) for the usual causes (a
  placeholder for "no limit") and `--scaling-guard warn` to run anyway.
- **"HiGHS reports non-optimal."** Read the console hint and open
  `solve_data/autoscale_<solve>.yaml`. The post-Layer-2 ranges show
  whether Layer 2 compressed the spread; the `layer3` section shows
  what `user_bound_scale` HiGHS received. Try pinning a different
  value with `--user-bound-scale N`.
- **"The solver runs slowly but says optimal."** Inspect `layer1`
  ranges. A wide cost or matrix range that survives Layer 2 (look at
  `layer2.post_l2_ranges`) is the most common cause.
- **"My model is a 10 kW heat pump next to a 10 GW grid."** This is
  the composite case. The autoscaler will record the wide
  cross-group ratio but cannot fix it numerically — aggregate the
  small-side units or stage the subsystems sequentially.
- **"I want the legacy row-scaling on top of autoscale."** The
  `solve.use_row_scaling` DB knob no longer exists; the legacy family is
  off by default and only reachable via the `FLEXTOOL_FORCE_ROW_SCALING`
  test hook. For production scaling use the `--scaling` CLI flag; the
  output writer un-scales whatever was applied.
