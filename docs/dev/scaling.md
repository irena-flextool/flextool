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
3. **Layer 3 — HiGHS native + escape-tier folding.** Picks an integer
   `user_bound_scale` exponent (within HiGHS's safe range) and folds
   the unbounded-slack escape valve into the same pass.

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

## The YAML audit

Every solve writes `solve_data/autoscale_<solve>.yaml`. Three top-level
sections:

| Section | Contents |
|---|---|
| `layer1` | matrix / cost / bound / RHS log10 ranges; `cross_group_max_ratio`; `trigger` flag (whether Layers 2/3 fired). |
| `layer2` | per-quantity-type power-of-2 exponents; post-Layer-2 ranges per type; list of quantity-types actually present in the LP. |
| `layer3` | `user_bound_scale` exponent applied to HiGHS; escape-tier plan. |

A single-line console summary echoes after each solve. When a solve is
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
  `OBJ_COEF_ZERO_REL_THRESHOLD = 1e-14` (`flextool/engine/model/core.py`,
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
