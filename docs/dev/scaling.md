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

A single-line console summary echoes after each solve. On HiGHS
non-optimal a poorly-scaled-LP hint also prints; the YAML shows which
layer (or which quantity-type) could not compress the spread.

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

## Precision rounding

Every numeric CSV cell is rounded to 10 significant figures by default
before it is written to `input/` or `solve_data/`. On very large
values (e.g. annual flows in the 1e9 range) this may shift downstream
objectives by `≤ 1e-11` relative — sub-microscopic but real and
deterministic.

Override with `--precision-digits 15` to effectively disable rounding.

## Troubleshooting checklist

- **"My objective doesn't match a pre-scaling reference."** Check
  `--precision-digits`. Default 10; was effectively 16 before.
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
