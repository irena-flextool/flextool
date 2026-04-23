# FlexTool scaling user guide

Short, practical reference for the numerical scaling that runs
automatically on every FlexTool solve. Paired with the deeper
reference in `flextool/SLACK_CONVENTION.md` and the project design
memo at
`~/.claude/projects/-home-jkiviluo-sources-flextool/memory/project_lp_scaling_2026-04.md`.

## What is scaling doing for me?

FlexTool builds an LP/MIP whose matrix, cost row, and bound vector
coefficients can span many orders of magnitude when a user mixes
entities of very different physical scale (tiny heat pumps next to
multi-GW coal plants, multi-year invest horizons next to
hourly dispatch, etc.). Wide coefficient spreads make the solver
scale the problem badly, miss symmetry, slow down, or — in extreme
cases — declare false infeasibility.

The scaling pipeline does three things on every solve, without user
intervention:

1. Applies a **unified slack convention** (primary `≤ K_rel` plus an
   unbounded escape tier) so every `vq_*` in the objective uses the
   same two-term pattern. Escape activity becomes a user-facing
   diagnostic instead of a false-infeasibility signal.
2. Runs a **ScaleAnalyzer** over the solve's input CSVs, summarises
   coefficient spreads per parameter family, and emits
   `solve_data/scaling_analysis.json`.
3. Writes a **diagnostic report** (`solve_data/scaling_report.txt`)
   covering matrix / cost / bound ranges, bimodal coefficient
   distributions, composite-scale mismatches, near-duplicate
   parameter clusters, and escape-tier slack activity. A one-line
   summary also echoes to stdout.

Nothing here changes the LP optimum. Every scaling operation is
paired with an output un-scaler so downstream consumers (parquet,
CSV, plots, scenario comparison) see values in the same absolute
units as before the project.

## How do I enable auto-scaling?

Two knobs; both default off:

### `--auto-scale` CLI flag

```
python run_flextool.py <input.sqlite> <output_info.sqlite> \
    --scenario-name <name> --auto-scale
```

Or set the environment variable:

```
export FLEXTOOL_AUTO_SCALE=1
```

With `--auto-scale`, the analyzer's `use_row_scaling` recommendation
is applied whenever the user's DB setting is missing. If the user
set `solve.use_row_scaling` to `"yes"` or `"no"` in the database,
the DB setting wins and a log line notes the override. The scalar
`scale_the_objective` is **not** auto-applied (see "What the scaling
report tells me" below for the reason).

### `solve.use_row_scaling` database parameter

Per-solve boolean (`"yes"` / `"no"`, default `"no"`). Writes to
`solve_data/p_use_row_scaling.csv` and flips
`node_capacity_for_scaling` / `group_capacity_for_scaling` inside
`flextool.mod` from `1.0` to the power-of-10 of the relevant
unitsizes. Intended for advanced users; the `--auto-scale` flag is
the normal entry point.

### When to enable row scaling

The scaling report's section 2 prints a recommendation:

```
use_row_scaling        recommended=yes  applied=no
```

- `recommended=yes` fires when `log10(max) − log10(min)` across the
  solve's unitsizes exceeds 3 decades.
- Applied column shows what actually happened. `applied=no` means
  the solve ran with row scaling off; flip `--auto-scale` on (or
  set the DB parameter) to apply the recommendation.

For the four in-tree benchmark scenarios:

- `small_building` (spread 2.0 decades) — analyzer says no.
- `continental` (spread 0.0 decades) — analyzer says no.
- `medium_national` (spread 5.0 decades) — analyzer says yes.
- `composite` (spread 8.0 decades) — analyzer says yes.

## What does the scaling report tell me?

`scaling_report.txt` has nine sections; the load-bearing ones are
5, 7, and 9.

| Section | Contents |
|---|---|
| 1. Header | Solve name, timestamp, HiGHS version. |
| 2. Scaling decisions | Recommended vs applied `use_row_scaling` / `scale_the_objective` and the rough objective-magnitude estimate. |
| 3. Coefficient-family ranges | Per-family log10 spread, percentiles, min/max. Spread > 5 decades earns a `!` mark. |
| 4. Bimodal coefficient distributions | Flags any family with a > 2-decade gap between two clusters, each holding ≥ 10 % of values. |
| 5. **Composite-scale mismatch** | Directly-connected entity pairs spanning > 3 decades in unitsize. The project's main user-facing diagnostic. |
| 6. Near-duplicate parameter clusters | Repeated values within a parameter file — useful for spotting copy-paste bugs. |
| 7. **Escape-tier / slack activity** | Every slack with non-zero activity, with the top-5 offending cells. Non-zero means the model is saturating a physical constraint. |
| 8. HiGHS matrix-range summary | Matrix / cost / bound / RHS ranges as reported by HiGHS, plus spread warnings. |
| 9. **Summary** | One-line verdict: `well-scaled`, `acceptably`, or `poorly` with warning count. |

`scaling_analysis.json` (same directory) is the machine-readable
twin — pure JSON, stdlib-parseable, suitable for regression tests
and automation.

## What if my model is composite?

A "composite" model mixes entities of wildly different physical
scale. Example: a 10 kW residential heat pump directly connected to
a 10 GW continental grid. The node-balance row for that connection
has coefficients spanning six orders of magnitude. Linear scaling
cannot eliminate this — both sides *have to* appear in the same
row for physical realism.

Section 5 of the scaling report lists the top-10 offending pairs
and prints a locked recommendation:

> Recommendations:
>   (1) Aggregate the small-side units: e.g., use 1000 buildings
>       instead of 1 to match the order of magnitude of the
>       connected system. Accept that aggregation introduces some
>       inaccuracy in the small-scale dynamics.
>   (2) Run the two subsystems as sequential models: optimise the
>       large system first, then use its results as boundary
>       conditions for a detailed small-system run (invest ->
>       dispatch handoff, or whichever staging fits your use case).

These are the two supported workarounds. Neither is automated.
The model change is physical, not numerical.

### Buffer-node false positives

A sink-only buffer node with a huge virtual capacity (e.g. a water
sink with `virtual_unitsize = 1e6` used for material balance
accounting) will trip the mismatch detector. The detector is
mathematically correct — the ratio *does* exist in the matrix —
but the node is removed by HiGHS's presolve so it does no
operational harm.

If the only mismatch section-5 entry is a sink-only buffer node
and no other family is poorly scaled, you can safely ignore the
warning after review.

## Precision rounding

Agent 7's precision module rounds every numeric CSV cell to 10
significant figures by default before it's written to
`input/` or `solve_data/`. On very large values (e.g. annual flows
in the 1e9 range), this may shift downstream objectives by
`≤ 1e-11` relative — sub-microscopic but real and deterministic.

Override with `--precision-digits 15` to effectively disable
rounding.

## Troubleshooting checklist

- **"My objective doesn't match a pre-scaling reference."**
  Check `--precision-digits`. Default 10; was effectively 16 before
  Agent 7.
- **"The diagnostic says poorly scaled but my model is fine."**
  Scan section 5. If it's a buffer-node false positive (sink-only,
  huge virtual capacity), you can ignore it.
- **"The diagnostic says well-scaled but HiGHS is slow."**
  Check section 3 family ranges for a family without a warning
  but with a spread of 3-5 decades — scaling is within tolerance
  but dense in a structural way (e.g., a large number of identical
  small entities). Symmetry detection and row scaling may still
  help; try `--auto-scale`.
- **"Slack escape tier fires."** Section 7 lists the top cells.
  Escape activity means the bounded primary slack saturated and the
  unbounded escape valve absorbed residual demand — the model is
  telling you the inputs are asking for more than the primary cap
  allows. Usually an input-data bug; sometimes a demand peak the
  model cannot meet.
