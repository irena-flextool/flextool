# Golden regeneration log

This file records cases where a `tests/expected/<scenario>/<csv>` golden
was regenerated against current code rather than left as the original
v3.32.0 reference.  The convention for this project is that
`tests/expected/` is the v3.32.0 frozen output of flextool's legacy
GMPL pipeline, used as the correctness oracle for the engine_polars
cascade at HEAD (see `tests/test_scenarios.py`).  Any departure from
that convention is documented here so future readers don't have to
re-investigate why the file moved.

## 2026-05-14 — `hyphenated_entity_names/costs__dt.csv`

**Context.** Δ.22 deleted `bin/glpsol` along with the GMPL solver
pipeline.  Six v3.32.0 test scenarios had `solver=glpsol` pinned per-
entity in `tests/fixtures/tests.json` as a documented temporary
workaround for an unrelated HiGHS handoff bug (see commit `d2018b3c`'s
message).  At HEAD these pins started raising `FlexToolUserError:
Solver 'glpsol' is not installed`.  Commit `e03e3047` switched the
pins to `solver=highs` (the planned cleanup per `d2018b3c`).

Five of the six scenarios then matched their v3.32.0 goldens
byte-identically under HiGHS — same LP vertex picked.  The sixth,
**`hyphenated_entity_names`**, has a degenerate optimum: there are
two equally-optimal LP solutions for the parallel `west-east-aux`
connection (VOM=0.1 cost would push flow through it only at exact
tie-breaks).  glpsol picked one vertex; HiGHS picks the other.

- **Total objective unchanged**: still `548.39350 M CUR`, matches
  the pinned `expected_objective` at `1e-4` tolerance.
- **Cost allocation shifted**: HiGHS's chosen vertex routes zero
  flow through `west-east-aux`, so `other_operational` is zero and
  `commodity_cost` absorbs the ~75 CUR/day previously appearing in
  `other_operational`.
- **`connection__dt.csv` for the same scenario was NOT regenerated**
  — it happens to be byte-identical under HiGHS.

**Decision** (recorded by the user, 2026-05-14): **accept the regen.**
Both LP vertices are equally optimal; v3.32.0's glpsol-specific
allocation has no physical preference over HiGHS's choice once
glpsol is gone.

Cross-references:
- `tests/scenarios.yaml` carries the same note inline next to the
  `hyphenated_entity_names` entry.
- `tests/fixtures/tests.json` solver pins for the six scenarios
  were updated in commit `e03e3047` (`"glpsol"` → `"highs"` for
  scenarios: `y2020_2029_1x10y`, `y2020_2029_2x5y`,
  `hyphenated_entity_names`, `years_represented_half`,
  `years_represented_2_5`, `unidirectional_connection`).

When to revisit:
- If a future change adds LP-side tie-breaking (e.g. ε-perturbation
  on the parallel connection) that forces a deterministic vertex,
  this golden may need a second regen to match the new vertex.
- If glpsol is ever reintroduced as a supported solver, the
  fixture pins could be reverted and the original v3.32.0 golden
  restored — at which point this log entry becomes historical.

## 2026-05-14 — `multi_year_wind_no_investment/unit_capacity__d.csv`

**Context.** v3.32.0's golden shipped `invested=0.0` on the
`wind_plant` rows for periods `p2030` and `p2035`, both of which
have `existing=0` (the wind plant had no presence in those periods —
no realized invest decision either).  Per FlexTool's "unrealized
cell semantics" convention (see project memory
`project_unrealized_cell_semantics.md`), an invest-eligible-but-not-
realized `(entity, period)` cell should be empty/NaN, NOT `0`.  The
distinction matters: `0` says "model decided zero investment",
empty/NaN says "no realized decision for this cell".  The engine_polars
cascade output already follows the empty convention (via the
intersection filter at `out_capacity.py` from commit `cd52d383`);
v3.32.0's `0.0` was the inconsistency.

**Diff** (- v3.32.0 golden, + regenerated, commit `d534116b`):

```
 wind_plant,y2020_2035_5week,p2020,1000,72.6275,,1072.6275
-wind_plant,y2020_2035_5week,p2025,1000,1277.4294,,2350.0568
-wind_plant,y2020_2035_5week,p2030,0,0.0,,1277.4294
-wind_plant,y2020_2035_5week,p2035,0,0.0,,0.0
+wind_plant,y2020_2035_5week,p2025,1000,1277.4293,,2350.0568
+wind_plant,y2020_2035_5week,p2030,0,,,1277.4293
+wind_plant,y2020_2035_5week,p2035,0,,,0.0
```

Two conceptual changes:
1. `p2030` and `p2035` `wind_plant.invested`: `0.0` → empty.
   Convention alignment, the load-bearing change.
2. `p2025` and `p2030` `wind_plant.invested` / `total`:
   `1277.4294` → `1277.4293` and `2350.0568` (unchanged) /
   `1277.4294` → `1277.4293`.  Fifth-decimal solver noise between
   glpsol (v3.32.0) and HiGHS (HEAD); well within
   `round_for_comparison + rtol=1e-4` tolerance.

**Decision** (user, 2026-05-14): accept.  The empty-cell convention
is semantically correct; the v3.32.0 ship of `0.0` was a writer-side
inconsistency (other v3.32.0 scenarios such as `y2020_2029_2x5y`
correctly ship empty for the same kind of cell).

**Not affected by this regen.** Other v3.32.0 goldens with
`invested=0.0` rows are NOT regenerated — most of those are realized
cells where the model genuinely chose zero investment.  Only the
`wind_plant.p2030` and `wind_plant.p2035` cells here matched the
unrealized-cell pattern (existing also 0 → no presence → no realized
decision).

## 2026-05-15 — `coal_co2_limit/unit__outputNode__dt.csv` + `coal_co2_limit/costs__dt.csv`

**Context.** Post-HiGHS-pin (`9205bf44`) the v3.32.0 glpsol golden for
`coal_co2_limit` no longer matches at HEAD because HiGHS picks a
different equally-optimal LP vertex on this CO2-capped dispatch
scenario.  The model has a binding CO2 cap of 3 t (`model_wide`
column in `co2.csv` agrees exactly between produced and golden), but
the per-timestep coal dispatch profile can be redistributed across
hours without changing the total energy delivered, total cost, or
total emissions.

**Verification (per Phase-2 regen protocol)**:
- LP objective (full horizon): 1780.16775 M CUR, identical between
  produced and v3.32.0 reference (computed from `costs__dt.csv` column
  sums × annualization).
- Column sums match exactly across all 3 CSVs:
  - `unit__outputNode__dt.csv` coal_plant|west: 19339.24255 MWh (same).
  - `costs__dt.csv` commodity_cost: 966962.127 CUR (same);
    upward slack penalty: 8787381.71 CUR (rel 1.9e-10).
  - `co2.csv` model_wide: 3.00 t (same — cap binds).
- Two independent pytest runs with separate basetemps produced
  byte-identical output, so the HiGHS pin in `9205bf44` is
  deterministic for this scenario.

**Diff summary** (`tests/expected/coal_co2_limit/`):
- `unit__outputNode__dt.csv`: 96 of 48 data rows changed (per-cell
  coal dispatch redistributed; column sum preserved).
- `costs__dt.csv`: 38 rows changed (per-cell commodity/slack cost
  shuffle that mirrors the dispatch profile).
- `co2.csv`: byte-identical, NOT regenerated.

**Decision**: accept the regen.  Both LP vertices are equally
optimal; v3.32.0's glpsol-flavoured allocation has no physical
preference over HiGHS's choice once glpsol is gone.

When to revisit:
- If a future change tightens HiGHS to pick the v3.32.0 vertex
  (e.g. an ε-perturbation in the cost coefficients), the golden may
  need a second regen.
- If glpsol is ever reintroduced as a supported solver, this golden
  could be reverted to the v3.32.0 reference.

## 2026-05-15 — Phase-2 candidates flagged (NOT regenerated)

Six candidates from the prior diagnosis cluster were investigated
under the same protocol but failed the vertex-swap criterion.  These
are flagged for follow-up rather than regenerated; see
`/tmp/real_bugs_flagged.md` (overnight run) for full evidence.

| Scenario | Symptom | Why not regen |
|---|---|---|
| `dr_shift_demand` | `dr_storage` 4.2% cells, column sum preserved | Run-to-run nondeterminism: regen and verify runs pick different equally-optimal vertices despite the HiGHS pin |
| `network_all_tech` | `dr_storage` 8.3% cells | Same as `dr_shift_demand` — contains the same dr_storage entity, same nondeterminism family |
| `multi_fullYear_battery` | `wind_plant` 2.5% cells | Run-to-run nondeterminism in per-cell wind dispatch |
| `test_a_lot` | `coal_plant` 1.25% cells | Run-to-run nondeterminism (wind↔CHP balance flips between runs) |
| `multi_year_one_solve_co2_limit` | `coal_plant` 11.25% cells | **NOT vertex swap** — objective itself differs by 0.9% (1532.30 vs 1546.07), CO2 emissions 85% below cap, coal output 26% lower.  Likely real engine bug. |
| `coal_wind_ev` | (passed at investigation time) | Already cleared by HiGHS pin; no regen needed |

The first four point to incomplete HiGHS determinism coverage for
scenarios containing `dr_storage` or `wind_plant`-heavy dispatch with
multiple optima.  The fifth points to a real constraint/annualization
bug that needs investigation rather than golden regen.

When to revisit (all five flagged):
- After the HiGHS determinism pin is extended (or after a more
  aggressive ε-perturbation lands), re-run all five and re-classify.
- The `multi_year_one_solve_co2_limit` divergence should be triaged
  separately — check that the model_wide CO2 cap reaches the solver
  for multi-year-one-solve scenarios.
