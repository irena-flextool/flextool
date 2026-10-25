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

## 2026-05-15 — Five alt-optima scenarios under canonical LP column ordering

**Context.** Before this regen the engine-side `Problem.add_var` and
`Problem.add_cstr` calls in `polar_high` were fed `index`/`over`
frames whose row order was driven by polars `unique()` /
hash-join bucket placement — and therefore by `PYTHONHASHSEED`.
Across processes the LP column / row order shifted run-to-run; HiGHS,
fed the same LP with different column orderings, picked different
vertices among alt-optima, which made several scenario goldens flip
pass/fail across runs.

A determinism wrapper installed in
`flextool/engine_polars/__init__.py::_install_polar_high_determinism`
now sorts every `add_var` index frame by its `dims` columns and every
`add_cstr` `over` frame by its axis columns before col/row ids get
assigned.  Cross-process verification (5 independent Python processes
per scenario) produced byte-identical LP `a_matrix_`, `col_cost_`,
`row_lower_`, `row_upper_`, plus identical objective and column/row
names.

Under this canonical ordering HiGHS now selects a single
deterministic vertex per scenario, which is in five cases NOT the
v3.32.0 glpsol vertex.  Total objective + per-column sums (per the
Phase-2 regen protocol) match v3.32.0 within `1e-4` for all five —
confirming pure alt-optima, not a numerical regression.

**Verification (per Phase-2 regen protocol):**
For each scenario below, all listed CSVs' numeric column sums match
v3.32.0 within absolute and relative tolerance `1e-4`:

- `dr_shift_demand`: unit__outputNode__dt.csv (6 cols), node__dt.csv
  (11 cols), costs__dt.csv (14 cols).  Cell-level diffs in dr_storage
  load shift between adjacent timesteps; column totals preserved.
- `multi_fullYear_battery`: unit__outputNode__dt.csv (5 cols),
  node_state__dt.csv (4 cols), costs__dt.csv (14 cols),
  costs_discounted.csv (2 cols).
- `network_all_tech`: unit__outputNode__dt.csv (10 cols), node__dt.csv
  (43 cols), connection__dt.csv (7 cols), costs__dt.csv (14 cols).
  Also bumped `time_budget_seconds` 4.5 → 6.0 — the per-add-var
  sort adds ~0.3 s on this 4.5 s scenario; legitimate cost of
  cross-process determinism.
- `coal_wind_min_uptime`: unit__outputNode__dt.csv (5 cols),
  unit_online__dt.csv (4 cols), costs__dt.csv (14 cols).  Per-cell
  coal_plant dispatch redistribution across hours that still respects
  the min-uptime constraint.
- `test_a_lot_but_not_multi_year`: unit__outputNode__dt.csv (9 cols),
  unit_capacity__d.csv (7 cols), node__dt.csv (43 cols),
  connection__dt.csv (7 cols), costs__dt.csv (14 cols).

`test_a_lot` (the multi-year sibling) was NOT regen'd — its column
sums diverge from v3.32.0 by 4.5 % in `('coal_plant', 'west')` (an
unflagged engine drift, not alt-optima), so it stays on the v3.32.0
golden until the underlying issue is investigated.

**Decision**: accept the five regens.  All confirmed equally optimal
(objective + column sums match v3.32.0 within `1e-4`); the new vertex
is what HiGHS selects under canonical LP column ordering and is now
deterministic across processes.

When to revisit:
- If `_install_polar_high_determinism` is ever removed or reordered,
  the LP column order will revert to hash-bucket order and these
  goldens will start flaking again.  Re-run determinism probes from
  the next session's handoff before regenerating against any new
  ordering.

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

## fullYear_roll — 2026-05-15

Regenerated `summary_solve.csv` after three bug fixes in the rolling-window
storage handoff path (commit forthcoming):

1. `flextool/engine_polars/input.py:4195-4253` (`build_handoff_from_flexpy`
   roll_end_state) — source changed from `period__time_last` (end-of-horizon,
   e.g. t0036 for roll_7) to `realized_dispatch` (end-of-realized-commitment,
   e.g. t0032). Mirrors v3.32.0 `_load_realized_period_time_last`. The
   docstring already documented the intended semantics; only the
   implementation diverged.

2. `flextool/engine_polars/model/core.py:805-826` (roll_continue start binding) —
   gate widened from `storage_bind_forward_only` to all `nodeState` nodes,
   matching v3.32.0 .mod:2201. Previously the constraint was silently
   skipped for `bind_within_timeset` storages (e.g. battery in this
   fixture), so v_state at first timestep of a continuation roll was free
   and the LP picked 0.

3. `flextool/engine_polars/input.py:3740-3761` (`_read_unitsize_long`) —
   now layers `input/p_entity_unitsize.csv` (full table) under
   `solve_data/p_entity_unitsize.csv` (overrides only). The handoff
   builder uses `unitsize.get(n, 1.0)`; for entities without an explicit
   override the default 1.0 was silently used, scaling `p_roll_continue_state`
   wrong by exactly the entity's unitsize (50× for battery).

Golden values match HEAD's output to ~1e-9 absolute (~1e-10 relative). Total
cost agrees with the previous golden to 4e-8 (well within the 1e-4 regen
tolerance). HEAD also now emits `"Investment discount factor",5,5,5,5`
where the previous golden left it empty — a v3.32.0 bug that HEAD fixes.

## multi_fullYear_battery_nested_24h_invest_one_solve, _multi_invest, _sample_invest_one_solve — 2026-05-15

Regenerated all 5 CSVs (`costs__dt.csv`, `costs_discounted.csv`,
`node_state__dt.csv`, `unit__outputNode__dt.csv`, `unit_capacity__d.csv`)
for all 3 nested scenarios after these engine + post-process fixes:

1. `flextool/engine_polars/_profiles.py` — `_profile_time_lf` now
   reads per-solve `solve_data/pt_profile.csv` (averaged values) when a
   workdir is available, falling back to the spine source only if the
   CSV is missing. Previously read raw hourly profile and inner-joined
   it on the sampled timestep set (24h-sampled invest), yielding wrong
   capacity-factor coefficients. Mirrors the existing `pt_node_inflow`
   path. Without this, wind invest LP coefficients were off by ~2.83×
   (0.38 vs 0.54875 for t0001 in 24h-sampled invest_24h), driving wind
   capacity to 814 MW vs 288.83 MW.

2. `flextool/engine_polars/_derived.py` — `_dt_period_active_steps`
   now prefers `_dt_period_active_steps_from_workdir` (reads
   `steps_in_use.csv`) over the spine source path. The spine source
   expands timesets against the raw hourly timeline duration, ignoring
   the solve's `new_stepduration`; that emitted 72 hourly steps for
   `invest_24h` instead of 3 (`t0001/t0025/t0049`). `dtttdt_forward_only`
   was built on the wrong grid, the state-lag join couldn't pair
   consecutive sampled steps, and `v_state` battery state-coupling was
   missing from the LP.

3. `flextool/process_outputs/read_parameters.py::read_parameters_multi`
   now filters each step's `entity_all_existing` to only that step's
   realized periods, and clears the densified-zero
   `entity_annual_*` frames on dispatch-only steps. In nested cascades
   the parent invest step's per-period values would otherwise be
   overwritten by child dispatch steps' last-wins dedup, zeroing the
   `unit investment & retirement` row in `costs_discounted.csv`.

4. `flextool/process_outputs/out_capacity.py` — `unit_capacity`,
   `connection_capacity`, and `node_capacity` now sort their period
   axis deterministically (`sorted(set(...get_level_values('period')))`)
   before the `from_product` index build. Without sorting, the
   row order in `unit_capacity__d.csv` was hash-partitioned and varied
   across runs.

## aggregate_outputs_network_coal_wind_chp — 2026-05-15

Regenerated `group_flows__dt.csv`, `connection__dt.csv`, `costs__dt.csv`,
`unit__outputNode__dt.csv` after `flextool/process_outputs/write_outputs.py`
`_backfill_group_indicator_sets` was extended to backfill the 12
``nodeGroupDispatch__*`` arc-union MultiIndex sets from the polars-LP
writer's ``solve_data/*.csv`` artefacts. Without these,
`calc_group_flows` found zero rows for unit/connection aggregator joins
and `out_group.nodeGroup_flows` emitted only slack/inflow/loss column
families — `group_flows__dt.csv` was missing
`from_unitGroup` / `from_unit` / `from_connectionGroup` /
`to_connectionGroup` / per-connection `internal_losses` columns
(23 cols vs golden 37).

`tests/test_scenarios.py` `_read_csv` extended to handle 3-row CSV
headers (group / parameter / item) used by `group_flows__dt.csv`.

## 5weeks_battery_intraperiod_blocks — 2026-05-15

Regenerated `node_state__dt.csv`, `node__dt.csv`, `costs__dt.csv`,
`summary_solve.csv` for alt-optima divergence. Objective matches
v3.32.0 exactly (1843.3814949). Only nonzero cost is the same
west-node penalty slack (485.25 created in p2020 → 956.43 M CUR).
Battery has zero operating cost so any v_state pattern that satisfies
the cyclic intra-block balance is LP-optimal. HEAD's solver picks
high-charge; v3.32.0 picked low-charge. Per the alt-optima regen
protocol (objective + col-sums match within 1e-4 → regen).

No code changes — `stateConstantWithinBlock_eq` and
`nodeBalanceBlock_eq` were already correctly implemented; the task
brief's "missing inter-block constraint" premise was wrong.

## test_a_lot — 2026-05-15

Regenerated 6 goldens after confirming LP coefficients match
v3.32.0 to formula precision. The fixture's row coefficients
(except the documented `co2_max_period` decarbonization path:
3M→2.4M→1.8M→1.2M tonnes across p2020..p2035) and invest-side
objective coefficients (scaling with `ed_entity_annual_discounted`
per the canonical `entity_annual_calc_params.py:231-243` formula)
are byte-identical between HEAD and v3.32.0. Total cost differs
~0.008% between HEAD and the prior golden (21287 vs 21289 M EUR);
the divergence is alt-optima from the LP-determinism reordering
(commits 39f2e503, f76c5123) shifting the simplex landing basis.

Per the alt-optima regen protocol, regenerated:
- `connection__dt.csv`, `costs__dt.csv`, `costs_discounted.csv`,
  `node__dt.csv`, `unit__outputNode__dt.csv`, `unit_capacity__d.csv`

Also bumped `tests/scenarios.yaml` `test_a_lot` time_budget 4.0 → 5.0s
to absorb determinism overhead (same pattern as commit 6f5f166f).

## 2_day_stochastic_dispatch — 2026-05-15

The stochastics fixture sets `model.output_horizon = "yes"` — explicitly
requesting horizon-extended outputs, which for a stochastic scenario
means all forecast-branch (period, step) rows in addition to the anchor.
HEAD's downstream readers were sourcing the canonical (period, step)
order from `realized_dispatch.csv` (anchor-only, 48 rows), so
`costs__dt.csv` and other dt-indexed outputs silently dropped the 144
forecast-branch rows.

Fix:
- `flextool/engine_polars/input.py:3409-3429` — prefer
  `solve_data/dt_realize_dispatch_set.csv` over `realized_dispatch.csv`
  when populating `flex_data.realized_dispatch` (used downstream as the
  canonical emission set). The polars cascade writes
  `dt_realize_dispatch_set.csv` honouring the `output_horizon` toggle
  via `_writer_per_solve.write_period_set_csvs:336-341`; for
  non-stochastic / non-output_horizon solves the two files carry the
  same rows (verified on fullYear_roll, still passes after this change).
- `flextool/process_outputs/read_highs_solution.py` — extended
  `_load_canonical_dt_order` with the matching dual-source fallback.

Regenerated `costs__dt.csv` (49 → 193 rows) and `summary_solve.csv`.
Gate at 64/65 (was 63/65).

## 2026-08-07 — `unit_capacity__d.csv` `existing` column on five multi-solve invest scenarios

**Files.** `multi_year`, `wind_battery_invest_lifetime_renew_4solve`,
`multi_fullYear_battery_nested_24h_invest_one_solve`,
`multi_fullYear_battery_nested_multi_invest`,
`multi_fullYear_battery_nested_sample_invest_one_solve` —
`unit_capacity__d.csv`.

**Context.** The capacity output's `existing` column read
`par.entity_all_existing` — the *cumulative* chain-sum
(`pre_existing + Σ prior-solve invest − Σ prior divest`).  On a multi-
solve run the invest→dispatch handoff folds each earlier solve's
`v_invest` into `p_entity_all_existing`, so a later (dispatch or
rolling) sub-solve reported previously-invested capacity under
`existing` — e.g. `multi_year` coal_plant `p2035` shipped
`existing=823.3178` when the true pre-invest baseline is `500`.  The
frozen v3.32.0 goldens encode this (the legacy GMPL writer folded the
same way), but it is misleading: "existing" should mean the original
pre-investment capacity, not "capacity present at the start of this
sub-solve".  Reported by the user as a bug.

**Fix** (`flextool/process_outputs/out_capacity.py` +
`read_parameters.py`).  `unit_capacity` / `connection_capacity` /
`node_capacity` now source the `existing` column from
`par.entity_pre_existing` (the static pre-invest baseline) instead of
`entity_all_existing`.  `invested` stays the per-sub-solve/roll
`v_invest` (marginal), `divested` and `total` are unchanged — so the
cumulative capacity still shows in `total`, consistent with how
single-solve multi-period runs already report (e.g. `y2020_2029_2x5y`,
unchanged).  `entity_pre_existing` is densified across the entity
universe in `read_parameters.py` so it is indexable exactly like
`entity_all_existing`.  The correction lands in the derived
`output_parquet/<scenario>/unit_capacity_ed_p.parquet` (the primary
source) and therefore propagates to CSV / Excel / SpineDB / plots /
GUI viewer / scenario-comparison in a fresh run.

**Diff shape** (- old cumulative, + regenerated baseline); only the
`existing` column moves (renew_4solve also picks up sub-1e-4 re-solve
jitter on `invested`/`total`, within `round_for_comparison + rtol=1e-4`):

```
-coal_plant,y2030_5week,p2030,567.8406,255.4772,,823.3178
-coal_plant,y2035_5week,p2035,823.3178,0.0,,823.3178
+coal_plant,y2030_5week,p2030,500.0,255.4772,,823.3178
+coal_plant,y2035_5week,p2035,500.0,0.0,,823.3178
```

**Decision** (user, 2026-08-07): accept — `existing` = original
pre-invest baseline is the intended semantics; the v3.32.0 fold was the
bug.  Single-solve scenarios and all non-capacity goldens are
byte-unaffected (full `tests/test_scenarios.py` sweep: only these five
`unit_capacity__d.csv` files changed).

**Follow-up (done same day).** The second, divergent producer —
`handoff_writers._write_capacity_per_period` / `write_entity_all_capacity`,
which wrote `output_raw/{unit,connection,node}_capacity__period.csv` and
`entity_all_capacity.csv` with `existing` from the cumulative
`entity_all_existing` — was a legacy phase-3 byte-parity artifact with no
live consumer (an exhaustive repo sweep found no production reader; the
engine, GUI, scenario-comparison and replay paths read only
`output_parquet/*.parquet`).  It was **removed**: the four writers, their
exclusive helpers, the four `_parquet_bundle` registry entries, and the
dispatch tuples in `write_all_handoffs`.  The solve-to-solve handoff
parameter feeders (`write_p_entity_period_existing_capacity`, the
`p_entity_*` chain params) and the in-memory `par.entity_all_capacity`
path are untouched.  The `period_capacity` / `periods_already_emitted`
dedup machinery is now vestigial and left for a separate teardown.

## test_a_lot — 2026-10-23 — CO2 caps count min-load section fuel

**Files.** `test_a_lot/` — `unit__outputNode__dt.csv`, `unit_capacity__d.csv`,
`node__dt.csv`, `connection__dt.csv`, `costs__dt.csv`, `costs_discounted.csv`,
`cost_unit_{annualized,discounted}__d.csv`, `cost_node_{annualized,discounted}__d.csv`.
The two `cost_connection_*` files are unchanged.

**Cause.** Fix "CO2 caps include min-load section fuel": `co2_max_period` /
`co2_max_total` now count `v_online·section·unitsize` of `min_load_efficiency`
units, as the CO2 price and the reported emissions already did.  In
`test_a_lot`, `coal_plant` is such a unit (linear UC) and the `co2_limit`
period cap of p2035 (1.2e6 t/a) binds.  The companion change to
`co2_max_total` (years represented) does not touch this scenario (its
`co2_method` is `period`); its outputs are byte-identical with and without it.

**Evidence** (run like `tests/test_scenarios.py`, HiGHS with `tests/highs.opt`):
- (a) `co2_max_period[co2_limit,p2035]` binds before and after (non-zero dual;
  reported `co2_price_period` 67.65 CUR/t both times); p2020–p2030 rows are slack.
- (b) reported `CO2_annualized[co2_limit,p2035]` (`group_CO2__d.csv`): before
  **1 208 683.2 t/a > cap 1.2e6** (the excess 8 683 t is the uncounted section
  fuel); after **1 200 000 t/a** = cap.  p2020–p2030 unchanged
  (1 667 474.4 / 1 670 022.2 / 1 670 022.2).
- (c) objective 20 655 120 922 → 20 656 442 644 CUR (+1.32e6, +0.0064 %);
  `costs_discounted.csv` totals move by the same +1.32 M (investment +4.86,
  storage investment +4.50, commodity −1.29, co2 −1.09, starts −3.98,
  upward slack −1.68).
- Every changed row is period p2035 (plus the horizon totals in
  `costs_discounted.csv`): coal_plant output falls (t0001 62.49 → 30.30 MW;
  `process__d_co2` coal_plant p2035 24 193 → 11 730 t/a), wind investment in
  p2035 rises 323.94 → 353.49 MW, battery storage investment and CHP output
  rise, the heat slack penalty falls. `connection__dt.csv`: only the
  `battery_inverter` column moves, p2035 t0001–t0004 (−142.31 / −158.19 /
  +158.19 / +158.19 → −166.64 / −149.91 / +166.64 / +166.64 MW), the larger
  p2035 battery cycling more; the transfer connections are unchanged.
- `test_a_lot_but_not_multi_year` (single period p2020, 3e6 t/a cap does not
  bind) and the other CO2 / min-load goldens (`coal_co2_limit`,
  `coal_co2_price`, `multi_year_one_solve_co2_limit`,
  `network_coal_wind_battery_co2_fullYear_availability`,
  `network_coal_wind_reserve_co2_capacity_margin`, `coal_min_load*`) pass
  unchanged.

Regenerated with `pytest tests/test_scenarios.py --regenerate test_a_lot`.

## network_coal_wind_reserve_n_1, all — 2026-10-24 — dynamic / n-1 reserves need no reservation

**Files.** `network_coal_wind_reserve_n_1/` — `unit__outputNode__dt.csv`,
`connection__dt.csv`, `process__reserve__upDown__node__dt.csv`, `costs__dt.csv`;
`all/` — `unit__outputNode__dt.csv`, `node__dt.csv`, `costs__dt.csv`.

**Cause.** Fix "dynamic and n-1 requirements activate their providers" (#40, owner
answer Q5 2026-10-24; reverses the reservation filter of 30e0c5a8): a reserve is active
when it has a requirement — a non-zero `reservation`, or a dynamic / large_failure
group driven by an `increase_reserve_ratio` / `large_failure_ratio` at one of its nodes
(`_reserve_methods.demanded_reserves`).  Both scenarios use the `reserve_n_1`
alternative: an `n_1` up reserve on `electricity` with method `large_failure_only`,
`coal_plant` failing with `large_failure_ratio` 0.1 and `wind_plant` (reliability 0.6,
`max_share` 1) and `coal_plant` (`max_share` 0.01) as providers, and no `reservation`.
Before the fix the reserve was inert (no `v_reserve`, no n-1 row); now coal's failure
(10 % of its output) must be covered, and wind holds the n-1 reserve of coal.

**Evidence** (LP of `tools/lp_snapshot.py`, solved with HiGHS):
- LP diff (both): + `v_reserve` (coal, wind), + `v_reserve_total[n_1]`, +
  `reserveBalance_up_n_1_eq[n_1,up,electricity,coal_plant,…]`, + `reserve_process_upward`,
  + `reserve_total_upper`, `v_reserve` terms in `maxFlow` and `profile_flow_upper_limit`
  (and in `maxFlow_online_linear` for `all`, where coal is linear UC), cost on
  `vq_reserve[n_1,…]` (scale now 50 MW).  Nothing else changes.
- `network_coal_wind_reserve_n_1`: objective 585.712458 → 833.076608 M CUR (+42.23 %).
  Holding the reserve on wind (1/0.6 MW of curtailment per MW of requirement) beats a
  shortfall at `penalty_reserve` 20 000, so the reserve slack stays 0 and the cost appears
  as east's unserved energy (`penalty_up` 500): wind output Σ 29 830 → 25 877 MWh (the
  3.953 unit-steps of wind `n_1` reserve × 1000 MW), coal output Σ 22 658.7 → 23 717.1
  (commodity cost +52 920 = +1 058 MWh × 50 CUR/MWh), `west_east` flow Σ 16 247 → 13 642,
  upward slack penalty Σ 2.076 M → 3.379 M (+2 605 MWh × 500).  Per-step cost change
  1.355 M × 182.5 (2 days → year) = 247.36 M CUR = the objective change.  The reserve
  CSV gets `coal_plant` / `wind_plant` `n_1` columns (wind Σ 3.953, coal 0).
- `all`: objective 10 585.024237 → 12 140.489869 M CUR (+14.70 %).  Coal commits more
  (output Σ 18 724 → 22 397 MWh; commodity, CO2 and start costs up), wind output
  Σ 11 940 → 8 207, the battery cycles more (west from/to connections 50 / −11 167 →
  260 / −11 399), `water_pump` runs 1 917 instead of 2 000 MWh and `water_source` gets a
  downward slack (83.4 MWh).  Upward reserve slack penalty stays 0.
  Reconciliation (objective terms of the solved LP, M CUR): `v_flow` costs (fuel, CO2,
  variable) 6 678.08 → 7 669.90 (+991.82), `v_online_linear` 2 675.83 → 3 069.99
  (+394.16), `v_startup_linear` 136.12 → 160.37 (+24.25), new battery storage investment
  `v_invest_n[battery,p2020]` 278.5 MWh (+72.14; `battery_inverter` 34.8 MW, no cost
  term), `vq_state_down` (the `water_source` slack) +73.10, `vq_state_up` unchanged
  1 095.00: Σ +1 555.47 = the objective change.  In the realized `costs__dt.csv` (p2020)
  the per-step sums move by +338 660 CUR: commodity +171 046, CO2 +145 389, starts
  +5 536, downward slack +16 689 (= 83.44 MWh × `penalty_down` 200 at `water_source`).
  Why the pump runs less: `water_pump` must draw 1 MWh of west electricity per MWh of
  water (`water_pump_fix`), and in t0001, t0012 and t0025 (25.3, 4.7 and 53.5 MWh) the
  marginal west electricity — coal now committed and dispatched with its n-1 reserve
  held on curtailed wind — costs more than leaving the water at the source at
  200 CUR/MWh.  The new battery capacity is what carries the extra battery cycling
  listed above.
- Coal's own `n_1` reserve is degenerate: it helps no row (it is excluded from its own
  failure row and coal is the only failing process), so its column is 0 at this solver
  but not unique.
- No other golden scenario has a dynamic or n-1 group (scan of every JSON under `tests/`:
  dynamic / n-1 data only in `tests.json:reserve_n_1`, used by these two scenarios, and in
  `scaling_stress/input.json`, which has no golden); the full `tests/test_scenarios.py` run
  passes with only these two regenerated.

Regenerated with `pytest tests/test_scenarios.py --regenerate network_coal_wind_reserve_n_1`
and `--regenerate all`.

## 2026-10-25 — `coal_ramp_limit/unit_ramp__outputNode__dt.csv` (cycle c14a, ramp predecessors #34)

**Context.** Ramp limits now apply only between a step and its adjacent predecessor in the un-rolled solve; the
first step of a solve has none (it used to wrap to the last step of the period).  The reported ramp follows the
same link (`s.ramp_link`), and is empty where there is no predecessor.

- Only change: `t0001` `0` → empty.  It was `flow[t0001] − flow[t0048]` (500 − 500, the cyclic wrap).
- Objective unchanged (1145.123625): the removed `ramp_sink_up_constraint[…,t0001]` row was slack (LP snapshot:
  the only diff of the fixture).  `unit__outputNode__dt.csv` is unchanged (verified; the regen rewrote it with the
  same values, so its committed text was kept).

Regenerated with `pytest tests/test_scenarios.py --regenerate coal_ramp_limit`.

## 2026-10-25 — `lh2_three_region/objective.json` and the Benders trajectory pin (cycle c15b, coarse groups Q1: #11, #12, #33, #36, #8)

**Context.** A node of a group whose `new_stepduration` is longer than the solve's step is now balanced once per
coarse step by summing all its hourly balance terms (`nodeBalanceCoarse_eq`); every flow stays hourly (owner
decision Q1).  Before, an arc from an hourly to a daily node (the electrolysers, elec_* → h2_*) entered the daily
H2 balance only at the first hour of each day (×24) — the other 23 hours turned electricity-free into hydrogen —
and daily↔daily arcs (liquefiers, pipes) had 23 free hours per day.

**The fixture is under-sized**: it needs penalty slack even when solved fully hourly (daily_group step 1:
objective 5 855 730 720.6, up-slack at battery_A 8 019, battery_B 5 214, h2_A 5 492, h2_B 3 038, lh2_A 3 931).
The daily-group objective is a relaxation of that hourly one, as it should be (5 494 349 607.5 < 5 855 730 720.6);
the old 4 815 813 387.9 was below both only because of the energy creation.

`lh2_three_region/objective.json`: 4 815 814 307.91 → 5 494 349 607.51 (+14.09 %).  Where it goes (summed solution,
MWh = v_flow × unitsize over the 168 h): the electrolysers now draw electricity every hour they make hydrogen —
electrolyser_A elec→h2 6 571 (was 659), electrolyser_B 8 153 (118), electrolyser_C 16 048 (396); coal_A and coal_B run
at their 400 MW maximum all week (67 200 each, were 62 022 / 55 721), coal_C stops (3 420 → 0), wind_C 25 850 →
47 559; the elec up-slack of A / B (3 845 + 2 127) is gone, battery_A's up-slack (penalty 3 000, the cheapest
energy source in region A) grows 6 619 → 9 902 and lh2_A gets a shortfall of 9 989 (penalty 5 000).  The
electrolysers also run in reverse (h2 → elec 3 845 / 2 127 / 3 952 at A / B / C) — the penalty arbitrage of the
under-sized fixture (battery-node slack at 3 000 turned into electricity worth 8 000), not a model artefact.  The LP
snapshot reproduces the objective exactly (`lh2_three_region/lh2_week.mps`, 4 815.813 → 5 494.350 scaled; columns
unchanged, 9 240; the 42 `nodeBalanceBlock_eq` rows become 42 `nodeBalanceCoarse_eq` rows with one coefficient per
hour, and the 966 `stateConstantWithinBlock_eq` rows become 483 `stateConstantCoarse_eq` rows — the other 483 were
empty rows for the non-storage h2_* nodes).

Regenerated by removing `tests/expected/lh2_three_region/objective.json` and running
`pytest tests/test_lh2_three_region.py` (the test writes a missing golden), then re-running it (green).

`tests/engine/test_benders_trajectory_pin.py` (`lh2_three_region_trade_invest`, λ = 0): re-captured with the
documented 13-process procedure (3 random + `PYTHONHASHSEED` 0–9; worst relative deviation 7.2e-16).  The
monolith optimum moves 8 544 247 283.47 → 12 217 878 452.23 (the same mechanism: region A's wind can no longer
feed C through free hydrogen, so pipe_BC is not built and C's LH2 demand of 5 760 MWh is served by slack) and the
loop now converges in 2 iterations (was 4) to 12 217 878 473.89 (gap 3.1e-7).  The other Benders tests on that
fixture are re-pinned to the same ground truth (`test_benders_phase0/2/3a`: M = 1.2217878e10; monolith invests
pipe_AB 15.7 MW, pipe_BC 0; `test_benders_existing_capacity_trade`: wind_C is not built; the RP fixture
`lh2_three_region_rp_invest`: M_rp 7 577 959 045.2 → 9 026 550 085.3 = the snapshot's 7 577.959 → 9 026.550).

## 2026-10-25 — `fullYear_roll/summary_solve.csv` (cycle c15e1, item 119: cyclic storage takes no roll carry)

**Context.** In the second and later rolls of a rolling solve the first-step balance row of a
`bind_within_timeblock` / `bind_within_period` / `bind_within_solve` storage held both the cyclic state change
(`v_state[wrap] − v_state[first]`) and the roll carry (`− v_state[first] + roll_continue_state`): −2·state_unitsize on
the first state, i.e. the start was forced halfway between the wrapped and the carried level.  how_to.md (rolling
notes) documents cycle-only behaviour, so cyclic nodes no longer take the carry.

`fullYear_roll` (battery `bind_within_timeblock`, rolling 8 h / 4 h over 72 h × 4 periods): total cost (full horizon
and realized) 6 047.34416824 → 5 979.70680794 M CUR (−1.12 %); 27 of the 72 roll objectives change (e.g. roll 4
594.174 → 591.850, roll 5 488.235 → 483.173) and, observed in this run, none rises — the new row is not a pure
relaxation of the old one (different feasible sets), so a lower cost is the observed result, not a guarantee.  The
"Created" (penalty energy) lines of west fall accordingly (p2020 2 764.7 → 2 740.8, p2025–p2035 2 760.2 → 2 727).
LP snapshot: only the continuation rolls of `coarse_roll`, `coarse_roll_jump12`, `conn_online_roll`, `ramp_rolling`
change (the battery's first-step row: −2·su → −su on `v_state[first]`, the carry leaves the bound).

Regenerated with `pytest tests/test_scenarios.py --regenerate fullYear_roll`.

## 2026-10-25 — `5weeks_battery_intraperiod_blocks` (cycle c15e1, item 117: block row = summed node balance)

**Context.** `nodeBalanceBlock_eq` (the per-timeset-block balance of a `bind_intraperiod_blocks` storage) is now
`nodeBalance_eq` summed over the block, with the state change through the block lag (the previous step of the
period, the period's first step continuing from its last step).  With the state constant within a block the state
terms telescope to `v_state[b⁻] − v_state[b]`, so `v_state` of a block is its state at the END of the block (before:
the state at its start, `S[b_next] − S[b]`), and the `fix_start` pin moves from `storage_state_start_binding
[battery,p2020,t0001]` to `storage_state_start_binding_cyclic_period[battery,p2020,t0040]` (the same physical
instant: the start of block 1 is the end of the period's last block).  The two LPs admit the same schedules and
differ only in which state each block's (linear, × Δt) self-discharge is charged on: the end state instead of the
start state.

- `node_state__dt.csv` battery column: p2020 block states (start labelling) 5 / 0 / 0 / 0 → (end labelling)
  0 / 7.3101 / 26.7731 / 5 MWh for blocks t0001 / t0009 / t0025 / t0037; p2025 unchanged.
- The solver lands on another vertex of a nearly flat optimum: objective 1 843.3814949 → 1 843.37775 (−2.0e-6
  relative; full-horizon total 956.4314949 → 956.42775 M CUR).  The battery now arbitrages in p2020 (charges at
  t0009, discharges at t0027–t0028, empty at t0039–t0040 with 10 MW more coal there), as p2025 already did:
  `node__dt.csv` 6 of 160 rows, `costs__dt.csv` 1 of 220 cells (t0004 commodity 15 526.71 → 15 525.0).
- Verified term by term against the design probe (`test_q5_block_rows.py`, the same objective 1 843.37775).
- LP snapshot: only `5weeks_battery_intraperiod_blocks` and `agg_intraperiod` change.

Regenerated with `pytest tests/test_scenarios.py --regenerate 5weeks_battery_intraperiod_blocks`.
