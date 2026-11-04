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
loop now converges in 2 iterations (was 4) to 12 217 878 473.89 (gap 3.1e-7).  On that fixture the monolith
invests pipe_AB 15.7 MW and pipe_BC 0, so the Benders phase tests that need both pipes built
(`test_benders_phase0/2/3a`) moved to a re-tuned sibling, `lh2_trade_invest_retuned` (tests/conftest.py: region A
wind 1 500 MW, electrolyser 1 000 MW, liquefier 800 MW), whose monolith builds both pipes again; they pin its
optimum, M = 4.6054965e9 (commit 28022f7c, c15b review follow-ups; the previous text of this entry said 1.2217878e10,
which is the trajectory pin's value on the un-tuned fixture, not what those tests pin).  The re-tuned monolith
(oracle batch F) is 4 605 499 297.6 (the pin's rtol is 1e-3): it builds pipe_AB 346.5 MW and pipe_BC 173.2 MW and
ships A→B→C (pipe_BC B→C 5 760 MWh = C's LH2 demand, lh2_C slack 0; remaining slack battery_A 954, battery_C 5 053,
h2_C 864 MWh).  Bound: the hourly twin (`daily_group.new_stepduration` = 1) gives 8 035 445 485.6 ≥ the daily M, as
a relaxation must.  It is a new pin on a new fixture, not a re-pin; no golden file changed.
`test_benders_existing_capacity_trade`: wind_C is not built (the daily h2_C node acts as same-day storage through
the reverse electrolyser, so elec_C's price is 0 whenever wind_C produces; the hourly twin builds ~838 MW); the RP fixture `lh2_three_region_rp_invest`: M_rp
7 577 959 045.2 → 9 026 550 085.3 = the snapshot's 7 577.959 → 9 026.550.

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

## 2026-10-26 — 11 goldens with self-discharging storage (cycle c15e1, FA-1 / item 3: exact self-discharge)

**Context.** Owner decision FA-1 option (a): the self-discharge of a step is `(α·S_t + β·S⁻)·su` with
h = Δt·ln(1 + loss), α = h/(1 − e^{−h}) − 1, β = 1 − h/(e^h − 1) (coarse steps / blocks: h of the whole step),
which is exact for idle storage and for flows spread evenly over the step at any step length, as long as the loss
is constant within the step (a coarse step / block with a time-varying loss: exact idle decay only, #152).  Before: `loss·S_t`
per step whatever its length.  The tests.json batteries lose 1e-4 per hour: the loss moves from `1e-4·S_t` to
≈ `0.5e-4·(S_t + S_{t−1})`, so states shift by up to 0.0025 MWh (= 50 MWh · 5e-5) and degenerate battery /
connection schedules move at the 1e-5 level; nested chains with the 6-h storage solve shift more (the loss itself
grows with Δt).  Numbers: old golden vs regenerated, counted with the test tolerance (round 4, rtol = atol = 1e-4).

- `wind_battery`: `node_state__dt` 9 / 48 cells (≤ 0.0025 MWh); `costs__dt` 1 / 528 (2.12 of 191 258).
- `network_coal_wind_battery_co2_fullYear_availability`: `node_state__dt` 5 / 72, `connection__dt` 4 / 288
  (≤ 0.0025).
- `network_wind_coal_battery_fullYear_invest`: `node_state__dt` 6 / 72, `connection__dt` 5 / 288 (≤ 0.0025).
- `multi_fullYear_battery_nested_24h_invest_one_solve`: `node_state__dt` 32 / 288 (≤ 0.0051),
  `unit__outputNode__dt` 10 / 576 (≤ 0.0077), `costs__dt` 11 / 3168 (≤ 1.10 of 2 648), `costs_discounted`
  1 / 18 (1.3e-5 rel), `cost_node_discounted__d` 1 / 48.
- `multi_fullYear_battery_nested_multi_invest`: `node_state__dt` 13 / 288, `unit__outputNode__dt` 3 / 576,
  `costs__dt` 3 / 3168.
- `multi_fullYear_battery_nested_sample_invest_one_solve`: `node_state__dt` 16 / 288.
- `5weeks_battery_intraperiod_blocks` (on top of item 117): objective 1 843.37775 → 1 843.37962 (+1.0e-6; the block
  loss is now the exact block form), `node_state__dt` 4 / 20 (≤ 0.0043 MWh), `node__dt` 2 / 160.
- `fullYear_roll`, `multi_year_wind_no_investment`, `multi_year_wind_growth_cap`: `summary_solve.csv` text only
  (compared as exact text; totals equal to 2e-8 relative).
- `test_a_lot` (not in the design's probe table — explained here): `costs_discounted` upward slack 15 185.603 →
  15 185.605 (the battery loss, +1.056 slack at p2020/t0002); in p2035 the solver swaps two equal-cost blocks of a
  degenerate optimum: 300 MW of upward slack (30 000) and 300 MW of coal (−10 000 commodity, −8 500 CO2 per step)
  move from t0038 / t0040 to t0010 / t0013 (and 1 244.93 slack / 414.98 coal between t0026 and t0011); the column
  sums of `costs__dt` are unchanged except the +1.056 slack.  `costs__dt` 18 / 880, `node__dt` 12 / 3 200,
  `unit__outputNode__dt` 18 / 560 (the moved 300 MW; `wind_plant` west 950.66 ↔ 1 100.66 at the swapped steps).
- No golden carries the reported self-discharge column (checked in the design), so the new output writer changes
  no golden by itself.
- LP snapshot: exactly the 37 fixtures with a lossy storage change (v_state coefficients of the balance rows; the
  downstream solves of nested / chained fixtures also through the changed upstream solution: maxFlow /
  profile_flow RHS = invested capacity, nested usage targets).

The other CSVs of these scenarios are rewritten by `--regenerate` in the current number formatting (`0.0` for `0`
etc.) with no value change at the test tolerance.

Regenerated with `pytest tests/test_scenarios.py --regenerate <scenario>` for each of the 11 scenarios.  Pins
re-captured in the same commit (fixtures with a lossy storage): `tests/engine/test_benders_trajectory_pin.py` (13-process
procedure, worst relative deviation 3.6e-16; M 1.22178785e10 → 1.22178811e10, still 2 iterations) and the
`bind_within_period_blended_weights` cost pin of `tests/engine/test_rp_multi_period.py` (16 367 662 856 →
16 367 709 994, +2.9e-6).

## 2026-10-26 — two re-captured pins on the rp lh2 fixtures (cycle c15e1, FA-3 / item 15: cyclic fix_start at the wrap)

No golden CSV changed; this records two test pins re-captured in commit ed4927ed.

**Context.** `storage_state_start_binding_cyclic_period` pinned every cyclic `fix_start` node at the last step of the
first period.  For a `bind_within_solve` node in a multi-period solve that is mid-horizon.  Since FA-3 the pin is on
the step the node's first step starts from (its own state-change lag): for `bind_within_solve` the solve's last step.

- `tests/engine/test_benders_phase3b_rp_loop.py` (`lh2_three_region_rp_invest`, RP Benders): `_M_RP_EXPECTED`
  9 026 550 085.3 → 9 437 466 952.7 (+4.6 %).  The "before" value is the pin as captured before FA-1 (f145c659);
  the actual parent commit of ed4927ed gives 9 026 593 938.7 (+4.9e-6 from FA-1's self-discharge, inside the
  pin's rtol 1e-4), so FA-3 alone moves the optimum 9 026 593 938.7 → 9 437 466 952.7 (+4.55 %) (oracle batch C).
- `tests/engine/test_rp_multi_period.py::test_variant_order_on_two_periods`, `bind_within_period_blended_weights`
  cost: 16 367 709 994 → 16 820 838 994 (+2.8 %).

**Mechanism** (probe outputs `c15e/impl1/probe_fa3_{old4,new4}`).  The batteries (200 MWh, `fix_start` 0.5, no
battery investment) swap their schedules between the two periods.  Before: y2030 ended at the 0.5 pin and y2040
ended full (B, C at 1.0), so y2030 started full through the wrap.  Now: y2040's end (the wrap target of the model
start) is pinned at 0.5 and y2030 ends full.  Ending y2040 at 0.5 means refilling the batteries during y2040, the
period in which lh2_C is already short of energy: lh2_C's upward slack grows 35.65 → 77.48, and that penalty accounts
for the M_rp rise.  The rp within_period cost rises for the same reason on the same batteries.  The lh2 rp nodes'
own pins (`storage_state_start_binding[lh2_*,y2030,t0001]`) are unchanged in the LP snapshot, so no pin was lost; the
change follows from pinning the right step.

## 2026-10-26 — 14 goldens with a `bind_within_timeblock` + `fix_start` storage (cycle c15f, N8: the start pin at the wrap)

**Context.** Owner decision Q-c15e-C: with `bind_within_timeblock`, `fix_start` pinned `v_state[t_first]`, the state at
the END of the solve's first step (after its flows), while the first step's state change runs from its wrap target
`v_state[t_wrap]`, `t_wrap` = `t_previous_within_timeset` of the first step (the last step of the first timeblock).
The cycle level of the first timeblock was therefore free and only the state after step 1 was 0.1·capacity, so the
model could start the block high and spend the difference in step 1.  The pin now sits on `v_state[t_wrap]`
(`storage_state_start_binding_cyclic_period`, as for the other cyclic methods since FA-3): the block starts and ends
at 0.1·capacity.  Not a relaxation, so the cost moves either way.  The tests.json battery is `bind_within_timeblock`
+ `fix_start` 0.1 in every scenario of the `battery` alternative.

LP snapshot (`_audit_reports/lp_c15f_n8` vs `lp_c15f_pre`): exactly the 19 fixtures of the design, one row each
(`storage_state_start_binding[n,d,t_first]` → `storage_state_start_binding_cyclic_period[n,d,t_wrap]`, same
coefficients and RHS); in `wind_battery_invest_lifetime_renew_4solve` all four solves move their row (at this commit
every solve of the sequence is still "first", item N5) and y2025–y2035 also get new maxFlow RHS through y2020's wind
investment.

Objectives (sum over the scenario's solves, M CUR, before → after) and the cells that move (test tolerance):
- `wind_battery`: 300.9921 → 301.7721 (+0.26 %).  `node_state__dt` battery t0048 9.9995 → 5 (the pinned start) and
  t0001 5 → 0; one upward-slack cell (t0002) and one wind cell (t0048).
- `fullYear_roll`: only roll 0 (the only roll with the pin) 247.0595 → 252.5345; full horizon 5 979.70694 →
  5 985.18194 (`summary_solve.csv`).
- `wind_battery_invest_lifetime_renew` 4 171.0812 → 4 262.2737 (+2.19 %) and `_choice` 6 305.9472 → 6 397.1397
  (+1.45 %): wind is built earlier, p2020 782.45 → 890.35 MW (p2025 253.61 → 145.71 MW in `_renew`); the p2020 t0001
  upward slack falls 37 828 → 27 253 and the wind output of the first block (t0001–t0004) moves.  The storage build is
  unchanged; only first-block cells and the capacities move.
- `wind_battery_invest_lifetime_renew_4solve`: y2020 2 043.7255 → 2 134.9180, y2025 938.9984 → 831.6442, y2030
  1 011.0613 → 878.5047, y2035 457.6541 → 420.2032 (the later solves cost less because y2020 builds 890.35 MW of wind
  instead of 782.45; y2025 / y2030 build 270.05 instead of 253.63).  Each solve's t0001 upward slack and its
  first-block wind output move (16 cells), plus the capacities.
- `all` 12 140.4899 → 12 142.2482 (+0.015 %): the battery is still 278.51 MWh in total but is built in p2025
  (180.11) instead of all in p2020 (p2020 98.40; the inverter 34.81 → 12.30 + 22.51 MW): with the pin at the block end
  a battery built in p2020 starts (and ends) its first block at 0.1 of its larger capacity.  p2020 coal / wind /
  water_pump cells move at 14 steps of p2020 only (30 / 220 `costs__dt` cells, 36 / 800 `node__dt`, 29 / 100
  `unit__outputNode__dt`).
- `test_a_lot` 20 656.4447 → 20 659.5278 (+0.015 %) and `test_a_lot_but_not_multi_year` 15 030.4657 → 15 032.7895
  (+0.016 %): the first block of each period's battery cycle (battery_inverter, wind, coal at t0001–t0004) and the
  investment split (test_a_lot: wind p2020 2 736.62 → 2 935.03 MW, p2035 353.49 → 155.08; storage investment moves
  from p2020 to p2035, `costs_discounted` storage 216.55 → 83.44, units 2 534.10 → 2 680.97).
- `network_all_tech` 324.2954 → 324.0930 (−0.06 %): battery_inverter / coal_chp / fusion / dr_shift cells at t0001–
  t0005 and t0046–t0048 (the end of the single 48-h block, i.e. the pinned wrap).
- `network_coal_wind_battery_co2_fullYear_availability` 2 259.6910 → 2 259.6199 (−0.003 %): battery state t0001
  5 → 15, t0072 (the wrap) 0 → 5; three slack / connection cells at t0001, t0002, t0072.
- `network_wind_coal_battery_fullYear_invest` 584.6455 → 584.6454: battery t0001 5 → 0 and two inverter / slack cells
  at t0001–t0002 (an equal-cost reshuffle of the first two steps), and `connection__dt` t0068 −0.9587 → −0.9592: the
  wrap target t0072 is now pinned at exactly 5 (before 5.0005), so the self-discharge top-up before the block end
  (battery t0068–t0072 5.0025…5.0005 → 5.002…5.0) moves by 5e-4.
- `wind_battery_invest`, `multi_year_one_solve_battery`, `network_coal_wind_battery_invest_cumulative`: objective
  unchanged to 1e-9; one or two battery state / inverter cells at t0001 / t0004 (the first timeblock t0001–t0004 of the
  5-week timeset; the pinned state moved from t0001 to t0004), and one upward-slack cell in `wind_battery_invest`
  (19 800 → 19 815.8 at t0004, an equal-cost move inside the block).

**Direction vs the design (c15f design §7.3).**  The design predicted `all` −0.27 % and
`test_a_lot_but_not_multi_year` −0.03 %; the objectives above rise (+0.015 %, +0.016 %).  This is not a sign flip:
the design's table compares "Total cost (calculated) full horizon" of `summary_solve.csv`, this entry the solve
objective, and N8 moves the two in opposite directions.  Re-run on `git archive` exports of 8a6ac407^ and 8a6ac407
(both metrics from the same runs):
- `all`: objective 12 140.4899 → 12 142.2482 (+1.758 M, +0.015 %); calculated full horizon 8 106.3154 → 8 084.7481
  (−0.27 %): operational 5 450.13 → 5 441.52, investment 72.14 → 25.49, penalty 584.05 → 617.74.
- `test_a_lot_but_not_multi_year`: objective 15 030.4657 → 15 032.7895 (+2.324 M, +0.016 %); calculated full horizon
  9 075.9825 → 9 073.4647 (−0.028 %): operational 904.72 → 914.20, investment 2 119.39 → 2 107.39, penalty unchanged.
The calculated figure is recomputed from annualised costs of the realised periods and follows the investment-timing
shift (the battery / storage built later, see above), so it falls while the LP objective, which the moved pin
constrains, rises.  The design's own runs on 2320ad67 show the same: objective 12 140.4899 → 12 142.2481 and
15 030.4655 → 15 032.7894 with the prototype.  The commits between the design's base and N8 (2320ad67..8a6ac407^:
FA-1 exact self-discharge, FA-3, #133 start half, FA-2, time-varying self-discharge, #113, #118) move these two
objectives by < 2e-8 relative (`all` 12 140.4898695 at 2320ad67 vs 12 140.4898693 at 8a6ac407^), so the whole change
is N8's: the pin moves from the end of step 1 to the wrap target.

**Sub-tolerance cells.**  Regenerating rewrote whole files, so the 14 scenarios' goldens also take the within-
tolerance drift of earlier commits (several had not been regenerated since the orphan root; mainly the exact
self-discharge, FA-1), e.g. 818 `costs__dt` cells and node_state p2025–p2035 t0028 / t0037 / t0039 / t0040 (~5e-4) in
`multi_year_one_solve_battery`, 792 cells in `network_coal_wind_battery_invest_cumulative`, ~800 cells in each
`wind_battery_invest_lifetime_*` file.  The cells beyond the test tolerance are only the ones listed above.

Regenerated with `pytest tests/test_scenarios.py --regenerate <scenario>` for each of the 14 scenarios; the full
`tests/test_scenarios.py` run before regenerating failed exactly these 14 (52 passed).

## 2026-10-26 — `wind_battery_invest_lifetime_renew_4solve`, `multi_fullYear_battery_nested_multi_invest` (cycle c15f, N5: later solves continue from the realised state)

**Context.** Owner decision Q-c15e-B (+ the nested clarification of 2026-10-25): `fix_start` applies to the first
step of the model and of each nested level's first appearance; every later solve or roll starts from the latest
storage state realised before it at its level or below (`solve/_carry_routing.py`).  Before, every top-level solve
of a sequence and every per-period instance of a nested level was "first" and re-pinned to `storage_state_start`.
LP snapshot (`lp_c15f_n5` vs `lp_c15f_n8`): only `wind_battery_invest_lifetime_renew_4solve` y2025–y2035 change (each
loses its `storage_state_start_binding_cyclic_period` row; the battery is cyclic, so it takes no carry either, and the
y2030 / y2035 maxFlow RHS follow the changed wind investment); every `nested_*` fixture is byte-identical (their
levels' first solves were already first, and roll k's carry was already roll k−1's).

- `wind_battery_invest_lifetime_renew_4solve` (solve objectives, M CUR, on top of the N8 entry above): y2025 831.6442
  → 734.5360, y2030 878.5047 → 1 011.0867, y2035 420.2032 → 457.6637 (y2020 unchanged).  Without the 0.1·capacity pin
  at the end of each solve's first block, y2025 builds 145.71 MW of wind instead of 270.05 (y2030 253.63 instead of
  270.05) — exactly the p2025 / p2030 builds of the single-solve `wind_battery_invest_lifetime_renew`, which pins only
  the first period (an independent check that the sequence now follows the documented start rule).  Cells: the t0001
  upward slack of p2025 / p2030 / p2035 (41 528 → 57 724, 55 803 → 77 613, 70 078 → 97 502) and the wind output of each
  solve's first block (t0001–t0004, 12 cells), plus the capacities.
- `multi_fullYear_battery_nested_multi_invest` (+0.095 % over the solve objectives): `invest_5weeks_p2025/30/35`,
  `storage_fullYear_6h_p2025/30/35` and the first dispatch roll of each period (rolls 18 / 36 / 54) now continue from
  the realised end of the previous period's last dispatch roll (0 MWh) instead of a re-pinned 5 MWh (0.1 × 50) of free
  energy: those nine solves rise slightly (e.g. roll 18 16.4842 → 16.5920), and the realised cells are the first step
  of p2025 / p2030 / p2035: coal +4.75 MW (= 5 MWh × 0.95, the inverter efficiency) at t0001, commodity cost
  +237.49 per cell, `costs_discounted` commodity 352.6505 → 352.9198.

Regenerated with `pytest tests/test_scenarios.py --regenerate <scenario>`; the full `tests/test_scenarios.py` run
before regenerating failed exactly these two (64 passed).

## 2026-10-26 — `coal_min_load`, `coal_min_load_wind`, `fullYear_roll` (cycle c15f, #140: unrounded conversion slope and section)

**Context.** The min_load_efficiency section was `round(1/η, 6) − round(slope, 6)` while the unrounded slope
multiplied `v_flow`, so slope + section ≠ 1/η (2.0000003 instead of 2 for η 0.5); and the rolling sub-solves read
`p_slope` from the emitted `pdtProcess_slope` seed, which used `round(1/η, 6)` (1.052632 for η 0.95), while single
solves used 1/η.  Now slope = (1/η − min_load/η_min)/(1 − min_load), section = 1/η − slope and the constant-efficiency
slope = 1/η, unrounded and identical on every path.  The rounding bought no solver benefit: HiGHS works on doubles,
where 1.785714 has a 53-bit mantissa just like 1.7857142857…; polar-high only floors |coef| < small_number_threshold
and scales by powers of two.  Re-solving the before/after MPS of 7 fixtures (test_a_lot, all, network_all_tech MIP,
co2_cap_minload, reserve_dynamic_demand, coal_wind_min_uptime_MIP, stochastic_1_week_rolling_wind) gave the same
simplex / IPM iteration and MIP node counts (±1 iteration).  LP coefficients move ≤ 1.3e-6 relative (section terms,
reserve scales) and ≤ 4e-7 (rolled 1/η costs and balances).

- `coal_min_load` (`unit_online__dt`): an exact tie.  The objective moves 1145.7410541 → 1145.7410833 (2.5e-8 rel,
  the section rounding).  The old vertex (online 1.0 at t0004–t0009 and t0024–t0029) has exactly the new LP's optimal
  cost, and it is feasible in it: the fixture's numbers make "stay fully online" and "go to online 0.964 / 0.938 and
  pay a 0.036 / 0.062 start-up later" cost the same, and the rounded section broke the tie.  HiGHS now picks the
  partial-online vertex.  Output flows unchanged (the file was rewritten in the regen format only).
- `coal_min_load_wind` (`costs__dt`, `unit__outputNode__dt`, `unit_online__dt`): the same tie.  Total cost 157 199.999
  → 157 200.0: commodity cost −1 200 (t0045–t0048 run at online 0.156 instead of 0.18, coal 39 instead of 45 MW at
  t0047–t0048, wind +6 MW) and start-ups +1 200 at t0001.  The old cells 4 099.9999 / 2 999.9997 were the rounding
  artefact itself; the new ones are 3 900 / 2 800 / 2 600.
- `fullYear_roll` (`summary_solve`): the rolls now read the battery inverter's exact 1/0.95 instead of 1.052632; roll
  objectives move ≤ 1.3e-8 relative (e.g. roll 5 483.173353846 → 483.173347607), the total 5 985.18193796 →
  5 985.18185996 M CUR.  This file is compared as text, so any digit change fails it.

Regenerated with `pytest tests/test_scenarios.py --regenerate <scenario>`; the golden subset with min_load_efficiency,
rolling, nested and multi-solve scenarios (22) failed exactly these three before regenerating (19 passed).

## 2026-10-27 — `coal_min_load`, `test_a_lot` (cycle c14c part C: ramp rows of online units on online capacity,
split by unit state)

**Context.** c14c part C (`model/core.py:_eq_ramp`) changes the ramp row of every process that has BOTH online
variables and a ramp limit; it adds `v_online_*` columns to the `ramp_*` family's LP wherever that combination
exists and gives `v_startup_*`/`v_shutdown_*` their physical coefficients there instead of the pre-c14c flat
+-1.  Neither `coal_min_load` nor `test_a_lot` has that combination (no process in either fixture carries a
`ramp_method`), so part C's own code never touches either fixture's rows — confirmed by re-solving each
scenario's full chain on the d77a21d4 pre-cycle baseline (before any of c14c's commits) and on this commit:
`coal_min_load`'s "Total cost (calculated) full horizon" is 1145.74108333 M CUR on both, `test_a_lot`'s is
21352.5012583 M CUR on both — byte-identical.  Both golden mismatches are therefore a PRE-EXISTING exact tie in
each fixture's own degenerate LP that something upstream of part C (c14c parts A/B, already landed as commits
`2f5e11ca`/`c34a2650`) nudges HiGHS onto the other side of — the design doc's own prediction ("coal_min_load
unmoved") did not anticipate this; `test_a_lot`'s move is the one the design doc DID predict (p2035 storage /
commodity reallocation), just with different exact cells than its worked estimate (a per-MPS estimate, per
design section 6.1's own disclaimer).

- `coal_min_load` (`unit_online__dt`): the SAME exact tie `coal_min_load_wind`/c15f's 2026-10-26 entry above
  already documented once (`unit_online` 1.0 vs 0.964/0.938 at t0004-t0009 / t0024-t0029 — "stay fully online"
  and "go to online 0.964/0.938 and pay a 0.036/0.062 start-up later" cost exactly the same in this fixture) —
  it has now flipped back to the fully-online vertex.  Output flows unchanged; objective unchanged (see above).
- `test_a_lot` (`unit__outputNode__dt`, `node__dt`, `costs__dt`): an exact-tie storage/commodity reallocation
  within p2035 (e.g. t0010/t0011/t0013/t0016/t0025/t0027: a battery charge/discharge split between two columns
  that sums to the same node balance either way) plus a handful of ~1e-6-relative `commodity_cost`/`co2` cells
  (22687.0 → 22687.001 at p2020 t0001 etc. — within this file's own rtol 1e-4 gate, carried along because the
  regen rewrites the whole csv in the current rounding format).  Full-horizon objective unchanged (see above);
  `costs_discounted.csv` and the `cost_*_annualized__d`/`cost_*_discounted__d` csvs in this scenario's golden
  set are untouched by the regen (not in the diff).

Regenerated with `pytest tests/test_scenarios.py --regenerate <scenario>`; the required c14c sub-cycle-end UC
golden set (`coal_min_load`, `coal_min_load_MIP_wind`, `coal_min_load_wind`, `coal_unit_size_MIP_wind`,
`coal_wind_min_uptime`, `coal_wind_min_uptime_MIP`, `network_all_tech`, `test_a_lot`, `all`) failed exactly these
two before regenerating (7 passed unchanged).

## 2026-10-28 — `network_coal_wind_battery_invest_cumulative`: `connection__dt.csv`, `costs__dt.csv`,
`node_state__dt.csv` (storage-node divestment history fix)

**Context.** The fix threads the node active-divest set (`ndd_divest_active`) through every node
consumer site that previously read the process-only `edd_divest_active` and cast the node id
against its process Enum (always 0 rows — CLAUDE.md invariant #2).  This scenario's `battery` node
(`invest_method = cumulative_limits`) has both `v_invest_n` and `v_divest_n` active across its four
periods, so it exercises several of the newly-wired sites at once: `maxState` (own + every later
period), `{max,min}Cumulative_capacity_n` (cross-period headroom), `storage_state_start_binding_
cyclic_period`, and the `fix_start` term of `nodeBalance_eq` — the last of these is the one that
moves this golden. Before the fix, the fix-start energy of `battery`'s very first steps
(`nodeState_first_dt`) used only `existing` capacity; it never netted the SAME-period invest/divest
the solve itself decides, so the start state ignored a swing the model had already made. After the
fix it correctly nets `(existing + invest − divest) · state_start`, giving a different (correct)
starting state of charge in p2020.

**Objective**: re-solved on the pre-fix code (consumer sites reverted to parent `c514465a`, same
input) `total_cost.val = 9023042606.5`; on the fix, `9023040929.57` — a **9023042606.5 − 9023040929.57
= 1677 M CUR** (rel. ≈ 1.86e-7) decrease, not an exact tie. This is not a degenerate re-shuffle: the
corrected fix-start energy and the loosened cumulative-capacity headroom (divesting now relaxes
`{max,min}Cumulative_capacity_n` by its own cross-period term, the same sign as the pre-existing
own-period term) let the solver reach a marginally cheaper invest/divest/dispatch combination that
was not reachable (or not recognised as cheaper) under the bug. `v_divest_n[battery,p2020]` moves
44.52 MWh either side of the fix with zero direct objective coefficient (no `salvage_value` is
configured for this node) — the change is entirely routed through capacity accounting, not a divest
cost/reward.

- `node_state__dt.csv` (`battery`): p2020 t0001–t0004 move 6.0118/6.0112/6.0106/6.0124 →
  1.5606/1.5604/1.5603/1.5607 (corrected fix-start energy); every other period/step unchanged.
- `costs__dt.csv`: one cell, p2020 t0004 `other operational` 45914.21 → 45913.367 (ripple of the
  SOC shift into that step's dispatch).
- `connection__dt.csv`: one row, p2020 t0004 `battery_inverter` 0.0024 → 0.0006 and `west_east`
  333.1716 → 333.1733 (same ripple).
- `unit__outputNode__dt.csv` and `unit_capacity__d.csv` are in the scenario's golden set but do not
  move (any change is below the comparison's rounding).

Regenerated with `pytest tests/test_scenarios.py --regenerate network_coal_wind_battery_invest_cumulative`.

## 2026-10-29 — `multi_year_wind_no_investment/summary_solve.csv` + `multi_year_wind_growth_cap/summary_solve.csv`

**Context.** `b4175321` fixed physical CO2 reporting to count emissions from every `co2_content`
commodity consumer regardless of whether a `group` prices or caps CO2 on its node (previously a
coal unit with no priced/capped group reported 0 Mt — a real under-count, not a reporting choice:
docs/results.md's "model" CO2 total is "across all groups and units"). Both scenarios burn coal
with no priced/capped CO2 group, so their `summary_solve.csv` "CO2 [Mt]" row moved off its stale
`0` once the fix landed:

- `multi_year_wind_no_investment`: `0` → `10.776`.
- `multi_year_wind_growth_cap`: `0` → `5.53675`.

**Side investigation (same commit, same `_physical_co2_emission_pieces` widening).** The widened
`process__commodity__node_co2` set also fed `calc_costs.py`'s per-process CO2 *cost* calculation
(`cost_process_co2_dt`), which joined on `s.group_node_co2` with pandas `Index.join`'s default
`how='left'` — so a process with no priced/capped group at all (like these two scenarios' coal
unit) kept a row with `group=NaN` instead of being dropped, and `group_co2_price` has no column for
a `NaN` group, so the price resolved to `NaN`; `cost_process_co2_dt`'s `groupby('process').sum()`
then turned that all-`NaN` column into a spurious **0.0** (pandas `sum()` skips `NaN`), emitting a
new all-zero `co2` category column in `cost_unit_annualized__d.csv` / `costs__dt.csv` etc. for 10
other scenarios in the full sweep. That is not a real reporting choice either — the column added
no information (always exactly 0) and existed only because of the left-join artifact. Fixed in
`flextool/process_outputs/calc_costs.py` by changing the join to `how='inner'` (a process with no
real co2-priced/capped group is simply excluded from the per-process CO2 *cost* breakdown, same as
before `b4175321` — the model-wide physical emissions output is untouched, since it is read from
`process_emissions_co2_dt` before this join runs). No golden regen needed for those 10 scenarios:
with the join fixed, their `cost_unit_annualized__d.csv` / `costs__dt.csv` output matches the
existing (pre-`b4175321`) golden shape exactly.

**Verification.** `--regenerate` on each of the two scenarios above also perturbed one `costs__dt.csv`
cell each (`commodity_cost` at `p2020 t0001`/`t0004`, ~1e-5 relative — solver-vertex noise, well
within the `rtol=1e-4` comparison and already passing without a regen); that unrelated drift was
discarded (restored from the committed golden) so only the intended `summary_solve.csv` CO2 change
is regenerated here.

Regenerated with `pytest tests/test_scenarios.py --regenerate multi_year_wind_no_investment` and
`pytest tests/test_scenarios.py --regenerate multi_year_wind_growth_cap`.

## 2026-10-29 — `2_day_stochastic_dispatch/summary_solve.csv`

**Context.** c18d (#52, "report model-wide and group CO2 per scenario path, not pooled across
branches") split `summary_solve.csv`'s single pooled `"CO2 [Mt]"` row into one row per scenario
path under recourse. This fixture's solve has several forecast branches (`__anchor`, `lower`,
`mid`, `realized`, `upper`), so its one `"CO2 [Mt]",0,"System-wide annualized CO2 emissions for
realized periods"` row became five: `"CO2 [Mt] (<path>)",0,"...realized periods, scenario path
<path>"` for each path — all still `0` (this scenario has no active `co2_content`/CO2 group, so
the value itself is unaffected, only the row count and label). No other row or file changed:
`costs__dt.csv` (also gated in this scenario's `csvs`) diffed only in float-literal formatting
(`0.0`/`-0` vs `0`) already inside the numeric comparison's tolerance — regenerating it re-wrote
byte-identical content, confirmed by `git diff --stat` showing zero changes to that file.

Regenerated with `pytest tests/test_scenarios.py --regenerate 2_day_stochastic_dispatch`.

## 2026-11-01 — `2_day_stochastic_dispatch/{costs__dt,summary_solve}.csv`

**Context.** A `realized: yes` designation whose fan reaches the solve's committed window now makes the engine add a
deterministic realization solve right after the stochastic solve (owner decision "Stochastic realization model
(2026-11-01)" point 5).  This fixture's `2day_dispatch` carries such a designation (branch `committed`, weight 0, in
`tests/fixtures/stochastics.json`), so the run now has a second solve `2day_dispatch__realized_dispatch`.  It realizes
`period1` with the stochastic solve's investments fixed and takes the stochastic solve's realized window through the
silent last-wins: the `costs__dt.csv` rows of `period1` move from `2day_dispatch` (the weight-0 period-label shadow
dispatch) to `2day_dispatch__realized_dispatch` and now hold the realized dispatch's own costs; the forecast-branch rows
(`period1_lower/mid/realized/upper`) of `2day_dispatch` are unchanged.  `summary_solve.csv` gains the second solve's
objective (307.573 M CUR); the calculated totals add it to the stochastic solve's forecast-branch costs, as the totals
already summed every reported row; the slack rows of `period1` are the realization solve's; the CO2 rows collapse to
one model-wide row because the output's scenario-path set (`d_leaf`) is read from the last solve of the run, now the
deterministic realization solve (one path) — all CO2 values stay 0.  The stochastic
solve's own LP is byte-identical (lp_snapshot).  This designation is a stand-in that the redesign removes later
together with the label's dispatch; this golden is regenerated again then.

Regenerated with `pytest tests/test_scenarios.py --regenerate 2_day_stochastic_dispatch`.

## 2026-11-01 — `2_day_stochastic_dispatch/{costs__dt,summary_solve}.csv` (a period label has no dispatch)

**Context.** The period that opens a stochastic fan (`period1` here) keeps its name only for period-level bookkeeping
and has no dispatch any more (stochastic realization redesign, owner decision "Stochastic realization model
(2026-11-01)" point 2).  Two effects on this fixture: (1) the stochastic solve `2day_dispatch` writes no `period1`
rows (they were the weight-0 shadow dispatch; the `period1` rows come only from the realization solve, as before);
(2) each forecast of this fan-in-the-first-period now starts the solve like the label did: its first-step storage
balance takes the solve-start form instead of wrapping to the forecast's own last step.  The stochastic objective
moves 66.9291187913 → 67.3517305373; independently, the parent LP (lp_snapshot of d9c2449d) with the label's rows and
columns removed and the four first-step rows in the label's form solves to the same 67.3517305373 (HiGHS on the edited
MPS).  The forecast-branch costs in `costs__dt.csv` change accordingly (first and later steps of each branch), and
the branch `downriver` slack drops 14700 → 14400, equal to the realization solve's own 14400 (same start state).  The
calculated totals follow (374.502452125 → 374.925063871 = 67.3517305373 + 307.573333333).  The realization solve's
rows are unchanged; `period1` now comes last in the period header and slack list (the union order of the two solves).

Regenerated with `pytest tests/test_scenarios.py --regenerate 2_day_stochastic_dispatch`.

## 2026-11-01 — `2_day_stochastic_dispatch/{costs__dt,summary_solve}.csv` (forecasts-only fixture again)

**Context.** The weight-0 `committed` (`realized: yes`) branch that `tests/fixtures/stochastics.json` gave
`2day_dispatch` as a #82 workaround is removed (owner decision "Stochastic realization model (2026-11-01)" point 8):
the fan is again four forecasts (`realized`, `upper`, `lower`, `mid`, 0.25 each) with no designation, so no
realization solve is added.  The stochastic solve's LP is byte-identical (lp_snapshot), so its objective stays the
independently grounded 67.3517305373 and every `2day_dispatch` row of `costs__dt.csv` is unchanged; the 48
`2day_dispatch__realized_dispatch,period1` rows are gone.  `summary_solve.csv`: the realization solve's objective
(307.573333333) is gone and the calculated totals drop 374.925063871 → 67.3517305373 (= the stochastic LP objective);
`period1` leaves the period header (no solve dispatches it); the realization solve's `downriver` slack row is gone; the
CO2 rows split per scenario path again (the path set is read from the last solve, now the stochastic solve with its
four paths; all 0).  `tests/scenarios.yaml` pins `expected_objective: 67.3517305373` again.

Regenerated with `pytest tests/test_scenarios.py --regenerate 2_day_stochastic_dispatch`.

## 2026-11-02 — `2_day_stochastic_dispatch/{costs__dt,summary_solve}.csv` (normal outputs are realized-only)

**Context.** `model.output_horizon` no longer widens the normal outputs (owner decision "output_horizon → separate
raw parquet (2026-11-02)"): forecast-branch rows go only to the raw `horizon/` dump, flag or not.  This fixture is a
forecasts-only fan (four forecasts `realized`, `upper`, `lower`, `mid` of 0.25 each, none designated `realized: yes`,
so no realization solve) that sets `output_horizon = yes`; every dispatch step of the solve is on a forecast branch
(`period1_<branch>`), and the period label `period1` has no dispatch, so the solve realizes no step.
`costs__dt.csv`: the 192 forecast-branch rows (48 steps × 4 branches) are gone; the file is the header only.
`summary_solve.csv`: the `2day_dispatch` Objective row stays 67.3517305373 — the LP is unchanged (lp_snapshot of
this fixture byte-identical to the parent: the flag never reached the LP of a single solve, and a single solve has
no carrier into a next solve); the calculated "full horizon" / "realized periods" / operational totals drop
67.3517305373 → 0 (nothing realized to add up; they were the probability-weighted forecast costs); the four branch
`Period` columns go (no realized period, so the period rows are empty); the four per-path CO2 lines become the one
`"CO2 [Mt]",0` line (no realized period on any path); the four branch `downriver` slack rows go.
`tests/scenarios.yaml` pins `expected_objective: 0.0` (it reads the calculated full-horizon total).

Regenerated with `pytest tests/test_scenarios.py --regenerate 2_day_stochastic_dispatch`.

## 2026-11-03 — `all/{unit__outputNode__dt,node__dt,costs__dt}.csv` (online commitment stops at timeline gaps)

**Context.** The online commitment no longer links across a gap in the timeline (owner decision "Online state at
timeline gaps (2026-11-02)"): a step without a ramp predecessor takes as its online predecessor the last step of its
own stretch of the period (`params._ramp_links.online_stretch_wrap`).  The `all` fixture's `5weeks` timeset has four
blocks per period (t0001-t0004, t0009-t0016, t0025-t0028, t0037-t0040).  lp_snapshot vs the parent: the only change
is the `online_change_linear` predecessor of the four block starts of each of p2020 / p2025 — `t0001: t0040 → t0004`,
`t0009: t0004 → t0016`, `t0025: t0016 → t0028`, `t0037: t0028 → t0040` (before, each block start took the step
before the gap, and t0001 the last step of the period).  Each block now cycles on its own, so block 4 no longer pays
a start-up at t0037 for having more units online than block 3 ended with (starts 36542.179 there → 0) and block 2
re-commits inside itself (a start-up of 4876.847 at t0012); dispatch, slack and emissions shift accordingly in the
blocks.  The calculated cost total drops 2766786.219 → 2729876.737.  `test_a_lot` (same blocks, its own LP changes the
same way) and `test_a_lot_but_not_multi_year` still match their goldens and are not regenerated.

Regenerated with `pytest tests/test_scenarios.py --regenerate all`.

## 2026-11-03 — `coal_unit_size_MIP_wind/{unit__outputNode__dt,unit_online__dt,costs__dt}.csv` (same change)

**Context.** Same change as the `all` entry above.  The scenario's investment solve `y2020_5week` uses the same
`5weeks` blocks with a binary `coal_plant`; an lp_snapshot of both solves vs the parent shows, in `y2020_5week`, only
the `online_change_integer` predecessors of the block starts of p2020 / p2025 (`t0001: t0040 → t0004`,
`t0009: t0004 → t0016`, `t0025: t0016 → t0028`, `t0037: t0028 → t0040`).  With each block cycling on its own the
investment solve builds a little less wind: the realized wind capacity in the `y2020_fullYear_dispatch` solve drops
1.17674 → 1.14737 units, its only LP change (72 `maxFlow` and 72 `profile_flow_upper_limit` RHS of `wind_plant`).
The dispatch therefore runs coal more (`unit__outputNode__dt`, `unit_online__dt`), and its costs rise
164813.932 → 174931.579.

Regenerated with `pytest tests/test_scenarios.py --regenerate coal_unit_size_MIP_wind`.

## 2026-11-04 — `*/unit_capacity__d.csv` (21 goldens: new empty `in_construction` column)

**Context.** The capacity results gain the column `in_construction` (capacity decided but not yet operating, with a
construction lead time), written between `invested` and `divested` and always present. No golden scenario has a
construction lead time, so the column is empty in every one of them, and `invested` stays the capacity invested in
the period (with no lead time it comes online in its decision period). The 21 `unit_capacity__d.csv` goldens were
updated by inserting the empty column into the header and every row; no value, row or other column changed (the
`connection_capacity__d` / `node_capacity__d` files are not goldens). The 21 scenarios of `tests/test_scenarios.py`
that pin a `unit_capacity__d.csv` pass against the updated files.

## 2026-11-04 — 16 cost / summary goldens of 11 scenarios (reported costs of a run from the run start, each once)

**Context.** Owner decisions "Chained costs" (2026-11-04): the reported costs read the
run-level fields (discounted from the start of the run over the run's periods); the invest / fixed-cost lumps of a
decision come from the solve realizing the decision period and sit where the capacity starts operating; the fixed
cost of existing capacity is charged on the user's existing capacity only; the investment columns use the
investment discount factor (before: the operations factor).  lp_snapshot vs the parent: 666/666 files identical
(no LP change).  Only the files below differ beyond the comparison tolerance; the other CSVs of these scenarios
were left as they were.

- `multi_year/{cost_unit_discounted__d,cost_unit_annualized__d,costs_discounted}.csv` (chain y2020 … y2035,
  r = 0.04): the discount base was each solve's start (p2025 commodity 589.08 = 484.18 · 1.04^5, now 484.18); the
  fixed cost of existing capacity was charged on 500 MW + the carried investments (p2030 56.78 M/yr, p2035 82.33),
  now on the 500 MW (50 M/yr); the p2025 lumps covered the solve's two periods from its start, now p2025 … p2035
  from the run start (fixed cost invested 56.11 → 63.22, investment 14.89 → 16.78).  Totals: investment 70.97 →
  54.66, fixed pre-existing 1085.58 → 692.97, fixed invested 267.43 → 205.98, commodity 2811.57 → 1978.01.
- `multi_fullYear_battery_nested_multi_invest/{cost_unit_discounted__d,cost_unit_annualized__d,
  costs_discounted}.csv` (an investment solve per period, each with the next period as look-ahead): the wind
  investment lumps covered the deciding solve's periods from its own start; now the run's later periods from the
  run start.  Each decision moves by Σ_{run window} F_inv / Σ_{solve} F_inv_solve: p2020 14.14 / 8.44 = 1.676
  (206.95 → 346.75), p2025 9.51 / 8.44 = 1.127, p2030 5.70 / 8.44 = 0.676, p2035 2.57 / 4.63 = 0.555.  Investment
  total 382.04 → 469.65.
- `cost_unit_annualized__d.csv` of `multi_year_one_solve`, `multi_year_one_solve_battery`,
  `multi_year_one_solve_co2_limit`, `multi_fullYear_battery`, `multi_fullYear_battery_nested_24h_invest_one_solve`,
  `multi_fullYear_battery_nested_sample_invest_one_solve`, and `test_a_lot/{cost_unit,cost_node}_annualized__d.csv`
  (single investment solves, discounted columns unchanged): the annualized investment is the lump ÷ the investment
  factor (before ÷ the operations factor), so it is 1.04^0.5 = 1.0198 times smaller (e.g. 78.8685 → 77.3369).
- `summary_solve.csv` of `multi_year_wind_no_investment`, `multi_year_wind_growth_cap`: the "Investment discount
  factor" row is the investment factor (4.62990, 3.80544, 3.12779, 2.57082; before the operations factor 4.53999 …).

Regenerated with `pytest tests/test_scenarios.py --regenerate <scenario>` per scenario; only the files that differ
beyond the test tolerance were kept.
