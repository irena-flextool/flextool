# Handoff: pre-existing issues observed during test-infra work

Captured on 2026-05-01 during the test-infrastructure batch
(commits `66bb583`..`e8417af` on `python-preprocessing`). Both items
predate this batch and are out of scope for it; flagged here so they
don't get lost.

> **Status update (2026-05-01):** Issue 1 is **fixed** in commit
> `<this-commit>`. The fix was a one-line path change in
> `_load_entity_class_set` (`solve_data/<set>.csv` →
> `input/<set>.csv`), matching the design intent that all consumers
> read from `input/` directly now that the redundant solve_data/
> printfs have been retired. Full Layer-1 suite goes from
> 53 failing / 11 passing → **64 passing** (5:23 wall-clock).
> Issue 2 (the timing-budget wiring still wants a human scan) and
> the stash-cleanup reference remain open.

---

## Issue 1: ~50 of 64 Layer-1 scenarios fail with `KeyError` on default-HiGHS solves [FIXED]

### Symptom

Running any of ~50 Layer-1 scenarios (e.g. `pytest tests/test_scenarios.py::test_scenario[coal] -v`) fails with:

```
KeyError: "None of [Index(['coal_plant', 'coal_chp', ...], dtype='object',
              name='entity')] are in the [columns]"
```

raised somewhere downstream of `flextool/process_outputs/handoff_writers.py::write_entity_all_capacity` or
`write_unit_capacity_period` etc. (the column-oriented per-period dump writers).

### Root cause

`flextool/process_outputs/handoff_writers.py:783-810` defines
`_load_entity_class_set(work_folder, set_name)`, which reads:

```python
path = work_folder / "solve_data" / f"{set_name}.csv"
```

But the `entity.csv` file lives at `work_folder / "input" / "entity.csv"`
in current runs (the `input_writer` writes it there, and per the
"post-solve cleanup" commit `4eb6b8b` the corresponding `mod` printfs
were dropped — so `solve_data/entity.csv` is no longer produced at all).

When `_load_entity_class_set(..., "entity")` returns `[]`, the
downstream wide-format dumps initialize a DataFrame with zero columns,
then fail when trying to assign per-entity values.

The sibling helper `_load_entity` at line 223-228 in the same file already reads from `input/entity.csv` correctly:

```python
def _load_entity(work_folder: Path) -> set[str]:
    """Return the full ``entity`` set from ``input/entity.csv``."""
    path = work_folder / "input" / "entity.csv"
    ...
```

So the two readers in the same file disagree on where `entity.csv` lives. `_load_entity_class_set` was last touched while the `solve_data/<set>.csv` printfs were still being emitted; commit `4eb6b8b` retired those printfs without updating the consumers.

### Affected scenarios

54 of 64 Layer-1 scenarios fail with this signature when run with the default solver path (HiGHS):

```
fullYear_roll, coal, coal_chp, coal_chp_extraction, coal_co2_limit,
coal_co2_price, coal_min_load_MIP_wind, coal_min_load_wind,
coal_wind_min_uptime, coal_wind_min_uptime_MIP, coal_ramp_limit,
coal_retire, coal_unit_size_MIP_wind, coal_wind_ev, coal_wind_inertia,
wind, wind_battery, wind_battery_invest, wind_battery_invest_lifetime_choice,
wind_battery_invest_lifetime_renew, wind_battery_invest_lifetime_renew_4solve,
network_coal_wind, network_all_tech, network_coal_wind_battery_co2_fullYear_availability,
network_coal_wind_battery_invest_cumulative, network_coal_wind_capacity_margin,
network_coal_wind_reserve, network_coal_wind_reserve_co2_capacity_margin,
network_coal_wind_reserve_n_1, network_wind_coal_battery_fullYear_invest,
dr_decrease_demand, dr_increase_demand, dr_shift_demand,
multi_year, multi_year_one_solve, multi_year_one_solve_battery,
multi_year_one_solve_co2_limit, multi_fullYear_battery,
multi_fullYear_battery_nested_24h_invest_one_solve,
multi_fullYear_battery_nested_multi_invest,
multi_fullYear_battery_nested_sample_invest_one_solve,
5weeks_invest_fullYear_dispatch_coal_wind, water_pump, water_pump_delayed,
fusion, aggregate_outputs_network_coal_wind_chp, all, test_a_lot,
test_a_lot_but_not_multi_year, 5weeks_battery_intraperiod_blocks,
multi_year_wind_growth_cap
```

The 10 scenarios that PASS today are all either:
- pinned to `solver=glpsol` in their alternative chain (4 promoted scenarios from commit `d2018b3`: `hyphenated_entity_names`, `years_represented_half`, `years_represented_2_5`, `unidirectional_connection`) — these dodge the bug because phase 3 still writes `solve_data/entity.csv` itself
- have no `entity` set members exercised by the writer (e.g. `base`, `base_weighted`, `capacity_margin`, `fullYear`, `coal_min_load`, `multi_year_wind_no_investment`) — the empty list short-circuits before the KeyError

### Suggested fix

One of:

1. **Update the reader** (smallest patch) — change `_load_entity_class_set` at line 804 to fall back to `input/<set_name>.csv` when the `solve_data/` copy is absent:

    ```python
    path = work_folder / "solve_data" / f"{set_name}.csv"
    if not path.exists():
        path = work_folder / "input" / f"{set_name}.csv"
    if not path.exists():
        return []
    ```
    Verify the column order in the input/-side files matches what
    phase 3 used to emit; if it doesn't, the per-period dumps may
    end up in a different column order than the existing goldens.

2. **Restore the writes** — re-emit `solve_data/entity.csv`,
    `solve_data/process_unit.csv`, `solve_data/process_connection.csv` from the runner side (Python preprocessing has all the needed sets). This may be the right call if the column order in input/ doesn't match what phase 3 was emitting and the goldens lock that order.

3. **Unify** — collapse `_load_entity_class_set` and `_load_entity` into a single helper that reads from `input/`. Walk through the call sites of `_load_entity_class_set` (lines 1034, 1152, plus any others) and check that switching to `input/` doesn't change semantics.

Once fixed, remove the `solver=glpsol` pin from the 4 scenarios in commit `d2018b3` (`tests/scenarios.yaml`, search for `# Pinned to solver=glpsol`) — they'll pass on the default HiGHS path.

### Verification after fix

```bash
pytest tests/test_scenarios.py -v 2>&1 | tail -3
```

Expect: 64 passed (or close to it — there may be a small number of
unrelated golden-CSV mismatches that should be regenerated separately).

---

## Issue 2: `time_budget_seconds` wiring landed silently from an orphan agent run

### Context

During the timing-budgets work in commit `e8417af`, the diff includes
~29 lines of new code in `tests/test_scenarios.py` for reading the
`time_budget_seconds` field from `scenarios.yaml` and asserting the
wall-clock budget. That code was actually written by an earlier agent
attempt that timed out before committing — the wiring landed in the
working tree but the timing measurements + YAML data + README docs
were finished by a later run.

The wiring is functional (verified: `pytest tests/test_scenarios.py -m smoke -v` passes the timing assertion on all 5 smoke scenarios), but it should be reviewed by a human to confirm:

- The placement of `t0 = time.perf_counter()` / `elapsed_seconds = ... - t0`
  (currently wraps `runner.write_input` + `runner.run_model` + `write_outputs`,
  i.e., the full pipeline — this is the documented choice)
- The order of assertions (CSV-diff → objective → timing) matches the
  documented intent that "the timing assertion is the LAST check, so
  a CSV/objective regression is reported first"
- The `_load_scenarios` tuple structure (5 elements:
  `scenario, csvs, expected_objective, expected_objective_tolerance, time_budget_seconds`) is what was intended

### Action

Quick scan of `tests/test_scenarios.py:127-244` to confirm the wiring
is correct. If something looks off, edit before the next batch of
work touches that file.

---

## Reference: stash entries to clean up

After all the test-infra work landed, two stash entries remain from
the batch-70 cleanup work (commits `7bf605c` / `f39318a` / `71046a6`):

```
stash@{0}: On python-preprocessing: user WIP - paused, restore after batch 70
stash@{1}: On python-preprocessing: batch 70 in progress
```

`stash@{1}` (batch 70) was applied + reverted carefully + committed in
`71046a6`; safe to drop.

`stash@{0}` (the user's earlier WIP at the time of stash) was applied
back via auto-merge after batch 70 landed; everything in it is now in
the user's working tree or already committed in `16238e4` /
`4eb6b8b`. Safe to drop.

Run `git stash drop stash@{0}` and `git stash drop stash@{1}` once
you've confirmed nothing is missing.
