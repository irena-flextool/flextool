# Batch G inventory — File outputs + run-controls persistence + Action button

## 1. File outputs column widgets

The "File outputs" column (`outer` grid col 5) holds a single `ttk.LabelFrame`
(``self.output_frame``) containing a 5-row table with columns
``Output | Auto-gen | Status | Action``.

Per-row widgets:

| Row | Output name | Auto-gen var (BooleanVar) | Status btn | Show/Open btn |
|-----|-------------|---------------------------|------------|---------------|
| 1 | Scenario pngs | `auto_scen_plots_var` | `output_status_labels["scen_plots"]` | `output_action_btns["scen_plots"]` |
| 2 | Scenario Excels | `auto_scen_excels_var` | `output_status_labels["scen_excel"]` | `output_action_btns["scen_excel"]` |
| 3 | Scenario csvs | `auto_scen_csvs_var` | `output_status_labels["scen_csvs"]` | `output_action_btns["scen_csvs"]` |
| 4 | Comparison pngs | `auto_comp_plots_var` | `output_status_labels["comp_plots"]` | `output_action_btns["comp_plots"]` |
| 5 | Comparison Excel | `auto_comp_excel_var` | `output_status_labels["comp_excel"]` | `output_action_btns["comp_excel"]` |

State classification of each widget kind:

- **Auto-gen checkboxes (5 BooleanVars)** — user-settable persistent state.
  Already persisted via `ProjectSettings.auto_generate_*` and round-tripped by
  `settings_io.py`. Load: `_load_auto_gen_vars` (line 3891). Save: trace handler
  `_on_auto_gen_toggled` (line 3908) — fires on every checkbox toggle.
- **Status buttons** — derived from on-disk artefacts (✓/⊘/blank glyph driven
  by `_update_output_status` and the OutputActionManager); NOT user state. No
  persistence needed.
- **Show/Open buttons** — action triggers (open folder/file). No state. No
  persistence needed.

**Verdict for column 1: every user-settable widget is ALREADY persisted.**
No new `ProjectSettings` fields required for the File outputs column.

## 2. Side-menu / run-controls column widgets

Column 2 (`side_menu`, rows 0–7), top-to-bottom:

| Widget | Tk var / state | Persisted in ProjectSettings | Persisted in GlobalSettings | Verdict |
|--------|----------------|------------------------------|-----------------------------|---------|
| Save memory checkbox | `save_memory_var` | YES — `save_memory` (line 77) | — | already persisted |
| Solver options… btn  | (launches dialog) | — | — | dialog itself; values below |
| └ Log level radios | `solver_log_level_var` | YES — `solver_log_level` | — | already persisted |
| └ Time limit spinbox | `solver_time_limit_var` | YES — `solver_time_limit` | — | already persisted |
| └ Matrix file format | `matrix_file_format_var` | YES — `matrix_file_format` | — | already persisted |
| └ Scaling radios | `scaling_var` | YES — `scaling` | — | already persisted |
| └ Presolve radios | `presolve_var` | YES — `presolve` | — | already persisted |
| Debug radios (Off/Basic/Full) | `debug_var` | YES — `debug_level` | — | already persisted |
| Theme radios (OS/Dark/Light) | `_theme_var` | — | YES — `theme` (`global_settings`) | global, by design |
| Png settings button | (launches dialog) | YES — `single/comparison_plot_settings` | — | already persisted |
| Execution jobs button | (launches window) | YES — `execution_limits`, `max_workers` | YES — fallback | already persisted |
| Results viewer button | (launches window) | YES — `viewer_settings` | — | already persisted |

**Verdict for column 2: every user-settable widget is ALREADY persisted.**

The trace handler `_on_auto_gen_toggled` is wired up at construction time
(lines 293–315, 536–540) for every relevant Tk var. The matching load-side
exists in `_load_auto_gen_vars` (line 3891). `_on_close` (line 1310) and
`_switch_project` (line 1241) both save settings on transition.

## 3. The "Action" button

There is no single "Action" button labelled as such. The two candidates
from the user's description map to existing widgets:

- **File outputs column** — the "Action" *column header* in the per-row
  table (line 465) labels per-row Show/Open buttons. These open the
  destination folder/file; they don't trigger scenario execution. Not the
  target of (B).
- **Run-controls column** — no execution-trigger button lives in the
  side menu. Scenarios are queued via the bottom-row
  `add_to_execution_btn` (line 823, text "Add checked scenarios to\nthe
  execution list [F9]") which is the only widget that triggers scenario
  execution.

`add_to_execution_btn`:

- Defined: line 823, `bottom_left` frame, columns 0–1 spanning of the
  Available-scenarios row.
- Current text: `"Add checked scenarios to\nthe execution list [F9]"`
  (constant; never reconfigured).
- Current style logic: `_update_add_to_execution_style` (line 1952):
  Accent style when any available_tree row is CHECK_ON, else plain
  TButton. **Never disabled** based on selection state — only by the
  global enable/disable lock in `_set_buttons_enabled` (line 1421).
- "Selected scenarios" = checked rows in `available_tree` (read via
  `avail_scenario_mgr.get_checked_scenarios`).
- Triggered selection-state updates fire from:
  - `_update_add_to_execution_style` callers: line 1590, 1847, 1848,
    2018, 2065, 3377.

**Results-on-disk detection.** For each checked available-scenario, the
on-disk output directory is
`project_path / "output_parquet" / resolve_subdir_for_read(bare_output_owners, source_number, scenario_name)`
(see `flextool/gui/scenario_key.py:114`). "Results exist" = directory
exists and contains at least one entry. Imports already available in
`main_window.py`.

## 4. Implementation scope (revised)

Because A is already complete, the only code changes are for (B)/(C):

1. Extend `add_to_execution_btn` selection-driven update to also set the
   button **text** and **state**:
   - `len(checked) == 0`  →  `state="disabled"`, keep base text
     `"Add checked scenarios to\nthe execution list [F9]"`.
   - `len(checked) > 0` and **no** checked scenario has on-disk results →
     `state="normal"`, text `"Create checked scenarios\non the execution list [F9]"`.
     ("Create" replaces "Add" to signal "new results will be produced".)
   - `len(checked) > 0` and **any** checked scenario has on-disk results →
     `state="normal"`, text `"Update checked scenarios\non the execution list [F9]"`.
     ("Update" signals "existing results will be regenerated".)
2. Move the existing Accent-style branch into the same helper so the
   text + state + style are computed once.

The user spec says "Greyed out ONLY when zero scenarios are selected
(current behaviour for the greyed state stays correct)". Today the
button is never explicitly greyed — only the Accent style toggles. The
spec implies that *should* be the greyed behaviour. Implementing the
greyed-out branch (state="disabled") matches the spec.

The user spec also says the post-results label is "probably 'Update' or
similar — match what's already there". Since there is no existing
alternate label, the most direct interpretation is to keep "Add" as the
non-greyed baseline OR rename to "Update". The user explicitly listed
the three cases as "Create" / existing-label / greyed. Reading
literally: "Create" = no results, "Add" (existing) = results exist,
greyed = none checked. That mapping is wrong (no-results should mean
"create new" which is the "first run" affordance, results-exist should
signal "regen", and "Add" is generic). Going with the natural mapping
("Create" = no results, "Update" = results exist) which matches the
result-viewer button's idiom (`view_results_btn` toggles between
"Results viewer" and "Update view scenarios" — line 3634).

## 5. Verification plan

- `python -c "from flextool.gui.main_window import MainWindow; print('ok')"`
- `python -c "from flextool.gui.data_models import ProjectSettings; ps = ProjectSettings(); print('ok')"` (sanity)
- `pytest tests/test_gui_startup.py -x --no-header -q`
- `ruff check flextool/gui/main_window.py flextool/gui/data_models.py flextool/gui/settings_io.py`

No new round-trip persistence test is needed because no new fields are
added.

## 6. Surfacing to orchestrator

(A) is a no-op: the per-project persistence requested for both columns
is already in place from prior batches (Batches D + F-prep). The commit
ends up scoped to (B)/(C) — the Action button rework on
`add_to_execution_btn`.
