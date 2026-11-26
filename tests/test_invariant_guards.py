"""Static guards for the CLAUDE.md invariants that no behavioural test pins at the point of the mistake.

Each guard scans source text and fails with the rule and the way out, so the message carries the guidance where it is
needed.  The counted ones are ratchets: a count may fall (lower its baseline here) but never rise without a decision.

* Axis detection by column name in the engine (invariant 2): a DB parameter's period / time axis comes from
  ``_param_shapes`` (``resolve_param_shape`` → ``broadcast_to_period_time`` → ``promote_param_to_dt``), never from
  ``"t" in df.columns``.  Spine names an unnamed Map index ``"x"``, so a column-name check misreads a time series as a
  scalar and cross-joins it against the whole timeline.
* Tests reading a generated template database (invariant 3): tests build their database from JSON (``json_to_db``,
  ``schema_db_url``); a template ``.sqlite`` lags the schema and concurrent xdist opens of one file can hang Windows CI.
* Value columns rendered with a cast in the vectorized emit (invariant 5): render with ``_render_value_column`` (the
  ``repr()`` loop); ``.cast(pl.Utf8)`` diverges on sci-notation padding and ``NaN``.
* Module size: a module stays under ``MAX_MODULE_LINES``; the larger ones may not grow more than ``GROWTH_SLACK``
  lines past their size when the ratchet was set.  Agents read and edit large modules badly; new code goes in a new,
  cohesive module.
* ``node_type`` in the ``Base`` alternative of a fixture or shipped example JSON: an old migration step wrote
  ``commodity`` there for every node, and a scenario that ranks ``Base`` above the real type turns the node into a
  commodity node.  Give a node its type in the alternative that defines it.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_AXIS_BY_NAME = re.compile(r"""["'](t|time|step|d|period)["'] (not )?in [A-Za-z_][\w.\[\]"']*\.columns""")

#: Column-name axis checks per engine module today.  They read model-internal frames whose columns the engine itself
#: wrote, which is fine; the count stops a new one appearing unnoticed.
AXIS_BY_NAME_BASELINE: dict[str, int] = {
    "flextool/engine/input/_inmemory_reader.py": 3,
    "flextool/engine/input/_param_shapes.py": 4,
    "flextool/engine/model/_commodity_ladder.py": 1,
    "flextool/engine/model/_cumulative_invest.py": 2,
    "flextool/engine/model/_period_rows.py": 2,
    "flextool/engine/model/core.py": 3,
    "flextool/engine/output/_reserve_report.py": 1,
    "flextool/engine/preprocess/_invest_seeds.py": 2,
    "flextool/engine/preprocess/_solve_context.py": 4,
    "flextool/engine/preprocess/flex_data.py": 21,
    "flextool/engine/preprocess/params/_branch.py": 5,
    "flextool/engine/preprocess/params/_coarse.py": 1,
    "flextool/engine/preprocess/params/_derived.py": 19,
    "flextool/engine/preprocess/params/_direct.py": 4,
    "flextool/engine/preprocess/params/_discount.py": 1,
    "flextool/engine/preprocess/params/_lifetime_costs.py": 1,
    "flextool/engine/solve/_benders.py": 1,
    "flextool/engine/solve/_warm.py": 2,
}

_TEMPLATE_DB = re.compile(
    r"""["'/](comparison_settings|examples|input_data_template|output_info|output_settings|results_template"""
    r"""|time_settings_only)\.sqlite["']""")

#: Test modules that name a template database today (most copy it to tmp_path or open it read-only first).
TEMPLATE_DB_READERS: frozenset[str] = frozenset({
    "tests/conftest.py",
    "tests/docs/test_parameter_catalogue.py",
    "tests/engine/test_param_shapes_phase_a.py",
    "tests/engine/test_spinedb_settings.py",
    "tests/gui/test_input_source_numbering.py",
    "tests/model/test_examples_e2e.py",
    "tests/spinedb_backend/test_memory_budget.py",
    "tests/spinedb_backend/test_parsed_value_eviction.py",
    "tests/test_ensure_settings_db.py",
    "tests/test_export_to_tabular.py",
    "tests/test_open_results_db.py",
    "tests/test_safe_names.py",
    "tests/update_flextool/test_ensure_runtime_files.py",
    # names the schema sources of old release tags as git paths; opens no file
    "tests/update_flextool/test_migration_sweep_sources.py",
    "tests/update_flextool/test_settings_param_sync.py",
})

MAX_MODULE_LINES = 1500
GROWTH_SLACK = 150

#: Lines of each module over MAX_MODULE_LINES when the ratchet was set (2026-11-24).  Lower an entry when the
#: module shrinks; drop it when it is split under the limit.
LARGE_MODULES: dict[str, int] = {
    "flextool/assistant/chat_cli.py": 1592,
    "flextool/assistant/session.py": 1948,
    "flextool/engine/input/_param_shapes.py": 3850,
    "flextool/engine/input/_solve_config.py": 1915,
    "flextool/engine/input/_spinedb_reader.py": 1890,
    "flextool/engine/input/_timeline.py": 1872,
    "flextool/engine/model/_group_slack.py": 1570,
    "flextool/engine/model/_reserve.py": 1897,
    "flextool/engine/model/core.py": 7349,
    "flextool/engine/preprocess/emit/_arc_unions.py": 1577,
    "flextool/engine/preprocess/emit/_inflow_scaling.py": 1555,
    "flextool/engine/preprocess/emit/_period_params.py": 2576,
    "flextool/engine/preprocess/flex_data.py": 8356,
    "flextool/engine/preprocess/params/_branch.py": 2119,
    "flextool/engine/preprocess/params/_derived.py": 9944,
    "flextool/engine/preprocess/params/_direct.py": 3243,
    "flextool/engine/preprocess/params/_projection.py": 2206,
    "flextool/engine/solve/_benders.py": 3657,
    "flextool/engine/solve/_orchestration.py": 6019,
    "flextool/engine/solve/_region_filter.py": 2874,
    "flextool/engine/solve/_run_model.py": 3134,
    "flextool/engine/solve/_solver_dispatch.py": 1874,
    "flextool/engine/solve/_subprocess_solve.py": 2951,
    "flextool/export_to_tabular/excel_writer.py": 3570,
    "flextool/gui/dialogs/plot_settings_picker.py": 2215,
    "flextool/gui/execution_manager.py": 2080,
    "flextool/gui/main_window.py": 6943,
    "flextool/gui/result_viewer.py": 4662,
    "flextool/input_derivation/_validators.py": 3303,
    "flextool/plot_outputs/plan.py": 1869,
    "flextool/plot_outputs/plot_bars.py": 1543,
    "flextool/process_inputs/read_tabular_with_specification.py": 1502,
    "flextool/process_inputs/write_old_flextool_to_db.py": 4055,
    "flextool/process_outputs/_output_meta.py": 2004,
    "flextool/process_outputs/read_highs_solution.py": 2162,
    "flextool/process_outputs/read_parameters.py": 2018,
    "flextool/process_outputs/write_outputs.py": 1566,
    "flextool/project_package.py": 1647,
    "tools/assistant_eval.py": 3531,
    "tools/lp_snapshot.py": 3403,
}

_VALUE_CAST = re.compile(r"""col\(\s*["']value["']\s*\)\s*\.cast\(\s*(pl\.)?(Utf8|String)\b""")


def _sources(folder: str) -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): p.read_text(encoding="utf-8")
            for p in sorted((ROOT / folder).rglob("*.py"))}


def test_no_new_axis_detection_by_column_name():
    grown = {}
    for rel, text in _sources("flextool/engine").items():
        n = len(_AXIS_BY_NAME.findall(text))
        if n > AXIS_BY_NAME_BASELINE.get(rel, 0):
            grown[rel] = (AXIS_BY_NAME_BASELINE.get(rel, 0), n)
    assert not grown, (
        f"New column-name axis checks (baseline, now): {grown}.  A DB parameter's period / time axis must come from "
        "flextool/engine/input/_param_shapes.py (resolve_param_shape -> broadcast_to_period_time -> "
        "promote_param_to_dt); Spine names an unnamed Map index 'x', so a column-name check misreads a time series. "
        "If the frame is model-internal (columns the engine wrote itself), raise the baseline in "
        "tests/test_invariant_guards.py and say why in the commit message.")


def test_no_new_test_reads_a_template_database():
    new = sorted(rel for rel, text in _sources("tests").items()
                 if _TEMPLATE_DB.search(text) and rel not in TEMPLATE_DB_READERS)
    assert not new, (
        f"{new} name a template .sqlite.  Build the test database from JSON in tmp_path (json_to_db, the "
        "session-scoped schema_db_url) instead.  If the test must read a template, open it read-only or from a "
        "tmp_path copy and add the module to TEMPLATE_DB_READERS in tests/test_invariant_guards.py.")


def test_template_reader_list_has_no_stale_entries():
    sources = _sources("tests")
    stale = sorted(rel for rel in TEMPLATE_DB_READERS if not _TEMPLATE_DB.search(sources.get(rel, "")))
    assert not stale, f"Drop from TEMPLATE_DB_READERS (no longer name a template database): {stale}"


def test_emit_renders_value_columns_without_a_cast():
    hits = [f"{rel}:{text.count(chr(10), 0, m.start()) + 1}"
            for rel, text in _sources("flextool/engine/preprocess/emit").items()
            for m in _VALUE_CAST.finditer(text)]
    assert not hits, (
        f"value column cast to a string in the vectorized emit: {hits}.  Render it with _render_value_column "
        "(flextool/engine/preprocess/emit/_vectorize.py, the repr() loop): .cast(pl.Utf8) diverges from the "
        "per-roll writer on sci-notation padding and NaN, which breaks byte parity.")


def test_modules_do_not_grow_past_their_ratchet():
    grown = {}
    for folder in ("flextool", "tools"):
        for rel, text in _sources(folder).items():
            lines = text.count("\n")
            limit = LARGE_MODULES[rel] + GROWTH_SLACK if rel in LARGE_MODULES else MAX_MODULE_LINES
            if lines > limit:
                grown[rel] = (limit, lines)
    assert not grown, (
        f"Modules over their size limit (limit, lines): {grown}.  Put the new code in a new module with one clear "
        f"topic (a new module stays under {MAX_MODULE_LINES} lines) instead of growing a large one.  If the growth "
        "really belongs there, raise the entry in LARGE_MODULES (tests/test_invariant_guards.py) and say why in the "
        "commit message.")


def test_large_module_list_has_no_stale_entries():
    stale = sorted(rel for rel in LARGE_MODULES if not (ROOT / rel).exists())
    assert not stale, f"Drop from LARGE_MODULES (the module no longer exists): {stale}"


def test_no_example_json_sets_node_type_in_base():
    files = sorted((ROOT / "tests" / "fixtures").glob("*.json")) + sorted(
        (ROOT / "flextool" / "schemas" / "canonical_databases").glob("*.json"))
    hits = {}
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("parameter_values", []) if isinstance(data, dict) else []
        nodes = sorted(row[1] for row in rows if row[0] == "node" and row[2] == "node_type" and row[4] == "Base")
        if nodes:
            hits[path.relative_to(ROOT).as_posix()] = nodes
    assert not hits, (
        f"node_type set in the Base alternative: {hits}.  A scenario that ranks Base above the node's own "
        "alternative makes the node a commodity node.  Set node_type in the alternative that defines the node.")
