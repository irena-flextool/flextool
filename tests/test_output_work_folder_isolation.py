"""Item 163 — output writers read only in-memory data (design_163.md).

Two fresh-run defects, both caused by an output writer reading the work folder from disk instead of
the sub-solve's :class:`FlexDataProvider` snapshot:

* ``v_dual_invest_{unit,connection,node}`` entity-class membership
  (:func:`flextool.process_outputs.read_highs_solution._entity_class_members`, formerly
  ``_load_entity_class``, which read ``input/{process_unit,process_connection,node}.csv`` — files a
  normal run never writes).  Fixed in commit 163-1 (T1, T2).  The ``--csv-dump`` round-trip (T5)
  pins that turning the debug dump on/off does not change this output.
* ``group_entity_invest`` (:func:`flextool.process_outputs.write_outputs._group_entity_invest`,
  which replaces a disk read of ``input/group_entity_invest.csv`` — a file only the retired GMPL
  model ever wrote).  Fixed in commit 163-2 (T3).

A third item-163 defect, unrelated to the two above: a REUSED work folder (the CLI-in-cwd rerun
case) used to corrupt outputs — ``_bootstrap_dirs`` didn't clear a prior run's ``output_raw/`` /
``solve_data/`` leftovers (design_163.md F6). Fixed in commit 163-5 (T4, T9).

Databases are built from JSON (CLAUDE.md invariant #3): ``tests/fixtures/tests.json``, via
``tests/db_utils.json_to_db`` + the standard migration, in a module-scoped ``tmp_path_factory`` dir.
``co2_period_rolling`` (T4a's dirtier) is not a plain ``tests.json`` scenario — it's one of
``tools/lp_snapshot.py``'s patched fixtures (an override alternative appended to a base scenario),
so T4a builds its own DB via that module's ``_build_fixture_db`` / ``_fixture_by_name`` (the same
helpers ``tools/output_snapshot.py`` reuses).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

TOOLS_DIR = TEST_DIR.parent / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine import run_chain_from_db  # noqa: E402
from flextool.process_outputs.write_outputs import _group_entity_invest, write_outputs  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402
from output_snapshot import _build_fixture_db, _fixture_by_name  # noqa: E402

TESTS_JSON = TEST_DIR / "fixtures" / "tests.json"
OUTPUT_CONFIG = TEST_DIR.parent / "flextool" / "schemas" / "default_plots.yaml"


@pytest.fixture(scope="module")
def tests_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    db_path = tmp_path_factory.mktemp("c24_163_db") / "tests.sqlite"
    url = json_to_db(TESTS_JSON, db_path)
    migrate_database(url)
    return url


def _run_cli_like(db_url: str, scenario: str, work: Path, *, csv_dump: bool = False):
    """Run one fixture the way the CLI does (``cmd_run_flextool.py`` l.966-985): the cascade, then
    ``write_outputs`` with the same call shape. Returns ``(last_step, work)``."""
    steps = run_chain_from_db(
        input_db_url=db_url, scenario_name=scenario, work_folder=work,
        warm=False, csv_dump=csv_dump,
    )
    assert steps, f"{scenario}: run_chain_from_db produced no orchestration steps"
    last = next(reversed(list(steps.values())))
    if csv_dump:
        assert last.flex_data_provider is not None
        last.flex_data_provider.snapshot_processed_inputs(work)
    write_outputs(
        scenario,
        output_location=str(work / "out"),
        subdir=scenario,
        output_config_path=str(OUTPUT_CONFIG),
        write_methods=["parquet"],
        raw_output_dir=str(work / "output_raw"),
        flex_data=last.flex_data,
        solution=last.solution,
        solve_name=last.solve_name,
        flex_data_provider=last.flex_data_provider,
    )
    return last


# ---------------------------------------------------------------------------
# T1 — v_dual_invest_* entity-class split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario, expected",
    [
        ("multi_year", {"unit": {"coal_plant"}, "connection": set(), "node": set()}),
        ("wind_battery_invest", {"unit": set(), "connection": {"battery_inverter"}, "node": {"battery"}}),
    ],
)
def test_fresh_run_v_dual_invest_has_entity_columns(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory, scenario: str, expected: dict,
) -> None:
    """A fresh run's ``v_dual_invest_{unit,connection,node}`` column sets are disjoint, their union
    is the ``v_invest`` column set, and the per-class membership matches the fixture JSON — item 163
    F2/T1 (before the fix every file had 0 columns)."""
    from flextool.lean_parquet import read_lean_parquet

    work = tmp_path_factory.mktemp(f"c24_163_t1_{scenario}")
    _run_cli_like(tests_db_url, scenario, work)

    raw = work / "output_raw"
    invest_parts = sorted(raw.glob("v_invest__*.parquet"))
    assert invest_parts, f"{scenario}: no v_invest parquet was written"

    for part in invest_parts:
        solve = part.name[len("v_invest__"):-len(".parquet")]
        invest_cols = set(read_lean_parquet(part).columns)
        if not invest_cols:
            continue
        by_class = {}
        for cls in ("unit", "connection", "node"):
            p = raw / f"v_dual_invest_{cls}__{solve}.parquet"
            by_class[cls] = set(read_lean_parquet(p).columns) if p.is_file() else set()
        pairwise = [(a, b) for i, a in enumerate(by_class) for b in list(by_class)[i + 1:]]
        for a, b in pairwise:
            assert not (by_class[a] & by_class[b]), (
                f"{scenario}/{solve}: {a} and {b} entity-class columns overlap: "
                f"{by_class[a] & by_class[b]}"
            )
        union = set().union(*by_class.values())
        assert union == invest_cols, (
            f"{scenario}/{solve}: v_dual_invest union {union} != v_invest columns {invest_cols}"
        )
        for cls, expected_members in expected.items():
            assert by_class[cls] == expected_members, (
                f"{scenario}/{solve}: v_dual_invest_{cls} columns {by_class[cls]} != "
                f"expected {expected_members}"
            )


def test_invest_marginal_includes_not_built_reduced_cost(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``multi_year`` coal_plant is not built (``v_invest`` = 0) in p2020: regime (a) of
    ``_synthesize_invest_dual`` (the not-built entity's own ``v_invest`` reduced cost) must now be
    populated and negative (unprofitable to build) — before the fix this cell was ``-0.0`` because
    the entity-class split was always empty. Bounded, not a copy of the probed value (independent-
    oracle rule): a genuinely unprofitable-to-build entity's reduced cost must exceed 1 CUR/kW in
    magnitude on this fixture's cost scale (probe: -574.47)."""
    from flextool.lean_parquet import read_lean_parquet

    work = tmp_path_factory.mktemp("c24_163_t2")
    _run_cli_like(tests_db_url, "multi_year", work)
    path = work / "out" / "output_parquet" / "multi_year" / "dual_invest_effective_unit_d_e.parquet"
    assert path.is_file(), "dual_invest_effective_unit_d_e.parquet was not written"
    df = read_lean_parquet(path)
    # Row index = period; columns = MultiIndex (scenario, entity).
    value = df.loc["p2020", ("multi_year", "coal_plant")]
    assert value < -1.0, f"expected a clearly negative not-built reduced cost, got {value}"


# ---------------------------------------------------------------------------
# T5 — --csv-dump does not change this output
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["5weeks_invest_fullYear_dispatch_coal_wind", "multi_year"])
def test_csv_dump_does_not_change_outputs(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory, scenario: str,
) -> None:
    """``--csv-dump`` must not change the output set (item 163; before the fix, the OLD
    ``_load_entity_class`` read ``input/*.csv`` — present only under ``--csv-dump`` — so
    ``v_dual_invest_*`` / ``dual_invest_effective_*`` differed between a plain run and a dump run)."""
    from tests._support.output_compare import diff_output_trees

    plain = tmp_path_factory.mktemp(f"c24_163_t5_plain_{scenario}")
    dumped = tmp_path_factory.mktemp(f"c24_163_t5_dump_{scenario}")
    _run_cli_like(tests_db_url, scenario, plain, csv_dump=False)
    _run_cli_like(tests_db_url, scenario, dumped, csv_dump=True)

    diffs = diff_output_trees(plain, dumped, scenario)
    assert diffs == [], f"{scenario}: csv_dump changed the output set: {diffs}"


# ---------------------------------------------------------------------------
# T3 — group_entity_invest from the provider
# ---------------------------------------------------------------------------


def test_group_entity_invest_comes_from_the_provider(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``multi_year_one_solve``'s ``group_entity_invest`` — the GMPL definition ``{(g,e) in
    group_entity : g in group_invest}`` — must equal the JSON: group ``wind_coal`` invests in
    ``coal_plant`` and ``wind_plant`` (``group__unit`` rows with ``invest_method`` ``invest_total``
    on ``wind_coal``). Before the fix this was always empty (no polars producer writes
    ``input/group_entity_invest.csv``)."""
    work = tmp_path_factory.mktemp("c24_163_t3")
    last = _run_cli_like(tests_db_url, "multi_year_one_solve", work)

    gei = _group_entity_invest(last.flex_data_provider)
    pairs = set(zip(gei["group"], gei["entity"]))
    assert pairs == {("wind_coal", "coal_plant"), ("wind_coal", "wind_plant")}


# ---------------------------------------------------------------------------
# T4 — a reused work folder gives fresh outputs (commit 163-5)
#
# critique_163.md §4: the parent's (pre-163-5) two pre-fix failure modes differ in KIND, not just
# in which files differ — one produces wrong-but-not-crashing diffs, the other crashes before any
# diff can even be computed — so they are two separate, explicitly named tests rather than one
# parametrized test whose pre-fix behaviour differs by sub-case.
# ---------------------------------------------------------------------------


def _dirty_then_run(
    dirtier_db_url: str, dirtier_scenario: str,
    target_db_url: str, target_scenario: str, work: Path,
):
    """Run *dirtier* (``csv_dump=True`` — the realistic leftover, design_163.md F6) in *work*
    first, then *target* (plain) in the SAME *work*. Returns the target's last step.

    The dirtier's own run may legitimately fail for reasons that have nothing to do with item 163
    (not every scenario in this table is designed to be solved standalone outside its own chain);
    only ITS ON-DISK LEFTOVERS matter to this test, so its exception is swallowed. The *target*
    call is NOT swallowed: on the parent commit it is exactly where F6's defects surface (wrong
    diffs, or — for the ``multi_year`` case — ``write_outputs`` itself crashing.
    """
    try:
        _run_cli_like(dirtier_db_url, dirtier_scenario, work, csv_dump=True)
    except Exception:
        pass
    return _run_cli_like(target_db_url, target_scenario, work)


def test_reused_work_folder_gives_fresh_outputs_wrong_diffs(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``base``, reused after ``co2_period_rolling`` (16 rolls) left stale per-roll parquets: on
    the parent commit this does NOT crash, but ``union_realized_slice.has_persisted_slices`` unions
    the stale per-roll parquets into the fresh run, producing 94 wrong-but-not-crashing final-output
    diffs (design_163.md F6). After the run-start clearing (163-5), the reused folder's output must
    equal a fresh folder's."""
    from tests._support.output_compare import diff_output_trees

    db_dir = tmp_path_factory.mktemp("c24_163_t4a_db")
    dirtier_fx = _fixture_by_name("co2_period_rolling")
    dirtier_url, _ = _build_fixture_db(dirtier_fx, db_dir, "dirtier")

    reused = tmp_path_factory.mktemp("c24_163_t4a_reused")
    fresh = tmp_path_factory.mktemp("c24_163_t4a_fresh")
    _dirty_then_run(dirtier_url, dirtier_fx.scenario, tests_db_url, "base", reused)
    _run_cli_like(tests_db_url, "base", fresh)

    diffs = diff_output_trees(reused, fresh, "base")
    assert diffs == [], f"reused work folder differs from fresh: {diffs}"


@pytest.mark.nightly_heavy  # PR CI skips it; runs nightly (pyproject markers)
def test_reused_work_folder_gives_fresh_outputs_crash(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``multi_year``, reused after ``multi_fullYear_battery_nested_24h_invest_one_solve`` (a
    battery/nested-storage scenario, a plain ``tests.json`` scenario) left its leftovers behind: on
    the parent commit ``write_outputs`` does not merely produce wrong diffs — it CRASHES
    (``KeyError: ['battery_inverter', 'battery'] not in index`` in ``calc_costs.compute_costs``)
    before any diff can even be computed (design_163.md F6). After the run-start clearing (163-5),
    the reused folder's output must equal a fresh folder's — no crash, no diffs."""
    from tests._support.output_compare import diff_output_trees

    reused = tmp_path_factory.mktemp("c24_163_t4b_reused")
    fresh = tmp_path_factory.mktemp("c24_163_t4b_fresh")
    _dirty_then_run(
        tests_db_url, "multi_fullYear_battery_nested_24h_invest_one_solve",
        tests_db_url, "multi_year", reused,
    )
    _run_cli_like(tests_db_url, "multi_year", fresh)

    diffs = diff_output_trees(reused, fresh, "multi_year")
    assert diffs == [], f"reused work folder differs from fresh: {diffs}"


# ---------------------------------------------------------------------------
# T9 — run-start clearing (commit 163-5)
# ---------------------------------------------------------------------------


def test_run_start_clears_output_raw(
    tests_db_url: str, tmp_path_factory: pytest.TempPathFactory, caplog,
) -> None:
    """``_bootstrap_dirs`` must clear a reused work folder's own stash before each run
    (design_163.md §4): a ghost per-roll parquet and manifest line from an earlier run must not
    survive into THIS run's ``output_raw/``, and a ghost ``solve_data/period_capacity.csv`` row
    must not survive either. The clearing is logged at INFO."""
    import logging

    work = tmp_path_factory.mktemp("c24_163_t9")
    output_raw = work / "output_raw"
    output_raw.mkdir(parents=True)
    (output_raw / "v_flow__ghost.parquet").write_bytes(b"not a real parquet")
    (output_raw / "_solve_order.txt").write_text("ghost\n")
    solve_data = work / "solve_data"
    solve_data.mkdir(parents=True)
    (solve_data / "period_capacity.csv").write_text("period\nX\n")

    with caplog.at_level(logging.INFO):
        steps = run_chain_from_db(
            input_db_url=tests_db_url, scenario_name="base", work_folder=work,
            warm=False,
        )
    assert steps, "base: run_chain_from_db produced no orchestration steps"

    assert not (output_raw / "v_flow__ghost.parquet").exists(), (
        "a ghost per-roll parquet from an earlier run survived into this run's output_raw/"
    )
    manifest = (output_raw / "_solve_order.txt").read_text()
    assert "ghost" not in manifest, f"the ghost manifest line survived: {manifest!r}"
    pc = (solve_data / "period_capacity.csv").read_text()
    assert pc.strip() == "period", f"period_capacity.csv should be header-only, got {pc!r}"
    assert any(
        "Removed" in r.message and "output_raw" in r.message for r in caplog.records
    ), "expected an INFO log line about clearing a reused work folder's leftovers"
