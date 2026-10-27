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

Databases are built from JSON (CLAUDE.md invariant #3): ``tests/fixtures/tests.json``, via
``tests/db_utils.json_to_db`` + the standard migration, in a module-scoped ``tmp_path_factory`` dir.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine import run_chain_from_db  # noqa: E402
from flextool.process_outputs.write_outputs import _group_entity_invest, write_outputs  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

TESTS_JSON = TEST_DIR / "fixtures" / "tests.json"
OUTPUT_CONFIG = TEST_DIR.parent / "templates" / "default_plots.yaml"


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
