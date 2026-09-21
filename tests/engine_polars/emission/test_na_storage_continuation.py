"""Tier-7: ``non_anticipativity_storage_use`` composes with
continuation cohort names (design
``specs/step05_continuation_fanout_design.md`` §5 / round-2 review
finding 1).

Fixture: an in-memory storage-bearing variant of
``tests/fixtures/stoch_two_period.json`` — a storage node
``reservoir`` (fed by inflow, drained by a free ``hydro`` unit into
``city``) is added to the stochastic group via ``group__node``.  The
variant is derived in-memory so the committed hand-calc fixture stays
the single source of truth (mirrors the ``lh2_master_build_db_url``
pattern); no hand calculation is needed — this is a structural
constraint-domain test.

Post-fix, ``period_branch_full`` gains the continuation pairs and
``period_in_use`` gains ``p2040_low``, so ``db_pairs``
(``model.py`` NA dispatch: ``d != b``, ``b ∈ period_in_use``) =
{(p2035, p2035_low), (p2040, p2040_low)} and the constraint domain
joins ``dt_non_anticipativity`` on ``d``:

    1 stoch node x [(p2035, p2035_low) x 3 t + (p2040, p2040_low) x 3 t]
    = 6 LP rows.

DEFECT PIN: pre-fix the continuation cohort had no active time, so
``period_in_use`` lacked ``p2040_low``, ``dt_non_anticipativity`` had
no p2040 rows, and the constraint collapsed to the branching-period
rows only (3), with no (p2040, p2040_low, t) domain row at all.
"""
from __future__ import annotations

import base64
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from tests.engine_polars.emission._helpers import (  # noqa: E402
    assert_cstr_absent,
    assert_cstr_row_count,
    build,
)


def _val(v: "str | float") -> list:
    return [
        base64.b64encode(json.dumps(v).encode()).decode("ascii"),
        "float" if isinstance(v, float) else "str",
    ]


@pytest.fixture(scope="module")
def storage_variant_workdir(tmp_path_factory, test_solver_config_dir):
    """Build + cascade-run the storage-bearing variant's ``stoch``
    scenario; returns the fully-preprocessed workdir."""
    from db_utils import json_to_db

    from flextool.engine_polars._orchestration import run_chain_from_db
    from flextool.update_flextool.db_migration import migrate_database

    spec = json.loads(
        (TESTS_DIR / "fixtures" / "stoch_two_period.json").read_text()
    )
    spec["entities"] += [
        ["node", "reservoir", None],
        ["unit", "hydro", None],
        ["group__node", ["stoch_g", "reservoir"], None],
        ["unit__inputNode", ["hydro", "reservoir"], None],
        ["unit__outputNode", ["hydro", "city"], None],
    ]
    spec["entity_alternatives"] += [
        ["node", ["reservoir"], "init", True],
        ["unit", ["hydro"], "init", True],
    ]
    spec["parameter_values"] += [
        ["node", "reservoir", "node_type", _val("storage"), "base"],
        ["node", "reservoir", "existing", _val(5000.0), "base"],
        ["node", "reservoir", "inflow", _val(50.0), "base"],
        ["node", "reservoir", "storage_start_end_method",
         _val("fix_start"), "base"],
        ["node", "reservoir", "storage_state_start", _val(0.5), "base"],
        ["node", "reservoir", "storage_solve_horizon_method",
         _val("use_reference_value"), "base"],
        ["node", "reservoir", "storage_state_reference_value",
         _val(0.5), "base"],
        ["unit", "hydro", "existing", _val(100.0), "base"],
        ["unit", "hydro", "efficiency", _val(1.0), "base"],
    ]

    root = tmp_path_factory.mktemp("_root_na_storage_continuation")
    json_path = root / "stoch_two_period_storage.json"
    json_path.write_text(json.dumps(spec))
    db_path = root / "stoch_two_period_storage.sqlite"
    url = json_to_db(json_path, db_path)
    migrate_database(url)

    wf = root / "work_stoch"
    wf.mkdir()
    steps = run_chain_from_db(
        input_db_url=url,
        scenario_name="stoch",
        work_folder=wf,
        solver_config_dir=test_solver_config_dir,
        csv_dump=True,
        keep_solutions=True,
    )
    shutil.copy(db_path, wf / "tests.sqlite")
    if steps:
        last_step = next(reversed(list(steps.values())))
        provider = getattr(last_step, "flex_data_provider", None)
        if provider is not None:
            provider.snapshot_processed_inputs(wf)
    return wf


@pytest.mark.emission
def test_na_storage_use_composes_with_continuation_cohorts(
    storage_variant_workdir,
) -> None:
    pb, data = build(storage_variant_workdir)

    # Fixture invariants — the continuation structures are populated.
    assert data.dt_non_anticipativity is not None
    assert data.period_branch_full is not None
    assert data.groupStochastic is not None
    assert data.period_in_use_set is not None
    piu = set(str(v) for v in data.period_in_use_set["d"].to_list())
    assert {"p2035", "p2035_low", "p2040", "p2040_low"} <= piu

    # dtna == the 6 realized (d, t) rows.
    assert data.dt_non_anticipativity.height == 6

    # Domain: 1 stoch node x [(p2035, p2035_low) x 3 + (p2040,
    # p2040_low) x 3] = 6 rows (DEFECT PIN — pre-fix: 3, no p2040
    # continuation pair).
    assert_cstr_row_count(pb, "non_anticipativity_storage_use", 6)

    recs = pb.cstrs_named("non_anticipativity_storage_use")
    assert len(recs) == 1
    over = recs[0].over
    assert set(over.columns) >= {"n", "d", "b", "t"}
    rows = sorted(
        (str(r["n"]), str(r["d"]), str(r["b"]), str(r["t"]))
        for r in over.rows(named=True)
    )
    assert rows == sorted(
        [("reservoir", "p2035", "p2035_low", t)
         for t in ("t0001", "t0002", "t0003")]
        + [("reservoir", "p2040", "p2040_low", t)
           for t in ("t0004", "t0005", "t0006")]
    )

    # No online / reserve providers in this fixture.
    assert_cstr_absent(pb, "non_anticipativity_online_integer")
    assert_cstr_absent(pb, "non_anticipativity_online_linear")
    assert_cstr_absent(pb, "non_anticipativity_reserve")
