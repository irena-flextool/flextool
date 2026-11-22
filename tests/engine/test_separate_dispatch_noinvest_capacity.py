"""OC1 capacity fix — the realized-dispatch sibling must resolve the
OPERATIONAL ``existing`` capacity, even with NO investment.

A ``dispatch_realization_method=separate_dispatch`` solve auto-spawns a
deterministic realized-dispatch SIBLING that consumes the parent's committed
capacity.  Its capacity is the existing cohort (the user's ``existing``) plus the investments carried in the
handoff (``realized_cohort``); for a NO-INVEST model nothing is carried, and the sibling must still see the
existing capacity — before the cohort handoff it resolved to ZERO → ``p_process_existing_count = 0`` → the unit
could not dispatch → all demand went unserved (VOLL) / a commitment pin was infeasible.

``_chain_params.emit_existing_capacity`` gives every solve ``existing + carried − retired``, so the carried
investments add to the existing capacity and never replace it.

Fixture (``separate_dispatch_noinvest.json``, built from JSON per CLAUDE.md
invariant 3): a fan-at-first stochastic separate_dispatch solve on ``p2035``
(branches ``rlz`` w=0 realized / ``high`` w=0.5 / ``low`` w=0.5) over 3 steps.
A committable ``peaker`` (existing 2 units of virtual_unitsize 50 MW = 100 MW,
min_load 0.5, startup_cost 1, other_operational_cost 10 CUR/MWh, NO invest)
backs ``wind`` (existing 100 MW, upper_limit, CF rlz 0.5 / high 0.8 / low 0.0)
against 100 MW demand.

Hand-calc (sibling ``stoch_rd__realized_dispatch``, realized ``rlz`` branch,
H = 8760/3 = 2920 h/step):
  * The sibling resolves peaker's capacity ``p_entity_all_existing`` =
    2 x 50 = 100 MW and ``p_process_existing_count`` = 100 / 50 = 2 units
    (NOT 0 — the fix).
  * Realized ``rlz`` wind CF 0.5 -> 50 MW; demand 100 -> peaker 50 MW.  The
    peaker is linear, so the sibling decides its units online again (the
    parent's hedge 2.0 is not pinned): 50 MW needs one unit, the min_load floor
    0.5*v*50 <= 50 allows two, and the first step is a free start, so any v in
    [1, 2] is optimal (a tie) — v = 2 needs the 2 units the fix resolves.
    Output is exactly 50 MW with ZERO unserved load (no VOLL).
  * Objective = 50 MW * 10 CUR/MWh * 2920 h/step * 3 steps = 4 380 000
    (no start-up: a free first step, no later start-up needed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "separate_dispatch_noinvest.json"

SCENARIO = "separate_dispatch_noinvest"
PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"

PEAKER_EXISTING_UNITS = 2.0       # the operational existing count the fix must resolve
PEAKER_CAPACITY_MW = 100.0        # = PEAKER_EXISTING_UNITS x virtual_unitsize 50
SIBLING_OBJECTIVE = 4_380_000.0   # realized rlz dispatch cost (hand-calc)


def _build_db(fixture_json: Path, tmp_path: Path) -> str:
    if not fixture_json.exists():
        pytest.skip(f"fixture JSON not present: {fixture_json}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{fixture_json.stem}.sqlite"
    url = json_to_db(fixture_json, db_path)
    migrate_database(url)
    return url


def _run_chain(tmp_path: Path):
    from flextool.engine import run_chain_from_db

    url = _build_db(FIXTURE, tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()
    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            url, SCENARIO, work_folder=work_folder,
            warm=False, keep_solutions=True,
        )
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    assert steps, "run_chain_from_db produced no orchestration steps"
    return steps, work_folder


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return _run_chain(tmp_path_factory.mktemp("separate_dispatch_noinvest"))


@pytest.mark.solver
def test_sibling_resolves_operational_existing_capacity(chain):
    """THE FIX: the sibling resolves peaker's operational ``existing``
    (2 units, 100 MW), NOT 0 — the existing cohort with nothing carried."""
    steps, _ = chain
    fd = steps[SIBLING].flex_data
    assert fd.p_process_existing_count is not None, (
        "sibling built no p_process_existing_count"
    )
    count = (fd.p_process_existing_count.frame
             .filter(pl.col("p") == "peaker")["value"].max())
    assert count == pytest.approx(PEAKER_EXISTING_UNITS, abs=1e-9), (
        f"sibling peaker existing_count = {count}, expected "
        f"{PEAKER_EXISTING_UNITS} (0.0 would be the pre-fix capacity bug)"
    )
    cap = (fd.p_entity_all_existing.frame
           .filter(pl.col("e") == "peaker")["value"].max())
    assert cap == pytest.approx(PEAKER_CAPACITY_MW, abs=1e-9)


@pytest.mark.solver
def test_sibling_dispatches_with_zero_unserved(chain):
    """The sibling solves OPTIMAL and serves all demand — ZERO up-slack
    (no VOLL).  Pre-fix (capacity 0) the unit could not generate and demand
    was unserved."""
    steps, _ = chain
    st = steps[SIBLING]
    assert st.optimal is True
    sol = st.solution
    if "vq_state_up" in sol._vars:
        up = sol.value("vq_state_up").filter(pl.col("value") > 1e-6)
        assert up.height == 0, (
            f"sibling has {up.height} unserved-load (vq_state_up) rows — "
            f"demand not fully served"
        )


@pytest.mark.solver
def test_sibling_commitment_and_objective(chain):
    """The sibling decides the linear peaker's units online again (not pinned:
    no p_fix_online), within [1, 2] units, and dispatches the realized rlz
    scenario at the hand-calc cost."""
    steps, wf = chain
    assert steps[SIBLING].flex_data.p_fix_online is None
    sol = steps[SIBLING].solution
    vo = sol.value("v_online_linear").filter(pl.col("p") == "peaker")
    assert vo["value"].min() >= 1.0 - 1e-9
    assert vo["value"].max() <= PEAKER_EXISTING_UNITS + 1e-9
    obj = float(
        pl.read_parquet(wf / "output_raw" / f"v_obj__{SIBLING}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(SIBLING_OBJECTIVE, rel=1e-6)
