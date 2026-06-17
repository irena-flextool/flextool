"""End-to-end coverage for the TIER 1 Lagrangian invest→dispatch handoff.

The base ``lh2_three_region`` fixture has zero investment, so the path
``solve_lagrangian → sol.subproblem_col_values → _assemble_invest_vars →
LagrangianResult.invest_solution_vars`` and the orchestrator deposit
(``_orchestration.py`` ``_run_lagrangian_solve``) were untested against a
real investing decomposition.  This module drives the
``lh2_three_region_invest`` scenario (an additive alternative + scenario
on the lh2 fixture, authored in ``tests/fixtures/regen_lh2_three_region``)
which:

* makes ``wind_<r>`` invest-eligible in every region
  (``invest_no_limit`` + a low ``invest_cost``) and tightens the binding
  ``coal_<r>`` capacity, so each region has a genuine in-region capacity
  shortage and the LP strictly prefers investing wind over paying the
  ``penalty_up=8000`` unserved-energy slack — i.e. it ACTUALLY invests;
* defines a two-solve chain ``[lh2_invest, lh2_dispatch]`` — a single-solve
  Lagrangian invest solve (``decomposition=lagrangian`` +
  ``invest_periods=[y2030]``) followed by a monolithic dispatch solve that
  consumes the invested capacity.

Three tiers (per ``specs/lagrangian_solution_assembly.md`` "Tier1 test
fixture plan"):

* Tier-1a — ``solve_lagrangian`` populates a non-empty, disjoint, ``>0``
  ``invest_solution_vars`` spanning ≥2 regions.
* Tier-1b — the carrier (``SnapshotSolution`` →
  ``build_handoff_from_solution``) yields a non-empty handoff.
* Tier-1c — ``run_chain_from_db`` deposits the handoff on the Lagrangian
  step and the downstream dispatch step consumes the invested capacity.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
import pytest

from flextool.engine_polars import load_flextool
from flextool.engine_polars._lagrangian import (
    _INVEST_VAR_NAMES,
    _resolve_entity_owner,
    solve_lagrangian,
)
from flextool.engine_polars import _region_filter


FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
LH2_FIXTURE_JSON = FLEXTOOL_ROOT / "tests" / "fixtures" / "lh2_three_region.json"

INVEST_SCENARIO = "lh2_three_region_invest"
REGIONS = ["region_A", "region_B", "region_C"]
# The three invest-eligible region-owned units the fixture authors.
INVEST_UNITS = {"wind_A", "wind_B", "wind_C"}


def _build_invest_db(tmp_path: Path) -> str:
    """Materialise the committed lh2 JSON fixture into a fresh SQLite and
    return its url (build-from-JSON per CLAUDE.md invariant 3)."""
    if not LH2_FIXTURE_JSON.exists():
        pytest.skip(f"LH2 JSON fixture not present: {LH2_FIXTURE_JSON}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / "lh2_invest.sqlite"
    url = json_to_db(LH2_FIXTURE_JSON, db_path)
    migrate_database(url)
    return url


def _set_single_solve(db_url: str, solve: str) -> None:
    """Override ``model.flexTool.solves`` under the invest alternative to a
    single solve so the chain runs only *solve* — used to snapshot the
    Lagrangian invest solve's FlexData (with the invest sets) for the
    direct ``solve_lagrangian`` Tier-1a/1b tests."""
    import spinedb_api as api

    with api.DatabaseMapping(db_url) as db:
        arr = {
            "type": "array", "value_type": "str",
            "data": [solve], "index_name": "sequence_index",
        }
        db_value, db_type = api.to_database(arr)
        db.add_update_item(
            "parameter_value",
            entity_class_name="model",
            entity_byname=("flexTool",),
            parameter_definition_name="solves",
            alternative_name=INVEST_SCENARIO,
            value=db_value,
            type=db_type,
        )
        db.commit_session("test: single-solve invest override")


@pytest.fixture(scope="module")
def invest_solve_data(tmp_path_factory):
    """FlexData for the ``lh2_invest`` solve, loaded from a workdir where
    the chain ran only that solve (so the snapshot carries the invest
    sets).  Returns ``(data, work_folder)``.
    """
    from flextool.engine_polars._orchestration import run_chain_from_db

    tmp_path = tmp_path_factory.mktemp("lh2_invest_solve")
    url = _build_invest_db(tmp_path)
    _set_single_solve(url, "lh2_invest")

    parent = tmp_path_factory.mktemp("_root_invest_solve")
    wf = parent / f"work_{INVEST_SCENARIO}"
    wf.mkdir()
    steps = run_chain_from_db(
        input_db_url=url,
        scenario_name=INVEST_SCENARIO,
        work_folder=wf,
        csv_dump=True,
        keep_solutions=True,
    )
    # Copy the SQLite + snapshot the last (only) sub-solve so
    # ``load_flextool(wf)`` reconstructs the invest solve's FlexData.
    sqlite_src = urlparse(url).path
    shutil.copy(sqlite_src, wf / "tests.sqlite")
    last_step = next(reversed(list(steps.values())))
    provider = getattr(last_step, "flex_data_provider", None)
    if provider is not None:
        provider.snapshot_processed_inputs(wf)
    data = load_flextool(wf)
    return data, wf


# ---------------------------------------------------------------------------
# Tier-1a — solve_lagrangian populates invest_solution_vars from a real
# investing decomposition.
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_invest_sets_are_non_none(invest_solve_data) -> None:
    """The invest solve's FlexData has non-None pd/nd_invest_set (the
    fixture's invest_method + invest_cost + solve.invest_periods make the
    set non-empty)."""
    data, _wf = invest_solve_data
    assert data.pd_invest_set is not None, (
        "pd_invest_set is None — the invest scenario did not create the "
        "process invest set (check invest_method / invest_periods)."
    )
    assert data.pd_invest_set.height > 0
    # The three invest-eligible wind units appear in the process invest set.
    pcol = data.pd_invest_set.columns[0]
    in_set = set(data.pd_invest_set[pcol].to_list())
    assert INVEST_UNITS <= in_set, (
        f"expected {INVEST_UNITS} in pd_invest_set, got {sorted(in_set)}"
    )


@pytest.mark.solver
def test_solve_lagrangian_invest_solution_vars_populated(
    invest_solve_data,
) -> None:
    """A real investing Lagrangian solve populates
    ``invest_solution_vars`` — non-empty, keyed by a subset of the
    invest/divest var names, disjoint ``(entity, d)`` keys, ``>0`` values
    spanning ≥2 regions, no virtual-flow tokens.
    """
    data, wf = invest_solve_data
    result = solve_lagrangian(
        data, work_dir=wf, alpha=10.0, max_iters=100, tol=0.5,
        initial_lambda=0.0, min_iters=20,
    )

    iv = result.invest_solution_vars
    assert iv, "invest_solution_vars is empty — the model did not invest"
    assert set(iv) <= set(_INVEST_VAR_NAMES), (
        f"unexpected keys {set(iv) - set(_INVEST_VAR_NAMES)}"
    )
    # The process invest var must be present (wind units are processes).
    assert "v_invest_p" in iv, (
        f"v_invest_p missing from assembled invest vars (keys={list(iv)})"
    )

    frame = iv["v_invest_p"]
    assert set(frame.columns) == {"p", "d", "value"}, (
        f"v_invest_p frame columns {frame.columns} != (p, d, value); "
        "must match polar_high.Solution.value(name) shape so "
        "SnapshotSolution can serve it."
    )

    # Disjoint (entity, d) keys — no double-count across regions.
    keys = frame.select("p", "d")
    assert keys.height == keys.unique().height, (
        "v_invest_p has duplicate (p, d) keys — owner-selection failed "
        "to de-duplicate across regions."
    )

    # No virtual half-flow tokens leak into the assembled frame.
    ents = frame["p"].to_list()
    for e in ents:
        s = str(e)
        assert not s.startswith("hf_"), f"virtual hf_ token leaked: {e!r}"
        assert "__export__" not in s and "__import__" not in s, (
            f"virtual half-flow token leaked: {e!r}"
        )

    # At least one positive invested value (the LP actually invested).
    positives = frame.filter(pl.col("value") > 1.0)
    assert positives.height >= 1, (
        f"no positive invest value — model invested nothing:\n{frame}"
    )

    # The invested entities are region-owned and span ≥2 regions.  Map
    # each invested entity to its owning region via the same membership
    # resolution the assembly uses.
    membership = _region_filter.load_region_membership(data, REGIONS)
    owner = _resolve_entity_owner(membership, REGIONS)
    invested_entities = set(positives["p"].to_list())
    owning_regions = {owner.get(e) for e in invested_entities}
    owning_regions.discard(None)
    assert len(owning_regions) >= 2, (
        f"invested entities {sorted(invested_entities)} span only "
        f"{owning_regions} — need ≥2 regions to exercise owner-selection."
    )
    # Every invested entity has a resolvable owner (no orphan tokens).
    assert all(owner.get(e) is not None for e in invested_entities), (
        f"some invested entity has no owning region: "
        f"{[e for e in invested_entities if owner.get(e) is None]}"
    )


# ---------------------------------------------------------------------------
# Tier-1b — the SnapshotSolution carrier yields a non-empty handoff.
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_invest_vars_through_handoff_carrier(invest_solve_data) -> None:
    """Feeding ``invest_solution_vars`` through
    ``build_handoff_from_solution`` via ``SnapshotSolution`` (mirroring the
    orchestrator deposit, ``flex_data=None``) yields a handoff whose
    ``realized_invest`` is populated and ``is_empty()`` is False.
    """
    from flextool.engine_polars._orchestration import SnapshotSolution
    from flextool.engine_polars.input import build_handoff_from_solution

    data, wf = invest_solve_data
    result = solve_lagrangian(
        data, work_dir=wf, alpha=10.0, max_iters=100, tol=0.5,
        initial_lambda=0.0, min_iters=20,
    )
    assert result.invest_solution_vars, "no invest vars to feed the carrier"

    snap = SnapshotSolution(_vars=dict(result.invest_solution_vars))
    handoff = build_handoff_from_solution(
        snap,
        wf,
        "lh2_invest",
        prior_handoff=None,
        flex_data=None,
        parent_handoff=None,
        provider=None,
    )
    assert handoff is not None
    assert not handoff.is_empty(), (
        "handoff.is_empty() is True despite a real investing solve"
    )
    ri = handoff.realized_invest
    assert ri is not None, "handoff.realized_invest is None"
    assert ri.height > 0, "handoff.realized_invest is empty"
    # At least one positive invested capacity carried forward.
    val_col = ri.columns[-1]
    assert ri.filter(pl.col(val_col) > 1.0).height >= 1, (
        f"realized_invest carries no positive capacity:\n{ri}"
    )


# ---------------------------------------------------------------------------
# Tier-1c — end-to-end chain: orchestrator deposits the handoff and the
# downstream dispatch consumes the invested capacity.
# ---------------------------------------------------------------------------


@pytest.mark.solver
def test_invest_dispatch_chain_handoff_reaches_dispatch(tmp_path) -> None:
    """``run_chain_from_db`` on the invest scenario:

    1. completes both solves without error;
    2. the Lagrangian invest step carries a non-empty
       ``handoff.realized_invest`` (orchestrator deposit works);
    3. the downstream dispatch step completed (real Solution, optimal);
    4. the dispatch consumed the invested capacity — its FlexData carries
       non-null ``p_entity_invested`` for the invested wind units.
    """
    from flextool.engine_polars import run_chain_from_db
    from flextool.engine_polars._orchestration import SnapshotSolution

    url = _build_invest_db(tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    steps = run_chain_from_db(
        url, INVEST_SCENARIO, work_folder=work_folder, keep_solutions=True,
    )
    assert steps, "run_chain_from_db produced no orchestration steps"

    keys = list(steps.keys())
    invest_key = next((k for k in keys if "lh2_invest" in k), None)
    dispatch_key = next((k for k in keys if "lh2_dispatch" in k), None)
    assert invest_key is not None, f"no lh2_invest step in {keys}"
    assert dispatch_key is not None, f"no lh2_dispatch step in {keys}"

    # --- (2) the Lagrangian invest step deposited a non-empty handoff ---
    invest_step = steps[invest_key]
    assert isinstance(invest_step.solution, SnapshotSolution), (
        "invest step should carry a SnapshotSolution invest carrier, got "
        f"{type(invest_step.solution).__name__}"
    )
    assert invest_step.handoff is not None, (
        "invest step has no handoff — orchestrator deposit did not fire"
    )
    assert not invest_step.handoff.is_empty(), (
        "invest step handoff is empty despite a real investing solve"
    )
    ri = invest_step.handoff.realized_invest
    assert ri is not None and ri.height > 0, (
        "invest step handoff.realized_invest is empty/None"
    )
    val_col = ri.columns[-1]
    invested = ri.filter(pl.col(val_col) > 1.0)
    assert invested.height >= 1, (
        f"invest handoff carries no positive capacity:\n{ri}"
    )
    # The invested entities include the wind units, spanning ≥2 regions.
    ent_col = ri.columns[0]
    invested_ents = set(invested[ent_col].to_list())
    assert invested_ents & INVEST_UNITS, (
        f"expected invested wind units, got {sorted(invested_ents)}"
    )

    # --- (3) the dispatch step completed with a real Solution ----------
    dispatch_step = steps[dispatch_key]
    assert dispatch_step.solution is not None, "dispatch produced no solution"
    assert not isinstance(dispatch_step.solution, SnapshotSolution), (
        "dispatch should carry a real Solution, not a SnapshotSolution"
    )
    assert dispatch_step.optimal, "dispatch solve not optimal"

    # --- (4) the dispatch CONSUMED the invested capacity ---------------
    fd = dispatch_step.flex_data
    pei = getattr(fd, "p_entity_invested", None)
    assert pei is not None, (
        "dispatch FlexData has no p_entity_invested — the handoff overlay "
        "did not reach the dispatch solve."
    )
    pei_frame = pei.frame if hasattr(pei, "frame") else pei
    # Restrict to the invested wind units and assert positive capacity.
    pei_val_col = pei_frame.columns[-1]
    pei_ent_col = pei_frame.columns[0]
    dispatch_invested = pei_frame.filter(
        (pl.col(pei_ent_col).is_in(list(INVEST_UNITS)))
        & (pl.col(pei_val_col) > 1.0)
    )
    assert dispatch_invested.height >= 1, (
        "dispatch p_entity_invested has no positive capacity for the "
        f"invested wind units:\n{pei_frame}"
    )
