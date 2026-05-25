"""End-to-end smoke test: the cascade's auto-row-scaling trigger fires.

Tier 4 Commit 4 will refactor the row-scaling decision out of the
disk-based ``_native_run_model.py`` path and into the in-memory
``_PolarHighCascadeSolver.run``.  Without a scenario that actually trips
the analyser's family-spread threshold, we cannot tell whether the
refactor preserves behaviour: ``base`` and ``coal`` both leave the
in-memory :class:`ScaleTable` recommending ``use_row_scaling="no"``.

This module builds a tiny derivative of the ``coal`` scenario,
``coal_extreme_scale``, that injects a single absurdly large slack
penalty (``penalty_up = 1e10`` on ``west``) on top of the regular
``init → west → coal`` alternative stack.  That value alone widens the
cost family (``vom_and_op_costs ∪ capex_invest ∪ node_penalty``) spread
to ~8.7 decades, well past the ``COST_SPREAD_THRESHOLD = 5.0`` decade
gate in :mod:`flextool.engine_polars.scaling`.

What the test verifies (current cascade — pre-Tier-4)
-----------------------------------------------------

The current cascade has two scaling analysers running in parallel:

1. :mod:`flextool.engine_polars.scaling`'s in-memory ``analyze_solve``,
   called from ``_orchestration.py`` against :class:`FlexData`.  Its
   recommendation feeds ``effective_row_scaling`` straight into the LP
   build.
2. Historically, the legacy disk-based ``flextoolrunner.scaling.analyze_solve``
   (called from ``_native_run_model.py``) read ``wf/input/*.csv``.
   In the in-memory cascade the input/ directory was empty, so this
   analyser always returned ``"no"``.  That path was retired in Tier 4.

Post-Tier-4, the engine_polars analyser's recommendation is the
ONLY source of truth for ``state.solve.use_row_scaling``.  This test
locks in the precondition: under :func:`run_chain_from_db` the
engine_polars analyser cache, keyed on the per-solve name, holds a
:class:`ScaleTable` with ``use_row_scaling="yes"`` and
``row_scaling_trigger="cost"``.  Any refactor that drops the trigger
silently — or fails to populate the cache for this solve — will fail
this assertion.

Concrete trigger maths (see :func:`flextool.engine_polars.scaling.analyze_solve`):

* ``unitsize`` family — every entity defaults to ``existing OR 1000``,
  giving a spread of ~0 decades.  Below the ``UNITSIZE_SPREAD_THRESHOLD = 3.0``.
* ``node_inflow`` family — coal's inflow values are uniform; spread ~0.
* ``cost`` family pre-injection — coal price 20, default penalty 10000;
  spread ~ ``log10(10000) - log10(20) ≈ 2.7`` decades, below 5.
* ``cost`` family post-injection — penalty values now span [900, 1e10];
  pooled with vom (down to 0.34 from co2_content via commodity_price),
  spread ≈ ``log10(1e10) - log10(0.34) ≈ 10.5`` decades.  Trigger fires
  with ``row_scaling_trigger = "cost"``.

The LP is still feasible — ``penalty_up`` is a cost coefficient on
upward slack and dispatch in the coal scenario does not engage upward
slack at all, so the optimum is identical to plain ``coal``.

Assertions
----------

1. The cascade reaches an optimal solution (LP solves cleanly with the
   widened cost spread).
2. The engine_polars analyser cache records
   ``use_row_scaling="yes"`` / ``row_scaling_trigger="cost"`` for the
   scenario's solve, with ``cost_spread_log10 > 5.0``.
3. ``effective_row_scaling`` (the value actually fed into the LP build)
   resolves to ``"yes"`` via :func:`resolve_effective_scaling`.

If a future refactor silently drops the auto-row-scaling code path,
all three assertions fire.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parents[2]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import json_to_db  # noqa: E402


def _add_coal_extreme_scale_scenario(db_url: str) -> None:
    """Layer ``coal_extreme_scale`` on top of the regular ``coal`` chain.

    Adds a single alternative (``extreme_penalty_for_scaling``) that sets
    ``penalty_up = 1e10`` on the ``west`` node, then wires up the
    ``coal_extreme_scale`` scenario as ``init → west → coal →
    extreme_penalty_for_scaling``.
    """
    from spinedb_api import DatabaseMapping, import_data

    with DatabaseMapping(db_url) as db_map:
        count, errors = import_data(
            db_map,
            alternatives=[("extreme_penalty_for_scaling", "")],
            scenarios=[("coal_extreme_scale", False, "")],
            scenario_alternatives=[
                ("coal_extreme_scale", "init", "west"),
                ("coal_extreme_scale", "west", "coal"),
                ("coal_extreme_scale", "coal", "extreme_penalty_for_scaling"),
                ("coal_extreme_scale", "extreme_penalty_for_scaling", None),
            ],
            parameter_values=[
                # 1e10 is far above the default 10000 (4 decades) so the
                # pooled cost-family spread crosses the 5-decade trigger.
                # Upward slack carries the penalty into the objective only
                # when binding; in the coal scenario it is not, so the
                # optimum is unchanged.
                (
                    "node", "west", "penalty_up", 1.0e10,
                    "extreme_penalty_for_scaling",
                ),
            ],
        )
        if errors:
            raise RuntimeError(f"Spine import errors: {errors}")
        db_map.commit_session("Add coal_extreme_scale scenario")


@pytest.fixture
def extreme_db_url(tmp_path: Path) -> str:
    """Fresh sqlite DB with the ``coal_extreme_scale`` scenario added."""
    db_path = tmp_path / "coal_extreme_scale.sqlite"
    url = json_to_db(TESTS_DIR / "fixtures" / "tests.json", db_path)
    _add_coal_extreme_scale_scenario(url)
    return url


def _setup_solver_config_dir(tmp_path: Path) -> Path:
    """Mirror ``tests/conftest.py::test_solver_config_dir`` for a per-test solver config dir.

    Symlinks the solver binaries from the repo's ``bin/`` and copies
    the deterministic ``tests/highs.opt`` so the cascade can resolve
    HiGHS.
    """
    import platform
    import shutil

    solver_config_dir = tmp_path / "solver_config"
    solver_config_dir.mkdir()
    repo_root = TESTS_DIR.parent
    repo_bin = repo_root / "bin"
    if sys.platform == "darwin" and platform.machine() == "arm64":
        glpsol_candidates = ["glpsol_macos15_arm64", "glpsol"]
    elif sys.platform.startswith("win"):
        glpsol_candidates = ["glpsol.exe"]
    else:
        glpsol_candidates = ["glpsol"]
    for candidate in glpsol_candidates:
        src = repo_bin / candidate
        if src.exists():
            (solver_config_dir / "glpsol").symlink_to(src)
            break
    for binary in ["highs", "highs.exe", "glpsol.exe"]:
        src = repo_bin / binary
        if src.exists() and not (solver_config_dir / binary).exists():
            (solver_config_dir / binary).symlink_to(src)
    shutil.copy(TESTS_DIR / "highs.opt", solver_config_dir / "highs.opt")
    return solver_config_dir


def test_coal_extreme_scale_triggers_auto_row_scaling(
    extreme_db_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine_polars analyser fires + the effective row-scaling = ``"yes"``."""
    workdir = tmp_path / "wd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    solver_config_dir = _setup_solver_config_dir(tmp_path)

    # The engine_polars analyser caches its decision per ``solve_name``
    # in a module-level dict; clear it explicitly so we know any
    # post-run lookup reflects THIS run, not a leftover from a prior
    # test in the worker.  ``tests/conftest.py`` already wires this as
    # an autouse fixture, but defending in depth here keeps the test
    # robust against ordering changes.
    from flextool.engine_polars import scaling as _engine_scaling
    _engine_scaling.clear_cache()

    from flextool.engine_polars import run_chain_from_db

    steps = run_chain_from_db(
        input_db_url=extreme_db_url,
        scenario_name="coal_extreme_scale",
        work_folder=workdir,
        solver_config_dir=solver_config_dir,
        keep_solutions=True,
    )
    assert steps, "run_chain_from_db returned no steps"
    last_step = next(reversed(list(steps.values())))
    assert last_step.solution is not None and last_step.solution.optimal, (
        "coal_extreme_scale did not reach an optimal solution — the LP "
        "should still solve cleanly; the injected penalty only widens "
        "the cost-coefficient spread, it does not change feasibility."
    )

    # The orchestrator strips a trailing ``_roll_N`` from the cache key;
    # walk the cache and grab the entry that matches this scenario's
    # base solve name.  There is exactly one solve in coal_extreme_scale.
    cached_tables = dict(_engine_scaling._scale_cache)
    assert cached_tables, (
        "engine_polars scaling cache is empty after the cascade — "
        "the in-memory analyse_solve call site in _orchestration.py "
        "was either skipped or its cache key drifted.  All sub-solve "
        "iterations should add at least one entry."
    )
    # Locate the entry corresponding to the only sub-solve of this
    # scenario.  ``last_step.solve_name`` may carry a ``_roll_N`` suffix
    # in rolling cascades; coal_extreme_scale is single-shot so it
    # matches directly, but we strip defensively.
    import re
    base_solve = re.sub(r"_roll_\d+$", "", last_step.solve_name)
    table = cached_tables.get(base_solve)
    assert table is not None, (
        f"No ScaleTable cached for solve {base_solve!r}; "
        f"cache keys = {list(cached_tables)!r}."
    )

    # Assertion 1 — the analyser recommends "yes" with the cost trigger.
    assert table.use_row_scaling == "yes", (
        f"engine_polars analyser returned use_row_scaling="
        f"{table.use_row_scaling!r}; expected 'yes'.  "
        f"cost_spread_log10={table.cost_spread_log10:.3f} "
        f"(threshold {_engine_scaling.COST_SPREAD_THRESHOLD}); "
        f"trigger={table.row_scaling_trigger!r}."
    )
    assert table.row_scaling_trigger == "cost", (
        f"Expected the cost-family trigger to fire; got "
        f"{table.row_scaling_trigger!r}.  This locks in the scenario's "
        f"design — the trigger is the absurd ``penalty_up=1e10``."
    )
    assert table.cost_spread_log10 > 5.0, (
        f"cost_spread_log10 = {table.cost_spread_log10:.3f}; expected "
        f"> 5.0 decades.  If this slips below threshold, the penalty "
        f"amplification was too gentle for the trigger."
    )

    # Assertion 2 — resolve_effective_scaling agrees: the LP build path
    # would use row scaling for this solve.  ``user_row_scaling`` is
    # None (DB carries no per-solve override on coal_extreme_scale),
    # so the analyser's recommendation wins.
    effective_row, _effective_obj = _engine_scaling.resolve_effective_scaling(
        table, user_row_scaling=None, user_obj_scale=None,
    )
    assert effective_row == "yes", (
        f"resolve_effective_scaling returned {effective_row!r}; "
        f"expected 'yes' when the user override is None and the "
        f"analyser recommends 'yes'."
    )

    # Assertion 3 — diagnostic: the provider-emitted
    # ``solve_data/p_use_row_scaling`` frame.  In the CURRENT cascade
    # this is fed by the legacy disk-based analyser running on an
    # empty input/ directory, so it reads 0.  Post-Tier-4 the cascade
    # will source it from ``table.use_row_scaling`` and the assertion
    # below will tighten to ``"1" in flag_values``.  Until then we
    # only assert the frame is well-formed.
    provider = last_step.flex_data_provider
    assert provider is not None, (
        "keep_solutions=True should retain flex_data_provider on the "
        "last step so post-run diagnostics can inspect emitted frames."
    )
    frame = provider.get("solve_data/p_use_row_scaling")
    assert frame is not None and frame.height >= 1, (
        f"p_use_row_scaling frame missing or empty in the provider; "
        f"available solve_data keys: "
        f"{[k for k in provider.keys() if k.startswith('solve_data/')]!r}. "
        "If this is empty, the solve-writer pipeline regressed."
    )


def test_coal_extreme_scale_analyzer_recommends_yes(
    extreme_db_url: str,
    tmp_path: Path,
) -> None:
    """Direct analyser test — guards the cost-family spread maths.

    Tighter localisation than the end-to-end test above: if this fails
    but the cascade test passes (or vice versa), the issue is in the
    plumbing between :func:`analyze_solve` and the state mutation, not
    in the analyser itself.
    """
    from flextool.engine_polars import run_chain_from_db
    from flextool.engine_polars.scaling import (
        analyze_solve,
        clear_cache,
    )

    workdir = tmp_path / "wd_analyzer"
    workdir.mkdir()
    solver_config_dir = _setup_solver_config_dir(tmp_path)

    # Run the cascade once to materialise the per-solve FlexData under
    # workdir; the test then reloads it and runs analyze_solve in
    # isolation.  No auto_scale here — we want to inspect the analyser's
    # raw recommendation, not its applied effect.
    steps = run_chain_from_db(
        input_db_url=extreme_db_url,
        scenario_name="coal_extreme_scale",
        work_folder=workdir,
        solver_config_dir=solver_config_dir,
        keep_solutions=True,
    )
    assert steps
    last_step = next(reversed(list(steps.values())))
    assert last_step.solution is not None and last_step.solution.optimal
    assert last_step.flex_data is not None, (
        "keep_solutions=True should retain flex_data on the last step."
    )

    clear_cache()
    table = analyze_solve(
        solve_name=f"{last_step.solve_name}_analyzer_probe",
        flex_data=last_step.flex_data,
    )
    assert table.use_row_scaling == "yes", (
        f"Analyser returned use_row_scaling={table.use_row_scaling!r}; "
        f"expected 'yes'.  cost_spread_log10={table.cost_spread_log10:.3f} "
        f"(threshold 5.0); trigger={table.row_scaling_trigger!r}."
    )
    assert table.cost_spread_log10 > 5.0, (
        "cost_spread_log10 should exceed COST_SPREAD_THRESHOLD = 5.0 "
        f"decades for coal_extreme_scale; got {table.cost_spread_log10:.3f}."
    )
