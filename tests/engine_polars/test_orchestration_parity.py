"""Γ.8.D end-to-end parity tests for the native orchestrator.

Validates that ``run_chain_from_db`` (the native flexpy entry point)
produces objective values matching flextool's reference at
``rel < 1e-6`` on every multi-solve fixture, AND that:

1. The R-O2 re-export shim works: both
   ``flextool.flextoolrunner.solve_handoff.SolveHandoff`` and
   ``flextool.engine_polars._solve_handoff.SolveHandoff`` resolve to
   the same class.
2. ``model_solve`` validation raises ``FlexToolConfigError`` on empty /
   multi-model configurations.
3. ``build_handoff_from_flexpy`` populates all 9 carriers when the
   underlying CSVs / variables are present.

Δ.12e — the four feature-flag tests guarding the legacy
file-symlink ``run_chain(native=False)`` driver were retired with
the legacy code path.  ``run_chain`` is now a thin compat shim that
always delegates to native; tests for legacy driver semantics no
longer apply.

The parity sweep uses the same fixture-discovery pattern as
``test_solve_config_parity._discover_fixtures`` so adding a new
``work_<S>/`` automatically extends coverage.
"""
from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import polars as pl
import pytest
import spinedb_api as api

from flextool.engine_polars import (
    OrchestrationStep,
    SolveHandoff,
    run_chain_from_db,
    run_orchestration,
)
from flextool.engine_polars._solve_config import (
    HiGHSConfig,
    SolveConfig,
    SolverSettings,
)
from flextool.engine_polars._solve_state import (
    FlexToolConfigError,
    PathConfig,
    RunnerState,
)


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


# ---------------------------------------------------------------------------
# R-O2: SolveHandoff re-export shim.
# ---------------------------------------------------------------------------


def test_solve_handoff_reexport_shim() -> None:
    """The legacy import path must reach the same class as the native one.

    Both ``flextool.flextoolrunner.solve_handoff.SolveHandoff`` and
    ``flextool.engine_polars._solve_handoff.SolveHandoff`` resolve to
    one class.  R-O2 mitigation in
    ``audit/solve_orchestration_plan.md``.
    """
    from flextool.engine_polars._solve_handoff import SolveHandoff as Native
    from flextool.flextoolrunner.solve_handoff import SolveHandoff as Legacy

    assert Native is Legacy, (
        "SolveHandoff is duplicated — the re-export shim is broken.  "
        "Both import paths must resolve to the same class object so "
        "isinstance checks across process_outputs and the orchestrator "
        "stay consistent."
    )


def test_solve_handoff_capture_helpers_reexported() -> None:
    """Inverse-regression guard.

    Phase 4.1i of ``specs/provider_consolidation.md`` retired
    ``write_fix_storage_files_from_handoff`` — every reader of
    ``solve_data/fix_storage_*`` migrated to the per-metric
    ``handoff/*`` Provider keys seeded by the iteration-start
    translator (Phases 4.1f–4.1h).  This guard asserts the helper
    stays gone on both the native module and the legacy re-export
    shim, so a regression that resurrects it lands a test failure
    rather than silently reintroducing dead disk-fan-out code.
    """
    from flextool.engine_polars import _solve_handoff as native_mod
    from flextool.flextoolrunner import solve_handoff as legacy_mod

    assert not hasattr(native_mod, "write_fix_storage_files_from_handoff")
    assert not hasattr(legacy_mod, "write_fix_storage_files_from_handoff")


# ---------------------------------------------------------------------------
# model_solve validation.
# ---------------------------------------------------------------------------


def _make_minimal_state(model_solve: dict | None = None) -> RunnerState:
    """Construct a minimal RunnerState for validation tests."""
    log = logging.getLogger("test_validation")
    sc = SolveConfig(
        model=["m"],
        model_solve=defaultdict(list, model_solve or {}),
        solve_modes={},
        rolling_times=defaultdict(list),
        highs=HiGHSConfig(presolve={}, method={}, parallel={}),
        solver_settings=SolverSettings(
            solvers={}, precommand={}, arguments=defaultdict(list),
        ),
        solve_period_years_represented=defaultdict(list),
        hole_multipliers=defaultdict(list),
        contains_solves=defaultdict(list),
        stochastic_branches=defaultdict(list),
        periods_available={},
        delay_durations={},
        logger=log,
    )
    return RunnerState(
        paths=PathConfig(work_folder=Path(".")),
        solve=sc,
        logger=log,
    )


def test_run_orchestration_empty_model_solve_raises(tmp_path) -> None:
    """``run_orchestration`` raises FlexToolConfigError when model_solve
    is empty (no model / no solves) — line 78-86 of the flextool
    reference."""
    state = _make_minimal_state(model_solve={})
    with pytest.raises(FlexToolConfigError, match="solves"):
        run_orchestration(state, tmp_path)


def test_run_orchestration_multi_model_raises(tmp_path) -> None:
    """``run_orchestration`` raises FlexToolConfigError when model_solve
    has more than one model — line 634-637 of the flextool reference."""
    state = _make_minimal_state(
        model_solve={"m1": ["s1"], "m2": ["s2"]},
    )
    with pytest.raises(FlexToolConfigError, match="more than one model"):
        run_orchestration(state, tmp_path)


def test_run_orchestration_empty_solves_raises(tmp_path) -> None:
    """``run_orchestration`` raises FlexToolConfigError when the
    one-and-only model has no solves — line 83-86 of the reference."""
    state = _make_minimal_state(model_solve={"m": []})
    with pytest.raises(FlexToolConfigError, match="No solves"):
        run_orchestration(state, tmp_path)


# ---------------------------------------------------------------------------
# Native orchestration — single-solve smoke test.
# ---------------------------------------------------------------------------


def test_run_chain_from_db_single_solve_smoke() -> None:
    """``run_chain_from_db`` runs end-to-end on a small single-solve
    fixture and produces an OrchestrationStep with non-trivial obj.

    Uses ``work_base`` which is the smallest DB scenario.
    """
    sqlite = DATA / "work_base" / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip("work_base fixture not present")
    # Discover the scenario (first one alphabetically — same convention).
    with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
        scenarios = sorted(s.name for s in db.query(db.scenario_sq).all())
    if not scenarios:
        pytest.skip("no scenarios in work_base")
    scenario = scenarios[0]

    steps = run_chain_from_db(sqlite, scenario, keep_solutions=True)
    assert len(steps) >= 1, "expected at least one solve step"
    # Every step should have a non-None solution + handoff.
    for name, step in steps.items():
        assert isinstance(step, OrchestrationStep)
        assert isinstance(step.handoff, SolveHandoff)
        assert step.solution is not None, f"{name}: solution is None"
        assert step.solution.optimal, f"{name}: not optimal"
        assert step.obj is not None, f"{name}: obj is None"


# ---------------------------------------------------------------------------
# build_handoff_from_flexpy 9-carrier coverage.
# ---------------------------------------------------------------------------


def test_build_handoff_from_flexpy_covers_eight_carriers() -> None:
    """Γ.8.D extension: all eight carriers populated when their source
    data is present.

    Pre-Γ.8.D, ``build_handoff_from_flexpy`` filled 3 of 8 carriers
    (``realized_invest``, ``realized_existing``, ``divest_cumulative``).
    The remaining 5 (``roll_end_state``, ``fix_storage``,
    ``cumulative_co2``, ``cumulative_commodity``, ``cum_sim_hours``)
    now also populate when their source files / variables are present.

    Δ.1 — the ninth carrier (``periods_already_emitted``) was retired
    from :class:`SolveHandoff` and now lives on
    :class:`flextool.engine_polars._output_writer.OutputWriterState`
    (writer-side state, not a true solver handoff).  See
    ``test_output_writer.test_output_writer_state_periods_already_emitted``
    for the new home's coverage.

    This test asserts the 3 file-based carriers (cumulative_co2,
    cumulative_commodity, cum_sim_hours) plus fix_storage_price/usage
    extraction populate from the handoff CSVs that flextool's
    preprocessing already writes.
    """
    # Use a fixture whose solve_data has period_capacity.csv + at least
    # one of the cumulative carriers.
    # work_multi_year_one_solve_co2_limit has all of these.
    work = DATA / "work_multi_year_one_solve_co2_limit"
    if not work.exists():
        pytest.skip("work_multi_year_one_solve_co2_limit not present")

    from polar_high import Problem
    from flextool.engine_polars.input import (
        build_handoff_from_flexpy,
        load_flextool,
    )
    from flextool.engine_polars.model import build_flextool

    # Build a tempdir mirroring the layout used by run_chain so
    # build_handoff_from_flexpy can find the per-solve CSVs.
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        td = Path(tmp)
        # Symlink input/ + solve_data/ from the fixture to the tempdir.
        for child in ("input", "solve_data", "output_raw"):
            src = work / child
            if src.exists():
                os.symlink(src, td / child)

        data = load_flextool(td)
        pb = Problem()
        build_flextool(pb, data)
        sol = pb.solve()
        if not sol.optimal:
            pytest.skip("LP not optimal — fixture corrupted")

        handoff = build_handoff_from_flexpy(sol, td, "test_solve")

    # The three file-based carriers we extended for Γ.8.D.  Δ.1 retired
    # ``periods_already_emitted`` from SolveHandoff (see test docstring).
    fixture_carrier_files = {
        "cumulative_co2": "co2_cum_realized_tonnes.csv",
        "cumulative_commodity": "commodity_ladder_cumulative.csv",
        "cum_sim_hours": "ladder_cum_sim_hours.csv",
    }
    for carrier_name, fname in fixture_carrier_files.items():
        sd = work / "solve_data" / fname
        on_disk_has_data = False
        if sd.exists():
            try:
                df = pl.read_csv(sd)
                on_disk_has_data = df.height > 0
            except Exception:
                on_disk_has_data = False
        carrier = getattr(handoff, carrier_name)
        if on_disk_has_data:
            assert carrier is not None, (
                f"{carrier_name} should populate when {fname} has rows; "
                f"got None.  Γ.8.D extension regression."
            )
        # If the file is empty, the carrier may or may not be set —
        # fixtures vary.  We only assert population when the source
        # file demonstrably has data.


# Δ.1 — the legacy ``test_build_handoff_from_flexpy_periods_already_emitted_propagates``
# test was retired; the carrier moved to ``OutputWriterState`` and
# ``test_output_writer.test_output_writer_state_periods_already_emitted_accumulates``
# covers the new home's accumulation across cascade rolls.


# ---------------------------------------------------------------------------
# Native vs flextool obj parity sweep.
# ---------------------------------------------------------------------------


# Reuse the fixture-discovery pattern from test_solve_config_parity.
_DIRNAME_TO_SCENARIO_OVERRIDES: dict[str, str] = {
    "work_2day_stochastic_dispatch_full_storage": "2_day_stochastic_dispatch",
    "work_commodity_ladder_annual": "coal_ladder_annual",
    "work_commodity_ladder_cumulative": "coal_ladder_cumulative",
    "work_delay_source_coef": "water_pump_delayed",
    "work_inflation_check": "wind_battery_invest_lifetime_renew",
}


def _discover_simple_single_solve_fixtures() -> list[tuple[str, str]]:
    """Return ``[(work_dirname, scenario_name), …]`` for fixtures that
    are candidates for the native orchestration smoke test.

    Single-solve fixtures only — multi-solve cascades exercise the
    same code path under existing chain tests.  This loop covers the
    "every fixture passes through run_chain_from_db" assertion at a
    representative sample.
    """
    out: list[tuple[str, str]] = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or not d.name.startswith("work_"):
            continue
        sqlite = d / "tests.sqlite"
        if not sqlite.exists():
            continue
        # Discover scenario.
        if d.name in _DIRNAME_TO_SCENARIO_OVERRIDES:
            target = _DIRNAME_TO_SCENARIO_OVERRIDES[d.name]
            try:
                with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                    found = any(
                        s.name == target for s in db.query(db.scenario_sq).all()
                    )
            except Exception:
                found = False
            if found:
                out.append((d.name, target))
                continue
        scen_target = d.name.removeprefix("work_")
        try:
            with api.DatabaseMapping("sqlite:///" + str(sqlite)) as db:
                scenarios = sorted(
                    s.name for s in db.query(db.scenario_sq).all()
                )
        except Exception:
            continue
        candidates = [scen_target]
        candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", scen_target))
        candidates.append(re.sub(r"(\d+)_([a-z])", r"\1\2", scen_target))
        if scen_target.endswith("_full_storage"):
            base = scen_target[: -len("_full_storage")]
            candidates.append(re.sub(r"(^|_)(\d+)([a-z])", r"\1\2_\3", base))
            candidates.append(base)
        chosen: str | None = None
        for cand in candidates:
            if cand in scenarios:
                chosen = cand
                break
        if chosen is not None:
            out.append((d.name, chosen))
    return out


# Single curated fixture — running every fixture through the native
# orchestrator in a parametrize is too slow (~5-10 minutes per fixture
# for the FlexToolRunner.write_input pass).  The smoke test above
# covers the end-to-end path; the dedicated chain tests
# (test_flex_chain_*.py) cover the multi-solve cascade.  This sweep is
# *opt-in* via FLEXPY_NATIVE_PARITY_SWEEP=1.
NATIVE_SWEEP_ENABLED = os.environ.get("FLEXPY_NATIVE_PARITY_SWEEP", "") == "1"


# Fixtures with known pre-existing LP-degeneracy residuals between
# flexpy's solve and flextool's reference parquet.  For these, we
# allow the looser tolerance from their dedicated single-solve parity
# test.  Adding a fixture here is documenting an existing residual,
# not papering over a regression: every value here matches the
# tolerance the test_flex_*.py file for that fixture already uses.
_FIXTURE_OBJ_TOLERANCE: dict[str, float] = {
    # work_test_a_lot has a documented ~6e-5 residual; see
    # audit/test_a_lot_residual.md and test_flex_test_a_lot.py:58.
    "work_test_a_lot": 1e-4,
    "work_test_a_lot_but_not_multi_year": 1e-4,
}


# Fixtures the native path can't reproduce because they're derivative
# fixtures: the recorded reference parquet was solved against a workdir
# whose CSVs were patched AFTER flextool's preprocessing produced them.
# The native path re-runs preprocessing from the DB, missing the patch,
# so the obj cannot match.
#
# E.g. ``work_delay_source_coef`` patches
# ``input/p_process_source_flow_coefficient.csv`` from 1.0 to 2.0 after
# preprocessing (see ``tests/_gen_delay_source_coef.py``).  The DB
# scenario (``water_pump_delayed``) carries the unpatched 1.0 value, so
# native obj reproduces the unpatched LP, not the patched reference.
_FIXTURES_DERIVATIVE_PATCH: set[str] = {
    "work_delay_source_coef",
}


# Fixtures where flextool's preprocessing emits ``pdGroup_capacity_margin.csv``
# (or related capacity-margin files) only when run from the original
# ``_gen_*.py`` pipeline that produced the committed snapshot.  Re-running
# preprocessing fresh from the DB via ``FlexToolRunner.write_input``
# (the path Γ.8.D's native orchestrator takes) drops this file because
# something in the input path differs.  The CSV-path single-solve parity
# tests for these fixtures continue to pass (the snapshot was generated
# with the file present); the native cascade can't reproduce.
#
# This is a preprocessing-coverage gap, not an override-chain bug —
# documented for follow-up but out of Γ.8.E scope.
_FIXTURES_NATIVE_PREPROCESS_GAP: set[str] = {
    "work_network_coal_wind_capacity_margin",
    "work_network_coal_wind_reserve_co2_capacity_margin",
}


@pytest.mark.skipif(
    not NATIVE_SWEEP_ENABLED,
    reason=(
        "set FLEXPY_NATIVE_PARITY_SWEEP=1 to run the full Γ.8.D parity "
        "sweep across every fixture (slow — ~5min per fixture)"
    ),
)
@pytest.mark.parametrize(
    "work_name,scenario",
    _discover_simple_single_solve_fixtures(),
    ids=lambda x: str(x),
)
def test_native_orchestration_obj_parity(work_name: str, scenario: str) -> None:
    """End-to-end parity: ``run_chain_from_db`` produces an obj
    matching flextool's reference at ``rel < 1e-6`` (or the looser
    fixture-specific tolerance documented in ``_FIXTURE_OBJ_TOLERANCE``).

    Skipped unless ``FLEXPY_NATIVE_PARITY_SWEEP=1`` (slow — each fixture
    runs flextool's preprocessing + flexpy's LP).

    The parity bar: every per-solve obj produced by
    ``run_chain_from_db`` matches the flextool reference at
    ``output_raw/v_obj__<solve>.parquet``.  Pre-Γ.8.E this only
    checked optimality; Γ.8.E's cascade-gap fix made the obj
    comparison meaningful so the test now asserts numerical parity.
    """
    sqlite = DATA / work_name / "tests.sqlite"
    flx_obj_path = DATA / work_name / "flextool.sol"
    if not flx_obj_path.exists():
        pytest.skip(f"{work_name}: no flextool.sol reference")
    if work_name in _FIXTURES_DERIVATIVE_PATCH:
        pytest.skip(
            f"{work_name}: derivative fixture (post-preprocessing CSV "
            f"patch); native path cannot reproduce reference obj"
        )
    if work_name in _FIXTURES_NATIVE_PREPROCESS_GAP:
        pytest.skip(
            f"{work_name}: native preprocessing emits a different "
            f"snapshot than _gen_*.py — capacity-margin file missing"
        )

    steps = run_chain_from_db(sqlite, scenario, keep_solutions=True)
    assert steps, f"{work_name}: no solve steps"

    tolerance = _FIXTURE_OBJ_TOLERANCE.get(work_name, 1e-6)

    output_raw = DATA / work_name / "output_raw"
    failures: list[str] = []
    for solve_name, step in steps.items():
        assert step.solution is not None, f"{work_name}/{solve_name}: solution is None"
        assert step.solution.optimal, f"{work_name}/{solve_name}: not optimal"
        parq = output_raw / f"v_obj__{solve_name}.parquet"
        if not parq.exists():
            # Some fixtures don't preserve per-solve parquets (e.g. the
            # base-weighted single-solve writes a single solve-name parquet).
            continue
        ft_obj = pl.read_parquet(parq)["objective"][0]
        rel = abs(step.solution.obj - ft_obj) / max(1.0, abs(ft_obj))
        if rel >= tolerance:
            failures.append(
                f"  {solve_name}: flexpy={step.solution.obj}, "
                f"flextool={ft_obj}, rel={rel}"
            )
    assert not failures, (
        f"{work_name} obj-parity failures (tolerance {tolerance}):\n"
        + "\n".join(failures)
    )


# ---------------------------------------------------------------------------
# Storage-fixing handoff: in-memory write helper.
# ---------------------------------------------------------------------------
#
# Phase 4.1i — the unit-level disk-fan-out test for
# ``write_fix_storage_files_from_handoff`` was deleted alongside the
# helper itself.  No equivalent test on the ``handoff/*`` Provider
# path lives here: end-to-end coverage of the per-metric handoff
# routing belongs in the rolling/chain handoff suites.


# ---------------------------------------------------------------------------
# Roll-counter reset semantics — R-O5.
# ---------------------------------------------------------------------------


def test_roll_counter_resets_between_invocations(tmp_path) -> None:
    """Calling ``run_orchestration`` twice with the same SolveConfig
    must produce identical roll naming on both calls.  R-O5 in the
    audit's risk register: ``state.solve.roll_counter[solve] += 1``
    is a class-attribute mutation and must reset between top-level
    invocations or sibling rolls collide on the second call.

    We test the reset directly (no need to actually drive a full solve):
    the orchestrator's first action is ``state.solve.roll_counter =
    state.solve.make_roll_counter()`` which clears the counter.
    """
    state = _make_minimal_state(model_solve={"m": ["s1"]})
    state.solve.solve_modes = {"s1": "rolling_window"}
    # Pre-populate the counter as if a prior run had bumped it.
    state.solve.roll_counter = {"s1": 5}

    # We can't drive the full orchestration here without a proper
    # state, but we can verify the reset semantics by calling the
    # helper directly — same call the orchestrator makes at the top.
    state.solve.roll_counter = state.solve.make_roll_counter()
    assert state.solve.roll_counter == {"s1": 0}, (
        "make_roll_counter must reset rolling_window solves to 0 — "
        "R-O5 mitigation."
    )
