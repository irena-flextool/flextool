"""flextool's in-memory solve-to-solve handoff (PoC, add-on).

These tests live in flexpy_spike but exercise flextool's runner.  The
PoC scope is *non-interference*: when ``RunnerState.handoffs`` is left
as ``None`` (default), flextool's behavior must be bit-identical to
pre-handoff flextool.  When it's set to ``{}``, the post-solve hook
populates it with the same ``realized_invest`` data that today's
file-based ``solve_data/p_entity_period_existing_capacity.csv`` carries.

The scenario used is ``coal_unit_size_MIP_wind`` because it's the only
test fixture exercising a multi-solve cascade (a y2020_5week invest
solve handing realized invest to a y2020_fullYear_dispatch dispatch
solve).

Δ.22 status — 9 of 11 tests skip-marked.  The cascade integration tests
in this file exercise ``FlexToolRunner.run_model()``, which goes through
``SolverRunner.run()``.  Δ.22 collapsed ``SolverRunner.run`` to
``raise NotImplementedError`` (the legacy GMPL/HiGHS pipeline retired);
the only surviving runtime path is the native cascade (override on
``_FlexpyCascadeSolver.run``).  The PoC handoff mechanism these tests
verify (``RunnerState.handoffs`` populated by ``capture_post_solve``)
belongs to flextool's legacy orchestration loop and is not the path
the engine_polars cascade uses.  The native cascade has its own
in-memory handoff via ``_solve_handoff.SolveHandoff``, exercised by
``test_chain_handoff_writers.py`` etc.  The 2 unit-level tests in this
module (``test_cumulative_loaders_consume_from_handoff`` /
``test_write_fix_storage_files_from_handoff``) exercise the
SolveHandoff dataclass + writer in isolation and remain active."""
from __future__ import annotations

import contextlib
import filecmp
import io
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest

# flextool runner is reachable via the same path append the DB-direct
# loader uses.  Append so flexpy's local ``flextool/`` package keeps
# precedence as the importable name.  FLEXTOOL_REPO is still needed
# because we consume flextool's Python package (FlexToolRunner) live.
FLEXTOOL_REPO = Path("/home/jkiviluo/sources/flextool")
if str(FLEXTOOL_REPO) not in sys.path:
    sys.path.append(str(FLEXTOOL_REPO))

from flextool.flextoolrunner.flextoolrunner import FlexToolRunner  # noqa: E402
from flextool.flextoolrunner.solve_handoff import (  # noqa: E402
    SolveHandoff,
    write_fix_storage_files_from_handoff,
)

# Vendored fixture data + json_to_db live alongside this test file.
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES = TESTS_DIR / "fixtures"
sys.path.insert(0, str(TESTS_DIR))
from _db_utils import json_to_db  # noqa: E402  vendored


@pytest.fixture(autouse=True)
def _quiet():
    logging.getLogger().setLevel(logging.ERROR)
    # When the broader pytest run executes ``test_load_from_db`` first,
    # something in flextool's preprocessing path initializes highspy's
    # *global scheduler* with the default 16 threads.  Subsequent runs
    # with ``threads=1`` from ``tests/highs.opt`` then fail with
    # "global scheduler has already been initialized".  Reset it
    # before each handoff test so the suite-order dependency goes away.
    try:
        import highspy
        highspy.Highs.resetGlobalScheduler(True)
    except Exception:
        pass
    yield


def _run(scenario: str, work: Path, *, handoffs_on: bool):
    """Execute write_input + run_model for ``scenario`` in ``work``.
    Returns the runner's RunnerState (so the test can inspect handoffs)."""
    db = work / "tests.sqlite"
    if db.exists():
        db.unlink()
    db_url = json_to_db(FIXTURES / "tests.json", db)

    runner = FlexToolRunner(
        input_db_url=db_url,
        scenario_name=scenario,
        flextool_dir=FLEXTOOL_REPO / "flextool",
        bin_dir=FLEXTOOL_REPO / "bin",
        work_folder=work,
    )
    if handoffs_on:
        runner.state.handoffs = {}  # opt-in: enable the post-solve hook

    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        runner.write_input(db_url, scenario)
        rc = runner.run_model()
    assert rc == 0
    return runner.state


def _csv_files(d: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for sub in ("input", "solve_data"):
        root = d / sub
        if not root.exists():
            continue
        for p in root.rglob("*.csv"):
            out[str(p.relative_to(d))] = p.read_bytes()
    return out


# Files that legitimately differ across two runs even with no handoff
# changes — timestamps, scaling reports, etc.  We exclude these from
# the byte-equality check.  Mostly empty unless a solver run logs a
# timestamp into the CSV.
_VOLATILE: set[str] = {
    # solve_progress.csv records wall-clock timing per solve — naturally
    # differs across two back-to-back runs even with no code change.
    "solve_data/solve_progress.csv",
    # timings.csv (added in newer flextool) — same reason.
    "solve_data/timings.csv",
}


# Δ.22 retirement banner reused for the 9 cascade-integration tests in
# this module.  The PoC handoff exercised here uses
# ``FlexToolRunner.run_model()`` → ``SolverRunner.run()`` which Δ.22
# collapsed to ``NotImplementedError``.  The native cascade has its own
# in-memory handoff (``_solve_handoff.SolveHandoff``); the PoC mechanism
# these tests exercise is not on the surviving runtime path.
_DELTA_22_RETIRED = pytest.mark.skip(
    reason=(
        "Δ.22 retired SolverRunner.run; this PoC handoff test exercises "
        "FlexToolRunner.run_model() which is no longer functional. "
        "The native cascade's handoff mechanism is exercised by "
        "test_chain_handoff_writers.py + test_solve_handoff's two "
        "unit-level tests below."
    ),
)


@_DELTA_22_RETIRED
def test_handoff_off_by_default(tmp_path):
    """Sanity: without setting state.handoffs, flextool runs exactly as
    before — runner state has no handoffs dict."""
    state = _run("coal", tmp_path, handoffs_on=False)
    assert state.handoffs is None


@_DELTA_22_RETIRED
def test_handoff_capture_realized_invest(tmp_path):
    """When handoffs={}, the post-solve hook populates one entry per
    completed solve, with realized_invest matching the file-based
    output."""
    state = _run("coal_unit_size_MIP_wind", tmp_path, handoffs_on=True)
    assert state.handoffs is not None
    # Two solves cascaded — both should have entries.
    assert {"y2020_5week", "y2020_fullYear_dispatch"} <= set(state.handoffs)
    # realized_invest for the invest solve should match the on-disk
    # ``p_entity_period_existing_capacity.csv`` invested column.
    h = state.handoffs["y2020_5week"]
    assert isinstance(h, SolveHandoff)
    assert h.realized_invest is not None
    # Wind plant invests ~177 in the first solve.
    wind_row = h.realized_invest.filter(pl.col("entity") == "wind_plant")
    assert wind_row.height == 1
    assert wind_row["value"][0] > 100  # ~177


@_DELTA_22_RETIRED
def test_handoff_on_preserves_csv_outputs(tmp_path):
    """The whole point of "add-on": turning on state.handoffs must not
    perturb a single byte of the CSV outputs.  Run the same scenario
    twice — once with handoffs off, once on — and assert every CSV
    in input/ and solve_data/ is byte-identical."""
    work_off = tmp_path / "off"
    work_on  = tmp_path / "on"
    work_off.mkdir()
    work_on.mkdir()

    _run("coal_unit_size_MIP_wind", work_off, handoffs_on=False)
    _run("coal_unit_size_MIP_wind", work_on,  handoffs_on=True)

    files_off = _csv_files(work_off)
    files_on  = _csv_files(work_on)

    # Same set of files must exist.
    assert set(files_off) - _VOLATILE == set(files_on) - _VOLATILE

    # Every shared file must be byte-identical.
    diffs = [name for name, off in files_off.items()
             if name not in _VOLATILE and files_on.get(name) != off]
    assert not diffs, f"handoff-on perturbed {len(diffs)} CSVs: {diffs[:5]}"


@_DELTA_22_RETIRED
def test_handoff_capture_realized_existing(tmp_path):
    """``realized_existing`` is the *resolved* existing-capacity per
    period — different from ``realized_invest`` because pre-existing
    decay + divest enter this column."""
    state = _run("coal_unit_size_MIP_wind", tmp_path, handoffs_on=True)
    h = state.handoffs["y2020_5week"]
    assert h.realized_existing is not None
    # coal_plant existing should be its base (500) — pre-existing,
    # invariant of solve.
    coal = h.realized_existing.filter(pl.col("entity") == "coal_plant")
    assert coal["value"][0] == pytest.approx(500.0)


@_DELTA_22_RETIRED
def test_handoff_capture_divest_cumulative(tmp_path):
    """``coal_retire`` exercises divest: the realized solution divests
    the coal plant.  ``p_entity_divested.csv`` carries the cumulative
    divested capacity per entity."""
    state = _run("coal_retire", tmp_path, handoffs_on=True)
    # Single-solve scenario — exactly one entry.
    assert len(state.handoffs) == 1
    h = next(iter(state.handoffs.values()))
    assert h.divest_cumulative is not None
    coal = h.divest_cumulative.filter(pl.col("entity") == "coal_plant")
    assert coal.height == 1
    assert coal["value"][0] > 0  # something divested


@_DELTA_22_RETIRED
def test_handoff_capture_roll_end_state(tmp_path):
    """``multi_year_wind_no_investment`` writes a non-empty
    ``p_roll_continue_state.csv`` for the battery storage node."""
    state = _run("multi_year_wind_no_investment", tmp_path, handoffs_on=True)
    h = next(iter(state.handoffs.values()))
    assert h.roll_end_state is not None
    assert "battery" in h.roll_end_state["node"].to_list()


@_DELTA_22_RETIRED
def test_period_capacity_csv_populates(tmp_path):
    """``period_capacity.csv`` is written by every solve.  Δ.1 moved the
    in-memory mirror from ``SolveHandoff.periods_already_emitted`` to
    :class:`OutputWriterState`; the on-disk file is unchanged.  This
    test asserts the CSV-level invariant that flextool's writers still
    bump the file (the in-memory carrier moves with the new home — see
    ``test_output_writer.test_output_writer_state_periods_already_emitted``)."""
    state = _run("coal", tmp_path, handoffs_on=True)
    pae = state.paths.work_folder / "solve_data" / "period_capacity.csv"
    assert pae.exists(), "period_capacity.csv missing — handoff_writers regression"
    df = pl.read_csv(pae)
    assert "period" in df.columns
    assert "p2020" in df["period"].to_list()


@_DELTA_22_RETIRED
def test_handoff_unexercised_carriers_are_none(tmp_path):
    """Carriers whose source files are empty/missing in a scenario
    should leave their slot as None (not an empty frame).  ``coal``
    doesn't exercise fix_storage, cumulative_co2, cumulative_commodity,
    or cum_sim_hours."""
    state = _run("coal", tmp_path, handoffs_on=True)
    h = next(iter(state.handoffs.values()))
    assert h.fix_storage is None
    assert h.cumulative_co2 is None
    assert h.cumulative_commodity is None
    assert h.cum_sim_hours is None


def _run_with_corrupting_capture(
    scenario: str, work: Path, *, files_to_corrupt: tuple[str, ...],
):
    """Like ``_run(handoffs_on=True)`` but after each post-solve capture
    deposits the in-memory ``SolveHandoff``, scribble over the parent's
    CSVs in ``files_to_corrupt`` so any subsequent file-based read of
    them returns garbage.

    This is the strong consume-side proof: if a downstream preprocessing
    module still reads from those CSVs, solve 2's preprocessing will
    pick up the corruption.  If the consume side is wired correctly,
    preprocessing reads from ``state.handoffs`` and the disk values are
    irrelevant.
    """
    db = work / "tests.sqlite"
    if db.exists():
        db.unlink()
    db_url = json_to_db(FIXTURES / "tests.json", db)

    runner = FlexToolRunner(
        input_db_url=db_url,
        scenario_name=scenario,
        flextool_dir=FLEXTOOL_REPO / "flextool",
        bin_dir=FLEXTOOL_REPO / "bin",
        work_folder=work,
    )
    runner.state.handoffs = {}

    # Wrap capture_post_solve at its imported binding inside orchestration
    # — that's the actual call site, so binding-level patching works.
    from flextool.flextoolrunner import orchestration as _orch
    real_capture = _orch.capture_post_solve

    def corrupting_capture(state, solve_name):
        real_capture(state, solve_name)
        sd = state.paths.work_folder / "solve_data"
        for fname in files_to_corrupt:
            p = sd / fname
            if p.exists():
                # Header-only corruption: a downstream reader sees zero
                # rows where it previously saw real per-entity data.
                with p.open() as fh:
                    header = fh.readline()
                p.write_text(header)

    _orch.capture_post_solve = corrupting_capture
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            runner.write_input(db_url, scenario)
            rc = runner.run_model()
        assert rc == 0
    finally:
        _orch.capture_post_solve = real_capture
    return runner.state


def _read_csv_rows(path: Path) -> list[tuple[str, ...]]:
    """Return all data rows (excluding header) as tuples."""
    import csv
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [tuple(r) for r in reader if r]


@_DELTA_22_RETIRED
def test_handoff_consume_realized_invest_chain(tmp_path):
    """Strong consume-side proof: corrupt
    ``p_entity_period_existing_capacity.csv`` and ``p_entity_divested.csv``
    after each capture.  If solve 2's preprocessing reads from disk it
    sees zero rows and ``p_entity_all_existing.csv`` collapses to all
    zeros; if it reads from the in-memory handoff, the file matches the
    baseline (clean) run byte-for-byte."""
    work_clean = tmp_path / "clean"
    work_corrupt = tmp_path / "corrupt"
    work_clean.mkdir()
    work_corrupt.mkdir()

    # Baseline: handoff-off, no corruption.  (Equivalently handoff-on
    # without corruption — the non-interference test above confirms
    # the two paths produce identical CSVs.)
    _run("coal_unit_size_MIP_wind", work_clean, handoffs_on=False)

    # Test: handoff-on, with corrupting capture.  Solve 2 must consume
    # from in-memory handoff (the disk CSVs are header-only post-capture).
    _run_with_corrupting_capture(
        "coal_unit_size_MIP_wind", work_corrupt,
        files_to_corrupt=(
            "p_entity_period_existing_capacity.csv",
            "p_entity_divested.csv",
        ),
    )

    # End-state ``p_entity_all_existing.csv`` reflects the LAST solve's
    # preprocessing — which under handoff-on read from the in-memory
    # frame, not the corrupted disk file.  If consume side were broken,
    # the file would be all-zero (later_existing = 0 with empty ppec).
    clean_rows = _read_csv_rows(
        work_clean / "solve_data" / "p_entity_all_existing.csv"
    )
    corrupt_rows = _read_csv_rows(
        work_corrupt / "solve_data" / "p_entity_all_existing.csv"
    )
    assert clean_rows == corrupt_rows, (
        "consume-side regression: solve 2's p_entity_all_existing.csv "
        "differs between clean baseline and post-capture-corruption run "
        "— preprocessing must be reading the corrupted disk file rather "
        "than the in-memory handoff."
    )

    # Sanity: the file is non-trivial (not just an all-zero collapse
    # that would also pass an equality check if both runs collapsed).
    nonzero = [r for r in clean_rows if float(r[2]) > 0.0]
    assert nonzero, "baseline p_entity_all_existing.csv was all zeros"


def test_cumulative_loaders_consume_from_handoff(tmp_path):
    """Unit-level: the three cumulative-handoff prior loaders read from
    a populated ``SolveHandoff`` instead of disk when the optional
    ``prior_handoff`` kwarg is supplied.  Verifies disk is bypassed by
    pointing the file path at a non-existent location — only the
    handoff path can produce non-empty output."""
    from flextool.process_outputs.cumulative_handoffs import (
        _load_prior_co2_cum_realized_tonnes,
        _load_prior_cum_realized_mwh,
        _load_prior_cum_sim_hours,
    )

    # Doesn't exist — file-only path would return {}.
    nope = tmp_path / "does_not_exist.csv"

    h = SolveHandoff(
        cumulative_co2=pl.DataFrame({
            "group":  ["g1", "g2"],
            "period": ["p2025", "p2025"],
            "value":  [12.5, 99.0],
        }),
        cumulative_commodity=pl.DataFrame({
            "commodity": ["coal"],
            "tier":      [1],
            "period":    ["p2025"],
            "mwh":       [42.0],
        }),
        cum_sim_hours=pl.DataFrame({
            "period": ["p2025", "p2030"],
            "value":  [8760.0, 4380.0],
        }),
    )

    co2 = _load_prior_co2_cum_realized_tonnes(nope, prior_handoff=h)
    assert co2 == {("g1", "p2025"): 12.5, ("g2", "p2025"): 99.0}

    mwh = _load_prior_cum_realized_mwh(nope, prior_handoff=h)
    assert mwh == {("coal", 1, "p2025"): 42.0}

    hrs = _load_prior_cum_sim_hours(nope, prior_handoff=h)
    assert hrs == {"p2025": 8760.0, "p2030": 4380.0}

    # Sanity: without the handoff, the same call returns empty (proving
    # the disk path is what the handoff is bypassing).
    assert _load_prior_co2_cum_realized_tonnes(nope) == {}
    assert _load_prior_cum_realized_mwh(nope) == {}
    assert _load_prior_cum_sim_hours(nope) == {}


def test_write_fix_storage_files_from_handoff(tmp_path):
    """Unit-level: the wide ``fix_storage`` handoff frame fans back out
    to three long-format ``fix_storage_*.csv`` files on disk, with NULL
    metric rows excluded per file and the time-axis column renamed
    from ``time`` to ``step`` (the on-disk convention).

    This is the only test for the fix_storage consume helper; no
    current fixture exercises the orchestration-level shutil.copy
    branch it replaces (all 18 scenarios have empty fix_storage_*.csv
    on disk)."""
    sd = tmp_path / "solve_data"
    sd.mkdir()

    # Wide row with mixed NULLs — the producer's natural output shape
    # per ``capture_post_solve``.
    fix_storage = pl.DataFrame({
        "node":     ["battery", "battery", "tank"],
        "period":   ["p2025",   "p2025",   "p2030"],
        "time":     ["t0001",   "t0002",   "t0001"],
        "quantity": [10.0, 20.0, None],
        "price":    [None, 5.0,  None],
        "usage":    [None, None, 0.7],
    })

    write_fix_storage_files_from_handoff(fix_storage, sd)

    # Each per-metric file should contain only its non-NULL rows, with
    # the on-disk column name and ``step`` (not ``time``) for the axis.
    qty = pl.read_csv(sd / "fix_storage_quantity.csv")
    assert qty.columns == ["node", "period", "step", "p_fix_storage_quantity"]
    assert qty.height == 2
    assert sorted(qty["p_fix_storage_quantity"].to_list()) == [10.0, 20.0]

    price = pl.read_csv(sd / "fix_storage_price.csv")
    assert price.columns == ["node", "period", "step", "p_fix_storage_price"]
    assert price.height == 1
    assert price["p_fix_storage_price"][0] == 5.0
    assert price["step"][0] == "t0002"

    usage = pl.read_csv(sd / "fix_storage_usage.csv")
    assert usage.columns == ["node", "period", "step", "p_fix_storage_usage"]
    assert usage.height == 1
    assert usage["node"][0] == "tank"
    assert usage["p_fix_storage_usage"][0] == 0.7
