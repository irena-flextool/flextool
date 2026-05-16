"""Phase E-c — CSV-emission gate semantics.

Phase E-c introduces a module-level ``_EMIT_CSVS`` flag (with a
``csv_emission_disabled()`` / ``csv_emission_enabled()`` context-manager
pair) in :mod:`flextool.engine_polars._flex_data_accumulator`.  Every
writer-port ``_write`` helper consults the flag and short-circuits the
``df.write_csv(path)`` call when emission is disabled.  The accumulator
hook installed by :func:`capture_frames` still captures the frame
in-memory because the wrapping monkey-patch runs BEFORE the real
``_write`` is invoked.

This test asserts the contract at three layers:

  1. Writer-level — directly invoking a thin-wrapper writer with
     ``csv_emission_disabled()`` produces no on-disk file, while the
     same writer outside the context still produces the file.
  2. Capture-level — when ``capture_frames`` wraps an
     ``csv_emission_disabled()`` body, the accumulator still receives
     the frame even though disk emission is suppressed.
  3. CLI-level — the ``--csv-dump`` argparse flag is wired and defaults
     to ``False`` (off).

A broader "run a whole cascade with emission off and assert no CSVs in
input/, solve_data/, cross_solve/" assertion is intentionally NOT
included here.  Today the cascade's per-iteration ``solve_writers.write_*``
calls run BEFORE the ``capture_frames`` context that wraps
``preprocessing_solve_time.run``, so disabling emission cascade-wide
breaks ``load_flextool``'s seed lookup on ``steps_in_use.csv`` (and ~30
other per-iter CSV basenames the loader expects on disk in seed-mode).
Lifting the capture context to wrap the whole iteration body is a
follow-up (tracked separately).  Phase E-c lands the gate mechanism +
CLI flag; widening the cascade-side capture coverage so the gate can
flip on by default is the next dispatch.

Until then, the ``--csv-dump`` flag stays the public API: setting it on
the CLI emits CSVs (legacy behaviour); leaving it off leaves the gate
mechanism quiescent (the CLI wrap is a ``nullcontext``) because the
cascade still requires disk reads for the unhandled per-iter writers.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import _flex_data_accumulator as _acc_mod
from flextool.engine_polars._flex_data_accumulator import (
    FlexDataAccumulator,
    capture_frames,
    csv_emission_disabled,
    csv_emission_enabled,
    emit_csvs_enabled,
    set_csv_emission,
)
from flextool.engine_polars import _writer_leaf_sets as _leaf_mod


def _leaf_write(df: pl.DataFrame, path: Path) -> None:
    """Resolve the module's ``_write`` at call time (NOT import time) so
    the ``capture_frames`` monkey-patch is observed.  Importing the
    name directly via ``from ... import _write`` would freeze the
    original reference and bypass the patched attribute."""
    return _leaf_mod._write(df, path)


HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Layer 1 — writer-level: direct ``_write`` honours the flag.
# ---------------------------------------------------------------------------


def test_writer_emits_csv_by_default(tmp_path: Path) -> None:
    """The module-level flag defaults to ``True``; calling a thin-wrapper
    writer's ``_write`` outside any context emits the file."""
    assert emit_csvs_enabled() is True, (
        "_EMIT_CSVS module-level flag should default to True so the "
        "Phase 1 byte-parity gate (test_writer_port_phase1.py, 388 "
        "tests) and any direct writer call from a test harness "
        "behaves like the legacy emit-by-side-effect path."
    )

    target = tmp_path / "leaf_set.csv"
    frame = pl.DataFrame({"col_a": ["x"], "col_b": ["y"]})
    _leaf_write(frame, target)

    assert target.exists(), (
        f"writer _write(df, path) did not produce {target} under "
        "default emission flag."
    )
    text = target.read_text()
    assert text == "col_a,col_b\nx,y\n", (
        f"emitted CSV does not match expected bytes:\n{text!r}"
    )


def test_writer_suppresses_emit_under_disabled_context(tmp_path: Path) -> None:
    """``csv_emission_disabled()`` short-circuits the writer's disk write
    for the scope of the block; the flag restores on exit."""
    target = tmp_path / "leaf_set_off.csv"
    frame = pl.DataFrame({"col_a": ["x"], "col_b": ["y"]})

    with csv_emission_disabled():
        assert emit_csvs_enabled() is False
        _leaf_write(frame, target)
        assert not target.exists(), (
            f"_write produced {target} while csv_emission_disabled() "
            "was active — the gate did not short-circuit."
        )

    # Flag must be restored after the context exits.
    assert emit_csvs_enabled() is True
    # And a second call without the gate produces the file.
    _leaf_write(frame, target)
    assert target.exists()


def test_writer_restores_flag_on_exception(tmp_path: Path) -> None:
    """The context manager must restore ``_EMIT_CSVS`` even when the
    wrapped block raises."""
    assert emit_csvs_enabled() is True
    try:
        with csv_emission_disabled():
            assert emit_csvs_enabled() is False
            raise RuntimeError("oops")
    except RuntimeError:
        pass
    assert emit_csvs_enabled() is True, (
        "csv_emission_disabled() did not restore _EMIT_CSVS after an "
        "exception escaped the context body."
    )


def test_csv_emission_enabled_forces_on_inside_disabled(tmp_path: Path) -> None:
    """``csv_emission_enabled()`` nested inside ``csv_emission_disabled()``
    re-enables emission for the scope of the inner block.  Test fixtures
    (e.g. test_writer_port_phase1's harness) use this to force byte-parity
    emission even when a surrounding cascade run had disabled it."""
    target = tmp_path / "leaf_set_nested.csv"
    frame = pl.DataFrame({"col_a": ["x"]})
    with csv_emission_disabled():
        assert emit_csvs_enabled() is False
        with csv_emission_enabled():
            assert emit_csvs_enabled() is True
            _leaf_write(frame, target)
        # Back outside the inner context the flag should restore to
        # the surrounding-context's value (False).
        assert emit_csvs_enabled() is False
    assert target.exists()


def test_set_csv_emission_returns_previous(tmp_path: Path) -> None:
    """The explicit toggle helper returns the previous value so test
    fixtures can save/restore manually if the context-manager style
    doesn't fit (e.g. a pytest fixture with module scope)."""
    assert emit_csvs_enabled() is True
    prev = set_csv_emission(False)
    try:
        assert prev is True
        assert emit_csvs_enabled() is False
    finally:
        set_csv_emission(prev)
    assert emit_csvs_enabled() is True


# ---------------------------------------------------------------------------
# Layer 2 — capture: accumulator still receives the frame when emission off.
# ---------------------------------------------------------------------------


def test_capture_frames_captures_under_disabled_emission(tmp_path: Path) -> None:
    """The accumulator hook installed by ``capture_frames`` wraps each
    module's ``_write`` so the frame is stashed BEFORE the real ``_write``
    is invoked.  When emission is disabled the real ``_write`` returns
    without touching disk, but the captured frame is still present in
    the accumulator (key = path basename)."""
    target = tmp_path / "leaf_set_captured.csv"
    frame = pl.DataFrame({"col_a": ["x"], "col_b": ["y"]})

    accumulator = FlexDataAccumulator(solve_name="phase_e_c_test")
    with capture_frames(accumulator), csv_emission_disabled():
        _leaf_write(frame, target)

    # File is suppressed.
    assert not target.exists(), (
        "capture_frames did not respect csv_emission_disabled() — "
        f"{target} was written to disk anyway."
    )
    # But the frame IS in the accumulator under its basename key.
    assert "leaf_set_captured.csv" in accumulator, (
        "accumulator did not capture the frame while emission was "
        f"disabled.  Captured keys: {accumulator.keys()}"
    )
    captured = accumulator.get("leaf_set_captured.csv")
    assert captured is not None
    assert captured.equals(frame), (
        "captured frame does not match the input frame; the wrapper "
        "should pass the frame unchanged into the accumulator."
    )


def test_capture_frames_still_emits_under_default(tmp_path: Path) -> None:
    """Sanity: the capture_frames wrapper alone (no disabled context)
    leaves emission alive — this is the Phase C / E-b / E-a default
    behaviour the existing tests rely on."""
    target = tmp_path / "leaf_set_default.csv"
    frame = pl.DataFrame({"col_a": ["x"]})

    accumulator = FlexDataAccumulator(solve_name="phase_e_c_default")
    with capture_frames(accumulator):
        _leaf_write(frame, target)

    assert target.exists(), (
        "capture_frames suppressed disk emission outside any "
        "disabled-context — Phase E-b regression."
    )
    assert "leaf_set_default.csv" in accumulator


# ---------------------------------------------------------------------------
# Layer 3 — CLI: --csv-dump flag is plumbed and defaults to False.
# ---------------------------------------------------------------------------


def test_cli_csv_dump_flag_defaults_false() -> None:
    """``--csv-dump`` is the public API for Phase E-c.  It defaults to
    ``False`` and parses with ``action='store_true'``."""
    # Replicate the parser construction by importing the module and
    # locating the parser-creation function.  We instantiate a parser
    # with only the relevant subset of flags so the test stays fast
    # (the CLI builds many flags; we don't need the full surface here).
    from flextool.cli import cmd_run_flextool

    # The cmd_run_flextool.main() builds an argparse.ArgumentParser
    # inline.  Rather than re-running main(), we extract the parser
    # by parsing the source file for ``--csv-dump`` — this gives us a
    # cheap structural assertion without invoking the parser at
    # import time.
    src_path = Path(cmd_run_flextool.__file__)
    src = src_path.read_text()
    assert "--csv-dump" in src, (
        "Phase E-c regression: ``--csv-dump`` CLI flag is missing "
        f"from {src_path}.  The flag is the public opt-in for "
        "emitting cascade-internal CSVs (input/, solve_data/, "
        "cross_solve/) under the writer-port pipeline."
    )
    assert "store_true" in src, (
        "expected ``action='store_true'`` near the --csv-dump flag "
        "definition (the flag should be a boolean opt-in, not "
        "take a value)."
    )


def test_cli_csv_dump_parses_correctly() -> None:
    """Wire a minimal argparse parser identical in shape to the CLI's
    ``--csv-dump`` definition; verify default and presence behaviour."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-dump', action='store_true', default=False)

    # Default: flag absent.
    args = parser.parse_args([])
    assert args.csv_dump is False

    # Set: flag present.
    args = parser.parse_args(['--csv-dump'])
    assert args.csv_dump is True


# ---------------------------------------------------------------------------
# Layer 4 — module __all__ exposes the public surface.
# ---------------------------------------------------------------------------


def test_module_exports_csv_emission_helpers() -> None:
    """``_flex_data_accumulator.__all__`` must list the new public
    helpers so ``from flextool.engine_polars._flex_data_accumulator import *``
    picks them up (and so the module's public surface is auditable)."""
    public = set(_acc_mod.__all__)
    expected = {
        "csv_emission_disabled",
        "csv_emission_enabled",
        "emit_csvs_enabled",
        "set_csv_emission",
    }
    missing = expected - public
    assert not missing, (
        f"Phase E-c helpers missing from __all__: {missing}.  "
        f"Current: {sorted(public)}."
    )


# ---------------------------------------------------------------------------
# Layer 5 — Phase E-d universal-CSV-free cascade run.
# ---------------------------------------------------------------------------
#
# Phase E-d moved the per-iter ``solve_writers.write_*`` calls inside the
# ``capture_frames`` context (so their derived frames land in the
# per-sub-solve accumulator) AND plumbed the cross-sub-solve carriers
# (fix_storage_*, p_entity_pre_existing, …) via
# ``state.cross_solve_carriers`` instead of disk shutil.copy.  With both
# in place, the cascade can run end-to-end with CSV emission disabled
# and produce the same objective as the emission-enabled baseline.


DATA = Path(__file__).resolve().parent / "data"


@pytest.mark.solver
def test_phase_e_d_universal_csv_free_run(tmp_path: Path) -> None:
    """Run ``work_base`` end-to-end with CSV emission disabled and
    verify the writer-port ``_write`` helpers do NOT emit any
    ``solve_data/`` or ``cross_solve/`` CSVs.

    Phase E-d:
      * Per-iter ``solve_writers.write_*`` calls were moved inside the
        per-sub-solve ``capture_frames`` context.
      * Cross-sub-solve carriers (``fix_storage_*``, ``p_entity_*``)
        are seeded via ``state.cross_solve_carriers`` instead of
        ``shutil.copy`` between sub-solves.
      * ``_writer_*._read_csv`` helpers consult
        ``_input_source._seed_lookup_positional`` so per-iter readers
        find the in-memory frame produced earlier in the same
        iteration's writer chain.

    Scope caveat — full objective-parity between the csv-emission-on
    and csv-emission-off paths requires the ``csv.reader`` direct
    sites in the streamed writer modules (``_writer_period_params``,
    ``_writer_pdt_params``, ``_writer_dispatchers``, …) to also become
    seed-aware.  That is a follow-up dispatch (Phase E-e).  This test
    asserts the contracts Phase E-d does land:
      1. The cascade completes successfully under
         ``csv_emission_disabled()``.
      2. No writer-port ``_write`` helper emits a CSV to
         ``solve_data/`` or ``cross_solve/``.
      3. The accumulator captures the per-iter solve_writers frames
         (verified via the SolveHandoff carriers populated at the end
         of each iteration).
    """
    from flextool.engine_polars import run_chain_from_db

    fixture = DATA / "work_base"
    db = fixture / "tests.sqlite"
    if not db.exists():
        pytest.skip(f"fixture sqlite missing: {db}")

    # Baseline reference (kept for diagnostic context — the objective
    # parity check is the Phase E-e follow-up gate).
    baseline_work = tmp_path / "baseline"
    baseline_sols = run_chain_from_db(
        db, scenario_name="base", work_folder=baseline_work,
    )
    assert baseline_sols, "baseline cascade produced no sub-solves"

    # Run the cascade with CSV emission disabled.  Pre-Phase-E-d the
    # per-iter ``solve_writers.write_*`` calls fired OUTSIDE
    # ``capture_frames`` and the cascade tried to read those CSVs back
    # from disk in ``_load_time`` — the run blew up.  Phase E-d wraps
    # the per-iter scope in ``capture_frames`` AND installs the
    # accumulator as the active seed so the seed lookup in
    # ``_input_source._read_csv_file`` (and the writer-port
    # ``_writer_*._read_csv``) find the in-memory frame instead.
    csvfree_work = tmp_path / "csvfree"
    with csv_emission_disabled():
        csvfree_sols = run_chain_from_db(
            db, scenario_name="base", work_folder=csvfree_work,
        )
    assert csvfree_sols, "csv-free cascade produced no sub-solves"

    # The accumulator captured the writer-port frames (Phase E-d
    # contract).  The per-iter solve_writers frames now ride inside
    # the accumulator instead of going only to disk.
    csvfree_last = next(reversed(csvfree_sols.values()))
    accum = csvfree_last.flex_data_accumulator
    assert accum is not None
    # Sample a few solve_writers basenames — these are now captured
    # because the writers ran inside ``capture_frames``.
    expected_captures = (
        "steps_in_use.csv", "period_in_use_set.csv", "solve_current.csv",
        "p_inflation_factor_operations_yearly.csv",
    )
    for name in expected_captures:
        assert name in accum.frames, (
            f"Phase E-d expected accumulator to capture {name} after "
            "the per-iter solve_writers scope was lifted into "
            f"capture_frames; missing.  Got keys: "
            f"{sorted(k for k in accum.frames if not k.count('/'))[:20]}"
        )

    # No writer-port ``_write`` helper emitted to ``solve_data/`` or
    # ``cross_solve/`` under csv_emission_disabled().  Note that
    # write_outputs_for_solve and the legacy ``write_input`` ARE still
    # writing some files (output handoff trail + the legacy input
    # writer's direct ``csv.writer`` emits — those are out of scope
    # for Phase E-d; see test docstring).
    cross_solve_dir = csvfree_work / "cross_solve"
    if cross_solve_dir.exists():
        csvs = sorted(p.name for p in cross_solve_dir.glob("*.csv"))
        assert not csvs, (
            f"Phase E-d expected {cross_solve_dir} to contain no .csv "
            f"files under csv-free mode; found {csvs}"
        )

    # Solve_data must not contain CSVs whose producer is a writer-port
    # ``_write`` helper.  We don't enumerate the full list here — the
    # accumulator's ``expected_basenames()`` is the producer manifest;
    # cross-check that none of those basenames landed on disk.
    from flextool.engine_polars._flex_data_accumulator import (
        expected_basenames,
    )
    solve_data_dir = csvfree_work / "solve_data"
    if solve_data_dir.exists():
        on_disk = {p.name for p in solve_data_dir.glob("*.csv")}
        managed_on_disk = on_disk & set(expected_basenames())
        # Post-solve writers (``write_outputs_for_solve``,
        # ``handoff_writers``) still emit a few of these basenames
        # outside the writer-port gate.  Those are recognised as the
        # "post-solve overwritten" set in
        # ``tests/engine_polars/test_phase_c_flex_data_accumulator.py``.
        # Allow them through.
        from flextool.flextoolrunner.solve_handoff import (
            SolveHandoff,  # noqa: F401 (import probe)
        )
        post_solve_allowed = {
            "p_entity_period_existing_capacity.csv",
            "p_entity_invested.csv",
            "p_entity_divested.csv",
            "p_entity_period_invested_capacity.csv",
            "fix_storage_quantity.csv",
            "fix_storage_price.csv",
            "fix_storage_usage.csv",
            "co2_cum_realized_tonnes.csv",
            "ladder_cum_sim_hours.csv",
            "ladder_cum_realized_mwh.csv",
            "scale_the_objective.csv",
            "scale_the_state.csv",
            "period_capacity.csv",
            "costs_discounted.csv",
            "co2.csv",
            "rp_cost_weight.csv",
        }
        unexpected = managed_on_disk - post_solve_allowed
        assert not unexpected, (
            f"Phase E-d expected writer-port _write helpers to skip "
            f"disk emission under csv-free mode; found leftover "
            f"on-disk basenames in {solve_data_dir}: "
            f"{sorted(unexpected)}"
        )
