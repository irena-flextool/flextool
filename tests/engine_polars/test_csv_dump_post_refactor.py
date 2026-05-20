"""Writer → emitter Phase 4 — ``--csv-dump`` round-trip guard.

After Phase 3 the cascade no longer writes any CSV during execution.
Every ``emit_*`` function pushes its frame into a
:class:`FlexDataProvider` via ``_emit(provider, key, df)`` which
dual-registers under both the bare basename and the parent-qualified
key (see :mod:`flextool.engine_polars._emit_provider_io`).  Disk
emission happens only via
:meth:`FlexDataProvider.snapshot_processed_inputs` (invoked by the
``--csv-dump`` CLI flow in
:mod:`flextool.cli.cmd_run_flextool`) and per-solve
:meth:`FlexData.dump_csvs` (orchestrator-side).

This module locks in three invariants that Phase 4 of the refactor
must preserve:

1. **Coverage** — every basename listed in
   :func:`expected_basenames()` materialises in the snapshot.  If a
   Phase 3a/3b call-site migration silently drops a key, this catches
   it.
2. **Dual-key parity (§2.3)** — when ``_emit`` registers a frame under
   both ``foo.csv`` and ``solve_data/foo.csv`` from a single emit
   site, the two files on disk are byte-identical.  (When two
   different parents collide on the same basename, the bare key
   inherits whichever was emitted last; that collision case is
   excluded from this assertion.)
3. **No off-flag leakage** — running the same cascade WITHOUT
   ``csv_dump=True`` must NOT populate the ``solve_data/`` directory
   on disk.  This proves the Phase 3b deletion of the legacy
   ``_write(df, path)`` plumbing left no stray disk-write site live.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.engine_polars import run_chain_from_db  # noqa: E402
from flextool.engine_polars._flex_data_accumulator import (  # noqa: E402
    expected_basenames,
)
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402


@pytest.fixture(scope="module")
def small_db_url(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Materialise the ``tests.json`` fixture to a sqlite DB once per module."""
    db_path = tmp_path_factory.mktemp("db_phase4") / "tests.sqlite"
    url = json_to_db(TESTS_DIR / "fixtures" / "tests.json", db_path)
    migrate_database(url, up_to=40)
    return url


def _run_coal(small_db_url: str, workdir: Path, *, csv_dump: bool):
    """Drive ``run_chain_from_db`` on the ``coal`` scenario inside *workdir*.

    Returns the step mapping so the test can pluck the Provider off the
    last step.  We chdir into *workdir* across the call because the
    cascade's per-solve scratch resolution is cwd-relative.
    """
    cwd0 = os.getcwd()
    os.chdir(workdir)
    try:
        return run_chain_from_db(
            small_db_url, "coal",
            work_folder=workdir,
            csv_dump=csv_dump,
            keep_solutions=True,
        )
    finally:
        os.chdir(cwd0)


def test_snapshot_covers_expected_basenames(
    small_db_url: str, tmp_path: Path,
) -> None:
    """Every basename in :func:`expected_basenames` materialises in the dump.

    This is the key-set coverage invariant: if a Phase 3 migration
    dropped an ``emit_*`` call, the affected basename would be missing
    from the Provider, and therefore from the on-disk snapshot.

    The :class:`FlexDataProvider` strips the ``.csv`` suffix when
    indexing, so the comparison is done over stems.
    """
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    steps = _run_coal(small_db_url, workdir, csv_dump=True)
    last_step = next(reversed(list(steps.values())))
    provider = last_step.flex_data_provider
    assert provider is not None, (
        "keep_solutions=True must retain the Provider on the last step"
    )

    snap = tmp_path / "snap"
    provider.snapshot_processed_inputs(snap)

    # Enumerate every basename written under *snap* — both top-level and
    # nested under ``input/`` or ``solve_data/``.
    produced_basenames = {p.name for p in snap.rglob("*.csv")}

    missing = set(expected_basenames()) - produced_basenames
    assert not missing, (
        f"snapshot_processed_inputs failed to materialise "
        f"{len(missing)} expected basename(s) — Phase 3 may have "
        f"dropped an emit_* call:\n" + "\n".join(sorted(missing))
    )


def test_snapshot_dual_key_byte_parity(
    small_db_url: str, tmp_path: Path,
) -> None:
    """For every basename emitted from a single parent dir, the bare and
    parent-qualified CSVs on disk are byte-identical.

    The :func:`_emit` helper registers each frame under both
    ``provider.put(p.name, df)`` and ``provider.put(str(p), df)`` so
    Phase-3 readers consulting either form get the same data.  The
    snapshot writer must preserve that parity on disk.

    When two distinct parent dirs (e.g. ``input/`` and ``solve_data/``)
    emit the same basename the bare key only retains the last-written
    frame, so this test excludes those collision cases — they're a
    documented intrinsic of the dual-key scheme, not a refactor bug.
    """
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    steps = _run_coal(small_db_url, workdir, csv_dump=True)
    last_step = next(reversed(list(steps.values())))
    provider = last_step.flex_data_provider

    snap = tmp_path / "snap"
    provider.snapshot_processed_inputs(snap)

    # Group every produced CSV by basename so we can detect collisions
    # vs. single-parent basenames.
    nested_by_basename: dict[str, list[Path]] = defaultdict(list)
    bare_by_basename: dict[str, Path] = {}
    for p in snap.rglob("*.csv"):
        rel = p.relative_to(snap)
        if rel.parent == Path("."):
            bare_by_basename[rel.name] = p
        else:
            nested_by_basename[rel.name].append(p)

    # Known-tolerated exception: the per-roll ladder cross-solve
    # accumulator fan-out at ``_native_run_model._fan_out_ladder_accumulators``
    # updates only the BARE Provider key (via a direct
    # ``provider.put(_basename, _frame)``) so the populated cross-roll
    # carrier can be read by ``_commodity_ladder.load_data`` on the
    # next roll, while leaving the empty header-only seed under the
    # ``solve_data/`` qualified key.  Pre-existing pre-Phase-3
    # behaviour (commit f7efe452), preserved by the refactor.  Audit
    # it separately if/when the dual-key semantics there get tightened.
    _DUAL_KEY_EXEMPT = {"ladder_cum_sim_hours.csv"}

    mismatches: list[str] = []
    checked = 0
    for basename, bare_path in bare_by_basename.items():
        if basename in _DUAL_KEY_EXEMPT:
            continue
        nested = nested_by_basename.get(basename, [])
        if len(nested) != 1:
            # Either no parent twin (cascade-side seeded raw input
            # without ``_emit``) or two-plus parents colliding on the
            # same basename — both excluded from the strict-parity
            # assertion per §2.3.
            continue
        bare_bytes = bare_path.read_bytes()
        nested_bytes = nested[0].read_bytes()
        if bare_bytes != nested_bytes:
            mismatches.append(
                f"{basename}: bare vs {nested[0].relative_to(snap)}"
            )
        checked += 1

    assert checked > 0, (
        "expected at least one dual-keyed pair in the snapshot; the "
        "Provider may not be wiring _emit correctly"
    )
    assert not mismatches, (
        f"dual-key byte-parity failed on {len(mismatches)} pair(s):\n"
        + "\n".join(mismatches)
    )


def test_no_csv_dump_means_no_emit_leak_on_disk(
    small_db_url: str, tmp_path: Path,
) -> None:
    """Running without ``csv_dump=True`` must not write any
    ``emit_*``-produced basename onto disk under ``solve_data/``.

    Phase 3b deleted every ``_write(df, path)`` helper from the
    cascade.  This test catches a regression where some emit_* site
    (or a stray helper) still writes to disk in the no-dump path.

    Out of scope: post-solve writers in
    :mod:`flextool.process_outputs.handoff_writers` and the
    ``scale_the_objective.csv`` reader-side stamp in
    :mod:`flextool.engine_polars._orchestration` deliberately drop
    their handoff state to ``solve_data/`` regardless of the flag —
    they're cross-roll carriers, not part of the writer→emitter
    refactor surface listed in ``specs/writer_to_emitter.md §6``.
    Those files are exempted via :data:`_POST_SOLVE_HANDOFF_LEAK_EXEMPT`.
    """
    workdir = tmp_path / "wd_no_dump"
    workdir.mkdir(parents=True, exist_ok=True)
    _run_coal(small_db_url, workdir, csv_dump=False)

    # Files dropped to ``solve_data/`` by the post-solve handoff
    # pipeline.  These predate the writer→emitter refactor and are
    # written by ``flextool.process_outputs.handoff_writers`` /
    # ``flextool.engine_polars._orchestration._dump_scale_the_objective``
    # — never via an ``emit_*`` site — so they're not in scope here.
    _POST_SOLVE_HANDOFF_LEAK_EXEMPT = {
        "fix_storage_price.csv",
        "fix_storage_quantity.csv",
        "fix_storage_usage.csv",
        "p_entity_divested.csv",
        "period_capacity.csv",
        "scale_the_objective.csv",
        "timeline_matching_map.csv",
        "timings.csv",
    }

    solve_data_dir = workdir / "solve_data"
    on_disk = (
        sorted(solve_data_dir.rglob("*.csv")) if solve_data_dir.exists() else []
    )
    emit_basenames = set(expected_basenames())
    leaks = [
        p for p in on_disk
        if p.name in emit_basenames and p.name not in _POST_SOLVE_HANDOFF_LEAK_EXEMPT
    ]
    assert not leaks, (
        f"csv_dump=False must not leak any emit_*-produced CSV to disk "
        f"but found {len(leaks)}:\n"
        + "\n".join(str(p.relative_to(workdir)) for p in leaks[:10])
    )

    # Surface the post-solve handoff files explicitly so a future change
    # there shows up in this test's diff rather than slipping past.
    handoff_on_disk = {p.name for p in on_disk if p.name in _POST_SOLVE_HANDOFF_LEAK_EXEMPT}
    unexpected_handoff_leak = (
        {p.name for p in on_disk} - emit_basenames - _POST_SOLVE_HANDOFF_LEAK_EXEMPT
    )
    assert not unexpected_handoff_leak, (
        f"unrecognised solve_data leak (neither emit_* nor known "
        f"handoff): {sorted(unexpected_handoff_leak)}.  Either add to "
        f"_POST_SOLVE_HANDOFF_LEAK_EXEMPT after auditing the writer "
        f"site, or fix the leak."
    )
    # handoff_on_disk is informational; assert nothing about its
    # content here — that's covered by the post-solve handoff tests.
    _ = handoff_on_disk
