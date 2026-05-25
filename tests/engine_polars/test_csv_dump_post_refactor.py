"""Writer → emitter Phase 4 — ``--csv-dump`` round-trip guard.

After Phase 3 the cascade no longer writes any CSV during execution.
Every ``emit_*`` function pushes its frame into a
:class:`FlexDataProvider` via ``_emit(provider, key, df)`` which —
since Phase 0a of ``specs/provider_consolidation.md`` —
single-registers under the parent-qualified key only (see
:mod:`flextool.engine_polars._emit_provider_io`).  Consumer queries
against the bare basename are resolved by the Provider's
bidirectional lookup (see :meth:`FlexDataProvider.get`).  Disk
emission happens only via
:meth:`FlexDataProvider.snapshot_processed_inputs` (invoked by the
``--csv-dump`` CLI flow in
:mod:`flextool.cli.cmd_run_flextool`) and per-solve
:meth:`FlexData.dump_csvs` (orchestrator-side).

This module locks in three invariants:

1. **Coverage** — every basename listed in
   :func:`expected_basenames()` materialises in the snapshot.  If a
   call-site migration silently drops a key, this catches it.
2. **Single-key registration (Phase 0a)** — every emitted basename
   appears under its parent-qualified path only; no bare top-level
   duplicate exists.  A regression to dual-key registration (or a
   stray ``provider.put(<bare_name>, df)`` from a producer site) shows
   up here as a duplicate file.
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


def test_snapshot_single_key_invariant(
    small_db_url: str, tmp_path: Path,
) -> None:
    """Every emitted basename appears under exactly one parent dir; no
    bare top-level duplicate.

    Phase 0a of ``specs/provider_consolidation.md`` retired the
    dual-key registration in :func:`_emit`.  Producers now store each
    frame under its parent-qualified key only; the Provider's
    bidirectional lookup resolves bare-form consumer queries to the
    qualified key.

    A regression would manifest as a basename appearing both at the
    snapshot root AND under a parent dir — that's the signal this
    test catches.
    """
    workdir = tmp_path / "wd"
    workdir.mkdir(parents=True, exist_ok=True)
    steps = _run_coal(small_db_url, workdir, csv_dump=True)
    last_step = next(reversed(list(steps.values())))
    provider = last_step.flex_data_provider

    snap = tmp_path / "snap"
    provider.snapshot_processed_inputs(snap)

    bare_by_basename: dict[str, Path] = {}
    nested_by_basename: dict[str, list[Path]] = defaultdict(list)
    for p in snap.rglob("*.csv"):
        rel = p.relative_to(snap)
        if rel.parent == Path("."):
            bare_by_basename[rel.name] = p
        else:
            nested_by_basename[rel.name].append(p)

    duplicates: list[str] = []
    for basename, bare_path in bare_by_basename.items():
        nested = nested_by_basename.get(basename, [])
        if nested:
            duplicates.append(
                f"{basename}: appears at {bare_path.relative_to(snap)} "
                f"AND {[str(p.relative_to(snap)) for p in nested]}"
            )

    assert not duplicates, (
        f"dual-key registration regression — {len(duplicates)} basename(s) "
        f"appear under both bare and parent-qualified paths:\n"
        + "\n".join(duplicates)
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
