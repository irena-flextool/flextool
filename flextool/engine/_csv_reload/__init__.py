"""Reload a FlexTool model from a CSV dump — UNSUPPORTED, opt-in.

The CSV dump writers (``FlexData.dump_csvs``, ``FlexDataProvider.snapshot_processed_inputs``, ``--csv-dump``) are
supported: they write a folder for inspection and stamp it with ``_dump_meta.json``
(:mod:`flextool.engine.preprocess._dump_stamp`).  Nothing in a run reads a dump back.

This package reads one back.  It is **not kept in sync with the engine**: a model reloaded from a dump may be stale
and can differ from the solved model (a dump has no solve-to-solve carry).  Every call warns with
:class:`CsvReloadWarning`, and a dump without a stamp, or with another dump format or database schema, is refused
unless ``force=True`` (:func:`check_dump`).

Dependency direction: this package imports the engine; nothing outside it may import it
(``tests/engine/test_meta_provider_invariants.py::test_csv_reload_is_a_leaf``).  Its tests live in
``tests/csv_reload/`` and run only with ``pytest -m csv_reload``.  Do not update it when you change the engine;
whoever needs the reload brings it up to date first and then updates ``VERIFIED_FLEXTOOL`` / ``VERIFIED_SCHEMA`` in
:mod:`._stamp`.

Public names:

* :func:`load_from_dump` / :func:`load_from_dump_with_provider` — rebuild a :class:`FlexData` from a dump folder,
  optionally with a DB reader (``db_url`` + ``scenario``).
* :func:`copy_runner_meta` — copy the run's metadata CSVs a ``dump_csvs`` folder lacks (prepares it for reload).
* :func:`seed_provider_from_dump` — the provider seeding step on its own.
* :func:`check_dump` — the stamp check on its own.
* :class:`CsvReloadWarning`.

Not re-exported (it imports the output writers): :mod:`.offline_outputs` — write a solve's output parquets from an
MPS + HiGHS solution file and the solve's dump (``python -m flextool.engine._csv_reload.offline_outputs``).
It replaces the removed ``python -m flextool.process_outputs.read_highs_solution``.  It currently needs the solve's
``scale_the_objective`` passed explicitly (``--scale``): a provider snapshot (``--csv-dump``, a run's work folder)
carries a header-only ``solve_data/scale_the_objective.csv``.
"""
from __future__ import annotations

from flextool.engine._csv_reload._meta import copy_runner_meta
from flextool.engine._csv_reload._seed import seed_provider_from_dump
from flextool.engine._csv_reload._stamp import (
    SUPPORTED_DUMP_FORMAT,
    VERIFIED_FLEXTOOL,
    VERIFIED_SCHEMA,
    CsvReloadWarning,
    check_dump,
)
from flextool.engine._csv_reload.load import load_from_dump, load_from_dump_with_provider

__all__ = [
    "SUPPORTED_DUMP_FORMAT",
    "VERIFIED_FLEXTOOL",
    "VERIFIED_SCHEMA",
    "CsvReloadWarning",
    "check_dump",
    "copy_runner_meta",
    "load_from_dump",
    "load_from_dump_with_provider",
    "seed_provider_from_dump",
]
