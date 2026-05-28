"""Tests for the v52 database migration (multi-solver dispatch, Phase 1).

History note (v56 test-cleanup, 2026-05)
----------------------------------------
The v52 migration originally added three solver-related value lists
(``solvers``, ``solver_io_apis``, ``solver_log_levels``) and seven
solver-selection parameter definitions on the ``solve`` entity class.
v56 subsequently **removed** four of those parameters in favour of CLI
flags:

- ``solver_io_api``    removed by b48cdaf4
- ``solver_time_limit`` removed by 7e96e9e6
- ``solver_log_level`` removed by 6198b056
- ``solver_threads``   removed by 6b98f9fc

Additionally the test fixtures (``tests/fixtures/*.json``) were
re-exported at DB v56 — there is no longer a pre-v52 fixture, so the
original ``migrate_database(url, up_to=52)`` staging is a no-op
(``FLEXTOOL_DB_VERSION`` reached in the import; ``next_version`` already
exceeds 52).  The original tests asserted v52 post-migration state
that v56 then dismantled, and the staging mechanism that exercised
them in isolation no longer functions.

The migration helpers themselves are unchanged and still correctly
handle older user databases when encountered in the wild.  The
remaining test below pins the version-constant contract, which is the
one piece of the v52 surface that survives unchanged through v56.
"""
from __future__ import annotations

from flextool.update_flextool import FLEXTOOL_DB_VERSION


def test_v52_version_constant_is_at_least_52():
    """The engine must report a schema version >= 52 — the multi-solver
    Phase 1 lower bound.  Later phases (storage_binding_method Phase 1
    bumped to 53, Phase 2 to 54, ...) keep raising the constant; an
    exact equality assertion would regress every time the chain grows.
    """
    assert FLEXTOOL_DB_VERSION >= 52
