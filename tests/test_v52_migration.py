"""Tests for the v52 database migration (multi-solver dispatch, Phase 1).

The v52 migration adds three new value lists (``solvers``,
``solver_io_apis``, ``solver_log_levels``) and seven solver-selection
parameter definitions on the ``solve`` entity class.  See
``flextool/update_flextool/db_migration.py:_migrate_v52_solver_selection``
for the schema rationale.

Test approach
-------------
Each test imports the JSON fixture ``stochastics.json`` (exported at
DB v25) into a fresh SQLite file and runs the full migration chain.
The chain naturally goes through every preceding migration step
before landing on v52, so these tests double as a chain-integrity
check for the newly-bumped ``FLEXTOOL_DB_VERSION = 52``.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
from spinedb_api import DatabaseMapping, from_database

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from db_utils import json_to_db  # noqa: E402

from flextool.update_flextool import FLEXTOOL_DB_VERSION  # noqa: E402
from flextool.update_flextool.db_migration import migrate_database  # noqa: E402

FIXTURE = TEST_DIR / "fixtures" / "stochastics.json"

EXPECTED_SOLVERS = {"highs", "gurobi", "cplex", "xpress", "copt"}
EXPECTED_IO_APIS = {"direct", "mps", "lp"}
EXPECTED_LOG_LEVELS = {"silent", "normal", "verbose"}

# (parameter_name, default_value, value_list_name_or_None)
EXPECTED_PARAMS: list[tuple[str, object, str | None]] = [
    ("solver", "highs", "solvers"),
    ("solver_io_api", "direct", "solver_io_apis"),
    ("solver_options", None, None),
    ("solver_time_limit", None, None),
    ("solver_mip_gap", None, None),
    ("solver_threads", None, None),
    ("solver_log_level", "normal", "solver_log_levels"),
]


@pytest.fixture
def migrated_db_url(tmp_path: Path) -> str:
    """Fresh DB migrated through v52."""
    db_path = tmp_path / "v52.sqlite"
    url = json_to_db(FIXTURE, db_path)
    migrate_database(url)
    return url


def _value_list_members(db: DatabaseMapping, name: str) -> set[str]:
    """Return the set of (parsed) string members of a value list."""
    vls = list(db.find_parameter_value_lists(name=name))
    assert vls, f"value list {name!r} not found"
    vl = vls[0]
    members: set[str] = set()
    for value in vl.get("parsed_value_list", []) or []:
        members.add(value)
    if members:
        return members
    # Fallback: walk the value rows directly when parsed_value_list isn't
    # populated by the mapped item (older spinedb_api versions).
    sq = db.parameter_value_list_sq
    list_rows = db.query(sq).filter(sq.c.name == name).all()
    if not list_rows:
        return members
    list_id = list_rows[0].id
    vsq = db.list_value_sq
    for row in db.query(vsq).filter(vsq.c.parameter_value_list_id == list_id).all():
        members.add(from_database(row.value, row.type))
    return members


def _find_solve_param(db: DatabaseMapping, name: str):
    """Locate the ``solve`` parameter_definition by name."""
    defs = list(
        db.find_parameter_definitions(entity_class_name="solve", name=name)
    )
    return defs[0] if defs else None


def test_v52_version_constant_is_52():
    """Phase 1 hard requirement: the engine reports schema version 52."""
    assert FLEXTOOL_DB_VERSION == 52


def test_v52_value_lists_present(migrated_db_url: str):
    """All three solver-related value lists exist with the expected members."""
    with DatabaseMapping(migrated_db_url) as db:
        assert _value_list_members(db, "solvers") == EXPECTED_SOLVERS
        assert _value_list_members(db, "solver_io_apis") == EXPECTED_IO_APIS
        assert _value_list_members(db, "solver_log_levels") == EXPECTED_LOG_LEVELS


def test_v52_solve_parameter_defs_present(migrated_db_url: str):
    """All seven solver-selection parameters exist on ``solve`` with
    expected defaults and value-list bindings."""
    with DatabaseMapping(migrated_db_url) as db:
        for name, expected_default, expected_vl in EXPECTED_PARAMS:
            pdef = _find_solve_param(db, name)
            assert pdef is not None, f"solve.{name} parameter_definition missing"

            default = from_database(pdef["default_value"], pdef["default_type"])
            assert default == expected_default, (
                f"solve.{name} default {default!r} != expected "
                f"{expected_default!r}"
            )

            actual_vl = pdef.get("parameter_value_list_name")
            assert actual_vl == expected_vl, (
                f"solve.{name} value list {actual_vl!r} != expected "
                f"{expected_vl!r}"
            )


def test_v52_solver_param_group_attachment(migrated_db_url: str):
    """All seven solver-selection parameters live under ``solve_advanced``."""
    with DatabaseMapping(migrated_db_url) as db:
        for name, _, _ in EXPECTED_PARAMS:
            pdef = _find_solve_param(db, name)
            assert pdef is not None, f"solve.{name} missing"
            group = pdef.get("parameter_group_name")
            assert group == "solve_advanced", (
                f"solve.{name} attached to {group!r}, expected "
                f"'solve_advanced'"
            )


def test_v52_idempotent(tmp_path: Path):
    """Re-running ``migrate_database`` on a fully-migrated DB is a no-op."""
    db_path = tmp_path / "v52_idempotent.sqlite"
    url = json_to_db(FIXTURE, db_path)
    migrate_database(url)
    # Second invocation must succeed; the function should detect the DB
    # already at FLEXTOOL_DB_VERSION and short-circuit without raising.
    migrate_database(url)
    # And the schema state must still be the expected one.
    with DatabaseMapping(url) as db:
        assert _value_list_members(db, "solvers") == EXPECTED_SOLVERS
        for name, _, _ in EXPECTED_PARAMS:
            assert _find_solve_param(db, name) is not None
