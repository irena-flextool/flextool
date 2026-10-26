"""Shared helpers for Tier-7 constraint-emission tests.

The pattern is:

  1. Load a fixture that exercises the target constraint family, through
     the production loader call (:func:`tests._support.load.load_scenario`
     on the cascade's last sub-solve).
  2. Build the LP — but **do not solve**.  Emission tests inspect what
     the model emits, not what HiGHS computes.
  3. Use :meth:`Problem.cstr_row_count` (engine API added in Phase 3) to
     assert the expected row count.

A failing emission test isolates a "constraint family was declared but
the wrong rows / no rows are bound to it" bug.
"""

from __future__ import annotations

from polar_high import Problem
from flextool.engine import build_flextool

from tests._support.load import ScenarioRun, load_scenario


def build(run: ScenarioRun):
    """Rebuild the last sub-solve of a finished cascade run and build the LP — without solving.

    ``run`` comes from the ``scenario_run`` fixture (or
    :func:`tests._support.load.run_scenario`); the FlexData is the one
    production builds (:func:`tests._support.load.load_scenario`).

    Returns ``(Problem, FlexData)``.  Emission tests only need to look
    at what was registered on the ``Problem`` (variables, constraints
    via ``pb.cstr_names()`` / ``pb.cstrs_named()``) and at the input
    sets on ``FlexData`` for cross-checks.
    """
    data = load_scenario(run)
    pb = Problem()
    build_flextool(pb, data)
    return pb, data


def assert_cstr_row_count(pb: Problem, name: str, expected: int) -> None:
    """Assert that the constraint family ``name`` emits exactly
    ``expected`` LP rows.  Reports all known constraint names on
    failure, which makes naming/typo issues immediately obvious."""
    actual = pb.cstr_row_count(name)
    assert actual == expected, (
        f"constraint family {name!r} row count: "
        f"expected {expected}, got {actual}.\n"
        f"All cstr names: {pb.cstr_names()}"
    )


def assert_cstr_present(pb: Problem, name: str) -> None:
    """Assert that the constraint family ``name`` was emitted with at
    least one LP row."""
    actual = pb.cstr_row_count(name)
    assert actual > 0, (
        f"constraint family {name!r} not emitted "
        f"(or emitted with zero rows).\n"
        f"All cstr names: {pb.cstr_names()}"
    )


def assert_cstr_absent(pb: Problem, name: str) -> None:
    """Assert that no constraint family matching ``name`` is emitted.

    Useful for negative checks like "the integer-online uptime variant
    must not be emitted on a linear-only fixture"."""
    actual = pb.cstr_row_count(name)
    assert actual == 0, (
        f"constraint family {name!r} unexpectedly emitted "
        f"({actual} rows).\n"
        f"All cstr names: {pb.cstr_names()}"
    )
