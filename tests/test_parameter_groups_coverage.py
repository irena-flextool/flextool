"""Regression tests for parameter_group coverage in the master template.

These tests guard the v44 parameter-group rollout (see
``rivendell/PROPOSAL_parameter_groups.md``).  They are defensive:
future parameter additions that forget to set a ``parameter_group_name``
should fail the coverage test here rather than silently slip into the
template.

The export 6-tuple for a parameter_definition is::

    [entity_class, name, default_value, default_type,
     description, parameter_group_name]

i.e. ``pd[5]`` is the group slot.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from flextool._resources import package_data_path

MASTER_TEMPLATE = package_data_path("schemas/spinedb_schema.json")

_GROUP_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


@pytest.fixture(scope="module")
def template() -> dict:
    with open(MASTER_TEMPLATE, encoding="utf-8") as f:
        return json.load(f)


def test_every_parameter_definition_has_group(template: dict) -> None:
    """Every parameter_definition row must have a non-null sixth slot
    (``parameter_group_name``).  New parameters added without a group
    assignment would silently slip through the migration otherwise."""
    missing: list[tuple[str, str]] = []
    for pd in template["parameter_definitions"]:
        # pd: [entity_class, name, default_value, default_type,
        #      description, parameter_group_name]
        if len(pd) < 6 or pd[5] is None or pd[5] == "":
            missing.append((pd[0], pd[1]))
    assert not missing, (
        f"{len(missing)} parameter_definition(s) lack a parameter_group_name: "
        f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
    )


def test_every_referenced_group_is_defined(template: dict) -> None:
    """Every ``parameter_group_name`` referenced by a parameter_definition
    must appear as a ``parameter_groups`` entry.  Protects against a
    typo'd group name creating a dangling reference."""
    defined_groups = {g[0] for g in template.get("parameter_groups", [])}
    referenced_groups = {
        pd[5]
        for pd in template["parameter_definitions"]
        if len(pd) >= 6 and pd[5] is not None
    }
    dangling = referenced_groups - defined_groups
    assert not dangling, (
        f"parameter_definition(s) reference undefined groups: {sorted(dangling)}; "
        f"defined groups are {sorted(defined_groups)}"
    )


def test_group_names_are_lowercase_snake_case(template: dict) -> None:
    """All parameter_group names must be lowercase snake_case.  Matches
    the FlexTool naming convention and blocks accidental reintroduction
    of ``Outputs``-style capitalisation."""
    bad = [g[0] for g in template.get("parameter_groups", [])
           if not _GROUP_NAME_RE.match(g[0])]
    assert not bad, (
        f"parameter_group names must match ^[a-z][a-z0-9_]*$; "
        f"offenders: {bad}"
    )


def test_fifteen_groups_present(template: dict) -> None:
    """Sanity check that the full v44 group scheme landed.  This is a
    smoke test — a full-coverage check is elsewhere — but a mismatch
    here is usually a sign that something reverted the migration."""
    expected = {
        "basics", "investment", "retirement", "storage", "tech_advanced",
        "reserve", "emission", "network", "flow_limit", "constraint",
        "model", "solve_basics", "solve_advanced", "timeline", "output",
    }
    actual = {g[0] for g in template.get("parameter_groups", [])}
    assert actual == expected, (
        f"parameter_groups mismatch.  Missing: {expected - actual}; "
        f"Unexpected: {actual - expected}"
    )
