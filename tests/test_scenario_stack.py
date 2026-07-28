"""Tests for :func:`add_alternative_to_scenario`.

Appending an alternative onto a scenario's ordered alternative stack at the
TOP rank (highest = applied last, so it overrides the baseline). The fixture
is built from the FlexTool JSON schema (CLAUDE.md invariant #3 — never read a
checked-in ``.sqlite``).
"""

from __future__ import annotations

import pytest
from spinedb_api import DatabaseMapping, import_data

from flextool._resources import package_data_path
from flextool.representative_periods.scenario_stack import (
    add_alternative_to_scenario,
)
from flextool.update_flextool import initialize_database

SCENARIO = "stack_scen"
BASE_ALT = "base_alt"
MID_ALT = "mid_alt"
RP_ALT = "rp_alt"


def _build_db(db_path: str) -> str:
    """Schema-complete DB: one scenario with two ranked alternatives plus a
    separate RP-style alternative NOT yet in the stack."""
    initialize_database(
        str(package_data_path("schemas/spinedb_schema.json")), db_path
    )
    url = f"sqlite:///{db_path}"
    with DatabaseMapping(url) as db:
        _, errors = import_data(
            db,
            alternatives=[
                (BASE_ALT, "baseline"),
                (MID_ALT, "override layer"),
                (RP_ALT, "representative periods"),
            ],
            scenarios=[(SCENARIO, True, "stack scenario")],
            # base_alt at rank 1, mid_alt at rank 2 (mid overrides base).
            scenario_alternatives=[
                (SCENARIO, BASE_ALT, MID_ALT),
                (SCENARIO, MID_ALT),
            ],
        )
        if errors:
            raise RuntimeError(f"fixture import errors: {errors[:5]}")
        db.commit_session("scenario stack fixture")
    return url


def _stack(url: str, scenario: str) -> list[str]:
    """Return the scenario's alternative names ordered by rank (ascending)."""
    with DatabaseMapping(url) as db:
        items = db.get_scenario_alternative_items(scenario_name=scenario)
        return [sa["alternative_name"] for sa in sorted(items, key=lambda x: x["rank"])]


def _ranks(url: str, scenario: str) -> list[int]:
    with DatabaseMapping(url) as db:
        items = db.get_scenario_alternative_items(scenario_name=scenario)
        return sorted(sa["rank"] for sa in items)


def test_appends_rp_alt_at_top(tmp_path):
    url = _build_db(str(tmp_path / "append.sqlite"))
    assert _stack(url, SCENARIO) == [BASE_ALT, MID_ALT]

    add_alternative_to_scenario(url, SCENARIO, RP_ALT)

    # RP alt lands at the TOP (highest rank / last in ascending order); the
    # pre-existing alternatives keep their relative order.
    assert _stack(url, SCENARIO) == [BASE_ALT, MID_ALT, RP_ALT]
    # Ranks stay contiguous 1..N.
    assert _ranks(url, SCENARIO) == [1, 2, 3]


def test_idempotent(tmp_path):
    url = _build_db(str(tmp_path / "idempotent.sqlite"))
    add_alternative_to_scenario(url, SCENARIO, RP_ALT)
    stack_after_first = _stack(url, SCENARIO)
    ranks_after_first = _ranks(url, SCENARIO)

    # Second call: no duplicate, no rank churn.
    add_alternative_to_scenario(url, SCENARIO, RP_ALT)
    assert _stack(url, SCENARIO) == stack_after_first
    assert _ranks(url, SCENARIO) == ranks_after_first
    assert stack_after_first.count(RP_ALT) == 1

    # An alternative already present but NOT at the top is also left untouched.
    add_alternative_to_scenario(url, SCENARIO, BASE_ALT)
    assert _stack(url, SCENARIO) == stack_after_first


def test_missing_scenario_raises(tmp_path):
    url = _build_db(str(tmp_path / "missing_scen.sqlite"))
    with pytest.raises(ValueError, match="no_such_scenario"):
        add_alternative_to_scenario(url, "no_such_scenario", RP_ALT)


def test_missing_alternative_raises(tmp_path):
    url = _build_db(str(tmp_path / "missing_alt.sqlite"))
    with pytest.raises(ValueError, match="no_such_alt"):
        add_alternative_to_scenario(url, SCENARIO, "no_such_alt")
