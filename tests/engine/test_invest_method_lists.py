"""The invest/retire method lists against the schema value lists.

``INVEST_NOT_ALLOWED`` / ``DIVEST_NOT_ALLOWED`` (and the group pair) in
``preprocess/params/_existing.py`` are the one definition of which
``invest_method`` values forbid investing / retiring.  Every value of the
schema list is classified by its name: a value that names ``invest`` (or is
``cumulative_limits``) allows investing, one that names ``retire`` (or is
``cumulative_limits``) allows retiring.  ``invest_period_total`` used to
allow retiring and ``retire_period_total`` investing, because the copies
left them out.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from flextool.engine.preprocess.params._existing import (
    DIVEST_NOT_ALLOWED,
    GROUP_DIVEST_NOT_ACTIVE,
    GROUP_INVEST_NOT_ACTIVE,
    INVEST_METHODS_ALL,
    INVEST_NOT_ALLOWED,
)

SCHEMA = Path(__file__).resolve().parents[2] / "flextool" / "schemas" / "spinedb_schema.json"


def _value_list(name: str) -> list[str]:
    lists = json.loads(SCHEMA.read_text())["parameter_value_lists"]
    values = [v for k, v in lists if k == name]
    assert values, name
    return values


def _allows(method: str, word: str) -> bool:
    return method == "cumulative_limits" or word in method.split("_")


@pytest.mark.parametrize("list_name, no_invest, no_divest", [
    ("invest_methods", INVEST_NOT_ALLOWED, DIVEST_NOT_ALLOWED),
    ("group_invest_methods", GROUP_INVEST_NOT_ACTIVE, GROUP_DIVEST_NOT_ACTIVE),
])
def test_method_lists_match_schema(list_name, no_invest, no_divest):
    values = _value_list(list_name)
    assert no_invest <= set(values)
    assert no_divest <= set(values)
    assert no_invest == {m for m in values if not _allows(m, "invest")}
    assert no_divest == {m for m in values if not _allows(m, "retire")}


def test_period_total_methods_are_one_sided():
    assert "retire_period_total" in INVEST_NOT_ALLOWED
    assert "invest_period_total" in DIVEST_NOT_ALLOWED
    assert "invest_period_total" not in INVEST_NOT_ALLOWED
    assert "retire_period_total" not in DIVEST_NOT_ALLOWED


def test_all_methods_is_the_schema_list():
    """``INVEST_METHODS_ALL`` (entityInvest / entityDivest are its complements) is every schema value: a new
    value missing from it would go silently into neither set."""
    assert INVEST_METHODS_ALL == set(_value_list("invest_methods"))
