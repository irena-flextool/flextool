"""Append an alternative onto a scenario's alternative stack.

A FlexTool scenario is defined by an *ordered* list of alternatives via the
``scenario_alternative`` relationship: each link carries a ``rank`` (integer,
higher = applied later, so a higher-rank alternative overrides the values of
lower-rank ones).  The representative-periods pre-processor writes its results
into a fresh RP alternative, but that alternative only influences a solve once
it is part of the scenario's alternative stack.

:func:`add_alternative_to_scenario` appends the RP alternative at the TOP of a
scenario's stack (the highest rank, so it wins over the baseline
``period_timeset``), computing the next rank from the scenario's existing max.
It mirrors the ``DatabaseMapping`` open/commit style of the RP writer
(``preprocess.py::_write_results_to_db``) and is idempotent — an alternative
already in the stack is left untouched.
"""

from __future__ import annotations

from spinedb_api import DatabaseMapping


def add_alternative_to_scenario(
    db_url: str,
    scenario_name: str,
    alternative_name: str,
) -> None:
    """Append *alternative_name* to *scenario_name*'s stack at the top rank.

    The alternative is added with ``rank = max(existing ranks) + 1`` so it is
    applied last and overrides the alternatives already in the scenario.

    Idempotent: if *alternative_name* is already in the scenario's stack, the
    function does nothing (no duplicate link, no rank churn, no error).

    Args:
        db_url: Spine database URL (e.g. ``'sqlite:///path.sqlite'``).
        scenario_name: Name of the scenario whose stack to extend. Must exist.
        alternative_name: Name of the alternative to append. Must exist.

    Raises:
        ValueError: If the scenario or the alternative does not exist.
    """
    with DatabaseMapping(db_url) as db:
        # ``get_*_item`` returns an empty dict (falsy) when nothing matches.
        if not db.get_scenario_item(name=scenario_name):
            raise ValueError(f"Scenario '{scenario_name}' does not exist in the database.")
        if not db.get_alternative_item(name=alternative_name):
            raise ValueError(
                f"Alternative '{alternative_name}' does not exist in the database."
            )

        existing = db.get_scenario_alternative_items(scenario_name=scenario_name)
        # Idempotent: already in the stack (at any rank) → leave it untouched.
        if any(sa["alternative_name"] == alternative_name for sa in existing):
            return

        next_rank = max((sa["rank"] for sa in existing), default=0) + 1
        # ``add_scenario_alternative`` returns the added item and raises
        # ``SpineDBAPIError`` on failure — let that surface rather than swallow.
        db.add_scenario_alternative(
            scenario_name=scenario_name,
            alternative_name=alternative_name,
            rank=next_rank,
        )

        db.commit_session(
            f"Append alternative '{alternative_name}' to scenario '{scenario_name}'"
        )


__all__ = ["add_alternative_to_scenario"]
