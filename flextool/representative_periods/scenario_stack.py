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


def existing_alternative_names(db_url: str) -> set[str]:
    """Return the set of alternative names already present in the database.

    A read-only helper the GUI uses to de-duplicate a freshly derived RP
    alternative name (append ``_2``, ``_3``, … when the base is taken) so a
    repeated build never silently overwrites an earlier one.

    Args:
        db_url: Spine database URL (e.g. ``'sqlite:///path.sqlite'``).

    Returns:
        The names of every alternative in the (unfiltered) database.
    """
    with DatabaseMapping(db_url) as db:
        return {alt["name"] for alt in db.get_alternative_items()}


def dedup_alternative_name(base: str, taken: set[str]) -> str:
    """Return *base*, or the first free ``base_2`` / ``base_3`` / … suffix.

    Pure helper (no DB access) so the GUI can compute the de-duplicated name
    against a cached name set both for the live CLI preview and for the launch,
    keeping the two in lock-step.

    Args:
        base: The desired alternative name.
        taken: Names already in use (case-sensitive).

    Returns:
        *base* if free, else ``f"{base}_{n}"`` for the smallest ``n >= 2`` that
        is not in *taken*.
    """
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


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


def create_scenario_with_alternative(
    db_url: str,
    base_scenario_name: str,
    new_scenario_name: str,
    alternative_name: str,
) -> None:
    """Clone *base_scenario_name* into *new_scenario_name* + append *alternative_name*.

    Creates *new_scenario_name* as a copy of *base_scenario_name*'s ordered
    alternative stack (same alternatives, same ranks) and then appends
    *alternative_name* at ``max(rank) + 1`` so the RP alternative overrides the
    baseline ``period_timeset``. The base scenario is left completely
    untouched, so the original and the representative-period runs can be
    compared side by side.

    Idempotent: if *new_scenario_name* already exists it is not recreated — the
    function only ensures *alternative_name* is present at the top of its stack
    (mirroring :func:`add_alternative_to_scenario`), so a repeated launch is a
    no-op rather than an error.

    Args:
        db_url: Spine database URL (e.g. ``'sqlite:///path.sqlite'``).
        base_scenario_name: Scenario to clone. Must exist.
        new_scenario_name: Name of the scenario to create / extend.
        alternative_name: Alternative to append on top. Must exist.

    Raises:
        ValueError: If the base scenario or the alternative does not exist.
    """
    with DatabaseMapping(db_url) as db:
        if not db.get_scenario_item(name=base_scenario_name):
            raise ValueError(
                f"Scenario '{base_scenario_name}' does not exist in the database."
            )
        if not db.get_alternative_item(name=alternative_name):
            raise ValueError(
                f"Alternative '{alternative_name}' does not exist in the database."
            )

        # Create the new scenario the first time, cloning the base stack. On a
        # repeat launch the scenario already exists, so only the top-rank append
        # below runs (idempotent).
        changed = False
        if not db.get_scenario_item(name=new_scenario_name):
            db.add_scenario(name=new_scenario_name)
            for sa in db.get_scenario_alternative_items(
                scenario_name=base_scenario_name
            ):
                db.add_scenario_alternative(
                    scenario_name=new_scenario_name,
                    alternative_name=sa["alternative_name"],
                    rank=sa["rank"],
                )
            changed = True

        existing = db.get_scenario_alternative_items(
            scenario_name=new_scenario_name
        )
        if not any(
            sa["alternative_name"] == alternative_name for sa in existing
        ):
            next_rank = max((sa["rank"] for sa in existing), default=0) + 1
            db.add_scenario_alternative(
                scenario_name=new_scenario_name,
                alternative_name=alternative_name,
                rank=next_rank,
            )
            changed = True

        # Nothing to persist on a fully idempotent repeat call — committing an
        # unchanged session raises ``NothingToCommit``.
        if changed:
            db.commit_session(
                f"Create scenario '{new_scenario_name}' from "
                f"'{base_scenario_name}' with alternative '{alternative_name}'"
            )


__all__ = [
    "add_alternative_to_scenario",
    "create_scenario_with_alternative",
    "dedup_alternative_name",
    "existing_alternative_names",
]
