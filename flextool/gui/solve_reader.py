"""Read-only helper: enumerate a scenario's solves and flag investment solves.

The Calibrate-investments GUI needs, for a given input DB + scenario, the
ordered list of solves that scenario runs and — for each — whether it is an
*investment solve*.  A solve is an investment solve when its ``invest_periods``
Array parameter is present and non-empty ("Array of periods where investments
are allowed."); an empty or absent ``invest_periods`` marks a dispatch-style
solve.

The scenario's solve list is read from ``model.solves`` (schema: "Sequence of
solves in the model. Array."), the authoritative ordered list of solves a model
runs, resolved under the scenario's alternative stack.  It is intersected with
the ``solve`` entities that actually exist so a dangling name in ``model.solves``
cannot fabricate a checklist row.  This mirrors ``SolveConfig.load_from_db``:
when no ``model.solves`` is defined and exactly one ``solve`` exists, that solve
is auto-wired as the model's single solve.

The DB is opened scenario-filtered and never written to.
"""

from __future__ import annotations

from dataclasses import dataclass

import spinedb_api as api
from spinedb_api import DatabaseMapping

from flextool.engine_polars._db_reader import (
    DictMode,
    get_single_entities,
    params_to_dict,
)


@dataclass(frozen=True)
class SolveInfo:
    """One solve referenced by a scenario.

    Attributes:
        name: The solve entity name.
        has_invest_periods: ``True`` when the solve's ``invest_periods`` Array
            is present and non-empty (an investment solve); ``False`` for a
            dispatch-style solve with an empty or absent ``invest_periods``.
    """

    name: str
    has_invest_periods: bool


def read_scenario_solves(db_url: str, scenario_name: str) -> list[SolveInfo]:
    """Return the solves *scenario_name* runs, flagged by investment status.

    Args:
        db_url: URL of a FlexTool input database (e.g. ``sqlite:///path.sqlite``).
        scenario_name: Name of the scenario to resolve.

    Returns:
        One :class:`SolveInfo` per solve the scenario runs, in the order the
        scenario's ``model.solves`` lists them (dangling names dropped).

    Raises:
        ValueError: If *scenario_name* is not a scenario in the database.
    """
    scen_config = api.filters.scenario_filter.scenario_filter_config(scenario_name)
    with DatabaseMapping(db_url) as db:
        # Existence check before filtering: an unknown scenario would otherwise
        # silently resolve to an empty stack rather than a loud error.
        scenario_names = {s["name"] for s in db.find_scenarios()}
        if scenario_name not in scenario_names:
            raise ValueError(
                f"No scenario named {scenario_name!r} in the database. "
                f"Available scenarios: {sorted(scenario_names)}."
            )

        api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        db.fetch_all("parameter_value")

        # Authoritative ordered solve list per model (Array → list of names).
        model_solves: dict = params_to_dict(
            db=db, cl="model", par="solves", mode=DictMode.DEFAULTDICT
        )
        existing_solves = get_single_entities(db=db, entity_class_name="solve")

        # Auto-wire the single-solve case exactly as SolveConfig.load_from_db.
        if len(model_solves) == 0 and len(existing_solves) == 1:
            model_solves = {"flextool": [existing_solves[0]]}

        # invest_periods Array per solve (Array → list of period names).
        invest_periods: dict = params_to_dict(
            db=db, cl="solve", par="invest_periods", mode=DictMode.DEFAULTDICT
        )

    existing = set(existing_solves)
    result: list[SolveInfo] = []
    seen: set[str] = set()
    for solve_names in model_solves.values():
        for solve_name in solve_names:
            if solve_name in seen or solve_name not in existing:
                continue
            seen.add(solve_name)
            has_invest = bool(invest_periods.get(solve_name))
            result.append(SolveInfo(name=solve_name, has_invest_periods=has_invest))
    return result
