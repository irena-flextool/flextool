"""Diagnostic for handoff-writer parity through ``run_chain_from_db``.

Mirrors flextool's ``test_handoff_writers.py`` — which unit-tests
flextool's CSV writers (``write_p_entity_divested``,
``write_p_roll_continue_state``, ``write_fix_storage_*``) AND
contains an end-to-end integration test asserting that those
writers' output matches flextool's phase-3 reference, byte-for-byte
after row-sort + 6-decimal rounding.

The polar_high analogue lives in :func:`flextool.engine.preprocess.flex_data.build_handoff_from_solution`
+ :func:`flextool.engine.preprocess.flex_data.apply_handoff` — the in-memory equivalents
of those CSV writers.  Parity is asserted indirectly by
``test_flex_chain_apply_handoff.py`` (objective-level parity end-
to-end).  This test goes one layer deeper: it walks the chain via
the native cascade and inspects the in-memory handoff carriers
polar_high emits at each sub-solve, comparing against flextool's
committed reference snapshots in ``solve_data_<sub>/``.

Δ.12e — migrated from the legacy ``run_chain(work)`` driver to the
native ``run_chain_from_db`` cascade.

Carriers exercised end-to-end:

* ``realized_cohort`` / ``realized_retire`` — one row per cohort (and per retirement period).
* ``divest_cumulative`` — populated only on fixtures whose
  ``entityDivest.csv`` is non-empty.
* ``roll_end_state`` — populated only on fixtures with nodeState +
  per-roll boundary.
* ``fix_storage`` (quantity) — populated only on fixtures with
  ``storage_nested_fix_method=fix_quantity``.

This test verifies that every populated carrier yields a unique-keyed frame.
"""
from __future__ import annotations

import pytest

from flextool.engine import run_chain_from_db

pytestmark = pytest.mark.solver


SCENARIO_NAME = "wind_battery_invest_lifetime_renew_4solve"


def test_chain_handoff_carriers_are_unique_keyed(scenario_workdir) -> None:
    """The in-memory handoff produced after each sub-solve carries unique-keyed frames: ``realized_cohort`` one
    row per (entity, cohort), ``realized_retire`` one per (entity, cohort, period), ``divest_cumulative`` one per
    entity.  The csv dump writes these carriers themselves (``produced_handoff/``); no writer mirrors them."""
    work = scenario_workdir(SCENARIO_NAME)
    db_path = work / "tests.sqlite"

    sols = run_chain_from_db(db_path, scenario_name=SCENARIO_NAME)
    chain_order = list(sols)
    assert len(chain_order) >= 2, (
        f"need ≥2 sub-solves for handoff parity; got {chain_order}")
    assert any(sols[sub].handoff.realized_cohort is not None for sub in chain_order), (
        "fixture invariant: the chain realizes investments")
    for sub in chain_order:
        h = sols[sub].handoff
        for name, df, keys in (("realized_cohort", h.realized_cohort, ["entity", "cohort"]),
                               ("realized_retire", h.realized_retire, ["entity", "cohort", "period"]),
                               ("divest_cumulative", h.divest_cumulative, ["entity"])):
            if df is not None:
                assert df.unique(keys).height == df.height, f"{sub}: {name} has duplicate {keys} rows"
