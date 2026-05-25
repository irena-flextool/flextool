"""Diagnostic for the CO2 rolling-handoff carrier through ``run_chain``.

Mirrors flextool's ``test_co2_rolling_handoff.py`` (which unit-tests
flextool's own ``write_co2_rolling_accumulators`` writer): exercises
the same CARRIER — the rolling cumulative-CO2 accumulator passed
between solves — but through polar_high's ``run_chain`` +
``build_handoff_from_solution`` instead of flextool's CSV writer.

polar_high's :class:`SolveHandoff` exposes the field ``cumulative_co2``
for this carrier.  The current polar_high build pipeline does NOT
populate it: no shipped fixture uses ``co2_method=total`` or
``co2_method=period_total`` (every shipped fixture either uses
``co2_method=period`` or has no CO2 group at all), and
``build_handoff_from_solution`` leaves the slot at ``None``.

This file pins down those facts as a regression guard:

* Multi-solve scenarios produce ``handoff.cumulative_co2 is None``
  at every sub-solve.
* The on-disk ``co2_cum_realized_tonnes.csv`` snapshot is header-
  only (no rolling CO2 group active) and is not clobbered by
  ``run_chain``.

When polar_high gains support for cumulative CO2 (Phase-2 work
documented in ``audit/handoff_full_parity_gaps.md``), this test
file is the right place to extend with the value-level checks
flextool's unit tests cover (uniform-split realized-MWh
attribution, prior-accumulator carryover, lookahead exclusion,
removal/sink-side subtraction).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars import run_chain_from_db

pytestmark = pytest.mark.solver


# Scenarios chained through both ``co2_method=period`` (5weeks) and
# no-CO2-group (multi_year, 4solve) paths.
SCENARIOS = (
    "5weeks_invest_fullYear_dispatch_coal_wind",  # has group__co2_method
    "wind_battery_invest_lifetime_renew_4solve",   # no CO2 group
    "multi_year",                                  # no CO2 group
)


def _co2_csv_text(work: Path, sub_solve: str) -> str:
    p = work / f"solve_data_{sub_solve}" / "co2_cum_realized_tonnes.csv"
    return p.read_text() if p.exists() else ""


def test_chain_cumulative_co2_handoff_diagnostic(scenario_workdir) -> None:
    """Three-part diagnostic on the CO2 rolling carrier across every
    available multi-solve fixture:

    1. ``handoff.cumulative_co2`` is ``None`` at every sub-solve
       (polar_high does not yet model ``co2_method=total``).
    2. The on-disk ``co2_cum_realized_tonnes.csv`` snapshot is header-
       only — mirrors flextool's
       ``test_writer_no_co2_groups_emits_header_only`` invariant.
    3. ``run_chain`` does not write that file (no byte drift between
       two consecutive runs).

    Becomes meaningful when a fixture with ``co2_method=total`` lands;
    parts (1) and (2) will then need updating to assert the carrier's
    value-level correctness.
    """
    for scenario_name in SCENARIOS:
        work = scenario_workdir(scenario_name)
        db_path = work / "tests.sqlite"
        sols = run_chain_from_db(db_path, scenario_name=scenario_name)
        assert sols, f"{scenario_name}: chain produced no sub-solves"

        # (1) Carrier is None — no fixture uses cumulative CO2 method.
        for sub_solve, step in sols.items():
            assert step.handoff.cumulative_co2 is None, (
                f"{scenario_name}/{sub_solve}: cumulative_co2 unexpectedly "
                f"populated ({step.handoff.cumulative_co2}). polar_high "
                f"doesn't model rolling cumulative CO2 yet — see "
                f"audit/handoff_full_parity_gaps.md §B3."
            )

        # (2) Snapshot file is header-only or absent.
        for sub_solve in sols:
            text = _co2_csv_text(work, sub_solve)
            if not text:
                continue
            lines = [ln for ln in text.strip().split("\n") if ln.strip()]
            assert len(lines) <= 1, (
                f"{scenario_name}/{sub_solve}: co2_cum_realized_tonnes.csv has "
                f"{len(lines)} lines, expected header-only:\n{text}"
            )

        # (3) No-clobber: re-running the chain doesn't alter the file.
        pre = {sub: _co2_csv_text(work, sub) for sub in sols}
        run_chain_from_db(db_path, scenario_name=scenario_name)
        post = {sub: _co2_csv_text(work, sub) for sub in sols}
        for sub in sols:
            assert pre[sub] == post[sub], (
                f"{scenario_name}/{sub}: co2_cum_realized_tonnes.csv drifted "
                f"across two run_chain_from_db invocations — polar_high must "
                f"not write this file."
            )
