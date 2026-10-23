"""Shared numeric constants for the polars engine.

FlexTool timelines use arbitrary indices (not datetime), so "a year" has
no calendar definition the engine can read off the timestamps.
``HOURS_PER_YEAR`` is the single canonical hours->years unit used for
*every* period-share / annualisation conversion in the engine
(``period_share_of_year``, ``timeline_duration_in_years``, the
``scale_to_annual_flow`` denominator, peak annualisation, ...).  Keeping
it in one place means the serve/report/cost annualisations stay mutually
consistent, and a future per-period override only has to replace one
symbol rather than ~two dozen scattered literals.
"""
from __future__ import annotations

# Hours in a reference year (365 * 24).  See module docstring.
HOURS_PER_YEAR = 8760.0
