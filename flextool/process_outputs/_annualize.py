"""Shared dt→d annualisation helper for the output-processing stage.

A single place that performs the *extensive* per-step → per-period→annual
aggregation so that no individual output site drifts from the cost-weighting
convention.

The LP objective annualises every per-timestep cost term with the
representative weight ``p_timestep_weight[d, t]`` (1.0 for a full / evenly
sampled timeline, a per-step share of the year for a representative timeset
with ``timeset_weights``).  Annual *energy / extensive* outputs must use the
**same** weight so that the energy the model is scaled to serve, reports, and
costs all coincide.  When ``w ≡ 1`` (the common full-timeline case) the result
is byte-identical to the unweighted formula.
"""

from __future__ import annotations

import pandas as pd


def annualize_dt_to_d(
    frame: pd.DataFrame,
    timestep_weight: pd.Series,
    complete_period_share_of_year: pd.Series,
    step_duration: "pd.Series | None" = None,
    *,
    div_level: "int | None" = None,
) -> pd.DataFrame:
    """Weight a per-``(d, t)`` extensive frame and annualise it to per-period.

    Steps (matching the objective's annualisation):

    1. If ``step_duration`` is given (the frame holds power, MW), multiply by
       it to obtain energy per step (MWh).  Inflow-type frames are already
       MWh/step — pass ``step_duration=None``.
    2. Multiply by ``timestep_weight`` (the per-``(d, t)`` representative weight,
       1.0 when no/uniform ``timeset_weights``).
    3. Sum over the ``period`` level.
    4. Divide by ``complete_period_share_of_year`` (the uniform annualiser
       ``Σ_t step_duration / 8760``).

    ``timestep_weight`` and ``step_duration`` share the identical
    ``(period, time)`` row index (both stripped of the ``solve`` level in
    ``drop_levels``), so ``.mul(..., axis=0)`` broadcasts the same way.

    ``div_level`` mirrors the per-site ``.div(..., axis=0, level=...)``
    convention: pass ``level=1`` where the caller's frame has a multi-level
    row index whose period sits at level 1 (e.g. ``out_node`` slacks).
    """
    f = frame if step_duration is None else frame.mul(step_duration, axis=0)
    f = f.mul(timestep_weight, axis=0)
    summed = f.groupby(level="period").sum()
    if div_level is None:
        return summed.div(complete_period_share_of_year, axis=0)
    return summed.div(complete_period_share_of_year, axis=0, level=div_level)
