"""Translators that fan typed orchestrator carriers into Provider entries.

Phase 2 of ``specs/provider_consolidation.md`` retires the
``prior_handoff`` parameter threading through preprocessing.  Instead of
passing the typed :class:`SolveHandoff` dataclass alongside the Provider
into every preprocessing call site, the orchestrator translates each
consumed handoff field into a dedicated Provider key
(``handoff/<field>.csv``) at iteration start.  Preprocessing consumers
then read via ``provider.get(K.HANDOFF_X)`` — one read interface, no
parameter threading.

The translator writes an **empty header-only frame** when a handoff
field is ``None`` so consumers can use a single check
(``provider.get(K.X).height > 0``) to distinguish "prior carrier
populated" from "no prior carrier".  This matches the
``_read_csv``-style empty-frame contract used elsewhere in the cascade.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from flextool.engine_polars import _provider_keys as K

if TYPE_CHECKING:
    from flextool.engine_polars._solve_handoff import SolveHandoff


# Schemas for each handoff carrier's empty-frame fallback.  Column names
# match what consumers iter_rows(named=True) keys; Utf8 dtype matches
# the canonical CSV-roundtrip shape (consumers cast to float on read).
_HANDOFF_EMPTY_SCHEMAS: dict[str, tuple[str, ...]] = {
    K.HANDOFF_REALIZED_INVEST:    ("entity", "period", "value"),
    K.HANDOFF_REALIZED_EXISTING:  ("entity", "period", "value"),
    K.HANDOFF_DIVEST_CUMULATIVE:  ("entity", "value"),
    K.HANDOFF_CUMULATIVE_CO2:     ("group", "period", "value"),
    K.HANDOFF_CUMULATIVE_COMMODITY: ("commodity", "tier", "period",
                                      "p_ladder_cum_realized_mwh"),
    K.HANDOFF_CUM_SIM_HOURS:      ("period", "p_ladder_cum_sim_hours"),
}


def _empty_handoff_frame(key: str) -> pl.DataFrame:
    cols = _HANDOFF_EMPTY_SCHEMAS[key]
    return pl.DataFrame(
        {c: [] for c in cols},
        schema={c: pl.Utf8 for c in cols},
    )


def translate_handoff_to_provider(
    handoff: "SolveHandoff | None", provider,
) -> None:
    """Write each consumed ``SolveHandoff`` field as a Provider entry.

    Called at iteration start *before* preprocessing runs.  When
    *handoff* is ``None`` (first sub-solve, or no prior carrier
    available) every key receives an empty header-only frame so
    consumers can read unconditionally and check ``height > 0`` to
    decide whether prior-carrier data is available.

    The translated keys are listed in
    ``flextool.engine_polars._provider_keys`` under the ``HANDOFF_*``
    constants.  See :data:`_HANDOFF_EMPTY_SCHEMAS` for column shapes.
    """
    field_to_key = (
        ("realized_invest",     K.HANDOFF_REALIZED_INVEST),
        ("realized_existing",   K.HANDOFF_REALIZED_EXISTING),
        ("divest_cumulative",   K.HANDOFF_DIVEST_CUMULATIVE),
        ("cumulative_co2",      K.HANDOFF_CUMULATIVE_CO2),
        ("cumulative_commodity", K.HANDOFF_CUMULATIVE_COMMODITY),
        ("cum_sim_hours",       K.HANDOFF_CUM_SIM_HOURS),
    )
    for field_name, key in field_to_key:
        frame = getattr(handoff, field_name, None) if handoff is not None else None
        if frame is None:
            frame = _empty_handoff_frame(key)
        provider.put(key, frame)


def read_handoff_frame(provider, key: str) -> "pl.DataFrame | None":
    """Return a populated handoff frame from the Provider, or ``None``.

    The translator writes an empty header-only frame when the
    corresponding handoff field is ``None``; this helper collapses the
    ``height == 0`` empty back to ``None`` so consumers can preserve
    their existing ``if frame is not None`` guard pattern.
    """
    if provider is None:
        return None
    frame = provider.get(key)
    if frame is None or frame.height == 0:
        return None
    return frame


__all__ = ["translate_handoff_to_provider", "read_handoff_frame"]
