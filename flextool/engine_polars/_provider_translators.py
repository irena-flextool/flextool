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

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine_polars import _provider_keys as K

if TYPE_CHECKING:
    from flextool.engine_polars._flex_data_provider import FlexDataProvider
    from flextool.engine_polars._solve_handoff import SolveHandoff


# Schemas for each handoff carrier's empty-frame fallback.  Column names
# match what consumers iter_rows(named=True) keys; Utf8 dtype matches
# the canonical CSV-roundtrip shape (consumers cast to float on read).
_HANDOFF_EMPTY_SCHEMAS: dict[str, tuple[str, ...]] = {
    K.HANDOFF_REALIZED_INVEST:    ("entity", "period", "value"),
    K.HANDOFF_REALIZED_EXISTING:  ("entity", "period", "value"),
    K.HANDOFF_DIVEST_CUMULATIVE:  ("entity", "value"),
    K.HANDOFF_ROLL_END_STATE:     ("node", "value"),
    K.HANDOFF_CUMULATIVE_CO2:     ("group", "period", "value"),
    K.HANDOFF_CUMULATIVE_COMMODITY: ("commodity", "tier", "period",
                                      "p_ladder_cum_realized_mwh"),
    K.HANDOFF_CUM_SIM_HOURS:      ("period", "p_ladder_cum_sim_hours"),
    K.HANDOFF_FIX_STORAGE_QUANTITY: ("node", "period", "step", "p_fix_storage_quantity"),
    K.HANDOFF_FIX_STORAGE_PRICE:    ("node", "period", "step", "p_fix_storage_price"),
    K.HANDOFF_FIX_STORAGE_USAGE:    ("node", "period", "step", "p_fix_storage_usage"),
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
        ("roll_end_state",      K.HANDOFF_ROLL_END_STATE),
        ("cumulative_co2",      K.HANDOFF_CUMULATIVE_CO2),
        ("cumulative_commodity", K.HANDOFF_CUMULATIVE_COMMODITY),
        ("cum_sim_hours",       K.HANDOFF_CUM_SIM_HOURS),
        ("fix_storage_quantity", K.HANDOFF_FIX_STORAGE_QUANTITY),
        ("fix_storage_price",    K.HANDOFF_FIX_STORAGE_PRICE),
        ("fix_storage_usage",    K.HANDOFF_FIX_STORAGE_USAGE),
    )
    for field_name, key in field_to_key:
        frame = getattr(handoff, field_name, None) if handoff is not None else None
        if frame is None:
            frame = _empty_handoff_frame(key)
        provider.put(key, frame)


# Mapping from user-facing handoff Provider keys to their override
# siblings.  Phase 5 of specs/provider_consolidation.md: external code
# supplies overrides keyed by ``K.HANDOFF_X`` (the user-facing identity);
# ``translate_overrides_to_provider`` writes each frame under the
# corresponding ``K.OVERRIDE_X`` key, and ``read_handoff_frame`` checks
# the override slot before falling back to the natural handoff carrier.
# Only the existing handoff carriers are overridable.
_HANDOFF_TO_OVERRIDE: dict[str, str] = {
    K.HANDOFF_REALIZED_INVEST:      K.OVERRIDE_REALIZED_INVEST,
    K.HANDOFF_REALIZED_EXISTING:    K.OVERRIDE_REALIZED_EXISTING,
    K.HANDOFF_DIVEST_CUMULATIVE:    K.OVERRIDE_DIVEST_CUMULATIVE,
    K.HANDOFF_ROLL_END_STATE:       K.OVERRIDE_ROLL_END_STATE,
    K.HANDOFF_CUMULATIVE_CO2:       K.OVERRIDE_CUMULATIVE_CO2,
    K.HANDOFF_CUMULATIVE_COMMODITY: K.OVERRIDE_CUMULATIVE_COMMODITY,
    K.HANDOFF_CUM_SIM_HOURS:        K.OVERRIDE_CUM_SIM_HOURS,
    K.HANDOFF_FIX_STORAGE_QUANTITY: K.OVERRIDE_FIX_STORAGE_QUANTITY,
    K.HANDOFF_FIX_STORAGE_PRICE:    K.OVERRIDE_FIX_STORAGE_PRICE,
    K.HANDOFF_FIX_STORAGE_USAGE:    K.OVERRIDE_FIX_STORAGE_USAGE,
}


def translate_overrides_to_provider(
    overrides: "dict[str, pl.DataFrame] | None", provider,
) -> None:
    """Write external overrides under ``override/<field>`` Provider keys.

    Phase 5 of ``specs/provider_consolidation.md``.  External code provides
    overrides keyed by ``K.HANDOFF_X`` constants (the user-facing identity);
    this translator writes each frame under the corresponding ``K.OVERRIDE_X``
    Provider key.  Existing :func:`read_handoff_frame` consumers automatically
    pick up overrides because the helper checks ``override/<field>`` before
    falling back to ``handoff/<field>``.

    When *overrides* is ``None`` or empty, no writes are made — the override
    layer simply isn't present.

    Raises
    ------
    ValueError
        If *overrides* contains a key that is not one of the whitelisted
        ``K.HANDOFF_*`` constants.  Arbitrary Provider-key writes through
        this translator are rejected so the override surface stays
        explicit.
    """
    if not overrides:
        return
    for handoff_key, frame in overrides.items():
        override_key = _HANDOFF_TO_OVERRIDE.get(handoff_key)
        if override_key is None:
            raise ValueError(
                f"translate_overrides_to_provider: key {handoff_key!r} is not "
                f"a whitelisted handoff carrier; only K.HANDOFF_* keys are "
                f"overridable.  Allowed keys: "
                f"{sorted(_HANDOFF_TO_OVERRIDE.keys())!r}"
            )
        # Phase 6a — tag every override write so the audit dump can
        # distinguish externally-supplied keys from natural-cascade
        # writes.  The override key itself is the discriminator at the
        # dump site, so a single stable tag is sufficient.
        provider.put(override_key, frame, source="external_override")


def read_handoff_frame(provider, key: str) -> "pl.DataFrame | None":
    """Return a populated handoff frame from the Provider, or ``None``.

    Phase 5: checks ``override/<field>`` first, falls back to
    ``handoff/<field>``.  External overrides written via
    :func:`translate_overrides_to_provider` shadow the natural handoff
    carrier; consumers stay key-stable.

    The translator writes an empty header-only frame when the
    corresponding handoff field is ``None``; this helper collapses the
    ``height == 0`` empty back to ``None`` so consumers can preserve
    their existing ``if frame is not None`` guard pattern.  The same
    ``height == 0 → None`` semantics apply to the override slot: an
    empty override frame is treated as "no override present" and we
    fall through to the handoff carrier.
    """
    if provider is None:
        return None
    override_key = _HANDOFF_TO_OVERRIDE.get(key)
    if override_key is not None:
        frame = provider.get(override_key)
        if frame is not None and frame.height > 0:
            return frame
    frame = provider.get(key)
    if frame is None or frame.height == 0:
        return None
    return frame


def dump_provider_sources(
    provider: "FlexDataProvider",
    dump_path: "str | Path",
    solve_name: str,
) -> None:
    """Append every source-tagged Provider key to *dump_path*.

    Phase 6b of ``specs/provider_consolidation.md``.  Iterates the
    Provider's keys; for each key whose :meth:`get_source` returns a
    non-``None`` tag, appends one tab-separated line in the format::

        <solve_name>\\t<key>\\t<source>\\n

    The file is opened in append mode so multiple sub-solves
    accumulate in a single log.  The parent directory is created if
    missing.

    Untagged (natural-cascade) keys are skipped — the dump is a
    minimal audit trail of writes whose origin differs from the
    default preprocessing chain, not a snapshot of the full Provider.
    """
    path = Path(dump_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted iteration keeps the log deterministic across runs so the
    # audit trail diffs cleanly between cascades that differ only in
    # dict-iteration order.
    with path.open("a", encoding="utf-8") as fh:
        for key in sorted(provider.keys()):
            source = provider.get_source(key)
            if source is None:
                continue
            fh.write(f"{solve_name}\t{key}\t{source}\n")


__all__ = [
    "translate_handoff_to_provider",
    "translate_overrides_to_provider",
    "read_handoff_frame",
    "dump_provider_sources",
]
