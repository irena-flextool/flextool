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

from flextool.engine.preprocess import _provider_keys as K

if TYPE_CHECKING:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider
    from flextool.engine.solve._handoff import SolveHandoff


def _cohort_carrier_columns() -> list[str]:
    from flextool.engine.preprocess.params._cohort_handoff import realized_cohort_schema
    return list(realized_cohort_schema())


# Schemas for each handoff carrier's empty-frame fallback.  Column names
# match what consumers iter_rows(named=True) keys; Utf8 dtype matches
# the canonical CSV-roundtrip shape (consumers cast to float on read).
_HANDOFF_EMPTY_SCHEMAS: dict[str, tuple[str, ...]] = {
    K.HANDOFF_REALIZED_COMMISSION: ("entity", "period", "commission_period", "value"),
    K.HANDOFF_DIVEST_CUMULATIVE:  ("entity", "value"),
    K.HANDOFF_REALIZED_COHORT:    tuple(_cohort_carrier_columns()),
    K.HANDOFF_REALIZED_RETIRE:    ("entity", "cohort", "period", "value"),
    K.HANDOFF_ROLL_END_STATE:     ("node", "value"),
    K.HANDOFF_UPWARD_ROLL_END_STATE: ("node", "value"),
    K.HANDOFF_ROLL_END_ONLINE:    ("process", "value"),
    K.HANDOFF_ROLL_END_FLOW:      ("process", "source", "sink", "period", "step", "duration", "value"),
    K.HANDOFF_ROLL_END_STEP:      ("period", "step", "duration",
                                   "end_rank", "end_root", "end_index", "end_branch"),
    K.HANDOFF_ONLINE_HISTORY:     ("process", "period", "step",
                                   "startup", "shutdown", "duration", "hours_to_end"),
    K.HANDOFF_FIX_ONLINE:         ("process", "period", "step", "value"),
    K.HANDOFF_CUMULATIVE_CO2:     ("group", "period", "value", "represented"),
    K.HANDOFF_CUMULATIVE_COMMODITY: ("commodity", "tier", "period",
                                      "p_ladder_cum_realized_mwh"),
    K.HANDOFF_CUM_SIM_HOURS:      ("period", "p_ladder_cum_sim_hours"),
    K.HANDOFF_CUMULATIVE_FLOWGROUP: ("group", "period", "flow", "hours", "flow_represented", "hours_represented"),
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


# The two carrier FLOWS, as partial translates that never clobber each
# other's keys: OC1 rolling separate_dispatch takes a dispatch roll's
# ROLL-FORWARD (realized trajectory) from the previous dispatch roll and its
# PIN (commitment + capacity) from its UC roll; a nested sub-solve takes only
# the PIN from its parent (the roll-forward carriers tied to the realised end
# are routed by solve._carry_routing).  A new carrier (e.g. a realization pin of
# another variable family) adds one row to SolveHandoff._FIELDS, to one of these
# sets, to _FIELD_TO_KEY and to _HANDOFF_EMPTY_SCHEMAS, plus a _provider_keys
# constant, a FlexData field with its loader, and for a pin its constraint (EQ.
# header, CONSTRAINT_FAMILIES entry) and its _reduce_detail field list entry:
# the full recipe is in docs/dev/architecture.md (realization solve carriers).
ROLL_FORWARD_FIELDS: frozenset = frozenset((
    "roll_end_state", "upward_roll_end_state", "roll_end_online", "roll_end_flow",
    "roll_end_step", "online_history", "cumulative_co2", "cumulative_commodity",
    "cum_sim_hours", "cumulative_flowgroup",
))
PIN_FIELDS: frozenset = frozenset((
    "realized_commission", "divest_cumulative", "realized_cohort", "realized_retire",
    "fix_online", "fix_storage_quantity", "fix_storage_price", "fix_storage_usage",
))


# SolveHandoff field → its ``handoff/<field>`` Provider key.
_FIELD_TO_KEY: tuple[tuple[str, str], ...] = (
    ("realized_commission", K.HANDOFF_REALIZED_COMMISSION),
    ("divest_cumulative",   K.HANDOFF_DIVEST_CUMULATIVE),
    ("realized_cohort",     K.HANDOFF_REALIZED_COHORT),
    ("realized_retire",     K.HANDOFF_REALIZED_RETIRE),
    ("roll_end_state",      K.HANDOFF_ROLL_END_STATE),
    ("upward_roll_end_state", K.HANDOFF_UPWARD_ROLL_END_STATE),
    ("roll_end_online",     K.HANDOFF_ROLL_END_ONLINE),
    ("roll_end_flow",       K.HANDOFF_ROLL_END_FLOW),
    ("roll_end_step",       K.HANDOFF_ROLL_END_STEP),
    ("online_history",      K.HANDOFF_ONLINE_HISTORY),
    ("fix_online",          K.HANDOFF_FIX_ONLINE),
    ("cumulative_co2",      K.HANDOFF_CUMULATIVE_CO2),
    ("cumulative_commodity", K.HANDOFF_CUMULATIVE_COMMODITY),
    ("cum_sim_hours",       K.HANDOFF_CUM_SIM_HOURS),
    ("cumulative_flowgroup", K.HANDOFF_CUMULATIVE_FLOWGROUP),
    ("fix_storage_quantity", K.HANDOFF_FIX_STORAGE_QUANTITY),
    ("fix_storage_price",    K.HANDOFF_FIX_STORAGE_PRICE),
    ("fix_storage_usage",    K.HANDOFF_FIX_STORAGE_USAGE),
)


def translate_handoff_to_provider(
    handoff: "SolveHandoff | None", provider,
    fields: "frozenset | set | None" = None,
) -> None:
    """Write each consumed ``SolveHandoff`` field as a Provider entry.

    Called at iteration start *before* preprocessing runs.  When
    *handoff* is ``None`` (first sub-solve, or no prior carrier
    available) every key receives an empty header-only frame so
    consumers can read unconditionally and check ``height > 0`` to
    decide whether prior-carrier data is available.

    ``fields`` (OC1 rolling separate_dispatch): when given, only those
    SolveHandoff field names are written — a PARTIAL translate that leaves
    every other Provider key untouched.  Used to compose a dispatch roll's
    handoff from two sources (pin from its UC roll + roll-forward from the
    previous dispatch roll) without each full translate clobbering the
    other's keys.  ``None`` (default) writes every field (the full translate).

    The translated keys are listed in
    ``flextool.engine.preprocess._provider_keys`` under the ``HANDOFF_*``
    constants.  See :data:`_HANDOFF_EMPTY_SCHEMAS` for column shapes.
    """
    for field_name, key in _FIELD_TO_KEY:
        if fields is not None and field_name not in fields:
            continue
        frame = getattr(handoff, field_name, None) if handoff is not None else None
        if frame is None:
            frame = _empty_handoff_frame(key)
        provider.put(key, frame)


def translate_produced_handoff_to_provider(handoff: "SolveHandoff", provider) -> None:
    """Write the solve's own (produced) ``SolveHandoff`` under the ``produced_handoff/<field>`` keys.

    Called after the solve.  The ``handoff/<field>`` keys keep what the solve consumed (the carriers translated and
    routed at iteration start), so a LP rebuilt from the post-solve Provider, or from its csv dump, is the LP that was
    solved; the produced carriers (the final cumulative totals, the realised end state) are exposed beside them for
    the dump and for inspection.  No cascade consumer reads the produced keys: the next sub-solve takes its carriers
    from ``state.handoffs`` through :func:`translate_handoff_to_provider`.
    """
    for field_name, key in _FIELD_TO_KEY:
        frame = getattr(handoff, field_name, None)
        if frame is None:
            frame = _empty_handoff_frame(key)
        provider.put(K.produced_handoff_key(key), frame)


# Mapping from user-facing handoff Provider keys to their override
# siblings.  Phase 5 of specs/provider_consolidation.md: external code
# supplies overrides keyed by ``K.HANDOFF_X`` (the user-facing identity);
# ``translate_overrides_to_provider`` writes each frame under the
# corresponding ``K.OVERRIDE_X`` key, and ``read_handoff_frame`` checks
# the override slot before falling back to the natural handoff carrier.
# Only the existing handoff carriers are overridable.
_HANDOFF_TO_OVERRIDE: dict[str, str] = {
    K.HANDOFF_DIVEST_CUMULATIVE:    K.OVERRIDE_DIVEST_CUMULATIVE,
    K.HANDOFF_REALIZED_COHORT:      K.OVERRIDE_REALIZED_COHORT,
    K.HANDOFF_REALIZED_RETIRE:      K.OVERRIDE_REALIZED_RETIRE,
    K.HANDOFF_ROLL_END_STATE:       K.OVERRIDE_ROLL_END_STATE,
    K.HANDOFF_UPWARD_ROLL_END_STATE: K.OVERRIDE_UPWARD_ROLL_END_STATE,
    K.HANDOFF_ROLL_END_ONLINE:      K.OVERRIDE_ROLL_END_ONLINE,
    K.HANDOFF_ROLL_END_FLOW:        K.OVERRIDE_ROLL_END_FLOW,
    K.HANDOFF_ROLL_END_STEP:        K.OVERRIDE_ROLL_END_STEP,
    K.HANDOFF_ONLINE_HISTORY:       K.OVERRIDE_ONLINE_HISTORY,
    K.HANDOFF_CUMULATIVE_CO2:       K.OVERRIDE_CUMULATIVE_CO2,
    K.HANDOFF_CUMULATIVE_COMMODITY: K.OVERRIDE_CUMULATIVE_COMMODITY,
    K.HANDOFF_CUM_SIM_HOURS:        K.OVERRIDE_CUM_SIM_HOURS,
    K.HANDOFF_FIX_STORAGE_QUANTITY: K.OVERRIDE_FIX_STORAGE_QUANTITY,
    K.HANDOFF_FIX_STORAGE_PRICE:    K.OVERRIDE_FIX_STORAGE_PRICE,
    K.HANDOFF_FIX_STORAGE_USAGE:    K.OVERRIDE_FIX_STORAGE_USAGE,
}


#: The retired override keys of the realized capacity, and the key that replaces them: an override names each
#: realized investment as a cohort (``realized_cohort``) and each retirement per cohort (``realized_retire``).
_RETIRED_OVERRIDE_KEYS: dict[str, str] = {
    "handoff/realized_invest": K.HANDOFF_REALIZED_COHORT,
    "handoff/realized_existing": K.HANDOFF_REALIZED_COHORT,
    "handoff/realized_divest": K.HANDOFF_REALIZED_RETIRE,
}


#: The value columns of the overridable carriers of units online / start-ups / shut-downs.
_ONLINE_OVERRIDE_COLUMNS: dict[str, tuple[str, ...]] = {
    K.HANDOFF_ROLL_END_ONLINE: ("value",),
    K.HANDOFF_ONLINE_HISTORY: ("startup", "shutdown"),
}


def _snap_override_online(handoff_key: str, frame: "pl.DataFrame | None") -> "pl.DataFrame | None":
    """An override of the carried units online or start-ups / shut-downs with its values near 0 set to 0, as the
    producer sets the values it carries (``flex_data.snap_carried_online``; the consumers' cuts compare with 0).
    An override comes from no LP, so "near" is HiGHS' default primal feasibility tolerance (1e-7 units); its other
    values are kept as given."""
    columns = _ONLINE_OVERRIDE_COLUMNS.get(handoff_key)
    if frame is None or columns is None or not set(columns) <= set(frame.columns):
        return frame
    from polar_high.autoscale._precedence import HIGHS_DEFAULT_FEASIBILITY_TOLERANCE

    from flextool.engine.preprocess.flex_data import snap_carried_online
    return snap_carried_online(frame, False, HIGHS_DEFAULT_FEASIBILITY_TOLERANCE, columns)


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
        If *overrides* contains a retired key (``handoff/realized_invest`` /
        ``handoff/realized_existing`` / ``handoff/realized_divest``; the message names
        the key that replaces it), or a key that is not one of the whitelisted
        ``K.HANDOFF_*`` constants.  Arbitrary Provider-key writes through
        this translator are rejected so the override surface stays
        explicit.
    """
    if not overrides:
        return
    for handoff_key, frame in overrides.items():
        if handoff_key in _RETIRED_OVERRIDE_KEYS:
            raise ValueError(
                f"translate_overrides_to_provider: the override key {handoff_key!r} is retired; give the realized "
                f"capacity as {_RETIRED_OVERRIDE_KEYS[handoff_key]!r} (one row per investment cohort, "
                f"columns {list(_HANDOFF_EMPTY_SCHEMAS[K.HANDOFF_REALIZED_COHORT])}; retirements per cohort in "
                f"{K.HANDOFF_REALIZED_RETIRE!r}, columns {list(_HANDOFF_EMPTY_SCHEMAS[K.HANDOFF_REALIZED_RETIRE])})."
            )
        override_key = _HANDOFF_TO_OVERRIDE.get(handoff_key)
        if override_key is None:
            raise ValueError(
                f"translate_overrides_to_provider: key {handoff_key!r} is not "
                f"a whitelisted handoff carrier; only K.HANDOFF_* keys are "
                f"overridable.  Allowed keys: "
                f"{sorted(_HANDOFF_TO_OVERRIDE.keys())!r}"
            )
        frame = _snap_override_online(handoff_key, frame)
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
    "translate_produced_handoff_to_provider",
    "translate_overrides_to_provider",
    "read_handoff_frame",
    "dump_provider_sources",
]
