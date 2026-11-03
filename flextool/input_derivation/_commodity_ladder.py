"""Commodity ladder input-derivation — cumulative + annual tabular emission.

Two ladder writers:

* :func:`derive_commodity_ladder_cumulative`
* :func:`derive_commodity_ladder_annual`

Both emit one canonical frame to the :class:`FlexDataProvider`:

* ``input/commodity_ladder_cumulative``  ``(commodity, tier, price, quantity)``
* ``input/commodity_ladder_annual``      ``(commodity, period, tier, price, quantity)``

The annual derivation auto-detects 2d vs 3d Spine maps and expands 2d
maps across the model period list (sourced from the Provider's
``input/periods_available`` key — populated by the
``_PARAMETER_SPECS`` loop — with DB fallbacks for setups that supply
periods only through ``solve.period_timeset``).

Tier labels are kept as written (text: ``01``, ``1`` and ``a`` are three tiers).  A tier without a price gets a
null price here; the engine's ladder reader (``_direct._ladder_split``) reports it.
"""
from __future__ import annotations

import logging

import polars as pl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tier_sort_key(t: str) -> tuple[int, str]:
    """Stable sort by integer tier when possible, else by string."""
    try:
        return (0, f"{int(t):020d}")
    except ValueError:
        return (1, t)


def tier_label_read_differently_before(label: str) -> bool:
    """Whether a tier label was read as another tier before labels were kept as written: earlier releases
    read a label as a whole number (``01`` became ``1``) and dropped labels that were not whole numbers."""
    try:
        return str(int(label)) != label
    except ValueError:
        return True


def _notice_relabelled_tiers(commodity: str, labels) -> None:
    if any(tier_label_read_differently_before(str(t)) for t in labels):
        from flextool.engine import _behaviour_changes
        _behaviour_changes.notice("ladder_tier_labels_as_written", "commodities", [commodity])


def _quantity_sentinel(quantity: str) -> str:
    """Convert user-facing infinite quantities into the 1e30 sentinel
    interpreted as the unbounded tail tier
    (see ``ladder_tier_cap_infinite_cum`` / ``_ann``).
    """
    try:
        q_float = float(quantity)
    except ValueError:
        q_float = float("inf")
    if q_float == float("inf") or q_float >= 1e30:
        return "1e30"
    return quantity


def _iter_flat_ladder_rows(api, value, commodity: str,
                            logger: logging.Logger) -> list[list]:
    """Flatten a Spine map ladder value.  Returns the raw list-of-lists
    from ``convert_map_to_table`` or an empty list on failure.
    """
    try:
        return api.convert_map_to_table(value)
    except Exception as exc:
        logger.warning(
            "Could not flatten ladder for commodity '%s': %s",
            commodity, exc,
        )
        return []


def _collect_periods(backend, provider, api) -> tuple[list[str], bool]:
    """Return the model's period list (for 1d-map → per-period expansion
    of ``price_ladder_annual``), and whether it came only from the solve
    periods of the Rule 1 / 5b default (fallback 3).

    Step 2.5-F Phase E
    ------------------

    Provider-first: the ``input/periods_available`` key is populated by
    :meth:`SpineDBBackend.parameter_values` during the
    ``_PARAMETER_SPECS`` pass for ``model.periods_available``.  DB
    fallbacks (in order) cover the case where the parameter is set as a
    raw map or where periods come exclusively from
    ``solve.period_timeset`` -- or, for a solve without one, from its
    invest / realized periods (the Rule 1 / 5b default).
    """
    periods: list[str] = []
    seen: set[str] = set()

    # Provider arm — frame layout: "model,period_from_model" (the legacy
    # input/periods_available.csv schema).  Take the rightmost column.
    if provider is not None and provider.has("input/periods_available"):
        df = provider.get("input/periods_available")
        if df is not None and df.height > 0:
            col = df.columns[-1]
            for value in df.get_column(col).to_list():
                v = str(value).strip()
                if v and v not in seen:
                    periods.append(v)
                    seen.add(v)
    if periods:
        return periods, False

    # Fallback 1: raw map on model.periods_available.
    for pv in backend.find_parameter_values(
        entity_class_name="model",
        parameter_definition_name="periods_available",
    ):
        if pv["type"] is None:
            continue
        val = pv["parsed_value"]
        try:
            flat = api.convert_map_to_table(val)
        except Exception:
            flat = []
        for entry in flat:
            for c in (str(x) for x in entry):
                if c and c not in seen:
                    periods.append(c)
                    seen.add(c)
    if periods:
        return periods, False

    # Fallback 2: scan solve.period_timeset map indexes.
    for pv in backend.find_parameter_values(
        entity_class_name="solve",
        parameter_definition_name="period_timeset",
    ):
        if pv["type"] is None:
            continue
        val = pv["parsed_value"]
        try:
            flat = api.convert_map_to_table(val)
        except Exception:
            flat = []
        for entry in flat:
            if len(entry) >= 1:
                p = str(entry[0])
                if p and p not in seen:
                    periods.append(p)
                    seen.add(p)

    # Fallback 3: the periods of the solves without ``period_timeset`` that
    # the Rule 1 / 5b default of ``create_assumptive_parts`` maps to a
    # timeset (item #175).  Imported here so that importing this module does
    # not initialise the ``flextool.engine`` package.
    from flextool.engine.input._timeline_rules import solve_period_timesets, time_structure_from_backend

    authored_only = len(periods)
    time_structure = time_structure_from_backend(backend)
    for solve in time_structure.solves:
        if solve in time_structure.period_timeset:
            continue
        for p, _ts in solve_period_timesets(solve, time_structure) or []:
            if p not in seen:
                periods.append(p)
                seen.add(p)
    return periods, authored_only == 0 and len(periods) > 0


# ---------------------------------------------------------------------------
# Phase D — commodity_ladder_cumulative
# ---------------------------------------------------------------------------


def derive_commodity_ladder_cumulative(
    backend, provider, logger: logging.Logger,
) -> None:
    """Run the cumulative-ladder derivation.

    Emits the Provider frame ``input/commodity_ladder_cumulative`` with
    columns ``(commodity, tier, price, quantity)`` — one row per
    (commodity, tier).

    Only the ``commodity.price_ladder_cumulative`` parameter is
    consulted (always a 2d map: ``Map(tier -> {price, quantity})`` —
    2d in Spine's counting because ``{price, quantity}`` is a second
    index layer).  The ``price_method`` filter happens mod-side via
    the ``commodity_with_ladder_cumulative`` set.
    """
    api = backend.api
    rows: list[tuple[str, str, str | None, str]] = []

    for pv in backend.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_cumulative",
    ):
        if pv["type"] is None:
            continue
        if pv["type"] != "map":
            logger.warning(
                "commodity.price_ladder_cumulative on '%s' has type %s "
                "(expected nested 1d map); skipping.",
                pv["entity_byname"][0], pv["type"],
            )
            continue
        commodity = pv["entity_byname"][0]
        flat = _iter_flat_ladder_rows(api, pv["parsed_value"], commodity, logger)
        per_tier: dict[str, dict[str, str]] = {}
        for entry in flat:
            # Expected layout: [tier_idx, facet, value] (length 3).
            if len(entry) < 3:
                continue
            tier_str = str(entry[0])
            facet = str(entry[1])
            val = entry[-1]
            per_tier.setdefault(tier_str, {})[facet] = str(val)

        _notice_relabelled_tiers(commodity, per_tier)
        for tier_str in sorted(per_tier.keys(), key=_tier_sort_key):
            facets = per_tier[tier_str]
            quantity = _quantity_sentinel(facets.get("quantity", "inf"))
            rows.append((commodity, tier_str, facets.get("price"), quantity))

    # All-Utf8 frame to match the legacy CSV's textual schema (consumers
    # read these through the writer-port ``_read_csv`` helper which
    # expects string columns).
    frame = pl.DataFrame(
        {
            "commodity": [r[0] for r in rows],
            "tier": [r[1] for r in rows],
            "price": [r[2] for r in rows],
            "quantity": [r[3] for r in rows],
        },
        schema={
            "commodity": pl.Utf8,
            "tier": pl.Utf8,
            "price": pl.Utf8,
            "quantity": pl.Utf8,
        },
    )
    provider.put("input/commodity_ladder_cumulative", frame)


# ---------------------------------------------------------------------------
# Phase E — commodity_ladder_annual (2d / 3d auto-detect + period expand)
# ---------------------------------------------------------------------------


def derive_commodity_ladder_annual(
    backend, provider, logger: logging.Logger,
) -> None:
    """Run the annual-ladder derivation.

    Emits the Provider frame ``input/commodity_ladder_annual`` with
    columns ``(commodity, period, tier, price, quantity)`` — one row
    per (commodity, period, tier).

    Reads ``commodity.price_ladder_annual``.  Auto-detects the map depth:

    * 2d form (Spine 2d_map): ``Map(tier -> {price, quantity})`` — the
      same (price, quantity) is expanded across every model period.
    * 3d form (Spine 3d_map): ``Map(period -> Map(tier -> {price,
      quantity}))`` — per-period rows are kept as-is.
    """
    api = backend.api
    rows: list[tuple[str, str, str, str | None, str]] = []
    periods_cache: list[str] | None = None
    periods_from_solves = False

    for pv in backend.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_annual",
    ):
        if pv["type"] is None:
            continue
        if pv["type"] != "map":
            logger.warning(
                "commodity.price_ladder_annual on '%s' has type %s "
                "(expected nested map); skipping.",
                pv["entity_byname"][0], pv["type"],
            )
            continue
        commodity = pv["entity_byname"][0]
        flat = _iter_flat_ladder_rows(api, pv["parsed_value"], commodity, logger)
        if not flat:
            continue

        # Depth detection via the flat table row length.  Spine's Map
        # dimension count matches the flat row length: 2d_map yields
        # length-3 rows, 3d_map yields length-4 rows.
        #   2d_map: [tier, facet, value]                      → len 3
        #   3d_map: [period, tier, facet, value]              → len 4
        max_len = max((len(row) for row in flat), default=0)
        if max_len == 3:
            # 2d_map → expand across all model periods.
            per_tier: dict[str, dict[str, str]] = {}
            for entry in flat:
                if len(entry) < 3:
                    continue
                tier_str = str(entry[0])
                facet = str(entry[1])
                val = entry[-1]
                per_tier.setdefault(tier_str, {})[facet] = str(val)
            _notice_relabelled_tiers(commodity, per_tier)
            if periods_cache is None:
                periods_cache, periods_from_solves = _collect_periods(backend, provider, api)
            if periods_from_solves:
                # Before, these periods were not found and the ladder was skipped.
                from flextool.engine import _behaviour_changes
                _behaviour_changes.notice("price_ladder_periods_from_solve_periods", "commodities", [commodity])
            if not periods_cache:
                logger.warning(
                    "commodity.price_ladder_annual on '%s' is 2d_map but "
                    "no model periods were available for expansion; "
                    "skipping.", commodity,
                )
                continue
            for period in periods_cache:
                for tier_str in sorted(per_tier.keys(), key=_tier_sort_key):
                    facets = per_tier[tier_str]
                    quantity = _quantity_sentinel(facets.get("quantity", "inf"))
                    rows.append((commodity, period, tier_str, facets.get("price"), quantity))
        elif max_len >= 4:
            # 3d_map → per-period.  Flat row layout [period, tier,
            # facet, value] — Spine nests Map(period -> Map(tier ->
            # {price, quantity})).
            per_period_tier: dict[tuple[str, str], dict[str, str]] = {}
            for entry in flat:
                if len(entry) < 4:
                    continue
                period = str(entry[0])
                tier_str = str(entry[1])
                facet = str(entry[2])
                val = entry[-1]
                per_period_tier.setdefault(
                    (period, tier_str), {}
                )[facet] = str(val)

            _notice_relabelled_tiers(commodity, {t for _p, t in per_period_tier})

            def _sort_key(k: tuple[str, str]) -> tuple:
                return (k[0], _tier_sort_key(k[1]))

            for (period, tier_str) in sorted(
                per_period_tier.keys(), key=_sort_key,
            ):
                facets = per_period_tier[(period, tier_str)]
                quantity = _quantity_sentinel(facets.get("quantity", "inf"))
                rows.append((commodity, period, tier_str, facets.get("price"), quantity))
        else:
            logger.warning(
                "commodity.price_ladder_annual on '%s' has unexpected "
                "flattened shape (max row length %d); skipping.",
                commodity, max_len,
            )
            continue

    frame = pl.DataFrame(
        {
            "commodity": [r[0] for r in rows],
            "period": [r[1] for r in rows],
            "tier": [r[2] for r in rows],
            "price": [r[3] for r in rows],
            "quantity": [r[4] for r in rows],
        },
        schema={
            "commodity": pl.Utf8,
            "period": pl.Utf8,
            "tier": pl.Utf8,
            "price": pl.Utf8,
            "quantity": pl.Utf8,
        },
    )
    provider.put("input/commodity_ladder_annual", frame)


__all__ = [
    "derive_commodity_ladder_cumulative",
    "derive_commodity_ladder_annual",
]
