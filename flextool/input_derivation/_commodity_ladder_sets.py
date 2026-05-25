"""Commodity-ladder filtered subsets of ``commodity``.

Three single-dimension sets selected by ``p_commodity_price_method``:

    set commodity_with_ladder            := {c : price_method[c] != 'price'};
    set commodity_with_ladder_annual     := {c : price_method[c] == 'price_ladder_annual'};
    set commodity_with_ladder_cumulative := {c : price_method[c] == 'price_ladder_cumulative'};

``p_commodity_price_method`` has ``default 'price'``, so commodities
not present in the price-method CSV are treated as ``'price'`` and
therefore excluded from all three sets.

Output frames (canonical CSV column = ``commodity``):

* ``solve_data/commodity_with_ladder``
* ``solve_data/commodity_with_ladder_annual``
* ``solve_data/commodity_with_ladder_cumulative``

The frames land under the ``solve_data/<name>`` Provider key, matching
the ``solve_data/<file>.csv`` paths that consumers
(``_emit_mid_sets``, ``_emit_per_solve``, ``model.py``, the
``_commodity_ladder`` cascade loader) resolve through the
Provider-aware ``_read_csv`` helper.
"""
from __future__ import annotations

import polars as pl


def derive_commodity_ladder_sets(backend, provider) -> None:
    """Run the commodity-ladder-sets derivation.

    Reads ``commodity.price_method`` via the
    :class:`SpineDBBackend` and emits three Provider frames.
    """
    # Read price_method per commodity directly from the Backend.
    price_methods: dict[str, str] = {}
    for pv in backend.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_method",
    ):
        if pv["type"] is None:
            continue
        price_methods[pv["entity_byname"][0]] = str(pv["parsed_value"])

    with_ladder = list(dict.fromkeys(
        c for c, m in price_methods.items() if m != "price"
    ))
    with_annual = list(dict.fromkeys(
        c for c, m in price_methods.items() if m == "price_ladder_annual"
    ))
    with_cum = list(dict.fromkeys(
        c for c, m in price_methods.items() if m == "price_ladder_cumulative"
    ))

    for key, rows in (
        ("solve_data/commodity_with_ladder", with_ladder),
        ("solve_data/commodity_with_ladder_annual", with_annual),
        ("solve_data/commodity_with_ladder_cumulative", with_cum),
    ):
        provider.put(
            key,
            pl.DataFrame(
                {"commodity": rows},
                schema={"commodity": pl.Utf8},
            ),
        )


__all__ = ["derive_commodity_ladder_sets"]
