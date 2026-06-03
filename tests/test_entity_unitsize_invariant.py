"""Regression: ``entity_unitsize`` must cover every entity in the
solve-invariant ``p_all_entity_unitsize`` carrier — including invest
candidates that are absent from the (pss-filtered) ``p_unitsize`` /
``p_state_unitsize`` reconstruction.

Before the carrier wiring, ``_build_entity_unitsize_series`` rebuilt the
Series only from the pss-filtered ``p_unitsize`` / ``p_state_unitsize``
plus a densify over the *current solve's* entity universe.  An entity
that was an invest candidate in some sub-solve but absent from the LAST
solve's pss/invest sets was therefore missing from ``entity_unitsize``,
and ``calc_costs`` (which indexes it by ``v.invest.columns`` — the union
of invest columns across ALL sub-solves) raised ``KeyError``.

This test proves an entity present ONLY in ``p_all_entity_unitsize``
(and absent from the filtered params and the densify universe) still
appears in the resulting ``entity_unitsize`` with its carrier value.
Solver-free and fast.
"""
from types import SimpleNamespace

import polars as pl
from polar_high import Param

from flextool.process_outputs.read_parameters import (
    _build_entity_unitsize_series,
)


def _fake_flex_data(*, with_carrier: bool):
    """Minimal ``flex_data`` shape that ``_build_entity_unitsize_series``
    reads: ``p_all_entity_unitsize`` (carrier), ``p_unitsize`` (process
    side, keyed ``p``), ``p_state_unitsize`` (node side, keyed ``n``).

    The carrier carries ``CaboVerde_wind`` (an invest candidate absent
    from the LAST solve), which the filtered params do NOT.
    """
    fd = SimpleNamespace()
    # Filtered per-solve params — DO NOT mention CaboVerde_wind.
    fd.p_unitsize = Param(
        ("p",), pl.DataFrame({"p": ["Aruba_coal"], "value": [500.0]})
    )
    fd.p_state_unitsize = Param(
        ("n",), pl.DataFrame({"n": ["Aruba_elec"], "value": [100.0]})
    )
    if with_carrier:
        # Whole-model carrier — includes the absent invest candidate.
        fd.p_all_entity_unitsize = Param(
            ("e",),
            pl.DataFrame(
                {
                    "e": ["Aruba_coal", "Aruba_elec", "CaboVerde_wind"],
                    "value": [500.0, 100.0, 250.0],
                }
            ),
        )
    else:
        fd.p_all_entity_unitsize = None
    return fd


def test_entity_unitsize_includes_carrier_only_entity():
    """The carrier-only entity must appear with its carrier value, even
    though it is absent from ``p_unitsize`` / ``p_state_unitsize`` and
    from the densify ``entity_universe``."""
    fd = _fake_flex_data(with_carrier=True)
    # An ``entity_universe`` that pointedly EXCLUDES CaboVerde_wind, to
    # prove the carrier — not the densify — is what surfaces it.
    s = _build_entity_unitsize_series(
        fd, entity_universe=["Aruba_coal", "Aruba_elec"]
    )
    assert "CaboVerde_wind" in s.index, (
        "invest candidate present in p_all_entity_unitsize but absent "
        "from filtered params/universe must still appear in entity_unitsize"
    )
    assert s.loc["CaboVerde_wind"] == 250.0
    # Sanity: carrier values for the other entities are intact.
    assert s.loc["Aruba_coal"] == 500.0
    assert s.loc["Aruba_elec"] == 100.0
    assert s.index.name == "entity"
    assert s.name == "entity"


def test_entity_unitsize_fallback_without_carrier():
    """Without the carrier (CSV/fixture path), the reconstruction +
    densify still works and does NOT invent the carrier-only entity."""
    fd = _fake_flex_data(with_carrier=False)
    s = _build_entity_unitsize_series(
        fd, entity_universe=["Aruba_coal", "Aruba_elec", "new_invest"]
    )
    # Reconstructed values present.
    assert s.loc["Aruba_coal"] == 500.0
    assert s.loc["Aruba_elec"] == 100.0
    # Densify default applies to universe entities missing from params.
    assert s.loc["new_invest"] == 1000.0
    # CaboVerde_wind is NOT in the fallback (no carrier, not in universe).
    assert "CaboVerde_wind" not in s.index
    assert s.index.name == "entity"
