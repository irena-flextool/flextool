"""Regression: the densify entity universe must cover every entity in
the solve-invariant ``p_all_entity_unitsize`` carrier — including invest
candidates absent from THIS sub-solve's per-step sets.

``read_parameters`` densifies ``entity_lifetime_fixed_cost`` /
``entity_lifetime_fixed_cost_divest`` over ``_build_entity_universe``.
That universe used to be built only from the current sub-solve's
``nodeBalance`` / ``process_source_sink`` / ``ed_invest_set`` /
``ed_divest_set``.  But ``v.invest`` / ``v.divest`` carry the
cross-solve column union: an entity that is invest-eligible in some
OTHER roll (e.g. ``CaboVerde_wind``) is absent from every step's
universe, so the concatenated ``entity_lifetime_fixed_cost`` columns
miss it and ``calc_costs`` (which indexes by ``v.invest.columns``)
raises ``KeyError``.  Sibling of the ``entity_unitsize`` carrier fix
(896263be); see ``test_entity_unitsize_invariant.py``.

This proves an entity present ONLY in the carrier still lands in the
universe.  Solver-free and fast.
"""
from types import SimpleNamespace

import polars as pl
from polar_high import Param

from flextool.process_outputs.read_parameters import _build_entity_universe


def _fake_flex_data(*, with_carrier: bool):
    """Minimal ``flex_data`` shape that ``_build_entity_universe`` reads.

    The per-solve sets mention only ``Aruba_*`` entities; the carrier
    additionally carries ``CaboVerde_wind`` — an invest candidate from a
    different roll that the per-solve sets do NOT see.
    """
    fd = SimpleNamespace()
    fd.nodeBalance = pl.DataFrame({"n": ["Aruba_elec"]})
    fd.process_source_sink = pl.DataFrame(
        {"p": ["Aruba_coal"], "source": ["Aruba_coal_src"], "sink": ["Aruba_elec"]}
    )
    fd.ed_invest_set = pl.DataFrame({"e": ["Aruba_coal"], "d": ["p2020"]})
    fd.ed_divest_set = pl.DataFrame({"e": [], "d": []}, schema={"e": pl.Utf8, "d": pl.Utf8})
    if with_carrier:
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


def test_universe_includes_carrier_only_entity():
    """The carrier-only invest candidate must appear in the universe,
    even though no per-solve set mentions it."""
    universe = _build_entity_universe(_fake_flex_data(with_carrier=True))
    assert "CaboVerde_wind" in universe, (
        "invest candidate present only in p_all_entity_unitsize must still "
        "land in the densify universe so entity_lifetime_fixed_cost covers it"
    )
    # Per-solve sources still contribute, and there are no duplicates.
    assert {"Aruba_elec", "Aruba_coal", "Aruba_coal_src"} <= set(universe)
    assert len(universe) == len(set(universe))


def test_universe_without_carrier_omits_other_roll_entity():
    """Without the carrier (CSV/fixture path), the universe is just the
    per-solve sets and does NOT invent the other-roll entity."""
    universe = _build_entity_universe(_fake_flex_data(with_carrier=False))
    assert "CaboVerde_wind" not in universe
    assert {"Aruba_elec", "Aruba_coal", "Aruba_coal_src"} <= set(universe)
