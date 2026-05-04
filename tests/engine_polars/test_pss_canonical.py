"""Δ.3 — Parity test for the canonical method-tagged
``process_source_sink`` frame.

The Δ.3 schematic offered two options for the partition family:
  (a-prime) keep ``process_source_sink_eff`` / ``_noEff`` as FlexData
            fields but compute them as projections of one canonical
            method-tagged frame.
  (b)       a single method-tagged frame consumed via ``.filter()``.

Δ.3 picks (a-prime) since the partitions appear in 17+ consumer sites
across model.py / _group_slack.py / _cumulative_invest.py /
_region_filter.py / chain.py / _dump_csvs.py — option (b) would
cascade widely.  The canonical frame is :func:`process_source_sink_canonical`
in ``_projection_params.py``; this test asserts the invariants:

  * ``method ∈ {'eff', 'noEff'}`` for every row.
  * The canonical frame's ``(p, source, sink)`` rows equal the union
    of ``_eff`` ∪ ``_noEff`` rows.
  * The ``method == 'eff'`` filter equals ``process_source_sink_eff``.
  * The ``method == 'noEff'`` filter equals ``process_source_sink_noEff``.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from flextool.engine_polars import SpineDbReader
from flextool.engine_polars._projection_params import (
    process_source_sink,
    process_source_sink_canonical,
    process_source_sink_eff,
    process_source_sink_noEff,
)


DATA = Path(__file__).resolve().parent / "data"


# Fixtures with diverse process_source_sink populations (units +
# connections) to exercise the canonical partition.
CANONICAL_FIXTURES: list[tuple[str, str]] = [
    ("work_base", "base"),
    ("work_coal", "coal"),
    ("work_dr_decrease_demand", "dr_decrease_demand"),
    ("work_capacity_margin", "capacity_margin"),
    ("work_lh2_three_region", "lh2_three_region"),
    ("work_network_wind_coal_battery_fullYear_invest",
     "network_wind_coal_battery_fullYear_invest"),
]


@pytest.mark.parametrize(
    "work_name,scenario",
    CANONICAL_FIXTURES,
    ids=[f"{w}::{s}" for w, s in CANONICAL_FIXTURES],
)
def test_canonical_partitions_match_eff_and_noEff(work_name: str,
                                                      scenario: str) -> None:
    """``process_source_sink_canonical`` filtered by ``method`` equals
    the projected ``_eff`` / ``_noEff`` helpers.  Plus the canonical
    frame's ``(p, source, sink)`` rows equal ``pss``."""
    work = DATA / work_name
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip(f"{sqlite} not present")

    reader = SpineDbReader(f"sqlite:///{sqlite}", scenario)

    pss = process_source_sink(reader)
    canonical = process_source_sink_canonical(reader, pss)
    eff = process_source_sink_eff(reader, pss)
    no_eff = process_source_sink_noEff(reader, pss)

    # 1. Schema invariant.
    assert set(canonical.columns) == {"p", "source", "sink", "method"}, \
        f"canonical schema: {canonical.columns}"

    # 2. method ∈ {'eff', 'noEff'} for every row.
    methods = set(canonical["method"].unique().to_list()) if canonical.height > 0 else set()
    assert methods <= {"eff", "noEff"}, f"unexpected methods: {methods}"

    # 3. Union invariant: canonical's (p,s,sink) rows == pss rows.
    canonical_keys = (canonical
        .select("p", "source", "sink")
        .unique()
        .sort("p", "source", "sink"))
    pss_sorted = pss.sort("p", "source", "sink")
    assert canonical_keys.equals(pss_sorted), \
        "canonical (p,source,sink) doesn't match pss"

    # 4. .filter(method=='eff') == process_source_sink_eff
    eff_from_canonical = (canonical
        .filter(pl.col("method") == "eff")
        .select("p", "source", "sink")
        .sort("p", "source", "sink"))
    assert eff_from_canonical.equals(eff.sort("p", "source", "sink")), \
        "filter(method='eff') mismatches process_source_sink_eff"

    # 5. .filter(method=='noEff') == process_source_sink_noEff
    no_eff_from_canonical = (canonical
        .filter(pl.col("method") == "noEff")
        .select("p", "source", "sink")
        .sort("p", "source", "sink"))
    assert no_eff_from_canonical.equals(no_eff.sort("p", "source", "sink")), \
        "filter(method='noEff') mismatches process_source_sink_noEff"

    # 6. eff and noEff are disjoint (sanity).
    if eff.height > 0 and no_eff.height > 0:
        overlap = (eff
            .join(no_eff, on=["p", "source", "sink"], how="inner"))
        assert overlap.height == 0, "_eff and _noEff overlap"


def test_canonical_empty_when_no_processes() -> None:
    """Empty ``pss`` → empty canonical frame with the right schema."""
    work = DATA / "work_base"
    sqlite = work / "tests.sqlite"
    if not sqlite.exists():
        pytest.skip(f"{sqlite} not present")
    reader = SpineDbReader(f"sqlite:///{sqlite}", "base")

    empty_pss = pl.DataFrame(schema={
        "p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    canonical = process_source_sink_canonical(reader, empty_pss)
    assert canonical.height == 0
    assert set(canonical.columns) == {"p", "source", "sink", "method"}
