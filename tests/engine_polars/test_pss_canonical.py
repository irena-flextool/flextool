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
    process_source_sink_collapsed,
    process_source_sink_eff,
    process_source_sink_noEff,
)


DATA = Path(__file__).resolve().parent / "data"


# Fixtures with diverse process_source_sink populations (units +
# connections) to exercise the canonical partition.  Each entry is
# ``(scenario_name, db_fixture)`` — ``main`` for scenarios in
# ``tests/fixtures/tests.json``, ``lh2`` for ``lh2_three_region.json``,
# etc.
CANONICAL_SCENARIOS: list[tuple[str, str]] = [
    ("base", "main"),
    ("coal", "main"),
    ("dr_decrease_demand", "main"),
    ("capacity_margin", "main"),
    ("network_wind_coal_battery_fullYear_invest", "main"),
    ("lh2_three_region", "lh2"),
]


def _assert_canonical_partitions(sqlite: Path, scenario: str) -> None:
    """Body of the partition-parity test — shared across the scenario-
    factory path and the disk-only LH2 path."""
    reader = SpineDbReader(f"sqlite:///{sqlite}", scenario)

    # Δ.17b Gap B: ``process_source_sink_canonical`` now produces flextool's
    # preprocessing-side **collapsed** shape (DIRECT methods cross-joined to
    # ``(p, source, sink)``; INDIRECT methods kept as 2-arc form).  Use
    # :func:`process_source_sink_collapsed` as the reference rather than
    # the *expanded* :func:`process_source_sink` (which mirrors the Spine
    # ``unit__inputNode`` ∪ ``unit__outputNode`` union).
    canonical = process_source_sink_canonical(reader)
    collapsed = process_source_sink_collapsed(reader)
    eff = process_source_sink_eff(reader)
    no_eff = process_source_sink_noEff(reader)

    # 1. Schema invariant.
    assert set(canonical.columns) == {"p", "source", "sink", "method"}, \
        f"canonical schema: {canonical.columns}"

    # 2. method ∈ {'eff', 'noEff'} for every row.
    methods = set(canonical["method"].unique().to_list()) if canonical.height > 0 else set()
    assert methods <= {"eff", "noEff"}, f"unexpected methods: {methods}"

    # 3. Union invariant: canonical's (p,s,sink) rows == collapsed rows.
    canonical_keys = (canonical
        .select("p", "source", "sink")
        .unique()
        .sort("p", "source", "sink"))
    collapsed_sorted = collapsed.sort("p", "source", "sink")
    assert canonical_keys.equals(collapsed_sorted), \
        "canonical (p,source,sink) doesn't match collapsed pss"

    # 4. .filter(method=='eff') == process_source_sink_eff
    eff_from_canonical = (canonical
        .filter(pl.col("method") == "eff")
        .select("p", "source", "sink")
        .unique()
        .sort("p", "source", "sink"))
    assert eff_from_canonical.equals(eff.sort("p", "source", "sink")), \
        "filter(method='eff') mismatches process_source_sink_eff"

    # 5. .filter(method=='noEff') == process_source_sink_noEff
    no_eff_from_canonical = (canonical
        .filter(pl.col("method") == "noEff")
        .select("p", "source", "sink")
        .unique()
        .sort("p", "source", "sink"))
    assert no_eff_from_canonical.equals(no_eff.sort("p", "source", "sink")), \
        "filter(method='noEff') mismatches process_source_sink_noEff"

    # 6. eff and noEff are disjoint (sanity).
    if eff.height > 0 and no_eff.height > 0:
        overlap = (eff
            .join(no_eff, on=["p", "source", "sink"], how="inner"))
        assert overlap.height == 0, "_eff and _noEff overlap"


@pytest.mark.parametrize(
    "scenario,db_fixture", CANONICAL_SCENARIOS,
    ids=lambda x: x,
)
def test_canonical_partitions_via_scenario_workdir(
    scenario: str, db_fixture: str, scenario_workdir,
) -> None:
    """``process_source_sink_canonical`` filtered by ``method`` equals
    the projected ``_eff`` / ``_noEff`` helpers.  Plus the canonical
    frame's ``(p, source, sink)`` rows equal ``pss``."""
    work = scenario_workdir(scenario, db_fixture=db_fixture)
    _assert_canonical_partitions(work / "tests.sqlite", scenario)


def test_canonical_empty_when_no_processes(scenario_workdir) -> None:
    """Empty ``pss`` → empty canonical frame with the right schema."""
    work = scenario_workdir("base")
    sqlite = work / "tests.sqlite"
    reader = SpineDbReader(f"sqlite:///{sqlite}", "base")

    empty_pss = pl.DataFrame(schema={
        "p": pl.Utf8, "source": pl.Utf8, "sink": pl.Utf8})
    canonical = process_source_sink_canonical(reader, empty_pss)
    assert canonical.height == 0
    assert set(canonical.columns) == {"p", "source", "sink", "method"}
