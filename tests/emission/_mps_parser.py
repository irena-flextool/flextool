"""Lightweight MPS row-family counter for Tier 7 emission tests.

Counts rows per constraint family in a free-MPS file written by
``glpsol --check --wfreemps`` (phase 1 of FlexToolRunner).

Empirical row format observed in ``flextool.mps`` (verified on the
``coal``, ``coal_wind_min_uptime`` and ``wind_battery`` scenarios)::

     N total_cost
     E nodeBalance_eq[y2020_2day_dispatch,west,default,p2020,t0001,...]
     L maxToSink[coal_plant,coal_market,west,p2020,t0001]
     ...

The objective row (type ``N``, name ``total_cost``) is the only
bracket-less row in the section, so we both skip it explicitly *and*
strip everything from the first ``[`` to the end of the row name to
recover the family name.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path


def parse_mps_row_families(mps_path: Path) -> dict[str, int]:
    """Return ``{family_name: row_count}`` for the MPS at ``mps_path``.

    Strips ``[...]`` index suffixes; ``family_name`` is the part before
    the first ``[``. The objective row (``N total_cost``) is excluded.

    Parameters
    ----------
    mps_path:
        Absolute path to a free-MPS file (the format written by
        ``glpsol --wfreemps``).

    Returns
    -------
    dict[str, int]
        Mapping from constraint family name to row count.
    """
    counts: Counter[str] = Counter()
    in_rows = False
    with open(mps_path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("*"):
                continue
            # Section markers are unindented in free MPS.
            if not line.startswith(" ") and not line.startswith("\t"):
                if stripped == "ROWS":
                    in_rows = True
                    continue
                # Any other top-level keyword (COLUMNS, RHS, RANGES,
                # BOUNDS, ENDATA, NAME, OBJSENSE, ...) ends the ROWS
                # section.
                if in_rows:
                    break
                continue
            if not in_rows:
                continue
            # Free-MPS row line: "<TYPE> <name>" with whitespace
            # separation. Skip the objective (type 'N').
            tokens = stripped.split()
            if len(tokens) < 2:
                continue
            row_type, row_name = tokens[0], tokens[1]
            if row_type == "N":
                continue
            family = row_name.split("[", 1)[0]
            counts[family] += 1
    return dict(counts)
