"""Guard: the group capacity-margin projection consumes the new
``group.capacity_margin_method`` enum (value ``'manual'``), not the
retired ``group.has_capacity_margin`` flag.

If this projection silently returns an empty frame, the downstream
capacity-margin constraint is never built (``_group_slack`` gates the
constraint on ``groupCapacityMargin.height > 0``).  This test pins that
the projection is non-empty for the ``capacity_margin`` scenario in
``tests/fixtures/tests.json``, which carries
``capacity_margin_method='manual'`` on the ``capacity_margin`` group.

The DB is built from the JSON fixture under a tmp path via the
session-scoped ``test_db_url`` fixture — never a checked-in ``.sqlite``.
"""
from __future__ import annotations

from flextool.engine_polars._projection_params import groupCapacityMargin
from flextool.engine_polars._spinedb_reader import SpineDbReader


def test_group_capacity_margin_non_empty_for_manual(test_db_url: str) -> None:
    """``groupCapacityMargin`` returns height > 0 for a scenario that
    sets ``capacity_margin_method='manual'`` — proving the constraint is
    no longer silently disabled after the flag→method rename."""
    reader = SpineDbReader(test_db_url, "capacity_margin")
    projected = groupCapacityMargin(reader)

    assert projected.columns == ["g"], projected.columns
    assert projected.height > 0, (
        "groupCapacityMargin is empty for the 'capacity_margin' scenario; "
        "the capacity-margin constraint would be silently disabled. "
        "Expected the projection to read group.capacity_margin_method == "
        "'manual'."
    )
