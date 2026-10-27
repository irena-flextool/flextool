"""``p_window_fraction`` survives a dump → CSV reload of a roll (#54).

Moved from ``tests/engine/test_co2_max_period_rolling.py`` (c24 S1-8): a
dumped roll is reloaded with a provider-less ``load_flextool`` (no DB
reader), which rebuilds the roll's own fraction at the load boundary.
The rolling runs come from that module's ``boundary_runs`` fixture.
"""
from __future__ import annotations

import pytest

from tests.engine import test_co2_max_period_rolling as _co2

pytestmark = pytest.mark.solver

# The module fixture and helpers of the source module (shared, not copied).
boundary_runs = _co2.boundary_runs
_fraction = _co2._fraction
_roll = _co2._roll


def test_window_fraction_survives_dump_reload(boundary_runs, tmp_path) -> None:
    """A dumped roll (CSV workdir, no DB reader) rebuilds its own fraction at the load boundary."""
    from flextool.engine import load_flextool

    for i, want in ((5, {"p2020": 18 / 48, "p2025": 12 / 48}), (15, {"p2025": 6 / 48})):
        fd = boundary_runs[False][_roll(i)].flex_data
        out = tmp_path / f"dumped_roll_{i}"
        fd.dump_csvs(out, copy_meta_from=boundary_runs["csv_workdir"])
        redo = load_flextool(out)
        assert _fraction(redo) == _fraction(fd), i
        assert _fraction(fd) == pytest.approx(want, rel=1e-12), i
