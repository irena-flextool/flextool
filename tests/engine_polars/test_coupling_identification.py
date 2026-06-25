"""Tests for cross-region coupling-column identification.

The cross-region coupling identification machinery
(:func:`flextool.engine_polars._benders._identify_coupling_cols` and the
:class:`flextool.engine_polars._benders.Coupling` dataclass) is SHARED:
it splits a whole-system model into regions and matches each cross-region
arc's export half-flow columns to the importing region's columns.  These
tests pin that contract on the LH2 three-region fixture.

(Previously lived in ``test_lagrangian.py``; the subgradient driver
``solve_lagrangian`` and its tests were deleted with the Benders
replacement — Phase 4 Chunk C.)
"""
from __future__ import annotations

import pytest

from polar_high import Problem, WarmProblem

from flextool.engine_polars import build_flextool, load_flextool
from flextool.engine_polars._benders import _identify_coupling_cols
from flextool.engine_polars._region_filter import split as region_split


# ---------------------------------------------------------------------------
# Coupling column identification
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lh2_workdir(scenario_workdir):
    return scenario_workdir("lh2_three_region", db_fixture="lh2")


@pytest.fixture(scope="module")
def lh2_data(lh2_workdir):
    return load_flextool(lh2_workdir)


@pytest.fixture(scope="module")
def lh2_warmproblems(lh2_data):
    splits = region_split(lh2_data, regions=["region_A", "region_B", "region_C"])
    warms = []
    for s in splits:
        pb = Problem()
        build_flextool(pb, s.data)
        wp = WarmProblem(pb)
        wp.solve()
        warms.append(wp)
    return splits, warms


class TestCouplingIdentification:
    def test_lh2_coupling_count(self, lh2_warmproblems) -> None:
        splits, warms = lh2_warmproblems
        couplings = _identify_coupling_cols(splits, warms)
        # pipe_AB has two directions × pipe_BC has two directions = 4
        # cross-region arcs = 4 couplings.
        assert len(couplings) == 4
        keys = {c.pipeline_key for c in couplings}
        assert keys == {
            ("pipe_AB", "lh2_A", "lh2_B"),
            ("pipe_AB", "lh2_B", "lh2_A"),
            ("pipe_BC", "lh2_B", "lh2_C"),
            ("pipe_BC", "lh2_C", "lh2_B"),
        }

    def test_lh2_coupling_columns_sized_correctly(self, lh2_warmproblems) -> None:
        splits, warms = lh2_warmproblems
        couplings = _identify_coupling_cols(splits, warms)
        # Each (d, t) of the original arc has its own coupling cell;
        # the LH2 fixture has 168 timesteps in 1 period.
        for cpl in couplings:
            assert cpl.export_cols.size == 168
            assert cpl.import_cols.size == 168

    def test_lh2_export_import_regions_correct(self, lh2_warmproblems) -> None:
        splits, warms = lh2_warmproblems
        couplings = _identify_coupling_cols(splits, warms)
        by_key = {c.pipeline_key: c for c in couplings}
        # pipe_AB(A→B): A exports, B imports.
        assert by_key[("pipe_AB", "lh2_A", "lh2_B")].export_region == "region_A"
        assert by_key[("pipe_AB", "lh2_A", "lh2_B")].import_region == "region_B"
        # pipe_AB(B→A): B exports, A imports.
        assert by_key[("pipe_AB", "lh2_B", "lh2_A")].export_region == "region_B"
        assert by_key[("pipe_AB", "lh2_B", "lh2_A")].import_region == "region_A"
        # pipe_BC(C→B): C exports, B imports.
        assert by_key[("pipe_BC", "lh2_C", "lh2_B")].export_region == "region_C"
        assert by_key[("pipe_BC", "lh2_C", "lh2_B")].import_region == "region_B"
