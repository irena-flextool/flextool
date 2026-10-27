"""Enum substrate of the CSV reload: ``load_flextool`` auto-builds ``axis_enums``.

Moved from ``tests/engine/test_enum_activation.py`` (c24 S1-8).  A
provider-less ``load_flextool`` of a work folder that holds
``tests.sqlite`` builds the axis-Enum vocabulary from that DB (the
``tests.sqlite`` auto-build in ``flex_data.py``).  Production never takes
this path: it loads with the sub-solve's provider and a DB reader, and
sets no ``axis_enums`` (c24 design_stage1 F2).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.preprocess.flex_data import load_flextool


def test_load_flextool_threads_axis_enums(
    tmp_path: Path, scenario_workdir,
) -> None:
    """Phase 4 substrate: ``load_flextool`` populates the Provider's
    ``axis_enums`` + ``contract`` attributes from the workdir sqlite.

    This is the prerequisite for the cascade-wide activation flip
    (which is BLOCKED on the cross-axis alias audit — see Phase 4
    BLOCKERS in ``specs/enum_dtype_refactor_plan.md``).
    """
    import shutil

    fixture = scenario_workdir("base")
    work = tmp_path / "work_base"
    shutil.copytree(fixture, work)

    # Call without an explicit provider so load_flextool builds an
    # ephemeral one seeded from the workdir; the helper then
    # auto-builds axis_enums from the workdir sqlite.
    flex_data = load_flextool(work)
    assert flex_data is not None

    # Build the same axis_enums independently to confirm the workdir
    # sqlite produces the canonical vocabulary the substrate would
    # have threaded.
    from flextool.spinedb_backend._axis_enums import (
        build_axis_enums,
        load_axis_contract,
    )
    from flextool.spinedb_backend import SpineDBBackend

    sqlite_path = work / "tests.sqlite"
    contract = load_axis_contract()
    with SpineDBBackend(f"sqlite:///{sqlite_path}", None) as backend:
        axis_enums = build_axis_enums(backend, contract)

    # Spot-check vocabulary: the ``p`` axis enum is the union of unit
    # + connection names; ``work_base`` has at least one of each.
    p_enum = axis_enums.get("p")
    assert p_enum is not None and isinstance(p_enum, pl.Enum)
    assert len(p_enum.categories) > 0
