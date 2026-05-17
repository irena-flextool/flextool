"""Lagrangian region decomposition wrapper (Agent 3.1).

Builds a self-contained ``input_region_<region>/`` directory for one
decomposition region's standalone GMPL solve.

Step 2.5 status
---------------

This is the LEGACY disk-based region-build path.  It calls
:func:`flextool.input_derivation.run` to populate a cascade-input
:class:`FlexDataProvider`, then expects the on-disk ``input/`` staging
area to be available for :mod:`flextool.flextoolrunner.region_filter`.
Provider-based projection is out of Step 2.6 scope; tracked separately
as "2.6 region decomposition full port" in the audit.

Until item 2.6 ports the region filter onto the Provider, this wrapper
ALSO needs the cascade workdir to contain real ``input/*.csv`` files.
With Step 2.5 the in-memory path no longer drops bytes to disk; this
function therefore explicitly snapshot-dumps the Provider before
delegating to the region filter.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.flextoolrunner.runner_state import FlexToolConfigError


def write_input_for_region(
    input_db_url: str,
    scenario_name: str | None,
    logger: logging.Logger,
    region_group: str,
    output_dir: Path,
    work_folder: Path | None = None,
    precision_digits: int = 0,
) -> dict:
    """Write a self-contained ``input_region_<region>/`` directory for one
    decomposition region's standalone GMPL solve.

    Pre-Step-2.5 this was a thin wrapper around
    ``flextool.flextoolrunner.input_writer.write_input`` + the regional
    filter in :mod:`flextool.flextoolrunner.region_filter`.  Item 18
    deleted the input_writer module; we now drive
    :func:`flextool.input_derivation.run` to populate the Provider and
    rely on the on-disk staging area produced as a side effect (the
    workdir ``input/`` directory).

    Parameters
    ----------
    input_db_url, scenario_name, logger, work_folder, precision_digits
        Standard cascade entry-point inputs.
    region_group
        The group name whose ``decomposition_method`` is
        ``lagrangian_region``.  Must exist in the database.
    output_dir
        Destination directory, typically
        ``work_folder / "input_region_<region>"``.  Created if missing.

    Returns a dict: ``{"region": ..., "half_flows": [...], "kept_nodes": ...,
    "kept_units": ..., "kept_connections": ...}``.
    """
    from flextool.flextoolrunner import region_filter
    from flextool.input_derivation import run as _input_derivation_run

    wf = work_folder if work_folder is not None else Path.cwd()
    # Produce the full input/ directory first.  Provider receives every
    # frame; downstream the region filter reads from the workdir disk
    # staging area (legacy contract — see 2.6 follow-up).
    provider = FlexDataProvider()
    _input_derivation_run(
        input_db_url,
        provider,
        logger,
        scenario_name=scenario_name,
        work_folder=work_folder,
        precision_digits=precision_digits,
    )
    all_regions = region_filter.discover_decomposition_regions_from_db(input_db_url)
    if region_group not in all_regions:
        raise FlexToolConfigError(
            f"Region '{region_group}' is not declared with "
            f"decomposition_method='lagrangian_region' in the database. "
            f"Available regions: {sorted(all_regions) or '(none)'}"
        )
    result = region_filter.build_region_directory(
        input_dir=wf / "input",
        output_dir=Path(output_dir),
        region=region_group,
        all_regions=all_regions,
    )
    region_filter.write_region_coupling_manifest(
        work_folder=wf,
        results=[result],
    )
    return result
