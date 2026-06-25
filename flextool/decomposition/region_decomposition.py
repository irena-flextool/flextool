"""Benders region decomposition wrapper.

Builds a self-contained ``input_region_<region>/`` directory for one
decomposition region's standalone solve.

Step 2.6 (this commit)
----------------------

The cascade-input :class:`FlexDataProvider` is the single source of
truth: :func:`flextool.input_derivation.run` populates it with every
``input/<name>`` frame, and the region filter
(:func:`flextool.decomposition.region_filter.build_region_provider`)
derives a region-scoped Provider from it entirely in-memory.  The CLI
deliverable — the on-disk ``input_region_<region>/`` directory — is
produced by one call to ``region_provider.snapshot_processed_inputs``
on the filtered Provider; no monolithic ``input/`` is ever written to
disk by this driver.
"""
from __future__ import annotations

import logging
from pathlib import Path

from flextool.engine_polars._flex_data_provider import FlexDataProvider
from flextool.engine_polars._solve_state import FlexToolConfigError


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
    decomposition region's standalone solve.

    Step 2.6: the pipeline is fully in-memory.

    1. :func:`flextool.input_derivation.run` populates the cascade-input
       :class:`FlexDataProvider`.
    2. :func:`region_filter.build_region_provider` derives a region-
       scoped Provider with rows filtered to the region's entities and
       virtual half-flow rows injected for cross-region pipelines.
    3. ``region_provider.snapshot_processed_inputs(output_dir)``
       materialises the CLI deliverable.
    4. The coupling manifest is stored in the cascade Provider as
       ``solve_data/region_coupling`` and snapshotted to
       ``work_folder/solve_data/region_coupling.csv``.

    No bridge: the cascade-input Provider is NOT snapshotted to disk.

    Parameters
    ----------
    input_db_url, scenario_name, logger, work_folder, precision_digits
        Standard cascade entry-point inputs.
    region_group
        The group name whose ``decomposition_method`` is
        ``benders_regional``.  Must exist in the database.
    output_dir
        Destination directory, typically
        ``work_folder / "input_region_<region>"``.  Created if missing.

    Returns a dict: ``{"region": ..., "half_flows": [...], "kept_nodes": ...,
    "kept_units": ..., "kept_connections": ...}``.
    """
    from flextool.decomposition import region_filter
    from flextool.input_derivation import run as _input_derivation_run

    wf = work_folder if work_folder is not None else Path.cwd()

    # 1. Populate the cascade-input Provider in-memory.  No disk staging.
    provider = FlexDataProvider()
    _input_derivation_run(
        input_db_url,
        provider,
        logger,
        scenario_name=scenario_name,
        work_folder=work_folder,
        precision_digits=precision_digits,
    )

    # 2. Validate region membership BEFORE filtering — surface a clear
    # error with the available regions if the caller picked a wrong one.
    all_regions = region_filter.discover_decomposition_regions_from_db(input_db_url)
    if region_group not in all_regions:
        raise FlexToolConfigError(
            f"Region '{region_group}' is not declared with "
            f"decomposition_method='benders_regional' in the database. "
            f"Available regions: {sorted(all_regions) or '(none)'}"
        )

    # 3. Derive the region-scoped Provider in-memory.
    region_provider, result = region_filter.build_region_provider(
        provider,
        region=region_group,
        all_regions=all_regions,
    )

    # 4. Snapshot the filtered Provider to the CLI-facing output dir.
    # ``snapshot_processed_inputs`` writes every ``input/<name>`` frame
    # under ``<output_dir>/input/<name>.csv``.  The region directory's
    # contract is the legacy "input/" subfolder, so we snapshot under
    # output_dir directly (the input/ prefix is already part of the key).
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Strip the leading ``input/`` from every key so the output lands as
    # ``output_dir/<file>.csv`` (matching the historical layout).
    flat_provider = FlexDataProvider()
    for key, frame in region_provider.items():
        if key.startswith("input/"):
            flat_provider.put(key.split("/", 1)[1], frame)
        else:
            # Non-input frames stay under their qualified parent.
            flat_provider.put(key, frame)
    flat_provider.snapshot_processed_inputs(output_dir)

    # 5. Write the coupling manifest.  Routes through the Provider for
    # symmetry with the rest of the pipeline; the snapshot lands at
    # ``work_folder/solve_data/region_coupling.csv``.
    manifest_provider = FlexDataProvider()
    region_filter.write_region_coupling_manifest_to_provider(
        manifest_provider,
        results=[result],
    )
    manifest_provider.snapshot_processed_inputs(wf)
    return result
