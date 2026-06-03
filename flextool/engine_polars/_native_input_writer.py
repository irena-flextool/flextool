"""Cascade-input Provider population from a Spine DB.

The live cascade's pre-solve "write_input" responsibility lives here:
:func:`write_workdir_inputs` reads the Spine database, runs the
:mod:`flextool.input_derivation` pipeline, and populates the
caller-supplied :class:`FlexDataProvider` with every derived frame the
cascade's downstream readers consume.

Pure in-memory
--------------

The :mod:`flextool.input_derivation` pipeline dispatches ``emit_*``
functions that push every derived frame into the caller-supplied
Provider directly — no monkey-patching, no disk I/O on this path.
Downstream readers
(:func:`flextool.engine_polars.input.load_flextool`,
:func:`flextool.engine_polars._output_writer.write_outputs_for_solve`,
the per-solve preprocessing dispatched by
:func:`flextool.engine_polars._native_run_model.native_run_model`)
resolve every input through the Provider.

For the ``--csv-dump`` debug path the cascade calls
:meth:`FlexDataProvider.snapshot_processed_inputs` separately; that is
the only on-disk emission of the derived frames the cascade produces.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flextool.engine_polars._flex_data_provider import FlexDataProvider


__all__ = ["write_workdir_inputs"]


def write_workdir_inputs(
    db_url: str,
    scenario_name: str | None,
    work_folder: Path,
    *,
    provider: "FlexDataProvider",
    logger: logging.Logger | None = None,
    precision_digits: int = 0,
    memory_recorder=None,
) -> None:
    """Populate *provider* with every cascade-input frame derived from
    the Spine database.

    Pure in-memory: each ``emit_*`` in the
    :mod:`flextool.input_derivation` pipeline pushes its derived
    frame into *provider* directly under its canonical key — no
    disk I/O on this path.

    Parameters
    ----------
    db_url : str
        Spine SQLite / postgres URL.  Already canonicalised to
        ``sqlite:///<path>`` / ``postgresql://...`` form.
    scenario_name : str | None
        Scenario filter; ``None`` triggers an auto-pick of the first
        scenario in the DB.
    work_folder : Path
        Workdir.  Created if absent.  Forwarded to
        :func:`flextool.input_derivation.run` for the not-yet-Provider-
        only writers' fallback path (becomes optional once every
        writer is Provider-only).
    provider : FlexDataProvider
        Required cascade-input Provider.  Every derivation in the
        :mod:`flextool.input_derivation` pipeline ``put``'s its
        materialised frames here; downstream cascade readers resolve
        them via
        :meth:`flextool.engine_polars._flex_data_provider.FlexDataProvider.get`.
    logger : logging.Logger, optional
        Logger to use during emission.  ``None`` builds a default
        named logger.
    precision_digits : int, default 0
        Float precision forwarded to ``SpineDBBackend.parameter_values``.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    work_folder = Path(work_folder)
    work_folder.mkdir(parents=True, exist_ok=True)

    from flextool.input_derivation import run as _input_derivation_run

    _input_derivation_run(
        db_url,
        provider,
        logger,
        scenario_name=scenario_name,
        work_folder=work_folder,
        precision_digits=precision_digits,
        memory_recorder=memory_recorder,
    )


