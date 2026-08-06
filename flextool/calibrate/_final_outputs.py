"""Regenerate the non-parquet output formats from the final surviving parquet.

The calibration loop solves each iteration with ``--write-methods parquet``
only (fast; the loop reads its signals straight from the parquet), and clears
the tree between iterations, so when :func:`~flextool.calibrate._loop.run_calibration`
returns the LAST iteration's full results survive at
``<out_root>/output_parquet/<scenario>/``.

This module turns that surviving parquet tree into the other regular output
formats (csv / excel / spinedb / plot) **without re-solving** — it drives the
engine's own disk-replay path (:func:`flextool.process_outputs.write_outputs.write_outputs`
with ``read_parquet_dir=True``), the same code the ``flextool-write-outputs``
CLI and the Toolbox re-plot flow use.  So the calibrated model's results are
available "just like regular results", but produced by re-reading parquet
rather than paying for another solve.

Known limitations of the replay path (inherent to reading back from parquet,
not to this module):

* ``spinedb`` replay cannot recover the two investment/operations
  discount-factor parameters (the writer needs the live ``par`` frame) nor the
  ordering of a two-way connection's (source, sink);
* ``csv`` replay writes the per-table frames only — it does not regenerate the
  native run's ``summary.csv``.

parquet is intentionally excluded from the accepted methods: it is already on
disk (re-writing it would be a no-op the engine guards against anyway).
"""

from __future__ import annotations

from pathlib import Path

# The formats this post-loop step can regenerate from parquet. ``parquet`` is
# excluded on purpose (already present); the rest are the regular output
# formats a normal run would emit.
FINAL_WRITE_METHOD_CHOICES = ("csv", "excel", "spinedb", "plot")


def write_final_outputs(
    out_root: Path,
    scenario: str,
    write_methods: "list[str] | tuple[str, ...]",
    *,
    results_db_url: str | None = None,
) -> list[str]:
    """Replay the final parquet tree into *write_methods*; return what was written.

    Reads ``<out_root>/output_parquet/<scenario>/`` (the surviving output of the
    last calibration solve) and emits each requested format alongside it —
    ``output_csv/<scenario>/``, ``output_excel/output_<scenario>.xlsx``,
    ``output_plots/<scenario>/`` and/or a results SpineDB — with **no** solve.

    Parameters
    ----------
    out_root:
        The calibration ``--output-location`` root: the parent that holds
        ``output_parquet/`` (and gains the sibling ``output_csv/`` … trees).
    scenario:
        Scenario name; also the ``output_parquet`` sub-folder to read back.
    write_methods:
        Which formats to regenerate (a subset of
        :data:`FINAL_WRITE_METHOD_CHOICES`).  Empty ⇒ nothing to do.
    results_db_url:
        Target SpineDB URL for the ``spinedb`` method; defaults inside the
        engine to ``<out_root>/results.sqlite`` when omitted.

    Returns
    -------
    The list of methods actually written (empty when *write_methods* is empty).

    Raises
    ------
    FileNotFoundError
        If the expected ``output_parquet/<scenario>/`` directory is absent —
        i.e. there is no final solve to replay.
    """
    methods = [m for m in write_methods if m]
    if not methods:
        return []

    out_root = Path(out_root)
    parquet_dir = out_root / "output_parquet" / scenario
    if not parquet_dir.is_dir():
        raise FileNotFoundError(
            f"no parquet output to replay for scenario '{scenario}': "
            f"expected {parquet_dir}/ (did the final solve write outputs?)"
        )

    # Deferred import: write_outputs pulls in the plotting/Excel stack, which we
    # do not want to load unless the operator actually asked for final outputs.
    from flextool.process_outputs.write_outputs import write_outputs

    write_outputs(
        scenario_name=scenario,
        output_location=str(out_root),
        # subdir is the folder name under output_parquet/ — the scenario — so
        # the replay reads exactly the tree the calibration loop left behind.
        subdir=scenario,
        read_parquet_dir=True,
        write_methods=methods,
        # Belt-and-suspenders: keep output_location non-empty even if the engine
        # ever tries to fall back to settings resolution on the replay path.
        fallback_output_location=str(out_root),
        results_db_url=results_db_url,
    )
    return methods


__all__ = ["FINAL_WRITE_METHOD_CHOICES", "write_final_outputs"]
