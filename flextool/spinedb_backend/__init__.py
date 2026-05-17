"""SpineDB → in-memory polars frames.

The :class:`SpineDBBackend` is the canonical materialiser for the
spec-driven entity / parameter / default-value tables that the legacy
:mod:`flextool.flextoolrunner.input_writer` used to emit as
``input/*.csv`` files.

Step 2.5 architecture
---------------------

::

    SpineDB → SpineDBBackend → input_derivation → FlexDataProvider → cascade

The Backend owns the EAV → tabular transformation.  Its methods return
canonical-schema :class:`polars.DataFrame` objects; **no disk writes**
happen anywhere in this package.  Disk emission is the Provider's job
(via :meth:`flextool.engine_polars._flex_data_provider.FlexDataProvider.snapshot_processed_inputs`).

See ``specs/step_2_5_audit.md`` Section 1 and Section 7 (items 1-4)
for the migration plan.
"""
from __future__ import annotations

from flextool.spinedb_backend._backend import SpineDBBackend

__all__ = ["SpineDBBackend"]
