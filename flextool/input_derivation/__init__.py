"""SpineDBBackend output → :class:`FlexDataProvider`.

The :mod:`input_derivation` tier sits between
:mod:`flextool.spinedb_backend` (raw EAV → polars materialisation) and
the in-cascade :class:`flextool.engine_polars._flex_data_provider.FlexDataProvider`
that the rest of the engine consumes.

Architecture
------------

::

    SpineDB → SpineDBBackend → input_derivation → FlexDataProvider → cascade

A derivation in this package reads its raw inputs **through the
:class:`SpineDBBackend`** (or, for derivations that compose other
derivation outputs, the Provider) and emits one or more canonical-
schema :class:`polars.DataFrame` objects to the Provider via
``provider.put(key, frame)``.

**No disk writes.**  The derivation tier is purely in-memory; CSV
emission is the Provider's job (``snapshot_processed_inputs`` /
``--csv-dump``).

Step 2.5-F scope
----------------

Phase A: package skeleton (this commit).

Phase B: :func:`flextool.input_derivation._dc_power_flow.derive_dc_power_flow`
— replaces :func:`flextool.flextoolrunner.input_writer._write_dc_power_flow_data`.

Phase C: ``_process_method.derive_process_method`` — replaces
``_write_process_method`` (consumes the ``derived/ct_method_overrides``
frame from Phase B).

Phase D-E: ``_commodity_ladder_cumulative.derive_commodity_ladder_cumulative``
and ``_commodity_ladder_annual.derive_commodity_ladder_annual``.

Phase F: ``_commodity_ladder_sets.derive_commodity_ladder_sets`` —
absorbs :mod:`flextool.flextoolrunner.preprocessing.commodity_ladder_sets`.

See ``specs/step_2_5_audit.md`` Section 7 (items 8-12) for full scope.

Hard rules
----------

* No disk writes anywhere in this package.
* No disk-fallback reads.  ``provider`` and ``backend`` are required
  keyword arguments; the derivation fails loudly if either is missing.
* Each derivation populates the Provider via ``provider.put(...)``;
  return values are reserved for derivation-internal carriers (e.g.
  ``ct_method_overrides`` from DC power flow, consumed by process
  method).
"""
from __future__ import annotations

__all__: list[str] = []
