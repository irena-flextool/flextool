"""Re-export shim for the legacy ``flextool.flextoolrunner.solve_handoff``
import path.

Γ.8.D moved the canonical :class:`SolveHandoff` dataclass + capture
function into ``flextool/engine_polars/_solve_handoff.py`` so the
native polars orchestrator (``_orchestration.py``) owns one source of
truth for the handoff carrier schema.

R-O2 mitigation
---------------

Two long-standing import sites still reference this module by absolute
path:

* ``flextool/process_outputs/handoff_writers.py:67``
* ``flextool/process_outputs/cumulative_handoffs.py:89``

…plus tests that pin the legacy import (e.g.
``tests/engine_polars/test_solve_handoff.py``).  Re-exporting the
class here keeps source compatibility without duplicating the
dataclass: both import paths resolve to the SAME class object.

History
-------

Pre-Γ.8.D this file held the full implementation; the contents now
live in ``flextool.engine_polars._solve_handoff``.  See that module's
docstring for the carrier schema and the capture / write helpers'
contract.
"""
from __future__ import annotations

from flextool.engine_polars._solve_handoff import (
    SolveHandoff,
    write_fix_storage_files_from_handoff,
)


__all__ = [
    "SolveHandoff",
    "write_fix_storage_files_from_handoff",
]
