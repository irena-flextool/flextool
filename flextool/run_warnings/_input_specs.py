"""The input-side warning codes (W-0100..W-0499) of the registry :mod:`flextool.run_warnings.registry`: the checks
of the input derivation (``flextool/input_derivation``) and the plots.  Same rules as the registry's own codes
(code order, a short lowercase ``message``, a ``body`` with **How to fix:**); the registry merges them.

* W-01xx groups, flowGroups and reserves
* W-02xx units and connections
* W-03xx nodes, profiles and the time / period keys of values
* W-04xx commodities, DC power flow and plots
"""
from __future__ import annotations

from flextool.run_warnings.registry import WarningSpec

INPUT_SPECS: tuple[WarningSpec, ...] = ()
