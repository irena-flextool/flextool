"""The warm-start gates of the variable-cost Params follow the objective terms that consume them.

``_warm._WARM_PARAM_GATES[param]`` lists the index sets whose rows decide whether ``param`` reaches the LP; a
Param whose gates are all empty is "dormant" and its change is ignored.  ``model/core._obj_variable_cost`` gates
each term as ``_has_rows(d.<index set>) and d.<param> is not None``, so the gate of each variable-cost Param is
exactly the index set of its term.  Before c10 ``p_pdt_varCost_process`` was gated on ``pssdt_varCost_noEff``
(it multiplies ``pssdt_varCost_eff_connection``), so a changing connection cost with no noEff rows could skip a
needed rebuild.
"""
from __future__ import annotations

import inspect
import re

from flextool.engine.model import core
from flextool.engine.solve._warm import _WARM_PARAM_GATES

_TERM_GATE = re.compile(r"_has_rows\(d\.(pssdt_varCost_\w+)\)\s*and\s+d\.(p_\w+)\s+is not None")


def test_varcost_gates_match_objective_terms() -> None:
    src = inspect.getsource(core._obj_variable_cost)
    pairs = _TERM_GATE.findall(src)
    assert len(pairs) == 4, pairs
    consumers: dict[str, set[str]] = {}
    for idx, param in pairs:
        consumers.setdefault(param, set()).add(idx)
    assert set(consumers) == {"p_pssdt_varCost", "p_pdt_varCost_source", "p_pdt_varCost_sink",
                              "p_pdt_varCost_process"}
    for param, idx in consumers.items():
        assert set(_WARM_PARAM_GATES[param]) == idx, (param, _WARM_PARAM_GATES[param], idx)
