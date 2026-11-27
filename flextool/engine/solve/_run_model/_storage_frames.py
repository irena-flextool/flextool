"""Storage-method frames and run-scoped storage bookkeeping shared by the storage steps of the cascade: the
per-solve ``storage_binding_method`` / ``storage_start_end_method`` readers, :class:`_RollingStorage` and the coded
storage warnings; the module logger."""
from __future__ import annotations

import logging

import polars as pl


# Advisories that must survive the cascade's silencing of ``state.logger`` (raised to ERROR during the run).
_MODULE_LOGGER = logging.getLogger(__name__)


_SBM_KEY = "input/node__storage_binding_method"
_SSE_KEY = "input/node__storage_start_end_method"


def _method_col(frame: pl.DataFrame, name: str) -> "str | None":
    """The method column of a per-solve ``input/node__<name>`` frame (``<name>`` or ``method``)."""
    for col in (name, "method"):
        if col in frame.columns:
            return col
    return None


def _node_methods(provider, key: str, name: str) -> dict[str, str]:
    """``{node: method}`` of the per-solve frame ``key`` (empty when absent)."""
    if not provider.has(key):
        return {}
    frame = provider.get(key)
    col = _method_col(frame, name) if frame is not None and frame.height > 0 else None
    if col is None or "node" not in frame.columns:
        return {}
    return {str(n): str(m) for n, m in zip(frame["node"].to_list(), frame[col].to_list()) if m is not None}


def _frame_nodes(frame) -> set[str]:
    """The ``node`` values of a handoff frame (empty for None)."""
    if frame is None or frame.height == 0 or "node" not in frame.columns:
        return set()
    return {str(n) for n in frame["node"].to_list()}


class _RollingStorage:
    """Run-scoped bookkeeping of the bind_within_solve storage carry of rolling solves
    (:func:`_anchor_cyclic_in_rolling`).

    ``start[instance][node]``: the level (MWh) the first roll of a solve instance (``(complete solve, parent
    sub-solve)``) started from, which the instance's closing rolls return to.  ``original[level key]``: the per-solve
    ``input/node__storage_binding_method`` frame of the level before a roll rewrote it (the level Provider is shared
    by the rolls of a level, so each roll starts from the methods the user gave).  ``pending[solve]``: ``(instance,
    nodes)`` whose start level is read from the solve's handoff (``cycle_start_state``) once it is solved.
    ``borrowed_check[solve]``: ``(nodes, upper start levels)`` of the nodes a nested first roll starts from
    fix_start while the upper solve fixes them: their start is compared with the upper solve's.
    """

    def __init__(self) -> None:
        self.start: dict = {}
        self.original: dict = {}
        self.pending: dict = {}
        self.seen: set = set()
        self.borrowed_check: dict = {}
        self.warned: set = set()
        # scope_start[(instance, node, period, first step)]: the level (MWh) a bind_within_period /
        # bind_intraperiod_blocks period started from in the roll that began it.
        self.scope_start: dict = {}
        self.scope_pending: dict = {}
        # scope_from_start[solve]: (scope key, node) of the nested first rolls whose scope starts from fix_start: the
        # start level is read from the handoff's cycle_start_state.
        self.scope_from_start: dict = {}
        # usage_start[(instance, period)]: where the period starts in the instance (the first sub-solve holding it);
        # usage_realised[(instance, period, node)]: the fix_usage energy the instance's sub-solves realised in it.
        self.usage_start: dict = {}
        self.usage_realised: dict = {}
        # model_start[(nesting level, node)]: the level (MWh) the node started the model from at that nesting level
        # (the first sub-solve of the level at the model start, :func:`_record_model_start`).
        self.model_start: dict = {}

    def model_start_of(self, level: int, node: str) -> "float | None":
        """The level (MWh) ``node`` started the model from at nesting ``level``; a level that did not start it
        (no sub-solve of it at the model start) takes the nearest level that did, the deeper one first."""
        if (level, node) in self.model_start:
            return self.model_start[(level, node)]
        known = sorted((abs(lv - level), -lv) for lv, n in self.model_start if n == node)
        return self.model_start[(-known[0][1], node)] if known else None


def _restore_methods(store: _RollingStorage, level_key, provider) -> "pl.DataFrame | None":
    """The per-solve ``input/node__storage_binding_method`` of the level as the downgrades left it, before a roll's
    rewrite (:func:`_anchor_cyclic_in_rolling`): put back in the shared level Provider and returned (None when the
    Provider has none)."""
    if not provider.has(_SBM_KEY):
        return None
    if level_key not in store.original:
        store.original[level_key] = provider.get(_SBM_KEY)
    sbm = store.original[level_key]
    provider.put(_SBM_KEY, sbm)
    return sbm


def _storages_named(names) -> str:
    from flextool.engine import _behaviour_changes
    return _behaviour_changes.storages_named(names)


def _warn_storages(code: str, complete_solve_name: str, names: list, details: str) -> None:
    """A coded storage warning: one line per code and complete solve, naming how many and the first
    (:func:`flextool.run_warnings.warn_grouped`); *details* (the roll, the period) go to the debug log."""
    from flextool import run_warnings
    run_warnings.warn_grouped(code, names, solve=complete_solve_name, details=details)
