"""Thin persistence helper for a live highspy.Highs instance.

Exposes name-indexed access for decomposition schemes that need to
modify a loaded LP/MIP between solves:

- Pattern-based column/row lookup (e.g. all columns whose name matches
  ``v_invest[*]``)
- Fix / unfix columns by index or by name-pattern
- Change column costs (Lagrangian price updates)
- Add rows dynamically (Benders cuts)
- Extract primal, row duals, reduced costs after each ``run()``

Intended usage (from e.g. a Lagrangian coordinator)::

    handle = HighsModelHandle(h)              # wraps a post-readModel Highs
    handle.build_name_maps()                  # post-presolve-safe indexing
    inv_cols = handle.cols_matching("v_invest[*]")
    handle.fix_cols(inv_cols, trial_invest)
    handle.solve()
    obj = handle.objective()
    reduced = handle.reduced_costs(inv_cols)
    handle.unfix_cols(inv_cols)               # restore original bounds
    handle.add_row("benders_cut_1", coeffs, lower=None, upper=rhs)
    handle.solve()

The helper is a pure persistence wrapper — it does not own the
``Highs`` lifetime, it does not decide solver options, and it does not
re-read the MPS.  The caller supplies an already-configured ``Highs``
instance (typically post-``readModel`` or post-``passModel``) and is
responsible for disposing of it when done.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

import highspy
import numpy as np


_ScalarOrSeq = float | Sequence[float]


def _compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a glob-style pattern to a regex.

    Only ``*`` (any sequence) and ``?`` (single char) are treated as
    wildcards; everything else — including the ``[`` / ``]`` characters
    commonly used in GMPL-derived column names like ``v_invest[alt_1]``
    — is matched literally.  This is a deliberate deviation from
    :mod:`fnmatch`, whose ``[...]`` character-class syntax would
    otherwise mis-parse our column names.
    """
    parts: list[str] = []
    for ch in pattern:
        if ch == "*":
            parts.append(".*")
        elif ch == "?":
            parts.append(".")
        else:
            parts.append(re.escape(ch))
    return re.compile("".join(parts) + r"\Z")


def _as_float_array(values: _ScalarOrSeq, n: int) -> np.ndarray:
    """Broadcast a scalar or length-n sequence to a float64 ndarray."""
    if isinstance(values, (int, float)):
        return np.full(n, float(values), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(
            f"Expected scalar or length-{n} sequence, got shape {arr.shape}"
        )
    return arr


@dataclass
class HighsModelHandle:
    """Persistence wrapper for a live ``highspy.Highs`` instance.

    Attributes
    ----------
    h : highspy.Highs
        The live solver instance.  Must already have a model loaded
        (e.g. via ``readModel`` or ``passModel``) before calling
        :meth:`build_name_maps` or :meth:`solve`.
    col_by_name : dict[str, int]
        Column-name → column-index map.  Populated by
        :meth:`build_name_maps`.
    row_by_name : dict[str, int]
        Row-name → row-index map.  Populated by :meth:`build_name_maps`.
    """

    h: "highspy.Highs"
    col_by_name: dict[str, int] = field(default_factory=dict)
    row_by_name: dict[str, int] = field(default_factory=dict)
    # Snapshot of original (lb, ub) captured on first fix_cols() per column,
    # so unfix_cols() with None-bounds can restore.
    _orig_bounds: dict[int, tuple[float, float]] = field(
        default_factory=dict, repr=False
    )

    # ------------------------------------------------------------------
    # Name maps
    # ------------------------------------------------------------------
    def build_name_maps(self) -> None:
        """Rebuild col/row name→index maps from the live ``Highs`` state.

        Safe to call after presolve — HiGHS may rename or eliminate
        columns during its internal presolve, so callers that need
        name lookups on the *solved* model should call this method
        **after** the first :meth:`solve` rather than before.

        Columns or rows with empty / missing names are skipped.
        Duplicate names resolve to the first occurrence.
        """
        lp = self.h.getLp()
        self.col_by_name = {}
        for idx, name in enumerate(lp.col_names_):
            if name and name not in self.col_by_name:
                self.col_by_name[name] = idx
        self.row_by_name = {}
        for idx, name in enumerate(lp.row_names_):
            if name and name not in self.row_by_name:
                self.row_by_name[name] = idx

    # ------------------------------------------------------------------
    # Pattern lookup
    # ------------------------------------------------------------------
    def cols_matching(self, pattern: str) -> list[int]:
        """Glob-style column-name lookup.

        ``v_invest[*]`` matches any column whose name starts with
        ``v_invest[``.  Only ``*`` and ``?`` are wildcards; ``[`` and
        ``]`` are matched literally (see :func:`_compile_glob`).
        Case-sensitive.  Returns column indices in ascending order.
        """
        regex = _compile_glob(pattern)
        matches = [
            idx
            for name, idx in self.col_by_name.items()
            if regex.match(name) is not None
        ]
        return sorted(matches)

    def rows_matching(self, pattern: str) -> list[int]:
        """Glob-style row-name lookup.  See :meth:`cols_matching`."""
        regex = _compile_glob(pattern)
        matches = [
            idx
            for name, idx in self.row_by_name.items()
            if regex.match(name) is not None
        ]
        return sorted(matches)

    # ------------------------------------------------------------------
    # Column bounds
    # ------------------------------------------------------------------
    def _snapshot_bounds(self, indices: Sequence[int]) -> None:
        """Capture original (lb, ub) for columns not already snapshotted."""
        missing = [i for i in indices if i not in self._orig_bounds]
        if not missing:
            return
        lp = self.h.getLp()
        for i in missing:
            self._orig_bounds[i] = (
                float(lp.col_lower_[i]),
                float(lp.col_upper_[i]),
            )

    def fix_cols(
        self, indices: Sequence[int], values: _ScalarOrSeq
    ) -> None:
        """Fix the given columns to ``values`` (set ``lb = ub = value``).

        ``values`` may be a scalar (broadcast to all columns) or one
        value per column.  Original bounds are snapshotted on first
        call per column so :meth:`unfix_cols` with default arguments
        can restore them.
        """
        if not indices:
            return
        self._snapshot_bounds(indices)
        idx_arr = np.asarray(indices, dtype=np.int32)
        v = _as_float_array(values, len(indices))
        self.h.changeColsBounds(len(indices), idx_arr, v, v)

    def unfix_cols(
        self,
        indices: Sequence[int],
        lb: _ScalarOrSeq | None = None,
        ub: _ScalarOrSeq | None = None,
    ) -> None:
        """Restore column bounds.

        If ``lb`` / ``ub`` are ``None``, restores the bounds captured
        on the most recent snapshot (i.e. the bounds the column had
        before the first :meth:`fix_cols` call).  Explicit ``lb`` /
        ``ub`` override the snapshot for that call; the snapshot
        itself is retained for a future default-restore.
        """
        if not indices:
            return
        n = len(indices)
        if lb is None or ub is None:
            # Need snapshotted values for any missing side.
            missing = [i for i in indices if i not in self._orig_bounds]
            if missing:
                raise RuntimeError(
                    f"unfix_cols: no snapshot available for columns {missing}; "
                    "either call fix_cols first or pass explicit lb/ub."
                )
        lb_arr = (
            np.array(
                [self._orig_bounds[i][0] for i in indices], dtype=np.float64
            )
            if lb is None
            else _as_float_array(lb, n)
        )
        ub_arr = (
            np.array(
                [self._orig_bounds[i][1] for i in indices], dtype=np.float64
            )
            if ub is None
            else _as_float_array(ub, n)
        )
        idx_arr = np.asarray(indices, dtype=np.int32)
        self.h.changeColsBounds(n, idx_arr, lb_arr, ub_arr)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def change_costs(
        self, indices: Sequence[int], costs: _ScalarOrSeq
    ) -> None:
        """Modify linear-objective coefficients for the given columns."""
        if not indices:
            return
        idx_arr = np.asarray(indices, dtype=np.int32)
        c = _as_float_array(costs, len(indices))
        self.h.changeColsCost(len(indices), idx_arr, c)

    # ------------------------------------------------------------------
    # Row addition
    # ------------------------------------------------------------------
    def add_row(
        self,
        name: str,
        coefficients: dict[int, float],
        lower: float | None = None,
        upper: float | None = None,
    ) -> int:
        """Add a new constraint row.

        Parameters
        ----------
        name : str
            Row name — registered in :attr:`row_by_name`.
        coefficients : dict[int, float]
            Mapping column-index → coefficient for the nonzero
            entries in the new row.
        lower, upper : float or None
            Row bounds.  ``None`` maps to ``±highspy.kHighsInf``
            (i.e. a one-sided constraint).  To add an equality,
            pass ``lower == upper``.

        Returns
        -------
        int
            The new row's index.
        """
        inf = highspy.kHighsInf
        lo = -inf if lower is None else float(lower)
        up = inf if upper is None else float(upper)
        if not coefficients:
            # Empty row: addRow with zero nonzeros.
            idx_arr = np.zeros(0, dtype=np.int32)
            val_arr = np.zeros(0, dtype=np.float64)
            nz = 0
        else:
            items = sorted(coefficients.items())
            idx_arr = np.asarray([i for i, _ in items], dtype=np.int32)
            val_arr = np.asarray([v for _, v in items], dtype=np.float64)
            nz = len(items)
        new_idx = self.h.getNumRow()
        self.h.addRow(lo, up, nz, idx_arr, val_arr)
        # Tag the row with the requested name so subsequent build_name_maps
        # (and the immediate row_by_name update below) can find it.
        self.h.passRowName(new_idx, name)
        self.row_by_name[name] = new_idx
        return new_idx

    # ------------------------------------------------------------------
    # Solve + status
    # ------------------------------------------------------------------
    def solve(self) -> "highspy.HighsStatus":
        """Call ``self.h.run()``.

        HiGHS warm-starts from the retained basis by default, so
        successive solves after small bound / cost / row modifications
        converge in few iterations.  Returns the raw ``HighsStatus``
        — callers should check :meth:`is_optimal` (which inspects
        ``HighsModelStatus``) for solution quality.
        """
        return self.h.run()

    def is_optimal(self) -> bool:
        """True iff the most recent solve ended with ``kOptimal``."""
        return self.h.getModelStatus() == highspy.HighsModelStatus.kOptimal

    def objective(self) -> float:
        """Objective value from the most recent solve."""
        return float(self.h.getInfo().objective_function_value)

    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------
    def _solution(self):
        return self.h.getSolution()

    def primal(self, indices: Sequence[int]) -> list[float]:
        """Primal values ``x_i`` for the given columns."""
        sol = self._solution()
        return [float(sol.col_value[i]) for i in indices]

    def row_duals(self, indices: Sequence[int]) -> list[float]:
        """Row duals (shadow prices) for the given constraints."""
        sol = self._solution()
        return [float(sol.row_dual[i]) for i in indices]

    def reduced_costs(self, indices: Sequence[int]) -> list[float]:
        """Reduced costs (column duals) for the given columns."""
        sol = self._solution()
        return [float(sol.col_dual[i]) for i in indices]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def iteration_count(self) -> int:
        """Simplex iteration count of the most recent solve.

        Useful for verifying that warm-starts reduce work on
        successive solves.
        """
        return int(self.h.getInfo().simplex_iteration_count)

    def num_cols(self) -> int:
        return int(self.h.getNumCol())

    def num_rows(self) -> int:
        return int(self.h.getNumRow())


__all__ = ["HighsModelHandle"]
