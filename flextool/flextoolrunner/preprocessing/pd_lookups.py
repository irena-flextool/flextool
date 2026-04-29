"""4-branch pd-style param lookups (pdProcess / pdNode / pdtCommodity-period).

Reusable helper for the calc-param migrations. The mod pattern at
flextool.mod:1216 (pdNode) and L1252 (pdProcess) is:

    param pdX {(e, param) in <class>__PeriodParam_in_use, d in period_with_history} :=
         + if (e, param, d) in <class>__param__period
              then pd_<class>[e, param, d]
           else if exists{(e, param, db) in <class>__param__period: (db, d) in period__branch} 1
                then sum{(db, d) in period__branch} pd_<class>[e, param, db]
           else if (e, param) in <class>__param
                then p_<class>[e, param]
           else 0;

    Branch 1: explicit (e, param, d) row from input/pd_<class>.csv.
    Branch 2: any (e, param, db) where db has d as a branch — sum those.
    Branch 3: per-class fall-back from input/p_<class>.csv.
    Branch 4: 0.
"""
from __future__ import annotations

import csv
from pathlib import Path


def _read_pd(path: Path) -> dict[tuple[str, str, str], float]:
    """4-col CSV: (entity, param, period, value)."""
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and row[0] and row[1] and row[2]:
                try:
                    out[(row[0], row[1], row[2])] = float(row[3])
                except ValueError:
                    continue
    return out


def _read_p(path: Path) -> dict[tuple[str, str], float]:
    """3-col CSV: (entity, param, value)."""
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1]:
                try:
                    out[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
    return out


def _read_period_branch(path: Path) -> dict[str, list[str]]:
    """Build d (branch column, mod's RHS of pair) → list of db (period column).

    period__branch.csv columns: (period, branch). Mod's
    ``(db, d) in period__branch`` reads ``db`` as period (col 0) and
    ``d`` as branch (col 1) — so for a given branch d, we want the list
    of source periods db.
    """
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.setdefault(row[1], []).append(row[0])
    return out


class PdLookup:
    """Replicates mod's 4-branch period-param resolution."""

    def __init__(
        self,
        pd_csv: Path,        # input/pd_<class>.csv (entity, param, period, value)
        p_csv: Path,         # input/p_<class>.csv  (entity, param, value)
        period_branch_csv: Path,  # solve_data/period__branch.csv
    ) -> None:
        self._pd: dict[tuple[str, str, str], float] = _read_pd(pd_csv)
        self._p: dict[tuple[str, str], float] = _read_p(p_csv)
        self._branches_for_d: dict[str, list[str]] = _read_period_branch(
            period_branch_csv
        )

    def get(self, e: str, param: str, d: str) -> float:
        """Replicates mod's pdProcess / pdNode value at (e, param, d)."""
        if (e, param, d) in self._pd:
            return self._pd[(e, param, d)]
        branches = self._branches_for_d.get(d, ())
        branched_vals = [
            self._pd[(e, param, db)] for db in branches
            if (e, param, db) in self._pd
        ]
        if branched_vals:
            return sum(branched_vals)
        if (e, param) in self._p:
            return self._p[(e, param)]
        return 0.0
