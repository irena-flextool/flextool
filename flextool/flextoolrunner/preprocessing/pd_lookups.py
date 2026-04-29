"""4- and 7-branch pd-style param lookups (pdProcess / pdNode / pdtCommodity-period
+ pdtProcess / pdtNode / pdtReserve_upDown_group).

Reusable helper for the calc-param migrations. The 4-branch mod pattern at
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

The 7-branch ``pdtX`` pattern at flextool.mod L1227 (pdtProcess) extends
the 4-branch pattern with a time-axis fallback and two stochastic
branch-folding contributions (see :class:`PdtLookup` for details).
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


def _read_pbt(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    """6-col CSV: (entity, param, branch, time_start, time, value)."""
    out: dict[tuple[str, str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 6 and row[0] and row[1] and row[2] and row[3] and row[4]:
                try:
                    out[(row[0], row[1], row[2], row[3], row[4])] = float(row[5])
                except ValueError:
                    continue
    return out


def _read_pt(path: Path) -> dict[tuple[str, str, str], float]:
    """4-col CSV: (entity, param, time, value)."""
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


def _read_pairs_to_dict(path: Path, key_col: int) -> dict[str, list[str]]:
    """Generic two-col CSV read into ``key_col → list[other_col]``."""
    out: dict[str, list[str]] = {}
    if not path.exists():
        return out
    other_col = 1 - key_col
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.setdefault(row[key_col], []).append(row[other_col])
    return out


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


class PdtLookup:
    """Replicates mod's 7-branch ``pdtProcess`` / ``pdtNode`` /
    ``pdtReserve_upDown_group`` resolution.

    Mod pattern (flextool.mod L1227, pdtProcess) — outer ``(p, param, d, t)``::

        if exists{(d,ts) in period__time_first, (d,tb) in solve_branch__time_branch:
                  (p, param, tb, ts, t) in process__param__branch__time} 1
              && exists{(g,p) in group_process: g in groupStochastic} 1
            then sum{(d,ts) in period__time_first, (d,tb) in solve_branch__time_branch}
                       pbt_process[p, param, tb, ts, t]
        else if exists{(pe,tb) in solve_branch__time_branch, (d,ts) in period__time_first:
                       (pe,d) in period__branch
                       && (p, param, tb, ts, t) in process__param__branch__time} 1
            then sum{(pe,tb) in solve_branch__time_branch, (d,ts) in period__time_first:
                     (pe,d) in period__branch}
                       pbt_process[p, param, tb, ts, t]
        else if (p, param, d) in process__param__period then pd_process[p, param, d]
        else if (p, param, t) in process__param__time   then pt_process[p, param, t]
        else if (p, param)    in process__param         then  p_process[p, param]
        else if param in processParam_def1              then 1
        else 0;

    GMPL semantics — outer indices ``(p, param, d, t)`` are pre-bound by
    the param's domain. Inside ``{(d, ts) in period__time_first}`` the
    first column ``d`` matches the outer ``d`` (filter); ``ts`` is a
    fresh local. Same for ``(d, tb) in solve_branch__time_branch`` and
    ``(pe, d) in period__branch`` (``pe`` fresh, ``d`` outer).

    Branch 1 (stochastic-process fold-in): fires when ``e`` belongs to
    any group flagged in ``groupStochastic``. Iterates ``ts`` over
    ``period__time_first[d]`` and ``tb`` over ``solve_branch[d]`` and
    sums any pbt entries at ``(e, param, tb, ts, t)``.

    Branch 2 (parent-period branch fold-in): for each parent period ``pe``
    of ``d`` (via ``period__branch``), iterate ``tb`` over
    ``solve_branch[pe]`` and ``ts`` over ``period__time_first[d]``, sum
    matching pbt rows. NB: branches come from the parent ``pe``, not
    ``d``.

    Branches 3-7: standard pd / pt / p / def1 / 0 fallback.
    """

    def __init__(
        self,
        pbt_csv: Path,                # input/pbt_<class>.csv (e, param, branch, ts, t, value)
        pd_csv: Path,                 # input/pd_<class>.csv  (e, param, period, value)
        pt_csv: Path,                 # input/pt_<class>.csv  (e, param, time, value)
        p_csv: Path,                  # input/p_<class>.csv   (e, param, value)
        period_time_first_csv: Path,  # solve_data/first_timesteps.csv [period, step]
        solve_branch_csv: Path,       # solve_data/solve_branch__time_branch.csv [period, branch]
        period_branch_csv: Path,      # solve_data/period__branch.csv [period, branch]
        group_entity_csv: Path,       # solve_data/group_<class>.csv [group, entity]
        group_stochastic_csv: Path,   # input/groupIncludeStochastics.csv [group]
        param_def1: frozenset[str],
    ) -> None:
        self._pbt = _read_pbt(pbt_csv)
        self._pd = _read_pd(pd_csv)
        self._pt = _read_pt(pt_csv)
        self._p = _read_p(p_csv)
        # d → list[ts] of first-timesteps of period d
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0)
        # d → list[tb] of branches associated with period d
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0)
        # d → list[pe] of parent periods (rows where col1 == d)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1)
        # entity → True if any of its groups is in groupStochastic
        groups_stoch = frozenset(_read_singles(group_stochastic_csv))
        self._stoch_entity: set[str] = set()
        if group_entity_csv.exists():
            with group_entity_csv.open() as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[0] in groups_stoch and row[1]:
                        self._stoch_entity.add(row[1])
        self._param_def1 = param_def1

    def get(self, e: str, param: str, d: str, t: str) -> float:
        # Branch 1: stochastic + outer-d's ts and tb
        if e in self._stoch_entity:
            ts_list = self._ts_for_d.get(d, ())
            tb_list = self._tb_for_d.get(d, ())
            total = 0.0
            hit = False
            for tb in tb_list:
                for ts in ts_list:
                    v = self._pbt.get((e, param, tb, ts, t))
                    if v is not None:
                        total += v
                        hit = True
            if hit:
                return total
        # Branch 2: parent period pe of d, tb from solve_branch[pe], ts from period__time_first[d]
        ts_list = self._ts_for_d.get(d, ())
        pe_list = self._pe_for_d.get(d, ())
        if pe_list and ts_list:
            total = 0.0
            hit = False
            for pe in pe_list:
                for tb in self._tb_for_d.get(pe, ()):
                    for ts in ts_list:
                        v = self._pbt.get((e, param, tb, ts, t))
                        if v is not None:
                            total += v
                            hit = True
            if hit:
                return total
        # Branch 3: period axis
        v = self._pd.get((e, param, d))
        if v is not None:
            return v
        # Branch 4: time axis
        v = self._pt.get((e, param, t))
        if v is not None:
            return v
        # Branch 5: scalar
        v = self._p.get((e, param))
        if v is not None:
            return v
        # Branch 6: default 1
        if param in self._param_def1:
            return 1.0
        # Branch 7: default 0
        return 0.0


PROCESS_PARAM_DEF1 = frozenset({"efficiency", "availability"})  # flextool_base.dat L154


def _read_pbt_3(path: Path) -> dict[tuple[str, str, str, str, str, str], float]:
    """7-col CSV: (e1, e2, param, branch, time_start, time, value).

    Used for pdtProcess_source / pdtProcess_sink whose entity key is
    ``(process, source)`` or ``(process, sink)``.
    """
    out: dict[tuple[str, str, str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 7 and all(row[i] for i in range(6)):
                try:
                    out[(row[0], row[1], row[2], row[3], row[4], row[5])] = float(row[6])
                except ValueError:
                    continue
    return out


def _read_pd_3(path: Path) -> dict[tuple[str, str, str, str], float]:
    """5-col CSV: (e1, e2, param, period, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and all(row[i] for i in range(4)):
                try:
                    out[(row[0], row[1], row[2], row[3])] = float(row[4])
                except ValueError:
                    continue
    return out


def _read_pt_3(path: Path) -> dict[tuple[str, str, str, str], float]:
    """5-col CSV: (e1, e2, param, time, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and all(row[i] for i in range(4)):
                try:
                    out[(row[0], row[1], row[2], row[3])] = float(row[4])
                except ValueError:
                    continue
    return out


def _read_p_3(path: Path) -> dict[tuple[str, str, str], float]:
    """4-col CSV: (e1, e2, param, value)."""
    out: dict[tuple[str, str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and all(row[i] for i in range(3)):
                try:
                    out[(row[0], row[1], row[2])] = float(row[3])
                except ValueError:
                    continue
    return out


class PdtLookupPerSide:
    """6-branch pdt lookup for ``pdtProcess_source`` / ``pdtProcess_sink``
    (flextool.mod L1265, L1279).

    Entity key is ``(process, side)`` where ``side`` is ``source`` or
    ``sink``. No ``processParam_def1`` branch (just ``else 0``).

    Stochastic gate is the same as ``pdtProcess``: process p must belong
    to a group flagged in groupStochastic via group_process.
    """

    def __init__(
        self,
        pbt_csv: Path,                # input/pbt_process_<side>.csv
        pd_csv: Path,                 # input/pd_process_<side>.csv
        pt_csv: Path,                 # input/pt_process_<side>.csv
        p_csv: Path,                  # input/p_process_<side>.csv
        period_time_first_csv: Path,
        solve_branch_csv: Path,
        period_branch_csv: Path,
        group_process_csv: Path,      # solve_data/group_process.csv [group, process]
        group_stochastic_csv: Path,
    ) -> None:
        self._pbt = _read_pbt_3(pbt_csv)
        self._pd = _read_pd_3(pd_csv)
        self._pt = _read_pt_3(pt_csv)
        self._p = _read_p_3(p_csv)
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0)
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1)
        groups_stoch = frozenset(_read_singles(group_stochastic_csv))
        self._stoch_process: set[str] = set()
        if group_process_csv.exists():
            with group_process_csv.open() as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[0] in groups_stoch and row[1]:
                        self._stoch_process.add(row[1])

    def get(self, p: str, side: str, param: str, d: str, t: str) -> float:
        if p in self._stoch_process:
            ts_list = self._ts_for_d.get(d, ())
            tb_list = self._tb_for_d.get(d, ())
            total = 0.0
            hit = False
            for tb in tb_list:
                for ts in ts_list:
                    v = self._pbt.get((p, side, param, tb, ts, t))
                    if v is not None:
                        total += v
                        hit = True
            if hit:
                return total
        ts_list = self._ts_for_d.get(d, ())
        pe_list = self._pe_for_d.get(d, ())
        if pe_list and ts_list:
            total = 0.0
            hit = False
            for pe in pe_list:
                for tb in self._tb_for_d.get(pe, ()):
                    for ts in ts_list:
                        v = self._pbt.get((p, side, param, tb, ts, t))
                        if v is not None:
                            total += v
                            hit = True
            if hit:
                return total
        v = self._pd.get((p, side, param, d))
        if v is not None:
            return v
        v = self._pt.get((p, side, param, t))
        if v is not None:
            return v
        v = self._p.get((p, side, param))
        if v is not None:
            return v
        return 0.0
