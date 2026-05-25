"""Per-(d, t) fallback lookups used by ``entity_period_calc_params``.

Three lookup classes implementing the 4-/6-/7-branch fallback chains:

* :class:`PdLookup`         — 4-branch period-only resolution (used by
  ``pdNode`` / ``pdProcess``).
* :class:`PdtLookup`         — 7- or 9-branch period+time fallback over
  pbt/pd/pt/p + class-level defaults.
* :class:`PdtLookupPerSide`  — 6-branch period+time fallback for the
  (process, side) per-arc tables.

Why a class (not a polars-lazy frame)?
--------------------------------------

Consumers call ``.get(entity, param, period, time)`` per output row in
the inner emission loop.  The lookup is a layered
``dict[tuple, float]`` cascade — optimal for the access pattern.
A polars-lazy join chain would be slower at the small fixture sizes
used by these writers and would obscure the 4-/7-branch semantics.

The CSV-read helpers below return plain dicts so the ``.get()`` hot
path stays pure Python (no polars overhead per call).

Branching semantics:
  * 4-branch (``PdLookup``): pd row → branch-sum → p scalar → 0.
  * 7-branch (``PdtLookup``): stochastic-fold → parent-period-fold →
    pd or pt (order toggled by ``time_first_priority``) → p scalar →
    ``param ∈ param_def1`` → class default → 0.
  * 6-branch (``PdtLookupPerSide``): stochastic-fold → parent-period-fold
    → pd → pt → p → 0.
"""
from __future__ import annotations

import csv
from pathlib import Path

from flextool.engine_polars._emit_provider_io import (
    _provider_key,
    _provider_open,
)


# ---------------------------------------------------------------------------
# Shared CSV readers (mirror legacy helpers byte-for-byte).
# ---------------------------------------------------------------------------

def _read_pd(path: Path,
              *, provider: "object | None" = None) -> dict[tuple[str, str, str], float]:
    """4-col CSV: (entity, param, period, value)."""
    out: dict[tuple[str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and row[0] and row[1] and row[2]:
                try:
                    out[(row[0], row[1], row[2])] = float(row[3])
                except ValueError:
                    continue
    return out


def _read_p(path: Path,
             *, provider: "object | None" = None) -> dict[tuple[str, str], float]:
    """3-col CSV: (entity, param, value)."""
    out: dict[tuple[str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1]:
                try:
                    out[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
    return out


def _read_pt(path: Path,
              *, provider: "object | None" = None) -> dict[tuple[str, str, str], float]:
    """4-col CSV: (entity, param, time, value)."""
    out: dict[tuple[str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and row[0] and row[1] and row[2]:
                try:
                    out[(row[0], row[1], row[2])] = float(row[3])
                except ValueError:
                    continue
    return out


def _read_pbt(path: Path,
               *, provider: "object | None" = None) -> dict[tuple[str, str, str, str, str], float]:
    """6-col CSV: (entity, param, branch, time_start, time, value)."""
    out: dict[tuple[str, str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 6 and row[0] and row[1] and row[2] and row[3] and row[4]:
                try:
                    out[(row[0], row[1], row[2], row[3], row[4])] = float(row[5])
                except ValueError:
                    continue
    return out


def _read_pd_3(path: Path,
                *, provider: "object | None" = None) -> dict[tuple[str, str, str, str], float]:
    """5-col CSV: (e1, e2, param, period, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and all(row[i] for i in range(4)):
                try:
                    out[(row[0], row[1], row[2], row[3])] = float(row[4])
                except ValueError:
                    continue
    return out


def _read_pt_3(path: Path,
                *, provider: "object | None" = None) -> dict[tuple[str, str, str, str], float]:
    """5-col CSV: (e1, e2, param, time, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 5 and all(row[i] for i in range(4)):
                try:
                    out[(row[0], row[1], row[2], row[3])] = float(row[4])
                except ValueError:
                    continue
    return out


def _read_p_3(path: Path,
               *, provider: "object | None" = None) -> dict[tuple[str, str, str], float]:
    """4-col CSV: (e1, e2, param, value)."""
    out: dict[tuple[str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 4 and all(row[i] for i in range(3)):
                try:
                    out[(row[0], row[1], row[2])] = float(row[3])
                except ValueError:
                    continue
    return out


def _read_pbt_3(path: Path,
                 *, provider: "object | None" = None) -> dict[tuple[str, str, str, str, str, str], float]:
    """7-col CSV: (e1, e2, param, branch, time_start, time, value)."""
    out: dict[tuple[str, str, str, str, str, str], float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 7 and all(row[i] for i in range(6)):
                try:
                    out[(row[0], row[1], row[2], row[3], row[4], row[5])] = float(row[6])
                except ValueError:
                    continue
    return out


def _read_pairs_to_dict(path: Path, key_col: int,
                         *, provider: "object | None" = None) -> dict[str, list[str]]:
    """Generic two-col CSV → ``key_col → list[other_col]``."""
    out: dict[str, list[str]] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    other_col = 1 - key_col
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.setdefault(row[key_col], []).append(row[other_col])
    return out


def _read_period_branch(path: Path,
                         *, provider: "object | None" = None) -> dict[str, list[str]]:
    """Build ``d → [db]`` from ``period__branch.csv`` (col0=db, col1=d)."""
    out: dict[str, list[str]] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.setdefault(row[1], []).append(row[0])
    return out


def _read_singles(path: Path,
                   *, provider: "object | None" = None) -> list[str]:
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return []
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


# ---------------------------------------------------------------------------
# Param-default enums (mirror flextool_base.dat — pinned here to avoid
# importing the legacy preprocessing tree).
# ---------------------------------------------------------------------------

PROCESS_PARAM_DEF1: frozenset[str] = frozenset({"efficiency", "availability"})
NODE_PARAM_DEF1: frozenset[str] = frozenset({"availability"})


def read_class_defaults(path: Path, class_name: str,
                         *, provider: "object | None" = None) -> dict[str, float]:
    """Read ``input/default_values.csv`` filtered to ``class_name``."""
    out: dict[str, float] = {}
    seeded = _provider_open(provider, _provider_key(path), path)
    if seeded is None:
        return out
    with seeded as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] == class_name and row[1]:
                try:
                    out[row[1]] = float(row[2])
                except ValueError:
                    continue
    return out


# ---------------------------------------------------------------------------
# PdLookup — 4-branch period-only resolution.
# ---------------------------------------------------------------------------

class PdLookup:
    """Mod's 4-branch period-param resolution (pdProcess / pdNode @ L1252/L1216).

    Branch 1: explicit (e, param, d) row in ``pd_<class>``.
    Branch 2: any ``(e, param, db)`` where ``(db, d)`` ∈ period__branch — sum.
    Branch 3: scalar fallback in ``p_<class>``.
    Branch 4: 0.
    """

    def __init__(
        self,
        pd_csv: Path,             # input/pd_<class>.csv
        p_csv: Path,              # input/p_<class>.csv
        period_branch_csv: Path,  # solve_data/period__branch.csv
        *,
        provider: "object | None" = None,
    ) -> None:
        self._pd = _read_pd(pd_csv, provider=provider)
        self._p = _read_p(p_csv, provider=provider)
        self._branches_for_d = _read_period_branch(period_branch_csv, provider=provider)

    def get(self, e: str, param: str, d: str) -> float:
        if (e, param, d) in self._pd:
            return self._pd[(e, param, d)]
        branches = self._branches_for_d.get(d, ())
        branched = [
            self._pd[(e, param, db)]
            for db in branches
            if (e, param, db) in self._pd
        ]
        if branched:
            return sum(branched)
        if (e, param) in self._p:
            return self._p[(e, param)]
        return 0.0


# ---------------------------------------------------------------------------
# PdtLookup — 7/9-branch period+time fallback.
# ---------------------------------------------------------------------------

class PdtLookup:
    """Mod's 7/9-branch ``pdtProcess`` / ``pdtNode`` resolution.

    Outer index ``(e, param, d, t)``.  Branches in order:

      1. stochastic-fold: e ∈ groupStochastic-flagged group — sum pbt rows
         over ts ∈ first_timesteps[d], tb ∈ solve_branch[d].
      2. parent-period-fold: for each pe with (pe, d) ∈ period__branch,
         sum pbt rows over ts ∈ first_timesteps[d], tb ∈ solve_branch[pe].
      3-4. pt or pd row (order toggled by ``time_first_priority``;
         pdtNode prefers time, pdtProcess prefers period).
      5. p scalar.
      6. ``param ∈ param_def1`` → 1.
      7. class-default (pdtNode only).
      8. 0.
    """

    def __init__(
        self,
        pbt_csv: Path,
        pd_csv: Path,
        pt_csv: Path,
        p_csv: Path,
        period_time_first_csv: Path,
        solve_branch_csv: Path,
        period_branch_csv: Path,
        group_entity_csv: Path,
        group_stochastic_csv: Path,
        param_def1: frozenset[str],
        time_first_priority: bool = False,
        class_default_values: dict[str, float] | None = None,
        *,
        provider: "object | None" = None,
    ) -> None:
        self._pbt = _read_pbt(pbt_csv, provider=provider)
        self._pd = _read_pd(pd_csv, provider=provider)
        self._pt = _read_pt(pt_csv, provider=provider)
        self._p = _read_p(p_csv, provider=provider)
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0, provider=provider)
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0, provider=provider)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1, provider=provider)
        groups_stoch = frozenset(_read_singles(group_stochastic_csv, provider=provider))
        self._stoch_entity: set[str] = set()
        ge_seeded = _provider_open(provider, _provider_key(group_entity_csv), group_entity_csv)
        if ge_seeded is not None:
            with ge_seeded as fh:
                reader = csv.reader(fh)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[0] in groups_stoch and row[1]:
                        self._stoch_entity.add(row[1])
        self._param_def1 = param_def1
        self._time_first = time_first_priority
        self._class_defaults = class_default_values or {}

    def get(self, e: str, param: str, d: str, t: str) -> float:
        # Branch 1: stochastic fold-in.
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
        # Branch 2: parent-period fold-in.
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
        # Branches 3-4: pt vs pd order toggle.
        if self._time_first:
            v = self._pt.get((e, param, t))
            if v is not None:
                return v
            v = self._pd.get((e, param, d))
            if v is not None:
                return v
        else:
            v = self._pd.get((e, param, d))
            if v is not None:
                return v
            v = self._pt.get((e, param, t))
            if v is not None:
                return v
        # Branch 5: scalar.
        v = self._p.get((e, param))
        if v is not None:
            return v
        # Branch 6: param_def1 default-1.
        if param in self._param_def1:
            return 1.0
        # Branch 7: class default.
        if param in self._class_defaults:
            return self._class_defaults[param]
        # Branch 8: 0.
        return 0.0


# ---------------------------------------------------------------------------
# PdtLookupPerSide — 6-branch for (process, side) entities.
# ---------------------------------------------------------------------------

class PdtLookupPerSide:
    """6-branch pdt lookup for ``pdtProcess_source`` / ``pdtProcess_sink``.

    Entity key is ``(process, side)``.  No def1 branch; otherwise same
    cascade as :class:`PdtLookup` minus the param_def1 / class-default
    layers.
    """

    def __init__(
        self,
        pbt_csv: Path,
        pd_csv: Path,
        pt_csv: Path,
        p_csv: Path,
        period_time_first_csv: Path,
        solve_branch_csv: Path,
        period_branch_csv: Path,
        group_process_csv: Path,
        group_stochastic_csv: Path,
        *,
        provider: "object | None" = None,
    ) -> None:
        self._pbt = _read_pbt_3(pbt_csv, provider=provider)
        self._pd = _read_pd_3(pd_csv, provider=provider)
        self._pt = _read_pt_3(pt_csv, provider=provider)
        self._p = _read_p_3(p_csv, provider=provider)
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0, provider=provider)
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0, provider=provider)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1, provider=provider)
        groups_stoch = frozenset(_read_singles(group_stochastic_csv, provider=provider))
        self._stoch_process: set[str] = set()
        gp_seeded = _provider_open(provider, _provider_key(group_process_csv), group_process_csv)
        if gp_seeded is not None:
            with gp_seeded as fh:
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
