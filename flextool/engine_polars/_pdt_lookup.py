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

from pathlib import Path

from flextool.engine_polars._emit_provider_io import _provider_key


# ---------------------------------------------------------------------------
# Native-frame row helpers.
#
# These read the in-memory polars frame directly from the Provider via
# ``provider.get(_provider_key(path))`` instead of round-tripping through
# CSV text (the legacy ``_provider_open`` + ``csv.reader`` path).
#
# Type-fidelity contract — reproduce *exactly* what ``csv.reader`` over a
# ``DataFrame.write_csv`` serialisation would have yielded:
#
#   * Key columns (the dict-key positions) were ``csv.reader`` strings.
#     ``write_csv`` serialises ``null`` → ``""`` and any scalar (Enum /
#     Int / Float / Utf8) → its string form.  We therefore coerce each
#     key cell with :func:`_cell_str` (``None`` → ``""``, else ``str``)
#     and apply the original truthiness guard to the *string* form so a
#     null cell is skipped (matching the legacy ``if row[i]`` test) while
#     a literal ``"0"`` is kept.
#   * Value columns were re-coerced with ``float(...)``.  We apply
#     ``float(...)`` to the native cell — harmless on an already-float
#     frame, necessary on an int frame, and identical to the legacy
#     ``float(str_cell)`` on a stringified-value frame.  A value that
#     cannot be parsed as a float is skipped (matching the legacy
#     ``except ValueError: continue``).
#
# ``provider.get`` returns data rows only (no header), so there is no
# header row to skip; an empty / missing frame yields the same empty
# dict / list the legacy loop produced.
# ---------------------------------------------------------------------------

def _cell_str(value: "object | None") -> str:
    """Reproduce a ``csv.reader`` cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; ``csv.reader`` then reads those
    strings back.  Mirror that here so dict keys stay byte-identical to
    the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


def _read_pd(path: Path,
              *, provider: "object | None" = None) -> dict[tuple[str, str, str], float]:
    """4-col frame: (entity, param, period, value)."""
    out: dict[tuple[str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 4:
            continue
        c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
        if c0 and c1 and c2:
            try:
                out[(c0, c1, c2)] = float(row[3])
            except (ValueError, TypeError):
                continue
    return out


def _read_p(path: Path,
             *, provider: "object | None" = None) -> dict[tuple[str, str], float]:
    """3-col frame: (entity, param, value)."""
    out: dict[tuple[str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            try:
                out[(c0, c1)] = float(row[2])
            except (ValueError, TypeError):
                continue
    return out


def _read_pt(path: Path,
              *, provider: "object | None" = None) -> dict[tuple[str, str, str], float]:
    """4-col frame: (entity, param, time, value)."""
    out: dict[tuple[str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 4:
            continue
        c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
        if c0 and c1 and c2:
            try:
                out[(c0, c1, c2)] = float(row[3])
            except (ValueError, TypeError):
                continue
    return out


def _read_pbt(path: Path,
               *, provider: "object | None" = None) -> dict[tuple[str, str, str, str, str], float]:
    """6-col frame: (entity, param, branch, time_start, time, value)."""
    out: dict[tuple[str, str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 6:
            continue
        c = [_cell_str(row[i]) for i in range(5)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3], c[4])] = float(row[5])
            except (ValueError, TypeError):
                continue
    return out


def _read_pd_3(path: Path,
                *, provider: "object | None" = None) -> dict[tuple[str, str, str, str], float]:
    """5-col frame: (e1, e2, param, period, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 5:
            continue
        c = [_cell_str(row[i]) for i in range(4)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3])] = float(row[4])
            except (ValueError, TypeError):
                continue
    return out


def _read_pt_3(path: Path,
                *, provider: "object | None" = None) -> dict[tuple[str, str, str, str], float]:
    """5-col frame: (e1, e2, param, time, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 5:
            continue
        c = [_cell_str(row[i]) for i in range(4)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3])] = float(row[4])
            except (ValueError, TypeError):
                continue
    return out


def _read_p_3(path: Path,
               *, provider: "object | None" = None) -> dict[tuple[str, str, str], float]:
    """4-col frame: (e1, e2, param, value)."""
    out: dict[tuple[str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 4:
            continue
        c = [_cell_str(row[i]) for i in range(3)]
        if all(c):
            try:
                out[(c[0], c[1], c[2])] = float(row[3])
            except (ValueError, TypeError):
                continue
    return out


def _read_pbt_3(path: Path,
                 *, provider: "object | None" = None) -> dict[tuple[str, str, str, str, str, str], float]:
    """7-col frame: (e1, e2, param, branch, time_start, time, value)."""
    out: dict[tuple[str, str, str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 7:
            continue
        c = [_cell_str(row[i]) for i in range(6)]
        if all(c):
            try:
                out[(c[0], c[1], c[2], c[3], c[4], c[5])] = float(row[6])
            except (ValueError, TypeError):
                continue
    return out


def _read_pairs_to_dict(path: Path, key_col: int,
                         *, provider: "object | None" = None) -> dict[str, list[str]]:
    """Generic two-col frame → ``key_col → list[other_col]``."""
    out: dict[str, list[str]] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    other_col = 1 - key_col
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.setdefault((c0, c1)[key_col], []).append((c0, c1)[other_col])
    return out


def _read_period_branch(path: Path,
                         *, provider: "object | None" = None) -> dict[str, list[str]]:
    """Build ``d → [db]`` from ``period__branch`` (col0=db, col1=d)."""
    out: dict[str, list[str]] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.setdefault(c1, []).append(c0)
    return out


def _read_singles(path: Path,
                   *, provider: "object | None" = None) -> list[str]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[str] = []
    for row in df.iter_rows():
        if not row:
            continue
        c0 = _cell_str(row[0])
        if c0:
            out.append(c0)
    return out


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
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 == class_name and c1:
            try:
                out[c1] = float(row[2])
            except (ValueError, TypeError):
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
        ge_df = provider.get(_provider_key(group_entity_csv))
        if ge_df is not None:
            for row in ge_df.iter_rows():
                if len(row) < 2:
                    continue
                c0, c1 = _cell_str(row[0]), _cell_str(row[1])
                if c0 in groups_stoch and c1:
                    self._stoch_entity.add(c1)
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
        gp_df = provider.get(_provider_key(group_process_csv))
        if gp_df is not None:
            for row in gp_df.iter_rows():
                if len(row) < 2:
                    continue
                c0, c1 = _cell_str(row[0]), _cell_str(row[1])
                if c0 in groups_stoch and c1:
                    self._stoch_process.add(c1)

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
