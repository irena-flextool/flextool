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
from typing import Iterable

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.preprocess._provider_io import _provider_key


# ---------------------------------------------------------------------------
# Native-frame row helpers.
#
# These read the in-memory polars frame directly from the Provider via
# ``provider.get(_provider_key(path))`` instead of round-tripping through
# CSV text (the former write-CSV + ``csv.reader`` path).
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


def _read_pdt(path: Path,
               *, provider: "object | None" = None) -> dict[tuple[str, str, str, str], float]:
    """5-col frame: (entity, param, period, time, value)."""
    out: dict[tuple[str, str, str, str], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 5:
            continue
        c0, c1, c2, c3 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2]), _cell_str(row[3])
        if c0 and c1 and c2 and c3:
            try:
                out[(c0, c1, c2, c3)] = float(row[4])
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
# A stochastic-time value of a fan: the forecast branch periods take their own
# branch's value; the period label (the reveal / continuation self-row) has no
# dispatch, so it needs no value of its own and a forecasts-only fan is valid.
# A stochastic-time value that authors only part of a fan's forecasts is an
# input error (never a silent class default).  See docs/how_to.md
# "Stochastics over multiple periods".
# ---------------------------------------------------------------------------

def _anchor_fan_members(period_branch_csv: Path,
                         *, provider: "object | None" = None) -> dict[str, list[str]]:
    """Anchor period -> its fan members (``period__branch`` rows, self row excluded).

    An anchor that branched (at the reveal or at a later continuation period) owns one row per
    fan member -- realized and zero-weight branches included, per ``_stochastic.py``'s main walk.
    A period that never branches (deterministic, or strictly pre-reveal) is absent.
    """
    out: dict[str, list[str]] = {}
    for anchor, members in _read_pairs_to_dict(period_branch_csv, key_col=0, provider=provider).items():
        kept = [m for m in members if m != anchor]
        if kept:
            out[anchor] = kept
    return out


def _check_pbt_branch_coverage(
    pbt: dict[tuple, float],
    n_ent: int,
    anchor_members: dict[str, list[str]],
    tb_for_d: dict[str, list[str]],
    ts_for_d: dict[str, list[str]],
) -> None:
    """Raise when a pbt entity/param authors SOME of a reveal cohort's active branches but not ALL
    of them -- the engine must never silently fall back to the class default for a branch an entity
    left out (owner decision #82, point 2).

    *pbt* keys are ``(*entity, param, branch, ts, time)`` (``n_ent`` entity-key columns).  A cohort's
    active-branch roster is the fan members' own tokens (*anchor_members* / *tb_for_d*); a roster of 0 or
    1 members can never be "partial", so those cohorts are skipped.  The cohort's time starts are its
    members' first steps (the period label has no steps of its own).
    """
    roster_by_ts: dict[str, frozenset[str]] = {}
    for anchor, members in anchor_members.items():
        tokens = frozenset(tb_for_d[m][0] for m in members if tb_for_d.get(m))
        if len(tokens) <= 1:
            continue
        starts = dict.fromkeys([*ts_for_d.get(anchor, ()), *(ts for m in members for ts in ts_for_d.get(m, ()))])
        for ts in starts:
            roster_by_ts[ts] = tokens
    if not roster_by_ts:
        return
    authored: dict[tuple, set[str]] = {}
    for key in pbt:
        ts = key[n_ent + 2]
        if ts not in roster_by_ts:
            continue
        branch = key[n_ent + 1]
        authored.setdefault((key[:n_ent], key[n_ent], ts), set()).add(branch)
    for (ent, param, ts), branches in authored.items():
        tokens = roster_by_ts[ts]
        if branches & tokens and branches < tokens:
            missing = sorted(tokens - branches)
            raise FlexToolConfigError(
                f"{'/'.join(ent)}.{param}: a stochastic-time value authors branch(es) "
                f"{sorted(branches & tokens)} of this reveal but is missing {missing} -- give every "
                "active forecast branch a value, or none; the engine never silently substitutes the "
                "class default for a branch left out."
            )


def _check_ungrouped_reads_have_realized_branch(
    pbt: dict[tuple, float],
    n_ent: int,
    stochastic_entities: "set[str] | frozenset[str]",
    anchor_members: dict[str, list[str]],
    tb_for_d: dict[str, list[str]],
    ts_for_d: dict[str, list[str]],
    *,
    member_pos: int = 0,
) -> None:
    """Raise when an entity outside every ``include_stochastics`` group authors a stochastic-time value
    that a forecasts-only fan of this solve would read.

    Such an entity reads, on every branch period, the single value of the fan's realized branch (the
    parent-period fold over the period label's ``solve_branch__time_branch`` rows).  A forecasts-only fan
    (no ``realized: yes`` designation) has no such row, so the value would silently become the class
    default; the engine never substitutes the class default for an authored value.

    *pbt* keys are ``(*entity, param, branch, ts, time)`` (``n_ent`` entity-key columns); group membership
    is tested on ``key[member_pos]``: the process (``0``) for the per-side tables, the group (``2``) for the
    ``(reserve, upDown, group)`` reservation table.  The fan's analysis times are its
    members' first steps (the period label has no steps of its own).
    """
    for anchor, members in anchor_members.items():
        if tb_for_d.get(anchor):
            continue  # a ``realized: yes`` row (or one projected from a parent) supplies the value.
        starts = frozenset(ts for m in members for ts in ts_for_d.get(m, ()))
        if not starts:
            continue
        ungrouped = sorted({(key[:n_ent], key[n_ent]) for key in pbt
                            if key[member_pos] not in stochastic_entities and key[n_ent + 2] in starts})
        if ungrouped:
            ent, param = ungrouped[0]
            raise FlexToolConfigError(
                f"{'/'.join(ent)}.{param}: a stochastic-time (rank-3) value of an entity in no "
                f"include_stochastics group, read in the forecasts-only fan of period {anchor!r}.  Outside "
                "such a group the value of the realized branch is read on every branch period, and no branch "
                "of this fan is designated realized: yes.  Add the entity to an include_stochastics group (for a "
                "reservation, set include_stochastics on its reserve group; each forecast then reads its own "
                "branch), designate a realized: yes branch, or give the parameter a non-stochastic value."
            )


# ---------------------------------------------------------------------------
# Branch periods inherit the anchor period's ``pd`` rows.
# ---------------------------------------------------------------------------

def inherit_pd_branch_periods(pd: dict, pe_for_d: "dict[str, list[str]]", *,
                              key_cols: "list[str]", period_pos: int,
                              periods: "Iterable[str]") -> dict:
    """``pd`` plus, for each branch period in ``periods``, its anchor period's entries.

    A stochastic branch period is minted per solve, so no ``pd`` entry is keyed by it; it takes the
    anchor's entry unless it has its own for that key (branch-native wins per key).  Pairs come from
    ``pe_for_d`` (branch -> anchors, ``period__branch`` read with ``key_col=1``); self rows are
    dropped and only anchors with at least one ``pd`` entry count, so a pair that is irrelevant to
    this dict never reaches the one-anchor-per-branch check of
    :func:`~flextool.engine.preprocess.params._branch.inherit_anchor_period` (which raises on a real
    conflict).  ``key_cols`` names the key positions, ``period_pos`` the period's.

    Returns ``pd`` itself when nothing is added (deterministic solves: every pair is a self row; no
    polars frame is built then).  Otherwise a new dict: the authored entries first, the inherited
    ones after.  Both the legacy loops and their vectorized twins read the result.
    """
    periods = set(periods)
    candidates = sorted({(a, br) for br, anchors in pe_for_d.items() if br in periods
                         for a in anchors if a != br})
    if not candidates:
        return pd
    anchors = {a for a, _ in candidates}
    anchors_with_rows = {k[period_pos] for k in pd if k[period_pos] in anchors}
    candidates = [(a, br) for a, br in candidates if a in anchors_with_rows]
    if not candidates:
        return pd
    import polars as pl

    from flextool.engine.preprocess.params._branch import inherit_anchor_period

    involved = anchors_with_rows | {br for _, br in candidates}
    rows = [(*k, v) for k, v in pd.items() if k[period_pos] in involved]
    frame = pl.DataFrame(rows, schema={**{c: pl.Utf8 for c in key_cols}, "value": pl.Float64},
                         orient="row")
    pairs = pl.DataFrame(candidates, schema={"anchor": pl.Utf8, "br": pl.Utf8}, orient="row")
    group_keys = [c for i, c in enumerate(key_cols) if i != period_pos]
    out = inherit_anchor_period(frame, pairs, group_keys=group_keys, period_col=key_cols[period_pos])
    if out.height == frame.height:
        return pd
    merged = dict(pd)
    for row in out.slice(frame.height).iter_rows():
        merged[tuple(row[:-1])] = row[-1]
    return merged


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
    """Mod's 7/9-branch ``pdtProcess`` / ``pdtNode`` resolution, plus the ``pdtProcess``-only genuine
    period × time (``pdt``) tier (Q13).

    Outer index ``(e, param, d, t)``.  Cascade:

        value[e, param, d, t] = fold | pdt[e, param, d, t] | pd[e, param, d] | pt[e, param, t]
                               | p[e, param] | def1 | class_default | 0

    Branches in order:

      1. stochastic-fold: e ∈ groupStochastic-flagged group — sum pbt rows
         over ts ∈ first_timesteps[d], tb ∈ solve_branch[d].
      2. parent-period-fold: for each pe with (pe, d) ∈ period__branch,
         sum pbt rows over ts ∈ first_timesteps[d], tb ∈ solve_branch[pe].
      2b. pdt row (``pdtProcess`` only — ``pdt_csv`` is ``None`` for
         ``pdtNode``): a genuine 2-D period × time map, read before
         pd/pt so it outranks the 1-D tiers for an entity that authors
         more than one shape of the same parameter.  A stochastic branch
         period without its own pdt row uses its anchor period's row
         (:func:`inherit_pd_branch_periods`, same as the pd tier).
      3-4. pt or pd row (order toggled by ``time_first_priority``;
         pdtNode prefers time, pdtProcess prefers period).  A stochastic
         branch period without a pd row of its own uses its anchor
         period's pd row (:func:`inherit_pd_branch_periods`).
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
        pdt_csv: "Path | None" = None,
        time_first_priority: bool = False,
        class_default_values: dict[str, float] | None = None,
        *,
        provider: "object | None" = None,
    ) -> None:
        self._pbt = _read_pbt(pbt_csv, provider=provider)
        self._pd = _read_pd(pd_csv, provider=provider)
        self._pt = _read_pt(pt_csv, provider=provider)
        self._p = _read_p(p_csv, provider=provider)
        self._pdt = _read_pdt(pdt_csv, provider=provider) if pdt_csv is not None else {}
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0, provider=provider)
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0, provider=provider)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1, provider=provider)
        anchor_members = _anchor_fan_members(period_branch_csv, provider=provider)
        _check_pbt_branch_coverage(self._pbt, 1, anchor_members, self._tb_for_d, self._ts_for_d)
        self._pd = inherit_pd_branch_periods(self._pd, self._pe_for_d, key_cols=["e", "param", "d"],
                                             period_pos=2, periods=list(self._ts_for_d))
        self._pdt = inherit_pd_branch_periods(self._pdt, self._pe_for_d, key_cols=["e", "param", "d", "t"],
                                              period_pos=2, periods=list(self._ts_for_d))
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
        _check_ungrouped_reads_have_realized_branch(self._pbt, 1, self._stoch_entity, anchor_members,
                                                    self._tb_for_d, self._ts_for_d)
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
        # Branch 2b: genuine period x time map (``pdtProcess`` only; always empty for ``pdtNode``).
        v = self._pdt.get((e, param, d, t))
        if v is not None:
            return v
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
    layers (branch periods inherit the anchor's pd rows the same way).
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
        anchor_members = _anchor_fan_members(period_branch_csv, provider=provider)
        _check_pbt_branch_coverage(self._pbt, 2, anchor_members, self._tb_for_d, self._ts_for_d)
        self._pd = inherit_pd_branch_periods(self._pd, self._pe_for_d,
                                             key_cols=["p", "side", "param", "d"], period_pos=3,
                                             periods=list(self._ts_for_d))
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
        _check_ungrouped_reads_have_realized_branch(self._pbt, 2, self._stoch_process, anchor_members,
                                                    self._tb_for_d, self._ts_for_d)

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
