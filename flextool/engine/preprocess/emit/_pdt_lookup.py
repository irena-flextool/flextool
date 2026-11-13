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
# Which forecast a branch period reads.  ``branch_time_start`` (period, step, read_first) has one row per period
# from the fan's period onwards (none for a pre-reveal trunk period, none in a solve without a fan); ``step`` is
# the solve's fan start, its analysis time.  Each fold reads, per value cell, the first analysis-time tier that
# authors a value for the cell:
#
# * a period from the fan's period onwards, in a single solve and in a roll of a rolling solve alike
#   (``read_first`` = ``fan``, item 255): the forecast authored at the fan start, else the one at the period's
#   own first step.  A solve or roll decides at its fan's analysis time and must not see a forecast made later,
#   so its later periods -- every row except the fan's own period, whose first step is the fan start -- read the
#   fan's forecast before one made at their own first step (a later roll's or a later solve's analysis time).  A
#   cell the fan-start forecast does not author but the own-step (later) forecast does is an input error, raised
#   once per reader before its value loop (:func:`_check_forecast_covers_horizon`): the fan's forecast must cover
#   the whole horizon.  The own tier therefore never supplies a value there; it stays in the tiers only so the
#   check and the folds share one tier order.  The realization sibling reads its parent's rows and follows the
#   same rule.  A frame that says ``read_first`` = ``own`` (or has no ``read_first`` column) reads the own
#   step first; no solve writes one since item 255.
# * a period before the reveal (no ``branch_time_start`` row) reads only the forecast authored at the solve's or
#   roll's start (its first time step, ``FanStarts.start``), under the realized branch -- every such period, not
#   only the one that holds the start; never the forecast made at a later period's own first step.  The start's
#   forecast must cover every step such a period reads, and a forecasts-only fan (no realized branch) has no
#   series for these periods at all; either is an input error raised once per reader before its value loop
#   (:func:`_check_pre_reveal_reads_have_forecast`), never a silent default.
#
# Non-stochastic (plain time-series) data has no analysis time and is not read through these tiers: unchanged.
# See docs/how_to/stochastics.md "Stochastics over multiple periods" and "Rolling horizon stochastics".
# ---------------------------------------------------------------------------

BRANCH_TIME_START_CSV = "branch_time_start.csv"
READ_FIRST_OWN = "own"  # only read: a hand-built frame; every solve writes ``fan`` (item 255)
READ_FIRST_FAN = "fan"


class FanStarts(dict):
    """``period -> [fan start step]`` of ``branch_time_start``, plus :attr:`fan_first`: the periods whose fold
    reads the fan-start forecast before the forecast at its own first step (``read_first`` = ``fan``, every
    period a solve writes since item 255), and :attr:`start`: the first time step of the solve or roll (the analysis time
    every period before the reveal reads, :func:`ts_tiers`).  A plain dict works wherever a :class:`FanStarts`
    is taken (no fan-first periods, no start: a period before the reveal reads its own first step)."""

    def __init__(self, *args, fan_first: "frozenset[str]" = frozenset(), start: "str | None" = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fan_first = frozenset(fan_first)
        self.start = start


def _solve_start(solve_data_dir: Path, *, provider: "object | None" = None) -> "str | None":
    """The first time step of the solve or roll: the first row of ``solve_data/steps_in_use``; ``None`` when
    absent or empty."""
    df = provider.get(_provider_key(solve_data_dir / "steps_in_use.csv"))
    if df is None or df.height == 0 or df.width < 2:
        return None
    return _cell_str(df.row(0)[1]) or None


def read_fan_starts(path: Path, *, provider: "object | None" = None) -> FanStarts:
    """Read a ``branch_time_start`` frame at *path* into a :class:`FanStarts`; empty when absent or without a
    fan.  A frame without the ``read_first`` column reads as own-first.  :attr:`FanStarts.start` is the first
    step of ``steps_in_use`` in the same folder."""
    out = FanStarts(start=_solve_start(path.parent, provider=provider))
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    fan_first: set[str] = set()
    has_read_first = "read_first" in df.columns
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        d, step = _cell_str(row[0]), _cell_str(row[1])
        if d and step:
            out.setdefault(d, []).append(step)
            if has_read_first and _cell_str(row[2]) == READ_FIRST_FAN:
                fan_first.add(d)
    out.fan_first = frozenset(fan_first)
    return out


def read_fan_for_d(solve_data_dir: Path,
                   *, provider: "object | None" = None) -> FanStarts:
    """``solve_data/branch_time_start`` as a :class:`FanStarts`; empty without a fan."""
    return read_fan_starts(solve_data_dir / BRANCH_TIME_START_CSV, provider=provider)


def ts_tiers(d: str, ts_for_d: "dict[str, list[str]]",
             fan_for_d: "dict[str, list[str]]") -> list[list[str]]:
    """The analysis-time tiers a fold of period *d* reads, in priority order.

    A period with a ``branch_time_start`` row (the fan's period and every later one): ``own`` (the period's
    own first steps, duplicates kept) and ``fan`` (the fan start steps that are not among the own steps), when
    there are any: ``[fan, own]`` for a period in ``fan_for_d.fan_first`` (``read_first`` = ``fan``: every
    period a solve writes, see the section comment above), else ``[own, fan]``.  A fold reads the first tier that has a value for the
    cell.  A period before the reveal (no ``branch_time_start`` row) reads only the forecast made at the
    solve's or roll's start (``fan_for_d.start``): ``[own]`` for the period that holds the start, ``[[start]]``
    for a later one -- never the forecast made at its own (later) first step.  Without a known start (a plain
    dict) it reads ``[own]``.  A period without steps of its own (a period label: no dispatch) reads nothing,
    not the fan start.
    """
    own = list(ts_for_d.get(d, ()))
    if not own:
        return [own]
    fan_steps = fan_for_d.get(d, ())
    if not fan_steps:
        start = getattr(fan_for_d, "start", None)
        return [own] if start is None or start in own else [[start]]
    fan = [ts for ts in fan_steps if ts not in own]
    if not fan:
        return [own]
    return [fan, own] if d in getattr(fan_for_d, "fan_first", ()) else [own, fan]


def fold_over_tiers(pbt: dict, prefix: tuple, tb_groups: "Iterable[Iterable[str]]",
                    tiers: "list[list[str]]", t: str) -> "float | None":
    """Sum ``pbt[(*prefix, tb, ts, t)]`` over the branches of *tb_groups* and the steps of the first tier of
    *tiers* that has at least one value; ``None`` when no tier has one.

    Summation order: tier, then each branch group in order (one per parent period for the parent-period fold),
    its branches, then the tier's steps -- the order the single-tier folds used.
    """
    tb_groups = [list(g) for g in tb_groups]
    for tier in tiers:
        total = 0.0
        hit = False
        for tbs in tb_groups:
            for tb in tbs:
                for ts in tier:
                    v = pbt.get((*prefix, tb, ts, t))
                    if v is not None:
                        total += v
                        hit = True
        if hit:
            return total
    return None


# ---------------------------------------------------------------------------
# A stochastic-time value of a fan: the forecast branch periods take their own
# branch's value; the period label (the reveal / continuation self-row) has no
# dispatch, so it needs no value of its own and a forecasts-only fan is valid.
# A stochastic-time value that authors only part of a fan's forecasts is an
# input error (never a silent class default).  See docs/how_to/stochastics.md
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


# The entity kinds of the stochastic-time input checks below (``kind=``): how a message names the entity
# and how the user puts it in a group with ``include_stochastics``.
def _subject(kind: str, ent: tuple) -> str:
    if kind == "reserve":
        reserve, up_down, group = ent
        return f"reserve '{reserve}' ({up_down}) of group '{group}'"
    if kind == "process_side":
        process, side = ent
        return f"unit or connection '{process}' ({side})"
    return f"{_KIND_NAME[kind]} '{ent[0]}'"


_KIND_NAME = {"node": "node", "process": "unit or connection", "profile": "profile"}

_NOT_GROUPED = {
    "node": "The node is not in a group with include_stochastics",
    "process": "The unit or connection is not in a group with include_stochastics",
    "process_side": "The unit or connection is not in a group with include_stochastics",
    "profile": "No unit, connection or node that uses this profile is in a group with include_stochastics",
    "reserve": "The group of this reserve does not have include_stochastics",
}

_GROUP_FIX = {
    "node": lambda ent: f"Add node '{ent[0]}' to a group that has include_stochastics",
    "process": lambda ent: f"Add '{ent[0]}' to a group that has include_stochastics",
    "process_side": lambda ent: f"Add '{ent[0]}' to a group that has include_stochastics",
    "profile": lambda ent: (f"Add a unit, connection or node that uses profile '{ent[0]}' to a group that has "
                            "include_stochastics"),
    "reserve": lambda ent: f"Set include_stochastics on group '{ent[2]}'",
}


def _check_pbt_branch_coverage(
    pbt: dict[tuple, float],
    n_ent: int,
    anchor_members: dict[str, list[str]],
    tb_for_d: dict[str, list[str]],
    ts_for_d: dict[str, list[str]],
    fan_for_d: "dict[str, list[str]] | None" = None,
    *,
    kind: str,
) -> None:
    """Raise when a pbt entity/param authors SOME of a reveal cohort's active branches but not ALL
    of them -- the engine must never silently fall back to the class default for a branch an entity
    left out (owner decision #82, point 2).

    *pbt* keys are ``(*entity, param, branch, ts, time)`` (``n_ent`` entity-key columns).  A cohort's
    active-branch roster is the fan members' own tokens (*anchor_members* / *tb_for_d*); a roster of 0 or
    1 members can never be "partial", so those cohorts are skipped.  The cohort's time starts are its
    members' first steps (the period label has no steps of its own) and the fan start they fall back to
    (*fan_for_d*, ``branch_time_start``).  *kind* (``node`` / ``process`` / ``process_side`` / ``profile`` /
    ``reserve``) names the entity in the message.
    """
    fan_for_d = fan_for_d or {}
    roster_by_ts: dict[str, frozenset[str]] = {}
    for anchor, members in anchor_members.items():
        tokens = frozenset(tb_for_d[m][0] for m in members if tb_for_d.get(m))
        if len(tokens) <= 1:
            continue
        starts = dict.fromkeys([*ts_for_d.get(anchor, ()), *fan_for_d.get(anchor, ()),
                                *(ts for m in members for ts in (*ts_for_d.get(m, ()), *fan_for_d.get(m, ())))])
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
                f"{_subject(kind, ent)}, parameter '{param}': the forecasts that start at time step '{ts}' "
                f"have values for {sorted(branches & tokens)} but are missing {missing}.  Give a value for "
                "every forecast of the solve, or for none of them: FlexTool does not fill a missing forecast "
                "with the default value.  Check which alternative of the scenario gives this value."
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
    fan_for_d: "dict[str, list[str]] | None" = None,
    kind: str,
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
    members' first steps (the period label has no steps of its own) and the fan start they fall back to
    (*fan_for_d*, ``branch_time_start``).  *kind* (as for :func:`_check_pbt_branch_coverage`) names the entity
    and the way to put it in a group with ``include_stochastics`` in the message.
    """
    fan_for_d = fan_for_d or {}
    for anchor, members in anchor_members.items():
        if tb_for_d.get(anchor):
            continue  # a ``realized: yes`` row (or one projected from a parent) supplies the value.
        starts = frozenset(ts for m in members for ts in (*ts_for_d.get(m, ()), *fan_for_d.get(m, ())))
        if not starts:
            continue
        ungrouped = sorted({(key[:n_ent], key[n_ent]) for key in pbt
                            if key[member_pos] not in stochastic_entities and key[n_ent + 2] in starts})
        if ungrouped:
            ent, param = ungrouped[0]
            raise FlexToolConfigError(
                f"{_subject(kind, ent)}, parameter '{param}': the value has forecasts, and the solve reads it "
                f"in the forecasts that start in period '{anchor}'.  {_NOT_GROUPED[kind]}, so every forecast "
                "reads the value of the forecast marked realized: yes, but no forecast of this solve is marked "
                f"realized: yes.  {_GROUP_FIX[kind](ent)} (then each forecast reads its own value), mark one "
                f"forecast realized: yes in the solve's stochastic_branches, or give '{param}' a value without "
                "forecasts.  Check which alternative of the scenario gives this value."
            )


def _tier_has_value(pbt: dict, prefix: tuple, tb_groups: "list[Iterable[str]]", tier: "list[str]",
                    t: str) -> bool:
    """Whether *tier* authors a value for the cell ``(prefix, t)`` on any branch of *tb_groups* (the hit test
    of :func:`fold_over_tiers`)."""
    return any((*prefix, tb, ts, t) in pbt for tbs in tb_groups for tb in tbs for ts in tier)


_UNCOVERED_SHOWN = 5


def _check_forecast_covers_horizon(
    pbt: dict[tuple, float],
    n_ent: int,
    stochastic_entities: "set[str] | frozenset[str]",
    tb_for_d: dict[str, list[str]],
    ts_for_d: dict[str, list[str]],
    pe_for_d: dict[str, list[str]],
    fan_for_d: "dict[str, list[str]]",
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
    member_pos: int = 0,
    kind: str,
) -> None:
    """Raise when a solve or a roll of a rolling solve would read, in a period after its fan's period, a forecast
    made later than its fan (owner decisions 2026-11-12, item 228, and 2026-11-13, item 255).

    A period after the fan's period (``read_first`` = ``fan``, :func:`ts_tiers` ``[fan, own]``) reads the
    forecast made at the fan start.  A value cell that forecast does not cover would otherwise fall back to the
    forecast made at the period's own first step -- a later roll's or a later solve's analysis time -- so the
    solve would decide on information it does not have yet.  That fallback is an input error: the fan's forecast
    must cover the whole horizon.  A cell neither forecast covers reads the plain value as before.  Without a
    ``fan_first`` period (no fan, or a frame that says ``read_first`` = ``own``) there is nothing to check and the
    function returns at once.

    The check mirrors the readers' folds exactly (branch 1, the stochastic fold over ``tb_for_d[d]``, when
    ``key[member_pos]`` is in *stochastic_entities*; then branch 2, the parent-period fold over the parents'
    branches) on every ``(d, t)`` of ``solve_data/steps_in_use``, so it runs once per reader, before the
    reader's value loop.  *pbt* keys are ``(*entity, param, branch, ts, time)`` (``n_ent`` entity-key
    columns); *kind* (as for :func:`_check_pbt_branch_coverage`) names the entity in the message.
    """
    fan_first = getattr(fan_for_d, "fan_first", frozenset())
    if not fan_first or not pbt:
        return
    times_by_d = _read_pairs_to_dict(solve_data_dir / "steps_in_use.csv", key_col=0, provider=provider)
    tiers_by_d: dict[str, list[list[str]]] = {}
    for d in times_by_d:
        if d in fan_first:
            tiers = ts_tiers(d, ts_for_d, fan_for_d)
            if len(tiers) == 2:
                tiers_by_d[d] = tiers
    if not tiers_by_d:
        return
    # Own (later) analysis time -> the look-ahead periods that would fall back to it.
    d_by_own_ts: dict[str, list[str]] = {}
    for d, (_fan, own) in tiers_by_d.items():
        for ts in dict.fromkeys(own):
            d_by_own_ts.setdefault(ts, []).append(d)
    time_sets = {d: set(times_by_d[d]) for d in tiers_by_d}
    candidates: dict[tuple, set[tuple[str, str]]] = {}
    for key in pbt:
        ts, t = key[n_ent + 2], key[n_ent + 3]
        for d in d_by_own_ts.get(ts, ()):
            if t in time_sets[d]:
                candidates.setdefault(key[:n_ent + 1], set()).add((d, t))
    uncovered: dict[tuple, list[tuple[str, str]]] = {}
    for prefix, cells in candidates.items():
        stochastic = prefix[member_pos] in stochastic_entities
        for d, t in cells:
            fan, own = tiers_by_d[d]
            folds = [[tb_for_d.get(d, ())]] if stochastic else []
            pe_list = pe_for_d.get(d, ())
            if pe_list:
                folds.append([tb_for_d.get(pe, ()) for pe in pe_list])
            for tb_groups in folds:
                if _tier_has_value(pbt, prefix, tb_groups, fan, t):
                    break
                if _tier_has_value(pbt, prefix, tb_groups, own, t):
                    uncovered.setdefault(prefix, []).append((d, t))
                    break
    if not uncovered:
        return
    prefix = min(uncovered)
    ent, param = prefix[:n_ent], prefix[n_ent]
    order = {dt: i for i, dt in enumerate((d, t) for d, ts_list in times_by_d.items() for t in ts_list)}
    cells = sorted(uncovered[prefix], key=order.__getitem__)
    steps = list(dict.fromkeys(t for _d, t in cells))
    periods = list(dict.fromkeys(d for d, _t in cells))
    shown = ", ".join(f"'{t}'" for t in steps[:_UNCOVERED_SHOWN])
    if len(steps) > _UNCOVERED_SHOWN:
        shown += f" and {len(steps) - _UNCOVERED_SHOWN} more"
    fan_steps = ", ".join(f"'{ts}'" for ts in dict.fromkeys(ts for d in periods for ts in tiers_by_d[d][0]))
    own_steps = ", ".join(f"'{ts}'" for ts in dict.fromkeys(ts for d in periods for ts in tiers_by_d[d][1]))
    solve = next(iter(_read_singles(solve_data_dir / "solve_current.csv", provider=provider)), None)
    solve_name = f"solve '{solve}'" if solve else "this solve"
    period_names = ", ".join(f"'{d}'" for d in periods)
    raise FlexToolConfigError(
        f"{_subject(kind, ent)}, parameter '{param}': {solve_name} branches into its forecasts at time step "
        f"{fan_steps} and reads the forecasts made then over its whole horizon, but those forecasts have no "
        f"value at time step{'s' if len(steps) > 1 else ''} {shown} (period{'s' if len(periods) > 1 else ''} "
        f"{period_names}).  Only the forecasts made later, at "
        f"{own_steps}, have values there, and a solve (or a roll of a rolling solve) must not use a forecast "
        f"made after it branches.  Give the forecasts that start at {fan_steps} values for the whole horizon, "
        "look-ahead included.  Check which alternative of the scenario gives this value."
    )


def _check_pre_reveal_reads_have_forecast(
    pbt: dict[tuple, float],
    n_ent: int,
    stochastic_entities: "set[str] | frozenset[str]",
    tb_for_d: dict[str, list[str]],
    ts_for_d: dict[str, list[str]],
    pe_for_d: dict[str, list[str]],
    fan_for_d: "dict[str, list[str]]",
    solve_data_dir: Path,
    *,
    provider: "object | None" = None,
    member_pos: int = 0,
    kind: str,
    steps: "Iterable[tuple[str, str]] | None" = None,
) -> None:
    """Raise when a period before the reveal of this solve or roll would read a stochastic-time value that the
    forecast made at the solve's or roll's start does not give (owner decisions 2026-11-13, item 250; the GMPL
    model had the same check).

    A period before the reveal (no ``branch_time_start`` row: the solve or roll has not fanned out yet) reads, per
    value cell, only the forecast made at the solve's or roll's start (:func:`ts_tiers`: its first time step),
    under the realized branch.  Two input errors, never a silent fallback to the non-stochastic value (usually
    0), to another forecast, or to a forecast made later:

    * the start's forecast has no value for a step such a period reads;
    * the solve or roll fans out into forecasts-only branches (no ``realized: yes`` branch), so the periods before
      the reveal have no branch to read a stochastic value through at all.

    The check mirrors the readers' folds (branch 1, the stochastic fold over ``tb_for_d[d]``, when
    ``key[member_pos]`` is in *stochastic_entities*; then branch 2, the parent-period fold) for every value
    ``(*entity, param)`` of *pbt* (keys ``(*entity, param, branch, ts, time)``, ``n_ent`` entity-key columns).
    *steps* is the reader's ``(period, time)`` grid (default ``solve_data/steps_in_use``); the start is the first
    step of ``steps_in_use``.  Without a period before the reveal (every solve or roll that fans out at its start)
    the function returns at once.  *kind* (as for :func:`_check_pbt_branch_coverage`) names the entity in the
    message.
    """
    if not pbt:
        return
    times_by_d = _read_pairs_to_dict(solve_data_dir / "steps_in_use.csv", key_col=0, provider=provider)
    if not times_by_d:
        return
    if steps is None:
        grid = times_by_d
    else:
        grid = {}
        for d, t in steps:
            grid.setdefault(d, []).append(t)
    pre = [d for d in grid if d not in fan_for_d and ts_for_d.get(d)]
    if not pre:
        return
    start = next(iter(times_by_d.values()))[0]
    solve = next(iter(_read_singles(solve_data_dir / "solve_current.csv", provider=provider)), None)
    solve_name = f"solve '{solve}'" if solve else "this solve"
    prefixes = sorted(dict.fromkeys(key[:n_ent + 1] for key in pbt))
    has_fan = bool(fan_for_d)
    tiers = {d: ts_tiers(d, ts_for_d, FanStarts(fan_for_d, start=start))[0] for d in pre}
    uncovered: dict[tuple, list[tuple[str, str]]] = {}
    read_branches: dict[tuple[tuple, str], list[str]] = {}
    for prefix in prefixes:
        stochastic = prefix[member_pos] in stochastic_entities
        for d in pre:
            folds = [[tb_for_d.get(d, ())]] if stochastic else []
            pe_list = pe_for_d.get(d, ())
            if pe_list:
                folds.append([tb_for_d.get(pe, ()) for pe in pe_list])
            branches = list(dict.fromkeys(tb for tb_groups in folds for tbs in tb_groups for tb in tbs))
            if not branches:
                if has_fan:
                    ent, param = prefix[:n_ent], prefix[n_ent]
                    shown = ", ".join(f"'{t}'" for t in grid[d][:_UNCOVERED_SHOWN])
                    if len(grid[d]) > _UNCOVERED_SHOWN:
                        shown += f" and {len(grid[d]) - _UNCOVERED_SHOWN} more"
                    raise FlexToolConfigError(
                        f"{_subject(kind, ent)}, parameter '{param}': the value has forecasts, and {solve_name} "
                        f"reads it in period '{d}' (time step{'s' if len(grid[d]) > 1 else ''} {shown}), before "
                        "its forecasts branch out.  No forecast of the solve is "
                        "marked realized: yes, so there is no series to read before the forecasts branch out; "
                        "FlexTool does not pick one of the forecasts, nor the default value.  Designate a realized "
                        "branch (realized: yes in the solve's stochastic_branches) -- it provides the shared data "
                        f"before the forecasts branch out, at analysis_time '{start}' -- or give '{param}' a value "
                        "without forecasts.  Check which alternative of the scenario gives this value."
                    )
                continue  # no fan and no realized branch: the readers' other checks.
            read_branches[(prefix, d)] = branches
            for t in grid[d]:
                if not any(_tier_has_value(pbt, prefix, tb_groups, tiers[d], t) for tb_groups in folds):
                    uncovered.setdefault(prefix, []).append((d, t))
    if not uncovered:
        return
    prefix = min(uncovered)
    ent, param = prefix[:n_ent], prefix[n_ent]
    cells = uncovered[prefix]
    steps_shown = list(dict.fromkeys(t for _d, t in cells))
    periods = list(dict.fromkeys(d for d, _t in cells))
    branches = ", ".join(f"'{tb}'" for tb in dict.fromkeys(tb for d in periods for tb in read_branches[(prefix, d)]))
    shown = ", ".join(f"'{t}'" for t in steps_shown[:_UNCOVERED_SHOWN])
    if len(steps_shown) > _UNCOVERED_SHOWN:
        shown += f" and {len(steps_shown) - _UNCOVERED_SHOWN} more"
    period_names = ", ".join(f"'{d}'" for d in periods)
    raise FlexToolConfigError(
        f"{_subject(kind, ent)}, parameter '{param}': {solve_name} starts at time step '{start}', and before its "
        f"forecasts branch out every period reads the forecast made then (analysis_time '{start}', branch "
        f"{branches}).  That forecast has no value at time step{'s' if len(steps_shown) > 1 else ''} {shown} "
        f"(period{'s' if len(periods) > 1 else ''} {period_names}), and FlexTool does not use a forecast made "
        f"later nor the default value instead.  Give forecasts with analysis_time '{start}' that cover these time "
        "steps, or align the solve or roll starts with the forecast times (in a rolling solve, "
        "rolling_solve_jump).  Check which alternative of the scenario gives this value."
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

    def has(self, e: str, param: str, d: str) -> bool:
        """Whether :meth:`get` finds a value for period *d* (branches 1-3) rather than falling back to 0."""
        if (e, param, d) in self._pd or (e, param) in self._p:
            return True
        return any((e, param, db) in self._pd for db in self._branches_for_d.get(d, ()))

    def explicit_rows(self, param: str) -> list[tuple[str, float]]:
        """``(entity, value)`` of every explicit value of *param*: one per period row of ``pd_<class>`` (a period
        Map) and the ``p_<class>`` constant, in file order (pd rows first)."""
        return ([(e, v) for (e, prm, _d), v in self._pd.items() if prm == param]
                + [(e, v) for (e, prm), v in self._p.items() if prm == param])


# ---------------------------------------------------------------------------
# PdtLookup — 7/9-branch period+time fallback.
# ---------------------------------------------------------------------------

class PdtLookup:
    """Mod's 7/9-branch ``pdtProcess`` / ``pdtNode`` resolution, plus the genuine period × time (``pdt``)
    tier.

    Outer index ``(e, param, d, t)``.  Cascade:

        value[e, param, d, t] = fold | pdt[e, param, d, t] | pd[e, param, d] | pt[e, param, t]
                               | p[e, param] | def1 | class_default | 0

    Branches in order:

      1. stochastic-fold: e ∈ groupStochastic-flagged group — sum pbt rows
         over ts ∈ first_timesteps[d], tb ∈ solve_branch[d]; a cell with no
         such row takes ts ∈ branch_time_start[d] (the fan start; in a roll
         of a rolling solve the fan start is read first, :func:`ts_tiers`).
      2. parent-period-fold: for each pe with (pe, d) ∈ period__branch,
         sum pbt rows over ts ∈ first_timesteps[d], tb ∈ solve_branch[pe]
         (same fan-start fallback per cell).
      2b. pdt row (``pdt_process`` / ``pdt_node``; empty when ``pdt_csv``
         is ``None``): a genuine 2-D period × time map, read before
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
        branch_time_start_csv: "Path | None" = None,
    ) -> None:
        self._pbt = _read_pbt(pbt_csv, provider=provider)
        self._pd = _read_pd(pd_csv, provider=provider)
        self._pt = _read_pt(pt_csv, provider=provider)
        self._p = _read_p(p_csv, provider=provider)
        self._pdt = _read_pdt(pdt_csv, provider=provider) if pdt_csv is not None else {}
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0, provider=provider)
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0, provider=provider)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1, provider=provider)
        self._fan_for_d = read_fan_starts(
            branch_time_start_csv if branch_time_start_csv is not None
            else period_time_first_csv.parent / BRANCH_TIME_START_CSV, provider=provider)
        self._tiers = {d: ts_tiers(d, self._ts_for_d, self._fan_for_d)
                       for d in dict.fromkeys([*self._ts_for_d, *self._fan_for_d])}
        anchor_members = _anchor_fan_members(period_branch_csv, provider=provider)
        kind = "node" if Path(group_entity_csv).name == "group__node.csv" else "process"
        _check_pbt_branch_coverage(self._pbt, 1, anchor_members, self._tb_for_d, self._ts_for_d,
                                   self._fan_for_d, kind=kind)
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
                                                    self._tb_for_d, self._ts_for_d, fan_for_d=self._fan_for_d,
                                                    kind=kind)
        _check_pre_reveal_reads_have_forecast(self._pbt, 1, self._stoch_entity, self._tb_for_d, self._ts_for_d,
                                              self._pe_for_d, self._fan_for_d, Path(period_time_first_csv).parent,
                                              provider=provider, kind=kind)
        _check_forecast_covers_horizon(self._pbt, 1, self._stoch_entity, self._tb_for_d, self._ts_for_d,
                                              self._pe_for_d, self._fan_for_d, Path(period_time_first_csv).parent,
                                              provider=provider, kind=kind)
        self._param_def1 = param_def1
        self._time_first = time_first_priority
        self._class_defaults = class_default_values or {}

    def get(self, e: str, param: str, d: str, t: str) -> float:
        tiers = self._tiers.get(d, ())
        # Branch 1: stochastic fold-in (the ts_tiers tiers: own first step and fan start -- per value cell).
        if e in self._stoch_entity:
            v = fold_over_tiers(self._pbt, (e, param), [self._tb_for_d.get(d, ())], tiers, t)
            if v is not None:
                return v
        # Branch 2: parent-period fold-in (same tiers).
        pe_list = self._pe_for_d.get(d, ())
        if pe_list:
            v = fold_over_tiers(self._pbt, (e, param), [self._tb_for_d.get(pe, ()) for pe in pe_list], tiers, t)
            if v is not None:
                return v
        # Branch 2b: genuine period x time map.
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
        branch_time_start_csv: "Path | None" = None,
    ) -> None:
        self._pbt = _read_pbt_3(pbt_csv, provider=provider)
        self._pd = _read_pd_3(pd_csv, provider=provider)
        self._pt = _read_pt_3(pt_csv, provider=provider)
        self._p = _read_p_3(p_csv, provider=provider)
        self._ts_for_d = _read_pairs_to_dict(period_time_first_csv, key_col=0, provider=provider)
        self._tb_for_d = _read_pairs_to_dict(solve_branch_csv, key_col=0, provider=provider)
        self._pe_for_d = _read_pairs_to_dict(period_branch_csv, key_col=1, provider=provider)
        self._fan_for_d = read_fan_starts(
            branch_time_start_csv if branch_time_start_csv is not None
            else period_time_first_csv.parent / BRANCH_TIME_START_CSV, provider=provider)
        self._tiers = {d: ts_tiers(d, self._ts_for_d, self._fan_for_d)
                       for d in dict.fromkeys([*self._ts_for_d, *self._fan_for_d])}
        anchor_members = _anchor_fan_members(period_branch_csv, provider=provider)
        _check_pbt_branch_coverage(self._pbt, 2, anchor_members, self._tb_for_d, self._ts_for_d,
                                   self._fan_for_d, kind="process_side")
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
                                                    self._tb_for_d, self._ts_for_d, fan_for_d=self._fan_for_d,
                                                    kind="process_side")
        _check_pre_reveal_reads_have_forecast(self._pbt, 2, self._stoch_process, self._tb_for_d, self._ts_for_d,
                                              self._pe_for_d, self._fan_for_d, Path(period_time_first_csv).parent,
                                              provider=provider, kind="process_side")
        _check_forecast_covers_horizon(self._pbt, 2, self._stoch_process, self._tb_for_d, self._ts_for_d,
                                              self._pe_for_d, self._fan_for_d, Path(period_time_first_csv).parent,
                                              provider=provider, kind="process_side")

    def get(self, p: str, side: str, param: str, d: str, t: str) -> float:
        tiers = self._tiers.get(d, ())
        if p in self._stoch_process:
            v = fold_over_tiers(self._pbt, (p, side, param), [self._tb_for_d.get(d, ())], tiers, t)
            if v is not None:
                return v
        pe_list = self._pe_for_d.get(d, ())
        if pe_list:
            v = fold_over_tiers(self._pbt, (p, side, param), [self._tb_for_d.get(pe, ()) for pe in pe_list],
                                tiers, t)
            if v is not None:
                return v
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
