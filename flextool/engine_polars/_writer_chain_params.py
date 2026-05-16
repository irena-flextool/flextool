"""Writer-port Phase 1 follow-up 8 — chain-cluster entity-period params.

Native ports of four ``write_*`` helpers in
:mod:`flextool.flextoolrunner.preprocessing.entity_period_calc_params`
that emit the per-(entity, period) existing/divest/capacity-max chain
that the LP build consumes via ``p_entity_*_existing_*`` and
``p_entity_*_capacity_*`` parameters:

* :func:`write_p_entity_pre_existing`           (legacy L205 — 12-branch).
* :func:`write_p_entity_divest_cumulative_max`  (legacy L354 — 3-branch).
* :func:`write_p_entity_existing_chain`         (legacy L1381 — 5 outputs).
* :func:`write_p_entity_capacity_max_chain`     (legacy L1594 — 4 outputs).

The chain-existing / chain-cap-max writers are the most state-dependent
writers in Phase 1: they straddle the per-solve handoff boundary (prior
solve's ``realized_existing``/``realized_invest``/``divest_cumulative``
either as in-memory ``SolveHandoff`` carriers or as CSVs left by the
parent solve's output writer).  We keep the legacy dict-keyed lookup
shape — it's already optimal for the per-row access pattern and the
intricate branching is much easier to verify against the mod source
when written procedurally.

Byte-for-byte parity with the legacy emitters is the gate; tests live
in ``tests/engine_polars/test_writer_port_phase1.py``.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from ._solve_handoff import SolveHandoff


# ---------------------------------------------------------------------------
# Canonical writer-port emitter — mirrors the ``_write(df, path)`` idiom
# in :mod:`._writer_arc_unions` and the four other patched modules.  All
# writers in this module funnel their derived ``(entity, period, value)``
# frames through this helper so :mod:`._flex_data_accumulator` can capture
# them in-memory via its monkey-patch.
# ---------------------------------------------------------------------------


def _write(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _ed_value_frame(
    rows: list[tuple[str, str, object]],
) -> pl.DataFrame:
    """Build a 3-col ``(entity, period, value)`` Utf8 frame whose ``value``
    cells are ``repr(v)`` strings — preserving the *Python type-as-emitted*
    semantics of the legacy ``f"{e},{d},{repr(v)}\\n"`` line builders.

    Importantly we call ``repr`` directly (NOT ``repr(float(v))``):
    ``write_p_entity_divest_cumulative_max`` accumulates a ``period_sum =
    sum(empty) → int(0)``, and the legacy CSV column for those rows is
    ``"0"``, not ``"0.0"``.  Similarly
    ``write_p_entity_existing_integer_count`` emits ``repr(round(cnt))``
    which is an int repr.  Coercing to float here would silently break
    byte-parity on either path — see commit notes for the symptom.
    Storing ``value`` as Utf8 preserves the bit-exact precision parity the
    legacy CSV writers achieved through ``repr`` (see
    :mod:`._writer_arc_unions` docstring).
    """
    return pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "period": [r[1] for r in rows],
            "value":  [repr(r[2]) for r in rows],
        },
        schema={"entity": pl.Utf8, "period": pl.Utf8, "value": pl.Utf8},
    )


# ---------------------------------------------------------------------------
# Shared CSV readers (mirror legacy helpers byte-for-byte).
# ---------------------------------------------------------------------------


def _read_singles(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        return [r[0] for r in reader if r and r[0]]


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] and row[1]:
                out.append((row[0], row[1]))
    return out


def _read_triples(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    out: list[tuple[str, str, str]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3 and row[0] and row[1] and row[2]:
                out.append((row[0], row[1], row[2]))
    return out


def _load_ed_value_csv(path: Path) -> dict[tuple[str, str], float]:
    """``[entity, period, value]`` (header + rows) → ``{(e, d): v}``."""
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 3 and r[0] and r[1]:
                try:
                    out[(r[0], r[1])] = float(r[2])
                except ValueError:
                    continue
    return out


def _load_e_value_csv(path: Path) -> dict[str, float]:
    """``[entity, value]`` (header + rows) → ``{e: v}``."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 2 and r[0]:
                try:
                    out[r[0]] = float(r[1])
                except ValueError:
                    continue
    return out


def _load_param_value_at(
    path: Path, param_col: int, param_value: str,
    key_cols: tuple[int, ...], value_col: int,
) -> dict[tuple[str, ...], float]:
    """``[..., param, ..., value]`` filtered to rows where column ``param_col``
    equals ``param_value`` → ``{tuple(row[c] for c in key_cols): value}``."""
    out: dict[tuple[str, ...], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) <= max(param_col, value_col, *key_cols):
                continue
            if r[param_col] != param_value:
                continue
            key = tuple(r[c] for c in key_cols)
            if not all(key):
                continue
            try:
                out[key] = float(r[value_col])
            except ValueError:
                continue
    return out


def _read_solve_first_flag(solve_data_dir: Path) -> bool:
    """Read ``solve_data/p_model.csv`` for the ``solveFirst`` flag.

    Mirrors :func:`flextool.flextoolrunner.preprocessing.entity_period_calc_params._read_solve_first_flag`.
    """
    pm_path = solve_data_dir / "p_model.csv"
    if not pm_path.exists():
        return False
    with pm_path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for r in reader:
            if len(r) >= 2 and r[0] == "solveFirst":
                try:
                    return bool(int(r[1]))
                except ValueError:
                    return False
    return False


# ---------------------------------------------------------------------------
# write_p_entity_pre_existing — mod L1886-1895 (12-branch lifetime-method
# gate × entity-kind × virtual_unitsize trichotomy).
# ---------------------------------------------------------------------------


def derive_p_entity_pre_existing(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """Pre-existing capacity per (entity, period) — 12-branch sum where
    exactly one branch fires per (e, d) given the method/kind/unitsize
    trichotomy.

    Equivalent simplified form (per legacy docstring):

        method = entity__lifetime_method[e]
        v_existing = pdProcess[e,'existing',d] if e in process
                     else pdNode[e,'existing',d] if e in node
                     else 0
        v_unit     = p_process[e,'virtual_unitsize'] if e in process
                     else p_node[e,'virtual_unitsize'] if e in node
                     else 0
        if method not in {reinvest_automatic, reinvest_choice, no_investment}: 0
        elif method ∈ {reinvest_choice, no_investment}
             and not (p_years_d[d] <
                      Σ_{d_first ∈ period_first} (p_years_d[d_first]
                          + edEntity_lifetime[e, d_first])):
            0
        else: v_existing * v_unit  if v_unit  else  v_existing

    Output domain: ``entity × period_in_use``.  Reads pdProcess /
    pdNode / edEntity_lifetime CSVs that ``write_entity_period_calc_params``
    has emitted earlier in the same per-solve pass.
    """
    process_set = frozenset(_read_singles(input_dir / "process.csv"))
    node_set = frozenset(_read_singles(input_dir / "node.csv"))
    entities = _read_singles(input_dir / "entity.csv")
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")
    period_first = _read_singles(solve_data_dir / "period_first.csv")

    lifetime_method: dict[str, str] = {
        e_: m for e_, m in _read_pairs(solve_data_dir / "entity__lifetime_method.csv")
    }
    p_years_d = _load_e_value_csv(solve_data_dir / "p_years_d.csv")

    ed_lifetime: dict[tuple[str, str], float] = {}
    for e_, d_, v_ in _read_triples(solve_data_dir / "edEntity_lifetime.csv"):
        try:
            ed_lifetime[(e_, d_)] = float(v_)
        except ValueError:
            continue

    # pdProcess.csv / pdNode.csv: [entity, param, period, value]; filter to
    # rows where param == 'existing'.
    pd_existing_proc = _load_param_value_at(
        solve_data_dir / "pdProcess.csv",
        param_col=1, param_value="existing",
        key_cols=(0, 2), value_col=3,
    )
    pd_existing_node = _load_param_value_at(
        solve_data_dir / "pdNode.csv",
        param_col=1, param_value="existing",
        key_cols=(0, 2), value_col=3,
    )

    # p_process.csv / p_node.csv: [entity, param, value]; filter to
    # param == 'virtual_unitsize'.
    p_process_vu = _load_param_value_at(
        input_dir / "p_process.csv",
        param_col=1, param_value="virtual_unitsize",
        key_cols=(0,), value_col=2,
    )
    p_node_vu = _load_param_value_at(
        input_dir / "p_node.csv",
        param_col=1, param_value="virtual_unitsize",
        key_cols=(0,), value_col=2,
    )

    # Per-entity lifetime gate sum:
    #   Σ_{d_first ∈ period_first} (p_years_d[d_first]
    #       + edEntity_lifetime[e, d_first])
    def _life_sum(e: str) -> float:
        return sum(
            p_years_d.get(d_first, 0.0) + ed_lifetime.get((e, d_first), 0.0)
            for d_first in period_first
        )

    rows: list[tuple[str, str, float]] = []
    for e in entities:
        method = lifetime_method.get(e, "")
        is_proc = e in process_set
        is_node = e in node_set
        if is_proc:
            v_unit = p_process_vu.get((e,), 0.0)
        elif is_node:
            v_unit = p_node_vu.get((e,), 0.0)
        else:
            v_unit = 0.0
        gate_sum = (
            _life_sum(e)
            if method in ("reinvest_choice", "no_investment")
            else None
        )
        for d in period_in_use:
            v: float = 0.0
            if method in ("reinvest_automatic", "reinvest_choice", "no_investment"):
                gate_passes = method == "reinvest_automatic" or (
                    gate_sum is not None and p_years_d.get(d, 0.0) < gate_sum
                )
                if gate_passes:
                    if is_proc:
                        pd_e = pd_existing_proc.get((e, d), 0.0)
                    elif is_node:
                        pd_e = pd_existing_node.get((e, d), 0.0)
                    else:
                        pd_e = 0.0
                    v = pd_e * v_unit if v_unit else pd_e
            rows.append((e, d, v))

    return _ed_value_frame(rows)


def write_p_entity_pre_existing(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Emit ``solve_data/p_entity_pre_existing.csv`` (see derive docstring)."""
    _write(
        derive_p_entity_pre_existing(input_dir, solve_data_dir),
        solve_data_dir / "p_entity_pre_existing.csv",
    )


# ---------------------------------------------------------------------------
# write_p_entity_divest_cumulative_max — mod L1920-1933 (3-branch sum,
# cumulative ceiling on v_divest by dispatch period).
# ---------------------------------------------------------------------------


def derive_p_entity_divest_cumulative_max(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """Cumulative ceiling on ``v_divest`` summed by dispatch period ``d``.

    Three-branch split per legacy comment (only one fires per (e, d)
    given the ``e_divest_total`` membership / cardinality of
    ``ed_divest_period`` for ``e``):

        if e ∉ e_divest_total:
            Σ_{(e, d_div) ∈ ed_divest_period,
               p_years_d[d_div] ≤ p_years_d[d]}
                ed_divest_max_period[e, d_div]
        elif ed_divest_period has no rows for e:
            e_divest_max_total[e]
        else:
            max(period_sum, e_divest_max_total[e])

    Domain: ``entityDivest × period_in_use``.
    """
    entityDivest = _read_singles(solve_data_dir / "entityDivest.csv")
    e_divest_total = frozenset(
        _read_singles(solve_data_dir / "e_divest_total.csv")
    )
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    ed_divest_period = _read_pairs(solve_data_dir / "ed_divest_period.csv")
    div_periods_for_e: dict[str, list[str]] = {}
    for e_, d_ in ed_divest_period:
        div_periods_for_e.setdefault(e_, []).append(d_)

    p_years_d = _load_e_value_csv(solve_data_dir / "p_years_d.csv")
    ed_divest_max = _load_ed_value_csv(
        solve_data_dir / "ed_divest_max_period.csv"
    )
    e_divest_max_total = _load_e_value_csv(
        solve_data_dir / "e_divest_max_total.csv"
    )

    rows: list[tuple[str, str, float]] = []
    for e in entityDivest:
        in_total = e in e_divest_total
        e_div_periods = div_periods_for_e.get(e, [])
        e_total_max = e_divest_max_total.get(e, 0.0)
        for d in period_in_use:
            d_years = p_years_d.get(d, 0.0)
            period_sum = sum(
                ed_divest_max.get((e, d_div), 0.0)
                for d_div in e_div_periods
                if p_years_d.get(d_div, 0.0) <= d_years
            )
            if not in_total:
                v = period_sum
            elif not e_div_periods:
                v = e_total_max
            else:
                v = max(period_sum, e_total_max)
            rows.append((e, d, v))

    return _ed_value_frame(rows)


def write_p_entity_divest_cumulative_max(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Emit ``solve_data/p_entity_divest_cumulative_max.csv`` (see derive)."""
    _write(
        derive_p_entity_divest_cumulative_max(input_dir, solve_data_dir),
        solve_data_dir / "p_entity_divest_cumulative_max.csv",
    )


# ---------------------------------------------------------------------------
# write_p_entity_existing_chain — mod L1680-1697 (5 cascading entity-capacity
# params; straddles the per-solve handoff boundary).
# ---------------------------------------------------------------------------


def _load_handoff_or_csv_realized(
    solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None",
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float], bool]:
    """Resolve (realized_existing, realized_invest) from either the
    in-memory ``SolveHandoff`` carriers (when populated) or the
    file-based ``p_entity_period_existing_capacity.csv`` left by the
    parent solve's output writer.

    Returns ``(ppec, ppic, used_handoff)`` where ``ppec`` maps
    ``(entity, period) → realized_existing`` and ``ppic`` maps
    ``(entity, period) → realized_invest``.  ``used_handoff`` mirrors
    the legacy ``used_handoff_existing`` flag — true when at least one
    of the two carriers was supplied in-memory.
    """
    ppec: dict[tuple[str, str], float] = {}
    ppic: dict[tuple[str, str], float] = {}
    used_handoff = (
        prior_handoff is not None
        and (
            prior_handoff.realized_existing is not None
            or prior_handoff.realized_invest is not None
        )
    )
    if used_handoff:
        if prior_handoff.realized_existing is not None:
            for r in prior_handoff.realized_existing.iter_rows(named=True):
                ppec[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        if prior_handoff.realized_invest is not None:
            for r in prior_handoff.realized_invest.iter_rows(named=True):
                ppic[(str(r["entity"]), str(r["period"]))] = float(r["value"])
        return ppec, ppic, True

    # CSV path: p_entity_period_existing_capacity.csv carries both
    # ``existing`` (col 2) and ``invested`` (col 3) for every (e, d).
    ppe_path = solve_data_dir / "p_entity_period_existing_capacity.csv"
    if ppe_path.exists():
        with ppe_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 4 and r[0] and r[1]:
                    try:
                        ppec[(r[0], r[1])] = float(r[2])
                        ppic[(r[0], r[1])] = float(r[3])
                    except ValueError:
                        continue
    return ppec, ppic, False


def _compute_p_entity_existing_chain(
    input_dir: Path, solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None",
) -> tuple[
    list[tuple[str, str, float]],  # later_solves rows
    list[tuple[str, str, float]],  # all_existing rows
    list[tuple[str, str, float]],  # count rows
    list[tuple[str, str, int]],    # integer_count rows (legacy emits int repr)
    list[tuple[str, str, float]],  # previously_invested rows
]:
    """Compute the five row-streams emitted by ``write_p_entity_existing_chain``
    in their canonical (entity-major, period-in-use-minor) order.

    Pulled out so each per-CSV ``derive_*`` can call this once via a module-
    level cache key keyed on the inputs; the orchestrator
    :func:`write_p_entity_existing_chain` runs the compute once and then
    funnels each row-stream through :func:`_write` so the per-sub-solve
    :mod:`._flex_data_accumulator` captures the resulting frames.
    """
    solve_first = _read_solve_first_flag(solve_data_dir)
    entities = _read_singles(input_dir / "entity.csv")
    periods_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    pre_existing = _load_ed_value_csv(
        solve_data_dir / "p_entity_pre_existing.csv"
    )
    unitsize = _load_e_value_csv(solve_data_dir / "p_entity_unitsize.csv")

    edd_by_ed: dict[tuple[str, str], list[str]] = {}
    for (e, d_h, d) in _read_triples(solve_data_dir / "edd_history.csv"):
        edd_by_ed.setdefault((e, d), []).append(d_h)

    ppec, ppic, _ = _load_handoff_or_csv_realized(
        solve_data_dir, prior_handoff,
    )

    ed_history_realized: set[tuple[str, str]] = set(ppec.keys())
    for e_, d_ in _read_pairs(solve_data_dir / "ed_history_realized_first.csv"):
        ed_history_realized.add((e_, d_))

    entity_divest = frozenset(_read_singles(solve_data_dir / "entityDivest.csv"))
    p_divested: dict[str, float] = {}
    if prior_handoff is not None and prior_handoff.divest_cumulative is not None:
        for r in prior_handoff.divest_cumulative.iter_rows(named=True):
            p_divested[str(r["entity"])] = float(r["value"])
    else:
        p_divested = _load_e_value_csv(solve_data_dir / "p_entity_divested.csv")

    later_existing: dict[tuple[str, str], float] = {}
    later_invested: dict[tuple[str, str], float] = {}
    if not solve_first:
        for e in entities:
            for d in periods_in_use:
                tot_e = 0.0
                tot_i = 0.0
                for d_h in edd_by_ed.get((e, d), ()):
                    if (e, d_h) in ed_history_realized:
                        tot_e += ppec.get((e, d_h), 0.0)
                        tot_i += ppic.get((e, d_h), 0.0)
                later_existing[(e, d)] = tot_e
                later_invested[(e, d)] = tot_i

    later_rows: list[tuple[str, str, float]] = []
    all_rows: list[tuple[str, str, float]] = []
    count_rows: list[tuple[str, str, float]] = []
    int_count_rows: list[tuple[str, str, int]] = []
    prev_inv_rows: list[tuple[str, str, float]] = []
    for e in entities:
        us = unitsize.get(e, 0.0)
        for d in periods_in_use:
            v_later = 0.0 if solve_first else later_existing.get((e, d), 0.0)
            later_rows.append((e, d, v_later))
            if solve_first:
                v_all = pre_existing.get((e, d), 0.0)
            else:
                v_all = later_existing.get((e, d), 0.0)
                if e in entity_divest:
                    v_all -= p_divested.get(e, 0.0)
            all_rows.append((e, d, v_all))
            cnt = v_all / us if us else 0.0
            count_rows.append((e, d, cnt))
            # round() with a single arg returns int in Py3 → legacy emits
            # ``repr(int)`` (e.g. ``"3"``, not ``"3.0"``).  Preserve.
            int_count_rows.append((e, d, round(cnt)))
            v_prev = 0.0 if solve_first else later_invested.get((e, d), 0.0)
            prev_inv_rows.append((e, d, v_prev))

    return later_rows, all_rows, count_rows, int_count_rows, prev_inv_rows


def derive_p_entity_existing_capacity_later_solves(
    input_dir: Path, solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None" = None,
) -> pl.DataFrame:
    later_rows, _, _, _, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, prior_handoff,
    )
    return _ed_value_frame(later_rows)


def derive_p_entity_all_existing(
    input_dir: Path, solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None" = None,
) -> pl.DataFrame:
    _, all_rows, _, _, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, prior_handoff,
    )
    return _ed_value_frame(all_rows)


def derive_p_entity_existing_count(
    input_dir: Path, solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None" = None,
) -> pl.DataFrame:
    _, _, count_rows, _, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, prior_handoff,
    )
    return _ed_value_frame(count_rows)


def derive_p_entity_existing_integer_count(
    input_dir: Path, solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None" = None,
) -> pl.DataFrame:
    _, _, _, int_rows, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, prior_handoff,
    )
    return _ed_value_frame(int_rows)


def derive_p_entity_previously_invested_capacity(
    input_dir: Path, solve_data_dir: Path,
    prior_handoff: "SolveHandoff | None" = None,
) -> pl.DataFrame:
    _, _, _, _, prev_rows = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, prior_handoff,
    )
    return _ed_value_frame(prev_rows)


def write_p_entity_existing_chain(
    input_dir: Path, solve_data_dir: Path,
    *, prior_handoff: "SolveHandoff | None" = None,
) -> None:
    """Five cascading entity-capacity params (mod L1680-1697):

      * ``p_entity_existing_capacity_later_solves{e, d}`` —
            0 on first solve; otherwise Σ over ``(e, d_history, d)
            ∈ edd_history`` filtered to ``(e, d_history) ∈
            ed_history_realized`` of
            ``p_entity_period_existing_capacity[e, d_history]``.
      * ``p_entity_all_existing{e, d}`` —
            pre-existing on first solve, ``later_solves`` on others,
            minus ``p_entity_divested[e]`` if not first solve and
            ``e ∈ entityDivest``.
      * ``p_entity_existing_count{e, d}``           = all_existing / unitsize.
      * ``p_entity_existing_integer_count{e, d}``   = round(count).
      * ``p_entity_previously_invested_capacity{e, d}`` —
            same shape as ``later_solves`` but using
            ``p_entity_period_invested_capacity`` from the handoff.

    Path-collision: mod also writes wide-format
    ``solve_data/p_entity_all_existing.csv``; the printf is retargeted
    to ``solve_data/solve__p_entity_all_existing.csv``.
    """
    later_rows, all_rows, count_rows, int_rows, prev_rows = (
        _compute_p_entity_existing_chain(
            input_dir, solve_data_dir, prior_handoff,
        )
    )
    _write(
        _ed_value_frame(later_rows),
        solve_data_dir / "p_entity_existing_capacity_later_solves.csv",
    )
    _write(
        _ed_value_frame(all_rows),
        solve_data_dir / "p_entity_all_existing.csv",
    )
    _write(
        _ed_value_frame(count_rows),
        solve_data_dir / "p_entity_existing_count.csv",
    )
    _write(
        _ed_value_frame(int_rows),
        solve_data_dir / "p_entity_existing_integer_count.csv",
    )
    _write(
        _ed_value_frame(prev_rows),
        solve_data_dir / "p_entity_previously_invested_capacity.csv",
    )


# ---------------------------------------------------------------------------
# write_p_entity_capacity_max_chain — mod L1699-1764 (4 cascading entity-
# capacity ceiling params).
# ---------------------------------------------------------------------------


def _compute_p_entity_capacity_max_chain(
    input_dir: Path, solve_data_dir: Path,
) -> tuple[
    list[tuple[str, str, float]],  # max_capacity rows (period_in_use only)
    list[tuple[str, str, float]],  # max_units rows (full period_set)
    list[tuple[str, str, float]],  # invest_cumulative_max rows
    list[tuple[str, str, float]],  # dispatch_capacity_max rows
]:
    """Compute the four row-streams emitted by
    :func:`write_p_entity_capacity_max_chain` in their canonical order.

    Extracted so each per-CSV ``derive_*`` returns a frame on demand and
    the orchestrator funnels each stream through :func:`_write` for
    accumulator capture.
    """
    entities = _read_singles(input_dir / "entity.csv")
    periods = _read_singles(solve_data_dir / "period_set.csv")
    period_in_use = frozenset(
        _read_singles(solve_data_dir / "period_in_use_set.csv")
    )

    all_existing = _load_ed_value_csv(solve_data_dir / "p_entity_all_existing.csv")
    cum_max_cap = _load_ed_value_csv(solve_data_dir / "ed_cumulative_max_capacity.csv")
    invest_max_period = _load_ed_value_csv(
        solve_data_dir / "ed_invest_max_period.csv"
    )
    unitsize = _load_e_value_csv(solve_data_dir / "p_entity_unitsize.csv")
    e_invest_max_total = _load_e_value_csv(
        solve_data_dir / "e_invest_max_total.csv"
    )

    invest_cumulative = frozenset(
        _read_pairs(solve_data_dir / "ed_invest_cumulative.csv")
    )
    invest_period = frozenset(
        _read_pairs(solve_data_dir / "ed_invest_period.csv")
    )
    invest_total = frozenset(_read_singles(solve_data_dir / "e_invest_total.csv"))
    invest_forbidden = frozenset(
        _read_pairs(solve_data_dir / "ed_invest_forbidden_no_investment.csv")
    )
    entity_invest = frozenset(_read_singles(solve_data_dir / "entityInvest.csv"))

    invest_method_pairs = frozenset(
        _read_pairs(input_dir / "entity__invest_method.csv")
    )

    p_unc = 1000000.0
    pmaxf_path = input_dir / "p_max_flow_for_unconstrained_variables.csv"
    if pmaxf_path.exists():
        max_v: float | None = None
        with pmaxf_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 2 and r[0]:
                    try:
                        v = float(r[1])
                    except ValueError:
                        continue
                    if max_v is None or v > max_v:
                        max_v = v
        if max_v is not None:
            p_unc = max_v

    edd_invest_by_ed: dict[tuple[str, str], list[str]] = {}
    for (e, d_inv, d) in _read_triples(solve_data_dir / "edd_invest.csv"):
        edd_invest_by_ed.setdefault((e, d), []).append(d_inv)

    max_capacity: dict[tuple[str, str], float] = {}
    mc_rows: list[tuple[str, str, float]] = []
    for e in entities:
        in_total = e in invest_total
        has_no_limit = (e, "invest_no_limit") in invest_method_pairs
        for d in periods:
            if d not in period_in_use:
                max_capacity[(e, d)] = 0.0
                continue
            if (e, d) in invest_cumulative:
                v = cum_max_cap.get((e, d), 0.0)
            else:
                v = all_existing.get((e, d), 0.0)
                in_period = (e, d) in invest_period
                imp = invest_max_period.get((e, d), 0.0)
                eim = e_invest_max_total.get(e, 0.0)
                if in_period and not in_total:
                    v += imp
                if in_total and not in_period:
                    v += eim
                if in_period and in_total:
                    v += max(imp, eim)
                if has_no_limit:
                    v += p_unc
            max_capacity[(e, d)] = v
            mc_rows.append((e, d, v))

    mu_rows: list[tuple[str, str, float]] = []
    for e in entities:
        us = unitsize.get(e, 0.0)
        for d in periods:
            mc = max_capacity[(e, d)]
            v = mc / us if us else 0.0
            mu_rows.append((e, d, v))

    invest_cum_max: dict[tuple[str, str], float] = {}
    icm_rows: list[tuple[str, str, float]] = []
    for e in entities:
        if e not in entity_invest:
            continue
        in_total = e in invest_total
        has_no_limit_method = (e, "invest_no_limit") in invest_method_pairs
        for d in periods:
            if d not in period_in_use:
                invest_cum_max[(e, d)] = 0.0
                continue
            if (e, d) in invest_cumulative:
                v = max(
                    0.0,
                    cum_max_cap.get((e, d), 0.0)
                    - all_existing.get((e, d), 0.0),
                )
            elif has_no_limit_method:
                v = p_unc
            else:
                v = 0.0
                in_period = (e, d) in invest_period
                if in_period and not in_total:
                    per_period_sum = sum(
                        invest_max_period.get((e, d_inv), 0.0)
                        for d_inv in edd_invest_by_ed.get((e, d), ())
                        if (e, d_inv) in invest_period
                        and (e, d_inv) not in invest_forbidden
                    )
                    v += per_period_sum
                if in_total and not in_period:
                    v += e_invest_max_total.get(e, 0.0)
                if in_period and in_total:
                    per_period_sum = sum(
                        invest_max_period.get((e, d_inv), 0.0)
                        for d_inv in edd_invest_by_ed.get((e, d), ())
                        if (e, d_inv) in invest_period
                        and (e, d_inv) not in invest_forbidden
                    )
                    v += max(
                        per_period_sum, e_invest_max_total.get(e, 0.0),
                    )
            invest_cum_max[(e, d)] = v
            icm_rows.append((e, d, v))

    dcm_rows: list[tuple[str, str, float]] = []
    for e in entities:
        for d in periods:
            if d not in period_in_use:
                continue
            v = all_existing.get((e, d), 0.0)
            if e in entity_invest:
                v += invest_cum_max.get((e, d), 0.0)
            dcm_rows.append((e, d, v))

    return mc_rows, mu_rows, icm_rows, dcm_rows


def derive_p_entity_max_capacity(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    mc_rows, _, _, _ = _compute_p_entity_capacity_max_chain(input_dir, solve_data_dir)
    return _ed_value_frame(mc_rows)


def derive_p_entity_max_units(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    _, mu_rows, _, _ = _compute_p_entity_capacity_max_chain(input_dir, solve_data_dir)
    return _ed_value_frame(mu_rows)


def derive_p_entity_invest_cumulative_max(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    _, _, icm_rows, _ = _compute_p_entity_capacity_max_chain(input_dir, solve_data_dir)
    return _ed_value_frame(icm_rows)


def derive_p_entity_dispatch_capacity_max(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    _, _, _, dcm_rows = _compute_p_entity_capacity_max_chain(input_dir, solve_data_dir)
    return _ed_value_frame(dcm_rows)


def write_p_entity_capacity_max_chain(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Four cascading entity-capacity ceiling params (mod L1699-1764):

      * ``p_entity_max_capacity{e, d ∈ period_in_use}`` — 5-branch fork:
            cumulative-cap if ``(e, d) ∈ ed_invest_cumulative``;
            otherwise ``all_existing`` plus per-period / total /
            max(per-period, total) extras + ``invest_no_limit`` slack.
      * ``p_entity_max_units{e, d ∈ period}`` —
            ``max_capacity / unitsize`` (0 outside ``period_in_use``).
      * ``p_entity_invest_cumulative_max{e ∈ entityInvest, d}`` —
            cumulative-cap minus existing, or per-period sum from
            ``edd_invest`` (with ``invest_no_limit`` short-circuit).
      * ``p_entity_dispatch_capacity_max{e, d}`` —
            ``all_existing + (invest_cumulative_max if entityInvest else 0)``.

    Reads ``p_entity_all_existing`` (just-written),
    ``p_entity_unitsize``, ``ed_invest_cumulative``,
    ``ed_cumulative_max_capacity``, ``ed_invest_period``, ``e_invest_total``,
    ``ed_invest_max_period``, ``e_invest_max_total``,
    ``ed_invest_forbidden_no_investment``, ``edd_invest``,
    ``entityInvest``, ``entity__invest_method``,
    ``p_max_flow_for_unconstrained_variables``.

    Path-collision: mod's wide-format printf for ``p_entity_max_units``
    is retargeted to ``solve__p_entity_max_units.csv``.
    """
    mc_rows, mu_rows, icm_rows, dcm_rows = _compute_p_entity_capacity_max_chain(
        input_dir, solve_data_dir,
    )
    _write(
        _ed_value_frame(mc_rows),
        solve_data_dir / "p_entity_max_capacity.csv",
    )
    _write(
        _ed_value_frame(mu_rows),
        solve_data_dir / "p_entity_max_units.csv",
    )
    _write(
        _ed_value_frame(icm_rows),
        solve_data_dir / "p_entity_invest_cumulative_max.csv",
    )
    _write(
        _ed_value_frame(dcm_rows),
        solve_data_dir / "p_entity_dispatch_capacity_max.csv",
    )
