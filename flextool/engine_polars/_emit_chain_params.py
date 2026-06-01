"""Chain-cluster entity-period parameter writers.

Three ``write_*`` helpers emitting the per-(entity, period)
existing/capacity-max chain that the LP build consumes via
``p_entity_*_existing_*`` and ``p_entity_*_capacity_*`` parameters:

* :func:`write_p_entity_pre_existing`           (12-branch).
* :func:`write_p_entity_existing_chain`         (3 live emits).
* :func:`write_p_entity_capacity_max_chain`     (2 live emits).

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

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine_polars._emit_provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine_polars import _provider_keys as K
from flextool.engine_polars._provider_translators import read_handoff_frame

if TYPE_CHECKING:
    from ._solve_handoff import SolveHandoff  # noqa: F401 — retained for legacy callers


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
    :mod:`._emit_arc_unions` docstring).
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
# Native-frame readers.
#
# These read the in-memory polars frame directly from the Provider via
# ``provider.get(_provider_key(path))`` instead of round-tripping through
# CSV text (the legacy ``_provider_open`` + ``csv.reader`` path).
#
# Type-fidelity contract — reproduce *exactly* what ``csv.reader`` over a
# ``DataFrame.write_csv`` serialisation would have yielded:
#
#   * Key columns (the dict-key / structural string positions) were
#     ``csv.reader`` strings.  ``write_csv`` serialises ``null`` → ``""``
#     and any scalar (Enum / Int / Float / Utf8) → its string form.  We
#     coerce each key cell with :func:`_cell_str` and apply the original
#     truthiness guard to the *string* form so a null cell is skipped
#     (matching the legacy ``if row[i]`` test) while a literal ``"0"`` is
#     kept.
#   * Value columns were re-coerced with ``float(...)`` / ``int(...)``.
#     We apply the same coercion to the native cell and widen the legacy
#     ``except ValueError`` to ``except (ValueError, TypeError)`` so a
#     null value cell is skipped (matching the legacy CSV behaviour where
#     the empty-string value cell raised ``ValueError``).
#
# ``provider.get`` returns data rows only (no header), so there is no
# header row to skip; an empty / missing frame yields the same empty
# output the legacy loop produced.
# ---------------------------------------------------------------------------


def _cell_str(value: "object | None") -> str:
    """Reproduce a ``csv.reader`` cell string for a native frame value.

    ``DataFrame.write_csv`` renders ``null`` as the empty string and every
    other scalar as its textual form; ``csv.reader`` then reads those
    strings back.  Mirror that here so dict keys / structural strings stay
    byte-identical to the legacy CSV round-trip.
    """
    return "" if value is None else str(value)


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


def _read_pairs(path: Path,
                *, provider: "object | None" = None) -> list[tuple[str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str]] = []
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0, c1 = _cell_str(row[0]), _cell_str(row[1])
        if c0 and c1:
            out.append((c0, c1))
    return out


def _read_triples(path: Path,
                  *, provider: "object | None" = None) -> list[tuple[str, str, str]]:
    df = provider.get(_provider_key(path))
    if df is None:
        return []
    out: list[tuple[str, str, str]] = []
    for row in df.iter_rows():
        if len(row) < 3:
            continue
        c0, c1, c2 = _cell_str(row[0]), _cell_str(row[1]), _cell_str(row[2])
        if c0 and c1 and c2:
            out.append((c0, c1, c2))
    return out


def _load_ed_value_csv(path: Path,
                       *, provider: "object | None" = None,
                       ) -> dict[tuple[str, str], float]:
    """``[entity, period, value]`` (data rows) → ``{(e, d): v}``."""
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


def _load_e_value_csv(path: Path,
                      *, provider: "object | None" = None,
                      ) -> dict[str, float]:
    """``[entity, value]`` (data rows) → ``{e: v}``."""
    out: dict[str, float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) < 2:
            continue
        c0 = _cell_str(row[0])
        if c0:
            try:
                out[c0] = float(row[1])
            except (ValueError, TypeError):
                continue
    return out


def _load_param_value_at(
    path: Path, param_col: int, param_value: str,
    key_cols: tuple[int, ...], value_col: int,
    *, provider: "object | None" = None,
) -> dict[tuple[str, ...], float]:
    """``[..., param, ..., value]`` filtered to rows where column ``param_col``
    equals ``param_value`` → ``{tuple(row[c] for c in key_cols): value}``."""
    out: dict[tuple[str, ...], float] = {}
    df = provider.get(_provider_key(path))
    if df is None:
        return out
    for row in df.iter_rows():
        if len(row) <= max(param_col, value_col, *key_cols):
            continue
        if _cell_str(row[param_col]) != param_value:
            continue
        key = tuple(_cell_str(row[c]) for c in key_cols)
        if not all(key):
            continue
        try:
            out[key] = float(row[value_col])
        except (ValueError, TypeError):
            continue
    return out


def _read_solve_first_flag(solve_data_dir: Path,
                           *, provider: "object | None" = None) -> bool:
    """Read ``solve_data/p_model.csv`` for the ``solveFirst`` flag."""
    pm_path = solve_data_dir / "p_model.csv"
    df = provider.get(_provider_key(pm_path))
    if df is None:
        return False
    for row in df.iter_rows():
        if len(row) >= 2 and _cell_str(row[0]) == "solveFirst":
            try:
                return bool(int(row[1]))
            except (ValueError, TypeError):
                return False
    return False


# ---------------------------------------------------------------------------
# write_p_entity_pre_existing — mod L1886-1895 (12-branch lifetime-method
# gate × entity-kind × virtual_unitsize trichotomy).
# ---------------------------------------------------------------------------


def derive_p_entity_pre_existing(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
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
    process_set = frozenset(_read_singles(input_dir / "process.csv",
                                            provider=provider))
    node_set = frozenset(_read_singles(input_dir / "node.csv",
                                        provider=provider))
    entities = _read_singles(input_dir / "entity.csv", provider=provider)
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv",
                                   provider=provider)
    period_first = _read_singles(solve_data_dir / "period_first.csv",
                                  provider=provider)

    lifetime_method: dict[str, str] = {
        e_: m for e_, m in _read_pairs(solve_data_dir / "entity__lifetime_method.csv",
                                         provider=provider)
    }
    p_years_d = _load_e_value_csv(solve_data_dir / "p_years_d.csv",
                                    provider=provider)

    ed_lifetime: dict[tuple[str, str], float] = {}
    for e_, d_, v_ in _read_triples(solve_data_dir / "edEntity_lifetime.csv",
                                      provider=provider):
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
        provider=provider,
    )
    pd_existing_node = _load_param_value_at(
        solve_data_dir / "pdNode.csv",
        param_col=1, param_value="existing",
        key_cols=(0, 2), value_col=3,
        provider=provider,
    )

    # p_process.csv / p_node.csv: [entity, param, value]; filter to
    # param == 'virtual_unitsize'.
    p_process_vu = _load_param_value_at(
        input_dir / "p_process.csv",
        param_col=1, param_value="virtual_unitsize",
        key_cols=(0,), value_col=2,
        provider=provider,
    )
    p_node_vu = _load_param_value_at(
        input_dir / "p_node.csv",
        param_col=1, param_value="virtual_unitsize",
        key_cols=(0,), value_col=2,
        provider=provider,
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


def emit_p_entity_pre_existing(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``p_entity_pre_existing`` to the Provider."""
    _emit(provider, "solve_data/p_entity_pre_existing.csv",
          derive_p_entity_pre_existing(input_dir, solve_data_dir,
                                         provider=provider))


# ---------------------------------------------------------------------------
# write_p_entity_existing_chain — mod L1680-1697 (5 cascading entity-capacity
# params; straddles the per-solve handoff boundary).
# ---------------------------------------------------------------------------


def _load_realized_from_handoff(
    *, provider: "object | None" = None,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    """Resolve ``(realized_existing, realized_invest)`` from the
    Provider's ``handoff/realized_*`` carriers, populated by the
    iteration-start translator.

    Returns ``(ppec, ppic)`` where ``ppec`` maps
    ``(entity, period) → realized_existing`` and ``ppic`` maps
    ``(entity, period) → realized_invest``.  An empty/None handoff
    frame yields an empty mapping — the translator pipeline always
    primes these carriers (header-only frame when no prior handoff
    exists, populated frame otherwise), so this is the sole source of
    truth.
    """
    ppec: dict[tuple[str, str], float] = {}
    ppic: dict[tuple[str, str], float] = {}
    realized_existing = read_handoff_frame(provider, K.HANDOFF_REALIZED_EXISTING)
    realized_invest = read_handoff_frame(provider, K.HANDOFF_REALIZED_INVEST)
    if realized_existing is not None:
        for r in realized_existing.iter_rows(named=True):
            ppec[(str(r["entity"]), str(r["period"]))] = float(r["value"])
    if realized_invest is not None:
        for r in realized_invest.iter_rows(named=True):
            ppic[(str(r["entity"]), str(r["period"]))] = float(r["value"])
    return ppec, ppic


def _compute_p_entity_existing_chain(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
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
    solve_first = _read_solve_first_flag(solve_data_dir, provider=provider)
    entities = _read_singles(input_dir / "entity.csv", provider=provider)
    periods_in_use = _read_singles(
        solve_data_dir / "period_in_use_set.csv", provider=provider,
    )

    pre_existing = _load_ed_value_csv(
        solve_data_dir / "p_entity_pre_existing.csv", provider=provider,
    )
    unitsize = _load_e_value_csv(
        solve_data_dir / "p_entity_unitsize.csv", provider=provider,
    )

    edd_by_ed: dict[tuple[str, str], list[str]] = {}
    for (e, d_h, d) in _read_triples(
        solve_data_dir / "edd_history.csv", provider=provider,
    ):
        edd_by_ed.setdefault((e, d), []).append(d_h)

    ppec, ppic = _load_realized_from_handoff(provider=provider)

    ed_history_realized: set[tuple[str, str]] = set(ppec.keys())
    for e_, d_ in _read_pairs(
        solve_data_dir / "ed_history_realized_first.csv", provider=provider,
    ):
        ed_history_realized.add((e_, d_))

    entity_divest = frozenset(_read_singles(
        solve_data_dir / "entityDivest.csv", provider=provider,
    ))
    p_divested: dict[str, float] = {}
    divest_cumulative = read_handoff_frame(provider, K.HANDOFF_DIVEST_CUMULATIVE)
    if divest_cumulative is not None:
        for r in divest_cumulative.iter_rows(named=True):
            p_divested[str(r["entity"])] = float(r["value"])

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
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    later_rows, _, _, _, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_value_frame(later_rows)


def derive_p_entity_all_existing(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    _, all_rows, _, _, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_value_frame(all_rows)


def derive_p_entity_existing_count(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    _, _, count_rows, _, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_value_frame(count_rows)


def derive_p_entity_existing_integer_count(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    _, _, _, int_rows, _ = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_value_frame(int_rows)


def derive_p_entity_previously_invested_capacity(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
) -> pl.DataFrame:
    _, _, _, _, prev_rows = _compute_p_entity_existing_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    return _ed_value_frame(prev_rows)


def emit_p_entity_existing_chain(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``p_entity_existing_chain`` to the Provider."""
    later_rows, all_rows, _count_rows, _int_rows, prev_rows = (
        _compute_p_entity_existing_chain(
            input_dir, solve_data_dir, provider=provider,
        )
    )
    _emit(provider,
          "solve_data/p_entity_existing_capacity_later_solves.csv",
          _ed_value_frame(later_rows))
    _emit(provider, "solve_data/p_entity_all_existing.csv",
          _ed_value_frame(all_rows))
    _emit(provider,
          "solve_data/p_entity_previously_invested_capacity.csv",
          _ed_value_frame(prev_rows))


# ---------------------------------------------------------------------------
# write_p_entity_capacity_max_chain — mod L1699-1764 (4 cascading entity-
# capacity ceiling params).
# ---------------------------------------------------------------------------


def _compute_p_entity_capacity_max_chain(
    input_dir: Path, solve_data_dir: Path,
    *, provider: "object | None" = None,
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
    entities = _read_singles(input_dir / "entity.csv", provider=provider)
    periods = _read_singles(solve_data_dir / "period_set.csv", provider=provider)
    period_in_use = frozenset(
        _read_singles(solve_data_dir / "period_in_use_set.csv", provider=provider)
    )

    all_existing = _load_ed_value_csv(solve_data_dir / "p_entity_all_existing.csv",
                                       provider=provider)
    cum_max_cap = _load_ed_value_csv(solve_data_dir / "ed_cumulative_max_capacity.csv",
                                      provider=provider)
    invest_max_period = _load_ed_value_csv(
        solve_data_dir / "ed_invest_max_period.csv",
        provider=provider,
    )
    unitsize = _load_e_value_csv(solve_data_dir / "p_entity_unitsize.csv",
                                  provider=provider)
    e_invest_max_total = _load_e_value_csv(
        solve_data_dir / "e_invest_max_total.csv",
        provider=provider,
    )

    invest_cumulative = frozenset(
        _read_pairs(solve_data_dir / "ed_invest_cumulative.csv", provider=provider)
    )
    invest_period = frozenset(
        _read_pairs(solve_data_dir / "ed_invest_period.csv", provider=provider)
    )
    invest_total = frozenset(_read_singles(solve_data_dir / "e_invest_total.csv",
                                            provider=provider))
    invest_forbidden = frozenset(
        _read_pairs(solve_data_dir / "ed_invest_forbidden_no_investment.csv",
                    provider=provider)
    )
    entity_invest = frozenset(_read_singles(solve_data_dir / "entityInvest.csv",
                                             provider=provider))

    invest_method_pairs = frozenset(
        _read_pairs(input_dir / "entity__invest_method.csv", provider=provider)
    )

    p_unc = 1000000.0
    pmaxf_path = input_dir / "p_max_flow_for_unconstrained_variables.csv"
    pmaxf_df = provider.get(_provider_key(pmaxf_path))
    if pmaxf_df is not None:
        max_v: float | None = None
        for row in pmaxf_df.iter_rows():
            if len(row) >= 2 and _cell_str(row[0]):
                try:
                    v = float(row[1])
                except (ValueError, TypeError):
                    continue
                if max_v is None or v > max_v:
                    max_v = v
        if max_v is not None:
            p_unc = max_v

    edd_invest_by_ed: dict[tuple[str, str], list[str]] = {}
    for (e, d_inv, d) in _read_triples(solve_data_dir / "edd_invest.csv",
                                         provider=provider):
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


def emit_p_entity_capacity_max_chain(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``p_entity_capacity_max_chain`` to the Provider."""
    _mc_rows, mu_rows, _icm_rows, dcm_rows = _compute_p_entity_capacity_max_chain(
        input_dir, solve_data_dir, provider=provider,
    )
    _emit(provider, "solve_data/p_entity_max_units.csv",
          _ed_value_frame(mu_rows))
    _emit(provider, "solve_data/p_entity_dispatch_capacity_max.csv",
          _ed_value_frame(dcm_rows))
