"""The capacity a solve starts from, and its capacity ceilings, per (entity, period).

* :func:`emit_existing_capacity` — the existing capacity from the cohorts (:func:`._cohorts.existing_capacity`):
  the run's existing cohort (the user's ``existing`` of each period, in service until its end of life), the
  investments earlier solves of the run realized (handoff ``realized_cohort``, each in service from its start of
  operation to its end of life) and their retirements (handoff ``realized_retire``).  One formula for every solve
  of a run (first, chained, roll, nested sub-solve):
  ``p_entity_pre_existing``, ``p_entity_all_existing``, ``p_entity_previously_invested_capacity``,
  ``p_entity_retired_carried``, ``p_entity_invested``.
* :func:`emit_p_entity_capacity_max_chain` — the capacity ceilings from ``p_entity_all_existing`` and the
  investment limits.

The values are rendered by :func:`_ed_value_frame` (``repr``), so a frame round-trips its floats exactly.
"""
from __future__ import annotations

import math

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from flextool.engine.preprocess._provider_io import (
    _emit,
    _provider_key,
)
from flextool.engine.preprocess import _provider_keys as K
from flextool.engine.preprocess._provider_translators import read_handoff_frame
from flextool.engine.preprocess.params._derived import INVEST_METHODS_INVEST_NO_LIMIT

if TYPE_CHECKING:
    from flextool.engine.preprocess.params._cohorts import ExistingCapacity


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
    :mod:`._arc_unions` docstring).
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
# CSV text (the former write-CSV + ``csv.reader`` path).
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


# ---------------------------------------------------------------------------
# The existing capacity of a solve, from the cohorts: the existing cohort of the run, the cohorts earlier solves
# realized (handoff ``realized_cohort``) and their retirements (handoff ``realized_retire``).
# ---------------------------------------------------------------------------


def _existing_mw(input_dir: Path, solve_data_dir: Path, entities: list[str], periods: list[str],
                 *, provider) -> dict[tuple[str, str], float]:
    """``{(e, d): MW}``: the user's ``existing`` of each period of the solve (``pdProcess`` / ``pdNode``, the map
    read for this solve), × ``virtual_unitsize`` when it is above 0 (a ``virtual_unitsize`` of 0 or below is
    ignored; ``validate_basic_sanity`` warns), 0 for an entity that is neither a process nor a node."""
    process_set = frozenset(_read_singles(input_dir / "process.csv", provider=provider))
    node_set = frozenset(_read_singles(input_dir / "node.csv", provider=provider))
    pd_existing_proc = _load_param_value_at(solve_data_dir / "pdProcess.csv", param_col=1, param_value="existing",
                                            key_cols=(0, 2), value_col=3, provider=provider)
    pd_existing_node = _load_param_value_at(solve_data_dir / "pdNode.csv", param_col=1, param_value="existing",
                                            key_cols=(0, 2), value_col=3, provider=provider)
    p_process_vu = _load_param_value_at(input_dir / "p_process.csv", param_col=1, param_value="virtual_unitsize",
                                        key_cols=(0,), value_col=2, provider=provider)
    p_node_vu = _load_param_value_at(input_dir / "p_node.csv", param_col=1, param_value="virtual_unitsize",
                                     key_cols=(0,), value_col=2, provider=provider)
    out: dict[tuple[str, str], float] = {}
    for e in entities:
        if e in process_set:
            v_unit, given = p_process_vu.get((e,), 0.0), pd_existing_proc
        elif e in node_set:
            v_unit, given = p_node_vu.get((e,), 0.0), pd_existing_node
        else:
            continue
        for d in periods:
            pd_e = given.get((e, d), 0.0)
            out[(e, d)] = pd_e * v_unit if v_unit > 0.0 else pd_e
    return out


def solve_run_years(provider, solve_data_dir: Path = Path("solve_data")) -> dict[str, float]:
    """``{period: run year}`` of the solve on ``provider``: its ``p_years_d`` (history, periods in use and branch
    periods) on the run calendar (:func:`._run_calendar.run_years`), the years the cohorts compare."""
    from flextool.engine.preprocess.params._run_calendar import read_run_calendar, run_years
    p_years_d: dict[str, float] = {}
    pyd = provider.get(_provider_key(solve_data_dir / "p_years_d.csv"))
    for d, y in ([] if pyd is None else pyd.select(pyd.columns[0], pyd.columns[-1]).iter_rows()):
        if _cell_str(d):
            p_years_d.setdefault(_cell_str(d), float(y))
    return run_years(p_years_d, read_run_calendar(provider))


def compute_existing_capacity(input_dir: Path, solve_data_dir: Path,
                              *, provider) -> "tuple[list[str], list[str], ExistingCapacity]":
    """``(entities, period_in_use, capacity)``: :func:`._cohorts.existing_capacity` of the solve from the
    Provider: the run's existing cohorts (:data:`._run_calendar.RUN_EXISTING_COHORT_KEY`, put by ``_run_model``),
    the solve's ``existing`` map (:func:`_existing_mw`), its period years on the run calendar
    (:func:`._run_calendar.run_years` of ``p_years_d``), the carried cohorts and retirements (handoff
    ``realized_cohort`` / ``realized_retire``, an override first) and ``entityDivest``."""
    from flextool.engine.preprocess.params._cohorts import cohort_frame_from_utf8, existing_capacity
    from flextool.engine.preprocess.params._run_calendar import RUN_EXISTING_COHORT_KEY
    if not provider.has(RUN_EXISTING_COHORT_KEY):
        raise ValueError("The existing capacity of a solve needs the run's existing cohorts "
                         f"({RUN_EXISTING_COHORT_KEY}), which the run puts on every sub-solve's Provider.")
    entities = _read_singles(input_dir / "entity.csv", provider=provider)
    periods = _read_singles(solve_data_dir / "period_in_use_set.csv", provider=provider)
    capacity = existing_capacity(
        entities, periods, solve_run_years(provider, solve_data_dir),
        cohort_frame_from_utf8(provider.get(RUN_EXISTING_COHORT_KEY)),
        _existing_mw(input_dir, solve_data_dir, entities, periods, provider=provider),
        read_handoff_frame(provider, K.HANDOFF_REALIZED_COHORT),
        read_handoff_frame(provider, K.HANDOFF_REALIZED_RETIRE),
        _read_singles(solve_data_dir / "entityDivest.csv", provider=provider))
    return entities, periods, capacity


def emit_existing_capacity(input_dir: Path, solve_data_dir: Path, *, provider) -> None:
    """Emit the solve's existing capacity (:func:`compute_existing_capacity`), each frame over
    ``entity × period_in_use`` (entity-major) and its values rendered by :func:`_ed_value_frame`:

    * ``p_entity_pre_existing`` — the existing cohort in service (the user's ``existing`` × virtual_unitsize);
    * ``p_entity_all_existing`` — the capacity in service at the start of the solve (existing + carried − retired);
    * ``p_entity_previously_invested_capacity`` — the carried investments in service, less their carried retirements;
    * ``p_entity_retired_carried`` — the retirements of the existing cohort earlier solves realized that still
      count (the rows with a retirement, sorted);
    * ``p_entity_invested`` ``[entity, p_entity_invested]`` — every carried investment of an entity.
    """
    entities, periods, cap = compute_existing_capacity(input_dir, solve_data_dir, provider=provider)
    grid = [(e, d) for e in entities for d in periods]
    _emit(provider, "solve_data/p_entity_pre_existing.csv", _ed_value_frame([(e, d, cap.pre[(e, d)]) for e, d in grid]))
    _emit(provider, "solve_data/p_entity_all_existing.csv",
          _ed_value_frame([(e, d, cap.all[(e, d)]) for e, d in grid]))
    _emit(provider, "solve_data/p_entity_previously_invested_capacity.csv",
          _ed_value_frame([(e, d, cap.carried[(e, d)]) for e, d in grid]))
    _emit(provider, "solve_data/p_entity_retired_carried.csv",
          _ed_value_frame([(e, d, v) for (e, d), v in sorted(cap.retired.items())]))
    _emit(provider, "solve_data/p_entity_invested.csv",
          pl.DataFrame({"entity": list(cap.invested), "p_entity_invested": [repr(v) for v in cap.invested.values()]},
                       schema={"entity": pl.Utf8, "p_entity_invested": pl.Utf8}))



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
    :func:`emit_p_entity_capacity_max_chain` in their canonical order.

    Kept as a single compute so the orchestrator funnels each stream
    through :func:`_emit` for accumulator capture.
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
    # As in `_derived.py`: entities that author
    # cumulative_max_capacity at all (any period) — an entity absent here never set the
    # parameter, as opposed to an authored-but-zero value.
    cmc_authors = {e for (e, _d) in cum_max_cap}
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

    # invest_max_period / cumulative_max_capacity = inf is no limit in that period (the model leaves the row
    # out): the bounds use the no-limit value p_unc, as an *_no_limit method does (``_derived`` does the same).
    invest_max_period = {k: (p_unc if math.isinf(v) and v > 0 else v) for k, v in invest_max_period.items()}
    cmc_unlimited = {k for k, v in cum_max_cap.items() if math.isinf(v) and v > 0}

    edd_invest_by_ed: dict[tuple[str, str], list[str]] = {}
    for (e, d_inv, d) in _read_triples(solve_data_dir / "edd_invest.csv",
                                         provider=provider):
        edd_invest_by_ed.setdefault((e, d), []).append(d_inv)

    max_capacity: dict[tuple[str, str], float] = {}
    mc_rows: list[tuple[str, str, float]] = []
    for e in entities:
        in_total = e in invest_total
        has_no_limit = any((e, m) in invest_method_pairs
                           for m in INVEST_METHODS_INVEST_NO_LIMIT)
        for d in periods:
            if d not in period_in_use:
                max_capacity[(e, d)] = 0.0
                continue
            if (e, d) in invest_cumulative:
                if (e, d) in cmc_unlimited:
                    v = all_existing.get((e, d), 0.0) + p_unc
                elif e in cmc_authors:
                    v = cum_max_cap.get((e, d), 0.0)
                else:
                    # A cumulative_limits entity that never authors
                    # cumulative_max_capacity (e.g. only cumulative_min_capacity, a floor with
                    # no ceiling) is unconstrained above, not pinned to v_invest = 0 — the floor
                    # would otherwise be unmeetable.
                    v = all_existing.get((e, d), 0.0) + p_unc
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
        has_no_limit_method = any((e, m) in invest_method_pairs
                                  for m in INVEST_METHODS_INVEST_NO_LIMIT)
        for d in periods:
            if d not in period_in_use:
                invest_cum_max[(e, d)] = 0.0
                continue
            if (e, d) in invest_cumulative:
                if (e, d) in cmc_unlimited:
                    v = p_unc
                elif e in cmc_authors:
                    v = max(
                        0.0,
                        cum_max_cap.get((e, d), 0.0)
                        - all_existing.get((e, d), 0.0),
                    )
                else:
                    # Unconstrained side has no existing-capacity subtraction
                    # to add back — matches the has_no_limit_method branch below.
                    v = p_unc
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
