"""Annuity / lifetime-fixed-cost per-solve emitters.

Called per-solve via ``write_entity_annual_calc_params``.

The six CSVs emitted are precision-sensitive: the values are float
annuities and discounted sums.  We pre-stringify with ``repr(float(v))``
to keep round-trip safety byte-for-byte.

Outputs (all written under ``solve_data/``):

  * ``ed_entity_annual.csv``                     (entity, period, value)
  * ``ed_entity_annual_discounted.csv``          (entity, period, value)
  * ``ed_entity_annual_divest.csv``              (entity, period, value)
  * ``ed_entity_annual_divest_discounted.csv``   (entity, period, value)
  * ``ed_lifetime_fixed_cost.csv``               (entity, period, value)
  * ``ed_lifetime_fixed_cost_divest.csv``        (entity, period, value)

Method-enum constants mirror flextool/flextool_base.dat:184 and
:211-212.  ``PdLookup`` is the native 4-branch resolver from
:mod:`._pdt_lookup` (same machinery used by ``_dispatchers`` and
``_period_params``).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.preprocess.emit._pdt_lookup import PdLookup
from flextool.engine.preprocess._annuity import (
    DEFAULT_DISCOUNT_RATE,
    annuity_factor,
    require_lifetime,
)
from flextool.engine.preprocess.params._branch import annuity_window_pairs
from flextool.engine.preprocess._provider_io import _emit
from flextool.engine.preprocess.params._existing import (
    DIVEST_NOT_ALLOWED as _DIVEST_NOT_ALLOWED,
    INVEST_NOT_ALLOWED as _INVEST_NOT_ALLOWED,
)


# ---------------------------------------------------------------------------
# Method-enum constants — mirror flextool/flextool_base.dat:211-212.
# ---------------------------------------------------------------------------

# The invest/retire method lists (_INVEST_NOT_ALLOWED / _DIVEST_NOT_ALLOWED)
# are imported from params/_existing.py, their one definition.


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as ``_per_solve``.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str],
              *, provider: "object | None" = None) -> pl.DataFrame:
    """Provider-only — returns empty all-Utf8 frame on Provider miss.
    Step 2.5 Phase C dropped the disk-fallback arm."""
    from flextool.engine.preprocess._provider_io import (
        _provider_key,
        _provider_lookup_positional,
    )
    seeded = _provider_lookup_positional(
        provider, _provider_key(path), path, columns,
    )
    if seeded is not None:
        return seeded
    return pl.DataFrame(
        {c: [] for c in columns},
        schema={c: pl.Utf8 for c in columns},
    )


def _read_singles(path: Path,
                   *, provider: "object | None" = None) -> list[str]:
    df = _read_csv(path, ["v"], provider=provider)
    return [v for v in df["v"].to_list() if v]


def _read_pairs(path: Path,
                 *, provider: "object | None" = None) -> list[tuple[str, str]]:
    df = _read_csv(path, ["a", "b"], provider=provider)
    return [(a, b) for a, b in zip(df["a"].to_list(), df["b"].to_list())
            if a and b]


def _read_keyed_value(path: Path,
                       *, provider: "object | None" = None) -> dict[str, float]:
    """``(key, value)`` 2-col CSV → ``dict[str, float]``, blanks skipped."""
    df = _read_csv(path, ["key", "value"], provider=provider)
    out: dict[str, float] = {}
    for k, v in zip(df["key"].to_list(), df["value"].to_list()):
        if not k or v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _read_pdv(path: Path,
               *, provider: "object | None" = None) -> dict[tuple[str, str], float]:
    """``(entity, period, value)`` 3-col CSV → dict."""
    df = _read_csv(path, ["entity", "period", "value"], provider=provider)
    out: dict[tuple[str, str], float] = {}
    for e, d, v in zip(df["entity"].to_list(),
                       df["period"].to_list(),
                       df["value"].to_list()):
        if not e or not d:
            continue
        try:
            out[(e, d)] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def _rows_to_frame(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """Materialise (entity, period, repr(value)) rows as an all-Utf8 frame.

    Pre-stringifies values via ``repr(float(v))`` so the polars
    ``write_csv`` output is byte-identical to the legacy
    ``f"{a},{b},{repr(float(v))}\\n"`` text emitter.
    """
    return pl.DataFrame(
        {
            "entity": [r[0] for r in rows],
            "period": [r[1] for r in rows],
            "value":  [repr(float(r[2])) for r in rows],
        },
        schema={"entity": pl.Utf8, "period": pl.Utf8, "value": pl.Utf8},
    )


# ---------------------------------------------------------------------------
# Numeric primitives.
# ---------------------------------------------------------------------------


def _annuity(lookup: PdLookup, e: str, d: str, cost_param: str) -> float:
    """``cost × 1000 × annuity_factor(r, n)`` of entity ``e`` in period ``d`` (:mod:`.._annuity`).

    A missing ``discount_rate`` is ``DEFAULT_DISCOUNT_RATE`` (the schema default); an explicit 0 stays 0 (the
    straight-line annuity ``cost / lifetime``).  A zero cost gives 0 whatever the lifetime; a non-zero cost
    without a positive lifetime is an error (:func:`require_lifetime`).
    """
    cost = lookup.get(e, cost_param, d)
    if cost == 0:
        return 0.0
    r = (lookup.get(e, "discount_rate", d) if lookup.has(e, "discount_rate", d)
         else DEFAULT_DISCOUNT_RATE)
    n = lookup.get(e, "lifetime", d) if lookup.has(e, "lifetime", d) else 0.0
    if not n > 0:
        require_lifetime([(e, d)], cost_param)
    return cost * 1000.0 * annuity_factor(r, n)


def _inflation_window_sum(
    pdy_d: float,
    life: float,
    period_in_use: list[str],
    p_discount_years: dict[str, float],
    inflation: dict[str, float],
    open_window: bool,
) -> float:
    """Σ inflation[d_all] over period_in_use with the lifetime window.

    ``open_window=True`` matches reinvest_automatic — sums forward
    from ``pdy_d`` with no upper bound.  ``open_window=False`` matches
    reinvest_choice / no_investment — sums only while
    ``pdy < pdy_d + life``.
    """
    s = 0.0
    upper = pdy_d + life
    for d_all in period_in_use:
        pdy = p_discount_years.get(d_all, 0.0)
        if pdy < pdy_d:
            continue
        if not open_window and pdy >= upper:
            continue
        s += inflation.get(d_all, 1.0)
    return s


def _window_periods(inp: dict, d: str) -> list[str]:
    """The ``period_in_use`` members the window sum of capacity built in ``d`` may count.

    In a stochastic fan each calendar period is counted once along a lineage
    (:func:`~flextool.engine.preprocess.params._branch.annuity_window_pairs`); a period outside
    ``period_in_use`` (history) and every period of a solve without a fan walk all of ``period_in_use``.
    """
    reach = inp["window_reach"]
    if reach is None or d not in reach:
        return inp["period_in_use"]
    return [x for x in inp["period_in_use"] if x in reach[d]]


# ---------------------------------------------------------------------------
# Top-level writer.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase E-b — derive_X family for each emitted CSV.
#
# The 6 CSVs share substantial input scans (PdLookup over pd_*, p_*, plus
# entity-class sets and period lists), so the wrapper builds the bundle
# once via :func:`_entity_annual_inputs` and threads it through private
# ``_compute_*`` helpers.  Public ``derive_*`` functions are thin
# rebuilders for standalone seed lookups.
# ---------------------------------------------------------------------------


def _entity_annual_inputs(input_dir: Path, solve_data_dir: Path,
                            *, provider: "object | None" = None) -> dict:
    """Shared input bundle for the 6 entity-annual derives."""
    pp = PdLookup(
        pd_csv=input_dir / "pd_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )
    pn = PdLookup(
        pd_csv=input_dir / "pd_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
        provider=provider,
    )
    methods_for_entity: dict[str, list[str]] = {}
    for e, m in _read_pairs(input_dir / "entity__invest_method.csv",
                             provider=provider):
        methods_for_entity.setdefault(e, []).append(m)
    lifetime_methods_for_entity: dict[str, list[str]] = {}
    for e, m in _read_pairs(solve_data_dir / "entity__lifetime_method.csv",
                             provider=provider):
        lifetime_methods_for_entity.setdefault(e, []).append(m)
    pairs = annuity_window_pairs(solve_data_dir.parent, provider=provider)
    window_reach: dict[str, set[str]] | None = None
    if pairs:
        window_reach = {}
        for x, y in pairs:
            window_reach.setdefault(x, set()).add(y)
    return {
        "window_reach": window_reach,
        "pp": pp,
        "pn": pn,
        "process_set": frozenset(_read_singles(input_dir / "process.csv", provider=provider)),
        "node_set": frozenset(_read_singles(input_dir / "node.csv", provider=provider)),
        "entityInvest": _read_singles(solve_data_dir / "entityInvest.csv", provider=provider),
        "entityDivest": _read_singles(solve_data_dir / "entityDivest.csv", provider=provider),
        "period_invest": _read_singles(
            solve_data_dir / "invest_periods_of_current_solve.csv", provider=provider),
        "period_in_use": _read_singles(
            solve_data_dir / "period_in_use_set.csv", provider=provider),
        "methods_for_entity": methods_for_entity,
        "lifetime_methods_for_entity": lifetime_methods_for_entity,
        "p_discount_years": _read_keyed_value(
            solve_data_dir / "p_discount_years.csv", provider=provider),
        "p_inflation_invest": _read_keyed_value(
            solve_data_dir / "p_inflation_factor_investment_yearly.csv", provider=provider),
        "p_inflation_ops": _read_keyed_value(
            solve_data_dir / "p_inflation_factor_operations_yearly.csv", provider=provider),
        "edEntity_lifetime": _read_pdv(
            solve_data_dir / "edEntity_lifetime.csv", provider=provider),
        "ed_fixed_cost": _read_pdv(solve_data_dir / "ed_fixed_cost.csv", provider=provider),
        "period_with_history": _read_singles(
            solve_data_dir / "period_with_history.csv", provider=provider),
        "entities": _read_singles(input_dir / "entity.csv", provider=provider),
    }


def _per_method_annuity_invest(inp: dict, e: str, d: str) -> float:
    v = 0.0
    for m in inp["methods_for_entity"].get(e, ()):
        if m in _INVEST_NOT_ALLOWED:
            continue
        if e in inp["node_set"]:
            v += _annuity(inp["pn"], e, d, "invest_cost")
        elif e in inp["process_set"]:
            v += _annuity(inp["pp"], e, d, "invest_cost")
    return v


def _per_method_annuity_divest(inp: dict, e: str, d: str) -> float:
    v = 0.0
    for m in inp["methods_for_entity"].get(e, ()):
        if m in _DIVEST_NOT_ALLOWED:
            continue
        if e in inp["node_set"]:
            v += _annuity(inp["pn"], e, d, "salvage_value")
        elif e in inp["process_set"]:
            v += _annuity(inp["pp"], e, d, "salvage_value")
    return v


def _ann_pair(inp: dict) -> tuple[
    list[tuple[str, str, float]], list[tuple[str, str, float]]
]:
    """Compute both ed_entity_annual and ed_entity_annual_discounted in one
    pass (they share the per-method annuity scan)."""
    rows_ann: list[tuple[str, str, float]] = []
    rows_ann_disc: list[tuple[str, str, float]] = []
    for e in inp["entityInvest"]:
        elm_set = frozenset(inp["lifetime_methods_for_entity"].get(e, ()))
        is_choice_or_no_invest = (
            "reinvest_choice" in elm_set or "no_investment" in elm_set
        )
        is_automatic = "reinvest_automatic" in elm_set
        for d in inp["period_invest"]:
            ann = _per_method_annuity_invest(inp, e, d)
            rows_ann.append((e, d, ann))

            disc = 0.0
            pdy_d = inp["p_discount_years"].get(d, 0.0)
            life = inp["edEntity_lifetime"].get((e, d), 0.0)
            if is_choice_or_no_invest:
                disc += ann * _inflation_window_sum(
                    pdy_d, life, _window_periods(inp, d),
                    inp["p_discount_years"], inp["p_inflation_invest"],
                    open_window=False,
                )
            if is_automatic:
                disc += ann * _inflation_window_sum(
                    pdy_d, life, _window_periods(inp, d),
                    inp["p_discount_years"], inp["p_inflation_invest"],
                    open_window=True,
                )
            rows_ann_disc.append((e, d, disc))
    return rows_ann, rows_ann_disc


def _div_pair(inp: dict) -> tuple[
    list[tuple[str, str, float]], list[tuple[str, str, float]]
]:
    """Compute both ed_entity_annual_divest and its _discounted variant."""
    rows_div: list[tuple[str, str, float]] = []
    rows_div_disc: list[tuple[str, str, float]] = []
    for e in inp["entityDivest"]:
        for d in inp["period_invest"]:
            ann = _per_method_annuity_divest(inp, e, d)
            rows_div.append((e, d, ann))

            pdy_d = inp["p_discount_years"].get(d, 0.0)
            disc = 0.0
            if e in inp["node_set"]:
                life = inp["pn"].get(e, "lifetime", d)
                disc = ann * _inflation_window_sum(
                    pdy_d, life, _window_periods(inp, d),
                    inp["p_discount_years"], inp["p_inflation_invest"],
                    open_window=False,
                )
            elif e in inp["process_set"]:
                life = inp["pp"].get(e, "lifetime", d)
                disc = ann * _inflation_window_sum(
                    pdy_d, life, _window_periods(inp, d),
                    inp["p_discount_years"], inp["p_inflation_invest"],
                    open_window=False,
                )
            rows_div_disc.append((e, d, disc))
    return rows_div, rows_div_disc


def _rows_lifetime_fixed_cost(inp: dict) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    for e in inp["entities"]:
        elm_set = frozenset(inp["lifetime_methods_for_entity"].get(e, ()))
        is_choice_or_no_invest = (
            "reinvest_choice" in elm_set or "no_investment" in elm_set
        )
        is_automatic = "reinvest_automatic" in elm_set
        for d in inp["period_with_history"]:
            fc = inp["ed_fixed_cost"].get((e, d), 0.0)
            v = 0.0
            pdy_d = inp["p_discount_years"].get(d, 0.0)
            life = inp["edEntity_lifetime"].get((e, d), 0.0)
            if is_choice_or_no_invest:
                v += fc * _inflation_window_sum(
                    pdy_d, life, _window_periods(inp, d),
                    inp["p_discount_years"], inp["p_inflation_ops"],
                    open_window=False,
                )
            if is_automatic:
                v += fc * _inflation_window_sum(
                    pdy_d, life, _window_periods(inp, d),
                    inp["p_discount_years"], inp["p_inflation_ops"],
                    open_window=True,
                )
            rows.append((e, d, v))
    return rows


def _rows_lifetime_fixed_cost_divest(inp: dict) -> list[tuple[str, str, float]]:
    # NB: mod L1651 deliberately uses p_inflation_factor_INVESTMENT_yearly
    # here (not operations-yearly) — asymmetric vs the non-divest variant.
    rows: list[tuple[str, str, float]] = []
    for e in inp["entityDivest"]:
        for d in inp["period_invest"]:
            fc = inp["ed_fixed_cost"].get((e, d), 0.0)
            pdy_d = inp["p_discount_years"].get(d, 0.0)
            if e in inp["node_set"]:
                life = inp["pn"].get(e, "lifetime", d)
            elif e in inp["process_set"]:
                life = inp["pp"].get(e, "lifetime", d)
            else:
                life = 0.0
            v = fc * _inflation_window_sum(
                pdy_d, life, _window_periods(inp, d),
                inp["p_discount_years"], inp["p_inflation_invest"],
                open_window=False,
            )
            rows.append((e, d, v))
    return rows


def emit_entity_annual_calc_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``entity_annual_calc_params`` to the Provider.
    Emits the same six frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration: basename and parent/basename).
    """
    inp = _entity_annual_inputs(input_dir, solve_data_dir, provider=provider)

    rows_ann, rows_ann_disc = _ann_pair(inp)
    _emit(provider, "solve_data/ed_entity_annual.csv",
          _rows_to_frame(rows_ann))
    _emit(provider, "solve_data/ed_entity_annual_discounted.csv",
          _rows_to_frame(rows_ann_disc))

    rows_div, rows_div_disc = _div_pair(inp)
    _emit(provider, "solve_data/ed_entity_annual_divest.csv",
          _rows_to_frame(rows_div))
    _emit(provider, "solve_data/ed_entity_annual_divest_discounted.csv",
          _rows_to_frame(rows_div_disc))

    _emit(provider, "solve_data/ed_lifetime_fixed_cost.csv",
          _rows_to_frame(_rows_lifetime_fixed_cost(inp)))
    _emit(provider, "solve_data/ed_lifetime_fixed_cost_divest.csv",
          _rows_to_frame(_rows_lifetime_fixed_cost_divest(inp)))
