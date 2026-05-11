"""Writer-port Phase 2 (sub-dispatch 2) — annuity / lifetime-fixed-cost.

Native polars port of
``flextool.flextoolrunner.preprocessing.entity_annual_calc_params``
(legacy ~348 LOC).  Called per-solve from
``flextool.flextoolrunner.preprocessing.solve_time.run`` via
``write_entity_annual_calc_params``.

The six CSVs emitted are precision-sensitive: the values are float
annuities and discounted sums.  We pre-stringify with ``repr(float(v))``
to match the legacy emitter byte-for-byte.

Outputs (all written under ``solve_data/``):

  * ``ed_entity_annual.csv``                     (entity, period, value)
  * ``ed_entity_annual_discounted.csv``          (entity, period, value)
  * ``ed_entity_annual_divest.csv``              (entity, period, value)
  * ``ed_entity_annual_divest_discounted.csv``   (entity, period, value)
  * ``ed_lifetime_fixed_cost.csv``               (entity, period, value)
  * ``ed_lifetime_fixed_cost_divest.csv``        (entity, period, value)

Method-enum constants mirror flextool/flextool_base.dat:184 and
:211-212.  ``PdLookup`` is the native 4-branch resolver from
:mod:`._pdt_lookup` (same machinery used by ``_writer_dispatchers`` and
``_writer_period_params``).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine_polars._pdt_lookup import PdLookup


# ---------------------------------------------------------------------------
# Method-enum constants — mirror flextool/flextool_base.dat:211-212.
# ---------------------------------------------------------------------------

_INVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))


# ---------------------------------------------------------------------------
# CSV I/O helpers — same conventions as ``_writer_per_solve``.
# ---------------------------------------------------------------------------


def _read_csv(path: Path, columns: list[str]) -> pl.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pl.DataFrame(
            {c: [] for c in columns},
            schema={c: pl.Utf8 for c in columns},
        )
    df = pl.read_csv(
        path,
        has_header=True,
        infer_schema_length=0,
        truncate_ragged_lines=True,
    )
    keep = df.columns[: len(columns)]
    df = df.select(keep)
    df.columns = columns
    return df


def _read_singles(path: Path) -> list[str]:
    df = _read_csv(path, ["v"])
    return [v for v in df["v"].to_list() if v]


def _read_pairs(path: Path) -> list[tuple[str, str]]:
    df = _read_csv(path, ["a", "b"])
    return [(a, b) for a, b in zip(df["a"].to_list(), df["b"].to_list())
            if a and b]


def _read_keyed_value(path: Path) -> dict[str, float]:
    """``(key, value)`` 2-col CSV → ``dict[str, float]``, blanks skipped."""
    df = _read_csv(path, ["key", "value"])
    out: dict[str, float] = {}
    for k, v in zip(df["key"].to_list(), df["value"].to_list()):
        if not k or v is None or v == "":
            continue
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def _read_pdv(path: Path) -> dict[tuple[str, str], float]:
    """``(entity, period, value)`` 3-col CSV → dict."""
    df = _read_csv(path, ["entity", "period", "value"])
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


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    """Emit (entity, period, value) CSV with ``repr(float)`` precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",".join(header) + "\n"
        + "".join(f"{a},{b},{repr(float(v))}\n" for a, b, v in rows)
    )


# ---------------------------------------------------------------------------
# Numeric primitives.
# ---------------------------------------------------------------------------


def _annuity(invest_value: float, discount_rate: float, lifetime: float) -> float:
    """Mod-faithful annuity:  inv * 1000 * r / (1 - 1/(1+r)^n).

    Falls back to ``r = 0.05`` when discount_rate ≤ 0 and ``n = 20`` when
    lifetime ≤ 0 (mirrors the ``else 0.05`` / ``else 20`` legs in
    ``flextool.mod``).
    """
    r = discount_rate if discount_rate > 0 else 0.05
    n = lifetime if lifetime > 0 else 20.0
    if r == 0:
        return 0.0
    return invest_value * 1000.0 * r / (1.0 - (1.0 / (1.0 + r)) ** n)


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


# ---------------------------------------------------------------------------
# Top-level writer.
# ---------------------------------------------------------------------------


def write_entity_annual_calc_params(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Native port of ``entity_annual_calc_params.write_entity_annual_calc_params``.

    Reads ``input/`` immutables (``pd_process``, ``p_process``,
    ``pd_node``, ``p_node``, ``entity__invest_method``, ``entity``,
    ``process``, ``node``) and per-solve ``solve_data/`` files
    (period sets, discount factors, lifetime / fixed-cost overrides).
    Emits the six CSVs documented at module level.
    """
    pp = PdLookup(
        pd_csv=input_dir / "pd_process.csv",
        p_csv=input_dir / "p_process.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
    )
    pn = PdLookup(
        pd_csv=input_dir / "pd_node.csv",
        p_csv=input_dir / "p_node.csv",
        period_branch_csv=solve_data_dir / "period__branch.csv",
    )
    process_set = frozenset(_read_singles(input_dir / "process.csv"))
    node_set = frozenset(_read_singles(input_dir / "node.csv"))

    entityInvest = _read_singles(solve_data_dir / "entityInvest.csv")
    entityDivest = _read_singles(solve_data_dir / "entityDivest.csv")
    period_invest = _read_singles(
        solve_data_dir / "invest_periods_of_current_solve.csv"
    )
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")

    # entity__invest_method lives in input/ (NOT solve_data/, which is
    # blank — see mod's load directive).
    methods_for_entity: dict[str, list[str]] = {}
    for e, m in _read_pairs(input_dir / "entity__invest_method.csv"):
        methods_for_entity.setdefault(e, []).append(m)

    lifetime_methods_for_entity: dict[str, list[str]] = {}
    for e, m in _read_pairs(solve_data_dir / "entity__lifetime_method.csv"):
        lifetime_methods_for_entity.setdefault(e, []).append(m)

    p_discount_years = _read_keyed_value(
        solve_data_dir / "p_discount_years.csv"
    )
    p_inflation_invest = _read_keyed_value(
        solve_data_dir / "p_inflation_factor_investment_yearly.csv"
    )
    p_inflation_ops = _read_keyed_value(
        solve_data_dir / "p_inflation_factor_operations_yearly.csv"
    )
    edEntity_lifetime = _read_pdv(solve_data_dir / "edEntity_lifetime.csv")
    ed_fixed_cost = _read_pdv(solve_data_dir / "ed_fixed_cost.csv")

    # ── Annuity sums (per-method, gated by entity class) ─────────────────
    # Mirrors mod L1542-1601: for each allowed method (entity__invest_method
    # ∩ ¬not_allowed), add one per-method annuity built from class-specific
    # cost / discount / lifetime lookups.  The sum collapses to
    # ``(#allowed_methods) × per_method_annuity`` — preserved exactly.

    def _per_method_annuity_invest(e: str, d: str) -> float:
        v = 0.0
        for m in methods_for_entity.get(e, ()):
            if m in _INVEST_NOT_ALLOWED:
                continue
            if e in node_set:
                v += _annuity(
                    pn.get(e, "invest_cost", d),
                    pn.get(e, "discount_rate", d),
                    pn.get(e, "lifetime", d),
                )
            elif e in process_set:
                v += _annuity(
                    pp.get(e, "invest_cost", d),
                    pp.get(e, "discount_rate", d),
                    pp.get(e, "lifetime", d),
                )
        return v

    def _per_method_annuity_divest(e: str, d: str) -> float:
        v = 0.0
        for m in methods_for_entity.get(e, ()):
            if m in _DIVEST_NOT_ALLOWED:
                continue
            if e in node_set:
                v += _annuity(
                    pn.get(e, "salvage_value", d),
                    pn.get(e, "discount_rate", d),
                    pn.get(e, "lifetime", d),
                )
            elif e in process_set:
                v += _annuity(
                    pp.get(e, "salvage_value", d),
                    pp.get(e, "discount_rate", d),
                    pp.get(e, "lifetime", d),
                )
        return v

    # ── ed_entity_annual + ed_entity_annual_discounted ───────────────────
    rows_ann: list[tuple[str, str, float]] = []
    rows_ann_disc: list[tuple[str, str, float]] = []
    for e in entityInvest:
        elm_set = frozenset(lifetime_methods_for_entity.get(e, ()))
        is_choice_or_no_invest = (
            "reinvest_choice" in elm_set or "no_investment" in elm_set
        )
        is_automatic = "reinvest_automatic" in elm_set
        for d in period_invest:
            ann = _per_method_annuity_invest(e, d)
            rows_ann.append((e, d, ann))

            disc = 0.0
            pdy_d = p_discount_years.get(d, 0.0)
            life = edEntity_lifetime.get((e, d), 0.0)
            if is_choice_or_no_invest:
                disc += ann * _inflation_window_sum(
                    pdy_d, life, period_in_use,
                    p_discount_years, p_inflation_invest,
                    open_window=False,
                )
            if is_automatic:
                disc += ann * _inflation_window_sum(
                    pdy_d, life, period_in_use,
                    p_discount_years, p_inflation_invest,
                    open_window=True,
                )
            rows_ann_disc.append((e, d, disc))
    _write_keyed_2(
        solve_data_dir / "ed_entity_annual.csv",
        ("entity", "period", "value"), rows_ann,
    )
    _write_keyed_2(
        solve_data_dir / "ed_entity_annual_discounted.csv",
        ("entity", "period", "value"), rows_ann_disc,
    )

    # ── ed_entity_annual_divest + _discounted ────────────────────────────
    # Divest uses pdNode_lifetime / pdProcess_lifetime directly (not the
    # edEntity_lifetime override).
    rows_div: list[tuple[str, str, float]] = []
    rows_div_disc: list[tuple[str, str, float]] = []
    for e in entityDivest:
        for d in period_invest:
            ann = _per_method_annuity_divest(e, d)
            rows_div.append((e, d, ann))

            pdy_d = p_discount_years.get(d, 0.0)
            disc = 0.0
            if e in node_set:
                life = pn.get(e, "lifetime", d)
                disc = ann * _inflation_window_sum(
                    pdy_d, life, period_in_use,
                    p_discount_years, p_inflation_invest,
                    open_window=False,
                )
            elif e in process_set:
                life = pp.get(e, "lifetime", d)
                disc = ann * _inflation_window_sum(
                    pdy_d, life, period_in_use,
                    p_discount_years, p_inflation_invest,
                    open_window=False,
                )
            rows_div_disc.append((e, d, disc))
    _write_keyed_2(
        solve_data_dir / "ed_entity_annual_divest.csv",
        ("entity", "period", "value"), rows_div,
    )
    _write_keyed_2(
        solve_data_dir / "ed_entity_annual_divest_discounted.csv",
        ("entity", "period", "value"), rows_div_disc,
    )

    # ── ed_lifetime_fixed_cost (e ∈ entity, d ∈ period_with_history) ─────
    period_with_history = _read_singles(
        solve_data_dir / "period_with_history.csv"
    )
    entities = _read_singles(input_dir / "entity.csv")
    rows_lfc: list[tuple[str, str, float]] = []
    for e in entities:
        elm_set = frozenset(lifetime_methods_for_entity.get(e, ()))
        is_choice_or_no_invest = (
            "reinvest_choice" in elm_set or "no_investment" in elm_set
        )
        is_automatic = "reinvest_automatic" in elm_set
        for d in period_with_history:
            fc = ed_fixed_cost.get((e, d), 0.0)
            v = 0.0
            pdy_d = p_discount_years.get(d, 0.0)
            life = edEntity_lifetime.get((e, d), 0.0)
            if is_choice_or_no_invest:
                v += fc * _inflation_window_sum(
                    pdy_d, life, period_in_use,
                    p_discount_years, p_inflation_ops,
                    open_window=False,
                )
            if is_automatic:
                v += fc * _inflation_window_sum(
                    pdy_d, life, period_in_use,
                    p_discount_years, p_inflation_ops,
                    open_window=True,
                )
            rows_lfc.append((e, d, v))
    _write_keyed_2(
        solve_data_dir / "ed_lifetime_fixed_cost.csv",
        ("entity", "period", "value"), rows_lfc,
    )

    # ── ed_lifetime_fixed_cost_divest ────────────────────────────────────
    # NB: mod L1651 deliberately uses p_inflation_factor_INVESTMENT_yearly
    # here (not operations-yearly) — asymmetric vs the non-divest variant.
    rows_lfcd: list[tuple[str, str, float]] = []
    for e in entityDivest:
        for d in period_invest:
            fc = ed_fixed_cost.get((e, d), 0.0)
            pdy_d = p_discount_years.get(d, 0.0)
            if e in node_set:
                life = pn.get(e, "lifetime", d)
            elif e in process_set:
                life = pp.get(e, "lifetime", d)
            else:
                life = 0.0
            v = fc * _inflation_window_sum(
                pdy_d, life, period_in_use,
                p_discount_years, p_inflation_invest,
                open_window=False,
            )
            rows_lfcd.append((e, d, v))
    _write_keyed_2(
        solve_data_dir / "ed_lifetime_fixed_cost_divest.csv",
        ("entity", "period", "value"), rows_lfcd,
    )
