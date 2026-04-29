"""Annuity/discounted calc params: ed_entity_annual + lifetime-fixed-cost family.

Migrated from flextool.mod:

    L1542 ed_entity_annual               (invest_cost annuity per allowed
                                           method, summed over methods)
    L1572 ed_entity_annual_discounted    (lifetime-aware inflation sum)
    L1591 ed_entity_annual_divest        (salvage_value annuity)
    L1602 ed_entity_annual_divest_discounted
    L1625 ed_lifetime_fixed_cost         (discounted lifetime fixed cost)
    L1644 ed_lifetime_fixed_cost_divest

Annuity formula:
    annuity = pdX[e, costParam, d] * 1000 * r / (1 - 1/(1+r)^n)
where r = pdX[e, 'discount_rate', d] (default 0.05 if ≤ 0) and
      n = pdX[e, 'lifetime', d]      (default 20   if ≤ 0).

Sum is over methods m ∈ entity__invest_method ∩ allowed-set, so the
final value is (#allowed_methods_for_e) × per_method_annuity. This
mirrors mod's `sum{m : (e,m) in entity__invest_method && ...}` shape.

The discounted-annual params accumulate the per-(d_all) inflation
factors over period_in_use, gated by lifetime windows.
"""
from __future__ import annotations

import csv
from pathlib import Path

from flextool.flextoolrunner.preprocessing.pd_lookups import PdLookup


# Mirror flextool/flextool_base.dat:184 — the full enum.
_INVEST_METHOD: tuple[str, ...] = (
    "not_allowed", "invest_no_limit", "invest_period", "invest_total",
    "invest_period_total",
    "retire_no_limit", "retire_period", "retire_total", "retire_period_total",
    "invest_retire_no_limit", "invest_retire_period", "invest_retire_total",
    "invest_retire_period_total", "cumulative_limits",
)

# Mirror flextool/flextool_base.dat:211-212.
_INVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "retire_period", "retire_total", "retire_no_limit",
))
_DIVEST_NOT_ALLOWED: frozenset[str] = frozenset((
    "not_allowed", "invest_period", "invest_total", "invest_no_limit",
))


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


def _read_keyed_value(path: Path, value_col_idx: int = 1) -> dict[str, float]:
    """Read CSV with first col as key, ``value_col_idx`` as float value."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) > value_col_idx and row[0]:
                try:
                    out[row[0]] = float(row[value_col_idx])
                except ValueError:
                    continue
    return out


def _write_keyed_2(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, float]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{repr(v)}\n" for a, b, v in rows))


def _annuity(invest_value: float, discount_rate: float, lifetime: float) -> float:
    """invest_value * 1000 * r / (1 - 1/(1+r)^n) with mod's ≤ 0 fallbacks."""
    r = discount_rate if discount_rate > 0 else 0.05
    n = lifetime if lifetime > 0 else 20.0
    if r == 0:  # would divide by zero — mod's `else 0.05` prevents this
        return 0.0
    return invest_value * 1000.0 * r / (1.0 - (1.0 / (1.0 + r)) ** n)


def write_entity_annual_calc_params(input_dir: Path, solve_data_dir: Path) -> None:
    """Six annuity/discounted params keyed on entityInvest×period_invest etc."""
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

    # entity__invest_method: (entity, method) — loaded by mod from
    # input/entity__invest_method.csv (NOT solve_data/, which is empty).
    eim = _read_pairs(input_dir / "entity__invest_method.csv")
    methods_for_entity: dict[str, list[str]] = {}
    for e, m in eim:
        methods_for_entity.setdefault(e, []).append(m)

    # entity__lifetime_method: (entity, lifetime_method).
    elm = _read_pairs(solve_data_dir / "entity__lifetime_method.csv")
    lifetime_methods_for_entity: dict[str, list[str]] = {}
    for e, m in elm:
        lifetime_methods_for_entity.setdefault(e, []).append(m)

    p_discount_years = _read_keyed_value(
        solve_data_dir / "p_discount_years.csv", value_col_idx=1
    )
    p_inflation_invest = _read_keyed_value(
        solve_data_dir / "p_inflation_factor_investment_yearly.csv",
        value_col_idx=1,
    )
    p_inflation_ops = _read_keyed_value(
        solve_data_dir / "p_inflation_factor_operations_yearly.csv",
        value_col_idx=1,
    )
    edEntity_lifetime: dict[tuple[str, str], float] = {}
    el_path = solve_data_dir / "edEntity_lifetime.csv"
    if el_path.exists():
        with el_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    try:
                        edEntity_lifetime[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue
    ed_fixed_cost: dict[tuple[str, str], float] = {}
    fc_path = solve_data_dir / "ed_fixed_cost.csv"
    if fc_path.exists():
        with fc_path.open() as fh:
            reader = csv.reader(fh)
            next(reader, None)
            for row in reader:
                if len(row) >= 3:
                    try:
                        ed_fixed_cost[(row[0], row[1])] = float(row[2])
                    except ValueError:
                        continue

    # ---- ed_entity_annual{e in entityInvest, d in period_invest} -------
    def _per_method_annuity_invest(e: str, d: str) -> float:
        # Sum the per-method annuity over methods m where:
        #   (e, m) ∈ entity__invest_method, m not in invest_method_not_allowed
        #   AND e is in node OR process (mod splits the sum into two
        #   class-specific blocks).
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

            # ed_entity_annual_discounted:
            #   for choice/no_investment: ann * sum_{d_all : pdy_d_all ∈ [pdy_d, pdy_d + lifetime)} infl_invest[d_all]
            #   + for reinvest_automatic: ann * sum_{d_all : pdy_d_all ≥ pdy_d} infl_invest[d_all]
            disc = 0.0
            pdy_d = p_discount_years.get(d, 0.0)
            life = edEntity_lifetime.get((e, d), 0.0)
            if is_choice_or_no_invest:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += p_inflation_invest.get(d_all, 1.0)
                disc += ann * s
            if is_automatic:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years.get(d_all, 0.0)
                    if pdy >= pdy_d:
                        s += p_inflation_invest.get(d_all, 1.0)
                disc += ann * s
            rows_ann_disc.append((e, d, disc))
    _write_keyed_2(solve_data_dir / "ed_entity_annual.csv",
                   ("entity", "period", "value"), rows_ann)
    _write_keyed_2(solve_data_dir / "ed_entity_annual_discounted.csv",
                   ("entity", "period", "value"), rows_ann_disc)

    # ---- ed_entity_annual_divest{e in entityDivest, d in period_invest} ----
    rows_div: list[tuple[str, str, float]] = []
    rows_div_disc: list[tuple[str, str, float]] = []
    for e in entityDivest:
        for d in period_invest:
            ann = _per_method_annuity_divest(e, d)
            rows_div.append((e, d, ann))

            # ed_entity_annual_divest_discounted:
            #   if e ∈ node: ann * sum_{d_all : pdy_d_all ∈ [pdy_d, pdy_d + pdNode_lifetime)} infl_invest[d_all]
            #   else if e ∈ process: ann * sum_{...pdProcess_lifetime} infl_invest[d_all]
            disc = 0.0
            pdy_d = p_discount_years.get(d, 0.0)
            if e in node_set:
                life = pn.get(e, "lifetime", d)
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += p_inflation_invest.get(d_all, 1.0)
                disc = ann * s
            elif e in process_set:
                life = pp.get(e, "lifetime", d)
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += p_inflation_invest.get(d_all, 1.0)
                disc = ann * s
            rows_div_disc.append((e, d, disc))
    _write_keyed_2(solve_data_dir / "ed_entity_annual_divest.csv",
                   ("entity", "period", "value"), rows_div)
    _write_keyed_2(solve_data_dir / "ed_entity_annual_divest_discounted.csv",
                   ("entity", "period", "value"), rows_div_disc)

    # ---- ed_lifetime_fixed_cost{e in entity, d in period_with_history} ----
    period_with_history = _read_singles(solve_data_dir / "period_with_history.csv")
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
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years.get(d_all, 0.0)
                    if pdy >= pdy_d and pdy < pdy_d + life:
                        s += p_inflation_ops.get(d_all, 1.0)
                v += fc * s
            if is_automatic:
                s = 0.0
                for d_all in period_in_use:
                    pdy = p_discount_years.get(d_all, 0.0)
                    if pdy >= pdy_d:
                        s += p_inflation_ops.get(d_all, 1.0)
                v += fc * s
            rows_lfc.append((e, d, v))
    _write_keyed_2(solve_data_dir / "ed_lifetime_fixed_cost.csv",
                   ("entity", "period", "value"), rows_lfc)

    # ---- ed_lifetime_fixed_cost_divest{e in entityDivest, d in period_invest}
    # NB: mod L1651 uses p_inflation_factor_INVESTMENT_yearly here, not the
    # operations-yearly factor — ASSYMETRIC vs the non-divest variant.
    rows_lfcd: list[tuple[str, str, float]] = []
    for e in entityDivest:
        for d in period_invest:
            fc = ed_fixed_cost.get((e, d), 0.0)
            v = 0.0
            pdy_d = p_discount_years.get(d, 0.0)
            if e in node_set:
                life = pn.get(e, "lifetime", d)
            elif e in process_set:
                life = pp.get(e, "lifetime", d)
            else:
                life = 0.0
            s = 0.0
            for d_all in period_in_use:
                pdy = p_discount_years.get(d_all, 0.0)
                if pdy >= pdy_d and pdy < pdy_d + life:
                    s += p_inflation_invest.get(d_all, 1.0)
            v = fc * s
            rows_lfcd.append((e, d, v))
    _write_keyed_2(solve_data_dir / "ed_lifetime_fixed_cost_divest.csv",
                   ("entity", "period", "value"), rows_lfcd)
