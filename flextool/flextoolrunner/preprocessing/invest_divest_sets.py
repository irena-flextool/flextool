"""ed_invest / ed_divest family — entity × period invest/divest domain sets.

Migrated from flextool.mod:1576-1605 (and related). These sets feed the
investment/retirement constraints' iteration domains.

Migrated (15 sets):
    ed_invest                  L1576  entityInvest × period_invest filtered
    ed_divest                  L1590  entityDivest × period_invest filtered
    ed_invest_period           L1577  ed_invest filtered by method
    ed_divest_period           L1591
    ed_invest_cumulative       L1580
    pd_invest, nd_invest       L1588-1589
    pd_divest, nd_divest       L1594-1595
    edd_history_choice         L1581
    edd_history_automatic      L1582
    edd_history_no_investment  L1583
    edd_history                L1584  union
    edd_history_invest         L1585
    edd_invest                 L1586

    gd_invest                  L1597
    gd_invest_period           L1598
    gd_divest                  L1601 (let me verify)
    gd_divest_period           L1602
"""
from __future__ import annotations

import csv
from pathlib import Path


# Method enums (used in filter conditions; stable model invariants).
_INVEST_PERIOD_METHODS = frozenset((
    "invest_period", "invest_period_total",
    "invest_retire_period", "invest_retire_period_total",
))
_DIVEST_PERIOD_METHODS = frozenset((
    "retire_period", "retire_period_total",
    "invest_retire_period", "invest_retire_period_total",
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


def _read_pdkv(path: Path) -> dict[tuple[str, str], float]:
    """3-col (entity, period, value)."""
    out: dict[tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                try:
                    out[(row[0], row[1])] = float(row[2])
                except ValueError:
                    continue
    return out


def _read_keyed_value(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0]:
                try:
                    out[row[0]] = float(row[1])
                except ValueError:
                    continue
    return out


def _write_pairs(path: Path, header: tuple[str, str],
                 rows: list[tuple[str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b}\n" for a, b in rows))


def _write_triples(path: Path, header: tuple[str, str, str],
                   rows: list[tuple[str, str, str]]) -> None:
    path.write_text(",".join(header) + "\n"
                    + "".join(f"{a},{b},{c}\n" for a, b, c in rows))


def write_invest_divest_sets(input_dir: Path, solve_data_dir: Path) -> None:
    entityInvest = _read_singles(solve_data_dir / "entityInvest.csv")
    entityDivest = _read_singles(solve_data_dir / "entityDivest.csv")
    period_invest = _read_singles(
        solve_data_dir / "invest_periods_of_current_solve.csv"
    )
    period_in_use = _read_singles(solve_data_dir / "period_in_use_set.csv")
    period_with_history = _read_singles(
        solve_data_dir / "period_with_history.csv"
    )
    process_set = frozenset(_read_singles(input_dir / "process.csv"))
    node_set = frozenset(_read_singles(input_dir / "node.csv"))
    entity_set = _read_singles(input_dir / "entity.csv")

    # Capacity-constraint sets (input/ — loaded as 2-col entity, constraint).
    pcc_inv = _read_pairs(input_dir / "p_process_constraint_invested_capacity_coefficient.csv")
    pcc_pre = _read_pairs(input_dir / "p_process_constraint_pre_built_capacity_coefficient.csv")
    ncc_inv = _read_pairs(input_dir / "p_node_constraint_invested_capacity_coefficient.csv")
    ncc_pre = _read_pairs(input_dir / "p_node_constraint_pre_built_capacity_coefficient.csv")
    has_pcc_inv = frozenset(p for p, _ in pcc_inv)
    has_pcc_pre = frozenset(p for p, _ in pcc_pre)
    has_ncc_inv = frozenset(n for n, _ in ncc_inv)
    has_ncc_pre = frozenset(n for n, _ in ncc_pre)

    eea = _read_pdkv(solve_data_dir / "ed_entity_annual.csv")
    eead = _read_pdkv(solve_data_dir / "ed_entity_annual_divest.csv")

    # entity__invest_method (input/)
    eim = _read_pairs(input_dir / "entity__invest_method.csv")
    methods_for_e: dict[str, frozenset[str]] = {}
    _eim_acc: dict[str, dict[str, None]] = {}
    for e, m in eim:
        _eim_acc.setdefault(e, {})[m] = None
    methods_for_e = {e: frozenset(d.keys()) for e, d in _eim_acc.items()}

    # entity__lifetime_method (Python)
    elm = _read_pairs(solve_data_dir / "entity__lifetime_method.csv")
    lm_for_e: dict[str, frozenset[str]] = {}
    _elm_acc: dict[str, dict[str, None]] = {}
    for e, m in elm:
        _elm_acc.setdefault(e, {})[m] = None
    lm_for_e = {e: frozenset(d.keys()) for e, d in _elm_acc.items()}

    p_years_d = _read_keyed_value(solve_data_dir / "p_years_d.csv")
    edEntity_lifetime = _read_pdkv(solve_data_dir / "edEntity_lifetime.csv")

    # group_invest, group_divest (Python)
    group_invest = _read_singles(solve_data_dir / "group_invest.csv")
    group_divest = _read_singles(solve_data_dir / "group_divest.csv")
    group_entity = _read_pairs(solve_data_dir / "group_entity.csv")
    entities_for_g: dict[str, list[str]] = {}
    for g, e in group_entity:
        entities_for_g.setdefault(g, []).append(e)
    # group__invest_method (input/)
    gim = _read_pairs(input_dir / "group__invest_method.csv")
    methods_for_g: dict[str, frozenset[str]] = {}
    _gim_acc: dict[str, dict[str, None]] = {}
    for g, m in gim:
        _gim_acc.setdefault(g, {})[m] = None
    methods_for_g = {g: frozenset(d.keys()) for g, d in _gim_acc.items()}

    # ---- ed_invest ------------------------------------------------------
    def _has_capacity_constraint_invest(e: str) -> bool:
        return e in has_pcc_inv or e in has_ncc_inv \
            or e in has_pcc_pre or e in has_ncc_pre

    ed_invest_pairs: list[tuple[str, str]] = []
    for e in entityInvest:
        for d in period_invest:
            if eea.get((e, d), 0.0) != 0.0 or _has_capacity_constraint_invest(e):
                ed_invest_pairs.append((e, d))
    _write_pairs(solve_data_dir / "ed_invest.csv",
                 ("entity", "period"), ed_invest_pairs)
    ed_invest_set = frozenset(ed_invest_pairs)

    # ---- ed_divest ------------------------------------------------------
    ed_divest_pairs: list[tuple[str, str]] = []
    for e in entityDivest:
        for d in period_invest:
            if eead.get((e, d), 0.0) != 0.0 or _has_capacity_constraint_invest(e):
                ed_divest_pairs.append((e, d))
    _write_pairs(solve_data_dir / "ed_divest.csv",
                 ("entity", "period"), ed_divest_pairs)
    ed_divest_set = frozenset(ed_divest_pairs)

    # ---- ed_invest_period -----------------------------------------------
    rows = [(e, d) for e, d in ed_invest_pairs
            if methods_for_e.get(e, frozenset()) & _INVEST_PERIOD_METHODS]
    _write_pairs(solve_data_dir / "ed_invest_period.csv",
                 ("entity", "period"), rows)

    # ---- ed_divest_period -- mod uses ed_invest (not ed_divest) as universe ----
    rows = [(e, d) for e, d in ed_invest_pairs
            if methods_for_e.get(e, frozenset()) & _DIVEST_PERIOD_METHODS]
    _write_pairs(solve_data_dir / "ed_divest_period.csv",
                 ("entity", "period"), rows)

    # ---- ed_invest_cumulative -------------------------------------------
    rows = [(e, d) for e, d in ed_invest_pairs
            if "cumulative_limits" in methods_for_e.get(e, frozenset())]
    _write_pairs(solve_data_dir / "ed_invest_cumulative.csv",
                 ("entity", "period"), rows)

    # ---- pd_invest / nd_invest / pd_divest / nd_divest ------------------
    _write_pairs(solve_data_dir / "pd_invest.csv",
                 ("process", "period"),
                 [(e, d) for e, d in ed_invest_pairs if e in process_set])
    _write_pairs(solve_data_dir / "nd_invest.csv",
                 ("node", "period"),
                 [(e, d) for e, d in ed_invest_pairs if e in node_set])
    _write_pairs(solve_data_dir / "pd_divest.csv",
                 ("process", "period"),
                 [(e, d) for e, d in ed_divest_pairs if e in process_set])
    _write_pairs(solve_data_dir / "nd_divest.csv",
                 ("node", "period"),
                 [(e, d) for e, d in ed_divest_pairs if e in node_set])

    # ---- edd_history_choice / _automatic / _no_investment ---------------
    edd_choice: list[tuple[str, str, str]] = []
    edd_auto: list[tuple[str, str, str]] = []
    edd_noinv: list[tuple[str, str, str]] = []
    for e in entity_set:
        e_lm = lm_for_e.get(e, frozenset())
        is_choice = "reinvest_choice" in e_lm
        is_auto = "reinvest_automatic" in e_lm
        is_noinv = "no_investment" in e_lm
        if not (is_choice or is_auto or is_noinv):
            continue
        for d_h in period_with_history:
            life = edEntity_lifetime.get((e, d_h), 0.0)
            pdy_dh = p_years_d.get(d_h, 0.0)
            for d in period_in_use:
                pdy_d = p_years_d.get(d, 0.0)
                if is_choice and pdy_d >= pdy_dh and pdy_d < pdy_dh + life:
                    edd_choice.append((e, d_h, d))
                if is_auto and pdy_d >= pdy_dh:
                    edd_auto.append((e, d_h, d))
                if is_noinv and pdy_d >= pdy_dh and pdy_d < pdy_dh + life:
                    edd_noinv.append((e, d_h, d))
    _write_triples(solve_data_dir / "edd_history_choice.csv",
                   ("entity", "period_history", "period"), edd_choice)
    _write_triples(solve_data_dir / "edd_history_automatic.csv",
                   ("entity", "period_history", "period"), edd_auto)
    _write_triples(solve_data_dir / "edd_history_no_investment.csv",
                   ("entity", "period_history", "period"), edd_noinv)

    # ---- edd_history (union), edd_history_invest, edd_invest ------------
    edd_seen: dict[tuple[str, str, str], None] = {}
    for r in edd_choice + edd_auto + edd_noinv:
        edd_seen.setdefault(r, None)
    edd_history = list(edd_seen.keys())
    _write_triples(solve_data_dir / "edd_history.csv",
                   ("entity", "period_history", "period"), edd_history)
    invest_set = frozenset(entityInvest)
    edd_history_invest = [r for r in edd_history if r[0] in invest_set]
    _write_triples(solve_data_dir / "edd_history_invest.csv",
                   ("entity", "period_history", "period"), edd_history_invest)
    edd_invest = [(e, d_inv, d) for (e, d_inv, d) in edd_history_invest
                  if (e, d_inv) in ed_invest_set]
    _write_triples(solve_data_dir / "edd_invest.csv",
                   ("entity", "period_history", "period"), edd_invest)

    # ---- gd_invest / gd_divest / period variants ------------------------
    # Mod L1597: {g in group_invest, d in period_invest :
    #              sum{(g, e) in group_entity : (e, d) in ed_invest} 1}
    gd_invest_pairs: list[tuple[str, str]] = []
    for g in group_invest:
        for d in period_invest:
            ents = entities_for_g.get(g, ())
            if any((e, d) in ed_invest_set for e in ents):
                gd_invest_pairs.append((g, d))
    _write_pairs(solve_data_dir / "gd_invest.csv",
                 ("group", "period"), gd_invest_pairs)

    # gd_divest: mod L1601 — uses group_invest as universe (verify) and
    # the same `(e, d) in ed_invest` filter. Re-reading mod ...
    # Actually mod L1601 reads:
    #   `{g in group_invest, d in period_invest : sum{(g, e) in group_entity : (e, d) in ed_invest} 1}`
    # — same as gd_invest. This is mod L1607: gd_divest is identical
    # except it iterates group_invest (not group_divest) — odd but
    # that's literally what mod says. Mirror it.
    _write_pairs(solve_data_dir / "gd_divest.csv",
                 ("group", "period"), gd_invest_pairs[:])

    rows = [(g, d) for g, d in gd_invest_pairs
            if methods_for_g.get(g, frozenset()) & _INVEST_PERIOD_METHODS]
    _write_pairs(solve_data_dir / "gd_invest_period.csv",
                 ("group", "period"), rows)
    rows = [(g, d) for g, d in gd_invest_pairs
            if methods_for_g.get(g, frozenset()) & _DIVEST_PERIOD_METHODS]
    _write_pairs(solve_data_dir / "gd_divest_period.csv",
                 ("group", "period"), rows)


def write_ed_invest_forbidden_no_investment(
    input_dir: Path, solve_data_dir: Path
) -> None:
    """flextool.mod L1627-1629 — ed_invest filtered to entities where the
    no_investment method's lifetime window has already ended at d.

        { (e, d) in ed_invest :
          (e, 'no_investment') in entity__lifetime_method
          && p_years_d[d] >= sum_{d_first in period_first}
                              (p_years_d[d_first] + edEntity_lifetime[e, d_first])
        }

    Used in fix_v_invest_no_investment_eq (mod L3930) to pin v_invest = 0
    after the lifetime expires.
    """
    ed_invest_pairs = _read_pairs(solve_data_dir / "ed_invest.csv")
    elm = _read_pairs(solve_data_dir / "entity__lifetime_method.csv")
    no_invest_set = frozenset(e for e, m in elm if m == "no_investment")
    p_years_d: dict[str, float] = {}
    for r in _read_pairs(solve_data_dir / "p_years_d.csv"):
        try:
            p_years_d[r[0]] = float(r[1])
        except ValueError:
            continue
    ed_lifetime: dict[tuple[str, str], float] = {}
    elf = solve_data_dir / "edEntity_lifetime.csv"
    if elf.exists():
        with elf.open() as fh:
            import csv as _csv
            reader = _csv.reader(fh)
            next(reader, None)
            for r in reader:
                if len(r) >= 3 and r[0] and r[1]:
                    try:
                        ed_lifetime[(r[0], r[1])] = float(r[2])
                    except ValueError:
                        continue
    period_first = _read_singles(solve_data_dir / "period_first.csv")

    def _life_sum(e: str) -> float:
        return sum(
            p_years_d.get(d_first, 0.0) + ed_lifetime.get((e, d_first), 0.0)
            for d_first in period_first
        )

    rows: list[tuple[str, str]] = []
    cached_sum: dict[str, float] = {}
    for e, d in ed_invest_pairs:
        if e not in no_invest_set:
            continue
        s = cached_sum.get(e)
        if s is None:
            s = _life_sum(e)
            cached_sum[e] = s
        if p_years_d.get(d, 0.0) >= s:
            rows.append((e, d))
    _write_pairs(solve_data_dir / "ed_invest_forbidden_no_investment.csv",
                 ("entity", "period"), rows)
