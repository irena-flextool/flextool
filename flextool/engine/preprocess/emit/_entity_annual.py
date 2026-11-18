"""Annuity / lifetime-fixed-cost per-solve emitters.

Called per-solve via ``write_entity_annual_calc_params``.

The CSVs emitted are precision-sensitive: the values are float
annuities and discounted sums.  We pre-stringify with ``repr(float(v))``
to keep round-trip safety byte-for-byte.

Outputs (all written under ``solve_data/``):

  * ``ed_entity_annual.csv``                     (entity, period, value)
  * ``ed_lifetime_fixed_cost_divest.csv``        (entity, period, value)

The salvage terms of a retirement are not emitted: ``_npv.apply_retirement_salvage`` derives them for a Spine
solve and a synthetic sub-solve alike.

``ed_entity_annual_discounted.csv`` and ``ed_lifetime_fixed_cost.csv`` are not emitted here: their windows start
at the start of operation of a decision (construction lead time) and run over the years in service at the level
price (:func:`..params._npv.npv_invest_discounted_lf`), which only :mod:`..params._npv` derives; ``_npv.apply_invest_lumps`` registers its values under
these keys (for a Spine solve and a synthetic sub-solve alike).

Method-enum constants mirror flextool/flextool_base.dat:184 and
:211-212.  ``PdLookup`` is the native 4-branch resolver from
:mod:`._pdt_lookup` (same machinery used by ``_dispatchers`` and
``_period_params``).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from flextool.engine.input._lifetime_roles import (
    LIFETIME_TECHNICAL,
    PAYBACK_TIME,
    lookup_roles,
    role_param,
)
from flextool.engine.preprocess.emit._pdt_lookup import PdLookup
from flextool.engine.preprocess._annuity import (
    DEFAULT_DISCOUNT_RATE,
    annuity_factor,
    require_lifetime,
)
from flextool.engine.preprocess.emit._period_calc import scalar_max
from flextool.engine.preprocess.params._branch import annuity_window_pairs
from flextool.engine.preprocess.params._npv import _INFL_OFFSET_INV_DEFAULT
from flextool.engine.preprocess._provider_io import _emit
from flextool.engine.preprocess.params._existing import (
    INVEST_NOT_ALLOWED as _INVEST_NOT_ALLOWED,
)


# ---------------------------------------------------------------------------
# Method-enum constants — mirror flextool/flextool_base.dat:211-212.
# ---------------------------------------------------------------------------

# The invest method list (_INVEST_NOT_ALLOWED) is imported from params/_existing.py, its one definition.


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


def _annuities(inp: dict, cost_param: str, rows: list[tuple[str, str, str | None, int]]) -> list[float]:
    """``count × cost × 1000 × annuity_factor(r, n, off)`` of each row ``(entity e, period d, kind, count)``
    (:mod:`.._annuity`), the values read from the lookup ``inp[kind]`` (``"pp"`` processes, ``"pn"`` nodes; ``None``
    or ``count`` 0: 0), ``off`` the model's ``inflation_offset_investment`` (``inp["offset_inv"]``).

    ``cost_param``: ``invest_cost`` or ``net_salvage_value``; ``n`` is the entity's payback time
    (:data:`PAYBACK_TIME`, its technical lifetime when it has none).  A missing ``discount_rate`` is
    ``DEFAULT_DISCOUNT_RATE`` (the schema default); an explicit 0 stays 0 (the straight-line annuity
    ``cost / n``).  A zero cost gives 0 whatever the payback time; a non-zero cost without a positive payback time
    is an error (:func:`require_lifetime`).
    """
    costs: list[float] = []
    rates: list[float] = []
    lives: list[float] = []
    missing: list[tuple[str, str]] = []
    for e, d, kind, count in rows:
        if kind is None or count == 0:
            costs.append(0.0)
            rates.append(0.0)
            lives.append(1.0)
            continue
        lookup, roles = inp[kind], inp["roles"][kind]
        cost = lookup.get(e, cost_param, d)
        r = (lookup.get(e, "discount_rate", d) if lookup.has(e, "discount_rate", d)
             else DEFAULT_DISCOUNT_RATE)
        life_name = role_param(roles, e, PAYBACK_TIME)
        n = lookup.get(e, life_name, d) if lookup.has(e, life_name, d) else 0.0
        if cost != 0 and not n > 0:
            missing.append((e, d))
        costs.append(float(cost) * count)
        rates.append(float(r))
        lives.append(float(n) if n > 0 else 1.0)
    require_lifetime(missing, cost_param, PAYBACK_TIME)
    if not rows:
        return []
    frame = pl.DataFrame({"cost": costs, "r": rates, "n": lives},
                         schema={"cost": pl.Float64, "r": pl.Float64, "n": pl.Float64})
    return frame.select(
        pl.when(pl.col("cost") == 0).then(0.0)
          .otherwise(pl.col("cost") * 1000.0 * annuity_factor(pl.col("r"), pl.col("n"), inp["offset_inv"]))
    ).to_series().to_list()


def _inflation_window_sum(
    pdy_d: float,
    life: float,
    period_in_use: list[str],
    p_discount_years: dict[str, float],
    inflation: dict[str, float],
) -> float:
    """Σ inflation[d_all] over the periods of period_in_use with ``pdy_d ≤ pdy < pdy_d + life`` (the retirement
    windows)."""
    s = 0.0
    upper = pdy_d + life
    for d_all in period_in_use:
        pdy = p_discount_years.get(d_all, 0.0)
        if pdy < pdy_d or pdy >= upper:
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
# The CSVs share substantial input scans (PdLookup over pd_*, p_*, plus
# entity-class sets and period lists), so the wrapper builds the bundle
# once via :func:`_entity_annual_inputs` and threads it through private
# ``_compute_*`` helpers.  Public ``derive_*`` functions are thin
# rebuilders for standalone seed lookups.
# ---------------------------------------------------------------------------


def _entity_annual_inputs(input_dir: Path, solve_data_dir: Path,
                            *, provider: "object | None" = None) -> dict:
    """Shared input bundle for the entity-annual derives."""
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
        "roles": {"pp": lookup_roles(pp), "pn": lookup_roles(pn)},
        "process_set": frozenset(_read_singles(input_dir / "process.csv", provider=provider)),
        "node_set": frozenset(_read_singles(input_dir / "node.csv", provider=provider)),
        "entityInvest": _read_singles(solve_data_dir / "entityInvest.csv", provider=provider),
        "entityDivest": _read_singles(solve_data_dir / "entityDivest.csv", provider=provider),
        "period_invest": _read_singles(
            solve_data_dir / "invest_periods_of_current_solve.csv", provider=provider),
        "period_in_use": _read_singles(
            solve_data_dir / "period_in_use_set.csv", provider=provider),
        "methods_for_entity": methods_for_entity,
        "p_discount_years": _read_keyed_value(
            solve_data_dir / "p_discount_years.csv", provider=provider),
        "p_inflation_invest": _read_keyed_value(
            solve_data_dir / "p_inflation_factor_investment_yearly.csv", provider=provider),
        "ed_fixed_cost": _read_pdv(solve_data_dir / "ed_fixed_cost.csv", provider=provider),
        "offset_inv": scalar_max(input_dir / "p_inflation_offset_investment.csv", _INFL_OFFSET_INV_DEFAULT,
                                 provider=provider),
    }


def _annuity_rows(inp: dict, pairs: list[tuple[str, str]], disallowed: frozenset[str]
                  ) -> list[tuple[str, str, str | None, int]]:
    """``(e, d, kind, count)`` of :func:`_annuities`: ``count`` the entity's invest methods outside ``disallowed``
    (each contributes the same annuity), ``kind`` its lookup (nodes ``"pn"``, processes ``"pp"``, else None)."""
    out = []
    for e, d in pairs:
        count = sum(1 for m in inp["methods_for_entity"].get(e, ()) if m not in disallowed)
        kind = "pn" if e in inp["node_set"] else "pp" if e in inp["process_set"] else None
        out.append((e, d, kind, count))
    return out


def _rows_annuity(inp: dict) -> list[tuple[str, str, float]]:
    """``ed_entity_annual``: the undiscounted annuity of each investment period (of that period's values)."""
    pairs = [(e, d) for e in inp["entityInvest"] for d in inp["period_invest"]]
    values = _annuities(inp, "invest_cost", _annuity_rows(inp, pairs, _INVEST_NOT_ALLOWED))
    return [(e, d, v) for (e, d), v in zip(pairs, values)]


def _rows_lifetime_fixed_cost_divest(inp: dict) -> list[tuple[str, str, float]]:
    # NB: mod L1651 deliberately uses p_inflation_factor_INVESTMENT_yearly
    # here (not operations-yearly) — asymmetric vs the non-divest variant.
    rows: list[tuple[str, str, float]] = []
    for e in inp["entityDivest"]:
        for d in inp["period_invest"]:
            fc = inp["ed_fixed_cost"].get((e, d), 0.0)
            pdy_d = inp["p_discount_years"].get(d, 0.0)
            if e in inp["node_set"]:
                life = inp["pn"].get(e, LIFETIME_TECHNICAL, d)
            elif e in inp["process_set"]:
                life = inp["pp"].get(e, LIFETIME_TECHNICAL, d)
            else:
                life = 0.0
            v = fc * _inflation_window_sum(
                pdy_d, life, _window_periods(inp, d),
                inp["p_discount_years"], inp["p_inflation_invest"],
            )
            rows.append((e, d, v))
    return rows


def emit_entity_annual_calc_params(
    input_dir: Path, solve_data_dir: Path,
    *, provider,
) -> None:
    """Emit ``entity_annual_calc_params`` to the Provider.
    Emits the two frames under ``solve_data/<basename>`` keys via
    :func:`_emit` (dual-key registration: basename and parent/basename).
    """
    inp = _entity_annual_inputs(input_dir, solve_data_dir, provider=provider)

    _emit(provider, "solve_data/ed_entity_annual.csv",
          _rows_to_frame(_rows_annuity(inp)))

    _emit(provider, "solve_data/ed_lifetime_fixed_cost_divest.csv",
          _rows_to_frame(_rows_lifetime_fixed_cost_divest(inp)))
