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
    """Emit (entity, period, value) CSV with ``repr(float)`` precision.

    Retained for compatibility with the legacy direct-text path; new code
    in this module routes through :func:`_write` (which feeds the
    accumulator hook).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ",".join(header) + "\n"
        + "".join(f"{a},{b},{repr(float(v))}\n" for a, b, v in rows)
    )


def _write(df: pl.DataFrame, path: Path) -> None:
    """Polars-frame emission funnel — patched by Phase E-b accumulator.

    Identical I/O contract to :mod:`._writer_dispatchers._write`: parents
    created, ``df.write_csv`` does the byte emission.  Every CSV emitted
    by this module flows through this single name so the accumulator
    monkey-patch can intercept ``(path.name -> df)``.

    Phase E-c — disk emission is gated behind ``emit_csvs_enabled()``.
    When disabled, the helper returns without touching disk; the
    accumulator hook still captures the frame because capture happens
    in the wrapping monkey-patch BEFORE this real ``_write`` is invoked.
    """
    from flextool.engine_polars._flex_data_accumulator import emit_csvs_enabled
    if not emit_csvs_enabled():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


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


# ---------------------------------------------------------------------------
# Phase E-b — derive_X family for each emitted CSV.
#
# The 6 CSVs share substantial input scans (PdLookup over pd_*, p_*, plus
# entity-class sets and period lists), so the wrapper builds the bundle
# once via :func:`_entity_annual_inputs` and threads it through private
# ``_compute_*`` helpers.  Public ``derive_*`` functions are thin
# rebuilders for standalone seed lookups.
# ---------------------------------------------------------------------------


def _entity_annual_inputs(input_dir: Path, solve_data_dir: Path) -> dict:
    """Shared input bundle for the 6 entity-annual derives."""
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
    methods_for_entity: dict[str, list[str]] = {}
    for e, m in _read_pairs(input_dir / "entity__invest_method.csv"):
        methods_for_entity.setdefault(e, []).append(m)
    lifetime_methods_for_entity: dict[str, list[str]] = {}
    for e, m in _read_pairs(solve_data_dir / "entity__lifetime_method.csv"):
        lifetime_methods_for_entity.setdefault(e, []).append(m)
    return {
        "pp": pp,
        "pn": pn,
        "process_set": frozenset(_read_singles(input_dir / "process.csv")),
        "node_set": frozenset(_read_singles(input_dir / "node.csv")),
        "entityInvest": _read_singles(solve_data_dir / "entityInvest.csv"),
        "entityDivest": _read_singles(solve_data_dir / "entityDivest.csv"),
        "period_invest": _read_singles(
            solve_data_dir / "invest_periods_of_current_solve.csv"),
        "period_in_use": _read_singles(
            solve_data_dir / "period_in_use_set.csv"),
        "methods_for_entity": methods_for_entity,
        "lifetime_methods_for_entity": lifetime_methods_for_entity,
        "p_discount_years": _read_keyed_value(
            solve_data_dir / "p_discount_years.csv"),
        "p_inflation_invest": _read_keyed_value(
            solve_data_dir / "p_inflation_factor_investment_yearly.csv"),
        "p_inflation_ops": _read_keyed_value(
            solve_data_dir / "p_inflation_factor_operations_yearly.csv"),
        "edEntity_lifetime": _read_pdv(
            solve_data_dir / "edEntity_lifetime.csv"),
        "ed_fixed_cost": _read_pdv(solve_data_dir / "ed_fixed_cost.csv"),
        "period_with_history": _read_singles(
            solve_data_dir / "period_with_history.csv"),
        "entities": _read_singles(input_dir / "entity.csv"),
    }


def _per_method_annuity_invest(inp: dict, e: str, d: str) -> float:
    v = 0.0
    for m in inp["methods_for_entity"].get(e, ()):
        if m in _INVEST_NOT_ALLOWED:
            continue
        if e in inp["node_set"]:
            v += _annuity(
                inp["pn"].get(e, "invest_cost", d),
                inp["pn"].get(e, "discount_rate", d),
                inp["pn"].get(e, "lifetime", d),
            )
        elif e in inp["process_set"]:
            v += _annuity(
                inp["pp"].get(e, "invest_cost", d),
                inp["pp"].get(e, "discount_rate", d),
                inp["pp"].get(e, "lifetime", d),
            )
    return v


def _per_method_annuity_divest(inp: dict, e: str, d: str) -> float:
    v = 0.0
    for m in inp["methods_for_entity"].get(e, ()):
        if m in _DIVEST_NOT_ALLOWED:
            continue
        if e in inp["node_set"]:
            v += _annuity(
                inp["pn"].get(e, "salvage_value", d),
                inp["pn"].get(e, "discount_rate", d),
                inp["pn"].get(e, "lifetime", d),
            )
        elif e in inp["process_set"]:
            v += _annuity(
                inp["pp"].get(e, "salvage_value", d),
                inp["pp"].get(e, "discount_rate", d),
                inp["pp"].get(e, "lifetime", d),
            )
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
                    pdy_d, life, inp["period_in_use"],
                    inp["p_discount_years"], inp["p_inflation_invest"],
                    open_window=False,
                )
            if is_automatic:
                disc += ann * _inflation_window_sum(
                    pdy_d, life, inp["period_in_use"],
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
                    pdy_d, life, inp["period_in_use"],
                    inp["p_discount_years"], inp["p_inflation_invest"],
                    open_window=False,
                )
            elif e in inp["process_set"]:
                life = inp["pp"].get(e, "lifetime", d)
                disc = ann * _inflation_window_sum(
                    pdy_d, life, inp["period_in_use"],
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
                    pdy_d, life, inp["period_in_use"],
                    inp["p_discount_years"], inp["p_inflation_ops"],
                    open_window=False,
                )
            if is_automatic:
                v += fc * _inflation_window_sum(
                    pdy_d, life, inp["period_in_use"],
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
                pdy_d, life, inp["period_in_use"],
                inp["p_discount_years"], inp["p_inflation_invest"],
                open_window=False,
            )
            rows.append((e, d, v))
    return rows


# ---- Public derive_* (each rebuilds its own input bundle) ----

def derive_ed_entity_annual(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_entity_annual.csv`` — per-(entity, period) annuity sum across
    allowed invest methods, gated by entity class (node vs process)."""
    rows, _ = _ann_pair(_entity_annual_inputs(input_dir, solve_data_dir))
    return _rows_to_frame(rows)


def derive_ed_entity_annual_discounted(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_entity_annual_discounted.csv`` — annuity scaled by the
    inflation-window sum keyed by reinvest_choice / no_investment / reinvest_automatic."""
    _, rows_disc = _ann_pair(_entity_annual_inputs(input_dir, solve_data_dir))
    return _rows_to_frame(rows_disc)


def derive_ed_entity_annual_divest(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_entity_annual_divest.csv`` — divest-side annuity sum."""
    rows, _ = _div_pair(_entity_annual_inputs(input_dir, solve_data_dir))
    return _rows_to_frame(rows)


def derive_ed_entity_annual_divest_discounted(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_entity_annual_divest_discounted.csv`` — divest annuity scaled
    by the inflation-window sum (always closed window)."""
    _, rows_disc = _div_pair(_entity_annual_inputs(input_dir, solve_data_dir))
    return _rows_to_frame(rows_disc)


def derive_ed_lifetime_fixed_cost(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_lifetime_fixed_cost.csv`` — per-(entity, period) lifetime
    fixed-cost using p_inflation_factor_OPERATIONS_yearly."""
    return _rows_to_frame(_rows_lifetime_fixed_cost(
        _entity_annual_inputs(input_dir, solve_data_dir)))


def derive_ed_lifetime_fixed_cost_divest(
    input_dir: Path, solve_data_dir: Path,
) -> pl.DataFrame:
    """``ed_lifetime_fixed_cost_divest.csv`` — divest variant; uses
    p_inflation_factor_INVESTMENT_yearly per mod L1651 (asymmetric)."""
    return _rows_to_frame(_rows_lifetime_fixed_cost_divest(
        _entity_annual_inputs(input_dir, solve_data_dir)))


def write_entity_annual_calc_params(
    input_dir: Path, solve_data_dir: Path,
) -> None:
    """Native port of ``entity_annual_calc_params.write_entity_annual_calc_params``.

    Reads ``input/`` immutables (``pd_process``, ``p_process``,
    ``pd_node``, ``p_node``, ``entity__invest_method``, ``entity``,
    ``process``, ``node``) and per-solve ``solve_data/`` files
    (period sets, discount factors, lifetime / fixed-cost overrides).
    Emits the six CSVs documented at module level.

    Each output flows through ``_write(_rows_to_frame(...), path)`` so the
    Phase E-b accumulator captures every emitted frame.  Shared scans
    (``_ann_pair`` / ``_div_pair``) avoid recomputing per-method annuities
    across the matched annual / annual_discounted CSV pairs.
    """
    inp = _entity_annual_inputs(input_dir, solve_data_dir)

    rows_ann, rows_ann_disc = _ann_pair(inp)
    _write(_rows_to_frame(rows_ann),
           solve_data_dir / "ed_entity_annual.csv")
    _write(_rows_to_frame(rows_ann_disc),
           solve_data_dir / "ed_entity_annual_discounted.csv")

    rows_div, rows_div_disc = _div_pair(inp)
    _write(_rows_to_frame(rows_div),
           solve_data_dir / "ed_entity_annual_divest.csv")
    _write(_rows_to_frame(rows_div_disc),
           solve_data_dir / "ed_entity_annual_divest_discounted.csv")

    _write(_rows_to_frame(_rows_lifetime_fixed_cost(inp)),
           solve_data_dir / "ed_lifetime_fixed_cost.csv")
    _write(_rows_to_frame(_rows_lifetime_fixed_cost_divest(inp)),
           solve_data_dir / "ed_lifetime_fixed_cost_divest.csv")
