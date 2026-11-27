"""§3.7.7/8 — the rolling-handoff state of investments (previously invested, invested and divested
capacity), the period history and :func:`apply_derived_f` (Γ.3.F)."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from polar_high import Param

from flextool.engine.input._axis_enums import rename_to_axis
from flextool.engine.input._param_shapes import solve_years_represented

if TYPE_CHECKING:
    from flextool.engine.input._input_source import InputSource
    from flextool.engine.preprocess._solve_context import SolveContext

from flextool.engine.preprocess.params._derived._scope import _provider_has_key, _provider_read, _try_param


# ---------------------------------------------------------------------------
# §3.7.7/8 — handoff state (rolling solve-chain feeds)
# ---------------------------------------------------------------------------


def _read_handoff_e_d_from_workdir(workdir: Path,
                                       name: str,
                                       *,
                                       provider: "object | None" = None,
                                       ) -> "Param | None":
    """Read a (entity, period, value) handoff CSV; mirror the loader's
    zero-filter (the flex_data loaders).  Returns None when the file is
    absent / empty / all-zero.
    """
    p = Path(workdir) / "solve_data" / f"{name}.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    df = _provider_read(provider, p)
    if df.height == 0:
        return None
    df = (df.pipe(rename_to_axis, {"entity": "e", "period": "d"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .filter(pl.col("value") != 0.0)
              .select("e", "d", "value"))
    if df.height == 0:
        return None
    return Param(("e", "d"), df)


def _read_handoff_e_from_workdir(workdir: Path, name: str,
                                    value_col: str | None = None,
                                    *,
                                    provider: "object | None" = None,
                                    ) -> "Param | None":
    """Wide-format (entity, p_entity_invested) → Param ((e,), value)."""
    p = Path(workdir) / "solve_data" / f"{name}.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return None
    df = _provider_read(provider, p)
    if df.height == 0:
        return None
    if value_col is None:
        non = [c for c in df.columns if c != "entity"]
        if not non:
            return None
        value_col = non[0]
    df = (df.pipe(rename_to_axis, {"entity": "e", value_col: "value"})
              .with_columns(value=pl.col("value")
                                       .cast(pl.Float64, strict=False)
                                       .fill_null(0.0))
              .filter(pl.col("value") != 0.0)
              .select("e", "value"))
    if df.height == 0:
        return None
    return Param(("e",), df)


def p_entity_previously_invested_capacity_from_workdir(
        workdir: Path,
        *,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Per-solve handoff carrier (§3.7.7).  Read-side mirror of the
    chain runner's write-side: ``solve_data/p_entity_previously_invested_capacity.csv``
    holds the cumulative prior-solve invest summed across the entity's
    historical periods.  In single-solve fixtures this CSV is
    header-only (or all-zero), and the Param collapses to ``None``.

    For chain runs, the chain runner is responsible for writing the
    fresh CSV between sub-solves (or for calling :func:`apply_handoff`
    with the in-memory carrier).  This helper only reads from disk —
    it doesn't manufacture state.
    """
    return _read_handoff_e_d_from_workdir(
        workdir, "p_entity_previously_invested_capacity", provider=provider)


def p_entity_invested_from_workdir(
        workdir: Path,
        *,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Per-entity scalar of cumulative prior-solve invest (§3.7.8): every investment earlier solves of the run
    realized (the carried cohorts), as the solve emitted it (``solve_data/p_entity_invested``,
    ``_chain_params.emit_existing_capacity``).  Absent / header-only / all-zero collapses to ``None``."""
    return _read_handoff_e_from_workdir(workdir, "p_entity_invested", provider=provider)


def p_entity_divested_from_handoff(
        *,
        provider: "object | None" = None,
        ) -> "Param | None":
    """Per-entity scalar of cumulative prior-solve divest (§3.7.8).

    Phase 4.2-1a — read the divest-cumulative carrier from the canonical
    ``handoff/divest_cumulative`` Provider key (schema
    ``[entity, value]``).  The translator populates this key from
    :pyattr:`SolveHandoff.divest_cumulative` at iteration start;
    absent / header-only / all-zero collapses to ``None``.  Mirrors
    the overlay logic of ``flex_data/_load.py::_overlay_handoff``.
    """
    from flextool.engine.preprocess import _provider_keys as K
    from flextool.engine.preprocess._provider_translators import read_handoff_frame
    df = read_handoff_frame(provider, K.HANDOFF_DIVEST_CUMULATIVE)
    if df is None or df.height == 0:
        return None
    df = (df
          .pipe(rename_to_axis, {"entity": "e"})
          .with_columns(value=pl.col("value").cast(pl.Float64, strict=False)
                                .fill_null(0.0))
          .filter(pl.col("value") != 0.0)
          .select("e", "value"))
    if df.height == 0:
        return None
    return Param(("e",), df)


# ---------------------------------------------------------------------------
# §3.7.5/6 — Lifetime fixed cost + entity annual discounted (annuity)
# ---------------------------------------------------------------------------


def _entity_invest_methods(source: "InputSource"
                              ) -> dict[str, list[str]]:
    """Return ``entity → [invest_method, ...]`` from
    ``entity__invest_method.csv``.  Spine source: per-entity-class
    ``invest_method`` Map, scalar-valued; flextool keeps the per-entity
    list (flextool.mod allows multi-method but every concrete model
    has 1).
    """
    out: dict[str, list[str]] = {}
    for ec in ("unit", "node", "connection"):
        df = _try_param(source, ec, "invest_method")
        if df is None:
            continue
        for r in df.iter_rows(named=True):
            out.setdefault(str(r["name"]), []).append(str(r["value"]))
    return out


def _p_discount_years_from_source(source: "InputSource",
                                       active_solve: str | None,
                                       ) -> dict[str, float]:
    """Derive ``solve_data/p_discount_years.csv`` (matches the
    ``emit_period_years`` output).

    Algorithm: for each period in solve.years_represented[active_solve]
    (iteration order = Spine Map order = canonical period_set order),
    emit ``(period, year_count)`` then ``year_count += R[d]``.

    Periods absent from years_represented (or with R=0) get value 0.0
    via the consumer's ``.get(d, 0.0)`` default.

    Returns ``{period: cumulative_years}`` dict.
    """
    out: dict[str, float] = {}
    if active_solve is None:
        return out
    yrp = solve_years_represented(source)
    if yrp is None:
        return out
    sub = yrp.filter(pl.col("name") == active_solve)
    if sub.height == 0:
        return out
    # Spine Map iteration order — preserve the row order from the source.
    # The Spine reader returns rows in DB-storage order which mirrors the
    # canonical Map index order.
    year_count = 0.0
    for r in sub.iter_rows(named=True):
        d = str(r["period"])
        out[d] = year_count
        try:
            year_count += float(r["value"])
        except Exception:
            pass
    return out


def _read_period_with_history(workdir: Path | None, *, provider: "object | None" = None) -> list[str]:
    """Read ``solve_data/period_with_history.csv`` from disk if present.

    For multi-solve chain runs flextool's per-solve preprocessing
    (``orchestration.py::main`` block at L262) writes this CSV with the
    accumulated period universe from all prior solves.  When the file
    is absent (e.g. tests using only the input/ side), the caller's
    fallback to ``period_in_use`` is used.

    Returns ``[period, ...]`` in CSV row order.
    """
    if workdir is None:
        return []
    p = Path(workdir) / "solve_data" / "period_with_history.csv"
    if not _provider_has_key(provider, p):  # Phase E-h — seed-aware
        return []
    df = _provider_read(provider, p)
    if df.height == 0 or "period" not in df.columns:
        return []
    return df["period"].cast(pl.Utf8, strict=False).to_list()


# ---------------------------------------------------------------------------
# Integration: apply_derived_f
# ---------------------------------------------------------------------------


def apply_derived_f(
    flex_data: object,
    source: "InputSource",
    workdir: Path,
    *,
    ctx: "SolveContext | None" = None,
    provider: "object | None" = None,
) -> None:
    """Apply Γ.3.F lifetime cascade family + handoff state + multi-year
    inflation cascade, mutating ``flex_data`` in place.

    Order:
      1. Cluster A (Δ.5): ``p_inflation_op``, ``p_ed_fixed_cost``,
         ``ed_entity_annual_discounted``, ``ed_divest_salvage``, ``ed_existing_salvage_credit``,
         ``ed_lifetime_fixed_cost``, ``ed_lifetime_fixed_cost_divest``
         — delegated to :func:`._lifetime_costs.apply_npv`, the lazy-polars
         port.  Replaces the eager ``ed_entity_annual_family_from_source``
         and ``p_inflation_op_full_cascade_from_source`` paths.
      2. ``p_entity_previously_invested_capacity`` / ``p_entity_invested``
         / ``p_entity_divested`` (handoff state).

    Δ.3 replaced the previous ``derived_overrides_f`` dict-return;
    Δ.4 deleted the deprecated wrapper alias; Δ.5 ports the lifetime
    cascade family to lazy polars.
    """
    from flextool.engine.preprocess.params import _lifetime_costs

    # 1. Cluster A (lazy NPV / inflation / fixed cost cascade).
    _lifetime_costs.apply_npv(flex_data, source, workdir, provider=provider)

    # 2. Handoff state -----------------------------------------------
    # Δ.12b — unconditional assignment.  Each helper returns ``None``
    # when the corresponding handoff CSV is missing / header-only
    # (single-solve fixtures, or first-of-chain) — ``None`` is the
    # explicit "no prior-solve state" signal.
    flex_data.p_entity_previously_invested_capacity = (
        p_entity_previously_invested_capacity_from_workdir(
            workdir, provider=provider))
    flex_data.p_entity_invested = p_entity_invested_from_workdir(
        workdir, provider=provider)
    flex_data.p_entity_divested = p_entity_divested_from_handoff(
        provider=provider)
