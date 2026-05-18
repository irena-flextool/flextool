"""Workdir-CSV seed readers for the invest/divest cascade.

These helpers exist for one reason: the **synthetic per-sub-solve**
case.  When ``_apply_db_overrides`` detects an active solve whose name
does not appear in Spine (per-period sub-solves like
``invest_5weeks_p2020`` synthesised at runtime by the orchestrator),
the per-solve override chain ``apply_derived_a..g`` is skipped — its
``_solve_periods(source, active_solve, ...)`` lookups would return
empty and wipe out the legitimate invest activity captured in the
workdir snapshot.

Post-Step-2.5 these helpers consume the canonical
``solve_data/*.csv`` frames exclusively through the
:class:`FlexDataProvider`.  The disk-fallback arms that previously
re-read ``<workdir>/solve_data/<name>.csv`` from disk are gone — the
writer cascade (``_writer_per_solve.write_invest_csvs`` and friends)
seeds every required key in the Provider before this loader runs.

When the active solve **is** in Spine, the override chain
(``apply_derived_c``) overlays its own values on top of these seeds,
so the helpers are functionally seeds-only on the non-synthetic path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ._axis_enums import cast_dim, schema_dtype
from ._writer_provider_io import _provider_key

# These helpers run at the workdir-CSV seed phase — before FlexData is
# materialised.  Phase 4 binds ``_enums`` to the live cascade-wide
# vocabulary proxy so ``schema_dtype(_enums, axis)`` and ``cast_dim(expr,
# _enums, axis)`` activate Enum allocation under ``load_flextool``.
# Falsy / pl.Utf8 fallback outside an active cascade.
from flextool.engine_polars._axis_enums import _LIVE_AXIS_ENUMS as _enums  # noqa: E402


def _provider_get(provider, path: "Path") -> "pl.DataFrame | None":
    """Provider-only fetch.  Returns ``None`` when the Provider is
    missing or doesn't carry *path*'s canonical key.
    """
    if provider is None:
        return None
    key = _provider_key(path)
    if not provider.has(key):
        return None
    return provider.get(key)


# ---------------------------------------------------------------------------
# (e, d) / (p, d) / (n, d) set frames
# ---------------------------------------------------------------------------


def read_invest_set(workdir_solve_data: Path, name: str,
                       kind_col: str, *, provider=None) -> pl.DataFrame:
    """Read ``ed_invest.csv`` / ``ed_divest.csv`` and rename the
    entity-axis column to *kind_col* (``e``).

    ``ed_invest.csv`` etc. are the canonical Python-preprocessing
    outputs that ``flextool.mod`` reads via ``table data IN``
    (flextool.mod:1428).  The ``solve__``-prefixed twins are .mod
    printf debug-exports of the *current solve's* subset and must NOT
    be used as inputs — using them silently drops invest variables for
    non-realized periods.
    """
    empty = pl.DataFrame(schema={kind_col: schema_dtype(_enums, kind_col),
                                  "d": schema_dtype(_enums, "d")})
    path = workdir_solve_data / f"{name}.csv"
    df = _provider_get(provider, path)
    if df is None or df.height == 0:
        return empty
    rename_src = ("entity" if "entity" in df.columns
                  else "node" if "node" in df.columns
                  else "process")
    return df.rename({rename_src: kind_col, "period": "d"}).select(
        cast_dim(pl.col(kind_col), _enums, kind_col),
        cast_dim(pl.col("d"), _enums, "d"),
    )


def read_forbidden_no_investment(workdir_solve_data: Path,
                                  *, provider=None) -> pl.DataFrame:
    """Read ``ed_invest_forbidden_no_investment.csv``.

    Entities that may NOT invest in specified periods
    (lifetime_method=no_investment combined with
    invest_method=invest_no_limit at periods where the lifetime window
    disallows new build).  flextool encodes this as
    ``fix_v_invest_no_investment_eq`` pinning the variable to 0; we
    achieve the same effect by removing the (entity, period) tuple
    from every invest set so the variable is never created.

    Returns an empty (e, d) frame when the Provider doesn't carry the
    key or it's empty.
    """
    empty = pl.DataFrame(schema={"e": schema_dtype(_enums, "e"),
                                  "d": schema_dtype(_enums, "d")})
    path = workdir_solve_data / "ed_invest_forbidden_no_investment.csv"
    df = _provider_get(provider, path)
    if df is None or df.height == 0:
        return empty
    return df.rename({"entity": "e", "period": "d"}).select(
        cast_dim(pl.col("e"), _enums, "e"),
        cast_dim(pl.col("d"), _enums, "d"),
    )


def read_set_seed(workdir_solve_data: Path, name: str,
                     kind_col: str, *, provider=None) -> pl.DataFrame:
    """Read ``pd_invest.csv`` / ``pd_divest.csv`` / ``nd_invest.csv``
    / ``nd_divest.csv``.  Each is a per-(entity, period) seed frame.
    """
    empty = pl.DataFrame(schema={kind_col: schema_dtype(_enums, kind_col),
                                  "d": schema_dtype(_enums, "d")})
    path = workdir_solve_data / f"{name}.csv"
    df = _provider_get(provider, path)
    if df is None or df.height == 0:
        return empty
    rename_src = ("entity" if "entity" in df.columns
                  else "node" if "node" in df.columns
                  else "process" if "process" in df.columns
                  else None)
    if rename_src is None or "period" not in df.columns:
        return empty
    return df.rename({rename_src: kind_col, "period": "d"}).select(
        cast_dim(pl.col(kind_col), _enums, kind_col),
        cast_dim(pl.col("d"), _enums, "d"),
    )


def read_edd_invest(workdir_solve_data: Path,
                     *, provider=None) -> pl.DataFrame:
    """Read ``edd_invest.csv`` — (entity, d_invest, period) triple set.

    Canonical CSV uses ``period_history`` for d_invest; tolerate both
    column names.
    """
    empty = pl.DataFrame(schema={
        "e": schema_dtype(_enums, "e"),
        "d_invest": schema_dtype(_enums, "d_invest"),
        "d": schema_dtype(_enums, "d")})
    path = workdir_solve_data / "edd_invest.csv"
    df = _provider_get(provider, path)
    if df is None or df.height == 0:
        return empty
    ren = {}
    if "entity" in df.columns:
        ren["entity"] = "e"
    if "period_history" in df.columns:
        ren["period_history"] = "d_invest"
    if "period" in df.columns:
        ren["period"] = "d"
    df = df.rename(ren)
    if not {"e", "d_invest", "d"}.issubset(df.columns):
        return empty
    return df.select(
        cast_dim(pl.col("e"), _enums, "e"),
        cast_dim(pl.col("d_invest"), _enums, "d_invest"),
        cast_dim(pl.col("d"), _enums, "d"),
    )


def read_period_set(workdir_solve_data: Path, name: str,
                       *, provider=None) -> pl.DataFrame | None:
    """Read ``ed_invest_period.csv`` / ``ed_divest_period.csv`` — the
    (entity, period) tuples with per-period invest / divest caps.

    Returns None (not empty) when the Provider doesn't carry the key
    or it's empty so the seed assignment in ``_load_invest`` mirrors
    the original ``None``-or-non-empty contract that downstream
    consumers (``model.py:1517``) gate on.
    """
    path = workdir_solve_data / f"{name}.csv"
    df = _provider_get(provider, path)
    if df is None or df.height == 0:
        return None
    return df.rename({"entity": "e", "period": "d"}).select(
        cast_dim(pl.col("e"), _enums, "e"),
        cast_dim(pl.col("d"), _enums, "d"),
    )
