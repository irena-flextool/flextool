"""Raw LP dump behind ``model.output_horizon`` (developer debug flag).

With the flag on, every sub-solve writes ALL its LP variables and constraints, over ALL rows (realized and
unrealized alike), as raw solver values: one parquet part per family per roll under
``<work>/output_raw/horizon/<lp_name>/<seq>__<roll>.parquet``.  :func:`assemble_horizon_dump` streams the parts of
each family into ``<output_parquet>/<scenario>/horizon/<lp_name>.parquet`` once the run's outputs are written.

Columns (one fixed schema per family; every part is built with the same dtypes):

* variables: ``<dims…>`` (Utf8, the LP dim names), ``solve``, ``roll``, ``realized`` (Boolean), ``value`` (primal,
  per unitsize as in the LP, Layer-2-unscaled) and ``reduced_cost`` (column dual × 1/scale_the_objective);
* constraints: ``<dims…>``, ``solve``, ``roll``, ``realized`` and ``dual`` (row dual × 1/scale_the_objective).

The dump is LP-shaped: a coarse process (one column per coarse step) has the rows of its variables of
``_coarse_flows.COARSE_VARS`` only at each coarse step's first fine step (``b_first``), unlike the normal output,
which writes them on every fine step (:mod:`flextool.engine.output._coarse_flows`).

Duals keep the HiGHS sign convention; no weight, price, unitsize or cost conversion.  A MIP solve, or a solve whose
duals are not valid, writes null duals and reduced costs (never 0).

``realized`` marks the rows that also reach normal output: rows with ``d`` and ``t`` when ``(d, t)`` is in
``solve_data/realized_dispatch``; investment / divestment families (``v_invest_*`` / ``v_divest_*``) when ``d`` is
in ``solve_data/realized_invest_periods_of_current_solve``; other ``d`` rows when ``d`` is in either; null for a
family without a ``d`` dim.  A fan solve's cells are ``False`` while the same cells are ``True`` in its realization
solve's dump, and period rows repeated across rolls are ``True`` in every roll (normal output keeps the last).

Constraint row keys come from a build-time snapshot of ``Problem._cstrs`` (:func:`snapshot_cstr_keys`), taken
before the solve releases it: HiGHS row names are not a reliable key source (commas in entity names, synthesized
``r*`` names on the subprocess path).  Rows are numbered in declaration order, ``over.height`` rows per family
(one for a family without ``over``), as polar-high builds them.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from flextool.safe_names import require_strictly_inside

if TYPE_CHECKING:
    from polar_high import Problem, Solution

_logger = logging.getLogger(__name__)

#: Sub-folder of ``output_raw/`` (parts) and of the scenario's ``output_parquet`` folder (assembled files).
HORIZON_DIR = "horizon"

_INVEST_PREFIXES = ("v_invest_", "v_divest_")
_KEY_SEP = "\x1f"

CstrKeys = list[tuple[str, "pl.DataFrame | None"]]


def output_horizon_enabled(work_folder: "Path | str | None", *, provider: "object | None" = None) -> bool:
    """True when ``model.output_horizon`` is enabled for this solve.

    The provider key is ``solve_data/enable_optional_outputs``.  ``work_folder`` only selects the key (no disk
    read); a ``None`` folder or provider, or a missing key, gives False.
    """
    if work_folder is None or provider is None:
        return False
    name = "solve_data/enable_optional_outputs"
    if not provider.has(name):
        return False
    try:
        frame = provider.get(name)
        # One column of flag names (its header is not part of the contract).
        return "output_horizon" in {str(v) for v in frame.to_series(0).to_list() if v is not None}
    except Exception:  # noqa: BLE001
        return False


def snapshot_cstr_keys(pb: "Problem") -> CstrKeys:
    """``[(name, over)]`` of every constraint family of a built ``Problem``, in declaration order.

    Holds references only (no copies).  Taken before the solve, which clears ``Problem._cstrs``.
    """
    return [(name, over) for name, _proto, over in pb._cstrs]


def _provider_frame(provider: "object | None", name: str) -> "pl.DataFrame | None":
    if provider is None or not provider.has(name):
        return None
    return provider.get(name)


def _realized_keys(provider: "object | None") -> tuple[pl.Series, pl.Series, pl.Series]:
    """``(dt keys, invest periods, d-only periods)`` of this solve as Utf8 Series."""
    rd = _provider_frame(provider, "solve_data/realized_dispatch")
    if rd is not None and rd.height and rd.width >= 2:
        # ``(period, step)``: positional, the step column's header varies (``step`` / ``time``).
        rd = rd.select(rd.to_series(0).cast(pl.Utf8).alias("period"),
                       rd.to_series(1).cast(pl.Utf8).alias("time")).drop_nulls()
    else:
        rd = pl.DataFrame({"period": [], "time": []}, schema={"period": pl.Utf8, "time": pl.Utf8})
    dt_keys = rd.select(pl.concat_str(["period", "time"], separator=_KEY_SEP).alias("key"))["key"]
    dispatch_periods = rd["period"].unique()
    inv = _provider_frame(provider, "solve_data/realized_invest_periods_of_current_solve")
    if inv is not None and inv.width >= 1:
        invest_periods = inv.to_series(0).cast(pl.Utf8).drop_nulls()
    else:
        # No invest-period set for this solve: the investment writer falls back to the dispatch periods.
        invest_periods = dispatch_periods
    d_only = pl.concat([dispatch_periods, invest_periods]).unique()
    return dt_keys, invest_periods, d_only


def _realized_expr(name: str, dims: list[str], keys: tuple[pl.Series, pl.Series, pl.Series]) -> pl.Expr:
    dt_keys, invest_periods, d_only = keys
    if "d" not in dims:
        return pl.lit(None, dtype=pl.Boolean)
    if "t" in dims:
        return pl.concat_str(["d", "t"], separator=_KEY_SEP).is_in(dt_keys.implode())
    if name.startswith(_INVEST_PREFIXES):
        return pl.col("d").is_in(invest_periods.implode())
    return pl.col("d").is_in(d_only.implode())


def _duals_valid(sol: "Solution") -> bool:
    if any(getattr(v, "integer", False) for v in sol._vars.values()):
        return False
    h = getattr(sol, "highs", None)
    if h is not None:
        try:
            if getattr(h.getSolution(), "dual_valid", True) is False:
                return False
        except Exception:  # noqa: BLE001
            pass
    return sol.row_dual is not None


def _key_frame(frame: "pl.DataFrame | None", dims: list[str], n_rows: int, solve_name: str) -> pl.DataFrame:
    """The family's key columns as Utf8 plus ``solve`` (a frame of ``n_rows`` rows when it has no dims)."""
    solve = pl.lit(solve_name, dtype=pl.Utf8).alias("solve")
    if frame is None or not dims:
        return pl.select(pl.repeat(solve_name, n_rows, dtype=pl.Utf8).alias("solve"))
    return frame.select([pl.col(c).cast(pl.Utf8) for c in dims]).with_columns(solve)


def _part_path(work_folder: Path, lp_name: str, seq: int, roll_name: str) -> Path:
    folder = work_folder / "output_raw" / HORIZON_DIR / lp_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{seq:05d}__{roll_name}.parquet"


def _manifest_seq(work_folder: Path) -> int:
    """The line count of ``output_raw/_solve_order.txt`` (this sub-solve's 1-based position in the run)."""
    path = work_folder / "output_raw" / "_solve_order.txt"
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def write_horizon_parts(
    sol: "Solution",
    cstr_keys: "CstrKeys | None",
    *,
    work_folder: "Path | str",
    solve_name: str,
    roll_name: str,
    scale_the_objective: "float | None",
    provider: "object | None",
) -> int:
    """Write this sub-solve's raw parts (one per non-empty family); returns the number of parts written.

    Call after ``append_solve_order`` so the part sequence number is this sub-solve's manifest position.  Each
    family is built, written and dropped in turn (RAM bounded by one family).  ``cstr_keys`` ``None`` skips the
    constraints (logged at ERROR: the caller did not take the build-time snapshot).
    """
    from flextool.process_outputs.read_highs_solution import _resolve_inv_scale_the_objective

    work_folder = Path(work_folder)
    inv_scale = _resolve_inv_scale_the_objective(scale_the_objective)
    seq = _manifest_seq(work_folder)
    keys = _realized_keys(provider)
    duals_ok = _duals_valid(sol)
    col_value = np.asarray(sol.col_value, dtype=np.float64)
    col_dual = np.asarray(sol.col_dual, dtype=np.float64) if duals_ok else None
    row_dual = np.asarray(sol.row_dual, dtype=np.float64) if duals_ok else None

    var_names = set(sol._vars)
    if cstr_keys is not None:
        clash = sorted(var_names & {name for name, _ in cstr_keys})
        if clash:
            raise ValueError(f"raw horizon dump: LP names used by both a variable and a constraint: {clash}")

    def _emit(name: str, frame: pl.DataFrame) -> int:
        if frame.height == 0:
            return 0
        frame.write_parquet(_part_path(work_folder, name, seq, roll_name))
        return 1

    def _tagged(frame: "pl.DataFrame | None", n_rows: int, name: str, dims: list[str]) -> pl.DataFrame:
        return _key_frame(frame, dims, n_rows, solve_name).with_columns(
            pl.lit(roll_name, dtype=pl.Utf8).alias("roll"),
            _realized_expr(name, dims, keys).alias("realized"),
        )

    written = 0
    for name, var in sol._vars.items():
        dims = list(var.dims)
        ids = var.frame["col_id"].to_numpy()
        part = _tagged(var.frame, len(ids), name, dims).with_columns(
            pl.Series("value", col_value[ids], dtype=pl.Float64),
            pl.Series("reduced_cost", col_dual[ids] * inv_scale, dtype=pl.Float64)
            if col_dual is not None else pl.lit(None, dtype=pl.Float64).alias("reduced_cost"),
        )
        written += _emit(name, part)
        del part

    if cstr_keys is None:
        _logger.error("raw horizon dump for solve '%s': no constraint keys; constraints not dumped", solve_name)
        return written
    total = sum(1 if over is None else int(over.height) for _name, over in cstr_keys)
    n_rows = len(sol.row_dual) if sol.row_dual is not None else total
    if total != n_rows:
        raise ValueError(
            f"raw horizon dump for solve '{solve_name}': {total} snapshot constraint rows but the solution has "
            f"{n_rows} rows"
        )
    base = 0
    for name, over in cstr_keys:
        height = 1 if over is None else int(over.height)
        dims = [] if over is None else list(over.columns)
        part = _tagged(over, height, name, dims).with_columns(
            pl.Series("dual", row_dual[base:base + height] * inv_scale, dtype=pl.Float64)
            if row_dual is not None else pl.lit(None, dtype=pl.Float64).alias("dual"),
        )
        base += height
        written += _emit(name, part)
        del part
    return written


def warn_benders_no_dump(logger: logging.Logger, solve_name: str, *, console: bool = True) -> None:
    """The warning a Benders solve gives under ``output_horizon`` (it writes no outputs at all).

    Logged for every Benders solve; with ``console`` also printed (the runner's logger is pinned at ERROR,
    so a solve-site warning would not reach the console otherwise) — the caller prints it once per run.
    """
    message = (f"output_horizon: raw horizon dump is not available for Benders solves; "
               f"solve '{solve_name}' is not dumped")
    logger.warning(message)
    if console:
        print(f"WARNING: {message}", flush=True)


def assemble_horizon_dump(raw_output_dir: "Path | str", parquet_dir: "Path | str", *, clear_stale: bool) -> int:
    """Stream every family's parts into ``<parquet_dir>/horizon/<lp_name>.parquet``; returns the file count.

    ``clear_stale`` removes an earlier run's ``<parquet_dir>/horizon/`` first (whatever the flag), so a flag-off
    run leaves none behind.  Row order is part order (roll order), then LP order within a part.
    """
    out = Path(parquet_dir) / HORIZON_DIR
    if clear_stale and out.is_dir():
        # Only the run's own horizon/ folder (not a link to another folder).
        require_strictly_inside(out, parquet_dir)
        shutil.rmtree(out)
    src = Path(raw_output_dir) / HORIZON_DIR
    if not src.is_dir():
        return 0
    count = 0
    for family in sorted(p for p in src.iterdir() if p.is_dir()):
        parts = sorted(str(p) for p in family.glob("*.parquet"))
        if not parts:
            continue
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{family.name}.parquet"
        tmp = target.with_suffix(".parquet.tmp")
        pl.scan_parquet(parts).sink_parquet(tmp, maintain_order=True)
        os.replace(tmp, target)
        count += 1
    return count
