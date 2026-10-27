"""Write a solve's output parquets offline from an MPS + HiGHS solution file and a CSV dump (unsupported).

``write_outputs_from_files`` rebuilds the solve's :class:`FlexData` from the dump (:func:`load_from_dump_with_provider`;
stamp check and :class:`CsvReloadWarning` as there), reads the MPS and the solution into a fresh ``highspy.Highs`` and
makes the same ``write_all_variables`` call as production (``flextool/engine/output/_writer.py``): the same
provider-gated ``solve_data`` CSV paths, ``flex_data`` and ``provider``, so the writers take their FlexData branches.
Two things production gets from the live solve are restored on the re-read instance: the unified ``v_invest`` /
``v_divest`` column names (``_writer._VAR_RENAME``) and the objective value (``readSolution`` does not set it; it is
recomputed as ``Σ col_cost · col_value + offset``).

Valid input.  The outputs are only right when all of these hold; none of them can be checked from the files:

* The MPS has row and column names (``Problem.write_mps(emit_names=True)``).  The subprocess solve path writes
  generic names (``c0``, ``r0``, ...) and gives no outputs.
* The MPS was built without autoscale Layer 2.  A Layer-2 MPS is in scaled variables
  (``flextool/engine/autoscale/_layer2.py``), and its solution gives silently wrong values.
* Its objective was scaled by the solve's ``scale_the_objective`` (``build_flextool(..., scale_the_objective=)``;
  production uses the solve's ``scale_the_objective`` DB parameter, default 1e-6).  It is read from
  ``<dump>/solve_data/scale_the_objective.csv``: the sub-solve provider holds the scale its LP was built with, so a
  provider snapshot (``--csv-dump``, a test run's work folder) carries it.  A missing or header-only file is an
  error, never a silent default.  ``scale_the_objective=`` (CLI ``--scale``) overrides it, for an MPS built at
  another scale.
* It is the same solve as the dump: the last sub-solve of the run whose work folder (or ``--csv-dump`` folder) the
  dump is.  A dump has no solve-to-solve carry, so the FlexData rebuilt from it is that of the last sub-solve only.

Known residual: an MPS written from an in-process solve whose entity names contain spaces carries HiGHS-mangled
names, while the dump's frames carry the original ones; those entities are missing from the outputs.

CLI: ``python -m flextool.engine._csv_reload.offline_outputs --mps M --sol S --solve NAME --out DIR --dump DUMP``
``[--db-url URL --scenario NAME --scale VALUE --force]``.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import polars as pl

from flextool.engine._csv_reload.load import _load
from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.engine.output._writer import _VAR_RENAME
from flextool.process_outputs.read_highs_solution import _actual_solve_name, write_all_variables

_SCALE_FILE = Path("solve_data") / "scale_the_objective.csv"


def read_scale_the_objective(dump_dir: Path | str) -> float:
    """The objective scale of the dump's solve, from ``<dump_dir>/solve_data/scale_the_objective.csv``.

    Raises :class:`FlexToolConfigError` when the file is missing, has no ``value`` row or holds a value that is
    not a positive number (no silent default).
    """
    path = Path(dump_dir) / _SCALE_FILE
    hint = "the dump does not hold the solve's objective scale; pass scale_the_objective= (CLI: --scale)"
    if not path.is_file():
        raise FlexToolConfigError(f"offline outputs: {path} is missing; {hint}")
    df = pl.read_csv(path)
    if "value" not in df.columns or df.height == 0:
        raise FlexToolConfigError(
            f"offline outputs: {path} has no 'value' row; {hint}"
        )
    try:
        value = float(df["value"][0])
    except (TypeError, ValueError):
        raise FlexToolConfigError(f"offline outputs: {path} value {df['value'][0]!r} is not a number; {hint}") from None
    if not value > 0:
        raise FlexToolConfigError(f"offline outputs: {path} value {value!r} is not positive; {hint}")
    return value


def _rename_invest_columns_by_name(h) -> None:
    """Rename ``v_invest_p[...]`` / ``v_invest_n[...]`` / ``v_divest_*[...]`` columns to the unified names.

    Production renames them on the live instance by column id (``_writer._rename_invest_columns``); the re-read
    instance has only the names, so match on the ``"<src>["`` prefix.
    """
    renames = {src: dst for src, dst in _VAR_RENAME.items() if src != dst}
    names = pl.DataFrame({"name": pl.Series(h.allVariableNames(), dtype=pl.Utf8)}).with_row_index("col")
    hits = (
        names.with_columns(head=pl.col("name").str.extract(r"^([^\[]+)\[", 1))
        .filter(pl.col("head").is_in(list(renames)))
        .with_columns(
            new=pl.col("head").replace_strict(renames, return_dtype=pl.Utf8)
            + pl.col("name").str.slice(pl.col("head").str.len_chars())
        )
    )
    for col, new_name in hits.select("col", "new").iter_rows():
        h.passColName(int(col), new_name)


def _objective_from_solution(h) -> float:
    """``Σ col_cost · col_value + offset`` of the loaded model and solution (scaled objective units)."""
    lp = h.getLp()
    cost = np.asarray(lp.col_cost_, dtype=float)
    value = np.asarray(h.getSolution().col_value, dtype=float)
    return float(cost @ value + lp.offset_)


def write_outputs_from_files(mps_file: Path | str, sol_file: Path | str, solve_name: str, out_dir: Path | str, *,
                             dump_dir: Path | str, db_url: str | None = None, scenario: str | None = None,
                             scale_the_objective: float | None = None, force: bool = False) -> list[Path]:
    """Write the output parquets of one solve into ``out_dir``; return the paths written.

    ``dump_dir`` is the solve's stamped CSV dump; ``db_url`` + ``scenario`` (together or neither) give the
    DB-direct parameters, as in :func:`load_from_dump_with_provider`.  ``scale_the_objective`` defaults to the
    dump's (:func:`read_scale_the_objective`, which raises when it is missing); pass it only to override the dump's
    value (an MPS built at another scale).  ``force`` loads a dump the stamp
    check would refuse.  See the module docstring for which MPS files are valid input.
    """
    import highspy  # local import, as in the production writer path

    dump_dir = Path(dump_dir)
    scale = float(scale_the_objective) if scale_the_objective is not None else read_scale_the_objective(dump_dir)
    if not scale > 0:
        raise FlexToolConfigError(f"offline outputs: scale_the_objective={scale!r} is not positive")
    fd, provider = _load(dump_dir, db_url=db_url, scenario=scenario, force=force, stacklevel=2)

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    if h.readModel(str(mps_file)) != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS failed to read MPS: {mps_file}")
    status = h.readSolution(str(sol_file), 0)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(
            f"HiGHS failed to read solution file: {sol_file} (status={status}).  Try re-solving instead."
        )
    _rename_invest_columns_by_name(h)
    h._flextool_unscaled_objective = _objective_from_solution(h)

    sd = dump_dir / "solve_data"
    roll = _actual_solve_name(solve_name, provider=provider)
    return write_all_variables(
        h,
        solve_name=roll,
        output_dir=out_dir,
        realized_dispatch_csv=(
            sd / "realized_dispatch.csv" if provider.has("solve_data/realized_dispatch") else None
        ),
        realized_periods_csv=(
            sd / "realized_invest_periods_of_current_solve.csv"
            if provider.has("solve_data/realized_invest_periods_of_current_solve") else None
        ),
        flex_data=fd,
        scale_the_objective=scale,
        provider=provider,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (``python -m flextool.engine._csv_reload.offline_outputs``)."""
    parser = argparse.ArgumentParser(
        description="Write a solve's output parquets from an MPS + HiGHS solution file and the solve's CSV dump "
                    "(unsupported; see the module docstring for valid MPS input).",
    )
    parser.add_argument("--mps", type=Path, required=True,
                        help="MPS file with names (write_mps(emit_names=True)), built without autoscale Layer 2")
    parser.add_argument("--sol", type=Path, required=True, help="HiGHS solution file (write_solution_style=0)")
    parser.add_argument("--solve", required=True, help="Solve name tag (the roll name is read from the dump)")
    parser.add_argument("--out", type=Path, required=True, help="Output dir")
    parser.add_argument("--dump", type=Path, required=True,
                        help="Stamped CSV dump of the same solve (work folder or --csv-dump folder)")
    parser.add_argument("--db-url", default=None, help="Input DB URL (give together with --scenario)")
    parser.add_argument("--scenario", default=None, help="Scenario name in --db-url")
    parser.add_argument("--scale", type=float, default=None,
                        help="override the scale_the_objective read from <dump>/solve_data/"
                             "scale_the_objective.csv (only for an MPS built at another scale)")
    parser.add_argument("--force", action="store_true", help="Load a dump the stamp check refuses")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    paths = write_outputs_from_files(
        args.mps, args.sol, args.solve, args.out, dump_dir=args.dump, db_url=args.db_url, scenario=args.scenario,
        scale_the_objective=args.scale, force=args.force,
    )
    for p in paths:
        print(f"Wrote {p}")


__all__ = ["main", "read_scale_the_objective", "write_outputs_from_files"]


if __name__ == "__main__":
    main()
