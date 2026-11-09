"""Empty output cells for the features a solve left out (``solve.reduce_*_detail_method``).

A sub-solve whose detail level left out a feature (units online, reserves, inertia limits, non-synchronous limits,
DC power flow; ``preprocess._reduce_detail.OUTPUT_FEATURES``) has no variables of it.  Its per-solve variable
parquets carry its rows without value columns, and the variable reader zero-fills them, so the tables of that
feature would show 0 in the steps the solve dispatched.  "Not modelled" is not 0: those cells are left empty (NaN),
as are the cells of investments that were not realised.  Solves without a reduction are not touched.

* The cascade writes one marker per reduced sub-solve, ``output_raw/reduced_detail__<solve>.parquet`` (column
  ``feature``), with :func:`write_reduced_marker`.
* :func:`reduced_cells_of` reads the markers and maps them onto the realised steps and periods, from the
  solve-keyed sets before ``drop_levels`` (the same last-solve-wins rule as the ``solve`` column of the tables).
* :func:`mask_reduced_cells` empties the rows of :data:`FEATURE_TABLES`: a per-step row when the solve that
  realised the step left the feature out; a per-period row when every solve that realised the period left it out
  (a period shared with a solve that kept the feature keeps its value).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

#: Stem of the per-solve marker parquet in ``output_raw/``.
MARKER = "reduced_detail"

#: The output tables each left-out feature empties.  Costs stay (a left-out feature costs nothing in the objective
#: of the solve), and so do the flows, ramps and the VRE share (the dispatch has them whatever the detail).
FEATURE_TABLES: dict[str, tuple[str, ...]] = {
    "online": ("unit_online_dt_e", "unit_online_average_d_e", "unit_startup_d_e",
               "connection_online_dt_e", "connection_online_average_d_e", "connection_startup_d_e"),
    "reserves": ("process_reserve_upDown_node_dt_eppe", "process_reserve_average_d_eppe",
                 "nodeGroup_slack_reserve_dt_eeg", "nodeGroup_slack_reserve_d_eeg", "reserve_prices_dt_ppg"),
    "inertia": ("nodeGroup_inertia_dt_g", "nodeGroup_unit_node_inertia_dt_gee", "nodeGroup_inertia_largest_flow_dt_g",
                "nodeGroup_slack_inertia_dt_g", "nodeGroup_slack_inertia_d_g"),
    "non_synchronous": ("nodeGroup_slack_nonsync_dt_g", "nodeGroup_slack_nonsync_d_g"),
    "dc_power_flow": ("dc_angle_dt_e", "dc_angle_diff_dt_e"),
}


def write_reduced_marker(output_raw: Path, solve: str, features: list[str]) -> None:
    """Record the features sub-solve ``solve`` left out; nothing is written when it left none out."""
    if not features:
        return
    output_raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"feature": list(features)}, schema={"feature": pl.Utf8}).write_parquet(
        output_raw / f"{MARKER}__{solve}.parquet")


def read_reduced_markers(output_raw: Path | str) -> dict[str, frozenset[str]]:
    """``{sub-solve: features it left out}`` from the markers in ``output_raw`` (empty without any)."""
    out: dict[str, frozenset[str]] = {}
    prefix = f"{MARKER}__"
    for path in Path(output_raw).glob(f"{prefix}*.parquet"):
        solve = path.name[len(prefix):-len(".parquet")]
        out[solve] = frozenset(pl.read_parquet(path)["feature"].to_list())
    return out


@dataclasses.dataclass
class ReducedCells:
    """Per left-out feature: the ``(period, time)`` steps, the periods and the sub-solves whose cells are empty."""

    steps: dict[str, set[tuple[str, str]]] = dataclasses.field(default_factory=dict)
    periods: dict[str, set[str]] = dataclasses.field(default_factory=dict)
    solves: dict[str, set[str]] = dataclasses.field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.solves)

    def step_mask(self, feature: str, index: pd.Index) -> np.ndarray:
        """True for the rows of ``index`` (with ``period`` and ``time`` levels) at a step of ``feature``."""
        steps = self.steps.get(feature)
        if not steps or not {"period", "time"} <= set(index.names or ()):
            return np.zeros(len(index), dtype=bool)
        pairs = zip(index.get_level_values("period"), index.get_level_values("time"))
        return np.fromiter(((str(d), str(t)) in steps for d, t in pairs), dtype=bool, count=len(index))


def reduced_cells_of(s, markers: dict[str, frozenset[str]]) -> ReducedCells:
    """Map the markers onto the realised steps and periods of the solve-keyed sets ``s`` (before ``drop_levels``).

    A step belongs to the last sub-solve that realised it (``s.dt_realize_dispatch``, as the ``solve`` column of
    the tables).  A period is empty for a feature only when every sub-solve that realised it
    (``s.d_realized_period``) left the feature out."""
    cells = ReducedCells()
    if not markers:
        return cells
    for solve, features in markers.items():
        for f in features:
            cells.solves.setdefault(f, set()).add(solve)
    dt = getattr(s, "dt_realize_dispatch", None)
    if dt is not None and len(dt) and "solve" in (dt.names or ()):
        owner: dict[tuple[str, str], str] = {}
        for solve, d, t in zip(dt.get_level_values("solve"), dt.get_level_values("period"),
                               dt.get_level_values("time")):
            owner[(str(d), str(t))] = str(solve)
        for step, solve in owner.items():
            for f in markers.get(solve, ()):
                cells.steps.setdefault(f, set()).add(step)
    dp = getattr(s, "d_realized_period", None)
    if dp is not None and len(dp) and "solve" in (dp.names or ()):
        by_period: dict[str, set[str]] = {}
        for solve, d in zip(dp.get_level_values("solve"), dp.get_level_values("period")):
            by_period.setdefault(str(d), set()).add(str(solve))
        for d, solves in by_period.items():
            for f in set.intersection(*(set(markers.get(x, ())) for x in solves)):
                cells.periods.setdefault(f, set()).add(d)
    return cells


def _row_mask(df: pd.DataFrame, feature: str, cells: ReducedCells) -> np.ndarray:
    names = set(df.index.names or ())
    if "solve" in names:
        solves = cells.solves.get(feature, set())
        return np.asarray([str(x) in solves for x in df.index.get_level_values("solve")], dtype=bool)
    if "time" in names:
        return cells.step_mask(feature, df.index)
    if "period" in names:
        periods = cells.periods.get(feature, set())
        return np.asarray([str(x) in periods for x in df.index.get_level_values("period")], dtype=bool)
    return np.zeros(len(df), dtype=bool)


def mask_reduced_cells(results: dict, cells: ReducedCells) -> None:
    """Empty (NaN), in place, the rows of :data:`FEATURE_TABLES` that belong to a sub-solve that left the feature
    out.  A no-op without reduced solves."""
    if not cells:
        return
    for feature, tables in FEATURE_TABLES.items():
        if feature not in cells.solves:
            continue
        for name in tables:
            df = results.get(name)
            if not isinstance(df, pd.DataFrame) or df.empty or df.shape[1] == 0:
                continue
            mask = _row_mask(df, feature, cells)
            if mask.any():
                df = df.astype(float)
                df.loc[mask, :] = np.nan
                results[name] = df
