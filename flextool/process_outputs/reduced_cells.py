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
    "online": ("unit_online_dt_e", "unit_online_average_d_e", "unit_startup_d_e", "unit_startup_energy_dt_ee",
               "unit_startup_energy_d_ee",
               "connection_online_dt_e", "connection_online_average_d_e", "connection_startup_d_e"),
    "reserves": ("process_reserve_upDown_node_dt_eppe", "process_reserve_average_d_eppe",
                 "nodeGroup_slack_reserve_dt_eeg", "nodeGroup_slack_reserve_d_eeg", "reserve_prices_dt_ppg",
                 "nodeGroup_reserve_dt_eegp", "nodeGroup_reserve_d_eegp", "nodeGroup_reserve_provider_dt_eegee"),
    "inertia": ("nodeGroup_inertia_dt_g", "nodeGroup_unit_node_inertia_dt_gee", "nodeGroup_inertia_largest_flow_dt_g",
                "nodeGroup_slack_inertia_dt_g", "nodeGroup_slack_inertia_d_g"),
    "non_synchronous": ("nodeGroup_slack_nonsync_dt_g", "nodeGroup_slack_nonsync_d_g"),
    "dc_power_flow": ("dc_angle_dt_e", "dc_angle_diff_dt_e"),
}


def write_reduced_marker(output_raw: Path, solve: str, features: list[str],
                         entities: "dict[str, list[str]] | None" = None) -> None:
    """Record the features sub-solve ``solve`` left out, whole (``features``) or for some entities only
    (``entities``: feature -> entity names; their rows carry the name in column ``entity``, null for a whole
    feature); nothing is written when it left none out."""
    rows = [(f, None) for f in features] + [(f, e) for f, names in (entities or {}).items() for e in names]
    if not rows:
        return
    output_raw.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"feature": [f for f, _ in rows], "entity": [e for _, e in rows]},
                 schema={"feature": pl.Utf8, "entity": pl.Utf8}).write_parquet(
        output_raw / f"{MARKER}__{solve}.parquet")


def read_reduced_markers(output_raw: Path | str) -> dict[str, frozenset]:
    """``{sub-solve: what it left out}`` from the markers in ``output_raw`` (empty without any): a whole feature
    as its name, a feature left out for one entity as ``(feature, entity)``."""
    out: dict[str, frozenset] = {}
    prefix = f"{MARKER}__"
    for path in Path(output_raw).glob(f"{prefix}*.parquet"):
        solve = path.name[len(prefix):-len(".parquet")]
        frame = pl.read_parquet(path)
        entity = frame["entity"].to_list() if "entity" in frame.columns else [None] * frame.height
        out[solve] = frozenset(f if e is None else (f, e) for f, e in zip(frame["feature"].to_list(), entity))
    return out


@dataclasses.dataclass
class ReducedCells:
    """Per left-out feature: the ``(period, time)`` steps (a ``MultiIndex``), the periods (an ``Index``) and the
    sub-solves whose cells are empty.  A feature left out for one entity only is keyed ``(feature, entity)``."""

    steps: dict[str, pd.MultiIndex] = dataclasses.field(default_factory=dict)
    periods: dict[str, pd.Index] = dataclasses.field(default_factory=dict)
    solves: dict[str, set[str]] = dataclasses.field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.solves)

    def step_mask(self, feature: "str | tuple[str, str]", index: pd.Index) -> np.ndarray:
        """True for the rows of ``index`` (with ``period`` and ``time`` levels) at a step of ``feature``."""
        steps = self.steps.get(feature)
        if steps is None or not len(steps) or not {"period", "time"} <= set(index.names or ()):
            return np.zeros(len(index), dtype=bool)
        rows = pd.MultiIndex.from_arrays([_str_level(index, "period"), _str_level(index, "time")])
        return np.asarray(rows.isin(steps), dtype=bool)


def _str_level(index: pd.Index, name: str) -> pd.Index:
    return index.get_level_values(name).astype(str)


def reduced_cells_of(s, markers: dict[str, frozenset]) -> ReducedCells:
    """Map the markers onto the realised steps and periods of the solve-keyed sets ``s`` (before ``drop_levels``).

    A step belongs to the last sub-solve that realised it (``s.dt_realize_dispatch``, as the ``solve`` column of
    the tables).  A period is empty for a feature only when every sub-solve that realised it
    (``s.d_realized_period``) left the feature out."""
    cells = ReducedCells()
    if not markers:
        return cells
    # A string per marker key (a feature, or a (feature, entity) pair) for the groupings; mapped back at the end.
    keys: dict[str, object] = {}
    for features in markers.values():
        for f in features:
            keys.setdefault(f if isinstance(f, str) else "\x1f".join(f), f)
    left_out = pd.DataFrame([(solve, f if isinstance(f, str) else "\x1f".join(f))
                             for solve, features in markers.items() for f in features],
                            columns=["solve", "feature"])
    cells.solves = {f: set(g["solve"]) for f, g in left_out.groupby("feature", sort=False)}
    dt = getattr(s, "dt_realize_dispatch", None)
    if dt is not None and len(dt) and "solve" in (dt.names or ()):
        owner = pd.DataFrame({k: _str_level(dt, k) for k in ("solve", "period", "time")}) \
            .drop_duplicates(["period", "time"], keep="last")
        hit = owner.merge(left_out, on="solve")
        cells.steps = {f: pd.MultiIndex.from_frame(g[["period", "time"]]) for f, g in hit.groupby("feature", sort=False)}
    dp = getattr(s, "d_realized_period", None)
    if dp is not None and len(dp) and "solve" in (dp.names or ()):
        realised = pd.DataFrame({k: _str_level(dp, k) for k in ("solve", "period")}).drop_duplicates()
        n_solves = realised.groupby("period").size().rename("n_solves")
        hit = realised.merge(left_out, on="solve").groupby(["feature", "period"]).size().rename("n_left_out") \
            .reset_index().merge(n_solves, left_on="period", right_index=True)
        every = hit[hit["n_left_out"] == hit["n_solves"]]
        cells.periods = {f: pd.Index(g["period"], name="period") for f, g in every.groupby("feature", sort=False)}
    cells.solves = {keys[k]: v for k, v in cells.solves.items()}
    cells.steps = {keys[k]: v for k, v in cells.steps.items()}
    cells.periods = {keys[k]: v for k, v in cells.periods.items()}
    return cells


def _row_mask(df: pd.DataFrame, feature: "str | tuple[str, str]", cells: ReducedCells) -> np.ndarray:
    names = set(df.index.names or ())
    if "solve" in names:
        return np.asarray(_str_level(df.index, "solve").isin(list(cells.solves.get(feature, ()))), dtype=bool)
    if "time" in names:
        return cells.step_mask(feature, df.index)
    if "period" in names:
        periods = cells.periods.get(feature, pd.Index([]))
        return np.asarray(_str_level(df.index, "period").isin(periods), dtype=bool)
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
            if df is None:
                continue
            assert isinstance(df, pd.DataFrame), f"{name}: a feature table must be a DataFrame to be masked"
            if df.empty or df.shape[1] == 0:
                continue
            mask = _row_mask(df, feature, cells)
            if mask.any():
                df = df.astype(float)
                df.loc[mask, :] = np.nan
                results[name] = df
    # A feature left out for some entities only: their column, in the rows of the solves that left them out.
    for key in cells.solves:
        if not isinstance(key, tuple):
            continue
        feature, entity = key
        for name in FEATURE_TABLES.get(feature, ()):
            df = results.get(name)
            if df is None or df.empty or entity not in df.columns:
                continue
            mask = _row_mask(df, key, cells)
            if mask.any():
                df = df.astype(float)
                df.loc[mask, entity] = np.nan
                results[name] = df
