"""Readers and helpers of the solve-to-solve handoff: the handoff auxiliaries, realized steps, carried
online rounding, roll-end flows and the tied-copy labels."""

from __future__ import annotations

from pathlib import Path
import polars as pl
from flextool.engine.preprocess._provider_io import _provider_key
from flextool.engine.preprocess.flex_data._readers import _provider_has, _provider_read, _read_unitsize
from flextool.engine.preprocess.flex_data._container import FlexData


def _load_handoff_aux_pair(path: Path, expected: tuple[str, str],
                            *,
                            provider: "object | None" = None) -> "pl.DataFrame | None":
    """Gap F final — load a 2-col handoff-auxiliary CSV into a polars
    frame, tolerating empty / missing files.  ``expected`` lists the two
    canonical column names the caller wants; we select them and drop
    anything else.  Returns ``None`` when the file is missing, empty,
    or doesn't carry the expected columns.
    """
    _p = Path(path)
    name = f"{_p.parent.name}/{_p.stem}" if _p.parent.name else _p.stem
    if not _provider_has(provider, name, path):
        return None
    try:
        df = _provider_read(provider, name, path)
    except pl.exceptions.NoDataError:
        return None
    if df.height == 0:
        return None
    a, b = expected
    if a not in df.columns or b not in df.columns:
        return None
    return df.select(a, b)


def _read_period_set(path: Path,
                       *,
                       provider: "object | None" = None) -> set[str]:
    """Read a single-column period CSV (header row, then one period per row)."""
    def _cell_str(value: "object | None") -> str:
        return "" if value is None else str(value)

    name = _provider_key(path)
    if not _provider_has(provider, name, path):
        return set()
    try:
        df = _provider_read(provider, name, path)
    except pl.exceptions.NoDataError:
        return set()
    out: set[str] = set()
    if df.width == 0:
        return out
    col0 = df.columns[0]
    for value in df.get_column(col0):
        c0 = _cell_str(value)
        if c0:
            out.add(c0)
    return out


def _read_realize_invest_periods(path: Path,
                                   *,
                                   provider: "object | None" = None) -> set[str]:
    """Read ``realized_invest_periods_of_current_solve.csv`` (single
    ``period`` column emitted per-solve by the periods emitter).

    Empty file or missing → empty set (treat as: nothing realized this solve).
    """
    return _read_period_set(path, provider=provider)


def _read_unitsize_long(work_folder: Path,
                         *,
                         provider: "object | None" = None) -> dict[str, float]:
    """Read entity unitsizes as ``{entity: value}``.

    Source: ``solve_data/p_entity_unitsize.csv`` (long, written for every solve by
    ``emit._dispatchers._compute_p_entity_unitsize``).  Callers like
    ``build_handoff_from_solution`` rely on ``unitsize.get(n, 1.0)``, so a missing entry
    collapses to a factor of 1.0.

    (An ``input/p_entity_unitsize`` layer used to be read as a base here, for the legacy
    .mod's wide printf.  Nothing has ever populated that provider key — dropped, c24 S4-6.)
    """
    out: dict[str, float] = {}
    src = work_folder / "solve_data" / "p_entity_unitsize.csv"
    key = "solve_data/p_entity_unitsize"
    if not _provider_has(provider, key, src):
        return out
    try:
        df = _read_unitsize(src, provider=provider)
    except Exception:  # noqa: BLE001 — best-effort fallback
        return out
    for r in df.iter_rows(named=True):
        try:
            out[str(r["e"])] = float(r["value"])
        except (TypeError, ValueError):
            continue
    return out


def _read_singles_csv(path: Path,
                       *,
                       provider: "object | None" = None) -> list[str]:
    """Read a single-column CSV (header row, then one value per row)."""
    def _cell_str(value: "object | None") -> str:
        return "" if value is None else str(value)

    name = _provider_key(path)
    if not _provider_has(provider, name, path):
        return []
    try:
        df = _provider_read(provider, name, path)
    except pl.exceptions.NoDataError:
        return []
    if df.width == 0:
        return []
    col0 = df.columns[0]
    out: list[str] = []
    for value in df.get_column(col0):
        c0 = _cell_str(value)
        if c0:
            out.append(c0)
    return out


def _step_duration_frame(
    sd: Path, flex_data: "FlexData | None",
    *,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Return the (period/d, time/t, value) frame for p_step_duration.

    Phase 4 (Gap F) — prefer ``flex_data.p_step_duration`` (in-memory)
    over the workdir's ``p_step_duration.csv``.  ``flex_data`` Param
    frames use (d, t, value); the CSV uses (period, time, value).
    Callers tolerate either schema.
    """
    if flex_data is not None and getattr(flex_data, "p_step_duration", None) is not None:
        return flex_data.p_step_duration.frame
    p = sd / "p_step_duration.csv"
    if not _provider_has(provider, "solve_data/p_step_duration", p):
        return None
    try:
        return _provider_read(provider, "solve_data/p_step_duration", p)
    except pl.exceptions.NoDataError:
        return None


def _realized_dispatch_frame(
    sd: Path, flex_data: "FlexData | None",
    *,
    provider: "object | None" = None,
) -> "pl.DataFrame | None":
    """Gap F final — prefer ``flex_data.realized_dispatch`` (in-memory)
    over the workdir's ``realized_dispatch.csv``.
    """
    if flex_data is not None and getattr(flex_data, "realized_dispatch", None) is not None:
        return flex_data.realized_dispatch
    p = sd / "realized_dispatch.csv"
    if not _provider_has(provider, "solve_data/realized_dispatch", p):
        return None
    try:
        return _provider_read(provider, "solve_data/realized_dispatch", p)
    except pl.exceptions.NoDataError:
        return None


def _last_realized_step(sd: Path, flex_data: "FlexData | None", *,
                        provider: "object | None" = None) -> "tuple[str, str] | None":
    """The last realized ``(d, t)`` of a solve in time: the last row of ``solve_data/realized_dispatch`` in its
    emitted (solve-step) order — the committed jump-end of the realized trunk.  Only when that key is absent
    (unit-test synthesis without a real solve) the ``(d, t)``-sorted last row of ``flex_data.realized_dispatch``
    (the forecast-expanded set, whose sort order need not be chronological).  The carriers of the next roll
    (``roll_end_state``, ``roll_end_online``, ``roll_end_flow``) are read at this step."""
    frame = None
    emitted = False
    rd_path = sd / "realized_dispatch.csv"
    if _provider_has(provider, "solve_data/realized_dispatch", rd_path):
        try:
            frame = _provider_read(provider, "solve_data/realized_dispatch", rd_path)
            emitted = True
        except pl.exceptions.NoDataError:
            frame = None
    if frame is None and flex_data is not None and getattr(flex_data, "realized_dispatch", None) is not None:
        frame = flex_data.realized_dispatch
    if frame is None or frame.height == 0:
        return None
    cols = frame.columns
    d_col = "period" if "period" in cols else "d"
    t_col = "step" if "step" in cols else "t"
    pairs = frame.select(pl.col(d_col).cast(pl.Utf8).alias("d"), pl.col(t_col).cast(pl.Utf8).alias("t"))
    if not emitted:
        pairs = pairs.unique().sort(["d", "t"])
    return pairs["d"][-1], pairs["t"][-1]


def _committed_fan_online_cells(sd: Path, flex_data: "FlexData | None", *,
                                provider: "object | None" = None) -> "pl.DataFrame | None":
    """``(label, d_ref, t)`` Utf8, chronological: the cells a solve commits inside a forecasts-only fan on which
    units online are tied, each at the forecast whose value the tie holds (the first sorted forecast ``d_ref``).

    Inside a fan the solve realizes no dispatch (``realized_dispatch`` holds no fan cell; a split realization
    roll's label dispatches its jump under its own name and commits nothing in the fan), but the families tied
    across its forecasts are de facto realized on the tied cells, so the next solve continues its unit
    commitment from them (the committed-window check, ``solve._realization.check_committed_window_realized``,
    stops a run whose untied families would need the same).  ``None`` when the solve commits no cell in a fan
    or no online tie covers it."""
    path = sd / "committed_in_fan.csv"
    if flex_data is None or not _provider_has(provider, "solve_data/committed_in_fan", path):
        return None
    try:
        committed = _provider_read(provider, "solve_data/committed_in_fan", path)
    except pl.exceptions.NoDataError:
        return None
    if committed is None or committed.height == 0:
        return None
    from flextool.engine.model._non_anticipativity import online_tie_cells
    ties = online_tie_cells(flex_data)
    if ties is None:
        return None
    order = committed.select(pl.col("period").cast(pl.Utf8).alias("label"), pl.col("step").cast(pl.Utf8).alias("t"),
                             pl.int_range(pl.len()).alias("__order"))
    cells = (order.join(ties.rename({"anchor": "label"}), on=["label", "t"], how="inner")
             .sort("__order").select("label", "d_ref", "t"))
    return cells if cells.height > 0 else None


def snap_carried_online(frame: pl.DataFrame, integer: bool, zero: float,
                        columns: "tuple[str, ...]" = ("value",)) -> pl.DataFrame:
    """*frame* with its *columns* (units online, start-ups or shut-downs read from a solution) rounded for the next
    solve: an *integer* column to the nearest integer, a continuous one to 0 where ``|value| <= zero``
    (``output._coarse_flows.var_zero_tolerance``: the solve's primal feasibility tolerance in its own scale, so the
    move to 0 is one the solver itself accepts as feasible; a nonzero value above it is kept as it is).  The
    solver's last-digit noise depends on the platform and the thread count; rounded, near-zero and whole-unit values
    are the same on every platform in the next LP (the realization pins, the carried units online and minimum up /
    down time) and in its outage cuts (W-0592).  ``+ 0.0`` turns a rounded ``-0.0`` into ``0.0``."""
    if integer:
        return frame.with_columns(pl.col(c).cast(pl.Float64).round(0) + 0.0 for c in columns)
    return frame.with_columns(
        pl.when(pl.col(c).abs() <= zero).then(0.0).otherwise(pl.col(c).cast(pl.Float64)).alias(c)
        for c in columns)


def round_whole_unit_events(frame: pl.DataFrame) -> pl.DataFrame:
    """*frame* with its ``startup`` / ``shutdown`` (of a process whose units online are integer) rounded to whole
    units so that ``startup − shutdown`` stays the change of the rounded units online (``online_change``: the
    LP's su − sd is that integer change, up to noise).  ``shutdown ← ⌊sd + ½⌋``, ``startup ← shutdown +
    round(su − sd)``: with ``su = sd + Δ`` (Δ integer) ``startup = ⌊su + ½⌋`` and both are ≥ 0; rounding each
    alone half-to-even would break the change (1.5 / 0.5 → 2 / 0)."""
    delta = (pl.col("startup") - pl.col("shutdown")).round(0)
    sd = (pl.col("shutdown") + 0.5).floor()
    return frame.with_columns((sd + delta + 0.0).alias("startup"), (sd + 0.0).alias("shutdown"))


def _step_duration_at(flex_data: "FlexData | None", step: "tuple[str, str]") -> float:
    """``p_step_duration`` of ``step`` ``(d, t)`` (1.0 when unknown)."""
    sd_param = getattr(flex_data, "p_step_duration", None) if flex_data is not None else None
    if sd_param is not None:
        hit = sd_param.frame.filter((pl.col("d").cast(pl.Utf8) == step[0]) & (pl.col("t").cast(pl.Utf8) == step[1]))
        if hit.height > 0:
            return float(hit["value"][0])
    return 1.0


def _roll_end_flow(sol, flex_data: "FlexData | None", last: "tuple[str, str] | None") -> "pl.DataFrame | None":
    """``[process, source, sink, period, step, duration, value]``: ``v_flow`` of the ramp-limited and the
    ramp-cost arcs at the solve's last realized step; ``None`` without either, realized steps or ``v_flow``."""
    if flex_data is None or "v_flow" not in sol._vars:
        return None
    sets = [getattr(flex_data, f"process_source_sink_ramp_limit_{k}", None)
            for k in ("sink_up", "sink_down", "source_up", "source_down")]
    sets.append(getattr(flex_data, "process_source_sink_ramp_cost", None))
    sets = [x.select(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink")) for x in sets
            if x is not None and x.height > 0]
    if not sets or last is None:
        return None
    arcs = pl.concat(sets).unique()
    from flextool.engine.output._coarse_flows import var_values_at

    # The last realized step, also of coarse processes (their column of the coarse step holding it).
    frame = var_values_at(sol, flex_data, "v_flow", last[0], last[1])
    if frame.height == 0:
        return None
    rows = (frame.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "source", "sink")),
                         pl.col("value").cast(pl.Float64))
            .join(arcs, on=["p", "source", "sink"], how="semi"))
    if rows.height == 0:
        return None
    dur = _step_duration_at(flex_data, last)
    rows = rows.with_columns(pl.lit(dur, dtype=pl.Float64).alias("duration"))
    steps = getattr(flex_data, "process_coarse_step", None)
    if steps is not None and steps.height > 0:
        # A coarse process's flow is the rate of its last realized coarse step (the run of coarse steps is cut at
        # the realized end, so the step ends at ``last``); the next roll ramps from it over that step's length.
        psd = flex_data.p_step_duration.frame.select(pl.col("d").cast(pl.Utf8), pl.col("t").cast(pl.Utf8),
                                                     pl.col("value").cast(pl.Float64).alias("__dt"))
        st = steps.select(*(pl.col(c).cast(pl.Utf8) for c in ("p", "d", "b_first", "t")))
        first = st.filter((pl.col("d") == last[0]) & (pl.col("t") == last[1])).select("p", "d", "b_first")
        coarse_dur = (st.join(first, on=["p", "d", "b_first"], how="inner")
                      .join(psd, on=["d", "t"], how="left")
                      .group_by("p").agg(pl.col("__dt").fill_null(dur).sum().alias("__coarse_dur")))
        rows = (rows.join(coarse_dur, on="p", how="left")
                .with_columns(pl.coalesce(pl.col("__coarse_dur"), pl.col("duration")).alias("duration"))
                .drop("__coarse_dur"))
    return rows.select(pl.col("p").alias("process"), "source", "sink",
                       pl.lit(last[0]).alias("period"), pl.lit(last[1]).alias("step"), "duration", "value")


def _tied_copy_labels(flex_data: "FlexData | None") -> tuple[dict[str, str], dict[str, str]]:
    """``({reference copy: period label}, {branch copy: period label})``.  The first holds the branch copies whose
    investment, retirement and extension decisions are one decision of every forecast (``pd_non_anticipativity``:
    ``d`` the reference copy of each forecast star, ``_branch.forecast_star_pairs``); the second every branch copy
    (``period_branch_full`` ``(label, copy)``, label ≠ copy).  Both empty when nothing is tied."""
    na = getattr(flex_data, "pd_non_anticipativity", None)
    pbf = getattr(flex_data, "period_branch_full", None)
    if na is None or na.height == 0 or pbf is None:
        return {}, {}
    refs = set(na["d"].cast(pl.Utf8).to_list())
    label_of = {copy: label for label, copy in
                pbf.select(pl.col("d").cast(pl.Utf8), pl.col("b").cast(pl.Utf8)).iter_rows()
                if copy is not None and label != copy}
    return {copy: label for copy, label in label_of.items() if copy in refs}, label_of


def _label_tied_cohorts(cohort: "pl.DataFrame | None", tied: dict[str, str],
                        label_of: dict[str, str]) -> "pl.DataFrame | None":
    """*cohort* (the solve's cohort table, Utf8 or axis labels) with, for each tied reference copy of *tied*, a copy
    of the rows whose cohort name holds it (its investment, the generations of it, its life extensions) under the
    label: the reference copy replaced by its label in ``cohort`` / ``root`` / ``parent``, and every branch copy of
    *label_of* by its label in the period columns (``d_build``, ``d_on``, ``d_eol``).  The tied decision is then
    realized under the real period name the later solves know."""
    if cohort is None or not tied:
        return cohort
    names = [c for c in ("cohort", "root", "parent") if c in cohort.columns]
    periods = [c for c in ("d_build", "d_on", "d_eol") if c in cohort.columns]
    utf8 = cohort.with_columns(pl.col(c).cast(pl.Utf8) for c in names + periods)

    def _rename(name: "str | None") -> "str | None":
        return None if name is None else ":".join(tied.get(tok, tok) for tok in name.split(":"))

    rows = []
    for row in utf8.iter_rows(named=True):
        if any(tok in tied for tok in str(row["cohort"]).split(":")):
            rows.append({**row, **{c: _rename(row[c]) for c in names},
                         **{c: label_of.get(row[c], row[c]) for c in periods}})
    if not rows:
        return cohort
    return pl.concat([utf8, pl.DataFrame(rows, schema=utf8.schema, orient="row")], how="vertical")
