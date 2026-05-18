"""``FlexData.dump_csvs(workdir)`` — debug oracle that writes a
materialised ``FlexData`` back to flextool's ``input/`` + ``solve_data/``
CSV layout.

**Use case.**  When the DB-direct loader's output for some Param
diverges from what flextool's preprocessing would have produced, this
helper lets a developer dump both views to disk and ``diff`` them.
The DB-loaded ``FlexData`` is the in-memory canonical;
``dump_csvs(tmpdir)`` materialises that canonical to CSV so a
side-by-side ``diff -r tmpdir tests/data/work_<fixture>`` surfaces the
divergence at file granularity.

**Round-trip contract.**  After::

    data = load_flextool(workdir, db_reader=reader)
    data.dump_csvs(tmp)
    redo = load_flextool(tmp)

every ``Param`` / ``DataFrame`` field on ``data`` and ``redo`` should
compare frame-equal up to row order.  This is the basis of
:func:`tests.test_flex_dump_csvs_roundtrip.test_dump_csvs_roundtrip`.

**Scope.**  The mapping covers every FlexData field that
``flextool/input.py`` reads from a single canonical CSV via a
direct-rename (the bulk of the surface).  A handful of Params today
read from sliced wide-by-param files (``pdtNode.csv``,
``pdtCommodity.csv``, ``pdtGroup.csv``, ``pdtProcess.csv``,
``pd_group.csv``, ``p_commodity.csv``, ``p_node.csv``, ``p_process.csv``,
``p_group.csv``) — those slice files we reconstruct by writing the
sliced rows back into a wide-by-param file (one row per (entity,
param, …)).  Where flextool emits both an ``input/`` and a
``solve_data/`` copy of the same logical CSV, ``dump_csvs`` writes
to ``solve_data/`` (the canonical post-preprocessing location) and
also to ``input/`` when ``input.py`` reads from there for that field.

**Not in scope.**  Per-solve metadata (``solve_current.csv``,
``period_first_of_solve.csv``, ``solve_mode.csv``, etc.) — these are
flextool-runner state, not FlexData fields, and the round-trip path
doesn't need them when the source ``workdir`` is also handed to
``load_flextool`` (we can copy them in advance).  ``dump_csvs`` will
copy them through if a ``copy_meta_from`` argument is given (used by
the round-trip test).
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import polars as pl

from polar_high import Param

from ._axis_enums import rename_to_axis, schema_dtype


# ---------------------------------------------------------------------------
# Heavy CSVs — gigabyte-scale on large fixtures (e.g. y2050 H2 supply
# curves: ~3.8 GB total per cascade iteration).  These files are NOT
# consumed by ``flextool/process_outputs/`` or ``flextool/plot_outputs/``
# post-processing; they were originally written so legacy preprocessing
# helpers could diff-roundtrip the in-memory FlexData against the
# flextoolrunner CSV writer's output.  The cascade's downstream
# consumers all read from ``flex_data.<field>.frame`` in-memory.
#
# Set ``FLEXTOOL_DUMP_CSVS=1`` to restore the debug-oracle behaviour
# (write the full ~50 CSVs).  When the env var is unset (default),
# these seven files are skipped — every other file in ``DIRECT_WRITES``
# and every sliced / metadata file continues to be emitted.
_HEAVY_CSV_FILES: frozenset[str] = frozenset({
    "p_flow_max.csv",
    "pdtProcess__source__sink__dt_varCost.csv",
    "pdtProcess_slope.csv",
    "pdtProcess.csv",
    "pdtNode.csv",
    "pdtNodeInflow.csv",
    "ptNode_inflow.csv",
})


def _heavy_dumps_enabled() -> bool:
    """Return True iff the heavy CSV writes are gated on (debug oracle).

    The default is OFF: the seven gigabyte-scale CSVs in
    :data:`_HEAVY_CSV_FILES` are not written by ``dump_csvs`` unless
    ``FLEXTOOL_DUMP_CSVS=1`` is set in the environment.
    """
    return os.environ.get("FLEXTOOL_DUMP_CSVS") == "1"


# ---------------------------------------------------------------------------
# Field → CSV mapping.
#
# One entry per FlexData field we know how to write.  Each entry is one of:
#
#   * ``("solve_data", "<file>.csv", {flex_col: csv_col, ...})`` — direct
#     write of the field's frame, applying the rename (renaming flexpy's
#     short names back to flextool's long names).
#
#   * ``("input", "<file>.csv", {...})`` — same, but to ``input/``.
#
#   * ``("slice", kind, file, entity_col, param_value)`` — write into a
#     wide-by-param file (e.g. ``pdtNode.csv`` rows with
#     ``param=penalty_up``).  The dump aggregates all slice rows for a
#     file across multiple FlexData fields in a single pass.
#
# Unknown fields (None on the FlexData) are skipped silently — the
# round-trip semantics are "every populated field round-trips"; we
# never fabricate empty CSVs.

# Direct frame → CSV writes (set & param fields).  Format:
#   field_name -> (kind, csv_filename, rename_to_csv)
DIRECT_WRITES: dict[str, tuple[str, str, dict[str, str]]] = {
    # ─── Time / weighting ─────────────────────────────────────────────
    "p_step_duration": ("solve_data", "steps_in_use.csv",
                        {"d": "period", "t": "step", "value": "step_duration"}),
    "p_rp_cost_weight": ("solve_data", "rp_cost_weight.csv",
                         {"d": "period", "t": "time", "value": "weight"}),
    "p_inflation_op": ("solve_data", "p_inflation_factor_operations_yearly.csv",
                        {"d": "period", "value": "value"}),
    "p_period_share": ("solve_data", "complete_period_share_of_year_calc.csv",
                        {"d": "period", "value": "value"}),
    # ─── Nodes ────────────────────────────────────────────────────────
    "nodeBalance": ("solve_data", "nodeBalance.csv", {"n": "node"}),
    "p_inflow": ("solve_data", "pdtNodeInflow.csv",
                 {"n": "node", "d": "period", "t": "time", "value": "value"}),
    # ─── Process topology ────────────────────────────────────────────
    "process_source_sink": ("solve_data", "process_source_sink.csv",
                             {"p": "process"}),
    "process_source_sink_eff": ("solve_data", "process_source_sink_eff.csv",
                                {"p": "process"}),
    "process_source_sink_noEff": ("solve_data", "process_source_sink_noEff.csv",
                                   {"p": "process"}),
    "p_unitsize": ("solve_data", "p_entity_unitsize.csv",
                   {"p": "entity", "value": "value"}),
    # p_flow_upper -> p_flow_max (long form: process, source, sink, period, time, value)
    "p_flow_upper": ("solve_data", "p_flow_max.csv",
                     {"p": "process", "d": "period", "t": "time"}),
    "p_slope": ("solve_data", "pdtProcess_slope.csv",
                {"p": "process", "d": "period", "t": "time"}),
    # ─── CO2 cap / price ─────────────────────────────────────────────
    "group_co2_max_period": ("solve_data", "group_co2_max_period.csv",
                              {"g": "group"}),
    "group_co2_max_total":  ("solve_data", "group_co2_max_total.csv",
                              {"g": "group"}),
    # ─── User-defined flow constraints ───────────────────────────────
    # ─── Profiles ────────────────────────────────────────────────────
    # ─── Invest / divest sets ────────────────────────────────────────
    "ed_invest_set": ("solve_data", "ed_invest.csv",
                       {"e": "entity", "d": "period"}),
    "ed_divest_set": ("solve_data", "ed_divest.csv",
                       {"e": "entity", "d": "period"}),
    "pd_invest_set": ("solve_data", "pd_invest.csv",
                       {"p": "process", "d": "period"}),
    "pd_divest_set": ("solve_data", "pd_divest.csv",
                       {"p": "process", "d": "period"}),
    "nd_invest_set": ("solve_data", "nd_invest.csv",
                       {"n": "node", "d": "period"}),
    "nd_divest_set": ("solve_data", "nd_divest.csv",
                       {"n": "node", "d": "period"}),
    "edd_invest_set": ("solve_data", "edd_invest.csv",
                        {"e": "entity", "d_invest": "d_invest", "d": "period"}),
    "e_invest_total": ("solve_data", "e_invest_total.csv",
                        {"e": "entity"}),
    "e_divest_total": ("solve_data", "e_divest_total.csv",
                        {"e": "entity"}),
    # ─── Online (UC) sets ────────────────────────────────────────────
    "process_online": ("solve_data", "process_online.csv", {"p": "process"}),
    "process_online_linear": ("solve_data", "process_online_linear.csv",
                               {"p": "process"}),
    "process_online_integer": ("solve_data", "process_online_integer.csv",
                                {"p": "process"}),
    "process_minload": ("solve_data", "process_minload.csv", {"p": "process"}),
    # ─── Storage ──────────────────────────────────────────────────────
    "nodeState": ("solve_data", "nodeState.csv", {"n": "node"}),
    "nodeStateBlock": ("solve_data", "nodeStateBlock.csv", {"n": "node"}),
    # ─── Variable cost ────────────────────────────────────────────────
    "p_pssdt_varCost": ("solve_data", "pdtProcess__source__sink__dt_varCost.csv",
                         {"p": "process", "d": "period", "t": "time",
                          "value": "value"}),
    # ─── Fixed cost / scaling ─────────────────────────────────────────
    "p_ed_fixed_cost": ("solve_data", "ed_fixed_cost.csv",
                         {"e": "entity", "d": "period"}),
    "p_entity_all_existing": ("solve_data", "p_entity_all_existing.csv",
                               {"e": "entity", "d": "period"}),
    "p_node_capacity_for_scaling": ("solve_data", "node_capacity_for_scaling.csv",
                                     {"n": "node", "d": "period"}),
    # ─── DC power flow ────────────────────────────────────────────────
    "node_dc_power_flow": ("input", "node_dc_power_flow.csv", {"n": "node"}),
    "connection_dc_power_flow": ("input", "connection_dc_power_flow.csv",
                                  {"p": "connection"}),
    "node_reference_angle": ("input", "node_reference_angle.csv", {"n": "node"}),
    # ─── Stochastics ──────────────────────────────────────────────────
    "pdt_branch_weight": ("solve_data", "pdt_branch_weight.csv",
                           {"d": "period", "t": "time", "value": "value"}),
    "pd_branch_weight": ("solve_data", "pd_branch_weight.csv",
                          {"d": "period", "value": "value"}),
    "period_in_use_set": ("solve_data", "period_in_use_set.csv",
                           {"d": "period"}),
}


def _frame_of(value: Any) -> pl.DataFrame | None:
    """Return the eager polars frame for a Param or DataFrame value.
    Returns None when the value is None or empty.
    """
    if value is None:
        return None
    if isinstance(value, Param):
        f = value.frame
    elif isinstance(value, pl.DataFrame):
        f = value
    elif isinstance(value, pl.LazyFrame):
        f = value.collect()
    else:
        return None
    if f.height == 0:
        return None
    return f


def _write_frame(frame: pl.DataFrame, path: Path,
                  rename: dict[str, str]) -> None:
    """Apply ``rename`` (flex columns → CSV columns), reorder columns
    so the rename-target order matches the original ``rename`` dict
    (insertion order = canonical column order), and write CSV.
    Columns NOT in the rename map stay in their relative position
    after the renamed ones.
    """
    df = frame
    # Apply the rename only for present columns.
    eff_rename = {k: v for k, v in rename.items() if k in df.columns}
    if eff_rename:
        df = df.pipe(rename_to_axis, eff_rename)
    # Order: rename targets first (in dict order), then any remaining cols.
    target_order = [v for v in rename.values() if v in df.columns]
    rest = [c for c in df.columns if c not in target_order]
    final_cols = target_order + rest
    df = df.select(final_cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def dump_csvs(data: "FlexData", workdir: Path | str,
               *, copy_meta_from: Path | str | None = None,
               include_heavy: bool | None = None) -> Path:
    """Materialise ``data`` to ``workdir/input/`` + ``workdir/solve_data/``
    in flextool's CSV layout.

    Parameters
    ----------
    data : FlexData
        The in-memory FlexData to dump.
    workdir : Path | str
        Target directory.  Created if missing.
    copy_meta_from : Path | str, optional
        When supplied, copies per-solve metadata files
        (``solve_current.csv``, ``period_first_of_solve.csv``,
        ``solve_mode.csv``, etc. — see :data:`_META_FILES_TO_COPY`)
        from this source workdir into the target.  This is the
        round-trip use case where the CSV reader needs metadata
        (timeline, solve list) that isn't on FlexData but is in the
        original workdir.
    include_heavy : bool, optional
        When ``True``, force-write the seven gigabyte-scale CSVs in
        :data:`_HEAVY_CSV_FILES` regardless of the
        ``FLEXTOOL_DUMP_CSVS`` env var.  When ``False``, force-skip
        them.  When ``None`` (default), consult the env var — set to
        ``"1"`` to enable.  The round-trip regression test
        (``tests/engine_polars/test_dump_csvs_roundtrip.py``) passes
        ``include_heavy=True`` to exercise the full debug-oracle
        round-trip contract.

    Returns
    -------
    Path
        The ``workdir`` (as a :class:`pathlib.Path`).
    """
    work = Path(workdir)
    inp_dir = work / "input"
    sd_dir = work / "solve_data"
    inp_dir.mkdir(parents=True, exist_ok=True)
    sd_dir.mkdir(parents=True, exist_ok=True)

    # ─── Direct field → CSV writes ───────────────────────────────────
    heavy_on = _heavy_dumps_enabled() if include_heavy is None else include_heavy
    for field, (kind, csv_name, rename) in DIRECT_WRITES.items():
        if not hasattr(data, field):
            continue
        if not heavy_on and csv_name in _HEAVY_CSV_FILES:
            # Skip gigabyte-scale CSVs that no post-processing reads
            # (see :data:`_HEAVY_CSV_FILES` for the list).
            continue
        value = getattr(data, field)
        f = _frame_of(value)
        if f is None:
            continue
        target_dir = sd_dir if kind == "solve_data" else inp_dir
        _write_frame(f, target_dir / csv_name, rename)

    # ─── dt set (special: drop value column from p_step_duration) ─
    # ``data.dt`` is the (d, t) set; flextool writes it implicitly via
    # steps_in_use.csv (handled by the p_step_duration write above).
    # If steps_in_use.csv was NOT written (no p_step_duration), but dt
    # exists, fall back to writing a header-only steps_in_use.csv.
    if (data.p_step_duration is None
            and getattr(data, "dt", None) is not None):
        f = data.dt.with_columns(value=pl.lit(1.0)) if data.dt.height > 0 else None
        if f is not None:
            _write_frame(f, sd_dir / "steps_in_use.csv",
                         {"d": "period", "t": "step", "value": "step_duration"})

    # ─── flow_to_n / flow_from_n: derived sets used by the loader ──
    # ``flow_to_n`` is just pss with sink renamed to n; we don't need to
    # write a separate CSV — the loader reconstructs from
    # process_source_sink.csv.

    # ─── Sliced (wide-by-param) files ─────────────────────────────────
    # pdtNode.csv carries (node, param, period, time, value) rows for
    # multiple FlexData Params: penalty_up, penalty_down, availability,
    # storage_state_reference_value.  Same pattern for pdtProcess.csv,
    # pdtCommodity.csv, pdtGroup.csv (period+time keys), and
    # pdProcess.csv / pdGroup.csv (period only).
    _write_pdt_sliced(data, sd_dir, heavy_on=heavy_on)
    _write_pd_sliced(data, sd_dir)
    _write_p_input_sliced(data, inp_dir)

    # ─── Capacity / unitsize composites ───────────────────────────────
    # p_entity_period_existing_capacity.csv has TWO value columns
    # (existing + invested).  flexpy's p_entity_all_existing carries
    # the 'all_existing' column; for round-trip we recompute existing =
    # all_existing - 0 (no prior invest) and invested = 0.  The reader
    # prefers p_entity_all_existing.csv when it exists, so this is a
    # stub for compatibility.
    if data.p_entity_all_existing is not None:
        f = _frame_of(data.p_entity_all_existing)
        if f is not None:
            stub = (f.pipe(rename_to_axis, {"e": "entity", "d": "period"})
                     .with_columns(
                         p_entity_period_existing_capacity=pl.col("value"),
                         p_entity_period_invested_capacity=pl.lit(0.0),
                     )
                     .select("entity", "period",
                             "p_entity_period_existing_capacity",
                             "p_entity_period_invested_capacity"))
            stub.write_csv(sd_dir / "p_entity_period_existing_capacity.csv")

    # ─── input/ entity-class set + wide-format unitsize (Δ.30) ────────
    # flextool's handoff_writers (process_outputs/handoff_writers.py) read
    # ``input/p_entity_unitsize.csv`` (wide-transposed: ``entity,<e1>,<e2>,…``
    # / ``value,<v1>,<v2>,…``) plus the entity-class set CSVs
    # ``input/{entity,process_unit,process_connection,node}.csv``
    # (single-column lists).  These are produced by flextool's
    # preprocessing pipeline on the slow path; the fast path skips
    # preprocessing, so without these the handoff writers fail with
    # ``[Errno 2]`` and the wide-format output writers downstream see
    # an incomplete workdir.  We synthesise them from FlexData here.
    _write_input_entity_set_csvs(data, inp_dir)
    _write_input_p_entity_unitsize_wide(data, inp_dir)

    # ─── solve_data/ handoff stub CSVs (Δ.30) ─────────────────────────
    # ``write_p_entity_period_existing_capacity`` reads
    # ``solve_data/solve__p_entity_pre_existing.csv`` (wide-by-entity,
    # indexed by (solve, period)).  In single-solve mode there's no
    # prior solve so no pre-existing capacity has been carried over;
    # write a header-only stub so ``pd.read_csv(... index_col=[0, 1])``
    # returns an empty frame and the writer's ``if df.empty: return {}``
    # branch fires.  Idempotent — overwrites only when missing.
    pre_existing_path = sd_dir / "solve__p_entity_pre_existing.csv"
    if not pre_existing_path.exists():
        pre_existing_path.write_text("solve,period\n")

    # ─── Optional metadata copy-through ───────────────────────────────
    if copy_meta_from is not None:
        _copy_meta(Path(copy_meta_from), work)

    return work


def _write_input_entity_set_csvs(data: "FlexData", inp_dir: Path) -> None:
    """Δ.30 — write the four single-column entity-class set CSVs that
    handoff_writers' ``_load_entity_class_set`` reads.

    ``input/node.csv`` — every row in ``data.nodeBalance`` (column ``n``).
    ``input/process_unit.csv`` — every row in ``data.process_unit``
        (column ``p``).  Header-only when the field is None / empty.
    ``input/process_connection.csv`` — every process in
        ``data.process_source_sink`` whose ``p`` is NOT in
        ``data.process_unit``.  Header-only when no connection processes
        exist (the common case for unit-only fixtures).
    ``input/entity.csv`` — union of nodes + processes.

    All four files MUST exist for handoff_writers to walk them; an
    empty (header-only) file is fine.
    """
    # node.csv
    nb = _frame_of(data.nodeBalance) if data.nodeBalance is not None else None
    if nb is not None:
        node_lf = nb.pipe(rename_to_axis, {"n": "node"}).select("node")
    else:
        _enums = getattr(data, "_axis_enums", None)
        node_lf = pl.DataFrame({"node": []},
                                schema={"node": schema_dtype(_enums, "node")})
    node_lf.write_csv(inp_dir / "node.csv")

    # process_unit.csv
    pu = _frame_of(data.process_unit) if data.process_unit is not None else None
    if pu is not None:
        pu_out = pu.rename({"p": "process_unit"}).select("process_unit")
    else:
        _enums = getattr(data, "_axis_enums", None)
        pu_out = pl.DataFrame({"process_unit": []},
                              schema={"process_unit": schema_dtype(_enums, "process_unit")})
    pu_out.write_csv(inp_dir / "process_unit.csv")

    # process_connection.csv = (every process in pss) - process_unit
    pss = (
        _frame_of(data.process_source_sink)
        if data.process_source_sink is not None
        else None
    )
    pu_set: set[str] = (
        set(pu["p"].cast(pl.Utf8).to_list()) if pu is not None else set()
    )
    if pss is not None and "p" in pss.columns:
        all_p = set(pss["p"].cast(pl.Utf8).to_list())
        conn_processes = sorted(all_p - pu_set)
    else:
        conn_processes = []
    pl.DataFrame({"process_connection": conn_processes}).write_csv(
        inp_dir / "process_connection.csv"
    )

    # entity.csv = nodes + processes (sorted, unique).
    nodes = (
        nb["n"].cast(pl.Utf8).to_list() if nb is not None else []
    )
    processes = (
        list(set(pss["p"].cast(pl.Utf8).to_list())) if pss is not None and "p" in pss.columns else []
    )
    entities = sorted(set(nodes) | set(processes))
    pl.DataFrame({"entity": entities}).write_csv(inp_dir / "entity.csv")


def _write_input_p_entity_unitsize_wide(data: "FlexData",
                                         inp_dir: Path) -> None:
    """Δ.30 — write the wide-transposed ``input/p_entity_unitsize.csv``
    that ``handoff_writers._load_unitsize`` consumes.

    Layout (read with ``pd.read_csv(path, index_col=0)``):

        entity,e1,e2,e3,...
        value,v1,v2,v3,...

    Source: union of ``data.p_unitsize`` (process unitsize, column ``p``)
    and ``data.p_state_unitsize`` (node-state unitsize, column ``n``).
    Entities not present in either Param fall back to 1000.0 — the same
    default flextool's preprocessing applies (see
    ``flextoolrunner/preprocessing/entity_period_calc_params.py:191``).
    """
    # Build the (entity → value) map.
    entity_value: dict[str, float] = {}
    pu_param = data.p_unitsize
    pu_frame = _frame_of(pu_param) if pu_param is not None else None
    if pu_frame is not None and "p" in pu_frame.columns and "value" in pu_frame.columns:
        for row in pu_frame.iter_rows(named=True):
            entity_value[str(row["p"])] = float(row["value"])
    psu_param = getattr(data, "p_state_unitsize", None)
    psu_frame = _frame_of(psu_param) if psu_param is not None else None
    if psu_frame is not None and "n" in psu_frame.columns and "value" in psu_frame.columns:
        for row in psu_frame.iter_rows(named=True):
            entity_value[str(row["n"])] = float(row["value"])

    # Discover the full entity universe so every node / process appears
    # in the wide CSV — even when the Params didn't carry it.  Default
    # 1000.0 mirrors flextool's preprocessing default.
    nb = _frame_of(data.nodeBalance) if data.nodeBalance is not None else None
    pss = (
        _frame_of(data.process_source_sink)
        if data.process_source_sink is not None
        else None
    )
    nodes = (
        nb["n"].cast(pl.Utf8).to_list() if nb is not None else []
    )
    processes = (
        list(set(pss["p"].cast(pl.Utf8).to_list())) if pss is not None and "p" in pss.columns else []
    )
    universe = sorted(set(nodes) | set(processes))
    rows: list[tuple[str, float]] = [
        (e, entity_value.get(e, 1000.0)) for e in universe
    ]

    # Wide-transposed: header row "entity,e1,e2,...", value row "value,v1,v2,...".
    if not rows:
        # Header-only file — handoff_writers' _load_unitsize would fail,
        # but _load_unitsize_map (the forgiving variant) tolerates a
        # missing 'value' index.  Emit a single-column header so
        # ``pd.read_csv(... index_col=0)`` returns an empty frame.
        (inp_dir / "p_entity_unitsize.csv").write_text("entity\nvalue\n")
        return
    header = "entity," + ",".join(e for e, _ in rows)
    values = "value," + ",".join(repr(v) for _, v in rows)
    (inp_dir / "p_entity_unitsize.csv").write_text(header + "\n" + values + "\n")


# ---------------------------------------------------------------------------
# Sliced-by-param writers

# Each entry: {param_value: (FlexData_field, dim_for_value)}.  The
# key is the literal `param=` value in the wide-by-param CSV.
_PDT_NODE_SLICES = {
    "penalty_up":   "p_penalty_up",
    "penalty_down": "p_penalty_down",
    "availability": "p_node_availability",
    "storage_state_reference_value": "p_storage_state_reference_value",
}

_PDT_PROCESS_SLICES = {
    "availability": "p_process_availability",
}

_PDT_COMMODITY_SLICES = {
    "price": "p_commodity_price",
}

_PDT_GROUP_SLICES = {
    "co2_price": "p_co2_price",
}


def _write_pdt_sliced(data: "FlexData", sd_dir: Path,
                       *, heavy_on: bool | None = None) -> None:
    """Write the wide-by-param ``pdt*.csv`` files (period + time keyed)."""
    if heavy_on is None:
        heavy_on = _heavy_dumps_enabled()
    # pdtNode.csv columns: node, param, period, time, value
    if heavy_on or "pdtNode.csv" not in _HEAVY_CSV_FILES:
        rows: list[pl.DataFrame] = []
        for param_value, field in _PDT_NODE_SLICES.items():
            f = _frame_of(getattr(data, field, None))
            if f is None:
                continue
            # Expected schema: (n, d, t, value) — rename to canonical and add
            # the literal ``param=`` column.
            ren = {"n": "node", "d": "period", "t": "time"}
            rows.append((f.pipe(rename_to_axis,
                                 {k: v for k, v in ren.items() if k in f.columns})
                           .with_columns(param=pl.lit(param_value))
                           .select("node", "param", "period", "time", "value")))
        if rows:
            out = pl.concat(rows, how="vertical_relaxed")
            out.write_csv(sd_dir / "pdtNode.csv")

    # pdtProcess.csv columns: process, param, period, time, value
    if heavy_on or "pdtProcess.csv" not in _HEAVY_CSV_FILES:
        rows = []
        for param_value, field in _PDT_PROCESS_SLICES.items():
            f = _frame_of(getattr(data, field, None))
            if f is None:
                continue
            ren = {"p": "process", "d": "period", "t": "time"}
            rows.append((f.pipe(rename_to_axis,
                                 {k: v for k, v in ren.items() if k in f.columns})
                           .with_columns(param=pl.lit(param_value))
                           .select("process", "param", "period", "time", "value")))
        if rows:
            out = pl.concat(rows, how="vertical_relaxed")
            out.write_csv(sd_dir / "pdtProcess.csv")

    # pdtCommodity.csv columns: commodity, param, period, time, value
    rows = []
    for param_value, field in _PDT_COMMODITY_SLICES.items():
        f = _frame_of(getattr(data, field, None))
        if f is None:
            continue
        ren = {"c": "commodity", "d": "period", "t": "time"}
        rows.append((f.pipe(rename_to_axis,
                             {k: v for k, v in ren.items() if k in f.columns})
                       .with_columns(param=pl.lit(param_value))
                       .select("commodity", "param", "period", "time", "value")))
    if rows:
        out = pl.concat(rows, how="vertical_relaxed")
        out.write_csv(sd_dir / "pdtCommodity.csv")

    # pdtGroup.csv columns: group, param, period, time, value
    rows = []
    for param_value, field in _PDT_GROUP_SLICES.items():
        f = _frame_of(getattr(data, field, None))
        if f is None:
            continue
        ren = {"g": "group", "d": "period", "t": "time"}
        rows.append((f.pipe(rename_to_axis,
                             {k: v for k, v in ren.items() if k in f.columns})
                       .with_columns(param=pl.lit(param_value))
                       .select("group", "param", "period", "time", "value")))
    if rows:
        out = pl.concat(rows, how="vertical_relaxed")
        out.write_csv(sd_dir / "pdtGroup.csv")


# pdProcess.csv columns: process, param, period, value
_PD_PROCESS_SLICES = {
    "startup_cost": "p_startup_cost",
}

# pdGroup.csv columns: group, param, period, value (1d_map period)
_PD_GROUP_SLICES: dict[str, str] = {}


def _write_pd_sliced(data: "FlexData", sd_dir: Path) -> None:
    """Write wide-by-param ``pd*.csv`` files (period only, no time)."""
    rows: list[pl.DataFrame] = []
    for param_value, field in _PD_PROCESS_SLICES.items():
        f = _frame_of(getattr(data, field, None))
        if f is None:
            continue
        ren = {"p": "process", "d": "period"}
        rows.append((f.pipe(rename_to_axis,
                             {k: v for k, v in ren.items() if k in f.columns})
                       .with_columns(param=pl.lit(param_value))
                       .select("process", "param", "period", "value")))
    if rows:
        out = pl.concat(rows, how="vertical_relaxed")
        out.write_csv(sd_dir / "pdProcess.csv")


# p_commodity.csv columns: commodity, commodityParam, p_commodity (long-by-param)
_P_COMMODITY_SLICES = {
    "co2_content": "p_co2_content",
    "unitsize":    "p_commodity_unitsize",
}

# p_node.csv columns: node, nodeParam, p_node
_P_NODE_SLICES = {
    "self_discharge_loss": "p_state_self_discharge",
    "storage_state_start": "p_state_start",
}

# p_process.csv columns: process, processParam, p_process
_P_PROCESS_SLICES = {
    "min_load": "p_min_load",
}


def _merge_param_slice(path: Path, new_rows: pl.DataFrame,
                         entity_col: str, param_col: str) -> pl.DataFrame:
    """Merge a freshly-built wide-by-param slice with the existing file.

    The legacy ``input_writer.write_parameter`` populates
    ``input/p_<class>.csv`` with the full entity-level scalar surface
    (e.g. for nodes: ``penalty_up``, ``existing``, ``lifetime``,
    ``self_discharge_loss``, …).  The override-chain-driven dump in
    :func:`_write_p_input_sliced` knows the canonical value for only a
    handful of slices (``_P_NODE_SLICES`` etc.), so a plain
    ``write_csv`` would wipe the rest of the entity-level surface.

    Read existing rows, drop those whose ``param_col`` is replaced by
    *new_rows*, and concatenate.  Both halves are normalised to Utf8
    so ``vertical_relaxed`` does not have to promote across numeric /
    string mismatches; ``write_csv`` produces text anyway.
    """
    if not path.exists():
        return new_rows
    existing = _read_csv_keep_columns(path)
    if existing is None or existing.height == 0:
        return new_rows
    cols = new_rows.columns
    keep = [c for c in cols if c in existing.columns]
    if len(keep) != len(cols):
        return new_rows
    existing = existing.select(keep)
    if param_col not in existing.columns:
        return new_rows
    replaced_params = new_rows.select(param_col).unique()
    survivors = (existing.lazy()
                  .join(replaced_params.lazy().with_columns(_drop=pl.lit(True)),
                        on=param_col, how="left")
                  .filter(pl.col("_drop").is_null())
                  .drop("_drop")
                  .collect())
    if survivors.height == 0:
        return new_rows
    new_rows_str = new_rows.with_columns(
        [pl.col(c).cast(pl.Utf8) for c in new_rows.columns]
    )
    return pl.concat([survivors, new_rows_str], how="vertical_relaxed")


def _read_csv_keep_columns(path: Path) -> pl.DataFrame | None:
    """Read a CSV with all columns as Utf8 to preserve original formatting.

    Used by :func:`_merge_param_slice` to avoid coercing existing
    numeric / string entity-level rows to a different dtype during the
    merge.  Returns ``None`` if the file is empty / unreadable.
    """
    try:
        return pl.read_csv(path, infer_schema_length=0)
    except Exception:  # noqa: BLE001 — defensive
        return None


def _write_p_input_sliced(data: "FlexData", inp_dir: Path) -> None:
    """Write wide-by-param ``input/p_*.csv`` files (scalar slices).

    These files carry the full entity-level scalar surface (e.g. for
    nodes: ``penalty_up``, ``existing``, ``lifetime``, …).  The
    override chain only feeds FlexData for a subset (``_P_*_SLICES``),
    so when a legacy ``input/p_<class>.csv`` already exists on disk
    we merge our slices INTO it rather than overwriting — otherwise
    legacy entity-level params would be silently dropped.
    """
    # p_commodity.csv
    rows: list[pl.DataFrame] = []
    for param_value, field in _P_COMMODITY_SLICES.items():
        f = _frame_of(getattr(data, field, None))
        if f is None:
            continue
        rec = (f.pipe(rename_to_axis, {"c": "commodity", "value": "p_commodity"})
                .with_columns(commodityParam=pl.lit(param_value))
                .select("commodity", "commodityParam", "p_commodity"))
        rows.append(rec)
    if rows:
        new_rows = pl.concat(rows, how="vertical_relaxed")
        merged = _merge_param_slice(inp_dir / "p_commodity.csv", new_rows,
                                       entity_col="commodity",
                                       param_col="commodityParam")
        merged.write_csv(inp_dir / "p_commodity.csv")

    rows = []
    for param_value, field in _P_NODE_SLICES.items():
        f = _frame_of(getattr(data, field, None))
        if f is None:
            continue
        rec = (f.pipe(rename_to_axis, {"n": "node", "value": "p_node"})
                .with_columns(nodeParam=pl.lit(param_value))
                .select("node", "nodeParam", "p_node"))
        rows.append(rec)
    if rows:
        new_rows = pl.concat(rows, how="vertical_relaxed")
        merged = _merge_param_slice(inp_dir / "p_node.csv", new_rows,
                                       entity_col="node",
                                       param_col="nodeParam")
        merged.write_csv(inp_dir / "p_node.csv")

    rows = []
    for param_value, field in _P_PROCESS_SLICES.items():
        f = _frame_of(getattr(data, field, None))
        if f is None:
            continue
        rec = (f.pipe(rename_to_axis, {"p": "process", "value": "p_process"})
                .with_columns(processParam=pl.lit(param_value))
                .select("process", "processParam", "p_process"))
        rows.append(rec)
    if rows:
        new_rows = pl.concat(rows, how="vertical_relaxed")
        merged = _merge_param_slice(inp_dir / "p_process.csv", new_rows,
                                       entity_col="process",
                                       param_col="processParam")
        merged.write_csv(inp_dir / "p_process.csv")


# ---------------------------------------------------------------------------
# Metadata copy-through

# These files are read by ``load_flextool`` but are NOT FlexData fields.
# When a round-trip test wants the dumped workdir to reload via the CSV
# path, it copies these from the original workdir verbatim.
_META_FILES_TO_COPY: tuple[tuple[str, str], ...] = (
    # (kind, filename)
    ("solve_data", "solve_current.csv"),
    ("solve_data", "period_first_of_solve.csv"),
    ("solve_data", "period_first.csv"),
    ("solve_data", "period_last.csv"),
    ("solve_data", "period__time_last.csv"),
    ("solve_data", "block_period_time_last.csv"),
    ("solve_data", "step_previous.csv"),
    ("solve_data", "p_years_d.csv"),
    ("solve_data", "p_nested_model.csv"),
    ("solve_data", "fix_storage_timesteps.csv"),
    ("solve_data", "realized_invest_periods_of_current_solve.csv"),
    ("solve_data", "realized_dispatch.csv"),
    ("solve_data", "p_entity_pre_existing.csv"),
    ("solve_data", "entity.csv"),
    ("solve_data", "entityDivest.csv"),
    ("solve_data", "process_side_block.csv"),
    ("solve_data", "entity_block.csv"),
    ("solve_data", "overlap_set.csv"),
    ("solve_data", "block_step_duration.csv"),
    ("solve_data", "period_block_set.csv"),
    ("solve_data", "period_block_succ.csv"),
    ("solve_data", "period_block_time.csv"),
    ("solve_data", "period__branch.csv"),
    ("solve_data", "first_timesteps.csv"),
    ("solve_data", "solve_branch_weight.csv"),
    ("input", "solve_mode.csv"),
    ("input", "p_model.csv"),
    ("input", "p_group.csv"),
    ("input", "pd_group.csv"),
    ("input", "process__ct_method.csv"),
    ("input", "groupIncludeStochastics.csv"),
    ("input", "commodity__node.csv"),
    ("input", "group__node.csv"),
    ("input", "group__entity.csv"),
    ("input", "group__process__node.csv"),
    ("input", "constraint__sense.csv"),
    ("input", "p_constraint_constant.csv"),
    ("input", "p_node_constraint_invested_capacity_coefficient.csv"),
    ("input", "p_process_constraint_invested_capacity_coefficient.csv"),
    ("input", "p_node_constraint_state_coefficient.csv"),
    ("input", "p_node_constraint_cumulative_pre_built_capacity_coefficient.csv"),
    ("input", "p_process_constraint_cumulative_pre_built_capacity_coefficient.csv"),
    ("input", "p_process_node_constraint_flow_coefficient.csv"),
    ("input", "p_process_source_flow_coefficient.csv"),
    ("input", "p_process_sink_flow_coefficient.csv"),
    ("input", "p_entity_unitsize.csv"),
    ("input", "process__source__sink__profile__profile_method.csv"),
    ("input", "node__profile__profile_method.csv"),
    ("input", "node__storage_start_end_method.csv"),
    ("input", "node__storage_solve_horizon_method.csv"),
    ("input", "node__storage_binding_method.csv"),
    ("input", "node__storage_nested_fix_method.csv"),
    ("input", "p_reserve__upDown__group.csv"),
    ("input", "p_process__reserve__upDown__node.csv"),
    ("input", "groupNonSync.csv"),
    ("input", "p_process_sink.csv"),
    ("input", "p_process_source.csv"),
    ("input", "node_dc_power_flow.csv"),
    ("input", "connection_dc_power_flow.csv"),
    ("input", "node_reference_angle.csv"),
    ("input", "p_connection_susceptance.csv"),
    ("input", "commodity_ladder_annual.csv"),
    ("input", "commodity_ladder_cumulative.csv"),
    ("input", "p_commodity_unitsize.csv"),
    # solve_data — additional preprocessed files
    ("solve_data", "process__method_indirect.csv"),
    ("solve_data", "commodity_node_co2.csv"),
    ("solve_data", "group_co2_price.csv"),
    ("solve_data", "p_online_dt_set.csv"),
    ("solve_data", "process_source_sink_ramp_limit_sink_up.csv"),
    ("solve_data", "process_source_sink_ramp_limit_sink_down.csv"),
    ("solve_data", "process_source_sink_ramp_limit_source_up.csv"),
    ("solve_data", "process_source_sink_ramp_limit_source_down.csv"),
    ("solve_data", "process_minload.csv"),
    ("solve_data", "process_online_linear.csv"),
    ("solve_data", "process_online_integer.csv"),
    ("solve_data", "process_online.csv"),
    ("solve_data", "pdt_uptime_set.csv"),
    ("solve_data", "pdt_downtime_set.csv"),
    ("solve_data", "uptime_lookback.csv"),
    ("solve_data", "downtime_lookback.csv"),
    ("solve_data", "ed_invest.csv"),
    ("solve_data", "ed_divest.csv"),
    ("solve_data", "edd_invest.csv"),
    ("solve_data", "ed_invest_period.csv"),
    ("solve_data", "ed_divest_period.csv"),
    ("solve_data", "ed_invest_max_period.csv"),
    ("solve_data", "ed_divest_max_period.csv"),
    ("solve_data", "ed_invest_min_period.csv"),
    ("solve_data", "ed_divest_min_period.csv"),
    ("solve_data", "e_invest_total.csv"),
    ("solve_data", "e_divest_total.csv"),
    ("solve_data", "e_invest_max_total.csv"),
    ("solve_data", "e_divest_max_total.csv"),
    ("solve_data", "e_invest_min_total.csv"),
    ("solve_data", "e_divest_min_total.csv"),
    ("solve_data", "ed_lifetime_fixed_cost.csv"),
    ("solve_data", "ed_lifetime_fixed_cost_divest.csv"),
    ("solve_data", "ed_entity_annual_discounted.csv"),
    ("solve_data", "ed_entity_annual_divest_discounted.csv"),
    ("solve_data", "p_entity_max_units.csv"),
    ("solve_data", "p_entity_period_existing_capacity.csv"),
    ("solve_data", "p_entity_previously_invested_capacity.csv"),
    ("solve_data", "p_entity_invested.csv"),
    ("solve_data", "p_entity_divested.csv"),
    ("solve_data", "ed_invest_forbidden_no_investment.csv"),
    ("solve_data", "ed_invest_cumulative.csv"),
    ("solve_data", "ed_cumulative_max_capacity.csv"),
    ("solve_data", "ed_cumulative_min_capacity.csv"),
    ("solve_data", "g_invest_total.csv"),
    ("solve_data", "g_divest_total.csv"),
    ("solve_data", "g_invest_cumulative.csv"),
    ("solve_data", "gd_invest_period.csv"),
    ("solve_data", "gd_divest_period.csv"),
    ("solve_data", "n_fix_storage_quantity_set.csv"),
    ("solve_data", "fix_storage_quantity.csv"),
    ("solve_data", "p_roll_continue_state.csv"),
    ("solve_data", "process_indirect.csv"),
    ("solve_data", "process_input_flows.csv"),
    ("solve_data", "process_output_flows.csv"),
    ("solve_data", "ed_fixed_cost.csv"),
    ("solve_data", "p_node.csv"),
    ("solve_data", "p_process.csv"),
    ("solve_data", "process_source_delayed.csv"),
    ("solve_data", "process_source_undelayed.csv"),
    ("solve_data", "process_source_sink_delayed.csv"),
    ("solve_data", "process_source_sink_undelayed.csv"),
    ("solve_data", "process_delayed.csv"),
    ("solve_data", "process_delayed__duration.csv"),
    ("solve_data", "p_process_delay_weight.csv"),
    ("solve_data", "dtt__delay_duration.csv"),
    ("solve_data", "commodity_with_ladder.csv"),
    ("solve_data", "commodity_with_ladder_annual.csv"),
    ("solve_data", "commodity_with_ladder_cumulative.csv"),
    ("solve_data", "cnd_ladder_set.csv"),
    ("solve_data", "cndi_ladder_set.csv"),
    ("solve_data", "cndi_ladder_ann_set.csv"),
    ("solve_data", "cndi_ladder_cum_set.csv"),
    ("solve_data", "ci_ladder_cumulative.csv"),
    ("solve_data", "commodity__tier_ann.csv"),
    ("solve_data", "f_d_k.csv"),
    ("solve_data", "ladder_cum_realized_mwh.csv"),
    ("solve_data", "node_capacity_for_scaling.csv"),
    ("solve_data", "group_capacity_for_scaling.csv"),
    ("solve_data", "inv_group_cap.csv"),
    ("solve_data", "group_node.csv"),
    ("solve_data", "group_entity.csv"),
    ("solve_data", "group_process_node.csv"),
    ("solve_data", "process_unit.csv"),
    ("solve_data", "process_sink_inertia.csv"),
    ("solve_data", "process_source_inertia.csv"),
    ("solve_data", "process__sink_nonSync.csv"),
    ("solve_data", "process__group_inside_group_nonSync.csv"),
    ("solve_data", "p_positive_inflow.csv"),
    ("solve_data", "p_negative_inflow.csv"),
    ("solve_data", "pdGroup.csv"),
    ("solve_data", "pdGroup_capacity_margin.csv"),
    ("solve_data", "pdGroup_inertia_limit.csv"),
    ("solve_data", "pdGroup_penalty_capacity_margin.csv"),
    ("solve_data", "pdGroup_penalty_inertia.csv"),
    ("solve_data", "pdGroup_penalty_non_synchronous.csv"),
    ("solve_data", "reserve__upDown__group.csv"),
    ("solve_data", "reserve__upDown__group__method_timeseries.csv"),
    ("solve_data", "reserve__upDown__group__method_dynamic.csv"),
    ("solve_data", "reserve__upDown__group__method_n_1.csv"),
    ("solve_data", "prundt.csv"),
    ("solve_data", "process_reserve_upDown_node_active.csv"),
    ("solve_data", "process_reserve_upDown_node_increase_reserve_ratio.csv"),
    ("solve_data", "process_reserve_upDown_node_large_failure_ratio.csv"),
    ("solve_data", "p_process_reserve_upDown_node_reliability.csv"),
    ("solve_data", "pdtReserve_upDown_group.csv"),
    ("solve_data", "dt_non_anticipativity_set.csv"),
    ("solve_data", "timeline_matching_map.csv"),
    ("solve_data", "fix_storage_quantity.csv"),
    ("solve_data", "edd_history.csv"),
    ("solve_data", "edd_history_invest.csv"),
    ("solve_data", "pssdt_varCost_noEff.csv"),
    ("solve_data", "pssdt_varCost_eff_unit_source.csv"),
    ("solve_data", "pssdt_varCost_eff_unit_sink.csv"),
    ("solve_data", "pssdt_varCost_eff_connection.csv"),
    ("solve_data", "pdtProcess_source.csv"),
    ("solve_data", "pdtProcess_sink.csv"),
)


def _copy_meta(src_workdir: Path, dst_workdir: Path) -> None:
    """Copy metadata files from ``src`` to ``dst``, preserving directory.

    Strategy: copy every ``.csv`` file from ``src/{input,solve_data}/``
    that does NOT already exist in ``dst/{input,solve_data}/``.  The
    direct-write pass has already populated the FlexData-derived files;
    this pass fills in metadata + structural files (timeline, period
    markers, method discriminators, header-only filter sets, etc.) that
    the CSV reader needs but that aren't FlexData fields.

    The intent is that the dumped tree is a *complete* workdir that
    ``load_flextool`` can consume — the CSV reader's many file-existence
    probes don't crash because every CSV in the source is mirrored.
    """
    for kind in ("input", "solve_data"):
        src_dir = src_workdir / kind
        dst_dir = dst_workdir / kind
        if not src_dir.is_dir():
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.csv"):
            dst = dst_dir / src.name
            if dst.exists():
                continue   # direct-write or earlier copy already populated
            shutil.copy(src, dst)
