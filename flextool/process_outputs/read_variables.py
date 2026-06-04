"""Read solver variable outputs into a namespace.

Supports two pathways:

* **CSV** — ``output_raw/*.csv`` (legacy path).  Kept alive for
  ``--use-old-raw-csv`` runs.
* **Parquet** — per-solve parquets in
  ``output_raw/<var>__{solve}.parquet`` written by
  :mod:`flextool.process_outputs.read_highs_solution`.  The parquets
  carry full MultiIndex information in compact ``flextool`` metadata,
  so they round-trip to the same DataFrame shape the CSV path
  produces (see ``lean_parquet``).

Both pathways use the same ``output_raw/`` folder.  Dispatch is by
file presence: if any ``v_flow__*.parquet`` exists and no
``v_flow.csv`` does, the parquet pathway is used; otherwise the CSV
pathway.
"""
import ast
import json
import os
import tempfile
from types import SimpleNamespace
from pathlib import Path
from typing import Sequence
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flextool.process_outputs.read_highs_solution import empty_variable_frame
from flextool.process_outputs.solve_order import load_solve_order


def read_variables(output_dir):
    """Read all variable outputs into a :class:`SimpleNamespace`.

    ``output_dir`` is the ``output_raw/`` path.  ``input/`` is resolved
    as a sibling folder (needed for ``group_entity_invest.csv``).
    """
    output_path = Path(output_dir)
    work_folder = output_path.parent
    input_path = work_folder / "input"

    # Prefer the CSV pathway when its output is present.  Fall back to
    # the per-solve parquets when ``v_flow.csv`` is absent.
    if (output_path / "v_flow.csv").exists():
        return _read_from_csv(output_path, input_path)
    if any(output_path.glob("v_flow__*.parquet")):
        return _read_from_parquet(output_path, input_path)
    return _read_from_csv(output_path, input_path)


def _stream_concat_parts(parts: list[Path]) -> pd.DataFrame:
    """Concatenate per-solve parquet *parts* without materialising them
    all at once.

    The previous implementation built ``frames = [read_lean_parquet(p)
    for p in parts]`` then ``pd.concat(frames).fillna(0.0)``.  On 72-roll
    SouthAfrica that peaked at ~50 GB for ``v_flow`` alone — the Python
    list held every per-solve frame simultaneously, then ``pd.concat``
    produced a second dense copy of the union.

    Phase E2 strategy:

    1. Footer-only pass over each part to gather the union of physical
       column names, the per-column canonical type (preferring concrete
       types over ``null`` from empty parts), and the row-index level
       info from the first non-empty part's ``flextool`` metadata.
    2. Open a single ``ParquetWriter`` on a temp file with the union
       schema.  For each part read its table, promote (add nulls / drop)
       columns to align with the union schema, and write it as one
       row-group.  Drop the table reference before reading the next
       part so we never hold more than ONE part's arrow data in memory
       during the write phase.
    3. Read the consolidated temp parquet back as a single pandas
       DataFrame via ``pq.read_table(...).to_pandas(self_destruct=True)``.
       arrow buffers are released column-by-column during the
       conversion.
    4. Apply the historical fixups (1-level MultiIndex collapse +
       ``.fillna(0.0)``).  The zero-fill matters for downstream
       consumers (e.g. ``calc_storage_vre`` uses
       ``v.state[n].reindex(...)``; a NaN there would break the
       arithmetic).

    Memory profile (per variable), where ``F = N × per_part`` is the
    union frame size:

        old : peak ≈ N × per_part + 2 × F   (Python list of N frames
                                             + ``pd.concat`` temp + final)
        new : peak ≈ max(2 × per_part,      (write phase: one part
                         2 × F)              + writer buffer)
                                            (read phase: arrow table
                                             + pandas frame during
                                             column-by-column convert)

    Concretely: the spec-cited 50 GB SouthAfrica peak (dominated by
    the N-frame list term) drops to roughly ``2 × F`` — the read-back
    floor.  When ``F ≈ N × per_part`` (parts share most columns) the
    win is ~halving peak; when parts share *no* columns (each part
    contributes disjoint entities) ``F`` shrinks proportionally and
    the win grows.

    The temp file lives in the same directory as the parts so it stays
    on the workspace filesystem (predictable disk locality, no
    cross-filesystem rename surprises).
    """
    # Filter out parts that wrote zero rows AND zero columns.  The old
    # path kept these only if they were the *only* parts; we mirror
    # that fallback here so an all-empty variable still returns the
    # same shape it did before.
    non_empty_parts: list[Path] = []
    for p in parts:
        pf = pq.ParquetFile(str(p))
        if pf.metadata.num_rows == 0 and pf.metadata.num_columns == 0:
            continue
        non_empty_parts.append(p)
    use_parts = non_empty_parts or parts

    if not use_parts:
        # Defensive — caller guarantees ``parts`` is non-empty.
        return pd.DataFrame()

    # Pass 1: union of columns + per-column canonical type + row-index
    # level info from the first non-empty part's metadata.  Reading
    # just the schema costs one footer read per file (cheap; KB not GB).
    union_cols: list[str] = []
    seen: set[str] = set()
    flextool_info: bytes | None = None
    null_type = pa.null()
    name_to_type: dict[str, pa.DataType] = {}
    for p in use_parts:
        pf = pq.ParquetFile(str(p))
        sch = pf.schema_arrow
        meta = sch.metadata or {}
        if flextool_info is None and b"flextool" in meta:
            flextool_info = meta[b"flextool"]
        for field in sch:
            if field.name not in seen:
                seen.add(field.name)
                union_cols.append(field.name)
                name_to_type[field.name] = field.type
            else:
                # Prefer any concrete type over ``null`` — a part with
                # zero rows may emit a column of ``null`` type, and
                # arrow can't cast a later concrete column *into* that.
                if (
                    name_to_type[field.name] == null_type
                    and field.type != null_type
                ):
                    name_to_type[field.name] = field.type

    union_fields = [pa.field(c, name_to_type[c]) for c in union_cols]
    # Preserve flextool metadata so the temp file is readable via
    # ``read_lean_parquet`` (round-trips the index / columns names).
    schema_meta = {b"flextool": flextool_info} if flextool_info else None
    union_schema = pa.schema(union_fields, metadata=schema_meta)

    # Pass 2: stream-write each part as a single row-group.
    parent = use_parts[0].parent
    # ``mkstemp`` creates the file with O_EXCL — no collision with
    # concurrent runs writing into the same folder.
    fd, tmp_path_str = tempfile.mkstemp(
        prefix=f".{use_parts[0].stem.split('__')[0]}__concat_", suffix=".parquet",
        dir=str(parent),
    )
    os.close(fd)  # ParquetWriter will reopen by path.
    tmp_path = Path(tmp_path_str)
    try:
        with pq.ParquetWriter(str(tmp_path), union_schema) as writer:
            for p in use_parts:
                table = pq.read_table(str(p))
                # Strip per-part metadata so the writer doesn't reject
                # the row-group on schema mismatch — only the file-level
                # ``union_schema`` carries the canonical ``flextool``
                # blob.
                table = table.replace_schema_metadata(None)
                # Align the part's columns to ``union_cols`` order,
                # filling absent columns with a null array (cheap —
                # arrow stores a single null reference, not a dense
                # ndarray).
                n_rows = table.num_rows
                arrays: list[pa.ChunkedArray | pa.Array] = []
                part_cols = {f.name: i for i, f in enumerate(table.schema)}
                for col_name in union_cols:
                    if col_name in part_cols:
                        arr = table.column(part_cols[col_name])
                        target_type = name_to_type[col_name]
                        if arr.type != target_type:
                            if arr.type == null_type:
                                # All-null part column → emit a properly
                                # typed null array of the same length.
                                arr = pa.nulls(arr.length(), type=target_type)
                            else:
                                # Concrete → target (e.g. int32 → int64).
                                arr = arr.cast(target_type)
                        arrays.append(arr)
                    else:
                        arrays.append(
                            pa.nulls(n_rows, type=name_to_type[col_name])
                        )
                aligned = pa.Table.from_arrays(
                    arrays, schema=union_schema,
                )
                writer.write_table(aligned)
                # Free part-level references before reading the next.
                del table, aligned, arrays

        # Pass 3: single read-back as pandas.  ``self_destruct=True``
        # discards arrow buffers as columns are materialised, so peak
        # during this conversion is one buffer + the growing pandas
        # frame, not two complete copies.  The temp file's row-group
        # layout is preserved (one per part) — multi-row-group reads
        # are still single-frame from the consumer's perspective.
        result_table = pq.read_table(str(tmp_path))
        df = result_table.to_pandas(self_destruct=True)
        del result_table
    finally:
        # Best-effort cleanup — never let a stray temp file accumulate.
        try:
            tmp_path.unlink()
        except OSError:
            pass

    # --- Rebuild row index from the flextool metadata ---
    # ``read_lean_parquet`` does this for a single file; we replicate
    # the logic against the union schema's metadata so the streaming
    # writer is interchangeable with the old eager path.
    if flextool_info is not None:
        info = json.loads(flextool_info)
        idx_names = info.get("idx")
        if idx_names:
            col_lookup: list[str] = []
            for i, lname in enumerate(idx_names):
                col_lookup.append(
                    f"__index_level_{i}__" if lname is None else lname
                )
            present = [c for c in col_lookup if c in df.columns]
            if present:
                df = df.set_index(present)
                if isinstance(df.index, pd.MultiIndex):
                    df.index.names = idx_names
                else:
                    df.index.name = idx_names[0]

        col_level_names = info.get("col")
        if col_level_names:
            tuples = [ast.literal_eval(c) for c in df.columns]
            df.columns = pd.MultiIndex.from_tuples(
                tuples, names=col_level_names,
            )
        else:
            single_col_name = info.get("col_name")
            if single_col_name is not None:
                df.columns.name = single_col_name

    # ``astype(float)`` matched the old behaviour — values are already
    # float64 from the writer side, but this preserves the historical
    # promotion for any oddly-typed (e.g. int-zero) all-null columns.
    df = df.astype(float)

    # Collapse 1-level MultiIndex columns to plain Index — same fixup
    # the old eager path applied (the CSV pathway produces a plain
    # Index here; downstream ``DataFrame.mul(..., level=0)`` treats
    # MultiIndex-with-one-level differently).
    if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 1:
        df.columns = pd.Index(
            [c[0] for c in df.columns], name=df.columns.names[0],
        )

    return df.fillna(0.0)


def _read_from_parquet(parquet_dir: Path, input_path: Path) -> SimpleNamespace:
    """New pathway: concat per-solve parquets for each variable.

    Per-solve parquets are concatenated in solve **creation** order —
    the same order the CSV pathway sees rows appended by phase-1
    printfs.  Reading in creation order avoids any post-concat sort;
    downstream ``DataFrame.mul(axis=1, level=0)`` aligns operands by
    row index, and both reader pathways producing the same row order
    is enough.
    """
    v = SimpleNamespace()
    work_folder = parquet_dir.parent
    solve_order = load_solve_order(work_folder)

    def _solve_from_filename(name: str, path: Path) -> str:
        # ``<output_name>__<solve>.parquet`` — strip prefix + extension.
        return path.name[len(name) + 2:-len(".parquet")]

    def _read(
        name: str,
        col_names: Sequence[str],
        *,
        has_period: bool = True,
        has_time: bool = True,
    ) -> pd.DataFrame:
        # Parquet carries full float64 precision; no rounding — that
        # belongs only at the final user-facing CSV write boundary.
        #
        # When no parquet exists (writer was never called for this variable),
        # fall back to a typed empty frame via ``empty_variable_frame`` so
        # downstream calc code still sees the expected ``columns.name`` /
        # row-index ``names``.
        parts = list(parquet_dir.glob(f"{name}__*.parquet"))
        if not parts:
            return empty_variable_frame(
                solve_name="", col_names=col_names,
                has_period=has_period, has_time=has_time,
            )
        # Order per-solve parts by solve creation order so the concat
        # result is already in the correct row order — no post-concat
        # sort needed.  Solves missing from ``solve_order`` sort first
        # (shouldn't happen normally; defensive).
        parts.sort(key=lambda p: solve_order.get(_solve_from_filename(name, p), -1))
        # Phase E2 — stream per-solve parts through pyarrow as one
        # row-group each, so we never hold all frames in a Python list
        # nor pay for ``pd.concat``'s second-copy.  Reader contract is
        # unchanged: the public return is still a ``pd.DataFrame`` with
        # the same row order, column union, MultiIndex/columns-name
        # semantics, and zero-filled missing cells.
        return _stream_concat_parts(parts)

    v.obj = _read("v_obj", ("objective",), has_period=False, has_time=False)
    v.flow = _read("v_flow", ("process", "source", "sink"))
    # Reverse-flow auxiliary for method_2way_1var_off arcs (empty frame
    # when no such arc is present).  Folded into the signed net flow by
    # calc_capacity_flows.
    v.flow_back = _read("v_flow_back", ("process", "source", "sink"))
    v.ramp = _read("v_ramp", ("process", "source", "sink"))
    v.reserve = _read("v_reserve", ("process", "reserve", "updown", "node"))
    v.state = _read("v_state", ("node",))
    v.online_linear = _read("v_online_linear", ("process",))
    v.startup_linear = _read("v_startup_linear", ("process",))
    v.shutdown_linear = _read("v_shutdown_linear", ("process",))
    v.online_integer = _read("v_online_integer", ("process",))
    v.startup_integer = _read("v_startup_integer", ("process",))
    v.shutdown_integer = _read("v_shutdown_integer", ("process",))
    v.q_state_up = _read("vq_state_up", ("node",))
    v.q_state_down = _read("vq_state_down", ("node",))
    v.q_reserve = _read("vq_reserve", ("reserve", "updown", "node_group"))
    v.q_inertia = _read("vq_inertia", ("group",))
    v.q_non_synchronous = _read("vq_non_synchronous", ("group",))
    v.q_state_up_group = _read("vq_state_up_group", ("group",))
    v.q_capacity_margin = _read("vq_capacity_margin", ("group",), has_time=False)
    v.invest = _read("v_invest", ("entity",), has_time=False)
    v.divest = _read("v_divest", ("entity",), has_time=False)
    v.dual_node_balance = _read("v_dual_node_balance", ("node",))
    v.dual_reserve_balance = _read(
        "v_dual_reserve__upDown__group__period__t",
        ("reserve", "updown", "node_group"),
    )
    v.angle = _read("v_angle", ("node",))
    v.dual_invest_unit = _read("v_dual_invest_unit", ("unit",), has_time=False)
    v.dual_invest_connection = _read("v_dual_invest_connection", ("connection",), has_time=False)
    v.dual_invest_node = _read("v_dual_invest_node", ("node",), has_time=False)
    v.dual_maxInvest_period = _read("v_dual_maxInvest_period", ("entity",), has_time=False)
    v.dual_maxInvest_total = _read("v_dual_maxInvest_total", ("entity",), has_time=False)
    v.dual_maxCumulative = _read("v_dual_maxCumulative", ("entity",), has_time=False)
    v.dual_maxInvestGroup_period = _read("v_dual_maxInvestGroup_period", ("group",), has_time=False)
    v.dual_maxInvestGroup_total = _read("v_dual_maxInvestGroup_total", ("group",), has_time=False)
    v.dual_maxInvestGroup_cumulative = _read("v_dual_maxInvestGroup_cumulative", ("group",), has_time=False)
    # Investment-floor (min-side) duals — mirror the max-side readers above.
    v.dual_minInvest_period = _read("v_dual_minInvest_period", ("entity",), has_time=False)
    v.dual_minInvest_total = _read("v_dual_minInvest_total", ("entity",), has_time=False)
    v.dual_minCumulative = _read("v_dual_minCumulative", ("entity",), has_time=False)
    v.dual_minInvestGroup_period = _read("v_dual_minInvestGroup_period", ("group",), has_time=False)
    v.dual_minInvestGroup_total = _read("v_dual_minInvestGroup_total", ("group",), has_time=False)
    v.dual_minInvestGroup_cumulative = _read("v_dual_minInvestGroup_cumulative", ("group",), has_time=False)
    v.dual_co2_max_period = _read("v_dual_co2_max_period", ("group",), has_time=False)
    v.dual_co2_max_total = _read("v_dual_co2_max_total", ("group",), has_period=False, has_time=False)

    # ``group_entity_invest`` is a static map (solveFirst-only) written
    # during phase 1 directly to ``input/`` — same as the CSV pathway.
    # Δ.31: tolerate absence (the polars cascade doesn't always emit
    # this file; downstream code only consumes it for the
    # ``maxInvestGroup`` family which is empty without group invest).
    gei_path = input_path / "group_entity_invest.csv"
    if gei_path.exists():
        v.group_entity_invest = pd.read_csv(gei_path)
    else:
        v.group_entity_invest = pd.DataFrame(columns=["group", "entity"])
    return v


def _read_from_csv(output_path: Path, input_path: Path) -> SimpleNamespace:
    """CSV pathway: read ``output_raw/<var>.csv`` files."""
    v = SimpleNamespace()

    # Variables with (solve, period, time) index
    v.obj = pd.read_csv(output_path / 'v_obj.csv', header=[0], index_col=[0]).astype(float)
    v.flow = pd.read_csv(output_path / 'v_flow.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    # Reverse-flow auxiliary for method_2way_1var_off arcs.  The legacy CSV
    # writer only emits this file when the variable exists; mirror v_flow's
    # shape with an empty frame otherwise.
    _flow_back_csv = output_path / 'v_flow_back.csv'
    if _flow_back_csv.exists():
        v.flow_back = pd.read_csv(_flow_back_csv, header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    else:
        v.flow_back = pd.DataFrame(index=v.flow.index, columns=v.flow.columns[:0])
    v.ramp = pd.read_csv(output_path / 'v_ramp.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    v.reserve = pd.read_csv(output_path / 'v_reserve.csv', header=[0, 1, 2, 3], index_col=[0, 1, 2]).astype(float)
    v.state = pd.read_csv(output_path / 'v_state.csv', index_col=[0, 1, 2]).astype(float)
    v.online_linear = pd.read_csv(output_path / 'v_online_linear.csv', index_col=[0, 1, 2]).astype(float)
    v.startup_linear = pd.read_csv(output_path / 'v_startup_linear.csv', index_col=[0, 1, 2]).astype(float)
    v.shutdown_linear = pd.read_csv(output_path / 'v_shutdown_linear.csv', index_col=[0, 1, 2]).astype(float)
    v.online_integer = pd.read_csv(output_path / 'v_online_integer.csv', index_col=[0, 1, 2]).astype(float)
    v.startup_integer = pd.read_csv(output_path / 'v_startup_integer.csv', index_col=[0, 1, 2]).astype(float)
    v.shutdown_integer = pd.read_csv(output_path / 'v_shutdown_integer.csv', index_col=[0, 1, 2]).astype(float)
    v.q_state_up = pd.read_csv(output_path / 'vq_state_up.csv', index_col=[0, 1, 2]).astype(float)
    v.q_state_down = pd.read_csv(output_path / 'vq_state_down.csv', index_col=[0, 1, 2]).astype(float)
    v.q_reserve = pd.read_csv(output_path / 'vq_reserve.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
    v.q_inertia = pd.read_csv(output_path / 'vq_inertia.csv', index_col=[0, 1, 2]).astype(float)
    v.q_non_synchronous = pd.read_csv(output_path / 'vq_non_synchronous.csv', index_col=[0, 1, 2]).astype(float)
    v.q_state_up_group = pd.read_csv(output_path / 'vq_state_up_group.csv', index_col=[0, 1, 2]).astype(float)
    v.q_capacity_margin = pd.read_csv(output_path / 'vq_capacity_margin.csv', index_col=[0, 1]).astype(float)
    v.invest = pd.read_csv(output_path / 'v_invest.csv', index_col=[0, 1]).astype(float)
    v.divest = pd.read_csv(output_path / 'v_divest.csv', index_col=[0, 1]).astype(float)
    v.dual_node_balance = pd.read_csv(output_path / 'v_dual_node_balance.csv', index_col=[0, 1, 2]).astype(float)
    v.dual_reserve_balance = pd.read_csv(output_path / 'v_dual_reserve__upDown__group__period__t.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)

    # DC power flow voltage angles (may be empty when no DC PF nodes exist)
    angle_path = output_path / 'v_angle.csv'
    if angle_path.exists():
        v.angle = pd.read_csv(angle_path, index_col=[0, 1, 2]).astype(float)
        if v.angle.empty:
            v.angle = pd.DataFrame()
        else:
            v.angle.index.names = ['solve', 'period', 'time']
            v.angle.columns.name = 'node'
    else:
        v.angle = pd.DataFrame()

    v.dual_invest_unit = pd.read_csv(output_path / 'v_dual_invest_unit.csv', index_col=[0, 1]).astype(float)
    v.dual_invest_connection = pd.read_csv(output_path / 'v_dual_invest_connection.csv', index_col=[0, 1]).astype(float)
    v.dual_invest_node = pd.read_csv(output_path / 'v_dual_invest_node.csv', index_col=[0, 1]).astype(float)
    # Investment constraint duals (per MW, unscaled)
    v.dual_maxInvest_period = pd.read_csv(output_path / 'v_dual_maxInvest_period.csv', index_col=[0, 1]).astype(float)
    v.dual_maxInvest_total = pd.read_csv(output_path / 'v_dual_maxInvest_total.csv', index_col=[0, 1]).astype(float)
    v.dual_maxCumulative = pd.read_csv(output_path / 'v_dual_maxCumulative.csv', index_col=[0, 1]).astype(float)
    v.dual_maxInvestGroup_period = pd.read_csv(output_path / 'v_dual_maxInvestGroup_period.csv', index_col=[0, 1]).astype(float)
    v.dual_maxInvestGroup_total = pd.read_csv(output_path / 'v_dual_maxInvestGroup_total.csv', index_col=[0, 1]).astype(float)
    v.dual_maxInvestGroup_cumulative = pd.read_csv(output_path / 'v_dual_maxInvestGroup_cumulative.csv', index_col=[0, 1]).astype(float)
    # Investment-floor (min-side) duals (per MW, unscaled)
    v.dual_minInvest_period = pd.read_csv(output_path / 'v_dual_minInvest_period.csv', index_col=[0, 1]).astype(float)
    v.dual_minInvest_total = pd.read_csv(output_path / 'v_dual_minInvest_total.csv', index_col=[0, 1]).astype(float)
    v.dual_minCumulative = pd.read_csv(output_path / 'v_dual_minCumulative.csv', index_col=[0, 1]).astype(float)
    v.dual_minInvestGroup_period = pd.read_csv(output_path / 'v_dual_minInvestGroup_period.csv', index_col=[0, 1]).astype(float)
    v.dual_minInvestGroup_total = pd.read_csv(output_path / 'v_dual_minInvestGroup_total.csv', index_col=[0, 1]).astype(float)
    v.dual_minInvestGroup_cumulative = pd.read_csv(output_path / 'v_dual_minInvestGroup_cumulative.csv', index_col=[0, 1]).astype(float)
    # CO2 emission-cap duals (raw dual is per /1000-scaled RHS; downstream * 1000 for tCO2)
    v.dual_co2_max_period = pd.read_csv(output_path / 'v_dual_co2_max_period.csv', index_col=[0, 1]).astype(float)
    v.dual_co2_max_total = pd.read_csv(output_path / 'v_dual_co2_max_total.csv', index_col=[0]).astype(float)
    # group_entity_invest lives in input/ (solveFirst-gated static).
    v.group_entity_invest = pd.read_csv(input_path / 'group_entity_invest.csv')

    v.flow.index.names = ['solve', 'period', 'time']
    v.ramp.index.names = ['solve', 'period', 'time']
    v.reserve.index.names = ['solve', 'period', 'time']
    v.state.index.names = ['solve', 'period', 'time']
    v.online_linear.index.names = ['solve', 'period', 'time']
    v.startup_linear.index.names = ['solve', 'period', 'time']
    v.shutdown_linear.index.names = ['solve', 'period', 'time']
    v.online_integer.index.names = ['solve', 'period', 'time']
    v.startup_integer.index.names = ['solve', 'period', 'time']
    v.shutdown_integer.index.names = ['solve', 'period', 'time']
    v.q_state_up.index.names = ['solve', 'period', 'time']
    v.q_state_down.index.names = ['solve', 'period', 'time']
    v.q_reserve.index.names = ['solve', 'period', 'time']
    v.q_inertia.index.names = ['solve', 'period', 'time']
    v.q_non_synchronous.index.names = ['solve', 'period', 'time']
    v.q_state_up_group.index.names = ['solve', 'period', 'time']
    v.q_capacity_margin.index.names = ['solve', 'period']
    v.invest.index.names = ['solve', 'period']
    v.divest.index.names = ['solve', 'period']
    v.dual_node_balance.index.names = ['solve', 'period', 'time']
    v.dual_reserve_balance.index.names = ['solve', 'period', 'time']
    v.dual_invest_unit.index.names = ['solve', 'period']
    v.dual_invest_connection.index.names = ['solve', 'period']
    v.dual_invest_node.index.names = ['solve', 'period']
    v.dual_maxInvest_period.index.names = ['solve', 'period']
    v.dual_maxInvest_total.index.names = ['solve', 'period']
    v.dual_maxCumulative.index.names = ['solve', 'period']
    v.dual_maxInvestGroup_period.index.names = ['solve', 'period']
    v.dual_maxInvestGroup_total.index.names = ['solve', 'period']
    v.dual_maxInvestGroup_cumulative.index.names = ['solve', 'period']
    v.dual_minInvest_period.index.names = ['solve', 'period']
    v.dual_minInvest_total.index.names = ['solve', 'period']
    v.dual_minCumulative.index.names = ['solve', 'period']
    v.dual_minInvestGroup_period.index.names = ['solve', 'period']
    v.dual_minInvestGroup_total.index.names = ['solve', 'period']
    v.dual_minInvestGroup_cumulative.index.names = ['solve', 'period']
    v.dual_co2_max_period.index.names = ['solve', 'period']
    v.dual_co2_max_total.index.name = 'solve'

    # Create multi-index for variables with single header row
    v.state.columns.name = 'node'
    v.online_linear.columns.name = 'process'
    v.startup_linear.columns.name = 'process'
    v.shutdown_linear.columns.name = 'process'
    v.online_integer.columns.name = 'process'
    v.startup_integer.columns.name = 'process'
    v.shutdown_integer.columns.name = 'process'
    v.q_state_up.columns.name = 'node'
    v.q_state_down.columns.name = 'node'
    v.q_inertia.columns.name = 'group'
    v.q_non_synchronous.columns.name = 'group'
    v.q_state_up_group.columns.name = 'group'
    v.q_capacity_margin.columns.name = 'group'
    v.invest.columns.name = 'entity'
    v.divest.columns.name = 'entity'
    v.dual_node_balance.columns.name = 'node'
    v.dual_invest_unit.columns.name = 'unit'
    v.dual_invest_connection.columns.name = 'connection'
    v.dual_invest_node.columns.name = 'node'
    v.dual_maxInvest_period.columns.name = 'entity'
    v.dual_maxInvest_total.columns.name = 'entity'
    v.dual_maxCumulative.columns.name = 'entity'
    v.dual_maxInvestGroup_period.columns.name = 'group'
    v.dual_maxInvestGroup_total.columns.name = 'group'
    v.dual_maxInvestGroup_cumulative.columns.name = 'group'
    v.dual_minInvest_period.columns.name = 'entity'
    v.dual_minInvest_total.columns.name = 'entity'
    v.dual_minCumulative.columns.name = 'entity'
    v.dual_minInvestGroup_period.columns.name = 'group'
    v.dual_minInvestGroup_total.columns.name = 'group'
    v.dual_minInvestGroup_cumulative.columns.name = 'group'
    v.dual_co2_max_period.columns.name = 'group'
    v.dual_co2_max_total.columns.name = 'group'

    # Add multi-index to variables with multiple header rows (this multi-index creation works also when the dataframe is empty)
    v.flow.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.flow.columns],
        names=['process', 'source', 'sink']
    )
    v.ramp.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.ramp.columns],
        names=['process', 'source', 'sink']
    )
    v.reserve.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2], col[3]) for col in v.reserve.columns],
        names=['process', 'reserve', 'updown', 'node']
    )
    v.q_reserve.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.q_reserve.columns],
        names=['reserve', 'updown', 'node_group']
    )
    v.dual_reserve_balance.columns = pd.MultiIndex.from_tuples(
        [(col[0], col[1], col[2]) for col in v.dual_reserve_balance.columns],
        names=['reserve', 'updown', 'node_group']
    )

    return v
