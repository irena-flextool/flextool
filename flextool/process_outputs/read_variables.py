"""Read solver variable outputs into a namespace.

Per-solve parquets in ``output_raw/<var>__{solve}.parquet``, written by
:mod:`flextool.process_outputs.read_highs_solution`.  The parquets carry
full MultiIndex information in compact ``flextool`` metadata (see
``lean_parquet``).

Item 163: the legacy ``output_raw/*.csv`` pathway (``--use-old-raw-csv``)
is gone — the cascade's output writers (:mod:`read_highs_solution`) only
ever write the per-solve parquets, so a work folder with no
``v_flow__*.parquet`` is an error, not a prompt to probe for a stale
``v_flow.csv``.
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


def read_variables(output_dir, *, group_entity_invest: pd.DataFrame):
    """Read all variable outputs into a :class:`SimpleNamespace`.

    ``output_dir`` is the ``output_raw/`` path.  Raises ``FileNotFoundError``
    when it holds no per-solve variable parquet (item 163: there is no
    CSV fallback to probe for any more).

    ``group_entity_invest`` (item 163 addendum) is the caller-supplied
    ``(group, entity)`` frame — the sub-solve's FlexDataProvider never
    reads ``input/group_entity_invest.csv`` from disk, because no
    normal run writes it (see :func:`flextool.process_outputs.write_outputs._group_entity_invest`).
    It is set on the returned namespace.
    """
    output_path = Path(output_dir)

    if not any(output_path.glob("v_flow__*.parquet")):
        raise FileNotFoundError(
            f"no per-solve parquet outputs in {output_path}; the cascade "
            f"writes output_raw/<var>__<solve>.parquet"
        )
    v = _read_from_parquet(output_path)
    v.group_entity_invest = group_entity_invest
    return v


def _unlink_with_retry(path: Path, *, attempts: int = 40, delay: float = 0.05) -> None:
    """Delete ``path``, retrying transient Windows sharing errors.

    On Windows a file that was just written and read back can stay briefly
    locked (an antivirus / indexer handle opened without ``FILE_SHARE_DELETE``),
    so ``unlink`` raises ``PermissionError`` (WinError 32) and a one-shot
    best-effort delete left the ``.<var>__concat_*.parquet`` temp in
    ``output_raw/``.  POSIX never hits this, so the first attempt succeeds."""
    import time

    for i in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if i == attempts - 1:
                import logging
                logging.getLogger(__name__).warning(
                    "could not remove temp file %s (still locked)", path,
                )
                return
            time.sleep(delay)
        except OSError:
            return


def _stream_concat_parts(parts: list[Path], *, zero_fill: bool = True) -> pd.DataFrame:
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
        # Never let a stray temp file accumulate in ``output_raw/``.
        _unlink_with_retry(tmp_path)

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

    # Prices keep their empty cells (a step the objective does not weigh has no price): ``zero_fill=False``.
    return df.fillna(0.0) if zero_fill else df


def _read_from_parquet(parquet_dir: Path) -> SimpleNamespace:
    """Concat per-solve parquets for each variable.

    Per-solve parquets are concatenated in solve **creation** order, so
    downstream ``DataFrame.mul(axis=1, level=0)`` aligns operands by row
    index with no post-concat sort needed.
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
        zero_fill: bool = True,
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
        return _stream_concat_parts(parts, zero_fill=zero_fill)

    v.obj = _read("v_obj", ("objective",), has_period=False, has_time=False)
    v.flow = _read("v_flow", ("process", "source", "sink"))
    # Reverse-flow auxiliary for method_2way_1var_off arcs (empty frame
    # when no such arc is present).  Folded into the signed net flow by
    # calc_capacity_flows.
    v.flow_back = _read("v_flow_back", ("process", "source", "sink"))
    # The ramp-cost split of the flow change (empty frames without a ramp cost); read by calc_costs.
    v.ramp_up = _read("v_ramp_up", ("process", "source", "sink"))
    v.ramp_down = _read("v_ramp_down", ("process", "source", "sink"))
    v.reserve = _read("v_reserve", ("process", "reserve", "updown", "node"))
    v.state = _read("v_state", ("node",))
    # Self-discharge per step (MWh) as the balance rows used it (engine writer write_v_self_discharge); empty
    # frame without the parquet.
    v.self_discharge = _read("v_self_discharge", ("node",))
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
    # The branch-weighted investment of a stochastic fan's branch periods at its label (recourse; empty frame
    # without the parquet): the reported investment costs inside a fan.
    v.invest_fan = _read("v_invest_fan", ("entity",), has_time=False)
    # Commodity price ladder (#57): tier purchase, period-only (no time).  Column order matches the
    # VariableSpec's ``col_names + trailing_col_names`` (commodity, node, tier).
    v.trade = _read("v_trade", ("commodity", "node", "tier"), has_time=False)
    v.dual_node_balance = _read("v_dual_node_balance", ("node",), zero_fill=False)
    v.dual_reserve_balance = _read(
        "v_dual_reserve__upDown__group__period__t",
        ("reserve", "updown", "node_group"),
        zero_fill=False,
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
    # The capacity margin of each solve with investment periods (engine.output._capacity_margin): long polars
    # frames, one solve after the other with no dedup (an invest solve keeps its rows under its own name); None
    # without a capacity margin.
    from flextool.engine.output._capacity_margin import ENTITY_PREFIX, REPORT_PREFIX, read_capacity_margin_parts

    v.capacity_margin_report = read_capacity_margin_parts(parquet_dir, REPORT_PREFIX, solve_order)
    v.capacity_margin_entity = read_capacity_margin_parts(parquet_dir, ENTITY_PREFIX, solve_order)
    # The reserve requirement of each reserve group (engine.output._reserve_report): long polars frames over the
    # realized steps, a step from the last solve that realizes it; None without a reserve requirement.
    from flextool.engine.output._reserve_report import PROVIDER_PREFIX, REPORT_PREFIX as RESERVE_PREFIX
    from flextool.engine.output._reserve_report import read_reserve_report_parts

    v.reserve_group = read_reserve_report_parts(parquet_dir, RESERVE_PREFIX, solve_order)
    v.reserve_group_provider = read_reserve_report_parts(parquet_dir, PROVIDER_PREFIX, solve_order)

    # ``group_entity_invest`` is set by the caller (``read_variables``) from the
    # sub-solve's FlexDataProvider — item 163 addendum; no polars producer
    # writes ``input/group_entity_invest.csv``, so a disk read here was always
    # empty on a fresh run.
    return v
