"""Read solver variable outputs into a namespace.

Supports two pathways:

* **Legacy** — ``output_raw/*.csv`` written by glpsol phase 3.  Kept
  alive for ``--use-old-raw-csv`` runs.
* **New** — per-solve parquets in ``output_raw/<var>__{solve}.parquet``
  written by :mod:`flextool.process_outputs.read_highs_solution`.  The
  parquets carry full MultiIndex information in compact ``flextool``
  metadata, so they round-trip to the same DataFrame shape the legacy
  CSV path produces (see ``lean_parquet``).

Both pathways use the same ``output_raw/`` folder — parquets coexist
with (or replace) the phase-3 CSVs.  Dispatch is by file presence: if
any ``v_flow__*.parquet`` exists, the parquet pathway is used;
otherwise the CSV pathway.
"""
from types import SimpleNamespace
from pathlib import Path
import pandas as pd

from flextool.lean_parquet import read_lean_parquet


def _round_to_sig(series: pd.Series, sig: int) -> pd.Series:
    """Round ``series`` to ``sig`` significant digits.

    Mimics the ``%.Ng`` formatting the phase-3 CSV writer applies to each
    cell — needed so the parquet pathway produces the same numeric values
    downstream tooling sees, not an extra trailing digit or two of
    float64 precision.
    """
    import numpy as np
    arr = series.to_numpy(dtype=float, copy=True)
    finite = np.isfinite(arr) & (arr != 0)
    if not finite.any():
        return series
    # %g rounds to N significant digits (away from zero).
    magnitude = np.zeros_like(arr)
    magnitude[finite] = np.floor(np.log10(np.abs(arr[finite])))
    factor = 10.0 ** (sig - 1 - magnitude)
    arr[finite] = np.round(arr[finite] * factor) / factor
    return pd.Series(arr, index=series.index, name=series.name)


def read_variables(output_dir):
    """Read all variable outputs into a :class:`SimpleNamespace`.

    ``output_dir`` is the ``output_raw/`` path.  ``input/`` is resolved
    as a sibling folder (needed for ``group_entity_invest.csv``).
    """
    output_path = Path(output_dir)
    work_folder = output_path.parent
    input_path = work_folder / "input"

    # Prefer the legacy CSV pathway when its output is present (phase 3
    # still runs by default).  Fall back to the per-solve parquets only
    # when there's no ``v_flow.csv`` — e.g. once phase 3 is retired.
    if (output_path / "v_flow.csv").exists():
        return _read_from_csv(output_path, input_path)
    if any(output_path.glob("v_flow__*.parquet")):
        return _read_from_parquet(output_path, input_path)
    return _read_from_csv(output_path, input_path)


def _read_from_parquet(parquet_dir: Path, input_path: Path) -> SimpleNamespace:
    """New pathway: concat per-solve parquets for each variable.

    ``read_lean_parquet`` restores the row / column MultiIndex that
    ``write_lean_parquet`` persisted, so no column-name post-processing
    is needed here — unlike the CSV reader, which has to manually
    install each ``columns.name`` / ``MultiIndex.names``.
    """
    v = SimpleNamespace()

    def _read(name: str, precision: int = 6) -> pd.DataFrame:
        # Concatenate every per-solve parquet for this variable, then
        # sort by the row MultiIndex.  Sorting matters: some downstream
        # pandas ops (``DataFrame.mul(..., level=0)``) raise
        # "Join on level between two MultiIndex objects is ambiguous"
        # unless BOTH operands have the same ``_lexsort_depth``.  Having
        # every DataFrame sorted guarantees that.
        #
        # ``precision`` rounds values to match the ``%.Ng`` format the
        # phase-3 CSV writer uses (most vars write ``%.6g``, some duals
        # use ``%.8g``).  Without this, parquet's full float64 precision
        # produces more decimal digits than the golden CSVs and the
        # downstream ``summary_solve.csv`` diff fails by a trailing digit.
        parts = sorted(parquet_dir.glob(f"{name}__*.parquet"))
        if not parts:
            return pd.DataFrame()
        frames = [read_lean_parquet(p) for p in parts]
        # Drop frames that came back entirely empty (0 rows AND 0 columns);
        # concatenating them in would widen the index dtype to object.
        frames = [f for f in frames if not (f.empty and f.shape[1] == 0)] or frames
        out = pd.concat(frames, axis=0).astype(float)
        if isinstance(out.index, pd.MultiIndex) and len(out.index) > 1:
            out = out.sort_index()
        # Collapse 1-level MultiIndex columns to plain Index.  The CSV
        # pathway produces a plain Index here; downstream code (e.g.
        # ``DataFrame.mul(..., level=0)``) treats MultiIndex-with-one-level
        # differently.
        if isinstance(out.columns, pd.MultiIndex) and out.columns.nlevels == 1:
            out.columns = pd.Index(
                [c[0] for c in out.columns], name=out.columns.names[0],
            )
        # Round to match the phase-3 CSV writer's ``%.Ng`` precision,
        # then fill missing combinations with 0 (same as the CSV reader).
        out = out.apply(lambda col: _round_to_sig(col, precision))
        return out.fillna(0.0)

    # Precision matches the phase-3 CSV format strings in flextool.mod:
    #   - ``%.6g`` for primal variables and slack penalties
    #   - ``%.8g`` for duals and voltage angles
    #   - ``%.10g`` for the scalar objective
    v.obj = _read("v_obj", precision=10)
    v.flow = _read("v_flow", precision=6)
    v.ramp = _read("v_ramp", precision=6)
    v.reserve = _read("v_reserve", precision=6)
    v.state = _read("v_state", precision=6)
    v.online_linear = _read("v_online_linear", precision=6)
    v.startup_linear = _read("v_startup_linear", precision=6)
    v.shutdown_linear = _read("v_shutdown_linear", precision=6)
    v.online_integer = _read("v_online_integer", precision=6)
    v.startup_integer = _read("v_startup_integer", precision=6)
    v.shutdown_integer = _read("v_shutdown_integer", precision=6)
    v.q_state_up = _read("vq_state_up", precision=6)
    v.q_state_down = _read("vq_state_down", precision=6)
    v.q_reserve = _read("vq_reserve", precision=6)
    v.q_inertia = _read("vq_inertia", precision=6)
    v.q_non_synchronous = _read("vq_non_synchronous", precision=6)
    v.q_state_up_group = _read("vq_state_up_group", precision=6)
    v.q_capacity_margin = _read("vq_capacity_margin", precision=6)
    v.invest = _read("v_invest", precision=6)
    v.divest = _read("v_divest", precision=6)
    v.dual_node_balance = _read("v_dual_node_balance", precision=8)
    v.dual_reserve_balance = _read("v_dual_reserve__upDown__group__period__t", precision=8)
    v.angle = _read("v_angle", precision=8)
    v.dual_invest_unit = _read("v_dual_invest_unit", precision=8)
    v.dual_invest_connection = _read("v_dual_invest_connection", precision=8)
    v.dual_invest_node = _read("v_dual_invest_node", precision=8)
    v.dual_maxInvest_period = _read("v_dual_maxInvest_period", precision=8)
    v.dual_maxInvest_total = _read("v_dual_maxInvest_total", precision=8)
    v.dual_maxCumulative = _read("v_dual_maxCumulative", precision=8)
    v.dual_maxInvestGroup_period = _read("v_dual_maxInvestGroup_period", precision=8)
    v.dual_maxInvestGroup_total = _read("v_dual_maxInvestGroup_total", precision=8)
    v.dual_maxInvestGroup_cumulative = _read("v_dual_maxInvestGroup_cumulative", precision=8)
    v.dual_co2_max_period = _read("v_dual_co2_max_period", precision=8)
    v.dual_co2_max_total = _read("v_dual_co2_max_total", precision=8)

    # ``group_entity_invest`` is a static map (solveFirst-only) written
    # during phase 1 directly to ``input/`` — same as the CSV pathway.
    v.group_entity_invest = pd.read_csv(input_path / "group_entity_invest.csv")
    return v


def _read_from_csv(output_path: Path, input_path: Path) -> SimpleNamespace:
    """Legacy pathway: CSVs in ``output_raw/`` written by glpsol phase 3."""
    v = SimpleNamespace()

    # Variables with (solve, period, time) index
    v.obj = pd.read_csv(output_path / 'v_obj.csv', header=[0], index_col=[0]).astype(float)
    v.flow = pd.read_csv(output_path / 'v_flow.csv', header=[0, 1, 2], index_col=[0, 1, 2]).astype(float)
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
    # CO2 emission-cap duals (raw dual is per /1000-scaled RHS; downstream * 1000 for tCO2)
    v.dual_co2_max_period = pd.read_csv(output_path / 'v_dual_co2_max_period.csv', index_col=[0, 1]).astype(float)
    v.dual_co2_max_total = pd.read_csv(output_path / 'v_dual_co2_max_total.csv', index_col=[0]).astype(float)
    # group_entity_invest moved from output_raw/ to input/ with the
    # derived-parameter printf migration (it's solveFirst-gated static).
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
