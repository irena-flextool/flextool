from types import SimpleNamespace
from pathlib import Path
import pandas as pd


def read_variables(output_dir):
    """
    Read all variable CSV files into a namespace.

    Returns:
        SimpleNamespace: Namespace with variables as attributes
    """
    output_path = Path(output_dir)
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
    v.dual_invest_unit = pd.read_csv(output_path / 'v_dual_invest_unit.csv', index_col=[0, 1]).astype(float)
    v.dual_invest_connection = pd.read_csv(output_path / 'v_dual_invest_connection.csv', index_col=[0, 1]).astype(float)
    v.dual_invest_node = pd.read_csv(output_path / 'v_dual_invest_node.csv', index_col=[0, 1]).astype(float)

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
