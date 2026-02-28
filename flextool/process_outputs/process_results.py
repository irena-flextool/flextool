from types import SimpleNamespace

from flextool.process_outputs.drop_levels import drop_levels
from flextool.process_outputs.calc_capacity_flows import compute_capacity_and_flows
from flextool.process_outputs.calc_connections import compute_connection_flows
from flextool.process_outputs.calc_storage_vre import compute_storage_and_vre
from flextool.process_outputs.calc_slacks import compute_slacks
from flextool.process_outputs.calc_costs import compute_costs
from flextool.process_outputs.calc_group_flows import compute_group_flows


def post_process_results(par, s, v):
    """Calculate post-processing results from variables, parameters, and sets"""
    r = SimpleNamespace()

    par, s, v = drop_levels(par, s, v)
    compute_capacity_and_flows(par, s, v, r)
    compute_connection_flows(par, s, v, r)
    compute_storage_and_vre(par, s, v, r)
    compute_slacks(par, s, v, r)
    compute_costs(par, s, v, r)
    compute_group_flows(par, s, v, r)

    return r
