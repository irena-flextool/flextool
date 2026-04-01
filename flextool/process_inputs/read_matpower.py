"""Parse MATPOWER .m case files and convert to FlexTool Spine DB.

This module provides:
- Dataclasses for MATPOWER bus, generator, branch and case data
- ``read_matpower(filepath)`` — parse a .m file into a ``MatpowerCase``
- ``create_flextool_db_from_matpower(case, db_path)`` — build a ready-to-solve
  FlexTool Spine DB from parsed MATPOWER data

The converter creates a DC-OPF-ready FlexTool database with:
- One commodity node per generator (fuel cost = linear cost coefficient)
- One unit per generator (constant_efficiency, efficiency=1.0)
- One node per bus (demand as negative inflow)
- One connection per branch (with reactance for DC power flow)
- A single AC network group with dc_power_flow_with_angles
- A minimal single-timestep timeline for dispatch
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from spinedb_api import Array, DatabaseMapping, Map, import_data

from flextool.update_flextool import FLEXTOOL_DB_VERSION
from flextool.update_flextool.db_migration import migrate_database
from flextool.update_flextool.initialize_database import initialize_database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MATPOWER data structures
# ---------------------------------------------------------------------------

@dataclass
class MatpowerBus:
    """Single bus from mpc.bus matrix."""
    bus_id: int
    bus_type: int   # 1=PQ, 2=PV, 3=ref (slack)
    pd: float       # real power demand (MW)
    qd: float       # reactive power demand (MVAr) — unused for DC


@dataclass
class MatpowerGen:
    """Single generator from mpc.gen + mpc.gencost matrices."""
    bus: int
    pmax: float     # MW
    pmin: float     # MW
    status: int     # 1=online, 0=offline
    cost_coeffs: list[float] = field(default_factory=list)
    # Polynomial cost coefficients [c2, c1, c0] where cost = c2*P^2 + c1*P + c0


@dataclass
class MatpowerBranch:
    """Single branch from mpc.branch matrix."""
    fbus: int
    tbus: int
    r: float        # resistance (p.u.)
    x: float        # reactance (p.u.)
    rate_a: float   # thermal limit (MVA), 0 = unlimited
    ratio: float    # transformer tap ratio (0 = line, nonzero = transformer)
    status: int     # 1=online


@dataclass
class MatpowerCase:
    """Complete parsed MATPOWER case."""
    name: str
    base_mva: float
    buses: list[MatpowerBus]
    generators: list[MatpowerGen]
    branches: list[MatpowerBranch]


# ---------------------------------------------------------------------------
# MATPOWER parser
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove MATLAB line comments (% ...) and inline comments."""
    lines = []
    for line in text.splitlines():
        # Remove everything after the first % that's not inside a string
        idx = line.find('%')
        if idx >= 0:
            line = line[:idx]
        lines.append(line)
    return '\n'.join(lines)


def _extract_matrix(text: str, name: str) -> list[list[float]]:
    """Extract a matrix block ``mpc.<name> = [ ... ];`` and parse rows.

    Each row is semicolon-terminated within the brackets.
    """
    # Match mpc.<name> = [ ... ];
    pattern = rf'mpc\.{name}\s*=\s*\[(.*?)\];'
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    rows: list[list[float]] = []
    for row_str in block.split(';'):
        row_str = row_str.strip()
        if not row_str:
            continue
        values = row_str.split()
        rows.append([float(v) for v in values])
    return rows


def _extract_scalar(text: str, name: str) -> float | None:
    """Extract ``mpc.<name> = <value>;``."""
    pattern = rf'mpc\.{name}\s*=\s*([\d.eE+-]+)\s*;'
    match = re.search(pattern, text)
    if match:
        return float(match.group(1))
    return None


def _extract_function_name(text: str) -> str:
    """Extract the function name from ``function mpc = <name>``."""
    match = re.search(r'function\s+mpc\s*=\s*(\w+)', text)
    if match:
        return match.group(1)
    return "unknown_case"


def read_matpower(filepath: str | Path) -> MatpowerCase:
    """Parse a MATPOWER .m file and return a ``MatpowerCase``.

    Parameters
    ----------
    filepath : str or Path
        Path to the .m file.

    Returns
    -------
    MatpowerCase
        Parsed case with buses, generators, branches and cost data.
    """
    filepath = Path(filepath)
    raw_text = filepath.read_text(encoding='utf-8')
    text = _strip_comments(raw_text)

    case_name = _extract_function_name(text)
    base_mva = _extract_scalar(text, 'baseMVA')
    if base_mva is None:
        base_mva = 100.0

    # Parse bus data
    # Columns: bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin
    bus_rows = _extract_matrix(text, 'bus')
    buses = [
        MatpowerBus(
            bus_id=int(row[0]),
            bus_type=int(row[1]),
            pd=row[2],
            qd=row[3],
        )
        for row in bus_rows
    ]

    # Parse generator data
    # Columns: bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin
    gen_rows = _extract_matrix(text, 'gen')
    generators = [
        MatpowerGen(
            bus=int(row[0]),
            pmax=row[8],     # Pmax is column 9 (0-indexed: 8)
            pmin=row[9],     # Pmin is column 10 (0-indexed: 9)
            status=int(row[7]),
        )
        for row in gen_rows
    ]

    # Parse generator cost data
    # Format type 2 (polynomial): type, startup, shutdown, n, c(n-1), ..., c0
    gencost_rows = _extract_matrix(text, 'gencost')
    for i, row in enumerate(gencost_rows):
        if i < len(generators):
            cost_type = int(row[0])
            if cost_type == 2:
                n = int(row[3])
                coeffs = row[4:4 + n]  # c(n-1), c(n-2), ..., c0
                generators[i].cost_coeffs = coeffs
            else:
                logger.warning("Generator %d has unsupported cost type %d", i, cost_type)

    # Parse branch data
    # Columns: fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    branch_rows = _extract_matrix(text, 'branch')
    branches = [
        MatpowerBranch(
            fbus=int(row[0]),
            tbus=int(row[1]),
            r=row[2],
            x=row[3],
            rate_a=row[5],   # rateA is column 6 (0-indexed: 5)
            ratio=row[8],    # ratio is column 9 (0-indexed: 8)
            status=int(row[10]),
        )
        for row in branch_rows
    ]

    return MatpowerCase(
        name=case_name,
        base_mva=base_mva,
        buses=buses,
        generators=generators,
        branches=branches,
    )


# ---------------------------------------------------------------------------
# FlexTool DB creation from MATPOWER data
# ---------------------------------------------------------------------------

def create_flextool_db_from_matpower(
    case: MatpowerCase,
    db_path: str,
    *,
    scenario_name: str = "dc_opf_test",
    alternative_name: str = "base",
    template_json: str | None = None,
) -> str:
    """Create a FlexTool Spine DB from parsed MATPOWER data.

    The DB is initialized from the FlexTool master template and then
    populated with entities/parameters for a single-timestep DC-OPF
    dispatch problem.

    Parameters
    ----------
    case : MatpowerCase
        Parsed MATPOWER case.
    db_path : str
        File path for the output .sqlite DB.
    scenario_name : str
        Name for the scenario (default ``dc_opf_test``).
    alternative_name : str
        Name for the alternative (default ``base``).
    template_json : str or None
        Path to the FlexTool template JSON. If None, uses the default
        ``version/flextool_template_master.json`` relative to the project root.

    Returns
    -------
    str
        The ``sqlite:///`` URL for the created database.
    """
    if template_json is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        template_json = str(project_root / "version" / "flextool_template_master.json")

    # Step 1: Initialize from template and migrate to latest schema
    initialize_database(template_json, db_path)
    migrate_database(db_path)

    url = f"sqlite:///{db_path}"

    # Step 2: Add MATPOWER data
    with DatabaseMapping(url) as db:
        _add_matpower_data(db, case, alternative_name)
        db.commit_session("Add MATPOWER case data")

    # Step 3: Add scenario
    with DatabaseMapping(url) as db:
        # The template already creates a 'base' alternative.
        # Add scenario and link it.
        _count, errors = import_data(
            db,
            scenarios=[(scenario_name, True)],
            scenario_alternatives=[(scenario_name, alternative_name)],
        )
        if errors:
            logger.warning("Scenario import errors: %s", errors)
        db.commit_session("Add scenario")

    return url


def _add_matpower_data(
    db: DatabaseMapping,
    case: MatpowerCase,
    alternative: str,
) -> None:
    """Populate a FlexTool DB with MATPOWER case entities and parameters."""
    entities: list[tuple] = []
    entity_alternatives: list[tuple] = []
    parameter_values: list[tuple] = []

    def add_entity(class_name: str, name: str | tuple) -> None:
        """Add an entity and its entity_alternative in one call."""
        entities.append((class_name, name))
        entity_alternatives.append((class_name, name, alternative, True))

    # Find the reference (slack) bus
    ref_bus_id: int | None = None
    for bus in case.buses:
        if bus.bus_type == 3:
            ref_bus_id = bus.bus_id
            break
    if ref_bus_id is None:
        # Fallback: use bus 1
        ref_bus_id = case.buses[0].bus_id if case.buses else 1

    # --- Bus nodes ---
    for bus in case.buses:
        node_name = f"bus_{bus.bus_id}"
        add_entity("node", node_name)

        # Enable node balance constraint
        parameter_values.append(("node", node_name, "has_balance", "yes", alternative))

        if bus.pd != 0.0:
            # Negative inflow = demand in FlexTool
            parameter_values.append(("node", node_name, "inflow", -bus.pd, alternative))

        # Large penalties to avoid slack usage
        parameter_values.append(("node", node_name, "penalty_up", 100000.0, alternative))
        parameter_values.append(("node", node_name, "penalty_down", 100000.0, alternative))

    # --- Generators (only those with Pmax > 0 and status=1) ---
    gen_index = 0
    for gen in case.generators:
        gen_index += 1
        if gen.pmax <= 0 or gen.status == 0:
            continue

        gen_name = f"gen_{gen_index}"
        commodity_node_name = f"commodity_gen{gen_index}"
        commodity_name = f"fuel_gen{gen_index}"
        bus_node_name = f"bus_{gen.bus}"

        # Linear cost coefficient (c1) — $/MWh
        c1 = 0.0
        if len(gen.cost_coeffs) >= 2:
            # For n=3: coeffs = [c2, c1, c0]
            c1 = gen.cost_coeffs[-2]  # c1 is the second-to-last

        # Commodity and commodity node
        add_entity("commodity", commodity_name)
        add_entity("node", commodity_node_name)
        add_entity("commodity__node", (commodity_name, commodity_node_name))

        # Set commodity price = marginal cost
        parameter_values.append(("commodity", commodity_name, "price", c1, alternative))

        # Unit
        add_entity("unit", gen_name)
        add_entity("unit__inputNode", (gen_name, commodity_node_name))
        add_entity("unit__outputNode", (gen_name, bus_node_name))

        # Unit parameters
        parameter_values.append(("unit", gen_name, "existing", gen.pmax, alternative))
        parameter_values.append(("unit", gen_name, "conversion_method", "constant_efficiency", alternative))
        parameter_values.append(("unit", gen_name, "efficiency", 1.0, alternative))

    # --- Branches / connections ---
    for branch in case.branches:
        if branch.status != 1:
            continue
        if branch.x == 0.0:
            # Zero reactance means infinite susceptance — skip
            logger.warning("Skipping branch %d-%d with zero reactance", branch.fbus, branch.tbus)
            continue

        conn_name = f"line_{branch.fbus}_{branch.tbus}"
        from_node = f"bus_{branch.fbus}"
        to_node = f"bus_{branch.tbus}"

        add_entity("connection", conn_name)
        add_entity("connection__node__node", (conn_name, from_node, to_node))

        parameter_values.append(("connection", conn_name, "existing", branch.rate_a, alternative))
        parameter_values.append(("connection", conn_name, "reactance", branch.x, alternative))
        parameter_values.append(("connection", conn_name, "transfer_method", "regular", alternative))

    # --- DC power flow group ---
    group_name = "ac_network"
    add_entity("group", group_name)

    for bus in case.buses:
        add_entity("group__node", (group_name, f"bus_{bus.bus_id}"))

    parameter_values.append(("group", group_name, "transfer_method", "dc_power_flow_with_angles", alternative))
    parameter_values.append(("group", group_name, "base_MVA", case.base_mva, alternative))
    parameter_values.append(("group", group_name, "reference_node", f"bus_{ref_bus_id}", alternative))

    # --- Timeline / timeset / solve / model ---
    timeline_name = "tl_dispatch"
    timeset_name = "ts_dispatch"
    solve_name = "dispatch"
    model_name = "flexTool"
    period_name = "p1"

    # Timeline entity with a single 1-hour timestep
    add_entity("timeline", timeline_name)
    timestep_duration_map = Map(["t0001"], [1.0])
    parameter_values.append(("timeline", timeline_name, "timestep_duration", timestep_duration_map, alternative))

    # Timeset entity linking to the timeline
    add_entity("timeset", timeset_name)
    parameter_values.append(("timeset", timeset_name, "timeline", timeline_name, alternative))
    # timeset_duration: map from first timestep to count of steps
    timeset_duration_map = Map(["t0001"], [1.0])
    parameter_values.append(("timeset", timeset_name, "timeset_duration", timeset_duration_map, alternative))

    # Solve entity
    add_entity("solve", solve_name)
    # period_timeset: map period -> timeset
    period_timeset_map = Map([period_name], [timeset_name])
    parameter_values.append(("solve", solve_name, "period_timeset", period_timeset_map, alternative))
    parameter_values.append(("solve", solve_name, "solve_mode", "single_solve", alternative))
    # realized_periods
    realized_periods = Array([period_name])
    parameter_values.append(("solve", solve_name, "realized_periods", realized_periods, alternative))

    # Model entity
    add_entity("model", model_name)
    parameter_values.append(("model", model_name, "version", FLEXTOOL_DB_VERSION, alternative))
    parameter_values.append(("model", model_name, "discount_rate", 0.0, alternative))
    # model.solves
    solves_array = Array([solve_name])
    parameter_values.append(("model", model_name, "solves", solves_array, alternative))

    # --- Import everything ---
    count, errors = import_data(
        db,
        entities=entities,
        entity_alternatives=entity_alternatives,
        parameter_values=parameter_values,
    )
    if errors:
        for err in errors:
            logger.error("import_data error: %s", err)
        raise RuntimeError(f"Failed to import MATPOWER data: {len(errors)} errors")
    logger.info("Imported %d items into FlexTool DB", count)
