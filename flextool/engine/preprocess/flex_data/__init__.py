"""Read flextool ``input/`` + ``solve_data/`` into a single
``FlexData`` bag.

All fields are optional — None / empty when the scenario doesn't
exercise that feature.  ``build_flextool(p, d)`` switches on field
presence to decide which constraints / variables / objective terms
to add.

Pipeline shape today:

    Spine DB → flextool preprocess → input/ + solve_data/ CSVs → load_flextool

Once flextool's preprocessing migration to Python is complete, this
module gains a parallel entry point that consumes the in-memory
preprocessing state directly, skipping the CSV roundtrip.

Modules (this package re-exports the public names; private helpers are imported from their module):

* ``_container`` — the :class:`FlexData` dataclass and its test-only cascade gate.
* ``_readers`` — provider access, the CSV-shape readers the loaders share, the load profiler.
* ``_load_topology``, ``_load_co2``, ``_load_features``, ``_load_ramp_online``, ``_load_storage`` — the per-feature
  loaders; ``_coarse`` — the coarse-step frames and the coarse-row rule.
* ``_load`` — :func:`load_flextool` and its post-load steps; ``_db_overrides`` — the DB-direct construction passes.
* ``_handoff`` — :func:`build_handoff_from_solution`, with ``_handoff_inputs``, ``_handoff_commodity`` and
  ``_handoff_storage``.
"""

from flextool.engine.preprocess.flex_data._readers import ROLL_END_TIED_KINDS
from flextool.engine.preprocess.flex_data._container import FlexData
from flextool.engine.preprocess.flex_data._coarse import (
    coarse_process_exclusions,
    coarse_process_reasons,
    coarse_tie_straddles,
    coarse_pin_varies,
    coarse_angle_steps,
    COARSE_ROW_FAMILIES,
    coarse_row_members,
    coarse_row_partition,
)
from flextool.engine.preprocess.flex_data._load_co2 import co2_capped_emitter_arcs
from flextool.engine.preprocess.flex_data._load_ramp_online import carried_obligations_yield_to_outages
from flextool.engine.preprocess.flex_data._handoff_inputs import snap_carried_online, round_whole_unit_events
from flextool.engine.preprocess.flex_data._handoff import build_handoff_from_solution
from flextool.engine.preprocess.flex_data._load import load_flextool, reference_start_shares

__all__ = [
    "ROLL_END_TIED_KINDS",
    "FlexData",
    "coarse_process_exclusions",
    "coarse_process_reasons",
    "coarse_tie_straddles",
    "coarse_pin_varies",
    "coarse_angle_steps",
    "COARSE_ROW_FAMILIES",
    "coarse_row_members",
    "coarse_row_partition",
    "co2_capped_emitter_arcs",
    "carried_obligations_yield_to_outages",
    "snap_carried_online",
    "round_whole_unit_events",
    "build_handoff_from_solution",
    "load_flextool",
    "reference_start_shares",
]
