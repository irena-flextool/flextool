"""Process method input-derivation — (ct, startup, fork) → method LUT.

The derivation reads raw input through
:class:`flextool.spinedb_backend.SpineDBBackend`, applies the
``METHODS_MAPPING`` LUT (model invariant; mirrors ``set methods`` in
``flextool_base.dat``), and emits three canonical frames into the
:class:`FlexDataProvider`:

* ``input/process_method``         (process, method)
* ``input/process_min_uptime``     (process_min_uptime,)
* ``input/process_min_downtime``   (process_min_downtime,)

Cross-derivation input
----------------------

Depends on
:func:`flextool.input_derivation._dc_power_flow.derive_dc_power_flow`
having already placed ``derived/ct_method_overrides`` on the Provider
(or having returned the dict via its return value).  The override
dict maps connection_name → effective ct_method (DC PF connections
get ``no_losses_no_variable_cost``; non-DC group transfer_method
groups get their group-level method).

See ``specs/step_2_5_audit.md`` Section 7 item 10 for the migration
plan.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping

import polars as pl

from flextool.input_derivation._specs import METHODS_MAPPING


def derive_process_method(
    backend,
    provider,
    logger: logging.Logger,
    *,
    ct_method_overrides: Mapping[str, str] | None = None,
) -> None:
    """Run the process_method derivation.

    Parameters
    ----------
    backend : SpineDBBackend
        The opened backend (raw EAV access via ``find_entities`` and
        ``find_parameter_values``).
    provider : FlexDataProvider
        The cascade-input Provider; receives the three output frames.
    logger : logging.Logger
        Routed to the input-writer logger for parity with legacy
        warning shapes.
    ct_method_overrides : Mapping[str, str] | None
        ``{process_name: overridden_ct_method}``.  Defaults to the
        Provider's ``derived/ct_method_overrides`` frame when omitted
        (the canonical channel; the explicit-kwarg path remains for
        callers that already hold the dict).
    """
    # ------------------------------------------------------------------
    # Resolve the ct_method override dict.  Explicit kwarg wins; else
    # the Provider's derived/ct_method_overrides carrier from Phase B.
    # ------------------------------------------------------------------
    if ct_method_overrides is None:
        ct_method_overrides_dict: dict[str, str] = {}
        if provider.has("derived/ct_method_overrides"):
            df = provider.get("derived/ct_method_overrides")
            if df is not None and df.height > 0:
                ct_method_overrides_dict = dict(zip(
                    df.get_column("process").to_list(),
                    df.get_column("ct_method").to_list(),
                ))
    else:
        ct_method_overrides_dict = dict(ct_method_overrides)

    # --- Collect ct_method per process ---
    ct_method_map: dict[str, str] = {}
    for cl, par in [
        ("unit", "conversion_method"),
        ("connection", "transfer_method"),
    ]:
        for pv in backend.find_parameter_values(
            entity_class_name=cl, parameter_definition_name=par,
        ):
            if pv["type"] is None:
                continue
            process_name = pv["entity_byname"][0]
            ct_method_map[process_name] = str(pv["parsed_value"])

    # Apply group-level ct_method overrides (from DC PF and other
    # group transfer_methods).
    for process_name, override_method in ct_method_overrides_dict.items():
        ct_method_map[process_name] = override_method

    # --- Collect startup_method per process ---
    startup_method_map: dict[str, str] = {}
    for cl in ["unit", "connection"]:
        for pv in backend.find_parameter_values(
            entity_class_name=cl, parameter_definition_name="startup_method",
        ):
            if pv["type"] is None:
                continue
            process_name = pv["entity_byname"][0]
            startup_method_map[process_name] = str(pv["parsed_value"])

    # --- Collect minimum_time_method per process ---
    minimum_time_method_map: dict[str, str] = {}
    for pv in backend.find_parameter_values(
        entity_class_name="unit",
        parameter_definition_name="minimum_time_method",
    ):
        if pv["type"] is None:
            continue
        process_name = pv["entity_byname"][0]
        minimum_time_method_map[process_name] = str(pv["parsed_value"])

    # --- Override startup_method if minimum_time_method requires online variables ---
    for process_name, mtm in minimum_time_method_map.items():
        if mtm in ("min_uptime", "min_downtime", "both"):
            current_startup = startup_method_map.get(process_name, "no_startup")
            if current_startup == "no_startup":
                startup_method_map[process_name] = "linear"
                logger.info(
                    "Process '%s': startup_method overridden to 'linear' "
                    "because minimum_time_method='%s' requires online variables",
                    process_name, mtm,
                )

    # --- Collect sources and sinks per process ---
    source_counts: dict[str, int] = {}
    for ent_class, dim_idx in [
        ("unit__inputNode", [0, 1]),
        ("connection__node__node", [0, 1]),
    ]:
        for entity in backend.find_entities(entity_class_name=ent_class):
            process_name = entity["entity_byname"][dim_idx[0]]
            source_counts[process_name] = source_counts.get(process_name, 0) + 1

    sink_counts: dict[str, int] = {}
    for ent_class, dim_idx in [
        ("unit__outputNode", [0, 1]),
        ("connection__node__node", [0, 2]),
    ]:
        for entity in backend.find_entities(entity_class_name=ent_class):
            process_name = entity["entity_byname"][dim_idx[0]]
            sink_counts[process_name] = sink_counts.get(process_name, 0) + 1

    # --- Collect delayed processes ---
    delayed_processes: set[str] = set()
    for cl in ["unit", "connection"]:
        for pv in backend.find_parameter_values(
            entity_class_name=cl, parameter_definition_name="delay",
        ):
            if pv["type"] is None:
                continue
            delayed_processes.add(pv["entity_byname"][0])

    # --- Collect all processes and which class they belong to ---
    all_processes: dict[str, str] = {}  # process_name -> "unit" or "connection"
    for entity in backend.find_entities(entity_class_name="unit"):
        all_processes[entity["entity_byname"][0]] = "unit"
    for entity in backend.find_entities(entity_class_name="connection"):
        all_processes[entity["entity_byname"][0]] = "connection"

    # --- Resolve method for each process ---
    rows: list[tuple[str, str]] = []
    for process_name, process_class in all_processes.items():
        # ct_method defaults must match flextool_base.dat:
        #   ct_method_constant (units) = "constant_efficiency"
        #   ct_method_regular (connections) = "regular"
        if process_name in ct_method_map:
            ct = ct_method_map[process_name]
        elif process_class == "connection":
            ct = "regular"
        else:
            ct = "constant_efficiency"

        # startup_method: default "no_startup"
        startup = startup_method_map.get(process_name, "no_startup")

        # fork_method: fork_yes if >1 source OR >1 sink OR delayed
        n_sources = source_counts.get(process_name, 0)
        n_sinks = sink_counts.get(process_name, 0)
        is_delayed = process_name in delayed_processes
        fork = "fork_yes" if (n_sources > 1 or n_sinks > 1 or is_delayed) else "fork_no"

        key = (ct, startup, fork)
        method = METHODS_MAPPING.get(key)
        if method is None:
            logger.warning(
                "process_method: no mapping for process '%s' with "
                "(ct_method=%s, startup_method=%s, fork_method=%s) — skipping",
                process_name, ct, startup, fork,
            )
            continue
        if method == "not_applicable":
            logger.warning(
                "process_method: method resolves to 'not_applicable' for "
                "process '%s' (ct_method=%s, startup_method=%s, fork_method=%s)",
                process_name, ct, startup, fork,
            )
        rows.append((process_name, method))

    # ------------------------------------------------------------------
    # process_method frame → Provider
    # ------------------------------------------------------------------
    process_method_frame = pl.DataFrame(
        {
            "process": [r[0] for r in rows],
            "method": [r[1] for r in rows],
        },
        schema={"process": pl.Utf8, "method": pl.Utf8},
    )
    provider.put("input/process_method", process_method_frame)

    # ------------------------------------------------------------------
    # process_min_uptime / process_min_downtime
    # ------------------------------------------------------------------
    min_uptime_values: dict[str, float] = {}
    min_downtime_values: dict[str, float] = {}
    for pv in backend.find_parameter_values(
        entity_class_name="unit", parameter_definition_name="min_uptime",
    ):
        if pv["type"] is not None and pv["parsed_value"]:
            min_uptime_values[pv["entity_byname"][0]] = float(pv["parsed_value"])
    for pv in backend.find_parameter_values(
        entity_class_name="unit", parameter_definition_name="min_downtime",
    ):
        if pv["type"] is not None and pv["parsed_value"]:
            min_downtime_values[pv["entity_byname"][0]] = float(pv["parsed_value"])

    process_min_uptime: list[str] = []
    process_min_downtime: list[str] = []
    for process_name, mtm in minimum_time_method_map.items():
        if mtm in ("min_uptime", "both") and min_uptime_values.get(process_name, 0) > 0:
            process_min_uptime.append(process_name)
        if mtm in ("min_downtime", "both") and min_downtime_values.get(process_name, 0) > 0:
            process_min_downtime.append(process_name)

    provider.put(
        "input/process_min_uptime",
        pl.DataFrame(
            {"process_min_uptime": sorted(process_min_uptime)},
            schema={"process_min_uptime": pl.Utf8},
        ),
    )
    provider.put(
        "input/process_min_downtime",
        pl.DataFrame(
            {"process_min_downtime": sorted(process_min_downtime)},
            schema={"process_min_downtime": pl.Utf8},
        ),
    )


__all__ = ["derive_process_method"]
