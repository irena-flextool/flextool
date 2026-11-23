"""Process method input-derivation — (ct, startup, fork) → method LUT.

The derivation reads raw input through
:class:`flextool.spinedb_backend.SpineDBBackend`, applies the
``METHODS_MAPPING`` LUT (model invariant; mirrors ``set methods`` in
``flextool_base.dat``), and emits three canonical frames into the
:class:`FlexDataProvider`:

* ``input/process_method``         (process, method)
* ``input/process_min_uptime``     (process_min_uptime,)
* ``input/process_min_downtime``   (process_min_downtime,)

and adds the automatic start-ups of ``piecewise_linear`` units to ``input/process__startup_method``.  The method
LUT is looked up with the startup class of each ``startup_method`` value
(:func:`~flextool.input_derivation._method_constants.startup_class`); :func:`write_startup_classes` puts the classes
into ``input/process__startup_method``.

``piecewise_linear`` units (``specs/piecewise_efficiency_final.md`` §2.2): a unit whose curve
(``unit.piecewise_linear_efficiency``) burns fuel at zero output -- the efficiency rises from the first point
to the second in some period -- and has no start-ups (after the ``minimum_time_method`` promotion) gets
``linear`` start-ups (an info line says so, W3); a unit with two or more lines in some period is
``fork_yes`` (as a delayed unit).  The flags come from
:func:`flextool.engine.input._piecewise_curve.unit_flags`, the function the engine classifier
(``params/_derived._classify_process_method``) calls too.

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

from flextool import run_warnings

from flextool.engine.input._piecewise_curve import curves_by_unit, rises_first, unit_flags
from flextool.input_derivation._method_constants import MIN_TIME_ENABLES, STARTUP_CLASSES, startup_class
from flextool.input_derivation._piecewise import parse_piecewise_curves
from flextool.input_derivation._specs import METHODS_MAPPING
from flextool.input_derivation._validators import _definition_default


def write_startup_classes(provider) -> None:
    """Rewrite the frame ``input/process__startup_method`` (the raw ``startup_method`` values) with the startup
    class of each value (:data:`STARTUP_CLASSES`), so that it holds only ``no_startup``, ``linear`` and ``binary``:
    its readers compare those strings.  An unknown value is kept."""
    if not provider.has("input/process__startup_method"):
        return
    sm = provider.get("input/process__startup_method")
    if sm is None or "startup_method" not in sm.columns:
        return
    classes = {raw: cls for raw, (cls, _fix) in STARTUP_CLASSES.items() if raw != cls}
    provider.put("input/process__startup_method", sm.with_columns(
        pl.col("startup_method").cast(pl.Utf8).replace(classes).cast(sm.schema["startup_method"])))


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

    # --- minimum_time_method per unit: explicit, else the definition default, else none (the engine's reading
    # rule, ``params/_derived._min_time_methods``) ---
    units = [entity["entity_byname"][0] for entity in backend.find_entities(entity_class_name="unit")]
    explicit_mtm = _explicit_values(backend, "minimum_time_method")
    default_mtm = _definition_default(backend, "unit", "minimum_time_method")
    default_mtm = "none" if default_mtm is None else str(default_mtm)
    minimum_time_method_map = {u: str(explicit_mtm.get(u, default_mtm)) for u in units}

    # --- Override startup_method if minimum_time_method requires online variables ---
    for process_name, mtm in minimum_time_method_map.items():
        if mtm in _ENABLES_ANY:
            current_startup = startup_method_map.get(process_name, "no_startup")
            if current_startup == "no_startup":
                startup_method_map[process_name] = "linear"
                run_warnings.note("W-0229", [process_name], details={process_name: f"minimum_time_method {mtm}"})

    # --- piecewise_linear: automatic linear start-ups (W3) and the forced fork of multi-line curves ---
    piecewise = sorted(p for p, ct in ct_method_map.items() if ct == "piecewise_linear")
    multi_line: set[str] = set()
    promoted: list[str] = []
    if piecewise:
        curves = parse_piecewise_curves(backend).frame
        flags = unit_flags(curves)
        by_unit = curves_by_unit(curves)
        for process_name in piecewise:
            f = flags.get(process_name)
            if f is None:
                continue
            if f.multi_line:
                multi_line.add(process_name)
            if f.auto_startup and startup_method_map.get(process_name, "no_startup") == "no_startup":
                startup_method_map[process_name] = "linear"
                promoted.append(process_name)
                by_d = by_unit[process_name]
                d = next(d for d in sorted(by_d, key=lambda x: (x is not None, x or "")) if rises_first(by_d[d]))
                pts = by_d[d]
                run_warnings.note("W-0230", [process_name], details={process_name: (
                    f"efficiency rises from load {pts[0][0]:g} to {pts[1][0]:g}"
                    + (f" in period {d}" if d is not None else ""))})
    if promoted and provider.has("input/process__startup_method"):
        sm = provider.get("input/process__startup_method")
        if sm is not None:
            add = pl.DataFrame({"process": promoted, "startup_method": ["linear"] * len(promoted)})
            sm = pl.concat([sm.filter(~pl.col("process").cast(pl.Utf8).is_in(promoted)),
                            add.select(pl.col(c).cast(sm.schema[c]) for c in sm.columns)])
            provider.put("input/process__startup_method", sm)

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
    # Units only: a connection ``delay`` is not implemented and is ignored
    # (``_validators.validate_connection_delay`` warns about it); it must
    # not make the connection ``fork_yes``.
    delayed_processes: set[str] = set()
    for cl in ["unit"]:
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
        is_multi_line = process_class == "unit" and process_name in multi_line
        fork = "fork_yes" if (n_sources > 1 or n_sinks > 1 or is_delayed or is_multi_line) else "fork_no"

        key = (ct, startup_class(startup)[0], fork)
        method = METHODS_MAPPING.get(key)
        if method is None:
            run_warnings.warn("W-0227", process_name, details={
                process_name: f"ct_method {ct}, startup_method {startup}, fork_method {fork}"})
            continue
        if method == "not_applicable":
            run_warnings.warn("W-0228", process_name, details={
                process_name: f"ct_method {ct}, startup_method {startup}, fork_method {fork}"})
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
    times = {name: _unit_times(backend, units, name) for name in MIN_TIME_ENABLES}
    gated = {name: [u for u in units if u in times[name] and minimum_time_method_map[u] in MIN_TIME_ENABLES[name]]
             for name in MIN_TIME_ENABLES}
    process_min_uptime, process_min_downtime = gated["min_uptime"], gated["min_downtime"]
    _warn_min_time_not_enabled(backend, times, gated, minimum_time_method_map, set(explicit_mtm),
                               {u for u in units if startup_class(startup_method_map.get(u, "no_startup"))[0]
                                != "no_startup"})

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


_ENABLES_ANY = MIN_TIME_ENABLES["min_uptime"] | MIN_TIME_ENABLES["min_downtime"]
#: The constraint a minimum time gives, for W-0231's details.
_TIME_ROWS = {"min_uptime": "no minimum up time constraint", "min_downtime": "no minimum down time constraint"}


def _explicit_values(backend, parameter: str) -> dict[str, object]:
    """``{unit: parsed value}`` of the scenario's explicit (non-null) values of ``unit.<parameter>``."""
    return {pv["entity_byname"][0]: pv["parsed_value"]
            for pv in backend.find_parameter_values(entity_class_name="unit", parameter_definition_name=parameter)
            if pv["type"] is not None}


def _time_value(value) -> float:
    """A minimum time read with the engine's test (``float(value)``; unreadable is 0)."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _unit_times(backend, units: list[str], parameter: str) -> dict[str, float]:
    """``{unit: time}`` of the *units* whose ``unit.<parameter>`` (explicit, else the definition default) is
    above 0, as the engine reads it (``params/_derived._process_min_time``)."""
    explicit = _explicit_values(backend, parameter)
    default = _definition_default(backend, "unit", parameter)
    out = {u: _time_value(explicit.get(u, default)) for u in units}
    return {u: v for u, v in out.items() if v > 0}


def _list_lacks_min_uptime(backend) -> bool:
    """The value list ``minimum_time_methods`` exists without ``min_uptime``: the fingerprint of an old upgrade
    that overwrote its first member (values that were ``min_uptime`` read ``none``)."""
    members = [lv["parsed_value"] for lv in backend.find_list_values(parameter_value_list_name="minimum_time_methods")]
    return bool(members) and "min_uptime" not in members


def _warn_min_time_not_enabled(backend, times: dict[str, dict[str, float]], gated: dict[str, list[str]],
                               methods: dict[str, str], explicit: set[str], online: set[str]) -> None:
    """W-0231: the units with a minimum time above 0 that their ``minimum_time_method`` does not enable.  An
    explicit ``none`` is a deliberate switch-off and gives no warning, except on a corrupted list or for a unit
    the database upgrade could not decide (the marker ``minimum_time_method_written``)."""
    missing = {name: [u for u in times[name] if u not in gated[name]] for name in times}
    if not any(missing.values()):
        return
    from flextool.engine import _behaviour_changes as bc
    rows = bc.marker_json("minimum_time_method_written").get("undecided") or []
    undecided = {str(row[0]): str(row[1]) for row in rows if isinstance(row, (list, tuple)) and len(row) >= 2}
    corrupted = _list_lacks_min_uptime(backend)
    details: dict[str, str] = {}
    for unit in dict.fromkeys(u for name in times for u in missing[name]):
        if methods[unit] == "none" and unit in explicit and not corrupted and unit not in undecided:
            continue
        method = methods[unit] if unit in explicit or methods[unit] != "none" else "none (not set)"
        lost = [name for name in times if unit in missing[name]]
        text = ", ".join(f"{name} {times[name][unit]:g} h" for name in lost) + f", minimum_time_method {method}; "
        text += ("; ".join(_TIME_ROWS[name] for name in lost) if unit in online
                 else "the unit has no online variables, so the time has no effect")
        if unit in undecided:
            text += f". The database upgrade could not write minimum_time_method for this unit: {undecided[unit]}."
        if corrupted:
            text += (". The value list minimum_time_methods has no min_uptime: an old database upgrade overwrote it, "
                     "and values that were min_uptime now read none. Check these units.")
        details[unit] = text
    if details:
        run_warnings.warn_grouped("W-0231", list(details), details=details, run_log=True)


__all__ = ["derive_process_method"]
