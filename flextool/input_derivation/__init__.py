"""SpineDBBackend output → :class:`FlexDataProvider`.

The :mod:`input_derivation` tier sits between
:mod:`flextool.spinedb_backend` (raw EAV → polars materialisation) and
the in-cascade :class:`flextool.engine.preprocess._flex_data_provider.FlexDataProvider`
that the rest of the engine consumes.

Architecture
------------

::

    SpineDB → SpineDBBackend → input_derivation → FlexDataProvider → cascade

A derivation in this package reads its raw inputs **through the
:class:`SpineDBBackend`** (or, for derivations that compose other
derivation outputs, the Provider) and emits one or more canonical-
schema :class:`polars.DataFrame` objects to the Provider via
``provider.put(key, frame)``.

**No disk writes.**  The derivation tier is purely in-memory; CSV
emission is the Provider's job (``snapshot_processed_inputs`` /
``--csv-dump``).

Hard rules
----------

* No disk writes anywhere in this package.
* No disk-fallback reads.  ``provider`` and ``backend`` are required
  keyword arguments; the derivation fails loudly if either is missing.
* Each derivation populates the Provider via ``provider.put(...)``;
  return values are reserved for derivation-internal carriers (e.g.
  ``ct_method_overrides`` from DC power flow, consumed by process
  method).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import spinedb_api as api
from flextool.update_flextool.db_handle import open_database_mapping

if TYPE_CHECKING:
    from flextool.engine.preprocess._flex_data_provider import FlexDataProvider


__all__ = ["run"]


def _provider_key(filename: str) -> str:
    """Canonical Provider key for ``input/<stem>.csv`` style paths.

    Matches the convention in :mod:`flextool.engine.preprocess._provider_io`:
    ``"<parent_dir_name>/<stem>"``.
    """
    p = Path(filename)
    return f"{p.parent.name}/{p.stem}" if p.parent.name else p.stem


def run(
    backend_or_db_url,
    provider: "FlexDataProvider",
    logger: "logging.Logger | None" = None,
    *,
    scenario_name: "str | None" = None,
    work_folder: "Path | None" = None,
    memory_recorder=None,
) -> None:
    """Populate *provider* with the canonical ``input/`` + derivation
    frames produced from the Spine database.

    Pipeline
    --------

    1. **SpineDBBackend spec loops** — three loops over the canonical
       spec lists (``_DEFAULT_VALUES_SPECS``, ``_ENTITY_SPECS``,
       ``_PARAMETER_SPECS``) materialise ~120 EAV frames into the
       Provider under their canonical ``input/<stem>`` key.
    2. **Validators** (timeline timestep duration, ladder methods,
       capacity margin storage exclusion, group output memberships).
    3. **DB-driven derivations** (DC power flow, process method,
       commodity ladder cumulative + annual + sets).
    4. **Native preprocessing emitters** — the dozen modules under
       :mod:`flextool.engine._emit_*` that produce the
       ``solve_data/*.csv`` artefacts the cascade consumes pre-solve.
       Each is invoked directly with ``provider=`` so its frames flow
       into the Provider and disk emission is skipped.

    Parameters
    ----------
    backend_or_db_url
        Either a Spine DB URL string (``sqlite:///...``) or a
        pre-constructed :class:`SpineDBBackend`.  The URL form opens an
        ephemeral DatabaseMapping for the duration of the call.
    provider
        Cascade-input :class:`FlexDataProvider`.  All frames land here.
    logger
        Optional logger.  Defaults to ``logging.getLogger(__name__)``.
    scenario_name
        Scenario filter applied to the DB mapping.  ``None`` runs
        without a filter.
    work_folder
        Workdir for the (still-disk) batches that have not yet been
        ported to Provider-only operation.  When ``None`` defaults to
        :func:`pathlib.Path.cwd`.  Becomes optional in a later step
        once every writer is Provider-only.
    """
    from flextool.spinedb_backend import SpineDBBackend
    from flextool.input_derivation._specs import (
        _ENTITY_SPECS,
        _PARAMETER_SPECS,
        _DEFAULT_VALUES_SPECS,
        PD_PROCESS_DROP_TIMESTEP_KEYS,
    )
    from flextool.input_derivation._validators import (
        validate_timeline_timestep_duration,
        validate_node_inflow_not_period,
        validate_ladder_methods,
        validate_group_output_memberships,
        validate_capacity_margin_groups,
        validate_group_limit_flags,
        validate_non_sync_one_variable_connections,
        validate_min_load_efficiency_inputs,
        validate_connection_node_memberships,
        validate_connection_delay,
        validate_connection_startup,
        validate_group_increase_reserve_ratio,
        warn_existing_not_whole_units,
        validate_unit_input_shares,
        validate_output_min_coeff,
    )
    from flextool.input_derivation._dc_power_flow import derive_dc_power_flow
    from flextool.input_derivation._process_method import derive_process_method
    from flextool.input_derivation._commodity_ladder import (
        derive_commodity_ladder_cumulative,
        derive_commodity_ladder_annual,
    )
    from flextool.input_derivation._commodity_ladder_sets import (
        derive_commodity_ladder_sets,
    )

    if logger is None:
        logger = logging.getLogger(__name__)
    if provider is None:
        raise TypeError("input_derivation.run requires a FlexDataProvider")

    def _mem(label: str, user_label: str) -> None:
        """Emit a phase checkpoint when a recorder was supplied; no-op
        otherwise.  Lets the user follow input-pipeline phase progress
        with section-delta accounting matching the cascade/build
        checkpoints emitted by :mod:`_orchestration`.

        The recorder's log lines are user-visible by default; full
        tracemalloc-peak diagnostics and CSV emission are opt-in via
        ``FLEXTOOL_MEMORY_DIAGNOSTICS=1`` (handled inside the recorder).
        """
        if memory_recorder is not None:
            memory_recorder.checkpoint(label, logger, user_label=user_label)

    wf = work_folder if work_folder is not None else Path.cwd()

    def _do(db) -> None:
        # Step 1 — SpineDBBackend spec loops.
        backend = SpineDBBackend.__new__(SpineDBBackend)
        backend._db = db                              # type: ignore[attr-defined]
        backend._api = api                            # type: ignore[attr-defined]
        backend._scenario_name = scenario_name        # type: ignore[attr-defined]

        for spec in _DEFAULT_VALUES_SPECS:
            frame = backend.parameter_defaults(
                cl_pars=spec["cl_pars"],
                header=spec["header"],
                filter_in_type=spec.get("filter_in_type"),
                only_value=spec.get("only_value", False),
            )
            provider.put(_provider_key(spec["filename"]), frame)

        for spec in _ENTITY_SPECS:
            frame = backend.entities(
                classes=spec.classes,
                header=spec.header,
                entity_dimens=spec.entity_dimens,
            )
            provider.put(_provider_key(spec.filename), frame)

        validate_timeline_timestep_duration(db)
        validate_node_inflow_not_period(backend)

        timesteps: list[str] | None = None
        for spec in _PARAMETER_SPECS:
            kwargs = {k: v for k, v in spec.items() if k != "filename"}
            frame = backend.parameter_values(**kwargs)
            if spec["filename"] == "input/timeline.csv":
                timesteps = frame["timestep"].unique().to_list()
            elif spec["filename"] == "input/pd_process.csv":
                if timesteps is None:
                    raise RuntimeError("_PARAMETER_SPECS: input/timeline.csv must precede input/pd_process.csv")
                # The period twins of the ``pt_process`` parameters (and ``startup_cost``) also catch silent-``x``
                # time maps; drop their timestep-keyed rows (a timestep is never a period) so they neither pollute
                # ``periodAll`` nor get copied into the per-roll ``pd_process`` lookups. An explicitly
                # period-labelled row (``index_name: "period"``) is never a silent-``x`` leak even when its key
                # happens to also be a timestep name (#76) — re-read the spec filtered to that index and keep any
                # row it names.
                key_cols = ["process", "processParam", "period"]
                explicit_period = backend.parameter_values(**{**kwargs, "filter_in_index": "period"})
                is_explicit = (
                    frame.join(
                        explicit_period.select(key_cols).unique().with_columns(
                            pl.lit(True).alias("__explicit")),
                        on=key_cols, how="left",
                    )["__explicit"].fill_null(False)
                )
                drop = (
                    pl.col("processParam").is_in(list(PD_PROCESS_DROP_TIMESTEP_KEYS))
                    & pl.col("period").is_in(timesteps)
                )
                frame = (
                    frame.with_columns(is_explicit.alias("__explicit"))
                    .filter(~drop | pl.col("__explicit"))
                    .drop("__explicit")
                )
            provider.put(_provider_key(spec["filename"]), frame)
        _mem("spine_db_read_end", "Spine DB loaded")

        # Step 2 — DB-driven derivations.
        ct_method_overrides = derive_dc_power_flow(backend, provider, logger)
        derive_process_method(
            backend, provider, logger,
            ct_method_overrides=ct_method_overrides,
        )
        # Once per scenario: input shares and output floors (need the
        # process method for the online / indirect classification).
        validate_unit_input_shares(db, provider, logger)
        validate_min_load_efficiency_inputs(db, logger)
        validate_output_min_coeff(db, provider, logger)
        validate_ladder_methods(db, logger)
        validate_connection_delay(db, logger)
        validate_connection_startup(db, provider)
        validate_group_increase_reserve_ratio(db, logger)
        derive_commodity_ladder_cumulative(backend, provider, logger)
        derive_commodity_ladder_annual(backend, provider, logger)
        derive_commodity_ladder_sets(backend, provider)
        _mem("db_derivations_end", "DB-driven derivations done")

        # Step 3 — write_input-time native preprocessing emitters.
        # Each emit_* threads the Provider directly so frames flow into
        # memory rather than disk.
        from flextool.engine.preprocess.emit import _leaf_sets as _leaf
        from flextool.engine.preprocess.emit import _mid_sets as _mid
        from flextool.engine.preprocess.emit import _calc_params as _calc
        from flextool.engine.preprocess.emit import _arc_unions as _arc
        from flextool.engine.preprocess.emit import _dispatchers as _disp
        input_dir = wf / "input"
        solve_data_dir = wf / "solve_data"
        os.makedirs(solve_data_dir, exist_ok=True)
        _leaf.emit_period_param_sets(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_invest_method_sets(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_co2_method_sets(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_optional_yes(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_reserve_upDown_group(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_group_loss_share(input_dir, solve_data_dir, provider=provider)
        _mid.emit_node_type_sets(input_dir, solve_data_dir, provider=provider)
        _mid.emit_entity_lifetime_method(input_dir, solve_data_dir, provider=provider)
        _mid.emit_process_ct_method(input_dir, solve_data_dir, provider=provider)
        _mid.emit_process_startup_method(input_dir, solve_data_dir, provider=provider)
        _mid.emit_node_inflow_method(input_dir, solve_data_dir, provider=provider)
        _mid.emit_node_storage_binding_method(input_dir, solve_data_dir, provider=provider)
        _mid.emit_node_penalty_method(input_dir, solve_data_dir, provider=provider)
        _mid.emit_process_group_inside_group_nonsync(input_dir, solve_data_dir, provider=provider)
        _mid.emit_process__sink_nonSync(input_dir, solve_data_dir, provider=provider)
        _mid.emit_group_entity(input_dir, solve_data_dir, provider=provider)
        _mid.emit_process_delayed__duration(input_dir, solve_data_dir, provider=provider)
        _calc.emit_entity_total_caps(input_dir, solve_data_dir, provider=provider)
        _calc.emit_process_method_projections(input_dir, provider=provider)
        warn_existing_not_whole_units(db, provider)
        _calc.emit_process_VRE(input_dir, provider=provider)
        _calc.emit_process_arc_method_joins(input_dir, provider=provider)
        _calc.emit_process_profile_method_joins(input_dir, provider=provider)
        _mid.emit_reserve_partitions(input_dir, solve_data_dir, provider=provider)
        _mid.emit_commodity_node_co2(input_dir, solve_data_dir, provider=provider)
        _mid.emit_process_coeff_zero_sets(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_def_optional_yes(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_process_delayed(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_simple_setof_projections(input_dir, solve_data_dir, provider=provider)
        # emit_period_solve depends on solve_data outputs from
        # emit_simple_setof_projections, so must run after.
        _leaf.emit_period_solve(solve_data_dir, provider=provider)
        _leaf.emit_time_set(input_dir, solve_data_dir, provider=provider)
        _leaf.emit_enable_optional_outputs(solve_data_dir, provider=provider)
        _leaf.emit_node_state_subsets(solve_data_dir, provider=provider)
        _mid.emit_invest_total_sets(input_dir, solve_data_dir, provider=provider)
        _mid.emit_ci_ladder_cumulative(input_dir, solve_data_dir, provider=provider)
        _disp.emit_process_arc_unions(input_dir, solve_data_dir, provider=provider)
        _arc.emit_param_in_use_sets(input_dir, solve_data_dir, provider=provider)

        _mem("preprocessing_writers_end", "Preprocessing writers done")

        # Step 4 — validators that need DB access.
        validate_capacity_margin_groups(db, logger)
        validate_group_limit_flags(db, logger)
        validate_non_sync_one_variable_connections(db, provider, logger)
        validate_group_output_memberships(db, logger)
        validate_connection_node_memberships(db, logger)

    # Accept either a backend, a DatabaseMapping or a URL string.
    if isinstance(backend_or_db_url, str):
        scen_config = (
            api.filters.scenario_filter.scenario_filter_config(scenario_name)
            if scenario_name else None
        )
        with open_database_mapping(backend_or_db_url) as db:
            # Hot-load the DB: fetching upfront is faster than repeated
            # parameter_value lookups.
            db.fetch_all("entity")
            db.fetch_all("parameter_value")
            if scen_config is not None:
                api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
            os.makedirs(wf / "input", exist_ok=True)
            _do(db)
    else:
        # Accept a SpineDBBackend (with ._db) or a DatabaseMapping
        # directly; both expose the find_* API we need.
        db = getattr(backend_or_db_url, "_db", backend_or_db_url)
        os.makedirs(wf / "input", exist_ok=True)
        _do(db)
