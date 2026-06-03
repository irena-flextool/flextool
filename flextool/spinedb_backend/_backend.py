""":class:`SpineDBBackend` — EAV → in-memory polars materialiser.

The Backend opens a :class:`spinedb_api.DatabaseMapping`, applies a
scenario filter, eagerly fetches the parameter / entity tables (same
``fetch_all`` pattern used by :mod:`flextool.engine_polars._db_loader`
and :mod:`flextool.engine_polars._solve_config`), and exposes per-spec
materialiser methods that each return a canonical
:class:`polars.DataFrame`.

Step 2.5 scope
--------------

Items 1-4 of the work plan (specs/step_2_5_audit.md §7) absorb the
three spec-driven flatteners from the legacy ``input_writer`` into
this Backend:

* :meth:`parameter_defaults` (Item 2) — replaces ``write_default_values``
* :meth:`entities` (Item 3) — replaces ``write_entity``
* :meth:`parameter_values` (Item 4) — replaces ``write_parameter``

Subsequent items port the four input-time derivations (DC power flow,
process method, commodity ladders) and rewire orchestration to consume
Backend output via the Provider.

Hard rules
----------

* **No disk writes anywhere in the Backend.**  Every method returns a
  polars frame.
* **No disk read fallbacks.**  The Backend reads SpineDB, full stop.

Note on :class:`flextool.engine_polars._spinedb_reader.SpineDbReader`
--------------------------------------------------------------------

The pre-existing ``SpineDbReader`` (used by the native cascade LP
build path) is a sibling abstraction over the same EAV schema, but with a
different consumer contract — it returns per-(entity_class,
parameter_name) frames in the canonical-axis schema used by the
``_axis_enums``-driven LP build path, not the spec-driven tabular
``input/*.csv`` layout the input_writer historically emitted.  Both
abstractions coexist; the Backend serves the ``input_writer`` spec
contract, ``SpineDbReader`` continues to serve the fast-path.
Unification is a follow-up beyond Step 2.5.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable

import polars as pl

from flextool.common_utils.precision import format_scalar_for_csv
from flextool.spinedb_backend._axis_enums import (
    AxisContract,
    cast_against_contract,
    load_axis_contract,
)


# ---------------------------------------------------------------------------
# Structural-parameter skip list — preserved verbatim from
# ``input_writer.write_parameter`` (lines 2166-2191).  These parameter
# names hold method names / structural identifiers / boolean flags
# rather than numerical coefficients; precision rounding would corrupt
# the model, so the effective precision is forced to 0 (passthrough).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# v56 Batch F — three entity classes carry an explicit ``is_enabled``
# parameter that replaces the pre-v56 ``entity_alternative.active``
# gating.  The Backend post-filters entities/parameter_values of these
# three classes against the resolved ``is_enabled`` value in the active
# scenario.  See ``flextool.update_flextool.db_migration.
# _migrate_v56_reactivate_is_enabled_parameter`` for the migration that
# materialises the parameter on legacy DBs.
# ---------------------------------------------------------------------------

_IS_ENABLED_CLASSES: frozenset[str] = frozenset({
    "constraint",
    "reserve__upDown__unit__node",
    "reserve__upDown__connection__node",
})


_STRUCTURAL_PARAM_NAMES: frozenset[str] = frozenset({
    # method names
    "ct_method", "transfer_method", "conversion_method",
    "startup_method", "fork_method", "inflow_method",
    "invest_method", "lifetime_method", "ramp_method",
    "minimum_time_method", "storage_binding_method",
    "storage_nested_fix_method", "storage_solve_horizon_method",
    "storage_start_end_method", "profile_method", "reserve_method",
    "co2_method", "loss_share_type", "price_method",
    "solver", "solve_mode",
    "solver_precommand", "solver_arguments",
    # structural flags / references
    "is_DC", "sense", "node_type", "has_capacity_margin",
    "has_inertia", "has_non_synchronous", "include_stochastics",
    "output_nodeGroup_dispatch", "output_nodeGroup_indicators",
    "output_flowGroup_indicators", "flow_aggregator",
    "output_horizon",
    # set membership / name references
    "solves", "contains_solves", "model",
    "realized_periods", "realized_invest_periods",
    "fix_storage_periods", "invest_periods", "periods_available",
    "version",
})


class SpineDBBackend:
    """In-memory SpineDB materialiser for the input_writer spec contract.

    Opens the database, applies the scenario filter once, and
    pre-fetches the entity / parameter_value tables.  Each materialiser
    method returns a polars frame matching the canonical CSV schema
    that the legacy ``input/*.csv`` files used (column names in
    ``spec["header"]`` order).

    Parameters
    ----------
    db_url : str
        SpineDB URL.  Bare paths are auto-prefixed with ``sqlite:///``.
    scenario_name : str | None
        Scenario filter name.  ``None`` disables filtering (matches the
        ``write_input`` semantics: when no scenario is supplied the DB
        is consumed as-is).
    precision_digits : int, default 0
        Precision-rounding policy passed to
        :func:`format_scalar_for_csv` during parameter materialisation.
        ``0`` disables rounding (byte-parity with the pre-precision-pass
        emissions).

    Notes
    -----
    The Backend holds an open :class:`spinedb_api.DatabaseMapping` only
    inside the ``with``-block.  Use it as a context manager:

    .. code-block:: python

        with SpineDBBackend(db_url, scenario) as backend:
            frame = backend.entities(
                classes=("commodity",),
                header=("commodity",),
            )
    """

    # Class-level defaults so ``__new__`` paths (e.g.
    # ``flextool.input_derivation.run`` bypasses ``__init__`` and attaches
    # ``_db`` / ``_api`` / ``_precision_digits`` manually) still expose the
    # attributes Phase-2 cast code reads.  ``_scenario_name`` defaults to
    # ``None`` (unfiltered backend); callers that *do* want a scenario
    # filter should construct via ``__init__``.
    _scenario_name: str | None = None
    _db_url: str = ""
    # Lazy cache for parameter_value rows, keyed by
    # ``(entity_class_name, parameter_definition_name)``.  Populated by
    # :meth:`_get_parameter_value_index` on first ``parameter_values``
    # call.  Replaces ~100 individual ``find_parameter_values(class=,
    # param=)`` calls (each ~1.85s under spinedb-api's scenario filter on
    # large databases) with one bulk ``find_parameter_values()`` call
    # plus a Python partition (~22ms total) — a ~60x speed-up.
    _parameter_value_index: dict | None = None
    # v56 Batch F — lazy cache of ``frozenset[int]`` disabled-entity ids
    # per entity_class in ``_IS_ENABLED_CLASSES``.  Populated on the first
    # ``find_entities`` / ``entities`` / ``parameter_values`` call for
    # that class.  Lifetime is tied to ``_open`` / ``close`` of this
    # Backend instance (single scenario filter).
    _disabled_entity_id_cache: dict | None = None

    def __init__(
        self,
        db_url: str,
        scenario_name: str | None = None,
        *,
        precision_digits: int = 0,
    ) -> None:
        url = str(db_url)
        if "://" not in url:
            url = f"sqlite:///{url}"
        self._db_url = url
        self._scenario_name = scenario_name
        self._precision_digits = int(precision_digits)
        self._db: Any | None = None
        self._open()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open the DB, apply the scenario filter, pre-fetch entities +
        parameter values.

        spinedb-api now requires the ``DatabaseMapping`` to be inside a
        ``with`` block before ``self.query()`` works — the underlying
        ``Session`` is only created in ``__enter__``.  Pre-v56 code
        could call ``scenario_filter_from_dict`` against a freshly
        constructed ``DatabaseMapping``; modern spinedb-api raises
        ``SpineDBAPIError("session is None; did you forget to use the
        DB map inside a 'with' block?")``.  We therefore enter the
        mapping's context here and keep it open for the lifetime of
        this Backend — :meth:`close` exits the context.  The
        ``_context_open_count`` machinery in ``DatabaseMapping`` lets
        nested call sites (e.g. ``fetch_all`` → internal ``with self``)
        re-enter without closing prematurely.
        """
        import spinedb_api as api  # late import: keeps cold-import cheap
        from spinedb_api import DatabaseMapping

        self._api = api
        db = DatabaseMapping(self._db_url)
        # Enter the mapping's context so ``self._session`` is created.
        # Required by ``scenario_filter_from_dict`` (which calls
        # ``db.query()`` while building the filter's subqueries) and by
        # downstream ``find_*`` calls.  Matched by :meth:`close`.
        db.__enter__()
        # ``fetch_all`` mirrors ``input_writer.write_input`` (line 1836-1837).
        # Pre-fetching here means subsequent ``find_*`` calls are pure
        # in-memory walks.
        db.fetch_all("entity")
        db.fetch_all("parameter_value")
        if self._scenario_name:
            scen_config = api.filters.scenario_filter.scenario_filter_config(
                self._scenario_name,
            )
            api.filters.scenario_filter.scenario_filter_from_dict(db, scen_config)
        self._db = db

    def close(self) -> None:
        """Close the underlying DB connection.  Idempotent."""
        if self._db is not None:
            # Exit the mapping's context manager entered by :meth:`_open`,
            # so the Session that was created in ``__enter__`` is closed.
            try:
                self._db.__exit__(None, None, None)
            except Exception:
                pass
            try:
                self._db.connection.close()
            except Exception:
                pass
            try:
                self._db.close()
            except Exception:
                pass
            self._db = None
        self._parameter_value_index = None
        self._disabled_entity_id_cache = None

    def __enter__(self) -> "SpineDBBackend":
        if self._db is None:
            self._open()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Diagnostic properties
    # ------------------------------------------------------------------

    @property
    def db_url(self) -> str:
        return self._db_url

    @property
    def scenario_name(self) -> str | None:
        return self._scenario_name

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"SpineDBBackend(db_url={self._db_url!r}, "
            f"scenario_name={self._scenario_name!r})"
        )

    # ------------------------------------------------------------------
    # Raw EAV accessors — for input_derivation consumers
    # ------------------------------------------------------------------

    def find_entities(self, *, entity_class_name: str) -> list:
        """Return all entity rows of *entity_class_name*.

        Thin passthrough over the underlying
        :meth:`spinedb_api.DatabaseMapping.find_entities`, exposed so
        consumers in :mod:`flextool.input_derivation` (DC power flow,
        process method, commodity ladder) can perform DB-driven
        derivations without reaching past the Backend boundary.

        v56 Batch F — for the three entity classes in
        :data:`_IS_ENABLED_CLASSES` the result is post-filtered against
        the explicit ``is_enabled`` parameter_value: entities whose
        ``is_enabled`` resolves to ``"no"`` in the active scenario are
        dropped.  Replaces the entity_alternative-based gating that
        spinedb_api's scenario_filter applied pre-v56 (the migration
        drops those rows for the three classes).
        """
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        entities = list(
            self._db.find_entities(entity_class_name=entity_class_name)
        )
        if entity_class_name in _IS_ENABLED_CLASSES:
            disabled_ids = self._disabled_entity_ids(entity_class_name)
            if disabled_ids:
                entities = [e for e in entities if e["id"] not in disabled_ids]
        return entities

    def _disabled_entity_ids(self, entity_class_name: str) -> frozenset[int]:
        """Return entity IDs of *entity_class_name* whose effective
        ``is_enabled`` parameter_value is ``"no"`` under the active
        scenario filter.

        Lazy-cached per class on first call.  The scenario filter is
        applied DB-wide at ``_open`` time (see
        :meth:`scenario_filter_from_dict`), so
        :meth:`spinedb_api.DatabaseMapping.find_parameter_values` already
        returns the scenario-priority winner per (entity, parameter):
        one row per entity, with the highest-priority alternative's
        value resolved.  We treat any returned row carrying
        ``parsed_value == "no"`` as disabling its entity.

        On a pre-v56 database the ``is_enabled`` parameter_definition
        does not exist; ``find_parameter_values`` then yields nothing
        and every entity is treated as enabled — preserving legacy
        behaviour for callers that open an un-migrated DB read-only.
        """
        cache = self._disabled_entity_id_cache
        if cache is None:
            cache = {}
            self._disabled_entity_id_cache = cache
        if entity_class_name in cache:
            return cache[entity_class_name]
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        try:
            pv_rows = self._db.find_parameter_values(
                entity_class_name=entity_class_name,
                parameter_definition_name="is_enabled",
            )
        except Exception:  # noqa: BLE001 — pre-v56 DBs lack the pdef
            pv_rows = []
        disabled: set[int] = set()
        for pv in pv_rows:
            if pv.get("type") != "str":
                continue
            if pv.get("parsed_value") == "no":
                disabled.add(pv["entity_id"])
        result = frozenset(disabled)
        cache[entity_class_name] = result
        return result

    def find_parameter_values(
        self,
        *,
        entity_class_name: str,
        parameter_definition_name: str,
    ) -> list:
        """Return all parameter-value rows for *(entity_class_name,
        parameter_definition_name)*.

        Same role as :meth:`find_entities`: a Backend-owned passthrough
        so :mod:`flextool.input_derivation` derivations don't have to
        re-implement the DB access layer.
        """
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        return self._db.find_parameter_values(
            entity_class_name=entity_class_name,
            parameter_definition_name=parameter_definition_name,
        )

    @property
    def api(self):
        """Expose the underlying :mod:`spinedb_api` module.

        Several derivations need ``api.convert_map_to_table`` /
        ``api.parameter_value.from_database_to_dimension_count`` to
        flatten nested Spine maps.  Re-exposing the same module the
        Backend imported avoids a duplicate import in every derivation
        site (and keeps that import path single-sourced).
        """
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        return self._api

    # ==================================================================
    # Materialiser methods — fully filled in by Items 2-4 of the work
    # plan.  Each method returns a polars DataFrame; the caller is
    # responsible for placing it in the Provider under the canonical
    # frame key.
    # ==================================================================

    # ------------------------------------------------------------------
    # parameter_defaults — Item 2
    # ------------------------------------------------------------------

    def parameter_defaults(
        self,
        *,
        cl_pars: Iterable[tuple[str, str]],
        header: str,
        filter_in_type: Iterable[str] | None = None,
        only_value: bool = False,
        axis_enums: dict[str, pl.Enum] | None = None,
        contract: AxisContract | None = None,
    ) -> pl.DataFrame:
        """Materialise the default-value rows for one ``_DEFAULT_VALUES_SPECS`` entry.

        Returns a polars frame.

        Parameters
        ----------
        cl_pars : iterable of (entity_class, parameter_name) tuples
            The parameter-definition (class, name) pairs to materialise.
        header : str
            Comma-separated header for the canonical CSV layout.  The
            frame's columns are derived from this header so downstream
            consumers see byte-identical schemas.
        filter_in_type : iterable of str, optional
            Restrict to parameter definitions whose ``default_type`` is
            in this set.  ``None`` accepts every supported scalar type.
        only_value : bool, default False
            When True the frame contains a single column (the value
            only).  Used by the ``db_version`` spec where the canonical
            CSV is a one-column scalar dump.
        """
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        cl_pars = list(cl_pars)
        cols = [c.strip() for c in header.split(",")]
        filter_set = (
            set(filter_in_type) if filter_in_type is not None else None
        )

        precision = self._precision_digits
        # Match write_default_values' structural-param policy: when the
        # default is one of the structural param names the precision is
        # forced to 0 so the emitted text round-trips byte-for-byte with
        # the on-disk legacy CSV.  (The legacy code did not branch on
        # the structural set for defaults — but the params in the only
        # default_values spec entry are ``penalty_up`` / ``penalty_down``
        # which are pure floats, so the policy is a strict superset.)
        if any(par in _STRUCTURAL_PARAM_NAMES for _, par in cl_pars):
            precision = 0

        api = self._api
        # ``find_parameter_definitions`` returns ALL definitions; filter
        # locally by (class, name) to mirror write_default_values exactly.
        definitions = self._db.find_parameter_definitions()
        rows: list[list[str]] = []
        # Origin breadcrumbs parallel to ``rows`` (Option C) — each entry
        # is a dict captured at row-emit time so a downstream
        # FlexDataIntegrityError can name the parameter / class that
        # generated the offending row.
        row_origins: list[dict[str, Any]] = []
        from flextool.engine_polars._solve_state import FlexToolConfigError

        for cl_par in cl_pars:
            for definition in definitions:
                if (
                    definition["entity_class_name"] != cl_par[0]
                    or definition["name"] != cl_par[1]
                ):
                    continue
                if (
                    filter_set is not None
                    and definition["default_type"] not in filter_set
                ):
                    continue
                dtype = definition["default_type"]
                if dtype in ("str", "float", "bool"):
                    raw = api.from_database(
                        definition["default_value"], dtype,
                    )
                    formatted = format_scalar_for_csv(raw, precision)
                    if only_value:
                        rows.append([formatted])
                    else:
                        rows.append([
                            definition["entity_class_name"],
                            definition["name"],
                            formatted,
                        ])
                    row_origins.append({
                        "parameter": definition["name"],
                        "entity": definition["entity_class_name"],
                        "scenario": self._scenario_name,
                    })
                else:
                    raise FlexToolConfigError(
                        "Default_value found in a parameter definition not "
                        "of supported default type\nParameter: "
                        + definition.get("parameter_definition_name", definition["name"])
                    )

        frame = _rows_to_frame(rows, cols)
        return self._maybe_cast(
            frame, rows, row_origins, axis_enums, contract,
        )

    # ------------------------------------------------------------------
    # entities — Item 3
    # ------------------------------------------------------------------

    def entities(
        self,
        *,
        classes: Iterable[str],
        header: str,
        entity_dimens: Iterable[Iterable[int]] | None = None,
        axis_enums: dict[str, pl.Enum] | None = None,
        contract: AxisContract | None = None,
    ) -> pl.DataFrame:
        """Materialise the entity rows for one ``_ENTITY_SPECS`` entry as a polars frame.

        Parameters
        ----------
        classes : iterable of str
            The SpineDB entity-class names to union over.  Each class
            contributes its full entity set; the result is the
            row-concatenation (in class-iteration order).
        header : str
            Comma-separated header for the canonical CSV layout.
        entity_dimens : iterable of iterable of int, optional
            Per-class re-ordering / projection of the
            ``entity_byname`` tuple — list of integer index lists, one
            per class.  Used to disambiguate multi-dimensional
            entity-class joins where the same dim appears more than
            once (e.g. ``connection__node__node``).  ``None`` keeps the
            byname tuple as-is.
        """
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        classes_list = list(classes)
        dimens_list: list[list[int] | None]
        if entity_dimens is None:
            dimens_list = [None] * len(classes_list)
        else:
            dimens_list = [list(d) if d is not None else None
                           for d in entity_dimens]
        cols = [c.strip() for c in header.split(",")]

        rows: list[list[str]] = []
        row_origins: list[dict[str, Any]] = []
        for i, ent_class in enumerate(classes_list):
            dim_proj = dimens_list[i] if i < len(dimens_list) else None
            # v56 Batch F — filter is_enabled="no" entities out of the
            # three affected classes.  Replaces the entity_alternative
            # gating spinedb_api's scenario_filter applied pre-v56.
            disabled_ids: frozenset[int] = frozenset()
            if ent_class in _IS_ENABLED_CLASSES:
                disabled_ids = self._disabled_entity_ids(ent_class)
            for entity in self._db.find_entities(entity_class_name=ent_class):
                if disabled_ids and entity["id"] in disabled_ids:
                    continue
                byname = entity["entity_byname"]
                if dim_proj is None:
                    rows.append(list(byname))
                else:
                    rows.append([byname[x] for x in dim_proj])
                row_origins.append({
                    "parameter": None,
                    "entity": "->".join(str(b) for b in byname),
                    "scenario": self._scenario_name,
                    "entity_class": ent_class,
                })
        frame = _rows_to_frame(rows, cols)
        return self._maybe_cast(
            frame, rows, row_origins, axis_enums, contract,
        )

    # ------------------------------------------------------------------
    # parameter_values — Item 4
    # ------------------------------------------------------------------

    def parameter_values(
        self,
        *,
        cl_pars: Iterable[tuple[str, str]],
        header: str,
        filter_in_type: Iterable[str] | None = None,
        filter_out_index: str | None = None,
        filter_in_value: Any = None,
        no_value: bool = False,
        param_print: bool = False,
        dimens: Iterable[int] | None = None,
        param_loc: int | None = None,
        no_entity: bool | None = None,
        axis_enums: dict[str, pl.Enum] | None = None,
        contract: AxisContract | None = None,
    ) -> pl.DataFrame:
        """Materialise the parameter-value rows for one ``_PARAMETER_SPECS`` entry.

        Handles the 1d / 2d / … map dimensionality logic, type / value /
        index filters, the structural-parameter precision policy, the
        ``no_value`` /
        ``no_entity`` / ``param_print`` / ``param_loc`` schema shaping
        flags — but returns a polars frame instead of writing CSV.
        """
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        from flextool.engine_polars._solve_state import FlexToolConfigError

        cl_pars = list(cl_pars)
        cols = [c.strip() for c in header.split(",")]
        dimens_list = list(dimens) if dimens is not None else None

        # Effective precision per the structural-param policy.
        effective_precision = (
            0 if any(par in _STRUCTURAL_PARAM_NAMES for _, par in cl_pars)
            else self._precision_digits
        )

        # Map-dimensionality filter parsing — exactly write_parameter's
        # rule.  At most one ``Nd_map`` entry; if present, also accept
        # the generic ``map`` type code via the filter.
        type_filter_map_dim: int | list = []
        filter_in_type_list: list[str] | None = None
        if filter_in_type is not None:
            filter_in_type_list = list(filter_in_type)
            map_found = False
            for tf in list(filter_in_type_list):
                if tf in ("1d_map", "2d_map", "3d_map", "4d_map", "5d_map"):
                    if map_found:
                        message = (
                            "Trying to have two different dimensionalities in "
                            "the same parameter to be written out"
                        )
                        logging.error(message)
                        raise FlexToolConfigError(message)
                    map_found = True
                    type_filter_map_dim = int(tf[0])
                    filter_in_type_list.remove(tf)
            if map_found:
                filter_in_type_list.append("map")

        api = self._api

        # Use the cached (class, param) → list[param] index built by
        # ``_get_parameter_value_index`` — one bulk ``find_parameter_values()``
        # call instead of N filtered ones.  Each filtered call is
        # ~1.85s under spinedb-api's scenario filter on large fixtures;
        # the bulk call + Python partition is ~2.8s total regardless of
        # N, so once N > ~2 this is a net win.
        index = self._get_parameter_value_index()
        params: list = []
        # v56 Batch F — when a (class, param) pair belongs to an
        # is_enabled-gated class, drop parameter_value rows for entities
        # whose ``is_enabled`` resolves to ``"no"`` in the active
        # scenario.  Without this filter the cascade would re-introduce
        # disabled entities through their non-is_enabled
        # parameter_values (e.g. ``constraint.constant`` /
        # ``reserve__upDown__unit__node.reliability``).
        for cl_par in cl_pars:
            bucket = index.get((cl_par[0], cl_par[1]), ())
            ent_class = cl_par[0]
            if ent_class in _IS_ENABLED_CLASSES:
                disabled_ids = self._disabled_entity_ids(ent_class)
                if disabled_ids:
                    bucket = [
                        p for p in bucket
                        if p["entity_id"] not in disabled_ids
                    ]
            params.extend(bucket)

        # Track A.6 — chunked row accumulator.  Some specs (notably
        # ``('profile', 'profile')`` on H2_trade) flatten to multi-million
        # Python row-lists, which dominate input_derivation peak RSS even
        # after Track A + A.5.  Flushing the row accumulator to a polars
        # sub-frame periodically and clearing the list keeps the
        # in-flight Python overhead bounded to ``_ROWS_FLUSH_THRESHOLD``
        # entries.  The sub-frames are concatenated at end; for ≤1 chunk
        # the path is identical to the pre-chunking code.
        #
        # Disabled when ``axis_enums`` is supplied: ``_maybe_cast``'s
        # error-breadcrumb logic walks ``rows`` and ``row_origins`` in
        # parallel, so partial-rows views would mis-attribute cast
        # failures.  axis_enum-supplied callsites are tests with small
        # fixtures where the peak doesn't bind; safe to fall back.
        _ROWS_FLUSH_THRESHOLD = 200_000
        _can_chunk = axis_enums is None
        chunk_frames: list[pl.DataFrame] = []

        rows: list[list[str]] = []
        # Origin breadcrumbs parallel to ``rows`` (Option C).  Each
        # entry is a dict of (parameter, entity, scenario, map_index)
        # captured at row-emit time.  Map rows append the map index
        # specifically; scalar / array rows leave map_index as None.
        #
        # Track A.5 — these breadcrumbs are *only* consulted by
        # :meth:`_maybe_cast` when an axis-enum cast fails.  When
        # ``axis_enums`` is ``None`` (the default for the
        # ``input_derivation`` callsites) ``_maybe_cast`` returns the
        # frame untouched and never reads ``row_origins``.  Building 3+
        # million-entry dict lists for that path is pure waste, so we
        # short-circuit append to a no-op when no cast is requested.
        # For axis-enum-enabled callsites the full origin dict list is
        # still constructed and the error breadcrumb quality is
        # unchanged.
        row_origins: list[dict[str, Any]] = []
        if axis_enums is None:
            def _origin_append(_origin: dict[str, Any]) -> None:
                pass
        else:
            _origin_append = row_origins.append  # type: ignore[assignment]
        scen = self._scenario_name
        # Track A — wrap ``params`` so each :class:`spinedb_api.PublicItem`
        # has its underlying ``MappedItem._parsed_value`` set to ``None``
        # as iteration advances past it.  ``parsed_value`` is a lazy
        # property in spinedb-api: dropping the cached object releases the
        # parsed ``Map`` / ``TimeSeries`` / ``Array`` and the next access
        # would re-parse via ``from_database(value, type)``.  Each
        # ``(class, param)`` is touched exactly once per
        # ``input_derivation`` pass, so this gives the full memory win at
        # zero re-parse cost.  The wrapper is safe even when callers
        # ``continue`` mid-iteration: the next ``next()`` call enters the
        # generator body BEFORE yielding the new item, so the previously
        # yielded item is evicted no matter how the caller exited the
        # body.  At any moment at most one parsed_value lives.
        # ``tests/spinedb_backend/test_parsed_value_eviction.py`` enforces
        # this contract and warns loudly on spinedb-api API drift.

        def _evict_as_we_go(items):
            prev = None
            for item in items:
                if prev is not None:
                    prev.mapped_item._parsed_value = None
                prev = item
                yield item
            if prev is not None:
                prev.mapped_item._parsed_value = None

        for param in _evict_as_we_go(params):
            if _can_chunk and len(rows) >= _ROWS_FLUSH_THRESHOLD:
                # Flush the in-flight rows into a sub-frame and start
                # fresh.  ``rows = []`` rebinds; the previous list
                # becomes garbage as soon as ``_rows_to_frame`` returns.
                chunk_frames.append(_rows_to_frame(rows, cols))
                rows = []
            if param["type"] is None:
                continue
            if filter_in_type_list and param["type"] not in filter_in_type_list:
                continue

            entity_byname = list(param["entity_byname"])
            if dimens_list is not None:
                tmp: list[Any] = [None] * len(entity_byname)
                for i, dim in enumerate(dimens_list):
                    tmp[dim] = entity_byname[i]
                entity_byname = tmp

            if param_print:
                if param_loc is not None:
                    collect = []
                    for i, byname in enumerate(entity_byname):
                        if i == param_loc:
                            collect.append(param["parameter_definition_name"])
                        collect.append(byname)
                    first_cols = collect
                else:
                    if no_entity:
                        first_cols = [param["parameter_definition_name"]]
                    else:
                        first_cols = list(entity_byname) + [
                            param["parameter_definition_name"]
                        ]
            else:
                first_cols = list(entity_byname)

            pname = param["parameter_definition_name"]
            ent_str = "->".join(
                str(b) for b in param["entity_byname"]
            )

            ptype = param["type"]
            if ptype == "map":
                if (
                    filter_out_index
                    and param["parsed_value"].index_name == filter_out_index
                ):
                    continue
                if (
                    filter_in_type_list
                    and isinstance(type_filter_map_dim, int)
                    and type_filter_map_dim
                    != api.parameter_value.from_database_to_dimension_count(
                        param["value"], param["type"],
                    )
                ):
                    continue
                value = param["parsed_value"]
                if (
                    api.parameter_value.from_database_to_dimension_count(
                        param["value"], param["type"],
                    ) <= 1
                ):
                    indexes = [str(ind) for ind in value.indexes]
                    pairs = list(zip(
                        indexes,
                        [format_scalar_for_csv(v, effective_precision)
                         for v in value.values],
                    ))
                    for idx, val in pairs:
                        if no_value:
                            rows.append(first_cols + [idx])
                        else:
                            rows.append(first_cols + [idx, val])
                        _origin_append({
                            "parameter": pname,
                            "entity": ent_str,
                            "scenario": scen,
                            "map_index": idx,
                        })
                else:
                    flat_map = api.convert_map_to_table(value)
                    for index in flat_map:
                        row = list(index)
                        if no_value:
                            rows.append(first_cols + row[:-1])
                        else:
                            row[-1] = format_scalar_for_csv(
                                row[-1], effective_precision,
                            )
                            rows.append(first_cols + row)
                        # The map index path is the row content before
                        # the value; render it as a joined-string for
                        # the breadcrumb.  Use index[:-1] to drop the
                        # leaf value.
                        idx_path = "->".join(
                            str(x) for x in list(index)[:-1]
                        )
                        _origin_append({
                            "parameter": pname,
                            "entity": ent_str,
                            "scenario": scen,
                            "map_index": idx_path,
                        })
            elif ptype in ("array", "time_series"):
                # Reject array-valued ``node.storage_binding_method`` at
                # ingestion.  The 2026-04 list-valued design (now reverted)
                # silently flattened arrays into one row per element,
                # which downstream additive logic in
                # ``calc_storage_vre.py`` and the constraint emitter
                # turned into double-counted state-change residuals.
                # Single-valued is the contract; arrays are a hard error
                # with an actionable message pointing at the v52->v53
                # migration (added in Phase 2).
                if (
                    param["entity_class_name"] == "node"
                    and pname == "storage_binding_method"
                ):
                    try:
                        arr_values = list(param["parsed_value"].values)
                    except Exception:
                        arr_values = []
                    arr_repr = "[" + ", ".join(
                        repr(v) for v in arr_values
                    ) + "]"
                    allowed = [
                        "bind_within_solve_blended_weights",
                        "bind_intraperiod_blocks",
                        "bind_within_solve",
                        "bind_within_period",
                        "bind_within_timeblock",
                        "bind_forward_only",
                    ]
                    message = (
                        "node.storage_binding_method must be a single "
                        "string, not an array.\n"
                        f"Offending entity: {ent_str}\n"
                        f"Array contents: {arr_repr}\n"
                        f"Allowed single-string values: {allowed}\n"
                        "Pick `bind_within_solve_blended_weights` if the node "
                        "uses representative-period blended weights, "
                        "otherwise pick the dominant non-RP method from "
                        "your array.\n"
                        "Run the v52->v53 migration tool (added in "
                        "Phase 2) to convert existing databases "
                        "automatically."
                    )
                    logging.error(message)
                    raise FlexToolConfigError(message)
                for i_arr, v in enumerate(param["parsed_value"].values):
                    rows.append(
                        list(entity_byname)
                        + [format_scalar_for_csv(v, effective_precision)]
                    )
                    _origin_append({
                        "parameter": pname,
                        "entity": ent_str,
                        "scenario": scen,
                        "map_index": str(i_arr),
                    })
            elif ptype in ("str", "float", "bool"):
                if filter_in_value is not None and param["parsed_value"] != filter_in_value:
                    continue
                if no_value:
                    rows.append(first_cols)
                else:
                    rows.append(
                        first_cols
                        + [format_scalar_for_csv(
                            param["parsed_value"], effective_precision,
                        )]
                    )
                _origin_append({
                    "parameter": pname,
                    "entity": ent_str,
                    "scenario": scen,
                })
            else:
                supported = (
                    filter_in_type_list
                    if filter_in_type_list
                    else ["bool", "str", "float", "array", "time_series", "map"]
                )
                message = (
                    f"Input data found in a parameter not of supported type."
                    f"\nEntity: {','.join(map(str, entity_byname))}"
                    f"\nParameter: {param['parameter_definition_name']}"
                    f"\nSupported types: {supported}"
                    f"\nParameter type: {param['type']}"
                )
                logging.error(message)
                raise FlexToolConfigError(message)

        # Track A.6 — assemble the final frame from any flushed chunks
        # plus the in-flight remainder.  Single-chunk path is identical
        # to the pre-chunking behaviour (one ``_rows_to_frame`` call,
        # one ``_maybe_cast``).  Multi-chunk path concatenates the
        # sub-frames *before* the cast so ``_maybe_cast`` sees a single
        # frame.  ``row_origins`` is empty in the chunked path (Track A.5
        # gated it to non-None axis_enums callers; chunked path is
        # axis_enums-None only), so ``_maybe_cast``'s breadcrumb walk is
        # a no-op there.
        if chunk_frames:
            if rows:
                chunk_frames.append(_rows_to_frame(rows, cols))
                rows = []
            if len(chunk_frames) == 1:
                frame = chunk_frames[0]
            else:
                frame = pl.concat(chunk_frames)
        else:
            frame = _rows_to_frame(rows, cols)
        return self._maybe_cast(
            frame, rows, row_origins, axis_enums, contract,
        )

    # ------------------------------------------------------------------
    # Internal parameter_value cache
    # ------------------------------------------------------------------

    def _get_parameter_value_index(self) -> dict:
        """Build (lazily) and return ``{(class, param): [param_rows]}``.

        ``find_parameter_values(entity_class_name=, parameter_definition_name=)``
        is ~1.85s per call under spinedb-api's scenario filter on large
        fixtures — proportional to the in-memory parameter_value table,
        not to the filter-selectivity.  A bulk ``find_parameter_values()``
        with no filter args returns *all* (scenario-filtered) rows in
        ~2.8s on the same fixture and partitions in ~22ms.  For the
        ``input_derivation.run`` call shape (~100 individual filtered
        calls per backend instance), the bulk-then-partition pattern
        is ~60x faster.

        The cache is invalidated by ``close()``.  It is rebuilt
        per-backend-instance, so callers that mutate the underlying
        ``DatabaseMapping`` mid-flight (no current caller does this)
        would see stale data — flag this if it ever becomes relevant.
        """
        if self._parameter_value_index is not None:
            return self._parameter_value_index
        if self._db is None:
            raise RuntimeError("SpineDBBackend is closed")
        all_params = self._db.find_parameter_values()
        index: dict = {}
        for p in all_params:
            key = (p["entity_class_name"], p["parameter_definition_name"])
            bucket = index.get(key)
            if bucket is None:
                index[key] = [p]
            else:
                bucket.append(p)
        self._parameter_value_index = index
        return index

    # ------------------------------------------------------------------
    # Internal cast helper
    # ------------------------------------------------------------------

    def _maybe_cast(
        self,
        frame: pl.DataFrame,
        rows: list[list[Any]],
        row_origins: list[dict[str, Any]],
        axis_enums: dict[str, pl.Enum] | None,
        contract: AxisContract | None,
    ) -> pl.DataFrame:
        """Cast *frame* against the axis contract when *axis_enums* is
        non-None.

        Origin breadcrumbs (Option C from the Phase 2 plan): the row
        builder kept a *row_origins* list parallel to *rows*; on a cast
        failure we identify the offending column / token by scanning
        the frame, then look up which row introduced that token and
        thread the row's origin dict into the raised
        :class:`FlexDataIntegrityError`.

        When *axis_enums* is None the method is a no-op (pre-Phase-2
        behaviour: callers that don't opt-in see Utf8 frames).
        """
        if axis_enums is None:
            return frame
        if contract is None:
            contract = load_axis_contract()
        # First pass: identify bad (column, token) before raising so
        # we can find its row in row_origins.
        bad_col: str | None = None
        bad_token: str | None = None
        for col in frame.columns:
            axis = contract.column_to_axis(col)
            if axis is None:
                continue
            dtype = axis_enums.get(axis.name)
            if dtype is None:
                continue
            vocab = set(dtype.categories.to_list())
            for v in frame[col].to_list():
                if v is None or v == "":
                    continue
                if v not in vocab:
                    bad_col = col
                    bad_token = str(v)
                    break
            if bad_col is not None:
                break

        if bad_col is None:
            # All values fit — let cast_against_contract do the cast.
            return cast_against_contract(
                frame,
                contract=contract,
                axis_enums=axis_enums,
                origin={"scenario": self._scenario_name},
                backend=self,
            )

        # Identify the row that introduced the bad token to thread the
        # right breadcrumb into the error.
        origin_for_error: dict[str, Any] = {
            "scenario": self._scenario_name,
        }
        try:
            col_idx = frame.columns.index(bad_col)
        except ValueError:  # pragma: no cover — defensive
            col_idx = -1
        if col_idx >= 0:
            for i, r in enumerate(rows):
                if col_idx < len(r) and str(r[col_idx]) == bad_token:
                    if i < len(row_origins):
                        origin_for_error = {
                            **row_origins[i],
                            "scenario": row_origins[i].get(
                                "scenario", self._scenario_name,
                            ),
                        }
                    break

        # Re-run cast_against_contract with the located breadcrumb so
        # the FlexDataIntegrityError carries the precise origin.
        return cast_against_contract(
            frame,
            contract=contract,
            axis_enums=axis_enums,
            origin=origin_for_error,
            backend=self,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rows_to_frame(rows: list[list[Any]], cols: list[str]) -> pl.DataFrame:
    """Build an all-Utf8 polars frame from string-row data.

    The legacy CSVs are pure text — every cell is the result of
    :func:`format_scalar_for_csv` or a raw entity name.  Reading them
    back yields :class:`polars.Utf8` columns; we materialise the
    frames with the same schema so downstream consumers see byte-
    identical bytes when the Provider is snapshotted to disk.

    A row with fewer cells than ``cols`` is padded with empty strings;
    rows with more cells are truncated.  In practice every row already
    matches ``len(cols)`` — the defensive padding documents the
    invariant.
    """
    schema = {c: pl.Utf8 for c in cols}
    if not rows:
        return pl.DataFrame(schema=schema)
    n = len(cols)
    norm: list[list[str]] = []
    for r in rows:
        if len(r) == n:
            norm.append([_to_str(x) for x in r])
        elif len(r) < n:
            norm.append([_to_str(x) for x in r] + [""] * (n - len(r)))
        else:
            norm.append([_to_str(x) for x in r[:n]])
    # Build column-oriented dict for cheap pl.DataFrame construction.
    data: dict[str, list[str]] = {c: [] for c in cols}
    for r in norm:
        for c, v in zip(cols, r):
            data[c].append(v)
    return pl.DataFrame(data, schema=schema)


def _to_str(x: Any) -> str:
    """Coerce a row cell to its CSV-form string.  Already-strings pass
    through; everything else goes via ``str(...)``.
    """
    if isinstance(x, str):
        return x
    if x is None:
        return ""
    return str(x)


__all__ = ["SpineDBBackend"]
