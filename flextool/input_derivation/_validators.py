"""SpineDB → Provider pipeline validators.

Each validator consumes a :class:`spinedb_api.DatabaseMapping` (or a
:class:`SpineDBBackend` exposing one as ``._db``) and either:

* raises :class:`FlexToolConfigError` for hard misconfigurations that
  would silently corrupt the solve (e.g. zero ``step_duration``), or
* logs a warning for soft inconsistencies (e.g. ``output_*: yes`` set
  on a group with no members of the required class).

The validators live as their own module because:

1. They are pure code paths with no derivation outputs — they don't
   ``provider.put`` anything.
2. They are called from :func:`flextool.input_derivation.run` early
   enough that a hard failure surfaces before the cascade builds any
   frames against malformed input.
3. They share the :func:`_get_commodity_price_methods` helper but are
   otherwise unrelated to the EAV → tabular materialiser specs in
   :mod:`flextool.input_derivation._specs`.

Pre-Step-2.5 these functions lived in the 2356-LOC ``input_writer.py``
monolith; Step 2.5 item 6 extracted them here.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from collections.abc import Iterable

import spinedb_api as api

from flextool.engine.input._solve_state import FlexToolConfigError
from flextool.input_derivation._method_constants import METHOD_2WAY_1VAR


__all__ = [
    "validate_timeline_timestep_duration",
    "validate_capacity_margin_groups",
    "validate_group_limit_flags",
    "validate_non_sync_one_variable_connections",
    "validate_min_load_efficiency_inputs",
    "validate_ladder_methods",
    "validate_group_output_memberships",
    "validate_connection_node_memberships",
    "validate_connection_delay",
    "validate_unit_input_shares",
    "validate_output_min_coeff",
    "validate_node_inflow_not_period",
    "validate_time_series_keys",
]


def _get_commodity_price_methods(db) -> dict[str, str]:
    """Return ``{commodity: price_method}`` for every commodity whose
    ``price_method`` is set.  Commodities without the param default to
    ``'price'`` in the mod (and do not appear here).
    """
    out: dict[str, str] = {}
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_method",
    ):
        if pv["type"] is None:
            continue
        out[pv["entity_byname"][0]] = str(pv["parsed_value"])
    return out


def validate_timeline_timestep_duration(db) -> None:
    """Raise FlexToolConfigError if any timeline entity is missing its
    ``timestep_duration`` map.  Without it, ``step_duration`` silently
    falls to 0 throughout the model and every time-weighted quantity
    (balances, costs, ramps) collapses to zero.  There is no sensible
    default, so the value must be present on every timeline.
    """
    timelines = [ent["entity_byname"][0]
                 for ent in db.find_entities(entity_class_name="timeline")]
    if not timelines:
        return
    have_duration: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="timeline",
        parameter_definition_name="timestep_duration",
    ):
        if pv["type"] is None:
            continue
        have_duration.add(pv["entity_byname"][0])
    missing = [t for t in timelines if t not in have_duration]
    if missing:
        raise FlexToolConfigError(
            "timeline 'timestep_duration' is not set for: "
            + ", ".join(sorted(missing))
            + ".  Every timeline needs a Map(timestep -> duration_in_hours); "
              "without it all time-weighted quantities collapse to zero."
        )


def _pv_rows(src, entity_class: str, parameter: str) -> Iterable:
    """Parameter-value items of ``(entity_class, parameter)``.  ``src`` is a ``SpineDBBackend`` (its bulk
    ``(class, param)`` index, built once per backend, through ``parameter_value_items``) or a ``DatabaseMapping``."""
    if hasattr(src, "parameter_value_items"):
        return src.parameter_value_items(entity_class, parameter)
    return src.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter)


def _parsed_then_evict(pv):
    """Parse ``pv``'s value and drop spinedb-api's cached parse right away (Track A eviction contract, see
    ``SpineDBBackend.parameter_values``), so a validator never keeps every series of a class parsed at once."""
    value = pv["parsed_value"]
    item = getattr(pv, "mapped_item", None)
    if item is not None:
        item._parsed_value = None
    return value


def _map_keys_all_levels(value, out: set[str]) -> None:
    """Add every index key of a (nested) Map to ``out``; Map values that are not Maps are not collected."""
    if not isinstance(value, api.Map):
        return
    for key, sub in zip(value.indexes, value.values):
        out.add(str(key))
        _map_keys_all_levels(sub, out)


_SOLVE_PERIOD_PARAMS = ("realized_periods", "invest_periods", "realized_invest_periods", "fix_storage_periods")


def _period_and_timestep_names(src) -> tuple[set[str], set[str]]:
    """``(periods, timesteps)`` of the scenario: see ``validate_node_inflow_not_period``."""
    periods, timelines = _scenario_time_vocab(src)
    timesteps: set[str] = set()
    for steps in timelines.values():
        timesteps.update(steps)
    return periods, timesteps


def _scenario_periods(src) -> set[str]:
    """The period names of the scenario: see ``validate_node_inflow_not_period``."""
    periods: set[str] = set()
    for pv in _pv_rows(src, "solve", "period_timeset"):
        if pv["type"] == "map":
            _map_keys_all_levels(_parsed_then_evict(pv), periods)
    for par in _SOLVE_PERIOD_PARAMS:
        for pv in _pv_rows(src, "solve", par):
            if pv["type"] == "array":
                periods.update(str(v) for v in _parsed_then_evict(pv).values)
            elif pv["type"] == "map":
                _map_keys_all_levels(_parsed_then_evict(pv), periods)
    return periods


def _keys_all_periods(indexes, periods: set[str], timesteps: set[str]) -> bool:
    """True when ``indexes`` is non-empty, every key is a period name and none is a timestep name.  Stops at the
    first key that is not a period or is a timestep, so a time series exits at its first key."""
    if len(indexes) == 0:
        return False
    for key in indexes:
        name = str(key)
        if name not in periods or name in timesteps:
            return False
    return True


def validate_node_inflow_not_period(src, vocab: tuple[set[str], dict[str, list[str]]] | None = None) -> None:
    """Raise FlexToolConfigError when a node's ``inflow`` is keyed by period.

    ``inflow`` is a constant, a time series (1-D map over time) or a stochastic 3-D map (branch, analysis time,
    time); it cannot be period-based (docs/reference/entities_and_time.md, 'Timesteps and periods').  A period-keyed value is not
    read by any path: before this check it silently became inflow 0.  Flagged:
      * a 1-D map with index_name 'period';
      * any 2-D map (the schema has no rank-2 inflow; a period x time map is the usual intent);
      * a 1-D map with another index_name whose keys are all period names and none a timestep name.

    Period names: the keys of every ``solve.period_timeset``, the entries of the 1-D period arrays of a solve
    (``realized_periods``, ``invest_periods``, ``realized_invest_periods``, ``fix_storage_periods``) and the index
    keys at every level of their Map variants (never Map values: nested-investment variants map period to a Map).
    Timestep names: the keys of every ``timeline.timestep_duration``.  Both sets are built first (only when some
    inflow is a map), then each map is classified right after its parse, which is evicted at once: the rank is
    read without parsing, rank-1 and rank-2 maps are parsed (rank 2 for its first-level keys only), and only the
    first 3 keys of an offending map are kept.

    ``src`` is the ``SpineDBBackend`` of ``input_derivation.run`` (or a scenario-filtered ``DatabaseMapping``);
    ``vocab`` the ``(periods, timelines)`` of :func:`_scenario_time_vocab`, shared with
    :func:`validate_time_series_keys` (built here when not given).
    """
    to_rank = api.parameter_value.from_database_to_dimension_count
    inflow_maps = [pv for pv in _pv_rows(src, "node", "inflow") if pv["type"] == "map"]
    if not inflow_maps:
        return
    if vocab is None:
        periods, timesteps = _period_and_timestep_names(src)
    else:
        periods = vocab[0]
        timesteps = {step for steps in vocab[1].values() for step in steps}

    problems: list[str] = []
    for pv in inflow_maps:
        rank = to_rank(pv["value"], pv["type"])
        if rank not in (1, 2):
            continue
        value = _parsed_then_evict(pv)
        indexes = value.indexes
        if rank == 2:
            reason = "a 2-D map"
        elif value.index_name == "period":
            reason = "index_name 'period'"
        elif _keys_all_periods(indexes, periods, timesteps):
            reason = "all map keys are period names"
        else:
            continue
        shown = ", ".join(str(k) for k in indexes[:3]) + (", ..." if len(indexes) > 3 else "")
        problems.append(
            f"node '{pv['entity_byname'][0]}' inflow (alternative '{pv['alternative_name']}') is keyed by period "
            f"({reason}" + (f"; first-level keys: {shown}" if len(indexes) else "") + ")")
    if problems:
        raise FlexToolConfigError(
            "\n".join(problems)
            + "\ninflow is a constant or a time series and cannot be period-based; to vary demand by period use "
              "annual_flow (constant or period) with an inflow_method that scales to it (scale_to_annual_flow, "
              "scale_in_proportion, scale_to_annual_and_peak_flow).")


# ---------------------------------------------------------------------------
# Time-series keys that are not time steps of any timeline (#205)

# The start of a nested value in a Map's database JSON, and the label of a level.
_RAW_NESTED = re.compile(rb'\{\s*"type"')
_RAW_LABEL = re.compile(rb'"index_name"\s*:\s*"([^"]*)"')
# The quoted strings of a Map's database JSON that are not keys.
_RAW_STRUCTURE = frozenset((b"type", b"map", b"index_type", b"str", b"date_time", b"duration", b"index_name",
                            b"data", b"rank"))
# Database JSON whose keys the fast path cannot see: an escape (the quotes no longer pair), a time series, a float
# index (unquoted keys; any ``float`` in the text takes the parse, one scan), a dict-form ``data``.  (A date_time
# or duration index has quoted keys: a non-canonical stamp is not a time step's bytes, so it takes the parse.)
_RAW_SLOW_MARKERS = (b"\\", b"time_series", b"float", b'"data": {', b'"data":{')


def _scenario_time_vocab(src) -> tuple[set[str], dict[str, list[str]]]:
    """``(periods, {timeline: [step, ...]})`` of the scenario: the period names of
    ``validate_node_inflow_not_period`` and the steps of every ``timeline.timestep_duration``, in timeline order
    (keys stringified as ``str(key)``, the timeline reader's form)."""
    periods = _scenario_periods(src)
    timelines: dict[str, list[str]] = {}
    for pv in _pv_rows(src, "timeline", "timestep_duration"):
        if pv["type"] in ("map", "time_series"):
            timelines[pv["entity_byname"][0]] = [str(k) for k in _parsed_then_evict(pv).indexes]
    return periods, timelines


class _DatabaseTimeVocab:
    """The periods and timelines of the **whole** database (every alternative, so every scenario), read on first
    use only (a run whose values all classify on the scenario's own vocabulary never opens it).

    :func:`validate_time_series_keys` uses it for two cases that are plain data of another scenario, not errors:
    an unnamed map level keyed by periods of another scenario (a period level: that entity gets no value in this
    solve, docs/reference/entities_and_time.md), and a series keyed on the steps of a timeline that is not one of
    this scenario's.  ``src`` is the scenario-filtered ``SpineDBBackend``; any other ``src`` (or a backend without
    a scenario) is taken as the whole database already.
    """

    def __init__(self, src, periods: set[str], timelines: dict[str, list[str]]) -> None:
        self._src = src
        self._loaded = False
        self.periods: set[str] = set(periods)
        # (timeline, alternative, steps) of the timestep_duration values that are not this scenario's timelines
        self.other_timelines: list[tuple[str, str, frozenset[str]]] = []
        self._own = {name: frozenset(steps) for name, steps in timelines.items()}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        url = getattr(self._src, "db_url", None)
        if not url or not getattr(self._src, "scenario_name", None):
            return
        from flextool.update_flextool.db_handle import open_database_mapping

        with open_database_mapping(url) as db:
            self.periods |= _scenario_periods(db)
            for pv in _pv_rows(db, "timeline", "timestep_duration"):
                if pv["type"] not in ("map", "time_series"):
                    continue
                name = pv["entity_byname"][0]
                steps = frozenset(str(k) for k in pv["parsed_value"].indexes)
                if self._own.get(name) != steps:
                    self.other_timelines.append((name, pv["alternative_name"], steps))

    def all_periods(self) -> set[str]:
        self._load()
        return self.periods

    def other_timelines_holding(self, keys) -> list[tuple[str, str]] | None:
        """``[(timeline, alternative), ...]`` of the timelines outside this scenario whose steps together hold
        every key of ``keys``; None when some key is a step of none of them."""
        self._load()
        left = set(keys)
        found: list[tuple[str, str]] = []
        for name, alt, steps in self.other_timelines:
            if not left.isdisjoint(steps):
                found.append((name, alt))
                left -= steps
                if not left:
                    return found
        return None


def _raw_keys_on_timeline(raw: bytes, periods_b: frozenset[bytes], base_b: frozenset[bytes],
                          unions: dict[frozenset[bytes], frozenset[bytes]]) -> bool:
    """Fast path of :func:`validate_time_series_keys`: True when the database JSON of a str-indexed Map shows,
    without parsing it, that no time key can be off the timeline: every quoted string is a time step, a word of the
    JSON structure, a level label, or a period name that keys a nested map (the period level of a period x time
    map).  A C-speed split and set comparison (the quoted strings are every other piece between quotes); False
    means "parse and classify" (not that a key is off).  An escape, a float index, a dict-form ``data``, a string
    value or a time series inside the value always takes the parse.  Known limit: a time key that is one of those
    words or a period name of the map's period level is not reported.

    ``base_b`` is the time steps, the structure words and the usual labels (built once per check); ``unions``
    caches ``base_b`` plus the other words of a value (its period keys, another label), so the values of the same
    periods share one set."""
    for marker in _RAW_SLOW_MARKERS:
        if marker in raw:
            return False
    quoted = raw.split(b'"')[1::2]
    extra: set[bytes] = set()
    # The keys of the nested maps: the last key before each nested value (few, one per period).
    for match in _RAW_NESTED.finditer(raw):
        pos = match.start()
        open_at = raw.rfind(b'["', 0, pos)
        close_at = raw.find(b'"', open_at + 2, pos)
        if open_at < 0 or close_at < 0:
            return False
        key = raw[open_at + 2:close_at]
        if key not in periods_b:
            return False
        extra.add(key)
    extra.difference_update(base_b)
    if not extra:
        allowed = base_b
    else:
        frozen = frozenset(extra)
        allowed = unions.get(frozen)
        if allowed is None:
            allowed = unions[frozen] = base_b | frozen
    if allowed.issuperset(quoted):
        return True
    # A level label other than time / period / t (rare): read the labels only now and try again.
    labels = set(_RAW_LABEL.findall(raw)).difference(allowed)
    return bool(labels) and allowed.union(labels).issuperset(quoted)


@functools.lru_cache(maxsize=65536)
def _index_key_str(key: str, index_type: str) -> str:
    """A Map key as the reader sees it: ``str()`` of the parsed index (spinedb_api parses a ``date_time`` or
    ``duration`` key, so ``2050-01-01 00:00:00`` and ``2050-01-01T00:00`` read as ``2050-01-01T00:00:00``)."""
    try:
        if index_type == "date_time":
            return str(api.DateTime(key))
        if index_type == "duration":
            return str(api.Duration(key))
    except Exception:  # noqa: BLE001 - a key spinedb_api cannot parse is compared as written
        return str(key)
    return str(key)


def _value_levels(value) -> list[tuple[str | None, list]]:
    """``[(index_name, keys), ...]`` per level of a parsed database JSON value (outer to inner): the depth and the
    labels follow the first entry of each level, as the reader's ``_discover_index_cols`` does, and the keys of a
    level are those of all its maps.  Keys are stringified as the reader does (``str()`` of the parsed index:
    :func:`_index_key_str`); a time series level is ``("time", stamps)``."""
    levels: list[tuple[str | None, list]] = []
    current = [value]
    while current:
        first = current[0]
        kind = first.get("type") if isinstance(first, dict) else None
        if kind == "time_series":
            keys: list = []
            for ts in current:
                if isinstance(ts, dict):
                    parsed = api.from_database(json.dumps(ts).encode(), "time_series")
                    keys.extend(str(k) for k in parsed.indexes)
            levels.append(("time", keys))
            break
        if not isinstance(first, dict) or "data" not in first:
            break
        keys = []
        children: list = []
        for m in current:
            if not isinstance(m, dict) or m.get("type") == "time_series":
                continue
            index_type = m.get("index_type")
            parse = index_type in ("date_time", "duration")
            data = m.get("data")
            entries = data.items() if isinstance(data, dict) else data or []
            for entry in entries:
                k, v = entry[0], entry[1]
                keys.append(_index_key_str(str(k), index_type) if parse else str(k))
                if isinstance(v, dict) and v.get("type") in ("map", "time_series"):
                    children.append(v)
        levels.append((first.get("index_name"), keys))
        current = children
    return levels


def _time_levels(levels: list[tuple[str | None, list]], allowed, periods: set[str],
                 timesteps: set[str], all_periods=None) -> list[int]:
    """The positions of the time levels in ``levels`` (:func:`_value_levels`), classified as the readers do:
    by depth, label and the parameter's allowed shapes (``PARAM_ALLOWED_SHAPES``), then by the keys.

    - 3 levels: a stochastic map (branch, analysis time, time) by position; the branch level is never checked.
    - A level labelled ``period`` is not checked, one labelled ``time`` / ``t`` is; another label is left to the
      shape checks (not checked).
    - An unnamed level takes the allowed shapes' unique label at that depth and position; otherwise it is a period
      level when its keys hit the periods and no time step, or (``all_periods``, a callable giving the periods of
      the whole database) hit no time step but a period of another scenario: such an entity gets no value in this
      solve, as documented; else it is a time level.  On 2 levels an unnamed outer level that hits no period is
      not checked (a branch level of a 2-level stochastic profile).
    """
    from flextool.engine.input._param_shapes import _infer_silent_default_labels, _normalise_label

    depth = len(levels)
    if depth == 3:
        return [1, 2]
    if depth not in (1, 2):
        return []
    labels = [_normalise_label(name) for name, _keys in levels]
    if allowed:
        labels = _infer_silent_default_labels(labels, allowed)
    out: list[int] = []
    for pos, (label, (_name, keys)) in enumerate(zip(labels, levels)):
        if label in ("time", "t"):
            out.append(pos)
        elif label is None:
            key_set = set(keys)
            hits_step = not key_set.isdisjoint(timesteps)
            hits_period = not key_set.isdisjoint(periods)
            if hits_period and not hits_step:
                continue
            if not hits_step and all_periods is not None and not key_set.isdisjoint(all_periods()):
                continue
            if depth == 2 and pos == 0 and not hits_period and not hits_step:
                continue
            out.append(pos)
    if depth == 2 and len(out) == 2:
        # Both unnamed and both look like time: the outer one is the period level of a period x time map.
        out = [1]
    return out


def _span(steps: list[str]) -> str:
    return f"{steps[0]} .. {steps[-1]}" if steps else "no steps"


def _timelines_text(timelines: dict[str, list[str]]) -> str:
    items = list(timelines.items())
    if len(items) == 1:
        name, steps = items[0]
        return f"Timeline {name!r}: {len(steps)} steps, {_span(steps)}."
    shown = "; ".join(f"{name!r}: {len(steps)} steps, {_span(steps)}" for name, steps in items[:3])
    more = f"; and {len(items) - 3} more" if len(items) > 3 else ""
    return f"Timelines {shown}{more}."


_ISO_STAMP = re.compile(r"^(\d{4})-\d\d-\d\d[T ]")


def _off_keys_hint(keys: list[str], lower_steps: dict[str, str], timelines: dict[str, list[str]]) -> str:
    """Why the keys of a value may be off: a case-insensitive match (``did you mean``), or date-time stamps of
    another year than the timelines'.  Empty when neither applies."""
    for k in keys:
        match = lower_steps.get(k.lower())
        if match is not None:
            return f" Did you mean {match!r} (for {k!r})? Time step names are case sensitive."
    key_years = {m.group(1) for k in keys[:50] if (m := _ISO_STAMP.match(k))}
    if key_years:
        step_years = {m.group(1) for steps in timelines.values() for s in steps[:1] + steps[-1:]
                      if (m := _ISO_STAMP.match(s))}
        if step_years and key_years.isdisjoint(step_years):
            return (f" Its stamps are of year {', '.join(sorted(key_years))}, the timelines' of "
                    f"{', '.join(sorted(step_years))}.")
    return ""


def validate_time_series_keys(src, vocab: tuple[set[str], dict[str, list[str]]] | None = None) -> None:
    """Warn about time-series keys that are not a time step of any timeline; error when none of a value's are (#205).

    The readers join a time series with the solve's time steps, so a key that is not a time step of a timeline is
    dropped without a word: hourly data on a 3 h timeline, a different year's stamps, a different case.  This check
    runs once per run (``input_derivation.run``) for every parameter of ``TIME_AGGREGATION`` (the complete
    time-series registry), on the Map and time series values, against the steps of **all** the scenario's
    timelines (a database that keeps series for two timelines gives no noise):

    - some keys of a value are off → one warning per (class, parameter), printed once, naming up to 3 entities,
      the number of off keys and the first 5;
    - no time key of a value is on a timeline → :class:`FlexToolConfigError` (every value of it would be dropped
      and the parameter would fall back to its default), naming the entity, the parameter, the alternative and the
      first keys (up to 5 values shown), with a case or year hint when one applies;
    - ... unless every key is a time step of a timeline that is not one of this scenario's (another alternative's
      ``timestep_duration``): plain data of another scenario → a warning (the value is not used).

    The time levels are classified as the readers do (:func:`_time_levels`): a stochastic branch name or a period
    key is never checked; an unnamed level keyed by periods of another scenario is a period level (no value in
    this solve, as documented). Date-time and duration keys are compared as the reader parses them
    (:func:`_index_key_str`). Known limit: a period-level key that is not a period of the scenario is not reported.

    Performance: a str-indexed Map is first read as raw database JSON (:func:`_raw_keys_on_timeline`, a C-speed
    regex and set comparison); only a value with a key that is not a period or a time step is parsed. Each
    distinct raw value is checked once (copies of one profile on many entities cost a dict lookup); the cache
    holds the database mapping's own bytes, so it adds no copy of them. The whole database's periods and timelines
    are read only when a value needs them (:class:`_DatabaseTimeVocab`).

    ``vocab`` is the ``(periods, timelines)`` of :func:`_scenario_time_vocab` (built here when not given).
    """
    from flextool.engine.input._param_shapes import PARAM_ALLOWED_SHAPES
    from flextool.engine.input._solve_step_aggregation import TIME_AGGREGATION

    periods, timelines = vocab if vocab is not None else _scenario_time_vocab(src)
    timesteps: set[str] = set()
    for steps in timelines.values():
        timesteps.update(steps)
    if not timesteps:
        return
    periods_b = frozenset(p.encode() for p in periods)
    base_b = frozenset(t.encode() for t in timesteps) | _RAW_STRUCTURE | {b"time", b"period", b"t"}
    unions: dict[frozenset[bytes], frozenset[bytes]] = {}
    database = _DatabaseTimeVocab(src, periods, timelines)
    lower_steps: dict[str, str] | None = None
    # raw value -> True (fast path: nothing off), or per allowed shapes -> the value's time keys (in order)
    fast_seen: dict[bytes, bool] = {}
    keys_seen: dict[tuple[bytes, frozenset | None], dict[str, None]] = {}

    errors: list[str] = []
    other_timeline_warnings: list[str] = []
    warnings: list[str] = []
    for (cls, par) in TIME_AGGREGATION:
        allowed = PARAM_ALLOWED_SHAPES.get((cls, par))
        allowed_key = frozenset(allowed) if allowed else None
        off_entities: list[str] = []
        all_keys: dict[str, None] = {}
        off_keys: dict[str, None] = {}
        for pv in _pv_rows(src, cls, par):
            kind = pv["type"]
            if kind not in ("map", "time_series"):
                continue
            raw = pv["value"]
            if isinstance(raw, str):
                raw = raw.encode()
            if kind == "map":
                fast = fast_seen.get(raw)
                if fast is None:
                    fast = fast_seen[raw] = _raw_keys_on_timeline(raw, periods_b, base_b, unions)
                if fast:
                    continue
            keys = keys_seen.get((raw, allowed_key))
            if keys is None:
                if kind == "time_series":
                    levels = [("time", [str(k) for k in api.from_database(raw, "time_series").indexes])]
                else:
                    levels = _value_levels(json.loads(raw))
                keys = {}
                for pos in _time_levels(levels, allowed, periods, timesteps, database.all_periods):
                    keys.update(dict.fromkeys(levels[pos][1]))
                keys_seen[(raw, allowed_key)] = keys
            if not keys:
                continue
            off = [k for k in keys if k not in timesteps]
            if not off:
                continue
            entity = "/".join(str(b) for b in pv["entity_byname"])
            where = f"{cls} {entity!r} parameter {par!r} (alternative {pv['alternative_name']!r})"
            if len(off) == len(keys):
                first = ", ".join(list(keys)[:3]) + (" ..." if len(keys) > 3 else "")
                others = database.other_timelines_holding(keys)
                if others is not None:
                    names = ", ".join(f"{name!r} (alternative {alt!r})" for name, alt in others[:2])
                    other_timeline_warnings.append(
                        f"Warning: {where}: its keys ({first}) are time steps of timeline {names}, which is not a "
                        "timeline of this scenario; the value is not used.")
                    continue
                if lower_steps is None:
                    lower_steps = {}
                    for s in timesteps:
                        lower_steps.setdefault(s.lower(), s)
                errors.append(
                    f"{where}: none of its {len(keys)} time keys ({first}) is a time step of a timeline."
                    + _off_keys_hint(list(keys), lower_steps, timelines))
                continue
            off_entities.append(entity)
            all_keys.update(keys)
            off_keys.update(dict.fromkeys(off))
        if off_entities:
            names = ", ".join(repr(e) for e in off_entities[:3])
            if len(off_entities) > 3:
                names += f" and {len(off_entities) - 3} more entities"
            first = list(off_keys)[:5]
            rest = len(off_keys) - len(first)
            shown = ", ".join(first) + (f" and {rest} more" if rest else "")
            warnings.append(
                f"Warning: {cls} {names} parameter {par!r}: {len(off_keys)} of {len(all_keys)} distinct time keys "
                f"are not time steps of any timeline ({shown}); their values are not used. "
                f"{_timelines_text(timelines)}")
    if errors:
        more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
        timeline_list = ", ".join(f"{name}: {_span(steps)}" for name, steps in list(timelines.items())[:3])
        raise FlexToolConfigError(
            "Time series with no key on a timeline: " + " ".join(errors[:5]) + more
            + f" (timelines {timeline_list}). Rename the keys to time steps of a timeline, or remove the value.")
    for text in other_timeline_warnings + warnings:
        print(text, flush=True)


def validate_capacity_margin_groups(db, logger: logging.Logger) -> None:
    """Storage nodes are excluded from the capacity-margin constraint.
    Raise if any capacity_margin_method='manual' group contains *only*
    storage nodes (constraint would have no valid members); warn if a
    mix is present.
    """
    capacity_margin_groups: dict[str, list[str]] = {}
    for pv in db.find_parameter_values(
        entity_class_name="group", parameter_definition_name="capacity_margin_method",
    ):
        if pv["parsed_value"] == "manual":
            capacity_margin_groups[pv["entity_byname"][0]] = []
    if not capacity_margin_groups:
        return
    for ent in db.find_entities(entity_class_name="group__node"):
        g, n = ent["entity_byname"][0], ent["entity_byname"][1]
        if g in capacity_margin_groups:
            capacity_margin_groups[g].append(n)
    storage_nodes: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="node", parameter_definition_name="node_type",
    ):
        if pv["parsed_value"] == "storage":
            storage_nodes.add(pv["entity_byname"][0])
    for g, nodes in capacity_margin_groups.items():
        storage_in_group = [n for n in nodes if n in storage_nodes]
        if storage_in_group and len(storage_in_group) == len(nodes):
            raise FlexToolConfigError(
                f"Capacity margin group '{g}' contains only storage nodes "
                f"({', '.join(storage_in_group)}). The capacity margin constraint "
                f"excludes storage nodes, so this group has no valid nodes."
            )
        elif storage_in_group:
            logger.warning(
                "Capacity margin group '%s' contains storage nodes (%s) which will "
                "be excluded from the capacity margin constraint.",
                g, ', '.join(storage_in_group),
            )


def _explicit_values(db, entity_class: str, parameter: str) -> dict[str, list[float]]:
    """``{entity: [numeric elements]}`` of the scenario's explicit values of a one-dimensional class's
    ``parameter`` (scalars and every value of a period / time / stochastic-time map)."""
    out: dict[str, list[float]] = {}
    for pv in db.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter):
        if pv["type"] is None:
            continue
        out[pv["entity_byname"][0]] = _numeric_values(pv["parsed_value"])
    return out


def _entities_with_value(db, entity_class: str, parameter: str, value: str) -> set[str]:
    """Entities of ``entity_class`` whose explicit ``parameter`` value is ``value``."""
    return {pv["entity_byname"][0]
            for pv in db.find_parameter_values(entity_class_name=entity_class, parameter_definition_name=parameter)
            if pv["type"] is not None and pv["parsed_value"] == value}


# (limit parameter, method/flag parameter, value that switches the limit on)
_GROUP_LIMIT_FLAGS: tuple[tuple[str, str, str], ...] = (
    ("capacity_margin", "capacity_margin_method", "manual"),
    ("inertia_limit", "has_inertia", "yes"),
    ("non_synchronous_limit", "has_non_synchronous", "yes"),
)

# What a flagged group without the limit gets (the model reads the limit as 0).
_GROUP_LIMIT_ABSENT: dict[str, str] = {
    "capacity_margin": "its capacity must cover the peak net load (margin 0)",
    "inertia_limit": "it has no inertia requirement",
    "non_synchronous_limit": "it reads as 0, so no non-synchronous infeed is allowed (violations are penalised)",
}


def validate_group_limit_flags(db, logger: logging.Logger) -> None:
    """Warn about group limits and their method / flag that do not go together.

    The capacity margin, inertia and non-synchronous constraints are built for the groups whose
    ``capacity_margin_method`` is ``manual`` / ``has_inertia`` is ``yes`` / ``has_non_synchronous``
    is ``yes``, and nowhere else.  Warnings (``db`` is scenario-filtered):

    * a non-zero limit on a group without its method / flag: the limit is ignored;
    * the method / flag on a group with no limit value at all: the model reads the limit as 0.  An
      explicit 0 is a deliberate input and is not reported.
    """
    for limit, flag, on in _GROUP_LIMIT_FLAGS:
        flagged = _entities_with_value(db, "group", flag, on)
        values = _explicit_values(db, "group", limit)
        for g in sorted(values):
            if g not in flagged and any(v != 0.0 for v in values[g]):
                logger.warning(
                    "Group '%s' has %s but %s is not %s: the limit is ignored.", g, limit, flag, on)
        for g in sorted(flagged - set(values)):
            logger.warning(
                "Group '%s' has %s = %s but no %s: %s.", g, flag, on, limit, _GROUP_LIMIT_ABSENT[limit])


def validate_non_sync_one_variable_connections(db, provider, logger: logging.Logger) -> None:
    """Warn about one-variable connections across the boundary of a group with a non-synchronous limit.

    The non-synchronous limit counts the flows of a connection with one node in the group both ways: the
    flow into the group as infeed, the flow out of it as supply.  A one-variable connection
    (``no_losses_no_variable_cost``, and the lines of a ``dc_power_flow_with_angles`` group, which use it
    internally) can flow both ways at once at no cost, so the optimiser can raise the group's supply side by
    up to the line capacity and loosen the limit.  Warnings (``db`` is scenario-filtered), one per connection
    and group, for groups with ``has_non_synchronous = yes``: the connection is classified ``2way_1var`` and
    exactly one of its nodes is in the group.  ``is_DC`` connections are left out: their infeed into the
    group counts as non-synchronous, so flowing both ways tightens the limit instead.  The same loophole
    exists for a ``regular`` connection (documented); ``exact`` closes it.
    """
    groups = _entities_with_value(db, "group", "has_non_synchronous", "yes")
    if not groups:
        return
    methods = _process_methods(provider)
    one_var = {c for c, m in methods.items() if m in METHOD_2WAY_1VAR}
    if not one_var:
        return
    is_dc = _entities_with_value(db, "connection", "is_DC", "yes")
    dc_pf: set[str] = set()
    if provider is not None and provider.has("input/connection_dc_power_flow"):
        frame = provider.get("input/connection_dc_power_flow")
        if frame is not None and frame.height > 0:
            dc_pf = set(frame.get_column(frame.columns[0]).cast(str).to_list())
    group_nodes: dict[str, set[str]] = {}
    for ent in db.find_entities(entity_class_name="group__node"):
        group_nodes.setdefault(ent["entity_byname"][0], set()).add(ent["entity_byname"][1])
    for ent in sorted(db.find_entities(entity_class_name="connection__node__node"),
                      key=lambda e: tuple(e["entity_byname"])):
        conn, n1, n2 = ent["entity_byname"][:3]
        if conn not in one_var or conn in is_dc:
            continue
        for g in sorted(groups):
            nodes = group_nodes.get(g, set())
            if (n1 in nodes) == (n2 in nodes):
                continue
            if conn in dc_pf:
                logger.warning(
                    "DC power-flow line '%s' crosses the boundary of group '%s' with a non-synchronous limit: its "
                    "flows both ways count, and simultaneous flow both ways can loosen the limit at no cost.",
                    conn, g)
            else:
                logger.warning(
                    "Connection '%s' crosses the boundary of group '%s' with a non-synchronous limit: its flows "
                    "both ways count, and simultaneous flow both ways can loosen the limit at no cost: use "
                    "transfer_method exact.", conn, g)


def validate_min_load_efficiency_inputs(db, logger: logging.Logger) -> None:
    """Check the conversion curve inputs of ``min_load_efficiency`` units.

    Errors (:class:`FlexToolConfigError`), for units with ``conversion_method = min_load_efficiency``, over every
    value of a scalar or a period / time / stochastic-time map:

    * ``efficiency_at_min_load`` not above 0 (the fuel use at minimum load would be infinite);
    * ``min_load`` not below 1 (the curve between minimum and full load would have no length).

    Warning: a unit with a ``min_load`` value above 0 in any value (over the whole database, not one solve) and no
    ``efficiency_at_min_load``.  Such a unit uses its ``efficiency`` at minimum load too (a flat curve: the fuel
    use is output / efficiency, as with constant_efficiency, while the minimum load still applies).  Cells that an
    ``efficiency_at_min_load`` map leaves out get the efficiency in the same way, without a warning.  ``db`` is
    scenario-filtered.
    """
    mle = _entities_with_value(db, "unit", "conversion_method", "min_load_efficiency")
    if not mle:
        return
    min_load = _explicit_values(db, "unit", "min_load")
    eta_min = _explicit_values(db, "unit", "efficiency_at_min_load")
    errors: list[str] = []
    for unit in sorted(mle):
        if any(v <= 0.0 for v in eta_min.get(unit, [])):
            errors.append(f"unit '{unit}': efficiency_at_min_load {_fmt(eta_min[unit])} must be above 0")
        if any(v >= 1.0 for v in min_load.get(unit, [])):
            errors.append(f"unit '{unit}': min_load {_fmt(min_load[unit])} must be below 1 for a "
                          "min_load_efficiency unit (use constant_efficiency for a unit that only runs at full load)")
    if errors:
        raise FlexToolConfigError("min_load_efficiency inputs: " + "; ".join(errors) + ".")
    with_min_load = {u for u, vals in min_load.items() if any(v > 0.0 for v in vals)}
    for unit in sorted((mle & with_min_load) - set(eta_min)):
        logger.warning(
            "Unit '%s' uses min_load_efficiency with min_load > 0 but has no efficiency_at_min_load: its "
            "efficiency is used at minimum load too (a flat curve; the minimum load still applies).", unit)


def validate_ladder_methods(db, logger: logging.Logger) -> None:
    """Raise FlexToolConfigError if any commodity declares a ladder
    ``price_method`` but does not have the corresponding ladder parameter
    set.  Runs before the ladder writers so errors name the offending
    commodity and expected parameter.
    """
    methods = _get_commodity_price_methods(db)
    ladder_methods = {"price_ladder_annual", "price_ladder_cumulative"}
    commodities_needing_ladder = {
        c: m for c, m in methods.items() if m in ladder_methods
    }
    if not commodities_needing_ladder:
        return

    # Collect commodities that HAVE each ladder param (non-None, non-empty).
    have_cumulative: set[str] = set()
    have_annual: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_cumulative",
    ):
        if pv["type"] is None:
            continue
        have_cumulative.add(pv["entity_byname"][0])
    for pv in db.find_parameter_values(
        entity_class_name="commodity",
        parameter_definition_name="price_ladder_annual",
    ):
        if pv["type"] is None:
            continue
        have_annual.add(pv["entity_byname"][0])

    for commodity, method in commodities_needing_ladder.items():
        expected_param = method  # parameter name matches method name
        if method == "price_ladder_cumulative" and commodity not in have_cumulative:
            raise FlexToolConfigError(
                f"commodity '{commodity}' has "
                f"price_method='price_ladder_cumulative' but no "
                f"'{expected_param}' value is set.  Add a "
                f"Map(tier -> {{price, quantity}}) on that parameter."
            )
        if method == "price_ladder_annual" and commodity not in have_annual:
            raise FlexToolConfigError(
                f"commodity '{commodity}' has "
                f"price_method='price_ladder_annual' but no "
                f"'{expected_param}' value is set.  Add either a 2d "
                f"Map(tier -> {{price, quantity}}) or a 3d "
                f"Map(period -> Map(tier -> {{price, quantity}}))."
            )


def validate_group_output_memberships(db, logger: logging.Logger) -> None:
    """Warn when a group-level output flag is set but the group lacks
    the membership class required for that output to produce any data.

    Three silent-no-op cases are detected (v58 vocabulary):

    * ``group.print_dispatch: yes`` with no ``group__node`` row
    * ``group.print_indicators: yes`` with no ``group__node`` row
    * ``flowGroup.flow_aggregator`` set to a non-``none`` method with no
      ``flowGroup__unit__node`` **or** ``flowGroup__connection__node`` row

    Additionally, a double-counting guard warns when a single flow arc
    ``(process, node)`` belongs to two or more *dispatch-bound* flowGroups
    (``flow_aggregator`` in ``{dispatch_plots_only, both}``): the arc would
    be summed once per such flowGroup into the nodeGroup dispatch table.

    Only warnings are emitted — a user may deliberately stage a partial
    configuration.
    """
    # Collect groups that are members of the relevant entity classes.
    groups_with_node_members: set[str] = set()
    for ent in db.find_entities(entity_class_name="group__node"):
        byname = ent["entity_byname"]
        if byname:
            groups_with_node_members.add(byname[0])

    flowgroups_with_flow_members: set[str] = set()
    for cls in ("flowGroup__unit__node", "flowGroup__connection__node"):
        for ent in db.find_entities(entity_class_name=cls):
            byname = ent["entity_byname"]
            if byname:
                flowgroups_with_flow_members.add(byname[0])

    # nodeGroup output flags: "yes" on the ``group`` class, need group__node.
    node_checks: list[tuple[str, str, set[str]]] = [
        ("print_dispatch", "group__node", groups_with_node_members),
        ("print_indicators", "group__node", groups_with_node_members),
    ]
    for param_name, required_members, membership_set in node_checks:
        for pv in db.find_parameter_values(
            entity_class_name="group", parameter_definition_name=param_name
        ):
            if pv["type"] is None:
                continue
            if pv["parsed_value"] != "yes":
                continue
            group_name = pv["entity_byname"][0]
            if group_name not in membership_set:
                logger.warning(
                    "Group '%s' has %s: yes but no %s members — output will be empty.",
                    group_name, param_name, required_members,
                )

    # flow_aggregator: an enum METHOD on the ``flowGroup`` class.  Any value
    # other than ``none`` requests dispatch bands and/or standalone
    # indicators, both of which need flowGroup__*__node members to produce
    # any data.
    #
    # We also record which flowGroups request dispatch *bands* — the values
    # ``dispatch_plots_only`` and ``both`` — for the overlap check below.
    _DISPATCH_BOUND = {"dispatch_plots_only", "both"}
    dispatch_bound_flowgroups: set[str] = set()
    for pv in db.find_parameter_values(
        entity_class_name="flowGroup", parameter_definition_name="flow_aggregator"
    ):
        if pv["type"] is None:
            continue
        if pv["parsed_value"] in (None, "none"):
            continue
        group_name = pv["entity_byname"][0]
        if pv["parsed_value"] in _DISPATCH_BOUND:
            dispatch_bound_flowgroups.add(group_name)
        if group_name not in flowgroups_with_flow_members:
            logger.warning(
                "flowGroup '%s' has flow_aggregator: %s but no "
                "flowGroup__unit__node or flowGroup__connection__node members "
                "— output will be empty.",
                group_name, pv["parsed_value"],
            )

    # Double-counting guard: a single flow arc ``(process, node)`` that
    # belongs to two (or more) dispatch-bound flowGroups (flow_aggregator in
    # {dispatch_plots_only, both}) contributes its flow as a band in the
    # nodeGroup dispatch table once per such flowGroup, so the dispatch
    # balance double-counts that arc.  Standalone-only overlap is legitimate
    # (separate aggregator series), so it is excluded here.  Only a warning
    # is emitted — the user may have intended overlapping membership.
    pn_to_aggregators: dict[tuple[str, str], list[str]] = {}
    for cls in ("flowGroup__unit__node", "flowGroup__connection__node"):
        for ent in db.find_entities(entity_class_name=cls):
            byname = ent["entity_byname"]
            if not byname or len(byname) < 3:
                continue
            flowgroup, process, node = byname[0], byname[1], byname[2]
            if flowgroup not in dispatch_bound_flowgroups:
                continue
            pn_to_aggregators.setdefault((process, node), []).append(flowgroup)
    for (process, node), flowgroups in pn_to_aggregators.items():
        if len(flowgroups) >= 2:
            logger.warning(
                "Flow (%s, %s) belongs to multiple flowGroups (%s) in effect "
                "double counting the flow in dispatch plots. Check "
                "flow_aggregator parameter.",
                process, node, ", ".join(sorted(flowgroups)),
            )


def validate_connection_node_memberships(db, logger: logging.Logger) -> None:
    """Warn when a ``connection`` entity has no ``connection__node__node``
    relationship defining its two endpoints.

    A connection's purpose is to transfer between two nodes; without a
    ``connection__node__node`` member it carries no source/sink rows, never
    enters ``process_source_sink``, and contributes nothing to any node
    balance.  If it nonetheless has an ``invest_method`` it still becomes a
    (degenerate, always-zero) investment variable in the LP, surfacing as a
    ``v_invest`` column the output post-processor has to tolerate (see
    ``read_parameters._entity_universe``).  ``db`` here is already
    scenario-filtered, so this fires either because the connection was
    never wired to its nodes, or because one of its endpoint nodes is not
    active in the current scenario (so the ``connection__node__node`` row
    is filtered out while the connection itself survives).  We cannot tell
    the two apart from the filtered set — the connection→node mapping
    lived only in the now-absent ``connection__node__node`` row — so the
    warning names both possibilities and tells the user to check the data.

    Only a warning is emitted: the solve proceeds, treating the connection
    as a no-op.
    """
    connected: set[str] = set()
    for ent in db.find_entities(entity_class_name="connection__node__node"):
        byname = ent["entity_byname"]
        if byname:
            connected.add(byname[0])
    for ent in db.find_entities(entity_class_name="connection"):
        byname = ent["entity_byname"]
        if not byname:
            continue
        name = byname[0]
        if name not in connected:
            logger.warning(
                "Connection '%s' excluded from this scenario: missing one/both endpoint "
                "nodes or its connection__node__node — '%s' not included. Check your data!",
                name, name,
            )


def validate_connection_delay(db, logger: logging.Logger) -> None:
    """Warn once per connection that has a non-zero ``delay``.

    Connection delay is not implemented: no connection term shifts the
    flow in time (the delayed input term exists only in the indirect
    conversion of units).  The delay is therefore ignored everywhere
    (``_specs`` delay tables, ``_process_method`` and the engine readers
    read ``unit.delay`` only), so the connection is modelled exactly as
    without a delay.  Before, the delay still marked the connection
    delayed, which dropped its efficiency losses and made a ``regular``
    connection one-way.  ``db`` is scenario-filtered; a delay of 0 (or a
    map of zeros) changes nothing and is not reported.
    """
    for pv in db.find_parameter_values(
        entity_class_name="connection", parameter_definition_name="delay",
    ):
        if pv["type"] is None:
            continue
        if not any(v != 0.0 for v in _numeric_values(pv["parsed_value"])):
            continue
        logger.warning(
            "Connection '%s' has a delay: connection delay is not "
            "implemented yet and is ignored (the connection is modelled "
            "without a delay). Use a unit with a delay for a delayed flow.",
            pv["entity_byname"][0],
        )


def validate_connection_startup(db, provider) -> None:
    """Raise when a connection in a ``dc_power_flow_with_angles`` group has a ``startup_method``.

    A DC power-flow line's flow is set by the voltage angles (``dc_flow_eq``); an offline line would still tie
    the angles, and switching it needs line-switching constraints FlexTool does not have.  Reads the DC line
    set from the Provider (``input/connection_dc_power_flow``, after :func:`derive_dc_power_flow`).  The model
    builder raises the same error for data built without the input derivation
    (``model.core._validate_no_online_dc_lines``).
    """
    from flextool.engine.model.core import DC_ONLINE_MESSAGE

    if not provider.has("input/connection_dc_power_flow"):
        return
    dc_frame = provider.get("input/connection_dc_power_flow")
    if dc_frame is None or dc_frame.height == 0:
        return
    dc = {str(x) for x in dc_frame.get_column(dc_frame.columns[0]).to_list()}
    hit = sorted(
        pv["entity_byname"][0]
        for pv in db.find_parameter_values(entity_class_name="connection",
                                           parameter_definition_name="startup_method")
        if pv["type"] is not None and str(pv["parsed_value"]) != "no_startup" and pv["entity_byname"][0] in dc)
    if hit:
        raise FlexToolConfigError(DC_ONLINE_MESSAGE.format(names=", ".join(hit)))


def warn_existing_not_whole_units(db, provider) -> None:
    """Print a warning for each unit / connection with integer online variables whose ``existing`` is not a whole
    number of units.

    With ``virtual_unitsize`` set, ``existing`` is the number of units (capacity = existing × virtual_unitsize; the
    ``params._existing`` capacity cascade), so the unit count is ``existing`` itself.  An integer online variable
    (``process_online_integer``: ``startup_method = binary``) counts whole units up to that count, so a count such
    as 2.5 cannot all be online.  Without ``virtual_unitsize`` the entity is one unit of its existing capacity, a
    whole number by construction, so it is not checked.  Nodes and investments have no integer variables, so they
    are not checked either.  Reads the integer set from the Provider (after ``emit_process_method_projections``).
    ``db`` is scenario-filtered, and the input derivation runs once per scenario, so each entity is reported once
    per run.  Results are unchanged.
    """
    key = "solve_data/process_online_integer"
    if not provider.has(key):
        return
    frame = provider.get(key)
    if frame is None or frame.height == 0:
        return
    integer = {str(x) for x in frame.get_column(frame.columns[0]).to_list()}
    for cls in ("unit", "connection"):
        unitsize = {name: vals[0] for name, vals in _explicit_values(db, cls, "virtual_unitsize").items()
                    if name in integer and vals and vals[0] > 0.0}
        for name, existing in sorted(_explicit_values(db, cls, "existing").items()):
            size = unitsize.get(name)
            if size is None:
                continue
            fractional = [e for e in dict.fromkeys(existing) if abs(e - round(e)) > 1e-6 * max(1.0, abs(e))]
            if not fractional:
                continue
            counts = ", ".join(f"{e:g}" for e in fractional)
            print(f"Warning: {cls} '{name}' has integer online variables (whole units), but existing gives "
                  f"{counts} units, not a whole number. With virtual_unitsize, existing is the number of units of "
                  f"virtual_unitsize ({size:g} MW each).", flush=True)


def validate_group_increase_reserve_ratio(db, logger: logging.Logger) -> None:
    """Warn once per reserve group with an ``increase_reserve_ratio`` but no dynamic ``reserve_method``.

    The group ratio adds the demand of the group's nodes to the dynamic requirement only
    (``model._reserve.group_demand_requirement``); with any other method (or none) it is ignored.
    ``db`` is scenario-filtered; a ratio of 0 changes nothing and is not reported.
    """
    from flextool.engine.preprocess._reserve_methods import RESERVE_DYNAMIC_METHODS

    methods = {
        tuple(pv["entity_byname"]): pv["parsed_value"]
        for pv in db.find_parameter_values(
            entity_class_name="reserve__upDown__group", parameter_definition_name="reserve_method",
        )
    }
    for pv in db.find_parameter_values(
        entity_class_name="reserve__upDown__group", parameter_definition_name="increase_reserve_ratio",
    ):
        if pv["type"] is None:
            continue
        if not any(v != 0.0 for v in _numeric_values(pv["parsed_value"])):
            continue
        key = tuple(pv["entity_byname"])
        if methods.get(key) in RESERVE_DYNAMIC_METHODS:
            continue
        logger.warning(
            "increase_reserve_ratio of reserve group '%s' is used only with a dynamic reserve_method "
            "(dynamic_only, timeseries_and_dynamic, dynamic_and_large_failure, all); it is ignored.",
            "__".join(key),
        )


def _numeric_values(parsed) -> list[float]:
    """Every numeric element of a parsed parameter value (a float, or the
    values of a Map / TimeSeries / array).  Non-numeric values give []."""
    if parsed is None or isinstance(parsed, (str, bool)):
        return []
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    values = getattr(parsed, "values", None)
    if values is None:
        return []
    out: list[float] = []
    for v in values:
        out.extend(_numeric_values(v))
    return out


def _arc_values(db, entity_class: str, parameter: str
                ) -> dict[tuple[str, str], list[float]]:
    """``{(unit, node): [numeric elements]}`` of the scenario's explicit
    values of ``entity_class.parameter``."""
    out: dict[tuple[str, str], list[float]] = {}
    for pv in db.find_parameter_values(entity_class_name=entity_class,
                                       parameter_definition_name=parameter):
        if pv["type"] is None:
            continue
        vals = _numeric_values(pv["parsed_value"])
        if vals:
            out[(pv["entity_byname"][0], pv["entity_byname"][1])] = vals
    return out


def _arcs(db, entity_class: str) -> dict[str, list[str]]:
    """``{unit: [nodes]}`` of the scenario's ``entity_class`` entities."""
    out: dict[str, list[str]] = {}
    for ent in db.find_entities(entity_class_name=entity_class):
        unit, node = ent["entity_byname"][0], ent["entity_byname"][1]
        out.setdefault(unit, []).append(node)
    return out


def _process_methods(provider) -> dict[str, str]:
    """``{process: method}`` from ``derive_process_method``'s output."""
    if provider is None or not provider.has("input/process_method"):
        return {}
    df = provider.get("input/process_method")
    if df is None or df.height == 0:
        return {}
    return dict(zip(df.get_column("process").to_list(),
                    df.get_column("method").to_list()))


def validate_unit_input_shares(db, provider, logger: logging.Logger) -> None:
    """Check ``unit__inputNode.input_share_min`` / ``input_share_max``.

    Runs once per scenario (after ``derive_process_method``).  Only inputs
    with ``conversion_flow_coeff ≠ 0`` count ("the unit's input" is their
    flow × conversion_flow_coeff).  Errors (:class:`FlexToolConfigError`):

    * ``input_share_min`` outside [0, 1];
    * ``input_share_max`` below 0;
    * ``input_share_min`` above the same input's ``input_share_max``: the
      input must supply at least that share of the unit's current input but
      may supply at most the smaller share of its full-load input, so the
      unit could not run at full load;
    * the ``input_share_min`` of a unit's inputs summing above 1 (the unit
      could never run);
    * ``input_share_min`` on a unit that has an input with a negative
      ``conversion_flow_coeff`` (the unit's input would no longer be an
      energy total).

    Warnings: ``input_share_max`` above 1 (same effect as 1);
    ``input_share_min`` on an input with ``conversion_flow_coeff = 0`` or
    on a unit with fewer than two inputs with ``conversion_flow_coeff ≠ 0``
    (no effect).
    """
    inputs = _arcs(db, "unit__inputNode")
    if not inputs:
        return
    conv = _arc_values(db, "unit__inputNode", "conversion_flow_coeff")
    share_min = _arc_values(db, "unit__inputNode", "input_share_min")
    share_max = _arc_values(db, "unit__inputNode", "input_share_max")
    if not share_min and not share_max:
        return
    errors: list[str] = []
    for unit in sorted(inputs):
        nodes = inputs[unit]
        conv_of = {n: conv.get((unit, n), [1.0]) for n in nodes}
        live = [n for n in nodes if any(v != 0.0 for v in conv_of[n])]
        for n in nodes:
            mins = share_min.get((unit, n), [])
            maxs = share_max.get((unit, n), [])
            if any(v < 0.0 or v > 1.0 for v in mins):
                errors.append(
                    f"unit '{unit}' input '{n}': input_share_min "
                    f"{_fmt(mins)} is outside [0, 1].  If this database was "
                    "migrated during development, the value may predate the "
                    "input_share_min meaning (it was carried over from the "
                    "old input-side capacity_min_coeff); re-author or "
                    "delete it")
            if any(v < 0.0 for v in maxs):
                errors.append(
                    f"unit '{unit}' input '{n}': input_share_max "
                    f"{_fmt(maxs)} is negative")
            elif any(v > 1.0 for v in maxs):
                logger.warning(
                    "Unit '%s' input '%s': input_share_max %s is above 1; "
                    "it has the same effect as 1 (no limit).",
                    unit, n, _fmt(maxs))
            if mins and maxs and max(mins) > min(maxs) + 1e-12:
                errors.append(
                    f"unit '{unit}' input '{n}' must supply at least "
                    f"{max(mins):g} of the unit's input (input_share_min) "
                    f"but may supply at most {min(maxs):g} of its full-load "
                    "input (input_share_max): the unit could not run at "
                    "full load")
            if any(v > 0.0 for v in mins):
                if n not in live:
                    logger.warning(
                        "Unit '%s' input '%s': input_share_min has no "
                        "effect on an input with conversion_flow_coeff = 0.",
                        unit, n)
                elif len(live) < 2:
                    logger.warning(
                        "Unit '%s' input '%s': input_share_min has no "
                        "effect on a unit with a single input.", unit, n)
        live_mins = [max(share_min.get((unit, n), [0.0])) for n in live]
        if len(live) >= 2 and sum(live_mins) > 1.0 + 1e-9:
            errors.append(
                f"unit '{unit}': input_share_min of its inputs sums to "
                f"{sum(live_mins):g} > 1; the unit could never run")
        if (len(live) >= 2 and any(v > 0.0 for v in live_mins)
                and any(min(conv_of[n]) < 0.0 for n in live)):
            neg = [n for n in live if min(conv_of[n]) < 0.0]
            errors.append(
                f"unit '{unit}': input_share_min is not defined for a unit "
                f"with a negative input conversion_flow_coeff "
                f"({', '.join(neg)})")
    if errors:
        raise FlexToolConfigError(
            "Invalid input shares:\n  " + "\n  ".join(errors))


def _fmt(values: list[float]) -> str:
    return (f"{values[0]:g}" if len(values) == 1
            else "[" + ", ".join(f"{v:g}" for v in values) + "]")


def validate_output_min_coeff(db, provider, logger: logging.Logger) -> None:
    """Check ``unit__outputNode.capacity_min_coeff`` against the per-unit
    minimum load.

    The minimum load is once per unit on the sum of its outputs
    (``Σ outputs ≥ min_load × online capacity``) and carries no
    coefficient; ``capacity_min_coeff`` is an optional extra floor on one
    output of a multi-output unit.  Runs once per scenario.

    * Error: an explicit ``capacity_min_coeff`` outside [0, 1].
    * Warning: a unit with online variables and ``min_load > 0`` whose
      explicit output coefficients are all below 1 (0 included — the old
      way to lower or switch off the minimum load).
    * Warning: a multi-output online unit whose per-output floors
      ``Σ_k capacity_min_coeff_k · min_load > 1`` or
      ``capacity_min_coeff_k · min_load > capacity_max_coeff_k`` may be
      unable to run while online (static values only; availability and
      time-varying values are not checked).
    * Warning: an online min-load unit whose outputs all have
      ``conversion_flow_coeff = 0`` has no output measure and no floor.
    """
    outputs = _arcs(db, "unit__outputNode")
    if not outputs:
        return
    cmin = _arc_values(db, "unit__outputNode", "capacity_min_coeff")
    bad = [f"unit '{u}' output '{n}': {_fmt(v)}"
           for (u, n), v in sorted(cmin.items())
           if any(x < 0.0 or x > 1.0 for x in v)]
    if bad:
        raise FlexToolConfigError(
            "capacity_min_coeff must lie in [0, 1]:\n  " + "\n  ".join(bad))
    min_load: dict[str, float] = {}
    for pv in db.find_parameter_values(entity_class_name="unit",
                                       parameter_definition_name="min_load"):
        vals = _numeric_values(pv["parsed_value"])
        if vals:
            min_load[pv["entity_byname"][0]] = max(vals)
    if not min_load:
        return
    from flextool.input_derivation._method_constants import (
        METHOD_LP, METHOD_MIP,
    )
    methods = _process_methods(provider)
    online = {p for p, mth in methods.items() if mth in METHOD_LP | METHOD_MIP}
    cmax = _arc_values(db, "unit__outputNode", "capacity_max_coeff")
    conv = _arc_values(db, "unit__outputNode", "conversion_flow_coeff")
    for unit in sorted(outputs):
        ml = min_load.get(unit, 0.0)
        if ml <= 0.0 or unit not in online:
            continue
        nodes = outputs[unit]
        explicit = [max(cmin[(unit, n)]) for n in nodes if (unit, n) in cmin]
        if explicit and max(explicit) < 1.0:
            logger.warning(
                "Unit '%s': capacity_min_coeff no longer lowers the unit's "
                "minimum load (the floor is min_load x online capacity on "
                "the sum of the outputs); set min_load lower (min_load = 0 "
                "switches it off) instead, or remove the coefficient.", unit)
        live = [n for n in nodes
                if any(v != 0.0 for v in conv.get((unit, n), [1.0]))]
        if not live:
            logger.warning(
                "Unit '%s': all outputs have conversion_flow_coeff = 0, so "
                "the unit has no minimum-load floor.", unit)
            continue
        if len(live) < 2:
            continue
        floors = {n: max(cmin.get((unit, n), [0.0])) for n in live}
        total = sum(floors.values()) * ml
        over_cap = [n for n in live
                    if floors[n] * ml > min(cmax.get((unit, n), [1.0])) + 1e-12]
        if total > 1.0 + 1e-9 or over_cap:
            logger.warning(
                "Unit '%s': its per-output floors (capacity_min_coeff x "
                "min_load) %s; the unit may be unable to run while online.",
                unit,
                (f"sum to {total:g} of the capacity" if total > 1.0 + 1e-9
                 else "exceed capacity_max_coeff on "
                 + ", ".join(over_cap)))
