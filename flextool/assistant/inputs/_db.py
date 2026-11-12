"""Input databases read once, read-only, into plain in-memory tables; the scenario view computed from them.

**Opening** (P0 §1, §4.3).  The database is opened through a read-only SQLite URI (:func:`db_uri`): ``mode=ro``
only counts with ``uri=true``, and the path is percent-encoded, because a raw ``#`` or ``%`` in it makes SQLite
open (and create) another file.  ``sqlite_timeout=5``: an exclusive lock held by a program that saves the
database makes the open wait at most 5 s, then the tools say the database is busy.  ``model.version`` and the
behaviour-change markers (``metadata``) are read while the mapping is open (their texts are made only when a tool
asks for them, :attr:`InputDB.behaviour_notices`: loading a database never imports :mod:`flextool.engine`); then
exactly :data:`FETCHED` is fetched,
the mapping is closed (no file handle stays open) and its items are copied into plain tables, reading each item
type of :data:`FETCHED` from the closed mapping.  The mapping itself is dropped: values stay as the stored bytes
and are parsed with ``spinedb_api.from_database`` when a tool needs them, so nothing parsed is kept.

**One unfiltered load per file** (P0-D10).  :class:`InputDB` holds every alternative's values; the scenario view
(:class:`ScenarioView`) is computed here and kept in a small LRU per database (:data:`VIEW_LRU_SIZE` scenarios;
a view costs about 40 bytes per stored value, and is rebuilt in well under a second per million): the alternatives in rank order, the active
entities (the highest-ranked ``entity_alternative`` of the scenario decides, else the class's
``active_by_default``; a multi-dimensional entity also needs every element active; FlexTool's ``is_enabled = no``
drops an entity of the classes it gates) and the winning value of each (class, entity, parameter): the value of
the highest-ranked alternative of the scenario.  spinedb-api's scenario filter is the oracle of these rules
(``tests/assistant/test_input_scenario_view_matches_filter.py``).

Loaded databases are cached by ``(path, mtime_ns, size)``: a database replaced or saved is loaded again.  At most
:data:`LRU_SIZE` are kept.  An :class:`InputDB` is never changed after it is loaded, so its index of the parameters
stored per entity (:meth:`InputDB.parameters_of`) never goes stale: a changed file is a new ``InputDB``.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

from flextool.assistant.errors import ToolError

FETCHED = (
    "alternative", "scenario", "scenario_alternative", "entity_class", "entity", "entity_alternative",
    "parameter_value_list", "list_value", "parameter_definition", "parameter_value",
)
SQLITE_TIMEOUT_S = 5
#: The metadata name prefix of a behaviour-change marker: ``flextool.engine._behaviour_changes.MARKER_PREFIX``
#: (pinned equal in ``tests/assistant/test_model_facts.py``), here so that a load does not import the engine.
MARKER_PREFIX = "flextool_behaviour_change:"
LRU_SIZE = 2
#: Scenario views kept per database: two is the most one tool uses at once (compare_inputs); a running call
#: holds its views by reference, so an eviction never affects it.
VIEW_LRU_SIZE = 2


def db_uri(path: Path) -> str:
    """The read-only SQLite URI of *path* for spinedb-api / SQLAlchemy (:func:`uri_of_posix`)."""
    return uri_of_posix(Path(path).absolute().as_posix())


def uri_of_posix(posix: str) -> str:
    """``sqlite:///file:/a/b%20c.sqlite?mode=ro&uri=true`` of an absolute POSIX-style path; a Windows drive path
    ``C:/a/b.sqlite`` gets a leading ``/`` (``file:/C:/a/b.sqlite``), the form SQLite reads as an absolute file
    URI there.  A UNC path ``//server/share/x.sqlite`` gets ``//`` more (``file:////server/share/x.sqlite``): an
    empty authority (SQLite refuses any other) and the UNC path as the path, which SQLite's Windows VFS opens as
    ``\\\\server\\share\\x.sqlite``.  Every part is percent-encoded, the server and share names too."""
    if posix.startswith("//"):
        posix = "//" + posix
    elif not posix.startswith("/"):
        posix = "/" + posix
    return "sqlite:///file:" + quote(posix, safe="/:") + "?mode=ro&uri=true"


@dataclass(frozen=True, slots=True)
class SourceKey:
    path: str
    mtime_ns: int
    size: int


def source_key(path: Path) -> SourceKey:
    st = os.stat(path)
    return SourceKey(str(Path(path).absolute()), st.st_mtime_ns, st.st_size)


@dataclass(frozen=True, slots=True)
class ValueItem:
    """One stored value: its alternative, the stored bytes and type (parsed on demand, never cached)."""

    alternative: str
    value: bytes | None
    type: str | None

    def parsed(self) -> Any:
        from spinedb_api import from_database

        return from_database(self.value, self.type)


@dataclass(frozen=True, slots=True)
class DefinitionItem:
    entity_class: str
    name: str
    default_value: bytes | None
    default_type: str | None
    value_list: str | None
    description: str | None
    group: str | None

    def default(self) -> Any:
        from spinedb_api import from_database

        return from_database(self.default_value, self.default_type)


@dataclass(frozen=True, slots=True)
class ClassItem:
    name: str
    dimensions: tuple[str, ...]
    active_by_default: bool
    description: str | None


@dataclass(frozen=True, slots=True)
class EntityItem:
    entity_class: str
    byname: tuple[str, ...]          # the entity's name per dimension (one name for a 0-dimensional entity)
    name: str
    elements: tuple[str, ...]        # element names of a multi-dimensional entity (empty for 0-dimensional)
    description: str | None


Byname = tuple[str, ...]
ValueKey = tuple[str, Byname, str]   # (class, entity byname, parameter)


@dataclass
class InputDB:
    """One input database, unfiltered, in plain tables."""

    path: Path
    key: SourceKey
    version: int | float | str | None
    markers: dict[str, str]                                 # behaviour-change id -> marker value
    alternatives: list[str]
    scenarios: dict[str, list[str]]                         # scenario -> alternatives, lowest rank first
    classes: dict[str, ClassItem]
    entities: dict[str, dict[Byname, EntityItem]]           # class -> byname -> entity
    entity_alternatives: dict[tuple[str, Byname], dict[str, bool]]
    definitions: dict[tuple[str, str], DefinitionItem]      # (class, parameter)
    values: dict[ValueKey, dict[str, ValueItem]]            # -> alternative -> value
    list_values: dict[str, list[tuple[bytes | None, str | None]]]
    _by_name: dict[str, dict[str, EntityItem]] = field(default_factory=dict, repr=False)
    _related: dict[tuple[str, str], list[EntityItem]] | None = field(default=None, repr=False)
    _params_of: dict[tuple[str, Byname], tuple[str, ...]] = field(default_factory=dict, repr=False)
    _views: "OrderedDict[str, ScenarioView]" = field(default_factory=OrderedDict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _notices: list[str] | None = field(default=None, repr=False)

    @property
    def behaviour_notices(self) -> list[str]:
        """One text per behaviour-change marker of the database, made on first use (the registry of the changes
        is in :mod:`flextool.engine`, which only a database with markers imports)."""
        if self._notices is None:
            self._notices = _behaviour_notices(self.markers) if self.markers else []
        return self._notices

    def __post_init__(self) -> None:
        self._by_name = {cls: {e.name: e for e in ents.values()} for cls, ents in self.entities.items()}
        stored: dict[tuple[str, Byname], set[str]] = {}
        for cls, byname, parameter in self.values:
            stored.setdefault((cls, byname), set()).add(parameter)
        self._params_of = {key: tuple(sorted(names)) for key, names in stored.items()}

    def scenario_names(self) -> list[str]:
        return list(self.scenarios)

    def entity(self, entity_class: str, byname: Byname) -> EntityItem | None:
        return self.entities.get(entity_class, {}).get(tuple(byname))

    def entity_by_name(self, entity_class: str, name: str) -> EntityItem | None:
        return self._by_name.get(entity_class, {}).get(name)

    def definition(self, entity_class: str, parameter: str) -> DefinitionItem | None:
        return self.definitions.get((entity_class, parameter))

    def class_parameters(self, entity_class: str) -> list[str]:
        return [p for (c, p) in self.definitions if c == entity_class]

    def values_for(self, entity_class: str, byname: Byname, parameter: str) -> dict[str, ValueItem]:
        """The value of every alternative that sets it (provenance)."""
        return self.values.get((entity_class, tuple(byname), parameter), {})

    def parameters_of(self, entity_class: str, byname: Byname) -> tuple[str, ...]:
        """The parameters with a value in some alternative for this entity, by name (from the index built at
        load)."""
        return self._params_of.get((entity_class, tuple(byname)), ())

    def stored_parameters(self, entity_class: str, byname: Byname) -> list[str]:
        """The parameters with a value in some alternative for this entity."""
        return list(self.parameters_of(entity_class, byname))

    def relationships_of(self, entity_class: str, name: str) -> list[EntityItem]:
        """The multi-dimensional entities that name the *entity_class* entity *name* as one of their elements."""
        with self._lock:
            if self._related is None:
                related: dict[tuple[str, str], list[EntityItem]] = {}
                for cls, ents in self.entities.items():
                    item = self.classes.get(cls)
                    if item is None or len(item.dimensions) < 2:
                        continue
                    for e in ents.values():
                        for dim_class, element in zip(item.dimensions, e.elements):
                            related.setdefault((dim_class, element), []).append(e)
                self._related = related
            return list(self._related.get((entity_class, name), []))

    def view(self, scenario: str) -> "ScenarioView":
        """The scenario's view, from the LRU of :data:`VIEW_LRU_SIZE` views."""
        with self._lock:
            view = self._views.get(scenario)
            if view is not None:
                self._views.move_to_end(scenario)
                return view
            view = ScenarioView(self, scenario)
            self._views[scenario] = view
            while len(self._views) > VIEW_LRU_SIZE:
                self._views.popitem(last=False)
            return view


class ScenarioView:
    """The input data as one scenario sees it."""

    def __init__(self, db: InputDB, scenario: str) -> None:
        from flextool.spinedb_backend._backend import _IS_ENABLED_CLASSES

        self.db = db
        self.scenario = scenario
        self.ranked: list[str] = list(db.scenarios[scenario])           # lowest rank first; the last wins
        self.rank: dict[str, int] = {alt: i for i, alt in enumerate(self.ranked)}
        self._winners: dict[ValueKey, ValueItem] = {}
        for key, by_alt in db.values.items():
            best: ValueItem | None = None
            best_rank = -1
            for alt, item in by_alt.items():
                r = self.rank.get(alt, -1)
                if r > best_rank:
                    best, best_rank = item, r
            if best is not None:
                self._winners[key] = best
        self._own: dict[tuple[str, Byname], tuple[bool, str | None]] = {}
        self._active: dict[tuple[str, Byname], tuple[bool, str | None]] = {}
        self._gated = frozenset(_IS_ENABLED_CLASSES)

    # -- values --------------------------------------------------------------------------------------------

    def value(self, entity_class: str, byname: Byname, parameter: str) -> ValueItem | None:
        """The value the scenario uses, or ``None`` when none of its alternatives sets one."""
        return self._winners.get((entity_class, tuple(byname), parameter))

    def overridden(self, entity_class: str, byname: Byname, parameter: str) -> list[ValueItem]:
        """The scenario's other alternatives that set this value, highest rank first (the winner excluded)."""
        by_alt = self.db.values_for(entity_class, byname, parameter)
        winner = self.value(entity_class, byname, parameter)
        others = [v for alt, v in by_alt.items() if alt in self.rank and v is not winner]
        return sorted(others, key=lambda v: self.rank[v.alternative], reverse=True)

    def outside(self, entity_class: str, byname: Byname, parameter: str) -> list[str]:
        """The alternatives outside the scenario that set this value."""
        return sorted(alt for alt in self.db.values_for(entity_class, byname, parameter) if alt not in self.rank)

    def winners(self) -> dict[ValueKey, ValueItem]:
        return self._winners

    # -- activity ------------------------------------------------------------------------------------------

    def _own_activity(self, entity_class: str, byname: Byname) -> tuple[bool, str | None]:
        key = (entity_class, byname)
        hit = self._own.get(key)
        if hit is not None:
            return hit
        choices = self.db.entity_alternatives.get(key, {})
        ranked = [(self.rank[alt], alt, active) for alt, active in choices.items() if alt in self.rank]
        if ranked:
            _, alt, active = max(ranked)
            out = (bool(active), None if active else f"it is set inactive in alternative '{alt}'")
        else:
            cls = self.db.classes.get(entity_class)
            by_default = bool(cls.active_by_default) if cls is not None else True
            out = (by_default, None if by_default else
                   f"no alternative of the scenario makes it active, and {entity_class} entities are inactive "
                   "unless an alternative makes them active")
        self._own[key] = out
        return out

    def spine_active(self, entity_class: str, byname: Byname) -> tuple[bool, str | None]:
        """Activity by the Spine rules only: the entity's own ``entity_alternative`` / class default, and every
        element of a multi-dimensional entity active by the same rule."""
        byname = tuple(byname)
        active, reason = self._own_activity(entity_class, byname)
        if not active:
            return active, reason
        entity = self.db.entity(entity_class, byname)
        cls = self.db.classes.get(entity_class)
        if entity is not None and cls is not None and len(cls.dimensions) > 1:
            for dim_class, element in zip(cls.dimensions, entity.elements):
                el = self.db.entity_by_name(dim_class, element)
                if el is None:
                    continue
                el_active, _ = self._own_activity(dim_class, el.byname)
                if not el_active:
                    return False, f"its {dim_class} '{element}' is not active in the scenario"
        return True, None

    def active(self, entity_class: str, byname: Byname) -> tuple[bool, str | None]:
        """``(active, reason when not)`` of an entity in this scenario, FlexTool's ``is_enabled`` included."""
        key = (entity_class, tuple(byname))
        hit = self._active.get(key)
        if hit is not None:
            return hit
        out = self.spine_active(*key)
        if out[0] and entity_class in self._gated:
            item = self.value(entity_class, key[1], "is_enabled")
            if item is not None and item.parsed() == "no":
                out = (False, f"is_enabled is 'no' (from alternative '{item.alternative}')")
        self._active[key] = out
        return out


# ---------------------------------------------------------------------------------------------------------
# Loading


def version_status(version: int | float | str | None) -> str:
    """``current`` · ``needs_migration`` · ``newer`` against ``FLEXTOOL_DB_VERSION``; ``unreadable`` when the
    version is not a number."""
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    if not isinstance(version, (int, float)) or isinstance(version, bool):
        return "unreadable"
    if version < FLEXTOOL_DB_VERSION:
        return "needs_migration"
    if version > FLEXTOOL_DB_VERSION:
        return "newer"
    return "current"


def _int_version(raw: Any) -> int | float | str | None:
    if raw is None:
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return str(raw)
    return int(number) if number.is_integer() else number


def _behaviour_notices(markers: dict[str, str]) -> list[str]:
    from flextool.engine._behaviour_changes import BEHAVIOUR_CHANGES, version_phrase

    out: list[str] = []
    for change_id in sorted(markers):
        change = BEHAVIOUR_CHANGES.get(change_id)
        if change is None:
            out.append(f"The database carries the behaviour-change marker {MARKER_PREFIX}{change_id}, which this "
                       "FlexTool does not know.")
        else:
            out.append(f"Behaviour changed in {version_phrase(change)}: {change.summary} {change.silence}")
    return out


def _is_locked(exc: BaseException) -> bool:
    seen: BaseException | None = exc
    while seen is not None:
        if "database is locked" in str(seen):
            return True
        seen = seen.__cause__ or seen.__context__
    return False


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def _read_markers(db: Any) -> dict[str, str]:
    """``{change id: marker value}`` of the behaviour-change markers of an open mapping (as
    ``flextool.engine._behaviour_changes.read_markers`` reads them)."""
    out: dict[str, str] = {}
    for md in db.get_items("metadata"):
        name = str(md["name"])
        if name.startswith(MARKER_PREFIX):
            out[name[len(MARKER_PREFIX):]] = str(md["value"])
    return out


def _release_abandoned_connections() -> None:
    """Close the SQLite connection a failed ``DatabaseMapping(...)`` leaves open.

    ``spinedb_api`` connects before it checks the file (``create_engine`` runs ``engine.begin()`` and the
    revision check), and when that check raises, the engine and its pooled connection are dropped without
    ``dispose()``.  The connection then holds the file open until the garbage collector finds it: on Windows
    the user cannot delete or replace the unreadable file meanwhile, and Python 3.13+ reports the finalised
    connection as an unclosed-database ``ResourceWarning`` wherever the collection happens to run.  Called
    after the ``except`` blocks, so the exception's traceback no longer keeps the engine alive; the warning
    is silenced because the release here is the intended close."""
    import gc
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        gc.collect()


def closed_mapping(path: Path) -> tuple[Any, Any, dict[str, str]]:
    """Open *path* read-only, read ``model.version`` and the behaviour-change markers, fetch :data:`FETCHED` and
    close the mapping.  Returns ``(closed mapping, raw version, markers)``; raises :class:`ToolError`."""
    from spinedb_api import DatabaseMapping, from_database
    from spinedb_api.exception import SpineDBAPIError, SpineDBVersionError

    name = Path(path).name
    refusal: ToolError | None = None
    try:
        db = DatabaseMapping(db_uri(path), create=False, upgrade=False, sqlite_timeout=SQLITE_TIMEOUT_S)
    except SpineDBVersionError:
        refusal = ToolError("old_spine_format")
    except SpineDBAPIError as exc:
        if _is_locked(exc):
            refusal = ToolError("busy")
        elif "revision" in str(exc) or "not a database" in str(exc):
            refusal = ToolError("not_flextool", name=name)
        else:
            refusal = ToolError("unreadable", name=name, reason=_first_line(exc))
    except Exception as exc:  # noqa: BLE001 — SQLAlchemy errors of a damaged or locked file
        if _is_locked(exc):
            refusal = ToolError("busy")
        elif "not a database" in str(exc):
            refusal = ToolError("not_flextool", name=name)
        else:
            refusal = ToolError("unreadable", name=name, reason=_first_line(exc))
    if refusal is not None:
        _release_abandoned_connections()
        raise refusal
    try:
        try:
            db.__enter__()
            version_item = db.get_parameter_definition_item(entity_class_name="model", name="version")
            raw_version = (from_database(version_item["default_value"], version_item["default_type"])
                           if version_item else None)
            markers = _read_markers(db)
            db.fetch_all(*FETCHED)
        finally:
            db.__exit__(None, None, None)
            db.close()
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_locked(exc):
            raise ToolError("busy") from None
        raise ToolError("unreadable", name=name, reason=_first_line(exc)) from None
    if not version_item:
        raise ToolError("not_flextool", name=name)
    return db, raw_version, markers


def _tables(db: Any) -> dict[str, Any]:
    """The plain tables of a closed mapping (every item type of :data:`FETCHED` is read from it)."""
    find = db.find_by_type
    alternatives = [a["name"] for a in find("alternative")]
    scenarios: dict[str, list[tuple[int, str]]] = {s["name"]: [] for s in find("scenario")}
    for sa in find("scenario_alternative"):
        scenarios.setdefault(sa["scenario_name"], []).append((sa["rank"], sa["alternative_name"]))
    classes = {c["name"]: ClassItem(c["name"], tuple(c["dimension_name_list"] or ()), bool(c["active_by_default"]),
                                    c["description"]) for c in find("entity_class")}
    entities: dict[str, dict[Byname, EntityItem]] = {name: {} for name in classes}
    for e in find("entity"):
        byname = tuple(e["entity_byname"])
        entities.setdefault(e["entity_class_name"], {})[byname] = EntityItem(
            e["entity_class_name"], byname, e["name"], tuple(e["element_name_list"] or ()), e["description"])
    entity_alternatives: dict[tuple[str, Byname], dict[str, bool]] = {}
    for ea in find("entity_alternative"):
        entity_alternatives.setdefault((ea["entity_class_name"], tuple(ea["entity_byname"])), {})[
            ea["alternative_name"]] = bool(ea["active"])
    list_names = {vl["name"] for vl in find("parameter_value_list")}
    list_values: dict[str, list[tuple[int, bytes | None, str | None]]] = {name: [] for name in list_names}
    for lv in find("list_value"):
        list_values.setdefault(lv["parameter_value_list_name"], []).append((lv["index"], lv["value"], lv["type"]))
    definitions = {
        (d["entity_class_name"], d["name"]): DefinitionItem(
            d["entity_class_name"], d["name"], d["default_value"], d["default_type"], d["parameter_value_list_name"],
            d["description"], d["parameter_group_name"])
        for d in find("parameter_definition")
    }
    values: dict[ValueKey, dict[str, ValueItem]] = {}
    for v in find("parameter_value"):
        key = (v["entity_class_name"], tuple(v["entity_byname"]), v["parameter_definition_name"])
        values.setdefault(key, {})[v["alternative_name"]] = ValueItem(v["alternative_name"], v["value"], v["type"])
    return {
        "alternatives": alternatives,
        "scenarios": {s: [alt for _, alt in sorted(rows)] for s, rows in scenarios.items()},
        "classes": classes,
        "entities": entities,
        "entity_alternatives": entity_alternatives,
        "definitions": definitions,
        "values": values,
        "list_values": {k: [(val, typ) for _, val, typ in sorted(rows, key=lambda r: r[0])]
                        for k, rows in list_values.items()},
    }


def load_uncached(path: Path) -> InputDB:
    """Read *path* into an :class:`InputDB` (no cache)."""
    key = source_key(path)
    db, raw_version, markers = closed_mapping(path)
    tables = _tables(db)
    del db
    return InputDB(path=Path(path), key=key, version=_int_version(raw_version),
                   markers=dict(markers), **tables)


_cache: "OrderedDict[SourceKey, InputDB]" = OrderedDict()
_cache_lock = threading.Lock()
_load_locks: dict[str, threading.Lock] = {}


def load(path: Path) -> InputDB:
    """The :class:`InputDB` of *path*, from the cache while the file is unchanged."""
    try:
        key = source_key(path)
    except FileNotFoundError:
        raise ToolError("source_missing", source=Path(path).name, path=Path(path).name) from None
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
        file_lock = _load_locks.setdefault(key.path, threading.Lock())
    with file_lock:
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                return hit
        db = load_uncached(path)
        with _cache_lock:
            for old in [k for k in _cache if k.path == db.key.path]:
                del _cache[old]
            _cache[db.key] = db
            while len(_cache) > LRU_SIZE:
                _cache.popitem(last=False)
        return db


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
