"""Build axis enums from the canonical contract + a SpineDBBackend.

This module is the EAV → axis-vocabulary bridge for the pl.Enum dtype
refactor. It reads the contract at version/flextool_axis_contract.json,
queries SpineDBBackend for entity-class vocabularies and parameter-map
keys, and emits a dict[str, pl.Enum] keyed by axis name.

The cast helper :func:`cast_against_contract` validates frames at the
Backend/SpineDbReader emit boundaries (see Phase 2). On vocabulary
miss, it raises :class:`FlexDataIntegrityError` with a beginner-friendly
4-paragraph message that names the offending token, where it appeared,
and a short list of next-step suggestions.

The contract is the authoritative source of axis names + sources;
this module never invents an axis. Synthetic-token allowlist entries
declared in the contract are folded into the relevant axis enum so
literal tokens introduced by cascade writers (e.g. ``eff`` / ``noEff``
for the branch axis, ``default`` for the block axis) round-trip
without integrity errors.

Public API
----------

* :class:`AxisSpec` — one parsed axis row.
* :class:`AxisContract` — full parsed contract.
* :func:`load_axis_contract` — parse the JSON file.
* :func:`build_axis_enums` — emit ``{axis_name: pl.Enum}``.
* :func:`cast_against_contract` — cast a frame's dim columns; raise
  :class:`FlexDataIntegrityError` on vocabulary miss.
* :class:`FlexDataIntegrityError` — beginner-friendly cast failure.

This is Phase 1 of the pl.Enum dtype refactor.  Phase 2 wires the
cast helper into the Backend's emit boundary; Phase 4 activates it
across the cascade.  See ``specs/enum_dtype_refactor_plan.md`` for the
full plan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl


# ---------------------------------------------------------------------------
# Contract path resolution
# ---------------------------------------------------------------------------


def _default_contract_path() -> Path:
    """Return the on-disk path to ``version/flextool_axis_contract.json``.

    The contract lives in the repo's ``version/`` directory; this module
    is at ``flextool/spinedb_backend/_axis_enums.py``, so the repo root
    is two parents up.
    """
    return (
        Path(__file__).resolve().parents[2]
        / "version"
        / "flextool_axis_contract.json"
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AxisSpec:
    """One axis entry parsed from the contract.

    Mirrors the JSON shape directly — every field corresponds to a
    ``axes[i]`` key, with sensible empties when the key is absent.

    Attributes
    ----------
    name : str
        The short axis name (e.g. ``"n"``, ``"p"``, ``"branch"``).
    label : str
        Human-readable label (e.g. ``"node"``, ``"process"``).  Used in
        :class:`FlexDataIntegrityError` to render beginner messages.
    source_type : str
        One of ``"entity_class"``, ``"entity_class_union"``,
        ``"parameter_keys"``, ``"parameter_value_list"``, ``"synthetic"``.
    source : Any
        Raw ``source`` field from the contract.  Shape depends on
        ``source_type`` (see contract JSON schema).
    filter : str | None
        Free-text filter description.  Informational only — vocabulary
        construction does not apply scenario filters here (the Backend
        does that upstream).
    tokens : list[str] | None
        Hardcoded tokens for ``source_type == "synthetic"`` axes.
    column_synonyms : list[str]
        Column names that should be cast against this axis enum.  The
        axis ``name`` itself is always implicit.
    note : str | None
        Free-text note for human readers.
    """

    name: str
    label: str
    source_type: str
    source: Any
    filter: str | None
    tokens: list[str] | None
    column_synonyms: list[str]
    note: str | None


@dataclass(frozen=True)
class AxisContract:
    """The full parsed contract.

    Attributes
    ----------
    axes : tuple[AxisSpec, ...]
        All axis rows in declaration order.
    synthetic_token_allowlist : tuple[dict, ...]
        Raw entries from ``synthetic_token_allowlist`` — each is a
        ``{"axis": str, "tokens": list[str], ...}`` dict.  The
        ``build_axis_enums`` folds these into the right axis enum.
    mixed_vocab_columns : dict
        Raw ``mixed_vocab_columns`` block — ``{"confirmed": [...],
        "pending_audit": [...]}``.  Columns in ``confirmed`` are cast
        against the ``e`` (entity union) axis enum.
    non_dim_columns : dict
        Raw ``non_dim_columns`` block — ``{"confirmed": [...]}``.
        These columns are NOT cast (they hold data values, not
        dimension tokens).
    """

    axes: tuple[AxisSpec, ...]
    synthetic_token_allowlist: tuple[dict, ...]
    mixed_vocab_columns: dict
    non_dim_columns: dict

    def by_name(self, name: str) -> AxisSpec:
        """Look up an axis by its short name.

        Raises
        ------
        KeyError
            If *name* is not declared in the contract.
        """
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise KeyError(
            f"axis {name!r} is not declared in the contract"
        )

    def column_to_axis(self, column_name: str) -> AxisSpec | None:
        """Map a column name to the axis whose enum should cast it.

        Resolution order:
        1. If *column_name* is in ``non_dim_columns.confirmed``, return
           ``None`` (the column is NOT a dim — leave its dtype alone).
        2. If *column_name* is in ``mixed_vocab_columns.confirmed``,
           return the ``e`` (entity union) axis.
        3. If a declared axis has *column_name* in its
           ``column_synonyms``, return that axis.
        4. If *column_name* matches an axis ``name`` directly, return
           that axis.
        5. Otherwise return ``None``.
        """
        if column_name in self.non_dim_columns.get("confirmed", []):
            return None
        if column_name in self.mixed_vocab_columns.get("confirmed", []):
            try:
                return self.by_name("e")
            except KeyError:
                return None
        for axis in self.axes:
            if column_name in axis.column_synonyms:
                return axis
        for axis in self.axes:
            if axis.name == column_name:
                return axis
        return None


# ---------------------------------------------------------------------------
# Contract loader
# ---------------------------------------------------------------------------


def load_axis_contract(path: Path | None = None) -> AxisContract:
    """Parse ``version/flextool_axis_contract.json`` into an :class:`AxisContract`.

    Parameters
    ----------
    path : Path | None
        Override the default contract path (used for testing).  When
        ``None``, defaults to ``<repo>/version/flextool_axis_contract.json``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    if path is None:
        path = _default_contract_path()
    path = Path(path)
    with path.open() as fh:
        raw = json.load(fh)
    axes_list = []
    for row in raw.get("axes", []):
        axes_list.append(AxisSpec(
            name=row["name"],
            label=row.get("label", row["name"]),
            source_type=row["source_type"],
            source=row.get("source"),
            filter=row.get("filter"),
            tokens=row.get("tokens"),
            column_synonyms=list(row.get("column_synonyms", [])),
            note=row.get("note"),
        ))
    return AxisContract(
        axes=tuple(axes_list),
        synthetic_token_allowlist=tuple(
            raw.get("synthetic_token_allowlist", [])
        ),
        mixed_vocab_columns=dict(raw.get("mixed_vocab_columns", {})),
        non_dim_columns=dict(raw.get("non_dim_columns", {})),
    )


# ---------------------------------------------------------------------------
# Vocabulary builders
# ---------------------------------------------------------------------------


def _dedup_keep_order(items: Iterable[str]) -> list[str]:
    """Return a list of *items* with duplicates removed; first
    occurrence wins.

    Used everywhere we union vocabularies — keeps the contract's
    declaration order observable in the resulting enum.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _entity_names(backend: Any, entity_class: str) -> list[str]:
    """Return the entity names for *entity_class*.

    ``backend.find_entities`` returns the raw spinedb_api rows; each
    row's ``"name"`` field is the entity name we want for the enum
    vocabulary.
    """
    rows = backend.find_entities(entity_class_name=entity_class)
    return [row["name"] for row in rows]


def _allowlist_tokens_for(contract: AxisContract, axis_name: str) -> list[str]:
    """Return the synthetic-allowlist tokens for *axis_name*, in
    declaration order, deduplicated.
    """
    out: list[str] = []
    for entry in contract.synthetic_token_allowlist:
        if entry.get("axis") == axis_name:
            out.extend(entry.get("tokens", []))
    return _dedup_keep_order(out)


def _map_keys_at_depth(parsed_value: Any, depths: list[int]) -> int:
    """Return the maximum nesting depth of a Spine Map ``parsed_value``.

    Walks ``parsed_value.indexes`` / ``parsed_value.values`` recursively,
    appending each level's depth to *depths*.  Used to discover the
    ``i`` axis vocabulary length (max depth of
    ``commodity.price_ladder_*`` parameter maps).
    """
    # If the value is itself a Map, walk one level deeper.
    if hasattr(parsed_value, "values") and hasattr(parsed_value, "indexes"):
        # Each level contributes 1; recurse into the nested maps to find
        # the deepest leaf.
        local_depth = 1
        children_max = 0
        for v in parsed_value.values:
            child_depth = _map_keys_at_depth(v, depths)
            if child_depth > children_max:
                children_max = child_depth
        return local_depth + children_max
    return 0


def _collect_parameter_map_keys(
    backend: Any,
    entity_class: str,
    parameter: str,
) -> list[str]:
    """Collect the top-level keys of all maps in
    ``(entity_class, parameter)``.

    For each parameter-value row whose ``type == "map"``, gather the
    ``parsed_value.indexes`` (top-level keys); union across all rows
    preserving first-occurrence order.
    """
    rows = backend.find_parameter_values(
        entity_class_name=entity_class,
        parameter_definition_name=parameter,
    )
    out: list[str] = []
    for param in rows:
        if param.get("type") != "map":
            continue
        pv = param.get("parsed_value")
        if pv is None or not hasattr(pv, "indexes"):
            continue
        for idx in pv.indexes:
            out.append(str(idx))
    return _dedup_keep_order(out)


def _collect_parameter_array_values(
    backend: Any,
    entity_class: str,
    parameter: str,
) -> list[str]:
    """Collect the values of all array parameters in
    ``(entity_class, parameter)``.

    Used for sources like ``solve.realized_periods`` whose value is a
    Spine Array of period labels.
    """
    rows = backend.find_parameter_values(
        entity_class_name=entity_class,
        parameter_definition_name=parameter,
    )
    out: list[str] = []
    for param in rows:
        if param.get("type") != "array":
            continue
        pv = param.get("parsed_value")
        if pv is None or not hasattr(pv, "values"):
            continue
        for v in pv.values:
            out.append(str(v))
    return _dedup_keep_order(out)


def _collect_parameter_scalar_values(
    backend: Any,
    entity_class: str,
    parameter: str,
) -> list[str]:
    """Collect the scalar string values of ``(entity_class, parameter)``.

    Used for the ``branch`` axis (``solve.stochastic_branches`` is a
    parameter whose scalar string value names the active branch).
    """
    rows = backend.find_parameter_values(
        entity_class_name=entity_class,
        parameter_definition_name=parameter,
    )
    out: list[str] = []
    for param in rows:
        ptype = param.get("type")
        if ptype not in ("str", "float", "bool"):
            continue
        pv = param.get("parsed_value")
        if pv is None:
            continue
        out.append(str(pv))
    return _dedup_keep_order(out)


def _discover_tier_vocabulary(
    backend: Any,
    entity_class: str,
    parameter_prefix: str,
) -> list[str]:
    """Discover the ``i`` (tier_index) vocabulary by collecting the
    distinct tier-level keys across all parameter definitions in
    *entity_class* whose name starts with *parameter_prefix*.

    For ``commodity.price_ladder_*``, the Map structure is
    ``period → tier → {price, quantity}``: the *tier* keys live at
    level 2 (one nesting under the top-level period keys).  This
    helper walks every map's level-2 indices and unions them, sorted
    numerically when all keys are integer-like.

    Returns
    -------
    list[str]
        Distinct tier keys in numeric-ascending order (or
        first-occurrence order if any key is non-numeric).  Empty
        when no map parameter rows exist.
    """
    db = backend._db
    if db is None:  # pragma: no cover — backend lifecycle
        return []
    definitions = db.find_parameter_definitions()
    candidates = [
        d["name"] for d in definitions
        if d.get("entity_class_name") == entity_class
        and d.get("name", "").startswith(parameter_prefix)
    ]
    if not candidates:
        return []
    tier_keys: list[str] = []
    for pname in candidates:
        rows = backend.find_parameter_values(
            entity_class_name=entity_class,
            parameter_definition_name=pname,
        )
        for param in rows:
            if param.get("type") != "map":
                continue
            pv = param.get("parsed_value")
            if pv is None or not hasattr(pv, "values"):
                continue
            # Walk one level in: each top-level value is the per-period
            # tier-map; gather its indexes.
            for v in pv.values:
                if hasattr(v, "indexes"):
                    for idx in v.indexes:
                        tier_keys.append(str(idx))
    unique = _dedup_keep_order(tier_keys)
    try:
        return sorted(unique, key=int)
    except ValueError:
        return unique


def _build_period_vocab(backend: Any, spec_source: dict) -> list[str]:
    """Build the ``d`` (period) axis vocabulary.

    Per the contract, ``d`` is sourced from
      * keys of ``solve.years_represented`` (Map),
      * values of ``solve.realized_periods`` (Array),
      * keys of ``solve.invest_periods`` (Map),
      * keys of ``solve.realized_invest_periods`` (Map).
    All four are unioned (deterministic order).
    """
    entity_class = spec_source.get("entity_class", "solve")
    params = spec_source.get("parameters", [])
    out: list[str] = []
    for param_name in params:
        # Maps first (keys), arrays second (values).
        keys = _collect_parameter_map_keys(backend, entity_class, param_name)
        out.extend(keys)
        vals = _collect_parameter_array_values(
            backend, entity_class, param_name
        )
        out.extend(vals)
    return _dedup_keep_order(out)


def _build_block_vocab(backend: Any, spec_source: dict) -> list[str]:
    """Build the ``block`` axis vocabulary.

    Per the contract (``axes/block.source``), the block axis source is
    ``group.new_stepduration`` with ``key_kind: "values_plus_default"``.
    Block names are the names of ``group`` entities that carry a
    non-null ``new_stepduration`` parameter value — regardless of
    whether the parameter is authored as a per-period Map or as a
    scalar-per-entity.  The synthetic ``default`` token is appended
    later by the ``tokens_default_extension`` allowlist (folded by the
    caller via :func:`_allowlist_tokens_for` for the ``block`` axis).

    When ``key_kind == "values_plus_default"`` we enumerate group
    entities whose ``new_stepduration`` value is set to anything (Map
    or scalar).  For other ``key_kind`` values (or none specified) we
    fall back to the original Map-key extraction for backwards
    compatibility.
    """
    entity_class = spec_source.get("entity_class", "group")
    parameter = spec_source.get("parameter", "new_stepduration")
    key_kind = spec_source.get("key_kind")
    if key_kind == "values_plus_default":
        rows = backend.find_parameter_values(
            entity_class_name=entity_class,
            parameter_definition_name=parameter,
        )
        out: list[str] = []
        for param in rows:
            if param.get("parsed_value") is None:
                continue
            ent = param.get("entity_byname")
            if not ent:
                continue
            # ``entity_byname`` is a tuple — for the single-dim 'group'
            # class the entity name lives at index 0.
            out.append(str(ent[0]))
        return _dedup_keep_order(out)
    return _collect_parameter_map_keys(backend, entity_class, parameter)


def build_axis_enums(
    backend: Any,
    contract: AxisContract,
) -> dict[str, pl.Enum]:
    """Build the ``{axis_name: pl.Enum}`` mapping from *contract* +
    *backend*.

    For each axis in the contract:
      * ``entity_class`` — vocabulary = ``find_entities(class)`` names.
      * ``entity_class_union`` — union across the listed classes
        (first-occurrence order preserved).
      * ``parameter_keys`` — vocabulary depends on the specific axis:
        - ``t``: keys of ``timeline.timestep_duration`` map.
        - ``d``: union of period keys/values across the four solve
          parameters (see :func:`_build_period_vocab`).
        - ``i``: integers ``"1"..."N"`` where N is the max map depth
          across ``commodity.price_ladder_*`` parameters.
        - ``block``: keys of ``group.new_stepduration`` maps.
        - ``d_anchor``: empty here — populated per-solve at Phase 3
          handoff.
      * ``parameter_value_list`` — scalar values of the named parameter
        (e.g. ``solve.stochastic_branches`` for ``branch``).
      * ``synthetic`` — use ``spec.tokens`` verbatim.

    Synthetic-allowlist tokens (e.g. ``eff``/``noEff`` for ``branch``,
    ``default`` for ``block``) are appended to every axis's vocabulary.

    Returns
    -------
    dict[str, pl.Enum]
        Keyed by axis ``name``.  Every contract axis appears in the
        result; axes with empty vocabulary get an empty enum (still a
        valid Enum dtype, just with zero categories).
    """
    out: dict[str, pl.Enum] = {}
    for axis in contract.axes:
        vocab: list[str] = []
        st = axis.source_type
        src = axis.source
        if st == "entity_class":
            vocab = _entity_names(backend, src)
        elif st == "entity_class_union":
            members: list[str] = []
            for cls in src or []:
                members.extend(_entity_names(backend, cls))
            vocab = _dedup_keep_order(members)
        elif st == "parameter_keys":
            spec_src = src if isinstance(src, dict) else {}
            if axis.name == "t":
                vocab = _collect_parameter_map_keys(
                    backend,
                    spec_src.get("entity_class", "timeline"),
                    spec_src.get("parameter", "timestep_duration"),
                )
            elif axis.name == "d":
                vocab = _build_period_vocab(backend, spec_src)
            elif axis.name == "i":
                # Per the contract i_axis_depth review note: discover
                # the tier vocabulary by walking the level-2 keys of
                # every ``commodity.price_ladder_*`` map.  This emits
                # the actual tier labels (typically integer strings
                # ``"1"``..``"N"``) rather than a hard-coded ceiling.
                vocab = _discover_tier_vocabulary(
                    backend,
                    spec_src.get("entity_class", "commodity"),
                    spec_src.get("parameter_prefix", "price_ladder_"),
                )
            elif axis.name == "block":
                vocab = _build_block_vocab(backend, spec_src)
            elif axis.name == "d_anchor":
                # Per-solve carrier; built at Phase 3 handoff time.
                vocab = []
            else:
                # Generic fallback: union over listed parameters.
                ec = spec_src.get("entity_class")
                params = spec_src.get("parameters") or []
                if "parameter" in spec_src:
                    params = [spec_src["parameter"]]
                gathered: list[str] = []
                for pname in params:
                    if ec is None:
                        continue
                    gathered.extend(
                        _collect_parameter_map_keys(backend, ec, pname)
                    )
                vocab = _dedup_keep_order(gathered)
        elif st == "parameter_value_list":
            spec_src = src if isinstance(src, dict) else {}
            ec = spec_src.get("entity_class")
            pn = spec_src.get("parameter")
            if ec is not None and pn is not None:
                vocab = _collect_parameter_scalar_values(backend, ec, pn)
            else:
                vocab = []
        elif st == "synthetic":
            vocab = list(axis.tokens or [])
        else:
            # Unknown source_type — empty vocabulary.  Contract schema
            # validation upstream should prevent this from happening in
            # practice; we degrade gracefully here.
            vocab = []

        # Merge synthetic-allowlist tokens (always last; dedup keeps
        # original tokens' priority).
        vocab = _dedup_keep_order(
            list(vocab) + _allowlist_tokens_for(contract, axis.name)
        )
        out[axis.name] = pl.Enum(vocab)
    return out


# ---------------------------------------------------------------------------
# Cast helper + integrity error
# ---------------------------------------------------------------------------


_BANNER = "=" * 72


class FlexDataIntegrityError(ValueError):
    """A column carried a token not in its axis enum vocabulary.

    Raised by :func:`cast_against_contract` when a strict cast fails.
    The message is a 4-paragraph beginner-friendly explanation: what
    the unknown token is, where it appeared, the size of the declared
    vocabulary, and a short list of next steps.
    """

    @classmethod
    def from_cast_failure(
        cls,
        *,
        axis_name: str,
        axis_friendly: str,
        bad_token: str,
        vocabulary_size: int,
        parameter: str | None = None,
        entity: str | None = None,
        map_index: str | None = None,
        scenario: str | None = None,
        suggestions: Sequence[str] | None = None,
    ) -> "FlexDataIntegrityError":
        """Render the canonical 4-paragraph error message.

        Parameters
        ----------
        axis_name : str
            Short axis name (e.g. ``"n"``) — used to build the
            ``axis_friendly`` plural in the third paragraph.
        axis_friendly : str
            Human-readable axis label (e.g. ``"node"``) — used in the
            opening line and the cardinality sentence.
        bad_token : str
            The token that wasn't in the enum vocabulary.
        vocabulary_size : int
            Number of declared tokens; rendered in the cardinality
            sentence (``"Your input database lists {N} nodes."``).
        parameter, entity, map_index, scenario : str | None
            Origin breadcrumbs.  Each becomes a line in the "Where it
            appeared" block; missing breadcrumbs are replaced with a
            ``(unknown)`` placeholder so the message shape stays
            stable.
        suggestions : sequence of str | None
            Up to 4 hints rendered as bullets in the final paragraph.
            None / empty means a generic "add the entity" fallback
            bullet is appended.
        """
        # Plural: just append 's' — the axis labels in the contract are
        # all simple nouns (node, process, commodity, ...) that
        # pluralise cleanly.  If a future label breaks this rule, the
        # cast helper still raises the error correctly — only the
        # cardinality sentence reads slightly off.
        plural = axis_friendly + "s" if not axis_friendly.endswith("s") else axis_friendly

        where_lines = [
            f"  In parameter:    {parameter or '(unknown)'}",
            f"  On entity:       {entity or '(unknown)'}",
            f"  Inside a map at: key index {map_index or '(unknown)'}",
            f"  Active scenario: {scenario or '(unknown)'}",
        ]

        if suggestions is None or len(suggestions) == 0:
            bullets = [
                f"  - Check {axis_friendly} name {bad_token!r} for a typo against your DB.",
                f"  - If {bad_token!r} is truly new, add it to the source DB under "
                f"the {axis_friendly} entity class.",
            ]
        else:
            bullets = [f"  - {hint}" for hint in suggestions[:4]]
            if not any("add" in b.lower() and "DB" in b for b in bullets):
                # Always close with the "add to DB" fallback if the
                # caller didn't already include it.
                if len(bullets) < 4:
                    bullets.append(
                        f"  - If {bad_token!r} is truly new, add it to the source "
                        f"DB under the {axis_friendly} entity class."
                    )

        para3_tail = (
            f' is not one of them. This token appeared while the cascade was '
            f'reading {parameter or "an input parameter"} and casting it to '
            f'the {axis_friendly} axis enum.'
        )

        msg = (
            f"\n{_BANNER}\n"
            f"Found an unknown {axis_friendly} name {bad_token!r} "
            f"in your input data.\n"
            f"\n"
            f"Where it appeared:\n"
            + "\n".join(where_lines) + "\n"
            f"\n"
            f"Your input database lists {vocabulary_size} {plural}. "
            f"{bad_token!r}{para3_tail}\n"
            f"\n"
            f"What to do:\n"
            + "\n".join(bullets) + "\n"
            f"{_BANNER}"
        )
        return cls(msg)


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein edit distance between *a* and *b*.

    Small standalone implementation — no external dep.  Used by
    :func:`_lookup_similar_classes` to surface "did you mean ...?"
    suggestions in the integrity-error message.
    """
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(
                curr[j - 1] + 1,        # insert
                prev[j] + 1,            # delete
                prev[j - 1] + cost,     # substitute
            ))
        prev = curr
    return prev[-1]


def _lookup_similar_classes(
    token: str,
    backend: Any,
    intended_axis: AxisSpec,
    all_classes: list[str],
) -> list[str]:
    """Build a short list of suggestion hints for the bad *token*.

    Cross-class lookup (most actionable):
      For each entity class in *all_classes*, check whether *token* is
      an entity in that class.  If found, prepend a hint like
      ``"the commodity 'token'"`` — beginner-helpful when the user
      typed e.g. a commodity name in a node column.

    Levenshtein typo suggestions:
      Compute edit distance from *token* against every entity name in
      the *intended_axis*'s source class.  Return up to 2 candidates
      with distance ≤ 2.

    The total list is capped at 3 hints so the
    :class:`FlexDataIntegrityError` rendering has room for the
    ``add this entity if truly new`` fallback as the 4th bullet.
    """
    hints: list[str] = []
    # 1. Cross-class lookup.
    for cls in all_classes:
        try:
            rows = backend.find_entities(entity_class_name=cls)
        except Exception:  # noqa: BLE001 — defensive
            continue
        for r in rows:
            if r.get("name") == token:
                hints.append(f"{token!r} exists as a {cls}, not a {intended_axis.label}.")
                # Cross-class hit is the strongest signal; stop after one.
                break
        if hints:
            break

    # 2. Levenshtein typo suggestions within the intended class.
    intended_classes: list[str] = []
    if intended_axis.source_type == "entity_class":
        intended_classes = [str(intended_axis.source)]
    elif intended_axis.source_type == "entity_class_union":
        intended_classes = list(intended_axis.source or [])
    typo_candidates: list[tuple[int, str]] = []
    for cls in intended_classes:
        try:
            rows = backend.find_entities(entity_class_name=cls)
        except Exception:  # noqa: BLE001 — defensive
            continue
        for r in rows:
            name = r.get("name")
            if not name or name == token:
                continue
            dist = _levenshtein(token, name)
            if dist <= 2:
                typo_candidates.append((dist, name))
    typo_candidates.sort()
    for _dist, name in typo_candidates[:2]:
        hints.append(f"did you mean {name!r}?")

    return hints[:3]


def cast_against_contract(
    frame: pl.DataFrame,
    *,
    contract: AxisContract,
    axis_enums: dict[str, pl.Enum],
    origin: dict | None = None,
    backend: Any | None = None,
) -> pl.DataFrame:
    """Cast every dim column of *frame* to its canonical enum dtype.

    For each column in ``frame.columns``:
      * If :meth:`AxisContract.column_to_axis` resolves it to an
        :class:`AxisSpec`, cast the column to ``axis_enums[axis.name]``
        with ``strict=True``.
      * On cast failure (a token not in the enum vocabulary), raise
        :class:`FlexDataIntegrityError` with the *origin* breadcrumbs
        threaded in.
      * If no axis maps the column, leave it alone.

    Parameters
    ----------
    frame : pl.DataFrame
        Eager frame to cast.  Returned unchanged if no dim columns are
        present.
    contract : AxisContract
        The canonical axis contract.
    axis_enums : dict[str, pl.Enum]
        Result of :func:`build_axis_enums`.
    origin : dict | None
        Optional breadcrumb dict — supported keys: ``parameter``,
        ``entity``, ``map_index``, ``scenario``.  Threaded into
        :meth:`FlexDataIntegrityError.from_cast_failure`.
    backend : Any | None
        Optional :class:`SpineDBBackend` — used by
        :func:`_lookup_similar_classes` to render
        "did you mean ...?" suggestions in the error message.  When
        ``None``, the error still renders, just without suggestions.

    Returns
    -------
    pl.DataFrame
        The cast frame (or the original if no dim columns were
        present).
    """
    origin = dict(origin or {})
    cast_exprs = []
    cast_pairs: list[tuple[str, AxisSpec]] = []
    for col in frame.columns:
        axis = contract.column_to_axis(col)
        if axis is None:
            continue
        dtype = axis_enums.get(axis.name)
        if dtype is None:
            continue
        cast_pairs.append((col, axis))
        cast_exprs.append(pl.col(col).cast(dtype, strict=True))

    if not cast_exprs:
        return frame

    try:
        return frame.with_columns(cast_exprs)
    except pl.exceptions.InvalidOperationError as exc:
        # Find which column / token failed.  Polars only reports the
        # column / value, not the axis — we walk the cast pairs ourself
        # and raise on the first miss.
        bad_axis: AxisSpec | None = None
        bad_token: str = "<unknown>"
        bad_col: str | None = None
        for col, axis in cast_pairs:
            dtype = axis_enums[axis.name]
            vocab = set(dtype.categories.to_list())
            values = frame[col].to_list()
            for v in values:
                if v is None:
                    continue
                if v not in vocab:
                    bad_axis = axis
                    bad_token = str(v)
                    bad_col = col
                    break
            if bad_axis is not None:
                break

        if bad_axis is None:
            # Couldn't identify the offending value — re-raise with a
            # softer integrity error using the original exception text.
            raise FlexDataIntegrityError(
                f"polars rejected the dim-column cast: {exc}"
            ) from exc

        all_classes = []
        for ax in contract.axes:
            if ax.source_type == "entity_class":
                all_classes.append(str(ax.source))
            elif ax.source_type == "entity_class_union":
                all_classes.extend(list(ax.source or []))
        all_classes = _dedup_keep_order(all_classes)

        if backend is not None:
            suggestions = _lookup_similar_classes(
                bad_token, backend, bad_axis, all_classes,
            )
        else:
            suggestions = []

        raise FlexDataIntegrityError.from_cast_failure(
            axis_name=bad_axis.name,
            axis_friendly=bad_axis.label,
            bad_token=bad_token,
            vocabulary_size=len(axis_enums[bad_axis.name].categories),
            parameter=origin.get("parameter"),
            entity=origin.get("entity"),
            map_index=origin.get("map_index"),
            scenario=origin.get("scenario"),
            suggestions=suggestions,
        ) from exc


__all__ = [
    "AxisSpec",
    "AxisContract",
    "load_axis_contract",
    "build_axis_enums",
    "cast_against_contract",
    "FlexDataIntegrityError",
]
