"""Phase A — per-parameter axis-tuple contract extension.

Tests the registry/contract additions made in ``_param_shapes.py``:

* New :class:`Shape` members ``SCALAR_STR``, ``MAP_TIER_FACET`` and
  ``MAP_PERIOD_TIER_FACET`` exist and have :class:`ParamAxes` decompositions
  in :data:`_SHAPE_AXES`.
* :class:`AxisName` enum (``D``, ``T``, ``I``) sourced from
  ``flextool_axis_contract.json``.
* :class:`LeafKind` enum (``NUMERIC``, ``STR_FROM_LIST``,
  ``FACET_PRICE_QUANTITY``).
* The five new registry entries
  (``commodity.price_ladder_cumulative``, ``commodity.price_ladder_annual``,
  ``{node, unit, connection}.invest_method``) carry the expected shapes.
* :func:`resolve_param_shape` recognises the new shape variants against a
  live SpineDbReader sourced from ``templates_examples.json`` — depth-2
  facet (cumulative), depth-3 period+facet (annual), and the scalar-string
  promotion (invest_method).

Phase A is contract-only.  Phase B will wire the readers/writers; this
test file pins the contract so the wiring has a stable target.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flextool.engine_polars._param_shapes import (
    PARAM_ALLOWED_SHAPES,
    AxisName,
    LeafKind,
    ParamAxes,
    Shape,
    _SHAPE_AXES,
    _SHAPE_LABELS,
    _shape_from_indices,
    facet_keys,
    resolve_param_shape,
    shape_to_axes,
)


FLEXTOOL_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES_JSON = (
    FLEXTOOL_ROOT / "flextool" / "schemas" / "canonical_databases"
    / "templates_examples.json"
)


# ---------------------------------------------------------------------------
# A. Enum / dataclass surface
# ---------------------------------------------------------------------------


def test_axis_name_members() -> None:
    """``AxisName`` must expose the three axis identifiers Phase A needs.

    Sourced from ``flextool/schemas/flextool_axis_contract.json``: ``d``
    (period), ``t`` (time), ``i`` (tier — the price-ladder axis).  The
    enum value MUST equal the contract's axis name so downstream code
    can use it as a column key.
    """
    assert AxisName.D.value == "d"
    assert AxisName.T.value == "t"
    assert AxisName.I.value == "i"


def test_leaf_kind_members() -> None:
    """``LeafKind`` must expose numeric, value-list-string, and facet-dict."""
    assert LeafKind.NUMERIC.value == "numeric"
    assert LeafKind.STR_FROM_LIST.value == "str_from_list"
    # Spelled out so a typo in the enum doesn't slip into Phase B silently.
    assert LeafKind.FACET_PRICE_QUANTITY.value == "facet[price,quantity]"


def test_new_shape_members_exist() -> None:
    """The three new Phase A shapes are registered on the enum.

    Pins names so renaming requires updating tests too.
    """
    assert Shape.SCALAR_STR.value == "scalar[str_from_list]"
    assert Shape.MAP_TIER_FACET.value == "2d_map[tier,facet{price,quantity}]"
    assert Shape.MAP_PERIOD_TIER_FACET.value == (
        "3d_map[period,tier,facet{price,quantity}]"
    )


def test_existing_shape_members_preserved() -> None:
    """Pre-existing :class:`Shape` members are NOT renamed/deleted.

    ``_direct_params.py`` and three test files transitively depend on
    these names — Phase A must keep them intact (structural choice (a)
    in the task brief).
    """
    assert Shape.SCALAR.value == "scalar"
    assert Shape.MAP_PERIOD.value == "1d_map[period]"
    assert Shape.MAP_TIME.value == "1d_map[time]"
    assert Shape.MAP_PERIOD_TIME.value == "2d_map[period,time]"
    assert Shape.MAP_TIME_PERIOD.value == "2d_map[time,period]"


# ---------------------------------------------------------------------------
# B. _SHAPE_AXES / shape_to_axes — derived (map_levels, leaf) view
# ---------------------------------------------------------------------------


def test_shape_axes_covers_every_shape() -> None:
    """Every :class:`Shape` member has a :class:`ParamAxes` decomposition.

    Drift here means a Phase B consumer querying ``shape_to_axes`` for
    a freshly-added shape would crash with :class:`KeyError`.
    """
    for shape in Shape:
        assert shape in _SHAPE_AXES, (
            f"Shape.{shape.name} missing from _SHAPE_AXES — add the "
            "(axes, leaf) decomposition when extending the enum."
        )


def test_param_axes_numeric_legacy_shapes() -> None:
    """Legacy shapes decompose to numeric-leaf axes only."""
    assert shape_to_axes(Shape.SCALAR) == ParamAxes((), LeafKind.NUMERIC)
    assert shape_to_axes(Shape.MAP_PERIOD) == ParamAxes(
        (AxisName.D,), LeafKind.NUMERIC)
    assert shape_to_axes(Shape.MAP_TIME) == ParamAxes(
        (AxisName.T,), LeafKind.NUMERIC)
    assert shape_to_axes(Shape.MAP_PERIOD_TIME) == ParamAxes(
        (AxisName.D, AxisName.T), LeafKind.NUMERIC)
    assert shape_to_axes(Shape.MAP_TIME_PERIOD) == ParamAxes(
        (AxisName.T, AxisName.D), LeafKind.NUMERIC)


def test_param_axes_phase_a_shapes() -> None:
    """Phase A shapes carry the expected (axes, leaf) decomposition."""
    assert shape_to_axes(Shape.SCALAR_STR) == ParamAxes(
        (), LeafKind.STR_FROM_LIST)
    assert shape_to_axes(Shape.MAP_TIER_FACET) == ParamAxes(
        (AxisName.I,), LeafKind.FACET_PRICE_QUANTITY)
    assert shape_to_axes(Shape.MAP_PERIOD_TIER_FACET) == ParamAxes(
        (AxisName.D, AxisName.I), LeafKind.FACET_PRICE_QUANTITY)


def test_facet_keys_price_quantity() -> None:
    """The facet-key accessor returns the documented ``{price, quantity}``."""
    assert facet_keys(LeafKind.FACET_PRICE_QUANTITY) == {"price", "quantity"}


def test_facet_keys_non_facet_raises() -> None:
    """Asking for facet keys on a non-facet leaf raises KeyError."""
    with pytest.raises(KeyError):
        facet_keys(LeafKind.NUMERIC)
    with pytest.raises(KeyError):
        facet_keys(LeafKind.STR_FROM_LIST)


def test_shape_labels_covers_every_shape() -> None:
    """``_SHAPE_LABELS`` parallels the enum — :func:`_infer_silent_default_labels`
    consults it to disambiguate silent ``index_name`` slots.  Drift means
    new shapes can't be reached via the silent-default recovery path.
    """
    for shape in Shape:
        assert shape in _SHAPE_LABELS, (
            f"Shape.{shape.name} missing from _SHAPE_LABELS — add the "
            "per-level canonical label tuple when extending the enum."
        )


def test_shape_labels_facet_positions_are_none() -> None:
    """Facet positions in :data:`_SHAPE_LABELS` are encoded as ``None``.

    ``index_name`` is user-set per parameter value — the author may
    write ``Map(<anything>)`` with any label they like (the
    spinedb_api default ``"x"`` is just one possibility).  Phase A
    therefore records every facet-shape slot as ``None``: silent-
    default disambiguation MUST NOT try to invent labels for these
    levels, and structural detection lives in
    :func:`_recognise_facet_shape` (registry-driven, label-agnostic).
    """
    assert _SHAPE_LABELS[Shape.MAP_TIER_FACET] == (None, None)
    assert _SHAPE_LABELS[Shape.MAP_PERIOD_TIER_FACET] == (
        None, None, None,
    )


# ---------------------------------------------------------------------------
# C. PARAM_ALLOWED_SHAPES — the 5 new entries
# ---------------------------------------------------------------------------


def test_commodity_price_ladder_cumulative_entry() -> None:
    """Spine declares ``("commodity", "price_ladder_cumulative", "map", 2)``
    in its parameter_value_types — the registry pins the depth-2 facet.
    """
    allowed = PARAM_ALLOWED_SHAPES[("commodity", "price_ladder_cumulative")]
    assert allowed == {Shape.MAP_TIER_FACET}


def test_commodity_price_ladder_annual_entry() -> None:
    """Spine declares BOTH ``("commodity", "price_ladder_annual", "map", 2)``
    and ``("commodity", "price_ladder_annual", "map", 3)``; the registry
    accepts the depth-2 (same per-year limit every period) and the
    depth-3 (per-period override) variants.
    """
    allowed = PARAM_ALLOWED_SHAPES[("commodity", "price_ladder_annual")]
    assert allowed == {
        Shape.MAP_TIER_FACET, Shape.MAP_PERIOD_TIER_FACET,
    }


@pytest.mark.parametrize("entity_class", ["node", "unit", "connection"])
def test_invest_method_entries(entity_class: str) -> None:
    """``{node, unit, connection}.invest_method`` are scalar strings
    constrained by the ``invest_methods`` parameter_value_list.
    """
    allowed = PARAM_ALLOWED_SHAPES[(entity_class, "invest_method")]
    assert allowed == {Shape.SCALAR_STR}


# ---------------------------------------------------------------------------
# D. Facet recognition — registry-driven, label-agnostic
# ---------------------------------------------------------------------------


def test_recognise_facet_shape_depth2_tier_facet() -> None:
    """Depth-2 raw labels match ``Shape.MAP_TIER_FACET`` when the
    registry's allow-list contains it as the unique facet shape at
    that depth — regardless of the actual label strings.

    Confirms the user's contract: ``index_name`` is user-set, so
    facet recognition MUST be label-agnostic — only the SHAPE
    (depth) + LOCATION matters.
    """
    from flextool.engine_polars._param_shapes import _recognise_facet_shape
    allowed = PARAM_ALLOWED_SHAPES[("commodity", "price_ladder_cumulative")]
    # Canonical: outer authored as "tier", inner empty.
    assert _recognise_facet_shape(["tier", None], allowed) is Shape.MAP_TIER_FACET
    # Silent default: spinedb_api inserts "x" when index_name is unset.
    assert _recognise_facet_shape(["x", "x"], allowed) is Shape.MAP_TIER_FACET
    # User-set label of any shape (the contract honours this): still depth-2.
    assert _recognise_facet_shape(
        ["my_tier_axis", "facet"], allowed,
    ) is Shape.MAP_TIER_FACET


def test_recognise_facet_shape_depth3_period_tier_facet() -> None:
    """Depth-3 raw labels match ``Shape.MAP_PERIOD_TIER_FACET`` when
    the registry's allow-list contains it as a unique depth-3 facet
    shape.
    """
    from flextool.engine_polars._param_shapes import _recognise_facet_shape
    allowed = PARAM_ALLOWED_SHAPES[("commodity", "price_ladder_annual")]
    assert _recognise_facet_shape(
        ["period", "tier", None], allowed,
    ) is Shape.MAP_PERIOD_TIER_FACET
    # Silent defaults at every level — still recognised.
    assert _recognise_facet_shape(
        ["x", "x", "x"], allowed,
    ) is Shape.MAP_PERIOD_TIER_FACET
    # Author-chosen labels: depth still decides.
    assert _recognise_facet_shape(
        ["year", "step", "kv"], allowed,
    ) is Shape.MAP_PERIOD_TIER_FACET


def test_recognise_facet_shape_no_match_outside_facet() -> None:
    """Non-facet allow-lists (e.g. ``MAP_PERIOD``) never produce a facet
    shape, regardless of the depth match.
    """
    from flextool.engine_polars._param_shapes import _recognise_facet_shape
    allowed = PARAM_ALLOWED_SHAPES[("group", "co2_max_period")]  # numeric
    assert _recognise_facet_shape([None], allowed) is None
    assert _recognise_facet_shape([None, None], allowed) is None


def test_shape_from_indices_legacy_shapes_unchanged() -> None:
    """Phase A's wiring MUST NOT regress the legacy structural
    detection for ``period`` / ``time`` axes.  ``_shape_from_indices``
    only handles those — facet recognition is now its own helper.
    """
    assert _shape_from_indices([]) is Shape.SCALAR
    assert _shape_from_indices(["period"]) is Shape.MAP_PERIOD
    assert _shape_from_indices(["time"]) is Shape.MAP_TIME
    assert _shape_from_indices(
        ["period", "time"]
    ) is Shape.MAP_PERIOD_TIME
    assert _shape_from_indices(
        ["time", "period"]
    ) is Shape.MAP_TIME_PERIOD


# ---------------------------------------------------------------------------
# E. Live-DB resolution against canonical examples
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def examples_db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialise ``templates_examples.json`` into a temp sqlite.

    Mirrors the session fixture in ``tests/test_xlsx_workflow.py`` —
    user-facing ``templates/examples.sqlite`` is left untouched.  Scoped
    to this module so the canonical examples DB is built once for the
    three DB-driven tests below.
    """
    from flextool.update_flextool.initialize_database import (
        initialize_database,
    )

    work = tmp_path_factory.mktemp("phase_a_examples_db")
    db = work / "examples.sqlite"
    initialize_database(str(TEMPLATES_JSON), str(db))
    return db


def _spine_source(db_path: Path, scenario: str):
    """Build a ``SpineDbReader`` against *db_path* under *scenario*."""
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    return SpineDbReader(f"sqlite:///{db_path}", scenario)


def test_resolve_price_ladder_cumulative_depth2(
    examples_db_path: Path,
) -> None:
    """``commodity.price_ladder_cumulative`` on the canonical examples DB
    resolves to :class:`Shape.MAP_TIER_FACET`.

    The ``coal_ladder_cumulative`` scenario binds the ``ladder_cum_on``
    alternative which authors ``coal.price_ladder_cumulative`` as
    ``Map(tier -> Map(facet -> value))`` — depth-2 with outer
    ``index_name="tier"``, inner unset.
    """
    source = _spine_source(examples_db_path, "coal_ladder_cumulative")
    info = source.parameter_shape_info(
        "commodity", "price_ladder_cumulative",
    )
    # ``parameter_shape_info`` returns raw DB labels; the inner facet
    # ``Map`` is encoded by Spine as ``"x"`` (silent-default index_name).
    # The resolver normalises ``"x"`` → ``None`` via ``_normalise_label``
    # before structural matching — assert depth + outer label here, and
    # let the resolver's final ``shape`` field confirm the canonical form.
    assert len(info) == 2 and info[0] == "tier", (
        f"Canonical examples DB no longer matches the Phase A fixture "
        f"shape (depth-2, outer 'tier'); got {info!r}.  Either the "
        "example was edited or the SpineDbReader.parameter_shape_info "
        "contract changed — either way the contract test must adapt."
    )
    resolved = resolve_param_shape(
        source, "commodity", "price_ladder_cumulative",
    )
    assert resolved is not None
    assert resolved.shape is Shape.MAP_TIER_FACET


def test_resolve_price_ladder_annual_depth3(
    examples_db_path: Path,
) -> None:
    """``commodity.price_ladder_annual`` on the canonical examples DB
    resolves to :class:`Shape.MAP_PERIOD_TIER_FACET`.

    The ``coal_ladder_annual`` scenario binds the ``ladder_ann_on``
    alternative which authors ``coal.price_ladder_annual`` as a depth-3
    Map ``Map(period -> Map(tier -> Map(facet -> value)))``.
    """
    source = _spine_source(examples_db_path, "coal_ladder_annual")
    info = source.parameter_shape_info(
        "commodity", "price_ladder_annual",
    )
    # Outer two levels are explicit ("period", "tier"); facet inner is
    # the silent default that ``_normalise_label`` collapses to None.
    assert len(info) == 3 and info[:2] == ["period", "tier"], (
        f"Canonical examples DB no longer matches the Phase A fixture "
        f"shape (depth-3, outer ['period', 'tier']); got {info!r}."
    )
    resolved = resolve_param_shape(
        source, "commodity", "price_ladder_annual",
    )
    assert resolved is not None
    assert resolved.shape is Shape.MAP_PERIOD_TIER_FACET


def test_resolve_invest_method_scalar_str(
    examples_db_path: Path,
) -> None:
    """``unit.invest_method`` on the canonical examples DB resolves to
    :class:`Shape.SCALAR_STR`.

    The ``5weeks_invest_fullYear_dispatch_coal_wind`` scenario binds the
    ``coal_invest`` alternative which authors
    ``unit.coal_plant.invest_method = "invest_total"``.  Spine stores the
    value as a plain ``str`` (depth 0), so :func:`_shape_from_indices`
    returns :class:`Shape.SCALAR` and the Phase A allow-list promotion
    in :func:`resolve_param_shape` maps it to :class:`Shape.SCALAR_STR`
    (the only allowed shape for this entry).
    """
    source = _spine_source(
        examples_db_path, "5weeks_invest_fullYear_dispatch_coal_wind",
    )
    info = source.parameter_shape_info("unit", "invest_method")
    assert info == [], (
        f"Expected depth-0 (scalar) shape info for unit.invest_method; "
        f"got {info!r}."
    )
    resolved = resolve_param_shape(source, "unit", "invest_method")
    assert resolved is not None
    assert resolved.shape is Shape.SCALAR_STR


# ---------------------------------------------------------------------------
# F. Stub-source fallback — promotion path without touching a live DB
# ---------------------------------------------------------------------------


class _ScalarStringStub:
    """Stub :class:`InputSource` that surfaces a single scalar-string row.

    Mirrors the existing ``_ScalarStubSource`` idiom in
    ``test_silent_default_index_ed.py``.  Used to test the
    ``SCALAR → SCALAR_STR`` promotion deterministically (no DB needed).
    """

    def __init__(self, entity_class: str, parameter_name: str,
                 entity_name: str, value: str) -> None:
        self._entity_class = entity_class
        self._parameter_name = parameter_name
        self._entity_name = entity_name
        self._value = value

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        import polars as pl
        if (entity_class == self._entity_class
                and parameter_name == self._parameter_name):
            return pl.DataFrame({
                "name": [self._entity_name],
                "value": [self._value],
            })
        raise KeyError((entity_class, parameter_name))

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        # Scalar — no Map levels.
        return []

    def entities(self, entity_class: str):
        import polars as pl
        if entity_class == self._entity_class:
            return pl.DataFrame({"name": [self._entity_name]})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


class _FacetStub:
    """Stub :class:`InputSource` for facet-leaf parameters.

    Lets us assert label-agnostic facet recognition without binding to
    spinedb_api's silent-default ``"x"`` — the test supplies whatever
    labels it likes for the outer Map levels and verifies the resolver
    still lands on the correct :class:`Shape`.
    """

    def __init__(self, entity_class: str, parameter_name: str,
                 entity_name: str,
                 index_names: "list[str | None]",
                 ) -> None:
        self._entity_class = entity_class
        self._parameter_name = parameter_name
        self._entity_name = entity_name
        self._index_names = list(index_names)

    def parameter_explicit(self, entity_class: str, parameter_name: str):
        import polars as pl
        if (entity_class != self._entity_class
                or parameter_name != self._parameter_name):
            raise KeyError((entity_class, parameter_name))
        # Build a minimal frame: one entity-dim col + the index cols
        # named after this stub's index_names (defaulting to "x_<i>"
        # where a label is None) + a value col.  The actual values are
        # not exercised by the resolver — just the columns/depth.
        cols: dict[str, list] = {"name": [self._entity_name]}
        for i, lab in enumerate(self._index_names):
            col = lab if lab else f"x_{i}"
            cols[col] = ["1"]
        cols["value"] = [1.0]
        return pl.DataFrame(cols)

    def parameter(self, entity_class: str, parameter_name: str):
        return self.parameter_explicit(entity_class, parameter_name)

    def parameter_shape_info(self, entity_class: str, parameter_name: str):
        return list(self._index_names)

    def entities(self, entity_class: str):
        import polars as pl
        if entity_class == self._entity_class:
            return pl.DataFrame({"name": [self._entity_name]})
        return pl.DataFrame({"name": []}, schema={"name": pl.Utf8})


@pytest.mark.parametrize("index_names", [
    ["tier", None],          # canonical labels, silent default at facet
    ["x", "x"],              # both silent defaults (spinedb_api default)
    ["my_outer", "my_inner"],  # author-chosen labels
])
def test_resolver_facet_depth2_label_agnostic(index_names) -> None:
    """The resolver lands on :class:`Shape.MAP_TIER_FACET` for the
    cumulative ladder regardless of the user's ``index_name`` choices.

    Confirms the user's contract that labels are user-set and shape
    detection should rely on depth + position alone.
    """
    stub = _FacetStub(
        "commodity", "price_ladder_cumulative",
        entity_name="coal", index_names=index_names,
    )
    resolved = resolve_param_shape(
        stub, "commodity", "price_ladder_cumulative",
    )
    assert resolved is not None
    assert resolved.shape is Shape.MAP_TIER_FACET


@pytest.mark.parametrize("index_names", [
    ["period", "tier", None],
    ["x", "x", "x"],
    ["year", "step", "kv"],
])
def test_resolver_facet_depth3_label_agnostic(index_names) -> None:
    """Depth-3 ``commodity.price_ladder_annual`` resolves to
    :class:`Shape.MAP_PERIOD_TIER_FACET` regardless of label choice.
    """
    stub = _FacetStub(
        "commodity", "price_ladder_annual",
        entity_name="coal", index_names=index_names,
    )
    resolved = resolve_param_shape(
        stub, "commodity", "price_ladder_annual",
    )
    assert resolved is not None
    assert resolved.shape is Shape.MAP_PERIOD_TIER_FACET


@pytest.mark.parametrize("entity_class", ["node", "unit", "connection"])
def test_resolver_promotes_scalar_to_scalar_str(entity_class: str) -> None:
    """When the allow-list contains only :class:`Shape.SCALAR_STR` and the
    DB carries a depth-0 (scalar) value, the resolver promotes ``SCALAR``
    to ``SCALAR_STR``.

    The depth signal is structural (``parameter_shape_info`` returns
    ``[]``), so the resolver can't tell numeric from string from
    structure alone.  Promotion is driven by the registry's
    allow-list exclusivity (``{SCALAR_STR}`` without ``SCALAR``).
    """
    stub = _ScalarStringStub(
        entity_class, "invest_method",
        entity_name=f"stub_{entity_class}", value="invest_no_limit",
    )
    resolved = resolve_param_shape(stub, entity_class, "invest_method")
    assert resolved is not None
    assert resolved.shape is Shape.SCALAR_STR
