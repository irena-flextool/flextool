"""Axis-Enum activation: a load with a vocabulary types every FlexData dim column as ``pl.Enum``.

Axis-Enum mode is experimental and off in production (every axis column ``pl.Utf8``).  It is switched on for a
whole run by ``FLEXTOOL_AXIS_ENUMS=1`` (``run_chain_from_db``; tested in ``test_axis_enums_env_switch.py``) and, in
tests, per load by ``load_scenario(axis_enums=True)``, which sets the scenario's vocabulary
(``build_scenario_axis_enums``) on a provider clone and on the reader; ``load_flextool`` then sets it as the
global axis vocabulary for the load (``set_global_axis_enums``).  The loader never builds a vocabulary itself:
a DB file in the work folder does not switch a run to Enum dtypes.

* ``test_load_flextool_emits_enum_typed_dim_columns`` — no dim column of the loaded FlexData stays Utf8.
* ``test_enum_load_derives_the_flow_caps_of_the_utf8_load`` — the in-load ``p_flow_upper`` derivation runs under
  axis Enums and gives the Utf8 load's caps.

The LP built in Enum mode is checked against the Utf8 LP by ``test_axis_enum_lp_parity_sites.py`` and the opt-in
``tests/engine/axis_enums/`` run.
"""
from __future__ import annotations

import dataclasses

import polars as pl

from tests._support.load import load_scenario


def _all_dim_dtypes(flex_data) -> list[tuple[str, str, pl.DataType]]:
    """Return ``[(field_name, column_name, dtype), …]`` for every dim
    column on every Param / DataFrame / LazyFrame field of *flex_data*.
    """
    from polar_high import Param

    dim_names = {
        "e", "n", "p", "c", "g", "d", "t", "f", "i", "branch", "block",
        "r", "ud", "constraint", "cn", "bk", "b", "b_f", "b_first",
        "b_next",
        # Synonyms (per _AXIS_SYNONYMS) that should also be Enum
        # when the cascade is fully Enum-aware.
        "node", "process", "commodity", "group", "entity",
        "d_invest", "d_divest", "d_previous", "d_upper", "d_back",
        "period", "anchor",
    }
    out: list[tuple[str, str, pl.DataType]] = []
    for f in dataclasses.fields(flex_data):
        val = getattr(flex_data, f.name, None)
        if val is None:
            continue
        if isinstance(val, Param):
            schema = val.lazy.collect_schema()
        elif isinstance(val, pl.DataFrame):
            schema = val.schema
        elif isinstance(val, pl.LazyFrame):
            schema = val.collect_schema()
        else:
            continue
        for col, dt in schema.items():
            if col in dim_names:
                out.append((f.name, col, dt))
    return out


def test_load_flextool_emits_enum_typed_dim_columns(scenario_run) -> None:
    """Activation contract: every FlexData dim column is Enum.

    With the vocabulary set by ``load_flextool``, every Param / DataFrame
    / LazyFrame field on the returned FlexData exposes its dim columns as
    ``pl.Enum``.  A remaining ``pl.Utf8`` dim column is a missed cast
    site; the failure names every one.

    Production loads without axis Enums; this test activates them
    (``axis_enums=True``: the scenario's vocabulary on a provider clone
    and on the reader) because it tests the Enum activation itself.
    """
    flex_data = load_scenario(scenario_run("base"), axis_enums=True)

    dim_dtypes = _all_dim_dtypes(flex_data)
    assert dim_dtypes, "load_flextool produced no dim columns at all"

    enum_columns = [
        (field, col, dt) for field, col, dt in dim_dtypes
        if isinstance(dt, pl.Enum)
    ]
    utf8_columns = [
        (field, col, dt) for field, col, dt in dim_dtypes
        if dt == pl.Utf8
    ]

    utf8_count = len(utf8_columns)
    assert utf8_count == 0, (
        f"Enum activation incomplete: {utf8_count} dim columns are "
        f"still Utf8.\n"
        f"First Utf8 site: "
        f"{utf8_columns[0] if utf8_columns else '<none>'}\n"
        f"All Utf8 sites: {utf8_columns}"
    )
    assert len(enum_columns) > 0, (
        "Activation produced no Enum dim columns at all — substrate is "
        "wired but no field's dims got cast."
    )




def _utf8_rows(frame: pl.DataFrame) -> list[tuple]:
    out = frame.with_columns(pl.col(c).cast(pl.Utf8) for c, t in frame.schema.items() if isinstance(t, pl.Enum))
    return sorted(out.rows(), key=lambda r: tuple(str(v) for v in r))


def _load_recording_flow_caps(run, monkeypatch, *, axis_enums: bool) -> list:
    """Load ``run``'s last step; return what each ``p_flow_upper_from_source`` call of the load gave (or raised)."""
    from flextool.engine.preprocess.params import _derived

    original = _derived.p_flow_upper_from_source
    calls: list = []

    def recording(*args, **kwargs):
        try:
            out = original(*args, **kwargs)
        except Exception as exc:
            calls.append(exc)
            raise
        calls.append(out)
        return out

    with monkeypatch.context() as mp:
        mp.setattr(_derived, "p_flow_upper_from_source", recording)
        load_scenario(run, axis_enums=axis_enums)
    return calls


def test_enum_load_derives_the_flow_caps_of_the_utf8_load(scenario_run, monkeypatch) -> None:
    """The load's ``p_flow_upper`` derivation runs under axis Enums and gives the Utf8 load's caps.

    It joined the capacity-per-unit term's Utf8 period column to the Enum one and raised; the load swallowed the
    error and kept the provider's frame, so in Enum mode the derivation never ran.
    """
    run = scenario_run("lh2_three_region_rp_invest", db_fixture="lh2_rp_invest")
    utf8_calls = _load_recording_flow_caps(run, monkeypatch, axis_enums=False)
    enum_calls = _load_recording_flow_caps(run, monkeypatch, axis_enums=True)
    assert len(utf8_calls) == 1 and utf8_calls[0] is not None and not isinstance(utf8_calls[0], Exception)
    assert len(enum_calls) == 1 and not isinstance(enum_calls[0], Exception), enum_calls
    enum_cap, utf8_cap = enum_calls[0], utf8_calls[0]
    assert isinstance(enum_cap.frame.schema["d"], pl.Enum)
    assert _utf8_rows(enum_cap.frame) == _utf8_rows(utf8_cap.frame)
