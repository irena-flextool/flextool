"""Pins of the realization-roll registries (G1-G4): ``PERIOD_ROW_KINDS`` / ``REVIEWED_STEP_LEVEL``
(``model._period_rows``) and ``STEP_RELATIONS`` (``preprocess._step_relations``).

The pins are over LP family *names*, not autoscale registry keys: ``lookup_cstr`` resolves any ``K_*`` name to a
reviewed prefix key ``K``, so a new prefix-named family (``maxFlow_period_cap`` over (p, d)) would pass a key pin.
Every constraint name of the engine (the static ``add_cstr`` literals and the pinned dynamic names of
``test_registry_coverage.py``) must be in ``PERIOD_ROW_KINDS`` or ``REVIEWED_STEP_LEVEL``.  ``build_flextool`` pins
the same, and the step-pair frames of the real FlexData, on every LP under ``FLEXTOOL_AUTOSCALE_STRICT=1`` (CI).
"""
from __future__ import annotations

import dataclasses
from collections import Counter
from types import SimpleNamespace

import polars as pl
import pytest
from polar_high import Problem

from flextool.engine.autoscale._layer2_types import CONSTRAINT_FAMILIES, lookup_cstr
from flextool.engine.model import _period_rows as pr
from flextool.engine.model._period_rows import PERIOD_ROW_KINDS, REVIEWED_STEP_LEVEL, RP_PREFIX
from flextool.engine.preprocess import _step_relations as sr
from flextool.engine.preprocess.flex_data import FlexData
from tests.engine.autoscale.test_registry_coverage import (
    _ADD_CSTR_PATT,
    _DYNAMIC_CONSTRAINT_SAMPLES,
    _engine_sources,
)


def _engine_family_names(extra_source: str = "") -> set[str]:
    """The constraint family names of the engine: static literals (+ ``extra_source``) and the dynamic samples."""
    texts = [p.read_text(encoding="utf-8") for p in _engine_sources()] + [extra_source]
    return {n for text in texts for n in _ADD_CSTR_PATT.findall(text)} | set(_DYNAMIC_CONSTRAINT_SAMPLES)


def test_every_period_row_name_is_a_registered_family():
    for name in PERIOD_ROW_KINDS:
        lookup_cstr(name)   # KeyError on a rename
    # Design rev 7.1 section 2.4: 8 path, 1 path_at_step, 1 fragment, 48 invest names.
    assert Counter(PERIOD_ROW_KINDS.values()) == {"path": 8, "path_at_step": 1, "fragment": 1, "invest": 48}
    assert any(k.startswith(RP_PREFIX) for k in CONSTRAINT_FAMILIES)
    assert not set(PERIOD_ROW_KINDS) & REVIEWED_STEP_LEVEL


def test_every_family_name_is_classified_or_reviewed():
    names = _engine_family_names()
    new = pr.unreviewed_families(names)
    assert not new, pr.unreviewed_message(new)
    assert not REVIEWED_STEP_LEVEL - names, sorted(REVIEWED_STEP_LEVEL - names)   # stale (renamed or removed)
    assert not set(PERIOD_ROW_KINDS) - names, sorted(set(PERIOD_ROW_KINDS) - names)


def test_a_new_prefix_named_family_fails_the_name_pin():
    # The review probe: maxFlow_period_cap resolves to the reviewed key maxFlow, so it passes the autoscale
    # registry; the name pin names it.
    probe = 'm.add_cstr("maxFlow_period_cap", over=pd_rows, sense="<=", lhs_terms=lhs, rhs_terms=rhs)'
    lookup_cstr("maxFlow_period_cap")
    assert pr.unreviewed_families(_engine_family_names(probe)) == ["maxFlow_period_cap"]
    assert "maxFlow_period_cap" in pr.unreviewed_message(["maxFlow_period_cap"])


# ── the build pin (FLEXTOOL_AUTOSCALE_STRICT=1) ───────────────────────────────────────────────────────────────

@dataclasses.dataclass
class _FlexDataWithSeam(FlexData):
    """FlexData with a new step-pair field, as the review probe added it."""
    dtttdt_seam: pl.DataFrame | None = None   # (d, t, t_previous_seam)


def _flex_data(**fields) -> FlexData:
    required = {f.name: None for f in dataclasses.fields(FlexData)
                if f.init and f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    return _FlexDataWithSeam(**{**required, **fields})


def _pin_build(family: str, over: pl.DataFrame, d) -> SimpleNamespace:
    m = Problem()
    x = m.add_var("x", tuple(over.columns), over)
    m.add_cstr(family, over=over, sense="<=", lhs_terms={"x": x}, rhs_terms={"one": 1.0})
    return SimpleNamespace(m=m, d=d, solve_name="s_roll_0")


def test_the_build_pin_raises_on_a_new_family_or_link_field_under_strict(monkeypatch):
    pd_rows = pl.DataFrame({"p": ["u1"], "d": ["p1"]})
    seam = pl.DataFrame({"d": ["p1"], "t": ["t2"], "t_previous_seam": ["t1"]})
    clean = _pin_build("maxFlow", pd_rows.with_columns(t=pl.lit("t1")), _flex_data())
    new_family = _pin_build("maxFlow_period_cap", pd_rows, _flex_data())
    new_field = _pin_build("maxFlow", pd_rows.with_columns(t=pl.lit("t1")), _flex_data(dtttdt_seam=seam))
    monkeypatch.delenv("FLEXTOOL_AUTOSCALE_STRICT", raising=False)
    for b in (clean, new_family, new_field):
        pr.check_registry_pins(b)   # a user run never stops on a registry gap
    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    pr.check_registry_pins(clean)
    with pytest.raises(RuntimeError, match=r"solve 's_roll_0'.*not in PERIOD_ROW_KINDS: \['maxFlow_period_cap'\].*"
                                           r"constraint families \['maxFlow_period_cap'\]: add each"):
        pr.check_registry_pins(new_family)
    with pytest.raises(RuntimeError, match=r"not in STEP_RELATIONS \(preprocess/_step_relations.py\): "
                                           r"\['dtttdt_seam'\]"):
        pr.check_registry_pins(new_field)


def test_build_flextool_pins_the_registries_on_the_real_lp(tmp_path_factory, monkeypatch):
    from tests.engine._unit_db import PLANT_CHAIN, build, solve

    monkeypatch.setenv("FLEXTOOL_AUTOSCALE_STRICT", "1")
    fd = solve(tmp_path_factory, alt="registry_pins", values=[], base_chain=PLANT_CHAIN).flex_data
    families = set(build(fd).cstr_names())
    assert "nodeBalance_eq" in families and fd.dtttdt is not None and fd.dtttdt.height > 0
    # A family name the review has not seen, a step-pair field not in STEP_RELATIONS: the build raises.
    monkeypatch.setattr(pr, "REVIEWED_STEP_LEVEL", REVIEWED_STEP_LEVEL - {"nodeBalance_eq"})
    with pytest.raises(RuntimeError, match=r"constraint families \['nodeBalance_eq'\]"):
        build(fd)
    monkeypatch.setattr(pr, "REVIEWED_STEP_LEVEL", REVIEWED_STEP_LEVEL)
    monkeypatch.setattr(sr, "STEP_RELATIONS", {k: v for k, v in sr.STEP_RELATIONS.items() if k != "dtttdt"})
    with pytest.raises(RuntimeError, match=r"not in STEP_RELATIONS .*\['dtttdt'\]"):
        build(fd)
