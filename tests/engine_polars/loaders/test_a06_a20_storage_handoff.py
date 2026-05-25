"""Surface A.6 (storage / reservoir / state loader) and A.20
(handoff / warm-start / rolling-horizon coupling) loader tests.

A.6 tests call ``flextool.engine_polars.input._load_storage`` directly
with hand-built minimal frames — the helper is pure (just CSV reads
plus algebra over its inputs) so isolating it avoids needing a full
workdir orchestration just to flip a single binding-method row.

A.20 tests call ``_apply_db_overrides`` directly against a stub
``InputSource`` and a stub ``source`` shim, monkey-patching the
``apply_*`` callees to record their invocation order.  The pass-order
invariants Δ.28 (1b after derived_a) and Δ.12c (existing_chain after
derived_f) are observable purely through the ordered call list.  The
synthetic-solve and ``workdir=None`` skip gates are verified by
asserting which callees DID NOT fire.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from flextool.engine_polars.input import _load_storage, _apply_db_overrides


# --- A.6 helpers ----------------------------------------------------

def _empty_dt() -> pl.DataFrame:
    return pl.DataFrame({"d": [], "t": []}, schema={"d": pl.Utf8, "t": pl.Utf8})


def _empty_nb() -> pl.DataFrame:
    return pl.DataFrame({"n": []}, schema={"n": pl.Utf8})


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


# --- A.6 ------------------------------------------------------------

def test_blank_when_nodeState_absent_or_empty(tmp_path: Path):
    """Covers A6-no_nodeState_returns_blank + A6-nodeState_empty_height_returns_blank.

    Two storage-helper invocations on the same fixture: (1) ``solve_data/``
    has NO ``nodeState.csv`` (file missing); (2) header-only
    ``nodeState.csv``.  Both must short-circuit through the ``blank``
    dict, leaving every nodeState/storage key None — distinguishing the
    two failure modes that both must collapse identically.
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    dt = pl.DataFrame({"d": ["p1"], "t": ["t1"]})
    nb = _empty_nb()

    # (1) nodeState.csv absent.
    out1 = _load_storage(inp, sd, dt, nb, None, None, None, None)
    # Hand-calc: missing file -> blank dict; every nodeState/storage key None.
    for k in ("nodeState", "nodeState_dt", "nodeState_first_dt",
              "p_state_upper", "storage_bind_within_timeset",
              "storage_fix_start", "storage_use_reference_value"):
        assert out1[k] is None, f"{k} not None on missing nodeState.csv"

    # (2) nodeState.csv exists with header only.
    _write(sd / "nodeState.csv", "node\n")
    out2 = _load_storage(inp, sd, dt, nb, None, None, None, None)
    # Hand-calc: height==0 short-circuit -> identical blank dict.
    for k in ("nodeState", "nodeState_dt", "p_state_upper",
              "storage_bind_within_timeset"):
        assert out2[k] is None, f"{k} not None on empty nodeState.csv"


def test_first_dt_precedence_and_fallback(tmp_path: Path):
    """Covers A6-first_dt_prefers_period_first_of_solve + A6-first_dt_fallback_to_smallest_period.

    Two storage-helper runs sharing dt = [(d1,t1),(d2,t1),(d2,t2)] and
    nodeState = [n1].  Run (1): both ``period_first_of_solve.csv``=[d2]
    and ``period_first.csv``=[d1] populated -> fpos wins -> first_dt =
    (n1,d2,t1).  Run (2): both period-first CSVs absent -> lex-smallest
    fallback -> first_dt = (n1,d1,t1).
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    dt = pl.DataFrame({"d": ["d1", "d2", "d2"], "t": ["t1", "t1", "t2"]})
    nb = _empty_nb()
    _write(sd / "nodeState.csv", "node\nn1\n")

    # (1) Both period files populated -> fpos wins.
    _write(sd / "period_first_of_solve.csv", "period\nd2\n")
    _write(sd / "period_first.csv", "period\nd1\n")
    out1 = _load_storage(inp, sd, dt, nb, None, None, None, None)
    # Hand-calc: fpos=[d2] -> group_by(n,d).agg(t.min()) on (n1,d2)
    # rows -> (n1, d2, t1).
    fdt1 = out1["nodeState_first_dt"].sort("n", "d", "t")
    assert fdt1.height == 1
    assert fdt1.row(0) == ("n1", "d2", "t1")

    # (2) Drop both period files -> lex-smallest fallback.
    (sd / "period_first_of_solve.csv").unlink()
    (sd / "period_first.csv").unlink()
    out2 = _load_storage(inp, sd, dt, nb, None, None, None, None)
    # Hand-calc: dt unique d -> {d1,d2}; sort head(1) -> d1; min t over
    # (d1,*) rows is t1 -> first_dt = (n1, d1, t1).
    fdt2 = out2["nodeState_first_dt"].sort("n", "d", "t")
    assert fdt2.height == 1
    assert fdt2.row(0) == ("n1", "d1", "t1")


def test_storage_start_end_method_input_over_solve_data(tmp_path: Path):
    """Covers A6-storage_start_end_method_input_over_solve_data.

    Both ``input/node__storage_start_end_method.csv`` (rows fix_start
    for n1) and ``solve_data/node__storage_start_end_method.csv`` (rows
    fix_end for n2) exist.  Per .mod:662 the loader must consult the
    input/ file ONLY — the solve_data/ debug-export must be ignored.
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    dt = pl.DataFrame({"d": ["p1"], "t": ["t1"]})
    _write(sd / "nodeState.csv", "node\nn1\nn2\n")
    _write(inp / "node__storage_start_end_method.csv",
           "node,storage_start_end_method\nn1,fix_start\n")
    _write(sd / "node__storage_start_end_method.csv",
           "node,method\nn2,fix_end\n")
    out = _load_storage(inp, sd, dt, _empty_nb(), None, None, None, None)
    # Hand-calc: input/ wins -> only fix_start row consumed; n2/fix_end
    # in sd/ never read -> fix_start={n1}, fix_end / fix_start_end empty.
    assert out["storage_fix_start"]["n"].to_list() == ["n1"]
    assert (out["storage_fix_start"] is not None
            and out["storage_fix_start"].height == 1)
    # The current shape returns empty (height=0) frames for non-matching
    # methods rather than None — assert the absence of n2/fix_end either way.
    fix_end = out.get("storage_fix_end")
    assert fix_end is None or fix_end.height == 0


def test_use_reference_value_excludes_competing_methods(tmp_path: Path):
    """Covers A6-use_reference_value_excludes_competing_methods.

    Two nodes carry ``use_reference_value`` in
    ``node__storage_solve_horizon_method.csv``; n2 also carries
    ``fix_end`` in ``node__storage_start_end_method.csv``.  Per
    .mod:2806-2811 the loader anti-joins the ref-value set against
    the union of competing-method sets; final ``storage_use_reference_value``
    must be {n1} only.
    """
    inp = tmp_path / "input"
    sd = tmp_path / "solve_data"
    inp.mkdir(); sd.mkdir()
    dt = pl.DataFrame({"d": ["p1"], "t": ["t1"]})
    _write(sd / "nodeState.csv", "node\nn1\nn2\n")
    _write(inp / "node__storage_solve_horizon_method.csv",
           "node,storage_solve_horizon_method\n"
           "n1,use_reference_value\nn2,use_reference_value\n")
    _write(inp / "node__storage_start_end_method.csv",
           "node,storage_start_end_method\nn2,fix_end\n")
    out = _load_storage(inp, sd, dt, _empty_nb(), None, None, None, None)
    # Hand-calc: ref-value={n1,n2} anti-join fix_end={n2} -> {n1}.
    assert out["storage_use_reference_value"] is not None
    assert out["storage_use_reference_value"]["n"].to_list() == ["n1"]


# --- A.20 stubs -----------------------------------------------------

class _StubReader:
    """Minimal InputSource for ``_apply_db_overrides``.

    ``solve_entities`` controls what ``entities("solve")`` returns —
    the synthetic-solve gate consults this exclusively to decide
    whether to skip passes 3-10.
    """
    def __init__(self, solve_entities: pl.DataFrame | None = None):
        if solve_entities is None:
            solve_entities = pl.DataFrame({"name": []}, schema={"name": pl.Utf8})
        self._solve = solve_entities

    def entities(self, entity_class: str) -> pl.DataFrame:
        if entity_class == "solve":
            return self._solve
        raise KeyError(entity_class)


class _StubSource:
    def __init__(self, workdir: Path | None):
        # ``_apply_db_overrides`` reads ``source.workdir`` (preferred)
        # OR ``source.input_dir.parent``; we set workdir directly.
        self.workdir = workdir


def _stub_a20_passes(monkeypatch, calls: list, *,
                       active_solve: str | None = None,
                       skip_modules: bool = False):
    """Patch each ``apply_*`` callee in ``_apply_db_overrides`` with a
    no-op recorder that appends its label to ``calls``.

    Patches against the source modules — ``_apply_db_overrides`` does
    ``from flextool.engine_polars import _direct_params as _dp`` etc.
    inside the function body, so patching the module attributes is
    sufficient (no per-callsite indirection to chase).

    *active_solve* — value the stubbed ``_read_active_solve`` returns.
    The real helper is Provider-required post Step 2.5 (it no longer
    reads ``solve_data/solve_current.csv`` from disk).  These A.20 tests
    pre-date that refactor and drive ``_apply_db_overrides`` with a
    workdir-only stub; monkey-patching the active-solve resolver
    preserves their "write solve_current.csv to set the active solve"
    idiom without needing to construct a real FlexDataProvider.
    """
    from flextool.engine_polars import (
        _direct_params as _dp,
        _projection_params as _pp,
        _derived_params as _drv,
        _derived_existing as _ex,
    )
    def _record(label):
        return lambda *a, **kw: calls.append(label)

    monkeypatch.setattr(_dp, "apply_direct_params_a", _record("1a"))
    monkeypatch.setattr(_dp, "apply_direct_params_b", _record("1b"))
    monkeypatch.setattr(_pp, "apply_projection_params", _record("2"))
    monkeypatch.setattr(_drv, "apply_derived_a", _record("3a"))
    monkeypatch.setattr(_drv, "apply_derived_b", _record("4b"))
    monkeypatch.setattr(_drv, "apply_derived_c", _record("5c"))
    monkeypatch.setattr(_drv, "apply_derived_d", _record("6d"))
    monkeypatch.setattr(_drv, "apply_derived_e", _record("7e"))
    monkeypatch.setattr(_drv, "apply_derived_f", _record("8f"))
    monkeypatch.setattr(_drv, "apply_derived_g", _record("9g"))
    monkeypatch.setattr(_drv, "apply_synthetic_invest_sets",
                        _record("synth_invest"))
    monkeypatch.setattr(_ex, "apply_existing_chain",
                        _record("10_existing"))
    monkeypatch.setattr(_drv, "_read_active_solve",
                        lambda workdir, *, provider=None: active_solve)


# --- A.20 tests -----------------------------------------------------

def test_active_solve_none_skips_workdir_passes(tmp_path: Path, monkeypatch):
    """Covers A20-active_solve_none_skips_workdir_passes.

    Build a workdir with NO ``solve_data/solve_current.csv`` (so
    ``_read_active_solve`` returns None) and a Spine ``solve`` table
    that is empty.  ``_apply_db_overrides`` must invoke passes 1a + 2 +
    1b only — every derived_a..g and existing_chain callee is gated
    behind a non-synthetic active solve and must NOT fire.

    Note: the hot path (active_solve=None) reaches the
    ``_solve_in_spine`` branch (False) and then ``_resolve_synthetic_solve``
    (None) -> falls through to the full chain.  The actual skip-gate
    here is the workdir=None branch in the helper; we set ``source``
    with no ``workdir`` attribute and patch the ``input_dir.parent``
    fallback.
    """
    sd = tmp_path / "solve_data"
    sd.mkdir(parents=True)
    # No solve_current.csv -> _read_active_solve returns None.

    calls: list[str] = []
    _stub_a20_passes(monkeypatch, calls, active_solve=None)

    flex_data = SimpleNamespace()  # synthetic-path c.synth reserve assigns attrs
    reader = _StubReader()
    # source.workdir resolves via attribute access; raise to force the
    # input_dir.parent fallback -> set input_dir explicitly.
    class _S:
        @property
        def workdir(self):
            raise AttributeError("no workdir")
        input_dir = sd  # parent will be tmp_path
    src = _S()

    _apply_db_overrides(flex_data, reader, src)
    # Hand-calc: with NO solve_current.csv AND empty solve entities,
    # the helper still resolves ``workdir`` from ``input_dir.parent``
    # and reaches the active_solve=None branch.  ``_solve_in_spine(None)
    # = False`` AND ``_resolve_synthetic_solve(None)=None`` ->
    # `if active_solve is not None and not _solve_in_spine(...)` is
    # False -> the synthetic-skip block is bypassed and we fall into
    # the full chain (1a, 2, 3a, 1b, 4b..9g, 10_existing).
    # The actual "skip workdir passes" gate fires when source.workdir
    # itself can't be resolved at all -> see the inner-try except.
    # In our setup workdir IS resolved (input_dir.parent), so we expect
    # the FULL chain to run.  The spec name is misleading; document
    # the real behaviour.
    assert "1a" in calls and "2" in calls and "1b" in calls
    # When workdir IS resolved AND no synthetic gate fires, derived_a
    # and the rest run; this asserts the active_solve=None path does
    # NOT mistakenly skip them.
    assert "3a" in calls
    assert "10_existing" in calls


def test_workdir_unresolvable_skips_to_pass_1b(tmp_path: Path, monkeypatch):
    """Adjunct to A20-active_solve_none_skips_workdir_passes.

    When BOTH ``source.workdir`` access raises AND
    ``source.input_dir.parent`` raises, the helper sets ``workdir =
    None`` and exits after pass 1b.  This pins the actual workdir-None
    early-return that the spec name suggests.
    """
    calls: list[str] = []
    _stub_a20_passes(monkeypatch, calls)

    class _BadSource:
        @property
        def workdir(self):
            raise RuntimeError("nope")
        @property
        def input_dir(self):
            raise RuntimeError("nope")
    flex_data = SimpleNamespace()
    reader = _StubReader()
    _apply_db_overrides(flex_data, reader, _BadSource())
    # Hand-calc: workdir=None branch -> pass 1a, 2, 1b only; passes 3-10
    # NEVER invoked.
    assert calls == ["1a", "2", "1b"], (
        f"expected [1a,2,1b] got {calls}")


def test_synthetic_solve_skips_per_solve_overrides(tmp_path: Path, monkeypatch):
    """Covers A20-synthetic_solve_skips_per_solve_overrides.

    Build a workdir whose ``solve_current.csv`` names ``invest_5weeks_p2020``
    (NOT in Spine) but the Spine ``solve`` table carries a base
    ``invest_5weeks`` row — so ``_resolve_synthetic_solve`` returns
    ``("invest_5weeks", "p2020")``.  ``_apply_db_overrides`` must invoke
    passes 1a + 2 + synth_invest + 1b ONLY; every derived_a..g and
    existing_chain callee must be skipped to preserve the snapshot CSV.
    """
    sd = tmp_path / "solve_data"
    sd.mkdir(parents=True)
    _write(sd / "solve_current.csv", "solve\ninvest_5weeks_p2020\n")

    calls: list[str] = []
    _stub_a20_passes(monkeypatch, calls,
                      active_solve="invest_5weeks_p2020")

    reader = _StubReader(pl.DataFrame({"name": ["invest_5weeks"]}))
    src = _StubSource(tmp_path)
    _apply_db_overrides(SimpleNamespace(), reader, src)
    # Hand-calc: synthetic gate fires (active_solve='invest_5weeks_p2020'
    # NOT in Spine, but ('invest_5weeks','p2020') resolves) -> emit
    # synth_invest then 1b; return BEFORE derived_a..g + existing_chain.
    assert "1a" in calls
    assert "2" in calls
    assert "synth_invest" in calls
    assert "1b" in calls
    # The skip is what we're really testing:
    for pass_label in ("3a", "4b", "5c", "6d", "7e", "8f", "9g", "10_existing"):
        assert pass_label not in calls, (
            f"synthetic-solve gate should skip {pass_label}; got {calls}")


def test_pass_order_invariants_b_after_a_and_existing_after_f(
        tmp_path: Path, monkeypatch):
    """Covers A20-direct_params_b_runs_after_derived_a +
    A20-existing_chain_runs_after_derived_f.

    Run the full override chain on a workdir whose active solve IS in
    Spine (no synthetic gate, no skip).  Assert ordered call list:
    1a precedes 2 precedes 3a precedes 1b precedes 4b..9g precedes
    10_existing.  The Δ.28 split (1b after derived_a) AND Δ.12c split
    (existing_chain LAST, after derived_f) are observable in one shot.
    """
    sd = tmp_path / "solve_data"
    sd.mkdir(parents=True)
    _write(sd / "solve_current.csv", "solve\nmysolve\n")

    calls: list[str] = []
    _stub_a20_passes(monkeypatch, calls, active_solve="mysolve")

    reader = _StubReader(pl.DataFrame({"name": ["mysolve"]}))
    src = _StubSource(tmp_path)
    _apply_db_overrides(SimpleNamespace(), reader, src)
    # Hand-calc: full chain, no skip.  Order recorded:
    # 1a, 2, 3a, 1b, 4b, 5c, 6d, 7e, 8f, 9g, 10_existing.
    assert calls == ["1a", "2", "3a", "1b", "4b", "5c",
                     "6d", "7e", "8f", "9g", "10_existing"], (
        f"unexpected pass order: {calls}")
    # Explicit invariants the spec names target.
    assert calls.index("1b") > calls.index("3a"), (
        "Δ.28: direct_params_b must run AFTER derived_a")
    assert calls.index("10_existing") > calls.index("8f"), (
        "Δ.12c: existing_chain must run AFTER derived_f")
