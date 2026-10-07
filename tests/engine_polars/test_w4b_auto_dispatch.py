"""W4b — engine-automatic realized-dispatch sibling for a single
top-level ``separate_dispatch`` solve.

W8.1 (``test_w8_realized_dispatch.py``) proved that realized operations
can come from a SEPARATE deterministic dispatch solve whose investments
are fixed from a stochastic solve — using an AUTHOR-DECLARED second solve.
W4b removes the author-declared solve: a single top-level solve that sets
``dispatch_realization_method = separate_dispatch`` causes the engine to
AUTOMATICALLY register a deterministic realized-dispatch SIBLING
``<parent>__realized_dispatch`` that takes the full override path
(invest→dispatch handoff fixes committed capacity; realized last-wins
empties the parent's realized window; the sibling owns the reported
realized operations).  The auto-spawned sibling must be INDISTINGUISHABLE
from W8.1's hand-declared dispatch solve.

This module pins:

  * the four W8.1 oracles, now against the AUTO-spawned
    ``stoch_rd__realized_dispatch`` step
    (``w4b_auto_dispatch`` fixture — W8.1 collapsed to one solve +
    ``dispatch_realization_method=separate_dispatch``);
  * a RESERVE full-path proof — a ``separate_dispatch`` solve carrying an
    up-reserve requirement; the sibling's LP must contain a reserve
    variable (the synthetic ``<base>_<anchor>`` fallback would silently
    drop the reserve subsystem), proving the sibling takes the real DB
    override path;
  * the two in-memory registration primitives
    (:meth:`SpineDbReader.register_derived_solve`,
    :meth:`SolveConfig.spawn_realized_dispatch_sibling`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl
import pytest

FLEXTOOL_ROOT = Path(__file__).resolve().parents[2]
AUTO_FIXTURE = FLEXTOOL_ROOT / "tests" / "fixtures" / "w4b_auto_dispatch.json"
RESERVE_FIXTURE = (
    FLEXTOOL_ROOT / "tests" / "fixtures" / "w4b_reserve_dispatch.json"
)

SCENARIO = "w8_realized_dispatch"
PARENT = "stoch_rd"
SIBLING = "stoch_rd__realized_dispatch"
REALIZED_STEPS = {"t0001", "t0002", "t0003"}

# Hand-calc constants (identical to W8.1 — the sibling reproduces them).
COMMITTED_PEAKER_MW = 60.0
HOURS_PER_YEAR = 8760.0
PEAKER_VAR_COST = 10.0
DISPATCH_OBJECTIVE = COMMITTED_PEAKER_MW * PEAKER_VAR_COST * HOURS_PER_YEAR
RESERVE_MW = 10.0


def _build_db(fixture_json: Path, tmp_path: Path) -> str:
    """Build-from-JSON + migrate (CLAUDE.md invariant 3; never read a
    checked-in .sqlite)."""
    if not fixture_json.exists():
        pytest.skip(f"fixture JSON not present: {fixture_json}")
    tests_dir = FLEXTOOL_ROOT / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from db_utils import json_to_db  # noqa: E402
    from flextool.update_flextool.db_migration import migrate_database

    db_path = tmp_path / f"{fixture_json.stem}.sqlite"
    url = json_to_db(fixture_json, db_path)
    migrate_database(url)
    return url


def _run_chain(fixture_json: Path, tmp_path: Path):
    """Run the single-solve chain once under ``FLEXTOOL_AUTOSCALE_STRICT=1``
    with ``warm=False`` + ``keep_solutions=True`` and return
    ``(steps, work_folder)``."""
    from flextool.engine_polars import run_chain_from_db

    url = _build_db(fixture_json, tmp_path)
    work_folder = tmp_path / "work"
    work_folder.mkdir()

    prev = os.environ.get("FLEXTOOL_AUTOSCALE_STRICT")
    os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = "1"
    try:
        steps = run_chain_from_db(
            url, SCENARIO, work_folder=work_folder,
            warm=False, keep_solutions=True,
        )
    finally:
        if prev is None:
            os.environ.pop("FLEXTOOL_AUTOSCALE_STRICT", None)
        else:
            os.environ["FLEXTOOL_AUTOSCALE_STRICT"] = prev
    assert steps, "run_chain_from_db produced no orchestration steps"
    return steps, work_folder


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("w4b_auto_dispatch")
    return _run_chain(AUTO_FIXTURE, tmp_path)


@pytest.fixture(scope="module")
def reserve_chain(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("w4b_reserve_dispatch")
    return _run_chain(RESERVE_FIXTURE, tmp_path)


def _step(steps, needle):
    key = next((k for k in steps if k == needle), None) or next(
        (k for k in steps if needle in k), None
    )
    assert key is not None, f"no {needle!r} step in {list(steps)}"
    return steps[key]


def _as_frame(param):
    if param is None:
        return None
    return param.frame if hasattr(param, "frame") else param


# ===========================================================================
# The sibling is auto-spawned at all
# ===========================================================================


@pytest.mark.solver
def test_sibling_is_auto_registered(chain) -> None:
    """The engine spawned ``<parent>__realized_dispatch`` even though the
    fixture declares only ONE solve — no author-declared dispatch solve."""
    steps, _wf = chain
    assert PARENT in steps, f"parent solve missing: {list(steps)}"
    assert SIBLING in steps, (
        f"engine did not auto-register the realized-dispatch sibling "
        f"{SIBLING!r}: {list(steps)}"
    )


# ===========================================================================
# (a) both solves optimal
# ===========================================================================


@pytest.mark.solver
def test_both_solves_optimal(chain) -> None:
    steps, _wf = chain
    stoch = _step(steps, PARENT)
    dispatch = _step(steps, SIBLING)
    assert stoch.solution is not None and stoch.optimal
    assert dispatch.solution is not None and dispatch.optimal


# ===========================================================================
# (b) sibling builds NO v_invest; capacity fixed at the committed 60 MW
# ===========================================================================


@pytest.mark.solver
def test_sibling_capacity_is_fixed_from_invest(chain) -> None:
    steps, _wf = chain
    stoch = _step(steps, PARENT)
    dispatch = _step(steps, SIBLING)

    assert stoch.flex_data is not None
    assert stoch.flex_data.pd_invest_set is not None
    pcol = stoch.flex_data.pd_invest_set.columns[0]
    assert "peaker" in set(stoch.flex_data.pd_invest_set[pcol].to_list())

    assert stoch.handoff is not None and not stoch.handoff.is_empty()
    ri = stoch.handoff.realized_invest
    assert ri is not None and ri.height > 0
    ent_col, val_col = ri.columns[0], ri.columns[-1]
    committed = {r[ent_col]: float(r[val_col]) for r in ri.iter_rows(named=True)}
    assert committed.get("peaker") == pytest.approx(COMMITTED_PEAKER_MW)

    # sibling builds no invest set / v_invest ...
    assert dispatch.flex_data is not None
    assert dispatch.flex_data.pd_invest_set is None
    disp_vars = set(getattr(dispatch.solution, "_vars", {}).keys())
    assert not any("v_invest" in v for v in disp_vars), (
        f"sibling built v_invest variables {sorted(disp_vars)}"
    )

    # ... yet carries the committed capacity as fixed existing capacity.
    pei = _as_frame(dispatch.flex_data.p_entity_invested)
    assert pei is not None
    pei_ent, pei_val = pei.columns[0], pei.columns[-1]
    dispatch_cap = {r[pei_ent]: float(r[pei_val]) for r in pei.iter_rows(named=True)}
    assert dispatch_cap.get("peaker") == pytest.approx(COMMITTED_PEAKER_MW)
    assert dispatch_cap["peaker"] == pytest.approx(committed["peaker"])


# ===========================================================================
# (c) last-wins: sibling owns the realized window
# ===========================================================================


@pytest.mark.solver
def test_realized_window_owned_by_sibling(chain) -> None:
    steps, _wf = chain
    stoch = _step(steps, PARENT)
    dispatch = _step(steps, SIBLING)

    stoch_rd = stoch.flex_data.realized_dispatch
    if stoch_rd is not None:
        remaining = stoch_rd.filter(pl.col("period") == "p2035")
        assert remaining.height == 0, (
            "parent still owns realized p2035 cells — last-wins did not hand "
            f"them to the auto sibling:\n{stoch_rd}"
        )

    disp_rd = dispatch.flex_data.realized_dispatch
    assert disp_rd is not None and disp_rd.height > 0
    owned = disp_rd.filter(pl.col("period") == "p2035")
    assert set(owned["step"].to_list()) == REALIZED_STEPS


# ===========================================================================
# (d) hand-calc of the sibling's objective
# ===========================================================================


@pytest.mark.solver
def test_sibling_objective_hand_calc(chain) -> None:
    steps, wf = chain
    dispatch = _step(steps, SIBLING)

    obj = float(
        pl.read_parquet(wf / "output_raw" / f"v_obj__{SIBLING}.parquet")
        ["objective"][0]
    )
    assert obj == pytest.approx(DISPATCH_OBJECTIVE, rel=1e-9)

    flows = dispatch.solution.value("v_flow")
    peaker = flows.filter(pl.col("source") == "peaker")
    assert peaker.height == 3
    assert all(v == pytest.approx(60.0) for v in peaker["value"].to_list())
    wind = flows.filter(pl.col("source") == "wind")
    assert wind.height == 3
    assert all(v == pytest.approx(0.40) for v in wind["value"].to_list())


# ===========================================================================
# RESERVE full-path proof — the synthetic path would drop the reserve
# ===========================================================================


@pytest.mark.solver
def test_reserve_present_in_sibling_lp(reserve_chain) -> None:
    """A ``separate_dispatch`` solve carrying an up-reserve requirement:
    the auto sibling's LP must CONTAIN the reserve subsystem (``v_reserve``
    and the reserve relationship frames), and the peaker must actually hold
    the 10 MW/step reserve on its FIXED capacity.

    This is a non-vacuous LP-completeness + provision check (it would catch
    a regression that dropped reserves from the sibling), but note it is NOT
    itself the full-vs-synthetic discriminator: the reserve relationship
    frames (``process_reserve_upDown_node_active`` /
    ``reserve_upDown_group``) are built unconditionally from the ENTITY
    source, not keyed on the solve name, so they survive either path.  The
    GENUINE full-vs-synthetic discriminator lives in the oracle tests
    above: on the synthetic ``<base>_<anchor>`` path
    ``_solve_periods(sibling, "realized_periods")`` returns ``None``, so
    last-wins would never empty the parent's p2035 window and the realized-
    ownership + exact-objective assertions would fail.  Here we additionally
    confirm the reserve subsystem is live on the sibling's fixed-capacity
    dispatch."""
    steps, _wf = reserve_chain
    stoch = _step(steps, PARENT)
    dispatch = _step(steps, SIBLING)
    assert stoch.optimal and dispatch.optimal

    disp_vars = set(getattr(dispatch.solution, "_vars", {}).keys())
    assert "v_reserve" in disp_vars, (
        "sibling built NO v_reserve — the reserve subsystem was dropped, so "
        f"the sibling fell back to the synthetic path. vars={sorted(disp_vars)}"
    )

    # The reserve relationship frames survived onto the sibling's FlexData.
    assert dispatch.flex_data.reserve_upDown_group is not None
    assert dispatch.flex_data.reserve_upDown_group.height > 0
    assert dispatch.flex_data.process_reserve_upDown_node_active is not None
    assert dispatch.flex_data.process_reserve_upDown_node_active.height > 0

    # The peaker provides the full 10 MW/step up-reserve on its fixed cap.
    vr = dispatch.solution.value("v_reserve")
    assert vr.height == 3
    assert all(v == pytest.approx(RESERVE_MW) for v in vr["value"].to_list()), (
        f"sibling reserve provision != {RESERVE_MW} MW/step:\n{vr}"
    )


# ===========================================================================
# Unit tests — the two in-memory registration primitives (no solver)
# ===========================================================================


def test_register_derived_solve_makes_sibling_a_real_spine_solve(tmp_path):
    """``register_derived_solve`` lists the sibling in ``entities('solve')``
    with a FRESH entity id, so ``_solve_in_spine`` is True and
    ``_solve_periods`` reads the cloned realized window — while NOT cloning
    ``invest_periods`` (the sibling must not invest)."""
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._derived_params import (
        _solve_in_spine,
        _solve_periods,
    )

    url = _build_db(AUTO_FIXTURE, tmp_path)
    reader = SpineDbReader(url, scenario=SCENARIO)

    assert not _solve_in_spine(reader, SIBLING)
    solve_cls = reader._class_name_to_id["solve"]
    before_ids = set(reader._entity_by_id)

    reader.register_derived_solve(PARENT, SIBLING)

    # Fresh id, listed in entities('solve'), _solve_in_spine True.
    new_ids = set(reader._entity_by_id) - before_ids
    assert len(new_ids) == 1
    new_eid = new_ids.pop()
    assert new_eid == max(before_ids) + 1
    assert reader._entity_by_id[new_eid] == (solve_cls, SIBLING)
    assert SIBLING in set(reader.entities("solve")["name"].to_list())
    assert _solve_in_spine(reader, SIBLING)

    # realized window cloned; invest_periods NOT cloned.
    assert _solve_periods(reader, SIBLING, "realized_periods") == ["p2035"]
    assert _solve_periods(reader, PARENT, "invest_periods") == ["p2035"]
    assert _solve_periods(reader, SIBLING, "invest_periods") is None

    # Idempotent — a second call is a no-op (no duplicate id).
    reader.register_derived_solve(PARENT, SIBLING)
    assert set(reader._entity_by_id) - before_ids == {new_eid}


def test_register_derived_solve_missing_parent_raises(tmp_path):
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._solve_state import FlexToolConfigError

    url = _build_db(AUTO_FIXTURE, tmp_path)
    reader = SpineDbReader(url, scenario=SCENARIO)
    with pytest.raises(FlexToolConfigError):
        reader.register_derived_solve("no_such_solve", "no_such__realized_dispatch")


def test_spawn_realized_dispatch_sibling_copies_sets_and_drops(tmp_path):
    """``spawn_realized_dispatch_sibling`` inserts the sibling right after
    the parent in ``model_solve``, COPIES the timeline scaffolding, SETS
    ``single_solve`` + the realized window, and DROPS the invest /
    stochastic / decomposition keys."""
    from flextool.engine_polars._solve_config import SolveConfig

    url = _build_db(AUTO_FIXTURE, tmp_path)
    sc = SolveConfig.load_from_db_url(url, SCENARIO)

    # Pre-populate a few per-solve dicts on the parent so COPY is observable.
    sc.timesets_used_by_solves[PARENT] = [("p2035", "ts1")]
    sc.realized_periods[PARENT] = [("p2035", "p2035")]
    sc.invest_periods[PARENT] = [("p2035", "p2035")]

    model = next(iter(sc.model_solve))
    assert sc.model_solve[model] == [PARENT]

    sc.spawn_realized_dispatch_sibling(PARENT, SIBLING)

    # Positional insert right AFTER the parent.
    assert sc.model_solve[model] == [PARENT, SIBLING]
    # SET.
    assert sc.solve_modes[SIBLING] == "single_solve"
    assert sc.realized_periods[SIBLING] == sc.realized_periods[PARENT]
    # COPY (deep — a distinct object).
    assert sc.timesets_used_by_solves[SIBLING] == sc.timesets_used_by_solves[PARENT]
    assert sc.timesets_used_by_solves[SIBLING] is not sc.timesets_used_by_solves[PARENT]
    # DROP — the sibling is investment-free / monolithic.  invest_periods is
    # a defaultdict(list): a missing key reads back as the empty default.
    assert SIBLING not in sc.invest_periods
    assert sc.invest_periods[SIBLING] == []
    assert SIBLING not in sc.realized_invest_periods
    assert SIBLING not in sc.fix_storage_periods
    assert SIBLING not in sc.decomposition
    assert SIBLING not in sc.stochastic_invest_method
    assert SIBLING not in sc.contains_solves
    # dispatch_realization_method absent → resolver yields within_solve →
    # the run()-site guard no-ops → no re-spawn.
    assert sc.dispatch_realization_method_for(SIBLING) == "within_solve"


def test_spawn_helper_registers_in_both_reader_and_config(tmp_path):
    """The orchestration helper performs the DUAL registration: the sibling
    lands in the reader's ``solve`` entities (full override path) AND right
    after the parent in the driving config's ``model_solve``."""
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._orchestration import (
        _spawn_realized_dispatch_siblings,
    )

    url = _build_db(AUTO_FIXTURE, tmp_path)
    reader = SpineDbReader(url, scenario=SCENARIO)
    sc = SolveConfig.load_from_db_url(url, SCENARIO)
    model = next(iter(sc.model_solve))

    _spawn_realized_dispatch_siblings(sc, reader, logger=None)

    assert SIBLING in set(reader.entities("solve")["name"].to_list())
    assert sc.model_solve[model] == [PARENT, SIBLING]
    assert sc.solve_modes[SIBLING] == "single_solve"


def test_spawn_helper_collision_raises(tmp_path):
    """A pre-existing solve named ``<parent>__realized_dispatch`` (here
    injected into the reader) makes the helper raise, naming the colliding
    owner, rather than silently clobbering it."""
    from flextool.engine_polars._spinedb_reader import SpineDbReader
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._orchestration import (
        _spawn_realized_dispatch_siblings,
    )
    from flextool.engine_polars._solve_state import FlexToolConfigError

    url = _build_db(AUTO_FIXTURE, tmp_path)
    reader = SpineDbReader(url, scenario=SCENARIO)
    # Inject a colliding 'solve' entity into the reader.
    solve_cls = reader._class_name_to_id["solve"]
    collide_eid = max(reader._entity_by_id) + 1
    reader._entities_by_class[solve_cls].append((collide_eid, SIBLING, None))
    reader._entity_by_id[collide_eid] = (solve_cls, SIBLING)

    sc = SolveConfig.load_from_db_url(url, SCENARIO)
    with pytest.raises(FlexToolConfigError) as exc:
        _spawn_realized_dispatch_siblings(sc, reader, logger=None)
    assert SIBLING in str(exc.value)


def test_spawn_helper_fix_storage_guard_raises():
    """Positive storage-pinning guard (investigation outcome): a
    ``separate_dispatch`` parent authoring ANY non-empty
    ``fix_storage_periods`` is rejected — the auto sibling drops
    ``fix_storage_periods`` (the only solve-level storage mechanism it
    sheds), which would silently re-float the pinned storage boundaries.
    Rejection is CONSERVATIVE and independent of realized-period count
    (gating on >1 realized period would UNDER-fire for a single realized
    period with a multi-period fix_storage authored).  Empty / unset
    ``fix_storage_periods`` is NOT rejected."""
    from flextool.engine_polars._solve_config import SolveConfig
    from flextool.engine_polars._orchestration import (
        _spawn_realized_dispatch_siblings,
    )
    from flextool.engine_polars._solve_state import FlexToolConfigError

    class _FakeReader:
        def __init__(self):
            self.registered = []

        def entities(self, _cls):
            return pl.DataFrame({"name": [PARENT]})

        def register_derived_solve(self, parent, sibling):
            self.registered.append((parent, sibling))

    def _cfg(realized, fix_storage):
        sc = object.__new__(SolveConfig)
        sc.model_solve = {"m": [PARENT]}
        sc.contains_solves = {}
        sc.solve_modes = {PARENT: "single_solve"}
        sc.dispatch_realization_method = {PARENT: "separate_dispatch"}
        sc.realized_periods = {PARENT: realized}
        sc.fix_storage_periods = {PARENT: fix_storage}
        return sc

    # Multi-period fix_storage_periods → reject.
    sc = _cfg([("p2035", "p2035"), ("p2040", "p2040")],
              [("p2035", "p2035"), ("p2040", "p2040")])
    with pytest.raises(FlexToolConfigError) as exc:
        _spawn_realized_dispatch_siblings(sc, _FakeReader(), logger=None)
    assert "fix_storage_periods" in str(exc.value)

    # SINGLE realized period but a (2-period) fix_storage authored → still
    # reject (this is exactly the case a realized-period-count gate would
    # miss — the single pinned boundary is dropped by the sibling).
    sc_single = _cfg([("p2035", "p2035")],
                     [("p2035", "p2035"), ("p2040", "p2040")])
    with pytest.raises(FlexToolConfigError) as exc:
        _spawn_realized_dispatch_siblings(sc_single, _FakeReader(), logger=None)
    assert "fix_storage_periods" in str(exc.value)

    # Multiple realized periods but NO fix_storage_periods → allowed
    # (the monolithic solve links storage via its own balance constraints).
    sc_ok = _cfg([("p2035", "p2035"), ("p2040", "p2040")], [])
    reader_ok = _FakeReader()
    _spawn_realized_dispatch_siblings(sc_ok, reader_ok, logger=None)
    assert reader_ok.registered == [(PARENT, SIBLING)]
