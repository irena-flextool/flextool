"""Every solve parameter reaches a solve that FlexTool makes from another one.

A solve made from another one (a solve of a 2-D period map, the realization solve ``<solve>__realized_dispatch``)
gets every per-solve setting of its parent through one step (``SolveConfig.derive_solve``, registry
``PER_SOLVE_FIELDS``), and its reader rows through ``SpineDbReader.register_derived_solve``; each path lists only
what it overrides or drops.  These tests set every schema ``solve`` parameter on one solve, make the copies, and
check each SolveConfig / TimelineConfig entry and each reader row of a copy against its parent's.  A new schema
parameter fails ``test_every_schema_solve_parameter_is_classified`` until it is classified here.
"""
from __future__ import annotations

import copy

import pytest

from tests.engine import _solve_copy_models as scm

#: Every schema ``solve`` parameter -> the SolveConfig / TimelineConfig field(s) it is read into.
SCHEMA_FIELDS: dict[str, tuple[str, ...]] = {
    "benders_in_out_weight": ("benders_in_out_weight",),
    "benders_max_iter": ("benders_max_iter",),
    "benders_tolerance": ("benders_tolerance",),
    "contains_solves": ("contains_solves",),
    "decomposition": ("decomposition",),
    "dispatch_realization_method": ("dispatch_realization_method",),
    "fix_storage_periods": ("fix_storage_periods",),
    "invest_periods": ("invest_periods",),
    "new_stepduration": ("timeline.new_step_durations",),
    "period_timeset": ("timesets_used_by_solves",),
    "realized_invest_periods": ("realized_invest_periods",),
    "realized_periods": ("realized_periods",),
    "reduce_system_detail_method": ("reduce_system_detail_method",),
    "reduce_unit_detail_method": ("reduce_unit_detail_method",),
    "rolling_duration": ("rolling_times",),
    "rolling_solve_horizon": ("rolling_times",),
    "rolling_solve_jump": ("rolling_times",),
    "scaling": ("scaling",),
    "shared_invest_periods": ("non_anticipativity_invest_periods",),
    "shared_operation_periods": ("non_anticipativity_periods",),
    "solve_mode": ("solve_modes", "roll_counter"),
    "solver": ("solver_settings.solvers", "solver_configs"),
    "solver_arguments": ("solver_settings.arguments", "solver_configs"),
    "solver_mip_gap": ("solver_configs",),
    "solver_precommand": ("solver_settings.precommand",),
    "stochastic_branches": ("stochastic_branches",),
    "stochastic_invest_method": ("stochastic_invest_method",),
    "timeline_hole_multiplier": ("hole_multipliers",),
    "years_represented": ("solve_period_years_represented",),
}

PERIOD_LISTS = ("invest_periods", "realized_periods", "realized_invest_periods", "fix_storage_periods")


def test_every_schema_solve_parameter_is_classified():
    """Every ``solve`` parameter of the schema is read into a field that a solve made from another one copies
    (``PER_SOLVE_FIELDS`` / the timeline's ``new_step_durations``) or that is not copied for a stated reason
    (``NOT_COPIED_FIELDS``)."""
    from flextool.engine.input._solve_config import NOT_COPIED_FIELDS, PER_SOLVE_FIELDS
    from flextool.update_flextool.fixture_schema import schema_sections

    schema = {row[1] for row in schema_sections("decoded")["parameter_definitions"] if row[0] == "solve"}
    assert set(SCHEMA_FIELDS) == schema, sorted(set(SCHEMA_FIELDS) ^ schema)
    known = set(PER_SOLVE_FIELDS) | set(NOT_COPIED_FIELDS) | {"timeline.new_step_durations"}
    assert {f for fields in SCHEMA_FIELDS.values() for f in fields} <= known


def test_every_solve_config_dict_is_classified(tmp_path):
    """Every dict attribute of a loaded SolveConfig (also inside its solver dataclasses) is a per-solve field that
    copies are made with, a per-solve field that is not copied (with its reason), or keyed by something else: a
    new per-solve attribute cannot be silently left out of the copies (``load_from_db`` runs the same check)."""
    from flextool.engine.input._solve_config import _check_field_registry

    sc, _tc, _reader = _load(_spec("realized_periods"), tmp_path)
    _check_field_registry(sc)
    sc.new_per_solve_setting = {"inv": "x"}
    with pytest.raises(RuntimeError, match="new_per_solve_setting"):
        _check_field_registry(sc)


def _all_parameters() -> list:
    """``[parameter, packed value]`` of every schema solve parameter (valid values; the 2-D trigger replaces one
    of the period lists)."""
    stochastic = scm.pmap([(p, scm.nested([("high", scm.nested([(t, scm.nested([("no", 1.0)]))])),
                                         ("low", scm.nested([(t, scm.nested([("no", 3.0)]))]))]))
                           for p, t in (("p1", "t01"), ("p2", "t03"))])
    return [
        ["benders_in_out_weight", scm.pack(0.3, "float")],
        ["benders_max_iter", scm.pack(7.0, "float")],
        ["benders_tolerance", scm.pack(0.01, "float")],
        ["contains_solves", scm.array(["child"])],
        ["decomposition", scm.pack("benders", "str")],
        ["dispatch_realization_method", scm.pack("separate_dispatch", "str")],
        ["new_stepduration", scm.pack(2.0, "float")],
        ["reduce_system_detail_method", scm.pack("reserves", "str")],
        ["reduce_unit_detail_method", scm.pack("integers", "str")],
        ["rolling_duration", scm.pack(4.0, "float")],
        ["rolling_solve_horizon", scm.pack(2.0, "float")],
        ["rolling_solve_jump", scm.pack(1.0, "float")],
        ["scaling", scm.pack("basic", "str")],
        ["shared_invest_periods", scm.array(["p1"])],
        ["shared_operation_periods", scm.array(["p2"])],
        ["solver", scm.pack("gurobi", "str")],
        ["solver_arguments", scm.pack(scm.pmap([("TimeLimit", "800")]), "map")],
        ["solver_mip_gap", scm.pack(0.2, "float")],
        ["solver_precommand", scm.pack("echo", "str")],
        ["stochastic_branches", scm.pack(stochastic, "map")],
        ["stochastic_invest_method", scm.pack("recourse", "str")],
        ["timeline_hole_multiplier", scm.pack(2.0, "float")],
    ]


def _split_params(trigger: str | None) -> tuple[str, ...]:
    """The period lists that are 2-D maps for *trigger*: a 2-D ``invest_periods`` needs a 2-D
    ``realized_invest_periods`` with the same solve names (``_check_realized_invest_periods``)."""
    if trigger is None:
        return ()
    return (trigger, "realized_invest_periods") if trigger == "invest_periods" else (trigger,)


def _spec(trigger: str | None) -> dict:
    """Model A whose solve ``inv`` sets every schema solve parameter: the period lists over both periods (1-D),
    except the :func:`_split_params` of *trigger*, the 2-D map ``{p1: [p1], p2: [p2]}``; solve_mode
    rolling_window (the last row wins over model A's single_solve)."""
    spec = scm.model_a("split", split_params=_split_params(trigger),
                       solve_rows=_all_parameters() + [
                           ["fix_storage_periods", scm.two_d({p: [p] for p in scm.PERIODS})
                            if trigger == "fix_storage_periods" else scm.array(list(scm.PERIODS))]])
    spec["entities"].append(["solve", "child", None])
    for row in spec["parameter_values"]:
        if row[:3] == ["solve", "inv", "solve_mode"]:
            row[3] = scm.pack("rolling_window", "str")
    return spec


def _load(spec: dict, folder):
    """``(SolveConfig, TimelineConfig, reader)`` of *spec*, as the cascade prepares them (copies registered)."""
    from flextool.engine.input._solve_config import SolveConfig, register_solve_copies
    from flextool.engine.input._spinedb_reader import SpineDbReader
    from flextool.engine.input._timeline import TimelineConfig

    url = scm.build_db(scm.write_spec(spec, folder), folder)
    sc = SolveConfig.load_from_db_url(url, "base")
    tc = TimelineConfig.load_from_db_url(url, "base")
    tc.inherit_solve_copies(sc)
    reader = SpineDbReader(url, scenario="base")
    register_solve_copies(sc, reader)
    return sc, tc, reader


def _field(sc, tc, name: str) -> dict:
    if name == "timeline.new_step_durations":
        return tc.new_step_durations
    obj = sc
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


def _reader_values(reader, solve: str) -> dict:
    """``{parameter: parsed value}`` of the ``solve`` entity *solve* in *reader*."""
    solve_cls = reader._class_name_to_id["solve"]
    eid = next(e for e, name, _ in reader._entities_by_class[solve_cls] if name == solve)
    names = {pdef["id"]: name for (cls, name), pdef in reader._pdef_by_class_name.items() if cls == solve_cls}
    return {names[pdef_id]: value for (cls, pdef_id), rows in reader._param_rows.items() if cls == solve_cls
            for e, value in rows if e == eid}


def _map_rows(value, names=("period", "branch", "analysis_time", "realized")) -> list[tuple]:
    """The rows of a nested map *value*, checking that its levels have the index names *names*."""
    assert value.index_name == names[0]
    out = []
    for key, inner in zip(value.indexes, value.values):
        if len(names) == 1:
            out.append((str(key), inner))
        else:
            out += [(str(key), *row) for row in _map_rows(inner, names[1:])]
    return out


def _periods(value) -> list[str]:
    return [str(v) for v in value.values]


@pytest.mark.parametrize("trigger", PERIOD_LISTS)
def test_a_split_solve_has_every_parameter_of_its_database_solve(tmp_path, trigger):
    """For each 2-D trigger: the split solves ``inv_p1`` / ``inv_p2`` equal ``inv`` in every field except their
    own period lists (the trigger's inner list; the 1-D lists cut to the solve's period; shared_invest_periods
    with no entry where the cut is empty, shared_operation_periods an empty list there) and timesets; their
    reader rows equal ``inv``'s except the same period lists, as arrays, and period_timeset / years_represented
    as maps of their own period with the index name ``period``."""
    from flextool.engine.input._solve_config import NOT_COPIED_FIELDS

    sc, tc, reader = _load(_spec(trigger), tmp_path)
    assert sc.model_solve["flextool"] == ["inv_p1", "inv_p2"]
    own_fields = {"invest_periods", "realized_periods", "realized_invest_periods", "fix_storage_periods",
                  "non_anticipativity_periods", "non_anticipativity_invest_periods", "timesets_used_by_solves",
                  "stochastic_branches"}
    for p in scm.PERIODS:
        new = f"inv_{p}"
        assert sc.map2d_duplicate_of[new] == ("inv", p)
        for par, fields in SCHEMA_FIELDS.items():
            for name in fields:
                values = _field(sc, tc, name)
                if name in NOT_COPIED_FIELDS:
                    assert new not in values, (par, name)
                elif name not in own_fields:
                    assert new in values and values[new] == values["inv"], (par, name)
        for field in PERIOD_LISTS:
            assert sc.__dict__[field][new] == [(p, p)], field
        assert sc.timesets_used_by_solves[new] == [(p, scm.PERIODS[p])]
        assert sc.non_anticipativity_invest_periods.get(new) == ([("p1", "p1")] if p == "p1" else None)
        assert sc.non_anticipativity_periods[new] == ([] if p == "p1" else [("p2", "p2")])
        # Each split solve keeps the forecast fan of the period it realizes (item 257).
        assert sc.stochastic_branches[new] == [row for row in sc.stochastic_branches["inv"] if row[0] == p]
        assert len(sc.stochastic_branches[new]) == 2

        parent_rows, rows = _reader_values(reader, "inv"), _reader_values(reader, new)
        replaced = {*PERIOD_LISTS, "period_timeset", "years_represented", "shared_invest_periods",
                    "shared_operation_periods", "stochastic_branches"}
        # The trigger's 2-D row is gone from the database solve; inv_p2 has no shared_invest_periods (cut empty).
        expected = (set(parent_rows) | set(_split_params(trigger))) - ({"shared_invest_periods"} if p == "p2"
                                                                       else set())
        assert set(rows) == expected, sorted(set(rows) ^ expected)
        for par, value in rows.items():
            if par not in replaced:
                assert value == parent_rows[par], par
        for par in PERIOD_LISTS:
            assert _periods(rows[par]) == [p], par
        assert _periods(rows["shared_operation_periods"]) == ([] if p == "p1" else ["p2"])
        assert ("shared_invest_periods" in rows) == (p == "p1")
        for par, expected in (("period_timeset", scm.PERIODS[p]), ("years_represented", 5.0)):
            assert rows[par].index_name == "period", par
            assert list(map(str, rows[par].indexes)) == [p] and list(rows[par].values) == [expected], par
        assert _map_rows(rows["stochastic_branches"]) == [
            (p, branch, t, "no", w) for branch, w in (("high", 1.0), ("low", 3.0)) for t in [scm.STEPS[2 * (p == "p2")]]]
    # The database solve's own 2-D rows are gone from the reader; its other rows stay.
    assert not set(_split_params(trigger)) & set(_reader_values(reader, "inv"))


def test_the_realization_solve_has_every_parameter_except_the_dropped_ones(tmp_path):
    """The realization solve gets every field of its parent except ``REALIZATION_SOLVE_DROPS`` (and the
    never-copied ``solver_precommand``): solve_mode is the parent's (rolling_window here), its realized window is
    the parent's; the solver, its arguments and gap, scaling, the detail levels, years_represented and the rolling
    times are copied.  The reader half drops ``REALIZATION_SOLVE_DROPPED_PARAMS`` and copies every other row."""
    from flextool.engine.input._solve_config import NOT_COPIED_FIELDS, REALIZATION_SOLVE_DROPS
    from flextool.engine.input._spinedb_reader import SpineDbReader

    sc, tc, reader = _load(_spec(None), tmp_path)
    parent, sibling = "inv", "inv__realized_dispatch"
    before = copy.deepcopy(sc.model_solve)
    sc.spawn_realized_dispatch_sibling(parent, sibling)
    reader.register_derived_solve(parent, sibling, drop=SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS)
    assert sc.model_solve["flextool"] == before["flextool"][:1] + [sibling] + before["flextool"][1:]
    for par, fields in SCHEMA_FIELDS.items():
        for name in fields:
            if name == "timeline.new_step_durations":
                continue  # set by _spawn_realized_dispatch_siblings, covered by test_realized_dispatch_coherence
            values = _field(sc, tc, name)
            if name in REALIZATION_SOLVE_DROPS or name in NOT_COPIED_FIELDS:
                assert sibling not in values, (par, name)
            else:
                assert values[sibling] == values[parent], (par, name)
    assert sc.solve_modes[sibling] == "rolling_window"
    parent_rows, rows = _reader_values(reader, parent), _reader_values(reader, sibling)
    assert set(rows) == set(parent_rows) - set(SpineDbReader.REALIZATION_SOLVE_DROPPED_PARAMS)
    assert all(rows[par] == parent_rows[par] for par in rows)
