"""model_builder offers investment only when the spec asks for it.

An ``invest_method`` alone offers investment even without ``invest_cost`` (free
investment), so the builder must not write one for a spec without a cost.
"""

from flextool.model_builder.build_model import ModelBuilder

_BASE = {
    "profile_lengths": 4,
    "node_locations": {"island": {"number": 2, "distribution": {
        "name": "pert", "min": 50, "mode": 100, "max": 150}}},
    "grids": {"elec": {"node_group": "island"}},
    "fuels": {"natgas": {"price": 10}},
}


def _params(spec: dict) -> dict[tuple[str, str, str], object]:
    result = ModelBuilder(spec, seed=1).build()
    return {(pv[0], pv[1] if isinstance(pv[1], str) else "__".join(pv[1]), pv[2]): pv[3]
            for pv in result["parameter_values"]}


def _spec(storage: dict, connection: dict, tech: dict) -> dict:
    return {**_BASE,
            "storages": {"elec": {"battery": {"storage": storage, "connection": connection}}},
            "technologies": {"natgas": {"elec": {"ocgt": {
                "conversion_method": "constant_efficiency", "efficiency": 0.4, **tech}}}}}


def _methods(params: dict, cls: str, prefix: str) -> dict[str, object]:
    return {name: v for (c, name, p), v in params.items()
            if c == cls and p == "invest_method" and name.startswith(prefix)}


def test_no_invest_cost_writes_no_invest_method():
    params = _params(_spec({}, {"efficiency": 0.9}, {}))
    assert _methods(params, "node", "battery_") == {}
    assert _methods(params, "connection", "battery_conn_") == {}
    assert _methods(params, "unit", "ocgt_") == {}


def test_invest_cost_writes_invest_no_limit():
    params = _params(_spec({"invest_cost": 300}, {"invest_cost": 50}, {"invest_cost": 700}))
    for cls, prefix in (("node", "battery_"), ("connection", "battery_conn_"), ("unit", "ocgt_")):
        methods = _methods(params, cls, prefix)
        assert len(methods) == 2, (cls, methods)
        assert set(methods.values()) == {"invest_no_limit"}


def test_explicit_invest_method_is_kept_without_cost():
    params = _params(_spec({"invest_method": "invest_total"}, {}, {"invest_method": "invest_period"}))
    assert set(_methods(params, "node", "battery_").values()) == {"invest_total"}
    assert _methods(params, "connection", "battery_conn_") == {}
    assert set(_methods(params, "unit", "ocgt_").values()) == {"invest_period"}
