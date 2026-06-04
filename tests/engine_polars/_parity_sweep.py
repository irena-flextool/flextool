"""Phase 3d — curated parity-sweep case list for engine_polars cluster tests.

Replaces the legacy disk-discovery functions in
``test_arithmetic_cluster.py``, ``test_block_cluster.py``,
``test_block_layout.py``, ``test_npv_cluster.py``,
``test_profile_cluster.py``, and the opt-in sweep in
``test_orchestration.py``.  Each entry is
``(legacy_workdir_dirname, scenario_name, db_fixture)``:

- ``legacy_workdir_dirname`` — preserved as a stable test ID so historical
  xfail keys (e.g. ``("work_delay_source_coef", "p_penalty_up")``) keep
  matching.
- ``scenario_name`` — passed to :func:`scenario_workdir` to build the
  workdir on demand.
- ``db_fixture`` — the ``db_fixture`` keyword arg to
  :func:`scenario_workdir` (which JSON fixture backs the DB).

Living in a sibling module rather than ``conftest.py`` so that test
files can ``from _parity_sweep import PARITY_SWEEP_CASES`` without
relying on pytest's conftest discovery (which doesn't fire on plain
module imports during e.g. linting or static analysis).
"""

PARITY_SWEEP_CASES: list[tuple[str, str, str]] = [
    # Main fixture scenarios
    ("work_base",                          "base",                          "main"),
    ("work_base_weighted",                 "base_weighted",                 "main"),
    ("work_capacity_margin",               "capacity_margin",               "main"),
    ("work_coal",                          "coal",                          "main"),
    ("work_coal_chp",                      "coal_chp",                      "main"),
    ("work_coal_chp_extraction",           "coal_chp_extraction",           "main"),
    ("work_coal_co2_limit",                "coal_co2_limit",                "main"),
    ("work_coal_co2_price",                "coal_co2_price",                "main"),
    ("work_coal_min_load",                 "coal_min_load",                 "main"),
    ("work_coal_min_load_wind",            "coal_min_load_wind",            "main"),
    ("work_coal_ramp_limit",               "coal_ramp_limit",               "main"),
    ("work_coal_retire",                   "coal_retire",                   "main"),
    ("work_coal_wind_ev",                  "coal_wind_ev",                  "main"),
    ("work_coal_wind_inertia",             "coal_wind_inertia",             "main"),
    ("work_coal_wind_min_uptime",          "coal_wind_min_uptime",          "main"),
    ("work_dr_decrease_demand",            "dr_decrease_demand",            "main"),
    ("work_dr_increase_demand",            "dr_increase_demand",            "main"),
    ("work_dr_shift_demand",               "dr_shift_demand",               "main"),
    ("work_fullYear",                      "fullYear",                      "main"),
    ("work_hyphenated_entity_names",       "hyphenated_entity_names",       "main"),
    ("work_multi_fullYear_battery",        "multi_fullYear_battery",        "main"),
    ("work_multi_year",                    "multi_year",                    "main"),
    ("work_multi_year_one_solve",          "multi_year_one_solve",          "main"),
    ("work_multi_year_one_solve_battery",  "multi_year_one_solve_battery",  "main"),
    ("work_multi_year_one_solve_co2_limit","multi_year_one_solve_co2_limit","main"),
    ("work_multi_year_wind_growth_cap",    "multi_year_wind_growth_cap",    "main"),
    ("work_multi_year_wind_no_investment", "multi_year_wind_no_investment", "main"),
    ("work_network_coal_wind",             "network_coal_wind",             "main"),
    ("work_network_all_tech",              "network_all_tech",              "main"),
    ("work_unidirectional_connection",     "unidirectional_connection",     "main"),
    ("work_lossless_2way",                  "lossless_2way",                 "main"),
    ("work_water_pump",                    "water_pump",                    "main"),
    ("work_water_pump_delayed",            "water_pump_delayed",            "main"),
    ("work_wind",                          "wind",                          "main"),
    ("work_wind_battery",                  "wind_battery",                  "main"),
    ("work_wind_battery_invest",           "wind_battery_invest",           "main"),
    # Phase 3d additions
    ("work_commodity_ladder_annual",       "coal_ladder_annual",            "main"),
    ("work_commodity_ladder_cumulative",   "coal_ladder_cumulative",        "main"),
    ("work_inflation_check",
        "wind_battery_invest_lifetime_renew_inflation_2pct",                "main"),
    # Stochastic / branched fixtures
    ("work_2day_stochastic_dispatch_full_storage",
        "2_day_stochastic_dispatch",                                        "stochastic"),
    # LH2 fixture
    ("work_lh2_three_region",              "lh2_three_region",              "lh2"),
]
