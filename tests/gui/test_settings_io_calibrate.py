"""Tests for Calibrate-investments dialog settings persistence.

Covers the ``calib_*`` fields added to :class:`ProjectSettings`:

* round-trip through ``save_project_settings`` / ``load_project_settings``
  preserves every non-default value;
* a settings.yaml lacking the keys yields the declared defaults;
* a hand-edited / malformed settings.yaml loads without raising and
  falls back to defaults / drops bad list elements.

``save_project_settings`` serialises via ``asdict`` so the save side is
automatic; only the tolerant manual load needs pinning.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from flextool.gui.data_models import ProjectSettings
from flextool.gui.settings_io import (
    SETTINGS_FILENAME,
    load_project_settings,
    save_project_settings,
)

_CALIB_FIELDS = (
    "calib_rp_n_rp",
    "calib_rp_period_length",
    "calib_rp_force_sustained",
    "calib_rp_force_peak",
    "calib_rp_force_window",
    "calib_rp_count_mode",
    "calib_rp_add_to_scenario",
    "calib_selected_solves",
    "calib_max_iterations",
    "calib_sizing",
    "calib_overshoot_pct",
    "calib_damping_first",
    "calib_damping_remaining",
    "calib_stall_fraction",
    "calib_keep_artifacts",
)


def test_calib_settings_round_trip(tmp_path: Path):
    """Every calib_* field survives a save/load round-trip."""
    settings = ProjectSettings(
        calib_rp_n_rp=42,
        calib_rp_period_length=200,
        calib_rp_force_sustained=False,
        calib_rp_force_peak=True,
        calib_rp_force_window=48,
        calib_rp_count_mode="fixed",
        calib_rp_add_to_scenario=False,
        calib_selected_solves=["solve_a", "solve_b"],
        calib_max_iterations=15,
        calib_sizing="uniform",
        calib_overshoot_pct=7.5,
        calib_damping_first=0.8,
        calib_damping_remaining=0.25,
        calib_stall_fraction=0.02,
        calib_keep_artifacts=True,
    )
    save_project_settings(tmp_path, settings)
    loaded = load_project_settings(tmp_path)

    for name in _CALIB_FIELDS:
        assert getattr(loaded, name) == getattr(settings, name), name


def test_calib_settings_defaults_when_absent(tmp_path: Path):
    """A settings.yaml lacking the calib_* keys yields the defaults."""
    (tmp_path / SETTINGS_FILENAME).write_text(
        yaml.safe_dump({"scaling": "basic"}), encoding="utf-8"
    )
    loaded = load_project_settings(tmp_path)
    defaults = ProjectSettings()

    for name in _CALIB_FIELDS:
        assert getattr(loaded, name) == getattr(defaults, name), name


def test_calib_settings_malformed_tolerated(tmp_path: Path):
    """Malformed calib_* values load without raising, falling back to
    defaults and dropping bad list elements."""
    (tmp_path / SETTINGS_FILENAME).write_text(
        yaml.safe_dump(
            {
                "calib_sizing": "bogus",
                "calib_rp_count_mode": "nonsense",
                "calib_max_iterations": True,  # bool, not a plain int
                "calib_overshoot_pct": True,   # bool, not a number
                "calib_keep_artifacts": "yes",  # not a bool
                "calib_rp_add_to_scenario": "sure",  # not a bool
                "calib_selected_solves": ["ok", 5, ""],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_project_settings(tmp_path)
    defaults = ProjectSettings()

    assert loaded.calib_sizing == defaults.calib_sizing
    assert loaded.calib_rp_count_mode == defaults.calib_rp_count_mode
    assert loaded.calib_max_iterations == defaults.calib_max_iterations
    assert loaded.calib_overshoot_pct == defaults.calib_overshoot_pct
    assert loaded.calib_keep_artifacts == defaults.calib_keep_artifacts
    assert loaded.calib_rp_add_to_scenario == defaults.calib_rp_add_to_scenario
    # Non-string / empty elements dropped; the valid one is kept.
    assert loaded.calib_selected_solves == ["ok"]
