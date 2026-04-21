"""Tests for the compound scenario-identity helpers and their consumers.

Covers the behaviour required to keep same-named scenarios from different
input sources from colliding on disk while keeping the common case
prefix-free:

* round-trip of suffix / compound-key encodings (including legacy names)
* bare-name ownership: first source to run a name "owns" the bare folder;
  later runs from other sources get ``<name>_<src#>`` suffix
* :class:`ExecutedScenarioManager` resolves bare folders back to their
  owner source via the ownership map
* :class:`ExecutionManager` keeps jobs for same-named scenarios from
  different sources distinct (pruning, work folder, input DB URL,
  output subdir)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from flextool.gui.data_models import ProjectSettings, ScenarioInfo
from flextool.gui.execution_manager import (
    ExecutionManager,
    JobStatus,
    JobType,
)
from flextool.gui.scenario_key import (
    choose_output_subdir_for_write,
    format_key,
    format_subdir,
    parse_key,
    parse_subdir,
    release_bare_owner,
    resolve_source_number,
    resolve_subdir_for_read,
)
from flextool.gui.scenario_lists import ExecutedScenarioManager


# ---------------------------------------------------------------------------
# scenario_key helpers — encoding round-trips
# ---------------------------------------------------------------------------

class TestScenarioKeyHelpers:
    def test_subdir_suffix_round_trip(self):
        assert format_subdir(1, "base") == "base_1"
        assert parse_subdir("base_1") == (1, "base")

    def test_subdir_preserves_underscores_in_name(self):
        # rpartition on "_" — only the trailing numeric chunk is the src#
        subdir = format_subdir(2, "foo_bar_baz")
        assert subdir == "foo_bar_baz_2"
        assert parse_subdir(subdir) == (2, "foo_bar_baz")

    def test_subdir_bare_name_is_legacy(self):
        # No trailing digit → caller must consult the ownership map.
        assert parse_subdir("legacy_name") == (0, "legacy_name")
        assert parse_subdir("foo") == (0, "foo")

    def test_key_round_trip(self):
        assert format_key(3, "scenario") == "3|scenario"
        assert parse_key("3|scenario") == (3, "scenario")

    def test_key_preserves_pipes_in_name(self):
        key = "4|weird|name|with|pipes"
        assert parse_key(key) == (4, "weird|name|with|pipes")


# ---------------------------------------------------------------------------
# Ownership-aware resolution
# ---------------------------------------------------------------------------

class TestOwnershipResolution:
    def test_bare_folder_without_owner_is_legacy(self):
        # No suffix, no ownership record → legacy source 0.
        assert resolve_source_number("base", {}) == (0, "base")

    def test_bare_folder_with_owner(self):
        owners = {"base": 1}
        assert resolve_source_number("base", owners) == (1, "base")

    def test_suffixed_folder_parses_without_owners_map(self):
        assert resolve_source_number("base_2", {}) == (2, "base")

    def test_name_ending_in_digits_is_rescued_by_ownership_map(self):
        # A scenario legitimately named "foo_1" would otherwise be parsed
        # as (src=1, name="foo"). An explicit ownership record fixes it.
        owners = {"foo_1": 3}
        assert resolve_source_number("foo_1", owners) == (3, "foo_1")

    def test_resolve_subdir_for_read_bare_when_owner_matches(self):
        owners = {"base": 1}
        assert resolve_subdir_for_read(owners, 1, "base") == "base"

    def test_resolve_subdir_for_read_suffixed_when_owner_differs(self):
        owners = {"base": 1}
        assert resolve_subdir_for_read(owners, 2, "base") == "base_2"

    def test_resolve_subdir_for_read_suffixed_when_no_owner_yet(self):
        # Read-only path can't claim; fall back to suffix so nothing
        # overwrites an unclaimed bare folder by accident.
        assert resolve_subdir_for_read({}, 2, "base") == "base_2"

    def test_choose_for_write_claims_bare_when_unclaimed(self, tmp_path):
        owners: dict[str, int] = {}
        result = choose_output_subdir_for_write(tmp_path, owners, 1, "base")
        assert result == "base"
        assert owners == {"base": 1}

    def test_choose_for_write_is_bare_for_owner(self, tmp_path):
        owners = {"base": 1}
        result = choose_output_subdir_for_write(tmp_path, owners, 1, "base")
        assert result == "base"
        assert owners == {"base": 1}

    def test_choose_for_write_suffixes_non_owner(self, tmp_path):
        owners = {"base": 1}
        result = choose_output_subdir_for_write(tmp_path, owners, 2, "base")
        assert result == "base_2"
        assert owners == {"base": 1}  # non-owner doesn't claim anything

    def test_choose_for_write_reuses_existing_suffixed_folder(self, tmp_path):
        # If a prior run produced a suffixed folder, reuse it instead of
        # orphaning those results with a fresh bare claim.
        (tmp_path / "output_parquet" / "base_1").mkdir(parents=True)
        owners: dict[str, int] = {}
        result = choose_output_subdir_for_write(tmp_path, owners, 1, "base")
        assert result == "base_1"
        assert owners == {}  # no claim — we reused the suffixed form

    def test_release_bare_owner_only_releases_when_source_matches(self):
        owners = {"base": 1}
        assert release_bare_owner(owners, 2, "base") is False
        assert owners == {"base": 1}
        assert release_bare_owner(owners, 1, "base") is True
        assert owners == {}


# ---------------------------------------------------------------------------
# ExecutedScenarioManager — ownership-aware scan + delete
# ---------------------------------------------------------------------------

def _touch_parquet(dir_path: Path, filename: str = "unit_flow.parquet") -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / filename).write_bytes(b"")


class TestExecutedScenarioManager:
    def test_scan_bare_folder_uses_ownership_map(self, tmp_path):
        parquet = tmp_path / "output_parquet"
        _touch_parquet(parquet / "base")      # bare, owned by src 1
        _touch_parquet(parquet / "base_2")    # suffixed, src 2
        _touch_parquet(parquet / "peak")      # bare, owned by src 1

        settings = ProjectSettings()
        settings.bare_output_owners = {"base": 1, "peak": 1}

        mgr = ExecutedScenarioManager(tmp_path, settings)
        executed = mgr.scan_executed()
        pairs = {(e.source_number, e.name) for e in executed}
        assert pairs == {(1, "base"), (2, "base"), (1, "peak")}

    def test_scan_bare_without_owner_is_legacy_zero(self, tmp_path):
        _touch_parquet(tmp_path / "output_parquet" / "old_run")
        mgr = ExecutedScenarioManager(tmp_path, ProjectSettings())
        executed = mgr.scan_executed()
        assert len(executed) == 1
        assert (executed[0].source_number, executed[0].name) == (0, "old_run")

    def test_check_outputs_uses_bare_folder_for_owner(self, tmp_path):
        (tmp_path / "output_plots" / "base").mkdir(parents=True)
        (tmp_path / "output_plots" / "base" / "a.png").write_bytes(b"")
        settings = ProjectSettings()
        settings.bare_output_owners = {"base": 1}
        mgr = ExecutedScenarioManager(tmp_path, settings)
        outputs = mgr.check_outputs([(1, "base")])
        assert outputs["1|base"]["has_plots"] is True

    def test_check_outputs_uses_suffix_for_non_owner(self, tmp_path):
        (tmp_path / "output_plots" / "base_2").mkdir(parents=True)
        (tmp_path / "output_plots" / "base_2" / "a.png").write_bytes(b"")
        settings = ProjectSettings()
        settings.bare_output_owners = {"base": 1}  # src 2 is not the owner
        mgr = ExecutedScenarioManager(tmp_path, settings)
        outputs = mgr.check_outputs([(2, "base")])
        assert outputs["2|base"]["has_plots"] is True

    def test_delete_releases_ownership_when_bare_deleted(self, tmp_path):
        _touch_parquet(tmp_path / "output_parquet" / "base")
        settings = ProjectSettings()
        settings.bare_output_owners = {"base": 1}
        mgr = ExecutedScenarioManager(tmp_path, settings)
        mgr.delete_results([(1, "base")])
        assert "base" not in settings.bare_output_owners

    def test_delete_keeps_ownership_when_suffix_deleted(self, tmp_path):
        _touch_parquet(tmp_path / "output_parquet" / "base")
        _touch_parquet(tmp_path / "output_parquet" / "base_2")
        settings = ProjectSettings()
        settings.bare_output_owners = {"base": 1}
        mgr = ExecutedScenarioManager(tmp_path, settings)
        mgr.delete_results([(2, "base")])
        # Src 2 only held the suffixed folder — src 1 keeps the bare.
        assert settings.bare_output_owners == {"base": 1}
        assert (tmp_path / "output_parquet" / "base").is_dir()
        assert not (tmp_path / "output_parquet" / "base_2").exists()


# ---------------------------------------------------------------------------
# ExecutionManager — compound identity through the pipeline
# ---------------------------------------------------------------------------

def _make_mgr(project_path: Path) -> ExecutionManager:
    return ExecutionManager(project_path, ProjectSettings())


def _make_scenario(name: str, source_number: int, source_name: str) -> ScenarioInfo:
    return ScenarioInfo(name=name, source_number=source_number, source_name=source_name)


class TestExecutionManagerCompoundIdentity:
    def test_first_source_gets_bare_output_subdir(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        job = mgr.add_jobs([_make_scenario("base", 1, "a.sqlite")])[0]
        assert job.output_subdir == "base"
        assert mgr.settings.bare_output_owners == {"base": 1}

    def test_second_source_with_same_name_gets_suffix(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        first = mgr.add_jobs([_make_scenario("base", 1, "a.sqlite")])[0]
        second = mgr.add_jobs([_make_scenario("base", 2, "b.sqlite")])[0]
        assert first.output_subdir == "base"
        assert second.output_subdir == "base_2"
        # Ownership sticks with src 1.
        assert mgr.settings.bare_output_owners == {"base": 1}

    def test_add_jobs_distinct_input_urls_for_same_name(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        jobs = mgr.add_jobs(
            [
                _make_scenario("base", 1, "a.sqlite"),
                _make_scenario("base", 2, "b.sqlite"),
            ]
        )
        assert jobs[0].input_db_url != jobs[1].input_db_url

    def test_prune_does_not_cross_source(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        old_src1 = mgr.add_jobs([_make_scenario("base", 1, "a.sqlite")])[0]
        old_src2 = mgr.add_jobs([_make_scenario("base", 2, "b.sqlite")])[0]
        old_src1.status = JobStatus.FAILED
        old_src2.status = JobStatus.FAILED

        new_src1 = mgr.add_jobs([_make_scenario("base", 1, "a.sqlite")])[0]
        new_src1.status = JobStatus.SUCCESS
        with mgr._lock:
            mgr._prune_old_jobs(new_src1)

        remaining = [(j.source_number, j.scenario_name, j.status) for j in mgr._jobs]
        assert (1, "base", JobStatus.FAILED) not in remaining
        assert (2, "base", JobStatus.FAILED) in remaining
        assert (1, "base", JobStatus.SUCCESS) in remaining

    def test_build_run_command_uses_job_output_subdir(self, tmp_path):
        mgr = _make_mgr(tmp_path)
        # Seed an ownership so src 3 must suffix.
        mgr.settings.bare_output_owners["base"] = 1
        job = mgr.add_jobs([_make_scenario("base", 3, "src.sqlite")])[0]
        assert job.output_subdir == "base_3"
        cmd = mgr._build_run_command(job, tmp_path / "work" / "base_3")
        assert "--output-subdir" in cmd
        assert cmd[cmd.index("--output-subdir") + 1] == "base_3"
