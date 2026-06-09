"""Project-root rooting for the scenario-comparison step.

The comparison run derives the project root from the participating scenarios'
``output_location`` values and uses it to (a) resolve the per-project
``plot_settings.yaml`` (``color_path``) and (b) re-root the *default*
``--plot-dir`` at ``<project_root>/output_plot_comparisons`` so Toolbox/GUI
comparison plots honour ``projects/<Name>/plot_settings.yaml``.  An explicit
``--plot-dir`` always wins; divergent/empty locations fall back to the old
CWD-relative behaviour and the bundled default template.
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from flextool.scenario_comparison.orchestrator import derive_project_root


# ---------------------------------------------------------------------------
# Pure predicate: derive_project_root
# ---------------------------------------------------------------------------

class TestDeriveProjectRoot(unittest.TestCase):
    def test_all_equal(self) -> None:
        """All scenarios share one folder → that folder is the root."""
        folders = {"a": "/proj/p1", "b": "/proj/p1"}
        self.assertEqual(derive_project_root(folders), Path("/proj/p1"))

    def test_single_scenario(self) -> None:
        self.assertEqual(derive_project_root({"a": "/proj/p1"}), Path("/proj/p1"))

    def test_distinct_subdirs_rejected(self) -> None:
        """Distinct per-scenario subdirs are NOT a shared root → None.

        FlexTool's scenario_folders map every scenario of one project to the
        same value, so distinct folders mean distinct projects."""
        folders = {"a": "/proj/p1/sA", "b": "/proj/p1/sB"}
        self.assertIsNone(derive_project_root(folders))

    def test_divergent_trees_rejected(self) -> None:
        """Different projects → no shared folder → None."""
        folders = {"a": "/proj/p1/out", "b": "/proj/p2/out"}
        self.assertIsNone(derive_project_root(folders))

    def test_sibling_projects_rejected(self) -> None:
        """Sibling project folders must not collapse to their parent."""
        folders = {"a": "/home/u/projA", "b": "/home/u/projB"}
        self.assertIsNone(derive_project_root(folders))

    def test_empty_mapping(self) -> None:
        self.assertIsNone(derive_project_root({}))
        self.assertIsNone(derive_project_root(None))

    def test_empty_values(self) -> None:
        self.assertIsNone(derive_project_root({"a": "", "b": ""}))

    def test_normalizes_trailing_and_dots(self) -> None:
        folders = {"a": "/proj/p1/", "b": "/proj/p1/./"}
        self.assertEqual(derive_project_root(folders), Path("/proj/p1"))


# ---------------------------------------------------------------------------
# run(): color_path + default plot-dir rooting
# ---------------------------------------------------------------------------

class TestRunProjectRooting(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        # A project folder with its own plot_settings.yaml.
        self.project = self.tmp / "MyProject"
        self.project.mkdir()
        self.project_settings = self.project / "plot_settings.yaml"
        self.project_settings.write_text("scenarios:\n  scen_a: '#ff0000'\n")
        # Minimal plots config consumed by run().
        self.config_path = self.tmp / "config.yaml"
        self.config_path.write_text("plots:\n  test:\n    plot_name: test\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_results(self):
        from flextool.scenario_comparison.data_models import TimeSeriesResults

        res = TimeSeriesResults()
        res.unit_flow__dt = pd.DataFrame({"a": [1, 2]})
        return res

    def _run(self, scenario_folders, *, plot_dir, plot_dir_is_default):
        captured: dict = {}

        def _capture_plot(*args, **kwargs):
            captured["color_path"] = kwargs.get("color_path")

        with (
            patch(
                "flextool.scenario_comparison.orchestrator.get_scenario_results",
                return_value=(scenario_folders, self._fake_results()),
            ),
            patch(
                "flextool.scenario_comparison.orchestrator.combine_dispatch_mappings",
            ),
            patch(
                "flextool.scenario_comparison.orchestrator.plot_dict_of_dataframes",
                side_effect=_capture_plot,
            ),
        ):
            from flextool.scenario_comparison.orchestrator import run

            run(
                db_url=None,
                parquet_subdir="",
                plot_dir=plot_dir,
                output_config_path=str(self.config_path),
                active_configs=["default"],
                plot_rows=[0, 167],
                write_to_xlsx=False,
                write_dispatch_xlsx=False,
                write_to_ods=False,
                show_plots=False,
                dispatch_plots=False,
                scenario_folders=scenario_folders,
                plot_dir_is_default=plot_dir_is_default,
            )
        return captured

    def test_shared_project_root_uses_project_settings_and_default_dir(self) -> None:
        """All scenarios share <project> → color_path is the project's own
        plot_settings.yaml and the default plot dir is rooted in the project."""
        folders = {"scen_a": str(self.project), "scen_b": str(self.project)}
        captured = self._run(
            folders,
            plot_dir="output_plot_comparisons",
            plot_dir_is_default=True,
        )
        # color_path chosen over the bundled default.
        self.assertEqual(Path(captured["color_path"]), self.project_settings)
        # Default plot dir re-rooted into the project.
        expected_dir = self.project / "output_plot_comparisons"
        self.assertTrue(
            expected_dir.is_dir(),
            f"expected {expected_dir} to be created; "
            f"contents: {list(self.project.iterdir())}",
        )

    def test_explicit_plot_dir_wins(self) -> None:
        """An explicit --plot-dir is used verbatim even with a shared root,
        but color_path still resolves against the project root."""
        explicit = self.tmp / "explicit_out"
        folders = {"scen_a": str(self.project), "scen_b": str(self.project)}
        captured = self._run(
            folders,
            plot_dir=str(explicit),
            plot_dir_is_default=False,
        )
        # Explicit dir created; project default dir NOT created.
        self.assertTrue(explicit.is_dir())
        self.assertFalse((self.project / "output_plot_comparisons").exists())
        # color_path still resolves to the project file (4-tier rooting).
        self.assertEqual(Path(captured["color_path"]), self.project_settings)

    def test_divergent_locations_fall_back_to_bundled_default(self) -> None:
        """Scenarios in genuinely different projects → no sensible common
        root → CWD-relative plot dir + bundled default template, no crash."""
        from flextool.plot_outputs.color_template import _default_path

        folders = {"scen_a": "/proj/pA/out", "scen_b": "/proj/pB/out"}
        old_cwd = Path.cwd()
        import os

        os.chdir(self.tmp)
        try:
            captured = self._run(
                folders,
                plot_dir="output_plot_comparisons",
                plot_dir_is_default=True,
            )
        finally:
            os.chdir(old_cwd)
        # Bundled default chosen (no per-project file found).
        self.assertEqual(Path(captured["color_path"]), Path(_default_path()))
        # CWD-relative dir created; nothing rooted into a phantom project.
        self.assertTrue((self.tmp / "output_plot_comparisons").is_dir())


if __name__ == "__main__":
    unittest.main()
