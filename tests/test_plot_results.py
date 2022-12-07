import inspect
from functools import partial
from io import StringIO
from pathlib import Path
import sys
import unittest
import numpy as np
from spinedb_api import (
    DatabaseMapping,
    import_object_classes,
    import_objects,
    import_object_parameters,
    import_object_parameter_values,
    Map,
    import_alternatives,
    import_relationship_classes,
    import_relationships,
    import_relationship_parameters,
    import_relationship_parameter_values,
)
from spinetoolbox.plotting import TreeNode, XYData

sys.path.insert(0, str(Path(inspect.getfile(inspect.currentframe())).parent.parent))

import plot_results


class QueryParameterValuesTest(unittest.TestCase):
    def setUp(self):
        self._db_map = DatabaseMapping("sqlite:///", create=True)

    def tearDown(self):
        self._db_map.connection.close()

    def test_empty_database(self):
        """query_parameter_values() returns empty dict when there are no parameters in the database."""
        value_tree = plot_results.query_parameter_values(
            plot_results.EntityType.OBJECT, (), None, self._db_map
        )
        expected = TreeNode("class")
        self.assertEqual(value_tree, expected)

    def test_single_object_value(self):
        """query_parameter_values() creates valid IndexComponent for single object value."""
        alternative = "my_scenario__Fake_Data_Store@2022-09-06T15:00:00"
        import_alternatives(self._db_map, (alternative,))
        import_object_classes(self._db_map, ("my_class",))
        import_objects(self._db_map, (("my_class", "my_object"),))
        import_object_parameters(self._db_map, (("my_class", "my_parameter"),))
        import_object_parameter_values(
            self._db_map,
            (
                (
                    "my_class",
                    "my_object",
                    "my_parameter",
                    Map(["T1", "T2"], [2.3, 23.0], index_name="my index"),
                    alternative,
                ),
            ),
        )
        self._db_map.commit_session("Add test data.")
        value_tree = plot_results.query_parameter_values(
            plot_results.EntityType.OBJECT, (), None, self._db_map
        )
        value_component = TreeNode("my index")
        value_component.content.update({"T1": 2.3, "T2": 23.0})
        scenario_component = TreeNode("alternative")
        scenario_component.content[
            "my_scenario__Fake_Data_Store@2022-09-06T15:00:00"
        ] = value_component
        entity_component = TreeNode("object")
        entity_component.content["my_object"] = scenario_component
        parameter_component = TreeNode("parameter")
        parameter_component.content["my_parameter"] = entity_component
        class_component = TreeNode("class")
        class_component.content["my_class"] = parameter_component
        self.assertEqual(value_tree, class_component)

    def test_single_relationship_value(self):
        """query_parameter_values() creates valid IndexComponent for single relationship value."""
        alternative = "my_scenario__Fake_Data_Store@2022-09-06T15:00:00"
        import_alternatives(self._db_map, (alternative,))
        import_object_classes(self._db_map, ("my_class",))
        import_objects(self._db_map, (("my_class", "my_object"),))
        import_relationship_classes(
            self._db_map, (("my_relationship_class", ("my_class",)),)
        )
        import_relationships(self._db_map, (("my_relationship_class", ("my_object",)),))
        import_relationship_parameters(
            self._db_map, (("my_relationship_class", "my_parameter"),)
        )
        import_relationship_parameter_values(
            self._db_map,
            (
                (
                    "my_relationship_class",
                    ("my_object",),
                    "my_parameter",
                    Map(["T1", "T2"], [2.3, 23.0], index_name="my index"),
                    alternative,
                ),
            ),
        )
        self._db_map.commit_session("Add test data.")
        value_tree = plot_results.query_parameter_values(
            plot_results.EntityType.RELATIONSHIP, (), None, self._db_map
        )
        value_component = TreeNode("my index")
        value_component.content.update({"T1": 2.3, "T2": 23.0})
        scenario_component = TreeNode("alternative")
        scenario_component.content[
            "my_scenario__Fake_Data_Store@2022-09-06T15:00:00"
        ] = value_component
        entity_component = TreeNode("my_class")
        entity_component.content["my_object"] = scenario_component
        parameter_component = TreeNode("parameter")
        parameter_component.content["my_parameter"] = entity_component
        class_component = TreeNode("class")
        class_component.content["my_relationship_class"] = parameter_component
        self.assertEqual(value_tree, class_component)


class CategoryTicksTest(unittest.TestCase):
    def test_simple(self):
        categories = {("imaginary",): ["a", "b"], ("real",): ["a", "b"]}
        dividers, labels = plot_results.category_ticks(categories, 0.0, 3.0)
        convert = partial(self._x_to_axis_units, x_min=0.0, x_max=3.0)
        expected_positions = list(map(convert, (0.5, 2.5)))
        for label_position, expected in zip(labels.values(), expected_positions):
            self.assertAlmostEqual(label_position, expected)
        self.assertEqual(list(labels), ["imaginary", "real"])
        expected_dividers = list(map(convert, (-0.5, 1.5, 3.5)))
        for divider, expected in zip(dividers, expected_dividers):
            self.assertAlmostEqual(divider, expected)

    def test_uneven_dividers(self):
        categories = {
            ("imaginary",): ["a", "b"],
            ("real",): ["c"],
            ("eerie",): ["d", "e", "f"],
        }
        dividers, labels = plot_results.category_ticks(categories, 0.0, 5.0)
        convert = partial(self._x_to_axis_units, x_min=0.0, x_max=5.0)
        expected_dividers = list(map(convert, (-0.5, 1.5, 2.5, 5.5)))
        for divider, expected in zip(dividers, expected_dividers):
            self.assertAlmostEqual(divider, expected)
        expected_locations = list(map(convert, (0.5, 2.0, 4.0)))
        for label_position, expected in zip(labels.values(), expected_locations):
            self.assertAlmostEqual(label_position, expected)
        self.assertEqual(list(labels), ["imaginary", "real", "eerie"])

    def test_long_axis(self):
        categories = {("imaginary",): ["a", "b"], ("real",): ["a", "b"]}
        dividers, labels = plot_results.category_ticks(categories, -0.7, 3.7)
        convert = partial(self._x_to_axis_units, x_min=-0.7, x_max=3.7)
        expected_dividers = list(map(convert, (-0.5, 1.5, 3.5)))
        for divider, expected in zip(dividers, expected_dividers):
            self.assertAlmostEqual(divider, expected)
        self.assertEqual(list(labels), ["imaginary", "real"])
        expected_positions = list(map(convert, (0.5, 2.5)))
        for label_position, expected in zip(labels.values(), expected_positions):
            self.assertAlmostEqual(label_position, expected)

    @staticmethod
    def _x_to_axis_units(x, x_min, x_max):
        width = x_max - x_min
        step = 1.0 / width
        return -x_min * step + x * step


class TileHorizontallyTest(unittest.TestCase):
    def test_two_pieces_of_xy_data(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["idx1"], ["name 1"]),
            XYData(["a", "b"], [3.3, 4.4], "x", "y", ["idx2"], ["name 1"]),
        ]
        tiled, categories = plot_results.tile_horizontally(data_list, 1)
        expected = [
            XYData([0, 1], [1.1, 2.2], "x", "y", ["idx1"], ["name 1"]),
            XYData([2, 3], [3.3, 4.4], "x", "y", ["idx2"], ["name 1"]),
        ]
        self.assertEqual(tiled, expected)
        expected_categories = {("idx1",): ["a", "b"], ("idx2",): ["a", "b"]}
        self.assertEqual(categories, expected_categories)

    def test_same_last_data_index_gets_grouped_to_same_category(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["idx1"], ["name 1"]),
            XYData(["a", "b"], [3.3, 4.4], "x", "y", ["idx1"], ["name 1"]),
        ]
        tiled, categories = plot_results.tile_horizontally(data_list, 1)
        expected = [
            XYData([0, 1], [1.1, 2.2], "x", "y", ["idx1"], ["name 1"]),
            XYData([0, 1], [3.3, 4.4], "x", "y", ["idx1"], ["name 1"]),
        ]
        self.assertEqual(tiled, expected)
        expected_categories = {("idx1",): ["a", "b"]}
        self.assertEqual(categories, expected_categories)

    def test_incompatible_x_lengths_longer_first(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["idx1"], ["name 1"]),
            XYData(["a"], [3.3], "x", "y", ["idx1"], ["name 1"]),
        ]
        tiled, categories = plot_results.tile_horizontally(data_list, 1)
        expected = [
            XYData([0, 1], [1.1, 2.2], "x", "y", ["idx1"], ["name 1"]),
            XYData([0], [3.3], "x", "y", ["idx1"], ["name 1"]),
        ]
        self.assertEqual(tiled, expected)
        expected_categories = {("idx1",): ["a", "b"]}
        self.assertEqual(categories, expected_categories)

    def test_incompatible_x_lengths_shorter_first(self):
        data_list = [
            XYData(["a"], [1.1], "x", "y", ["idx1"], ["name 1"]),
            XYData(["a", "b"], [2.2, 3.3], "x", "y", ["idx1"], ["name 1"]),
        ]
        tiled, categories = plot_results.tile_horizontally(data_list, 1)
        expected = [
            XYData([0], [1.1], "x", "y", ["idx1"], ["name 1"]),
            XYData([0, 1], [2.2, 3.3], "x", "y", ["idx1"], ["name 1"]),
        ]
        self.assertEqual(tiled, expected)
        expected_categories = {("idx1",): ["a", "b"]}
        self.assertEqual(categories, expected_categories)

    def test_same_tiling_data_index(self):
        data_list = [
            XYData(["a"], [1.1], "x", "y", ["idx1"], ["name 1"]),
            XYData(["b"], [2.2], "x", "y", ["idx1"], ["name 1"]),
        ]
        tiled, categories = plot_results.tile_horizontally(data_list, 1)
        expected = [
            XYData([0], [1.1], "x", "y", ["idx1"], ["name 1"]),
            XYData([1], [2.2], "x", "y", ["idx1"], ["name 1"]),
        ]
        self.assertEqual(tiled, expected)
        expected_categories = {("idx1",): ["a", "b"]}
        self.assertEqual(categories, expected_categories)

    def test_extra_tiling_dimension(self):
        data_list = [
            XYData(
                ["a", "b"], [1.1, 2.2], "x", "y", ["idx", "A"], ["name 1", "name 2"]
            ),
            XYData(
                ["a", "b"], [3.3, 4.4], "x", "y", ["idx", "B"], ["name 1", "name 2"]
            ),
        ]
        tiled, categories = plot_results.tile_horizontally(data_list)
        expected = [
            XYData([0, 1], [1.1, 2.2], "x", "y", ["idx", "A"], ["name 1", "name 2"]),
            XYData([2, 3], [3.3, 4.4], "x", "y", ["idx", "B"], ["name 1", "name 2"]),
        ]
        self.assertEqual(tiled, expected)
        expected_categories = {("idx", "A"): ["a", "b"], ("idx", "B"): ["a", "b"]}
        self.assertEqual(categories, expected_categories)


class CategorizeFurtherTest(unittest.TestCase):
    def test_single_subcategory(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["A", "B"], ["index 1", "index 2"])
        ]
        categories = plot_results.categorize_further(
            {("A", "B"): ["a", "b"]}, data_list
        )
        expected = {("A",): ["a", "b"]}
        self.assertEqual(categories, expected)

    def test_two_subcategories_within_single_category(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["A", "B1"], ["name 1", "name 2"]),
            XYData(["a", "b"], [3.3, 4.4], "x", "y", ["A", "B2"], ["name 1", "name 2"]),
        ]
        categories = plot_results.categorize_further(
            {("A", "B1"): ["a", "b"], ("A", "B2"): ["a", "b"]}, data_list
        )
        expected = {("A",): ["a", "b", "a", "b"]}
        self.assertEqual(categories, expected)


class ShuffleDimensionsTest(unittest.TestCase):
    def test_shuffle_data_indices(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["A", "B"], ["index 1", "index 2"])
        ]
        inserted_list = plot_results.shuffle_dimensions({"index 1": 1}, data_list)
        expected = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["B", "A"], ["index 2", "index 1"])
        ]
        self.assertEqual(inserted_list, expected)

    def test_negative_target_moves_to_end(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["A", "B"], ["index 1", "index 2"])
        ]
        inserted_list = plot_results.shuffle_dimensions({"index 1": -1}, data_list)
        expected = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["B", "A"], ["index 2", "index 1"])
        ]
        self.assertEqual(inserted_list, expected)

    def test_x_target_moves_to_x_axis(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["A", "B"], ["index 1", "index 2"])
        ]
        inserted_list = plot_results.shuffle_dimensions({"index 1": "x"}, data_list)
        expected = [
            XYData(["A"], [1.1], "index 1", "", ["B", "a"], ["index 2", "x"]),
            XYData(["A"], [2.2], "index 1", "", ["B", "b"], ["index 2", "x"]),
        ]
        self.assertEqual(inserted_list, expected)

    def test_move_twice_works_as_expected(self):
        data_list = [
            XYData(["a", "b"], [1.1, 2.2], "x", "y", ["A", "B"], ["index 1", "index 2"])
        ]
        inserted_list = plot_results.shuffle_dimensions(
            {"index 2": "x", "index 1": -1}, data_list
        )
        expected = [
            XYData(["B"], [1.1], "index 2", "", ["a", "A"], ["x", "index 1"]),
            XYData(["B"], [2.2], "index 2", "", ["b", "A"], ["x", "index 1"]),
        ]
        self.assertEqual(inserted_list, expected)


class InsertAsXTest(unittest.TestCase):
    def test_switch_two_indexes(self):
        data_list = [XYData(["a", "b"], [1.1, 2.2], "x", "y", ["idx"], ["my_index"])]
        inserted_list = plot_results.insert_as_x("my_index", data_list)
        expected = [
            XYData(["idx"], [1.1], "my_index", "", ["a"], ["x"]),
            XYData(["idx"], [2.2], "my_index", "", ["b"], ["x"]),
        ]
        self.assertEqual(inserted_list, expected)

    def test_y_values_from_different_xy_data_get_merged_logically(self):
        data_list = [
            XYData(
                ["a", "b"],
                [1.1, 2.2],
                "x",
                "y",
                ["idx1", "cat 1"],
                ["my index", "animate"],
            ),
            XYData(
                ["a", "b"],
                [3.3, 4.4],
                "x",
                "y",
                ["idx2", "cat 1"],
                ["my index", "animate"],
            ),
        ]
        inserted_list = plot_results.insert_as_x("my index", data_list)
        expected = [
            XYData(
                ["idx1", "idx2"],
                [1.1, 3.3],
                "my index",
                "",
                ["cat 1", "a"],
                ["animate", "x"],
            ),
            XYData(
                ["idx1", "idx2"],
                [2.2, 4.4],
                "my index",
                "",
                ["cat 1", "b"],
                ["animate", "x"],
            ),
        ]
        self.assertEqual(inserted_list, expected)


class RelabelXAxisTest(unittest.TestCase):
    def test_tick_gap_smaller_than_unity(self):
        categories = {
            "y2020_5week": ["p2020"],
            "y2025_5week": ["p2025"],
            "y2030_5week": ["p2030"],
            "y2035_5week": ["p2035"],
        }
        x_ticks = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        tick_positions, labels = plot_results.relabel_x_axis(categories, x_ticks)
        self.assertEqual(list(tick_positions), [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(labels, ["p2020", "p2025", "p2030", "p2035"])

    def test_tick_gap_greater_than_unity(self):
        categories = {
            "y2020_5week": [str(i) for i in range(10)],
            "y2025_5week": [str(i) for i in range(10, 20)],
            "y2030_5week": [str(i) for i in range(20, 40)],
            "y2035_5week": [str(i) for i in range(30, 40)],
        }
        x_ticks = np.array(
            [-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        )
        tick_positions, labels = plot_results.relabel_x_axis(categories, x_ticks)
        self.assertEqual(
            list(tick_positions), [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
        )
        self.assertEqual(labels, [str(i) for i in range(0, 40, 5)])

    def test_single_tick(self):
        categories = {"y2020_2day_dispatch": ["p2020"]}
        x_ticks = np.array([-0.1, 0.0, 0.1])
        tick_positions, labels = plot_results.relabel_x_axis(categories, x_ticks)
        self.assertEqual(list(tick_positions), [0.0])
        self.assertEqual(labels, ["p2020"])


class CheckEntityClassesTest(unittest.TestCase):
    def test_nothing_gets_printed_if_everything_is_ok(self):
        settings = {"plots": [{"selection": {"entity_class": ["my_class"]}}]}
        entity_class_types = {"my_class": plot_results.EntityType.OBJECT}
        my_out = StringIO()
        plot_results.check_entity_classes(settings, entity_class_types, my_out)
        self.assertEqual(my_out.getvalue(), "")

    def test_warns_about_missing_class(self):
        settings = {
            "plots": [{"selection": {"entity_class": ["my_non_existent_class"]}}]
        }
        entity_class_types = {"my_class": plot_results.EntityType.OBJECT}
        my_out = StringIO()
        plot_results.check_entity_classes(settings, entity_class_types, my_out)
        self.assertEqual(
            my_out.getvalue(),
            "entity class 'my_non_existent_class' not in database; ignoring\n",
        )


if __name__ == "__main__":
    unittest.main()
