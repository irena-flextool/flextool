from datetime import datetime
import unittest
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
from .. import plot_results


class GatherPlottingDataTest(unittest.TestCase):
    def test_simple_data(self):
        """gather_plotting_data() yields correct data with single plottable dataset."""
        leaf_component = plot_results.IndexComponent("index name 2")
        leaf_component.content.update({"T1": 2.3, "T2": 23.0})
        values = plot_results.IndexComponent("index name 1")
        values.content["index 1,1"] = leaf_component
        result = [xy_data for xy_data in plot_results.gather_plotting_data(values)]
        expected = plot_results.XYData(
            ["T1", "T2"],
            [2.3, 23.0],
            "index name 2",
            "",
            ["index 1,1"],
            ["index name 1"],
        )
        self.assertEqual(result, [expected])

    def test_two_leaves(self):
        """gather_plotting_data() can handle multiple leaf components."""
        leaf_component1 = plot_results.IndexComponent("x label 1")
        leaf_component1.content.update({"T1": 2.3, "T2": 23.0})
        leaf_component2 = plot_results.IndexComponent("x label 2")
        leaf_component2.content.update({"T1": -2.3, "T2": -23.0})
        intermediate_component1 = plot_results.IndexComponent("index name 2,1")
        intermediate_component1.content["value 1"] = leaf_component1
        intermediate_component1.content["value 2"] = leaf_component2
        values = plot_results.IndexComponent("index name 1,1")
        values.content["index 1,1"] = intermediate_component1
        result = [xy_data for xy_data in plot_results.gather_plotting_data(values)]
        expected = [
            plot_results.XYData(
                ["T1", "T2"],
                [2.3, 23.0],
                "x label 1",
                "",
                ["index 1,1", "value 1"],
                ["index name 1,1", "index name 2,1"],
            ),
            plot_results.XYData(
                ["T1", "T2"],
                [-2.3, -23.0],
                "x label 2",
                "",
                ["index 1,1", "value 2"],
                ["index name 1,1", "index name 2,1"],
            ),
        ]
        self.assertEqual(result, expected)

    def test_two_roots(self):
        """gather_plotting_data() can handle multiple root components."""
        leaf_component1 = plot_results.IndexComponent("x label 1")
        leaf_component1.content.update({"T1": 2.3, "T2": 23.0})
        leaf_component2 = plot_results.IndexComponent("x label 2")
        leaf_component2.content.update({"T1": -2.3, "T2": -23.0})
        intermediate_component1 = plot_results.IndexComponent("index name 2,1")
        intermediate_component1.content["value 1"] = leaf_component1
        intermediate_component2 = plot_results.IndexComponent("index name 2,2")
        intermediate_component2.content["value 2"] = leaf_component2
        values = plot_results.IndexComponent("index name 1,1")
        values.content["index 1,1"] = intermediate_component1
        values.content["index 1,2"] = intermediate_component2
        result = [xy_data for xy_data in plot_results.gather_plotting_data(values)]
        expected = [
            plot_results.XYData(
                ["T1", "T2"],
                [2.3, 23.0],
                "x label 1",
                "",
                ["index 1,1", "value 1"],
                ["index name 1,1", "index name 2,1"],
            ),
            plot_results.XYData(
                ["T1", "T2"],
                [-2.3, -23.0],
                "x label 2",
                "",
                ["index 1,2", "value 2"],
                ["index name 1,1", "index name 2,2"],
            ),
        ]
        self.assertEqual(result, expected)


class QueryParameterValuesTest(unittest.TestCase):
    def setUp(self):
        self._db_map = DatabaseMapping("sqlite:///", create=True)

    def tearDown(self):
        self._db_map.connection.close()

    def test_empty_database(self):
        """query_parameter_values() returns empty dict when there are no parameters in the database."""
        value_tree = plot_results.query_parameter_values(
            plot_results.EntityType.OBJECT, self._db_map
        )
        expected = plot_results.IndexComponent("class")
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
            plot_results.EntityType.OBJECT, self._db_map
        )
        value_component = plot_results.IndexComponent("my index")
        value_component.content.update({"T1": 2.3, "T2": 23.0})
        scenario_component = plot_results.IndexComponent("scenario")
        scenario_component.content["my_scenario"] = value_component
        entity_component = plot_results.IndexComponent("object")
        entity_component.content["my_object"] = scenario_component
        parameter_component = plot_results.IndexComponent("parameter")
        parameter_component.content["my_parameter"] = entity_component
        class_component = plot_results.IndexComponent("class")
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
            plot_results.EntityType.RELATIONSHIP, self._db_map
        )
        value_component = plot_results.IndexComponent("my index")
        value_component.content.update({"T1": 2.3, "T2": 23.0})
        scenario_component = plot_results.IndexComponent("scenario")
        scenario_component.content["my_scenario"] = value_component
        entity_component = plot_results.IndexComponent("my_class")
        entity_component.content["my_object"] = scenario_component
        parameter_component = plot_results.IndexComponent("parameter")
        parameter_component.content["my_parameter"] = entity_component
        class_component = plot_results.IndexComponent("class")
        class_component.content["my_relationship_class"] = parameter_component
        self.assertEqual(value_tree, class_component)


if __name__ == "__main__":
    unittest.main()
