from argparse import ArgumentParser
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum, unique
from operator import attrgetter
from typing import Dict, Generator, List, Mapping, MutableSequence, Optional, Sequence, Tuple
import matplotlib
import matplotlib.pyplot as plt
from PySide2 import QtCore  # Makes matplotlib pick the correct Qt5Agg backend.
from matplotlib.ticker import MaxNLocator
from spinedb_api import convert_containers_to_maps, DatabaseMapping, from_database, Map
from spinedb_api.db_mapping_base import DatabaseMappingBase
from spinedb_api.filters.scenario_filter import SCENARIO_FILTER_TYPE
from spinedb_api.filters.tools import name_from_dict, pop_filter_configs

matplotlib.use("Qt5Agg")

Plottable = Tuple[Sequence[str], Sequence[float]]


@unique
class EntityType(IntEnum):
    OBJECT = 1
    RELATIONSHIP = 2


@dataclass(frozen=True)
class ResultScenario:
    alternative_id: int
    time_stamp: datetime


@dataclass(frozen=True)
class XYData:
    x: Sequence[str]
    y: Sequence[float]
    x_label: str
    y_label: str
    data_index: Sequence[str]
    index_names: Sequence[str]


@dataclass
class IndexComponent:
    label: str
    content: Dict = field(default_factory=dict)


def make_argument_parser() -> ArgumentParser:
    """Creates a command line argument parser."""
    parser = ArgumentParser(description="Plot FlexTool results.")
    parser.add_argument(
        "url", metavar="URL", help="URL pointing to the result database"
    )
    return parser


def convert_map_to_index_components(map_: Map) -> IndexComponent:
    """Converts Maps to nested dictionaries."""
    d = IndexComponent(map_.index_name)
    for index, x in zip(map_.indexes, map_.values):
        if isinstance(x, Map):
            x = convert_map_to_index_components(x)
        d.content[index] = x
    return d


def get_model_scenario(value_row) -> str:
    """Parses model scenario name from the alternative name in parameter value row."""
    scenario, separator, remainder = value_row.alternative_name.partition("__")
    return scenario


def query_parameter_values(
    entity_type: EntityType,
    db_map: DatabaseMappingBase
) -> IndexComponent:
    """Reads parameter values from database."""
    value_tree = IndexComponent("class")
    class_name_fields = {
        EntityType.OBJECT: "object_class_name",
        EntityType.RELATIONSHIP: "relationship_class_name",
    }
    object_lists = {
        EntityType.OBJECT: lambda r: [r.object_name],
        EntityType.RELATIONSHIP: lambda r: r.object_name_list.split(","),
    }
    object_labels = {
        EntityType.OBJECT: lambda r: ["object"],
        EntityType.RELATIONSHIP: lambda r: r.object_class_name_list.split(","),
    }
    get_class_name = attrgetter(class_name_fields[entity_type])
    get_object_names = object_lists[entity_type]
    get_object_labels = object_labels[entity_type]
    subquery = {
        EntityType.OBJECT: db_map.object_parameter_value_sq,
        EntityType.RELATIONSHIP: db_map.relationship_parameter_value_sq,
    }[entity_type]
    for row in db_map.query(subquery):
        class_name = get_class_name(row)
        objects = get_object_names(row)
        model_scenario = get_model_scenario(row)
        parameter_value = from_database(row.value, row.type)
        parameter_subtree = value_tree.content.setdefault(
            class_name, IndexComponent("parameter")
        )
        object_labels = get_object_labels(row)
        entity_subtree = parameter_subtree.content.setdefault(
            row.parameter_name, IndexComponent(object_labels[0])
        )
        for entity, label in zip(objects[:-1], object_labels[1:]):
            entity_subtree = entity_subtree.content.setdefault(
                entity, IndexComponent(label)
            )
        scenario_subtree = entity_subtree.content.setdefault(objects[-1], IndexComponent("scenario"))
        if not isinstance(parameter_value, Map):
            parameter_value = convert_containers_to_maps(parameter_value)
        scenario_subtree.content[model_scenario] = convert_map_to_index_components(
            parameter_value
        )
    return value_tree


def gather_plotting_data(
    index_component: IndexComponent,
    index_names: MutableSequence[str] = None,
    indexes: MutableSequence[str] = None,
) -> Generator[XYData, None, None]:
    """Constructs plottable data and indexes recursively."""
    if index_names is None:
        index_names = []
    if indexes is None:
        indexes = []
    current_index_names = index_names + [index_component.label]
    content_type = type(next(iter(index_component.content.values()), None))
    if content_type == float:
        x = [x for x in index_component.content.keys()]
        if len(x) < 2 or not x[0].startswith("t"):
            return
        y = [y for y in index_component.content.values()]
        x_label = current_index_names[-1]
        y_label = ""
        yield XYData(x, y, x_label, y_label, indexes, current_index_names[:-1])
    else:
        for index, subcomponent in index_component.content.items():
            current_indexes = indexes + [index]
            yield from gather_plotting_data(
                subcomponent, current_index_names, current_indexes
            )


def pop_filters(url: str) -> Tuple[Optional[str], str]:
    """Pops filters from URL and parses active scenario name."""
    configs, bare_url = pop_filter_configs(url)
    for config in configs:
        if config["type"] == SCENARIO_FILTER_TYPE:
            return name_from_dict(config), bare_url
    return None, bare_url


def prepare_component_figures(component: IndexComponent):
    """Plots data from index component on matplotlib figures."""
    for entity_class, parameter_component in component.content.items():
        for parameter, entity_component in parameter_component.content.items():
            figure = None
            axes = None
            legend_axes = None
            legend_handles = []
            for data in gather_plotting_data(entity_component):
                if figure is None:
                    figure, (axes, legend_axes) = plt.subplots(1, 2, figsize=(12.0, 4.8), tight_layout=True, gridspec_kw={"width_ratios": [1, 0]})
                    title = f"{entity_class} - {parameter}"
                    figure.canvas.manager.set_window_title(f"FlexTool [{title}]")
                    axes.set_title(title)
                    axes.set_xlabel(data.x_label)
                    axes.set_ylabel(data.y_label)
                    axes.xaxis.set_major_locator(MaxNLocator(10))
                    legend_axes.axis("off")
                line, = axes.plot(data.x, data.y, label=" | ".join(data.data_index))
                legend_handles.append(line)
            if legend_axes is not None:
                legend_axes.legend(handles=legend_handles, loc="upper center")


def plot(url: str) -> None:
    """Plots all time series results."""
    db_map = DatabaseMapping(url)
    try:
        object_parameter_values = query_parameter_values(
            EntityType.OBJECT, db_map
        )
        relationship_parameter_values = query_parameter_values(
            EntityType.RELATIONSHIP, db_map
        )
    finally:
        db_map.connection.close()
    prepare_component_figures(object_parameter_values)
    prepare_component_figures(relationship_parameter_values)
    plt.show()


def main() -> None:
    """Main entry point to the script."""
    arg_parser = make_argument_parser()
    args = arg_parser.parse_args()
    plot(args.url)


if __name__ == "__main__":
    main()
