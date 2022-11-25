"""
This script plots FlexTool data from a Spine database following instructions given in a JSON file.

The structure of the JSON file is as follows:

{
  "plots": [
    {
      "plot_type": <plot type>,
      "selection": {
        "entity_class": <entity class list>,
        "entity": <entity list>,
        "parameter": <parameter list>,
        "alternative": <alternative list>,
        "solve": <solve list>,
        "period": <period list>
        ...
      },
      "dimensions": {
        "separate_window": <item type>,
        "x1": <item type>,
        "x2": <item type>,
        "x3": <item type>
      }
    }
  ]
}

<plot type>: one of: "line", "stacked line", "bar", "stacked bar", "heatmap"

The entries in the selection object choose what is included in the plot.
<entity class list>: a list of entity class names to include in the plot
<entity list>: a list of lists of entity names to include in the plot; empty list includes all
    Examples:
        include all: []
        include selected objects: [["coal", "oil", "peat"]]
        select 1st relationship dimension, include all from 2nd dimension: [["coal_plant"], []]
<parameter list>: a list of parameter names; empty list includes all
<alternative list>: a list of alternative names; empty list includes all
<solve list>: a list of solve names; empty list includes all
<period list>: a list of period names; empty list includes all
It is possible to include other dimensions such as "cost_type" or "flow_type" in similar fashion.

The entries in "dimensions" object accept one of the following <item type> values:
    null, object class name, "parameter", "alternative", "solve", "period", "cost_type", "flow_type" etc.
separate_window: each item of this type will get its own plot window
x1: which item to use as the x-axis; defaults to the last dimension, e.g. time for *_t parameters
x2: regroups or categorises the x-axis by this item
x3: minor x-axis regrouping item; works only if x2 is defined
"""
import subprocess
import sys
import traceback
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, replace
from enum import IntEnum, unique
from io import TextIOWrapper
from itertools import accumulate, zip_longest
import json
from operator import attrgetter
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep
from typing import Any, Callable, Dict, Iterator, Optional, List, Tuple
import matplotlib
import numpy as np
from PySide2.QtCore import QTimer
from PySide2.QtWidgets import QApplication
from matplotlib.ticker import MaxNLocator
from sqlalchemy.sql.expression import Alias, and_
from spinedb_api import convert_containers_to_maps, DatabaseMapping, from_database, Map
from spinedb_api.db_mapping_base import DatabaseMappingBase
from spinetoolbox.plotting import (
    combine_data_with_same_indexes,
    convert_indexed_value_to_tree,
    plot_data,
    PlottingError,
    PlotType,
    reduce_indexes,
    TreeNode,
    turn_node_to_xy_data,
    XYData,
)
from spinetoolbox.widgets.plot_widget import PlotWidget

matplotlib.use("Qt5Agg")


BASIC_PLOT_TYPES = {"line", "stacked line", "bar", "stacked bar"}


@unique
class EntityType(IntEnum):
    """Spine database entity type enums."""

    OBJECT = 1
    RELATIONSHIP = 2


@dataclass(frozen=True)
class ImageData:
    """Image data and metadata."""

    image: np.ndarray
    row_labels: List[str]
    column_labels: List[str]


def make_argument_parser() -> ArgumentParser:
    """Creates a command line argument parser."""
    parser = ArgumentParser(description="Plot FlexTool results.")
    parser.add_argument(
        "--use-subprocess",
        action="store_true",
        help="create an independent process for plotting",
    )
    parser.add_argument(
        "url", metavar="URL", help="URL pointing to the result database"
    )
    parser.add_argument("settings", help="path to settings JSON file")
    parser.add_argument(
        "notification_file",
        nargs="?",
        default=None,
        help="file to create when the application starts",
    )
    return parser


def reject_objects(objects: List[str], acceptable_objects: List[List[str]]) -> bool:
    """
    Returns True if any object in objects list is not
    in any corresponding list of acceptable_objects.
    """
    for object_, acceptables in zip(objects, acceptable_objects):
        if acceptables and object_ not in acceptables:
            return True
    return False


def entity_handling_functions(
    entity_type: EntityType
) -> Tuple[
    Callable[[object], str], Callable[[Any], List[str]], Callable[[Any], List[str]]
]:
    """
    Generates callable suitable for retrieving information
    from database row of given entity type.
    """
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
    return get_class_name, get_object_names, get_object_labels


def query_parameter_values(
    entity_type: EntityType,
    filter_conditions: Tuple,
    accept_objects: Optional[List[List[str]]],
    db_map: DatabaseMappingBase,
) -> TreeNode:
    """Reads parameter values from database."""
    value_tree = TreeNode("class")
    get_class_name, get_object_names, get_object_labels = entity_handling_functions(
        entity_type
    )
    subquery = {
        EntityType.OBJECT: db_map.object_parameter_value_sq,
        EntityType.RELATIONSHIP: db_map.relationship_parameter_value_sq,
    }[entity_type]
    for row in db_map.query(subquery).filter(and_(*filter_conditions)):
        objects = get_object_names(row)
        if (
            entity_type == EntityType.RELATIONSHIP
            and accept_objects
            and reject_objects(objects, accept_objects)
        ):
            continue
        object_labels = get_object_labels(row)
        entity_subtree = value_tree.content.setdefault(
            get_class_name(row), TreeNode("parameter")
        ).content.setdefault(row.parameter_name, TreeNode(object_labels[0]))
        for entity, label in zip(objects[:-1], object_labels[1:]):
            entity_subtree = entity_subtree.content.setdefault(entity, TreeNode(label))
        alternative_subtree = entity_subtree.content.setdefault(
            objects[-1], TreeNode("alternative")
        )
        parameter_value = from_database(row.value, row.type)
        if not isinstance(parameter_value, Map):
            parameter_value = convert_containers_to_maps(parameter_value)
        alternative_subtree.content[
            row.alternative_name
        ] = convert_indexed_value_to_tree(parameter_value)
    return value_tree


def make_image(data_list: List[XYData]) -> ImageData:
    """Merges xy data into heat map."""
    rows = []
    row_labels = []
    column_labels = []
    for xy_data in data_list:
        rows.append(np.array(xy_data.y))
        row_labels.append(" | ".join(xy_data.data_index))
        if not column_labels:
            column_labels = xy_data.x
        elif xy_data.x != column_labels:
            raise PlottingError("x-axes mismatch")
    heat_map = np.vstack(rows) if rows else np.array([])
    return ImageData(heat_map, row_labels, column_labels)


def tile_horizontally(
    data_list: List[XYData]
) -> Tuple[List[XYData], Dict[str, List[str]]]:
    """Creates 'ghost' x-axis such that each xy data is plotted next to each other."""
    tiled = []
    categories = {}
    x_lookups_per_category = {}
    for xy_data in data_list:
        x_lookup = x_lookups_per_category.setdefault(xy_data.data_index[-1], {})
        for x in xy_data.x:
            if x not in x_lookup:
                x_lookup[x] = len(x_lookup)
    offsets = {}
    current_offset = 0
    for category, x_lookup in x_lookups_per_category.items():
        offsets[category] = current_offset
        current_offset += len(x_lookup)
    for category, x_lookup in x_lookups_per_category.items():
        categories[category] = list(x_lookup)
    for xy_data in data_list:
        category = xy_data.data_index[-1]
        x_lookup = x_lookups_per_category[category]
        offset = offsets[category]
        tiled.append(replace(xy_data, x=[offset + x_lookup[x] for x in xy_data.x]))
    return tiled, categories


def categorize_further(
    subcategories: Dict[str, List[str]], data_list: List[XYData]
) -> Dict[str, List[str]]:
    """Creates 'ghost' x-axis such that each xy data is plotted next to each other."""
    categories = {}
    subcategory_lookup = {}
    for xy_data in data_list:
        category = xy_data.data_index[-2]
        subcategory = xy_data.data_index[-1]
        subcategory_lookup.setdefault(category, set()).add(subcategory)
    current_subcategories = subcategories
    for category, contained_subcategories in subcategory_lookup.items():
        unused_subcategories = {}
        for subcategory, x in current_subcategories.items():
            if subcategory not in contained_subcategories:
                unused_subcategories[subcategory] = x
                continue
            categories.setdefault(category, []).extend(x)
        current_subcategories = unused_subcategories
    return categories


def drop_data_index_tail(data_list: List[XYData], count: int) -> List[XYData]:
    """Deletes data indices from the end."""
    return [
        replace(
            xy_data,
            data_index=xy_data.data_index[:-count],
            index_names=xy_data.index_names[:-count],
        )
        for xy_data in data_list
    ]


def category_ticks(
    categories: Dict[str, List[str]], x_min: float, x_max: float
) -> Tuple[List[float], Dict[str, float]]:
    """Calculates major and minor tick positions for category x-axis."""
    x_width = x_max - x_min
    first_divider = -x_min / x_width
    category_sizes = list(accumulate(len(labels) for labels in categories.values()))
    count = category_sizes[-1]
    category_width = 1.0 - (x_max - x_min - (count - 1)) / x_width
    step = category_width / count
    category_dividers = [first_divider] + list(
        first_divider + size * step for size in category_sizes
    )
    category_labels = {}
    for i, name in enumerate(categories):
        first_major = category_dividers[i]
        category_labels[name] = (first_major + category_dividers[i + 1]) / 2.0
    return category_dividers, category_labels


def add_category_spine(
    offset: int,
    category_labels: Dict[str, float],
    category_dividers: List[float],
    plot_widget: [PlotWidget],
):
    """Adds a category spine to plot widget."""
    category_axis = plot_widget.canvas.axes.twiny()
    point_offset = 35 + offset * 17
    category_axis.spines["bottom"].set_position(("outward", point_offset))
    category_axis.tick_params("both", length=0, width=0, which="minor")
    category_axis.tick_params("both", direction="in", which="major")
    category_axis.xaxis.set_ticks_position("bottom")
    category_axis.xaxis.set_label_position("bottom")
    if (
        len(category_dividers) == 2
        and category_dividers[0] - category_dividers[1] == 0.0
    ):
        category_axis.set_xticks([category_dividers[0]])
        category_axis.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(list(category_labels))
        )
        return
    category_axis.set_xticks(category_dividers)
    category_axis.xaxis.set_major_formatter(matplotlib.ticker.NullFormatter())
    category_axis.xaxis.set_minor_locator(
        matplotlib.ticker.FixedLocator(list(category_labels.values()))
    )
    category_axis.xaxis.set_minor_formatter(
        matplotlib.ticker.FixedFormatter(list(category_labels))
    )


def relabel_x_axis(
    categories: Dict[str, List[str]], x_ticks: np.ndarray
) -> Tuple[List[float], List[str]]:
    """Replaces numerical x-axis by string labels."""
    all_labels = sum((labels for labels in categories.values()), [])
    if len(all_labels) == 1:
        return [0.0], all_labels[0:1]
    begin = max(0.0, round(x_ticks[0]))
    end = min(len(all_labels), round(x_ticks[-1]))
    tick_positions = [i for i in x_ticks if begin <= i < end and i == round(i)]
    labels = [all_labels[int(pos)] for pos in tick_positions]
    return tick_positions, labels


def fetch_entity_class_types(db_map: DatabaseMappingBase) -> Dict[str, EntityType]:
    """Connects entity class names to whether they are object or relationship classes."""
    return {
        row.name: EntityType.OBJECT if row.type_id == 1 else EntityType.RELATIONSHIP
        for row in db_map.query(db_map.entity_class_sq)
    }


def make_object_filter(
    object_classes: List[str],
    objects: List[List[str]],
    parameters: List[str],
    alternatives: List[str],
    subquery: Alias,
) -> Tuple:
    """Creates object parameter value query filters."""
    filters = ()
    if object_classes:
        filters = filters + (subquery.c.object_class_name.in_(object_classes),)
    if objects:
        filters = filters + (subquery.c.object_name.in_(objects[0]),)
    if parameters:
        filters = filters + (subquery.c.parameter_name.in_(parameters),)
    if alternatives:
        filters = filters + (subquery.c.alternative_name.in_(alternatives),)
    return filters


def make_relationship_filter(
    relationship_classes: List[str],
    parameters: List[str],
    alternatives: List[str],
    subquery: Alias,
) -> Tuple:
    """Creates relationship parameter value query filters."""
    filters = ()
    if relationship_classes:
        filters = filters + (
            subquery.c.relationship_class_name.in_(relationship_classes),
        )
    if parameters:
        filters = filters + (subquery.c.parameter_name.in_(parameters),)
    if alternatives:
        filters = filters + (subquery.c.alternative_name.in_(alternatives),)
    return filters


def build_parameter_value_tree(
    plot_selection: Dict,
    db_map: DatabaseMappingBase,
    entity_class_types: Dict[str, EntityType],
) -> TreeNode:
    """Builds data tree according to given plot settings."""
    object_parameter_values = None
    relationship_parameter_values = None
    entity_classes = plot_selection.get("entity_class", [])
    object_classes = [
        class_
        for class_ in entity_classes
        if entity_class_types.get(class_) == EntityType.OBJECT
    ]
    if object_classes:
        filter_conditions = make_object_filter(
            object_classes,
            plot_selection.get("entity", []),
            plot_selection.get("parameter", []),
            plot_selection.get("alternative", []),
            db_map.object_parameter_value_sq,
        )
        object_parameter_values = query_parameter_values(
            EntityType.OBJECT, filter_conditions, None, db_map
        )
    relationship_classes = [
        class_
        for class_ in entity_classes
        if entity_class_types.get(class_) == EntityType.RELATIONSHIP
    ]
    if relationship_classes:
        filter_conditions = make_relationship_filter(
            relationship_classes,
            plot_selection.get("parameter", []),
            plot_selection.get("alternative", []),
            db_map.relationship_parameter_value_sq,
        )
        relationship_parameter_values = query_parameter_values(
            EntityType.RELATIONSHIP,
            filter_conditions,
            plot_selection.get("entity", []),
            db_map,
        )
    if object_parameter_values is not None and object_parameter_values.content:
        parameter_values = object_parameter_values
        if (
            relationship_parameter_values is not None
            and relationship_parameter_values.content
        ):
            merged_content = dict(
                **parameter_values.content, **relationship_parameter_values.content
            )
            return replace(parameter_values, content=merged_content)
        return parameter_values
    if (
        relationship_parameter_values is not None
        and relationship_parameter_values.content
    ):
        return relationship_parameter_values
    return None


def filter_by_data_index(
    data_list: List[XYData], index_label: str, accepted_values: List[str]
) -> List[XYData]:
    """Removes xy data that does not have an acceptable index data under given label."""
    filtered_list = []
    for xy_data in data_list:
        try:
            i = xy_data.index_names.index(index_label)
        except ValueError:
            continue
        if xy_data.data_index[i] in accepted_values:
            filtered_list.append(xy_data)
    return filtered_list


def separate(separate_window: str, data_list: List[XYData]) -> Iterator[List[XYData]]:
    """Yields chunks of data list that should be plotted separately."""
    if not separate_window:
        yield data_list
        return
    baskets: Dict[str, List[XYData]] = {}
    for xy_data in data_list:
        baskets.setdefault(data_index_at(separate_window, xy_data), []).append(xy_data)
    yield from baskets.values()


def make_shuffle_instructions(plot_dimensions: Dict) -> Dict:
    """Generates shuffle instructions based on plot settings."""
    instructions = {}
    x1 = plot_dimensions.get("x1")
    if x1 is not None:
        instructions[x1] = "x"
    x2 = plot_dimensions.get("x2")
    if x2 is not None:
        instructions[x2] = -1
        x3 = plot_dimensions.get("x3")
        if x3 is not None:
            instructions[x3] = -2
    return instructions


def shuffle_dimensions(instructions: Dict, data_list: List[XYData]) -> List[XYData]:
    """Moves xy data indexes around."""
    current_list = data_list
    for index_name, target in instructions.items():
        if target == "x":
            if index_name == current_list[0].x_label:
                continue
            current_list = insert_as_x(index_name, current_list)
            continue
        new_list = []
        for xy_data in current_list:
            usable_target = target if target >= 0 else target + len(xy_data.data_index)
            source = xy_data.index_names.index(index_name)
            if source == usable_target:
                new_list.append(xy_data)
                continue
            data_index = xy_data.data_index[source]
            new_data_index = [
                i for n, i in enumerate(xy_data.data_index) if n != source
            ]
            new_data_index.insert(usable_target, data_index)
            new_index_names = [
                name for n, name in enumerate(xy_data.index_names) if n != source
            ]
            new_index_names.insert(usable_target, index_name)
            new_list.append(
                replace(xy_data, data_index=new_data_index, index_names=new_index_names)
            )
        current_list = new_list
    return current_list


def insert_as_x(index_name: str, data_list: List[XYData]) -> List[XYData]:
    """Moves given data index to x-axis."""
    root_node = None
    y_label_position = "undefined"
    for xy_data in data_list:
        if index_name not in xy_data.index_names:
            raise RuntimeError(f"unknown dimension '{index_name}'")
        if y_label_position == "undefined":
            try:
                y_label_position = xy_data.index_names.index(xy_data.y_label)
            except ValueError:
                y_label_position = None
        moved_data_index = data_index_at(index_name, xy_data)
        new_indices = [
            (i, name)
            for i, name in zip(xy_data.data_index, xy_data.index_names)
            if name != index_name
        ]
        last = (moved_data_index, index_name)
        table = [new_indices + [(x, xy_data.x_label), last] for x in xy_data.x]
        root_label = table[0][0][1]
        if root_node is None:
            root_node = TreeNode(root_label)
        elif root_node.label != root_label:
            raise RuntimeError("root node label mismatch")
        for row, indices in enumerate(table):
            current_node = root_node
            for column, index in enumerate(indices[:-1]):
                current_node = current_node.content.setdefault(
                    index[0], TreeNode(indices[column + 1][1])
                )
            current_node.content[indices[-1][0]] = xy_data.y[row]
    return list(turn_node_to_xy_data(root_node, y_label_position))


def toolbox_plot_type(plot_type: str) -> PlotType:
    """Converts plot type to the one used in Spine Toolbox."""
    return {
        "line": PlotType.LINE,
        "stacked line": PlotType.STACKED_LINE,
        "bar": PlotType.BAR,
        "stacked bar": PlotType.STACKED_BAR,
    }[plot_type]


def filtered_data_list(
    plot_selection: Dict, parameter_values: TreeNode
) -> List[XYData]:
    """Turns parameter value tree into data list and filters by nodes and periods."""
    data_list = list(turn_node_to_xy_data(parameter_values, 1))
    used_index_names = {"entity_class", "entity", "parameter", "alternative"}
    for index_name, accepted_values in plot_selection.items():
        if index_name in used_index_names or not accepted_values:
            continue
        data_list = filter_by_data_index(data_list, index_name, accepted_values)
    return data_list


def plot_basic(
    plot_type: PlotType, plot_dimensions: Dict, data_list: List[XYData]
) -> PlotWidget:
    """Plots basic plot types, e.g. line, scatter etc."""
    category_list = []
    if plot_dimensions.get("x2") is not None:
        data_list, categories_1 = tile_horizontally(data_list)
        category_list.append(categories_1)
        if plot_dimensions.get("x3") is not None:
            categories_2 = categorize_further(categories_1, data_list)
            category_list.append(categories_2)
        data_list = drop_data_index_tail(data_list, len(category_list))
    plot_widget = plot_data(data_list, plot_type=toolbox_plot_type(plot_type))
    if category_list:
        x_min, x_max = plot_widget.canvas.axes.get_xlim()

        x_ticks, x_labels = relabel_x_axis(
            category_list[0], plot_widget.canvas.axes.get_xticks()
        )
        plot_widget.canvas.axes.set_xticks(x_ticks, labels=x_labels)
        for offset, categories in enumerate(category_list):
            category_dividers, category_labels = category_ticks(
                categories, x_min, x_max
            )
            add_category_spine(offset, category_labels, category_dividers, plot_widget)
    return plot_widget


def check_entity_classes(
    settings: Dict,
    entity_class_types: Dict[str, EntityType],
    file: TextIOWrapper = sys.stdout,
):
    """Prints warnings if settings contain unknown entity classes."""
    for plot_settings in settings["plots"]:
        entity_classes = plot_settings["selection"].get("entity_class", [])
        for class_ in entity_classes:
            if class_ not in entity_class_types:
                print(f"entity class '{class_}' not in database; ignoring", file=file)


def plot(url: str, settings: Dict) -> bool:
    """Plots data as defined in settings."""
    db_map = DatabaseMapping(url)
    did_plot = False
    try:
        entity_class_types = fetch_entity_class_types(db_map)
        check_entity_classes(settings, entity_class_types)
        for plot_settings in settings["plots"]:
            plot_selection = plot_settings["selection"]
            parameter_values = build_parameter_value_tree(
                plot_selection, db_map, entity_class_types
            )
            if parameter_values is None:
                continue
            data_list = filtered_data_list(plot_selection, parameter_values)
            plot_type = plot_settings["plot_type"]
            plot_dimensions = plot_settings["dimensions"]
            shuffle_instructions = make_shuffle_instructions(plot_dimensions)
            if shuffle_instructions:
                data_list = shuffle_dimensions(shuffle_instructions, data_list)
            for list_chunk in separate(plot_dimensions["separate_window"], data_list):
                if plot_type in BASIC_PLOT_TYPES:
                    plot_widget = plot_basic(plot_type, plot_dimensions, list_chunk)
                elif plot_type == "heatmap":
                    plot_widget = draw_image(list_chunk)
                else:
                    raise RuntimeError(f"Unknown plot type '{plot_type}'")
                plot_widget.use_as_window(None, "FlexTool results - periods")
                plot_widget.show()
                did_plot = True
        return did_plot
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc(file=sys.stdout)
        return did_plot
    finally:
        db_map.connection.close()


def data_index_at(index_name: str, xy_data: XYData) -> str:
    """Returns data index under given name."""
    return xy_data.data_index[xy_data.index_names.index(index_name)]


def draw_image(data_list: List[XYData]) -> PlotWidget:
    """Draws an image to plot widget."""
    plot_widget = PlotWidget()
    squeezed_data, common_indexes = reduce_indexes(data_list)
    squeezed_data = combine_data_with_same_indexes(squeezed_data)
    if len(squeezed_data) > 1 and any(not data.data_index for data in squeezed_data):
        unsqueezed_index = common_indexes.pop(-1) if common_indexes else "<root>"
        for data in squeezed_data:
            data.data_index.insert(0, unsqueezed_index)
    axes = plot_widget.canvas.axes
    image_data = make_image(squeezed_data)
    axes.imshow(image_data.image, interpolation="none", aspect="auto")
    axes.set_xticks(
        np.arange(len(image_data.column_labels)), labels=image_data.column_labels
    )
    axes.set_yticks(np.arange(len(image_data.row_labels)), labels=image_data.row_labels)
    if len(image_data.column_labels) > 20:
        axes.xaxis.set_major_locator(MaxNLocator(20))
    plot_title = " | ".join(map(str, common_indexes))
    plot_widget.canvas.axes.set_title(plot_title)
    return plot_widget


def start_subprocess(args: Namespace) -> None:
    """Restarts the script as subprocess and waits for the application to start."""
    with TemporaryDirectory() as temp_dir:
        notification_file = Path(temp_dir, ".plotting_started")
        # pylint: disable=consider-using-with
        subprocess.Popen(
            [
                sys.executable,
                sys.argv[0],
                args.url,
                args.settings,
                str(notification_file),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        while True:
            if notification_file.exists():
                break
            sleep(0.5)


def notify_via_file(notification_file: str) -> None:
    """Creates an empty notification file."""
    Path(notification_file).touch()


def main() -> None:
    """Main entry point to the script."""
    arg_parser = make_argument_parser()
    args = arg_parser.parse_args()
    if args.use_subprocess:
        start_subprocess(args)
        return
    with open(args.settings, encoding="utf-8") as settings_file:
        settings = json.load(settings_file)
    # The QApplication instance may already exist when running on Toolbox console
    app = QApplication.instance()
    if app is None:
        app = QApplication()
    app.setApplicationName("FlexTool results")
    did_plot = plot(args.url, settings)
    if not did_plot:
        print("Nothing to plot.")
        if args.notification_file is not None:
            notify_via_file(args.notification_file)
        return
    if args.notification_file is not None:
        QTimer.singleShot(0, lambda: notify_via_file(args.notification_file))
    return_code = app.exec_()
    if return_code != 0:
        raise RuntimeError(f"Unexpected exit status {return_code}")


if __name__ == "__main__":
    main()
