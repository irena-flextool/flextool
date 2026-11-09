"""
db_reader — Pure functions for reading from spinedb_api.
No state, no side effects. All functions take explicit parameters.
"""
import logging
from collections import defaultdict
from enum import Enum

import spinedb_api as api

from flextool import run_record
from flextool.engine.input._solve_state import FlexToolConfigError


class DictMode(Enum):
    DICT = "dict"
    DEFAULTDICT = "defaultdict"
    LIST = "list"


def check_version(db: api.DatabaseMapping, logger: logging.Logger) -> None:
    """Verify that the input database version is compatible with this tool (:class:`FlexToolConfigError` if not)."""
    db_version_item = db.get_parameter_definition_item(entity_class_name="model", name="version")
    if not db_version_item:
        raise FlexToolConfigError(
            "No version information found in the FlexTool input database, check you have a correct database."
        )
    database_version = api.from_database(db_version_item["default_value"], db_version_item["default_type"])
    try:
        _version = float(database_version)
        run_record.active().note_input(db_version=int(_version) if _version.is_integer() else _version)
    except (TypeError, ValueError):
        run_record.active().note_input(db_version=str(database_version))
    from flextool.update_flextool import FLEXTOOL_DB_VERSION

    try:
        found = float(database_version)
    except (TypeError, ValueError):
        raise FlexToolConfigError(
            f"The FlexTool version of the input database ({database_version!r}) is not a number; "
            "check you have a correct database."
        ) from None
    if found == FLEXTOOL_DB_VERSION:
        return
    location = _database_location(db)
    shown = int(found) if found.is_integer() else found
    if found < FLEXTOOL_DB_VERSION:
        raise FlexToolConfigError(
            f"The input database is at FlexTool data version {shown}; this FlexTool needs version "
            f"{FLEXTOOL_DB_VERSION}. FlexTool does not run a database that has not been migrated.\n"
            f"Migrate it first:  flextool-migrate-database {location}\n"
            "A frozen database is not migrated in place: add --copy-to NEW.sqlite to migrate a copy and run that.\n"
            "(The FlexTool window migrates its input sources when it refreshes them, and Spine Toolbox runs the "
            "'Migrate database' tool before FlexTool.)"
        )
    raise FlexToolConfigError(
        f"The input database is at FlexTool data version {shown}, newer than this FlexTool (version "
        f"{FLEXTOOL_DB_VERSION}). Update FlexTool (flextool-update, or pip install --upgrade flextool) to run it."
    )


def _database_location(db: api.DatabaseMapping) -> str:
    """The file of a sqlite input database for the migrate hint (its URL otherwise, ``<database>`` if unknown)."""
    url = getattr(db, "sa_url", None)
    if url is None:
        return "<database>"
    database = getattr(url, "database", None)
    if getattr(url, "drivername", "").startswith("sqlite") and database:
        return database
    return str(url)


def get_single_entities(db: api.DatabaseMapping, entity_class_name: str) -> list[str]:
    """Return a list of entity names for a single-dimension entity class."""
    return [entity["entity_byname"][0] for entity in db.find_entities(entity_class_name=entity_class_name)]


def entities_to_dict(
    db: api.DatabaseMapping, cl: str, mode: DictMode
) -> dict | defaultdict:
    """
    Read multi-dimension entities of class *cl* and return as a dict.

    mode="defaultdict" → defaultdict(list)
    mode="dict"        → plain dict
    Each entity maps entity_byname[0] → list of remaining byname elements.
    """
    entities = db.find_entities(entity_class_name=cl)
    if mode == DictMode.DEFAULTDICT:
        result: dict | defaultdict = defaultdict(list)
    else:
        result = dict()
    for entity in entities:
        if len(entity["entity_byname"]) > 1:
            result[entity["entity_byname"][0]] = list(entity["entity_byname"][1:])
        else:
            raise ValueError(
                "Only one dimension in the entity, cannot make into a dict in entities_to_dict"
            )
    return result


def params_to_dict(
    db: api.DatabaseMapping,
    cl: str,
    par: str,
    mode: DictMode,
    str_to_list: bool = False,
) -> dict | defaultdict | list:
    """
    Read parameter values of *par* on entity class *cl* and return as a
    dict, defaultdict, or list depending on *mode*.

    mode="defaultdict" → defaultdict(list)
    mode="dict"        → plain dict
    mode="list"        → list of [entity_name, value] pairs
    str_to_list        → wrap string values in a list (only for dict/defaultdict modes)
    """
    all_params = db.find_parameter_values(
        entity_class_name=cl, parameter_definition_name=par
    )
    if mode == DictMode.DEFAULTDICT:
        result: dict | defaultdict | list = defaultdict(list)
    elif mode == DictMode.DICT:
        result = dict()
    elif mode == DictMode.LIST:
        result = []
    for param in all_params:
        param_value = api.from_database(param["value"], param["type"])
        if mode in (DictMode.DEFAULTDICT, DictMode.DICT):
            if isinstance(param_value, api.Map):
                if isinstance(param_value.values[0], float):
                    result[param["entity_name"]] = list(
                        zip(list(param_value.indexes), list(map(float, param_value.values)))
                    )
                elif isinstance(param_value.values[0], str):
                    result[param["entity_name"]] = list(
                        zip(list(param_value.indexes), param_value.values)
                    )
                elif isinstance(param_value.values[0], api.Map):
                    result[param["entity_name"]] = api.convert_map_to_table(param_value)
                else:
                    raise TypeError(
                        "params_to_dict function does not handle other values than floats and strings"
                    )
            elif isinstance(param_value, api.Array):
                result[param["entity_name"]] = param_value.values
            elif isinstance(param_value, float):
                result[param["entity_name"]] = str(param_value)
            elif isinstance(param_value, str):
                if str_to_list:
                    result[param["entity_name"]] = [param_value]
                else:
                    result[param["entity_name"]] = param_value
        elif mode == DictMode.LIST:
            if isinstance(param_value, (float, str)):
                result.append([param["entity_name"], param_value])  # type: ignore[union-attr]
    return result
