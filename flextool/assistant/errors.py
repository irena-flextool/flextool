"""Every message the assistant tools give the user, in one table, and the exception that carries one.

A tool raises :class:`ToolError` with a key of :data:`MESSAGES` and the fields its text needs; ``call_tool``
turns it into an error outcome.  The texts are plain English for a FlexTool user: no abbreviations, no
internal names, and a next step where there is one (``tests/assistant/test_tool_wording.py``).
"""
from __future__ import annotations

MESSAGES: dict[str, str] = {
    # The registry
    "unknown_tool": "There is no tool named '{name}'. The tools are: {names}.",
    "bad_arguments": "The arguments of {name} are not valid: {problem}.",
    "not_in_level": ("The tool {name} is not part of the basic level. Restart the FlexTool connection without "
                     "--tier basic to use it."),
    "fault": ("FlexTool could not answer this request ({type}). This is a fault in the assistant tools; please "
              "report it."),
    # Documentation
    "unknown_doc": "There is no document named '{doc}'. Available documents: {docs}.",
    "unknown_section": "The document {doc} has no section '{section}'. Its sections include: {sections}.",
    "docs_unavailable": "{text}",
    "unknown_formulation": "There is no equation, variable or objective term named '{name}'.{similar}",
    # Input sources
    "no_sources": ("The project has no input sources (spreadsheets or databases in its input_sources folder, "
                   "or files added to the project)."),
    "unknown_source": "There is no input source '{source}' in this project. The input sources are: {sources}.",
    "choose_source": "This project has several input sources; name one of them: {sources}.",
    "source_missing": "The file of input source '{source}' is missing: {path}.",
    "not_converted": ("This input source is a spreadsheet. FlexTool converts it when the scenario runs. Run (or "
                      "convert) it first."),
    "converting": "This spreadsheet is being converted now; ask again in a moment.",
    "busy": "The input database is busy (another program is saving to it). Try again in a moment.",
    "old_spine_format": ("This database uses an older Spine database format. Open it once in the FlexTool "
                         "window, which updates it, then ask again."),
    "not_flextool": "'{name}' is not a FlexTool input database.",
    "unreadable": "The input database '{name}' could not be read: {reason}.",
    "no_scenarios": "The input source '{source}' has no scenarios.",
    "unknown_scenario": "There is no scenario '{scenario}' in '{source}'. Its scenarios include: {scenarios}.",
    "choose_scenario": "The input source '{source}' has {n} scenarios; name one of them: {scenarios}.",
    # Input data
    "unknown_class": "There is no entity class '{entity_class}'.{similar}",
    "unknown_parameter": "The entity class '{entity_class}' has no parameter '{parameter}'.{similar}",
    "unknown_entity": "There is no {entity_class} named {entity} in '{source}'.{similar}",
    "entity_not_found": "No entity named {entity} exists in '{source}' in any entity class.{similar}",
    "entity_dimensions": ("A {entity_class} is named by {n} names ({dimensions}); give the entity as a list of "
                          "{n} names."),
    "older_database": ("This database is {relation} than this FlexTool; the value may be stored under another "
                       "parameter name. Parameters stored for this entity: {names}."),
    "inactive_entity": "'{entity}' exists in the input data but is not active in scenario '{scenario}' ({reason}).",
    "no_value": "No alternative of the scenario sets this value and the parameter has no default.",
    "bad_time_range": "time_range takes {{\"from\": first index, \"to\": last index}}, for example {example}.",
    # Runs
    "no_runs": "The project has no scenario results (no folders in output_parquet).",
    "bad_run_name": "'{run}' is not a result folder name; use a folder name that list_runs gives.",
    "unknown_run": "There is no result folder '{run}'. The folders include: {runs}.",
    "run_running": "Scenario folder {run} is running now; its results are not complete.",
    "run_abandoned": ("The latest run of {run} stopped without finishing. Read the run log (get_run_log) and run "
                      "it again."),
    "run_failed": ("The latest run of {run} failed: {error}. Result files left in the folder belong to an earlier "
                   "run. Read the run log (get_run_log) and get the scenario to run successfully first."),
    "run_no_parquet": ("The latest run of {run} wrote no result files to read (its write methods: {methods}). Run "
                       "it with parquet output."),
    # Result tables
    "unknown_table": "There is no result table '{table}' in {run}.{similar}",
    "unknown_topic": "There is no topic '{topic}' among the result tables of {run}. The topics are: {topics}.",
    "result_damaged": ("This result file is being written or is damaged. If a run is in progress, wait for it to "
                       "finish."),
    "unknown_level": "The table {table} has no level '{level}' (in {what}). Its levels are: {levels}.",
    "no_match": "No value of the level {level} of {table} matches the filter. Its values include: {values}.",
    "no_match_combination": "No column of {table} matches all the filters together.",
    "no_time_steps": "{what} needs a table with time steps; {table} has none.",
    "bad_time_step": ("time_range: the table {table} has no time step '{value}'. Its time steps run from {first} "
                      "to {last}."),
    "budget": "This would return about {n} values. Add filters, group_by or top_n.",
    "bad_sort": "sort.by must be one of the result's columns: {columns} (not '{by}').",
    "membership_table": ("{table} is a set of names with no values; it can be listed (with filters) but not "
                         "aggregated."),
    "mixed_units": ("The values of {table} combined here have different units ({units}). Keep the level {level} "
                    "in group_by or filter one of its values."),
    "sum_rate_over_time": ("This table holds power at each timestep. Its sum over time is not energy unless each "
                           "step is weighted by its duration, which the result files do not record.{use}"),
    "sum_rate_over_periods": ("This table holds power at each timestep; adding it across periods gives no "
                              "meaningful total. Compare periods side by side (keep period in group_by).{use}"),
    "sum_annual_over_periods": ("These are annual values; adding them across periods does not give a total over "
                                "the horizon. Use the discounted cost table, or compare periods side by side."),
    "sum_per_period_over_periods": ("These values are sums over the sampled time steps of each period, not "
                                    "scaled to the years a period represents; adding them across periods does not "
                                    "give a total over the horizon. Compare periods side by side."),
    "sum_level": ("This table holds levels or averages (a capacity, a storage state, a mean over the period); "
                  "adding them over {span} gives no meaningful total. Use mean, min or max, or keep the {span} "
                  "apart in group_by."),
    "sum_ratio": "This table holds {what}; adding them gives no meaningful total. Use mean, min or max.",
    "sum_unknown": ("FlexTool does not record what kind of quantity {table} holds, so a sum may mean nothing. Use "
                    "mean, min or max."),
    "compare_needs_runs": ("compare_runs needs two or more result folders from list_runs, or [\"comparison\"] for "
                           "the comparison files."),
    "no_comparison": ("The project has no comparison files (output_parquet_comparison). Name the scenario folders "
                      "to compare."),
    "table_in_no_run": "None of the runs {runs} has the result table '{table}'.",
    # Run logs and slacks
    "no_run_record": ("This run has no run record (it ran before FlexTool kept run logs, or with an older "
                      "FlexTool). Run the scenario again to get one."),
    "unknown_solve": "The run {run} has no solve named '{solve}'. Its solves include: {solves}.",
    "needs_migration": ("This database is at version {db}; this FlexTool uses version {tool}. Some values may be "
                        "stored under older parameter names. Open the project in the FlexTool window to update it."),
    "newer_database": ("This database is at version {db}, newer than this FlexTool (version {tool}). Update "
                       "FlexTool to read all its parameters."),
}


class ToolError(Exception):
    """A request the tool cannot answer, with the message for the user."""

    def __init__(self, key: str, **fields: object) -> None:
        self.key = key
        self.fields = fields
        super().__init__(message(key, **fields))


def message(key: str, **fields: object) -> str:
    """The text of :data:`MESSAGES` ``[key]`` with *fields* filled in."""
    return MESSAGES[key].format(**fields)


def close_names(word: str, candidates: list[str], n: int = 3) -> list[str]:
    """Up to *n* names of *candidates* close to *word* (spelling, then containing it)."""
    import difflib

    close = difflib.get_close_matches(word, candidates, n=n, cutoff=0.6)
    if not close:
        lower = word.lower()
        close = [c for c in candidates if lower in c.lower()][:n]
    return close


def similar_text(word: str, candidates: list[str], *, label: str = "Similar names") -> str:
    """`` Similar names: a, b, c.`` for up to three close matches of *word*, else an empty string."""
    close = close_names(word, candidates)
    return f" {label}: {', '.join(close)}." if close else ""


def name_list(names: list[str], limit: int = 20) -> str:
    """*names* joined with commas, at most *limit* of them, with the count of the rest."""
    shown = list(names[:limit])
    rest = len(names) - len(shown)
    text = ", ".join(shown) if shown else "(none)"
    return f"{text} and {rest} more" if rest > 0 else text
