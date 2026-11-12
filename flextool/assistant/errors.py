"""Every message the assistant tools give the user, in one table, and the exception that carries one.

A tool raises :class:`ToolError` with a key of :data:`MESSAGES` and the fields its text needs; ``call_tool``
turns it into an error outcome.  The texts are plain English for a FlexTool user: no abbreviations, no
internal names, and a next step where there is one (``tests/assistant/test_tool_wording.py``).

A message is rendered in one place, :func:`render`, which treats each field by its class (review G7): a name from
the user's files is cleaned, put on one line, redacted, cut to :data:`~flextool.assistant.caps.NAME_MAX` and
JSON-quoted (:func:`quoted`), so it can neither leave its quotes nor start a line of its own; a list of names
(:class:`NameList`) quotes each name; a failure message from outside FlexTool is quoted as one long name; a
sentence FlexTool builds is one-lined, redacted and cut; FlexTool's own words and numbers are cut.  The rendered
text is :class:`~flextool.assistant.caps.Rendered`: it is redacted once, here, and never again.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from string import Formatter

from flextool.assistant.caps import LOG_LINE_MAX, NAME_MAX, Rendered, cap_text, clean

#: How many result folders the filters of a list_runs call hide (list_runs' ``hidden_by_filters`` line, and the
#: overview's refusal when nothing passes): a filter a model meant as "no filter" (``results_usable: false``) shows.
_HIDDEN_BY_FILTERS = "Result folders hidden by the filters ({what}): {n}, {usable} of them with usable results."

MESSAGES: dict[str, str] = {
    # The registry
    "unknown_tool": "There is no tool named {name}. The tools are: {names}.",
    "bad_arguments": "The arguments of {name} are not valid: {problem}.",
    "not_in_level": ("The tool {name} is not part of the basic level. Restart the FlexTool connection without "
                     "--tier basic to use it."),
    "not_at_level": "The tool {name} is not available at this assistant level. Available tools: {names}.",
    "bad_json": "The tool arguments were not valid JSON: {start}",
    "unknown_prompt": "There is no prompt {name}. The prompts are: {names}.",
    "fault": ("FlexTool could not answer this request ({type}). This is a fault in the assistant tools; please "
              "report it."),
    # Documentation
    "unknown_doc": "There is no document named {doc}. Available documents: {docs}.",
    "unknown_section": "The document {doc} has no section {section}. Its sections include: {sections}.",
    "docs_not_included": "The FlexTool documentation is not included in this installation.",
    "docs_build_failed": "The documentation index could not be built; it failed with the message {reason}.",
    "unknown_formulation": "There is no equation, variable or objective term named {name}.{similar}",
    # Input sources
    "no_sources": ("The project has no input sources (spreadsheets or databases in its input_sources folder, "
                   "or files added to the project)."),
    "unknown_source": "There is no input source {source} in this project. The input sources are: {sources}.",
    "choose_source": "This project has several input sources; name one of them: {sources}.",
    "source_missing": "The file of input source {source} is missing: {path}.",
    "not_converted": ("This input source is a spreadsheet. FlexTool converts it when the scenario runs. Run (or "
                      "convert) it first."),
    "converting": "This spreadsheet is being converted now; ask again in a moment.",
    "busy": "The input database is busy (another program is saving to it). Try again in a moment.",
    "old_spine_format": ("This database uses an older Spine database format. Open it once in the FlexTool "
                         "window, which updates it, then ask again."),
    "not_flextool": "{name} is not a FlexTool input database.",
    "unreadable": "The input database {name} could not be read; reading it failed with the message {reason}.",
    "no_scenarios": "The input source {source} has no scenarios.",
    "unknown_scenario": "There is no scenario {scenario} in {source}. Its scenarios include: {scenarios}.",
    "scenario_in_no_source": "No input source has a scenario {scenario}. The scenarios include: {scenarios}.",
    "unknown_compare_alternative": ("Neither scenario has the alternative {value}. Their alternatives include: "
                                    "{values}."),
    "choose_scenario": "The input source {source} has {n} scenarios; name one of them: {scenarios}.",
    # Input data
    "unknown_class": "There is no entity class {entity_class}.{similar}",
    "unknown_parameter": "The entity class {entity_class} has no parameter {parameter}.{similar}",
    "unknown_entity": "There is no {entity_class} named {entity} in {source}.{similar}",
    "entity_not_found": "No entity named {entity} exists in {source} in any entity class.{similar}",
    "entity_dimensions": ("A {entity_class} is named by {n} names ({dimensions}); give the entity as a list of "
                          "{n} names."),
    "older_database": ("This database is {relation} than this FlexTool; the value may be stored under another "
                       "parameter name. Parameters stored for this entity: {names}."),
    # {consequence}: INACTIVE_CONSEQUENCE of the entity's class, or "" where results are shown beside the note.
    "inactive_entity": ("{entity} is not part of scenario {scenario} ({reason}). Its stored values are not used "
                        "there{consequence}."),
    "no_entity_matches": ("No {entity_class} matches the filters given ({what}). The class has {n} entities in the "
                          "input data; leave the filters out to list them."),
    "none_active": ("No {entity_class} listed here is part of scenario {scenario}: inactive_in_scenario lists them. "
                    "Their stored values are not used in this scenario."),
    "no_value": "No alternative of the scenario sets this value and the parameter has no default.",
    "no_parameter_named": ("Name the parameter to read (parameter), or several (parameters). get_entity lists every "
                           "value an entity has."),
    "too_many_names": ("One get_parameter call reads at most {max_parameters} parameters and {max_entities} "
                       "entities; this one names {parameters} parameters and {entities} entities. Split it into "
                       "smaller calls, which can go in the same step."),
    "bad_time_range": "time_range takes {{\"from\": first index, \"to\": last index}}, for example {example}.",
    "bad_input_time_step": ("time_range: the value has no index {value} at that level. Its index runs from {first} "
                            "to {last}."),
    # Runs
    "no_runs": "The project has no scenario results (no folders in output_parquet).",
    "bad_run_name": "{run} is not a result folder name; use a folder name that list_runs gives.",
    "unknown_run": "There is no result folder {run}. The folders include: {runs}.",
    "no_run_matches": "No result folder matches {run}. The folders include: {runs}.",
    "overview_one_run": ("detail overview needs one result folder: folders with one name, or scenario alone (its "
                         "newest run). The latest folders are: {runs}."),
    "overview_other_scenario": ("The result folder {run} is a run of scenario {value}, not of {scenario}. Leave "
                                "scenario out for its overview, or name a result folder of {scenario}."),
    "overview_not_of_scenario": ("The result folder {run} is not a run of scenario {scenario}. Leave scenario out "
                                 "for its overview, or name a result folder of {scenario}."),
    "overview_no_scenario_run": ("No result folder is a run of scenario {scenario}. The scenarios with result "
                                 "folders include: {scenarios}."),
    "overview_filtered_out": ("No result folder passes every filter given. " + _HIDDEN_BY_FILTERS + " Leave out "
                              "source, state or results_usable for the overview, or list_runs with the same "
                              "filters to see what passes."),
    "hidden_by_filters": _HIDDEN_BY_FILTERS + " Leave a filter out to see them.",
    "unknown_run_state": "There is no run state {value}. The states of this project's runs are: {values}.",
    "run_running": "Scenario folder {run} is running now; its results are not complete.",
    "run_abandoned": ("The latest run of {run} stopped without finishing. Read the run log (get_run_log) and run "
                      "it again."),
    "run_failed": ("The latest run of {run} failed with the message {error}. Result files left in the folder "
                   "belong to an earlier run. Read the run log (get_run_log) and get the scenario to run successfully first."),
    "run_no_parquet": ("The latest run of {run} wrote no result files to read (its write methods: {methods}). Run "
                       "it with parquet output."),
    "run_unreadable": ("The run record of {run} could not be read right now (it may be being written). Ask again in "
                       "a moment; if it stays so, read its run log (get_run_log)."),
    "run_malformed": ("The run record of {run} is damaged, so FlexTool cannot say whether its latest run succeeded. "
                      "Read its run log (get_run_log) and run the scenario again."),
    # Result tables
    "unknown_table": "There is no result table {table} in {run}.{similar}",
    "unknown_topic": "There is no topic {topic} among the result tables of {run}. The topics are: {topics}.",
    "result_damaged": ("This result file is being written or is damaged. If a run is in progress, wait for it to "
                       "finish."),
    "unknown_level": "The table {table} has no level {level} (in {what}). Its levels are: {levels}.",
    "no_match": "No value of the level {level} of {table} matches the filter. Its values include: {values}.",
    "no_match_combination": "No column of {table} matches all the filters together.",
    "no_time_steps": "{what} needs a table with time steps; {table} has none.",
    "bad_time_step": ("time_range: the table {table} has no time step {value}. Its time steps run from {first} "
                      "to {last}."),
    "budget": "This would return about {n} values. Add filters, group_by or top_n.",
    "bad_sort": "sort.by must be one of the result's columns: {columns} (not {by}).",
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
                                    "give a total over the horizon. Compare periods side by side.{use}"),
    "level_name_clash": ("The table {table} uses the name {level} for a row level and a column level, so it "
                         "cannot be queried."),
    "combine_quantities": ("The level {level} of {table} holds different quantities ({values}); they cannot be "
                           "added up or combined. Keep {level} in group_by, or filter it to one value; or keep the "
                           "level and add the values you need yourself."),
    "mean_of_parts": ("The {level} values of {table} are parts of one total; their mean is not a meaningful "
                      "figure. Use sum, or keep {level} in group_by."),
    "sum_over_groups": ("Groups of {table} can share members, so adding them up may count a member more than "
                        "once. Keep {level} in group_by or filter it to one group; for a system total use a table "
                        "by unit, node or process."),
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
    "table_in_no_run": "None of the runs {runs} has the result table {table}.",
    # One entity of a run (explain_entity)
    "explain_run_or_scenario": ("explain_entity needs a result folder (run, from list_runs) or a scenario (its "
                                "newest run). The latest folders are: {runs}."),
    "explain_other_scenario": ("The result folder {run} is a run of scenario {value}, not of {scenario}. Leave "
                               "scenario out, or name a result folder of {scenario}."),
    "explain_other_source": ("The result folder {run} is not from the input source {source}. Leave source out, or "
                             "name a result folder of that source."),
    "explain_no_source_run": ("No result folder of scenario {scenario} is from the input source {source}. The "
                              "scenario's result folders are: {runs}."),
    "explain_not_explained": ("{entity} is not a unit, connection or node in {source} (it is in {names}). "
                              "explain_entity explains units, connections and nodes; get_entity shows any entity."),
    "explain_several_classes": ("{entity} names entities of several classes ({names}) in {source}. Give "
                                "entity_class, one of them."),
    "explain_not_in_results": ("The run {run} has no results of {entity}, and its input data cannot be read to say "
                               "what it is. Check the name with list_entities, or name entity_class."),
    # Run logs and slacks
    "no_run_record": ("This run has no run record (it ran before FlexTool kept run logs, or with an older "
                      "FlexTool). Run the scenario again to get one."),
    "unknown_solve": "The run {run} has no solve named {solve}. Its solves include: {solves}.",
    "needs_migration": ("This database is at version {db}; this FlexTool uses version {tool}. Some values may be "
                        "stored under older parameter names. Open the project in the FlexTool window to update it."),
    "newer_database": ("This database is at version {db}, newer than this FlexTool (version {tool}). Update "
                       "FlexTool to read all its parameters."),
}


#: What an inactive entity's stored values not being used means, by entity class (``inactive_entity``).
INACTIVE_CONSEQUENCE = ": it has no capacity, flows or results in this scenario"
INACTIVE_GROUP_CONSEQUENCE = ": its limits do not apply and it has no results in this scenario"


def inactive_consequence(entity_class: str) -> str:
    """The ``consequence`` field of ``inactive_entity`` for an entity of *entity_class*."""
    return INACTIVE_GROUP_CONSEQUENCE if entity_class == "group" else INACTIVE_CONSEQUENCE


#: The classes of the message fields (:func:`render`); every placeholder of :data:`MESSAGES` is in exactly one.
#: A name from the user's files (or the model's request): quoted, one line, redacted, cut to NAME_MAX.
NAMES_QUOTED = frozenset({"name", "doc", "section", "scenario", "source", "entity_class", "parameter", "entity",
                          "table", "level", "value", "first", "last", "run", "solve", "by", "topic", "path"})
#: Several names (a :class:`NameList`, or a list): each name quoted.
LISTS = frozenset({"names", "scenarios", "sources", "values", "similar", "solves", "runs", "topics", "docs",
                   "sections", "columns", "dimensions", "units", "levels", "methods"})
#: A failure message from outside FlexTool (a database driver, a run): quoted as data, cut to LOG_LINE_MAX.
MESSAGES_QUOTED = frozenset({"error", "reason"})
#: A sentence FlexTool builds that may hold the request's words (the schema check, a broken JSON start): one line,
#: redacted, cut to LOG_LINE_MAX, not quoted.
FREE_TEXT = frozenset({"problem", "start"})
#: FlexTool's own words and numbers: cut to NAME_MAX.
PLAIN = frozenset({"what", "span", "relation", "use", "type", "example", "n", "db", "tool", "max_parameters",
                   "max_entities", "parameters", "entities", "usable", "consequence"})
_CLASSES = (NAMES_QUOTED, LISTS, MESSAGES_QUOTED, FREE_TEXT, PLAIN)

Redact = Callable[[str], str]


def _same(text: str) -> str:
    return text


def one_line(value: object) -> str:
    """*value* as text without invisible characters, on one line (every run of white space one space)."""
    return " ".join(clean(str(value)).split())


def quoted(value: object, redact: Redact = _same, limit: int = NAME_MAX) -> str:
    """*value* as a quoted name: cleaned, on one line (a name cannot start a line), redacted, cut to *limit*, then
    JSON-quoted (an inner ``"`` becomes ``\\"``), so the name can never close its quotes.  Redaction comes before
    the quoting: the quoting doubles backslashes, which a path pattern would then no longer read as the text
    meant, and before the cut: a path cut first could leave part of a folder name.

    The name is redacted inside the quotes it will have, so a name that is a path (``/srv/Acme Ltd/x``) is read as
    a quoted path, with its spaces, as a later reader of the quoted text would read it."""
    text = one_line(value)
    wrapped = redact(f'"{text}"')
    text = wrapped[1:-1] if len(wrapped) >= 2 and wrapped[0] == wrapped[-1] == '"' else redact(text)
    return json.dumps(cap_text(text, limit), ensure_ascii=False)


@dataclass(frozen=True)
class NameList:
    """Several names for a message, each quoted when rendered.

    *form*: ``list`` — ``"a", "b" and 3 more`` (at most *limit* names; ``(none)`` for none); ``join`` — every name,
    comma-separated; ``similar`` — `` <label>: "a", "b".`` (empty for none); ``did_you_mean`` —
    `` Did you mean: "a"? Other close names: "b", "c".`` (empty for none)."""

    items: tuple[str, ...]
    form: str = "list"
    limit: int | None = None
    label: str = "Similar names"

    def render(self, redact: Redact = _same) -> str:
        items = list(self.items)
        if self.form == "did_you_mean":
            if not items:
                return ""
            text = f" Did you mean: {quoted(items[0], redact)}?"
            rest = ", ".join(quoted(i, redact) for i in items[1:])
            return text + (f" Other close names: {rest}." if rest else "")
        if self.form == "similar":
            return f" {self.label}: {', '.join(quoted(i, redact) for i in items)}." if items else ""
        shown = items if self.limit is None else items[: self.limit]
        text = ", ".join(quoted(i, redact) for i in shown) if shown else "(none)"
        rest = len(items) - len(shown)
        return f"{text} and {rest} more" if rest > 0 else text

    def __str__(self) -> str:
        """The names unquoted, for a payload text (which ``fit_json`` cleans and redacts)."""
        if self.form in ("did_you_mean", "similar"):
            return self.render().replace('"', "")
        shown = list(self.items) if self.limit is None else list(self.items[: self.limit])
        text = ", ".join(shown) if shown else "(none)"
        rest = len(self.items) - len(shown)
        return f"{text} and {rest} more" if rest > 0 else text


def field_class(name: str) -> frozenset[str] | None:
    """The class of the message field *name* (one of :data:`NAMES_QUOTED` … :data:`PLAIN`), ``None`` if none."""
    return next((c for c in _CLASSES if name in c), None)


def _render_field(name: str, value: object, redact: Redact) -> str:
    if isinstance(value, NameList):
        return value.render(redact)
    cls = field_class(name)
    if cls is None:
        raise KeyError(f"the message field {name!r} has no class")
    if cls is LISTS:
        if isinstance(value, (list, tuple)):
            return NameList(tuple(str(v) for v in value), form="join").render(redact)
        return quoted(value, redact)            # one string in a list field: one quoted name (the safe default)
    if cls is NAMES_QUOTED:
        return quoted(value, redact)
    if cls is MESSAGES_QUOTED:
        return quoted(value, redact, LOG_LINE_MAX)
    if cls is FREE_TEXT:
        return cap_text(redact(one_line(value)), LOG_LINE_MAX)
    return cap_text(clean(str(value)), NAME_MAX)             # FlexTool's words: kept as written (" Use …")


def placeholders(template: str) -> list[str]:
    """The field names of a message template."""
    return [f for _, f, _, _ in Formatter().parse(template) if f]


def render(key: str, fields: dict[str, object], redact: Redact = _same) -> Rendered:
    """The text of :data:`MESSAGES` ``[key]`` with *fields* filled in by their class, redacted once (*redact*)."""
    template = MESSAGES[key]
    values = {f: _render_field(f, fields[f], redact) for f in placeholders(template)}
    return Rendered(template.format(**values))


class ToolError(Exception):
    """A request the tool cannot answer, with the message for the user (:meth:`render`)."""

    def __init__(self, key: str, **fields: object) -> None:
        self.key = key
        self.fields = fields
        super().__init__(message(key, **fields))

    def render(self, redact: Redact = _same) -> Rendered:
        """The message, redacted with *redact* (the context's redaction for a text that is sent)."""
        return render(self.key, self.fields, redact)


def message(key: str, **fields: object) -> str:
    """The text of :data:`MESSAGES` ``[key]`` with *fields* filled in, not redacted (for ``str(exc)`` and logs;
    a text that is sent uses :func:`render` with the context's redaction)."""
    return str(render(key, fields))


def docs_error(exc: Exception) -> ToolError:
    """The tool error for a :class:`~flextool.assistant.docs_store.DocsUnavailable`: its failure message quoted as
    data, or "not included"."""
    reason = getattr(exc, "reason", None)
    return ToolError("docs_build_failed", reason=reason) if reason else ToolError("docs_not_included")


def close_names(word: str, candidates: list[str], n: int = 3) -> list[str]:
    """Up to *n* names of *candidates* close to *word* (spelling, then containing it)."""
    import difflib

    close = difflib.get_close_matches(word, candidates, n=n, cutoff=0.6)
    if not close:
        lower = word.lower()
        close = [c for c in candidates if lower in c.lower()][:n]
    return close


def _name_parts(name: str) -> set[str]:
    return {p for p in re.split(r"[_\W]+", name.lower()) if p}


def closest_names(word: str, candidates: list[str], n: int = 3) -> list[str]:
    """Up to *n* names of *candidates* closest to *word*, closest first: the spelling's similarity plus the share
    of name parts in common, so ``existing_capacity`` finds ``existing`` first.  A name equal to *word* but for
    case is the only answer."""
    import difflib

    lower = word.lower()
    parts = _name_parts(word)
    scored: list[tuple[float, str]] = []
    for c in candidates:
        if c.lower() == lower:
            return [c]
        ratio = difflib.SequenceMatcher(None, lower, c.lower()).ratio()
        theirs = _name_parts(c)
        shared = len(parts & theirs) / len(parts | theirs) if parts | theirs else 0.0
        if ratio >= 0.6 or shared > 0:
            scored.append((ratio + shared, c))
    scored.sort(key=lambda sc: (-sc[0], sc[1]))
    return [c for _, c in scored[:n]]


def did_you_mean(word: str, candidates: list[str]) -> NameList:
    """`` Did you mean: "existing"? Other close names: "a", "b".`` for the closest names of *word*, else empty."""
    return NameList(tuple(closest_names(word, candidates)), form="did_you_mean")


def similar_text(word: str, candidates: list[str], *, label: str = "Similar names") -> NameList:
    """`` Similar names: "a", "b", "c".`` for up to three close matches of *word*, else empty."""
    return NameList(tuple(close_names(word, candidates)), form="similar", label=label)


def name_list(names: Iterable[str], limit: int = 20) -> NameList:
    """*names*, at most *limit* of them, with the count of the rest (``"a", "b" and 3 more``)."""
    return NameList(tuple(names), form="list", limit=limit)
