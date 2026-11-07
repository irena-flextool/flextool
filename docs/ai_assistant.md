# Connecting an AI agent to FlexTool

An AI agent, such as Claude Code, can answer questions about a FlexTool project: what a parameter is set to in a
scenario, what a result table shows, why a run failed, where slack appeared, how two scenarios differ. It does this
through a set of FlexTool tools that FlexTool starts for the agent. The tools only read. They cannot change input
data, settings or results, and they cannot start a run.

The tools give the agent:

- the input data of the project's input sources, scenario by scenario, as FlexTool would use it (the winning value
  of each parameter after the alternatives are ranked);
- the results of the scenarios that have been run (the parquet files in `output_parquet/`), with filters, sums and
  averages over the dimensions that allow them, and comparisons between scenarios;
- the run log and run status of each run, including which solves were accepted and the slack (penalty) in the
  results;
- the FlexTool documentation, including the model's equations and the list of result tables.

## Where your data goes

Everything the tools return — input data, results, run logs and documentation — is sent to the AI service that your
agent uses. FlexTool cannot say how that service stores or uses it; check its policies yourself before you connect
a project with data you may not share. FlexTool takes local folder paths, your user name and the computer's name out
of what the tools return, but it does not remove anything else: names of units, nodes and scenarios, and all
values, are sent as they are.

## What you need

- FlexTool installed (see [Install with FlexTool GUI](install_gui.md) or [Install with Spine Toolbox](install_toolbox.md)).
  The tools come with FlexTool; after an update, run "Update FlexTool" (or `pip install -e .` in a source checkout)
  so that the `flextool-assistant` command exists.
- An AI agent that can start tools on your computer through the Model Context Protocol (MCP), an open standard for
  connecting AI agents to tools. Claude Code is one; many other agents and editors support it too.
- A FlexTool project folder: the folder of a project of the FlexTool window (`projects/<project name>/`), or any
  folder with an `input_sources/` folder and the result folders FlexTool writes.

## Connect Claude Code

Give Claude Code the command that starts the tools for one project. Use the absolute path of the project folder:

```bash
claude mcp add flextool -- flextool-assistant --mcp --project-path /home/me/flextool/projects/MyProject
```

On Windows, if `flextool-assistant` is not found, give the full path of the command in the Python environment where
FlexTool is installed, for example:

```bash
claude mcp add flextool -- C:\flextool\venv\Scripts\flextool-assistant.exe --mcp --project-path C:\flextool\projects\MyProject
```

With the Spine Toolbox bundle, start the tools with the bundle's own Python (the path depends on where you unpacked
the bundle):

```bash
claude mcp add flextool -- "C:\Spine Toolbox\_internal\Python\python.exe" -m flextool_assistant_entry --mcp --project-path C:\flextool\projects\MyProject
```

Then start Claude Code and ask, for example, "What is the existing capacity of coal_plant in scenario coal?" or
"Did the latest run of scenario base succeed, and if not, why?". Claude Code starts the tools itself when it needs
them; you do not start anything by hand. `claude mcp list` shows whether the connection works.

To connect another project, add it under another name (`claude mcp add flextool-other -- …`), or remove the first
one (`claude mcp remove flextool`).

## Connect another agent

Agents that read a configuration file usually take the command and its arguments separately:

```json
{
  "mcpServers": {
    "flextool": {
      "command": "flextool-assistant",
      "args": ["--mcp", "--project-path", "/home/me/flextool/projects/MyProject"]
    }
  }
}
```

The tools talk to the agent through the command's standard input and output, so the agent must start the command
itself; there is no network port.

## Options

| Option | What it does |
|---|---|
| `--mcp` | Serve the tools to an AI agent. The agent starts the command with this option. |
| `--project-path FOLDER` | The project folder, as an absolute path. |
| `--project NAME` | The project `projects/NAME` under the folder the command starts in (the command prints the folder it uses). Use `--project-path` with agents: they may start the command in any folder. |
| `--input-source SOURCE` | The input source the tools use when a question names none: its file name or its number in the FlexTool window. Without it, the agent names a source when the project has several. |
| `--scenario NAME` | The scenario the tools use when a question names none. |
| `--result-cap small\|medium\|large` | How much one tool result may hold (rows, characters). Default: `medium`. A smaller cap saves the agent's context; the agent can still ask for more in one call, up to a fixed limit. |
| `--tier basic\|standard\|advanced` | The assistant level (see below). `basic` offers only seven tools and starts with the small result cap; `advanced` starts with the large one. |
| `--threads N` | Threads for FlexTool's own processing in the tools. Default: 1, so the tools never slow down a model that is running. |
| `--verbose` | More messages on the standard error stream (the agent usually shows them in its log). |
| `--call TOOL [ARGUMENTS]` | Run one tool from the terminal and print its result (see below). |

## Levels

The tools offer three prompts, one per assistant level. A prompt gives the agent instructions for working at that
level for the rest of the conversation:

- **basic** — look up and explain values with a few tool calls; for small models;
- **standard** — interpret results and compare scenarios;
- **advanced** — trace slack and penalties to their cause, read the model's equations, explain investment
  differences.

In Claude Code they are `/mcp__flextool__basic`, `/mcp__flextool__standard` and `/mcp__flextool__advanced`, and you
can add your question after the name. Without a prompt the agent works with the general instructions the tools
send when they connect. `--tier basic` limits the tools to the basic set, for agents that run a small model.

## Units

FlexTool does not check units. Its documented default units are MW for power, MWh for energy and CUR for the model's
currency (see [Units of measure](reference.md#units-of-measure)), and the result labels always use them. A model may
use other units consistently, for example kg/h and kg for a hydrogen node. The tools therefore word every unit as the
default convention ("MW (default convention)"), and the agent is told to ask, or to say that values are in the
model's own units, when the description of a node, carrier or parameter suggests other units. Describing such
entities (their `description` in the input data) helps the agent.

## Try a tool from the terminal

`--call` runs one tool and prints its result as the agent would see it, which helps when something does not work as
expected:

```bash
flextool-assistant --project-path /home/me/flextool/projects/MyProject --call list_runs
flextool-assistant --project-path /home/me/flextool/projects/MyProject --call get_entity '{"name": ["coal_plant"], "scenario": "coal"}'
```

The arguments are a JSON object, or `@file.json` to read them from a file. `flextool-assistant --help` lists the
options. A chat with an AI model directly in the terminal and in the FlexTool window will come in a later
release.

## When something goes wrong

- **The agent says FlexTool could not start, and gives a command.** On some older processors FlexTool needs the
  compatible build of polars, its data library. The tools then start with only one tool, which gives the reason
  and the command that installs it. Run the command (or "Update FlexTool"), then restart the agent.
- **"The connection for AI assistants could not load the package mcp"** (in the agent's log). FlexTool was updated
  without its new requirements. Run "Update FlexTool", or `pip install -e .` in a source checkout.
- **The agent says a run's results are not used.** The tools read results only from a scenario whose latest run
  succeeded; files left in the folder by an earlier run are not used. The agent can read the run log to help get
  the scenario to run.
- **A spreadsheet input source is "not converted".** The tools read a spreadsheet through the database FlexTool
  makes from it when one of its scenarios runs. Run (or convert) a scenario of it first.
