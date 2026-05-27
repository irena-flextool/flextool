# Choosing an interface

FlexTool offers **two parallel ways to orchestrate runs** and **two parallel ways to edit data**.
This page helps you pick the right combination — most users settle in under two minutes by
scanning the table below.

The two axes are orthogonal: any orchestration GUI can be paired with any data editor, and a
terminal workflow can be layered on top of either combination for batches and automation.

## Pick by use case

| You want to… | Recommended stack |
| --- | --- |
| Get started quickly / build a first model | **FlexTool GUI** + **Spine database editor** |
| Keep editing data in spreadsheets | **FlexTool GUI** + **Excel / LibreOffice** |
| Build multi-tool DAG workflows (FlexTool combined with other models) | **Spine Toolbox** + **Spine database editor** |
| Run batches on a server or in CI | **Terminal CLI** (data editor irrelevant once the input DB is built) |
| You're an experienced FlexTool v3 user / already use Spine Toolbox | **Spine Toolbox** + **Spine database editor** |

The rest of this page explains each choice and links to the installation and usage guides.

## Execution / orchestration GUI

### FlexTool GUI

A standalone Tkinter application launched with `python -m flextool.gui`. It is the recommended
starting point for new users and for anyone whose workflow centres on FlexTool itself.

What it gives you:

- Project management with one or more input sources (`.sqlite` or `.xlsx`).
- Double-click an `.sqlite` source to open it in the Spine database editor.
- Scenario selection and an execution queue with a memory watchdog (per-job memory budget,
  system-memory floor, automatic kill of the most-over-budget job under pressure).
- Auto-generated outputs after each run: scenario plots, scenario Excels, scenario CSVs,
  comparison plots, and comparison Excels — each toggleable.
- Result viewer with keyboard navigation between scenarios and plot variants, plus a dedicated
  scenario comparison mode and a network graph view.

When to choose FlexTool GUI:

- You are new to FlexTool, or your work is mostly FlexTool-centric.
- You want one place to manage inputs, run scenarios, and inspect results.
- You want the lightest installation footprint.

See [Install with FlexTool GUI](install_gui.md) and the
[FlexTool GUI reference](flextool_gui_interface.md).

### Spine Toolbox

Spine Toolbox is a separate workflow manager that runs FlexTool as one node in a directed
acyclic graph (DAG) of tools and data stores.

When to choose Spine Toolbox:

- You build **multi-tool pipelines** (custom pre-processing, FlexTool, post-processing,
  another model downstream).
- You need to integrate FlexTool with other Spine-based models.
- You already use Spine Toolbox and don't want a second orchestration GUI.
- You are more comfortable with the visual workflow than predefined options

See [Install with Spine Toolbox](install_toolbox.md) and the
[Spine Toolbox interface guide](spine_toolbox.md).

## Data editor

### Spine database editor (default)

The Spine database editor is the native editor for FlexTool's input format. It opens from
either GUI: double-click an `.sqlite` source in the FlexTool GUI, or open the data store from
inside Spine Toolbox.

It understands FlexTool's entities, alternatives, scenarios, and parameter-value structure
directly, so all relationships (e.g. a unit's input/output nodes, a node's commodity) are
validated as you edit. This is the recommended editor for anything beyond trivial changes.

See the [Spine database guide](spine_database.md).

### Excel / LibreOffice (alternative)

If you would rather not learn a new editor, you can export an Excel template from the FlexTool
GUI, edit the data in Excel or LibreOffice, and import it back. The same workflow is supported
from the terminal.

See the [Excel / spreadsheet interface](excel_interface.md).

!!! note "Mix and match freely"
    You can edit some scenarios in the Spine database editor and others in Excel within the
    same project — they round-trip through the same input database. Use whichever fits the
    edit at hand.

## Scripting / automation (terminal CLI)

For batch jobs, CI pipelines, parameter sweeps, and headless servers, drive FlexTool from the
command line. The CLI subcommands live under `python -m flextool.cli.cmd_*`, and the
repository ships convenience wrappers such as `run_flextool.py` and
`execute_flextool_workflow.py`.

When to choose the terminal CLI:

- You want repeatable, scripted runs that don't need a GUI.
- You're running on a remote server, a cluster, or in continuous integration.
- You're sweeping many scenarios or chaining FlexTool into a larger automation.

The terminal workflow is **orthogonal to the GUI axis**: you can edit data in either the
Spine database editor or Excel and still execute the same project from the terminal.

See the [terminal workflow guide](terminal_workflow.md).

## What's not an interface any more

The FlexTool **web / browser interface has been retired**. It is preserved for historical
reference in the [archived browser interface notes](archived/browser_interface.md), but it is
no longer maintained and is not a current option for running FlexTool. This is also why some
older video tutorials look quite different from the screenshots in this documentation — they
were recorded against the retired web UI.

!!! tip "Still unsure?"
    Start with **FlexTool GUI + Spine database editor**. You can always switch later — the
    underlying data and scenarios are the same regardless of which orchestrator and editor
    you use.
