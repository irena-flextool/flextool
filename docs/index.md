![IRENA FlexTool logo](./img/irena_flextool_logo.png)

IRENA FlexTool is an energy systems optimisation model for power and energy systems with high shares of wind and solar power, designed to find cost-effective sources of flexibility across the whole system. It performs multi-year capacity expansion as well as unit commitment and economic dispatch in a user-defined sequence of solves. The aim has been to make it fast to learn and easy to use while covering the time scales relevant for investment planning and operational scheduling. The data structure allows you to build scenarios up from alternative datasets (no data repetition, think of base data + easily combinable variations - or multiple base data sets if you want).

# Getting started

New to FlexTool? Pick the row that matches what you want to do next — each link points to the relevant page in this site.

| I want to… | Start here |
| --- | --- |
| Install FlexTool and run my first model | [Install with FlexTool GUI](install_gui.md) then [Tutorial](tutorial.md) |
| Compare my model needs to the available interfaces | [Choosing an interface](interface_overview.md) |
| Look up a parameter or output | [Model parameters](reference.md) · [Model outputs](results.md) |
| Find an example for a specific feature | [How-to recipes](how_to.md) |
| Understand the solver internals or contribute | [Developer guide → Architecture overview](dev/architecture.md) |

# Two interfaces

The **FlexTool GUI** is the default, recommended front-end — a standalone Tkinter application launched with `python -m flextool.gui` that walks you through building, solving and inspecting a model ([install](install_gui.md), [interface guide](flextool_gui_interface.md)). Can use both Excel/LibreOffice input files (FlexTool specific template) or Spine DB input files (FlexTool specific, has all the entity classes and parameters baked in).

The **Spine Toolbox workflow** is the alternative for advanced users who combine FlexTool with other models or build custom input/output processing pipelines ([install](install_toolbox.md), [workflow guide](spine_toolbox.md)).

**Excel / LibreOffice** spreadsheets are supported for spreadsheet-driven data entry and bulk edits alongside either interface ([Excel interface](excel_interface.md)).

# Pure-Python optimisation core

FlexTool's optimisation core is **pure Python**: it builds the LP/MIP matrix in [polars](https://pola.rs/) DataFrames with [polar-high](https://github.com/nodal-tools/polar-high) and solves it directly with [HiGHS](https://highs.dev/). This `engine_polars` backend has replaced the older GLPSOL-based pre-processing and is a lot (~3-10x) faster on the model-build side (the solver itself is unchanged).

# Video tutorials

!!! warning "Video tutorials may be out of date"
    Some of the video tutorials linked from this site predate the FlexTool GUI and the `engine_polars` solver backend. The core modelling concepts still apply, but the screens and CLI flags shown may differ from the current release. Treat them as supplementary; the written tutorial and how-to pages are authoritative.

# Monthly user support telcos

The monthly user support telco is held on the last Monday of each month at 12-13 UTC (skipping December and July). (Notice that time is according to UTC and in places where day-light saving time is applied, the time of the meeting may change between winter/summer) Each 1 h session starts with a ~15 min presentation on simple IRENA FlexTool demos or tutorials, followed by 45 min Q&A session.

[Teams link](https://teams.microsoft.com/l/meetup-join/19%3ameeting_MmRkYzAyNzktOTVhZS00NzAyLWI5OTItZTg4ZjhlM2I3NDc5%40thread.v2/0?context=%7b%22Tid%22%3a%2268d6b592-5008-43b5-9b04-23bec4e86cf7%22%2c%22Oid%22%3a%225138c2b5-7b5a-472e-9793-addd3b524ae7%22%7d)

Please contact anni.niemi@vtt.fi for an Outlook calendar invitation.

Recordings and presentations of the past support calls can be found from [here](https://drive.google.com/drive/folders/1cqEqCRpAEjZ24by3BjiWSxPv6cXed7Ib).

# Background for the FlexTool modelling approach

The theory slides below give some background how FlexTool is formulated. There are also examples that show some ways how FlexTool can be used (including examples from other similar models). The slides were made for training in the OASES project (funded by LEAP-RE, project no: 963530, co-funding from European Commission and national funding agencies). The files can also be found in the folder docs/theory_slides.

[1: Energy planning and types of modelling approaches](./theory_slides/Session1_Energy_planning_and_types_of_modelling_approaches.pdf)

[2: Modelling tools process and IRENA Flextool approach](./theory_slides/Session2_Modelling_tools_process_and_IRENA_FlexTool_approach.pdf)

[3: IRENA Flextool in practice](./theory_slides/Session3_IRENA_FlexTool_in_practice.pdf)

[4: Examples of studies done with IRENA Flextool approach](./theory_slides/Examples_of_studies_done_with_IRENA_FlexTool_approach.pdf)

# Reporting issues and getting help

Bug reports and feature requests go to the [GitHub issue tracker](https://github.com/irena-flextool/flextool/issues). For modelling questions or design discussion, join the [monthly user support telco](#monthly-user-support-telcos) — Anni Niemi (anni.niemi@vtt.fi) can add you to the calendar invite.
