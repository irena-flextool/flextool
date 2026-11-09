"""What the FlexTool documentation ships: the curated Markdown files and the generated pages.

Standard library only, and no imports from ``flextool``: the build step reads this file by path (importing
``flextool`` runs its ``__init__``).  Paths are relative to the Markdown root (``docs/`` in a clone).
"""
from __future__ import annotations

# The user documentation, in reading order.
USER_DOCS: tuple[str, ...] = (
    "index.md",
    "interface_overview.md",
    "install_gui.md",
    "install_toolbox.md",
    "tutorial.md",
    "reference/index.md",
    "reference/entities_and_time.md",
    "reference/nodes.md",
    "reference/units.md",
    "reference/connections.md",
    "reference/commodities.md",
    "reference/groups.md",
    "reference/flow_groups.md",
    "reference/reserves.md",
    "reference/user_constraints.md",
    "reference/parameter_groups.md",
    "how_to/index.md",
    "how_to/basics.md",
    "how_to/storage_hydro.md",
    "how_to/thermal_chp.md",
    "how_to/investment.md",
    "how_to/solve_sequences.md",
    "how_to/stochastics.md",
    "how_to/emissions.md",
    "how_to/outputs.md",
    "how_to/performance.md",
    "results.md",
    "advanced_concepts.md",
    "flextool_gui_interface.md",
    "spine_toolbox.md",
    "spine_database.md",
    "excel_interface.md",
    "terminal_workflow.md",
    "ai_assistant.md",
    "env_vars.md",
    "solvers/index.md",
    "solvers/highs.md",
    "solvers/gurobi.md",
    "solvers/cplex.md",
    "solvers/xpress.md",
    "solvers/copt.md",
)

# Developer pages shipped whole.
DEV_DOCS: tuple[str, ...] = (
    "dev/architecture.md",
    "dev/slack_convention.md",
)

# Developer pages of which only some sections are shipped, by heading text.  The formulation page also reads
# its preamble from ``dev/engine.md`` (FORMULATION_PREAMBLE_SECTION).
ENGINE_DOC = "dev/engine.md"
ENGINE_SECTIONS: tuple[str, ...] = (
    "Solve modes",
    "Solve-to-solve handoff",
    "Flex-temporal decomposition",
    "Stochastic branches",
)
FORMULATION_PREAMBLE_SECTION = "Capacity symbols used in the headers"

# The pages generated from the installed code (flextool.docs_build.render_page).
FORMULATION_PAGE = "formulation.md"
RESULTS_CATALOGUE_PAGE = "results_catalogue.md"
GENERATED_PAGES: tuple[str, ...] = (FORMULATION_PAGE, RESULTS_CATALOGUE_PAGE)

# Every Markdown file the documentation copies from the Markdown root.
CURATED_DOCS: tuple[str, ...] = USER_DOCS + DEV_DOCS + (ENGINE_DOC,)

# The local, git-ignored list of private names (one case-insensitive regular expression per line, ``#``
# comments), at the repository root.  Present: a shipped or generated page that matches a line fails the
# build; absent: the check is skipped.
PRIVATE_NAMES_FILE = ".private_names"
