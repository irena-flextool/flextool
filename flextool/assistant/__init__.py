"""Read-only tools that let an AI assistant answer questions about a FlexTool project.

Nothing is imported here: the connection server must start before heavy packages load, and each tool module
imports what it needs when it is first used.
"""

#: The batch of the assistant's tool behaviour (what the tools return and how a question goes on), bumped by each
#: batch that changes it without changing the tool descriptions or the instructions (so without changing their
#: hashes): the evaluation runner records it with each cell (tools/assistant_eval.py ``setting_key``) and does not
#: continue a cell checkpointed under another one.  b1: assistant tools batch B1 (2026-11-12); b2: the
#: owner-approved level texts of 2026-11-12 (how answers are written; the instructions' hash changes with them);
#: b3: the model facts (2026-11-12): the context block and get_gui_context give the scenario's solves, time steps,
#: periods and latest run, and the units note no longer says time steps are in hours.
ASSISTANT_BEHAVIOUR = "b3"
