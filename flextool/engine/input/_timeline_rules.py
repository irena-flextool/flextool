"""Time-structure fallback rules shared by the CSV path and the DB-side readers.

``TimelineConfig.create_assumptive_parts`` (``_timeline.py``) fills in the parts of the time structure a model
leaves out. The names it invents are defined here so the DB-side readers that mirror those rules use the same ones.
"""
from __future__ import annotations

__all__ = ["SYNTHETIC_TIMESET"]

#: Rule 1: the timeset FlexTool creates when the model has no timeset and exactly one timeline. It covers the
#: whole timeline (Rule 2) and every period of the solve (Rule 5b).
SYNTHETIC_TIMESET = "full_timeline"
