"""The cost lines of the reported investment-type costs: one name per kind of event the objective charges.

Each investment-type term of the objective (an investment's price over its years in service, a retirement's
settlement, a life extension's price, the constants of existing capacity) is split into the events it is made of,
each reported in the period it happens (the lifetime-cost modules ``_run_costs``, ``_retire_settlement``,
``_existing_salvage``, ``_extend``: the line frames ``edl_*`` / ``cdl_*``; ``process_outputs.calc_costs``: their
sums).  The lines of one term add up to the term (the objective's coefficient or lump).

* ``investment`` — the capital of a plant (a_cap over its years in service inside the horizon), at its start of
  operation; ``replacement`` — the same of a later plant of a reinvest_automatic chain, and a retirement before
  that plant comes online takes its capital back here, in the same period; ``extension`` — the extension-cost
  part of a life extension's price.
* ``retirement`` — a retirement's contract settlement of the plant in service (the payments due before it and the
  loan balance at its start) less the capital its years in service were charged, in the retirement period.
* ``salvage end of life`` / ``salvage replaced`` — the salvage credit of a plant (a_salv over its years in service
  inside the horizon; negative for a positive salvage value), at its end of life (the last period of the run when
  it lives to or past the end of the horizon), for a plant whose life simply ends / one a reinvest_automatic
  replacement takes over; a retirement takes the credit of the years it no longer serves back here.
  ``salvage extended`` — the salvage value a life extension gives up (the parent's salvage at its end of life),
  financed at discount_rate over life_extension_years: its share inside the horizon, part of the extension's
  price.  ``salvage economic`` — the salvage value received at a retirement.
* ``investment pre-existing`` / ``salvage pre-existing`` — the replacement of existing reinvest_automatic capacity
  and the salvage credits of existing capacity: objective constants (counted only with
  ``include_existing_fixed_cost``).

The cost tables (:data:`TABLE_LINES`) show the four salvage lines of new capacity as one ``salvage`` line; the text
summary keeps them apart (the cause is also in the retirement-by-cause tables).
"""
from __future__ import annotations

INVESTMENT = "investment"
REPLACEMENT = "replacement"
EXTENSION = "extension"
RETIREMENT = "retirement"
SALVAGE_END_OF_LIFE = "salvage end of life"
SALVAGE_REPLACED = "salvage replaced"
SALVAGE_EXTENDED = "salvage extended"
SALVAGE_ECONOMIC = "salvage economic"
INVESTMENT_PRE_EXISTING = "investment pre-existing"
SALVAGE_PRE_EXISTING = "salvage pre-existing"

#: Every cost line, in the order the cost tables show them.
COST_LINES: tuple[str, ...] = (INVESTMENT, REPLACEMENT, EXTENSION, RETIREMENT, SALVAGE_END_OF_LIFE, SALVAGE_REPLACED,
                               SALVAGE_EXTENDED, SALVAGE_ECONOMIC, INVESTMENT_PRE_EXISTING, SALVAGE_PRE_EXISTING)
#: The lines of the objective constants of existing capacity (counted only with include_existing_fixed_cost).
PRE_EXISTING_LINES: tuple[str, ...] = (INVESTMENT_PRE_EXISTING, SALVAGE_PRE_EXISTING)
#: The salvage of new capacity, by cause (the text summary's lines), and their sum in the cost tables.
SALVAGE_CAUSES: tuple[str, ...] = (SALVAGE_END_OF_LIFE, SALVAGE_REPLACED, SALVAGE_EXTENDED, SALVAGE_ECONOMIC)
SALVAGE = "salvage"
#: The lines of the cost tables (costs_discounted, the annualized and per-entity cost tables), in their order.
TABLE_LINES: tuple[str, ...] = (INVESTMENT, REPLACEMENT, EXTENSION, RETIREMENT, SALVAGE, INVESTMENT_PRE_EXISTING,
                                SALVAGE_PRE_EXISTING)
#: The columns of a line frame of a decision (CUR per MW of the decision; Utf8 labels).
LINE_COLUMNS: tuple[str, ...] = ("e", "d", "line", "d_report", "value")
