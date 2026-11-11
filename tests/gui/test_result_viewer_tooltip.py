"""Result-viewer hover tooltip: every temporal variant gets its own line
with the variant-adjusted unit + semantics.

Guards the bug where the ``a`` (sum_periods → total) and ``w`` (chunks →
weekly) variants were dropped from the tooltip — they share their base
``result_key`` with the ``p``/``h`` variants, so the old dedup-by-result_key
collapsed them away, and even un-collapsed they showed the unadjusted base
unit/semantics.

``ResultViewer._entry_tooltip_text`` is a pure method (it reads only the
passed ``PlotEntry`` and imports the metadata helper lazily — no ``self``
state, no Tk).  So this test binds it to a bare object and never creates a
Tk root; nothing here needs a display / xvfb.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flextool.gui.result_viewer import ResultViewer


@dataclass
class _Variant:
    letter: str
    result_key: str
    sub_config: str = "default"
    full_name: str = ""


@dataclass
class _Entry:
    number: str
    full_name: str
    variants: list = field(default_factory=list)


def _tooltip(entry: _Entry) -> str:
    """Call the real method body without constructing a Tk-bearing viewer."""
    return ResultViewer._entry_tooltip_text(object(), entry)


def test_tooltip_p_and_a_share_key_both_get_adjusted_lines():
    # p (base) and a (sum_periods → total) on the SAME annual-rate key
    entry = _Entry(
        number="1.0", full_name="Annualized costs",
        variants=[
            _Variant("p", "annualized_costs_d_p", "default"),
            _Variant("a", "annualized_costs_d_p", "sum_periods"),
        ],
    )
    text = _tooltip(entry)
    lines = text.splitlines()
    assert lines[0] == "1.0 Annualized costs"
    # both variants present — a no longer dropped by dedup
    assert any(line.startswith("   [p]") for line in lines)
    assert any(line.startswith("   [a]") for line in lines)
    # p keeps the annual-rate unit + semantics; a is the stripped total
    p_line = next(line for line in lines if line.startswith("   [p]"))
    a_line = next(line for line in lines if line.startswith("   [a]"))
    assert "M CUR/a" in p_line and "annualized" in p_line
    assert "M CUR " in a_line and "/a" not in a_line and "total" in a_line


def test_tooltip_h_and_w_share_key_weekly_keeps_unit():
    entry = _Entry(
        number="2.0", full_name="Node output",
        variants=[
            _Variant("h", "unit_outputNode_dt_ee", "lines"),
            _Variant("w", "unit_outputNode_dt_ee", "chunks"),
        ],
    )
    lines = _tooltip(entry).splitlines()
    h_line = next(line for line in lines if line.startswith("   [h]"))
    w_line = next(line for line in lines if line.startswith("   [w]"))
    assert "MW" in h_line and "instantaneous" in h_line
    # weekly keeps the base unit, only the semantics label changes
    assert "MW" in w_line and "weekly" in w_line


class _Mode:
    def __init__(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


def test_tooltip_and_dispatch_follow_the_node_units(tmp_path):
    """unit_of_measure: with a _unit_context.json in the selected scenario's folder the hover shows the units the
    run wrote into the parquet footer (m3/h for a water node), and the dispatch y-axis takes the node's unit; a
    folder without it keeps the static MW."""
    import pandas as pd

    from flextool.lean_parquet import write_lean_parquet
    from flextool.plot_outputs.unit_labels import dispatch_ylabel
    from flextool.process_outputs._output_meta import UnitContext, derive_column_meta, write_unit_context

    folder = tmp_path / "output_parquet" / "S"
    folder.mkdir(parents=True)
    ctx = UnitContext(node_units={"water": "m3"}, group_nodes={"both": ("elec", "water")})
    cols = pd.MultiIndex.from_tuples([("desal", "water")], names=["unit", "node"])
    df = pd.DataFrame([[1.0]], columns=cols,
                      index=pd.MultiIndex.from_tuples([("p1", "t1")], names=["period", "time"]))
    meta = {k: v.as_dict() for k, v in derive_column_meta("unit_outputNode_dt_ee", df.columns, ctx=ctx).items()}
    write_lean_parquet(pd.concat({"S": df}, axis=1, names=["scenario"]), folder / "unit_outputNode_dt_ee.parquet",
                       column_metadata=meta)
    write_unit_context(str(folder), ctx)

    rv = ResultViewer.__new__(ResultViewer)
    rv._project_path = tmp_path
    rv._mode = _Mode("single")
    rv._get_selected_scenarios = lambda: ["S"]
    rv._unit_ctx_cache = {}
    entry = _Entry(number="2.0", full_name="Unit output", variants=[_Variant("h", "unit_outputNode_dt_ee")])
    h_line = next(line for line in ResultViewer._entry_tooltip_text(rv, entry).splitlines()
                  if line.startswith("   [h]"))
    assert "m3/h" in h_line and "instantaneous" in h_line
    rv._mode = _Mode("comparison")
    assert "m3/h" not in ResultViewer._entry_tooltip_text(rv, entry)

    from flextool.gui.result_viewer import _scenario_unit_context

    loaded = _scenario_unit_context(rv, "S")
    assert loaded.node_units == {"water": "m3"}
    assert dispatch_ylabel(loaded, node="water") == "m3/h"
    assert dispatch_ylabel(loaded, group="both") == "Mixed units of measure!"
    assert dispatch_ylabel(_scenario_unit_context(rv, "missing"), group="both") == "MWh/h"
