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
