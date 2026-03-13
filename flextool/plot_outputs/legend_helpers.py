"""Legend sizing, positioning, and label formatting utilities."""
from __future__ import annotations


def estimate_legend_width(labels, title: str = '', base_width: float = 1.5) -> float:
    """Estimate the width in inches needed for a legend based on label content.

    Args:
        labels: List of label strings
        title: Legend title string
        base_width: Minimum width in inches (default: 1.5)

    Returns:
        Estimated width in inches
    """
    if not labels:
        return base_width

    max_label_len = max(len(str(label)) for label in labels)
    title_len = len(str(title)) if title else 0

    # ~0.09 inches per character + base padding (accounts for typical 8-10pt font)
    char_width = 0.07
    label_width = max_label_len * char_width
    title_width = title_len * char_width

    estimated_width = max(label_width, title_width, base_width) + 0.8
    return estimated_width


def _format_legend_labels(items: list) -> list[str]:
    """Convert legend items (possibly tuples/lists) to display strings.

    Tuples and lists are joined with ' | '; all other items are str()-converted.
    """
    result = []
    for item in items:
        if isinstance(item, (tuple, list)):
            result.append(' | '.join(str(v) for v in item))
        else:
            result.append(str(item))
    return result


def _should_show_legend(
    legend_position: str,
    sub_levels: list,
    idx: int,
    n_cols: int,
    n_subs: int,
) -> bool:
    """Return True if the legend should be shown on subplot idx.

    'all'   → show on every subplot.
    'right' → show only on the rightmost column or the last subplot.
    """
    if legend_position == 'all':
        return True
    if legend_position == 'right':
        if not sub_levels:
            return True
        col = idx % n_cols
        if col == n_cols - 1 or idx == n_subs - 1:
            return True
    return False
