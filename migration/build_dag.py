"""Build a dependency DAG over in-scope derived sets and params.

Reads ``migration/inventory.csv`` (produced by
``migration.inventory_mathprog_derivations``), restricts to in-scope
items, and emits:

- ``migration/dag.json``  — adjacency (parent -> [children]) + reverse
- ``migration/order.txt`` — topo-sorted layered order with one item per
  line, blank lines between layers. Agents step through this list.

Layers reflect dependency depth: L0 has no in-scope dependencies,
L1 depends only on L0, etc. Within a layer, items are alphabetical for
determinism.

CLI:
    python -m migration.build_dag
        [--inventory migration/inventory.csv]
        [--dag migration/dag.json]
        [--order migration/order.txt]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path


def _is_in_scope(row: dict) -> bool:
    return (
        row["has_derivation"] == "True"
        and row["references_variables"] == "False"
        and row["already_loaded"] == "False"
    )


def build_dag(inventory_path: Path) -> tuple[dict[str, list[str]], dict[str, dict]]:
    """Return (adj parent->children, item-info dict)."""
    items: dict[str, dict] = {}
    with inventory_path.open() as fh:
        for row in csv.DictReader(fh):
            if _is_in_scope(row):
                items[row["name"]] = row

    in_scope_names = set(items)

    # Edge: parent depended-on -> child depends-on-it
    adj: dict[str, list[str]] = defaultdict(list)
    for child, info in items.items():
        refs = [r for r in info["references"].split(",") if r and r in in_scope_names]
        for parent in refs:
            adj[parent].append(child)
    return dict(adj), items


def topo_layers(items: dict[str, dict], adj: dict[str, list[str]]
               ) -> list[list[str]]:
    """Return list-of-layers, each layer is alphabetically sorted names.

    Layer L0: no in-scope dependencies.
    Layer Ln: depends only on items in L0..L(n-1).
    """
    in_scope = set(items)
    in_degree: dict[str, int] = {n: 0 for n in in_scope}
    for parent, children in adj.items():
        for c in children:
            in_degree[c] += 1

    layers: list[list[str]] = []
    placed: set[str] = set()
    remaining = set(in_scope)
    while remaining:
        layer = sorted(n for n in remaining if in_degree[n] == 0)
        if not layer:
            # Cycle — unusual but possible if derivations are mutually recursive
            # via ``setof`` chains. Surface them so we can investigate.
            raise RuntimeError(
                f"Dependency cycle detected — remaining items: "
                f"{sorted(remaining)[:10]}..."
            )
        layers.append(layer)
        placed.update(layer)
        remaining -= set(layer)
        for n in layer:
            for c in adj.get(n, []):
                if c in remaining:
                    in_degree[c] -= 1
    return layers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path,
                        default=Path("migration/inventory.csv"))
    parser.add_argument("--dag", type=Path,
                        default=Path("migration/dag.json"))
    parser.add_argument("--order", type=Path,
                        default=Path("migration/order.txt"))
    args = parser.parse_args(argv)

    adj, items = build_dag(args.inventory)
    layers = topo_layers(items, adj)

    # Reverse adjacency for quick "who depends on me?" lookups
    reverse: dict[str, list[str]] = defaultdict(list)
    for parent, children in adj.items():
        for c in children:
            reverse[c].append(parent)
    reverse = {k: sorted(v) for k, v in reverse.items()}

    args.dag.parent.mkdir(parents=True, exist_ok=True)
    args.dag.write_text(json.dumps({
        "n_items": len(items),
        "n_layers": len(layers),
        "adjacency": {k: sorted(v) for k, v in adj.items()},
        "reverse_adjacency": reverse,
        "layers": layers,
    }, indent=2) + "\n")

    with args.order.open("w") as fh:
        fh.write(f"# python-preprocessing migration order\n")
        fh.write(f"# {len(items)} items in {len(layers)} layers\n")
        fh.write(f"# Each layer's items have no dependencies on later layers.\n")
        fh.write(f"# Within a layer, items are alphabetical (any order is safe).\n\n")
        for i, layer in enumerate(layers):
            fh.write(f"## Layer L{i} ({len(layer)} items)\n")
            for name in layer:
                info = items[name]
                fh.write(f"  {info['kind']:5s}  {name:45s}  "
                         f"L{info['line']:>5s}  {info['complexity']}\n")
            fh.write("\n")

    print(f"Wrote {args.dag} and {args.order}")
    print(f"  {len(items)} in-scope items in {len(layers)} layers")
    for i, layer in enumerate(layers):
        kind_breakdown = {}
        for n in layer:
            k = items[n]["kind"]
            kind_breakdown[k] = kind_breakdown.get(k, 0) + 1
        breakdown = ", ".join(f"{n} {k}{'s' if n != 1 else ''}"
                              for k, n in sorted(kind_breakdown.items()))
        print(f"  L{i}: {len(layer):3d} items  ({breakdown})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
