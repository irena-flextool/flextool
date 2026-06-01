#!/usr/bin/env python3
"""Render a memory-shape PNG from a flextool memory-sampler log.

Usage:
    plot_mem_shape.py <mem_log_path> <png_path>

Plots priv_dirty (live set, red filled area) and rss (blue line) against
wall time. Legend reports the lower (min) and upper (peak) value of each
series; no on-canvas text annotations.
"""
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_FIELD = re.compile(r"(\w+)=([-+0-9.eE]+)")


def parse(path):
    t, rss, priv = [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ts="):
                continue
            f = dict(_FIELD.findall(line))
            try:
                t.append(float(f["mono_s"]))
                rss.append(float(f["rss_gb"]))
                priv.append(float(f["priv_dirty_gb"]))
            except (KeyError, ValueError):
                continue
    return t, rss, priv


def main():
    if len(sys.argv) != 3:
        sys.exit(f"usage: {sys.argv[0]} <mem_log_path> <png_path>")
    mem_path, png_path = sys.argv[1], sys.argv[2]

    t, rss, priv = parse(mem_path)
    if not t:
        sys.exit(f"no samples parsed from {mem_path}")

    rss_lo, rss_hi = min(rss), max(rss)
    priv_lo, priv_hi = min(priv), max(priv)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(t, priv, color="red", alpha=0.12)
    ax.plot(
        t, priv, color="red", linewidth=1.0,
        label=f"priv_dirty (live set)  {priv_lo:.1f}–{priv_hi:.1f} GB",
    )
    ax.plot(
        t, rss, color="tab:blue", linewidth=1.0,
        label=f"rss  {rss_lo:.1f}–{rss_hi:.1f} GB",
    )

    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0, max(rss_hi, priv_hi) * 1.15)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("GB")
    ax.margins(x=0)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    print(f"wrote {png_path}  ({len(t)} samples, "
          f"priv_dirty {priv_lo:.1f}–{priv_hi:.1f} GB, "
          f"rss {rss_lo:.1f}–{rss_hi:.1f} GB)")


if __name__ == "__main__":
    main()
