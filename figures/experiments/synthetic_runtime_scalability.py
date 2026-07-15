"""Synthetic planning layout for runtime and graph-size scaling."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_style import (
    DATASET_COLORS,
    METHOD_COLORS,
    METHOD_MARKERS,
    add_synthetic_notice,
    apply_style,
    save_outputs,
)


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "synthetic_runtime_scalability.csv"
DATASETS = ["Netscience", "NetFacebookEgo", "DoubanRandom", "EmailEnron", "network.douban"]
METHODS = ["CIM-RIS", "IC-RIS", "1Hop-Sort", "Alpha-Sort"]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA_FILE.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "nodes": int(row["nodes"]),
                    "k": int(row["k"]),
                    "method": row["method"],
                    "time": float(row["time_seconds"]),
                }
            )
    return rows


def main() -> None:
    apply_style()
    rows = load_rows()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))

    ax = axes[0]
    for dataset in DATASETS:
        selected = sorted(
            (
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == "CIM-RIS"
            ),
            key=lambda row: row["k"],
        )
        ax.plot(
            [row["k"] for row in selected],
            [row["time"] for row in selected],
            color=DATASET_COLORS[dataset],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=dataset,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Coupon budget, $k$")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("(a) CIM-RIS budget scaling", loc="left", pad=4)
    ax.legend(frameon=False, ncol=1, loc="upper left")

    ax = axes[1]
    for method in METHODS:
        selected = sorted(
            (row for row in rows if row["k"] == 100 and row["method"] == method),
            key=lambda row: row["nodes"],
        )
        ax.plot(
            [row["nodes"] for row in selected],
            [row["time"] for row in selected],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markerfacecolor="white" if method == "IC-RIS" else METHOD_COLORS[method],
            markeredgewidth=0.8,
            label=method,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Wall-clock time at $k=100$ (s)")
    ax.set_title("(b) Graph-size scaling", loc="left", pad=4)
    ax.legend(frameon=False, loc="upper left")

    fig.subplots_adjust(top=0.91, bottom=0.17, left=0.09, right=0.985, wspace=0.30)
    add_synthetic_notice(fig)
    save_outputs(fig, __file__)


if __name__ == "__main__":
    main()
