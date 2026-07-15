"""Synthetic planning layout for adoption spread versus budget."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_style import METHOD_COLORS, METHOD_MARKERS, add_synthetic_notice, apply_style, save_outputs


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "synthetic_quality_vs_budget.csv"
DATASETS = ["Netscience", "NetFacebookEgo", "DoubanRandom", "EmailEnron"]
METHODS = ["MC-CELF", "CIM-RIS", "1Hop-Sort", "Alpha-Sort", "IMM", "Random"]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA_FILE.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "dataset": row["dataset"],
                    "k": int(row["k"]),
                    "method": row["method"],
                    "mean": float(row["mean"]),
                    "ci95": float(row["ci95"]),
                }
            )
    return rows


def main() -> None:
    apply_style()
    rows = load_rows()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), sharex=True, sharey=True)
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for ax, dataset, letter in zip(axes.flat, DATASETS, letters):
        for method in METHODS:
            selected = sorted(
                (row for row in rows if row["dataset"] == dataset and row["method"] == method),
                key=lambda row: row["k"],
            )
            if not selected:
                continue
            x = [row["k"] for row in selected]
            y = [row["mean"] for row in selected]
            ci = [row["ci95"] for row in selected]
            ax.plot(
                x,
                y,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                markerfacecolor="white" if method in {"MC-CELF", "IMM"} else METHOD_COLORS[method],
                markeredgewidth=0.8,
                label=method,
            )
            if method in {"MC-CELF", "CIM-RIS"}:
                ax.fill_between(
                    x,
                    [value - width for value, width in zip(y, ci)],
                    [value + width for value, width in zip(y, ci)],
                    color=METHOD_COLORS[method],
                    alpha=0.09,
                    linewidth=0,
                )
        ax.set_title(f"{letter} {dataset}", loc="left", pad=4)
        ax.set_xlim(5, 205)
        ax.set_ylim(0, 150)
        ax.set_xticks([10, 50, 100, 150, 200])

    for ax in axes[-1, :]:
        ax.set_xlabel("Coupon budget, $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Distinct adopters")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.995))
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.09, right=0.985, hspace=0.28, wspace=0.18)
    add_synthetic_notice(fig)
    save_outputs(fig, __file__)


if __name__ == "__main__":
    main()

