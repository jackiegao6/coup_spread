"""Synthetic planning layout for the conditioned-root sampler ablation."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from _plot_style import METHOD_COLORS, METHOD_MARKERS, add_synthetic_notice, apply_style, save_outputs


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "synthetic_importance_sampling.csv"
SAMPLERS = ["Uniform root", "Conditioned root"]


def load_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with DATA_FILE.open(newline="", encoding="ascii") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "p_a": float(row["p_a"]),
                    "sampler": row["sampler"],
                    "zero": float(row["zero_fraction"]),
                    "cv": float(row["coefficient_of_variation"]),
                    "useful": float(row["useful_samples_per_10000"]),
                }
            )
    return rows


def main() -> None:
    apply_style()
    rows = load_rows()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75))
    fields = [
        ("zero", "Zero-contribution samples", "(a) Rejected work"),
        ("cv", "Estimator coefficient of variation", "(b) Estimator variability"),
        ("useful", "Useful samples per 10,000", "(c) Effective samples"),
    ]

    for ax, (field, ylabel, title) in zip(axes, fields):
        for sampler in SAMPLERS:
            selected = sorted(
                (row for row in rows if row["sampler"] == sampler),
                key=lambda row: row["p_a"],
            )
            ax.plot(
                [row["p_a"] for row in selected],
                [row[field] for row in selected],
                color=METHOD_COLORS[sampler],
                marker=METHOD_MARKERS[sampler],
                markerfacecolor="white" if sampler == "Uniform root" else METHOD_COLORS[sampler],
                markeredgewidth=0.8,
                label=sampler,
            )
        ax.set_xscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=4)
        ax.set_xticks([0.01, 0.02, 0.05, 0.1, 0.2, 0.4])
        ax.set_xticklabels([".01", ".02", ".05", ".1", ".2", ".4"])

    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[2].set_ylim(0, 10500)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.01))
    fig.supxlabel("Homogeneous adoption probability, $p^a$ ($k=50$)", x=0.5, y=0.115)
    fig.subplots_adjust(top=0.82, bottom=0.25, left=0.09, right=0.99, wspace=0.43)
    add_synthetic_notice(fig)
    save_outputs(fig, __file__)


if __name__ == "__main__":
    main()
