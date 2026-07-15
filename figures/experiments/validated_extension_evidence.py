"""Render the strong-reference, sensitivity, and capacity evidence figure."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_style import DATASET_COLORS, apply_style, save_outputs


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results/validated-v2/extensions"
SCENARIO_COLORS = {
    "balanced": "#0077BB",
    "adoption-heavy": "#009988",
    "forwarding-heavy": "#CC3311",
}
SCENARIO_LABELS = {
    "balanced": "Balanced",
    "adoption-heavy": "Adoption-heavy",
    "forwarding-heavy": "Forwarding-heavy",
}


def read_real(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(
        row["status"] != "REAL_EXPERIMENT"
        or row["protocol_version"] != "validated-v2.3-extension"
        for row in rows
    ):
        raise ValueError(f"Invalid extension source: {name}")
    return rows


def main() -> None:
    benchmark = read_real("strong_benchmark_summary.csv")
    sensitivity = read_real("sensitivity_summary.csv")
    capacity = read_real("capacity_summary.csv")
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.7))

    ax = axes[0, 0]
    for scenario in SCENARIO_COLORS:
        selected = sorted(
            (
                row
                for row in benchmark
                if row["dataset"] == "NetFacebookEgo"
                and row["scenario"] == scenario
            ),
            key=lambda row: int(row["k"]),
        )
        ax.plot(
            [int(row["k"]) for row in selected],
            [float(row["mean_gap_to_mc_greedy_percent"]) for row in selected],
            color=SCENARIO_COLORS[scenario],
            marker="o",
            markerfacecolor="white",
            label=SCENARIO_LABELS[scenario],
        )
    ax.set_title("(a) Larger-graph strong reference", loc="left", pad=4)
    ax.set_xlabel("Coupon budget, $k$")
    ax.set_ylabel("Gap to MC-Greedy (%)")
    ax.legend(frameon=False, fontsize=6.0)

    for ax, dataset, letter in [
        (axes[0, 1], "Netscience", "b"),
        (axes[1, 0], "NetFacebookEgo", "c"),
    ]:
        for budget, color, marker in [
            (10, "#0077BB", "o"),
            (50, "#EE7733", "s"),
            (200, "#009988", "^"),
        ]:
            selected = sorted(
                (
                    row
                    for row in sensitivity
                    if row["dataset"] == dataset and int(row["k"]) == budget
                ),
                key=lambda row: int(row["rr_samples"]),
            )
            ax.plot(
                [int(row["rr_samples"]) for row in selected],
                [float(row["mean_gap_to_mc_greedy_percent"]) for row in selected],
                color=color,
                marker=marker,
                markerfacecolor="white",
                label=f"$k={budget}$",
            )
        ax.set_xscale("log")
        ax.set_xticks([5_000, 20_000, 50_000, 100_000])
        ax.set_xticklabels(["5k", "20k", "50k", "100k"])
        ax.set_xlabel("Joint RR samples")
        ax.set_ylabel("Gap to MC-Greedy (%)")
        ax.set_title(f"({letter}) Sample sensitivity: {dataset}", loc="left", pad=4)
        ax.legend(frameon=False, fontsize=6.0, ncol=3)

    ax = axes[1, 1]
    policies = ["capacity-2", "unrestricted"]
    labels = ["$c_v=2$", "$c_v=k$"]
    x = np.arange(len(policies), dtype=float)
    width = 0.34
    index = {
        (row["dataset"], row["scenario"], row["capacity_policy"]): row
        for row in capacity
    }
    for offset, dataset in [(-width / 2, "Netscience"), (width / 2, "NetFacebookEgo")]:
        base = float(
            index[(dataset, "forwarding-heavy", "distinct")][
                "mean_cim_mean_adopters"
            ]
        )
        gains = []
        repeats = []
        for policy in policies:
            row = index[(dataset, "forwarding-heavy", policy)]
            gains.append(
                100.0 * (float(row["mean_cim_mean_adopters"]) - base) / base
            )
            repeats.append(float(row["mean_cim_repeated_placements"]))
        bars = ax.bar(
            x + offset,
            gains,
            width,
            color=DATASET_COLORS[dataset],
            label=dataset,
        )
        for bar, repeat in zip(bars, repeats):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.06,
                f"r={repeat:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gain over $c_v=1$ (%)")
    ax.set_title("(d) Repeated placement at $k=200$", loc="left", pad=4)
    ax.legend(frameon=False, fontsize=6.0)
    ax.set_ylim(0, 2.55)

    fig.subplots_adjust(
        top=0.97, bottom=0.11, left=0.09, right=0.99, hspace=0.38, wspace=0.28
    )
    save_outputs(
        fig,
        str(Path(__file__)),
        "Validated strong-reference, sample-sensitivity, and capacity evidence.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
