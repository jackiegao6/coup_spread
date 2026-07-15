"""Render publication figures exclusively from REAL_EXPERIMENT CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from _plot_style import METHOD_COLORS, METHOD_MARKERS, apply_style, save_outputs


ROOT = Path(__file__).resolve().parents[2]
DATASETS = ["Netscience", "NetFacebookEgo", "DoubanRandom", "EmailEnron"]
METHODS = [
    "MC-Greedy",
    "CIM-RIS",
    "1Hop-Sort",
    "Alpha-Sort",
    "IC-RIS",
    "DegreeTopM",
    "PageRank",
    "Random",
]
LINESTYLES = {
    "MC-Greedy": "--",
    "CIM-RIS": "-",
    "1Hop-Sort": "-",
    "Alpha-Sort": "-.",
    "IC-RIS": "--",
    "DegreeTopM": ":",
    "PageRank": "-.",
    "Random": ":",
}


def read_real_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(row.get("status") != "REAL_EXPERIMENT" for row in rows):
        raise ValueError(f"Non-real or empty data source: {path}")
    return rows


def plot_quality(scenario: str, source: Path, output_stem: str) -> None:
    rows = read_real_csv(source)
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.65), sharex=True)
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for ax, dataset, letter in zip(axes.flat, DATASETS, letters):
        for method in METHODS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset and row["method"] == method
                ),
                key=lambda row: int(row["k"]),
            )
            if not selected:
                continue
            x = [int(row["k"]) for row in selected]
            y = [float(row["mean_adopters"]) for row in selected]
            color = METHOD_COLORS[method]
            ax.plot(
                x,
                y,
                color=color,
                linestyle=LINESTYLES[method],
                linewidth=2.1 if method == "CIM-RIS" else 1.15,
                marker=METHOD_MARKERS[method],
                markersize=4.5 if method == "CIM-RIS" else 3.1,
                markerfacecolor=color if method == "CIM-RIS" else "white",
                markeredgewidth=0.75,
                label=method,
                zorder=8 if method == "CIM-RIS" else 3,
            )
        ax.set_title(f"{letter} {dataset}", loc="left", pad=4)
        ax.set_xlim(5, 205)
        ax.set_ylim(bottom=0)
        ax.set_xticks([10, 50, 100, 150, 200])

    for ax in axes[-1, :]:
        ax.set_xlabel("Coupon budget, $k$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Distinct adopters")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        columnspacing=1.25,
        handlelength=2.4,
    )
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.09, right=0.985, hspace=0.29, wspace=0.22)
    save_outputs(fig, str(Path(__file__).with_name(output_stem + ".py")))
    plt.close(fig)


def plot_runtime() -> None:
    balanced = read_real_csv(
        ROOT / "experiments/results/balanced/real_quality_balanced.csv"
    )
    scalability = read_real_csv(
        ROOT / "experiments/results/scalability/real_quality_balanced.csv"
    )
    rows = balanced + scalability
    datasets = DATASETS + ["network.douban"]
    colors = {
        "Netscience": "#007C7A",
        "NetFacebookEgo": "#4C78A8",
        "DoubanRandom": "#D18B00",
        "EmailEnron": "#B24C4C",
        "network.douban": "#6F7782",
    }
    apply_style()
    fig, ax = plt.subplots(figsize=(3.35, 2.55))

    for dataset in datasets:
        selected = sorted(
            (
                row
                for row in rows
                if row["dataset"] == dataset and row["method"] == "CIM-RIS"
            ),
            key=lambda row: int(row["k"]),
        )
        ax.plot(
            [int(row["k"]) for row in selected],
            [float(row["selection_seconds"]) for row in selected],
            color=colors[dataset],
            marker="o",
            markerfacecolor="white",
            markeredgewidth=0.8,
            label=dataset,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Coupon budget, $k$")
    ax.set_ylabel("Selection time (s)")
    ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
        fontsize=5.8,
        columnspacing=0.8,
        handlelength=1.6,
    )

    fig.subplots_adjust(top=0.97, bottom=0.18, left=0.18, right=0.985)
    save_outputs(fig, str(Path(__file__).with_name("real_runtime_scalability.py")))
    plt.close(fig)


def plot_sampler_ablation() -> None:
    rows = read_real_csv(
        ROOT / "experiments/results/sampler/real_sampler_ablation.csv"
    )
    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 3.7), sharex=True)
    datasets = ["Netscience", "EmailEnron"]
    colors = {"Netscience": "#4C78A8", "EmailEnron": "#B24C4C"}

    for dataset in datasets:
        uniform = sorted(
            (row for row in rows if row["dataset"] == dataset and row["sampler"] == "Uniform root"),
            key=lambda row: int(row["k"]),
        )
        axes[0].plot(
            [int(row["k"]) for row in uniform],
            [100.0 * float(row["mean_zero_fraction"]) for row in uniform],
            color=colors[dataset],
            marker="o",
            label=dataset,
        )
    axes[0].set_ylabel("Zero-contribution\nsamples (%)")
    axes[0].set_title("(a) Uniform-root waste", loc="left", pad=4)

    for dataset in datasets:
        for sampler, marker, style in [
            ("Uniform root", "s", "--"),
            ("Conditioned root", "o", "-"),
        ]:
            selected = sorted(
                (row for row in rows if row["dataset"] == dataset and row["sampler"] == sampler),
                key=lambda row: int(row["k"]),
            )
            axes[1].plot(
                [int(row["k"]) for row in selected],
                [float(row["coefficient_of_variation"]) for row in selected],
                color=colors[dataset],
                marker=marker,
                linestyle=style,
                markerfacecolor=colors[dataset] if sampler == "Conditioned root" else "white",
                label=f"{dataset}: {sampler.replace(' root', '')}",
            )
    axes[1].set_ylabel("Coefficient of\nvariation")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Estimator variability", loc="left", pad=4)

    for dataset in datasets:
        ratios = []
        budgets = []
        for budget in [10, 50, 200]:
            uniform = next(row for row in rows if row["dataset"] == dataset and row["sampler"] == "Uniform root" and int(row["k"]) == budget)
            conditioned = next(row for row in rows if row["dataset"] == dataset and row["sampler"] == "Conditioned root" and int(row["k"]) == budget)
            uniform_cost = float(uniform["std_spread_estimate"]) ** 2 * float(uniform["mean_batch_seconds"])
            conditioned_cost = float(conditioned["std_spread_estimate"]) ** 2 * float(conditioned["mean_batch_seconds"])
            budgets.append(budget)
            ratios.append(uniform_cost / conditioned_cost)
        axes[2].plot(budgets, ratios, color=colors[dataset], marker="o", label=dataset)
    axes[2].axhline(1.0, color="#777777", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Coupon budget, $k$")
    axes[2].set_xticks([10, 50, 100, 150, 200])
    axes[2].set_ylabel("Relative efficiency")
    axes[2].set_title("(c) Variance-time gain", loc="left", pad=4)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.005),
        fontsize=5.2,
        columnspacing=0.9,
        handlelength=1.7,
    )
    fig.subplots_adjust(top=0.86, bottom=0.12, left=0.23, right=0.985, hspace=0.42)
    save_outputs(fig, str(Path(__file__).with_name("real_sampler_ablation.py")))
    plt.close(fig)


def main() -> None:
    plot_quality(
        "balanced",
        ROOT / "experiments/results/balanced/real_quality_balanced.csv",
        "real_quality_balanced",
    )
    plot_quality(
        "forwarding-heavy",
        ROOT / "experiments/results/forwarding-heavy/real_quality_forwarding-heavy.csv",
        "real_quality_forwarding_heavy",
    )
    plot_runtime()
    plot_sampler_ablation()


if __name__ == "__main__":
    main()
