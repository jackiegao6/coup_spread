"""Render the validated-v2.5 adaptive parameter-regime sweep."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from _plot_style import apply_style, save_outputs


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "experiments/results/validated-v2/adaptive-grid/comparison_summary.csv"
)
PROTOCOL = "validated-v2.5-adaptive-grid"
DATASETS = ["Netscience", "NetFacebookEgo"]
TRANSFERS = [0.30, 0.50, 0.70, 0.85, 0.93]
SHARES = [0.20, 0.40, 0.60, 0.80]
BUDGETS = [25, 100, 200]
BUDGET_COLORS = {25: "#4C78A8", 100: "#D18B00", 200: "#007C7A"}
BUDGET_MARKERS = {25: "o", 100: "s", 200: "^"}


def read_rows() -> list[dict[str, str]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(
        row["status"] != "REAL_EXPERIMENT"
        or row["protocol_version"] != PROTOCOL
        for row in rows
    ):
        raise ValueError("Invalid appendix parameter-sweep source")
    return rows


def matrix(
    rows: list[dict[str, str]],
    dataset: str,
    field: str,
) -> np.ndarray:
    index = {
        (
            float(row["central_redemption_share"]),
            float(row["transfer_probability"]),
        ): float(row[field])
        for row in rows
        if row["dataset"] == dataset and int(row["k"]) == 100
    }
    return np.asarray(
        [[index[(share, transfer)] for transfer in TRANSFERS] for share in SHARES]
    )


def annotated_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    norm: colors.Normalize,
    cmap: str,
    title: str,
    colorbar_label: str,
    formatter=lambda value: f"{value:+.1f}",
    flagged: np.ndarray | None = None,
    sequential: bool = False,
) -> None:
    image = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(TRANSFERS)))
    ax.set_xticklabels([f"{value:.2f}" for value in TRANSFERS])
    ax.set_yticks(range(len(SHARES)))
    ax.set_yticklabels([f"{value:.2f}" for value in SHARES])
    ax.set_xlabel("Transfer probability, $t$")
    ax.set_ylabel("Central redemption share, $r$")
    ax.set_title(title, loc="left", pad=4)
    ax.grid(False)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            normalized = norm(value)
            if sequential:
                text_color = "#202020" if normalized < 0.62 else "white"
            else:
                text_color = (
                    "white"
                    if normalized < 0.22 or normalized > 0.78
                    else "#202020"
                )
            ax.text(
                column,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=6.2,
                color=text_color,
            )
            if flagged is not None and flagged[row, column] > 0:
                ax.add_patch(
                    patches.Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#B22222",
                        linewidth=1.2,
                    )
                )
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label(colorbar_label, fontsize=7.2)
    colorbar.ax.tick_params(labelsize=6.5)


def main() -> None:
    rows = read_rows()
    apply_style()
    plt.rcParams["svg.hashsalt"] = "appendix-parameter-sweep-v2.5"
    fig, axes = plt.subplots(4, 2, figsize=(7.15, 9.3))
    gain_norm = colors.TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=5.0)
    overlap_norm = colors.TwoSlopeNorm(vmin=-8.0, vcenter=0.0, vmax=8.0)
    sample_norm = colors.Normalize(vmin=0.0, vmax=3.0)

    letters = ["a", "b"]
    for column, dataset in enumerate(DATASETS):
        annotated_heatmap(
            axes[0, column],
            matrix(rows, dataset, "relative_gain_best_pct"),
            gain_norm,
            "RdBu",
            f"({letters[column]}) Spread gain: {dataset}",
            "Gain over best baseline (%)",
        )
        annotated_heatmap(
            axes[1, column],
            matrix(rows, dataset, "mean_final_rr_samples") / 1_000_000.0,
            sample_norm,
            "YlGnBu",
            f"({chr(ord('c') + column)}) Adaptive RR budget: {dataset}",
            "Mean final samples (millions)",
            formatter=lambda value: f"{value:.2f}",
            flagged=matrix(rows, dataset, "unresolved_instability_runs"),
            sequential=True,
        )
        annotated_heatmap(
            axes[2, column],
            matrix(rows, dataset, "duplicate_reduction_pp"),
            overlap_norm,
            "RdBu",
            f"({chr(ord('e') + column)}) Duplicate reduction: {dataset}",
            "Reduction (percentage points)",
        )

        ax = axes[3, column]
        for budget in BUDGETS:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["dataset"] == dataset
                    and int(row["k"]) == budget
                    and float(row["central_redemption_share"]) == 0.60
                ),
                key=lambda row: float(row["transfer_probability"]),
            )
            x = [float(row["transfer_probability"]) for row in selected]
            y = [float(row["relative_gain_best_pct"]) for row in selected]
            error = [float(row["paired_gain_ci95_pct"]) for row in selected]
            ax.errorbar(
                x,
                y,
                yerr=error,
                color=BUDGET_COLORS[budget],
                marker=BUDGET_MARKERS[budget],
                markerfacecolor="white",
                capsize=2,
                label=f"$k={budget}$",
            )
        ax.axhline(0.0, color="#5F6368", linewidth=0.8, linestyle="--")
        ax.set_xticks(TRANSFERS)
        ax.set_xticklabels([f"{value:.2f}" for value in TRANSFERS])
        ax.set_xlabel("Transfer probability, $t$")
        ax.set_ylabel("Gain over best baseline (%)")
        ax.set_title(
            f"({chr(ord('g') + column)}) Budget interaction: {dataset}",
            loc="left",
            pad=4,
        )
        ax.legend(frameon=False, ncol=3, fontsize=6.5)

    fig.subplots_adjust(
        top=0.985,
        bottom=0.065,
        left=0.09,
        right=0.985,
        hspace=0.52,
        wspace=0.34,
    )
    save_outputs(
        fig,
        str(Path(__file__)),
        (
            "Complete adaptive-sampling transfer-by-redemption-share sweep, "
            "sampling budget, duplicate-redemption comparison, and budget "
            "interaction. Red cell outlines mark unresolved stability runs."
        ),
    )
    plt.close(fig)
    svg_path = Path(__file__).resolve().with_suffix(".svg")
    svg = svg_path.read_text(encoding="utf-8")
    svg = re.sub(
        r"<dc:date>.*?</dc:date>",
        "<dc:date>2026-07-16</dc:date>",
        svg,
    )
    svg_path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
