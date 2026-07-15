"""Shared restrained style for synthetic planning figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


METHOD_COLORS = {
    "CIM-RIS": "#007C7A",
    "MC-CELF": "#252525",
    "MC-Greedy": "#252525",
    "1Hop-Sort": "#D18B00",
    "Alpha-Sort": "#4C78A8",
    "IC-RIS": "#B24C4C",
    "DegreeTopM": "#7A5195",
    "PageRank": "#4F8A5B",
    "Random": "#7A7A7A",
    "Uniform root": "#6F7782",
    "Conditioned root": "#007C7A",
}
METHOD_MARKERS = {
    "CIM-RIS": "o",
    "MC-CELF": "s",
    "MC-Greedy": "s",
    "1Hop-Sort": "^",
    "Alpha-Sort": "D",
    "IC-RIS": "v",
    "DegreeTopM": "P",
    "PageRank": "h",
    "Random": "X",
    "Uniform root": "s",
    "Conditioned root": "o",
}
DATASET_COLORS = {
    "Netscience": "#007C7A",
    "NetFacebookEgo": "#4C78A8",
    "DoubanRandom": "#D18B00",
    "EmailEnron": "#B24C4C",
    "network.douban": "#6F7782",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.45,
            "lines.markersize": 3.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D5D8DA",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "savefig.dpi": 450,
            "svg.fonttype": "none",
        }
    )


def add_synthetic_notice(fig: plt.Figure) -> None:
    fig.text(
        0.5,
        0.50,
        "SYNTHETIC",
        ha="center",
        va="center",
        color="#9C3D3D",
        fontsize=28,
        fontweight="bold",
        alpha=0.07,
        rotation=18,
        zorder=-10,
    )
    fig.text(
        0.5,
        0.012,
        "SYNTHETIC PLANNING DATA - NOT FOR SUBMISSION",
        ha="center",
        va="bottom",
        color="#8E3030",
        fontsize=7.2,
        fontweight="bold",
        bbox={"boxstyle": "square,pad=0.25", "facecolor": "#FFF4F2", "edgecolor": "#C9837A"},
    )


def save_outputs(fig: plt.Figure, script_path: str) -> None:
    output = Path(script_path).resolve().with_suffix("")
    metadata = {
        "Title": output.name,
        "Description": "SYNTHETIC PLANNING DATA - NOT FOR SUBMISSION",
    }
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", dpi=450, metadata=metadata)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", metadata=metadata)
