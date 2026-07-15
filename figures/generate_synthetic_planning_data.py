"""Generate deterministic planning data. Never use these values as evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(name: str, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path = DATA_DIR / name
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quality_rows() -> list[dict[str, object]]:
    datasets = {
        "Netscience": (379, 0.77, 0.43),
        "NetFacebookEgo": (2888, 0.72, 0.30),
        "DoubanRandom": (4723, 0.70, 0.27),
        "EmailEnron": (33696, 0.68, 0.20),
    }
    budgets = [10, 25, 50, 100, 150, 200]
    methods = {
        "MC-CELF": (1.000, 0.00005),
        "CIM-RIS": (0.988, 0.00006),
        "1Hop-Sort": (0.925, 0.00014),
        "Alpha-Sort": (0.885, 0.00012),
        "IC-RIS": (0.765, 0.00016),
        "Random": (0.625, 0.00010),
    }
    rows: list[dict[str, object]] = []
    for dataset, (nodes, base, collision) in datasets.items():
        for budget in budgets:
            overlap = max(0.62, 1.0 - collision * (budget / nodes) ** 0.72)
            oracle = min(float(budget), base * budget * overlap)
            for method, (ratio, decay) in methods.items():
                if dataset == "EmailEnron" and method == "MC-CELF":
                    continue
                method_ratio = max(0.45, ratio - decay * budget)
                mean = min(float(budget), oracle * method_ratio)
                ci95 = 0.05 + 0.006 * mean + 0.012 * math.sqrt(budget)
                rows.append(
                    {
                        "dataset": dataset,
                        "k": budget,
                        "method": method,
                        "mean": f"{mean:.4f}",
                        "ci95": f"{ci95:.4f}",
                        "status": "SYNTHETIC_NOT_FOR_SUBMISSION",
                    }
                )
    return rows


def runtime_rows() -> list[dict[str, object]]:
    datasets = {
        "Netscience": (379, 1828),
        "NetFacebookEgo": (2888, 5962),
        "DoubanRandom": (4723, 11774),
        "EmailEnron": (33696, 361622),
        "network.douban": (154907, 654206),
    }
    budgets = [10, 25, 50, 100, 150, 200]
    rows: list[dict[str, object]] = []
    for dataset, (nodes, edges) in datasets.items():
        graph_scale = (nodes + edges) / 100000.0
        for budget in budgets:
            values = {
                "CIM-RIS": 0.015 + graph_scale * (0.15 + 0.00022 * budget**2),
                "IC-RIS": 0.010 + 0.000010 * (nodes + edges) * (1.0 + 0.002 * budget),
                "1Hop-Sort": 0.006 + 0.0000025 * (nodes + edges),
                "Alpha-Sort": 0.003 + 0.0000015 * nodes,
            }
            for method, seconds in values.items():
                ci_ratio = 0.055 if method in {"CIM-RIS", "IC-RIS"} else 0.020
                rows.append(
                    {
                        "dataset": dataset,
                        "nodes": nodes,
                        "edges": edges,
                        "k": budget,
                        "method": method,
                        "time_seconds": f"{seconds:.6f}",
                        "ci95": f"{max(0.0005, seconds * ci_ratio):.6f}",
                        "status": "SYNTHETIC_NOT_FOR_SUBMISSION",
                    }
                )
    return rows


def importance_rows() -> list[dict[str, object]]:
    adoption_probabilities = [0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
    coupon_count = 50
    event_given_gate = 0.40
    rows: list[dict[str, object]] = []
    for adoption_probability in adoption_probabilities:
        gate_probability = 1.0 - (1.0 - adoption_probability) ** coupon_count
        event_probability = event_given_gate * gate_probability
        uniform_cv = math.sqrt((1.0 - event_probability) / event_probability)
        conditioned_cv = math.sqrt((1.0 - event_given_gate) / event_given_gate)
        for sampler, zero_fraction, cv, useful in [
            (
                "Uniform root",
                1.0 - gate_probability,
                uniform_cv,
                10000.0 * gate_probability,
            ),
            ("Conditioned root", 0.0, conditioned_cv, 10000.0),
        ]:
            rows.append(
                {
                    "p_a": f"{adoption_probability:.4f}",
                    "k": coupon_count,
                    "sampler": sampler,
                    "zero_fraction": f"{zero_fraction:.6f}",
                    "coefficient_of_variation": f"{cv:.6f}",
                    "useful_samples_per_10000": f"{useful:.2f}",
                    "status": "SYNTHETIC_NOT_FOR_SUBMISSION",
                }
            )
    return rows


def main() -> None:
    write_csv(
        "synthetic_quality_vs_budget.csv",
        ["dataset", "k", "method", "mean", "ci95", "status"],
        quality_rows(),
    )
    write_csv(
        "synthetic_runtime_scalability.csv",
        ["dataset", "nodes", "edges", "k", "method", "time_seconds", "ci95", "status"],
        runtime_rows(),
    )
    write_csv(
        "synthetic_importance_sampling.csv",
        [
            "p_a",
            "k",
            "sampler",
            "zero_fraction",
            "coefficient_of_variation",
            "useful_samples_per_10000",
            "status",
        ],
        importance_rows(),
    )


if __name__ == "__main__":
    main()
