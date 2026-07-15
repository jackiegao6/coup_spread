"""Correctness checks for the coupon forward and reverse processes."""

from __future__ import annotations

import itertools
import random
import unittest

import numpy as np

from run_real_submission import (
    Graph,
    cim_ris_seeds,
    conditioned_gates,
    reverse_coupon_set,
    single_coupon_adopter,
)


ADOPT = -2
DISCARD = -1


def make_graph(neighbors: list[list[int]], name: str = "tiny") -> Graph:
    indices = np.asarray([v for row in neighbors for v in row], dtype=np.int32)
    indptr = np.zeros(len(neighbors) + 1, dtype=np.int64)
    for i, row in enumerate(neighbors, start=1):
        indptr[i] = indptr[i - 1] + len(row)
    in_neighbors = [[] for _ in neighbors]
    for source, row in enumerate(neighbors):
        for target in row:
            in_neighbors[target].append(source)
    return Graph(
        name=name,
        n=len(neighbors),
        m=len(indices),
        indices=indices,
        indptr=indptr,
        in_neighbors=in_neighbors,
        degrees=np.asarray([len(row) for row in neighbors], dtype=np.int32),
    )


def exact_adoption_matrix(
    graph: Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
) -> np.ndarray:
    """Enumerate all deterministic node-action realizations on a tiny graph."""
    node_options: list[list[tuple[int, float]]] = []
    for node in range(graph.n):
        options = [(ADOPT, float(alpha[node])), (DISCARD, float(discard[node]))]
        neighbors = graph.out_neighbors(node)
        transfer = 1.0 - float(alpha[node]) - float(discard[node])
        if len(neighbors):
            options.extend((int(v), transfer / len(neighbors)) for v in neighbors)
        elif transfer > 0:
            options.append((DISCARD, transfer))
        node_options.append([(action, p) for action, p in options if p > 0])

    q = np.zeros((graph.n, graph.n), dtype=np.float64)
    for world in itertools.product(*node_options):
        probability = float(np.prod([entry[1] for entry in world]))
        actions = [entry[0] for entry in world]
        for start in range(graph.n):
            current = start
            visited: set[int] = set()
            while current not in visited:
                visited.add(current)
                action = actions[current]
                if action == ADOPT:
                    q[start, current] += probability
                    break
                if action == DISCARD:
                    break
                current = action
    return q


def exact_spread(q: np.ndarray, seeds: list[int]) -> float:
    survival = np.ones(q.shape[1], dtype=np.float64)
    for seed in seeds:
        survival *= 1.0 - q[seed]
    return float(np.sum(1.0 - survival))


def rr_spread_estimate(
    graph: Graph,
    alpha: np.ndarray,
    discard: np.ndarray,
    seeds: list[int],
    samples: int,
    seed: int,
) -> float:
    rng = random.Random(seed)
    k = len(seeds)
    weights = 1.0 - np.power(1.0 - alpha, k)
    total_weight = float(np.sum(weights))
    roots = rng.choices(range(graph.n), weights=weights, k=samples)
    covered = 0
    for root in roots:
        sample_rng = random.Random(rng.getrandbits(64))
        gates = conditioned_gates(float(alpha[root]), k, sample_rng)
        for coupon_index, gate in enumerate(gates):
            if gate and seeds[coupon_index] in reverse_coupon_set(
                graph, root, alpha, discard, sample_rng
            ):
                covered += 1
                break
    return total_weight * covered / samples


class CouponCoreTests(unittest.TestCase):
    def test_forward_simulation_matches_exact_realization_enumeration(self) -> None:
        graph = make_graph([[1, 2], [2], [0]])
        alpha = np.asarray([0.2, 0.4, 0.6])
        discard = np.asarray([0.3, 0.2, 0.1])
        exact = exact_adoption_matrix(graph, alpha, discard)
        simulations = 100_000
        for start in range(graph.n):
            rng = random.Random(8000 + start)
            counts = np.zeros(graph.n, dtype=np.float64)
            for _ in range(simulations):
                adopter = single_coupon_adopter(graph, start, alpha, discard, rng)
                if adopter >= 0:
                    counts[adopter] += 1
            empirical = counts / simulations
            np.testing.assert_allclose(empirical, exact[start], atol=0.007)

    def test_conditioned_joint_rr_estimator_matches_exact_spread(self) -> None:
        graph = make_graph([[1, 2], [2], [0]])
        alpha = np.asarray([0.2, 0.4, 0.6])
        discard = np.asarray([0.3, 0.2, 0.1])
        q = exact_adoption_matrix(graph, alpha, discard)
        seeds = [0, 1]
        estimate = rr_spread_estimate(
            graph, alpha, discard, seeds, samples=150_000, seed=9917
        )
        self.assertAlmostEqual(estimate, exact_spread(q, seeds), delta=0.02)

    def test_cim_ris_recovers_clear_distinct_seed_optimum(self) -> None:
        graph = make_graph([[], [], []], name="isolates")
        alpha = np.asarray([0.9, 0.6, 0.2])
        discard = 1.0 - alpha
        seeds, _, _, _ = cim_ris_seeds(
            graph, alpha, discard, k=2, samples=20_000, seed=125
        )
        self.assertEqual(seeds, [0, 1])


if __name__ == "__main__":
    unittest.main()
