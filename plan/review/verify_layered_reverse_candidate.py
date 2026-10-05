"""Standalone checks for the single-coupon random-horizon candidate.

Uses only the Python standard library. This is a research validation
prototype, not the paper's experimental implementation or a scalability test.
Run: py -3.12 plan/review/verify_layered_reverse_candidate.py --samples 20000
"""

import argparse
import bisect
from fractions import Fraction as F
import json
import math
from pathlib import Path
import random

CONSUME, DROP = -2, -1


def categorical(entries):
    entries = [(action, probability) for action, probability in entries if probability]
    assert all(probability > 0 for _, probability in entries)
    assert sum(probability for _, probability in entries) == 1
    actions, cumulative, total = [], [], F(0)
    for action, probability in entries:
        total += probability
        actions.append(action)
        cumulative.append(float(total))
    cumulative[-1] = 1.0
    return actions, cumulative


def draw(distribution, rng):
    actions, cumulative = distribution
    return actions[bisect.bisect_right(cumulative, rng.random())]


class Model:
    def __init__(self, pa, pd, edges):
        self.pa, self.pd = list(map(F, pa)), list(map(F, pd))
        self.n = len(pa)
        self.edges = [{v: F(p) for v, p in row.items() if p} for row in edges]
        self.c = [a + d for a, d in zip(self.pa, self.pd)]
        self.beta = min(self.c)
        assert self.beta > 0
        self.incoming = [[] for _ in pa]
        for u, row in enumerate(self.edges):
            assert self.c[u] + sum(row.values()) == 1
            for v, p in row.items():
                assert 0 <= v < self.n and p > 0
                self.incoming[v].append(u)
        self.terminal = [categorical([(CONSUME, a / c), (DROP, d / c)])
                         for a, d, c in zip(self.pa, self.pd, self.c)]
        self.residual = []
        if self.beta < 1:
            for u in range(self.n):
                scale = 1 - self.beta
                a = (self.pa[u] - self.beta * self.pa[u] / self.c[u]) / scale
                d = (self.pd[u] - self.beta * self.pd[u] / self.c[u]) / scale
                entries = [(CONSUME, a), (DROP, d)]
                entries += [(v, p / scale) for v, p in self.edges[u].items()]
                self.residual.append(categorical(entries))

    def horizon(self, rng):
        if self.beta == 1:
            return 0
        return math.floor(math.log1p(-rng.random()) / math.log1p(-float(self.beta)))

    def reverse(self, root, rng, frozen=None):
        """Explore only incoming edges and cache actions within one layer."""
        horizon = len(frozen) - 1 if frozen is not None else self.horizon(rng)
        terminal = frozen[horizon][root] if frozen is not None else draw(self.terminal[root], rng)
        active = {root} if terminal == CONSUME else set()
        for layer in range(horizon - 1, -1, -1):
            cache = {}

            def action(w):
                if w not in cache:
                    cache[w] = frozen[layer][w] if frozen is not None else draw(self.residual[w], rng)
                return cache[w]

            previous = {root} if action(root) == CONSUME else set()
            for v in active:
                for w in self.incoming[v]:
                    if action(w) == v:
                        previous.add(w)
            active = previous
        return active

    def freeze(self, horizon, rng):
        assert self.beta < 1 or horizon == 0
        return [[draw(self.residual[u], rng) for u in range(self.n)]
                for _ in range(horizon)] + [[draw(d, rng) for d in self.terminal]]

    @staticmethod
    def endpoint(seed, frozen):
        """Reference forward traversal used only for checking a frozen world."""
        u = seed
        for layer in frozen:
            action = layer[u]
            if action == CONSUME:
                return u
            if action == DROP:
                return None
            u = action
        raise AssertionError("The terminal layer did not terminate")

    def exact_matrix(self):
        """Rational elimination: Q=(I-P)^(-1)D_a; M=Q transpose."""
        n = self.n
        augmented = []
        for u in range(n):
            left = [F(u == v) - self.edges[u].get(v, F(0)) for v in range(n)]
            right = [self.pa[u] if u == v else F(0) for v in range(n)]
            augmented.append(left + right)
        for col in range(n):
            pivot = next(row for row in range(col, n) if augmented[row][col])
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
            divisor = augmented[col][col]
            augmented[col] = [x / divisor for x in augmented[col]]
            for row in range(n):
                if row != col:
                    multiplier = augmented[row][col]
                    augmented[row] = [x - multiplier * y for x, y in zip(augmented[row], augmented[col])]
        return [[augmented[s][n + u] for s in range(n)] for u in range(n)]


def cases():
    return {
        "two_node_cycle": Model([F(1, 2), 0], [0, F(1, 2)], [{1: F(1, 2)}, {0: F(1, 2)}]),
        "heterogeneous_cycle_and_sink": Model(
            [F(1, 5), F(1, 10), F(7, 20), F(3, 5)],
            [F(1, 10), F(3, 10), F(3, 20), F(2, 5)],
            [{0: F(1, 10), 1: F(3, 5)}, {0: F(1, 5), 2: F(2, 5)},
             {0: F(1, 10), 1: F(1, 5), 3: F(1, 5)}, {}]),
        "beta_one": Model([1, F(1, 4), 0], [0, F(3, 4), 1], [{}, {}, {}]),
        "zero_consumption": Model([0, 0], [F(1, 4), F(1, 2)], [{1: F(3, 4)}, {0: F(1, 2)}]),
        "self_loop": Model([F(1, 10)], [F(1, 10)], [{0: F(4, 5)}]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20000, help="Samples per target per model")
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert args.samples > 0
    rng = random.Random(args.seed)
    report = {"seed": args.seed, "samples_per_target": args.samples, "cases": {}}
    models = cases()
    assert models["two_node_cycle"].exact_matrix() == [[F(2, 3), F(1, 3)], [F(0), F(0)]]
    for name, model in models.items():
        oracle = model.exact_matrix()
        checked = 0
        # Realization-wise check includes merging paths and repeated visits.
        for h in ([0] if model.beta == 1 else range(7)):
            for _ in range(40):
                frozen = model.freeze(h, rng)
                endpoints = [model.endpoint(s, frozen) for s in range(model.n)]
                for u in range(model.n):
                    expected = {s for s in range(model.n) if endpoints[s] == u}
                    assert model.reverse(u, rng, frozen) == expected
                    checked += 1
        errors = []
        empirical = []
        for u in range(model.n):
            counts = [0] * model.n
            for _ in range(args.samples):
                for s in model.reverse(u, rng):
                    counts[s] += 1
            empirical.append([count / args.samples for count in counts])
            for s in range(model.n):
                p = float(oracle[u][s])
                error = abs(empirical[u][s] - p)
                tolerance = 7 * math.sqrt(p * (1 - p) / args.samples) + .002
                assert error <= tolerance, (name, u, s, error, tolerance)
                errors.append(error)
        report["cases"][name] = {
            "beta": str(model.beta), "frozen_world_checks": checked,
            "exact_M": [[str(x) for x in row] for row in oracle],
            "empirical_M": empirical, "max_absolute_error": max(errors),
        }
        print(f"PASS {name}: {checked} frozen-world checks; max probability error {max(errors):.6f}")
    # An empty later layer cannot be used as an early stopping rule.
    model = models["heterogeneous_cycle_and_sink"]
    forced = [[distribution[0][0] for distribution in model.residual], [DROP] * model.n]
    assert CONSUME in model.residual[2][0]
    forced[0][2] = CONSUME
    assert model.reverse(2, rng, forced) == {2}
    # A single shared reverse set does not model two independent coupons.
    cycle = models["two_node_cycle"]
    for _ in range(500):
        assert len(cycle.reverse(0, rng)) == 1
    report["shared_set_warning"] = {
        "shared_reverse_union_probability": "1",
        "two_independent_coupons_consumption_probability": str(1 - (1 - F(2, 3)) * (1 - F(1, 3))),
    }
    report["empty_later_layer_test"] = "passed"
    report["status"] = "passed"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS boundary checks: no early empty-layer stopping; shared-set union differs from independent coupons (1 vs 7/9).")


if __name__ == "__main__":
    main()
