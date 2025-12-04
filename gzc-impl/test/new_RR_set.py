import random
from collections import deque, defaultdict

def reverse_bfs(root, a, d, p, in_neighbors):
    RR_set = {root}
    Q = deque([root])
    while Q:
        u = Q.popleft()
        for w in in_neighbors[u]:
            r = random.random()
            if r > a[w] + d[w]:  # w chooses to forward
                # choose one outgoing neighbor according to p[w]
                neighs, probs = zip(*p[w].items())
                chosen = random.choices(neighs, probs)[0]
                if chosen == u and w not in RR_set:
                    RR_set.add(w)
                    Q.append(w)
    return RR_set

def estimate_sigma_rr(G, a, d, p, S, R):
    """ G: adjacency list dict, S: list of seeds [s1,...,sk] """
    V = list(G.keys())
    n = len(V)
    in_neighbors = defaultdict(list)
    for u in G:
        for v in G[u]:
            in_neighbors[v].append(u)

    success_count = 0
    k = len(S)
    for _ in range(R):
        v = random.choice(V)  # uniform root
        hit = False
        for j in range(k):
            r_vj = random.random()
            if r_vj <= a[v]:
                RR_set = reverse_bfs(v, a, d, p, in_neighbors)
                if S[j] in RR_set:
                    hit = True
                    break
        if hit:
            success_count += 1
    return n * success_count / R

# ====== 示例运行 ======
if __name__ == "__main__":
    # 构造一个小图 G
    G = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': ['D'],
        'D': []
    }
    a = {'A': 0.3, 'B': 0.4, 'C': 0.2, 'D': 0.5}  # adoption prob
    d = {'A': 0.2, 'B': 0.3, 'C': 0.1, 'D': 0.2}  # discard prob
    # forwarding probs (conditioned on forwarding)
    p = {
        'A': {'B': 0.5, 'C': 0.5},
        'B': {'C': 1.0},
        'C': {'D': 1.0},
        'D': {}
    }
    S = ['A', 'B']  # 两张券，分别从 A, B 出发
    # S = ['A']  # 两张券，分别从 A, B 出发
    est_sigma = estimate_sigma_rr(G, a, d, p, S, R=10000)
    print("Estimated sigma(S) ≈", est_sigma)
