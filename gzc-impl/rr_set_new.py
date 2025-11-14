import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any

import numpy as np
import scipy.sparse as sp
import networkx as nx


# ==============================================================================
# 1. 复用您的RR集生成逻辑，并适配到我们的并行框架中
# ==============================================================================

def run_single_ssr_generation_from_existing(args: Tuple) -> List[Set[int]]:
    """
    这是适配您代码的并行工作单元。
    它执行一次完整的SSR（Super Set of RR-sets）的生成。

    Args:
        args: 一个元组，包含 (nodes, reversed_graph, alpha, k, default_p)
    """
    nodes, reversed_graph, alpha, k, default_p = args

    # a. 随机选择一个根节点 (与您的代码一致)
    root_node_v = random.choice(nodes)

    ssr = []

    # b. 为 k 张优惠券生成 k 个 RR-set
    for _ in range(k):
        # c. 检查根节点的领券概率 (这是新模型引入的)
        if random.random() <= alpha[root_node_v]:
            # d. 调用您修改后的 _generate_single_rr_set 逻辑
            #    注意：我们需要将您的函数稍作修改，以接受反向图
            rr_set = _generate_single_rr_set_adapted(root_node_v, reversed_graph, default_p)
            ssr.append(rr_set)
        else:
            ssr.append(set())

    return ssr


def _generate_single_rr_set_adapted(
        start_node: int,
        reversed_graph: Dict[int, Dict[int, float]],
        default_p: float
) -> set:
    """
    这是您 _generate_single_rr_set 函数的适配版本。
    它直接使用我们预先构建好的反向图字典，以提高效率。
    """
    rr_set = {start_node}
    queue = [start_node]

    head = 0
    while head < len(queue):
        current_node = queue[head]
        head += 1

        # 遍历所有“入邻居”（在反向图中就是普通邻居）
        for predecessor, edge_prob in reversed_graph.get(current_node, {}).items():
            if predecessor not in rr_set:
                # 以传播概率 p 进行“反向穿越”
                if random.random() < edge_prob:
                    rr_set.add(predecessor)
                    queue.append(predecessor)

    return rr_set


# ==============================================================================
# 2. 将您的逻辑整合到 CouponInfluenceMaximizer 类中
# ==============================================================================

class CouponInfluenceMaximizer:

    def __init__(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray, alpha: Dict[int, float], k: int):
        """
        初始化函数，复用您构建图的逻辑。
        """
        if k <= 0:
            raise ValueError("优惠券数量k必须为正整数。")

        self.k = k
        self.alpha = alpha

        # a. 从稀疏矩阵构建图 (复用您的代码)
        self.graph = nx.from_scipy_sparse_array(adj, create_using=nx.DiGraph)
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)

        # b. 构建反向图 (内部数据结构，为了效率)
        self.reversed_graph = self._build_reversed_graph(tranProMatrix)

        self.all_ssrs: List[List[Set[int]]] = []
        print(f"图已初始化，包含 {self.num_nodes} 个节点，将选择 {k} 个种子。")

    def _build_reversed_graph(self, tranProMatrix: np.ndarray) -> Dict[int, Dict[int, float]]:
        """
        根据您的 tranProMatrix 构建反向图字典。
        注意：tranProMatrix[v, u] 是 u->v 的概率。
        在反向图中，我们要找 v 的入邻居 u，所以边是 (u,v)，存储为 reversed_graph[v][u]。
        """
        reversed_graph = defaultdict(dict)
        num_edges_processed = 0
        for u, v in self.graph.edges():
            # tranProMatrix[v, u] 存储的是 u->v 的概率
            probability = tranProMatrix[v, u]
            if probability > 0:
                reversed_graph[v][u] = probability
                num_edges_processed += 1
        print(f"反向图构建完成，处理了 {num_edges_processed} 条边。")
        return dict(reversed_graph)

    def generate_rr_sets_parallel(self, N: int, default_p:   float = 0.1, workers: int = -1):
        """
        并行生成SSR样本，现在它会调用我们适配好的函数。
        """
        if N <= 0:
            raise ValueError("抽样次数N必须为正整数。")

        if workers == -1:
            workers = cpu_count()

        print(f"\n开始生成 {N} 组SSR样本，使用 {workers} 个并行进程...")
        start_time = time.time()

        args_list = [(self.nodes, self.reversed_graph, self.alpha, self.k, default_p) for _ in range(N)]

        with Pool(processes=workers) as pool:
            results = pool.map(run_single_ssr_generation_from_existing, args_list)

        self.all_ssrs = results
        end_time = time.time()
        print(f"SSR样本生成完毕。耗时: {end_time - start_time:.2f} 秒。")

    def select_seeds(self) -> Tuple[List[Tuple[int, int]], float]:
        """
        种子选择函数，适配了您的贪心逻辑，并扩展到多优惠券模型。
        """
        if not self.all_ssrs:
            raise RuntimeError("请先调用 generate_rr_sets_parallel() 生成RR-set样本。")

        print("\n开始选择最优的种子节点...")
        start_time = time.time()

        # a. 构建倒排索引 (与我之前代码一致，但现在处理SSR)
        node_to_ssr_indices = defaultdict(list)
        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                for node in rr_set:
                    node_to_ssr_indices[node].append((ssr_idx, coupon_j))

        selected_seeds = []
        covered_ssr_indices = np.zeros(len(self.all_ssrs), dtype=bool)

        # b. 贪心选择 k 个种子 (复用您的边际增益思想)
        for i in range(self.k):
            max_marginal_gain = -1
            best_seed_combo = None  # (node, coupon_slot)

            # 遍历所有可能的 (节点, 券) 组合来寻找最大边际增益
            for node, appearances in node_to_ssr_indices.items():
                # 按优惠券类型分组，计算每个组合的边际增益
                gains_per_coupon = defaultdict(list)
                for ssr_idx, coupon_j in appearances:
                    if not covered_ssr_indices[ssr_idx]:
                        gains_per_coupon[coupon_j].append(ssr_idx)

                for coupon_j, indices in gains_per_coupon.items():
                    gain = len(indices)
                    if gain > max_marginal_gain:
                        max_marginal_gain = gain
                        best_seed_combo = (node, coupon_j)

            if best_seed_combo is None:
                print(f"在第 {i + 1} 轮没有找到有增益的种子。")
                break

            selected_seeds.append(best_seed_combo)

            # c. 更新已覆盖的SSR (复用您的 is_rr_set_covered 数组)
            best_node, best_coupon = best_seed_combo
            for ssr_idx, coupon_j in node_to_ssr_indices[best_node]:
                if coupon_j == best_coupon:
                    covered_ssr_indices[ssr_idx] = True

            print(f"  - 第 {i + 1}/{self.k} 个种子: 节点 {best_seed_combo[0]} (用于券 {best_seed_combo[1] + 1})，"
                  f"新增覆盖 {max_marginal_gain} 个样本。")

        estimated_influence = (np.sum(covered_ssr_indices) / len(self.all_ssrs)) * self.num_nodes

        end_time = time.time()
        print(f"种子节点选择完毕。耗时: {end_time - start_time:.2f} 秒。")

        return selected_seeds, estimated_influence


# ==============================================================================
# 3. 主执行模块：如何调用新的类
# ==============================================================================

if __name__ == "__main__":

    # a. 创建和您代码中类似的输入数据
    NUM_NODES = 50
    K_COUPONS = 5
    NUM_SAMPLES = 20000

    # 创建随机的邻接矩阵和转移概率矩阵
    adj_matrix = sp.lil_matrix((NUM_NODES, NUM_NODES))
    trans_prob_matrix = np.zeros((NUM_NODES, NUM_NODES))

    p_edge = 0.1
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j and random.random() < p_edge:
                prob = random.uniform(0.01, 0.1)
                adj_matrix[i, j] = 1
                trans_prob_matrix[j, i] = prob  # trans_prob_matrix[v, u] = p(u->v)

    adj_matrix_csr = adj_matrix.tocsr()

    # 每个节点的领券概率 (新模型需要)
    alpha_probabilities = {node: random.uniform(0.1, 0.5) for node in range(NUM_NODES)}

    print("=" * 40)
    print("融合后的多优惠券影响力最大化")
    print(f"网络规模: {NUM_NODES} 节点, {adj_matrix_csr.nnz} 条边")
    print(f"参数: k={K_COUPONS}, N={NUM_SAMPLES}")
    print("=" * 40)

    # b. 使用新类来解决问题
    try:
        maximizer = CouponInfluenceMaximizer(
            adj=adj_matrix_csr,
            tranProMatrix=trans_prob_matrix,
            alpha=alpha_probabilities,
            k=K_COUPONS
        )

        maximizer.generate_rr_sets_parallel(N=NUM_SAMPLES)

        selected_seeds, influence = maximizer.select_seeds()

        print("\n" + "=" * 40)
        print("最终结果")
        print("=" * 40)
        print(f"估算出的最大用户激活期望数: {influence:.2f}")
        print("选择的最优种子组合 (节点, 分配的优惠券编号):")
        for seed, coupon_slot in selected_seeds:
            print(f"  - 节点 {seed} <-- 优惠券 {coupon_slot + 1}")
        print("=" * 40)

    except Exception as e:
        print(f"\n程序运行出错: {e}")