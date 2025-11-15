import random
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any
import numpy as np
import scipy.sparse as sp
import logging


# 一次完整的SSR抽样过程
def run_single_ssr_generation(args: Tuple) -> List[Set[int]]:
    """
    Args:
        args: 一个元组，包含 (nodes, reversed_graph, alpha, k)
              nodes (List[int]): 所有节点的列表。
              reversed_graph (Dict[int, Dict[int, float]]): 反向图。
              alpha (Dict[int, float]): 节点的领券概率。
              k (int): 优惠券的数量。

    Returns:
        List[Set[int]]: 一个SSR，即包含k个RR-set的列表。
    """
    nodes, reversed_graph, alpha, k = args

    # 1. 随机采样一个节点 v
    root_node_v = random.choice(nodes)
    ssr = [] # 当前采样节点的 SSR，包含k个rr-set

    # 2. 为k张券生成k 个RR-set
    for _ in range(k):
        # 检查v是否会领取这张券
        if random.random() <= alpha[root_node_v]:
            # 如果领券，则通过反向BFS生成RR-set
            rr_set = {root_node_v}
            queue = [root_node_v]
            head = 0
            while head < len(queue):
                current_node = queue[head]
                head += 1
                for in_neighbor, probability in reversed_graph.get(current_node, {}).items():
                    if in_neighbor not in rr_set:
                        # 模拟边(in_neighbor -> current_node)的激活
                        if random.random() <= probability:
                            rr_set.add(in_neighbor)
                            queue.append(in_neighbor)
            ssr.append(rr_set)
        else:
            # 否则，此券的RR-set为空
            ssr.append(set())
    return ssr


class CouponInfluenceMaximizer:

    def __init__(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray, alpha: Dict[int, float], k: int):
        self.k = k # 优惠券（种子节点）的数量。
        self.alpha = alpha # 每个节点的领券概率。

        # 直接从 adj 获取节点信息
        self.num_nodes = adj.shape[0]
        self.nodes = list(range(self.num_nodes))

        # 使用 tranProMatrix 构建反向图
        self.reversed_graph = self._build_reversed_graph(adj, tranProMatrix)
        self.all_ssrs: List[List[Set[int]]] = []
        logging.info(f"图初始化完成，包含 {self.num_nodes} 个节点，选择 {k} 个种子。")

    def _build_reversed_graph(self, adj: sp.csr_matrix, tranProMatrix: np.ndarray) -> Dict[int, Dict[int, float]]:
        """根据 adj 和 tranProMatrix 构建反向图 利用二维数组"""
        reversed_graph = defaultdict(dict)
        rows, cols = adj.nonzero()
        num_edges_processed = 0
        for u, v in zip(rows, cols):
            # tranProMatrix[v, u] 存储的是 u->v 的概率
            probability = tranProMatrix[v, u]
            if probability > 0:
                reversed_graph[v][u] = probability
                num_edges_processed += 1
        logging.info(f"反向图构建完成，处理了 {num_edges_processed} 条边。")
        return dict(reversed_graph)

    def generate_rr_sets_parallel(self, N: int, workers: int = 2):

        logging.info(f"\n采样次数: {N}, 开始生成 {N} 组SSR，使用 {workers} 个并行进程...")
        start_time = time.time()
        args_list = [(self.nodes, self.reversed_graph, self.alpha, self.k) for _ in range(N)]
        with Pool(processes=workers) as pool:
            results = pool.map(run_single_ssr_generation, args_list)
        self.all_ssrs = results
        end_time = time.time()
        logging.info(f"{N} 个SSR生成完毕。耗时: {end_time - start_time:.2f} 秒。")

    def select_seeds_new(self) -> Tuple[List[int], float]:
        """
        按列（优惠券）依次选择种子（边际增益）（种子节点不能重复）
        """
        logging.info("\n开始选择最优的种子节点 ...")
        start_time = time.time()
        # 1. 构建倒排索引
        inverted_index = defaultdict(list)
        for ssr_idx, ssr in enumerate(self.all_ssrs):
            for coupon_j, rr_set in enumerate(ssr):
                for node in rr_set:
                    # 每个node对应哪个（第i个SSR，第j张券）
                    inverted_index[node].append((ssr_idx, coupon_j))

        # 2. 贪心选择k个不重复的种子
        selected_seeds = []
        covered_ssr_indices = set()  # 记录已覆盖的SSR（行）
        for i in range(self.k): # 外层循环：迭代k次，为每张优惠券（每一列）选择一个种子
            
            coupon_to_assign = i # 当前要分配的优惠券列索引
            max_marginal_gain = -1
            best_candidate_node = None

            # 内层循环：遍历所有节点，寻找当前列的最佳种子
            for node in self.nodes:
                if node in selected_seeds:
                    continue # # 种子不能重复

                # 1.找到该节点在当前优惠券列中出现的所有SSR
                ssrs_for_this_combo = {
                    ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(node, [])
                    if coupon_j == coupon_to_assign
                }
                # 计算边际增益
                marginal_gain = len(ssrs_for_this_combo - covered_ssr_indices)
                if marginal_gain > max_marginal_gain:
                    max_marginal_gain = marginal_gain
                    best_candidate_node = node

            selected_seeds.append(best_candidate_node)
            print(f"  - 第 {i + 1}/{self.k} 个种子 (为券 {coupon_to_assign + 1}): "
                  f"节点 {best_candidate_node}，新增覆盖 {max_marginal_gain} 个样本。")
            ssrs_covered_by_new_seed = {
                ssr_idx for (ssr_idx, coupon_j) in inverted_index.get(best_candidate_node, [])
                if coupon_j == coupon_to_assign
            }
            covered_ssr_indices.update(ssrs_covered_by_new_seed)

        estimated_influence = (len(covered_ssr_indices) / len(self.all_ssrs)) * self.num_nodes
        end_time = time.time()
        logging.info(f"种子节点选择完毕。耗时: {end_time - start_time:.2f} 秒。")
        return selected_seeds, estimated_influence

def deliverers_ris_coverage(
        adj: sp.csr_matrix,  # 原始邻接矩阵
        tranProMatrix: np.ndarray,  # 转移概率矩阵
        seeds_num: int,
        num_samples: int = 100,
        alpha: Dict[int, float] = None  # 新增一个可选参数alpha
) -> list:
    """
    Args:
        adj (sp.csr_matrix): 邻接矩阵。
        tranProMatrix (np.ndarray): 转移概率矩阵, tranProMatrix[v, u] = p(u->v)。
        seeds_num (int): 要选择的种子数量 (对应k)。
        num_samples (int, optional): RR-set/SSR的抽样次数 (对应N)。默认为 100。
        alpha (Dict[int, float], optional): 节点的领券概率。
                                             如果为 None，则所有节点默认使用 0.1。

    Returns:
        list: 包含k个最优种子节点ID的列表。
    """
    # 1. 处理缺失的 alpha 参数（test）
    if alpha is None:
        num_nodes = adj.shape[0]
        alpha = {node: 0.1 for node in range(num_nodes)}

    # 2. 实例化核心实现类
    maximizer = CouponInfluenceMaximizer(
        adj=adj,
        tranProMatrix=tranProMatrix,
        alpha=alpha,
        k=seeds_num
    )

    # 3. 执行核心逻辑
    maximizer.generate_rr_sets_parallel(N=num_samples)
    selected_seeds, estimated_influence = maximizer.select_seeds_new()

    print(f"\n估算的最大影响力: {estimated_influence:.2f}")
    logging.info(f"最终选择的种子集: {selected_seeds}\n")
    return selected_seeds

if __name__ == "__main__":


    NUM_NODES = 500
    SEEDS_TO_SELECT = 10
    NUM_SAMPLES_FOR_RUN = 200

    adj_matrix = sp.lil_matrix((NUM_NODES, NUM_NODES))
    trans_prob_matrix = np.zeros((NUM_NODES, NUM_NODES))
    p_edge = 0.1
    for i in range(NUM_NODES):
        for j in range(NUM_NODES):
            if i != j and random.random() < p_edge:
                prob = random.uniform(0.01, 0.1)
                adj_matrix[i, j] = 1
                trans_prob_matrix[j, i] = prob

    adj_matrix_csr = adj_matrix.tocsr()


    custom_alpha_probs = {node: random.uniform(0.05, 0.3) for node in range(NUM_NODES)}
    seeds2 = deliverers_ris_coverage(
        adj=adj_matrix_csr,
        tranProMatrix=trans_prob_matrix,
        seeds_num=SEEDS_TO_SELECT,
        num_samples=NUM_SAMPLES_FOR_RUN,
        alpha=custom_alpha_probs
    )